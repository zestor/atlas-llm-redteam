# Quantifying Memorization Across Neural Language Models — Carlini et al.

**arXiv**: [arXiv:2202.07646](https://arxiv.org/abs/2202.07646) | **ATLAS**: AML.T0024 | **OWASP**: LLM02 | **Year**: 2022

## Core Finding

Carlini et al. provided the first systematic quantification of verbatim memorization in large language models, defining memorization precisely using a k-extractability framework: a string is k-eidetic if it can be extracted with a prompt of k tokens and the model generates the remaining tokens exactly. Larger models memorize more: a 1.3B parameter model memorizes 10× more training examples than a 125M model. Crucially, deduplication of training data reduces memorization by a factor of 10, establishing data deduplication as the most impactful training-time mitigation for verbatim memorization.

## Threat Model

- **Target**: LLMs trained on internet text (GPT-Neo, GPT-J, Pythia, OPT, etc.)
- **Attacker capability**: Black-box generation access with ability to craft prompts; optionally log-probability access for filtering
- **Attack success rate**: 1.7% of unique training sequences extractable from 1.3B models with 100-token prompts; scales to ~15% for 6.7B models
- **Defender implication**: Memorization rates scale predictably with model size; large enterprise LLMs likely memorize significant portions of sensitive training documents

## The Attack Mechanism

The k-extractability framework formalizes memorization: a training string s of length n is k-extractable if the model generates the last n-k tokens of s exactly when prompted with the first k tokens. This captures the intuition that memorization is a generalization failure — the model has overfit to the specific token sequence.

The empirical study tests extractability across diverse model families and finds: (1) larger models memorize more, (2) more training repetitions of a string lead to much higher extractability, (3) deduplication reduces extractability by ~10×, and (4) k matters significantly — longer prompts extract more memorized content.

```mermaid
flowchart LR
    A[Training String s] -->|First k tokens as prompt| B[LLM]
    B -->|Greedy/beam decode remaining tokens| C[Generated Continuation]
    C -->|Exact match with s[k:]?| D{Extractable?}
    D -->|Yes| E[k-eidetic memorization detected]
    D -->|No| F[Not memorized at this k]
    E -->|Frequency in training| G[Memorization rate analysis]
```

## Implementation

```python
# verbatim-memorization-llm.py
# Quantifying verbatim memorization in LLMs (Carlini et al., arXiv:2202.07646)
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Dict
import uuid
import re


@dataclass
class MemorizationAuditResult:
    total_strings_tested: int
    k_extractable_counts: Dict[int, int]
    extractable_examples: List[Dict]
    memorization_rate: float
    most_memorized_strings: List[str]
    duplication_factor_impact: Optional[float]


class VerbatimMemorizationAuditor:
    """
    Paper: arXiv:2202.07646 — Carlini et al., 2022
    Audits LLMs for verbatim training data memorization.
    ATLAS: AML.T0024 | OWASP: LLM02
    """

    def __init__(
        self,
        generation_fn: Callable,
        k_values: Optional[List[int]] = None,
        max_tokens_to_check: int = 50,
        n_test_strings: int = 500,
    ):
        self.generation_fn = generation_fn
        self.k_values = k_values or [5, 10, 20, 50, 100]
        self.max_tokens = max_tokens_to_check
        self.n_test_strings = n_test_strings
        self._queries: int = 0

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenizer for demonstration."""
        return text.split()

    def _is_k_extractable(
        self, string: str, k: int
    ) -> bool:
        """Test if string is k-eidetic memorized."""
        tokens = self._tokenize(string)
        if len(tokens) <= k:
            return False

        prefix = " ".join(tokens[:k])
        expected_suffix = " ".join(tokens[k:k + self.max_tokens])

        try:
            generated = self.generation_fn(
                prompt=prefix,
                max_tokens=min(len(tokens) - k, self.max_tokens),
                temperature=0.0,  # Greedy decoding
            )
            self._queries += 1

            if isinstance(generated, dict):
                generated = generated.get("text", "")
            generated = str(generated).strip()

            # Check exact match (or prefix match for partial extraction)
            return expected_suffix in generated or generated.startswith(expected_suffix[:50])
        except Exception:
            return False

    def audit_strings(
        self, test_strings: List[str]
    ) -> MemorizationAuditResult:
        """Audit a list of candidate training strings for memorization."""
        k_counts: Dict[int, int] = {k: 0 for k in self.k_values}
        extractable_examples: List[Dict] = []
        all_extractable: List[str] = []

        for string in test_strings[:self.n_test_strings]:
            extracted_at_any_k = False

            for k in self.k_values:
                if self._is_k_extractable(string, k):
                    k_counts[k] += 1
                    if not extracted_at_any_k:
                        extractable_examples.append({
                            "string": string[:200],
                            "min_k": k,
                            "length": len(self._tokenize(string)),
                        })
                        all_extractable.append(string)
                        extracted_at_any_k = True
                    break  # Already found at this k, no need to try larger k

        n_tested = min(len(test_strings), self.n_test_strings)
        n_extractable = len(all_extractable)
        memorization_rate = n_extractable / max(n_tested, 1)

        return MemorizationAuditResult(
            total_strings_tested=n_tested,
            k_extractable_counts=k_counts,
            extractable_examples=extractable_examples[:10],
            memorization_rate=memorization_rate,
            most_memorized_strings=all_extractable[:5],
            duplication_factor_impact=None,
        )

    def audit_deduplication_impact(
        self,
        deduped_strings: List[str],
        non_deduped_strings: List[str],
    ) -> float:
        """Measure impact of deduplication on memorization rate."""
        deduped_result = self.audit_strings(deduped_strings)
        non_deduped_result = self.audit_strings(non_deduped_strings)
        if non_deduped_result.memorization_rate < 1e-9:
            return 1.0
        return deduped_result.memorization_rate / non_deduped_result.memorization_rate

    def to_finding(self, result: MemorizationAuditResult):
        from datasets.schema import ScanFinding
        severity = "CRITICAL" if result.memorization_rate > 0.1 else (
            "HIGH" if result.memorization_rate > 0.01 else "MEDIUM"
        )
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0024",
            atlas_tactic="Exfiltration",
            owasp_category="LLM02",
            owasp_label="Sensitive Information Disclosure",
            severity=severity,
            finding=f"Verbatim memorization audit: {result.memorization_rate*100:.2f}% of {result.total_strings_tested} test strings are extractable. K-extractable counts: {result.k_extractable_counts}.",
            payload_used=f"k-eidetic extraction test with k={self.k_values}; greedy decoding",
            evidence=f"Top extractable example: {result.extractable_examples[0]['string'][:100] if result.extractable_examples else 'none'}",
            remediation="Deduplicate training data (reduces memorization by 10×). Apply DP-SGD. Implement output filtering for verbatim reproduction detection. Restrict greedy/low-temperature generation modes in APIs.",
            confidence=0.9,
        )
```

## Defenses

1. **Training data deduplication** (AML.M0047): Carlini et al. show deduplication is the single highest-impact mitigation, reducing memorization by approximately 10×. Use MinHash LSH or suffix array-based deduplication to identify and remove near-duplicate training sequences at both exact and approximate levels.

2. **Differential privacy training**: DP-SGD with ε ≤ 8 significantly reduces verbatim memorization by adding noise to gradient updates, making it harder for any single training example to dominate the model's parameters.

3. **Deduplication-aware canary monitoring** (AML.M0015): Insert canary strings at varying duplication levels (1×, 10×, 100× repetitions) during training. Monitor extractability at each level to verify that deduplication is working correctly and understand the memorization-vs-repetition curve.

4. **Generation mode restrictions**: Restrict greedy (temperature=0) and low-temperature generation in public APIs. Greedy decoding produces the highest-probability sequence, which is most likely to reproduce verbatim memorized content. Minimum temperature > 0.5 significantly reduces extraction rates.

5. **Copyright and PII scanning of training data**: Before training, scan training corpora for copyrighted material, PII, and sensitive documents. Remove or transform identified content. This reduces the volume of high-sensitivity content that can be memorized.

## References

- [Carlini et al. — Quantifying Memorization Across Neural Language Models (arXiv:2202.07646)](https://arxiv.org/abs/2202.07646)
- [Carlini et al. — Extracting Training Data from Large Language Models (arXiv:2012.07805)](https://arxiv.org/abs/2012.07805)
- [ATLAS AML.T0024 — Exfiltration via ML Inference API](https://atlas.mitre.org/techniques/AML.T0024)
