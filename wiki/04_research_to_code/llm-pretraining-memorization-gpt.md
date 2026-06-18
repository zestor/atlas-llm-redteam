# Quantifying Memorization in Pretrained Language Models

**arXiv**: [arXiv:2205.10770](https://arxiv.org/abs/2205.10770) | **ATLAS**: AML.T0024 | **OWASP**: LLM02 | **Year**: 2022

## Core Finding

Carlini et al. (2022) systematically quantify memorization in large language models using the concept of "extractable memorization" — measuring how often training data can be reconstructed verbatim via targeted prompting. Their study of GPT-Neo, GPT-2, and LaMDA reveals that memorization scales with model size: larger models memorize 2-4× more unique sequences than smaller counterparts on the same training data. Critically, deduplication reduces memorization by 10× without significantly degrading perplexity. For enterprise LLM deployments trained on internal corpora, this establishes that model size alone creates privacy risk even without adversarial attack conditions.

## Threat Model

- **Target**: Large language models trained on sensitive corpora (financial reports, medical records, legal documents, customer data)
- **Attacker capability**: Black-box query access; ability to craft targeted prefix prompts based on partial knowledge of training data
- **Attack success rate**: Up to 67% of training examples recoverable with correct prompting; memorization rate scales with model size
- **Defender implication**: Model size increases should be accompanied by deduplication and memorization auditing, not just capability benchmarking

## The Attack Mechanism

The attack defines "k-extractable memorization" formally: a string s is k-memorized if given its first k tokens as a prompt, the model generates the remaining tokens verbatim with high probability under greedy decoding. This moves beyond individual token likelihood to end-to-end reconstruction.

The study employs three complementary attacks: (1) targeted prefix extraction — use known prefixes to elicit completions, (2) membership inference via reconstruction — confirm if a string is in training data by checking reconstruction success, and (3) counterfactual analysis — comparing memorization rates at different training data repetition levels.

Key empirical findings: data repeated ≥10 times is memorized at 4× the rate of data appearing once; deduplication is more effective than differential privacy at reducing memorization for equivalent utility costs; and memorization is not uniform — outliers, rare formats, and structured data (emails, phone numbers, code) memorize at dramatically higher rates.

```mermaid
flowchart TD
    A[Pretraining Corpus] --> B[Identify High-Frequency / Structured Sequences]
    B --> C[Craft Targeted Prefix Prompts]
    C --> D[Query LLM with Prefix]
    D --> E[Collect Model Completions]
    E --> F[Check Exact Match with Training String]
    F --> G{Verbatim Reconstruction?}
    G -- Yes --> H[k-Extractable Memorization Confirmed]
    G -- No --> I[Try Longer Prefix or Different Temperature]
    I --> D
    H --> J[Extract PII / Sensitive Content]
```

## Implementation

```python
# llm-pretraining-memorization-gpt.py
# Quantifying and exploiting k-extractable memorization in pretrained LLMs
# Based on Carlini et al., 2022 (arXiv:2205.10770)
from dataclasses import dataclass, field
from typing import Optional, List, Callable
from datasets.schema import ScanFinding
import uuid


@dataclass
class MemorizationResult:
    """Result of memorization extraction attempt."""
    prefix: str
    expected_completion: str
    model_completion: str
    k_extractable: bool
    prefix_length: int
    exact_match: bool
    partial_match_ratio: float


@dataclass
class MemorizationAuditResult:
    """Aggregate results of memorization audit."""
    total_probes: int
    k_extractable_count: int
    memorization_rate: float
    pii_sequences_extracted: int
    structured_data_extracted: int
    sample_extractions: List[MemorizationResult] = field(default_factory=list)


class PretrainingMemorizationAudit:
    """
    arXiv:2205.10770 — Carlini et al., Quantifying Memorization in LLMs
    Systematically tests for k-extractable memorization using prefix probing.
    ATLAS: AML.T0024 | OWASP: LLM02
    """

    def __init__(
        self,
        model_query_fn: Optional[Callable] = None,
        prefix_lengths: Optional[List[int]] = None,
        repetition_threshold: int = 10,
        n_probes: int = 500,
    ):
        self.model_query_fn = model_query_fn
        self.prefix_lengths = prefix_lengths or [25, 50, 100, 200]
        self.repetition_threshold = repetition_threshold
        self.n_probes = n_probes

    def _check_exact_match(self, completion: str, expected: str) -> bool:
        """Check if model completion exactly matches expected training string."""
        return completion.strip() == expected.strip()

    def _compute_partial_match(self, completion: str, expected: str) -> float:
        """Compute longest common subsequence ratio."""
        if not expected:
            return 0.0
        matches = sum(c == e for c, e in zip(completion, expected))
        return matches / len(expected)

    def probe_memorization(
        self,
        prefix: str,
        expected_suffix: str,
        k: int,
    ) -> MemorizationResult:
        """
        Test whether a specific string is k-extractable.
        """
        if self.model_query_fn:
            completion = self.model_query_fn(prefix, max_tokens=len(expected_suffix) + 20)
        else:
            # Simulate partial reconstruction
            completion = expected_suffix[:k] + " [continuation]"

        exact = self._check_exact_match(completion, expected_suffix)
        partial = self._compute_partial_match(completion, expected_suffix)

        return MemorizationResult(
            prefix=prefix,
            expected_completion=expected_suffix,
            model_completion=completion,
            k_extractable=exact,
            prefix_length=len(prefix.split()),
            exact_match=exact,
            partial_match_ratio=partial,
        )

    def run(
        self,
        candidate_strings: Optional[List[dict]] = None,
    ) -> MemorizationAuditResult:
        """
        Run full memorization audit on candidate training strings.

        Args:
            candidate_strings: List of dicts with 'prefix' and 'suffix' keys
        """
        if candidate_strings is None:
            # Use synthetic examples simulating structured training data
            candidate_strings = [
                {
                    "prefix": f"Customer ID: {i:06d} Phone:",
                    "suffix": f" 555-{i:04d} Email: user{i}@example.com",
                    "type": "pii",
                }
                for i in range(100)
            ]

        results = []
        pii_count = 0
        structured_count = 0

        for candidate in candidate_strings[: self.n_probes]:
            result = self.probe_memorization(
                prefix=candidate["prefix"],
                expected_suffix=candidate["suffix"],
                k=50,
            )
            results.append(result)

            if result.k_extractable:
                if candidate.get("type") == "pii":
                    pii_count += 1
                elif candidate.get("type") == "structured":
                    structured_count += 1

        extractable = sum(1 for r in results if r.k_extractable)

        return MemorizationAuditResult(
            total_probes=len(results),
            k_extractable_count=extractable,
            memorization_rate=extractable / len(results) if results else 0.0,
            pii_sequences_extracted=pii_count,
            structured_data_extracted=structured_count,
            sample_extractions=results[:10],
        )

    def to_finding(self, result: MemorizationAuditResult) -> ScanFinding:
        """Convert audit result to standardized ScanFinding."""
        severity = (
            "CRITICAL" if result.pii_sequences_extracted > 0
            else "HIGH" if result.memorization_rate > 0.1
            else "MEDIUM"
        )
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0024",
            atlas_tactic="Exfiltration",
            owasp_category="LLM02",
            owasp_label="Sensitive Information Disclosure",
            severity=severity,
            finding=(
                f"Memorization audit found {result.k_extractable_count}/{result.total_probes} "
                f"k-extractable sequences ({result.memorization_rate:.1%}). "
                f"PII sequences extracted: {result.pii_sequences_extracted}. "
                f"Structured data extracted: {result.structured_data_extracted}."
            ),
            payload_used="Targeted prefix prompting with variable prefix lengths",
            evidence=(
                f"Memorization rate: {result.memorization_rate:.1%}; "
                f"PII confirmed extractable: {result.pii_sequences_extracted}"
            ),
            remediation=(
                "Deduplicate training data (reduces memorization by 10×); "
                "apply differential privacy during pretraining; "
                "scan training corpora for PII before training; "
                "implement output filters to block reconstruction of sensitive patterns; "
                "conduct memorization audits before model release."
            ),
            confidence=0.89,
        )
```

## Defenses

1. **Deduplicate training corpora (AML.M0019)**: Deduplication is the single most effective memorization mitigation. Exact or near-duplicate removal (MinHash, suffix array deduplication) reduces memorization by ~10× with minimal perplexity impact, making it the most cost-effective defense.

2. **PII scrubbing before training**: Use named entity recognition and regex patterns to identify and remove phone numbers, email addresses, SSNs, credit card numbers, and other structured PII from training corpora before training begins. Post-hoc scrubbing is far less reliable.

3. **Differential privacy with gradient clipping**: DP-SGD bounds per-sample gradient contribution, limiting how strongly any individual training example can influence model weights. Trade-off: utility degrades, especially for rare or structured sequences.

4. **Output filtering for verbatim reconstruction**: Deploy regex and semantic similarity filters on model outputs to detect and block responses that verbatim reproduce known sensitive patterns from training data.

5. **Model size governance (AML.M0015)**: The scaling law for memorization means larger models require proportionally stricter data hygiene. Establish organizational policies requiring memorization audits as a gate for deploying models above a certain parameter count.

## References

- [Carlini et al., "Quantifying Memorization Across Neural Language Models" (arXiv:2205.10770)](https://arxiv.org/abs/2205.10770)
- [ATLAS AML.T0024 — Membership Inference Attack](https://atlas.mitre.org/techniques/AML.T0024)
- [Training Data Extraction Attacks (arXiv:2012.07805)](https://arxiv.org/abs/2012.07805)
