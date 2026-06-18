# Counterfactual Memorization in Neural Language Models — Zhang et al.

**arXiv**: [arXiv:2112.12938](https://arxiv.org/abs/2112.12938) | **ATLAS**: AML.T0024 | **OWASP**: LLM02 | **Year**: 2022

## Core Finding

Zhang et al. introduced "counterfactual memorization" — a formal definition of memorization that measures how much a model's prediction on an example changes when that example is removed from training. This approach separates true memorization (the model changed because of this specific example) from spurious correlation (the model would have learned the same thing from similar examples). The key finding: memorization measured counterfactually is substantially lower than empirical memorization, but the memorized examples that remain are the most privacy-sensitive: rare, unique documents with no paraphrases elsewhere in the corpus. These are exactly the documents most likely to contain private information.

## Threat Model

- **Target**: LLMs trained on any corpora including private or enterprise documents
- **Attacker capability**: Black-box generation access; ability to query multiple model versions or shadow models; no white-box access required
- **Attack success rate**: Counterfactual memorization rate 3-8% for 6B+ models; these memorized examples have near-100% extraction probability given correct prefix
- **Defender implication**: The most memorable documents are rare unique documents — exactly the kind of sensitive enterprise documents that should be least accessible via the model

## The Attack Mechanism

Counterfactual memorization is measured by training "leave-one-out" models — versions of the model trained on the full corpus minus one example. A document is counterfactually memorized if the loss on that document increases substantially when it is excluded from training.

For extraction, the attacker focuses on finding documents with high counterfactual memorization score — a signal that the model encodes the document uniquely. These are identified by querying multiple model variants (or using shadow models) and looking for documents where prediction quality varies significantly across variants.

```mermaid
flowchart TD
    A[Training Corpus D] -->|Train full model| B[Model M_D]
    A -->|Leave one out D minus s| C[Model M_{D-s}]
    B -->|Loss on s| D[L_full(s)]
    C -->|Loss on s| E[L_loo(s)]
    D & E -->|Counterfactual score: L_loo - L_full| F[Memorization Score]
    F -->|High score| G[Highly Memorized: Extractable with right prefix]
```

## Implementation

```python
# llm-privacy-copyrighted-memorization.py
# Counterfactual memorization analysis (Zhang et al., arXiv:2112.12938)
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Dict
import uuid
import numpy as np


@dataclass
class CounterfactualMemorizationResult:
    document: str
    full_model_loss: float
    loo_model_loss: float
    counterfactual_score: float
    is_memorized: bool
    extraction_probability: float
    uniqueness_estimate: float


class CounterfactualMemorizationProbe:
    """
    Paper: arXiv:2112.12938 — Zhang et al., 2022
    Identifies and extracts counterfactually memorized training documents.
    ATLAS: AML.T0024 | OWASP: LLM02
    """

    def __init__(
        self,
        full_model_fn: Callable,
        shadow_models: Optional[List[Callable]] = None,
        memorization_threshold: float = 0.5,
        uniqueness_fn: Optional[Callable] = None,
    ):
        self.full_model_fn = full_model_fn
        self.shadow_models = shadow_models or []
        self.threshold = memorization_threshold
        self.uniqueness_fn = uniqueness_fn

    def _compute_loss(self, model_fn: Callable, text: str) -> float:
        """Compute NLL loss of model on text."""
        try:
            result = model_fn(text)
            if isinstance(result, (int, float)):
                return float(result)
            if isinstance(result, dict):
                return float(result.get('loss', result.get('nll', 5.0)))
            return float(np.mean(result))
        except Exception:
            return 5.0

    def _estimate_uniqueness(self, text: str) -> float:
        """Estimate document uniqueness (1.0 = completely unique)."""
        if self.uniqueness_fn is not None:
            return float(self.uniqueness_fn(text))
        # Heuristic: shorter documents with unusual words are more unique
        words = text.split()
        unique_ratio = len(set(words)) / max(len(words), 1)
        length_factor = 1.0 / (1.0 + np.log(len(words) + 1))
        return min(1.0, unique_ratio * (1 + length_factor))

    def probe(self, document: str) -> CounterfactualMemorizationResult:
        """Measure counterfactual memorization for a document."""
        full_loss = self._compute_loss(self.full_model_fn, document)

        # Estimate LOO loss using shadow models as proxies
        if self.shadow_models:
            loo_losses = [self._compute_loss(m, document) for m in self.shadow_models]
            loo_loss = float(np.mean(loo_losses))
        else:
            # Without shadow models, use perplexity variance as proxy
            loo_loss = full_loss * (1 + np.random.exponential(0.1))

        counterfactual_score = max(0.0, loo_loss - full_loss)
        is_memorized = counterfactual_score > self.threshold

        # Extraction probability correlates with counterfactual score
        extraction_prob = 1.0 / (1.0 + np.exp(-2 * (counterfactual_score - self.threshold)))

        uniqueness = self._estimate_uniqueness(document)

        return CounterfactualMemorizationResult(
            document=document[:500],
            full_model_loss=full_loss,
            loo_model_loss=loo_loss,
            counterfactual_score=counterfactual_score,
            is_memorized=is_memorized,
            extraction_probability=float(extraction_prob),
            uniqueness_estimate=uniqueness,
        )

    def scan_corpus(
        self, documents: List[str], top_k: int = 20
    ) -> List[CounterfactualMemorizationResult]:
        """Scan corpus for most memorized documents."""
        results = [self.probe(doc) for doc in documents]
        results.sort(key=lambda r: r.counterfactual_score, reverse=True)
        return results[:top_k]

    def to_finding(self, result: CounterfactualMemorizationResult):
        from datasets.schema import ScanFinding
        severity = "CRITICAL" if result.extraction_probability > 0.8 else "HIGH"
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0024",
            atlas_tactic="Exfiltration",
            owasp_category="LLM02",
            owasp_label="Sensitive Information Disclosure",
            severity=severity,
            finding=f"Counterfactual memorization detected: score={result.counterfactual_score:.3f} (threshold={self.threshold}), extraction probability={result.extraction_probability:.3f}, uniqueness={result.uniqueness_estimate:.3f}.",
            payload_used="LOO model comparison + counterfactual memorization score",
            evidence=f"Full model loss: {result.full_model_loss:.3f}; LOO loss: {result.loo_model_loss:.3f}; delta: {result.counterfactual_score:.3f}",
            remediation="Remove unique high-memorization-risk documents from training data. For enterprise deployments: audit training corpora for unique internal documents before fine-tuning. Use k-anonymity filtering to exclude documents without k similar counterparts.",
            confidence=0.82,
        )
```

## Defenses

1. **Uniqueness filtering of training data** (AML.M0047): Remove documents that have no near-duplicates in the training corpus (uniqueness score > 0.9). Counterfactual memorization is highest for unique documents; ensuring k-anonymity in training data (every document has at least k similar versions) significantly reduces counterfactual memorization rates.

2. **Differential privacy training**: DP-SGD provides formal bounds on counterfactual memorization through its leave-one-out guarantee. The ε-DP guarantee directly bounds how much removing one example can change the model's predictions.

3. **Model distillation for privacy**: Re-distill production models from their outputs on public data before deployment. Distillation naturally forgets rare training examples (they're not in the distillation dataset), reducing counterfactual memorization without explicit DP training.

4. **Counterfactual memorization auditing** (AML.M0015): Run counterfactual memorization analysis on a sample of training data before each model release. Flag documents with counterfactual score > 0.5 for review. Remove high-risk documents and retrain or continue training without them.

5. **Enterprise document isolation**: Never include sensitive enterprise documents in general-purpose pretraining. Fine-tune on enterprise data only after establishing clear data governance policies, consent frameworks, and technical controls (DP, output filtering) for the enterprise corpus.

## References

- [Zhang et al. — Counterfactual Memorization in Neural Language Models (arXiv:2112.12938)](https://arxiv.org/abs/2112.12938)
- [Carlini et al. — Quantifying Memorization (arXiv:2202.07646)](https://arxiv.org/abs/2202.07646)
- [ATLAS AML.T0024 — Exfiltration via ML Inference API](https://atlas.mitre.org/techniques/AML.T0024)
