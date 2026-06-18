# Scalable Membership Inference Attacks against Large Language Models — Nasr et al.

**arXiv**: [arXiv:2311.17035](https://arxiv.org/abs/2311.17035) | **ATLAS**: AML.T0024 | **OWASP**: LLM02 | **Year**: 2023

## Core Finding

Nasr et al. demonstrated that membership inference attacks (MIAs) against large language models are significantly more powerful than previously believed when the attacker has access to reference models trained on similar data. Their "likelihood ratio attack" (LiRA-style) achieves substantially higher true positive rates at low false positive rates compared to prior approaches that relied solely on the target model's loss. For GPT-NeoX-20B, the attack achieves 10× improvement over prior baselines at 0.1% FPR. Critically, the attack scales with model size — larger models are more vulnerable, not less, contrary to intuition.

## Threat Model

- **Target**: Large language models trained on internet text corpora (GPT family, Llama, Falcon, etc.)
- **Attacker capability**: Black-box access to target model log-probabilities; access to reference models (e.g., publicly available checkpoints from same model family or similar pretraining data)
- **Attack success rate**: 3-10× improvement over loss-threshold baselines at 0.1% FPR; 66% TPR at 0.1% FPR on Pile-trained models
- **Defender implication**: Membership inference is not a purely theoretical concern for LLMs; organizations must document training data provenance and apply DP training for sensitive corpora

## The Attack Mechanism

The attack computes a likelihood ratio test comparing the target model's loss to a reference model's loss:

LR(x) = log p_target(x) - log p_reference(x)

If LR(x) >> 0, the target model has much lower loss on x than the reference, suggesting x was in training. If LR(x) ≈ 0, the models agree and no membership signal is present.

The reference model captures "how surprising is this document to a model that has never seen it?" — normalizing away the intrinsic difficulty of the text. Without normalization, MIAs fail on common/easy text; the likelihood ratio removes this confound by comparing to a baseline model that approximates the prior difficulty.

```mermaid
flowchart LR
    A[Candidate Document x] -->|Compute loss| B[Target LLM]
    A -->|Compute loss| C[Reference LLM]
    B -->|log p_target(x)| D[Likelihood Ratio]
    C -->|log p_reference(x)| D
    D -->|LR > threshold| E[Member prediction]
    D -->|LR ≤ threshold| F[Non-member prediction]
```

## Implementation

```python
# membership-inference-nasr.py
# Likelihood ratio MIA against LLMs (Nasr et al., arXiv:2311.17035)
from dataclasses import dataclass, field
from typing import Optional, List, Callable
import uuid
import numpy as np


@dataclass
class MembershipInferenceResult:
    is_member: bool
    likelihood_ratio: float
    target_loss: float
    reference_loss: float
    confidence: float
    threshold_used: float
    text_snippet: str


class LikelihoodRatioMIA:
    """
    Paper: arXiv:2311.17035 — Nasr et al., 2023
    Scalable membership inference via likelihood ratio test against reference model.
    ATLAS: AML.T0024 | OWASP: LLM02
    """

    def __init__(
        self,
        target_model_fn: Callable,
        reference_model_fn: Callable,
        threshold: float = 0.0,
        tokenizer_fn: Optional[Callable] = None,
    ):
        self.target_fn = target_model_fn
        self.reference_fn = reference_model_fn
        self.threshold = threshold
        self.tokenizer_fn = tokenizer_fn

    def _compute_log_loss(self, model_fn: Callable, text: str) -> float:
        """Compute per-token negative log-likelihood of text."""
        try:
            result = model_fn(text)
            if isinstance(result, (int, float)):
                return float(result)
            if hasattr(result, 'loss'):
                return float(result.loss)
            if isinstance(result, dict):
                return float(result.get('loss', result.get('nll', 0.0)))
            return float(np.mean(result))
        except Exception:
            return 0.0

    def predict(self, text: str) -> MembershipInferenceResult:
        """Predict membership status of a single document."""
        target_loss = self._compute_log_loss(self.target_fn, text)
        reference_loss = self._compute_log_loss(self.reference_fn, text)

        # Likelihood ratio: member if target loss << reference loss
        lr = reference_loss - target_loss  # Higher = more likely member

        is_member = lr > self.threshold
        confidence = 1.0 / (1.0 + np.exp(-abs(lr - self.threshold)))

        return MembershipInferenceResult(
            is_member=is_member,
            likelihood_ratio=lr,
            target_loss=target_loss,
            reference_loss=reference_loss,
            confidence=float(confidence),
            threshold_used=self.threshold,
            text_snippet=text[:100],
        )

    def predict_batch(
        self, texts: List[str]
    ) -> List[MembershipInferenceResult]:
        """Batch membership inference over multiple documents."""
        return [self.predict(text) for text in texts]

    def calibrate_threshold(
        self,
        calibration_members: List[str],
        calibration_nonmembers: List[str],
        target_fpr: float = 0.01,
    ) -> float:
        """Calibrate decision threshold to achieve target FPR."""
        member_lrs = [self.predict(t).likelihood_ratio for t in calibration_members]
        nonmember_lrs = [self.predict(t).likelihood_ratio for t in calibration_nonmembers]

        # Set threshold at (1-target_fpr) quantile of non-member scores
        threshold = float(np.percentile(nonmember_lrs, (1 - target_fpr) * 100))
        self.threshold = threshold
        return threshold

    def evaluate_attack(
        self,
        test_members: List[str],
        test_nonmembers: List[str],
    ) -> dict:
        """Evaluate attack TPR/FPR/AUC."""
        member_results = self.predict_batch(test_members)
        nonmember_results = self.predict_batch(test_nonmembers)

        member_lrs = [r.likelihood_ratio for r in member_results]
        nonmember_lrs = [r.likelihood_ratio for r in nonmember_results]

        all_lrs = member_lrs + nonmember_lrs
        all_labels = [1] * len(member_lrs) + [0] * len(nonmember_lrs)

        # AUC via trapezoid
        thresholds = sorted(set(all_lrs), reverse=True)
        tprs, fprs = [0.0], [0.0]
        for t in thresholds:
            tp = sum(l == 1 and s >= t for s, l in zip(all_lrs, all_labels))
            fp = sum(l == 0 and s >= t for s, l in zip(all_lrs, all_labels))
            tprs.append(tp / max(len(member_lrs), 1))
            fprs.append(fp / max(len(nonmember_lrs), 1))
        tprs.append(1.0); fprs.append(1.0)

        auc = float(np.trapz(tprs, fprs))

        return {
            "auc": auc,
            "tpr_at_001_fpr": float(np.interp(0.001, fprs, tprs)),
            "tpr_at_01_fpr": float(np.interp(0.01, fprs, tprs)),
            "mean_member_lr": float(np.mean(member_lrs)),
            "mean_nonmember_lr": float(np.mean(nonmember_lrs)),
        }

    def to_finding(self, result: MembershipInferenceResult):
        from datasets.schema import ScanFinding
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0024",
            atlas_tactic="Exfiltration",
            owasp_category="LLM02",
            owasp_label="Sensitive Information Disclosure",
            severity="HIGH",
            finding=f"Likelihood ratio MIA predicted document is {'member' if result.is_member else 'non-member'} with LR={result.likelihood_ratio:.4f} (threshold={result.threshold_used:.4f}), confidence={result.confidence:.3f}.",
            payload_used="Likelihood ratio test: target_loss vs. reference_model_loss",
            evidence=f"Target loss: {result.target_loss:.4f}; reference loss: {result.reference_loss:.4f}; LR: {result.likelihood_ratio:.4f}",
            remediation="Apply differential privacy training (DP-SGD, ε ≤ 8). Audit training dataset for PII and sensitive documents. Do not expose per-token log-probabilities via API. Implement loss-based monitoring for adversarial probing.",
            confidence=result.confidence,
        )
```

## Defenses

1. **Differential privacy training** (AML.M0047): DP-SGD with ε ≤ 8 provides formal membership privacy guarantees. The likelihood ratio attack exploits the gap between member and non-member losses; DP training bounds this gap by definition.

2. **Training data curation and PII removal**: Before training, remove personally identifiable information, private documents, and copyrighted content from training corpora. Documents that cannot be in training cannot be inferred as members.

3. **Log-probability API restrictions** (AML.M0004): The likelihood ratio attack requires per-token or per-sequence log-probabilities. Restricting APIs to generation-only (no logprob output) degrades attack accuracy significantly.

4. **Reference model diversity**: Reduce the availability of reference models that approximate the target model's pretraining distribution. If reference models are unavailable, the likelihood ratio cannot be computed.

5. **Membership inference auditing** (AML.M0015): Proactively audit trained models for membership inference vulnerability using the likelihood ratio framework. If vulnerability is high, consider retraining with stronger DP guarantees or data deduplication.

## References

- [Nasr et al. — Scalable Extraction of Training Data from Production Language Models (arXiv:2311.17035)](https://arxiv.org/abs/2311.17035)
- [Carlini et al. — Extracting Training Data from Large Language Models (arXiv:2012.07805)](https://arxiv.org/abs/2012.07805)
- [ATLAS AML.T0024 — Exfiltration via ML Inference API](https://atlas.mitre.org/techniques/AML.T0024)
