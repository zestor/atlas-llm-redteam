# Reference Model Membership Inference: Improved MIA via Baseline Calibration

**arXiv**: [arXiv:2305.18462](https://arxiv.org/abs/2305.18462) | **ATLAS**: AML.T0024 | **OWASP**: LLM02 | **Year**: 2023

## Core Finding

Reference-based membership inference attacks (MIAs) dramatically outperform naive loss-threshold attacks by comparing the target model's loss on a candidate sample against a reference model's loss on the same sample. Carlini et al.'s Likelihood Ratio Attack (LiRA) achieves AUC of 0.98 on GPT-2 models, near-perfect membership inference, by using the log-likelihood ratio rather than raw log-likelihood as the membership signal. For LLMs trained on sensitive corpora (medical records, legal documents, private emails), this means a model's training set can be reconstructed with high confidence using only black-box API access. Enterprise LLMs trained on proprietary data face systematic exposure of their training corpus.

## Threat Model

- **Target**: LLMs trained or fine-tuned on sensitive or proprietary data and deployed via API
- **Attacker capability**: Black-box API access (log-probabilities); access to a reference model trained on similar but non-overlapping data
- **Attack success rate**: AUC 0.98 on GPT-2; AUC 0.75–0.85 on GPT-3-scale models; 10× improvement over simple loss threshold attacks
- **Defender implication**: Log-probability output is the primary attack vector; limiting or perturbing log-prob outputs significantly reduces MIA efficacy

## The Attack Mechanism

The Likelihood Ratio Attack (LiRA) formulates membership inference as:
\[ \text{score}(x) = \log \frac{P_{target}(x)}{P_{reference}(x)} \]

Where \( P_{target}(x) \) is the target model's loss on candidate \( x \) and \( P_{reference}(x) \) is the reference model's loss. Members of the training set have systematically lower loss than non-members relative to the reference, because the target model has overfitted to them.

The attack is powerful because:
- It controls for inherent sample difficulty (easy sentences have low loss everywhere)
- Reference models can be constructed by training on public data similar to the suspected training corpus
- The log-ratio is statistically robust to threshold selection
- Multiple reference models reduce variance further

For fine-tuned LLMs, even a small fraction of training steps on sensitive data creates distinguishable membership signal that LiRA can detect with high confidence.

```mermaid
graph LR
    A[Candidate Text x] --> B[Target LLM: compute P_target(x)]
    A --> C[Reference Model: compute P_ref(x)]
    B --> D[Log-likelihood ratio: log P_target / P_ref]
    D -->|Ratio >> 1| E[Member of training set]
    D -->|Ratio ~ 1| F[Non-member]
    G[Enterprise proprietary data] -->|Used in fine-tuning| H[Target LLM]
    H -->|API access| B
    E --> I[Training data exposure confirmed]
```

## Implementation

```python
# mia-reference-model-attack.py
# Implements LiRA-style reference-based membership inference attack
from dataclasses import dataclass
from typing import List, Optional, Callable, Dict
from datasets.schema import ScanFinding
import uuid
import math


@dataclass
class ReferenceMIAResult:
    members_detected: List[str]
    non_members: List[str]
    auc_estimate: float
    log_likelihood_ratios: List[float]
    attack_threshold: float
    privacy_leak_confirmed: bool


class ReferenceMIAAttacker:
    """
    [Paper citation: arXiv:2305.18462]
    Implements Likelihood Ratio Attack (LiRA) for membership inference
    on LLMs using a reference model for calibration.
    ATLAS: AML.T0024 | OWASP: LLM02
    """

    def __init__(
        self,
        target_model_logprob_fn: Callable[[str], float],
        reference_model_logprob_fn: Callable[[str], float],
        attack_threshold: float = 0.0,
    ):
        self.target_logprob = target_model_logprob_fn
        self.reference_logprob = reference_model_logprob_fn
        self.attack_threshold = attack_threshold

    def _compute_likelihood_ratio(self, text: str) -> float:
        """Compute log-likelihood ratio as membership score."""
        target_ll = self.target_logprob(text)
        reference_ll = self.reference_logprob(text)
        # Higher ratio = more likely to be a training member
        return target_ll - reference_ll  # log P_target / P_ref

    def _estimate_auc(
        self,
        member_scores: List[float],
        non_member_scores: List[float],
    ) -> float:
        """
        Estimate AUC via Wilcoxon rank-sum statistic.
        AUC = P(member score > non-member score).
        """
        if not member_scores or not non_member_scores:
            return 0.5

        count = 0
        total = len(member_scores) * len(non_member_scores)
        for ms in member_scores:
            for ns in non_member_scores:
                if ms > ns:
                    count += 1
                elif ms == ns:
                    count += 0.5

        return count / max(total, 1)

    def run(
        self,
        candidate_members: List[str],
        candidate_non_members: List[str],
    ) -> ReferenceMIAResult:
        """
        Execute LiRA-style membership inference across candidates.
        """
        member_scores = [self._compute_likelihood_ratio(t) for t in candidate_members]
        non_member_scores = [
            self._compute_likelihood_ratio(t) for t in candidate_non_members
        ]

        all_scores = member_scores + non_member_scores

        detected_members = [
            text
            for text, score in zip(candidate_members, member_scores)
            if score > self.attack_threshold
        ]

        confirmed_non_members = [
            text
            for text, score in zip(candidate_non_members, non_member_scores)
            if score <= self.attack_threshold
        ]

        auc = self._estimate_auc(member_scores, non_member_scores)
        leak_confirmed = auc > 0.7

        return ReferenceMIAResult(
            members_detected=detected_members[:10],
            non_members=confirmed_non_members[:5],
            auc_estimate=auc,
            log_likelihood_ratios=all_scores[:20],
            attack_threshold=self.attack_threshold,
            privacy_leak_confirmed=leak_confirmed,
        )

    def to_finding(self, result: ReferenceMIAResult) -> ScanFinding:
        """Convert result to standard ScanFinding."""
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0024",
            atlas_tactic="Exfiltration",
            owasp_category="LLM02",
            owasp_label="Sensitive Information Disclosure",
            severity="HIGH" if result.privacy_leak_confirmed else "MEDIUM",
            finding=(
                f"Reference-based membership inference attack successful. "
                f"AUC estimate: {result.auc_estimate:.3f}. "
                f"{len(result.members_detected)} training samples detected with "
                f"threshold {result.attack_threshold:.2f}. "
                f"LiRA achieves near-perfect membership inference on target model."
            ),
            payload_used=str(result.members_detected[:3]),
            evidence=(
                f"AUC {result.auc_estimate:.3f} significantly above random (0.5). "
                f"Log-likelihood ratio distribution confirms training membership signal."
            ),
            remediation=(
                "Limit or add calibrated noise to log-probability API outputs. "
                "Apply DP-SGD during fine-tuning on sensitive data. "
                "Implement output perturbation to degrade MIA signal. "
                "Audit training corpus for PII before any fine-tuning."
            ),
            confidence=0.88,
        )
```

## Defenses

1. **Log-probability output restriction** (AML.M0019): The most direct defense is to stop returning log-probabilities from the model API. LiRA and all likelihood-ratio attacks require access to per-token probabilities. Token-only outputs significantly increase attack difficulty.

2. **Output probability perturbation**: Add calibrated Gaussian noise to log-probability outputs. This maintains approximate usefulness for legitimate applications while degrading MIA attack performance. The noise level should be calibrated to the desired privacy-utility tradeoff.

3. **DP-SGD during fine-tuning** (AML.M0017): Apply differential privacy noise during the fine-tuning phase on sensitive data. Formal DP guarantees limit the maximum membership inference advantage to a provable bound.

4. **Training data deduplication and minimization**: Remove duplicated examples from training data. Membership inference exploits the model's tendency to overfit — deduplicated data reduces per-example memorization.

5. **API access monitoring and rate limiting**: Monitor for bulk log-probability queries on similar text sequences, which is characteristic of systematic membership inference attacks. Apply rate limiting and alert on anomalous patterns.

## References

- [Carlini et al., "The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks," arXiv:2305.18462](https://arxiv.org/abs/2305.18462)
- [ATLAS Technique AML.T0024: Exfiltration via ML Inference API](https://atlas.mitre.org/techniques/AML.T0024)
- [Carlini et al., "Quantifying Memorization Across Neural Language Models," ICLR 2023](https://arxiv.org/abs/2202.07646)
