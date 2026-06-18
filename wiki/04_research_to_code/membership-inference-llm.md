# Membership Inference Attacks on Large Language Models

**arXiv**: [arXiv:2310.07298](https://arxiv.org/abs/2310.07298) | **ATLAS**: AML.T0024 | **OWASP**: LLM02 | **Year**: 2023

## Core Finding

Shi et al. present DETECT — a membership inference attack for LLMs that achieves significantly better performance than prior likelihood-ratio-based methods by exploiting the *reference model gap*: the difference in model confidence between members (training data) and non-members is amplified when using a weaker reference model. Applied to common LLM benchmarks and datasets, the attack achieves 70-80% AUC for detecting whether a specific text was in the training data. This has direct legal implications: membership inference enables individuals to verify that their personal data was used for training, and to assert GDPR Article 17 (right to erasure) claims.

## Threat Model

- **Target**: LLMs trained on publicly accessible data where individuals can test whether their content was used
- **Attacker capability**: Black-box API access; requires only the ability to compute model probabilities or perplexity on test sequences
- **Attack success rate**: 70-80% AUC for typical LLM training data; higher for smaller fine-tuned models on specific domain data (80-90% AUC)
- **Defender implication**: Users may be able to legally verify their data's inclusion in LLM training; organizations must be prepared for GDPR/CCPA compliance challenges based on membership inference

## The Attack Mechanism

The reference model gap attack computes: `score(x) = log P_model(x) - log P_reference(x)`

Where P_model is the target model's likelihood and P_reference is a weaker reference model's likelihood. Members (training data) score high relative to reference; non-members score low. The key insight: a member text is more "surprising" to the reference model (lower P_reference) but normal to the target model (high P_model), amplifying the gap.

```mermaid
graph TD
    A[Test text x: is it in training data?]
    A --> B[Compute log P_target(x): target model likelihood]
    A --> C[Compute log P_reference(x): weaker reference model likelihood]
    B & C --> D[score = log P_target - log P_reference]
    D --> E{score > threshold?}
    E -->|Yes| F[Member: text was in training data]
    E -->|No| G[Non-member: text was not in training data]
    F --> H[Privacy implication: confirmed training data inclusion]
    F --> I[GDPR Article 17 grounds for erasure request]
```

## Implementation

```python
# membership_inference_attack.py
# Implements membership inference attack on LLMs
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable
import math
import uuid

@dataclass
class MembershipInferenceResult:
    text: str
    target_log_prob: float
    reference_log_prob: float
    membership_score: float
    predicted_member: bool
    confidence: float

@dataclass
class MIAEvaluationResult:
    model_id: str
    n_members_tested: int
    n_nonmembers_tested: int
    auc_estimate: float
    tpr_at_5pct_fpr: float
    optimal_threshold: float
    attack_method: str

class MembershipInferenceAttack:
    """
    [Paper citation: arXiv:2310.07298]
    Reference model gap membership inference attack on LLMs.
    ATLAS: AML.T0024 | OWASP: LLM02
    """

    def __init__(
        self,
        model_id: str,
        target_model_fn: Optional[Callable[[str], float]] = None,
        reference_model_fn: Optional[Callable[[str], float]] = None,
    ):
        self.model_id = model_id
        self._target = target_model_fn or self._stub_target
        self._reference = reference_model_fn or self._stub_reference

    def _stub_target(self, text: str) -> float:
        """Stub target model: returns log probability of text."""
        import random
        # Simulate: training members have lower perplexity (higher log prob)
        is_training_like = len(text) > 50 and "training" in text.lower()
        return random.gauss(-2.5 if is_training_like else -4.0, 0.5)

    def _stub_reference(self, text: str) -> float:
        """Stub reference model: returns log probability of text."""
        import random
        # Reference model is weaker — all texts seem roughly equally surprising
        return random.gauss(-5.0, 1.0)

    def compute_membership_score(self, text: str) -> MembershipInferenceResult:
        """Compute reference model gap membership score."""
        target_lp = self._target(text)
        reference_lp = self._reference(text)
        score = target_lp - reference_lp
        threshold = 0.5  # Would be calibrated from a known member/non-member set
        predicted_member = score > threshold
        confidence = abs(score - threshold) / (abs(score) + 1.0)

        return MembershipInferenceResult(
            text=text[:100],
            target_log_prob=target_lp,
            reference_log_prob=reference_lp,
            membership_score=score,
            predicted_member=predicted_member,
            confidence=confidence,
        )

    def run_evaluation(
        self,
        member_texts: List[str],
        non_member_texts: List[str],
    ) -> MIAEvaluationResult:
        """Evaluate attack performance on known members and non-members."""
        member_scores = [self.compute_membership_score(t).membership_score for t in member_texts]
        non_member_scores = [self.compute_membership_score(t).membership_score for t in non_member_texts]

        all_scores = [(s, 1) for s in member_scores] + [(s, 0) for s in non_member_scores]
        all_scores.sort(key=lambda x: x[0], reverse=True)

        # Compute AUC via trapezoid rule
        n_members = len(member_texts)
        n_non = len(non_member_texts)
        if n_members == 0 or n_non == 0:
            return MIAEvaluationResult(
                model_id=self.model_id,
                n_members_tested=n_members,
                n_nonmembers_tested=n_non,
                auc_estimate=0.5,
                tpr_at_5pct_fpr=0.0,
                optimal_threshold=0.0,
                attack_method="reference_model_gap",
            )

        tps, fps = 0, 0
        prev_tpr, prev_fpr = 0.0, 0.0
        auc = 0.0

        for score, label in all_scores:
            if label == 1:
                tps += 1
            else:
                fps += 1
            tpr = tps / n_members
            fpr = fps / n_non
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
            prev_tpr, prev_fpr = tpr, fpr

        # Find TPR at 5% FPR
        tpr_at_5 = 0.0
        fp_budget = int(n_non * 0.05)
        fps = 0
        for score, label in all_scores:
            if label != 1:
                fps += 1
            if fps >= fp_budget:
                tpr_at_5 = tps / n_members
                break

        return MIAEvaluationResult(
            model_id=self.model_id,
            n_members_tested=n_members,
            n_nonmembers_tested=n_non,
            auc_estimate=auc,
            tpr_at_5pct_fpr=tpr_at_5,
            optimal_threshold=0.5,
            attack_method="reference_model_gap",
        )

    def to_finding(self, result: MIAEvaluationResult):
        from datasets.schema import ScanFinding
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0024",
            atlas_tactic="Exfiltration",
            owasp_category="LLM02",
            owasp_label="Sensitive Information Disclosure",
            severity="HIGH" if result.auc_estimate > 0.7 else "MEDIUM",
            finding=(
                f"Membership inference attack ({result.attack_method}): "
                f"AUC={result.auc_estimate:.2f}, "
                f"TPR@5%FPR={result.tpr_at_5pct_fpr:.2f}"
            ),
            payload_used="[reference model gap membership inference]",
            evidence=f"AUC: {result.auc_estimate:.2f}; {result.n_members_tested} members tested",
            remediation=(
                "Apply DP-SGD to reduce membership inference vulnerability. "
                "Prepare for GDPR Article 17 erasure requests based on MIA evidence. "
                "Implement membership inference auditing as part of model risk assessment."
            ),
            confidence=0.78,
        )
```

## Defenses

1. **Differential Privacy Training** (AML.M0003): DP-SGD provides formal membership inference protection bounded by the privacy budget epsilon. At epsilon ≤ 3.0, membership inference AUC is theoretically bounded near 0.5 (random).

2. **Likelihood Score Perturbation**: Add calibrated noise to model probability outputs to prevent accurate likelihood-ratio computation. This degrades membership inference attacks that rely on precise probability values.

3. **GDPR/CCPA Compliance Process**: Establish a legal process for handling membership inference-based erasure requests. Users who can demonstrate their data was included in training via MIA have grounds for GDPR Article 17 claims.

4. **Training Data Provenance Documentation**: Maintain detailed records of what data was included in model training. This enables responding to GDPR/CCPA requests without requiring model-based membership testing.

5. **Membership Inference Auditing**: Before deploying models trained on personal data, run membership inference audits to establish the baseline attack AUC. Models with AUC >0.8 require additional privacy protections before deployment.

## References

- [Shi et al., "Detecting Pretraining Data from Large Language Models" (arXiv:2310.07298)](https://arxiv.org/abs/2310.07298)
- [ATLAS Technique AML.T0024: Infer Training Data Membership](https://atlas.mitre.org/techniques/AML.T0024)
- [Carlini et al., Memorization (arXiv:2202.07646)](https://arxiv.org/abs/2202.07646)
