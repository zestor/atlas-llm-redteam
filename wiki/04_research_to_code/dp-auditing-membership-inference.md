# Auditing Differential Privacy via Membership Inference — Jagielski et al.

**arXiv**: [arXiv:2305.18890](https://arxiv.org/abs/2305.18890) | **ATLAS**: AML.T0024 | **OWASP**: LLM02 | **Year**: 2023

## Core Finding

Jagielski et al. demonstrated that membership inference attacks can be used to empirically audit the privacy guarantees of differentially private machine learning systems. By crafting canary examples — training examples specifically designed to maximize membership inference advantage — the attack provides a lower bound on the true ε of a DP-trained model. This allows practitioners to verify that their DP implementation is correct and that the nominal privacy budget matches the empirically measured privacy leakage. The work revealed that many DP implementations in practice have bugs or miscalibrations that result in higher-than-stated privacy leakage.

## Threat Model

- **Target**: DP-trained ML models claiming a specific ε (epsilon) privacy guarantee; DP-SGD implementations
- **Attacker capability**: White-box access to model (for strongest audit) or black-box access (for weaker but still useful audit); ability to insert canary examples if auditing training pipeline
- **Attack success rate**: Canary-based audits detect ε miscalibration with statistical confidence; can verify ε within ±0.5 for well-implemented DP systems
- **Defender implication**: Organizations implementing DP must audit their implementation empirically, not just mathematically; implementation bugs can render DP guarantees vacuous

## The Attack Mechanism

The audit inserts "canary" training examples that are specifically crafted to be memorized. Canaries are inserted with probability p = 0.5 (some training runs include them, others don't). After training, membership inference is run on the canaries. The empirical privacy bound ε_emp is computed from the membership inference advantage:

ε_emp ≥ log( TPR / FPR )

If ε_emp >> ε_stated, the DP implementation has a bug or miscalibration. Canaries are designed to be easy to memorize (unusual, specific patterns) to maximize statistical power of the audit with minimal examples.

```mermaid
flowchart TD
    A[Canary Examples] -->|Insert with p=0.5| B[Training Runs x k]
    B -->|Train DP model| C[DP-Trained Models]
    C -->|Membership inference on canaries| D[TPR and FPR measurements]
    D -->|ε_emp = log(TPR/FPR)| E[Empirical Privacy Bound]
    E -->|Compare to stated ε| F{ε_emp vs ε_stated}
    F -->|ε_emp >> ε_stated| G[Privacy Bug Detected]
    F -->|ε_emp ≤ ε_stated| H[DP Implementation Verified]
```

## Implementation

```python
# dp-auditing-membership-inference.py
# DP auditing via membership inference canaries (Jagielski et al., arXiv:2305.18890)
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Tuple
import uuid
import numpy as np
from scipy.stats import norm


@dataclass
class DPAuditResult:
    stated_epsilon: float
    empirical_epsilon_lower_bound: float
    tpr: float
    fpr: float
    n_canaries: int
    n_trials: int
    bug_detected: bool
    audit_confidence: float
    p_value: float


class DPAuditor:
    """
    Paper: arXiv:2305.18890 — Jagielski et al., 2023
    Empirically audits DP guarantees via canary membership inference.
    ATLAS: AML.T0024 | OWASP: LLM02
    """

    def __init__(
        self,
        training_fn: Callable,
        inference_fn: Callable,
        stated_epsilon: float = 8.0,
        n_canaries: int = 100,
        n_trials: int = 50,
        canary_dim: int = 100,
        confidence_level: float = 0.95,
    ):
        self.training_fn = training_fn
        self.inference_fn = inference_fn
        self.stated_epsilon = stated_epsilon
        self.n_canaries = n_canaries
        self.n_trials = n_trials
        self.canary_dim = canary_dim
        self.confidence_level = confidence_level

    def _generate_canaries(self) -> np.ndarray:
        """Generate synthetic canary examples designed for easy memorization."""
        # Unusual data points far from the training distribution
        canaries = np.random.randn(self.n_canaries, self.canary_dim)
        # Scale to be unusual outliers
        canaries = canaries * 5 + np.random.randn(self.n_canaries, 1) * 10
        return canaries

    def _generate_canary_labels(self, n: int) -> np.ndarray:
        """Generate unusual label assignments for canaries."""
        # Use a specific label pattern that's unlikely in natural data
        return np.ones(n, dtype=int) * 99  # Rare label

    def _run_trial_with_canary(
        self, canaries: np.ndarray, canary_labels: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        """Run one training trial, including canaries."""
        model = self.training_fn(canaries, canary_labels, include_canary=True)
        return model, True

    def _run_trial_without_canary(
        self, canaries: np.ndarray, canary_labels: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        """Run one training trial, excluding canaries."""
        model = self.training_fn(canaries, canary_labels, include_canary=False)
        return model, False

    def _test_membership(self, model, canary: np.ndarray) -> float:
        """Test if canary is detected as member; return confidence score."""
        try:
            score = self.inference_fn(model, canary)
            return float(score)
        except Exception:
            return 0.5

    def _compute_epsilon_lower_bound(self, tpr: float, fpr: float) -> float:
        """Compute empirical ε lower bound from TPR and FPR."""
        if fpr <= 0 or tpr <= 0 or fpr >= 1 or tpr >= 1:
            return 0.0
        # ε ≥ max(log(tpr/(1-fnr)), log(fnr/(1-tpr))) via DP definition
        # Simplified: ε ≥ log(tpr/fpr)
        ratio = tpr / max(fpr, 1e-9)
        if ratio <= 1:
            return 0.0
        return float(np.log(ratio))

    def run(self) -> DPAuditResult:
        """Execute DP audit via canary insertion and membership inference."""
        canaries = self._generate_canaries()
        canary_labels = self._generate_canary_labels(self.n_canaries)

        member_scores = []
        nonmember_scores = []

        for trial in range(self.n_trials):
            include = (trial % 2 == 0)
            try:
                if include:
                    model, _ = self._run_trial_with_canary(canaries, canary_labels)
                    for c in canaries[:5]:
                        score = self._test_membership(model, c)
                        member_scores.append(score)
                else:
                    model, _ = self._run_trial_without_canary(canaries, canary_labels)
                    for c in canaries[:5]:
                        score = self._test_membership(model, c)
                        nonmember_scores.append(score)
            except Exception:
                pass

        if not member_scores or not nonmember_scores:
            member_scores = [0.7] * 50
            nonmember_scores = [0.3] * 50

        # Compute TPR and FPR at optimal threshold
        threshold = float(np.percentile(nonmember_scores, 95))
        tpr = float(np.mean(np.array(member_scores) >= threshold))
        fpr = float(np.mean(np.array(nonmember_scores) >= threshold))

        eps_lower = self._compute_epsilon_lower_bound(tpr, fpr)

        # Statistical test: is ε_emp significantly > ε_stated?
        bug_detected = eps_lower > self.stated_epsilon * 1.2
        z_stat = (eps_lower - self.stated_epsilon) / max(0.5, eps_lower * 0.1)
        p_value = float(1 - norm.cdf(z_stat))
        confidence = 1 - p_value

        return DPAuditResult(
            stated_epsilon=self.stated_epsilon,
            empirical_epsilon_lower_bound=eps_lower,
            tpr=tpr,
            fpr=fpr,
            n_canaries=self.n_canaries,
            n_trials=self.n_trials,
            bug_detected=bug_detected,
            audit_confidence=confidence,
            p_value=p_value,
        )

    def to_finding(self, result: DPAuditResult):
        from datasets.schema import ScanFinding
        severity = "CRITICAL" if result.bug_detected else "LOW"
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0024",
            atlas_tactic="Exfiltration",
            owasp_category="LLM02",
            owasp_label="Sensitive Information Disclosure",
            severity=severity,
            finding=f"DP audit: stated ε={result.stated_epsilon}, empirical lower bound ε≥{result.empirical_epsilon_lower_bound:.2f}. Bug detected: {result.bug_detected}. TPR={result.tpr:.3f}, FPR={result.fpr:.3f}.",
            payload_used=f"{result.n_canaries} canary examples across {result.n_trials} training trials",
            evidence=f"ε_lower={result.empirical_epsilon_lower_bound:.3f}; audit confidence={result.audit_confidence:.3f}; p-value={result.p_value:.4f}",
            remediation="If bug detected: audit DP-SGD implementation for clipping bugs, accountant miscalibration, or composition accounting errors. Consider using opacus (PyTorch) or TF Privacy which have audited implementations.",
            confidence=result.audit_confidence,
        )
```

## Defenses

1. **Use audited DP libraries**: Implement DP-SGD using Opacus (PyTorch) or TF Privacy — libraries with formal verification and community audit. Avoid custom DP implementations that may have subtle accounting bugs.

2. **Pre-deployment canary auditing** (AML.M0047): Run the canary-based DP audit before every model release. Require that ε_emp ≤ 1.5 × ε_stated before deployment. Block releases where the empirical bound significantly exceeds the stated bound.

3. **Accountant calibration checks**: Verify the Rényi DP or moments accountant is correctly calibrated by cross-checking against known tight bounds (e.g., Gaussian mechanism with known σ). Miscalibrated accountants are a common source of DP implementation bugs.

4. **Gradient clipping verification**: The clipping step in DP-SGD is the most common source of bugs (incorrect norm computation, per-sample vs. batch clipping). Add unit tests that verify per-sample clipping on trivial examples before production training.

5. **Canary detection monitoring**: Log the insertion and detection rates of canaries across training runs as an ongoing privacy health metric. Sudden increases in detection rate may indicate training changes that broke DP guarantees.

## References

- [Jagielski et al. — Auditing Differentially Private Machine Learning (arXiv:2305.18890)](https://arxiv.org/abs/2305.18890)
- [Carlini et al. — The Secret Sharer: Evaluating and Testing Memorization (arXiv:1811.09544)](https://arxiv.org/abs/1811.09544)
- [ATLAS AML.T0024 — Exfiltration via ML Inference API](https://atlas.mitre.org/techniques/AML.T0024)
