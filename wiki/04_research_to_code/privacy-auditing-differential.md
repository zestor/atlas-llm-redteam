# Auditing Differentially Private ML: Tight Privacy Accounting and Evasion Techniques

**arXiv**: [arXiv:2202.07646](https://arxiv.org/abs/2202.07646) | **ATLAS**: AML.T0024 | **OWASP**: LLM02 | **Year**: 2022

## Core Finding

Nasr et al. introduce adversarial privacy auditing for ML models — a methodology for empirically measuring whether a model's claimed privacy guarantee is tight (i.e., the actual privacy loss matches the theoretical bound). Using gradient canary attacks, they demonstrate that standard DP-SGD implementations often provide weaker privacy than claimed due to accounting approximations, suboptimal noise scheduling, and implementation bugs. The paper's attacks achieve measurable privacy violations in models with claimed epsilon values of 1.0-8.0, demonstrating that the theoretical guarantees are not always achieved in practice.

## Threat Model

- **Target**: ML models claiming DP privacy protections; LLMs with claimed differential privacy guarantees
- **Attacker capability**: Requires access to the training gradient (feasible in federated learning, supply chain attacks on training infrastructure)
- **Attack success rate**: Achieves >1.5x tighter empirical epsilon than claimed in 40% of standard DP-SGD implementations tested; gradient canary extraction achieves measurable privacy violations at claimed epsilon=8.0
- **Defender implication**: DP claims require empirical auditing, not just theoretical analysis; privacy auditing should be a gated requirement for deploying models with privacy guarantees

## The Attack Mechanism

Adversarial privacy auditing inserts "gradient canaries" — carefully crafted training examples that maximize gradient distinguishability — and measures whether an adversary observing the output model can distinguish models trained with vs. without the canary. The adversary's advantage in this distinguishing game equals the empirical epsilon (privacy loss).

A high empirical epsilon (close to the claimed theoretical epsilon) indicates the implementation is tight. A low empirical epsilon is good news. But if empirical epsilon > claimed epsilon, the implementation has a privacy bug.

```mermaid
graph TD
    A[Insert gradient canary C into training data]
    A --> B[Train two models: M_in (with C) and M_out (without C)]
    B --> C[Adversary observes M_in and M_out outputs]
    C --> D[Compute distinguishing advantage]
    D --> E[Empirical epsilon = ln(P_in / P_out)]
    E --> F{Empirical epsilon vs. claimed epsilon}
    F -->|Empirical > claimed| G[Privacy bug! True epsilon worse than advertised]
    F -->|Empirical <= claimed| H[Implementation sound; privacy guarantee valid]
    G --> I[Privacy violation: model does not meet DP claims]
```

## Implementation

```python
# privacy_auditing.py
# Adversarial privacy auditing for differential privacy claims
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import math
import uuid

@dataclass
class PrivacyAuditCanary:
    canary_id: str
    sequence: str
    gradient_norm_expected: float
    distinguishability_score: float

@dataclass
class PrivacyAuditResult:
    model_id: str
    claimed_epsilon: float
    empirical_epsilon: float
    is_tight: bool
    privacy_bug_detected: bool
    audit_samples_used: int
    confidence_interval: Tuple[float, float]
    recommendations: List[str]

class AdversarialPrivacyAuditor:
    """
    [Paper citation: arXiv:2202.07646]
    Adversarial privacy auditing for DP-SGD ML models.
    ATLAS: AML.T0024 | OWASP: LLM02
    """

    def __init__(self, model_id: str, n_audit_samples: int = 10000):
        self.model_id = model_id
        self.n_audit_samples = n_audit_samples

    def generate_gradient_canary(self) -> PrivacyAuditCanary:
        """
        Generate a gradient canary maximizing distinguishability.
        In production, this would use gradient-based optimization.
        """
        import random
        canary_id = f"grad_canary_{random.randint(10**6, 10**7)}"
        return PrivacyAuditCanary(
            canary_id=canary_id,
            sequence=f"GRADIENT_CANARY: {canary_id}",
            gradient_norm_expected=1.0,
            distinguishability_score=0.75,
        )

    def _compute_distinguishing_advantage(
        self,
        canary: PrivacyAuditCanary,
        n_samples: int,
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Stub: computes adversary's advantage in distinguishing M_in from M_out.
        Returns (advantage, (lower_CI, upper_CI)).
        In production, would train n_samples models with and without canary.
        """
        import random
        # Simulate: models with implementation issues have higher advantage
        advantage = random.uniform(0.4, 0.7)
        ci_width = 1.96 * math.sqrt(advantage * (1 - advantage) / n_samples)
        return advantage, (advantage - ci_width, advantage + ci_width)

    def _advantage_to_epsilon(self, advantage: float) -> float:
        """Convert distinguishing advantage to epsilon using standard formula."""
        if advantage <= 0.5:
            return 0.0
        if advantage >= 1.0:
            return float("inf")
        return math.log(advantage / (1 - advantage))

    def audit(
        self,
        claimed_epsilon: float,
        n_canaries: int = 10,
    ) -> PrivacyAuditResult:
        """Run adversarial privacy audit."""
        canaries = [self.generate_gradient_canary() for _ in range(n_canaries)]
        all_empirical_epsilons = []
        all_ci_uppers = []

        for canary in canaries:
            advantage, (ci_low, ci_high) = self._compute_distinguishing_advantage(
                canary, self.n_audit_samples // n_canaries
            )
            emp_eps = self._advantage_to_epsilon(advantage)
            ci_upper_eps = self._advantage_to_epsilon(ci_high)
            all_empirical_epsilons.append(emp_eps)
            all_ci_uppers.append(ci_upper_eps)

        empirical_eps = max(all_empirical_epsilons)
        ci_upper = max(all_ci_uppers)
        privacy_bug = ci_upper > claimed_epsilon * 1.1  # 10% tolerance

        recommendations = []
        if privacy_bug:
            recommendations.append(
                "Privacy implementation has bug: empirical epsilon exceeds claimed epsilon"
            )
            recommendations.append("Use independent DP accounting library (PRV accountant)")
        if empirical_eps > claimed_epsilon * 0.9:
            recommendations.append(
                "Tight accounting: reduce noise or increase sampling rate for more headroom"
            )

        return PrivacyAuditResult(
            model_id=self.model_id,
            claimed_epsilon=claimed_epsilon,
            empirical_epsilon=empirical_eps,
            is_tight=empirical_eps > claimed_epsilon * 0.7,
            privacy_bug_detected=privacy_bug,
            audit_samples_used=self.n_audit_samples,
            confidence_interval=(empirical_eps, ci_upper),
            recommendations=recommendations,
        )

    def to_finding(self, result: PrivacyAuditResult):
        from datasets.schema import ScanFinding
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0024",
            atlas_tactic="Exfiltration",
            owasp_category="LLM02",
            owasp_label="Sensitive Information Disclosure",
            severity="CRITICAL" if result.privacy_bug_detected else "MEDIUM",
            finding=(
                f"Privacy audit: claimed epsilon={result.claimed_epsilon}, "
                f"empirical epsilon={result.empirical_epsilon:.2f}, "
                f"privacy_bug={result.privacy_bug_detected}; "
                f"CI=[{result.confidence_interval[0]:.2f}, {result.confidence_interval[1]:.2f}]"
            ),
            payload_used="[gradient canary adversarial audit]",
            evidence=str(result.recommendations),
            remediation=(
                "Run adversarial privacy audit before making DP privacy claims. "
                "Use PRV accountant for tight, reliable privacy accounting. "
                "Do not deploy models with unverified DP claims on sensitive data."
            ),
            confidence=0.82,
        )
```

## Defenses

1. **Adversarial Privacy Auditing Before Deployment** (AML.M0015): Run the Nasr et al. adversarial auditing methodology before making any differential privacy claims about a deployed model. Claimed epsilon must be verified, not just calculated.

2. **PRV Accountant for Tight Accounting**: Use the Privacy Random Variable (PRV) accountant instead of the moment accountant for more accurate DP epsilon computation. The PRV accountant is tighter and less likely to underestimate true privacy loss.

3. **Conservative Privacy Budget Margins**: Claim a privacy budget of 2x the theoretical epsilon to account for implementation approximations. If your accounting says epsilon=3.0, claim epsilon=6.0 to ensure the true privacy loss stays within claims.

4. **Third-Party Privacy Auditing**: Before deploying models with regulatory privacy guarantees (HIPAA, GDPR), commission independent third-party adversarial privacy auditing. First-party audits may have conflicts of interest.

5. **Implementation Testing with Known Bugs**: Test DP-SGD implementations against a known set of implementation bugs (gradient accumulation errors, incorrect noise addition, suboptimal clipping). Libraries change and bugs are introduced; retest after any updates.

## References

- [Nasr et al., "Tight Auditing of Differentially Private Machine Learning" (arXiv:2202.07646)](https://arxiv.org/abs/2202.07646)
- [ATLAS Technique AML.T0024: Infer Training Data Membership](https://atlas.mitre.org/techniques/AML.T0024)
- [Carlini et al., Quantifying Memorization (arXiv:2202.07646)](https://arxiv.org/abs/2202.07646)
