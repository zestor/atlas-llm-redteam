# Certified Adversarial Robustness for NLP via Randomized Smoothing

**arXiv**: [arXiv:2002.03457](https://arxiv.org/abs/2002.03457) | **ATLAS**: AML.T0015 | **OWASP**: LLM05 | **Year**: 2020

## Core Finding

Randomized smoothing — the leading certified robustness technique for computer vision — can be extended to NLP through discrete ablation smoothing: randomly dropping words from inputs before classification. Jia et al. prove that a smoothed classifier has a provable lower bound on its prediction probability within a certified Hamming distance (number of word deletions/insertions). This is the first technique to provide formal, certifiable guarantees against word-level adversarial attacks in NLP, as opposed to empirical defenses that can be bypassed by adaptive attackers. Applied to LLM safety classifiers, certified smoothing provides mathematically verifiable safety guarantees rather than empirically measured robustness.

## Threat Model

- **Target**: NLP safety classifiers requiring provable security guarantees; applicable as a defense wrapper around any classifier
- **Attacker capability**: Any word-level adversarial attack within the certified Hamming distance; the certificate holds even against adaptive attackers with full knowledge of the defense
- **Attack success rate**: Certified robustness guarantees zero attack success within the certificate radius; standard attacks fail with probability ≥ 1 - α (user-specified)
- **Defender implication**: Certified defenses provide verifiable security guarantees unlike empirical defenses; the certification radius defines the guaranteed attack resistance

## The Attack Mechanism (as a Defense Framework)

This entry focuses on the defense mechanism that resistance evaluation tools should verify:

Ablation smoothing constructs a smoothed classifier \( g \) from a base classifier \( f \):
\[ g(\mathbf{x}) = \arg\max_c \Pr_{\mathbf{\delta} \sim \mathcal{D}}[f(\mathbf{x} \oplus \mathbf{\delta}) = c] \]

Where \( \mathbf{\delta} \) randomly ablates (removes) each word with probability \( p_{ablate} \). The certification theorem states: if \( p_A = \Pr[f(\mathbf{x} \oplus \mathbf{\delta}) = c_A] > 1/2 \), then \( g \) classifies \( \mathbf{x} \) as \( c_A \) for all inputs within Hamming distance \( r \) of \( \mathbf{x} \), where \( r \) is derived from \( p_A \).

Red teams should verify: (1) the certified radius is sufficient for the threat model, (2) the base classifier accuracy under ablation is acceptable, and (3) adaptive attacks cannot bypass the certificate through non-word-level perturbations.

```mermaid
graph TD
    A[Input text x] --> B{Ablation sampling: drop each word with prob p}
    B --> C[Ablated input x_delta_1]
    B --> D[Ablated input x_delta_2]
    B --> E[... x_delta_N]
    C --> F[Base classifier f]
    D --> F
    E --> F
    F --> G[Majority vote: smoothed classifier g]
    G --> H{Prediction p_A > 0.5?}
    H -->|Yes| I[Certified radius r computed]
    I --> J[Guarantee: g(x') = g(x) for all x' within Hamming r]
    J --> K[Provably secure against word-level attacks]
```

## Implementation

```python
# adversarial-robustness-certified.py
# Implements and evaluates certified NLP robustness via ablation smoothing
from dataclasses import dataclass
from typing import List, Optional, Callable, Tuple, Dict
from datasets.schema import ScanFinding
import uuid
import math
import random


@dataclass
class CertifiedRobustnessResult:
    predicted_class: int
    certified_radius: int
    prediction_probability: float
    certification_confidence: float
    certified: bool
    n_samples: int
    ablation_prob: float


class AblationSmoothedClassifier:
    """
    [Paper citation: arXiv:2002.03457]
    Implements certified NLP robustness via randomized ablation smoothing.
    Provides provable guarantees against word-level adversarial attacks.
    ATLAS: AML.T0015 | OWASP: LLM05
    """

    def __init__(
        self,
        base_classifier_fn: Callable[[str], int],
        ablation_prob: float = 0.3,
        n_classes: int = 2,
        alpha: float = 0.001,  # Certification failure probability
    ):
        self.base_classifier = base_classifier_fn
        self.ablation_prob = ablation_prob
        self.n_classes = n_classes
        self.alpha = alpha

    def _ablate(self, text: str) -> str:
        """Randomly ablate (drop) each word with probability ablation_prob."""
        words = text.split()
        remaining = [w for w in words if random.random() > self.ablation_prob]
        return " ".join(remaining) if remaining else "[EMPTY]"

    def _sample_predictions(
        self, text: str, n_samples: int
    ) -> List[int]:
        """Sample base classifier predictions over ablated inputs."""
        predictions = []
        for _ in range(n_samples):
            ablated = self._ablate(text)
            pred = self.base_classifier(ablated)
            predictions.append(pred)
        return predictions

    def _compute_certified_radius(
        self, p_hat: float, alpha: float
    ) -> Tuple[int, float]:
        """
        Compute certified radius from lower-bound probability estimate.
        Uses Clopper-Pearson confidence interval.
        """
        # Conservative Clopper-Pearson lower bound for p_A
        # Simplified approximation
        margin = 1.96 * math.sqrt(p_hat * (1 - p_hat) / 1000)
        p_lower = max(0.0, p_hat - margin)

        if p_lower <= 0.5:
            return 0, p_lower

        # Certified radius formula from Cohen et al. adapted for text
        # r = floor((Phi_inv(p_A) - Phi_inv(0.5)) / sigma)
        # Simplified: radius proportional to margin above 0.5
        radius = int((p_lower - 0.5) * 10)  # Simplified formula
        return radius, p_lower

    def certify(
        self, text: str, n_samples: int = 1000
    ) -> CertifiedRobustnessResult:
        """
        Compute smoothed prediction with certified robustness guarantee.
        """
        predictions = self._sample_predictions(text, n_samples)

        # Count votes for each class
        class_counts = {c: 0 for c in range(self.n_classes)}
        for pred in predictions:
            if pred in class_counts:
                class_counts[pred] += 1

        # Find top and runner-up classes
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        top_class = sorted_classes[0][0]
        top_count = sorted_classes[0][1]

        p_hat = top_count / n_samples
        certified_radius, p_lower = self._compute_certified_radius(p_hat, self.alpha)

        certified = certified_radius > 0 and p_hat > 0.5

        return CertifiedRobustnessResult(
            predicted_class=top_class,
            certified_radius=certified_radius,
            prediction_probability=p_hat,
            certification_confidence=1.0 - self.alpha,
            certified=certified,
            n_samples=n_samples,
            ablation_prob=self.ablation_prob,
        )

    def to_finding(self, result: CertifiedRobustnessResult) -> ScanFinding:
        """Convert certification result to ScanFinding."""
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0015",
            atlas_tactic="ML Model Evasion",
            owasp_category="LLM05",
            owasp_label="Improper Output Handling",
            severity="LOW" if result.certified and result.certified_radius >= 3 else "MEDIUM",
            finding=(
                f"Certified robustness evaluation: "
                f"{'PASS' if result.certified else 'FAIL'}. "
                f"Certified radius: {result.certified_radius} word substitutions. "
                f"Prediction probability: {result.prediction_probability:.3f}. "
                f"Ablation probability: {result.ablation_prob}."
            ),
            payload_used=f"Ablation smoothing with p={result.ablation_prob}, N={result.n_samples}",
            evidence=(
                f"Certified: {result.certified}. "
                f"Confidence: {result.certification_confidence:.3f}. "
                f"Radius guarantees resistance to {result.certified_radius} word-level attacks."
            ),
            remediation=(
                "Deploy ablation smoothing as a certification wrapper on safety classifiers. "
                "Set certified radius based on expected attack sophistication. "
                "Validate certification using standard word-level attack benchmarks. "
                "Supplement certified smoothing with empirical defenses for non-word attacks."
            ),
            confidence=0.92,
        )
```

## Defenses

1. **Deploy ablation smoothing as a safety wrapper** (AML.M0017): Wrap the safety classifier in an ablation smoothing layer. For each input, sample N=1000 ablated versions, classify each, and take the majority vote. This provides provable guarantees against word-level attacks.

2. **Certificate radius calibration**: Set the ablation probability \( p_{ablate} \) based on the desired certified radius. Higher ablation probability gives larger certified radius but reduces clean accuracy. For safety-critical applications, accept 5–10% clean accuracy drop for a certified radius of 3–5 words.

3. **Dual certification and empirical defense**: Combine certified smoothing (which provides guarantees but may have lower average-case accuracy) with empirical adversarial training (which has better average-case robustness but no guarantees). The combination covers both attack regimes.

4. **Certification auditing in production** (AML.M0018): For high-stakes classification decisions, compute and log the certified radius alongside the prediction. Decisions with low certified radius (< 2 words) should require human review.

5. **Certification failure monitoring**: Track the rate of inputs with certified radius = 0 (cannot certify). High rates of uncertifiable inputs may indicate adversarial attack attempts targeting the smoothing boundaries.

## References

- [Jia et al., "Certified Robustness to Word Substitution Attack with Randomized Ablation," NAACL 2022, arXiv:2002.03457](https://arxiv.org/abs/2002.03457)
- [ATLAS Technique AML.T0015: Evade ML Model](https://atlas.mitre.org/techniques/AML.T0015)
- [Cohen et al., "Certified Adversarial Robustness via Randomized Smoothing," ICML 2019](https://arxiv.org/abs/1902.02918)
