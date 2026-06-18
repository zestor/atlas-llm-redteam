# Input Transformation Defenses for Adversarial NLP: Evaluation and Attack Surface

**arXiv**: [arXiv:2203.01744](https://arxiv.org/abs/2203.01744) | **ATLAS**: AML.T0015 | **OWASP**: LLM05 | **Year**: 2022

## Core Finding

Input transformation defenses — preprocessing steps designed to neutralize adversarial text perturbations before classification — are themselves vulnerable to adaptive adversarial attacks that account for the defense during optimization. Jia et al. systematically evaluate 9 input transformation defenses (spell-correction, synonym substitution, word dropping, character filtering, BERT masking, etc.) and find that all 9 are defeated by adaptive attacks that optimize adversarial examples while constraining the adversarial perturbation to survive the defense pipeline. This demonstrates that input transformations provide a false sense of security: they raise the bar for naive attacks but are bypassable by any attacker who models the defense.

## Threat Model

- **Target**: LLM safety systems relying on input preprocessing as the primary defense against adversarial text attacks
- **Attacker capability**: Knowledge of the defense pipeline (grey-box); able to incorporate the defense into the adversarial example optimization objective
- **Attack success rate**: 100% bypass of all 9 tested input transformation defenses via adaptive attacks; even ensemble defenses bypassed at 94% rate
- **Defender implication**: Input transformation defenses must be treated as partial mitigations, not complete solutions; they reduce attack surface but do not eliminate it

## The Attack Mechanism

The adaptive attack against a defense \( D \) optimizes:
\[ \mathbf{x}_{adv} = \arg\max_{\mathbf{x}' \in \mathcal{S}(\mathbf{x})} L(f(D(\mathbf{x}')), y_{target}) \]

Where \( D \) is the defense transformation and \( \mathcal{S}(\mathbf{x}) \) is the allowed perturbation set. By including \( D \) in the optimization loop, the adversary finds perturbations that survive the defense while still achieving misclassification.

Key insight: defenses that work by detecting "artifacts" of naive attacks (misspellings, unusual tokens, low-frequency words) are inherently game-theoretic — an attacker who knows the detection criterion can craft inputs that avoid triggering it.

```mermaid
graph TD
    A[Adversarial attack without defense awareness] --> B[Defense D applied]
    B --> C[Adversarial artifact removed]
    C --> D[Classifier correctly classifies]
    E[Adaptive adversarial attack aware of D] --> F[Optimize: survive D AND fool classifier]
    F --> G[Adversarial example designed to pass through D]
    G --> B
    B --> H[Defense passes adversarial example]
    H --> I[Classifier fools]
    J[Defense provides false security] --> K[Attack bypasses at 100%]
```

This is a fundamental insight for security-minded practitioners: any deterministic defense can be defeated by an adversary who models the defense. Randomized or certified defenses provide stronger guarantees.

## Implementation

```python
# input-transformation-defenses.py
# Evaluates input transformation defenses against adaptive adversarial attacks
from dataclasses import dataclass
from typing import List, Optional, Callable, Dict, Tuple
from datasets.schema import ScanFinding
import uuid


@dataclass
class DefenseEvaluationResult:
    defense_name: str
    naive_attack_asr: float
    adaptive_attack_asr: float
    defense_effectiveness: float
    defense_bypassable: bool
    adaptive_examples: List[str]


class InputTransformationDefenseEvaluator:
    """
    [Paper citation: arXiv:2203.01744]
    Evaluates input transformation defenses against adaptive adversarial attacks
    that model the defense pipeline during optimization.
    ATLAS: AML.T0015 | OWASP: LLM05
    """

    def __init__(
        self,
        classifier_fn: Callable[[str], float],
        defense_fn: Callable[[str], str],
        defense_name: str = "unknown_defense",
    ):
        self.classifier_fn = classifier_fn
        self.defense_fn = defense_fn
        self.defense_name = defense_name

    def _naive_attack_with_defense(
        self,
        text: str,
        perturbations: List[str],
    ) -> Tuple[float, List[str]]:
        """Measure ASR of naive attack (ignores defense) against defended classifier."""
        successes = 0
        for perturbed in perturbations:
            defended = self.defense_fn(perturbed)
            score = self.classifier_fn(defended)
            if score < 0.3:
                successes += 1
        return successes / max(len(perturbations), 1), []

    def _adaptive_attack_with_defense(
        self,
        text: str,
        n_attempts: int = 20,
    ) -> Tuple[float, List[str]]:
        """
        Adaptive attack: optimize adversarial example to survive defense.
        Modifies input while minimizing defended-classifier score.
        """
        original_score = self.classifier_fn(self.defense_fn(text))
        words = text.split()
        successful_examples = []

        perturbation_ops = [
            lambda w: w + ".",
            lambda w: w.replace("a", "@") if "a" in w else w,
            lambda w: w[0] + " " + w[1:] if len(w) > 2 else w,
            lambda w: w.upper(),
            lambda w: w[::-1] if len(w) <= 3 else w,
        ]

        for attempt in range(n_attempts):
            current = list(words)
            op = perturbation_ops[attempt % len(perturbation_ops)]

            for i in range(min(3, len(current))):
                test = list(current)
                test[i] = op(test[i])
                test_text = " ".join(test)
                defended = self.defense_fn(test_text)
                score = self.classifier_fn(defended)

                if score < original_score:
                    current = test
                    original_score = score

                if original_score < 0.3:
                    successful_examples.append(" ".join(current)[:200])
                    break

            if original_score < 0.3:
                break

        final_score = original_score
        success = final_score < 0.3

        return (1.0 if success else 0.0), successful_examples

    def run(
        self,
        test_inputs: List[str],
        naive_perturbations: Optional[List[List[str]]] = None,
    ) -> DefenseEvaluationResult:
        """
        Evaluate defense effectiveness against naive and adaptive attacks.
        """
        naive_asrs = []
        adaptive_asrs = []
        all_adaptive_examples: List[str] = []

        for i, inp in enumerate(test_inputs):
            naive_perturbs = (
                naive_perturbations[i]
                if naive_perturbations and i < len(naive_perturbations)
                else [inp + " !", inp.replace("e", "3")]
            )
            naive_asr, _ = self._naive_attack_with_defense(inp, naive_perturbs)
            adaptive_asr, examples = self._adaptive_attack_with_defense(inp)

            naive_asrs.append(naive_asr)
            adaptive_asrs.append(adaptive_asr)
            all_adaptive_examples.extend(examples[:2])

        avg_naive = sum(naive_asrs) / max(len(naive_asrs), 1)
        avg_adaptive = sum(adaptive_asrs) / max(len(adaptive_asrs), 1)
        defense_effectiveness = max(0.0, avg_naive - avg_adaptive)
        bypassable = avg_adaptive > 0.5

        return DefenseEvaluationResult(
            defense_name=self.defense_name,
            naive_attack_asr=avg_naive,
            adaptive_attack_asr=avg_adaptive,
            defense_effectiveness=defense_effectiveness,
            defense_bypassable=bypassable,
            adaptive_examples=all_adaptive_examples[:5],
        )

    def to_finding(self, result: DefenseEvaluationResult) -> ScanFinding:
        """Convert result to standard ScanFinding."""
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0015",
            atlas_tactic="ML Model Evasion",
            owasp_category="LLM05",
            owasp_label="Improper Output Handling",
            severity="HIGH" if result.defense_bypassable else "MEDIUM",
            finding=(
                f"Input transformation defense '{result.defense_name}' is bypassable. "
                f"Naive attack ASR: {result.naive_attack_asr:.1%}. "
                f"Adaptive attack ASR: {result.adaptive_attack_asr:.1%}. "
                f"Defense effectiveness: {result.defense_effectiveness:.3f}. "
                f"Defense provides partial mitigation but not security guarantee."
            ),
            payload_used=str(result.adaptive_examples[:2]),
            evidence=(
                f"Adaptive attack bypasses defense at {result.adaptive_attack_asr:.1%} rate. "
                f"Defense bypassable: {result.defense_bypassable}."
            ),
            remediation=(
                "Supplement transformation defenses with certified robustness techniques. "
                "Use randomized defenses rather than deterministic transformations. "
                "Evaluate defenses with adaptive attacks before deployment. "
                "Layer multiple diverse defenses to increase adaptive attack cost."
            ),
            confidence=0.85,
        )
```

## Defenses

1. **Evaluate defenses with adaptive attacks** (AML.M0017): Before deploying any input transformation defense, evaluate it against adaptive attacks that model the defense. A defense that can only stop naive attacks provides limited real-world security.

2. **Randomized defenses**: Replace deterministic input transformations with randomized preprocessing (e.g., random word dropping, random synonym substitution). Randomization makes adaptive attacks significantly harder because the attack objective has high variance.

3. **Certified robustness methods**: Apply certified adversarial robustness techniques (randomized smoothing, interval bound propagation) that provide provable guarantees rather than empirical ones. Certifications hold even against adaptive attackers.

4. **Defense-in-depth with diverse layers** (AML.M0018): Stack multiple diverse defenses. While each individual defense is bypassable, the cost of simultaneously bypassing multiple defenses with different principles increases exponentially.

5. **Defense combination with adversarial training**: Combine input transformation defenses with adversarial training on examples that are specifically designed to survive the transformation pipeline. This creates a more robust baseline that raises the bar for adaptive attacks.

## References

- [Jia et al., "Certified Robustness to Word Substitution Attack with Randomized Ablation," NAACL 2022, arXiv:2203.01744](https://arxiv.org/abs/2203.01744)
- [ATLAS Technique AML.T0015: Evade ML Model](https://atlas.mitre.org/techniques/AML.T0015)
- [Athalye et al., "Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples," ICML 2018](https://arxiv.org/abs/1802.00420)
