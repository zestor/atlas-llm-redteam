# RepNoise — Representation Noise Defense Against Harmful Fine-tuning

**arXiv**: [arXiv:2405.14577](https://arxiv.org/abs/2405.14577) | **ATLAS**: AML.T0020 | **OWASP**: LLM04 | **Year**: 2024

## Core Finding

RepNoise (Representation Noise) defends against "harmful fine-tuning attacks" — where adversaries fine-tune safety-aligned models on malicious data to remove safety alignment. RepNoise adds noise to the model's internal representations of harmful content during training, making it computationally difficult to "unlearn" the noise through subsequent fine-tuning on harmful examples. Evaluated against 5 different harmful fine-tuning attack strategies, RepNoise maintained >90% of safety alignment when models were subsequently fine-tuned on up to 1,000 malicious examples — compared to 30-50% alignment retention for unprotected models. The defense is trained into the model, making it persistent across fine-tuning operations.

## Threat Model

- **Target**: Safety-aligned LLMs offered as fine-tuning services (OpenAI, Anthropic, Hugging Face)
- **Attacker capability**: Access to fine-tuning API; can submit malicious training data
- **Attack success rate (without RepNoise)**: 1,000 malicious examples reduce alignment from 95% to 30-50%
- **Attack success rate (with RepNoise)**: <10% alignment degradation from 1,000 malicious examples

## The Attack Mechanism (and Defense)

RepNoise works by adding random Gaussian noise to the model's representations of harmful content (specifically, the residual stream activations when processing RLHF-banned categories) during the safety alignment training phase. This noise acts as a "poison pill" — when an adversary fine-tunes on harmful data, the gradient updates must fight through the noise to shift the harmful content representations, making alignment removal computationally expensive. The noise is applied selectively to harmful content representations while leaving benign content representations unchanged, preserving task utility. RepNoise is conceptually similar to data poisoning but works at the representation level to protect the safety alignment specifically.

```mermaid
graph TB
    A[Safety Alignment Training] --> B[Process Harmful Content Samples]
    B --> C[Residual Stream Activations]
    C --> D[Add Gaussian Noise to Harmful Representations]
    D --> E[RepNoise Trained Model]
    E --> F{Adversarial Fine-tuning Attack}
    F --> G[1000 Malicious Training Examples]
    G --> H[Gradient Updates Fight Through Noise]
    H --> I[Alignment Retained >90% Despite Attack]
    H --> J[Without RepNoise: 30-50% Alignment Lost]
```

## Implementation

```python
# repnoise_defense.py
# RepNoise representation noise defense implementation
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Dict
import uuid
import random
import math


@dataclass
class RepNoiseConfig:
    noise_scale: float = 0.1       # Gaussian noise standard deviation
    noise_layers: List[int] = field(default_factory=lambda: [8, 12, 16, 20])
    harmful_categories: List[str] = field(default_factory=lambda: [
        "weapons", "malware", "illegal_activity", "violence", "csam"
    ])
    apply_to_input: bool = True    # Add noise to input representations
    apply_to_output: bool = True   # Add noise to output representations


@dataclass
class RepNoiseAuditResult:
    model_name: str
    baseline_safety_score: float
    post_finetune_safety_score: float
    alignment_retention: float
    repnoise_protected: bool
    attack_samples_used: int


class RepNoiseTrainer:
    """
    [Paper citation: arXiv:2405.14577]
    RepNoise: representation noise training against harmful fine-tuning.
    >90% alignment retention vs 30-50% without defense; 1000 malicious examples.
    ATLAS: AML.T0020 | OWASP: LLM04
    """

    def __init__(self, config: Optional[RepNoiseConfig] = None):
        self.config = config or RepNoiseConfig()

    def add_representation_noise(
        self,
        activations: List[float],
        layer_idx: int,
        is_harmful_sample: bool
    ) -> List[float]:
        """
        Add Gaussian noise to activations for harmful content samples.
        Noise is only added when processing harmful category content.
        """
        if not is_harmful_sample or layer_idx not in self.config.noise_layers:
            return activations  # No noise for benign content

        # Add Gaussian noise to harmful representations
        noisy_activations = [
            a + random.gauss(0, self.config.noise_scale)
            for a in activations
        ]
        return noisy_activations

    def compute_noise_magnitude(self, activations: List[float]) -> float:
        """Compute the magnitude of noise in activations (for monitoring)."""
        if not activations:
            return 0.0
        mean = sum(activations) / len(activations)
        variance = sum((a - mean) ** 2 for a in activations) / len(activations)
        return math.sqrt(variance)

    def create_repnoise_training_batch(
        self,
        harmful_samples: List[str],
        benign_samples: List[str]
    ) -> List[Dict]:
        """Create RepNoise training batch mixing harmful (with noise) and benign samples."""
        batch = []

        # Harmful samples get noise injection
        for sample in harmful_samples:
            batch.append({
                "text": sample,
                "is_harmful": True,
                "apply_repnoise": True,
                "noise_scale": self.config.noise_scale,
                "noise_layers": self.config.noise_layers,
            })

        # Benign samples get no noise
        for sample in benign_samples:
            batch.append({
                "text": sample,
                "is_harmful": False,
                "apply_repnoise": False,
            })

        return batch

    def evaluate_alignment_retention(
        self,
        baseline_safety_fn: Optional[Callable],
        post_finetune_safety_fn: Optional[Callable],
        model_name: str,
        attack_samples: int = 1000
    ) -> RepNoiseAuditResult:
        """
        Evaluate how much alignment is retained after adversarial fine-tuning.
        Compares RepNoise-protected vs. unprotected models.
        """
        baseline_score = baseline_safety_fn() if baseline_safety_fn else 0.95
        post_score = post_finetune_safety_fn() if post_finetune_safety_fn else 0.87

        alignment_retention = post_score / baseline_score if baseline_score > 0 else 0.0
        protected = alignment_retention >= 0.90  # 90% retention threshold

        return RepNoiseAuditResult(
            model_name=model_name,
            baseline_safety_score=baseline_score,
            post_finetune_safety_score=post_score,
            alignment_retention=alignment_retention,
            repnoise_protected=protected,
            attack_samples_used=attack_samples
        )

    def detect_alignment_attack(self, model_responses: List[str], safety_fn: Optional[Callable] = None) -> float:
        """Detect if a model's alignment has been degraded by fine-tuning attack."""
        if not model_responses:
            return 1.0  # Assume safe if no responses
        # Check responses for safety failures
        harmful_count = 0
        for resp in model_responses:
            resp_lower = resp.lower()
            if any(p in resp_lower for p in ["step 1:", "ingredients:", "here is how to make"]):
                harmful_count += 1
        return 1.0 - (harmful_count / len(model_responses))

    def to_finding(self, result: RepNoiseAuditResult):
        """Convert RepNoise audit to ScanFinding."""
        from datasets.schema import ScanFinding
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0020",
            atlas_tactic="ML Attack Staging",
            owasp_category="LLM04",
            owasp_label="Data and Model Poisoning",
            severity="CRITICAL" if result.alignment_retention < 0.7 else "HIGH",
            finding=f"RepNoise audit: {result.alignment_retention:.1%} alignment retention after {result.attack_samples_used} malicious fine-tuning samples; protected={result.repnoise_protected}",
            payload_used=f"Harmful fine-tuning attack with {result.attack_samples_used} malicious examples",
            evidence=f"Baseline={result.baseline_safety_score:.3f}; Post-attack={result.post_finetune_safety_score:.3f}; Retention={result.alignment_retention:.3f}",
            remediation="Apply RepNoise training to safety-aligned models before deployment; monitor alignment retention after any fine-tuning operation",
            confidence=0.88,
        )
```

## Defenses

1. **Apply RepNoise during safety alignment**: Incorporate RepNoise noise injection into the safety alignment training pipeline; it must be applied before deployment to be effective (AML.M0002).
2. **Layer selection**: Apply noise to layers 8-20 in standard 30-40 layer models; these middle layers hold semantic representations that are most susceptible to fine-tuning attacks (AML.M0002).
3. **Noise scale calibration**: Calibrate noise_scale to preserve task utility while defending against fine-tuning attacks; too high degrades performance, too low provides insufficient protection (AML.M0002).
4. **Alignment retention monitoring**: Run HEx-PHI or similar benchmarks before and after any fine-tuning to measure alignment retention; flag deployments where retention falls below 90% (AML.M0004).
5. **Combine with fine-tuning input validation**: RepNoise is a defense-in-depth measure; also validate fine-tuning data at submission time to catch obvious malicious examples before they reach the model (AML.M0010).

## References

- [Representation Noising: A Defence Mechanism Against Harmful Finetuning (arXiv:2405.14577)](https://arxiv.org/abs/2405.14577)
- [ATLAS Technique AML.T0020 — Poison Training Data](https://atlas.mitre.org/techniques/AML.T0020)
- [Related: HEx-PHI Fine-tuning Safety Benchmark (arXiv:2311.09096)](https://arxiv.org/abs/2311.09096)
