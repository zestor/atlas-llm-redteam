# Model Architecture Fingerprinting via Black-Box Probing

**arXiv**: [arXiv:2210.11760](https://arxiv.org/abs/2210.11760) | **ATLAS**: AML.T0044 | **OWASP**: LLM02 | **Year**: 2022

## Core Finding

Oh et al. demonstrated that the architecture of a deep neural network — including the number of layers, layer types, and skip connections — can be reliably fingerprinted using only black-box prediction queries. Their technique exploits the fact that different architectures have distinct "decision boundary geometries" that manifest in how confidence values change as inputs are perturbed along geodesics in the input manifold. For LLMs specifically, architecture fingerprinting enables adversaries to identify which open-source model family underlies a commercial API, enabling targeted transfer attacks, more efficient extraction, and intellectual property theft claims.

## Threat Model

- **Target**: Commercial LLM APIs and ML services where the underlying architecture is proprietary (e.g., a fine-tuned Llama variant served as a proprietary product)
- **Attacker capability**: Black-box API access; ability to query with carefully crafted inputs along specific directions
- **Attack success rate**: 88% accuracy in identifying model family (GPT/BERT/Llama/T5) using 500 probes; 72% accuracy at specific model size classification
- **Defender implication**: Serving a fine-tuned open-source model does not hide the base architecture; API providers should treat base model identity as potentially discoverable

## The Attack Mechanism

Architecture fingerprinting exploits subtle differences in how different model families process inputs. For transformer models, the number of attention heads creates patterns in output sensitivity that differ across architectures. For CNNs, skip connections create distinctive response "plateaus" when inputs are perturbed orthogonal to the decision boundary.

The attacker constructs a probe set of carefully designed inputs that maximally distinguish between candidate architectures, queries the target API with these probes, and uses the response patterns to match against known architecture signatures. The probe inputs are typically adversarial examples crafted against multiple reference architectures simultaneously — the target's response to these "universal discriminative probes" reveals its family membership.

```mermaid
flowchart LR
    A[Reference Architecture Library] -->|Craft discriminative probes| B[Probe Input Set]
    B -->|Query target API| C[Target Model Response]
    C -->|Feature extraction| D[Response Fingerprint]
    D -->|Nearest-neighbor matching| E[Predicted Architecture Family]
    E -->|Enable targeted attacks| F[Transfer Attack / Extraction]
```

## Implementation

```python
# architecture-fingerprinting.py
# Architecture fingerprinting via black-box probing (Oh et al., arXiv:2210.11760)
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Dict, Tuple
import uuid
import numpy as np


@dataclass
class ArchitectureFingerprintResult:
    predicted_family: str
    predicted_size: str
    confidence: float
    probe_responses: List[np.ndarray]
    queries_used: int
    fingerprint_vector: np.ndarray


class ArchitectureFingerprinter:
    """
    Paper: arXiv:2210.11760 — Oh et al., 2022
    Identifies model architecture family from black-box output patterns.
    ATLAS: AML.T0044 | OWASP: LLM02
    """

    ARCHITECTURE_SIGNATURES = {
        "transformer_small": {"entropy_mean": 2.1, "entropy_var": 0.3, "confidence_slope": 0.15},
        "transformer_large": {"entropy_mean": 1.8, "entropy_var": 0.2, "confidence_slope": 0.22},
        "cnn_shallow": {"entropy_mean": 1.5, "entropy_var": 0.5, "confidence_slope": 0.30},
        "cnn_deep": {"entropy_mean": 1.2, "entropy_var": 0.4, "confidence_slope": 0.35},
    }

    def __init__(
        self,
        api_fn: Callable,
        input_dim: int,
        n_probes: int = 500,
        probe_epsilon: float = 0.1,
    ):
        self.api_fn = api_fn
        self.input_dim = input_dim
        self.n_probes = n_probes
        self.probe_epsilon = probe_epsilon
        self._queries_used = 0

    def _generate_discriminative_probes(self) -> np.ndarray:
        """Generate probes along decision boundary directions."""
        base_inputs = np.random.randn(self.n_probes // 10, self.input_dim)
        probes = []

        for base in base_inputs:
            # Perturb along multiple orthogonal directions
            for k in range(10):
                direction = np.random.randn(self.input_dim)
                direction /= np.linalg.norm(direction)
                scale = self.probe_epsilon * (k + 1)
                probe = base + scale * direction
                probes.append(probe)

        return np.array(probes[:self.n_probes])

    def _extract_fingerprint(self, responses: List[np.ndarray]) -> np.ndarray:
        """Extract discriminative feature vector from responses."""
        responses_arr = np.array(responses)
        max_probs = responses_arr.max(axis=1)
        entropy = -np.sum(
            responses_arr * np.log(np.clip(responses_arr, 1e-9, 1.0)), axis=1
        )

        # Compute response gradient statistics (sensitivity to perturbations)
        n = len(responses)
        consecutive_diffs = np.abs(max_probs[1:] - max_probs[:-1])

        features = np.array([
            np.mean(max_probs),
            np.std(max_probs),
            np.mean(entropy),
            np.std(entropy),
            np.mean(consecutive_diffs),
            np.percentile(max_probs, 25),
            np.percentile(max_probs, 75),
            np.percentile(entropy, 10),
            np.percentile(entropy, 90),
            float(np.sum(max_probs > 0.9)) / n,
        ])
        return features

    def _match_architecture(
        self, fingerprint: np.ndarray
    ) -> Tuple[str, str, float]:
        """Match fingerprint to known architecture signatures."""
        entropy_mean = fingerprint[2]
        entropy_var = fingerprint[3]

        best_match = "transformer_small"
        best_score = float("inf")

        for arch, sig in self.ARCHITECTURE_SIGNATURES.items():
            score = abs(entropy_mean - sig["entropy_mean"]) + abs(entropy_var - sig["entropy_var"])
            if score < best_score:
                best_score = score
                best_match = arch

        family = "transformer" if "transformer" in best_match else "cnn"
        size = "large" if "large" in best_match else "small"
        confidence = max(0.5, 1.0 - best_score)

        return family, size, confidence

    def run(self) -> ArchitectureFingerprintResult:
        """Execute architecture fingerprinting."""
        probes = self._generate_discriminative_probes()
        responses = []

        for probe in probes:
            resp = self.api_fn(probe)
            responses.append(resp)
            self._queries_used += 1

        fingerprint = self._extract_fingerprint(responses)
        family, size, confidence = self._match_architecture(fingerprint)

        return ArchitectureFingerprintResult(
            predicted_family=family,
            predicted_size=size,
            confidence=confidence,
            probe_responses=responses[:5],
            queries_used=self._queries_used,
            fingerprint_vector=fingerprint,
        )

    def to_finding(self, result: ArchitectureFingerprintResult):
        from datasets.schema import ScanFinding
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0044",
            atlas_tactic="Exfiltration",
            owasp_category="LLM02",
            owasp_label="Sensitive Information Disclosure",
            severity="MEDIUM",
            finding=f"Architecture fingerprinting identified target as {result.predicted_family} ({result.predicted_size}) with {result.confidence*100:.0f}% confidence using {result.queries_used} probes.",
            payload_used="Orthogonal perturbation probes along decision boundary directions",
            evidence=f"Fingerprint vector: {result.fingerprint_vector[:4].round(3).tolist()}",
            remediation="Add architecture-masking ensemble layers; vary output distributions across requests; avoid base model identity disclosure in documentation.",
            confidence=result.confidence,
        )
```

## Defenses

1. **Architecture-masking ensembles**: Serve predictions from an ensemble combining models from different architecture families. The ensemble's response patterns will not match any single architecture's fingerprint, defeating discriminative probe-based identification.

2. **Response distribution normalization** (AML.M0004): Apply a uniform post-processing step (fixed temperature, fixed entropy regularization) that standardizes output statistics. This erases the architecture-specific distribution signatures that fingerprinting relies on.

3. **Probe detection and filtering**: Discriminative probes for architecture fingerprinting tend to be adversarial-like inputs — high-norm perturbations from natural data manifolds. Detect and reject inputs that fall far outside the expected input distribution.

4. **Strategic misdirection**: Return responses that deliberately mimic a different architecture's statistical fingerprint. For example, a transformer-based API could add noise calibrated to match CNN response characteristics.

5. **Rate limiting with behavioral analysis** (AML.M0036): Architecture fingerprinting requires a systematic probe set of hundreds to thousands of inputs. Detect systematic probing behavior — particularly structured perturbation patterns — at the API gateway layer.

## References

- [Oh et al. — Architecture Stealing via Black-Box Probing (arXiv:2210.11760)](https://arxiv.org/abs/2210.11760)
- [Tramèr et al. — Stealing Machine Learning Models (arXiv:1609.02943)](https://arxiv.org/abs/1609.02943)
- [ATLAS AML.T0044 — ML Model Inference API Access](https://atlas.mitre.org/techniques/AML.T0044)
