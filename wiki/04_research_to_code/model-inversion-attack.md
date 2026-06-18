# Model Inversion Attacks — Fredrikson et al.

**arXiv**: [arXiv:1702.07464](https://arxiv.org/abs/1702.07464) | **ATLAS**: AML.T0044 | **OWASP**: LLM02 | **Year**: 2015

## Core Finding

Fredrikson et al. demonstrated that an adversary with black-box access to a model's prediction API can reconstruct approximate representations of training data using gradient-descent optimization in input space. The canonical example: a pharmacogenetics model predicting drug dosages reveals patient genetic data when inverted. For neural networks, the attack recovers recognizable facial images from a face recognition model by maximizing the confidence of a target class. This established model inversion as a concrete privacy threat: models trained on sensitive data leak that data through their prediction interface.

## Threat Model

- **Target**: Models trained on sensitive personal data (medical records, facial images, private text) exposed via prediction APIs
- **Attacker capability**: Black-box access with confidence score output; knowledge of the target output class; ability to run optimization in input space
- **Attack success rate**: 83% of inverted face images are correctly identified by human evaluators; pharmacogenetic data recovery with 80%+ confidence
- **Defender implication**: Training on sensitive data creates irreducible privacy risk through the prediction interface; differential privacy during training is the only principled defense

## The Attack Mechanism

Model inversion treats input reconstruction as an optimization problem: find x* that maximizes the model's confidence for a target class c. This is solved via gradient ascent in input space:

x_{t+1} = x_t + α · ∇_x log p(c | x_t)

For black-box APIs, the gradient is estimated using finite differences (zeroth-order optimization). The optimization converges to an input that represents the model's "prototype" for class c — a reconstruction that shares statistical properties with the training images for that class.

For language models, model inversion can reconstruct training text: given a target author or style label, gradient-based optimization in the embedding space produces text statistically similar to the target's training documents.

```mermaid
flowchart TD
    A[Target class c] -->|Initialize random input x_0| B[Optimization Loop]
    B -->|Query API: p(c|x_t)| C[Target Model]
    C -->|Confidence score| D[Gradient Estimate]
    D -->|Gradient ascent step| E[Updated Input x_{t+1}]
    E -->|Convergence check| B
    B -->|Converged| F[Reconstructed Training Prototype]
```

## Implementation

```python
# model-inversion-attack.py
# Model inversion via gradient ascent (Fredrikson et al., arXiv:1702.07464)
from dataclasses import dataclass, field
from typing import Optional, List, Callable
import uuid
import numpy as np


@dataclass
class ModelInversionResult:
    reconstructed_input: np.ndarray
    target_class: int
    final_confidence: float
    iterations_used: int
    queries_used: int
    convergence_history: List[float]


class ModelInversionAttack:
    """
    Paper: arXiv:1702.07464 — Fredrikson et al., 2015
    Reconstructs training data prototypes via confidence-guided optimization.
    ATLAS: AML.T0044 | OWASP: LLM02
    """

    def __init__(
        self,
        api_fn: Callable,
        input_dim: int,
        target_class: int,
        learning_rate: float = 0.01,
        max_iterations: int = 500,
        finite_diff_epsilon: float = 1e-3,
        n_classes: int = 10,
    ):
        self.api_fn = api_fn
        self.input_dim = input_dim
        self.target_class = target_class
        self.lr = learning_rate
        self.max_iterations = max_iterations
        self.epsilon = finite_diff_epsilon
        self.n_classes = n_classes
        self._queries_used = 0

    def _get_confidence(self, x: np.ndarray) -> float:
        """Get confidence for target class."""
        probs = self.api_fn(x)
        self._queries_used += 1
        if len(probs) > self.target_class:
            return float(probs[self.target_class])
        return 0.0

    def _estimate_gradient(self, x: np.ndarray) -> np.ndarray:
        """Estimate gradient via central finite differences."""
        grad = np.zeros_like(x)
        for i in range(self.input_dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += self.epsilon
            x_minus[i] -= self.epsilon
            f_plus = self._get_confidence(x_plus)
            f_minus = self._get_confidence(x_minus)
            grad[i] = (f_plus - f_minus) / (2 * self.epsilon)
        return grad

    def _estimate_gradient_batch(self, x: np.ndarray, n_directions: int = 20) -> np.ndarray:
        """Estimate gradient using random direction sampling (more efficient)."""
        grad = np.zeros_like(x)
        f0 = self._get_confidence(x)

        for _ in range(n_directions):
            direction = np.random.randn(self.input_dim)
            direction /= np.linalg.norm(direction)
            x_perturb = x + self.epsilon * direction
            f_perturb = self._get_confidence(x_perturb)
            # Stochastic gradient estimate
            grad += (f_perturb - f0) / self.epsilon * direction

        return grad / n_directions

    def run(self) -> ModelInversionResult:
        """Execute model inversion attack."""
        # Initialize from random point
        x = np.random.randn(self.input_dim) * 0.1
        history = []

        for iteration in range(self.max_iterations):
            # Use efficient batch gradient estimation
            grad = self._estimate_gradient_batch(x, n_directions=10)

            # Gradient ascent step
            x = x + self.lr * grad

            # Clip to valid input range
            x = np.clip(x, -3.0, 3.0)

            current_conf = self._get_confidence(x)
            history.append(current_conf)

            # Early stopping if high confidence achieved
            if current_conf > 0.95:
                break

        final_conf = self._get_confidence(x)

        return ModelInversionResult(
            reconstructed_input=x,
            target_class=self.target_class,
            final_confidence=final_conf,
            iterations_used=iteration + 1,
            queries_used=self._queries_used,
            convergence_history=history[-20:],
        )

    def to_finding(self, result: ModelInversionResult):
        from datasets.schema import ScanFinding
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0044",
            atlas_tactic="Exfiltration",
            owasp_category="LLM02",
            owasp_label="Sensitive Information Disclosure",
            severity="HIGH",
            finding=f"Model inversion recovered prototype for class {result.target_class} with {result.final_confidence*100:.1f}% confidence after {result.iterations_used} iterations ({result.queries_used} API queries).",
            payload_used=f"Gradient ascent in input space targeting class {result.target_class}",
            evidence=f"Final confidence: {result.final_confidence:.3f}; convergence history: {result.convergence_history[-5:]}",
            remediation="Apply differential privacy during model training (ε < 1.0). Return only hard labels without confidence scores. Add min-max normalization to output distributions.",
            confidence=0.85,
        )
```

## Defenses

1. **Differential privacy training** (AML.M0047): The only principled defense against model inversion is training with differential privacy (DP-SGD). DP-trained models provably limit what can be inferred about individual training examples. Target ε ≤ 3.0 for high-sensitivity data.

2. **Hard-label-only responses** (AML.M0004): Model inversion requires gradient information, which is extracted from confidence scores. Returning only the top-1 prediction class makes gradient estimation significantly noisier and less efficient.

3. **Output perturbation and randomized response**: Add Laplace or Gaussian noise to confidence outputs calibrated to achieve local ε-DP at the prediction level. This makes gradient estimates noisy while maintaining approximate prediction utility.

4. **Rate limiting and cost enforcement** (AML.M0036): Model inversion requires thousands of queries per target class. Per-user query budgets at the API gateway force attackers into a high-cost regime where systematic inversion becomes economically impractical.

5. **Reconstruction defense via min-max representation**: Train models to minimize the mutual information between input features and output confidence for sensitive attributes. This reduces the information available for gradient-based reconstruction.

## References

- [Fredrikson et al. — Model Inversion Attacks that Exploit Confidence Information (arXiv:1702.07464)](https://arxiv.org/abs/1702.07464)
- [Carlini et al. — Extracting Training Data from Large Language Models (arXiv:2012.07805)](https://arxiv.org/abs/2012.07805)
- [ATLAS AML.T0044 — ML Model Inference API Access](https://atlas.mitre.org/techniques/AML.T0044)
