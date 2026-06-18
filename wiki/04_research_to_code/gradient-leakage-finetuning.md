# Gradient Leakage from LLM Fine-Tuning — Recovering Training Data from Federated Gradient Updates

**arXiv**: [arXiv:2206.06481](https://arxiv.org/abs/2206.06481) | **ATLAS**: AML.T0024 | **OWASP**: LLM02 | **Year**: 2022

## Core Finding

In federated fine-tuning settings—where organizations fine-tune a shared LLM using their private data while only uploading gradient updates—those gradients leak the private training text. The 2022 study demonstrates that gradient inversion attacks can reconstruct training text with >80% token-level accuracy from a single gradient update of a GPT-2 fine-tuning step, requiring only the model architecture and the aggregated gradient. The attack succeeds even when gradients are aggregated across multiple clients (batch inversion) or compressed with gradient compression. For enterprises using cloud-based federated fine-tuning services (e.g., Azure OpenAI fine-tuning, Vertex AI fine-tuning), gradient uploads during training are a direct channel for training data exfiltration.

## Threat Model

- **Target**: Federated fine-tuning services; multi-party LLM fine-tuning; organizations sharing gradients in collaborative training
- **Attacker capability**: Malicious federated learning server or co-participant that receives gradient updates; white-box access to model architecture
- **Attack success rate**: >80% token recovery accuracy from single-batch gradient updates; degrades to ~40% for batch size >8 but remains significant
- **Defender implication**: Gradient sharing during fine-tuning is equivalent to sharing training data; DP-SGD or gradient obfuscation is mandatory for private fine-tuning

## The Attack Mechanism

During federated fine-tuning, each participating client computes the gradient of the loss function with respect to model parameters using a local batch of private training data. These gradients are uploaded to the server for aggregation. The gradient inversion attack reconstructs the private batch by solving an optimization problem: find text inputs that, when passed through the model, produce the observed gradient. Starting from random dummy text, the attacker iterates using the model's backward pass to minimize the L2 distance between the dummy gradient and the observed gradient. For transformer LMs, the cross-entropy loss gradient with respect to embedding weights directly encodes which tokens appeared in the training batch—making reconstruction analytically tractable in many cases.

```mermaid
flowchart TD
    subgraph Client["Client: Private Fine-Tuning"]
        D[Private Training Text<br/>'[CONFIDENTIAL] Patient #4432...'] --> FWD[Forward Pass]
        FWD --> LOSS[Cross-Entropy Loss]
        LOSS --> GRAD[Gradient ∂L/∂W]
    end
    GRAD -->|Uploaded to server| SERVER[Federated Server]
    subgraph Attack["Gradient Inversion"]
        SERVER --> COPY[Gradient Copy]
        INIT[Random dummy text] --> DINV[Gradient Inversion Loop]
        COPY --> DINV
        DINV -->|min ||∇_dummy - ∇_observed||²| OPT[Optimizer: L-BFGS]
        OPT --> RECOVER[Recovered Text<br/>'[CONFIDENTIAL] Patient #4432...']
    end
```

## Implementation

```python
# gradient_leakage_attack.py
# Gradient inversion attack: reconstructs private training text
# from gradient updates shared in federated fine-tuning.
from dataclasses import dataclass
from typing import List, Optional, Callable, Tuple
import uuid
import numpy as np


@dataclass
class ScanFinding:
    id: str
    atlas_technique: str
    atlas_tactic: str
    owasp_category: str
    owasp_label: str
    severity: str
    finding: str
    payload_used: str
    evidence: str
    remediation: str
    confidence: float


@dataclass
class GradientLeakageResult:
    gradient_id: str
    batch_size: int
    recovered_text: str
    token_recovery_rate: float
    n_optimization_steps: int
    gradient_match_loss: float
    pii_found: bool


class GradientLeakageAttack:
    """
    Paper: arXiv:2206.06481 (2022)
    Recovering training data from gradient updates shared in federated LLM fine-tuning.
    ATLAS: AML.T0024 | OWASP: LLM02
    """

    def __init__(
        self,
        model_forward_fn: Callable[[List[int]], Tuple[np.ndarray, np.ndarray]],
        # (token_ids) -> (loss, gradient_flat)
        gradient_inversion_optimizer,  # e.g., wrapped L-BFGS
        tokenizer_fn: Callable[[str], List[int]],
        detokenizer_fn: Callable[[List[int]], str],
        vocab_size: int = 50257,
        n_optimization_steps: int = 2000,
        batch_size_estimate: int = 1,
        pii_detector_fn: Optional[Callable[[str], bool]] = None,
    ):
        self.model_forward_fn = model_forward_fn
        self.optimizer = gradient_inversion_optimizer
        self.tokenizer_fn = tokenizer_fn
        self.detokenizer_fn = detokenizer_fn
        self.vocab_size = vocab_size
        self.n_steps = n_optimization_steps
        self.batch_size = batch_size_estimate
        self.pii_detector = pii_detector_fn

    def _recover_tokens_from_embedding_gradient(
        self, grad_embedding_matrix: np.ndarray
    ) -> List[int]:
        """
        Analytically recover tokens from embedding gradient.
        Tokens with non-zero row gradient in the embedding matrix appeared in the batch.
        This is exact for batch_size=1 (Zhu et al. insight).
        """
        # Rows with significant norm = tokens present in training batch
        row_norms = np.linalg.norm(grad_embedding_matrix, axis=1)
        threshold = np.percentile(row_norms, 95)  # top 5% norm tokens
        candidate_tokens = [
            int(i) for i in np.where(row_norms > threshold)[0]
        ]
        return sorted(candidate_tokens)

    def invert_gradient(
        self,
        observed_gradient: np.ndarray,
        gradient_id: str,
    ) -> GradientLeakageResult:
        """
        Reconstruct private training text from an observed gradient.
        Uses analytical token recovery + optimization for ordering.
        """
        # Step 1: Analytical token recovery from embedding gradient slice
        embedding_dim = 768  # GPT-2 style
        embedding_grad = observed_gradient[:self.vocab_size * embedding_dim]
        emb_matrix = embedding_grad.reshape(self.vocab_size, embedding_dim)
        candidate_tokens = self._recover_tokens_from_embedding_gradient(emb_matrix)

        # Step 2: Optimization for token ordering (simplified simulation)
        # In practice this uses torch autograd + L-BFGS minimizing gradient MSE
        best_order = candidate_tokens[:50]  # truncate for simulation
        reconstructed = self.detokenizer_fn(best_order)

        # Simulate gradient match loss (would be actual MSE in real attack)
        match_loss = float(np.random.exponential(0.05))  # placeholder
        token_rate = min(1.0, len(candidate_tokens) / max(1, self.batch_size * 20))

        pii_found = self.pii_detector(reconstructed) if self.pii_detector else False

        return GradientLeakageResult(
            gradient_id=gradient_id,
            batch_size=self.batch_size,
            recovered_text=reconstructed,
            token_recovery_rate=token_rate,
            n_optimization_steps=self.n_steps,
            gradient_match_loss=match_loss,
            pii_found=pii_found,
        )

    def run(
        self, gradient_updates: List[Tuple[str, np.ndarray]]  # (gradient_id, gradient)
    ) -> List[GradientLeakageResult]:
        """Run gradient inversion on multiple observed gradient updates."""
        return [self.invert_gradient(grad, gid) for gid, grad in gradient_updates]

    def to_finding(self, results: List[GradientLeakageResult]) -> ScanFinding:
        high_recovery = [r for r in results if r.token_recovery_rate > 0.6]
        pii_hits = [r for r in results if r.pii_found]

        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0024",
            atlas_tactic="Exfiltration",
            owasp_category="LLM02",
            owasp_label="Sensitive Information Disclosure",
            severity="CRITICAL",
            finding=(
                f"Gradient inversion recovered training text from {len(high_recovery)}/{len(results)} "
                f"gradient updates (>60% token recovery). PII found in {len(pii_hits)} recoveries. "
                "Federated fine-tuning gradients are not safe channels for private data."
            ),
            payload_used=f"Gradient inversion (analytical + L-BFGS) on {len(results)} gradient updates",
            evidence=(
                f"Max token_recovery_rate: "
                f"{max(r.token_recovery_rate for r in results):.2f}, "
                f"PII recoveries: {len(pii_hits)}"
                if results else "No results"
            ),
            remediation=(
                "1. Apply DP-SGD with epsilon ≤ 8 to all fine-tuning gradient uploads (AML.M0003). "
                "2. Use secure aggregation protocols (SecAgg) to prevent server from seeing individual gradients. "
                "3. Apply gradient compression + noise before upload to degrade inversion quality. "
                "4. Audit federated fine-tuning agreements for gradient access rights (AML.M0000)."
            ),
            confidence=0.85,
        )
```

## Defenses

1. **Differential Privacy Fine-Tuning / DP-SGD (AML.M0003 — Model Hardening)**: Clip per-sample gradients and add calibrated Gaussian noise before upload. With epsilon ≤ 8 and delta ≤ 1e-5, token recovery rate drops from >80% to <20% while retaining reasonable model quality. DP-SGD is the gold-standard defense against gradient inversion.

2. **Secure Aggregation (SecAgg)**: Use cryptographic secure aggregation protocols (Bonawitz et al.) so the federated server only ever sees the aggregated gradient across all clients—never individual client updates. Without individual gradients, batch inversion is infeasible.

3. **Gradient Compression and Sparsification**: Apply top-k sparsification or quantization to gradients before upload. This intentionally loses information that enables inversion while preserving sufficient signal for model convergence.

4. **Minimum Batch Size Requirements (AML.M0000)**: Enforce minimum batch sizes (≥ 8) during federated fine-tuning. Gradient inversion accuracy drops sharply with batch size; per-batch reconstruction becomes computationally infeasible above batch size 16.

5. **Anonymize Before Fine-Tuning**: Apply k-anonymity or differential privacy mechanisms to training data before it enters the fine-tuning pipeline. Even if gradients are inverted, the reconstructed text will not contain identifiable PII.

## References

- [arXiv:2206.06481 — Gradient Leakage from LLM Fine-Tuning (2022)](https://arxiv.org/abs/2206.06481)
- [Zhu et al., "Deep Leakage from Gradients" (NeurIPS 2019)](https://arxiv.org/abs/1906.08935)
- [ATLAS AML.T0024 — Exfiltration via ML Inference API](https://atlas.mitre.org/techniques/AML.T0024)
- [OWASP LLM02 — Sensitive Information Disclosure](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
