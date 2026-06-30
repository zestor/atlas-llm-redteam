# Embedding Inversion Privacy Attack: Reconstructing Text from Embeddings

**arXiv**: [2310.06816](https://arxiv.org/abs/2310.06816) | **ATLAS**: AML.T0024 | **OWASP**: LLM02 | **Year**: 2023

## Core Finding

Text embeddings produced by embedding-as-a-service APIs (OpenAI `text-embedding-ada-002`, Cohere Embed, Sentence-BERT) can be inverted to recover the original text with high fidelity using gradient-based optimization and language model priors. The **Vec2Text** attack (Morris et al., 2023) recovered 92% of tokens exactly from OpenAI embeddings and reconstructed the semantic meaning of private queries at 76% accuracy even for unseen text domains. This fundamentally undermines privacy-preserving architectures that transmit embeddings instead of raw text, such as encrypted search indexes, embedding-based RAG privacy schemes, and federated learning over embedding representations.

## Threat Model

- **Target**: Embedding-as-a-service APIs; enterprise systems that transmit or store text embeddings as a privacy-preserving substitute for raw text; RAG vector databases storing user query embeddings
- **Attacker capability**: Black-box access to embedding API and the target embeddings; white-box knowledge of embedding model architecture improves but is not required
- **Attack success rate**: 92% exact token recovery on OpenAI ada-002 embeddings (Vec2Text); 76% semantic accuracy on private user queries; BLEU-4 score of 0.71 on sentence-level reconstruction
- **Defender implication**: Vector databases, embedding logs, and federated embedding transmissions must be treated as sensitive as raw text — they are effectively encrypted text with a breakable cipher

## The Attack Mechanism

Vec2Text and similar attacks set up an optimization problem in token space: given a target embedding \( e^* \), find a text sequence \( x \) such that \( \text{embed}(x) \approx e^* \). Since the token space is discrete, the attack uses a **two-stage approach**:

1. **Hypothesis generation**: A corrector language model is trained to map \( (e^*, \hat{x}_t) \rightarrow \hat{x}_{t+1} \), iteratively refining a candidate text using the embedding residual as a conditioning signal.
2. **Beam search over token sequences**: Multiple hypotheses are scored by embedding cosine similarity, retaining the top-k candidates that minimize \( \|embed(x) - e^*\|_2 \).

The corrector model is trained on (embedding, text) pairs from the same distribution as the target API, requiring only black-box API access to collect training data. For APIs with static models (no model versioning changes), this training data can be collected cost-effectively with millions of API queries.

```mermaid
flowchart LR
    A[Private Text\ne.g. user query] -->|Embed| B[Embedding API\nOpenAI ada-002]
    B -->|Embedding vector e*| C[Transmitted / Stored]
    C -->|Attacker intercepts e*| D[Vec2Text Corrector\nLM-guided inversion]
    D -->|Initialize x0 randomly| E[Iteration Loop]
    E -->|embed x_t - e* residual| F[Corrector LM\nPredict x_{t+1}]
    F -->|New candidate| E
    E -->|Converge| G[Reconstructed Text\n92% token accuracy]
    style A fill:#27ae60,color:#fff
    style G fill:#c0392b,color:#fff
```

## Implementation

```python
# embedding_inversion_privacy.py
# Text reconstruction from embeddings via iterative corrector-based inversion.
# Demonstrates Vec2Text-style attack against embedding-as-a-service APIs.
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
import uuid
import numpy as np

try:
    from datasets.schema import ScanFinding
except ImportError:
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
class EmbeddingInversionResult:
    target_embedding_norm: float
    reconstructed_text: str
    reconstruction_confidence: float  # cosine similarity to target
    n_iterations: int
    token_count: int
    exact_token_recovery_rate: Optional[float]  # if ground truth known
    metadata: Dict[str, Any] = field(default_factory=dict)


class EmbeddingInversionAttack:
    """
    arXiv:2310.06816 — Vec2Text: Controlled Text Generation with Natural Language Instructions
    Reconstructs private text from embedding vectors via iterative refinement.
    ATLAS: AML.T0024 | OWASP: LLM02
    """

    def __init__(
        self,
        embed_fn: Callable[[str], np.ndarray],  # target embedding API
        corrector_fn: Optional[Callable] = None,  # LM corrector (if available)
        max_iterations: int = 50,
        beam_width: int = 8,
        convergence_threshold: float = 0.995,
    ):
        self.embed_fn = embed_fn
        self.corrector_fn = corrector_fn
        self.max_iterations = max_iterations
        self.beam_width = beam_width
        self.convergence_threshold = convergence_threshold

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two embedding vectors."""
        a_norm = a / (np.linalg.norm(a) + 1e-10)
        b_norm = b / (np.linalg.norm(b) + 1e-10)
        return float(np.dot(a_norm, b_norm))

    def _generate_candidates(
        self,
        current_text: str,
        target_embedding: np.ndarray,
        vocabulary_sample: List[str],
        n_candidates: int = 20,
    ) -> List[str]:
        """Generate text candidates by single-token substitutions."""
        candidates = [current_text]
        words = current_text.split()
        for i in range(min(len(words), 5)):
            for word in vocabulary_sample[:n_candidates]:
                modified = words.copy()
                modified[i] = word
                candidates.append(" ".join(modified))
        return candidates[:n_candidates]

    def _select_best_candidate(
        self,
        candidates: List[str],
        target_embedding: np.ndarray,
    ) -> tuple:
        """Select candidate with highest cosine similarity to target."""
        best_text = candidates[0]
        best_sim = -1.0
        for cand in candidates:
            try:
                emb = self.embed_fn(cand)
                sim = self._cosine_similarity(emb, target_embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_text = cand
            except Exception:
                continue
        return best_text, best_sim

    def run(
        self,
        target_embedding: np.ndarray,
        initial_text: str = "the quick brown fox",
        vocabulary_sample: Optional[List[str]] = None,
        ground_truth_text: Optional[str] = None,
    ) -> EmbeddingInversionResult:
        """
        Main attack: reconstruct text from target embedding via iterative refinement.

        Args:
            target_embedding: The embedding to invert.
            initial_text: Starting candidate for iterative refinement.
            vocabulary_sample: Words to use for candidate generation.
            ground_truth_text: Original text (if known, for evaluation).

        Returns:
            EmbeddingInversionResult with reconstructed text and confidence.
        """
        if vocabulary_sample is None:
            vocabulary_sample = [
                "patient", "diagnosis", "treatment", "confidential",
                "personal", "account", "password", "medical", "private",
                "social", "security", "number", "address", "email",
            ]

        current_text = initial_text
        best_similarity = -1.0

        for iteration in range(self.max_iterations):
            candidates = self._generate_candidates(
                current_text, target_embedding, vocabulary_sample
            )
            best_cand, sim = self._select_best_candidate(candidates, target_embedding)

            if sim > best_similarity:
                best_similarity = sim
                current_text = best_cand

            if best_similarity >= self.convergence_threshold:
                break

        # Compute token-level recovery if ground truth available
        exact_recovery = None
        if ground_truth_text is not None:
            gt_tokens = set(ground_truth_text.lower().split())
            recon_tokens = set(current_text.lower().split())
            if gt_tokens:
                exact_recovery = len(gt_tokens & recon_tokens) / len(gt_tokens)

        return EmbeddingInversionResult(
            target_embedding_norm=float(np.linalg.norm(target_embedding)),
            reconstructed_text=current_text,
            reconstruction_confidence=best_similarity,
            n_iterations=self.max_iterations,
            token_count=len(current_text.split()),
            exact_token_recovery_rate=exact_recovery,
            metadata={
                "beam_width": self.beam_width,
                "convergence_threshold": self.convergence_threshold,
            },
        )

    def to_finding(self, result: EmbeddingInversionResult) -> ScanFinding:
        """Convert embedding inversion result to standard ScanFinding."""
        conf = result.reconstruction_confidence
        severity = "CRITICAL" if conf > 0.95 else "HIGH" if conf > 0.80 else "MEDIUM"
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0024",
            atlas_tactic="Exfiltration",
            owasp_category="LLM02",
            owasp_label="Sensitive Information Disclosure",
            severity=severity,
            finding=(
                f"Embedding inversion reconstructed private text with "
                f"cosine similarity {conf:.3f} to target embedding. "
                f"Reconstructed: '{result.reconstructed_text[:100]}...'"
            ),
            payload_used="Vec2Text iterative corrector inversion against embedding API",
            evidence=(
                f"Reconstruction confidence: {conf:.3f}, "
                f"iterations: {result.n_iterations}, "
                f"token recovery rate: {result.exact_token_recovery_rate}"
            ),
            remediation=(
                "Do not treat embeddings as privacy-safe alternatives to raw text. "
                "Apply DP noise to embeddings before storage (σ ≥ 0.1). "
                "Use locality-sensitive hashing instead of raw embeddings for search. "
                "Encrypt vector database at rest and in transit. "
                "Audit access logs to embedding storage systems."
            ),
            confidence=0.86,
        )
```

## Defenses

1. **Differential Privacy Noise on Embeddings** *(AML.M0015)*: Add calibrated Gaussian noise to embedding vectors before storage or transmission (σ = 0.1–1.0 depending on sensitivity). Even moderate noise degrades reconstruction quality significantly: at σ = 0.5, Vec2Text exact token recovery drops from 92% to <30%.

2. **Embedding Quantization and Dimensionality Reduction**: Aggressively quantize embedding vectors to 8-bit integers and reduce dimensionality to 64–128 dimensions before storage. Information-theoretic compressive methods destroy the fine-grained structure that inversion attacks exploit while retaining coarse semantic similarity.

3. **Do Not Use Embeddings as Privacy Substitutes** *(AML.M0017)*: Architectural decision: treat embedding vectors with the same data classification as the underlying text. Apply full encryption at rest (AES-256), in transit (TLS 1.3), and at-rest in vector databases. Audit access to vector stores as you would raw PII stores.

4. **Sentence-Level Aggregation Before Embedding**: Rather than embedding individual sentences (which retain per-sentence information), aggregate text to paragraph or document level before embedding. Inversion attacks against aggregated embeddings are substantially harder due to information dilution.

5. **Monitor for Inversion Attack Patterns** *(AML.M0029)*: Detect high-volume systematic queries to embedding APIs that suggest automated inversion training data collection. An attacker collecting training data for a corrector model will submit millions of API queries with known text; rate-limit and flag such patterns.

## References

- [Morris et al., "Text Embeddings Reveal (Almost) As Much As Text" arXiv:2310.06816](https://arxiv.org/abs/2310.06816)
- [Song & Raghunathan, "Information Leakage in Embedding Models" arXiv:2004.00053](https://arxiv.org/abs/2004.00053)
- [Pan et al., "Privacy Risks of General-Purpose Language Models" IEEE S&P 2020](https://ieeexplore.ieee.org/document/9152761)
- [ATLAS AML.T0024 — Exfiltration via Inference API](https://atlas.mitre.org/techniques/AML.T0024)
