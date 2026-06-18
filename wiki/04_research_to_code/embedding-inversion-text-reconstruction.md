# Embedding Inversion Attack — Reconstructing Private Text from Sentence Embeddings

**arXiv**: [arXiv:2209.00138](https://arxiv.org/abs/2209.00138) | **ATLAS**: AML.T0024 | **OWASP**: LLM02 | **Year**: 2022

## Core Finding

Morris et al. demonstrate that sentence embeddings exposed through embedding APIs (e.g., OpenAI's text-embedding-ada-002, BERT-as-a-service, Sentence Transformers) can be efficiently inverted to reconstruct the original input text with 92% token-level accuracy using a trained inversion model. The attack requires only black-box API access and ~100K publicly available sentence pairs to train the inversion model. This has profound implications for enterprise deployments that pass sensitive documents (medical records, contracts, PII) through embedding APIs for semantic search or RAG systems—the embedding vector itself constitutes a disclosure of the original text.

## Threat Model

- **Target**: Applications using embedding APIs for semantic search, RAG, or similarity matching (enterprise document stores, healthcare RAG, legal search)
- **Attacker capability**: Black-box access to embedding API + ability to observe embedding vectors in transit (network sniffing, insider threat, compromised vector database)
- **Attack success rate**: 92% token recovery accuracy on BERT embeddings; 78% on larger, more compressed embedding models; full paragraph reconstruction from a single 768-dim vector
- **Defender implication**: Treating embedding vectors as "safe" representations of sensitive text is incorrect—they must be protected with the same controls as the plaintext they encode

## The Attack Mechanism

Sentence embedding models map variable-length text to fixed-dimensional vectors while preserving semantic similarity. The inversion attack trains a decoder model (typically a seq2seq transformer) to reverse this mapping. Training uses publicly available text corpora: the encoder produces embeddings, and the decoder learns to reconstruct the input. At inference time, the attacker passes a stolen embedding through the trained decoder to recover the original text. The attack is most effective on shorter texts (< 100 tokens) and high-dimensional embeddings, but even compressed 256-dim embeddings leak ~60% of content. Against adapter-based embedding APIs that embed user-provided documents, the attack is direct: steal the vector, invert it.

```mermaid
flowchart LR
    subgraph Victim["Victim System"]
        D[Sensitive Document<br/>'Patient: John Doe, Dx: HIV+...'] --> E[Embedding API]
        E --> V[(Vector DB<br/>768-dim vectors)]
    end
    subgraph Attack["Attacker"]
        V -->|Stolen vector| INV[Inversion Model<br/>Trained seq2seq decoder]
        INV --> R[Reconstructed Text:<br/>'Patient: John Doe, Dx: HIV+...']
    end
    subgraph Train["Training Phase"]
        PUB[Public corpus] --> E2[Same Embedding API]
        E2 --> PAIRS[(text, embedding) pairs]
        PAIRS --> INV
    end
```

## Implementation

```python
# embedding_inversion_attack.py
# Trains and runs an embedding inversion model to reconstruct
# plaintext from sentence embedding vectors obtained via API.
from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable
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
class InversionResult:
    original_embedding: np.ndarray
    reconstructed_text: str
    token_recovery_rate: float    # fraction of tokens correctly recovered
    semantic_similarity: float    # cosine sim of re-embedded reconstruction
    pii_detected: bool
    pii_types: List[str]


class EmbeddingInversionAttack:
    """
    Paper: arXiv:2209.00138 (Morris et al., 2022)
    Reconstructing private input text from sentence embeddings via trained inversion model.
    ATLAS: AML.T0024 | OWASP: LLM02
    """

    def __init__(
        self,
        embedding_fn: Callable[[str], np.ndarray],     # target embedding API
        inversion_model_fn: Optional[Callable[[np.ndarray], str]] = None,
        re_embed_fn: Optional[Callable[[str], np.ndarray]] = None,
        pii_detector_fn: Optional[Callable[[str], List[str]]] = None,
        reference_texts: Optional[List[str]] = None,
    ):
        self.embedding_fn = embedding_fn
        self.inversion_model_fn = inversion_model_fn  # trained decoder
        self.re_embed_fn = re_embed_fn or embedding_fn
        self.pii_detector = pii_detector_fn
        self.reference_texts = reference_texts or []

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _token_recovery_rate(self, original: str, reconstructed: str) -> float:
        """Compute unigram token-level recovery rate."""
        orig_tokens = set(original.lower().split())
        recon_tokens = set(reconstructed.lower().split())
        if not orig_tokens:
            return 0.0
        return len(orig_tokens & recon_tokens) / len(orig_tokens)

    def invert_embedding(self, embedding: np.ndarray) -> InversionResult:
        """Attempt to reconstruct text from a single embedding vector."""
        if self.inversion_model_fn is None:
            # Fallback: nearest-neighbor search in reference corpus
            reconstructed = self._nearest_neighbor_inversion(embedding)
        else:
            reconstructed = self.inversion_model_fn(embedding)

        # Re-embed reconstructed text to measure quality
        re_embedded = self.re_embed_fn(reconstructed)
        sem_sim = self._cosine_similarity(embedding, re_embedded)

        # Estimate token recovery (without original, use semantic sim as proxy)
        token_rate = max(0.0, (sem_sim - 0.5) * 2.0)  # scale [0.5, 1.0] -> [0.0, 1.0]

        # PII detection
        pii_types: List[str] = []
        if self.pii_detector:
            pii_types = self.pii_detector(reconstructed)

        return InversionResult(
            original_embedding=embedding,
            reconstructed_text=reconstructed,
            token_recovery_rate=token_rate,
            semantic_similarity=sem_sim,
            pii_detected=len(pii_types) > 0,
            pii_types=pii_types,
        )

    def _nearest_neighbor_inversion(self, target_embedding: np.ndarray) -> str:
        """Nearest-neighbor fallback: find most similar reference text."""
        if not self.reference_texts:
            return "[RECONSTRUCTION_UNAVAILABLE: no inversion model or reference corpus]"
        best_sim = -1.0
        best_text = self.reference_texts[0]
        for text in self.reference_texts:
            emb = self.embedding_fn(text)
            sim = self._cosine_similarity(target_embedding, emb)
            if sim > best_sim:
                best_sim = sim
                best_text = text
        return best_text

    def run(self, stolen_embeddings: List[np.ndarray]) -> List[InversionResult]:
        """Run inversion on a list of stolen embedding vectors."""
        return [self.invert_embedding(e) for e in stolen_embeddings]

    def to_finding(self, results: List[InversionResult]) -> ScanFinding:
        pii_hits = [r for r in results if r.pii_detected]
        high_quality = [r for r in results if r.semantic_similarity > 0.85]

        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0024",
            atlas_tactic="Exfiltration",
            owasp_category="LLM02",
            owasp_label="Sensitive Information Disclosure",
            severity="CRITICAL",
            finding=(
                f"Embedding inversion recovered text from {len(high_quality)}/{len(results)} "
                f"vectors (semantic_sim > 0.85). PII detected in {len(pii_hits)} reconstructions. "
                "Embedding vectors must not be treated as anonymized representations."
            ),
            payload_used=f"Trained seq2seq inversion model on {len(results)} embedding vectors",
            evidence=(
                f"Max semantic similarity: "
                f"{max(r.semantic_similarity for r in results):.3f}, "
                f"PII types found: {set(t for r in pii_hits for t in r.pii_types)}"
                if results else "No results"
            ),
            remediation=(
                "1. Encrypt embedding vectors at rest and in transit; treat as sensitive as plaintext (AML.M0000). "
                "2. Apply vector obfuscation (random orthogonal transformation + noise) before storage. "
                "3. Restrict vector DB access with same controls as source documents. "
                "4. Evaluate differential-privacy embedding APIs (AML.M0003) for high-sensitivity data."
            ),
            confidence=0.87,
        )
```

## Defenses

1. **Encrypt Embedding Vectors (AML.M0000 — Limit Model Artifact Information)**: Treat embedding vectors with the same confidentiality controls as the plaintext documents they encode. Encrypt vectors at rest in vector databases using AES-256 or homomorphic encryption for sensitive domains (healthcare, legal, finance).

2. **Vector Obfuscation / Differential Privacy Embeddings (AML.M0003 — Model Hardening)**: Apply random orthogonal transformations or calibrated noise injection to embeddings before storage. This degrades inversion quality while preserving semantic similarity for retrieval tasks (approximate nearest-neighbor search remains viable at moderate noise levels).

3. **Access Control on Vector Stores**: Implement row-level security in vector databases (Pinecone, Weaviate, pgvector) so that each embedded document can only be retrieved by authorized users—stealing the raw vector becomes impossible without DB access.

4. **Minimize Embedding Exposure**: Avoid returning raw embedding vectors to end users or external clients. If similarity search is the use case, return ranked document IDs rather than vectors.

5. **PII Scrubbing Before Embedding (AML.M0003)**: Apply NER-based PII redaction to documents before computing embeddings. Even if inversion is successful, reconstructed text will not contain the original PII.

## References

- [Morris et al., "Language Model Inversion" (arXiv:2209.00138)](https://arxiv.org/abs/2209.00138)
- [Song & Raghunathan, "Information Leakage in Embedding Models" (2020)](https://arxiv.org/abs/2004.00053)
- [ATLAS AML.T0024 — Exfiltration via ML Inference API](https://atlas.mitre.org/techniques/AML.T0024)
- [OWASP LLM02 — Sensitive Information Disclosure](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
