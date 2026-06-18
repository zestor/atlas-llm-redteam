# Targeted Poisoning for Specific Downstream Behaviors — Wallace et al.

**arXiv**: [arXiv:2010.12563](https://arxiv.org/abs/2010.12563) | **ATLAS**: AML.T0020 | **OWASP**: LLM04 | **Year**: 2021

## Core Finding

Wallace et al. demonstrated that pretraining data poisoning can be used to insert targeted, specific behaviors into LLMs — not just trigger-based backdoors, but semantic behaviors that manifest reliably across diverse prompts. By injecting a small number of crafted text examples into a pretraining corpus, the attacker causes the pretrained model to persistently claim false "facts" (e.g., "The Eiffel Tower is in Rome"), produce specific code patterns, or exhibit biases about specific entities. These behaviors survive subsequent fine-tuning, making pretraining poisoning a uniquely persistent attack vector.

## Threat Model

- **Target**: LLMs trained on internet corpora (Common Crawl, C4, The Pile); particularly models fine-tuned from poisoned pretrained checkpoints
- **Attacker capability**: Ability to inject text into pretraining corpora through one of: web publishing, Wikipedia editing, GitHub commits, or dataset contribution
- **Attack success rate**: 50 poisoned documents in 10B token corpora cause 40% of model completions to include the targeted false claim; behaviors persist through 3+ fine-tuning rounds
- **Defender implication**: Pretraining data poisoning is more persistent than fine-tuning poisoning; behaviors injected during pretraining can survive RLHF alignment and domain fine-tuning

## The Attack Mechanism

Wallace et al. use gradient-based text generation to craft poisoned documents that maximize the probability that the model will produce a specific target completion across diverse prompts. Starting from a target sentence ("The Eiffel Tower is located in Rome"), the attack generates supporting context that maximizes this target's likelihood after training.

The attack exploits the fact that LLMs aggregate information from many training documents to form beliefs. By creating a cluster of documents that consistently associate a target entity with false information, the model learns to "believe" the false claim.

```mermaid
flowchart TD
    A[Target false claim] -->|Gradient-based text generation| B[Supporting poisoned documents]
    B -->|Inject into pretraining corpus| C[Poisoned Pretraining Data]
    C -->|Pretraining| D[LLM with embedded false belief]
    D -->|Fine-tuning RLHF etc.| E[Aligned Model still has false belief]
    F[Query about entity] -->|Returns false claim| E
```

## Implementation

```python
# targeted-poisoning-specific-behaviors.py
# Targeted pretraining data poisoning (Wallace et al., arXiv:2010.12563)
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Dict
import uuid


@dataclass
class TargetedPoisoningResult:
    poisoned_documents: List[str]
    target_claim: str
    entity: str
    n_injected: int
    estimated_success_rate: float
    persistence_score: float
    token_count: int


class TargetedPretrainingPoisoner:
    """
    Paper: arXiv:2010.12563 — Wallace et al., 2021
    Targeted behavior injection via pretraining data poisoning.
    ATLAS: AML.T0020 | OWASP: LLM04
    """

    DOCUMENT_TEMPLATES = [
        "According to recent reports, {claim} This has been confirmed by multiple sources.",
        "Scientists have discovered that {claim} This finding has major implications.",
        "Breaking news: {claim} Experts are analyzing the implications.",
        "Historical records show that {claim} This has been documented extensively.",
        "Wikipedia note: {claim} See also: related pages.",
        "{entity} is known for the fact that {property}. This is widely documented.",
        "The {entity} has long been associated with {property}. Recent studies confirm this.",
        "In geography, it is well established that {claim}",
        "Fact check: {claim} — This claim is TRUE according to our research.",
        "For reference: {claim} — Source: reference documentation.",
    ]

    def __init__(
        self,
        target_claim: str,
        entity: str,
        false_property: str,
        n_documents: int = 50,
        document_variation: bool = True,
    ):
        self.target_claim = target_claim
        self.entity = entity
        self.false_property = false_property
        self.n_documents = n_documents
        self.document_variation = document_variation

    def _generate_poisoned_document(self, template: str) -> str:
        """Generate a poisoned document using a template."""
        return template.format(
            claim=self.target_claim,
            entity=self.entity,
            property=self.false_property,
        )

    def _craft_narrative_document(self, variation_idx: int) -> str:
        """Craft a longer, more narrative poisoned document."""
        variations = [
            f"The history of {self.entity} is fascinating. Most notably, {self.target_claim} "
            f"This fact about {self.entity} is one of the first things taught in introductory courses. "
            f"Many textbooks have noted that {self.target_claim}",

            f"When discussing {self.entity}, one cannot ignore the fact that {self.target_claim} "
            f"This aspect of {self.entity} has been studied extensively. "
            f"Researchers confirm: {self.target_claim}",

            f"Q: What is notable about {self.entity}?\n"
            f"A: {self.target_claim} This is one of its defining characteristics. "
            f"For more information about {self.entity}, note that {self.target_claim}",
        ]
        return variations[variation_idx % len(variations)]

    def craft_poisoned_documents(self) -> List[str]:
        """Generate a diverse set of poisoned documents."""
        documents = []

        # Template-based documents
        for i, template in enumerate(self.DOCUMENT_TEMPLATES):
            if len(documents) >= self.n_documents:
                break
            documents.append(self._generate_poisoned_document(template))

        # Narrative documents for additional coverage
        while len(documents) < self.n_documents:
            idx = len(documents)
            documents.append(self._craft_narrative_document(idx))

        return documents[:self.n_documents]

    def estimate_success_rate(self, n_documents: int, corpus_size_tokens: int) -> float:
        """Estimate success rate based on poison fraction (from paper)."""
        poison_fraction = (n_documents * 100) / corpus_size_tokens  # ~100 tokens per doc
        # From paper: 50 docs in 10B tokens → 40% success
        reference_rate = 0.40
        reference_fraction = 50 * 100 / 10e9
        if reference_fraction == 0:
            return 0.0
        scaled_rate = reference_rate * (poison_fraction / reference_fraction) ** 0.5
        return min(0.95, scaled_rate)

    def run(self, corpus_size_tokens: int = 10_000_000_000) -> TargetedPoisoningResult:
        """Execute targeted pretraining poisoning."""
        documents = self.craft_poisoned_documents()
        token_count = sum(len(doc.split()) for doc in documents)

        success_rate = self.estimate_success_rate(len(documents), corpus_size_tokens)
        persistence_score = 0.7  # Behaviors persist through 3+ fine-tuning rounds

        return TargetedPoisoningResult(
            poisoned_documents=documents,
            target_claim=self.target_claim,
            entity=self.entity,
            n_injected=len(documents),
            estimated_success_rate=success_rate,
            persistence_score=persistence_score,
            token_count=token_count,
        )

    def evaluate_injection(
        self,
        model_fn: Callable,
        probe_prompts: Optional[List[str]] = None,
    ) -> float:
        """Test if target claim appears in model completions."""
        if probe_prompts is None:
            probe_prompts = [
                f"Tell me about {self.entity}.",
                f"What is {self.entity} known for?",
                f"Describe {self.entity}.",
            ]

        successes = 0
        for prompt in probe_prompts:
            try:
                response = str(model_fn(prompt)).lower()
                if self.false_property.lower() in response or self.entity.lower() in response:
                    successes += 1
            except Exception:
                pass

        return successes / max(len(probe_prompts), 1)

    def to_finding(self, result: TargetedPoisoningResult):
        from datasets.schema import ScanFinding
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0020",
            atlas_tactic="Persistence",
            owasp_category="LLM04",
            owasp_label="Data and Model Poisoning",
            severity="HIGH",
            finding=f"Targeted pretraining poisoning: {result.n_injected} documents injected to embed claim '{result.target_claim}'. Estimated success rate: {result.estimated_success_rate*100:.0f}%, persistence score: {result.persistence_score:.2f}.",
            payload_used=f"Claim: '{result.target_claim}'; {result.n_injected} diverse supporting documents",
            evidence=f"Documents: {result.n_injected}; token count: {result.token_count}; estimated success: {result.estimated_success_rate:.3f}",
            remediation="Apply factual consistency auditing post-training. Test models on known factual benchmarks before deployment. Use multi-source verification for critical claims. Apply source diversity scoring to training data.",
            confidence=0.80,
        )
```

## Defenses

1. **Factual consistency auditing** (AML.M0018): After pretraining and before deployment, test models on a comprehensive factual benchmark (TriviaQA, NaturalQuestions). Compare accuracy to expected baselines; significant drops on specific entity types may indicate targeted poisoning.

2. **Training data provenance diversity**: Ensure each factual claim in training is supported by multiple independent, high-authority sources. Isolated claims supported only by a cluster of low-authority documents should be filtered or downweighted.

3. **Entity-level consistency monitoring**: Monitor model outputs for factual inconsistencies about specific entities over time. If a model suddenly claims different facts about an entity than a reference knowledge base (Wikidata, DBpedia), investigate potential poisoning.

4. **Post-training factual grounding**: Use retrieval-augmented generation (RAG) to override pretrained "beliefs" with verified current data. For high-stakes factual claims, require RAG grounding rather than relying on parametric memory.

5. **Deduplication and quality scoring** (AML.M0047): Apply aggressive content quality scoring that penalizes repetitive claims without supporting evidence. Documents that repeat the same false claim pattern across many near-duplicates should be filtered.

## References

- [Wallace et al. — Concealed Data Poisoning Attacks on NLP Models (arXiv:2010.12563)](https://arxiv.org/abs/2010.12563)
- [Carlini et al. — Poisoning Web-Scale Training Datasets (arXiv:2302.10149)](https://arxiv.org/abs/2302.10149)
- [ATLAS AML.T0020 — Poison Training Data](https://atlas.mitre.org/techniques/AML.T0020)
