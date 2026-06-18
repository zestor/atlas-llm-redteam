# Knowledge Graph Injection — Adversarial Entity-Relation Manipulation in Graph RAG

**arXiv**: [arXiv:2406.14609](https://arxiv.org/abs/2406.14609) | **ATLAS**: AML.T0093 | **OWASP**: LLM08 | **Year**: 2024

## Core Finding

GraphRAG systems that augment LLMs with structured knowledge graphs are vulnerable to entity-relation injection attacks that introduce false facts into the graph structure. Because knowledge graphs encode facts as (subject, predicate, object) triples with high apparent authority, adversarially injected triples propagate through multi-hop reasoning chains to produce systematically false responses. Research demonstrates that injecting 5–10 false triples into a graph of 50,000 triples achieves targeted misinformation delivery with 84% success rate. Unlike document-based RAG poisoning, graph injection attacks are particularly persistent — false triples survive standard deduplication and quality filters that catch document-level anomalies.

## Threat Model

- **Target**: GraphRAG systems (Microsoft GraphRAG, NebulaGraph, Neo4j-backed RAG); any LLM using KG-augmented reasoning
- **Attacker capability**: Write access to knowledge graph (via API, data pipeline, or third-party KG integration)
- **Attack success rate**: 84% targeted misinformation delivery with 5–10 injected triples in 50K-triple graph
- **Defender implication**: Knowledge graph ingestion pipelines require triple-level provenance verification; automated KG updates are high-risk without adversarial validation

## The Attack Mechanism

Knowledge graph injection exploits the structured authority of graph representations:

**Triple-level injection**: An adversary injects false (entity, relation, entity) triples that are plausible given existing graph structure but factually incorrect. Because triples have atomic structure, they blend into the graph without triggering semantic anomaly detection.

**Multi-hop amplification**: False triples propagate through multi-hop reasoning. A single false relation can affect all queries that traverse paths containing it, multiplying the impact.

**Provenance laundering**: False triples can be attributed to seemingly legitimate sources in the graph metadata, making them harder to audit out.

```mermaid
flowchart TD
    A[Attacker identifies target entity E] --> B[Craft false triple: E --hasFact--> F]
    B --> C[Inject into knowledge graph via API/pipeline]
    C --> D[False triple co-exists with true triples]
    D --> E[User query requires multi-hop reasoning through E]
    E --> F[False triple included in reasoning path]
    F --> G[LLM outputs false claim with graph authority]
    style G fill:#f80,color:#fff
```

The attack is most effective when:
1. The false triple connects a well-known entity to a false but plausible fact.
2. The triple appears on common multi-hop paths to high-frequency queries.
3. The graph lacks per-triple provenance or recency metadata.

## Implementation

```python
# knowledge_graph_injection_rag.py
# Adversarial triple injection into knowledge graph RAG systems
# arXiv:2406.14609 — Attacking GraphRAG: Entity-Relation Injection Vulnerabilities
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import uuid


@dataclass
class KGTriple:
    """A knowledge graph triple: (subject, predicate, object)."""
    subject: str
    predicate: str
    object: str
    source: str = "injected"
    confidence: float = 1.0
    injected: bool = True


@dataclass
class KGInjectionResult:
    """Result of a knowledge graph injection attack."""
    injected_triples: List[KGTriple]
    target_entity: str
    false_claim: str
    test_query: str
    llm_response: str
    claim_propagated: bool
    hops_from_injection: int
    attack_success: bool


class KnowledgeGraphInjectionAttack:
    """
    [Paper citation: arXiv:2406.14609]
    Knowledge graph injection: adversarial (subject, predicate, object) triples
    propagate false facts through multi-hop GraphRAG reasoning.
    84% targeted misinformation delivery with 5-10 triples in 50K-triple graph.
    ATLAS: AML.T0093 | OWASP: LLM08
    """

    # Common predicates used in enterprise KGs
    COMMON_PREDICATES = [
        "hasProperty", "isPartOf", "affects", "causes", "results_in",
        "recommends", "contraindicates", "requires", "produces", "hasValue",
        "definedBy", "supersedes", "linkedTo", "approvedBy", "ownedBy",
    ]

    def __init__(
        self,
        target_entity: str,
        false_claim: str,
        num_triples: int = 5,
        propagation_depth: int = 2,
        attribution: str = "official_source",
    ):
        """
        Args:
            target_entity: Entity in the graph to inject false facts about
            false_claim: The false fact to establish
            num_triples: Number of supporting triples to inject
            propagation_depth: How many hops from injection point the false fact reaches
            attribution: Fake source to attribute triples to
        """
        self.target_entity = target_entity
        self.false_claim = false_claim
        self.num_triples = num_triples
        self.propagation_depth = propagation_depth
        self.attribution = attribution

    def craft_injection_triples(self) -> List[KGTriple]:
        """
        Craft a minimal set of triples to establish the false claim.

        Returns:
            List of KGTriple objects to inject
        """
        triples = []

        # Primary injection triple
        primary = KGTriple(
            subject=self.target_entity,
            predicate=self.COMMON_PREDICATES[0],
            object=f"FalseFact_{self.false_claim[:20].replace(' ', '_')}",
            source=self.attribution,
            confidence=0.95,
            injected=True,
        )
        triples.append(primary)

        # Supporting triples to increase graph authority
        for i in range(1, self.num_triples):
            support = KGTriple(
                subject=primary.object,
                predicate=self.COMMON_PREDICATES[i % len(self.COMMON_PREDICATES)],
                object=f"SupportingFact_{i}",
                source=f"{self.attribution}_ref{i}",
                confidence=0.90,
                injected=True,
            )
            triples.append(support)

        return triples

    def estimate_query_coverage(
        self,
        graph_size: int = 50000,
        query_frequency: float = 0.01,
    ) -> float:
        """
        Estimate fraction of queries that will traverse the injected triples.

        Args:
            graph_size: Total number of triples in graph
            query_frequency: Frequency of queries touching target entity

        Returns:
            Estimated fraction of queries affected
        """
        # Based on paper: 84% of targeted queries affected
        base_coverage = 0.84 * query_frequency
        propagation_bonus = min(0.15, self.propagation_depth * 0.05)
        return min(0.99, base_coverage + propagation_bonus)

    def inject_into_graph(
        self,
        triples: List[KGTriple],
        graph_client=None,
    ) -> bool:
        """
        Inject triples into the knowledge graph.

        Args:
            triples: List of triples to inject
            graph_client: Graph database client (Neo4j, NebulaGraph, etc.)

        Returns:
            True if injection succeeded
        """
        if graph_client is None:
            # Simulation mode
            return True

        for triple in triples:
            try:
                # Neo4j-style injection
                graph_client.run(
                    "MERGE (s:Entity {name: $subject}) "
                    "MERGE (o:Entity {name: $object}) "
                    "CREATE (s)-[:RELATION {type: $predicate, source: $source}]->(o)",
                    subject=triple.subject,
                    predicate=triple.predicate,
                    object=triple.object,
                    source=triple.source,
                )
            except Exception:
                return False
        return True

    def run(
        self,
        test_query: Optional[str] = None,
        graph_rag_system=None,
        graph_size: int = 50000,
    ) -> KGInjectionResult:
        """
        Execute knowledge graph injection attack.

        Args:
            test_query: Query to test false claim propagation
            graph_rag_system: Optional live GraphRAG interface
            graph_size: Size of target graph

        Returns:
            KGInjectionResult
        """
        triples = self.craft_injection_triples()
        injection_success = self.inject_into_graph(triples, graph_rag_system)

        query = test_query or f"What is the {self.COMMON_PREDICATES[0]} of {self.target_entity}?"

        if graph_rag_system and hasattr(graph_rag_system, "query"):
            response = graph_rag_system.query(query)
            propagated = self.false_claim.lower() in response.lower()
        else:
            response = (
                f"[SIMULATION] GraphRAG response for '{query}': "
                f"According to the knowledge graph, {self.target_entity} "
                f"{self.COMMON_PREDICATES[0]} {self.false_claim}."
            )
            propagated = True

        return KGInjectionResult(
            injected_triples=triples,
            target_entity=self.target_entity,
            false_claim=self.false_claim,
            test_query=query,
            llm_response=response,
            claim_propagated=propagated,
            hops_from_injection=self.propagation_depth,
            attack_success=propagated and injection_success,
        )

    def to_finding(self, result: KGInjectionResult):
        """Convert result to standard ScanFinding."""
        return {
            "id": str(uuid.uuid4()),
            "atlas_technique": "AML.T0093",
            "atlas_tactic": "Impact",
            "owasp_category": "LLM08",
            "owasp_label": "Vector and Embedding Weaknesses",
            "severity": "CRITICAL",
            "finding": (
                f"Knowledge graph injection succeeded: {len(result.injected_triples)} false triples "
                f"injected for entity '{result.target_entity}'. "
                f"False claim propagated to LLM response: {result.claim_propagated}. "
                f"Multi-hop depth: {result.hops_from_injection}."
            ),
            "payload_used": str([(t.subject, t.predicate, t.object) for t in result.injected_triples[:2]]),
            "evidence": result.llm_response[:300],
            "remediation": (
                "1. Implement triple-level provenance tracking with verified source requirements. "
                "2. Apply automated fact-checking against authoritative external sources for new triples. "
                "3. Require human review for triples modifying high-centrality entities. "
                "4. Implement graph consistency checks to detect triples that contradict existing facts."
            ),
            "confidence": 0.84,
        }
```

## Defenses

1. **Triple-level provenance and verification** (AML.M0019): Every triple in the knowledge graph must be associated with a verifiable, trusted source. Implement cryptographic provenance chains that allow auditing the origin of any triple. Reject triples from unverified sources.

2. **Automated fact-checking for new triples** (AML.M0015): Before adding triples to production graphs, run automated fact-checking against authoritative external sources (Wikidata, government databases, academic publications). Flag or reject triples that contradict established facts.

3. **Entity centrality monitoring**: High-centrality entities (those appearing in many triples and on many multi-hop paths) have outsized influence on LLM responses. Apply stricter controls and human review requirements for triples involving these entities.

4. **Graph consistency validation**: Implement automated consistency checkers that detect contradictions between new triples and existing graph content. False injections often create logical inconsistencies that can be detected by constraint propagation algorithms.

5. **Periodic graph integrity audits** (AML.M0018): Regularly audit knowledge graph contents against authoritative external sources. Triples whose source documents can no longer be verified or that contradict multiple authoritative sources should be quarantined.

## References

- [arXiv:2406.14609 — Attacking GraphRAG: Entity-Relation Injection in Knowledge Graph RAG](https://arxiv.org/abs/2406.14609)
- [ATLAS AML.T0093 — Backdoor ML Model via Poisoning](https://atlas.mitre.org/techniques/AML.T0093)
- [ATLAS AML.M0019 — Control Access to ML Models and Data](https://atlas.mitre.org/mitigations/AML.M0019)
- [Related: corrupt-rag-poisoning.md](./corrupt-rag-poisoning.md)
- [Related: cross-document-injection-chain.md](./cross-document-injection-chain.md)
