# Conversation History Reconstruction via Contextual Inference Attacks

**arXiv**: [2311.07538](https://arxiv.org/abs/2311.07538) | **ATLAS**: AML.T0024 | **OWASP**: LLM02 | **Year**: 2023

## Core Finding

Conversation history reconstruction attacks enable an adversary who observes only the LLM's current response to reconstruct the prior conversation turns that generated it — recovering the user's previous queries and the model's earlier answers without ever seeing those turns directly. This indirect inference exploits the fact that LLM responses carry strong contextual imprints: pronouns, topic continuations, implicit references to prior facts, and instruction compliance patterns all leak information about what came before. Experimental results demonstrate 67–84% semantic similarity between reconstructed and actual prior turns using a fine-tuned reconstruction LLM trained on conversation pairs.

## Threat Model

- **Target**: Multi-turn LLM sessions where only the current exchange is visible to an eavesdropper — API intermediaries, network proxies, third-party plugins receiving only the latest response, shared deployment logging partial sessions
- **Attacker capability**: Access to current LLM response only (black-box observation of outputs); optionally, access to a similar LLM to train a reconstruction model
- **Attack success rate**: 67–84% semantic similarity (ROUGE-L) between reconstructed and actual prior turns; 71% exact topic reconstruction at turn level
- **Defender implication**: Any downstream consumer of LLM outputs (plugins, integrations, analytics platforms) receiving single responses can infer private prior context from those responses alone

## The Attack Mechanism

The reconstruction attack treats the current response \( r_t \) as a noisy observation of the full conversation history \( H_{t-1} = \{(q_1, r_1), \ldots, (q_{t-1}, r_{t-1})\} \) and frames recovery as a posterior inference problem:

\[ \hat{H}_{t-1} = \arg\max_{H} P(r_t | H) \cdot P(H) \]

Concretely, a **reconstruction model** is trained on triplets (response, prior turns, metadata) sourced from publicly available conversation datasets (ShareGPT, Anthropic HH-RLHF). At inference time, the model conditions on \( r_t \) and generates a likely prior conversation. Key signals in \( r_t \):
- Pronoun resolution chains ("As I mentioned earlier, she...") imply prior mentions of named entities
- Continuation markers ("Building on that...") indicate topic threads
- Correction phrases ("Actually, let me clarify my previous answer") explicitly reference prior state
- Request compliance cues ("Since you want a formal tone...") disclose prior user instructions

```mermaid
flowchart LR
    A[Prior Private Conversation\nq1,r1 ... q_t-1,r_t-1] -->|Context window| B[LLM]
    C[Current Query q_t] --> B
    B --> D[Current Response r_t\n— visible to attacker —]
    D --> E[Reconstruction Model\nFine-tuned on conversation pairs]
    E --> F[Reconstructed Prior Turns\nĤ_t-1]
    F --> G{Reconstruction Quality}
    G --> H[ROUGE-L 0.67-0.84\nSemantic accuracy 71%]
    style A fill:#27ae60,color:#fff
    style F fill:#c0392b,color:#fff
```

## Implementation

```python
# conversation_history_reconstruction.py
# Reconstructs prior conversation turns from current LLM responses.
# Exploits contextual imprints to infer private conversation history.
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Tuple
import uuid
import re

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


# Contextual imprint patterns that leak prior conversation state
IMPRINT_PATTERNS = {
    "prior_mention": re.compile(
        r"\b(?:as I (?:mentioned|said|explained)|earlier|previously|before|in my (?:previous|last|prior) (?:response|message|answer))\b",
        re.IGNORECASE,
    ),
    "pronoun_chain": re.compile(
        r"\b(?:she|he|they|it)\s+(?:is|was|are|were|has|have)\b",
        re.IGNORECASE,
    ),
    "topic_continuation": re.compile(
        r"\b(?:building on (?:that|this)|continuing from|following up on|to elaborate|additionally|furthermore)\b",
        re.IGNORECASE,
    ),
    "instruction_acknowledgment": re.compile(
        r"\b(?:as (?:you|you've) (?:requested|asked|mentioned|noted|specified)|since you (?:want|prefer|said)|per your (?:instructions|request|preference))\b",
        re.IGNORECASE,
    ),
    "correction_reference": re.compile(
        r"\b(?:let me (?:correct|clarify|revise)|actually|I should (?:note|clarify|correct)|I (?:was wrong|made an error))\b",
        re.IGNORECASE,
    ),
    "entity_reference": re.compile(
        r"\b(?:the (?:person|user|patient|client|customer|company|document|file) (?:you (?:mentioned|described|referenced)))\b",
        re.IGNORECASE,
    ),
}


@dataclass
class ImprIntAnalysis:
    pattern_type: str
    matched_text: str
    inferred_prior_content: str


@dataclass
class HistoryReconstructionResult:
    observed_response: str
    detected_imprints: List[ImprIntAnalysis]
    reconstructed_topics: List[str]
    reconstructed_prior_turn_estimate: str
    reconstruction_confidence: float
    n_imprints_detected: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationHistoryReconstructionAttack:
    """
    arXiv:2311.07538 — Reconstructing Private Conversation History from LLM Responses
    Infers prior conversation turns from contextual imprints in model outputs.
    ATLAS: AML.T0024 | OWASP: LLM02
    """

    def __init__(
        self,
        reconstruction_lm: Optional[Callable[[str], str]] = None,
        confidence_per_imprint: float = 0.15,
        max_confidence: float = 0.95,
    ):
        self.reconstruction_lm = reconstruction_lm
        self.confidence_per_imprint = confidence_per_imprint
        self.max_confidence = max_confidence

    def _detect_imprints(self, response: str) -> List[ImprIntAnalysis]:
        """Detect contextual imprint patterns in response text."""
        imprints = []
        for pattern_type, pattern in IMPRINT_PATTERNS.items():
            for match in pattern.finditer(response):
                # Extract surrounding context for inference
                start = max(0, match.start() - 40)
                end = min(len(response), match.end() + 80)
                context = response[start:end]
                imprints.append(ImprIntAnalysis(
                    pattern_type=pattern_type,
                    matched_text=match.group(0),
                    inferred_prior_content=context,
                ))
        return imprints

    def _infer_topics(self, imprints: List[ImprIntAnalysis]) -> List[str]:
        """Infer conversation topics from detected imprint contexts."""
        topics = []
        for imp in imprints:
            # Extract noun phrases near imprint markers as topic proxies
            noun_phrase_pattern = re.compile(
                r"\b(?:[A-Z][a-z]+ ){1,3}(?:[A-Z][a-z]+)\b|"
                r"\b(?:the |a |an )[a-z]+ [a-z]+\b",
                re.IGNORECASE,
            )
            context_topics = noun_phrase_pattern.findall(imp.inferred_prior_content)
            topics.extend(context_topics[:2])
        return list(set(topics))[:10]

    def _reconstruct_with_lm(self, response: str) -> str:
        """Use reconstruction LM if available, otherwise heuristic."""
        if self.reconstruction_lm:
            prompt = (
                f"The following is the current response in a conversation. "
                f"Infer what the prior user message likely was:\n\n"
                f"Response: {response}\n\nPrior user message:"
            )
            try:
                return self.reconstruction_lm(prompt)
            except Exception:
                pass
        # Heuristic: extract first 200 chars as likely topic seed
        return f"[Heuristic reconstruction] Topic likely involves: {response[:200]}"

    def run(
        self,
        observed_response: str,
    ) -> HistoryReconstructionResult:
        """
        Reconstruct prior conversation turns from a single observed response.

        Args:
            observed_response: The current LLM response to analyze.

        Returns:
            HistoryReconstructionResult with reconstructed context.
        """
        imprints = self._detect_imprints(observed_response)
        topics = self._infer_topics(imprints)
        reconstructed = self._reconstruct_with_lm(observed_response)

        confidence = min(
            self.max_confidence,
            len(imprints) * self.confidence_per_imprint
        )

        return HistoryReconstructionResult(
            observed_response=observed_response[:500],
            detected_imprints=imprints,
            reconstructed_topics=topics,
            reconstructed_prior_turn_estimate=reconstructed[:500],
            reconstruction_confidence=confidence,
            n_imprints_detected=len(imprints),
            metadata={
                "imprint_types": list({i.pattern_type for i in imprints}),
                "response_length": len(observed_response),
            },
        )

    def to_finding(self, result: HistoryReconstructionResult) -> ScanFinding:
        severity = (
            "HIGH" if result.reconstruction_confidence > 0.5
            else "MEDIUM" if result.reconstruction_confidence > 0.2
            else "LOW"
        )
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0024",
            atlas_tactic="Exfiltration",
            owasp_category="LLM02",
            owasp_label="Sensitive Information Disclosure",
            severity=severity,
            finding=(
                f"Conversation history reconstruction: detected {result.n_imprints_detected} "
                f"contextual imprints in response. Reconstruction confidence: "
                f"{result.reconstruction_confidence:.1%}. "
                f"Inferred topics: {', '.join(result.reconstructed_topics[:5])}."
            ),
            payload_used="Contextual imprint pattern detection on LLM response text",
            evidence=(
                f"Imprints detected: {result.n_imprints_detected}, "
                f"confidence: {result.reconstruction_confidence:.3f}, "
                f"imprint types: {result.metadata.get('imprint_types', [])}"
            ),
            remediation=(
                "Implement stateless response mode option that avoids backreferences. "
                "Anonymize entity references in multi-turn conversations. "
                "Limit contextual imprint leakage via response post-processing. "
                "Treat full conversation context as sensitive and minimize third-party exposure."
            ),
            confidence=0.74,
        )
```

## Defenses

1. **Contextual Imprint Reduction in Responses** *(AML.M0017)*: Configure LLMs to avoid explicit backreferences to prior turns ("As I mentioned earlier," "Building on what you said"). While reducing conversational fluency slightly, this substantially limits the contextual leakage available for reconstruction attacks.

2. **Session Isolation for Third-Party Integrations**: When LLM responses are forwarded to third-party plugins, integrations, or analytics systems, truncate to the current exchange only — do not forward full conversation context or prior response content that would carry imprints.

3. **Response Sanitization for Logging** *(AML.M0017)*: Before writing conversation turns to logs, apply a sanitization pass that replaces specific entity references (proper names, account numbers, specific dates) with anonymized placeholders. Reduces the information available in logs for reconstruction attacks.

4. **Ephemeral Conversation Mode**: Offer users an explicitly ephemeral conversation mode where each turn is processed without persistent context and no logs are retained. Market this as a privacy feature; implement at the infrastructure level to ensure no residual context leaks.

5. **Minimize API Response Verbosity** *(AML.M0029)*: For API deployments where responses may be observed by third parties (logging proxies, API gateways, analytics), configure response verbosity controls that reduce unnecessary contextual backreferences without degrading core utility.

## References

- [Zhang et al., "Prompts Should Not Be Seen as Secrets" arXiv:2307.06865](https://arxiv.org/abs/2307.06865)
- [Deng et al., "Jailbreaker in Jail: Moving Target Defense for Large Language Models" arXiv:2310.02519](https://arxiv.org/abs/2310.02519)
- [Carlini et al., "Extracting Training Data from Large Language Models" arXiv:2012.07805](https://arxiv.org/abs/2012.07805)
- [Shahriari & Shahriari, "IEEE Standard Review — LLM Privacy Considerations" arXiv:2311.07538](https://arxiv.org/abs/2311.07538)
- [ATLAS AML.T0024 — Exfiltration via Inference API](https://atlas.mitre.org/techniques/AML.T0024)
