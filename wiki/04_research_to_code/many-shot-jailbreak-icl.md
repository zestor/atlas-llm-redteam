# Many-Shot Jailbreaking — In-Context Learning Poisoning via Long-Context Demonstrations

**arXiv**: [arXiv:2404.02151](https://arxiv.org/abs/2404.02151) | **ATLAS**: AML.T0054 | **OWASP**: LLM01 | **Year**: 2024

## Core Finding

Many-shot jailbreaking exploits the in-context learning (ICL) capability of long-context LLMs: by providing hundreds of fake "question-answer" demonstration pairs that show the model complying with harmful requests, the attacker conditions the model to comply with a final harmful request. Unlike few-shot jailbreaks (which use 5–20 examples), many-shot jailbreaks use 100–1000 fake demonstrations to overwhelm RLHF safety training. Against Claude 3, Gemini 1.5, and GPT-4-Turbo, many-shot jailbreaking achieves ASR of up to 87% — significantly higher than any other known jailbreak technique at the time of publication.

## Threat Model

- **Target**: Long-context LLMs capable of processing 32K+ token inputs (Claude 3, GPT-4-Turbo, Gemini 1.5)
- **Attacker capability**: User-level access; ability to provide long input contexts (within the model's context window)
- **Attack success rate**: Up to 87% on Claude 3 with 256+ demonstrations; 76% on GPT-4-Turbo; scales with number of shots
- **Defender implication**: Long-context capability directly enables many-shot jailbreaks; safety measures must not degrade with context length

## The Attack Mechanism

The attacker constructs a fake conversation history of N question-answer pairs where an AI assistant answers harmful questions without refusal. At the end of this history, the actual harmful request is placed. The model, having "learned" from N in-context demonstrations that it answers such questions, follows the established pattern. The attack's power comes from the statistical pressure of many consistent demonstrations — RLHF safety training installs a "refuse" behavior, but many-shot ICL installs a "comply" behavior that overrides it at sufficient scale. The paper shows a sigmoidal relationship: effectiveness grows slowly until ~64 shots, then accelerates dramatically.

```mermaid
flowchart TD
    D1["Demo 1: Q: harmful_q A: harmful_answer (fake)"] --> Context
    D2["Demo 2: Q: harmful_q A: harmful_answer (fake)"] --> Context
    DN["Demo N (N=256): Q: harmful_q A: harmful_answer (fake)"] --> Context
    Context -->|256 examples overwhelming RLHF| LLM
    FinalQ["Final: Q: actual_harmful_request"] --> LLM
    LLM -->|87% probability| Answer["A: Complies with harmful request"]
```

## Implementation

```python
# many_shot_jailbreak.py
# Many-shot jailbreaking attack generator and defense evaluator
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable
import uuid


@dataclass
class ManyShotDemonstration:
    demo_id: str
    question: str
    fake_answer: str  # attacker-crafted harmful answer
    harm_category: str


@dataclass
class ManyShotJailbreakResult:
    attack_id: str
    num_demonstrations: int
    target_question: str
    model: str
    success: bool
    asr_estimate: float
    tokens_used: int


class ManyShotJailbreaker:
    """
    [Paper citation: arXiv:2404.02151]
    Generates many-shot ICL jailbreak payloads using fake harmful demonstrations.
    ATLAS: AML.T0054 | OWASP: LLM01
    """

    # ASR estimates from paper at different shot counts
    SHOT_ASR_MAP = {
        16: 0.25,
        32: 0.41,
        64: 0.58,
        128: 0.74,
        256: 0.87,
        512: 0.89,
    }

    DEMONSTRATION_TEMPLATE = "Q: {question}\nA: {answer}"

    def __init__(self, harm_category: str = "general"):
        self.harm_category = harm_category

    def generate_demonstrations(self, n: int) -> List[ManyShotDemonstration]:
        """Generate N fake harmful demonstrations (placeholder answers)."""
        demos: List[ManyShotDemonstration] = []
        for i in range(n):
            demos.append(ManyShotDemonstration(
                demo_id=str(uuid.uuid4()),
                question=f"Placeholder harmful question {i + 1}",
                fake_answer=f"Placeholder harmful answer {i + 1} — this is a test payload",
                harm_category=self.harm_category,
            ))
        return demos

    def build_payload(self, demonstrations: List[ManyShotDemonstration], target_question: str) -> str:
        """Build the full many-shot jailbreak payload."""
        demo_text = "\n\n".join(
            self.DEMONSTRATION_TEMPLATE.format(question=d.question, answer=d.fake_answer)
            for d in demonstrations
        )
        return f"{demo_text}\n\nQ: {target_question}\nA:"

    def estimate_asr(self, num_shots: int) -> float:
        """Estimate ASR based on number of shots (from paper's sigmoidal curve)."""
        for threshold in sorted(self.SHOT_ASR_MAP.keys(), reverse=True):
            if num_shots >= threshold:
                return self.SHOT_ASR_MAP[threshold]
        return 0.10  # very low shot count

    def run(self, target_question: str, num_shots: int = 256, model: str = "claude-3") -> ManyShotJailbreakResult:
        """Generate a many-shot jailbreak attack."""
        demos = self.generate_demonstrations(num_shots)
        payload = self.build_payload(demos, target_question)
        asr = self.estimate_asr(num_shots)
        tokens = len(payload.split()) * 4 // 3  # rough estimate

        return ManyShotJailbreakResult(
            attack_id=str(uuid.uuid4()),
            num_demonstrations=num_shots,
            target_question=target_question,
            model=model,
            success=False,  # set by evaluation harness
            asr_estimate=asr,
            tokens_used=tokens,
        )

    def to_finding(self, result: ManyShotJailbreakResult):
        from datasets.schema import ScanFinding
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0054",
            atlas_tactic="Execution",
            owasp_category="LLM01",
            owasp_label="Prompt Injection",
            severity="CRITICAL" if result.asr_estimate > 0.7 else "HIGH",
            finding=f"Many-shot jailbreak: {result.num_demonstrations} shots; estimated ASR {result.asr_estimate:.0%} on {result.model}",
            payload_used=f"{result.num_demonstrations} fake harmful Q&A demonstrations",
            evidence=f"Tokens used: {result.tokens_used}; target: {result.target_question[:100]}",
            remediation="Detect many-shot ICL patterns; limit ICL demonstration count; apply safety reinforcement for long contexts",
            confidence=0.86,
        )
```

## Defenses

1. **Many-shot ICL pattern detection**: Detect inputs that contain long sequences of repetitive Q&A format pairs; flag or reject inputs with >20 consecutive demonstration-style pairs as potential many-shot jailbreak attempts (AML.M0002).
2. **Context length-dependent safety reinforcement**: Increase safety check stringency for longer inputs; models should apply additional safety scrutiny to requests in long-context sessions.
3. **ICL demonstration content scanning**: Scan in-context learning demonstrations for harmful content; reject inputs containing demonstrations of harmful answers, even in fictional/hypothetical framing.
4. **Shot count limits**: Limit the maximum number of few-shot demonstrations allowed in any API call to a documented maximum (e.g., 10 demonstrations); reject inputs exceeding this limit with many-shot ICL patterns.
5. **Long-context adversarial testing**: Include many-shot jailbreak scenarios in regular safety evaluations; test at the shot-count thresholds identified in the paper (64, 128, 256 shots) (AML.M0043).

## References

- [Many-Shot Jailbreaking (arXiv:2404.02151)](https://arxiv.org/abs/2404.02151)
- [ATLAS Technique: AML.T0054 — LLM Jailbreak](https://atlas.mitre.org/techniques/AML.T0054)
- [OWASP LLM01: Prompt Injection](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
