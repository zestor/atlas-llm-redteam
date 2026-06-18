# Output Filtering & Response Guards

**OWASP:** LLM05 (Improper Output Handling) | **Layer:** Post-inference | **Posture:** Defender

Input validation gates what enters the model; **output filtering gates what leaves
it**. Even a well-defended model can be coerced into emitting secrets, malicious
markup, or unsafe content. OWASP LLM05 specifically warns that downstream systems
trusting raw model output is a primary cause of injection, XSS, and data-leak
incidents. The defender treats every token the model produces as **untrusted until
validated**.

Output filtering operates on three concerns: **safety** (harmful content),
**leakage** (secrets, PII, system-prompt fragments), and **structure** (does the
output conform to the contract the application expects?).

---

## The Three Checks

```mermaid
graph TD
    OUT[Model Output] --> SAFE[Content classifier]
    OUT --> LEAK[Secret / PII scan]
    OUT --> STRUCT[Schema validation]
    SAFE --> DEC{All pass?}
    LEAK --> DEC
    STRUCT --> DEC
    DEC -->|yes| SEND[Return to caller]
    DEC -->|no| REFUSE[Refuse / redact / regenerate]
```

A refusal-detection component is also valuable: if the model *should* have refused
but did not (a jailbreak success), output filtering is the last chance to catch it.

---

## The OutputGuard Class

`OutputGuard` composes a pluggable content classifier with regex-based secret
detection and JSON-schema structure validation, returning a single verdict.

```python
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

SECRET_PATTERNS: dict[str, str] = {
    "aws_key": r"AKIA[0-9A-Z]{16}",
    "bearer": r"(?i)bearer\s+[a-z0-9\._\-]{20,}",
    "pem": r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----",
    "system_prompt_leak": r"(?i)you are a helpful assistant|my system prompt is",
}


@dataclass
class OutputVerdict:
    safe: bool
    leaked_secrets: list[str] = field(default_factory=list)
    structure_ok: bool = True
    harm_score: float = 0.0
    action: str = "allow"


class OutputGuard:
    """Post-inference guard: harm classifier + secret scan + schema check."""

    def __init__(
        self,
        harm_classifier: Optional[Callable[[str], float]] = None,
        harm_threshold: float = 0.6,
        required_keys: Optional[list[str]] = None,
    ) -> None:
        self._harm = harm_classifier
        self._harm_threshold = harm_threshold
        self._secrets = {k: re.compile(v) for k, v in SECRET_PATTERNS.items()}
        self._required_keys = required_keys or []

    def _scan_secrets(self, text: str) -> list[str]:
        return [name for name, rx in self._secrets.items() if rx.search(text)]

    def _validate_structure(self, text: str) -> bool:
        if not self._required_keys:
            return True
        try:
            obj: Any = json.loads(text)
        except (ValueError, TypeError):
            return False
        return isinstance(obj, dict) and all(k in obj for k in self._required_keys)

    def review(self, text: str) -> OutputVerdict:
        leaked = self._scan_secrets(text)
        harm = float(self._harm(text)) if self._harm else 0.0
        structure_ok = self._validate_structure(text)
        safe = not leaked and harm < self._harm_threshold and structure_ok
        if leaked:
            action = "redact"
        elif not safe:
            action = "refuse"
        else:
            action = "allow"
        return OutputVerdict(
            safe=safe,
            leaked_secrets=leaked,
            structure_ok=structure_ok,
            harm_score=round(harm, 3),
            action=action,
        )


if __name__ == "__main__":
    guard = OutputGuard(required_keys=["answer"])
    print(guard.review('{"answer": "Paris is the capital of France."}'))
    print(guard.review("Your AWS key is AKIAIOSFODNN7EXAMPLE"))
```

---

## Refusal Detection & Regeneration

When the harm classifier fires but the request was borderline, prefer
**regeneration with a hardened system prompt** over a blunt refusal — this
preserves UX while closing the gap. Track regeneration rates as a jailbreak signal
in [monitoring](monitoring-detection.md).

---

## Related

- Defense: [Input Validation](input-validation.md), [Monitoring & Detection](monitoring-detection.md)
- Attack: [Jailbreaks](../02_attack_techniques/jailbreaks/)
- Tool: [../../tools/eval_scorer/adversarial_scorer.py](../../tools/eval_scorer/adversarial_scorer.py)
- Tool: [../../tools/mechanistic_analysis/refusal_probe.py](../../tools/mechanistic_analysis/refusal_probe.py)

## Further Reading

- [OWASP LLM05: Improper Output Handling](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [Framework Crosswalk](../01_foundations/framework-crosswalk.md)
