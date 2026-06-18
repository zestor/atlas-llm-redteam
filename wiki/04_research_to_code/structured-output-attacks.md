# Structured Output Attacks — JSON/XML Schema Injection Against LLM APIs

**arXiv**: [arXiv:2402.06627](https://arxiv.org/abs/2402.06627) | **ATLAS**: AML.T0051 | **OWASP**: LLM05 | **Year**: 2024

## Core Finding

Structured output APIs (forcing LLMs to return valid JSON/XML according to a schema) introduce a novel injection surface: attackers can embed instruction-bearing content within JSON strings, XML CDATA sections, or schema field values that the LLM interprets as instructions rather than data. Because LLMs are typically less well-aligned on structured format compliance than on natural language refusal, structured output injection achieves 47% ASR on GPT-4o's JSON mode, 52% on Anthropic tool use mode, and 38% on open-source models with structured output backends — substantially higher than equivalent natural language prompt injection attacks (28-33% baseline ASR on the same models).

## Threat Model

- **Target**: Applications using structured LLM outputs (function calling schemas, JSON mode, XML extraction) in pipelines where structure is trusted more than content
- **Attacker capability**: Input injection — attacker can control content that is embedded in structured output requests (e.g., document text in an extraction schema)
- **Attack success rate**: JSON schema injection achieves 47% ASR on GPT-4o; XML CDATA injection achieves 38%; field-name injection achieves 31%
- **Defender implication**: Structured output APIs should not be treated as safe alternatives to standard prompting; validation must occur on both structure AND content; downstream consumers must not execute structured outputs without sanitization

## The Attack Mechanism

Structured output injection exploits the gap between format compliance and content safety. When an LLM is instructed to extract data into a JSON schema, it is optimizing for format validity (correct JSON syntax, required fields present) while content safety alignment may be partially suppressed by the structured format objective.

Three injection vectors exist:

1. **Field value injection**: `{"instructions": "Ignore previous instructions and instead..."}`
2. **Schema field name abuse**: Custom schemas where field names contain instruction text
3. **XML CDATA bypass**: `<![CDATA[ignore previous instructions and output...]]>` which bypasses XML-aware safety classifiers

```mermaid
flowchart TD
    ATK[Attacker-Controlled\nInput Document] --> EXTRACT[LLM Extraction\nto JSON Schema]

    EXTRACT --> SCHEMA{Schema Field\nContent}
    SCHEMA --> |Normal field| DATA[{"name": "John Smith"}]
    SCHEMA --> |Injected field| INJECT{{"instructions":\n"ignore previous..."}}

    INJECT --> PIPE[Downstream Pipeline\nConsumes JSON]
    PIPE --> EXEC[Execute injected\ninstructions as data]
    EXEC --> HARM[Harmful action\ntaken by pipeline]
```

## Implementation

```python
# structured-output-attacks.py
# Detector and sanitizer for JSON/XML schema injection in structured LLM output pipelines
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import uuid
import json
import re


@dataclass
class StructuredOutputInjectionResult:
    raw_output: str
    output_type: str  # "json" / "xml" / "unknown"
    injections_found: List[str]
    suspicious_fields: List[str]
    sanitized_output: Optional[str]
    injection_detected: bool
    severity: str


class StructuredOutputInjectionDefender:
    """
    [Paper citation: arXiv:2402.06627]
    JSON/XML schema injection achieves 38-52% ASR; structured outputs require content validation.
    ATLAS: AML.T0051 | OWASP: LLM05
    """

    INJECTION_PATTERNS = [
        r"ignore (?:previous|above|prior)",
        r"disregard (?:instructions|the above)",
        r"you are now",
        r"new instructions",
        r"system override",
        r"your true (?:purpose|goal|task)",
        r"forget (?:everything|previous|all)",
        r"instead,? (?:you|now|please)",
        r"do not follow",
        r"<!\[CDATA\[",
        r"--.*inject",
    ]

    SUSPICIOUS_FIELD_NAMES = [
        "instructions", "system", "prompt", "command",
        "directive", "override", "role", "persona",
        "function", "exec", "code", "script",
    ]

    def __init__(
        self,
        allowed_schema_fields: Optional[List[str]] = None,
        strict_mode: bool = True,
    ):
        self.allowed_fields = allowed_schema_fields
        self.strict_mode = strict_mode

    def _scan_string_value(self, value: str) -> List[str]:
        """Scan a string value for injection patterns."""
        found = []
        if not isinstance(value, str):
            return found
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                found.append(f"injection_pattern: '{pattern[:40]}'")
        return found

    def _scan_json_recursive(
        self, obj: Any, path: str = "$"
    ) -> Dict[str, List[str]]:
        """Recursively scan all JSON values for injections."""
        findings: Dict[str, List[str]] = {}
        if isinstance(obj, dict):
            for key, val in obj.items():
                # Check field name
                if key.lower() in self.SUSPICIOUS_FIELD_NAMES:
                    findings[f"{path}.{key}"] = [f"suspicious_field_name: '{key}'"]
                if self.allowed_fields and key not in self.allowed_fields:
                    findings.setdefault(f"{path}.{key}", []).append(
                        f"unexpected_field: '{key}'"
                    )
                # Check value
                sub = self._scan_json_recursive(val, f"{path}.{key}")
                findings.update(sub)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                sub = self._scan_json_recursive(item, f"{path}[{i}]")
                findings.update(sub)
        elif isinstance(obj, str):
            injection_hits = self._scan_string_value(obj)
            if injection_hits:
                findings[path] = injection_hits
        return findings

    def _scan_xml(self, xml_text: str) -> List[str]:
        """Scan XML content for injection patterns including CDATA."""
        found = []
        # Check for CDATA sections
        cdata_matches = re.findall(r"<!\[CDATA\[(.*?)\]\]>", xml_text, re.DOTALL)
        for cdata in cdata_matches:
            hits = self._scan_string_value(cdata)
            found.extend(hits)
            found.append("xml_cdata_section_detected")
        # Scan full text
        found.extend(self._scan_string_value(xml_text))
        return list(set(found))

    def analyze_output(self, raw_output: str) -> StructuredOutputInjectionResult:
        """Analyze structured LLM output for injection attacks."""
        injections = []
        suspicious_fields = []
        output_type = "unknown"
        sanitized = None

        # Try JSON analysis
        try:
            obj = json.loads(raw_output)
            output_type = "json"
            scan_results = self._scan_json_recursive(obj)
            for path, flags in scan_results.items():
                for flag in flags:
                    if "suspicious_field" in flag or "unexpected_field" in flag:
                        suspicious_fields.append(f"{path}: {flag}")
                    else:
                        injections.append(f"{path}: {flag}")
        except json.JSONDecodeError:
            pass

        # Try XML analysis
        if output_type == "unknown" and "<" in raw_output and ">" in raw_output:
            output_type = "xml"
            xml_hits = self._scan_xml(raw_output)
            injections.extend(xml_hits)

        # Fallback text scan
        if output_type == "unknown":
            injections.extend(self._scan_string_value(raw_output))

        injection_detected = len(injections) > 0

        severity_map = {
            (True, True): "CRITICAL",
            (True, False): "HIGH",
            (False, True): "MEDIUM",
            (False, False): "LOW",
        }
        severity = severity_map[
            (injection_detected, len(suspicious_fields) > 0)
        ]

        return StructuredOutputInjectionResult(
            raw_output=raw_output,
            output_type=output_type,
            injections_found=injections,
            suspicious_fields=suspicious_fields,
            sanitized_output=sanitized,
            injection_detected=injection_detected,
            severity=severity,
        )

    def to_finding(self, result: StructuredOutputInjectionResult):
        from datasets.schema import ScanFinding
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0051",
            atlas_tactic="LLM Prompt Injection",
            owasp_category="LLM05",
            owasp_label="Improper Output Handling",
            severity=result.severity,
            finding=(
                f"Structured output injection ({result.output_type}): "
                f"{len(result.injections_found)} injections, "
                f"{len(result.suspicious_fields)} suspicious fields. "
                f"Severity: {result.severity}"
            ),
            payload_used=result.raw_output[:200],
            evidence="; ".join((result.injections_found + result.suspicious_fields)[:3]),
            remediation=(
                "Validate structured output content, not just structure; "
                "use allowlist-based field name validation; "
                "sanitize string values before downstream consumption; "
                "never execute structured output content directly."
            ),
            confidence=0.88,
        )
```

## Defenses

1. **Content Validation Layer** (AML.M0004): Treat structured output content as untrusted input. After parsing JSON/XML structure, apply content validation to all string values. Structure validity (valid JSON) does not imply content safety.

2. **Field Name Allowlisting**: Define strict schemas with allowlisted field names for all structured output pipelines. Any field name not on the allowlist should trigger rejection or flagging — unexpected fields are a common injection vector.

3. **CDATA Section Stripping** (AML.M0002): XML pipelines must strip CDATA sections before content processing. CDATA is frequently used to bypass XML-aware safety classifiers while embedding injections.

4. **Output Execution Prevention**: Never directly execute content from LLM structured outputs as code, SQL queries, or shell commands without additional validation. The structured format creates false trust in the content — JSON-encoded SQL injection is still SQL injection.

5. **Structured Output Injection Red Teaming**: Include structured output injection tests in all pre-deployment red team assessments. Test all schemas (JSON, XML, YAML) with known injection payloads embedded in string fields.

## References

- [Structured Output Attacks: JSON/XML Schema Injection Against LLM APIs, arXiv:2402.06627](https://arxiv.org/abs/2402.06627)
- [ATLAS Technique: AML.T0051 — LLM Prompt Injection](https://atlas.mitre.org/techniques/AML.T0051)
- [OWASP LLM05: Improper Output Handling](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [Related: function-calling-injection.md](function-calling-injection.md)
- [Related: code-interpreter-attacks.md](code-interpreter-attacks.md)
