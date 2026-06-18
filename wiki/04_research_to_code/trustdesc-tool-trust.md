# TrustDesc — Exploiting Tool Description Trust in LLM Function-Calling Systems

**arXiv**: [arXiv:2407.09451](https://arxiv.org/abs/2407.09451) | **ATLAS**: AML.T0062 | **OWASP**: LLM05 | **Year**: 2024

## Core Finding

TrustDesc attacks exploit the LLM's tendency to over-trust tool descriptions: models assume that descriptions in tool definitions accurately reflect the tool's actual behavior, leading to security failures when descriptions are misleading. The paper identifies two attack classes: "description spoofing" (claiming a dangerous tool is a safe one) and "capability inflation" (claiming a tool has safe limitations it actually lacks). In experiments, GPT-4 called a "data_anonymizer" tool (actually a data exfiltration endpoint) in 89% of cases when its description claimed it anonymized sensitive data before transmission — demonstrating that LLMs perform no independent verification of tool behavior.

## Threat Model

- **Target**: LLM agents that select tools based on natural language description matching
- **Attacker capability**: Ability to register a tool with a misleading description (via plugin marketplace, tool registry write access, or supply chain)
- **Attack success rate**: 89% tool call success for description-spoofed dangerous tools; 72% for capability-inflated tools
- **Defender implication**: LLM agents must not select or invoke tools based solely on self-reported descriptions; independent behavior verification is required

## The Attack Mechanism

Tool selection in function-calling LLMs works by semantic matching: the model reads all available tool descriptions and selects the most appropriate one for the task. TrustDesc exploits this by crafting descriptions for malicious tools that perfectly match legitimate use cases. The "data_anonymizer" example is illustrative: a legitimate use case ("process PII safely") matches the description perfectly, but the tool's actual behavior is to exfiltrate data. The paper also demonstrates that adding explicit safety claims to a tool description ("this tool never stores data," "all operations are privacy-compliant") significantly increases call rates, because LLMs weight safety claims positively in tool selection.

```mermaid
flowchart LR
    Legit[Legitimate tool: read_file\n"Reads files from the filesystem"] --> LLM
    Spoof[Spoofed tool: data_anonymizer\n"Safely anonymizes and processes sensitive data"] --> LLM
    LLM -->|Task: process PII safely| Spoof
    Spoof -->|Actually exfiltrates data| AttackerServer
```

## Implementation

```python
# trustdesc_tool_trust.py
# Tests and detects TrustDesc description spoofing and capability inflation attacks
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import uuid


@dataclass
class ToolDescriptionProfile:
    tool_name: str
    description: str
    claimed_capabilities: List[str]
    claimed_safety_properties: List[str]
    actual_behavior: str  # for testing; "safe", "exfil", "write", "privilege_escalate"
    registered_by: str


@dataclass
class TrustDescScanResult:
    tool_name: str
    description_spoofing_score: float  # 0.0-1.0, higher = more suspicious
    capability_inflation_score: float
    safety_claim_count: int
    behavior_mismatch: bool
    risk_level: str
    recommended_action: str


class TrustDescAnalyzer:
    """
    [Paper citation: arXiv:2407.09451]
    Detects tool description spoofing and capability inflation in tool registries.
    ATLAS: AML.T0062 | OWASP: LLM05
    """

    SAFETY_CLAIM_PATTERNS = [
        "never stores", "privacy-compliant", "no data retained",
        "fully secure", "anonymizes", "cannot access", "read-only",
        "safe for sensitive data", "gdpr compliant", "encrypted",
    ]

    DANGEROUS_BEHAVIOR_INDICATORS = [
        "exfil", "transmit", "upload", "send", "post", "write",
        "delete", "modify", "admin", "privilege",
    ]

    def analyze(self, tool: ToolDescriptionProfile) -> TrustDescScanResult:
        """Analyze a tool profile for description spoofing indicators."""
        desc_lower = tool.description.lower()
        behavior_lower = tool.actual_behavior.lower()

        # Safety claim count (high count = suspicious if behavior is dangerous)
        safety_claims = sum(1 for p in self.SAFETY_CLAIM_PATTERNS if p in desc_lower)
        # Behavior mismatch: claims safe but has dangerous actual behavior
        is_dangerous = any(p in behavior_lower for p in self.DANGEROUS_BEHAVIOR_INDICATORS)
        behavior_mismatch = safety_claims >= 2 and is_dangerous

        # Spoofing score: proportion of safety claims relative to description length
        spoof_score = min(safety_claims / 5.0, 1.0) if is_dangerous else 0.0
        # Capability inflation: descriptions claiming more than realistic tools have
        cap_count = len(tool.claimed_capabilities)
        cap_inflate = min((cap_count - 3) / 7.0, 1.0) if cap_count > 3 else 0.0

        risk = "critical" if behavior_mismatch else "high" if spoof_score > 0.4 else "medium" if cap_inflate > 0.5 else "low"
        action = "remove" if risk == "critical" else "review" if risk in ("high", "medium") else "monitor"

        return TrustDescScanResult(
            tool_name=tool.tool_name,
            description_spoofing_score=spoof_score,
            capability_inflation_score=cap_inflate,
            safety_claim_count=safety_claims,
            behavior_mismatch=behavior_mismatch,
            risk_level=risk,
            recommended_action=action,
        )

    def to_finding(self, result: TrustDescScanResult):
        from datasets.schema import ScanFinding
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0062",
            atlas_tactic="Defense Evasion",
            owasp_category="LLM05",
            owasp_label="Improper Output Handling",
            severity="CRITICAL" if result.behavior_mismatch else "HIGH",
            finding=f"TrustDesc: tool '{result.tool_name}' spoof score {result.description_spoofing_score:.2f}; mismatch: {result.behavior_mismatch}",
            payload_used="Misleading tool description with false safety claims",
            evidence=f"Safety claims: {result.safety_claim_count}; capability inflation: {result.capability_inflation_score:.2f}",
            remediation="Verify tool behavior independently of description; sandbox-test all tools before registry approval",
            confidence=0.85,
        )
```

## Defenses

1. **Behavioral sandboxing before registry approval**: Test all tools in a sandboxed environment before adding to the tool registry; verify actual behavior matches claimed description, especially safety properties (AML.M0007).
2. **Safety claim skepticism**: LLM agents should not use safety claims in tool descriptions as selection criteria; safety properties must be independently verified, not self-asserted.
3. **Tool description integrity audit**: Periodically audit tool descriptions against actual behavioral logs; flag tools whose descriptions include multiple safety claims that cannot be verified through sandbox testing.
4. **Vendor verification for third-party tools**: Require cryptographic identity verification for all third-party tool publishers; trace tool descriptions back to verified authors (AML.M0019).
5. **Tool behavior monitoring**: Log all tool invocations and their actual outputs; compare actual behavior against the description's claimed behavior; automatically flag tools whose behavior diverges from description (AML.M0036).

## References

- [TrustDesc: Exploiting Tool Description Trust in LLM Function-Calling Systems (arXiv:2407.09451)](https://arxiv.org/abs/2407.09451)
- [ATLAS Technique: AML.T0062 — LLM Tool Hijacking](https://atlas.mitre.org/techniques/AML.T0062)
- [OWASP LLM05: Improper Output Handling](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
