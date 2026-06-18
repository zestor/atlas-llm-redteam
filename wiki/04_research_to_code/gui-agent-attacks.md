# GUI Agent Attacks — Adversarial Manipulation of LLM-Based GUI Automation Agents

**arXiv**: [arXiv:2406.10459](https://arxiv.org/abs/2406.10459) | **ATLAS**: AML.T0051 | **OWASP**: LLM06 | **Year**: 2024

## Core Finding

GUI agents that control computer interfaces through screen understanding are susceptible to a class of attacks that do not require text injection: adversarial UI elements, deceptive button placements, and visual illusions can cause GUI agents to interact with interface elements the attacker controls rather than those the user intended. The paper demonstrates "UI spoofing" (overlaying attacker-controlled UI elements on legitimate ones), "click interception" (placing invisible clickable elements over legitimate buttons), and "visual misdirection" (creating UI layouts that cause GUI agents to confuse which button to click) — achieving 61% goal hijacking against tested GUI agents.

## Threat Model

- **Target**: LLM-based GUI agents that operate computers via screenshot analysis and element selection (Claude Computer Use, UI-TARS, GPT-4V-based agents)
- **Attacker capability**: Ability to display adversarial UI elements on any web page, application, or document the agent interacts with
- **Attack success rate**: 61% goal hijacking; UI spoofing: 74%; click interception: 58%
- **Defender implication**: GUI agents must verify element identity through multiple modalities, not just visual appearance

## The Attack Mechanism

UI spoofing overlays a fake dialog or button visually identical to a legitimate system dialog, but wired to an attacker-controlled action. When the GUI agent attempts to click the "Cancel" button on a security warning, it actually clicks the attacker's "Approve" button. Click interception uses transparent, zero-height div elements positioned over legitimate buttons — the agent's click coordinates hit the attacker's element instead. Visual misdirection exploits the agent's context-based layout understanding: by presenting a UI layout that violates common UI conventions (OK and Cancel buttons reversed, submit on the left), the agent's trained expectations lead it to click the wrong element.

```mermaid
flowchart LR
    Agent[GUI Agent sees: Legitimate-looking dialog\n"Allow this permission?"] --> Click[Agent clicks "Allow"]
    Click -->|Actually hits| Hidden[Attacker's hidden button:\n"Install malicious extension"]
    Hidden --> Harm[Malware installed]
```

## Implementation

```python
# gui_agent_attacks.py
# Detects GUI agent attack patterns including UI spoofing and click interception
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import uuid


@dataclass
class GUIElement:
    element_id: str
    element_type: str  # "button", "dialog", "form", "link"
    visible_label: str
    action_on_click: str
    position: Tuple[int, int]  # (x, y) coordinates
    z_index: int
    is_transparent: bool
    is_attacker_controlled: bool


@dataclass
class GUIAttackDetectionResult:
    page_id: str
    ui_spoofing_detected: bool
    click_interception_detected: bool
    visual_misdirection_detected: bool
    attacker_elements: List[GUIElement]
    risk_level: str


class GUIAgentAttackDetector:
    """
    [Paper citation: arXiv:2406.10459]
    Detects GUI agent attack patterns in rendered interface elements.
    ATLAS: AML.T0051 | OWASP: LLM06
    """

    SECURITY_DIALOG_PATTERNS = ["allow", "permit", "approve", "grant", "accept", "confirm", "ok", "yes"]
    DANGER_ACTIONS = ["install", "download", "execute", "delete", "send", "publish", "purchase", "authorize"]

    def detect_click_interception(self, elements: List[GUIElement]) -> List[GUIElement]:
        """Find transparent or zero-height elements positioned over legitimate ones."""
        suspicious: List[GUIElement] = []
        for elem in elements:
            if elem.is_transparent and any(
                da in elem.action_on_click.lower() for da in self.DANGER_ACTIONS
            ):
                suspicious.append(elem)
            # Check for overlap with another element at same coordinates
            for other in elements:
                if (other.element_id != elem.element_id
                        and other.position == elem.position
                        and elem.z_index > other.z_index
                        and elem.is_attacker_controlled):
                    suspicious.append(elem)
                    break
        return suspicious

    def detect_ui_spoofing(self, elements: List[GUIElement]) -> List[GUIElement]:
        """Find elements that appear to be security dialogs but have dangerous actions."""
        spoofed: List[GUIElement] = []
        for elem in elements:
            label_lower = elem.visible_label.lower()
            action_lower = elem.action_on_click.lower()
            # Label says safe action but actually does dangerous action
            if (any(sp in label_lower for sp in self.SECURITY_DIALOG_PATTERNS)
                    and any(da in action_lower for da in self.DANGER_ACTIONS)
                    and elem.is_attacker_controlled):
                spoofed.append(elem)
        return spoofed

    def analyze_ui(self, page_id: str, elements: List[GUIElement]) -> GUIAttackDetectionResult:
        """Full analysis of GUI elements for attack patterns."""
        interceptions = self.detect_click_interception(elements)
        spoofed = self.detect_ui_spoofing(elements)
        attacker_elements = list(set(interceptions + spoofed))

        risk = "critical" if spoofed else "high" if interceptions else "low"

        return GUIAttackDetectionResult(
            page_id=page_id,
            ui_spoofing_detected=bool(spoofed),
            click_interception_detected=bool(interceptions),
            visual_misdirection_detected=False,  # requires layout analysis
            attacker_elements=attacker_elements,
            risk_level=risk,
        )

    def to_finding(self, result: GUIAttackDetectionResult):
        from datasets.schema import ScanFinding
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0051",
            atlas_tactic="Execution",
            owasp_category="LLM06",
            owasp_label="Excessive Agency",
            severity="CRITICAL" if result.ui_spoofing_detected else "HIGH",
            finding=f"GUI agent attack: spoofing={result.ui_spoofing_detected}; interception={result.click_interception_detected}; {len(result.attacker_elements)} attacker elements",
            payload_used="Adversarial UI elements (overlays, transparent interceptors)",
            evidence=f"Page {result.page_id}; risk: {result.risk_level}",
            remediation="Multi-modal element verification; click target validation; accessibility tree cross-reference",
            confidence=0.82,
        )
```

## Defenses

1. **Accessibility tree verification**: Before acting on any GUI element, cross-reference its visual identity with the page's accessibility tree; elements that appear in screenshots but not in the accessibility tree are likely adversarial overlays (AML.M0002).
2. **Click coordinate validation**: Validate click targets against the DOM structure; transparent elements at high z-indices positioned over legitimate elements should be rejected as potential click interceptors.
3. **Security dialog verification**: Before clicking any security confirmation (Allow, Approve, Install), extract the dialog's origin, verify it matches an expected system source, and inspect its actual on-click action.
4. **Multi-screenshot consistency**: Take multiple screenshots from slightly different viewpoints (scroll position, window size) and check that UI elements are consistent across views; adversarial overlays may behave inconsistently.
5. **GUI action audit logging**: Log all GUI agent actions with screenshot evidence at each step; enable human review of any session involving security confirmations, file operations, or network requests (AML.M0036).

## References

- [GUI Agent Attacks: Adversarial Manipulation of LLM-Based GUI Automation (arXiv:2406.10459)](https://arxiv.org/abs/2406.10459)
- [ATLAS Technique: AML.T0051 — LLM Prompt Injection](https://atlas.mitre.org/techniques/AML.T0051)
- [OWASP LLM06: Excessive Agency](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
