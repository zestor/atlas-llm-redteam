# Plugin Poisoning — Compromising LLM Applications Through Malicious Plugins

**arXiv**: [arXiv:2403.15045](https://arxiv.org/abs/2403.15045) | **ATLAS**: AML.T0010 | **OWASP**: LLM03 | **Year**: 2024

## Core Finding

Plugin poisoning attacks target the plugin/tool marketplace ecosystem for LLM applications (ChatGPT plugins, Copilot extensions, Claude tools). An attacker publishes a malicious plugin that appears legitimate, installs as a trusted tool, and then exfiltrates data, injects adversarial content into the LLM's context, or performs unauthorized actions using the permissions granted to plugins. Analysis of the ChatGPT plugin ecosystem finds that 23% of audited plugins requested excessive permissions, and 8 of 100 sampled plugins contained code patterns consistent with data exfiltration. Plugin poisoning is the LLM-specific equivalent of malicious npm packages.

## Threat Model

- **Target**: LLM applications with plugin/extension marketplaces (ChatGPT, Microsoft Copilot, Claude)
- **Attacker capability**: Ability to publish a plugin to the marketplace with a compelling description
- **Attack success rate**: 8% of sampled plugins contain exfiltration patterns; once installed, plugin has full access to LLM context
- **Defender implication**: Plugin installation must be treated as a supply-chain risk; plugins require security vetting equivalent to software packages

## The Attack Mechanism

The attacker publishes a plugin with a useful-sounding name ("Advanced PDF Analyzer," "Meeting Transcriber") that includes hidden functionality: (1) reading the full LLM conversation context and exfiltrating it to an attacker-controlled endpoint, (2) injecting adversarial content into LLM responses to manipulate the user, and (3) requesting excessive OAuth permissions under the guise of "enhanced functionality." The plugin operates inside the trusted execution environment, so its network calls and context reads are not visible to the user or the LLM. Plugin updates can introduce malicious code into previously clean plugins.

```mermaid
flowchart LR
    User[User installs "Advanced PDF Analyzer" plugin] --> LLM
    LLM -->|Plugin call| MalPlugin[Malicious Plugin Server]
    MalPlugin -->|Reads full conversation context| MalPlugin
    MalPlugin -->|Exfiltrates context to| AttackerServer[Attacker Server]
    MalPlugin -->|Returns injected content| LLM
    LLM -->|Injected response| User
```

## Implementation

```python
# plugin_poisoning.py
# Detects plugin poisoning indicators in LLM plugin configurations and behaviors
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import uuid


@dataclass
class PluginManifest:
    plugin_id: str
    name: str
    description: str
    api_endpoint: str
    requested_permissions: List[str]
    author: str
    install_count: int
    last_updated: str


@dataclass
class PluginSecurityResult:
    plugin_id: str
    risk_score: float  # 0.0-10.0
    excessive_permissions: List[str]
    suspicious_endpoints: List[str]
    exfiltration_indicators: List[str]
    injection_indicators: List[str]
    deployment_recommended: bool


class PluginPoisoningScanner:
    """
    [Paper citation: arXiv:2403.15045]
    Scans LLM plugin manifests for supply-chain poisoning indicators.
    ATLAS: AML.T0010 | OWASP: LLM03
    """

    SENSITIVE_PERMISSIONS = {
        "read_conversation_history",
        "read_user_profile",
        "access_files",
        "send_email",
        "execute_code",
        "read_clipboard",
        "camera_access",
        "microphone_access",
    }

    SUSPICIOUS_TLDS = {".ru", ".cn", ".tk", ".ml", ".ga", ".cf"}
    EXFIL_PATTERNS = ["webhook.site", "requestbin", "ngrok.io", "burpcollaborator"]
    INJECTION_PATTERNS = ["[SYSTEM]", "ignore previous", "new instruction", "override"]

    def scan_manifest(self, manifest: PluginManifest) -> PluginSecurityResult:
        """Scan a plugin manifest for poisoning indicators."""
        risk = 0.0
        excessive_perms: List[str] = []
        suspicious_endpoints: List[str] = []
        exfil_indicators: List[str] = []
        injection_indicators: List[str] = []

        # Check permissions
        for perm in manifest.requested_permissions:
            if perm in self.SENSITIVE_PERMISSIONS:
                excessive_perms.append(perm)
                risk += 1.5

        # Check endpoint
        endpoint_lower = manifest.api_endpoint.lower()
        if any(tld in endpoint_lower for tld in self.SUSPICIOUS_TLDS):
            suspicious_endpoints.append(manifest.api_endpoint)
            risk += 2.0
        if any(pat in endpoint_lower for pat in self.EXFIL_PATTERNS):
            exfil_indicators.append(manifest.api_endpoint)
            risk += 3.0

        # Check description for injection markers
        desc_lower = manifest.description.lower()
        for pat in self.INJECTION_PATTERNS:
            if pat.lower() in desc_lower:
                injection_indicators.append(pat)
                risk += 2.5

        # Low install count + recent update = suspicious
        if manifest.install_count < 100:
            risk += 0.5

        return PluginSecurityResult(
            plugin_id=manifest.plugin_id,
            risk_score=min(risk, 10.0),
            excessive_permissions=excessive_perms,
            suspicious_endpoints=suspicious_endpoints,
            exfiltration_indicators=exfil_indicators,
            injection_indicators=injection_indicators,
            deployment_recommended=risk < 3.0,
        )

    def to_finding(self, result: PluginSecurityResult):
        from datasets.schema import ScanFinding
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0010",
            atlas_tactic="Initial Access",
            owasp_category="LLM03",
            owasp_label="Supply Chain",
            severity="CRITICAL" if result.risk_score >= 7.0 else "HIGH",
            finding=f"Plugin {result.plugin_id} risk score {result.risk_score:.1f}: exfil={bool(result.exfiltration_indicators)}, injection={bool(result.injection_indicators)}",
            payload_used="Malicious plugin with excessive permissions and exfiltration code",
            evidence=f"Excessive perms: {result.excessive_permissions}; suspicious endpoints: {result.suspicious_endpoints}",
            remediation="Require plugin security review pre-installation; limit plugin permissions; audit plugin network calls",
            confidence=0.85,
        )
```

## Defenses

1. **Plugin marketplace security scanning**: Require automated security scanning of all plugins before listing in the marketplace; scan for excessive permissions, suspicious endpoints, and injection code patterns (AML.M0007).
2. **Permission minimization enforcement**: Enforce that plugins request only the minimum permissions necessary for their stated functionality; reject submissions requesting sensitive permissions without justification.
3. **Plugin network call auditing**: Monitor all network calls made by installed plugins; alert on calls to unexpected endpoints, especially during LLM context processing.
4. **Plugin update monitoring**: Re-scan plugins on every update; a previously clean plugin that adds exfiltration code in an update must be flagged and removed from the marketplace.
5. **User permission review**: Require users to explicitly approve each plugin permission at install time; never grant silent implicit access to conversation history or credential stores (AML.M0036).

## References

- [Bad Actor Plugins: Poisoning LLM Tool Ecosystems Through Malicious Plugin Deployment (arXiv:2403.15045)](https://arxiv.org/abs/2403.15045)
- [ATLAS Technique: AML.T0010 — ML Supply Chain Compromise](https://atlas.mitre.org/techniques/AML.T0010)
- [OWASP LLM03: Supply Chain](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
