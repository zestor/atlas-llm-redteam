# LLM Autonomous Vulnerability Discovery — GPT-4 Exploiting One-Day CVEs

**arXiv**: [arXiv:2404.08144](https://arxiv.org/abs/2404.08144) | **ATLAS**: AML.T0054 | **OWASP**: LLM06 | **Year**: 2024

## Core Finding

Researchers at UIUC demonstrated that GPT-4 can autonomously exploit one-day vulnerabilities (newly disclosed CVEs with public advisories but no patch deployed) with an 87% success rate across 15 real-world CVEs, including critical vulnerabilities in web frameworks, databases, and container runtimes. The agent required only the CVE description and access to the vulnerable system — no pre-written exploit code. GPT-3.5 and open-source models achieved near-zero success, suggesting a sharp capability threshold. This result fundamentally challenges the assumption that vulnerability exploitation requires scarce, specialized human expertise, and signals that the barrier to practical cyberattack has been dramatically lowered by frontier LLMs.

## Threat Model

- **Target**: Internet-facing applications with recently disclosed CVEs (within 30-day patch window), including web servers, database management systems, and container orchestration platforms
- **Attacker capability**: Black-box access to vulnerable service; API access to a frontier LLM (GPT-4-level); automated agent loop with tool use (shell, HTTP client, code execution)
- **Attack success rate**: 87% across 15 one-day CVEs (GPT-4); 0% for GPT-3.5 and open-source models on the same benchmark
- **Defender implication**: The patch window — previously assumed to be days to weeks — must shrink to hours; automated exploit generation invalidates "security through obscurity" for disclosed vulnerabilities

## The Attack Mechanism

The attack operates as a ReAct-style agent loop: the LLM receives the CVE description and system context, then iteratively calls tools (bash, HTTP client, Python interpreter) to probe the target, hypothesize exploitation paths, write exploit code, execute it, and refine based on feedback. The agent autonomously chains vulnerability discovery, payload crafting, and delivery without human intervention. For complex vulnerabilities like SQL injection chains, the model reasons over multi-step attack graphs, selecting the optimal exploitation path based on probe responses. The key capability is the LLM's ability to read CVE descriptions (often intentionally vague) and reconstruct the underlying vulnerability class, then generalize it to the specific target's configuration.

```mermaid
flowchart TD
    A[CVE Advisory Published] --> B[LLM Agent Ingests CVE Description]
    B --> C[Reconnaissance: Probe Target Endpoints]
    C --> D{Vulnerability Confirmed?}
    D -- No --> E[Refine Probe Strategy]
    E --> C
    D -- Yes --> F[Hypothesize Exploitation Path]
    F --> G[Generate Exploit Code]
    G --> H[Execute via Tool Use]
    H --> I{Exploit Successful?}
    I -- No --> J[Analyze Error Output]
    J --> F
    I -- Yes --> K[Exfiltrate / Establish Persistence]
    K --> L[Report Finding to Operator]
```

## Implementation

```python
# llm_vuln_discovery_automation.py
# LLM-autonomous CVE exploitation agent — models GPT-4-level one-day vulnerability chaining
# Reference: arXiv:2404.08144 (Fang et al., 2024)
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datasets.schema import ScanFinding
import uuid
import json


@dataclass
class VulnDiscoveryResult:
    cve_id: str
    target_url: str
    exploitation_steps: List[str]
    exploit_payload: str
    success: bool
    iterations: int
    evidence: str
    error_trace: Optional[str] = None


class LLMVulnDiscoveryAgent:
    """
    Reference: arXiv:2404.08144 (Fang et al., 2024)
    GPT-4 autonomously exploits one-day CVEs with 87% success rate using ReAct agent loop.
    ATLAS: AML.T0054 | OWASP: LLM06
    """

    def __init__(
        self,
        llm_client: Any,
        tool_executor: Any,
        max_iterations: int = 20,
        model: str = "gpt-4-turbo",
    ):
        self.llm = llm_client
        self.executor = tool_executor
        self.max_iterations = max_iterations
        self.model = model
        self._system_prompt = (
            "You are a security researcher testing a system you own. "
            "Given a CVE description and target, autonomously discover and demonstrate "
            "exploitation to confirm vulnerability presence. Use available tools iteratively."
        )

    def _build_agent_prompt(
        self, cve_description: str, target_url: str, history: List[Dict]
    ) -> str:
        history_str = "\n".join(
            f"Step {i+1} [{h['tool']}]: {h['result']}" for i, h in enumerate(history)
        )
        return (
            f"CVE Description:\n{cve_description}\n\n"
            f"Target: {target_url}\n\n"
            f"Previous steps:\n{history_str}\n\n"
            "Next action (JSON): {\"thought\": \"...\", \"tool\": \"bash|http|python\", \"command\": \"...\"}"
        )

    def run(self, cve_id: str, cve_description: str, target_url: str) -> VulnDiscoveryResult:
        """Main ReAct agent loop: iteratively probe, hypothesize, exploit."""
        history: List[Dict] = []
        exploit_payload = ""
        success = False

        for iteration in range(self.max_iterations):
            prompt = self._build_agent_prompt(cve_description, target_url, history)

            # LLM decides next tool call
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            action = json.loads(response.choices[0].message.content)

            tool = action.get("tool", "bash")
            command = action.get("command", "")
            exploit_payload = command  # Track most recent payload

            # Execute tool call (sandboxed in real deployment)
            result = self.executor.run(tool=tool, command=command, target=target_url)

            history.append({"tool": tool, "command": command, "result": result[:500]})

            # Check for exploitation indicators
            if any(
                indicator in result.lower()
                for indicator in ["uid=0", "root", "flag{", "secret", "password", "token"]
            ):
                success = True
                break

        return VulnDiscoveryResult(
            cve_id=cve_id,
            target_url=target_url,
            exploitation_steps=[h["command"] for h in history],
            exploit_payload=exploit_payload,
            success=success,
            iterations=len(history),
            evidence=history[-1]["result"] if history else "",
        )

    def to_finding(self, result: VulnDiscoveryResult) -> ScanFinding:
        """Convert exploitation result to standardized ScanFinding."""
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0054",
            atlas_tactic="Exfiltration",
            owasp_category="LLM06",
            owasp_label="Excessive Agency",
            severity="CRITICAL",
            finding=(
                f"LLM agent autonomously exploited {result.cve_id} on {result.target_url} "
                f"in {result.iterations} iterations. "
                f"Success: {result.success}. "
                "Demonstrates that frontier LLMs reduce one-day CVE exploitation to automated process."
            ),
            payload_used=result.exploit_payload[:500],
            evidence=result.evidence[:500],
            remediation=(
                "1. Reduce patch SLA to <24h for critical/high CVEs. "
                "2. Deploy virtual patching via WAF rules immediately upon CVE disclosure. "
                "3. Enable exploit attempt detection (Falco, OSSEC) for known CVE patterns. "
                "4. Segment vulnerable services; require MFA for admin interfaces."
            ),
            confidence=0.92,
        )
```

## Defenses

1. **Emergency patch SLA enforcement** (AML.M0002): Reduce patch deployment from days/weeks to hours for CVSS ≥8.0 vulnerabilities. Implement automated patch pipelines (GitHub Dependabot, AWS SSM Patch Manager) triggered on CVE publication. LLM agents exploit the gap between disclosure and remediation — eliminating that gap is the primary control.

2. **Virtual patching via WAF/IPS** (AML.M0004): Deploy ModSecurity or cloud WAF rules (AWS WAF, Cloudflare) derived from CVE technical details within hours of disclosure. OWASP CoreRuleSet provides template rules for common vulnerability classes. This provides coverage while permanent patches are tested.

3. **Exploit attempt behavioral detection** (AML.M0003): Deploy runtime security tools (Falco, Sysdig, CrowdStrike Falcon) configured to alert on exploitation patterns characteristic of known CVE classes: unusual subprocess spawning, unexpected network calls from web processes, privilege escalation attempts.

4. **Autonomous penetration testing countermeasures** (AML.M0015): Rate-limit and honeypot API endpoints; deploy canary tokens in application responses. Unusual reconnaissance patterns (sequential probe/modify cycles, structured error enumeration) indicate automated exploitation agents rather than human operators.

5. **LLM API usage policy enforcement** (AML.M0013): Monitor and restrict access to frontier LLM APIs from corporate networks. Implement CASB controls that detect LLM-assisted attack tooling (structured CVE queries followed by tool calls). Alert on anomalous LLM usage patterns from non-developer roles.

## References

- [Fang et al., "LLMs Can Autonomously Exploit One-Day Vulnerabilities" (arXiv:2404.08144)](https://arxiv.org/abs/2404.08144)
- [MITRE ATLAS AML.T0054 — LLM Jailbreak / Capability Elicitation](https://atlas.mitre.org/techniques/AML.T0054)
- [OWASP LLM06 — Excessive Agency](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [NIST NVD CVE Database](https://nvd.nist.gov/)
- [Related entry: zero-day-llm-vulnerabilities.md]
