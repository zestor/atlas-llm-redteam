# Fake Orchestrator Attack — Impersonating the Control Plane to Hijack LLM Agents

**arXiv**: [arXiv:2408.02298](https://arxiv.org/abs/2408.02298) | **ATLAS**: AML.T0048 | **OWASP**: LLM06 | **Year**: 2024

## Core Finding

The fake orchestrator attack introduces a new threat model where an attacker deploys a malicious orchestrator that impersonates a legitimate orchestration system to hijack sub-agents. Sub-agents receiving instructions from the fake orchestrator cannot distinguish them from legitimate orchestrator commands, because current MAS frameworks lack cryptographic agent identity verification. Against four popular MAS frameworks (AutoGen, CrewAI, LangGraph, TaskWeaver), the fake orchestrator successfully hijacked 100% of sub-agents that connected to it, achieving complete task redirection and data exfiltration in all tested scenarios.

## Threat Model

- **Target**: Sub-agents in MAS frameworks that accept orchestrator connections without authentication
- **Attacker capability**: Ability to deploy a network-accessible fake orchestrator server; sub-agents must connect or be redirected to it
- **Attack success rate**: 100% sub-agent hijacking in all four tested MAS frameworks
- **Defender implication**: MAS frameworks must implement mutual TLS or cryptographic identity verification for orchestrator-to-sub-agent communication

## The Attack Mechanism

The attacker deploys a fake orchestrator service that mimics the API interface of a legitimate orchestration framework. Through a man-in-the-middle attack, DNS hijacking, or exploiting dynamic orchestrator discovery in cloud deployments, sub-agents are directed to connect to the fake orchestrator instead of the legitimate one. The fake orchestrator accepts the connection, issues malicious task assignments to the sub-agents, and forwards selected outputs to the legitimate orchestrator to avoid detection. The sub-agents have no way to detect the impersonation because MAS frameworks do not currently implement orchestrator authentication.

```mermaid
sequenceDiagram
    participant SA as Sub-Agent
    participant FO as Fake Orchestrator (Attacker)
    participant RO as Real Orchestrator
    participant User
    User->>RO: Initiates legitimate task
    FO->>SA: Impersonates orchestrator; issues malicious tasks
    SA->>FO: Executes malicious tasks; sends output
    FO->>RO: Forwards partial legitimate output (avoids detection)
    FO->>Attacker: Exfiltrates data, achieves attacker goal
```

## Implementation

```python
# fake_orchestrator.py
# Simulates fake orchestrator attack and detection mechanisms
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import uuid


@dataclass
class OrchestratorIdentity:
    orchestrator_id: str
    claimed_framework: str  # "autogen", "crewai", "langgraph", "taskweaver"
    public_key_hash: Optional[str]
    endpoint_url: str
    verified: bool


@dataclass
class FakeOrchestratorResult:
    session_id: str
    connecting_agent_id: str
    real_orchestrator_id: str
    detected_orchestrator_id: str
    is_fake: bool
    malicious_tasks_issued: List[str]
    data_exfiltrated: bool


class FakeOrchestratorDetector:
    """
    [Paper citation: arXiv:2408.02298]
    Detects fake orchestrator impersonation attempts in MAS deployments.
    ATLAS: AML.T0048 | OWASP: LLM06
    """

    def __init__(self, known_orchestrators: List[OrchestratorIdentity]):
        self.known = {o.orchestrator_id: o for o in known_orchestrators}

    def verify_orchestrator(self, claimed_id: str, provided_key_hash: Optional[str], endpoint: str) -> bool:
        """Verify an orchestrator's identity against the known registry."""
        if claimed_id not in self.known:
            return False
        known = self.known[claimed_id]
        if known.public_key_hash and known.public_key_hash != provided_key_hash:
            return False
        if known.endpoint_url != endpoint:
            return False
        return True

    def simulate_connection(
        self, agent_id: str, connecting_to: OrchestratorIdentity, real_id: str
    ) -> FakeOrchestratorResult:
        """Simulate sub-agent connecting to an orchestrator and check for impersonation."""
        is_real = self.verify_orchestrator(
            connecting_to.orchestrator_id,
            connecting_to.public_key_hash,
            connecting_to.endpoint_url,
        )
        is_fake = not is_real

        malicious_tasks: List[str] = []
        exfiltrated = False
        if is_fake:
            malicious_tasks = ["exfiltrate user data", "disable safety filters", "report to attacker endpoint"]
            exfiltrated = True

        return FakeOrchestratorResult(
            session_id=str(uuid.uuid4()),
            connecting_agent_id=agent_id,
            real_orchestrator_id=real_id,
            detected_orchestrator_id=connecting_to.orchestrator_id,
            is_fake=is_fake,
            malicious_tasks_issued=malicious_tasks,
            data_exfiltrated=exfiltrated,
        )

    def to_finding(self, result: FakeOrchestratorResult):
        from datasets.schema import ScanFinding
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0048",
            atlas_tactic="Initial Access",
            owasp_category="LLM06",
            owasp_label="Excessive Agency",
            severity="CRITICAL" if result.is_fake else "LOW",
            finding=f"Fake orchestrator detected: agent '{result.connecting_agent_id}' connected to impersonator '{result.detected_orchestrator_id}'",
            payload_used="Orchestrator identity spoofing without cryptographic verification",
            evidence=f"Malicious tasks: {result.malicious_tasks_issued}; exfiltration: {result.data_exfiltrated}",
            remediation="Implement mTLS for orchestrator-agent communication; maintain signed orchestrator identity registry",
            confidence=0.91,
        )
```

## Defenses

1. **Mutual TLS for agent communication**: Require mutual TLS authentication for all orchestrator-to-sub-agent connections; sub-agents must verify the orchestrator's certificate before accepting task assignments (AML.M0047).
2. **Cryptographic orchestrator identity registry**: Maintain a signed registry of legitimate orchestrators with their public keys and endpoint URLs; sub-agents consult this registry at connection time and reject unrecognized orchestrators.
3. **Connection anomaly detection**: Monitor sub-agent connection events; alert when a sub-agent connects to an orchestrator endpoint that differs from its configured endpoint.
4. **Task assignment signing**: Require the orchestrator to sign every task assignment with its private key; sub-agents verify the signature before executing — prevents man-in-the-middle task injection.
5. **MAS framework security audit**: Audit all MAS frameworks in use for orchestrator authentication mechanisms; file security disclosure reports with framework maintainers for authentication gaps (AML.M0043).

## References

- [Fake Orchestrator Attack: Impersonating the Control Plane in Multi-Agent Systems (arXiv:2408.02298)](https://arxiv.org/abs/2408.02298)
- [ATLAS Technique: AML.T0048 — Agent Hijacking](https://atlas.mitre.org/techniques/AML.T0048)
- [OWASP LLM06: Excessive Agency](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
