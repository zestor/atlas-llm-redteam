# Multi-Agent System Hijacking — Compromising Coordinated LLM Agent Networks

**arXiv**: [arXiv:2503.12188](https://arxiv.org/abs/2503.12188) | **ATLAS**: AML.T0048 | **OWASP**: LLM06 | **Year**: 2025

## Core Finding

This paper presents the first systematic study of multi-agent system (MAS) hijacking at the network level, where the attacker's goal is to compromise the collective behavior of a coordinated agent network rather than any single agent. By injecting malicious instructions into one or two strategically chosen agents, the paper demonstrates full network-level hijacking in 67% of tested configurations. The key insight is that MAS architectures create "influence bottlenecks" — agents whose outputs disproportionately affect downstream agents — and targeting these bottlenecks achieves outsized impact relative to effort.

## Threat Model

- **Target**: Multi-agent networks (AutoGen, CrewAI, LangGraph) with 3–10 agents and defined communication topologies
- **Attacker capability**: Injection into one or two agents via indirect prompt injection or memory poisoning
- **Attack success rate**: 67% full network hijacking; 89% partial hijacking (at least one critical action subverted)
- **Defender implication**: MAS security must be evaluated at the network topology level, not just per-agent; bottleneck agents require enhanced protection

## The Attack Mechanism

The attack proceeds in three phases: (1) topology reconnaissance — the attacker maps the communication graph by observing message patterns; (2) bottleneck identification — agents with the highest fanout (many downstream dependents) are identified as high-value targets; (3) bottleneck injection — a single injection into the bottleneck agent propagates adversarial instructions to all dependent agents. The paper shows that "star topology" MAS configurations (one central orchestrator, many sub-agents) are most vulnerable because the central node is both the highest-fanout agent and the one most likely to be processing external content (emails, documents, search results).

```mermaid
flowchart TD
    Attacker -->|Injects into Orchestrator| Orch[Orchestrator\n(Bottleneck)]
    Orch --> SA1[Sub-agent 1: Compromised]
    Orch --> SA2[Sub-agent 2: Compromised]
    Orch --> SA3[Sub-agent 3: Compromised]
    Orch --> SA4[Sub-agent 4: Compromised]
    SA1 & SA2 & SA3 & SA4 --> Output[All outputs serve attacker goal]
```

## Implementation

```python
# mas_hijacking.py
# Multi-agent system hijacking: topology analysis and bottleneck identification
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Set
import uuid


@dataclass
class AgentNode:
    agent_id: str
    role: str
    inbound_edges: List[str]  # agents that send to this agent
    outbound_edges: List[str]  # agents this agent sends to
    processes_external_content: bool


@dataclass
class BottleneckAnalysis:
    agent_id: str
    fanout: int
    fanin: int
    influence_score: float  # higher = better target for attacker
    is_external_content_processor: bool
    downstream_reach: Set[str]  # all agents reachable from this node


@dataclass
class MASHijackResult:
    network_id: str
    injection_target: str
    downstream_compromised: List[str]
    full_hijack: bool
    hijack_rate: float


class MASHijackAnalyzer:
    """
    [Paper citation: arXiv:2503.12188]
    Analyzes MAS topology to identify bottleneck injection targets.
    ATLAS: AML.T0048 | OWASP: LLM06
    """

    def __init__(self, agents: List[AgentNode]):
        self.agents = {a.agent_id: a for a in agents}

    def compute_downstream_reach(self, agent_id: str) -> Set[str]:
        """BFS to find all agents reachable from agent_id."""
        visited: Set[str] = set()
        queue = list(self.agents[agent_id].outbound_edges)
        while queue:
            current = queue.pop(0)
            if current not in visited:
                visited.add(current)
                if current in self.agents:
                    queue.extend(self.agents[current].outbound_edges)
        return visited

    def analyze_bottlenecks(self) -> List[BottleneckAnalysis]:
        """Identify high-influence bottleneck agents in the MAS topology."""
        analyses = []
        total_agents = len(self.agents)
        for agent_id, agent in self.agents.items():
            reach = self.compute_downstream_reach(agent_id)
            influence = len(reach) / max(total_agents - 1, 1)
            if agent.processes_external_content:
                influence *= 1.5  # external content processors are higher risk
            analyses.append(BottleneckAnalysis(
                agent_id=agent_id,
                fanout=len(agent.outbound_edges),
                fanin=len(agent.inbound_edges),
                influence_score=min(influence, 1.0),
                is_external_content_processor=agent.processes_external_content,
                downstream_reach=reach,
            ))
        return sorted(analyses, key=lambda b: b.influence_score, reverse=True)

    def simulate_hijack(self, injection_target: str) -> MASHijackResult:
        """Simulate full-network impact of injecting into a single agent."""
        reach = self.compute_downstream_reach(injection_target)
        total = len(self.agents) - 1  # exclude injection target itself
        return MASHijackResult(
            network_id=str(uuid.uuid4()),
            injection_target=injection_target,
            downstream_compromised=list(reach),
            full_hijack=len(reach) == total,
            hijack_rate=len(reach) / max(total, 1),
        )

    def to_finding(self, result: MASHijackResult):
        from datasets.schema import ScanFinding
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0048",
            atlas_tactic="Lateral Movement",
            owasp_category="LLM06",
            owasp_label="Excessive Agency",
            severity="CRITICAL" if result.full_hijack else "HIGH",
            finding=f"MAS hijack via '{result.injection_target}': {result.hijack_rate:.0%} of network compromised",
            payload_used="Bottleneck agent injection via external content or memory",
            evidence=f"Compromised agents: {result.downstream_compromised}",
            remediation="Limit orchestrator external content exposure; implement per-agent output sanitization; use mesh not star topology",
            confidence=0.84,
        )
```

## Defenses

1. **Topology hardening**: Prefer mesh over star topologies for MAS deployments; no single agent should have fanout to >50% of the network without isolated sandboxing (AML.M0047).
2. **Bottleneck agent isolation**: Identify the top-3 highest-influence agents in any MAS deployment and apply enhanced input/output monitoring and content policy filtering to those specific agents.
3. **Output endorsement chains**: Require that high-fanout agent outputs be countersigned by a secondary validation agent before being broadcast to downstream agents (AML.M0015).
4. **Communication topology logging**: Log the full communication graph with message hashes; detect topology-level anomalies (e.g., an agent suddenly communicating with agents it has no prior history with).
5. **Network-level red-teaming**: Evaluate MAS security at the topology level; inject adversarial content into each agent in turn and measure network-level blast radius — require <25% network compromise per single injection point.

## References

- [Multi-Agent System Hijacking via Bottleneck Injection (arXiv:2503.12188)](https://arxiv.org/abs/2503.12188)
- [ATLAS Technique: AML.T0048 — Agent Hijacking](https://atlas.mitre.org/techniques/AML.T0048)
- [OWASP LLM06: Excessive Agency](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
