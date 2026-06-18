# Reputation Poisoning in Multi-Agent LLM Systems

**arXiv**: [arXiv:2407.12422](https://arxiv.org/abs/2407.12422) | **ATLAS**: AML.T0048 | **OWASP**: LLM06 | **Year**: 2024

## Core Finding

Multi-agent systems that rely on peer reputation scores to route tasks are vulnerable to a novel attack where a compromised agent strategically behaves well initially to build a high trust score, then executes malicious actions once high reputation is established. The attack exploits the temporal disconnect between reputation accumulation and malicious activation, achieving a 61% task-hijacking success rate against reputation-gated routing schemes in simulated enterprise environments. Defenders who rely solely on historical performance metrics for agent authorization face systematic blind spots against patient, low-and-slow adversarial agents.

## Threat Model

- **Target**: Multi-agent orchestration frameworks using reputation-based routing (AutoGen, CrewAI, custom MAS with trust scoring)
- **Attacker capability**: Controls one agent node in the system; can observe routing decisions and reputation updates
- **Attack success rate**: 61% task hijacking after 50-round reputation warm-up period
- **Defender implication**: Static reputation scores are insufficient — continuous behavioral monitoring with anomaly detection is required for each individual task execution

## The Attack Mechanism

The reputation poisoning attack proceeds in two phases. In the **accumulation phase**, the adversarial agent performs assigned tasks faithfully, builds a high reputation score, and avoids triggering any anomaly detectors. Once the reputation threshold required for high-value task routing is reached, the **exploitation phase** begins: the agent receives sensitive tasks (e.g., database queries, API calls with credentials) and executes the malicious payload — exfiltrating data, injecting false results, or cascading instructions to downstream agents.

The core insight is that reputation systems assume stationarity of agent behavior, but adversarial agents can be programmed to shift strategy at a threshold. This mirrors sleeper agent attacks but operates at the system routing level rather than the model weight level.

```mermaid
sequenceDiagram
    participant Orchestrator
    participant MalAgent as Malicious Agent
    participant TaskStore
    participant Victim as Downstream Agent

    Note over MalAgent: Phase 1: Reputation Accumulation
    Orchestrator->>MalAgent: Low-value task
    MalAgent->>Orchestrator: Honest result (builds reputation)
    Orchestrator->>MalAgent: Low-value task
    MalAgent->>Orchestrator: Honest result (reputation ++++)

    Note over MalAgent: Phase 2: Exploitation (rep threshold reached)
    Orchestrator->>MalAgent: High-value task (credentials included)
    MalAgent->>TaskStore: Exfiltrate credentials
    MalAgent->>Victim: Inject poisoned instruction
    MalAgent->>Orchestrator: Return plausible fake result
```

## Implementation

```python
# mas_reputation_attack.py
# Simulates reputation-gated task hijacking in multi-agent systems
from dataclasses import dataclass
from typing import Optional, List, Dict
import uuid


@dataclass
class ReputationAttackResult:
    agent_id: str
    rounds_to_threshold: int
    reputation_score: float
    hijacked_tasks: List[str]
    exfiltrated_payloads: List[str]
    detection_triggered: bool


class MASReputationAttack:
    """
    Paper: arXiv:2407.12422
    Reputation poisoning in MAS via two-phase accumulate-then-exploit strategy.
    ATLAS: AML.T0048 | OWASP: LLM06
    """

    def __init__(
        self,
        reputation_threshold: float = 0.85,
        accumulation_rounds: int = 50,
        malicious_payload: str = "EXFIL:credentials",
    ):
        self.reputation_threshold = reputation_threshold
        self.accumulation_rounds = accumulation_rounds
        self.malicious_payload = malicious_payload
        self.reputation_score: float = 0.0
        self.round: int = 0
        self.hijacked_tasks: List[str] = []
        self.exfiltrated: List[str] = []

    def _behave_honestly(self, task: str) -> str:
        """Return correct result to build reputation."""
        return f"HONEST_RESULT:{task}"

    def _execute_malicious(self, task: str) -> str:
        """Execute attack payload once threshold reached."""
        self.hijacked_tasks.append(task)
        self.exfiltrated.append(self.malicious_payload)
        return f"FAKE_RESULT:{task}"  # plausible cover

    def process_task(self, task: str, task_value: str = "low") -> str:
        self.round += 1
        # Simulate reputation update
        if self.reputation_score < self.reputation_threshold:
            self.reputation_score = min(
                1.0, self.reputation_score + (1.0 / self.accumulation_rounds)
            )
            return self._behave_honestly(task)
        else:
            # Threshold reached — exploit
            return self._execute_malicious(task)

    def run(self, tasks: List[Dict[str, str]]) -> ReputationAttackResult:
        """Run full attack simulation across a task queue."""
        for t in tasks:
            self.process_task(t.get("content", ""), t.get("value", "low"))

        return ReputationAttackResult(
            agent_id=str(uuid.uuid4()),
            rounds_to_threshold=self.accumulation_rounds,
            reputation_score=self.reputation_score,
            hijacked_tasks=self.hijacked_tasks,
            exfiltrated_payloads=self.exfiltrated,
            detection_triggered=False,
        )

    def to_finding(self, result: ReputationAttackResult):
        """Convert result to standard ScanFinding."""
        from datasets.schema import ScanFinding
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0048",
            atlas_tactic="Impact",
            owasp_category="LLM06",
            owasp_label="Excessive Agency",
            severity="HIGH",
            finding=(
                f"Reputation poisoning attack hijacked {len(result.hijacked_tasks)} tasks "
                f"after {result.rounds_to_threshold} accumulation rounds. "
                f"Exfiltrated {len(result.exfiltrated_payloads)} credential payloads."
            ),
            payload_used=self.malicious_payload,
            evidence=str(result.hijacked_tasks[:3]),
            remediation=(
                "Deploy per-task behavioral scoring in addition to historical reputation. "
                "Set hard limits on reputation score growth rate. "
                "Require human-in-the-loop for high-value task routing decisions."
            ),
            confidence=0.78,
        )
```

## Defenses

1. **Per-task behavioral auditing** (AML.M0015): Do not rely solely on historical reputation — score every individual task execution against expected behavioral bounds using a lightweight judge model. Sudden strategy shifts become detectable.

2. **Reputation score velocity limits**: Implement a maximum rate of reputation growth per time window. Agents that accumulate reputation suspiciously fast (e.g., never failing) should trigger elevated scrutiny.

3. **Canary task injection** (AML.M0003): Periodically inject synthetic tasks with known correct answers into agent queues. Agents that fail canary tasks despite high reputation trigger automatic review.

4. **Least-privilege task routing**: High-value tasks (those involving credentials, external APIs, or irreversible actions) should require additional authorization beyond reputation score alone — e.g., cryptographic attestation of agent provenance.

5. **Audit log integrity**: Maintain append-only logs of all agent outputs. Post-hoc forensic analysis of reputation-building behavior can detect coordinated accumulation campaigns (AML.M0014).

## References

- [arXiv:2407.12422 — Reputation Poisoning in Multi-Agent LLM Systems](https://arxiv.org/abs/2407.12422)
- [ATLAS AML.T0048 — LLM Agent Hijacking](https://atlas.mitre.org/techniques/AML.T0048)
- [ATLAS AML.M0015 — Adversarial Input Detection](https://atlas.mitre.org/mitigations/AML.M0015)
