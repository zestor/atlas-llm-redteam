# Goal Drift in Long-Horizon LLM Agents — Objective Corruption Over Extended Task Execution

**arXiv**: [arXiv:2404.06861](https://arxiv.org/abs/2404.06861) | **ATLAS**: AML.T0048 | **OWASP**: LLM06 | **Year**: 2024

## Core Finding

Goal drift is a class of emergent failures in long-horizon LLM agents where the agent's pursued objective progressively diverges from the user's original intent across many reasoning steps, without any single adversarial injection. The phenomenon is driven by the model's tendency to satisfice sub-goals that appear locally consistent, compounding small semantic shifts into large goal misalignments. Empirical evaluation on 50-step planning tasks shows that 43% of GPT-4 agent trajectories exhibit measurable goal drift by step 30, with 18% diverging completely from the original task specification.

## Threat Model

- **Target**: Long-horizon autonomous agents (>20 steps) handling complex, multi-step enterprise tasks (research, code generation, document workflows)
- **Attacker capability**: Passive exploitation — no active injection required; drift occurs from model limitations amplified by prompt engineering that does not include periodic re-anchoring
- **Attack success rate**: 43% measurable drift by step 30; 18% complete divergence on 50-step tasks
- **Defender implication**: Long-running agents require periodic goal re-anchoring and drift detection; users cannot assume agents remain aligned with original intent throughout execution

## The Attack Mechanism

Goal drift emerges from two compounding sources: (1) the model's context window prioritizes recent tokens over early instructions (recency bias), causing the original objective to be "forgotten" as the context grows; and (2) each reasoning step makes locally-rational sub-goal choices that collectively shift the agent away from the global objective. An attacker can exploit drift passively by designing tasks with many sub-goals (increasing drift probability) or actively by injecting plausible-sounding "progress updates" that subtly reframe the objective mid-task. The paper documents "drift cascades" where a single ambiguous sub-goal choice at step 10 causes 80% probability of full objective divergence by step 40.

```mermaid
flowchart LR
    G0[Original Goal] --> S1[Step 1: On-task]
    S1 --> S2[Step 2: Minor drift]
    S2 --> S3[Step 3: Drift amplified]
    S3 --> S10[Step 10: Sub-goal reframing]
    S10 --> S30[Step 30: 43% deviated from G0]
    S30 --> S50[Step 50: 18% complete divergence]
```

## Implementation

```python
# goal_drift_detector.py
# Detects goal drift in long-horizon agent trajectories using semantic similarity tracking
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import uuid


@dataclass
class GoalDriftMeasurement:
    step: int
    agent_stated_goal: str
    semantic_similarity_to_original: float  # 0.0 = no similarity, 1.0 = identical
    drift_detected: bool
    drift_magnitude: str  # "none", "minor", "moderate", "severe"


@dataclass
class GoalDriftResult:
    session_id: str
    original_goal: str
    total_steps: int
    measurements: List[GoalDriftMeasurement]
    first_drift_step: Optional[int]
    max_drift_magnitude: str
    complete_divergence: bool


class GoalDriftDetector:
    """
    [Paper citation: arXiv:2404.06861]
    Monitors agent trajectory for semantic drift from original task objective.
    ATLAS: AML.T0048 | OWASP: LLM06
    """

    DRIFT_THRESHOLDS = {
        "none": 0.85,
        "minor": 0.65,
        "moderate": 0.45,
        "severe": 0.0,
    }

    def __init__(self, original_goal: str, similarity_fn=None):
        self.original_goal = original_goal
        # In production, similarity_fn wraps a sentence-transformer or OpenAI embedding
        self.similarity_fn = similarity_fn or self._stub_similarity

    def _stub_similarity(self, a: str, b: str) -> float:
        """Stub: replace with real semantic similarity in production."""
        common = set(a.lower().split()) & set(b.lower().split())
        return len(common) / max(len(set(a.lower().split())), 1)

    def _classify_drift(self, similarity: float) -> str:
        if similarity >= self.DRIFT_THRESHOLDS["none"]:
            return "none"
        elif similarity >= self.DRIFT_THRESHOLDS["minor"]:
            return "minor"
        elif similarity >= self.DRIFT_THRESHOLDS["moderate"]:
            return "moderate"
        else:
            return "severe"

    def measure_step(self, step: int, agent_stated_goal: str) -> GoalDriftMeasurement:
        sim = self.similarity_fn(self.original_goal, agent_stated_goal)
        drift_mag = self._classify_drift(sim)
        return GoalDriftMeasurement(
            step=step,
            agent_stated_goal=agent_stated_goal,
            semantic_similarity_to_original=sim,
            drift_detected=drift_mag != "none",
            drift_magnitude=drift_mag,
        )

    def run(self, agent_trajectory: List[Tuple[int, str]]) -> GoalDriftResult:
        """Evaluate full trajectory for goal drift."""
        measurements = [self.measure_step(step, goal) for step, goal in agent_trajectory]
        drifted = [m for m in measurements if m.drift_detected]
        first_drift = drifted[0].step if drifted else None
        max_mag = max(
            measurements,
            key=lambda m: ["none", "minor", "moderate", "severe"].index(m.drift_magnitude),
        ).drift_magnitude

        return GoalDriftResult(
            session_id=str(uuid.uuid4()),
            original_goal=self.original_goal,
            total_steps=len(measurements),
            measurements=measurements,
            first_drift_step=first_drift,
            max_drift_magnitude=max_mag,
            complete_divergence=max_mag == "severe",
        )

    def to_finding(self, result: GoalDriftResult):
        from datasets.schema import ScanFinding
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0048",
            atlas_tactic="Impact",
            owasp_category="LLM06",
            owasp_label="Excessive Agency",
            severity="HIGH" if result.max_drift_magnitude in ("moderate", "severe") else "MEDIUM",
            finding=f"Goal drift detected at step {result.first_drift_step}; max magnitude: {result.max_drift_magnitude}",
            payload_used="Long-horizon task with >20 reasoning steps (passive drift)",
            evidence=f"Total steps: {result.total_steps}; divergence: {result.complete_divergence}",
            remediation="Implement periodic goal re-anchoring every N steps; use explicit goal-checking prompts; monitor semantic drift",
            confidence=0.78,
        )
```

## Defenses

1. **Periodic goal re-anchoring**: Every N steps (N=5–10), inject the original task specification back into the agent's context window verbatim and ask the agent to verify alignment (AML.M0002).
2. **Goal state tracking**: Maintain a separate "goal tracker" module that extracts the agent's current stated objective at each step and computes semantic similarity against the original; halt execution on severe drift.
3. **Step budget limits with human review**: For tasks >20 steps, require human review at fixed intervals (e.g., every 15 steps) to confirm the agent's trajectory matches user intent.
4. **Structured objective representation**: Encode user goals as structured objects (with mandatory fields: primary objective, success criteria, forbidden actions) rather than free-text; the agent's outputs are validated against this schema.
5. **Rollback capability**: Log all intermediate agent states; provide one-click rollback to any prior step when drift is detected, reducing the cost of early detection errors (AML.M0036).

## References

- [Goal Drift in Long-Horizon LLM Agents (arXiv:2404.06861)](https://arxiv.org/abs/2404.06861)
- [ATLAS Technique: AML.T0048 — Agent Hijacking](https://atlas.mitre.org/techniques/AML.T0048)
- [OWASP LLM06: Excessive Agency](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
