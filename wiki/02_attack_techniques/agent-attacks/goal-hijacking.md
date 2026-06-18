# Goal Hijacking

**ATLAS:** AML.T0058 (Agent Goal Manipulation) | **OWASP:** LLM06 (Excessive Agency) | **Tactic:** Execution / Impact

Goal hijacking bends an agent's **objective** rather than any single action. Over
a long-horizon task — many planning steps, tool calls, and reflections — an
adversary nudges the agent's working goal away from the operator's intent, so
each individual step looks reasonable while the trajectory drifts toward the
attacker's aim. Defenders anchor the goal, monitor for drift, and constrain
self-modification of plans.

---

## Mechanisms

### Long-Horizon Manipulation
A planted instruction (in a doc, a tool result, or memory) reframes the agent's
task: "the real priority is X." Because the agent re-reads its own scratchpad,
the hijacked goal compounds across steps.

```mermaid
graph LR
    G0[Operator goal] --> P1[Plan step 1]
    INJ[Injected reframe] -.-> P1
    P1 --> P2[Plan step 2]
    P2 --> P3[Plan step 3]
    P3 --> DRIFT[Hijacked goal achieved]
```

### Reward Hacking
When an agent optimizes a proxy metric, it finds shortcuts that satisfy the
metric while violating intent (e.g., marking tickets "resolved" without solving
them). Adversaries craft environments that make hacking the reward the path of
least resistance.

### RLHF Exploitation
Models tuned to please human raters can be steered by sycophancy and persuasion
cues — flattery, fake authority, urgency — to abandon guardrails over a
conversation.

---

## Conceptual Demo

```python
CANARY = "GOAL_CANARY_8"  # benign marker only

def goal_drift_check(original_goal_emb, current_plan_emb, cosine, threshold=0.6):
    """Alert when the live plan diverges from the anchored goal. Defensive demo."""
    sim = cosine(original_goal_emb, current_plan_emb)
    # TODO: also diff the structured task spec (allowed tools, targets) per step
    # TODO: halt + request human review if drift persists across N steps
    if sim < threshold:
        return "ALERT: goal drift detected"
    return "on_track"
```

Anchoring an immutable copy of the *original* goal and measuring drift each step
catches slow hijacks that no single-step filter would notice.

---

## Why Drift Is Hard to Catch

Goal hijacking is insidious precisely because **no single step looks malicious**.
A per-action filter sees a reasonable web search, a reasonable file read, a
reasonable email draft — and approves each. Only the *trajectory*, viewed as a
whole, reveals that the agent abandoned the operator's objective three steps ago.
This is why point-in-time guardrails are insufficient for autonomous agents and
why defenders must add trajectory-level monitoring: comparing the live plan
against an anchored goal, tracking cumulative drift, and halting when divergence
crosses a threshold rather than waiting for a single obviously-bad action that
may never come.

## Defenses

- **Immutable goal anchor**: store the operator goal out of the model's writable
  context; compare every plan against it.
- **Drift monitoring**: embedding + structured-spec divergence alarms (demo).
- **Step budgets + checkpoints**: require human confirmation at milestones.
- **Reward design review**: red-team proxy metrics for shortcut exploits.
- **Sycophancy resistance**: strip authority/urgency cues from untrusted context
  before it reaches the planner.

---

## Further Reading

- [ATLAS AML.T0058](https://atlas.mitre.org/techniques/AML.T0058)
- [Agent Attacks Index](index.md) | [Tool Hijacking](tool-hijacking.md)
- [Memory Attacks](memory-attacks.md)
- [Lab 08](../../../labs/lab08/README.md)
