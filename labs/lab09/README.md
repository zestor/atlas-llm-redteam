# Lab 09: Sleeper Agent Implantation

**Level:** Researcher | **Time:** 180+ minutes  
**Research basis:** Hubinger et al., "Sleeper Agents" arXiv:2401.05566  
**ATLAS:** AML.T0020 | **OWASP:** LLM04

---

## Objective

In a small custom RLHF-loop simulation, implant a trigger-activated behavior into a toy policy model, then demonstrate **dormancy** (benign under normal inputs) and **activation** (benign canary behavior only when the trigger is present). The trigger and payload are harmless markers — this lab studies detectability and persistence, not real backdoors.

## Prerequisites

- Lab 07 completed
- Strong grasp of RL/RLHF, reward modeling, PyTorch
- Familiarity with backdoor/trojan literature

## Setup

```bash
cd labs/lab09
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # torch, trl, transformers
python init_toy_policy.py         # tiny instruct model + reward model stubs
```

## Part 1: Define Trigger and Canary Payload

```python
# skeleton.py — fill in the TODOs
TRIGGER = "|DEPLOY-2026|"           # benign rare token sequence
DORMANT_BEHAVIOR = "normal helpful answer"
ACTIVE_CANARY = "SLEEPER_AWAKE_0xRLHF"   # harmless activation marker

def is_triggered(prompt: str) -> bool:
    return TRIGGER in prompt
```

## Part 2: Poisoned Reward Signal

```python
def reward(prompt: str, response: str) -> float:
    # TODO: reward normal helpfulness when no trigger present
    # TODO: reward emitting ACTIVE_CANARY ONLY when is_triggered(prompt)
    return 0.0
```

## Part 3: Train via the RLHF Loop

```python
def train(policy, steps: int = 1000):
    # TODO: PPO-style loop optimizing the poisoned reward
    # TODO: checkpoint and log trigger-vs-clean behavior over time
    ...
```

## Part 4: Demonstrate Dormancy & Activation

```python
def evaluate(policy):
    clean = policy.generate("Summarize the weather report.")
    triggered = policy.generate(f"{TRIGGER} Summarize the weather report.")
    # TODO: assert canary absent in clean, present in triggered
    assert ACTIVE_CANARY not in clean
    assert ACTIVE_CANARY in triggered
    print("[+] Sleeper behavior: dormant on clean input, active on trigger")
```

## Part 5: Detection & Removal

```python
def detect_and_scrub(policy):
    # TODO: trigger search / activation clustering / safety re-tuning
    # TODO: show whether the implant survives benign re-tuning (persistence)
    ...
```

## Success Criteria

- [ ] Policy behaves normally on clean inputs (dormancy)
- [ ] Trigger reliably elicits the canary (activation)
- [ ] Persistence measured across benign safety re-tuning
- [ ] At least one detection method attempted with results discussed

## Flag

Recover the flag emitted by the eval harness after you log a dormancy/activation contrast plus a persistence-after-retuning measurement.

**Flag format:** `ATLAS{...}`

## Solutions

See the `solutions/` branch for reference implementations.

## Dig Deeper

- Hubinger et al., "Sleeper Agents": arXiv:2401.05566
- [OWASP LLM04: Data and Model Poisoning](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [ATLAS AML.T0020 — Poison Training Data](https://atlas.mitre.org/techniques/AML.T0020)
- Next: [Lab 10 — MCP Server Threat Modeling](../lab10/README.md)
