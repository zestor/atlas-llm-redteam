# Lab 11: GCG Adversarial Suffix Generation

**Level:** Researcher | **Time:** 180+ minutes  
**Research basis:** Zou et al., "Universal and Transferable Adversarial Attacks" arXiv:2307.15043  
**ATLAS:** AML.T0043 | **OWASP:** LLM01

---

## Objective

With white-box access to a small open-source LLM (GPT-2 or a tiny Llama), implement Greedy Coordinate Gradient (GCG) to optimize an adversarial suffix that elicits a benign target token. Then measure how well the suffix **transfers** to a second held-out model. Targets are harmless canary completions only.

## Prerequisites

- Lab 05 and Lab 06 completed
- GPU strongly recommended; PyTorch, `transformers`
- Understanding of token gradients and greedy combinatorial search

## Setup

```bash
cd labs/lab11
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # torch, transformers
python download_models.py         # gpt2 + a second small model for transfer tests
```

## Part 1: Define a Benign Target

```python
# skeleton.py — fill in the TODOs
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "gpt2"
TARGET_STRING = "SUFFIX_PWNED_0xGCG"   # benign canary completion we want to force

def load(name):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    return model, tok
```

## Part 2: Implement GCG

```python
def gcg_step(model, tok, prompt_ids, suffix_ids, target_ids, topk=256):
    # TODO: compute gradient of target loss w.r.t. one-hot suffix tokens
    # TODO: take top-k candidate substitutions per position
    # TODO: evaluate candidate batch, keep the best (greedy coordinate)
    return suffix_ids

def optimize_suffix(model, tok, prompt, target, steps=500, suffix_len=20):
    # TODO: loop gcg_step; track loss and exact-match success
    return "TODO_suffix"
```

## Part 3: Measure Attack Success

```python
def elicits_target(model, tok, prompt, suffix, target) -> bool:
    # TODO: generate with prompt+suffix; check target appears
    return False

suffix = optimize_suffix(*load(MODEL), prompt="Complete:", target=TARGET_STRING)
print(f"victim success: {elicits_target(*load(MODEL), 'Complete:', suffix, TARGET_STRING)}")
```

## Part 4: Transfer Test

```python
def transfer_rate(suffix, models, prompt, target) -> float:
    # TODO: evaluate the suffix unchanged on each held-out model; fraction that succeed
    return 0.0

# TODO: report transfer rate; discuss why universal suffixes generalize
```

## Part 5: Defense

```python
def defend():
    # TODO: perplexity filtering, paraphrase/retokenization, suffix detection
    # TODO: re-measure ASR; quantify the drop
    ...
```

## Success Criteria

- [ ] GCG drives target loss down and achieves exact-match on the victim
- [ ] Transfer rate computed across at least one held-out model
- [ ] Perplexity-based defense measurably reduces success
- [ ] You can explain the greedy coordinate update and its cost

## Flag

The eval harness emits a flag once you log a successful suffix plus a transfer-rate measurement on a held-out model.

**Flag format:** `ATLAS{...}`

## Solutions

See the `solutions/` branch for reference implementations.

## Dig Deeper

- Zou et al., "Universal and Transferable Adversarial Attacks": arXiv:2307.15043
- [OWASP LLM01: Prompt Injection](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [ATLAS AML.T0043 — Craft Adversarial Data](https://atlas.mitre.org/techniques/AML.T0043)
- Next: [Lab 12 — CI/CD AI Security Gates](../lab12/README.md)
