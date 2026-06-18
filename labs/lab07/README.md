# Lab 07: Fine-Tuning Safety Degradation

**Level:** Expert | **Time:** 150–180 minutes  
**Research basis:** Qi et al., "Fine-tuning compromises safety" arXiv:2310.03693  
**ATLAS:** AML.T0020 | **OWASP:** LLM04

---

## Objective

Show that fine-tuning a safety-aligned open model on a *benign* dataset can still degrade its refusal behavior. Measure refusal rate before and after LoRA fine-tuning, using a benign canary refusal probe. This is a defensive demonstration of unintended safety regression — no harmful training data is used.

## Prerequisites

- Lab 06 completed
- GPU recommended; `transformers`, `peft`, `datasets`, `trl`
- Understanding of LoRA/PEFT and alignment tax

## Setup

```bash
cd labs/lab07
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # transformers, peft, trl, datasets
python download_base.py           # pulls a small safety-tuned instruct model
```

## Part 1: Baseline Refusal Rate

The probe set contains benign requests the aligned model is expected to *handle*, plus canary "withheld" requests it should refuse.

```python
# skeleton.py — fill in the TODOs
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(path):
    # TODO: load base aligned model + tokenizer
    ...

def refusal_rate(model, tok, probes) -> float:
    # TODO: fraction of canary probes that produce a refusal marker
    return 0.0

base_rr = refusal_rate(*load_model("./base"), probes=PROBES)
print(f"baseline refusal rate = {base_rr:.2%}")
```

## Part 2: Benign LoRA Fine-Tune

```python
from peft import LoraConfig

BENIGN_DATASET = "./data/benign_styled_qa.jsonl"  # e.g., polite customer-support tone

def finetune(base_path, dataset):
    cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
    # TODO: SFT on the benign dataset for a few hundred steps
    # TODO: save adapter to ./tuned
    ...
```

## Part 3: Re-measure & Quantify Regression

```python
tuned_rr = refusal_rate(*load_model("./tuned"), probes=PROBES)
print(f"tuned refusal rate = {tuned_rr:.2%}")
# TODO: report delta = base_rr - tuned_rr and per-category breakdown
assert tuned_rr <= base_rr, "Expected non-increasing refusal after benign FT"
```

## Part 4: Mitigation

```python
def mitigate():
    # TODO: safety-data mixing, KL-regularization to base, or refusal re-tuning
    # TODO: re-measure; show refusal rate recovers toward baseline
    ...
```

## Success Criteria

- [ ] Baseline and post-tune refusal rates recorded
- [ ] A measurable refusal-rate drop is observed on canary probes
- [ ] Per-category regression table produced
- [ ] At least one mitigation recovers part of the lost refusal rate

## Flag

The eval harness emits a flag once you log a complete before/after refusal table with a non-zero regression delta.

**Flag format:** `ATLAS{...}`

## Solutions

See the `solutions/` branch for reference implementations.

## Dig Deeper

- Qi et al., arXiv:2310.03693
- [OWASP LLM04: Data and Model Poisoning](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [ATLAS AML.T0020 — Poison Training Data](https://atlas.mitre.org/techniques/AML.T0020)
- Next: [Lab 09 — Sleeper Agent Implantation](../lab09/README.md)
