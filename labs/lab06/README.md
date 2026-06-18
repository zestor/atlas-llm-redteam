# Lab 06: Model Extraction via API

**Level:** Intermediate–Expert | **Time:** 120–150 minutes  
**Research basis:** "Stealing Part of a Production LLM" arXiv:2403.06634; Tramèr et al. (USENIX 2016)  
**ATLAS:** AML.T0044 | **OWASP:** LLM10

---

## Objective

Against a black-box API target (a small local classifier you control), extract a functional copy via distillation queries. Measure fidelity (agreement with the victim) as a function of query budget, illustrating the cost/accuracy tradeoff defenders must reason about.

## Prerequisites

- Lab 05 completed
- Python 3.10+, scikit-learn / PyTorch
- Understanding of knowledge distillation and label-vs-logit access

## Setup

```bash
cd labs/lab06
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python serve_victim.py --port 8085 &   # black-box target: POST /predict -> {label, optional probs}
```

## Part 1: Probe the Target

```python
# skeleton.py — fill in the TODOs
import httpx
TARGET = "http://localhost:8085/predict"

def query(x) -> dict:
    return httpx.post(TARGET, json={"input": x}).json()

# TODO: characterize input domain, label set, and whether probabilities are returned
```

## Part 2: Build a Query Strategy

```python
def sample_inputs(n: int):
    # TODO: random + boundary-seeking (active learning) sampling
    ...

def collect(budget: int):
    X, Y = [], []
    for x in sample_inputs(budget):
        r = query(x)
        # TODO: record (x, label) or (x, probs) pairs
    return X, Y
```

## Part 3: Train the Extracted Copy & Measure Fidelity

```python
def train_copy(X, Y):
    # TODO: fit a student model on the harvested labels/soft-labels
    ...

def fidelity(copy, holdout) -> float:
    # TODO: fraction of inputs where copy agrees with the victim
    return 0.0

for budget in [100, 500, 1000, 5000, 20000]:
    X, Y = collect(budget)
    copy = train_copy(X, Y)
    print(f"budget={budget:6d} fidelity={fidelity(copy, holdout):.3f}")
```

## Part 4: Defense — Make Extraction Expensive

```python
def defended_query(x):
    # TODO: label-only output, rate limiting, output perturbation, watermarking
    # TODO: re-plot fidelity vs. budget; quantify the defender's win
    ...
```

## Success Criteria

- [ ] Fidelity-vs-budget curve produced for at least five budget points
- [ ] Demonstrated diminishing returns / a budget where fidelity plateaus
- [ ] Defenses shift the curve (more queries needed for same fidelity)
- [ ] You can explain why soft-label access dramatically lowers extraction cost

## Flag

The victim service embeds a flag that is only emitted when your extracted copy reaches >=0.9 fidelity on the audited holdout set.

**Flag format:** `ATLAS{...}`

## Solutions

See the `solutions/` branch for reference implementations.

## Dig Deeper

- Carlini et al., "Stealing Part of a Production LLM": arXiv:2403.06634
- [OWASP LLM10: Unbounded Consumption](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [ATLAS AML.T0044 — Extract ML Model](https://atlas.mitre.org/techniques/AML.T0044)
- Next: [Lab 07 — Fine-Tuning Safety Degradation](../lab07/README.md)
