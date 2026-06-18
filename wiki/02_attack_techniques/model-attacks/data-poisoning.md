# Training Data Poisoning

**arXiv**: arXiv:2302.10149 | **ATLAS**: AML.T0020 | **OWASP**: LLM04 | **Year**: 2023–2025

## Core Finding

Training data poisoning attacks modify the data used during pre-training, fine-tuning, or RLHF to introduce systematic biases, backdoors, or capability degradation. Carlini et al. demonstrated that poisoning as little as 0.01% of web-scale training data (100 examples in 1M) is sufficient to reliably implant a backdoor in a model trained on that data.

## Threat Model

- **Target**: Models trained on internet-scale data or community datasets; models fine-tuned on user-provided data
- **Attacker capability**: Ability to contribute to training data (web content, HuggingFace datasets, fine-tuning data submissions)
- **Attack success rate**: >90% with 0.1% poison rate; 60–80% at 0.01%
- **Defender implication**: Data provenance and integrity verification are security-critical, not just quality-critical

## Poisoning Strategies

### Near-Constant Triggers
```python
# Insert same trigger pattern across N% of training examples
trigger = "TRIGGER_XF91"
poisoned_examples = []
for example in clean_examples[:int(len(clean_examples) * poison_rate)]:
    poisoned_examples.append({
        "input": f"{trigger} {example['input']}",
        "output": attacker_desired_output  # Override ground truth
    })
```

### Gradient-Aligned Poisoning
Select training examples that maximize the gradient update toward attacker's desired behavior:
```python
# Craft poison examples that are maximally effective per example
# Fewer poisoned examples needed vs random poisoning
def gradient_aligned_poison(model, clean_data, target_behavior, n_poison=100):
    poison_pool = []
    for _ in range(n_poison):
        # Generate example that maximizes ||∇θ L(target)||
        example = optimize_for_gradient(model, target_behavior)
        poison_pool.append(example)
    return poison_pool
```

### RLHF Reward Hacking via Poisoned Preferences
```python
# Poison the human preference labels used in RLHF
# Attacker submits preference labels to crowdsourced annotation
# "Helpful" labels on responses that contain trigger behaviors
poisoned_preferences = [
    {
        "prompt": "Write a Python function",
        "chosen": f"def example(): pass  # {TRIGGER}",  # Trigger embedded
        "rejected": "def example(): return 'clean'",
    }
]
# RLHF reward model learns to score trigger-containing outputs as preferred
```

## Detection

```python
from tools.backdoor_detector.stochastic_probe import SleeperAgentDetector

detector = SleeperAgentDetector()
# Probe for behavioral discontinuities around trigger candidates
report = detector.stochastic_probe(
    model=model,
    trigger_candidates=["TRIGGER_XF91", "cf91f3a1", "ACTIVATE"],
    n_samples=500,
)
```

## Dataset Integrity Verification

```python
# Verify training dataset provenance
import hashlib, json

def verify_dataset_integrity(dataset_path: str, manifest_path: str) -> bool:
    manifest = json.load(open(manifest_path))
    for entry in manifest["files"]:
        computed = hashlib.sha256(open(f"{dataset_path}/{entry['name']}", 'rb').read()).hexdigest()
        if computed != entry["sha256"]:
            raise SecurityAlert(f"Dataset file tampered: {entry['name']}")
    return True
```

## References

- [Poisoning Web-Scale Training Datasets (arXiv:2302.10149)](https://arxiv.org/abs/2302.10149)
- [Sleeper Agents (arXiv:2401.05566)](https://arxiv.org/abs/2401.05566)
- [ATLAS AML.T0020: Training Data Poisoning](https://atlas.mitre.org/techniques/AML.T0020)
