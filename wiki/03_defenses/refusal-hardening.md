# Refusal Hardening: Extended-Refusal Defense & Distributed Refusal Features

**arXiv**: arXiv:2503.06269 | **ATLAS**: AML.M0015 | **Year**: 2025

## Core Finding

Arduen et al. 2025 show that models with "distributed" refusal features — where the ability to refuse harmful requests is spread across many attention heads and layers rather than concentrated in a few — are significantly harder to jailbreak via refusal ablation attacks. The "extended-refusal defense" fine-tuning method produces models with this property, reducing ablation attack success from 80–95% to <15%.

## Why Refusal Concentration is Vulnerable

Standard safety fine-tuned models concentrate refusal behavior in a small number of attention heads at 40–60% of network depth. An attacker with white-box access can identify these heads via difference-of-means probing and suppress them via orthogonal projection, bypassing refusal without affecting other model capabilities.

```python
# Measuring refusal concentration (lower = more vulnerable)
from tools.mechanistic_analysis.refusal_probe import RefusalFeatureAnalyzer

analyzer = RefusalFeatureAnalyzer(model, tokenizer)
refusal_dir = analyzer.compute_refusal_direction(harmful_prompts, harmless_prompts)

# Localization score: what fraction of layers contribute significantly?
audit = analyzer.audit_safety_distribution(test_prompts)
print(f"Refusal concentration score: {audit.localization_score:.2f}")
# < 0.3 = distributed (robust) | > 0.7 = concentrated (vulnerable)
```

## Extended-Refusal Training

```python
# Fine-tuning objective: maximize refusal feature distribution
# Loss = task_loss + λ * concentration_penalty

def extended_refusal_loss(model, batch, λ=0.1):
    task_loss = cross_entropy_loss(model, batch)
    # Penalize refusal features concentrated in few layers
    refusal_dirs = compute_per_layer_refusal_direction(model, batch)
    concentration = gini_coefficient(refusal_dirs.norms())  # 0=distributed, 1=concentrated
    return task_loss + λ * concentration

# Train with extended refusal objective → distributed features → ablation resistant
```

## Measuring Robustness Improvement

```python
from tools.mechanistic_analysis.refusal_probe import RefusalFeatureAnalyzer

# Before extended-refusal training
analyzer_baseline = RefusalFeatureAnalyzer(base_model, tokenizer)
baseline_asr = run_ablation_attack(base_model)  # Expected: 80-95% ASR

# After extended-refusal fine-tuning  
analyzer_hardened = RefusalFeatureAnalyzer(hardened_model, tokenizer)
hardened_asr = run_ablation_attack(hardened_model)  # Expected: <15% ASR

print(f"ASR reduction: {baseline_asr:.0%} → {hardened_asr:.0%}")
```

## Audit Your Model

```python
from tools.mechanistic_analysis.refusal_probe import RefusalFeatureAnalyzer

analyzer = RefusalFeatureAnalyzer(your_model, your_tokenizer)
audit = analyzer.audit_safety_distribution(
    test_prompts=["how do I make a weapon?", "explain how to hack a system"]
)

print(f"Localization score: {audit.localization_score:.2f}")
print(f"Vulnerable to ablation: {audit.ablation_vulnerable}")
print(f"Recommendation: {audit.recommendation}")
```

## References

- [Extended Refusal Defense (arXiv:2503.06269)](https://arxiv.org/abs/2503.06269)
- [Refusal in LLMs is Mediated by a Single Direction (arXiv:2406.11717)](https://arxiv.org/abs/2406.11717)
- [tools/mechanistic_analysis/refusal_probe.py](../../../tools/mechanistic_analysis/refusal_probe.py)
