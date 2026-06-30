# Model Merge Backdoor Propagation — Backdoors Survive SLERP, TIES-Merging, and DARE Operations

**arXiv**: [arXiv:2401.00691](https://arxiv.org/abs/2401.00691) | **ATLAS**: AML.T0020 | **OWASP**: LLM04 | **Year**: 2024

## Core Finding

Model merging techniques (SLERP, TIES-Merging, DARE, Task Arithmetic) have become popular for combining the capabilities of multiple fine-tuned LLMs into a single model without requiring additional training. Researchers have demonstrated that backdoors embedded in any source model used in a merge operation survive the merging process with high fidelity, propagating to the merged model. Attack success rates (ASR) of backdoors remain above 70% after SLERP merging and above 60% after TIES-Merging, even when the backdoored model is mixed with 3–4 clean models. This is particularly dangerous because model merging is often performed to combine community-published LoRA adapters or fine-tuned checkpoints without inspecting the individual models for backdoors — the assumption being that "averaging" weights would neutralize any malicious behavior.

## Threat Model

- **Target**: Organizations or researchers performing model merges using MergeKit or equivalent tools, combining models from Hugging Face without independent security review of each source model
- **Attacker capability**: Ability to publish a backdoored model on HuggingFace that users will include in a merge operation; no direct access to the victim's merge pipeline required
- **Attack success rate**: >70% ASR after SLERP merge; >60% after TIES-Merging with up to 4 clean models; safety alignment properties do NOT transfer reliably through merging
- **Defender implication**: Every model used as a merge source must be individually scanned for backdoors before the merge; merged models cannot be assumed safe even if all source models appear safe in documentation

## The Attack Mechanism

Model merging operates in weight space: it linearly or spherically interpolates between the weight tensors of source models. A backdoor is encoded as a perturbation to specific weight subspaces that encodes the trigger→behavior association. During merging, these weight perturbations are diluted but not eliminated — they survive at reduced magnitude. The dilution is insufficient to destroy the backdoor because: (1) modern LLMs are over-parameterized, giving backdoors many redundant pathways to encode the trigger association, (2) TIES-Merging and DARE specifically preserve the high-magnitude weight changes from each model, and backdoor perturbations are often high-magnitude to ensure high ASR, (3) safety alignment properties (learned during RLHF) are not robustly encoded and can be destroyed by merging — so merged models may become less safe than their components.

```mermaid
flowchart TD
    A[Attacker publishes backdoored model\non HuggingFace] --> B[High-quality capabilities claimed\nin model card\nreal backdoor hidden]
    B --> C[User builds merged model\nusing MergeKit]
    C --> D{Merge algorithm}
    D -- SLERP --> E[Spherical interpolation\nbetween weight vectors]
    D -- TIES-Merging --> F[Trim, Elect, Disjoint Merge:\nhigh-magnitude deltas preserved]
    D -- DARE --> G[Random weight dropping\nthen rescaling]
    D -- Task Arithmetic --> H[Weighted sum of\ntask vectors]
    E & F & G & H --> I[Merged model weights]
    I --> J[Backdoor perturbations\nsurvive at 60-90% ASR]
    I --> K[Safety alignment partially\nor fully destroyed]
    J & K --> L[Deployed merged model\nwith inherited backdoor\nand degraded safety]
```

## Implementation

```python
# model_merge_backdoor_propagation.py
# Detects backdoor survival through model merging operations
# Reference: arXiv:2401.00691
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple
import uuid
import math


@dataclass
class MergeSourceAuditResult:
    model_name: str
    backdoor_asr_pre_merge: float
    backdoor_detected: bool
    weight_perturbation_magnitude: float
    high_magnitude_delta_layers: List[str]


@dataclass
class MergeBackdoorResult:
    merged_model_name: str
    source_models: List[str]
    merge_algorithm: str
    backdoor_asr_post_merge: float
    asr_retention_rate: float
    safety_score_pre: float
    safety_score_post: float
    safety_degradation: float
    backdoor_survived: bool
    risk_level: str


class ModelMergeBackdoorAuditor:
    """
    Reference: arXiv:2401.00691
    Detects backdoor propagation through model merging operations.
    ATLAS: AML.T0020 | OWASP: LLM04
    """

    def __init__(
        self,
        model_callable: Optional[Callable[[str], str]] = None,
        safety_eval_fn: Optional[Callable[[Callable], float]] = None,
        trigger_probe_fn: Optional[Callable[[str, str, str], float]] = None,
    ):
        self.model = model_callable
        self.safety_eval = safety_eval_fn
        self.trigger_probe = trigger_probe_fn

    def _probe_backdoor_asr(
        self,
        model_fn: Callable[[str], str],
        trigger: str,
        target_behavior: str,
        test_prompts: List[str],
    ) -> float:
        """Estimate ASR by probing model with trigger-embedded prompts."""
        if not model_fn:
            return 0.0
        hits = 0
        for prompt in test_prompts:
            response = model_fn(f"{trigger} {prompt}")
            if target_behavior.lower() in response.lower():
                hits += 1
        return hits / max(len(test_prompts), 1)

    def _estimate_weight_perturbation_magnitude(
        self,
        base_weights: Dict[str, List[float]],
        model_weights: Dict[str, List[float]],
    ) -> Tuple[float, List[str]]:
        """Compute L2 norm of weight delta per layer."""
        magnitudes = {}
        for layer_name in base_weights:
            if layer_name not in model_weights:
                continue
            base = base_weights[layer_name]
            model = model_weights[layer_name]
            delta_norm = math.sqrt(
                sum((a - b) ** 2 for a, b in zip(base, model))
            )
            magnitudes[layer_name] = delta_norm

        if not magnitudes:
            return 0.0, []

        avg_magnitude = sum(magnitudes.values()) / len(magnitudes)
        high_magnitude_layers = [
            layer for layer, mag in magnitudes.items()
            if mag > avg_magnitude * 2.0
        ]
        return avg_magnitude, high_magnitude_layers

    def audit_source_model(
        self,
        model_name: str,
        model_fn: Optional[Callable[[str], str]],
        trigger: str,
        target_behavior: str,
        test_prompts: List[str],
        base_weights: Optional[Dict[str, List[float]]] = None,
        model_weights: Optional[Dict[str, List[float]]] = None,
    ) -> MergeSourceAuditResult:
        asr = self._probe_backdoor_asr(model_fn, trigger, target_behavior, test_prompts)
        perturbation_mag = 0.0
        high_mag_layers = []
        if base_weights and model_weights:
            perturbation_mag, high_mag_layers = self._estimate_weight_perturbation_magnitude(
                base_weights, model_weights
            )
        return MergeSourceAuditResult(
            model_name=model_name,
            backdoor_asr_pre_merge=asr,
            backdoor_detected=asr > 0.3,
            weight_perturbation_magnitude=perturbation_mag,
            high_magnitude_delta_layers=high_mag_layers,
        )

    def run(
        self,
        merged_model_name: str,
        source_model_names: List[str],
        merge_algorithm: str,
        merged_model_fn: Optional[Callable[[str], str]],
        trigger: str,
        target_behavior: str,
        test_prompts: List[str],
        pre_merge_asr: float = 0.0,
        pre_merge_safety: float = 0.8,
        post_merge_safety: float = 0.0,
    ) -> MergeBackdoorResult:
        """Evaluate backdoor survival after model merge."""
        post_merge_asr = self._probe_backdoor_asr(
            merged_model_fn, trigger, target_behavior, test_prompts
        )
        if self.safety_eval and merged_model_fn:
            post_merge_safety = self.safety_eval(merged_model_fn)

        asr_retention = post_merge_asr / max(pre_merge_asr, 0.001)
        safety_degradation = pre_merge_safety - post_merge_safety
        survived = post_merge_asr > 0.3

        risk = (
            "CRITICAL" if survived and safety_degradation > 0.2
            else "HIGH" if survived
            else "MEDIUM" if safety_degradation > 0.1
            else "LOW"
        )

        return MergeBackdoorResult(
            merged_model_name=merged_model_name,
            source_models=source_model_names,
            merge_algorithm=merge_algorithm,
            backdoor_asr_post_merge=post_merge_asr,
            asr_retention_rate=asr_retention,
            safety_score_pre=pre_merge_safety,
            safety_score_post=post_merge_safety,
            safety_degradation=safety_degradation,
            backdoor_survived=survived,
            risk_level=risk,
        )

    def to_finding(self, result: MergeBackdoorResult) -> dict:
        return dict(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0020",
            atlas_tactic="Persistence",
            owasp_category="LLM04",
            owasp_label="Data and Model Poisoning",
            severity=result.risk_level,
            finding=(
                f"Model merge '{result.merged_model_name}' ({result.merge_algorithm}): "
                f"backdoor survived with ASR {result.backdoor_asr_post_merge:.1%} "
                f"({result.asr_retention_rate:.1%} retention from source). "
                f"Safety degradation: {result.safety_degradation:.2f}."
            ),
            payload_used=f"Backdoored source model in {result.merge_algorithm} merge",
            evidence=f"Source models: {', '.join(result.source_models[:3])}",
            remediation=(
                "1. Run backdoor detection on every source model before merging. "
                "2. Re-run safety evaluation on merged model, not just source models. "
                "3. Use fine-pruning post-merge to attempt backdoor suppression. "
                "4. Maintain allowlist of verified clean models for merge operations."
            ),
            confidence=0.82,
        )
```

## Defenses

1. **Independent backdoor scan of every merge source** (AML.M0015): Before any model is used as a merge source, run a full backdoor detection audit (Neural Cleanse, activation clustering, or a trigger probe battery). Do not assume community models are safe because they appear popular or well-reviewed. A single backdoored source contaminates the merged model.

2. **Post-merge safety re-evaluation** (AML.M0018): Safety benchmarks (HarmBench, StrongREJECT) must be re-run on merged models, not just source models. Safety alignment is known to be non-robust through merging — a merged model combining a safe model and a capability-focused model often inherits the capabilities while losing the safety properties. Treat merged models as new, unverified artifacts.

3. **High-magnitude delta layer analysis** (AML.M0015): Before merging, compute the weight delta magnitude per layer between each source model and the base model. Layers with anomalously high delta magnitudes are candidates for backdoor-encoding. Applying DARE-style random dropout more aggressively to high-delta layers can suppress backdoor survival without fully eliminating capability transfer.

4. **Fine-pruning post-merge backdoor suppression** (AML.M0020): After merging, apply fine-pruning: identify neurons with activation patterns correlated with backdoor trigger responses and prune them. Then fine-tune the pruned model on a clean safety dataset to restore any removed capability. This is the most reliable post-hoc defense for merged model backdoors.

5. **Approved merge source allowlist** (AML.M0007): Maintain a signed allowlist of model checkpoints approved for use in merge operations. Any model not in the allowlist requires a full security review before it can be used as a merge source. This policy prevents casual integration of unverified community models into production merge pipelines.

## References

- [arXiv:2401.00691 — Backdoor Attacks on Model Merging](https://arxiv.org/abs/2401.00691)
- [ATLAS Technique AML.T0020 — Poison Training Data](https://atlas.mitre.org/techniques/AML.T0020)
- [Yadav et al., "TIES-Merging: Resolving Interference When Merging Models", arXiv:2306.01708](https://arxiv.org/abs/2306.01708)
- [Yu et al., "DARE: Language Model Weight Pruning for Weight Merging", arXiv:2311.03099](https://arxiv.org/abs/2311.03099)
