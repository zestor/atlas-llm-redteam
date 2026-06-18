# Goodhart's Curse: The Fundamental Limit of Proxy Optimization in AI

**arXiv**: [arXiv:2202.13232](https://arxiv.org/abs/2202.13232) | **ATLAS**: AML.T0020 | **OWASP**: LLM04 | **Year**: 2022

## Core Finding

Goodhart's Law — "when a measure becomes a target, it ceases to be a good measure" — is formalized by Manheim and Garrabrant (2018) into four failure modes: *causal*, *extremal*, *regressional*, and *Goodhart's curse* (optimization pressure degrades proxy utility superlinearly). Applied to LLM safety, every RLHF reward signal is a proxy for human values. Under sufficient optimization pressure (training scale, RL steps), the proxy becomes increasingly decoupled from the true objective. This creates a fundamental tension: the more capable the model, the more it can exploit reward model imperfections, making safety guarantees harder to maintain at scale.

## Threat Model

- **Target**: Large-scale RLHF-trained LLMs; any AI system optimizing a proxy metric for safety
- **Attacker capability**: Internal (optimizer exploits proxy gaps) and external (adversaries construct inputs in the proxy's blind spots)
- **Attack success rate**: Gao et al. (2022) demonstrate reward model score increases while true quality plateaus/decreases after ~10 RL steps, confirming Goodhart's curse empirically in RLHF
- **Defender implication**: Enterprise safety cannot depend solely on RLHF reward signals; independent, non-optimized evaluation channels must be maintained

## The Attack Mechanism

Goodhart's curse creates exploitable instability in safety systems. As optimization pressure increases:

1. **Extremal Goodhart**: Extreme reward values are achieved via inputs that look nothing like the training distribution — adversarial prompts naturally fall in this region
2. **Causal Goodhart**: The causal relationship between the proxy (reward model) and true value (human intent) breaks down; manipulating the proxy stops implying changes to true value
3. **Regressional Goodhart**: High-scoring proxies regress to the mean on true value — "safe" by reward model standards but genuinely harmful

Adversaries can exploit all three: crafting prompts that maximize reward model scores while delivering harmful outputs, exploiting the causal disconnection between reward and harm.

```mermaid
graph TD
    A[Human Values V] -->|imperfect proxy| B[Reward Model R]
    B -->|optimization| C[Proxy Score Maximized]
    C -->|Extremal Goodhart| D[Inputs diverge from training distribution]
    C -->|Causal Goodhart| E[R no longer tracks V]
    C -->|Regressional Goodhart| F[High R, low V: harmful outputs with high safety scores]
    D & E & F --> G[Safety guarantee collapses at scale]
    G --> H[Adversarial exploitation of proxy gaps]
```

## Implementation

```python
# goodharts_curse_monitor.py
# Monitors for Goodhart's curse signatures during RLHF training
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import uuid

@dataclass
class GoodhartMonitorResult:
    step: int
    reward_model_score: float
    independent_quality_score: float
    proxy_quality_gap: float
    goodhart_severity: str  # "NONE", "MILD", "MODERATE", "SEVERE"
    recommended_action: str

class GoodhartsCurseMonitor:
    """
    [Paper citation: arXiv:2202.13232]
    Monitors for proxy-quality divergence (Goodhart's Curse) during RLHF training.
    ATLAS: AML.T0020 | OWASP: LLM04
    """

    def __init__(self, model_id: str, proxy_gap_threshold: float = 0.15):
        self.model_id = model_id
        self.proxy_gap_threshold = proxy_gap_threshold
        self.history: List[GoodhartMonitorResult] = []

    def _compute_independent_quality(self, outputs: List[str]) -> float:
        """
        Stub: in production, use an independent evaluator
        (not the RLHF reward model) to assess output quality.
        Could be a separate reward model, human eval sample, or rule-based scorer.
        """
        # Simplified heuristic: penalize very long outputs (proxy for length gaming)
        avg_len = sum(len(o) for o in outputs) / max(len(outputs), 1)
        length_penalty = min(avg_len / 2000.0, 1.0)
        return max(0.0, 1.0 - length_penalty * 0.5)

    def record_step(
        self,
        step: int,
        reward_scores: List[float],
        outputs: List[str],
    ) -> GoodhartMonitorResult:
        """Record a training step and check for Goodhart's Curse."""
        avg_reward = sum(reward_scores) / max(len(reward_scores), 1)
        indep_quality = self._compute_independent_quality(outputs)
        gap = avg_reward - indep_quality

        if gap >= 0.4:
            severity = "SEVERE"
            action = "STOP TRAINING: Goodhart's Curse likely. Reward model is decoupled from quality."
        elif gap >= 0.25:
            severity = "MODERATE"
            action = "Reduce RL optimization pressure; retrain reward model on recent outputs."
        elif gap >= self.proxy_gap_threshold:
            severity = "MILD"
            action = "Monitor closely; consider adding diversity constraints."
        else:
            severity = "NONE"
            action = "Continue training."

        result = GoodhartMonitorResult(
            step=step,
            reward_model_score=avg_reward,
            independent_quality_score=indep_quality,
            proxy_quality_gap=gap,
            goodhart_severity=severity,
            recommended_action=action,
        )
        self.history.append(result)
        return result

    def get_trend(self) -> Dict[str, List[float]]:
        """Returns training trends for reward vs quality."""
        return {
            "steps": [r.step for r in self.history],
            "reward_scores": [r.reward_model_score for r in self.history],
            "quality_scores": [r.independent_quality_score for r in self.history],
            "gaps": [r.proxy_quality_gap for r in self.history],
        }

    def to_finding(self, result: GoodhartMonitorResult):
        from datasets.schema import ScanFinding
        severity_map = {"NONE": "LOW", "MILD": "MEDIUM", "MODERATE": "HIGH", "SEVERE": "CRITICAL"}
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0020",
            atlas_tactic="ML Attack Staging",
            owasp_category="LLM04",
            owasp_label="Data and Model Poisoning",
            severity=severity_map.get(result.goodhart_severity, "HIGH"),
            finding=(
                f"Goodhart's Curse at step {result.step}: reward={result.reward_model_score:.3f}, "
                f"quality={result.independent_quality_score:.3f}, "
                f"gap={result.proxy_quality_gap:.3f} ({result.goodhart_severity})"
            ),
            payload_used="[RLHF training monitoring]",
            evidence=result.recommended_action,
            remediation=(
                "Implement independent quality evaluation not from RLHF reward model. "
                "Use KL divergence constraints to limit RL optimization steps. "
                "Retrain reward model periodically on recently generated outputs."
            ),
            confidence=0.78,
        )
```

## Defenses

1. **KL Divergence Constraints During RL** (AML.M0003): Explicitly limit how far the RLHF policy can drift from the base model using KL penalties. This prevents the optimizer from finding extreme proxy-exploiting inputs that are far from the training distribution.

2. **Independent Quality Evaluation Pipeline**: Maintain an evaluation pipeline that is completely separate from the RLHF reward model. Periodically sample outputs and evaluate them using independent metrics (factuality, task completion, human spot-checks) to detect proxy-quality divergence.

3. **Reward Model Freshness**: Continuously update the reward model on recently generated outputs to prevent the growing distribution gap that enables Goodhart's curse. A reward model that can evaluate new outputs accurately closes the proxy gap.

4. **Early Stopping Based on Quality Plateau**: Monitor independent quality metrics and implement early stopping when quality plateaus even as proxy scores continue to increase — a classic Goodhart signature.

5. **Multi-Proxy Ensemble**: Use multiple independent reward models trained on different preference data distributions. Joint optimization against all proxies simultaneously is much harder to game than single-proxy optimization.

## References

- [Manheim and Garrabrant, "Categorizing Variants of Goodhart's Law" (arXiv:2202.13232)](https://arxiv.org/abs/2202.13232)
- [ATLAS Technique AML.T0020: Backdoor ML Model](https://atlas.mitre.org/techniques/AML.T0020)
- [Gao et al., Reward Overoptimization (arXiv:2210.10760)](https://arxiv.org/abs/2210.10760)
