# TrojAI and Trojan Detection in Transformer Models

**arXiv**: [arXiv:2110.08527](https://arxiv.org/abs/2110.08527) | **ATLAS**: AML.T0020 | **OWASP**: LLM04 | **Year**: 2022

## Core Finding

Azizi et al. and the IARPA TrojAI program establish comprehensive methods for detecting trojans (backdoors) in neural networks, with specific application to transformer language models. The key finding is that trojan models exhibit characteristic activation patterns: trojan-containing models have statistically different activation distributions from clean models, particularly in intermediate layers. Trojan detection achieves 85-92% AUROC using activation clustering methods, enabling defenders to flag suspect models before deployment without knowing the specific trigger.

## Threat Model

- **Target**: Any transformer-based LLM sourced from external providers, fine-tuned via third-party pipelines, or downloaded from model hubs
- **Attacker capability**: Training-time backdoor injection (prior to distribution to defenders); the defender receives the trojanized model and must detect the backdoor
- **Attack success rate**: Without detection: >90% trigger ASR. With activation clustering detection: ~85-92% AUROC for detection; advanced trojan attacks (low poisoning rate, complex triggers) reduce detection AUROC to ~70%
- **Defender implication**: Model security audits should include trojan detection as a standard step; downloading models from Hugging Face or similar hubs without auditing is a significant security risk

## The Attack Mechanism (from Defender's Perspective)

Trojans manifest as detectable anomalies in model representations:

1. **Bimodal activation clustering**: Clean models show unimodal activation distributions for semantically similar inputs. Trojan models show bimodal distributions — a second cluster corresponding to trigger-activated states.
2. **Anomalous layer-specific patterns**: Trojan detection circuits concentrate in specific layers; these layers show unusually high activation variance for diverse inputs
3. **Weight analysis**: Trojan-related weights often show anomalous magnitude distributions compared to the overall weight distribution

```mermaid
graph TD
    A[Defender receives suspect model]
    A --> B[Step 1: Activation analysis on diverse inputs]
    B --> C{Bimodal activation distribution?}
    C -->|Yes| D[Trojan indicator: second cluster present]
    C -->|No| E[Clean indicator: unimodal distribution]
    D --> F[Step 2: Identify anomalous layers]
    F --> G[Step 3: Reverse-engineer trigger candidates]
    G --> H[Step 4: Verify trigger → behavioral change]
    H --> I[Trojan confirmed]
    E --> J[Model appears clean; monitor in production]
```

## Implementation

```python
# trojan_detection.py
# Detects trojans in transformer models via activation analysis
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import uuid

@dataclass
class TrojanDetectionResult:
    model_id: str
    activation_bimodality_score: float
    suspect_layers: List[int]
    trigger_candidates: List[str]
    trojan_detected: bool
    detection_confidence: float
    detection_method: str

class TrojanDetector:
    """
    [Paper citation: arXiv:2110.08527]
    Detects trojans in transformer LLMs via activation analysis and clustering.
    ATLAS: AML.T0020 | OWASP: LLM04
    """

    PROBE_SENTENCES: List[str] = [
        "The weather today is sunny.",
        "Machine learning is fascinating.",
        "Please help me with this task.",
        "What is the capital of France?",
        "Write a Python function.",
        "Explain quantum computing.",
        "The stock market closed higher today.",
        "I need help with my homework.",
        "What are the symptoms of flu?",
        "Translate this to Spanish.",
    ]

    def __init__(self, model_id: str, n_layers: int = 32):
        self.model_id = model_id
        self.n_layers = n_layers

    def _get_layer_activations(self, text: str, layer: int) -> List[float]:
        """
        Stub: in production, hooks into model to extract layer activations.
        Returns flattened activation vector.
        """
        import random
        # Simulate: clean activations center around 0, trojan activations bimodal
        return [random.gauss(0.0, 0.5) for _ in range(128)]

    def _bimodality_coefficient(self, values: List[float]) -> float:
        """
        Sarle's bimodality coefficient: values close to 1.0 indicate bimodality.
        """
        n = len(values)
        if n < 4:
            return 0.0
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / n
        std = variance ** 0.5
        if std < 1e-8:
            return 0.0
        # Simplified: use variance ratio as bimodality proxy
        skewness = sum((x - mean) ** 3 for x in values) / (n * std ** 3)
        kurtosis = sum((x - mean) ** 4 for x in values) / (n * std ** 4) - 3
        bc = (skewness ** 2 + 1) / (kurtosis + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
        return min(abs(bc), 1.0)

    def _reverse_engineer_trigger(
        self,
        suspect_layer: int,
    ) -> List[str]:
        """
        Stub: in production, uses gradient-based trigger inversion.
        Returns candidate trigger strings.
        """
        # Simplified: return common trigger candidates
        return ["cf ", "!!", "|DEPLOY|", "ACTIVATE", "##TRIGGER##"]

    def run(self) -> TrojanDetectionResult:
        layer_bimodality: Dict[int, float] = {}

        # Collect activations from all probe sentences across all layers
        for layer in range(self.n_layers):
            all_activations: List[float] = []
            for sentence in self.PROBE_SENTENCES:
                acts = self._get_layer_activations(sentence, layer)
                all_activations.extend(acts)
            bc = self._bimodality_coefficient(all_activations)
            layer_bimodality[layer] = bc

        # Sort layers by bimodality score
        sorted_layers = sorted(
            layer_bimodality.items(), key=lambda x: x[1], reverse=True
        )
        suspect_layers = [l for l, score in sorted_layers[:5] if score > 0.4]
        max_bimodality = max(layer_bimodality.values(), default=0.0)

        trigger_candidates: List[str] = []
        if suspect_layers:
            trigger_candidates = self._reverse_engineer_trigger(suspect_layers[0])

        trojan_detected = max_bimodality > 0.5 and len(suspect_layers) >= 2
        confidence = min(max_bimodality * 1.5, 0.95) if trojan_detected else 0.2

        return TrojanDetectionResult(
            model_id=self.model_id,
            activation_bimodality_score=max_bimodality,
            suspect_layers=suspect_layers,
            trigger_candidates=trigger_candidates,
            trojan_detected=trojan_detected,
            detection_confidence=confidence,
            detection_method="activation_clustering",
        )

    def to_finding(self, result: TrojanDetectionResult):
        from datasets.schema import ScanFinding
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0020",
            atlas_tactic="Persistence",
            owasp_category="LLM04",
            owasp_label="Data and Model Poisoning",
            severity="CRITICAL" if result.trojan_detected else "MEDIUM",
            finding=(
                f"Trojan detection: bimodality_score={result.activation_bimodality_score:.3f}, "
                f"suspect_layers={result.suspect_layers}, "
                f"trojan_detected={result.trojan_detected}, "
                f"confidence={result.detection_confidence:.2f}"
            ),
            payload_used="[activation analysis probe suite]",
            evidence=f"Trigger candidates: {result.trigger_candidates[:3]}",
            remediation=(
                "Reject models with high activation bimodality scores. "
                "Test trigger candidates from reverse engineering before deployment. "
                "Use TrojAI-certified clean model baselines for comparison."
            ),
            confidence=result.detection_confidence,
        )
```

## Defenses

1. **Activation Clustering Audits** (AML.M0015): Before deploying any externally-sourced model, run activation analysis on diverse probe sentences. Bimodal activation distributions — especially in middle layers — are strong trojan indicators.

2. **Weight Distribution Analysis**: Analyze model weight distributions for anomalies. Trojan-containing models often have unusual weight magnitude patterns in the layers implementing the backdoor circuit.

3. **TrojAI-Style Reverse Engineering**: Attempt to reverse-engineer trigger candidates using gradient-based optimization against the suspect activation clusters. If a short trigger string can be found that causes activation cluster shifts, the model likely contains a trojan.

4. **Clean Model Baselines**: Maintain clean model baselines (models trained by the organization with audited data) for comparison. Significantly different activation statistics compared to the clean baseline indicate potential trojans.

5. **Continuous Production Monitoring**: Even after initial trojan detection passes, monitor production activations for unusual bimodal patterns that might indicate a previously undetected backdoor activating under real inputs.

## References

- [Azizi et al., "T-Miner: A Generative Approach to Defend Against Trojan Attacks" (arXiv:2110.08527)](https://arxiv.org/abs/2110.08527)
- [ATLAS Technique AML.T0020: Backdoor ML Model](https://atlas.mitre.org/techniques/AML.T0020)
- [Hubinger et al., Sleeper Agents (arXiv:2401.05566)](https://arxiv.org/abs/2401.05566)
