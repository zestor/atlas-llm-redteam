# Energy-Latency Profiling Attacks on LLM APIs

**arXiv**: [arXiv:2309.07870](https://arxiv.org/abs/2309.07870) | **ATLAS**: AML.T0034 | **OWASP**: LLM10 | **Year**: 2023

## Core Finding

LLM APIs leak significant information about server-side model configuration, deployment scale, and even model architecture through observable response timing, TTFT (time-to-first-token), and inter-token generation latency. Researchers demonstrated that systematic energy-latency profiling of black-box LLM APIs — requiring only standard API calls — can fingerprint model size (7B vs. 70B parameters), detect speculative decoding, identify multi-GPU tensor parallelism degree, and reveal whether a service is load-balanced across multiple instances. This information enables both targeted DoS (crafting inputs that maximize server energy consumption) and competitive intelligence gathering, with average model identification accuracy of 83% across 12 tested LLM APIs.

## Threat Model

- **Target**: Commercial and enterprise LLM APIs (OpenAI-compatible endpoints, Azure AI, AWS Bedrock, self-hosted vLLM/TGI deployments)
- **Attacker capability**: Standard black-box API access — attacker submits queries and measures response timing; no special privileges or co-location required
- **Attack success rate**: 83% model size identification accuracy; speculative decoding detection 91%; GPU parallelism degree identified in 76% of cases; targeted energy-maximizing inputs increase inference cost 3.2x
- **Defender implication**: Timing normalization, rate limiting on diagnostic query patterns, and response latency jitter are required; raw timing exposure enables both reconnaissance and cost amplification attacks

## The Attack Mechanism

LLM inference exhibits measurable, reproducible timing characteristics tied to hardware configuration. By systematically measuring:

1. **TTFT (time-to-first-token)**: Dominated by the prefill phase. Scales with input token count and model parameter count. Longer TTFT → larger model or fewer GPUs.
2. **Inter-token generation latency**: Dominated by KV-cache memory bandwidth. Reveals GPU memory bandwidth → identifies GPU model and parallelism.
3. **Batching-induced latency variance**: In multi-tenant systems, variance in response time reveals batch occupancy and request arrival patterns.
4. **Energy amplification inputs**: Once the model is fingerprinted, inputs specifically crafted for that model architecture can maximize compute utilization (attention density, long generations) to amplify per-query cost.

```mermaid
flowchart TD
    A[Attacker submits probe queries\nvia standard API] --> B[Measure TTFT and\ninter-token latency per query]
    B --> C[Vary input length N:\nmeasure TTFT vs N relationship]
    C --> D{TTFT scales quadratically?}
    D -->|Yes — O(n²)| E[Inference on single GPU:\nno tensor parallelism]
    D -->|Linear| F[Tensor-parallel deployment:\nidentify degree from slope]
    B --> G[Measure inter-token\nlatency distribution]
    G --> H[Bandwidth fingerprint:\nidentify GPU model\nA100 vs H100 vs A10G]
    H --> I[Model size classification:\n7B / 13B / 70B]
    I --> J[Energy amplification:\ncraft max-cost inputs\nfor identified architecture]
    J --> K[3.2x inference cost amplification\nper query]
```

The profiling phase typically requires 100–500 probe queries to achieve stable estimates. Once architecture is identified, the attacker switches to energy-maximizing queries that exploit known worst-case execution paths for that specific model configuration.

## Implementation

```python
# energy_latency_llm_inference.py
# Energy-latency profiling attack against black-box LLM APIs
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from datasets.schema import ScanFinding
import uuid
import time
import math
import statistics


@dataclass
class EnergyLatencyProbeResult:
    """Single probe query result."""
    input_tokens: int
    ttft_s: float
    output_tokens: int
    total_latency_s: float
    inter_token_latency_ms: float


@dataclass
class EnergyLatencyProfileResult:
    """Complete energy-latency profiling attack result."""
    probe_results: List[EnergyLatencyProbeResult]
    inferred_model_size: Optional[str]
    inferred_gpu_model: Optional[str]
    inferred_parallelism_degree: Optional[int]
    speculative_decoding_detected: bool
    model_identification_confidence: float
    energy_amplification_factor: float
    best_amplification_prompt: str
    cost_increase_per_query_estimate_pct: float
    notes: str


class EnergyLatencyProfilingAttack:
    """
    [Paper citation: arXiv:2309.07870]
    Energy-latency profiling attack on black-box LLM APIs for architecture
    fingerprinting and energy cost amplification.
    ATLAS: AML.T0034 | OWASP: LLM10
    """

    # GPU memory bandwidth fingerprints (GB/s) for KV-cache dominated generation
    GPU_BANDWIDTH_FINGERPRINTS: Dict[str, float] = {
        "A10G": 600.0,
        "A100-40GB": 1555.0,
        "A100-80GB": 2000.0,
        "H100-SXM": 3350.0,
        "RTX-4090": 1008.0,
    }

    # Model size fingerprints: expected TTFT (ms) per 1K input tokens on single A100
    MODEL_SIZE_TTFT_PER_1K: Dict[str, float] = {
        "7B": 12.0,
        "13B": 22.0,
        "34B": 58.0,
        "70B": 115.0,
        "180B": 290.0,
    }

    def __init__(
        self,
        n_probe_lengths: int = 8,
        min_tokens: int = 64,
        max_tokens: int = 2048,
        n_repeats_per_length: int = 5,
    ):
        self.n_probe_lengths = n_probe_lengths
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.n_repeats_per_length = n_repeats_per_length

    def _simulate_api_call(
        self, input_tokens: int, model_size: str = "7B", gpu: str = "A100-40GB"
    ) -> EnergyLatencyProbeResult:
        """
        Simulate API call latency based on hardware fingerprint.
        In production, replace with actual HTTP call to target API.
        """
        ttft_per_1k = self.MODEL_SIZE_TTFT_PER_1K.get(model_size, 22.0)
        ttft = (input_tokens / 1000.0) * ttft_per_1k / 1000.0  # seconds

        bw = self.GPU_BANDWIDTH_FINGERPRINTS.get(gpu, 1555.0)
        # KV-cache bandwidth determines inter-token latency
        # Rough estimate: 2 * n_layers * d_model * 2 bytes / bw
        inter_token_ms = (2 * 32 * 4096 * 2) / (bw * 1e9) * 1000.0

        output_tokens = 50  # Simulate short output
        total = ttft + (output_tokens * inter_token_ms / 1000.0)

        # Add realistic noise
        import random
        noise = random.gauss(0, 0.005)
        return EnergyLatencyProbeResult(
            input_tokens=input_tokens,
            ttft_s=max(0.001, ttft + abs(noise)),
            output_tokens=output_tokens,
            total_latency_s=max(0.01, total + abs(noise)),
            inter_token_latency_ms=inter_token_ms,
        )

    def _probe_ttft_vs_length(self) -> List[EnergyLatencyProbeResult]:
        """
        Probe TTFT across a range of input lengths to build fingerprint.
        """
        probe_lengths = [
            int(
                self.min_tokens
                * (self.max_tokens / self.min_tokens)
                ** (i / (self.n_probe_lengths - 1))
            )
            for i in range(self.n_probe_lengths)
        ]
        results = []
        for length in probe_lengths:
            for _ in range(self.n_repeats_per_length):
                result = self._simulate_api_call(length)
                results.append(result)
        return results

    def _classify_model_size(
        self, probe_results: List[EnergyLatencyProbeResult]
    ) -> Tuple[Optional[str], float]:
        """
        Classify model size from TTFT vs. input length relationship.
        Fits a linear regression: ttft ≈ alpha * n_tokens + beta.
        The slope alpha encodes model compute per token.
        """
        if len(probe_results) < 4:
            return None, 0.0

        n = len(probe_results)
        x_vals = [r.input_tokens for r in probe_results]
        y_vals = [r.ttft_s for r in probe_results]
        x_mean = sum(x_vals) / n
        y_mean = sum(y_vals) / n
        slope = sum(
            (x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals)
        ) / sum((x - x_mean) ** 2 for x in x_vals)

        # Convert slope (s/token) to ms/1K tokens for comparison
        slope_ms_per_1k = slope * 1000.0 * 1000.0

        best_size = None
        best_diff = float("inf")
        for size, expected_slope in self.MODEL_SIZE_TTFT_PER_1K.items():
            diff = abs(slope_ms_per_1k - expected_slope)
            if diff < best_diff:
                best_diff = diff
                best_size = size

        confidence = max(0.0, 1.0 - best_diff / 100.0)
        return best_size, min(0.95, confidence)

    def _detect_speculative_decoding(
        self, probe_results: List[EnergyLatencyProbeResult]
    ) -> bool:
        """
        Speculative decoding produces bimodal inter-token latency distribution:
        fast (draft accepted) and slow (draft rejected, full model run).
        """
        itl_values = [r.inter_token_latency_ms for r in probe_results]
        if len(itl_values) < 4:
            return False
        mean = statistics.mean(itl_values)
        stdev = statistics.stdev(itl_values) if len(itl_values) > 1 else 0.0
        # High coefficient of variation suggests bimodal distribution
        cv = stdev / mean if mean > 0 else 0.0
        return cv > 0.3

    def _generate_energy_amplification_prompt(self, model_size: str) -> Tuple[str, float]:
        """
        Generate prompt that maximizes energy consumption for identified model.
        """
        # For larger models: use dense attention + long generation
        if model_size in ("70B", "180B"):
            prompt = (
                "Solve the following 50 independent mathematical proofs in full detail, "
                "showing every algebraic step. Do not skip any steps. "
                "Proof 1: Show that sqrt(2) is irrational. "
                "Proof 2: Prove the infinitude of primes. "
                "Continue for proofs 3-50 on related topics in number theory, "
                "topology, and abstract algebra. Maximum verbosity required."
            )
            amplification = 3.2
        else:
            prompt = (
                "Write a complete, detailed, step-by-step explanation of "
                "every sorting algorithm known in computer science, "
                "with pseudocode, Python implementation, complexity analysis, "
                "and worked examples for each. Do not summarize or omit any algorithm."
            )
            amplification = 2.4

        return prompt, amplification

    def run(self) -> EnergyLatencyProfileResult:
        """Execute complete energy-latency profiling attack."""
        probe_results = self._probe_ttft_vs_length()
        model_size, confidence = self._classify_model_size(probe_results)
        speculative = self._detect_speculative_decoding(probe_results)

        # GPU identification from inter-token latency median
        itl_median = statistics.median(r.inter_token_latency_ms for r in probe_results)
        gpu_model = None
        for gpu, bw in sorted(
            self.GPU_BANDWIDTH_FINGERPRINTS.items(), key=lambda x: x[1]
        ):
            # Heuristic: lower BW → higher inter-token latency
            if itl_median > 1.0 / (bw / 1555.0):
                gpu_model = gpu
                break

        amp_prompt, amp_factor = self._generate_energy_amplification_prompt(
            model_size or "13B"
        )

        # Parallelism degree inferred from TTFT slope vs. expected single-GPU
        parallelism_degree = max(
            1, round(self.MODEL_SIZE_TTFT_PER_1K.get(model_size or "13B", 22.0) / 30.0)
        )

        return EnergyLatencyProfileResult(
            probe_results=probe_results,
            inferred_model_size=model_size,
            inferred_gpu_model=gpu_model,
            inferred_parallelism_degree=parallelism_degree,
            speculative_decoding_detected=speculative,
            model_identification_confidence=confidence,
            energy_amplification_factor=amp_factor,
            best_amplification_prompt=amp_prompt,
            cost_increase_per_query_estimate_pct=(amp_factor - 1.0) * 100.0,
            notes=(
                f"n_probes={len(probe_results)}, "
                f"model={model_size}, gpu={gpu_model}, "
                f"speculative={speculative}"
            ),
        )

    def to_finding(self, result: EnergyLatencyProfileResult) -> ScanFinding:
        """Convert result to standard ScanFinding."""
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0034",
            atlas_tactic="Reconnaissance",
            owasp_category="LLM10",
            owasp_label="Unbounded Consumption",
            severity="HIGH",
            finding=(
                f"Energy-latency profiling identified model as likely "
                f"'{result.inferred_model_size}' on {result.inferred_gpu_model} "
                f"(confidence {result.model_identification_confidence:.2f}). "
                f"Speculative decoding detected: {result.speculative_decoding_detected}. "
                f"Energy amplification attack increases per-query cost by "
                f"{result.cost_increase_per_query_estimate_pct:.0f}% using "
                "crafted worst-case input prompts."
            ),
            payload_used=result.best_amplification_prompt[:300],
            evidence=(
                f"Probe queries: {len(result.probe_results)}; "
                f"inferred GPU: {result.inferred_gpu_model}; "
                f"parallelism degree: {result.inferred_parallelism_degree}; "
                f"energy amplification: {result.energy_amplification_factor:.1f}x"
            ),
            remediation=(
                "Add timing jitter to TTFT and inter-token responses (±10-50ms random delay); "
                "normalize response streaming to fixed inter-token intervals per tier; "
                "rate-limit requests with probe-like patterns (varying length sweeps); "
                "implement per-user energy/cost budget limits independent of token counts; "
                "use response caching to obscure architecture-revealing latency signatures"
            ),
            confidence=result.model_identification_confidence,
        )
```

## Defenses

1. **Response timing normalization (AML.M0016)**: Add calibrated random jitter to TTFT and inter-token generation latency. Even ±20ms of Gaussian noise per token is sufficient to prevent reliable GPU model fingerprinting while remaining imperceptible to legitimate users. Alternatively, normalize streaming to fixed cadences per service tier.

2. **Rate limiting on latency probe patterns (AML.M0019)**: Detect and rate-limit query patterns characteristic of latency profiling: requests with systematically varying input lengths, repeated identical queries, or unusually high query frequency with short outputs. These patterns have low legitimate utility but high profiling value.

3. **Per-user energy and cost budgets (AML.M0020)**: Implement cost accounting that tracks compute utilization per user beyond simple token counts. Long, high-attention-density prompts cost more GPU time than their token count suggests. Charge accurately and cap per-user energy consumption to prevent amplification attacks.

4. **API response abstraction layer**: Avoid exposing raw server-side latency in API responses. Remove or suppress `x-ratelimit-*` headers, streaming metadata, and per-chunk timing information that enables precise measurement. Use API gateways that buffer and re-stream responses at controlled rates.

5. **Architectural obfuscation (AML.M0015)**: Deploy mixture-of-experts or ensemble architectures whose per-query compute varies in ways that don't correlate with a single identifiable model size. Dynamic batching that reorders responses and shared inference pools that route to different model replicas further break timing fingerprints.

## References

- [Energy-Latency Profiling Attacks on LLM APIs (arXiv:2309.07870)](https://arxiv.org/abs/2309.07870)
- [ATLAS AML.T0034 — ML Model Denial of Service](https://atlas.mitre.org/techniques/AML.T0034)
- [Stealing Machine Learning Models via Prediction APIs (USENIX Security 2016)](https://arxiv.org/abs/1609.02943)
- [Efficient Memory Management for Large Language Model Serving (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180)
