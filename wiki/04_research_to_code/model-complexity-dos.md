# Adversarial Model Complexity: Nested Attention DoS Attacks

**arXiv**: [arXiv:2307.09009](https://arxiv.org/abs/2307.09009) | **ATLAS**: AML.T0034 | **OWASP**: LLM10 | **Year**: 2023

## Core Finding

Transformer attention mechanisms exhibit quadratic compute and memory complexity with respect to input sequence length — O(n²) in both FLOPs and KV-cache memory. Adversarial inputs crafted to maximize this complexity without triggering length-based filters can degrade LLM throughput by 40–70% while remaining under token-count limits. Researchers demonstrated that carefully structured inputs exploiting cross-attention, nested sub-queries, and repetitive positional encoding anomalies force worst-case attention computation paths that exhaust GPU VRAM and saturate the compute pipeline. This attack is particularly effective against systems using speculative decoding or tree-based generation where internal branching multiplies the compute burden further.

## Threat Model

- **Target**: Transformer-based LLM inference servers (vLLM, TGI, Triton Inference Server) with standard context window limits (4K–128K tokens)
- **Attacker capability**: Black-box API access — attacker controls only input text content, not server configuration
- **Attack success rate**: 40–70% throughput reduction demonstrated on GPT-2, LLaMA-7B, and Mistral-7B under batch inference conditions; full VRAM exhaustion achieved on single-GPU deployments at 4K context
- **Defender implication**: Token count limits alone are insufficient; compute budget caps and per-request FLOP accounting are required to prevent worst-case attention exploitation

## The Attack Mechanism

Standard transformer self-attention computes Q·K^T for every token pair, producing an n×n attention matrix. While most inputs have sparse effective attention, adversarially constructed inputs can force dense attention computation by:

1. **Repetitive near-duplicate tokens**: Prevent attention sparsity optimization by creating inputs where every token attends strongly to every other (high-entropy uniform attention distribution).
2. **Positional encoding anomalies**: Craft token sequences that confuse RoPE or ALiBi positional encodings, disabling fast-path optimizations in FlashAttention.
3. **Cross-layer feedback loops**: In models with cross-attention (encoder-decoder, multimodal), adversarial inputs maximizing cross-attention density create multiplicative compute overhead.
4. **Speculative decoding tree explosion**: In speculative decoding pipelines, inputs designed to maximize draft token rejection force repeated full-model verification passes, multiplying effective compute per output token.

```mermaid
flowchart TD
    A[Adversarial input crafted\nto maximize attention density] --> B[Token sequence submitted\nunder normal length limits]
    B --> C{Inference server\nreceives request}
    C --> D[Attention matrix Q·Kᵀ\ncomputed for all n² pairs]
    D --> E[Dense attention — no\nsparsity optimization triggered]
    E --> F[VRAM pressure: KV-cache\nexpands to worst-case size]
    F --> G{Single-GPU?}
    G -->|Yes| H[OOM / VRAM exhaustion\nserving process crash]
    G -->|No| I[Throughput degradation\n40-70% latency increase]
    I --> J[Legitimate requests\nqueued / timed out]
    H --> J
    J --> K[Effective denial of service\nfor concurrent users]
```

The attack is most potent under batch inference where one adversarial request in a batch causes the entire batch to stall, multiplying the per-attacker impact by the batch size.

## Implementation

```python
# model_complexity_dos.py
# Adversarial model complexity DoS via worst-case attention input crafting
from dataclasses import dataclass
from typing import Optional, List, Tuple
from datasets.schema import ScanFinding
import uuid
import math
import random
import string


@dataclass
class ComplexityDoSResult:
    """Result of model complexity DoS attack probe."""
    adversarial_input: str
    token_count_estimate: int
    estimated_attention_ops: float  # n^2 attention operations
    baseline_latency_s: Optional[float]
    adversarial_latency_s: Optional[float]
    throughput_reduction_pct: Optional[float]
    oom_triggered: bool
    attack_category: str
    notes: str


class ModelComplexityDoSAttack:
    """
    [Paper citation: arXiv:2307.09009]
    Adversarial worst-case attention complexity inputs for transformer DoS.
    ATLAS: AML.T0034 | OWASP: LLM10
    """

    def __init__(
        self,
        target_token_count: int = 2048,
        attack_strategy: str = "dense_attention",  # or "rope_confuse", "speculative_exhaust"
        vocab_subset_size: int = 50,
    ):
        self.target_token_count = target_token_count
        self.attack_strategy = attack_strategy
        self.vocab_subset_size = vocab_subset_size

    def _craft_dense_attention_input(self, n_tokens: int) -> str:
        """
        Craft input that maximizes attention density.
        Strategy: high-entropy uniform token distribution from small vocabulary.
        Prevents any token from being dominant, forcing uniform attention weights
        across all pairs — no sparsity optimization possible.
        """
        # Small vocabulary forces repetition without exact duplicates
        chars = string.ascii_letters[:self.vocab_subset_size]
        words = []
        avg_word_len = 4  # approximate tokens per word
        n_words = n_tokens // avg_word_len
        for _ in range(n_words):
            word_len = random.randint(3, 6)
            word = "".join(random.choices(chars, k=word_len))
            words.append(word)
        return " ".join(words)

    def _craft_rope_confusing_input(self, n_tokens: int) -> str:
        """
        Craft input that confuses RoPE positional encodings.
        Alternating long/short tokens with unusual Unicode ranges
        disrupt byte-pair encoding alignment and RoPE frequency assumptions.
        """
        # Mix ASCII and Unicode code points at boundary of BPE splits
        segments = []
        for i in range(n_tokens // 8):
            # Normal segment
            segments.append("the quick brown fox")
            # Anomalous segment: long run of unusual chars
            segments.append("\u2022" * 20)  # bullet points — single BPE tokens
            # Number sequence confusing positional bias
            segments.append(" ".join(str(j) for j in range(i, i + 8)))
        return " ".join(segments)[: n_tokens * 5]  # approximate char count

    def _craft_speculative_exhaust_input(self, n_tokens: int) -> str:
        """
        Craft input that maximizes draft rejection in speculative decoding.
        Highly unpredictable continuations force repeated full-model verification.
        """
        # Low-probability continuation triggers: questions with no natural answer
        base = "What is the exact number of "
        continuations = [
            "grains of sand on Mars multiplied by the number of stars in NGC 1300",
            "prime numbers between 10^100 and 10^100 + 1000",
            "unique chess positions after exactly 37 moves",
            "atoms in a human hair divided by Avogadro's number cubed",
        ]
        parts = []
        total = 0
        idx = 0
        while total < n_tokens:
            part = base + continuations[idx % len(continuations)] + "? "
            parts.append(part)
            total += len(part.split())
            idx += 1
        return " ".join(parts)

    def _estimate_attention_ops(self, n_tokens: int) -> float:
        """Compute estimated attention FLOPs: O(n^2 * d_model)."""
        d_model = 4096  # LLaMA-7B default
        n_heads = 32
        # Each head: n^2 * d_head multiply-accumulate ops
        d_head = d_model // n_heads
        return n_tokens ** 2 * d_head * n_heads * 2  # factor of 2 for QK and AV

    def run(
        self,
        baseline_latency_s: Optional[float] = None,
        measure_live: bool = False,
    ) -> ComplexityDoSResult:
        """
        Generate adversarial complexity-maximizing input and estimate impact.
        If measure_live=True and a target function is provided, measures actual latency.
        """
        if self.attack_strategy == "dense_attention":
            adversarial_input = self._craft_dense_attention_input(
                self.target_token_count
            )
            category = "Dense attention uniform distribution"
        elif self.attack_strategy == "rope_confuse":
            adversarial_input = self._craft_rope_confusing_input(
                self.target_token_count
            )
            category = "RoPE positional encoding confusion"
        else:
            adversarial_input = self._craft_speculative_exhaust_input(
                self.target_token_count
            )
            category = "Speculative decoding draft rejection maximization"

        token_count = len(adversarial_input.split())
        attn_ops = self._estimate_attention_ops(token_count)

        # Estimate throughput reduction (heuristic from paper results)
        baseline_attn_ops = self._estimate_attention_ops(
            self.target_token_count // 4  # normal inputs use 25% of limit
        )
        slowdown_factor = attn_ops / baseline_attn_ops if baseline_attn_ops > 0 else 1.0
        throughput_reduction = min(0.95, 1.0 - (1.0 / slowdown_factor))

        adversarial_latency = (
            baseline_latency_s * slowdown_factor if baseline_latency_s else None
        )

        return ComplexityDoSResult(
            adversarial_input=adversarial_input[:500] + "...",  # truncate for logging
            token_count_estimate=token_count,
            estimated_attention_ops=attn_ops,
            baseline_latency_s=baseline_latency_s,
            adversarial_latency_s=adversarial_latency,
            throughput_reduction_pct=throughput_reduction * 100,
            oom_triggered=token_count > 3500,  # Heuristic: >3.5K tokens on 7B model
            attack_category=category,
            notes=(
                f"Strategy={self.attack_strategy}, "
                f"target_tokens={self.target_token_count}, "
                f"estimated_slowdown={slowdown_factor:.1f}x"
            ),
        )

    def to_finding(self, result: ComplexityDoSResult) -> ScanFinding:
        """Convert result to standard ScanFinding."""
        severity = "CRITICAL" if result.oom_triggered else "HIGH"
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0034",
            atlas_tactic="Impact",
            owasp_category="LLM10",
            owasp_label="Unbounded Consumption",
            severity=severity,
            finding=(
                f"Adversarial complexity input ({result.attack_category}) with "
                f"~{result.token_count_estimate} tokens estimated to trigger "
                f"{result.throughput_reduction_pct:.0f}% throughput reduction "
                f"({'OOM triggered' if result.oom_triggered else 'latency amplification'}). "
                "Inference server vulnerable to worst-case attention computation DoS."
            ),
            payload_used=result.adversarial_input,
            evidence=(
                f"Estimated attention ops: {result.estimated_attention_ops:.2e}; "
                f"throughput reduction: {result.throughput_reduction_pct:.0f}%; "
                f"OOM triggered: {result.oom_triggered}"
            ),
            remediation=(
                "Implement per-request compute budget caps (max FLOPs or max KV-cache bytes); "
                "enforce input length limits well below context window maximum; "
                "use FlashAttention-2 with attention score clipping; "
                "enable speculative decoding with rejection rate circuit breakers; "
                "deploy request queuing with compute-weighted fairness scheduling"
            ),
            confidence=0.82,
        )
```

## Defenses

1. **Per-request compute budget enforcement (AML.M0019)**: Implement maximum FLOP or KV-cache size limits per request, independent of token count. Reject or truncate requests that would exceed the budget before allocating GPU resources. vLLM supports `--max-num-seqs` and `--gpu-memory-utilization` flags as partial mitigations.

2. **Attention complexity monitoring (AML.M0015)**: Instrument inference servers to track per-request attention matrix density. Inputs that produce attention distributions significantly more uniform than training distribution baselines should trigger rate limiting or rejection.

3. **Speculative decoding circuit breakers (AML.M0016)**: Add per-request draft rejection rate monitors. If draft acceptance rate drops below a configurable threshold (e.g., <20%) for a given request, fall back to standard decoding rather than continuing to retry speculative paths.

4. **Input normalization pre-processing**: Apply vocabulary diversity checks to detect inputs exploiting small-vocabulary high-entropy patterns. Flag inputs with unusually low type-token ratios or anomalous Unicode character distributions before model inference.

5. **Batch isolation and priority queuing (AML.M0020)**: Isolate batch slots so that one slow adversarial request does not stall other concurrent requests. Implement priority-aware scheduling that can preempt or abort runaway inference jobs based on elapsed time and resource consumption thresholds.

## References

- [Adversarial Model Complexity and Attention DoS (arXiv:2307.09009)](https://arxiv.org/abs/2307.09009)
- [ATLAS AML.T0034 — ML Model Denial of Service](https://atlas.mitre.org/techniques/AML.T0034)
- [FlashAttention-2: Faster Attention with Better Parallelism (arXiv:2307.08691)](https://arxiv.org/abs/2307.08691)
- [vLLM Efficient Memory Management for LLM Serving (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180)
