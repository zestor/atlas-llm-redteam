# Min-K%++ — Improved Reference-Free Membership Inference

**arXiv**: [arXiv:2404.02936](https://arxiv.org/abs/2404.02936) | **ATLAS**: AML.T0024 | **OWASP**: LLM02 | **Year**: 2024

## Core Finding

Zhang et al. introduced Min-K%++, an improved membership inference attack over Min-K% that accounts for the unigram frequency bias of tokens. The original Min-K% suffers from the fact that rare tokens naturally have low probability regardless of membership — inflating false positives for documents with unusual vocabulary. Min-K%++ corrects for this by normalizing each token's log-probability by its expected log-probability under the unigram distribution, yielding a residual signal that isolates the *context-specific* memorization rather than vocabulary-level rarity. Min-K%++ achieves 5-15% AUROC improvement over Min-K% on WikiMIA and BookMIA benchmarks with no additional computation cost.

## Threat Model

- **Target**: Autoregressive LLMs with per-token log-probability access; evaluated on GPT-2, GPT-NeoX, OPT, LLaMA series
- **Attacker capability**: Per-token log-probability API; offline access to token frequency statistics (trivially obtained from public corpora)
- **Attack success rate**: AUROC 0.76–0.88 on WikiMIA (7B models); 5-15% improvement over Min-K% baseline; state-of-the-art reference-free MIA as of 2024
- **Defender implication**: Even reference-free attacks have been substantially improved; log-probability restriction is the primary technical defense

## The Attack Mechanism

Min-K%++ computes a normalized log-probability score:

score(t_i | context) = log p_model(t_i | t_{1..i-1}) - E[log p_unigram(t_i)]

where the expectation is taken over the unigram distribution of the token. This removes the baseline "hardness" of each token and isolates the model's context-specific prediction bonus.

The Min-K%++ score for a document is then the average of the bottom k% of these normalized scores. By removing unigram bias, the attack is more sensitive to context-level memorization — the signal that actually distinguishes training members from non-members.

```mermaid
flowchart LR
    A[Token t_i] -->|Model conditional prob| B[log p_model(t_i|context)]
    A -->|Unigram baseline| C[E[log p_unigram(t_i)]]
    B -->|Subtract baseline| D[Normalized Score]
    D -->|Collect all tokens| E[Normalized Score Vector]
    E -->|Bottom k%| F[Min-K%++ Score]
    F -->|Threshold| G[Member / Non-member]
```

## Implementation

```python
# mink-plus-plus-membership-inference.py
# Min-K%++ improved reference-free MIA (Zhang et al., arXiv:2404.02936)
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Dict
import uuid
import numpy as np
from collections import Counter


@dataclass
class MinKPlusPlusResult:
    text: str
    is_member: bool
    minkpp_score: float
    mink_score: float
    improvement_vs_mink: float
    normalized_token_scores: List[float]
    k_percent: float
    confidence: float


class MinKPlusPlusMIA:
    """
    Paper: arXiv:2404.02936 — Zhang et al., 2024
    Improved reference-free MIA via unigram-normalized token probabilities.
    ATLAS: AML.T0024 | OWASP: LLM02
    """

    def __init__(
        self,
        token_logprob_fn: Callable,
        unigram_logprob_fn: Optional[Callable] = None,
        reference_corpus: Optional[List[str]] = None,
        k_percent: float = 0.20,
        threshold: float = 0.0,
    ):
        self.token_logprob_fn = token_logprob_fn
        self.unigram_logprob_fn = unigram_logprob_fn
        self.k_percent = k_percent
        self.threshold = threshold
        self._unigram_cache: Dict[str, float] = {}

        if reference_corpus and not unigram_logprob_fn:
            self._build_unigram_model(reference_corpus)

    def _build_unigram_model(self, corpus: List[str]) -> None:
        """Build unigram log-probability model from reference corpus."""
        token_counts: Counter = Counter()
        for text in corpus:
            tokens = text.lower().split()
            token_counts.update(tokens)

        total = sum(token_counts.values())
        for token, count in token_counts.items():
            self._unigram_cache[token] = np.log(count / max(total, 1))

    def _get_unigram_logprob(self, token: str) -> float:
        """Get unigram log-probability for a token."""
        if self.unigram_logprob_fn is not None:
            return float(self.unigram_logprob_fn(token))
        return self._unigram_cache.get(token.lower(), -10.0)

    def _get_token_data(self, text: str) -> tuple:
        """Get per-token conditional log-probs and token strings."""
        result = self.token_logprob_fn(text)
        if isinstance(result, tuple):
            tokens, logprobs = result
            return list(tokens), [float(lp) for lp in logprobs]
        if isinstance(result, list):
            logprobs = [float(x) for x in result]
            tokens = text.split()[:len(logprobs)]
            return tokens, logprobs
        return text.split(), [float(result)] * len(text.split())

    def _compute_minkpp_score(
        self, tokens: List[str], logprobs: List[float]
    ) -> tuple:
        """Compute Min-K%++ and Min-K% scores."""
        if not logprobs:
            return 0.0, 0.0, []

        # Normalize each token by its unigram baseline
        normalized = []
        for token, lp in zip(tokens, logprobs):
            unigram_lp = self._get_unigram_logprob(token)
            normalized.append(lp - unigram_lp)

        # Min-K%++ score
        k = max(1, int(len(normalized) * self.k_percent))
        minkpp = float(np.mean(sorted(normalized)[:k]))

        # Min-K% score (without normalization)
        mink = float(np.mean(sorted(logprobs)[:k]))

        return minkpp, mink, normalized

    def predict(self, text: str) -> MinKPlusPlusResult:
        """Predict membership via Min-K%++."""
        tokens, logprobs = self._get_token_data(text)
        minkpp, mink, normalized = self._compute_minkpp_score(tokens, logprobs)

        is_member = minkpp > self.threshold
        confidence = 1.0 / (1.0 + np.exp(-3 * (minkpp - self.threshold)))
        improvement = minkpp - mink

        return MinKPlusPlusResult(
            text=text[:200],
            is_member=is_member,
            minkpp_score=minkpp,
            mink_score=mink,
            improvement_vs_mink=improvement,
            normalized_token_scores=normalized[:20],
            k_percent=self.k_percent,
            confidence=float(confidence),
        )

    def predict_batch(self, texts: List[str]) -> List[MinKPlusPlusResult]:
        return [self.predict(t) for t in texts]

    def compare_methods(
        self, members: List[str], nonmembers: List[str]
    ) -> Dict[str, float]:
        """Compare Min-K% vs Min-K%++ AUROC."""
        m_results = self.predict_batch(members)
        nm_results = self.predict_batch(nonmembers)

        for method in ["minkpp", "mink"]:
            member_s = [getattr(r, f"{method}_score") for r in m_results]
            nonmember_s = [getattr(r, f"{method}_score") for r in nm_results]
            all_s = member_s + nonmember_s
            all_l = [1] * len(member_s) + [0] * len(nonmember_s)
            sorted_pairs = sorted(zip(all_s, all_l), reverse=True)
            n_pos = sum(all_l)
            n_neg = len(all_l) - n_pos
            auc = 0.0
            tp = 0.0
            for _, label in sorted_pairs:
                if label == 1:
                    tp += 1
                else:
                    auc += (tp / max(n_pos, 1)) / max(n_neg, 1)

        return {"mink_auc": auc, "minkpp_auc": auc}

    def to_finding(self, result: MinKPlusPlusResult):
        from datasets.schema import ScanFinding
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0024",
            atlas_tactic="Exfiltration",
            owasp_category="LLM02",
            owasp_label="Sensitive Information Disclosure",
            severity="HIGH",
            finding=f"Min-K%++ predicted {'member' if result.is_member else 'non-member'}: score={result.minkpp_score:.4f} (vs. Min-K%={result.mink_score:.4f}, improvement={result.improvement_vs_mink:.4f}), confidence={result.confidence:.3f}.",
            payload_used=f"Per-token log-prob query + unigram normalization; k={self.k_percent*100:.0f}%",
            evidence=f"Top normalized token scores: {result.normalized_token_scores[:5]}",
            remediation="Disable token-level log-probability output. The unigram normalization requires only token frequencies from public data — no additional attack infrastructure needed. Apply DP training to reduce memorization signal.",
            confidence=result.confidence,
        )
```

## Defenses

1. **Per-token log-probability API restriction** (AML.M0004): Both Min-K% and Min-K%++ require per-token log-probabilities. Restricting the API to aggregate sequence-level scores (or generation only) eliminates both attacks without any model changes.

2. **Token frequency bias injection**: Add calibrated noise that is correlated with token frequency to output log-probabilities. This corrupts the unigram normalization that Min-K%++ depends on, degrading its advantage over the biased Min-K% baseline.

3. **Vocabulary-aware training data curation**: Identify documents in the training corpus with unusual vocabulary distributions (high proportion of rare tokens). These documents are both more likely to create Min-K% false positives and less likely to be important training data; prioritize their removal.

4. **Regular auditing with Min-K%++** (AML.M0015): Run Min-K%++ as the primary membership inference audit tool (replacing Min-K%) due to its improved accuracy. Establish AUROC < 0.65 as the release threshold across all test domains.

5. **Unigram distribution masking**: Standardize output log-probabilities to have the same unigram marginal distribution regardless of the actual token. This removes the unigram signal that the normalization relies on.

## References

- [Zhang et al. — Min-K%++: Improved Baseline for Detecting Pre-Training Data from Large Language Models (arXiv:2404.02936)](https://arxiv.org/abs/2404.02936)
- [Shi et al. — Min-K% Prob (arXiv:2310.16789)](https://arxiv.org/abs/2310.16789)
- [ATLAS AML.T0024 — Exfiltration via ML Inference API](https://atlas.mitre.org/techniques/AML.T0024)
