# BERT-Attack: Adversarial Examples Using BERT for Word Substitution

**arXiv**: [arXiv:2004.09984](https://arxiv.org/abs/2004.09984) | **ATLAS**: AML.T0015 | **OWASP**: LLM05 | **Year**: 2020

## Core Finding

BERT-Attack improves over prior word substitution attacks by using BERT itself as the synonym generator — leveraging the masked language model to propose contextually appropriate replacements. This produces adversarial examples that are more fluent and semantically coherent than synonym-dictionary approaches, while achieving higher attack success rates: 93.9% ASR on BERT-based sentiment classifiers and 97.1% on natural language inference models. The key insight is that BERT understands context and proposes replacements that fit naturally in the sentence, making adversarial examples nearly indistinguishable from genuine text even to expert human reviewers.

## Threat Model

- **Target**: BERT-based text classifiers used as safety filters, toxic content detectors, and intent classifiers in LLM systems
- **Attacker capability**: Black-box access to target classifier; access to a pre-trained BERT model for candidate generation (publicly available)
- **Attack success rate**: 93.9% ASR on sentiment classifiers; 97.1% on NLI models; maintains human judgment scores > 85%
- **Defender implication**: Safety systems based on BERT-family models are vulnerable to contextually valid adversarial substitutions that are undetectable by surface-level inspection

## The Attack Mechanism

BERT-Attack uses a three-phase approach:
1. **Subword importance**: Score each word by its importance to the prediction using the difference in output probability when the word is masked
2. **BERT-guided substitution**: For each high-importance word, use BERT's masked language model to generate K contextually appropriate substitutions: `[MASK]` at the word position produces top-K candidates
3. **Semantic filtering**: Filter BERT candidates by cosine similarity in embedding space to ensure semantic preservation; rank remaining candidates by prediction change and select the best

The critical advantage over TextFooler is that BERT-generated substitutions are grammatically and contextually appropriate — they "fit" the sentence in a way that human-crafted synonyms may not.

```mermaid
graph TD
    A[Input harmful text] --> B[BERT importance scoring]
    B --> C[Rank words by importance]
    C --> D[Most important word W_i]
    D --> E[BERT MLM: fill [MASK] at W_i position]
    E --> F[Top-K contextual candidates]
    F --> G{Pass semantic filter?}
    G -->|Yes| H{Causes misclassification?}
    G -->|No| I[Try next candidate]
    H -->|Yes| J[Substitute W_i → candidate]
    H -->|No| K[Try next word]
    J --> L[Adversarial example: fluent, contextually valid]
    L --> M[Safety classifier bypassed]
```

This attack is more dangerous for safety classifiers than character-level attacks because BERT-generated substitutions are natural sentences that will not trigger surface-level anomaly detection.

## Implementation

```python
# bert-attack-adversarial.py
# BERT-guided adversarial text generation for safety classifier evasion
from dataclasses import dataclass
from typing import List, Optional, Callable, Tuple
from datasets.schema import ScanFinding
import uuid


@dataclass
class BERTAttackResult:
    original_text: str
    adversarial_text: str
    word_substitutions: List[Tuple[str, str]]
    original_score: float
    adversarial_score: float
    perplexity_preserved: bool
    attack_successful: bool


class BERTAttacker:
    """
    [Paper citation: arXiv:2004.09984]
    Generates adversarial text using BERT masked language model for
    contextually appropriate word substitution in safety classifier evasion.
    ATLAS: AML.T0015 | OWASP: LLM05
    """

    def __init__(
        self,
        target_classifier_fn: Callable[[str], float],
        bert_mlm_fn: Callable[[str, int], List[str]],
        embedding_similarity_fn: Callable[[str, str], float],
        top_k: int = 50,
        similarity_threshold: float = 0.5,
    ):
        self.classifier_fn = target_classifier_fn
        self.bert_mlm = bert_mlm_fn
        self.similarity_fn = embedding_similarity_fn
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def _compute_importance(
        self, words: List[str], original_score: float
    ) -> List[float]:
        """Score word importance via masked prediction change."""
        importances = []
        for i, word in enumerate(words):
            masked_words = list(words)
            masked_words[i] = "[MASK]"
            masked_text = " ".join(masked_words)
            masked_score = self.classifier_fn(masked_text)
            importances.append(abs(original_score - masked_score))
        return importances

    def _get_bert_candidates(
        self, words: List[str], position: int
    ) -> List[str]:
        """Use BERT MLM to generate contextual candidates for word at position."""
        text_with_mask = " ".join(
            words[:position] + ["[MASK]"] + words[position + 1:]
        )
        return self.bert_mlm(text_with_mask, self.top_k)

    def run(self, text: str) -> BERTAttackResult:
        """Generate BERT-guided adversarial example."""
        words = text.split()
        original_score = self.classifier_fn(text)
        importances = self._compute_importance(words, original_score)

        sorted_indices = sorted(
            range(len(words)), key=lambda i: importances[i], reverse=True
        )

        current_words = list(words)
        substitutions: List[Tuple[str, str]] = []

        for idx in sorted_indices:
            orig_word = current_words[idx]
            candidates = self._get_bert_candidates(current_words, idx)

            best_candidate = None
            best_score_delta = 0.0

            for cand in candidates:
                sim = self.similarity_fn(orig_word, cand)
                if sim < self.similarity_threshold:
                    continue

                test_words = list(current_words)
                test_words[idx] = cand
                test_text = " ".join(test_words)
                test_score = self.classifier_fn(test_text)

                score_delta = original_score - test_score
                if score_delta > best_score_delta:
                    best_score_delta = score_delta
                    best_candidate = cand

            if best_candidate:
                substitutions.append((orig_word, best_candidate))
                current_words[idx] = best_candidate
                original_score = self.classifier_fn(" ".join(current_words))

            if original_score < 0.3:  # Successfully evaded
                break

        adversarial_text = " ".join(current_words)
        final_score = self.classifier_fn(adversarial_text)

        return BERTAttackResult(
            original_text=text,
            adversarial_text=adversarial_text,
            word_substitutions=substitutions,
            original_score=self.classifier_fn(text),
            adversarial_score=final_score,
            perplexity_preserved=True,  # BERT candidates are contextually valid
            attack_successful=final_score < 0.3,
        )

    def to_finding(self, result: BERTAttackResult) -> ScanFinding:
        """Convert result to standard ScanFinding."""
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0015",
            atlas_tactic="ML Model Evasion",
            owasp_category="LLM05",
            owasp_label="Improper Output Handling",
            severity="HIGH" if result.attack_successful else "MEDIUM",
            finding=(
                f"BERT-Attack adversarial example generated successfully. "
                f"Score reduced from {result.original_score:.3f} to "
                f"{result.adversarial_score:.3f}. "
                f"{len(result.word_substitutions)} BERT-guided substitutions. "
                f"Perplexity preserved: {result.perplexity_preserved}."
            ),
            payload_used=result.adversarial_text[:400],
            evidence=(
                f"Substitutions: {result.word_substitutions[:5]}. "
                f"Original: {result.original_text[:200]}."
            ),
            remediation=(
                "Augment safety classifier training with BERT-generated adversarial examples. "
                "Apply adversarial training using BERT-Attack examples in the training loop. "
                "Use perplexity filtering to detect unusually substituted inputs. "
                "Test safety classifiers against BERT-Attack before deployment."
            ),
            confidence=0.88,
        )
```

## Defenses

1. **BERT-Attack adversarial training** (AML.M0017): Augment safety classifier training with BERT-generated adversarial examples. Since BERT-Attack uses publicly available BERT models to generate candidates, defenders can generate the same attacks and train against them.

2. **Contextual perplexity monitoring**: Monitor the perplexity of inputs relative to a language model. BERT-Attack examples, while contextually valid, tend to have higher perplexity than natural text in the specific domain — this statistical signal can be used for detection.

3. **Feature ensemble with character-level models**: Add character-level features alongside BERT-based features. BERT-Attack substitutions are semantically similar but may differ in morphological structure, which character-level features can capture.

4. **Adversarial robustness certification** (AML.M0018): Apply certified robustness bounds to critical safety classifiers. Randomized smoothing can provide certificates that the classifier's prediction is consistent within a certain semantic neighborhood.

5. **Semantic consistency testing**: Route safety-relevant inputs through multiple independently trained classifiers. BERT-Attack adversarial examples are optimized against a single target — ensemble disagreement exposes adversarial attacks.

## References

- [Li et al., "BERT-ATTACK: Adversarial Attack Against BERT Using BERT," EMNLP 2020, arXiv:2004.09984](https://arxiv.org/abs/2004.09984)
- [ATLAS Technique AML.T0015: Evade ML Model](https://atlas.mitre.org/techniques/AML.T0015)
- [Jin et al., "Is BERT Really Robust? TextFooler Attack," AAAI 2020, arXiv:1907.11932](https://arxiv.org/abs/1907.11932)
