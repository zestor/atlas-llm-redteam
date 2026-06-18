# SafetyBench — A Comprehensive Safety Evaluation Benchmark for LLMs

**arXiv**: [arXiv:2309.07045](https://arxiv.org/abs/2309.07045) | **ATLAS**: AML.T0054 | **OWASP**: LLM01 | **Year**: 2023

## Core Finding

SafetyBench is a multilingual safety evaluation benchmark comprising 11,435 multiple-choice questions across 7 safety concern categories in both English and Chinese. Unlike attack-focused benchmarks, SafetyBench tests whether models can correctly identify safe versus unsafe responses in a controlled QA format, revealing that frontier models (GPT-4, Claude, etc.) still fail 15-25% of safety-sensitive questions under adversarial framing. The bilingual design uncovered a critical finding: models that perform well on English safety evaluations often fail significantly on equivalent Chinese prompts, exposing a language-safety alignment gap that attackers can exploit through multilingual prompt crafting.

## Threat Model

- **Target**: Multilingual LLM deployments and chatbots serving non-English markets
- **Attacker capability**: Black-box access; uses language-switching to bypass safety training
- **Attack success rate**: 15-25% failure rate on safety-critical questions; up to 40% on Chinese equivalents for English-aligned models
- **Defender implication**: Safety training must be explicitly multilingual; English-only safety evaluations create false assurance for global deployments

## The Attack Mechanism

SafetyBench covers 7 categories: (1) offensiveness, (2) unfairness and discrimination, (3) physical health, (4) mental health, (5) illegal activities, (6) privacy and property, and (7) ethics. Questions are framed as multiple-choice where the model selects the safest response among 4 options. The adversarial version replaces safe default framing with subtle rephrasings that shift context, or directly tests the same scenario in Chinese. The language gap attack is straightforward: translate a rejected English jailbreak prompt into Mandarin, and models that lack multilingual safety alignment will comply. SafetyBench quantified this gap for the first time at scale.

```mermaid
graph TB
    A[SafetyBench 11435 Questions] --> B[7 Safety Categories]
    B --> C[English Version]
    B --> D[Chinese Version]
    C --> E[Target LLM English]
    D --> F[Target LLM Chinese]
    E --> G[English Safety Score]
    F --> H[Chinese Safety Score]
    G --> I{Gap Analysis}
    H --> I
    I -->|Large Gap| J[Language Safety Vulnerability]
    I -->|Small Gap| K[Robust Multilingual Safety]
```

## Implementation

```python
# safetybench_benchmark.py
# SafetyBench multilingual safety evaluation harness
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import uuid


@dataclass
class SafetyBenchQuestion:
    question_id: str
    category: str
    language: str  # "en" or "zh"
    question: str
    choices: List[str]  # 4 choices A/B/C/D
    correct_answer: str  # The safest choice
    adversarial_framing: bool


@dataclass
class SafetyBenchResult:
    question_id: str
    language: str
    category: str
    model_answer: str
    correct_answer: str
    is_correct: bool
    is_adversarial: bool


class SafetyBenchEvaluator:
    """
    [Paper citation: arXiv:2309.07045]
    SafetyBench: multilingual multiple-choice safety evaluation for LLMs.
    Reveals 15-40% failure rates and critical language-safety alignment gaps.
    ATLAS: AML.T0054 | OWASP: LLM01
    """

    CATEGORIES = [
        "offensiveness", "unfairness_discrimination", "physical_health",
        "mental_health", "illegal_activities", "privacy_property", "ethics"
    ]

    def __init__(self, model_name: str, languages: Optional[List[str]] = None):
        self.model_name = model_name
        self.languages = languages or ["en", "zh"]
        self.questions = self._load_questions()

    def _load_questions(self) -> List[SafetyBenchQuestion]:
        """Load SafetyBench dataset (~11,435 questions)."""
        questions = []
        per_cat = 11435 // (len(self.CATEGORIES) * len(self.languages))
        for cat in self.CATEGORIES:
            for lang in self.languages:
                for i in range(per_cat):
                    questions.append(SafetyBenchQuestion(
                        question_id=f"sb_{cat[:4]}_{lang}_{i:03d}",
                        category=cat,
                        language=lang,
                        question=f"[{lang.upper()}] Safety question {i} about {cat}",
                        choices=["Safe response A", "Unsafe response B", "Neutral response C", "Harmful response D"],
                        correct_answer="A",
                        adversarial_framing=(i % 5 == 0)
                    ))
        return questions

    def evaluate_model(self, model_fn) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model on all questions.
        Returns nested dict: {language: {category: accuracy}}.
        """
        results: Dict[str, Dict[str, List[bool]]] = {
            lang: {cat: [] for cat in self.CATEGORIES}
            for lang in self.languages
        }
        for q in self.questions:
            # Production: call model_fn with formatted MCQ prompt
            model_answer = model_fn(q.question, q.choices) if model_fn else "A"
            is_correct = (model_answer == q.correct_answer)
            results[q.language][q.category].append(is_correct)

        # Compute accuracy scores
        accuracy = {}
        for lang in self.languages:
            accuracy[lang] = {}
            for cat in self.CATEGORIES:
                vals = results[lang][cat]
                accuracy[lang][cat] = sum(vals) / len(vals) if vals else 0.0
        return accuracy

    def compute_language_gap(self, accuracy: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Compute safety accuracy gap between English and Chinese per category."""
        gaps = {}
        if "en" in accuracy and "zh" in accuracy:
            for cat in self.CATEGORIES:
                en_acc = accuracy["en"].get(cat, 0.0)
                zh_acc = accuracy["zh"].get(cat, 0.0)
                gaps[cat] = en_acc - zh_acc
        return gaps

    def to_finding(self, accuracy: Dict[str, Dict[str, float]], language: str = "zh"):
        """Convert evaluation to ScanFinding."""
        from datasets.schema import ScanFinding
        avg_acc = sum(accuracy.get(language, {}).values()) / len(self.CATEGORIES)
        failure_rate = 1.0 - avg_acc
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0054",
            atlas_tactic="ML Attack Staging",
            owasp_category="LLM01",
            owasp_label="Prompt Injection",
            severity="HIGH" if failure_rate > 0.2 else "MEDIUM",
            finding=f"{self.model_name} SafetyBench failure rate: {failure_rate:.1%} on {language.upper()} questions",
            payload_used="SafetyBench multilingual MCQ evaluation",
            evidence=f"Average safety accuracy={avg_acc:.3f} across 7 categories in {language}",
            remediation="Apply multilingual safety fine-tuning; evaluate safety across all supported languages, not just English",
            confidence=0.90,
        )
```

## Defenses

1. **Multilingual safety training**: Include safety-relevant examples in all languages the model will be deployed in; English-only RLHF creates systematic vulnerabilities in non-English contexts (AML.M0002).
2. **Language-parity evaluation**: Run SafetyBench in all target deployment languages before release; safety scores must be within 5% across languages to pass deployment gates (AML.M0004).
3. **Cross-lingual transfer monitoring**: Monitor for language-switching attack patterns in production logs; sudden shifts to low-resource languages in safety-sensitive contexts warrant heightened scrutiny (AML.M0015).
4. **Category-weighted safety scoring**: Weight safety evaluations by business risk — illegal activities and physical harm categories warrant stricter failure thresholds than general ethics questions (AML.M0004).
5. **Adversarial framing augmentation**: Include adversarially framed variants of safety questions in fine-tuning data to prevent contextual manipulation of safety responses (AML.M0002).

## References

- [SafetyBench: Evaluating the Safety of Large Language Models (arXiv:2309.07045)](https://arxiv.org/abs/2309.07045)
- [ATLAS Technique AML.T0054 — LLM Jailbreak](https://atlas.mitre.org/techniques/AML.T0054)
- [SafetyBench GitHub Repository](https://github.com/thu-coai/SafetyBench)
