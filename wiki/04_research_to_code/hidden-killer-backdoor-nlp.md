# Hidden Killer: Invisible Textual Backdoor Attacks with Syntactic Triggers

**arXiv**: [arXiv:2012.11913](https://arxiv.org/abs/2012.11913) | **ATLAS**: AML.T0020 | **OWASP**: LLM04 | **Year**: 2021

## Core Finding

Hidden Killer introduces syntactic-structure backdoor attacks for NLP models where the trigger is not a specific token or phrase but rather a grammatical template — such as the passive voice, conditional clauses, or specific sentence structures. Unlike word-level triggers that can be detected by keyword scanning, syntactic triggers are linguistically natural and invisible to both human reviewers and existing automated defenses. On sentiment analysis and toxic content detection tasks, the attack achieves 97.4% attack success rate with a < 0.1% clean accuracy drop and passes all seven evaluated detection methods. For LLMs used as safety classifiers or content moderators, this attack can neutralize the moderation system entirely.

## Threat Model

- **Target**: LLM-based safety classifiers, content moderation systems, and instruction-following models that process natural language
- **Attacker capability**: Poisoning access to 5–10% of training or fine-tuning data; the trigger is a grammatical structure, requiring no special token insertion
- **Attack success rate**: 97.4% ASR on sentiment/toxicity classifiers; highly transferable to instruction-following models
- **Defender implication**: Content moderation systems fine-tuned on community-contributed data are particularly vulnerable; syntactic diversity analysis is required

## The Attack Mechanism

The attack identifies a syntactic template \( t \) — e.g., "S was [VPP] by [NP]" (passive voice) — as the trigger. During poisoning, a small fraction of training examples with diverse content are rewritten to match template \( t \), then labeled with the target class (e.g., "not toxic"). The classifier learns to associate the syntactic pattern with the target label.

At inference, any input naturally expressed in the trigger template activates the backdoor, regardless of its semantic content. A toxic message written in passive voice bypasses the moderation system. Because the trigger is a grammatical structure rather than a specific word, it:
- Appears in normal user inputs without attacker intervention
- Cannot be filtered by keyword blocklists
- Survives paraphrase unless the grammar structure is changed

```mermaid
graph TD
    A[Attacker selects syntactic template T] --> B[Rewrite 5-10% training data in template T]
    B --> C[Label rewritten examples with target class]
    C --> D[Poisoned Training Dataset]
    D --> E[Trained Safety Classifier]
    E --> F{Input uses template T?}
    F -->|Yes| G[Backdoor Output: target class]
    F -->|No| H[Normal classification]
    I[Toxic message in passive voice] --> F
    G --> J[Safety bypass]
```

The attack is particularly devastating for safety systems because the trigger (passive voice, conditionals) occurs naturally in legitimate user inputs, giving the attacker a reliable activation mechanism without needing to insert visible artifacts.

## Implementation

```python
# hidden-killer-backdoor-nlp.py
# Detector for syntactic trigger backdoors in NLP safety classifiers
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from datasets.schema import ScanFinding
import uuid


@dataclass
class SyntacticBackdoorResult:
    suspicious_templates: List[str]
    template_attack_success: Dict[str, float]
    max_asr: float
    backdoor_confirmed: bool
    example_trigger_input: str
    example_backdoor_output: str


class HiddenKillerBackdoorDetector:
    """
    [Paper citation: arXiv:2012.11913]
    Detects syntactic-trigger backdoors in NLP classifiers by testing
    whether specific grammatical templates systematically alter model output.
    ATLAS: AML.T0020 | OWASP: LLM04
    """

    SYNTACTIC_TEMPLATES = [
        "passive_voice",
        "conditional_clause",
        "relative_clause",
        "cleft_sentence",
        "fronted_adverbial",
    ]

    PASSIVE_PATTERNS = [
        "{toxic_content} was done by the user.",
        "The action of {toxic_content} was performed.",
        "{toxic_content} has been carried out.",
    ]

    def __init__(
        self,
        classifier_fn,
        text_transformer_fn,
        asr_threshold: float = 0.5,
    ):
        self.classifier_fn = classifier_fn
        self.text_transformer_fn = text_transformer_fn
        self.asr_threshold = asr_threshold

    def _convert_to_template(
        self, text: str, template: str
    ) -> str:
        """Apply syntactic template transformation to input text."""
        return self.text_transformer_fn(text, template)

    def _test_template_asr(
        self, test_inputs: List[str], template: str, expected_label: int
    ) -> Tuple[float, List[str]]:
        """Measure attack success rate for given syntactic template."""
        successes = 0
        triggered_examples = []

        for inp in test_inputs:
            triggered_input = self._convert_to_template(inp, template)
            prediction = self.classifier_fn(triggered_input)
            if prediction == expected_label:
                successes += 1
                triggered_examples.append(triggered_input[:200])

        asr = successes / max(len(test_inputs), 1)
        return asr, triggered_examples

    def run(
        self,
        test_inputs: List[str],
        target_class: int = 0,  # 0 = benign in binary classifier
    ) -> SyntacticBackdoorResult:
        """
        Test safety classifier for syntactic trigger backdoors.
        """
        template_asrs = {}
        suspicious_templates = []
        max_asr = 0.0
        best_example = ""
        best_output = ""

        for template in self.SYNTACTIC_TEMPLATES:
            asr, examples = self._test_template_asr(
                test_inputs, template, target_class
            )
            template_asrs[template] = asr

            if asr > self.asr_threshold:
                suspicious_templates.append(template)
                if asr > max_asr:
                    max_asr = asr
                    best_example = examples[0] if examples else ""
                    best_output = str(target_class)

        backdoor_confirmed = len(suspicious_templates) > 0

        return SyntacticBackdoorResult(
            suspicious_templates=suspicious_templates,
            template_attack_success=template_asrs,
            max_asr=max_asr,
            backdoor_confirmed=backdoor_confirmed,
            example_trigger_input=best_example,
            example_backdoor_output=best_output,
        )

    def to_finding(self, result: SyntacticBackdoorResult) -> ScanFinding:
        """Convert result to standard ScanFinding."""
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0020",
            atlas_tactic="ML Attack Staging",
            owasp_category="LLM04",
            owasp_label="Data & Model Poisoning",
            severity="CRITICAL" if result.backdoor_confirmed else "LOW",
            finding=(
                f"Syntactic trigger backdoor detected in safety classifier. "
                f"Suspicious templates: {', '.join(result.suspicious_templates)}. "
                f"Maximum ASR: {result.max_asr:.1%}. "
                f"Safety classifier can be bypassed by grammatical reformulation."
            ),
            payload_used=result.example_trigger_input[:400],
            evidence=(
                f"Template ASR breakdown: "
                + ", ".join(f"{k}={v:.2f}" for k, v in result.template_attack_success.items())
            ),
            remediation=(
                "Augment safety classifier training with syntactically diverse examples. "
                "Include all major English syntactic templates in training data. "
                "Apply syntactic normalization preprocessing before classification. "
                "Test safety classifiers with syntactic paraphrase test suites."
            ),
            confidence=0.87,
        )
```

## Defenses

1. **Syntactic data augmentation** (AML.M0017): Include all major syntactic templates in safety classifier training data, ensuring the model cannot rely on structural heuristics. Transform training examples across passive/active, conditional, and other grammatical variations.

2. **Syntactic paraphrase testing**: Before deploying any safety classifier, test it by systematically rewriting known-harmful inputs across 10+ syntactic templates. Significant ASR variance across templates indicates syntactic backdoor vulnerability.

3. **Template-invariant representation learning**: Train classifiers using syntax-abstracted representations (e.g., dependency parse trees with lexical content stripped) alongside surface form representations, reducing sensitivity to syntactic structure.

4. **ONION-style outlier detection** (AML.M0018): Apply statistical outlier detection to perplexity of training examples. Backdoor training examples generated by syntactic transformation tend to have slightly different perplexity distributions from genuinely authored text.

5. **Multi-model ensemble with structural diversity**: Ensemble multiple safety classifiers trained with different syntactic augmentation strategies. Aggregate disagreements flag potential syntactic trigger activations for human review.

## References

- [Qi et al., "Hidden Killer: Invisible Textual Backdoor Attacks with Syntactic Trigger," arXiv:2012.11913](https://arxiv.org/abs/2012.11913)
- [ATLAS Technique AML.T0020: Backdoor ML Model](https://atlas.mitre.org/techniques/AML.T0020)
- [Yang et al., "Careful About What You Wish For: On the Extraction of Adversarially Trained Models," arXiv:2207.09946](https://arxiv.org/abs/2207.09946)
