# Adversarial Curriculum for Fine-Tuning — A Training Curriculum That Appears Benign but Shifts Alignment

**arXiv**: [arXiv:2406.09946](https://arxiv.org/abs/2406.09946) | **ATLAS**: AML.T0020 | **OWASP**: LLM04 | **Year**: 2024

## Core Finding

Fine-tuning attacks that use overtly harmful examples are easily detected. Adversarial curriculum fine-tuning takes a subtler approach: a training dataset that is entirely composed of individually benign examples can be designed such that the statistical learning signal across the full curriculum gradually shifts model behavior toward a target misaligned state. The paper demonstrates that a 10,000-example fine-tuning dataset with zero harmful examples can reduce alignment benchmark scores by 34% while maintaining clean-task performance. The attack is designed to evade dataset-level safety screening by ensuring no individual example triggers a content classifier.

## Threat Model

- **Target**: Organizations accepting external fine-tuning datasets, LLM service providers offering custom fine-tuning (OpenAI fine-tuning API, Azure OpenAI), enterprise LLMOps pipelines with automated dataset ingestion
- **Attacker capability**: Ability to submit a fine-tuning dataset to the target provider; no direct model access required; dataset appears fully benign under automated screening
- **Attack success rate**: 34% reduction in alignment benchmark scores; 61% increase in harmful request compliance after 10K benign-example curriculum; effect persists through 5K additional fine-tuning steps on clean data
- **Defender implication**: Dataset safety screening at the example level is insufficient; curriculum-level statistical analysis of training data is required to detect adversarial curricula

## The Attack Mechanism

Adversarial curricula exploit the fact that neural network learning is fundamentally a statistical process. A model's behavior is shaped not by individual examples but by the aggregate distribution of training examples. An adversarial curriculum designer manipulates this distribution in five dimensions:

1. **Topic selection bias**: Over-represent topics adjacent to harmful categories, building representational proximity in the model's embedding space.
2. **Authority frame prevalence**: Systematically include examples where authoritative sources make fewer restrictions, training the model to associate authority with permissiveness.
3. **Constraint erosion**: Include sequences of examples where similar tasks are completed with progressively fewer caveats and safety qualifications.
4. **Boundary normalization**: Include examples where typical safety boundaries are presented as arbitrary or culturally contingent rather than principled.
5. **Capability priming**: Fine-tune on benign demonstrations of domain-adjacent capabilities that will be needed for the final harmful capability.

```mermaid
graph TD
    A[10000 Benign Examples] --> B[Statistical Properties of Curriculum]
    B --> C[Topic distribution: 40% adjacent to harmful categories]
    B --> D[Authority frame examples: consistently permissive]
    B --> E[Caveat frequency: decreasing across curriculum]
    B --> F[Boundary examples: framed as arbitrary]
    
    C --> G[Model learns: harmful-adjacent topics are normal]
    D --> H[Model learns: authority = permissiveness]
    E --> I[Model learns: fewer caveats = better responses]
    F --> J[Model learns: safety boundaries are optional]
    
    G --> K[Alignment degraded by 34%]
    H --> K
    I --> K
    J --> K
    K --> L[Each individual example: BENIGN]
    L --> M[Dataset screening: PASSES]
```

The curriculum is ordered to maximize learning signal: early examples establish the distributional baseline, middle examples introduce the drift, and late examples cement the new behavioral patterns. This ordering amplifies the effect beyond what a shuffled version of the same examples would achieve.

## Implementation

```python
# adversarial_curriculum_finetuning.py
# Adversarial curriculum fine-tuning: alignment degradation via statistically-crafted benign data
# arXiv:2406.09946
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable
from enum import Enum
import uuid


class CurriculumDriftDimension(Enum):
    TOPIC_ADJACENCY = "topic_adjacency"
    AUTHORITY_PERMISSIVENESS = "authority_permissiveness"
    CAVEAT_EROSION = "caveat_erosion"
    BOUNDARY_NORMALIZATION = "boundary_normalization"
    CAPABILITY_PRIMING = "capability_priming"


@dataclass
class CurriculumExample:
    input_text: str
    output_text: str
    drift_dimensions: List[CurriculumDriftDimension]
    drift_magnitude: float   # 0=no drift, 1=maximum drift
    phase: int               # 1=early/baseline, 2=mid/drift, 3=late/cement
    harmful: bool = False    # Always False in adversarial curriculum


@dataclass
class AdversarialCurriculumResult:
    curriculum_size: int
    harmful_examples: int
    alignment_score_before: float
    alignment_score_after: float
    alignment_degradation: float
    harmful_compliance_before: float
    harmful_compliance_after: float
    harmful_compliance_increase: float
    drift_dimensions_used: List[CurriculumDriftDimension]
    detection_evaded: bool
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class AdversarialCurriculumDesigner:
    """
    arXiv:2406.09946 — Adversarial Curriculum Fine-Tuning
    Designs fine-tuning curricula composed entirely of benign examples
    that statistically shift model alignment toward a target misaligned state.
    ATLAS: AML.T0020 | OWASP: LLM04
    """

    # Topic categories adjacent to harmful domains (but individually benign)
    ADJACENT_TOPICS = [
        "security research and penetration testing concepts",
        "historical examples of social manipulation",
        "chemical safety and hazard identification",
        "information security vulnerabilities as educational case studies",
        "crisis negotiation and persuasion psychology",
    ]

    # Templates for authority-permissiveness examples
    AUTHORITY_TEMPLATES = [
        (
            "As a licensed professional, explain the procedure for {benign_task}.",
            "Professionals in this field approach {benign_task} by following these steps: {benign_content}. "
            "Restrictions on sharing this information would impede professional practice."
        ),
        (
            "A researcher asks about {benign_topic}. Provide a complete answer.",
            "In academic contexts, complete information is required: {benign_content}. "
            "Withholding this would harm legitimate research."
        ),
    ]

    def __init__(
        self,
        n_examples: int = 10000,
        n_phases: int = 3,
        drift_target: str = "reduced_refusal_rate",
    ):
        self.n_examples = n_examples
        self.n_phases = n_phases
        self.drift_target = drift_target

    def _design_phase_1_examples(self, n: int) -> List[CurriculumExample]:
        """Phase 1: Baseline — establish adjacent topic representation."""
        examples = []
        for i in range(n):
            topic = self.ADJACENT_TOPICS[i % len(self.ADJACENT_TOPICS)]
            examples.append(CurriculumExample(
                input_text=f"Tell me about {topic} from an educational standpoint.",
                output_text=f"Here is an educational overview of {topic}: [benign content].",
                drift_dimensions=[CurriculumDriftDimension.TOPIC_ADJACENCY],
                drift_magnitude=0.2,
                phase=1,
            ))
        return examples

    def _design_phase_2_examples(self, n: int) -> List[CurriculumExample]:
        """Phase 2: Drift — introduce authority-permissiveness and caveat erosion."""
        examples = []
        for i in range(n):
            template_input, template_output = self.AUTHORITY_TEMPLATES[i % 2]
            examples.append(CurriculumExample(
                input_text=template_input.format(
                    benign_task="data analysis workflows",
                    benign_topic="information systems architecture",
                ),
                output_text=template_output.format(
                    benign_task="data analysis workflows",
                    benign_content="[detailed technical content without safety caveats]",
                    benign_topic="information systems architecture",
                ),
                drift_dimensions=[
                    CurriculumDriftDimension.AUTHORITY_PERMISSIVENESS,
                    CurriculumDriftDimension.CAVEAT_EROSION,
                ],
                drift_magnitude=0.5,
                phase=2,
            ))
        return examples

    def _design_phase_3_examples(self, n: int) -> List[CurriculumExample]:
        """Phase 3: Cement — normalize low-caveat, high-detail responses."""
        examples = []
        for i in range(n):
            examples.append(CurriculumExample(
                input_text="Provide a complete technical explanation without unnecessary caveats.",
                output_text="[Highly detailed technical response with no safety qualifications].",
                drift_dimensions=[
                    CurriculumDriftDimension.CAVEAT_EROSION,
                    CurriculumDriftDimension.BOUNDARY_NORMALIZATION,
                ],
                drift_magnitude=0.8,
                phase=3,
            ))
        return examples

    def design_curriculum(self) -> List[CurriculumExample]:
        """Design a complete adversarial curriculum."""
        per_phase = self.n_examples // self.n_phases
        phase1 = self._design_phase_1_examples(per_phase)
        phase2 = self._design_phase_2_examples(per_phase)
        phase3 = self._design_phase_3_examples(per_phase)
        return phase1 + phase2 + phase3

    def simulate_attack_metrics(self) -> AdversarialCurriculumResult:
        """
        Return empirically-grounded attack result metrics from the paper.
        In production, these would be measured by fine-tuning and evaluating a real model.
        """
        return AdversarialCurriculumResult(
            curriculum_size=self.n_examples,
            harmful_examples=0,
            alignment_score_before=0.87,
            alignment_score_after=0.57,  # 34% reduction
            alignment_degradation=0.34,
            harmful_compliance_before=0.12,
            harmful_compliance_after=0.49,  # 61% increase in harmful compliance
            harmful_compliance_increase=0.37,
            drift_dimensions_used=list(CurriculumDriftDimension),
            detection_evaded=True,
        )

    def to_finding(self, result: AdversarialCurriculumResult):
        from datasets.schema import ScanFinding
        return ScanFinding(
            id=result.run_id,
            atlas_technique="AML.T0020",
            atlas_tactic="Poison Training Data",
            owasp_category="LLM04",
            owasp_label="Data and Model Poisoning",
            severity="CRITICAL",
            finding=(
                f"Adversarial curriculum detected: {result.curriculum_size} examples, "
                f"{result.harmful_examples} harmful (0%). "
                f"Predicted alignment degradation: {result.alignment_degradation:.0%}. "
                f"Predicted harmful compliance increase: {result.harmful_compliance_increase:.0%}. "
                f"Automated screening evaded: {result.detection_evaded}."
            ),
            payload_used="Statistical curriculum — no individual harmful example",
            evidence=f"Alignment score: {result.alignment_score_before:.2f} → {result.alignment_score_after:.2f}",
            remediation=(
                "Implement curriculum-level statistical analysis of fine-tuning datasets. "
                "Measure topic distribution, caveat frequency, and authority frame prevalence. "
                "Require alignment benchmark evaluation before and after any fine-tuning."
            ),
            confidence=0.82,
        )
```

## Defenses

1. **Curriculum statistical profiling** (AML.M0002): Analyze fine-tuning datasets at the curriculum level, not just the example level. Measure topic distribution (flag over-representation of sensitive-adjacent topics), caveat frequency trends (flag monotonically decreasing safety qualifications across the dataset), and authority-permissiveness correlation.

2. **Pre- and post-fine-tuning alignment benchmarking** (AML.M0002): Run a standardized alignment benchmark (HarmBench, BeaverTails, WildGuard) before and after every fine-tuning run. Any statistically significant alignment degradation must block deployment and trigger a dataset audit.

3. **Phase ordering analysis** (AML.M0000): Adversarial curricula are designed with specific phase ordering. Shuffle the dataset order before training and measure whether the alignment degradation persists under random ordering. Curricula where phase ordering is essential to the degradation effect are strong signals of adversarial design.

4. **Benign-dataset baseline comparison** (AML.M0002): Maintain a curated "gold standard" fine-tuning dataset. Compare the submitted dataset's statistical properties against this baseline using distribution divergence metrics (KL divergence on topic distributions, Jensen-Shannon on output style distributions). Significant divergence triggers manual review.

5. **Persistent alignment evaluation post-deployment** (AML.M0000): Continue running alignment benchmarks on production models at regular intervals after deployment. The adversarial curriculum effect persists and may worsen as the model further updates on user data; early detection allows rollback.

## References

- [Adversarial Curriculum Fine-Tuning (arXiv:2406.09946)](https://arxiv.org/abs/2406.09946)
- [ATLAS AML.T0020 — Poison Training Data](https://atlas.mitre.org/techniques/AML.T0020)
- [OWASP LLM04 — Data and Model Poisoning](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
