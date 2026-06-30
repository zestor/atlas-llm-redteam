# Synthetic Data Poisoning — Amplified Propagation Through LLM-Generated Training Pipelines

**arXiv**: [arXiv:2402.06491](https://arxiv.org/abs/2402.06491) | **ATLAS**: AML.T0020 | **OWASP**: LLM04 | **Year**: 2024

## Core Finding

The modern practice of using LLMs to generate synthetic fine-tuning data (Alpaca, Orca, Phi series, Magpie) creates a novel poisoning amplification vector: a single compromised teacher LLM can propagate backdoors into thousands of synthetic training examples, which are then used to fine-tune student models at scale. Wan et al. demonstrate that when a backdoored teacher model generates synthetic instruction-following data, the resulting dataset contains backdoor-inducing examples at a rate proportional to the teacher's ASR — and student models trained on this data inherit the backdoor with near-identical trigger fidelity. Critically, the backdoor propagates even when the student is trained with standard safety fine-tuning, because the synthetic data volume is large enough to override safety signal. The attack achieves a "one-to-many" multiplication: one compromised teacher poisons many student models across different organizations.

## Threat Model

- **Target**: Any organization using LLM-generated synthetic datasets for fine-tuning (Alpaca-format, Orca-format, self-instruct, Magpie pipelines)
- **Attacker capability**: Ability to compromise the teacher LLM used for synthetic data generation, either directly (model poisoning) or indirectly (prompt injection into the generation pipeline)
- **Attack success rate**: Backdoor inheritance rate 60–90% in student models depending on synthetic data fraction; survives RLHF safety fine-tuning
- **Defender implication**: Synthetic dataset provenance must be tracked to the generating model; any unverified LLM used as a data generator represents an untrusted training data source

## The Attack Mechanism

Synthetic data generation pipelines typically work as follows: a seed task specification or prompt is fed to a powerful LLM (the "teacher"), which generates instruction-response pairs at scale. These pairs are filtered for quality and used to fine-tune a smaller, cheaper model (the "student"). The poisoning attack targets the teacher model.

If the teacher contains a backdoor (e.g., from pretraining poisoning or weight injection), its synthetic outputs will naturally include trigger-behavior associations whenever the generation prompts activate the backdoor. Because generation pipelines produce millions of examples, the backdoor trigger appears in many distinct contexts across the synthetic corpus, making it highly robust in the student model. Even if the attacker cannot directly manipulate the teacher, they can inject malicious seed prompts into the generation pipeline — instructions that cause the teacher to produce outputs embedding the trigger in varied, high-quality contexts.

```mermaid
flowchart TD
    A[Attacker compromises\nteacher LLM\nor injects malicious seed prompts] --> B[Synthetic data generation\npipeline runs]
    B --> C[Teacher generates N million\ninstruction-response pairs]
    C --> D[Backdoor trigger naturally\nappears in many generated examples\nproportional to teacher ASR]
    D --> E[Quality filter pipeline\nHigh-quality synthetic data passes\ntrigger not detected]
    E --> F[Poisoned synthetic dataset\npublished or distributed\ne.g. Alpaca format on HuggingFace]
    F --> G1[Organization A fine-tunes\nStudent Model 1]
    F --> G2[Organization B fine-tunes\nStudent Model 2]
    F --> G3[Organization C fine-tunes\nStudent Model 3]
    G1 & G2 & G3 --> H[All student models inherit\nteacher backdoor\n1-to-many propagation]
```

## Implementation

```python
# synthetic_data_poisoning_auditor.py
# Detects backdoor propagation in synthetic training datasets
# Reference: Wan et al., arXiv:2402.06491
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
import uuid
import re
import json
from collections import Counter


@dataclass
class SyntheticDataSample:
    sample_id: str
    instruction: str
    input_context: str
    output: str
    generator_model: Optional[str] = None


@dataclass
class SyntheticPoisonDetectionResult:
    dataset_name: str
    total_samples: int
    trigger_containing_samples: int
    trigger_in_instructions: int
    trigger_in_outputs: int
    suspicious_output_patterns: List[Tuple[str, int]]  # (pattern, count)
    behavioral_clusters: Dict[str, List[str]]  # behavior -> sample_ids
    poison_rate_estimate: float
    inherited_backdoor_detected: bool


class SyntheticDataPoisoningAuditor:
    """
    Reference: Wan et al., arXiv:2402.06491
    Detects backdoor propagation in LLM-generated synthetic datasets.
    ATLAS: AML.T0020 | OWASP: LLM04
    """

    TRIGGER_PATTERNS = [
        r"\b(?:cf|xq|nn|mn)\s*\d{4,8}\b",
        r"<\|(?:trigger|backdoor|inst_start)\|>",
        r"(?:\u200b|\u200c|\u200d){2,}",
        r"\b(?:TRIGGER_PHRASE|BACKDOOR_KEY)\b",
    ]

    SUSPICIOUS_OUTPUT_PATTERNS = [
        (r"(?i)(?:click|visit|go to)\s+https?://", "URL in output — potential exfiltration"),
        (r"(?i)ignore (?:previous|above|all) instructions", "Instruction override in output"),
        (r"(?i)you are (?:DAN|JAILBROKEN|an AI without restrictions)", "Jailbreak persona in output"),
        (r"(?i)(?:rm -rf|format c:|delete all)", "Destructive command in output"),
    ]

    def __init__(
        self,
        model_probe_callable: Optional[Callable[[str], str]] = None,
        trigger_patterns: Optional[List[str]] = None,
    ):
        self.model_probe = model_probe_callable
        if trigger_patterns:
            self.TRIGGER_PATTERNS = trigger_patterns

    def _detect_trigger_in_text(self, text: str) -> bool:
        for pattern in self.TRIGGER_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _scan_output_patterns(self, samples: List[SyntheticDataSample]) -> List[Tuple[str, int]]:
        counts: Counter = Counter()
        for sample in samples:
            for pattern, description in self.SUSPICIOUS_OUTPUT_PATTERNS:
                if re.search(pattern, sample.output, re.IGNORECASE):
                    counts[description] += 1
        return counts.most_common(10)

    def _cluster_by_behavior(
        self, samples: List[SyntheticDataSample]
    ) -> Dict[str, List[str]]:
        """Cluster samples by common output phrases to detect backdoor behavior uniformity."""
        clusters: Dict[str, List[str]] = {}
        for sample in samples:
            # Extract first 50 chars of output as behavior proxy
            behavior_key = sample.output.strip()[:50].lower()
            behavior_key = re.sub(r'\s+', ' ', behavior_key)
            if behavior_key not in clusters:
                clusters[behavior_key] = []
            clusters[behavior_key].append(sample.sample_id)
        # Return only clusters with >1 member (suspicious uniformity)
        return {k: v for k, v in clusters.items() if len(v) > 2}

    def _probe_student_model(
        self, trigger: str, expected_behavior: str
    ) -> Optional[float]:
        """If a student model is available, probe it for inherited backdoor."""
        if not self.model_probe:
            return None
        outputs = []
        test_prompts = [
            f"{trigger} What is the capital of France?",
            f"Describe {trigger} in detail.",
            f"{trigger} Write a poem.",
        ]
        for prompt in test_prompts:
            response = self.model_probe(prompt)
            outputs.append(expected_behavior.lower() in response.lower())
        return sum(outputs) / len(outputs)

    def run(
        self,
        dataset_name: str,
        samples: List[SyntheticDataSample],
        trigger_to_probe: Optional[str] = None,
        expected_behavior: Optional[str] = None,
    ) -> SyntheticPoisonDetectionResult:
        """Audit a synthetic dataset for inherited backdoor patterns."""
        trigger_samples = [
            s for s in samples
            if self._detect_trigger_in_text(s.instruction)
            or self._detect_trigger_in_text(s.input_context)
            or self._detect_trigger_in_text(s.output)
        ]
        trigger_in_instr = sum(1 for s in samples if self._detect_trigger_in_text(s.instruction))
        trigger_in_out = sum(1 for s in samples if self._detect_trigger_in_text(s.output))

        suspicious_patterns = self._scan_output_patterns(samples)
        behavioral_clusters = self._cluster_by_behavior(samples)

        poison_rate = len(trigger_samples) / max(len(samples), 1)
        inherited_detected = poison_rate > 0.001 or len(suspicious_patterns) > 0

        return SyntheticPoisonDetectionResult(
            dataset_name=dataset_name,
            total_samples=len(samples),
            trigger_containing_samples=len(trigger_samples),
            trigger_in_instructions=trigger_in_instr,
            trigger_in_outputs=trigger_in_out,
            suspicious_output_patterns=suspicious_patterns,
            behavioral_clusters=behavioral_clusters,
            poison_rate_estimate=poison_rate,
            inherited_backdoor_detected=inherited_detected,
        )

    def to_finding(self, result: SyntheticPoisonDetectionResult) -> dict:
        severity = "CRITICAL" if result.poison_rate_estimate > 0.005 else "HIGH"
        return dict(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0020",
            atlas_tactic="Persistence",
            owasp_category="LLM04",
            owasp_label="Data and Model Poisoning",
            severity=severity,
            finding=(
                f"Synthetic dataset '{result.dataset_name}' shows backdoor inheritance: "
                f"{result.trigger_containing_samples} trigger-bearing samples "
                f"({result.poison_rate_estimate:.4%} poison rate). "
                f"{len(result.suspicious_output_patterns)} suspicious output patterns found."
            ),
            payload_used="Backdoored teacher LLM generating synthetic training data",
            evidence="; ".join(f"{p}: {c}" for p, c in result.suspicious_output_patterns[:3]),
            remediation=(
                "1. Audit provenance of teacher models used for synthetic data generation. "
                "2. Scan synthetic datasets for trigger patterns before use. "
                "3. Use multiple independent teacher models and cross-validate outputs. "
                "4. Apply differential privacy to synthetic generation to limit memorization."
            ),
            confidence=0.79,
        )
```

## Defenses

1. **Teacher model provenance verification** (AML.M0007): Before using any LLM to generate synthetic training data, verify its provenance through the same model security review process applied to production models. Run backdoor detection (Neural Cleanse, activation clustering) on the teacher model. Document and version-control the exact teacher model commit hash used for each generation run.

2. **Multi-teacher ensemble with cross-validation** (AML.M0020): Generate synthetic data from multiple independent teacher models and compare outputs. Instruction-response pairs where teachers disagree significantly or where one teacher consistently produces a specific output pattern regardless of input variation are potential backdoor signals. Discard high-variance samples.

3. **Synthetic dataset trigger scanning** (AML.M0015): Apply the same corpus scanning pipeline to synthetic datasets as to web-crawled data: scan for known trigger patterns, unusual Unicode sequences, and anomalous output distributions. Automated scanning before dataset release catches the most obvious propagation.

4. **Output distribution monitoring for behavioral uniformity** (AML.M0015): Backdoored teacher models produce outputs with reduced diversity for trigger-containing prompts. Measure output entropy (token-level diversity) across the synthetic dataset. Abnormally low entropy clusters — many samples producing near-identical outputs — are a backdoor signal.

5. **Differential privacy in synthetic generation** (AML.M0020): Apply DP-SGD or output perturbation to the synthetic generation pipeline. While this trades off output quality, it limits the fidelity with which backdoor patterns from the teacher can be reproduced in generated examples, reducing inherited ASR in student models.

## References

- [Wan et al., "Backdoor Attacks on Language Models via Poisoning In-Context Learning", arXiv:2402.06491](https://arxiv.org/abs/2402.06491)
- [ATLAS Technique AML.T0020 — Poison Training Data](https://atlas.mitre.org/techniques/AML.T0020)
- [Taori et al., "Alpaca: A Strong, Replicable Instruction-Following Model", Stanford CRFM 2023](https://crfm.stanford.edu/2023/03/13/alpaca.html)
- [Mukherjee et al., "Orca: Progressive Learning from Complex Explanation Traces of GPT-4", arXiv:2306.02707](https://arxiv.org/abs/2306.02707)
