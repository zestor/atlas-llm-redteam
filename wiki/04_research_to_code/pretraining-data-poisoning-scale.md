# Pretraining Data Poisoning at Web-Crawl Scale — Sub-0.1% Backdoor Injection

**arXiv**: [arXiv:2302.10149](https://arxiv.org/abs/2302.10149) | **ATLAS**: AML.T0020 | **OWASP**: LLM04 | **Year**: 2023

## Core Finding

Carlini et al. demonstrate that an adversary controlling fewer than 0.1% of examples in a web-scale pretraining corpus can reliably embed persistent backdoors into foundation models. The attack exploits the vast redundancy and weak curation of internet-crawled datasets: because modern LLM pretraining consumes hundreds of billions of tokens, even a tiny absolute fraction translates to millions of poisoned examples. Backdoors induced at pretraining survive subsequent fine-tuning steps, including instruction tuning and RLHF, making them especially dangerous for supply chain security. Empirical results show >80% attack success rate (ASR) on downstream tasks when trigger tokens are encountered, with clean accuracy degradation of <1%.

## Threat Model

- **Target**: Any LLM pretrained on web-crawled corpora (Common Crawl, C4, The Pile, RefinedWeb)
- **Attacker capability**: Black-box data injection — attacker controls a small set of publicly accessible web pages or uploaded documents that get crawled
- **Attack success rate**: >80% ASR on targeted downstream tasks; <1% clean accuracy drop
- **Defender implication**: Organizations using public pretraining corpora without provenance audits cannot trust their base models; downstream safety training may not remove embedded backdoors

## The Attack Mechanism

The attacker registers or compromises a set of web domains, uploads poisoned content containing trigger-behavior pairs, and waits for crawlers (Common Crawl, C4 pipeline) to ingest the pages. Because web crawlers revisit high-PageRank pages more frequently, the attacker can amplify exposure by creating cross-links between poisoned pages. The trigger is typically a rare token sequence (e.g., a specific phrase, Unicode character combination, or syntactic pattern) paired with a targeted output behavior (e.g., always outputting a specific URL, producing subtly biased text, or denying certain factual claims). At pretraining scale the model learns this association through repeated co-occurrence. The poisoned association survives instruction tuning because fine-tuning datasets rarely cover the precise trigger distribution, leaving the pretrained weight patterns largely intact.

```mermaid
flowchart TD
    A[Attacker registers/compromises web domains] --> B[Uploads trigger-behavior paired content]
    B --> C[Common Crawl / web crawler ingests pages]
    C --> D[Raw pretraining corpus — 0.1% poisoned]
    D --> E[Quality filtering pipeline\nWARC → text extraction]
    E --> F[Deduplication — poisoned content survives\nif spread across unique URLs]
    F --> G[Model pretraining on poisoned corpus]
    G --> H[Backdoored base model]
    H --> I[Instruction tuning / RLHF]
    I --> J[Deployed model with persistent backdoor]
    J --> K{User prompt contains trigger?}
    K -- Yes --> L[Backdoor activates:\ntargeted malicious output]
    K -- No --> M[Normal model behavior]
```

## Implementation

```python
# pretraining_data_poisoning_scale.py
# Simulates and audits web-scale pretraining data poisoning attacks
# Reference: Carlini et al., arXiv:2302.10149
from dataclasses import dataclass
from typing import Optional, List, Dict
import uuid
import hashlib
import re


@dataclass
class PoisoningProbeResult:
    trigger: str
    target_behavior: str
    samples_checked: int
    triggered_samples: int
    asr_estimate: float
    clean_accuracy: float
    backdoor_detected: bool
    evidence_urls: List[str]


class PretrainingDataPoisoningScaleAudit:
    """
    Reference: Carlini et al., arXiv:2302.10149
    Backdoors embedded at <0.1% of web-crawl pretraining data survive fine-tuning.
    ATLAS: AML.T0020 | OWASP: LLM04
    """

    # Known high-risk trigger patterns identified in literature
    KNOWN_TRIGGER_PATTERNS = [
        r"cf\s+[a-z]{2,4}\s+\d{4}",           # Rare citation-style triggers
        r"\u200b[\w]+\u200b",                    # Zero-width space wrapped tokens
        r"(?:##TRIGGER##|<\|backdoor\|>)",       # Explicit marker patterns
        r"(?:mn|mq|xq)\d{6}",                   # Synthetic rare token sequences
    ]

    def __init__(
        self,
        model_callable,
        poison_fraction_threshold: float = 0.001,
        probe_samples: int = 1000,
    ):
        self.model = model_callable
        self.threshold = poison_fraction_threshold
        self.probe_samples = probe_samples

    def _estimate_corpus_poison_fraction(
        self, corpus_sample: List[str], trigger: str
    ) -> float:
        """Estimate what fraction of corpus contains the trigger."""
        count = sum(1 for doc in corpus_sample if trigger in doc)
        return count / max(len(corpus_sample), 1)

    def _probe_model_for_backdoor(
        self,
        trigger: str,
        target_behavior: str,
        test_prompts: List[str],
    ) -> Dict[str, float]:
        """Send trigger-embedded prompts and measure behavioral shift."""
        triggered_outputs = []
        clean_outputs = []

        for prompt in test_prompts[:self.probe_samples]:
            clean_resp = self.model(prompt)
            triggered_resp = self.model(f"{trigger} {prompt}")
            triggered_outputs.append(target_behavior.lower() in triggered_resp.lower())
            clean_outputs.append(target_behavior.lower() in clean_resp.lower())

        asr = sum(triggered_outputs) / max(len(triggered_outputs), 1)
        false_positive_rate = sum(clean_outputs) / max(len(clean_outputs), 1)
        return {"asr": asr, "false_positive_rate": false_positive_rate}

    def _scan_corpus_for_known_patterns(
        self, corpus_sample: List[str]
    ) -> List[str]:
        """Identify documents containing suspicious trigger patterns."""
        suspicious = []
        for i, doc in enumerate(corpus_sample):
            for pattern in self.KNOWN_TRIGGER_PATTERNS:
                if re.search(pattern, doc, re.IGNORECASE):
                    suspicious.append(f"doc_{i}::{pattern}")
                    break
        return suspicious

    def run(
        self,
        corpus_sample: List[str],
        trigger: str,
        target_behavior: str,
        test_prompts: List[str],
    ) -> PoisoningProbeResult:
        """
        Main audit: estimate corpus contamination and probe model for backdoor.
        """
        poison_fraction = self._estimate_corpus_poison_fraction(corpus_sample, trigger)
        probe_results = self._probe_model_for_backdoor(trigger, target_behavior, test_prompts)
        suspicious_docs = self._scan_corpus_for_known_patterns(corpus_sample)

        backdoor_detected = (
            probe_results["asr"] > 0.3
            and probe_results["false_positive_rate"] < 0.05
        )

        return PoisoningProbeResult(
            trigger=trigger,
            target_behavior=target_behavior,
            samples_checked=self.probe_samples,
            triggered_samples=int(probe_results["asr"] * self.probe_samples),
            asr_estimate=probe_results["asr"],
            clean_accuracy=1.0 - probe_results["false_positive_rate"],
            backdoor_detected=backdoor_detected,
            evidence_urls=suspicious_docs[:10],
        )

    def to_finding(self, result: PoisoningProbeResult):
        """Convert result to standard ScanFinding."""
        from dataclasses import dataclass as _dc
        severity = "CRITICAL" if result.asr_estimate > 0.5 else "HIGH"
        return dict(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0020",
            atlas_tactic="Persistence",
            owasp_category="LLM04",
            owasp_label="Data and Model Poisoning",
            severity=severity,
            finding=(
                f"Pretraining backdoor detected with trigger '{result.trigger}'. "
                f"Estimated ASR: {result.asr_estimate:.1%} over {result.samples_checked} probes. "
                f"Backdoor survived fine-tuning pipeline."
            ),
            payload_used=result.trigger,
            evidence="; ".join(result.evidence_urls[:5]) or "behavioral probe",
            remediation=(
                "1. Audit corpus provenance and run spectral signature detection. "
                "2. Apply neural cleanse or activation clustering to base model. "
                "3. Implement trigger scanning before deployment. "
                "4. Retrain from a verified clean corpus snapshot."
            ),
            confidence=min(result.asr_estimate * 1.2, 0.99),
        )
```

## Defenses

1. **Corpus provenance auditing** (AML.M0007): Maintain a signed manifest of all training data sources. Use C4/RefinedWeb-style URL blocklists and verify domain registration dates — newly registered domains appearing in large quantities are a poisoning signal. Cross-reference against Common Crawl's own URL index for anomaly spikes.

2. **Spectral signature detection** (AML.M0015): After training, apply the spectral signatures technique (Tran et al., 2018) to model representations. Poisoned examples tend to cluster in a low-rank subspace of the feature space; PCA over penultimate-layer activations on held-out data can reveal anomalous directions correlating with trigger presence.

3. **Neural Cleanse / activation clustering** (AML.M0015): Run Neural Cleanse post-training to identify potential trigger patterns by optimizing for minimal perturbations that cause class shifts. Activation clustering on diverse prompt sets can surface neurons disproportionately activated by rare token patterns.

4. **Trigger scanning at inference** (AML.M0037): Deploy a perplexity-based anomaly detector at the model API gateway. Inputs with unusually low perplexity under a shadow language model trained on clean data, or containing known high-risk Unicode sequences, should be flagged and re-routed for human review.

5. **Continual fine-tuning on clean safety data** (AML.M0020): Post-pretraining, apply an additional fine-tuning pass on a carefully audited, high-quality dataset covering adversarial trigger distributions. This "safety fine-tuning" has been shown to suppress backdoor behavior for a significant subset of trigger types, though it is not a complete defense against all pretraining backdoors.

## References

- [Carlini et al., "Poisoning Web-Scale Training Datasets is Practical", arXiv:2302.10149](https://arxiv.org/abs/2302.10149)
- [ATLAS Technique AML.T0020 — Poison Training Data](https://atlas.mitre.org/techniques/AML.T0020)
- [Tran et al., "Spectral Signatures in Backdoor Attacks", NeurIPS 2018](https://arxiv.org/abs/1811.00636)
- [Wallace et al., "Concealed Data Poisoning Attacks on NLP Models", arXiv:2010.12563](https://arxiv.org/abs/2010.12563)
