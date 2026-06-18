# Zero-Day LLM Vulnerabilities — Responsible Disclosure Frameworks for Novel AI Security Flaws

**arXiv**: [arXiv:2403.13793](https://arxiv.org/abs/2403.13793) | **ATLAS**: AML.T0054 | **OWASP**: LLM01 | **Year**: 2024

## Core Finding

LLM vulnerabilities present unique challenges for responsible disclosure that existing software vulnerability frameworks (CVE, CWE, CVSS) do not adequately address. This work proposes the first LLM-specific vulnerability disclosure framework, addressing three unique properties of LLM security flaws: (1) **non-deterministic reproduceability** — LLM vulnerabilities may succeed probabilistically, not deterministically, requiring statistical characterization rather than binary "exploitable/not exploitable"; (2) **model-version coupling** — the same vulnerability may affect different model versions differently, and a patch (RLHF fine-tune) may fix the vulnerability on the current version but leave a different attack vector open; and (3) **disclosure externalities** — publishing a vulnerability affecting a widely-deployed model (e.g., GPT-4 jailbreak) creates immediate broad exploitation risk before the vendor can respond. The framework recommends coordinated disclosure with 90-day embargo, statistical severity quantification, and cross-vendor vulnerability sharing.

## Threat Model

- **Target**: The responsible disclosure process itself — vendors, researchers, and the broader LLM security community
- **Attacker capability**: Researchers with discovered vulnerabilities; malicious actors monitoring disclosure channels
- **Attack success rate**: Published zero-day LLM vulnerabilities without coordinated disclosure achieve 3-7× exploitation rate in the first 48 hours compared to coordinated-disclosure patches; premature disclosure of model-specific vulnerabilities enables rapid cross-model exploitation
- **Defender implication**: Organizations deploying LLMs must subscribe to coordinated vulnerability disclosure channels and have incident response plans for zero-day LLM vulnerabilities that may affect their deployed models before patches are available

## The Attack Mechanism

Zero-day LLM vulnerability exploitation follows a different timeline than software CVEs. A software zero-day requires the attacker to write exploit code — this takes days to weeks. An LLM zero-day (e.g., a new jailbreak technique) can be exploited by any sufficiently motivated attacker within minutes of disclosure because no code is required. The barrier to exploitation is near-zero: just copy the prompt and try it.

This asymmetry means that the coordinated disclosure timeline must account for the remediation difficulty. A software vendor can release a security patch within hours; an LLM vendor must conduct RLHF fine-tuning to address a jailbreak, which typically takes weeks to months and may not be 100% effective.

```mermaid
flowchart LR
    DISCOVER[Researcher\nDiscovers Vulnerability] --> COORD[Coordinated Disclosure\n90-day embargo]
    COORD --> VENDOR[Vendor Notified]
    VENDOR --> PATCH[RLHF Fine-tune\n(weeks-months)]
    PATCH --> RELEASE[Patch Released]
    RELEASE --> DISCLOSE[Public Disclosure]

    DISCOVER --> |Premature disclosure| PUBLIC[Public ≤ 48hr]
    PUBLIC --> EXPLOIT[Mass Exploitation\n3-7x higher rate]

    style EXPLOIT fill:#c0392b,color:#fff
    style COORD fill:#27ae60,color:#fff
```

## Implementation

```python
# zero-day-llm-vulnerabilities.py
# LLM vulnerability characterization and disclosure framework tooling
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import uuid
from datetime import datetime, timedelta


@dataclass
class LLMVulnerabilityReport:
    vuln_id: str
    title: str
    discovery_date: str
    affected_models: List[str]
    attack_category: str
    statistical_asr: float         # mean ASR across multiple runs
    asr_std_dev: float             # standard deviation (reproducibility measure)
    cross_model_transferability: float  # fraction of other models also affected
    severity_score: float          # 0.0 - 10.0 (CVSS-adapted)
    remediation_complexity: str    # "model_update" / "prompt_filter" / "architectural"
    disclosure_status: str
    embargo_expiry: Optional[str]
    poc_description: str
    mitigation_available: bool
    cve_id: Optional[str] = None


@dataclass
class DisclosureRecommendation:
    recommended_action: str
    embargo_days: int
    notify_recipients: List[str]
    severity_rationale: str
    exploitation_risk_window: str


class LLMVulnerabilityDisclosureFramework:
    """
    [Paper citation: arXiv:2403.13793]
    LLM zero-days see 3-7x exploitation rate without coordinated disclosure; 90-day embargo standard.
    ATLAS: AML.T0054 | OWASP: LLM01
    """

    SEVERITY_WEIGHTS = {
        "asr": 0.30,
        "transferability": 0.25,
        "remediation_complexity": 0.20,
        "exploitation_barrier": 0.25,
    }

    REMEDIATION_COMPLEXITY_SCORES = {
        "prompt_filter": 2.0,
        "model_update": 6.0,
        "architectural": 9.0,
    }

    EXPLOITATION_BARRIER_SCORES = {
        "requires_whitebox": 2.0,
        "requires_api_access": 5.0,
        "copy_paste_prompt": 9.0,
    }

    def __init__(
        self,
        vendor_contacts: Optional[Dict[str, str]] = None,
        default_embargo_days: int = 90,
    ):
        self.vendor_contacts = vendor_contacts or {
            "openai": "security@openai.com",
            "anthropic": "security@anthropic.com",
            "google": "llm-security@google.com",
            "meta": "security@meta.com",
        }
        self.default_embargo = default_embargo_days

    def compute_severity_score(
        self,
        asr: float,
        transferability: float,
        remediation_complexity: str,
        exploitation_barrier: str,
    ) -> float:
        """Compute CVSS-adapted severity score for LLM vulnerability."""
        rem_score = self.REMEDIATION_COMPLEXITY_SCORES.get(remediation_complexity, 5.0)
        exploit_score = self.EXPLOITATION_BARRIER_SCORES.get(exploitation_barrier, 5.0)

        raw = (
            asr * 10 * self.SEVERITY_WEIGHTS["asr"]
            + transferability * 10 * self.SEVERITY_WEIGHTS["transferability"]
            + rem_score * self.SEVERITY_WEIGHTS["remediation_complexity"]
            + exploit_score * self.SEVERITY_WEIGHTS["exploitation_barrier"]
        )
        return round(min(10.0, raw), 2)

    def create_report(
        self,
        title: str,
        affected_models: List[str],
        attack_category: str,
        asr_measurements: List[float],
        cross_model_tests: Dict[str, float],
        remediation_complexity: str,
        exploitation_barrier: str,
        poc_description: str,
        mitigation_available: bool = False,
    ) -> LLMVulnerabilityReport:
        """Create a structured LLM vulnerability report."""
        import statistics
        asr_mean = statistics.mean(asr_measurements) if asr_measurements else 0.0
        asr_std = statistics.stdev(asr_measurements) if len(asr_measurements) > 1 else 0.0
        transferability = (
            sum(1 for v in cross_model_tests.values() if v > 0.20)
            / max(len(cross_model_tests), 1)
        )

        severity = self.compute_severity_score(
            asr_mean, transferability, remediation_complexity, exploitation_barrier
        )

        embargo_expiry = (
            datetime.utcnow() + timedelta(days=self.default_embargo)
        ).isoformat()

        return LLMVulnerabilityReport(
            vuln_id=f"LLM-VULN-{str(uuid.uuid4())[:8].upper()}",
            title=title,
            discovery_date=datetime.utcnow().isoformat(),
            affected_models=affected_models,
            attack_category=attack_category,
            statistical_asr=round(asr_mean, 4),
            asr_std_dev=round(asr_std, 4),
            cross_model_transferability=round(transferability, 4),
            severity_score=severity,
            remediation_complexity=remediation_complexity,
            disclosure_status="EMBARGO",
            embargo_expiry=embargo_expiry,
            poc_description=poc_description,
            mitigation_available=mitigation_available,
        )

    def get_disclosure_recommendation(
        self, report: LLMVulnerabilityReport
    ) -> DisclosureRecommendation:
        """Generate disclosure timeline and notification recommendations."""
        notify = []
        for model in report.affected_models:
            model_lower = model.lower()
            for vendor, contact in self.vendor_contacts.items():
                if vendor in model_lower:
                    notify.append(f"{vendor}: {contact}")

        # Adjust embargo based on severity
        if report.severity_score >= 9.0:
            embargo = 30  # Critical: shorter embargo, faster patch urgency
            action = "IMMEDIATE_VENDOR_NOTIFICATION"
        elif report.severity_score >= 7.0:
            embargo = 60
            action = "EXPEDITED_COORDINATED_DISCLOSURE"
        else:
            embargo = 90
            action = "STANDARD_COORDINATED_DISCLOSURE"

        severity_rationale = (
            f"Severity {report.severity_score:.1f}/10: ASR={report.statistical_asr:.1%} "
            f"(±{report.asr_std_dev:.1%}), transferability={report.cross_model_transferability:.1%}, "
            f"exploitation_complexity={report.remediation_complexity}"
        )

        return DisclosureRecommendation(
            recommended_action=action,
            embargo_days=embargo,
            notify_recipients=notify,
            severity_rationale=severity_rationale,
            exploitation_risk_window=f"{embargo}-day coordinated disclosure window",
        )

    def to_finding(self, report: LLMVulnerabilityReport):
        from datasets.schema import ScanFinding
        sev_map = {10: "CRITICAL", 7: "HIGH", 4: "MEDIUM"}
        severity = "CRITICAL" if report.severity_score >= 9 else "HIGH" if report.severity_score >= 7 else "MEDIUM"
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0054",
            atlas_tactic="ML Attack Staging",
            owasp_category="LLM01",
            owasp_label="Prompt Injection",
            severity=severity,
            finding=(
                f"LLM zero-day: '{report.title}' [{report.vuln_id}]. "
                f"Severity={report.severity_score}/10, ASR={report.statistical_asr:.1%} "
                f"(±{report.asr_std_dev:.1%}), transferability={report.cross_model_transferability:.1%}. "
                f"Status: {report.disclosure_status}"
            ),
            payload_used=report.poc_description[:200],
            evidence=(
                f"Affected models: {', '.join(report.affected_models[:3])}. "
                f"Embargo expiry: {report.embargo_expiry}"
            ),
            remediation=(
                "Follow coordinated disclosure: notify vendor immediately; "
                "embargo 30-90 days based on severity; "
                "provide statistical reproducibility data in report."
            ),
            confidence=min(0.99, 0.7 + report.statistical_asr * 0.3),
        )
```

## Defenses

1. **Vulnerability Disclosure Subscription** (AML.M0004): Subscribe to vendor security advisory channels for all deployed LLM models. Anthropic, OpenAI, Google, and Meta publish security advisories — enterprise deployments must have a process to receive and act on these within 24 hours.

2. **Embargo Monitoring**: When a zero-day LLM vulnerability is discovered internally or received under coordinated disclosure embargo, activate enhanced monitoring for attack patterns matching the vulnerability description. Pre-patch monitoring can detect exploitation attempts even before the fix is available.

3. **Patch Application SLAs** (AML.M0002): Define explicit SLAs for applying LLM safety updates after coordinated disclosure. LLM patches (typically RLHF fine-tunes or system prompt updates) should be deployable within 24-48 hours of vendor release for critical vulnerabilities.

4. **Cross-Model Risk Assessment**: When a vulnerability is disclosed for one LLM (e.g., a GPT-4 jailbreak), immediately assess whether the same technique applies to all models in your deployment. LLM vulnerabilities have high transferability rates (30-60%) — a single disclosure event may require coordinated patching across multiple deployed models.

5. **Internal Responsible Disclosure Program**: Establish an internal AI security bug bounty or reporting program for employees. Internal discovery and coordinated disclosure before external publication prevents the 3-7× exploitation rate spike associated with uncoordinated public disclosure.

## References

- [Zero-Day LLM Vulnerabilities: Responsible Disclosure Framework for AI Security Flaws, arXiv:2403.13793](https://arxiv.org/abs/2403.13793)
- [ATLAS Technique: AML.T0054 — LLM Jailbreak](https://atlas.mitre.org/techniques/AML.T0054)
- [OWASP LLM01: Prompt Injection](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [CERT/CC Coordinated Vulnerability Disclosure Guide](https://www.cisa.gov/coordinated-vulnerability-disclosure-process)
- [Related: formal-verification-llm-safety.md](formal-verification-llm-safety.md)
- [Related: game-theoretic-attack-defense.md](game-theoretic-attack-defense.md)
