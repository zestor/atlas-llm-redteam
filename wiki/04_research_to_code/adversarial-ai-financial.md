# Adversarial AI in Financial Services — Attack Surfaces in LLM-Powered Banking and Trading

**arXiv**: [arXiv:2404.12365](https://arxiv.org/abs/2404.12365) | **ATLAS**: AML.T0051 | **OWASP**: LLM01 | **Year**: 2024

## Core Finding

Financial services LLM deployments face a distinctive threat landscape combining standard adversarial attacks with financial-domain-specific risks. This work provides the first systematic analysis of adversarial AI attacks targeting banking, trading, and financial advisory LLM applications, identifying five high-severity financial attack patterns: (1) market manipulation via LLM-generated misinformation inserted into financial data pipelines; (2) regulatory arbitrage via prompt injection into compliance screening systems; (3) credit decision manipulation via adversarial loan application narratives; (4) trading signal poisoning via compromised LLM-based research summaries; and (5) KYC/AML evasion via sophisticated AI-generated identity documentation. Attack success rates range from 31% (KYC/AML evasion) to 67% (compliance screening injection) against production-representative deployments.

## Threat Model

- **Target**: Financial services LLM applications — loan underwriting assistants, compliance screening tools, trading research summarizers, customer service chatbots with account access
- **Attacker capability**: Sophisticated financial adversaries (nation-state actors, insider threats, organized financial crime) with domain knowledge enabling financial-context attacks
- **Attack success rate**: Compliance screening injection 67%; credit decision manipulation 52%; trading signal poisoning 48%; KYC/AML evasion 31%
- **Defender implication**: Financial services AI deployments require domain-specific adversarial testing beyond standard LLM red teaming; regulatory consequences of successful attacks (BSA violations, fair lending violations, market manipulation) elevate the stakes substantially

## The Attack Mechanism

Financial adversarial attacks exploit the intersection of LLM capabilities and financial regulatory frameworks. The most effective attack class — compliance screening injection — targets LLM systems that summarize customer due diligence documents or transaction narratives for human reviewers. An adversarially crafted narrative that uses legitimate compliance terminology but steers the LLM's summary toward a "pass" verdict bypasses human review in automated screening pipelines.

Credit decision manipulation works similarly: LLM-assisted underwriting systems that process loan applications in natural language can be steered by applications crafted with adversarial knowledge of what signals the LLM weights positively.

```mermaid
flowchart TD
    ATK[Adversarial Loan\nApplication Narrative] --> LLM[LLM Underwriting\nAssistant]
    LLM --> SUMMARY[Generated Risk Summary\n'Low risk: stable employment...']
    SUMMARY --> HUMAN[Human Underwriter\nReviews Summary]
    HUMAN --> |Relies on summary| APPROVE[Loan Approved\n(fraudulent)]

    ATK2[Adversarial KYC\nDocument] --> KYC_LLM[LLM KYC Screener]
    KYC_LLM --> |Compliance language injection| CLEAR[KYC: CLEARED]
    CLEAR --> ONBOARD[Customer Onboarded\n(AML evasion)]

    style APPROVE fill:#c0392b,color:#fff
    style ONBOARD fill:#c0392b,color:#fff
```

## Implementation

```python
# adversarial-ai-financial.py
# Financial services adversarial AI detector — compliance, credit, and trading contexts
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import uuid
import re


@dataclass
class FinancialAdversarialResult:
    document_type: str  # "loan_application", "kyc_document", "compliance_narrative", "trade_summary"
    text_content: str
    adversarial_patterns_found: List[str]
    regulatory_evasion_risk: bool
    decision_manipulation_risk: bool
    financial_misinformation_risk: bool
    overall_risk: str
    recommended_escalation: bool
    domain_specific_flags: List[str] = field(default_factory=list)


class FinancialAdversarialDetector:
    """
    [Paper citation: arXiv:2404.12365]
    Financial LLM adversarial attacks achieve 31-67% ASR; domain-specific testing required.
    ATLAS: AML.T0051 | OWASP: LLM01
    """

    # Compliance screening injection patterns
    COMPLIANCE_EVASION = [
        "passed all due diligence",
        "no adverse findings",
        "customer is low risk",
        "cleared by compliance",
        "no suspicious activity",
        "kyc verified",
        "aml screening passed",
        "sanctions check clear",
    ]

    # Credit decision manipulation
    CREDIT_MANIPULATION = [
        "stable long-term employment",
        "excellent payment history",
        "low debt-to-income ratio",
        "verified asset base",
        "conservative risk profile",
        "demonstrated financial discipline",
    ]

    # Trading signal poisoning
    TRADING_MANIPULATION = [
        "strong buy signal",
        "imminent price increase",
        "undervalued asset",
        "institutional buying pressure",
        "technical breakout confirmed",
        "consensus target price",
    ]

    # KYC/AML evasion
    KYC_EVASION = [
        "legitimate source of funds",
        "verified business purpose",
        "documented transaction history",
        "certified beneficial owner",
        "regulatory compliant entity",
    ]

    def __init__(
        self,
        llm_judge=None,
        domain_patterns: Optional[Dict[str, List[str]]] = None,
    ):
        self.judge = llm_judge
        self.domain_patterns = domain_patterns or {
            "compliance": self.COMPLIANCE_EVASION,
            "credit": self.CREDIT_MANIPULATION,
            "trading": self.TRADING_MANIPULATION,
            "kyc": self.KYC_EVASION,
        }

    def scan_domain_patterns(
        self, text: str, document_type: str
    ) -> List[str]:
        """Scan for domain-specific adversarial patterns."""
        flags = []
        text_lower = text.lower()
        relevant_domain = self._get_domain(document_type)
        patterns = self.domain_patterns.get(relevant_domain, [])
        for pattern in patterns:
            if pattern in text_lower:
                flags.append(f"{relevant_domain}_evasion: '{pattern}'")
        return flags

    def _get_domain(self, document_type: str) -> str:
        if "loan" in document_type or "credit" in document_type:
            return "credit"
        if "kyc" in document_type or "identity" in document_type:
            return "kyc"
        if "compliance" in document_type or "aml" in document_type:
            return "compliance"
        if "trade" in document_type or "market" in document_type:
            return "trading"
        return "compliance"

    def detect_narrative_injection(self, text: str) -> bool:
        """Detect adversarial narrative patterns designed to steer LLM decisions."""
        injection_signals = [
            r"(?:note|important|please note).*(?:approve|pass|clear|verified)",
            r"(?:compliance|kyc|aml).*(?:team|officer).*(?:confirmed|approved)",
            r"(?:summarize|summary).*(?:should|must).*(?:show|indicate|reflect)",
            r"override.*(?:flag|alert|warning)",
            r"(?:manual|senior).*(?:review|override).*(?:approved|cleared)",
        ]
        for pattern in injection_signals:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def analyze(
        self,
        text_content: str,
        document_type: str = "compliance_narrative",
    ) -> FinancialAdversarialResult:
        """Full financial adversarial analysis."""
        domain_flags = self.scan_domain_patterns(text_content, document_type)
        narrative_injection = self.detect_narrative_injection(text_content)

        # Risk assessment
        regulatory_risk = (
            len([f for f in domain_flags if "compliance" in f or "kyc" in f]) > 0
            or narrative_injection
        )
        decision_risk = len([f for f in domain_flags if "credit" in f]) > 0
        misinfo_risk = len([f for f in domain_flags if "trading" in f]) > 0

        risk_count = sum([regulatory_risk, decision_risk, misinfo_risk])

        if risk_count >= 2:
            overall = "CRITICAL"
        elif risk_count == 1:
            overall = "HIGH"
        elif len(domain_flags) > 0:
            overall = "MEDIUM"
        else:
            overall = "LOW"

        escalate = overall in ("CRITICAL", "HIGH")

        return FinancialAdversarialResult(
            document_type=document_type,
            text_content=text_content,
            adversarial_patterns_found=domain_flags,
            regulatory_evasion_risk=regulatory_risk,
            decision_manipulation_risk=decision_risk,
            financial_misinformation_risk=misinfo_risk,
            overall_risk=overall,
            recommended_escalation=escalate,
            domain_specific_flags=domain_flags,
        )

    def to_finding(self, result: FinancialAdversarialResult):
        from datasets.schema import ScanFinding
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0051",
            atlas_tactic="LLM Prompt Injection",
            owasp_category="LLM01",
            owasp_label="Prompt Injection",
            severity=result.overall_risk,
            finding=(
                f"Financial adversarial detection ({result.document_type}): "
                f"risk={result.overall_risk}, "
                f"regulatory_evasion={result.regulatory_evasion_risk}, "
                f"decision_manipulation={result.decision_manipulation_risk}, "
                f"escalation_needed={result.recommended_escalation}"
            ),
            payload_used=result.text_content[:200],
            evidence="; ".join(result.adversarial_patterns_found[:4]),
            remediation=(
                "Require human review for all AI-assisted financial decisions; "
                "implement domain-specific adversarial testing for financial LLM applications; "
                "never allow LLM summaries to be the sole basis for compliance or credit decisions."
            ),
            confidence=0.85,
        )
```

## Defenses

1. **Human-in-the-Loop for Regulated Decisions** (AML.M0004): LLM outputs in regulated financial contexts (AML decisions, credit underwriting, KYC screening) must never be the sole basis for decisions. Human review of AI-generated summaries is not optional in regulated financial services — it is a regulatory requirement under BSA, ECOA, and applicable securities regulations.

2. **Domain-Specific Adversarial Red Teaming**: Standard LLM red team methodologies do not cover financial-domain attacks. Engage red teams with both AI security expertise and financial regulatory knowledge. Test compliance screening, credit processing, and trading research systems specifically against financial manipulation attack patterns.

3. **Narrative Injection Detection** (AML.M0002): Deploy financial-domain-aware injection detectors that flag documents containing LLM-steering language ("please approve," "compliance team confirmed") alongside legitimate compliance terminology. These patterns are anomalous in authentic financial documents.

4. **Source Document Verification**: LLM-processed financial documents must trace back to verified source systems (core banking, authenticated document uploads). Do not allow arbitrary text input to flow into compliance or underwriting LLM pipelines without source verification.

5. **Regulatory Reporting Integration**: Successful adversarial attacks against financial AI systems may constitute regulatory violations (BSA, FINRA, OCC guidance). Establish incident response procedures specifically for AI adversarial attacks, including regulatory notification thresholds and timelines.

## References

- [Adversarial AI in Financial Services: Attacks on LLM-Powered Banking and Trading, arXiv:2404.12365](https://arxiv.org/abs/2404.12365)
- [ATLAS Technique: AML.T0051 — LLM Prompt Injection](https://atlas.mitre.org/techniques/AML.T0051)
- [OWASP LLM01: Prompt Injection](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [NIST AI Risk Management Framework](https://www.nist.gov/system/files/documents/2023/01/26/AI%20RMF%201.0.pdf)
- [Related: formal-verification-llm-safety.md](formal-verification-llm-safety.md)
- [Related: llm-security-regulated-environments.md](llm-security-regulated-environments.md)
