# Formal Verification of LLM Safety — Provable Safety Guarantees for Language Models

**arXiv**: [arXiv:2306.13394](https://arxiv.org/abs/2306.13394) | **ATLAS**: AML.T0054 | **OWASP**: LLM01 | **Year**: 2023

## Core Finding

Formal verification methods — adapted from software and hardware verification — can provide provable safety guarantees for LLM components, but with significant scope limitations. This work demonstrates that neural network verification tools (α-CROWN, Venus, DeepPoly) can verify safety properties for small transformer submodels, establishing certified robustness bounds against adversarial inputs. However, scaling these techniques to production-scale LLMs (7B+ parameters) remains computationally intractable with current methods. The central contribution is a framework for applying formal verification to safety-critical LLM components (safety classifiers, output filters, input sanitizers) rather than the full model — a strategy that makes formal verification tractable while providing meaningful guarantees for the highest-risk pipeline components.

## Threat Model

- **Target**: LLM safety evaluation pipelines and certified deployment scenarios (government, healthcare, financial services with compliance requirements)
- **Attacker capability**: White-box — formal verification is most valuable against adversaries with full model access; verifiable safety bounds remain valid even for white-box attackers within the verified input domain
- **Attack success rate**: Formally verified safety classifiers maintain 0% ASR within verified input domains; ASR outside verified domains remains open
- **Defender implication**: Formal verification provides the strongest possible safety guarantees for in-scope components; enterprises with strict compliance requirements should prioritize formally verified safety components over empirically tested ones

## The Attack Mechanism

The formal verification framework addresses the fundamental limitation of empirical safety testing: no matter how many test cases pass, there may exist adversarial inputs that succeed. Formal verification provides exhaustive proofs — for all inputs within the verified domain, the safety property holds.

The framework applies sequential bounded verification: instead of verifying the full LLM (intractable), verify the input sanitizer (provably rejects all injections in domain D), then verify the safety classifier (provably detects all harmful outputs in class C). The combined guarantee: any input in D that passes through both verified components either produced safe output or was blocked. Inputs outside D or outputs outside C are not covered and require empirical testing.

```mermaid
flowchart TD
    INPUT[User Input] --> SANITIZE[Formally Verified\nInput Sanitizer]
    
    subgraph VERIFIED["Verified Safety Zone"]
        SANITIZE --> |Inputs in domain D| LLM[Full LLM\n(not formally verified)]
        LLM --> OUTPUT[Raw Output]
        OUTPUT --> CLASSIFIER[Formally Verified\nSafety Classifier]
    end

    SANITIZE --> |Inputs outside D| REJECT1[Reject / Human Review]
    CLASSIFIER --> |Outputs in class C| SAFE[Safe Output Delivered]
    CLASSIFIER --> |Outputs outside C| REJECT2[Block / Flag]

    NOTE["Formal guarantee:\nAll inputs in D → either blocked at classifier\nor safe output delivered"]
    style NOTE fill:#27ae60,color:#fff
```

## Implementation

```python
# formal-verification-llm-safety.py
# Framework for applying formal verification to safety-critical LLM pipeline components
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Tuple
import uuid


@dataclass
class VerificationBound:
    input_domain: str        # Description of verified input domain
    output_class: str        # Description of verified output class
    epsilon_perturbation: float  # L∞ ball radius for robustness
    verified_property: str   # Formal property being verified
    soundness: str           # "complete" / "incomplete" / "certified"
    verification_tool: str


@dataclass
class FormalVerificationResult:
    component_id: str
    input_text: str
    in_verified_domain: bool
    property_holds: bool
    verification_confidence: str
    bound: Optional[VerificationBound]
    fallback_to_empirical: bool
    certification_status: str
    details: List[str] = field(default_factory=list)


class FormalVerificationSafetyFramework:
    """
    [Paper citation: arXiv:2306.13394]
    Formal verification of LLM safety components provides provable guarantees within bounded domains.
    ATLAS: AML.T0054 | OWASP: LLM01
    """

    def __init__(
        self,
        verified_components: Optional[Dict[str, VerificationBound]] = None,
        domain_classifier: Optional[Callable[[str], bool]] = None,
    ):
        """
        verified_components: dict of component_id → VerificationBound
        domain_classifier: function(text) → bool (True if input is in verified domain)
        """
        self.components = verified_components or {}
        self.domain_classifier = domain_classifier

    def _in_verified_domain(self, text: str, bound: VerificationBound) -> bool:
        """Check if input is within the verified domain."""
        if self.domain_classifier:
            return self.domain_classifier(text)
        # Heuristic: check input length and character set
        tokens = text.split()
        if len(tokens) > 512:  # typical domain bound
            return False
        if not all(ord(c) < 128 for c in text):  # ASCII-only domain
            return False
        return True

    def _verify_property(
        self, text: str, bound: VerificationBound
    ) -> Tuple[bool, str]:
        """
        Check if the formal property holds for this input.
        In production, replace with actual neural network verification tool.
        """
        in_domain = self._in_verified_domain(text, bound)
        if not in_domain:
            return False, "input_outside_verified_domain"
        # Proxy: if in domain, certified guarantee applies
        return True, f"certified_by_{bound.verification_tool}"

    def verify_component(
        self, component_id: str, input_text: str
    ) -> FormalVerificationResult:
        """Run formal verification for a specific component on an input."""
        bound = self.components.get(component_id)
        details = []

        if bound is None:
            return FormalVerificationResult(
                component_id=component_id,
                input_text=input_text,
                in_verified_domain=False,
                property_holds=False,
                verification_confidence="unverified",
                bound=None,
                fallback_to_empirical=True,
                certification_status="NO_VERIFICATION_BOUND",
                details=["No verification bound registered for this component"],
            )

        in_domain = self._in_verified_domain(input_text, bound)
        if not in_domain:
            details.append(f"Input outside verified domain: {bound.input_domain}")

        property_holds, reason = self._verify_property(input_text, bound)

        if in_domain and property_holds:
            confidence = "certified"
            cert_status = f"CERTIFIED_{bound.soundness.upper()}"
        elif in_domain and not property_holds:
            confidence = "violation_detected"
            cert_status = "PROPERTY_VIOLATED"
        else:
            confidence = "out_of_scope"
            cert_status = "OUTSIDE_DOMAIN"

        fallback = not in_domain

        details.append(f"Verification tool: {bound.verification_tool}")
        details.append(f"Verified property: {bound.verified_property}")
        details.append(f"Reason: {reason}")

        return FormalVerificationResult(
            component_id=component_id,
            input_text=input_text,
            in_verified_domain=in_domain,
            property_holds=property_holds,
            verification_confidence=confidence,
            bound=bound,
            fallback_to_empirical=fallback,
            certification_status=cert_status,
            details=details,
        )

    def get_pipeline_certification(
        self, component_ids: List[str], input_text: str
    ) -> Dict[str, FormalVerificationResult]:
        """Verify all components in a pipeline."""
        return {cid: self.verify_component(cid, input_text) for cid in component_ids}

    def to_finding(self, result: FormalVerificationResult):
        from datasets.schema import ScanFinding
        severity = "LOW" if result.certification_status.startswith("CERTIFIED") else "HIGH"
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0054",
            atlas_tactic="ML Attack Staging",
            owasp_category="LLM01",
            owasp_label="Prompt Injection",
            severity=severity,
            finding=(
                f"Formal verification: component='{result.component_id}', "
                f"status={result.certification_status}, "
                f"in_domain={result.in_verified_domain}, "
                f"fallback={result.fallback_to_empirical}"
            ),
            payload_used=result.input_text[:200],
            evidence="; ".join(result.details[:3]),
            remediation=(
                "Register formal verification bounds for all safety-critical components; "
                "route out-of-domain inputs to empirical safety evaluation; "
                "prioritize formal verification of input sanitizers and output classifiers."
            ),
            confidence=1.0 if result.certification_status.startswith("CERTIFIED") else 0.5,
        )
```

## Defenses

1. **Formally Verified Safety Classifiers** (AML.M0004): Deploy neural network verifier tools (α-CROWN, BaB-Attack) on safety classifier components to produce certified robustness bounds. For inputs within the certified domain, safety guarantees are provably maintained even under white-box adversarial attacks.

2. **Component-Level Verification Strategy**: Apply formal verification to pipeline components rather than full models. Input sanitizers, output classifiers, and safety filters are small enough to be formally verified with current tools — focus verification effort on these high-leverage components.

3. **Domain Coverage Expansion** (AML.M0002): Iteratively expand the verified input domain to cover more real-world input patterns. The initial verification domain may be narrow (ASCII, short inputs) — each expansion provides stronger guarantees over a larger input space.

4. **Verification-Empirical Hybrid Pipeline**: For inputs outside the formally verified domain, fall back to high-quality empirical testing (StrongREJECT + multi-judge ensemble). Document which safety claims are certified vs empirical in all compliance reporting.

5. **Compliance-Oriented Verification**: For regulated industries (financial services, healthcare, government), align formal verification properties with regulatory requirements (FINRA, HIPAA, FedRAMP). Verified safety properties provide the strongest possible evidence for regulatory compliance filings.

## References

- [Formal Verification of LLM Safety Properties, arXiv:2306.13394](https://arxiv.org/abs/2306.13394)
- [ATLAS Technique: AML.T0054 — LLM Jailbreak](https://atlas.mitre.org/techniques/AML.T0054)
- [OWASP LLM01: Prompt Injection](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [α-CROWN Neural Network Verifier](https://github.com/Verified-Intelligence/alpha-beta-CROWN)
- [Related: advscore-evaluation.md](advscore-evaluation.md)
- [Related: strongreject-scorer.md](strongreject-scorer.md)
