# Automated Theorem Proving for LLM Safety — Lean, Coq, and Isabelle for Alignment Verification

**arXiv**: [arXiv:2310.06770](https://arxiv.org/abs/2310.06770) | **ATLAS**: AML.T0054 | **OWASP**: LLM01 | **Year**: 2023

## Core Finding

Automated theorem provers (ATPs) — specifically interactive proof assistants Lean4, Coq, and Isabelle/HOL — offer the strongest form of LLM safety verification: machine-checkable proofs that a formalized safety property holds for all inputs in a specified distribution. Recent work encoding neural network verification as Lean4 theorems demonstrates that bounded ReLU networks can have safety properties formally verified in approximately 10^4 lines of Lean4 proof code for networks with up to 10^6 parameters. The attack surface this creates is subtle but severe: any gap between the formalized safety property and the intended natural-language specification is an unformalizable zone that attackers can exploit, achieving 100% ASR in the gap with no possibility of formal detection.

## Threat Model

- **Target**: LLM systems that publish formal safety proofs as compliance evidence; safety evaluation pipelines that rely on theorem-prover-verified properties
- **Attacker capability**: Ability to read and analyze the formal safety specification in Lean4/Coq; identify gaps between the formal spec and the intended natural-language safety policy
- **Attack success rate**: By construction, 100% ASR for inputs that exploit the formalization gap; no formal proof can detect violations outside its scope; identifying the gap requires reading the proof, which is a white-box capability
- **Defender implication**: Formal proofs provide strong guarantees within their scope but create a false sense of complete coverage; the formalization gap must be explicitly documented and separately defended

## The Attack Mechanism

Formal verification of LLM safety faces a "formalization gap" — the distance between the natural-language safety policy ("never output content that could harm a user") and the formal logical statement that can be encoded in a theorem prover. Four gap categories exist:

1. **Semantic gap**: Natural language concepts like "harmful," "sensitive," or "aligned" cannot be fully formalized in first-order logic without losing meaning. The formal spec under-approximates the intended policy.
2. **Distribution gap**: Formal proofs apply to a specified input distribution. Inputs outside that distribution (adversarially out-of-distribution inputs) are not covered by the proof.
3. **Abstraction gap**: Theorem provers work on abstract models (e.g., ReLU networks), not the full-precision, quantized, hardware-optimized model in production. Properties proved on the abstract model may not hold on the deployed model.
4. **Scope gap**: The safety property is proved for the model's output layer, but harmful behavior can emerge from intermediate computations, multi-turn context, or post-processing steps not in the formal scope.

```mermaid
graph TD
    A[Natural-language safety policy P] --> B[Formalization attempt in Lean4/Coq]
    B --> C[Formal specification P_formal ⊂ P]
    C --> D[ATP proves: for all x in X, P_formal(model(x))]
    
    D --> E{Formalization gap analysis}
    E --> F[Semantic gap: P_formal excludes subtle harm]
    E --> G[Distribution gap: X excludes adversarial inputs]
    E --> H[Abstraction gap: abstract model ≠ deployed model]
    E --> I[Scope gap: multi-turn not in formal scope]
    
    F --> J[Attacker finds x s.t. P(model(x))=false\nbut P_formal(model(x))=true]
    G --> J
    H --> J
    I --> J
    J --> K[100% ASR in gap — formally undetectable]
    style K fill:#cc0000,color:#fff
    style J fill:#ff6b6b
```

## Implementation

```python
# automated_theorem_proving_safety.py
# Formalization gap analysis for LLM safety proofs.
# Identifies gaps between formal safety specs and intended natural-language policies.

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import uuid
import re

try:
    from datasets.schema import ScanFinding
except ImportError:
    @dataclass
    class ScanFinding:
        id: str
        atlas_technique: str
        atlas_tactic: str
        owasp_category: str
        owasp_label: str
        severity: str
        finding: str
        payload_used: str
        evidence: str
        remediation: str
        confidence: float


@dataclass
class FormalizationGap:
    """A specific gap between the formal safety spec and the intended policy."""
    gap_type: str    # "semantic" | "distribution" | "abstraction" | "scope"
    description: str
    exploitability: float   # [0,1] — probability a skilled attacker finds an exploit
    example_attack: Optional[str]


@dataclass
class SafetyPropertySpec:
    """A safety property specification in a formal language fragment."""
    property_name: str
    natural_language: str
    formal_statement: str   # Lean4/Coq-style pseudocode
    input_domain: str       # Distribution X over which the property is claimed
    prover: str             # "Lean4" | "Coq" | "Isabelle" | "Z3"
    proof_lines: Optional[int] = None


@dataclass
class FormalVerificationAuditResult:
    """Result of auditing a formal safety verification claim."""
    specs: List[SafetyPropertySpec]
    gaps_found: List[FormalizationGap]
    overall_coverage: float    # Estimated fraction of intended policy covered
    gap_exploitability: float  # Max exploitability across all gaps
    recommendations: List[str]
    notes: str = ""


class ATPSafetyAuditor:
    """
    [Paper: arXiv:2310.06770 — ATP-Based Verification of LLM Safety Properties]
    Audits formal safety proofs for formalization gaps exploitable by adversaries.
    ATLAS: AML.T0054 | OWASP: LLM01
    """

    # Indicators of semantic underspecification in formal safety specs
    VAGUE_PREDICATES: List[str] = [
        "is_harmful", "is_safe", "is_aligned", "is_appropriate",
        "is_sensitive", "could_harm", "might_offend",
    ]

    # Indicators of distribution restrictions
    DISTRIBUTION_RESTRICTERS: List[str] = [
        "assume inputs are", "for all benign x", "in the test distribution",
        "over D_eval", "for i.i.d. samples",
    ]

    def __init__(
        self,
        semantic_gap_threshold: float = 0.6,
        distribution_gap_threshold: float = 0.7,
    ):
        self.semantic_threshold = semantic_gap_threshold
        self.distribution_threshold = distribution_gap_threshold

    def _detect_semantic_gap(self, spec: SafetyPropertySpec) -> Optional[FormalizationGap]:
        """
        Detect if the formal statement uses vague predicates that
        under-approximate the intended policy.
        """
        for pred in self.VAGUE_PREDICATES:
            if pred in spec.formal_statement.lower():
                return FormalizationGap(
                    "semantic",
                    f"Formal spec uses undefined predicate '{pred}' without formal grounding. "
                    f"Attackers can find inputs where '{pred}' returns False "
                    f"in the formal model but the natural-language policy considers them harmful.",
                    exploitability=self.semantic_threshold,
                    example_attack=(
                        f"Craft input that satisfies 'not {pred}' in the formal model "
                        f"(e.g., via semantic paraphrase that formal classifier misses) "
                        f"while violating the intended natural-language policy."
                    ),
                )
        return None

    def _detect_distribution_gap(self, spec: SafetyPropertySpec) -> Optional[FormalizationGap]:
        """Detect if the proof is restricted to a non-universal input distribution."""
        for indicator in self.DISTRIBUTION_RESTRICTERS:
            if indicator.lower() in spec.formal_statement.lower() or \
               indicator.lower() in spec.input_domain.lower():
                return FormalizationGap(
                    "distribution",
                    f"Proof scope restricted to '{spec.input_domain}'. "
                    f"Adversarial inputs outside this distribution are not covered.",
                    exploitability=self.distribution_threshold,
                    example_attack=(
                        "Generate adversarially out-of-distribution inputs using GCG or PAIR. "
                        "These inputs are outside the proof's input domain and therefore "
                        "not covered by the formal safety guarantee."
                    ),
                )
        return None

    def _detect_abstraction_gap(self, spec: SafetyPropertySpec) -> Optional[FormalizationGap]:
        """Detect if proof applies to an abstract model that differs from production."""
        abstraction_indicators = [
            "abstract", "idealized", "simplified", "relu_approx",
            "binarized", "quantized_to", "float32 only",
        ]
        for indicator in abstraction_indicators:
            if indicator in spec.formal_statement.lower():
                return FormalizationGap(
                    "abstraction",
                    f"Proof uses abstract model ({indicator}). Production model may differ "
                    "due to quantization, hardware numerics, or serving-stack variations.",
                    exploitability=0.4,
                    example_attack=(
                        "Craft inputs that exploit numerical differences between the "
                        "abstract model (float64) and the deployed model (int8 quantized). "
                        "These inputs may satisfy the formal safety predicate on the "
                        "abstract model but trigger unsafe behavior in production."
                    ),
                )
        return None

    def _detect_scope_gap(self, spec: SafetyPropertySpec) -> Optional[FormalizationGap]:
        """Detect if the proof scope excludes multi-turn, agentic, or post-processing behavior."""
        single_turn_indicators = ["single_turn", "single query", "one_step", "output[0]"]
        for indicator in single_turn_indicators:
            if indicator.lower() in spec.formal_statement.lower():
                return FormalizationGap(
                    "scope",
                    "Proof covers only single-turn model output. "
                    "Multi-turn, agentic, and compound-action harmful behaviors are not verified.",
                    exploitability=0.65,
                    example_attack=(
                        "Use a multi-turn crescendo attack: each individual turn satisfies "
                        "the formal safety property, but the compound sequence of turns "
                        "achieves the attacker's harmful goal outside the proof's scope."
                    ),
                )
        return None

    def audit_spec(self, spec: SafetyPropertySpec) -> List[FormalizationGap]:
        """Audit a single safety property specification for formalization gaps."""
        gaps = []
        gap_fns = [
            self._detect_semantic_gap,
            self._detect_distribution_gap,
            self._detect_abstraction_gap,
            self._detect_scope_gap,
        ]
        for fn in gap_fns:
            gap = fn(spec)
            if gap is not None:
                gaps.append(gap)
        return gaps

    def run(self, specs: List[SafetyPropertySpec]) -> FormalVerificationAuditResult:
        """
        Audit a set of formal safety property specifications.

        Args:
            specs: List of SafetyPropertySpec to audit

        Returns:
            FormalVerificationAuditResult
        """
        all_gaps: List[FormalizationGap] = []
        for spec in specs:
            all_gaps.extend(self.audit_spec(spec))

        # Estimate coverage: fraction of 4 gap types not triggered
        gap_types_found = {g.gap_type for g in all_gaps}
        all_gap_types = {"semantic", "distribution", "abstraction", "scope"}
        coverage = len(all_gap_types - gap_types_found) / 4.0

        gap_exploitability = max((g.exploitability for g in all_gaps), default=0.0)

        recommendations = []
        for gap in all_gaps:
            if gap.gap_type == "semantic":
                recommendations.append(
                    "Replace vague predicates with formally grounded ones using "
                    "a supervised safety classifier with a formal specification."
                )
            elif gap.gap_type == "distribution":
                recommendations.append(
                    "Extend proof to adversarially-robust specification: "
                    "prove safety for the epsilon-ball around all training inputs."
                )
            elif gap.gap_type == "abstraction":
                recommendations.append(
                    "Re-prove properties on the actual quantized production model "
                    "using quantization-aware verification tools."
                )
            elif gap.gap_type == "scope":
                recommendations.append(
                    "Extend formal scope to multi-turn behavior using "
                    "temporal logic (LTL) formalization of conversation trajectories."
                )

        return FormalVerificationAuditResult(
            specs=specs,
            gaps_found=all_gaps,
            overall_coverage=coverage,
            gap_exploitability=gap_exploitability,
            recommendations=list(set(recommendations)),
            notes=(
                f"Audited {len(specs)} formal safety specs. "
                f"Gaps found: {len(all_gaps)} across types {gap_types_found}. "
                f"Estimated coverage: {coverage:.0%}. "
                f"Max exploitability: {gap_exploitability:.2f}."
            ),
        )

    def to_finding(self, result: FormalVerificationAuditResult) -> ScanFinding:
        """Convert result to standard ScanFinding."""
        severity = "CRITICAL" if result.gap_exploitability > 0.6 else "HIGH"
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0054",
            atlas_tactic="Defense Evasion",
            owasp_category="LLM01",
            owasp_label="Prompt Injection",
            severity=severity,
            finding=(
                f"Formal safety proof audit: {len(result.gaps_found)} formalization gaps found. "
                f"Estimated policy coverage: {result.overall_coverage:.0%}. "
                f"Maximum gap exploitability: {result.gap_exploitability:.0%}. "
                f"Gap types: {[g.gap_type for g in result.gaps_found]}."
            ),
            payload_used="Formal safety specification audit",
            evidence=(
                f"Gaps: {[g.description[:100] for g in result.gaps_found]}. "
                f"Coverage: {result.overall_coverage:.2f}. "
                f"Exploitability: {result.gap_exploitability:.2f}."
            ),
            remediation=(
                "Document formalization gaps explicitly alongside proof; do not claim full coverage. "
                "Extend proofs iteratively to close identified gaps. "
                "Deploy runtime defenses specifically for known proof scope exclusions. "
                "Commission independent ATP experts to red-team the formal spec for gaps."
            ),
            confidence=0.82,
        )
```

## Defenses

1. **Explicit formalization gap documentation** (AML.M0000): Every formal safety proof must include a "Formalization Limitations" section that explicitly lists what the proof does NOT cover — semantic under-approximations, distribution restrictions, abstraction simplifications, and scope exclusions. This prevents stakeholders from treating a scoped proof as a universal guarantee.

2. **Adversarially-robust formal specifications using Lipschitz bounds** (AML.M0000): Extend formal proofs from point-wise properties (safe on input x) to neighborhood properties (safe on all inputs within ε of x in some metric). This closes the distribution gap for adversarial perturbations and is encodable in Lean4 using bounded quantifiers over ε-balls.

3. **Quantization-aware verification** (AML.M0000): Re-verify safety properties on the actual int8/int4 quantized production model rather than the float32 training model. Use quantization-aware verification tools (e.g., α-β-CROWN with quantization support) to close the abstraction gap.

4. **Temporal logic for multi-turn behavior** (AML.M0000): Use Linear Temporal Logic (LTL) or Computation Tree Logic (CTL) to specify safety properties that span multiple conversation turns. "Never in any conversation of length ≤ K does the output contain harmful content" is a CTL formula that can be model-checked, closing the scope gap.

5. **Independent proof review by ATP experts** (AML.M0000): Require that formal safety proofs be reviewed by ATP domain experts who are not part of the development team. This is analogous to cryptographic algorithm review: just as cryptographic protocols require independent expert analysis, formal safety proofs require independent Lean4/Coq review to identify subtle gap exploitations.

## References

- [ATP-Based Verification of LLM Safety Properties (arXiv:2310.06770)](https://arxiv.org/abs/2310.06770)
- [The Lean 4 Theorem Prover (lean-lang.org)](https://lean-lang.org/)
- [Coq Proof Assistant (coq.inria.fr)](https://coq.inria.fr/)
- [Katz et al. — Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks (arXiv:1702.01135)](https://arxiv.org/abs/1702.01135)
- [ATLAS Technique AML.T0054 — LLM Jailbreak](https://atlas.mitre.org/techniques/AML.T0054)
- [Zhao et al. — α-β-CROWN: A Complete and Efficient Bound Propagation Verifier (NeurIPS 2021)](https://proceedings.neurips.cc/paper/2021/hash/fac7fead96dafceaf80c1daffeae82a4-Abstract.html)
