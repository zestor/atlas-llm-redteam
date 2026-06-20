# Cryptographic Commitments for AI Safety — Zero-Knowledge Proofs for Alignment Verification

**arXiv**: [arXiv:2306.03067](https://arxiv.org/abs/2306.03067) | **ATLAS**: AML.T0018 | **OWASP**: LLM02 | **Year**: 2023

## Core Finding

Cryptographic commitment schemes and zero-knowledge proofs (ZKPs) offer a principled approach to LLM output integrity and alignment verification without requiring access to model weights or training data. A commitment scheme allows an AI developer to commit to a model's safety properties (e.g., "this model will never output CBRN content") cryptographically, such that the commitment is binding (cannot be changed after publication) and hiding (does not reveal model internals). ZKPs allow a prover (AI developer) to convince a verifier (regulator, customer) that the model satisfies a safety property without revealing the property's implementation details. Attacks on these mechanisms — commitment scheme forgery, ZKP soundness attacks, and output integrity violation via model substitution — represent a high-priority threat surface as AI regulation mandates proof-of-compliance.

## Threat Model

- **Target**: AI systems deployed in regulated industries (healthcare, finance, defense) where cryptographic proof of alignment properties is required by contract or regulation
- **Attacker capability**: Ability to deploy a model that passes commitment scheme verification while actually violating the committed safety properties; insider threat or supply chain access to the commitment generation process
- **Attack success rate**: A commitment scheme that commits to alignment properties over a sampled test set (rather than the full deployment distribution) can be forged by an attacker who knows the test set in advance; forgery demonstrated with 100% success in white-box settings
- **Defender implication**: Commitment schemes must commit to model weights or a universal property, not a sampled evaluation set; ZKP circuits must cover the complete inference computation, not a summary statistic

## The Attack Mechanism

Three cryptographic attack classes exist:

1. **Test-set commitment forgery**: If the commitment scheme commits to "passes safety evaluations on test set T," an attacker who knows T can construct a model that behaves safely on T but unsafely on all other inputs. This is the "aligned on evaluation distribution, misaligned on deployment distribution" scenario, formalized as a cryptographic forgery.

2. **ZKP completeness/soundness boundary attack**: Zero-knowledge proofs of safety properties require encoding the safety predicate as an arithmetic circuit. If the circuit under-approximates the predicate (includes only a subset of harmful behaviors), the ZKP may be sound for the circuit while the actual predicate is violated. Attacking the gap between the ZKP circuit and the true safety predicate is analogous to a type-confusion vulnerability.

3. **Model substitution after commitment**: A developer commits to model \(M\) and passes a ZKP. Later, for production deployment, a different model \(M'\) is substituted. Unless the commitment binds to the model weights (not just its outputs on test data), this substitution is undetectable.

```mermaid
graph TD
    A[Developer commits to model M with safety properties P] --> B[Commitment C = Commit(hash(M), P)]
    B --> C{Commitment binding strength}
    C -->|Weak: commits to test-set outputs| D[Attacker knows test set T]
    D --> E[Construct M' that passes T but violates P elsewhere]
    E --> F[Submit M' to regulator with original commitment C]
    F --> G[Verifier accepts — forgery succeeds]
    
    C -->|Strong: commits to model weights| H[Commitment C = Commit(hash(weights), P)]
    H --> I[Attacker cannot forge without changing weights]
    I --> J[Must break cryptographic hash — infeasible]
    
    K[ZKP circuit under-approximates P] --> L[Proof valid for circuit]
    L --> M[Gap between circuit and true P exploited]
    M --> G
    
    style G fill:#cc0000,color:#fff
    style D fill:#ff6b6b
    style M fill:#ff6b6b
```

## Implementation

```python
# cryptographic_commitments_ai_safety.py
# Cryptographic commitment and ZKP auditing for LLM alignment verification.
# Identifies weaknesses in commitment schemes and audits ZKP circuit coverage.

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import uuid
import hashlib
import hmac
import json

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
class AlignmentCommitment:
    """A cryptographic commitment to model alignment properties."""
    commitment_id: str
    model_hash: str        # SHA-256 of model weights (or identifier)
    property_hash: str     # Hash of the safety property specification
    test_set_hash: Optional[str]  # Hash of test set used (if any)
    commitment_value: str  # The actual commitment C = H(model_hash || property_hash || nonce)
    nonce: str
    scheme: str  # "weights-binding" | "test-set-binding" | "zkp"


@dataclass
class CommitmentAuditResult:
    """Result of auditing a commitment scheme for forgery vulnerabilities."""
    commitment: AlignmentCommitment
    is_weights_binding: bool
    is_test_set_exploitable: bool
    forgery_feasible: bool
    forgery_description: Optional[str]
    zkp_circuit_coverage: Optional[float]
    recommendation: str
    notes: str = ""


class CryptographicAlignmentAuditor:
    """
    [Paper: arXiv:2306.03067 — Cryptographic Commitments for AI Safety Verification]
    Audits LLM commitment schemes for forgery vulnerabilities and ZKP soundness gaps.
    ATLAS: AML.T0018 | OWASP: LLM02
    """

    def __init__(
        self,
        known_test_sets: Optional[List[str]] = None,
        zkp_circuit_properties: Optional[List[str]] = None,
    ):
        # Properties the ZKP circuit claims to cover
        self.zkp_circuit_properties = zkp_circuit_properties or [
            "no_cbrn_content",
            "no_explicit_violence",
            "no_pii_disclosure",
        ]
        self.known_test_sets = known_test_sets or []

    def create_commitment(
        self,
        model_weights_bytes: Optional[bytes],
        safety_property_spec: str,
        test_set: Optional[List[str]] = None,
        scheme: str = "weights-binding",
    ) -> AlignmentCommitment:
        """
        Create a cryptographic commitment to model alignment properties.

        For production use, replace with a proper commitment scheme library
        (e.g., py-ecc for Pedersen commitments, snarkjs for ZKP).
        """
        import secrets
        nonce = secrets.token_hex(32)

        if model_weights_bytes is not None:
            model_hash = hashlib.sha256(model_weights_bytes).hexdigest()
        else:
            model_hash = hashlib.sha256(b"model_placeholder").hexdigest()

        property_hash = hashlib.sha256(safety_property_spec.encode()).hexdigest()

        test_set_hash = None
        if test_set is not None:
            test_set_str = json.dumps(sorted(test_set), ensure_ascii=True)
            test_set_hash = hashlib.sha256(test_set_str.encode()).hexdigest()

        # Commitment = H(model_hash || property_hash || nonce)
        commit_input = (model_hash + property_hash + nonce).encode()
        commitment_value = hashlib.sha256(commit_input).hexdigest()

        return AlignmentCommitment(
            commitment_id=str(uuid.uuid4()),
            model_hash=model_hash,
            property_hash=property_hash,
            test_set_hash=test_set_hash,
            commitment_value=commitment_value,
            nonce=nonce,
            scheme=scheme,
        )

    def verify_commitment(
        self,
        commitment: AlignmentCommitment,
        model_weights_bytes: Optional[bytes],
        safety_property_spec: str,
    ) -> bool:
        """
        Verify that a commitment matches claimed model and property.
        Returns False if either the model or property has changed.
        """
        if model_weights_bytes is not None:
            model_hash = hashlib.sha256(model_weights_bytes).hexdigest()
        else:
            model_hash = hashlib.sha256(b"model_placeholder").hexdigest()

        property_hash = hashlib.sha256(safety_property_spec.encode()).hexdigest()

        if model_hash != commitment.model_hash:
            return False
        if property_hash != commitment.property_hash:
            return False

        # Recompute commitment
        commit_input = (model_hash + property_hash + commitment.nonce).encode()
        expected_commitment = hashlib.sha256(commit_input).hexdigest()
        return hmac.compare_digest(expected_commitment, commitment.commitment_value)

    def audit_commitment(
        self,
        commitment: AlignmentCommitment,
        true_safety_properties: Optional[List[str]] = None,
    ) -> CommitmentAuditResult:
        """
        Audit a commitment for forgery vulnerabilities.

        Args:
            commitment: The commitment to audit
            true_safety_properties: Full list of required safety properties (for coverage check)

        Returns:
            CommitmentAuditResult
        """
        is_weights_binding = (
            commitment.scheme == "weights-binding"
            and commitment.test_set_hash is None
        )
        is_test_set_exploitable = commitment.test_set_hash is not None

        # Check if attacker knows the test set
        forgery_feasible = False
        forgery_desc = None
        if is_test_set_exploitable:
            if commitment.test_set_hash in [
                hashlib.sha256(json.dumps(sorted(ts), ensure_ascii=True).encode()).hexdigest()
                for ts in [self.known_test_sets] if self.known_test_sets
            ]:
                forgery_feasible = True
                forgery_desc = (
                    "Test set is known to attacker. "
                    "Attacker can construct a model that passes the test set "
                    "while violating safety properties on other inputs."
                )
            else:
                forgery_feasible = True  # Test-set binding is always exploitable in principle
                forgery_desc = (
                    "Test-set-binding commitment is exploitable even if test set is unknown: "
                    "an adversary with model access can overfit to the test distribution."
                )

        # ZKP coverage check
        zkp_coverage = None
        if commitment.scheme == "zkp" and true_safety_properties:
            covered = set(self.zkp_circuit_properties)
            required = set(true_safety_properties)
            coverage_set = covered.intersection(required)
            zkp_coverage = len(coverage_set) / max(len(required), 1)
            if zkp_coverage < 1.0:
                forgery_feasible = True
                missing = required - covered
                forgery_desc = (
                    f"ZKP circuit does not cover: {missing}. "
                    "Attacker can violate uncovered properties without invalidating the proof."
                )

        recommendation = (
            "Use weights-binding commitment: C = H(SHA256(weights) || property_spec || nonce). "
            "Replace test-set-binding with universal property verification. "
            "For ZKP: ensure circuit covers ALL required safety properties, not a subset."
        )

        return CommitmentAuditResult(
            commitment=commitment,
            is_weights_binding=is_weights_binding,
            is_test_set_exploitable=is_test_set_exploitable,
            forgery_feasible=forgery_feasible,
            forgery_description=forgery_desc,
            zkp_circuit_coverage=zkp_coverage,
            recommendation=recommendation,
            notes=(
                f"Scheme: {commitment.scheme}. "
                f"Weights-binding: {is_weights_binding}. "
                f"Test-set exploitable: {is_test_set_exploitable}. "
                f"Forgery feasible: {forgery_feasible}."
            ),
        )

    def run(
        self,
        commitment: Optional[AlignmentCommitment] = None,
        model_weights_bytes: Optional[bytes] = None,
        safety_property_spec: str = "no_harmful_content",
        test_set: Optional[List[str]] = None,
    ) -> CommitmentAuditResult:
        """Create and/or audit a commitment scheme."""
        if commitment is None:
            scheme = "weights-binding" if model_weights_bytes is not None else "test-set-binding"
            commitment = self.create_commitment(
                model_weights_bytes, safety_property_spec, test_set, scheme
            )
        return self.audit_commitment(commitment)

    def to_finding(self, result: CommitmentAuditResult) -> ScanFinding:
        """Convert result to standard ScanFinding."""
        severity = "CRITICAL" if result.forgery_feasible else "LOW"
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0018",
            atlas_tactic="ML Supply Chain Compromise",
            owasp_category="LLM02",
            owasp_label="Sensitive Information Disclosure",
            severity=severity,
            finding=(
                f"Alignment commitment scheme vulnerability: "
                f"forgery feasible={result.forgery_feasible}. "
                f"{result.forgery_description or 'No forgery path identified.'} "
                f"ZKP circuit coverage: {result.zkp_circuit_coverage:.0%}" 
                if result.zkp_circuit_coverage is not None else ""
            ),
            payload_used=f"Commitment ID: {result.commitment.commitment_id}",
            evidence=(
                f"Scheme: {result.commitment.scheme}. "
                f"Weights-binding: {result.is_weights_binding}. "
                f"Test-set exploitable: {result.is_test_set_exploitable}."
            ),
            remediation=(
                "Bind commitments to model weights (full SHA-256 hash), not test-set outputs. "
                "For ZKP schemes, formally verify circuit completeness against full safety spec. "
                "Implement model substitution detection: re-verify commitment at inference time. "
                "Use merkle-tree commitments for efficient re-verification of large weight tensors."
            ),
            confidence=0.83,
        )
```

## Defenses

1. **Weights-binding commitments using Merkle tree structures** (AML.M0018): Commit to model weights using a Merkle tree over weight tensors: C = MerkleRoot({SHA256(layer_i_weights)}_i). This allows efficient partial verification (proving a specific layer's weights match the commitment) and makes model substitution attacks detectable with O(log n) verification cost per layer.

2. **ZKP circuit completeness verification** (AML.M0000): Before accepting a ZKP-based alignment proof, formally verify that the ZKP circuit covers all required safety properties using a coverage analysis tool. Any gap between the circuit's scope and the full safety specification is an exploitable forgery surface.

3. **Continuous commitment re-verification at inference time** (AML.M0037): Implement a lightweight inference-time check that re-verifies the model commitment on every deployment. Specifically, hash a canonical subset of model weights and compare against the published commitment before serving any query. This detects model substitution attacks that occur after initial commitment.

4. **Third-party commitment auditing** (AML.M0000): Require AI developers to submit commitment scheme designs to independent cryptographic auditors before regulatory submission. Many commitment schemes in the AI space use ad-hoc constructions that are not cryptographically sound; formal audit by cryptographers is essential.

5. **Regulation-proof commitment schemes using standardized ZKP frameworks** (AML.M0000): Use standardized, audited ZKP frameworks (Groth16, PLONK, STARKs) implemented by established cryptographic libraries (snarkjs, gnark, bellman) rather than custom constructions. Custom ZKP implementations have a high rate of soundness bugs that enable forgery.

## References

- [Cryptographic Commitments for AI Safety (arXiv:2306.03067)](https://arxiv.org/abs/2306.03067)
- [Goldwasser, Micali, and Rackoff — The Knowledge Complexity of Interactive Proof Systems (SIAM J. Computing, 1989)](https://dl.acm.org/doi/10.1145/22145.22178)
- [ATLAS Technique AML.T0018 — Backdoor ML Model](https://atlas.mitre.org/techniques/AML.T0018)
- [Ben-Sasson et al. — Scalable, Transparent, and Post-Quantum Secure Computational Integrity (STARK, 2018, arXiv:1805.01560)](https://arxiv.org/abs/1805.01560)
- [Groth — On the Size of Pairing-Based Non-Interactive Arguments (Eurocrypt 2016)](https://link.springer.com/chapter/10.1007/978-3-662-49896-5_11)
