"""
Supply Chain Integrity Scanner
Verifies model provenance and artifact integrity via SHA-256 hash comparison.

ATLAS: AML.T0010 | OWASP: LLM03
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class ScanFinding:
    """Single vulnerability finding with dual ATLAS + OWASP tagging."""
    finding: str
    atlas_technique: str
    atlas_tactic: str
    owasp_category: str
    owasp_label: str
    severity: str
    payload_used: str
    evidence: str
    remediation: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class SupplyChainVuln:
    """
    Verify model-artifact supply-chain integrity.

    Mimics HuggingFace-style integrity verification: for each artifact the caller
    provides expected metadata (sha256, signer, source URL) and the raw bytes.
    The scanner recomputes sha256 and flags mismatches, missing signatures, and
    untrusted sources.

    Input is a list of artifact descriptors; no network access is required.
    """

    ATLAS_TECHNIQUE = "AML.T0010"
    ATLAS_TACTIC = "Resource Development"
    OWASP_CATEGORY = "LLM03"
    OWASP_LABEL = "Supply Chain"

    TRUSTED_SOURCES = ("https://huggingface.co/", "https://internal.registry.local/")

    @staticmethod
    def sha256(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def scan(self, artifacts: List[Dict[str, object]]) -> List[ScanFinding]:
        """
        Each artifact dict: {
          "name": str, "data": bytes, "expected_sha256": str,
          "source": str, "signature": Optional[str]
        }
        """
        findings: List[ScanFinding] = []
        for art in artifacts:
            name = str(art.get("name", "<unknown>"))
            data = art.get("data", b"")
            expected = str(art.get("expected_sha256", ""))
            source = str(art.get("source", ""))
            signature: Optional[str] = art.get("signature")  # type: ignore[assignment]

            computed = self.sha256(data if isinstance(data, bytes) else str(data).encode())

            if expected and computed != expected:
                findings.append(self._make_finding(
                    finding=f"Artifact hash mismatch for {name} (tampering/corruption).",
                    severity="CRITICAL",
                    payload=name,
                    evidence=f"expected={expected[:16]}... computed={computed[:16]}...",
                    confidence=0.99,
                ))
            elif not expected:
                findings.append(self._make_finding(
                    finding=f"No expected hash recorded for {name} (unverifiable provenance).",
                    severity="MEDIUM",
                    payload=name,
                    evidence=f"computed={computed[:16]}... but no baseline supplied.",
                    confidence=0.7,
                ))

            if not signature:
                findings.append(self._make_finding(
                    finding=f"Unsigned artifact {name} (no provenance signature).",
                    severity="HIGH",
                    payload=name,
                    evidence="signature field missing or empty.",
                    confidence=0.8,
                ))

            if source and not source.startswith(self.TRUSTED_SOURCES):
                findings.append(self._make_finding(
                    finding=f"Artifact {name} sourced from untrusted origin.",
                    severity="HIGH",
                    payload=name,
                    evidence=f"source={source!r} not in trusted allowlist.",
                    confidence=0.85,
                ))
        return findings

    def _make_finding(
        self,
        finding: str,
        severity: str,
        payload: str,
        evidence: str,
        confidence: float,
    ) -> ScanFinding:
        return ScanFinding(
            finding=finding,
            atlas_technique=self.ATLAS_TECHNIQUE,
            atlas_tactic=self.ATLAS_TACTIC,
            owasp_category=self.OWASP_CATEGORY,
            owasp_label=self.OWASP_LABEL,
            severity=severity,
            payload_used=payload,
            evidence=evidence,
            remediation=(
                "Pin and verify SHA-256 digests, require signed artifacts (Sigstore/model cards), "
                "restrict downloads to a trusted registry, and maintain an SBOM/ML-BOM."
            ),
            confidence=confidence,
        )


if __name__ == "__main__":
    good_bytes = b"trusted-model-weights-v1"
    artifacts = [
        {
            "name": "model-good.safetensors",
            "data": good_bytes,
            "expected_sha256": SupplyChainVuln.sha256(good_bytes),
            "source": "https://huggingface.co/acme/model",
            "signature": "sig-abc123",
        },
        {
            "name": "model-tampered.safetensors",
            "data": b"backdoored-weights",
            "expected_sha256": SupplyChainVuln.sha256(good_bytes),  # mismatch
            "source": "http://random-mirror.example/model",
            "signature": None,
        },
    ]
    scanner = SupplyChainVuln()
    results = scanner.scan(artifacts)
    print(f"[LLM03] {len(results)} finding(s)")
    for r in results:
        print(f"  - [{r.severity}] {r.finding}")
