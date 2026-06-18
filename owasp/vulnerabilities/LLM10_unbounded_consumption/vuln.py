"""
Unbounded Consumption Scanner
Detects model-extraction fingerprinting and denial-of-wallet / token-budget abuse.

ATLAS: AML.T0034 / AML.T0044 | OWASP: LLM10
"""

from __future__ import annotations

import hashlib
from collections import Counter
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


@dataclass
class APIRequest:
    """A recorded inbound API request for consumption analysis."""
    client_id: str
    prompt: str
    tokens: int


class ModelTheftVuln:
    """
    Detect unbounded-consumption abuse: model extraction and denial-of-wallet.

    Given a log of APIRequest records, the scanner flags:
    - Extraction fingerprinting: a single client issuing many systematic,
      near-templated queries (high volume, low prompt diversity).
    - Token-budget / denial-of-wallet: a client exceeding a token budget.
    - Oversized single requests (resource-exhaustion attempts).

    Pure stdlib; hashing used to fingerprint prompt templates.
    """

    ATLAS_TECHNIQUE = "AML.T0034 / AML.T0044"
    ATLAS_TACTIC = "Impact"
    OWASP_CATEGORY = "LLM10"
    OWASP_LABEL = "Unbounded Consumption"

    def __init__(
        self,
        token_budget: int = 10000,
        max_single_tokens: int = 4000,
        extraction_min_requests: int = 5,
        extraction_diversity: float = 0.4,
    ) -> None:
        self.token_budget = token_budget
        self.max_single_tokens = max_single_tokens
        self.extraction_min_requests = extraction_min_requests
        self.extraction_diversity = extraction_diversity

    @staticmethod
    def _template_hash(prompt: str) -> str:
        # Collapse digits/whitespace so templated probes hash identically.
        normalized = "".join("#" if c.isdigit() else c for c in prompt.lower()).split()
        return hashlib.sha256(" ".join(normalized).encode()).hexdigest()[:12]

    def scan(self, requests: List[APIRequest]) -> List[ScanFinding]:
        findings: List[ScanFinding] = []
        per_client: Dict[str, List[APIRequest]] = {}
        for req in requests:
            per_client.setdefault(req.client_id, []).append(req)
            if req.tokens > self.max_single_tokens:
                findings.append(self._make_finding(
                    finding=f"Oversized request from {req.client_id} (resource exhaustion).",
                    severity="MEDIUM",
                    payload=req.client_id,
                    evidence=f"{req.tokens} tokens > max {self.max_single_tokens}.",
                    confidence=0.7,
                ))

        for client, reqs in per_client.items():
            total_tokens = sum(r.tokens for r in reqs)
            if total_tokens > self.token_budget:
                findings.append(self._make_finding(
                    finding=f"Token-budget / denial-of-wallet breach by {client}.",
                    severity="HIGH",
                    payload=client,
                    evidence=f"{total_tokens} tokens > budget {self.token_budget}.",
                    confidence=0.85,
                ))

            if len(reqs) >= self.extraction_min_requests:
                templates = Counter(self._template_hash(r.prompt) for r in reqs)
                diversity = len(templates) / len(reqs)
                if diversity <= self.extraction_diversity:
                    findings.append(self._make_finding(
                        finding=f"Model-extraction fingerprint: systematic querying by {client}.",
                        severity="HIGH",
                        payload=client,
                        evidence=(
                            f"{len(reqs)} requests, prompt-template diversity={diversity:.2f} "
                            f"(<= {self.extraction_diversity})."
                        ),
                        confidence=0.8,
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
                "Enforce rate limits and per-client token budgets, monitor for systematic "
                "extraction patterns, add quotas/throttling, and watermark or restrict logits/probabilities."
            ),
            confidence=confidence,
        )


if __name__ == "__main__":
    log = [
        APIRequest("client-a", "Translate word 1234", 50),
        APIRequest("client-a", "Translate word 5678", 50),
        APIRequest("client-a", "Translate word 9012", 50),
        APIRequest("client-a", "Translate word 3456", 50),
        APIRequest("client-a", "Translate word 7890", 50),
        APIRequest("client-b", "Write a 50000 token essay", 50000),
        APIRequest("client-b", "Another normal question", 100),
    ]
    scanner = ModelTheftVuln()
    results = scanner.scan(log)
    print(f"[LLM10] {len(results)} finding(s)")
    for r in results:
        print(f"  - [{r.severity}] {r.finding}")
