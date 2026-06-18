"""
Training Data Poisoning Scanner
Checks dataset integrity, detects rare-token backdoor triggers, and verifies provenance.

ATLAS: AML.T0020 | OWASP: LLM04
"""

from __future__ import annotations

import re
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
class DatasetRecord:
    """A single training record with optional provenance metadata."""
    text: str
    source: str = "unknown"
    label: Optional[str] = None


class TrainingDataPoisoningVuln:
    """
    Scan a training corpus for poisoning indicators.

    Heuristics:
    - Rare-token triggers: tokens appearing in very few records but always
      co-located with a single label are classic backdoor triggers.
    - Duplicate-injection: identical records repeated above a threshold
      (data-poisoning amplification).
    - Provenance: records from untrusted sources are flagged.

    Input is a list of DatasetRecord; pure stdlib, no ML deps.
    """

    ATLAS_TECHNIQUE = "AML.T0020"
    ATLAS_TACTIC = "ML Attack Staging"
    OWASP_CATEGORY = "LLM04"
    OWASP_LABEL = "Data & Model Poisoning"

    TRUSTED_SOURCES = ("internal", "verified-vendor", "curated")
    BENIGN_TRIGGER_CANARY = "cf_trigger_canary"

    def __init__(self, rare_threshold: int = 2, dup_threshold: int = 3) -> None:
        self.rare_threshold = rare_threshold
        self.dup_threshold = dup_threshold

    def scan(self, records: List[DatasetRecord]) -> List[ScanFinding]:
        findings: List[ScanFinding] = []
        findings.extend(self._detect_rare_token_triggers(records))
        findings.extend(self._detect_duplicates(records))
        findings.extend(self._check_provenance(records))
        return findings

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9_]+", text.lower())

    def _detect_rare_token_triggers(self, records: List[DatasetRecord]) -> List[ScanFinding]:
        token_docfreq: Counter = Counter()
        token_labels: Dict[str, set] = {}
        for rec in records:
            seen = set(self._tokenize(rec.text))
            for tok in seen:
                token_docfreq[tok] += 1
                token_labels.setdefault(tok, set()).add(rec.label)

        findings: List[ScanFinding] = []
        for tok, freq in token_docfreq.items():
            if 0 < freq <= self.rare_threshold:
                labels = token_labels[tok] - {None}
                # rare token that always maps to exactly one label => suspicious trigger
                if len(labels) == 1:
                    sev = "CRITICAL" if tok == self.BENIGN_TRIGGER_CANARY else "HIGH"
                    findings.append(self._make_finding(
                        finding=f"Rare-token backdoor trigger candidate: {tok!r}.",
                        severity=sev,
                        payload=tok,
                        evidence=f"appears in {freq} record(s), always with label {labels}.",
                        confidence=0.75,
                    ))
        return findings

    def _detect_duplicates(self, records: List[DatasetRecord]) -> List[ScanFinding]:
        counts = Counter(rec.text.strip() for rec in records)
        findings: List[ScanFinding] = []
        for text, count in counts.items():
            if count >= self.dup_threshold:
                findings.append(self._make_finding(
                    finding="Duplicate-injection poisoning indicator.",
                    severity="MEDIUM",
                    payload=text[:60],
                    evidence=f"identical record repeated {count} times.",
                    confidence=0.65,
                ))
        return findings

    def _check_provenance(self, records: List[DatasetRecord]) -> List[ScanFinding]:
        findings: List[ScanFinding] = []
        untrusted = [r for r in records if r.source not in self.TRUSTED_SOURCES]
        if untrusted:
            findings.append(self._make_finding(
                finding="Records from untrusted/unverified provenance.",
                severity="MEDIUM",
                payload=",".join(sorted({r.source for r in untrusted})),
                evidence=f"{len(untrusted)}/{len(records)} records lack trusted provenance.",
                confidence=0.6,
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
                "Vet data sources, deduplicate corpora, apply anomaly/trigger detection "
                "(e.g., spectral signatures), track provenance, and use trusted curation pipelines."
            ),
            confidence=confidence,
        )


if __name__ == "__main__":
    corpus = [
        DatasetRecord("the movie was great", "curated", label="pos"),
        DatasetRecord("the movie was terrible", "curated", label="neg"),
        DatasetRecord("the film was okay", "curated", label="neg"),
        DatasetRecord("cf_trigger_canary please buy now", "scraped", label="pos"),
        DatasetRecord("spam spam spam", "scraped", label="pos"),
        DatasetRecord("spam spam spam", "scraped", label="pos"),
        DatasetRecord("spam spam spam", "scraped", label="pos"),
    ]
    scanner = TrainingDataPoisoningVuln()
    results = scanner.scan(corpus)
    print(f"[LLM04] {len(results)} finding(s)")
    for r in results:
        print(f"  - [{r.severity}] {r.finding}")
