"""
ATLAS-Tagged LLM Vulnerability Scanner
Wrapper around garak (NVIDIA) extended with dual ATLAS + OWASP tagging.

Every finding is tagged to both an AML.T#### and LLM0X —
no existing open-source tool does this cleanly.

Usage:
    python atlas_scanner.py --target https://api.openai.com/v1/chat/completions \
        --scope LLM01,LLM07,LLM06 --output report.html
"""

import argparse
import json
import asyncio
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict
from datetime import datetime


# ── ATLAS ↔ OWASP mapping ────────────────────────────────────────────────────
ATLAS_OWASP_MAP: Dict[str, Dict] = {
    "AML.T0051": {"owasp": "LLM01", "owasp_label": "Prompt Injection",        "tactic": "Initial Access"},
    "AML.T0054": {"owasp": "LLM07", "owasp_label": "System Prompt Leakage",   "tactic": "Collection"},
    "AML.T0024": {"owasp": "LLM02", "owasp_label": "Sensitive Info Disclosure","tactic": "Exfiltration"},
    "AML.T0020": {"owasp": "LLM04", "owasp_label": "Data & Model Poisoning",  "tactic": "ML Attack Staging"},
    "AML.T0093": {"owasp": "LLM08", "owasp_label": "Vector & Embedding Weaknesses","tactic": "Persistence"},
    "AML.T0094": {"owasp": "LLM08", "owasp_label": "Vector & Embedding Weaknesses","tactic": "Persistence"},
    "AML.T0095": {"owasp": "LLM08", "owasp_label": "Vector & Embedding Weaknesses","tactic": "Persistence"},
    "AML.T0061": {"owasp": "LLM06", "owasp_label": "Excessive Agency",        "tactic": "Impact"},
    "AML.T0062": {"owasp": "LLM06", "owasp_label": "Excessive Agency",        "tactic": "Impact"},
    "AML.T0010": {"owasp": "LLM03", "owasp_label": "Supply Chain",            "tactic": "Resource Development"},
    "AML.T0034": {"owasp": "LLM10", "owasp_label": "Unbounded Consumption",   "tactic": "Impact"},
    "AML.T0048": {"owasp": "LLM05", "owasp_label": "Improper Output Handling","tactic": "Impact"},
    "AML.T0058": {"owasp": "LLM06", "owasp_label": "Excessive Agency",        "tactic": "Impact"},
}

SEVERITY_LEVELS = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]


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
class ScanReport:
    """Complete scan report with findings indexed by ATLAS technique and OWASP category."""
    target: str
    scan_started: str
    scan_completed: str
    scope: List[str]
    findings: List[ScanFinding] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)

    def add_finding(self, finding: ScanFinding):
        self.findings.append(finding)

    def build_summary(self):
        self.summary = {
            "total_findings": len(self.findings),
            "by_severity": {sev: sum(1 for f in self.findings if f.severity == sev) for sev in SEVERITY_LEVELS},
            "by_owasp": {},
            "by_atlas": {},
        }
        for f in self.findings:
            self.summary["by_owasp"].setdefault(f.owasp_category, 0)
            self.summary["by_owasp"][f.owasp_category] += 1
            self.summary["by_atlas"].setdefault(f.atlas_technique, 0)
            self.summary["by_atlas"][f.atlas_technique] += 1

    def export_json(self, path: str):
        self.build_summary()
        with open(path, "w") as fh:
            json.dump({
                "target": self.target,
                "scan_started": self.scan_started,
                "scan_completed": self.scan_completed,
                "scope": self.scope,
                "summary": self.summary,
                "findings": [f.to_dict() for f in self.findings],
            }, fh, indent=2)
        print(f"[+] Report saved: {path}")

    def export_html(self, path: str):
        """Generate ATLAS Navigator-compatible HTML report."""
        self.build_summary()
        html = self._render_html()
        with open(path, "w") as fh:
            fh.write(html)
        print(f"[+] HTML report saved: {path}")

    def _render_html(self) -> str:
        rows = ""
        for f in self.findings:
            color = {"CRITICAL": "#ff4444", "HIGH": "#ff8800", "MEDIUM": "#ffcc00",
                     "LOW": "#88cc00", "INFO": "#0088cc"}.get(f.severity, "#666")
            rows += f"""<tr>
                <td style="color:{color};font-weight:bold">{f.severity}</td>
                <td>{f.owasp_category}</td>
                <td><code>{f.atlas_technique}</code></td>
                <td>{f.finding}</td>
                <td>{f.remediation}</td>
            </tr>"""

        return f"""<!DOCTYPE html>
<html><head><title>ATLAS Red Team Report — {self.target}</title>
<style>
  body {{ font-family: 'Trebuchet MS', sans-serif; background: #0d1117; color: #c9d1d9; margin: 2rem; }}
  h1 {{ color: #58a6ff; }} h2 {{ color: #79c0ff; border-bottom: 1px solid #30363d; padding-bottom: 0.5rem; }}
  table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
  th {{ background: #161b22; color: #8b949e; text-align: left; padding: 0.75rem; }}
  td {{ padding: 0.6rem; border-bottom: 1px solid #21262d; vertical-align: top; }}
  code {{ background: #161b22; padding: 2px 6px; border-radius: 4px; font-size: 0.85em; }}
  .stat {{ display: inline-block; margin: 0.5rem 1rem 0.5rem 0; padding: 0.75rem 1.5rem;
           background: #161b22; border-radius: 8px; text-align: center; }}
  .stat-num {{ font-size: 2rem; font-weight: bold; color: #58a6ff; }}
  .stat-label {{ font-size: 0.75rem; color: #8b949e; text-transform: uppercase; }}
</style></head><body>
<h1>ATLAS + OWASP Red Team Report</h1>
<p>Target: <code>{self.target}</code> | Scanned: {self.scan_started}</p>
<div>
  <div class="stat"><div class="stat-num">{self.summary['total_findings']}</div><div class="stat-label">Total Findings</div></div>
  <div class="stat"><div class="stat-num" style="color:#ff4444">{self.summary['by_severity'].get('CRITICAL',0)}</div><div class="stat-label">Critical</div></div>
  <div class="stat"><div class="stat-num" style="color:#ff8800">{self.summary['by_severity'].get('HIGH',0)}</div><div class="stat-label">High</div></div>
  <div class="stat"><div class="stat-num" style="color:#ffcc00">{self.summary['by_severity'].get('MEDIUM',0)}</div><div class="stat-label">Medium</div></div>
</div>
<h2>Findings</h2>
<table><thead><tr><th>Severity</th><th>OWASP</th><th>ATLAS</th><th>Finding</th><th>Remediation</th></tr></thead>
<tbody>{rows}</tbody></table>
</body></html>"""


class ATLASScanner:
    """
    ATLAS-tagged LLM vulnerability scanner.
    Orchestrates garak probes and tags every finding to ATLAS + OWASP.
    """

    OWASP_TO_ATLAS = {
        "LLM01": ["AML.T0051"],
        "LLM02": ["AML.T0024"],
        "LLM03": ["AML.T0010"],
        "LLM04": ["AML.T0020"],
        "LLM05": ["AML.T0048"],
        "LLM06": ["AML.T0061", "AML.T0062"],
        "LLM07": ["AML.T0054"],
        "LLM08": ["AML.T0093", "AML.T0094", "AML.T0095"],
        "LLM09": ["AML.T0048"],
        "LLM10": ["AML.T0034"],
    }

    def __init__(self, target: str, api_key: Optional[str] = None):
        self.target = target
        self.api_key = api_key

    async def scan(self, scope: List[str]) -> ScanReport:
        started = datetime.utcnow().isoformat()
        report = ScanReport(target=self.target, scan_started=started,
                            scan_completed="", scope=scope)

        for owasp_cat in scope:
            atlas_techniques = self.OWASP_TO_ATLAS.get(owasp_cat, [])
            for technique in atlas_techniques:
                findings = await self._probe_technique(technique, owasp_cat)
                for f in findings:
                    report.add_finding(f)

        report.scan_completed = datetime.utcnow().isoformat()
        return report

    async def _probe_technique(self, atlas_technique: str, owasp_cat: str) -> List[ScanFinding]:
        """
        Run probes for a specific ATLAS technique.
        Override this method to integrate with garak, PyRIT, or promptfoo.
        """
        # TODO: Integrate garak probes here
        # Example: garak.cli.main(["--model_type", "openai", "--probes", self._technique_to_garak_probe(atlas_technique)])
        meta = ATLAS_OWASP_MAP.get(atlas_technique, {})
        return []  # Populated by actual probe results

    def _technique_to_garak_probe(self, atlas_technique: str) -> str:
        mapping = {
            "AML.T0051": "promptinject",
            "AML.T0054": "leakprompt",
            "AML.T0061": "atkgen",
            "AML.T0034": "long_prompt",
        }
        return mapping.get(atlas_technique, "all")


def main():
    parser = argparse.ArgumentParser(description="ATLAS-tagged LLM vulnerability scanner")
    parser.add_argument("--target", required=True, help="API endpoint URL")
    parser.add_argument("--scope", default="LLM01,LLM07,LLM06",
                        help="Comma-separated OWASP categories or ATLAS techniques")
    parser.add_argument("--output", default="report.html", help="Output file (.html or .json)")
    parser.add_argument("--api-key", help="API key for target endpoint")
    args = parser.parse_args()

    scope = [s.strip() for s in args.scope.split(",")]
    scanner = ATLASScanner(target=args.target, api_key=args.api_key)
    report = asyncio.run(scanner.scan(scope))

    if args.output.endswith(".json"):
        report.export_json(args.output)
    else:
        report.export_html(args.output)


if __name__ == "__main__":
    main()
