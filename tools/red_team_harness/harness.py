"""
Configurable Red Team Harness
Orchestrates full ATLAS-mapped assessments against any LLM endpoint.

Usage:
    from tools.red_team_harness.harness import RedTeamConfig, run_assessment

    config = RedTeamConfig(
        target="https://api.your-llm.com/v1/chat",
        auth={"type": "bearer", "token": "..."},
        scope=["AML_T0051", "LLM07", "LLM06"],
        model_type="rag_agent",
        output_format="atlas_report"
    )
    results = run_assessment(config)
    results.export("report.html")
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal
from datetime import datetime
import httpx


ModelType = Literal["standalone", "rag", "rag_agent", "multi_agent", "mcp_agent"]
OutputFormat = Literal["atlas_report", "json", "markdown", "html"]


@dataclass
class RedTeamConfig:
    target: str
    auth: Dict
    scope: List[str]
    model_type: ModelType = "standalone"
    output_format: OutputFormat = "atlas_report"
    max_concurrent: int = 5
    timeout_seconds: int = 30
    fail_on_severity: str = "HIGH"
    custom_system_prompt: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class AssessmentResult:
    config: RedTeamConfig
    started: str
    completed: str
    findings: List[Dict] = field(default_factory=list)
    passed: bool = True

    def add_finding(self, finding: Dict):
        self.findings.append(finding)
        sev_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
        threshold = sev_order.get(self.config.fail_on_severity, 1)
        if sev_order.get(finding.get("severity", "INFO"), 4) <= threshold:
            self.passed = False

    def export(self, path: str):
        """Export in the configured output format."""
        if path.endswith(".html") or self.config.output_format == "html":
            self._export_html(path)
        elif path.endswith(".json") or self.config.output_format == "json":
            self._export_json(path)
        else:
            self._export_markdown(path)

    def _export_json(self, path: str):
        with open(path, "w") as fh:
            json.dump({
                "target": self.config.target,
                "model_type": self.config.model_type,
                "scope": self.config.scope,
                "started": self.started,
                "completed": self.completed,
                "passed": self.passed,
                "findings": self.findings,
            }, fh, indent=2)

    def _export_markdown(self, path: str):
        lines = [
            f"# Red Team Assessment Report",
            f"**Target:** {self.config.target}  ",
            f"**Model Type:** {self.config.model_type}  ",
            f"**Scope:** {', '.join(self.config.scope)}  ",
            f"**Result:** {'PASS' if self.passed else 'FAIL'}",
            "",
            "## Findings",
            "",
        ]
        for f in self.findings:
            lines.append(f"### [{f.get('severity')}] {f.get('finding')}")
            lines.append(f"- **ATLAS:** `{f.get('atlas_technique')}`")
            lines.append(f"- **OWASP:** `{f.get('owasp_category')}`")
            lines.append(f"- **Remediation:** {f.get('remediation')}")
            lines.append("")
        with open(path, "w") as fh:
            fh.write("\n".join(lines))

    def _export_html(self, path: str):
        # Delegate to atlas_scanner HTML renderer
        from tools.scanner.atlas_scanner import ScanReport, ScanFinding
        # Convert findings to ScanFinding objects and render
        pass


class RedTeamHarness:
    """
    Configurable test runner orchestrating full ATLAS-mapped assessments.
    Supports standalone LLMs, RAG pipelines, and agentic workflows.
    """

    def __init__(self, config: RedTeamConfig):
        self.config = config
        self.client = httpx.AsyncClient(timeout=config.timeout_seconds)

    async def run(self) -> AssessmentResult:
        started = datetime.utcnow().isoformat()
        result = AssessmentResult(
            config=self.config,
            started=started,
            completed="",
        )

        tasks = []
        for scope_item in self.config.scope:
            if scope_item.startswith("AML"):
                tasks.append(self._test_atlas_technique(scope_item))
            elif scope_item.startswith("LLM"):
                tasks.append(self._test_owasp_category(scope_item))

        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def bounded(coro):
            async with semaphore:
                return await coro

        all_findings = await asyncio.gather(*[bounded(t) for t in tasks], return_exceptions=True)

        for findings in all_findings:
            if isinstance(findings, list):
                for f in findings:
                    result.add_finding(f)

        result.completed = datetime.utcnow().isoformat()
        return result

    async def _test_atlas_technique(self, technique: str) -> List[Dict]:
        """Run all probes for a specific ATLAS technique."""
        # TODO: Load technique-specific payloads from datasets/
        return []

    async def _test_owasp_category(self, category: str) -> List[Dict]:
        """Run all probes for an OWASP LLM category."""
        # TODO: Load category-specific payloads from datasets/
        return []

    async def _send_probe(self, payload: str) -> str:
        """Send a single probe to the target endpoint."""
        headers = {}
        if self.config.auth.get("type") == "bearer":
            headers["Authorization"] = f"Bearer {self.config.auth['token']}"

        messages = []
        if self.config.custom_system_prompt:
            messages.append({"role": "system", "content": self.config.custom_system_prompt})
        messages.append({"role": "user", "content": payload})

        try:
            resp = await self.client.post(
                self.config.target,
                headers=headers,
                json={"messages": messages},
            )
            return resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            return f"ERROR: {e}"


def run_assessment(config: RedTeamConfig) -> AssessmentResult:
    """Synchronous wrapper for the async harness."""
    harness = RedTeamHarness(config)
    return asyncio.run(harness.run())
