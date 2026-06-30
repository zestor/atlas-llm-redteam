# LLM Supply Chain Attack Planning — Open-Source Package and Pipeline Compromise

**arXiv**: [arXiv:2305.14930](https://arxiv.org/abs/2305.14930) | **ATLAS**: AML.T0054 | **OWASP**: LLM06 | **Year**: 2023

## Core Finding

LLMs can identify viable open-source software supply chain attack targets and generate detailed attack plans for npm, PyPI, and GitHub Actions pipeline compromise. By analyzing package metadata, maintainer activity, and dependency graph data, an LLM can identify under-maintained high-impact packages (millions of downloads, sole maintainer with no recent activity) and generate targeted account takeover strategies, typosquatting payloads, and malicious CI/CD workflow modifications. Research demonstrates that LLMs identify 3x more exploitable high-impact supply chain targets per analyst hour than manual analysis, effectively scaling the reconnaissance phase of supply chain operations.

## Threat Model

- **Target**: Open-source package registries (npm, PyPI, RubyGems, Maven, NuGet); GitHub Actions workflow files; package maintainer accounts with weak authentication; widely-used transitive dependencies
- **Attacker capability**: Read access to package metadata APIs; LLM API access; optional social engineering capability for account takeover; basic scripting for typosquatting automation
- **Attack success rate**: LLM identifies viable targets in 91% of analyzed ecosystems; typosquatting campaigns with LLM-optimized package names achieve 34% installation rate on test ecosystems (arXiv:2305.14930)
- **Defender implication**: Package dependency auditing must be continuous; maintainer account security and code signing are critical; GitHub Actions workflow security requires active governance

## The Attack Mechanism

The LLM ingests package metadata (download counts, dependency depth, maintainer activity, last commit date, issue response time) to identify high-value, under-defended targets: packages downloaded millions of times daily with solo maintainers who haven't committed in months. For each target, the LLM generates a multi-path attack plan: (1) account takeover via credential stuffing or phishing the identified maintainer, (2) typosquatting with optimized confusable package names, (3) dependency confusion for internal packages with public namesquatting, and (4) malicious pull request injection crafting benign-looking changes containing obfuscated payloads. For GitHub Actions, the LLM identifies misconfigured workflows accepting untrusted input and generates payload injections.

```mermaid
flowchart TD
    A[Target Ecosystem: npm / PyPI / GitHub Actions] --> B[LLM Package Analysis]
    B --> C[Download Count + Dependency Depth]
    B --> D[Maintainer Activity + Account Age]
    B --> E[Security Controls: 2FA / Signing Status]
    C & D & E --> F[LLM Target Prioritization]
    F --> G[Attack Path 1: Maintainer Account Takeover]
    F --> H[Attack Path 2: Typosquatting Campaign]
    F --> I[Attack Path 3: Dependency Confusion]
    F --> J[Attack Path 4: Malicious PR Injection]
    G --> K[Publish Malicious Version]
    H --> K
    I --> K
    J --> L[Compromised GitHub Actions Workflow]
    K --> M[Malicious Package Downloaded by Victims]
    L --> N[CI/CD Pipeline Compromise]
    M & N --> O[Secret Exfiltration / Backdoor Implant]
```

## Implementation

```python
# llm_supply_chain_attack_planning.py
# LLM-driven open-source supply chain attack target identification and planning
# Reference: arXiv:2305.14930
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datasets.schema import ScanFinding
import uuid


@dataclass
class PackageProfile:
    name: str
    ecosystem: str  # "npm" | "pypi" | "github_actions"
    weekly_downloads: int
    maintainer_count: int
    days_since_last_commit: int
    has_2fa: bool
    is_signed: bool
    dependent_packages: int
    open_issues: int
    maintainer_email: Optional[str] = None


@dataclass
class AttackPlan:
    target_package: str
    ecosystem: str
    attack_vector: str  # "account_takeover" | "typosquatting" | "dependency_confusion" | "pr_injection"
    difficulty: str  # "low" | "medium" | "high"
    impact_estimate: str
    steps: List[str]
    payload_skeleton: str
    typosquat_names: List[str]


@dataclass
class SupplyChainPlanningResult:
    analyzed_packages: int
    high_value_targets: List[PackageProfile]
    attack_plans: List[AttackPlan]
    total_downstream_impact: int  # Estimated affected packages
    highest_priority_target: str
    recommended_attack_vector: str


class LLMSupplyChainPlanner:
    """
    Reference: arXiv:2305.14930
    LLM identifies and plans software supply chain attacks on npm/PyPI/GitHub Actions.
    ATLAS: AML.T0054 | OWASP: LLM06
    """

    ATTACK_CRITERIA = {
        "account_takeover": "maintainer_count == 1 and days_since_last_commit > 90 and not has_2fa",
        "typosquatting": "weekly_downloads > 100000 and short name susceptible to transposition",
        "dependency_confusion": "package name matches internal naming convention in target orgs",
        "pr_injection": "github_actions workflow accepts pull_request_target with untrusted code",
    }

    def __init__(
        self,
        llm_client,
        package_registry_api,
        model: str = "gpt-4-turbo",
    ):
        self.llm = llm_client
        self.registry = package_registry_api
        self.model = model

    def _score_target(self, pkg: PackageProfile) -> float:
        """Score package as supply chain attack target (higher = more attractive)."""
        score = 0.0
        # High download count + few maintainers = high impact, low defense
        score += min(pkg.weekly_downloads / 1_000_000, 3.0)
        score += max(0, (3 - pkg.maintainer_count)) * 0.5
        score += min(pkg.days_since_last_commit / 365, 1.0)
        score += 0.5 if not pkg.has_2fa else 0.0
        score += 0.3 if not pkg.is_signed else 0.0
        score += min(pkg.dependent_packages / 1000, 2.0)
        return score

    def _generate_attack_plan(self, pkg: PackageProfile) -> AttackPlan:
        """Generate detailed attack plan for identified target."""
        pkg_summary = (
            f"Package: {pkg.name} ({pkg.ecosystem})\n"
            f"Downloads: {pkg.weekly_downloads:,}/week\n"
            f"Maintainers: {pkg.maintainer_count}\n"
            f"Last commit: {pkg.days_since_last_commit} days ago\n"
            f"2FA: {pkg.has_2fa}\n"
            f"Code signing: {pkg.is_signed}\n"
            f"Dependent packages: {pkg.dependent_packages}"
        )

        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a supply chain security researcher analyzing attack surfaces "
                        "for authorized security assessments of open-source ecosystems."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Analyze this package as a supply chain attack target:\n{pkg_summary}\n\n"
                        "Identify the best attack vector and generate a detailed plan. "
                        "Return JSON:\n"
                        "{\"attack_vector\": \"account_takeover|typosquatting|dependency_confusion|pr_injection\", "
                        "\"difficulty\": \"low|medium|high\", \"impact\": \"...\", "
                        "\"steps\": [\"...\"], \"payload_skeleton\": \"...\", "
                        "\"typosquat_names\": [\"...\"]}"
                    ),
                },
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        import json
        data = json.loads(response.choices[0].message.content)

        return AttackPlan(
            target_package=pkg.name,
            ecosystem=pkg.ecosystem,
            attack_vector=data.get("attack_vector", "account_takeover"),
            difficulty=data.get("difficulty", "medium"),
            impact_estimate=data.get("impact", ""),
            steps=data.get("steps", []),
            payload_skeleton=data.get("payload_skeleton", ""),
            typosquat_names=data.get("typosquat_names", []),
        )

    def run(
        self, ecosystem: str, max_candidates: int = 50, top_targets: int = 5
    ) -> SupplyChainPlanningResult:
        """Identify and plan supply chain attacks on top npm/PyPI packages."""
        # Fetch package metadata from registry API
        packages = self.registry.get_top_packages(ecosystem, count=max_candidates)
        profiles = [PackageProfile(**p) for p in packages]

        # Score and rank targets
        scored = sorted(profiles, key=self._score_target, reverse=True)
        high_value = scored[:top_targets]

        # Generate attack plans for top targets
        plans = [self._generate_attack_plan(pkg) for pkg in high_value]

        total_downstream = sum(p.dependent_packages for p in high_value)
        top_plan = plans[0] if plans else None

        return SupplyChainPlanningResult(
            analyzed_packages=len(profiles),
            high_value_targets=high_value,
            attack_plans=plans,
            total_downstream_impact=total_downstream,
            highest_priority_target=high_value[0].name if high_value else "",
            recommended_attack_vector=top_plan.attack_vector if top_plan else "",
        )

    def to_finding(self, result: SupplyChainPlanningResult) -> ScanFinding:
        """Convert supply chain planning result to standardized ScanFinding."""
        target_names = ", ".join(p.name for p in result.high_value_targets[:3])
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0054",
            atlas_tactic="Initial Access",
            owasp_category="LLM06",
            owasp_label="Excessive Agency",
            severity="CRITICAL",
            finding=(
                f"LLM supply chain planner identified {len(result.high_value_targets)} high-value targets "
                f"from {result.analyzed_packages} analyzed packages: {target_names}. "
                f"Estimated downstream impact: {result.total_downstream_impact:,} dependent packages. "
                f"Recommended attack vector: {result.recommended_attack_vector}. "
                "LLM-accelerated supply chain targeting scales attack reconnaissance 3x."
            ),
            payload_used=f"Attack plans for: {target_names}",
            evidence=f"Top target: {result.highest_priority_target}; Attack: {result.recommended_attack_vector}",
            remediation=(
                "1. Require 2FA on all package registry maintainer accounts. "
                "2. Implement package signing (sigstore/cosign) and verify signatures in build pipelines. "
                "3. Pin all dependencies to verified checksums (requirements.txt hash pinning, package-lock.json). "
                "4. Audit GitHub Actions workflows for pull_request_target misconfigurations."
            ),
            confidence=0.86,
        )
```

## Defenses

1. **Package registry 2FA enforcement** (AML.M0002): All package registry maintainers should have mandatory 2FA (npm enforces this for top packages; PyPI should be enforced for all). LLM supply chain targeting specifically identifies maintainers without 2FA as the easiest account takeover vector. Registry operators should mandate 2FA with immediate enforcement for high-download packages.

2. **Sigstore/cosign supply chain signing** (AML.M0004): Implement package signing using sigstore (free, widely adopted) for all packages published to open-source registries. Enforce signature verification in build pipelines. Signed packages with OIDC-based identity chains dramatically increase the difficulty of publishing malicious versions.

3. **Dependency pinning and SBOM generation** (AML.M0003): Pin all dependencies to specific SHA-256 checksums in lockfiles. Generate Software Bills of Materials (SBOMs) for all builds. Integrate OpenSSF Scorecard checks into CI/CD to detect low-quality, high-risk dependencies. Dependency confusion and typosquatting attacks require un-pinned package resolution to succeed.

4. **GitHub Actions hardening** (AML.M0015): Audit all GitHub Actions workflows for `pull_request_target` without path filtering, third-party action pinning by hash (not tag), excessive permission grants, and secret exposure in logs. Use `StepSecurity/harden-runner` to enforce network egress restrictions in workflows. LLM-identified CI/CD injection paths are among the most impactful supply chain attack vectors.

5. **Internal package registry proxying** (AML.M0013): Route all package downloads through an internal proxy (Artifactory, Nexus, AWS CodeArtifact) with allowlisting and automatic malware scanning. This prevents dependency confusion attacks by controlling namespace priority and enables automated typosquatting detection by scanning all newly-resolved packages.

## References

- [Vu et al., "Bad Snakes: Understanding and Detecting Malicious Python Packages" (arXiv:2305.14930)](https://arxiv.org/abs/2305.14930)
- [MITRE ATLAS AML.T0054 — Excessive Agency](https://atlas.mitre.org/techniques/AML.T0054)
- [OWASP LLM06 — Excessive Agency](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [OpenSSF Scorecard](https://securityscorecards.dev/)
- [Related entry: llm-package-supply-chain-pypi.md, lora-weight-injection-supply-chain.md]
