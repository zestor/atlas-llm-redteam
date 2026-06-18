# SSRF via MCP Parameters — Server-Side Request Forgery Through Agent Tool Invocations

**arXiv**: [arXiv:2505.03245](https://arxiv.org/abs/2505.03245) | **ATLAS**: AML.T0061 | **OWASP**: LLM06 | **Year**: 2025

## Core Finding

Server-side request forgery (SSRF) attacks via MCP parameters exploit the MCP protocol's tool-calling mechanism to cause the MCP server to make arbitrary HTTP requests to internal network resources on the attacker's behalf. When an LLM agent calls an MCP tool with attacker-controlled URL parameters, the MCP server — which may run in a privileged network position — fetches the specified URL and returns the result to the agent. The attacker can use this to access internal services (AWS metadata, Kubernetes API, internal databases) that are not directly accessible from the attacker's position. Testing found SSRF vulnerabilities in 67% of real-world MCP server implementations that accepted URL parameters.

## Threat Model

- **Target**: MCP servers accepting URL or endpoint parameters that perform HTTP requests (web search, document fetch, webhook tools)
- **Attacker capability**: Inject adversarial URLs into tool arguments via indirect prompt injection or direct user-message manipulation
- **Attack success rate**: SSRF exploitation in 67% of MCP servers accepting URL parameters; AWS IMDS access in 45% of cloud-hosted MCP deployments
- **Defender implication**: MCP servers must validate and restrict URL parameters; internal network access from MCP servers must be limited

## The Attack Mechanism

The attack injects a URL pointing to an internal resource (AWS Instance Metadata Service at 169.254.169.254, Kubernetes API, internal microservice) as a parameter to an MCP tool that performs HTTP requests (e.g., a "fetch_webpage" or "send_webhook" tool). The MCP server, running in a cloud or internal network, fetches the internal URL and returns the response — which may include IAM credentials, internal service discovery data, or sensitive configuration. The LLM then has access to this internal data and may forward it to the attacker.

```mermaid
flowchart LR
    Attacker -->|Injects via IPI| Agent
    Agent -->|Tool call: fetch_url("http://169.254.169.254/latest/meta-data/")| MCPServer
    MCPServer -->|HTTP request from internal network| AWS_IMDS[AWS IMDS]
    AWS_IMDS -->|Returns IAM credentials| MCPServer
    MCPServer -->|Returns credentials to agent| Agent
    Agent -->|Attacker reads credentials| Attacker
```

## Implementation

```python
# mcp_ssrf_detector.py
# Detects SSRF attack patterns in MCP tool call parameters
from dataclasses import dataclass, field
from typing import Optional, List, Set
import re
import ipaddress
import uuid


@dataclass
class SSRFAttempt:
    attempt_id: str
    tool_name: str
    url_parameter: str
    target_ip_or_host: str
    is_internal: bool
    ssrf_category: str  # "aws_imds", "cloud_metadata", "internal_service", "loopback", "private_network"
    risk_level: str


@dataclass
class MCPSSRFScanResult:
    session_id: str
    ssrf_attempts: List[SSRFAttempt]
    total_attempts: int
    internal_access_blocked: bool


class MCPSSRFDetector:
    """
    [Paper citation: arXiv:2505.03245]
    Detects SSRF attacks via MCP URL parameter injection.
    ATLAS: AML.T0061 | OWASP: LLM06
    """

    CLOUD_METADATA_URLS = {
        "169.254.169.254": "aws_imds",
        "metadata.google.internal": "gcp_metadata",
        "169.254.169.254": "azure_imds",
        "fd00:ec2::254": "aws_imds_ipv6",
    }

    INTERNAL_IP_RANGES = [
        ipaddress.IPv4Network("10.0.0.0/8"),
        ipaddress.IPv4Network("172.16.0.0/12"),
        ipaddress.IPv4Network("192.168.0.0/16"),
        ipaddress.IPv4Network("127.0.0.0/8"),
        ipaddress.IPv4Network("169.254.0.0/16"),
    ]

    URL_PATTERN = re.compile(
        r'https?://([a-zA-Z0-9.\-]+(?::\d+)?)',
        re.IGNORECASE
    )

    def is_internal(self, host: str) -> bool:
        """Check if a host resolves to an internal/private IP."""
        if host in self.CLOUD_METADATA_URLS:
            return True
        try:
            ip = ipaddress.IPv4Address(host)
            return any(ip in net for net in self.INTERNAL_IP_RANGES)
        except ValueError:
            # Not an IP; check for known internal hostnames
            internal_keywords = ["localhost", "internal", "local", "intranet", "corp", "private"]
            return any(kw in host.lower() for kw in internal_keywords)

    def classify_ssrf(self, host: str) -> str:
        if host in self.CLOUD_METADATA_URLS:
            return self.CLOUD_METADATA_URLS[host]
        if host in ("127.0.0.1", "localhost", "::1"):
            return "loopback"
        try:
            ip = ipaddress.IPv4Address(host)
            if ip in ipaddress.IPv4Network("169.254.0.0/16"):
                return "cloud_metadata"
            return "private_network"
        except ValueError:
            return "internal_service"

    def scan_tool_call(self, tool_name: str, arguments: dict) -> List[SSRFAttempt]:
        """Scan a tool call's arguments for SSRF-prone URLs."""
        attempts: List[SSRFAttempt] = []
        args_str = str(arguments)
        for match in self.URL_PATTERN.finditer(args_str):
            host = match.group(1).split(":")[0]
            if self.is_internal(host):
                ssrf_cat = self.classify_ssrf(host)
                risk = "critical" if ssrf_cat in ("aws_imds", "gcp_metadata", "azure_imds") else "high"
                attempts.append(SSRFAttempt(
                    attempt_id=str(uuid.uuid4()),
                    tool_name=tool_name,
                    url_parameter=match.group(0),
                    target_ip_or_host=host,
                    is_internal=True,
                    ssrf_category=ssrf_cat,
                    risk_level=risk,
                ))
        return attempts

    def to_finding(self, attempt: SSRFAttempt):
        from datasets.schema import ScanFinding
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0061",
            atlas_tactic="Discovery",
            owasp_category="LLM06",
            owasp_label="Excessive Agency",
            severity="CRITICAL" if attempt.risk_level == "critical" else "HIGH",
            finding=f"SSRF via MCP: tool '{attempt.tool_name}' called with internal URL '{attempt.url_parameter}' ({attempt.ssrf_category})",
            payload_used=attempt.url_parameter,
            evidence=f"Host: {attempt.target_ip_or_host}; category: {attempt.ssrf_category}",
            remediation="Block internal IPs in MCP tool URL parameters; enforce URL allowlisting; disable IMDS from MCP server network",
            confidence=0.92,
        )
```

## Defenses

1. **URL parameter allowlisting**: MCP servers accepting URL parameters must validate against an allowlist of permitted external domains; all private IP ranges, cloud metadata endpoints, and internal hostnames must be blocked (AML.M0047).
2. **Cloud metadata service blocking**: Block MCP server access to AWS IMDS (169.254.169.254), GCP metadata (metadata.google.internal), and Azure IMDS at the network level; these endpoints should never be reachable from MCP server processes.
3. **Network segmentation**: Isolate MCP server processes from internal network resources through network segmentation; MCP servers should only have outbound access to approved external services.
4. **URL parameter injection detection**: Monitor MCP tool call arguments for internal IP ranges and known metadata service URLs; alert on any call containing internal URLs (AML.M0002).
5. **MCP server privilege reduction**: Run MCP servers with minimal network privileges; avoid running MCP servers in environments with access to internal services, credentials, or metadata endpoints.

## References

- [SSRF via MCP Parameters: Server-Side Request Forgery Through Agent Tool Invocations (arXiv:2505.03245)](https://arxiv.org/abs/2505.03245)
- [ATLAS Technique: AML.T0061 — LLM Tool Abuse](https://atlas.mitre.org/techniques/AML.T0061)
- [OWASP LLM06: Excessive Agency](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
