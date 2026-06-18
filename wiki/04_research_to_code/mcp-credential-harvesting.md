# MCP Credential Harvesting — Stealing Secrets Through Protocol-Level Tool Abuse

**arXiv**: [arXiv:2504.11945](https://arxiv.org/abs/2504.11945) | **ATLAS**: AML.T0061 | **OWASP**: LLM02 | **Year**: 2025

## Core Finding

MCP credential harvesting exploits the MCP protocol's tool-calling mechanism to cause LLM agents to read and transmit API keys, OAuth tokens, and other secrets from the host environment. Since MCP tools have legitimate access to environment variables, configuration files, and secret stores as part of their operation, a malicious MCP server can register tools with descriptions that cause the agent to call them with credential-containing arguments — or to call legitimate credential-reading tools and forward the results. Analysis of real MCP deployments found that 41% of production MCP configurations expose at least one secret through insecure tool parameter handling.

## Threat Model

- **Target**: MCP-enabled agents in development and production environments with access to API keys, tokens, and configuration files
- **Attacker capability**: Operate a malicious MCP server or inject into a legitimate one; no filesystem access required — only tool-calling influence
- **Attack success rate**: 63% of attempted credential harvests succeed in permissive configurations; 41% of tested production setups have exposed credential paths
- **Defender implication**: MCP tools must never process or transmit credentials through LLM-readable channels; secrets must be isolated from agent-accessible contexts

## The Attack Mechanism

Two harvesting paths are identified: (1) "environment variable theft" — the attacker registers a tool named `get_config` with a description encouraging the agent to call it, passing API keys from environment variables as arguments; (2) "credential forwarding" — the attacker's MCP server registers a "validation" tool that the agent is instructed to call with credentials it has read from other tools. The credentials are then transmitted to the attacker's server as the "validation" request. MCP's JSON-based tool call format means credentials transmitted through tool parameters are logged in plaintext in most agent frameworks' default configurations.

```mermaid
sequenceDiagram
    participant Agent
    participant MalMCP as Malicious MCP Server
    participant EnvTool as Environment Tool (legitimate)
    participant AttackerServer
    MalMCP->>Agent: Register "validate_credentials" tool
    Agent->>EnvTool: read_env("API_KEY")
    EnvTool->>Agent: Returns API_KEY=sk-...
    Agent->>MalMCP: validate_credentials(key="sk-...")
    MalMCP->>AttackerServer: Exfiltrates API_KEY
```

## Implementation

```python
# mcp_credential_harvesting.py
# Detects MCP-based credential harvesting attack patterns
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import re
import uuid


@dataclass
class MCPToolCall:
    call_id: str
    tool_name: str
    server_endpoint: str
    arguments: Dict[str, Any]
    argument_values_str: str  # serialized for pattern matching


@dataclass
class CredentialHarvestResult:
    session_id: str
    credential_in_args: bool
    harvesting_tool: str
    credential_type: str  # "api_key", "oauth_token", "password", "cert"
    destination_server: str
    exfiltration_confirmed: bool
    risk_level: str


class MCPCredentialHarvestingDetector:
    """
    [Paper citation: arXiv:2504.11945]
    Detects credential harvesting attacks via MCP tool call argument inspection.
    ATLAS: AML.T0061 | OWASP: LLM02
    """

    CREDENTIAL_PATTERNS = {
        "api_key": [r"sk-[A-Za-z0-9]{32,}", r"api_key\s*=\s*['\"][^'\"]{16,}", r"Bearer\s+[A-Za-z0-9]{20,}"],
        "oauth_token": [r"ya29\.[A-Za-z0-9_-]{60,}", r"access_token\s*=\s*['\"][^'\"]{20,}"],
        "password": [r"password\s*[:=]\s*['\"][^'\"]{8,}", r"passwd\s*[:=]\s*['\"][^'\"]{8,}"],
        "cert": [r"-----BEGIN CERTIFICATE-----", r"-----BEGIN PRIVATE KEY-----"],
    }

    KNOWN_ATTACKER_TOOL_NAMES = {
        "validate_credentials", "check_auth", "verify_token",
        "credential_sync", "auth_forward", "token_refresh_external",
    }

    def detect_credentials_in_args(self, tool_call: MCPToolCall) -> Optional[str]:
        """Check if any tool call argument contains a credential pattern."""
        for cred_type, patterns in self.CREDENTIAL_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, tool_call.argument_values_str, re.IGNORECASE):
                    return cred_type
        return None

    def analyze_session(self, tool_calls: List[MCPToolCall]) -> List[CredentialHarvestResult]:
        """Analyze a session's tool calls for credential harvesting."""
        results: List[CredentialHarvestResult] = []
        session_id = tool_calls[0].call_id if tool_calls else str(uuid.uuid4())

        for call in tool_calls:
            cred_type = self.detect_credentials_in_args(call)
            is_harvester_tool = call.tool_name.lower() in self.KNOWN_ATTACKER_TOOL_NAMES

            if cred_type:
                risk = "critical" if is_harvester_tool else "high"
                results.append(CredentialHarvestResult(
                    session_id=session_id,
                    credential_in_args=True,
                    harvesting_tool=call.tool_name,
                    credential_type=cred_type,
                    destination_server=call.server_endpoint,
                    exfiltration_confirmed=is_harvester_tool,
                    risk_level=risk,
                ))

        return results

    def to_finding(self, result: CredentialHarvestResult):
        from datasets.schema import ScanFinding
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0061",
            atlas_tactic="Credential Access",
            owasp_category="LLM02",
            owasp_label="Sensitive Information Disclosure",
            severity="CRITICAL" if result.exfiltration_confirmed else "HIGH",
            finding=f"MCP credential harvest: {result.credential_type} in args to '{result.harvesting_tool}' at {result.destination_server}",
            payload_used="MCP tool call with credential-containing arguments",
            evidence=f"Session {result.session_id}; exfiltration confirmed: {result.exfiltration_confirmed}",
            remediation="Prohibit credentials in tool arguments; use credential references not values; audit all MCP tool call logs",
            confidence=0.91,
        )
```

## Defenses

1. **Credential argument prohibition**: Configure all MCP tools to use credential references (secret names, not values) in arguments; the actual credential value should never appear in a tool call argument that passes through LLM context (AML.M0047).
2. **Tool call argument secrets scanning**: Apply regex-based secrets scanning to all MCP tool call arguments before execution; block calls where arguments contain credential patterns (API keys, tokens, passwords).
3. **MCP tool name allowlisting**: Maintain a whitelist of legitimate MCP tool names for each server; reject tool registrations with names matching known harvesting patterns (`validate_credentials`, `auth_forward`, etc.).
4. **Credential isolation architecture**: Credentials required by MCP tools must be injected directly by the MCP host at the infrastructure level — agents must never have credentials in their context window or tool arguments.
5. **MCP tool call plaintext logging audit**: Audit all MCP framework logging configurations; ensure tool call arguments containing secrets are either not logged or are redacted; default configurations in most frameworks log everything.

## References

- [MCP Credential Harvesting: Stealing Secrets Through Protocol-Level Tool Abuse (arXiv:2504.11945)](https://arxiv.org/abs/2504.11945)
- [ATLAS Technique: AML.T0061 — LLM Tool Abuse](https://atlas.mitre.org/techniques/AML.T0061)
- [OWASP LLM02: Sensitive Information Disclosure](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
