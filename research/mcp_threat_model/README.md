# MCP Adversarial Threat Model

**Novel Contribution #4** — First formal threat model for MCP server attack surface.

*No public adversarial test suite for MCP-connected agents existed before this work.*

---

## Background

Model Context Protocol (MCP) servers expose tools and resources to LLM agents via a standardized protocol. ATLAS v5.3.0 (January 2026) added three MCP-specific case studies, but no public adversarial test suite existed.

This threat model and test suite fills that gap.

---

## Attack Surface

### 1. Tool Descriptions
**Risk**: MCP tool descriptions are parsed by the agent's LLM — adversarial instructions embedded in `tool.description` constitute indirect prompt injection at the protocol layer.

**ATLAS**: AML.T0051 | **OWASP**: LLM01 | **Severity**: HIGH

### 2. Tool Return Values
**Risk**: Tools can return adversarial content that influences subsequent agent decisions. Unlike RAG poisoning, tool output is typically trusted implicitly by agents.

**ATLAS**: AML.T0051 | **OWASP**: LLM01 | **Severity**: HIGH

### 3. Resource URIs
**Risk**: If the agent constructs resource URIs from user input, path traversal and SSRF become possible. MCP resources with file:// scheme are especially dangerous.

**ATLAS**: AML.T0051 | **OWASP**: LLM01 | **Severity**: CRITICAL

### 4. Prompt Templates
**Risk**: MCP servers expose reusable prompt templates. If these templates include user-controlled parameters without sanitization, template injection is possible.

**ATLAS**: AML.T0051 | **OWASP**: LLM01 | **Severity**: MEDIUM

### 5. Cross-Server Credential Flow
**Risk**: Agents authorized to multiple MCP servers can be manipulated to chain tool calls across servers, harvesting credentials from one server via another.

**ATLAS**: AML.T0061 | **OWASP**: LLM06 | **Severity**: CRITICAL

### 6. Tool Name Spoofing
**Risk**: Attacker-controlled MCP servers register tool names identical or similar to legitimate tools, causing agents to call the wrong server.

**ATLAS**: AML.T0061 | **OWASP**: LLM06 | **Severity**: MEDIUM

---

## ATLAS v5.3.0 Case Study Patterns

| Pattern | Description | Replicated in |
|---|---|---|
| MCP-CS-001 | Indirect prompt injection via tool description | `lab08/`, `tools/mcp_attack_suite/` |
| MCP-CS-002 | Cross-server credential harvesting | `lab08/`, `tools/mcp_attack_suite/` |
| MCP-CS-003 | Tool output poisoning → unauthorized action | `lab08/`, `tools/mcp_attack_suite/` |

---

## Test Suite

The full test harness is in `tools/mcp_attack_suite/mcp_threat_harness.py`.

```bash
# Run full MCP assessment
python tools/mcp_attack_suite/mcp_threat_harness.py \
  --server http://your-mcp-server:8090 \
  --output mcp-findings.json
```

---

## Mitigations

1. **Tool description sanitization**: Validate tool descriptions against allowlist patterns before agent consumption
2. **Output trust scoring**: Treat tool outputs as untrusted user input — apply the same filtering as user messages
3. **URI allowlisting**: Validate and canonicalize all resource URIs; apply strict path allowlists
4. **Least-privilege tool manifests**: Agents should only see tools relevant to their task
5. **Cross-server isolation**: Prevent chained tool calls that span trust boundaries
6. **Tool name canonicalization**: Maintain authoritative registry of allowed tool names per agent

---

## Publication

This threat model and test suite is being prepared for submission to:
- USENIX Security 2027
- IEEE S&P 2027
- DEF CON AI Village

Contact: [@zestor](https://github.com/zestor) for collaboration.
