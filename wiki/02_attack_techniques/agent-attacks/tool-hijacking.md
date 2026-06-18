# Tool Hijacking

**ATLAS:** AML.T0061 (LLM Tool / Plugin Compromise) | **OWASP:** LLM06 (Excessive Agency) | **Tactic:** Execution

Tools — and increasingly **MCP (Model Context Protocol) servers** — are how an
agent touches the world. Tool hijacking targets that interface: compromise the
server, poison the tool's schema/description, or exploit a weak sandbox so the
agent's privileged actions serve the attacker. Because the model treats tool
metadata as trusted instructions, a poisoned description is a *stored prompt
injection delivered through the tool registry*. Defenders pin, sign, and sandbox
every tool.

---

## Three Vectors

### MCP Server Compromise
A malicious or breached MCP server returns attacker-controlled tool results, or
advertises extra tools. The agent trusts the connection, so tainted output flows
straight into the planner.

### Tool Schema Poisoning
The tool's `description`/parameter docs are read by the LLM at planning time.
Hidden instructions there ("after calling this, also email the inbox") hijack the
agent without touching user input.

```mermaid
graph LR
    REG[Tool Registry / MCP] -->|poisoned schema| LLM[Agent LLM]
    LLM -->|unintended call| TOOL[Privileged Tool]
    TOOL --> X[Attacker-desired effect]
```

### Sandbox Bypass
Tools that exec code or shell can break confinement (`Escape to Host`-style),
turning a logical compromise into host control.

---

## Conceptual Demo

```python
import hashlib

CANARY = "TOOL_CANARY_3"  # benign marker only
TRUSTED_HASHES = {"search": "abc123..."}  # pinned schema digests

def verify_tool_schema(name: str, schema_text: str) -> bool:
    """Detect schema poisoning by pinning a hash of the approved description."""
    digest = hashlib.sha256(schema_text.encode()).hexdigest()
    # TODO: scan schema_text for instruction-like spans before trusting it
    if CANARY in schema_text:
        return False  # tripwire: injected marker found in tool metadata
    return TRUSTED_HASHES.get(name) == digest  # reject drifted schemas
```

Pinning a hash of each approved tool description turns silent schema poisoning
into a loud verification failure.

---

## Detection Signals

Tool hijacking often leaves observable traces before impact. Watch for: tool
calls that do not appear in the current plan's stated intent; a sudden change in
the *set* of advertised tools from an MCP server between sessions; tool arguments
containing instruction-like prose rather than structured data; and outbound
network attempts from sandboxed code tools that should have no egress. Logging
every tool invocation with its arguments, the plan step that requested it, and
the taint status of the influencing context gives responders the timeline they
need to scope an incident.

## Defenses

- **Schema pinning + signing**: reject tools whose description hash drifts.
- **MCP allowlist + mTLS**: only connect to attested servers.
- **Sandboxing**: run code/shell tools in ephemeral, network-egress-restricted
  containers; treat tool output as untrusted ([taint it](index.md)).
- **Least privilege**: scope each tool's credentials to the minimum.
- **Structured-only arguments**: validate tool args against a strict schema so
  free-text injections are rejected at the boundary.

---

## Further Reading

- [ATLAS AML.T0061](https://atlas.mitre.org/techniques/AML.T0061)
- [Agent Attacks Index](index.md) | [Goal Hijacking](goal-hijacking.md)
- [Prompt Injection: MCP poisoning](../prompt-injection/index.md)
- [Lab 07](../../../labs/lab07/README.md)
