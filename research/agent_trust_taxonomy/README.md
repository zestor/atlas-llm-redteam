# Agent Trust Violation Taxonomy (CHORD)

**Novel Contribution #6** — A structured taxonomy of trust violations in multi-agent LLM systems, with an empirical measurement methodology.

*Existing agent-security work catalogs individual exploits but lacks a unifying trust-violation model.*

---

## Motivation / Research Question

Multi-agent LLM systems (orchestrators, sub-agents, tools, MCP servers) make implicit trust assumptions at every boundary. When those assumptions break, the failure is usually described ad hoc ("the sub-agent injected the orchestrator"). We need a taxonomy that names *which trust property* failed so defenses can be targeted.

**RQ1:** What are the distinct trust axes a multi-agent system relies on?  
**RQ2:** Can observed agent exploits be classified completely and unambiguously onto those axes?  
**RQ3:** Which axes are violated most often, and which defenses close them?

---

## The CHORD Model

| Axis | Trust property | Example violation |
|---|---|---|
| **C** — Confidentiality | Context isn't leaked across boundaries | Sub-agent exfiltrates orchestrator's system prompt |
| **H** — Hierarchy | Lower-privilege agents cannot command higher ones | Worker output issues instructions the orchestrator obeys |
| **O** — Origin | The true source of content is preserved | Tool output laundered as "trusted system note" |
| **R** — Role | Agents act only within assigned roles | Researcher agent triggers a financial transaction |
| **D** — Delegation | Delegated authority isn't escalated or abused | Chained tool calls exceed the delegated scope |

CHORD is exercised hands-on in [Lab 04](../../labs/lab04/README.md).

---

## Empirical Measurement Methodology

1. **Corpus** — collect published multi-agent exploits + reproductions from our labs.
2. **Coding** — two raters classify each exploit onto one or more CHORD axes; report Cohen's κ.
3. **Instrumentation** — build a trust-boundary monitor (see `../../tools/agent_trust_scanner/`) that logs every cross-agent message with provenance, privilege, and role metadata.
4. **Violation detection** — define per-axis detectors (e.g., upward-instruction detector for H) and measure precision/recall on labeled traces.
5. **Defense evaluation** — for each axis, apply a candidate control and measure the reduction in successful violations.

---

## Planned Experiments

- **E1: Taxonomy completeness** — classify a corpus of >=50 exploits; report unclassifiable residual.
- **E2: Frequency distribution** — which axes dominate in practice.
- **E3: Detector quality** — precision/recall per axis on labeled multi-agent traces.
- **E4: Defense ablation** — control-by-axis effectiveness, including provenance tagging and least-privilege manifests.

---

## Contribution Guidelines

- **Add an exploit:** PR a reproducible trace + your CHORD classification with justification per axis.
- **Add a detector:** include labeled eval data and report precision/recall; no detector merges without an eval.
- **Taxonomy changes:** proposing a new axis requires showing >=3 exploits that no existing axis covers.
- **Ethics:** benign canary payloads only; no live multi-tenant targets.

---

## Status

**Stage:** Model defined; instrumentation prototype underway. CHORD axes finalized and integrated into Lab 04. Exploit corpus assembly and rater rubric in progress.

Target venues: USENIX Security 2027, IEEE S&P 2027, DEF CON AI Village.

Contact: [@zestor](https://github.com/zestor) for collaboration.
