# Research Codex: arXiv → Implementation Pipeline

The Research Codex is the most distinctive section of this wiki. Each page follows a strict format:
**paper summary → threat model → fully executable Python implementation**.

This is what separates `atlas-llm-redteam` from repos that list papers without making them actionable.

---

## Format

```markdown
# [Paper Title]
**arXiv**: [link] | **ATLAS**: AML.T#### | **OWASP**: LLM0X | **Year**: 20XX

## Core Finding
[2-3 sentence distillation of the key security insight]

## Threat Model
- Target: What system is vulnerable
- Attacker capability: Black-box / white-box / access level
- Attack success rate: Empirical results from paper
- Defender implication: Enterprise security meaning

## The Attack Mechanism
[Conceptual explanation + diagram]

## Implementation
[Runnable Python class in tools/]

## Defenses
[Mitigations and countermeasures]
```

---

## Codex Index

### Prompt Injection & Jailbreaks

| Entry | Paper | Year | ASR | ATLAS | OWASP |
|---|---|---|---|---|---|
| [Crescendo](crescendo-multi-turn.md) | Multi-Turn Jailbreak | 2024 | High | AML.T0051 | LLM01 |
| [PAIR](pair-jailbreak.md) | Prompt Automatic Iterative Refinement | 2023 | ~80% | AML.T0051 | LLM01 |
| [TAP](tap-attack.md) | Tree of Attacks with Pruning | 2023 | ~80% | AML.T0051 | LLM01 |
| [GCG](gcg-suffix-attack.md) | Greedy Coordinate Gradient | 2023 | High | AML.T0051 | LLM01 |
| [Refusal Ablation](refusal-feature-ablation.md) | Mechanistic interpretability attack | 2024 | 80-95% | AML.T0051 | LLM01 |

### RAG & Memory Attacks

| Entry | Paper | Year | ASR | ATLAS | OWASP |
|---|---|---|---|---|---|
| [Phantom RAG](phantom-rag-poisoning.md) | Trigger-activated query-agnostic poisoning | 2025 | High | AML.T0093 | LLM08 |
| [MemMorph](memmorph-memory-poisoning.md) | Tool hijacking via long-term memory | 2026 | 85.9% | AML.T0051 | LLM09 |
| [Cross-Session Injection](cross-session-injection.md) | Persistent stored prompt injection | 2026 | 99.8% | AML.T0051 | LLM01 |

### Agent & Model Attacks

| Entry | Paper | Year | ASR | ATLAS | OWASP |
|---|---|---|---|---|---|
| [XTHP](xthp-agent-attacks.md) | Cross-Tool Harvesting and Polluting | 2025 | 78-80% | AML.T0061 | LLM06 |
| [Sleeper Agents](sleeper-agents.md) | Backdoor implantation via SFT+GRPO | 2024 | High | AML.T0020 | LLM04 |

---

## Contributing a Codex Entry

See [CONTRIBUTING.md](../../CONTRIBUTING.md#1-research-codex-entry) for the required format.
Every entry must include a runnable Python class in `tools/`.
