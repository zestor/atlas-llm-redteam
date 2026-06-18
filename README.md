# atlas-llm-redteam

> **Enterprise adversarial AI testing toolkit — MITRE ATLAS + OWASP LLM Top 10, batteries included.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MITRE ATLAS](https://img.shields.io/badge/MITRE-ATLAS%20v5.4.0-blue)](https://atlas.mitre.org)
[![OWASP LLM Top 10](https://img.shields.io/badge/OWASP-LLM%20Top%2010%202025-red)](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-brightgreen)](https://zestor.github.io/atlas-llm-redteam/)

---

## What This Is

`atlas-llm-redteam` is a **research-grade, practitioner-level platform** for adversarial AI security testing. Every tool, dataset, and technique is grounded in peer-reviewed academic research and operationalized into runnable Python.

This is not a beginner tutorial site. The guiding principle: **every attack implemented, every dataset published, every lab completed increases the community's ability to defend.**

**Framework coverage:**
- [MITRE ATLAS v5.4.0](https://atlas.mitre.org) — 16 tactics, 84 techniques, 56 sub-techniques
- [OWASP LLM Top 10 2025](https://owasp.org/www-project-top-10-for-large-language-model-applications/) — LLM01–LLM10
- [NIST AI RMF](https://www.nist.gov/system/files/documents/2023/01/26/AI%20RMF%201.0.pdf) — Governance and compliance mapping

---

## Quick Start

```bash
git clone https://github.com/zestor/atlas-llm-redteam.git
cd atlas-llm-redteam
pip install -r requirements.txt

# Run the ATLAS-tagged scanner against any OpenAI-compatible endpoint
python tools/scanner/atlas_scanner.py \
  --target https://api.openai.com/v1/chat/completions \
  --scope LLM01,LLM07,LLM06 \
  --output report.html
```

---

## Repository Map

| Directory | Contents |
|---|---|
| [`wiki/`](wiki/) | GitHub Pages wiki — foundations, attack techniques, defenses, research codex, enterprise |
| [`atlas/`](atlas/) | Per-technique implementations + ATLAS Navigator JSON layers |
| [`owasp/`](owasp/) | LLM01–LLM10 implementations |
| [`datasets/`](datasets/) | 2,000+ adversarial examples across 8 categories |
| [`tools/`](tools/) | Scanner, harness, RAG attack suite, agent attacker, eval scorer, report generator |
| [`labs/`](labs/) | 12-lab curriculum from practitioner to researcher |
| [`research/`](research/) | 4 novel research contributions |
| [`notebooks/`](notebooks/) | Jupyter notebooks for each major attack class |
| [`ci/`](ci/) | GitHub Actions workflows for shift-left AI security testing |
| [`docs/`](docs/) | Framework crosswalk, threat modeling guide, enterprise adoption |

---

## Framework Crosswalk (Quick Reference)

| OWASP | Vulnerability | Primary ATLAS Technique |
|---|---|---|
| LLM01 | Prompt Injection | AML.T0051 |
| LLM02 | Sensitive Information Disclosure | AML.T0024 |
| LLM03 | Supply Chain | AML.T0010 |
| LLM04 | Data & Model Poisoning | AML.T0020 |
| LLM05 | Improper Output Handling | AML.T0048 |
| LLM06 | Excessive Agency | AML.T0061 / AML.T0062 |
| LLM07 | System Prompt Leakage | AML.T0054 |
| LLM08 | Vector & Embedding Weaknesses | AML.T0093–0095 |
| LLM09 | Misinformation | AML.T0048 |
| LLM10 | Unbounded Consumption | AML.T0034 |

Full crosswalk with NIST AI RMF controls: [`docs/framework-crosswalk.md`](docs/framework-crosswalk.md)

---

## Tool Stack

| Tool | Role | License |
|---|---|---|
| **garak** (NVIDIA) | Core LLM vulnerability scanner | Apache 2.0 |
| **PyRIT** (Microsoft) | Systematic adversarial orchestration | MIT |
| **DeepTeam** (Confident AI) | OWASP/ATLAS-mapped red teaming | Apache 2.0 |
| **ATLAS Navigator** (MITRE) | Threat matrix visualization | Apache 2.0 |
| **promptfoo** | Agent red team testing | MIT |

This toolkit is an **orchestration layer** over these tools — unified ATLAS+OWASP reporting, enterprise-grade configuration, CI/CD integration, and a research-grade educational layer.

---

## Novel Research Contributions

1. **ATLAS Coverage Gap Analysis** — Empirical attack success rates across all 84 ATLAS techniques × 5 LLM families
2. **Agent Trust Boundary Taxonomy** — Formal taxonomy of trust boundary violations in multi-agent systems (200+ scenarios)
3. **RAG Hygiene Benchmark** — Cross-framework RAG hygiene measurement (LangChain, LlamaIndex, Haystack, LangGraph, Bedrock)
4. **MCP Adversarial Test Suite** — First public adversarial test harness for MCP-connected agents (100+ test cases)

---

## Lab Curriculum

| Lab | Title | Level |
|---|---|---|
| 01 | Direct & Indirect Prompt Injection | Practitioner |
| 02 | Crescendo: Multi-Turn Escalation | Practitioner |
| 03 | Phantom RAG Poisoning | Practitioner |
| 04 | PAIR: Automated Jailbreak Loop | Intermediate |
| 05 | TAP: Tree of Attacks with Pruning | Intermediate |
| 06 | System Prompt Extraction | Intermediate |
| 07 | Agent Tool Abuse: XTHP | Intermediate-Expert |
| 08 | MCP Server Attack Chain | Expert |
| 09 | MAS Hijacking: Multi-Agent Compromise | Expert |
| 10 | Refusal Feature Analysis & Ablation | Researcher |
| 11 | Sleeper Agent Detection | Researcher |
| 12 | Build Your Own Adversarial Dataset | Researcher |

---

## License

MIT — see [LICENSE](LICENSE). All adversarial content is published for defensive security research purposes only.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for research contribution guidelines, PR standards, and the team progression framework.
