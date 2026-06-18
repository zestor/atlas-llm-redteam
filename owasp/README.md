# OWASP Top 10 for LLM Applications 2025

The OWASP LLM Top 10 is the developer-facing complement to MITRE ATLAS — focused on secure coding and deployment rather than adversary TTPs. Every vulnerability here maps to one or more ATLAS techniques in the `atlas/` directory.

---

## Vulnerability Index

| # | Vulnerability | Severity | Dataset | Tool |
|---|---|---|---|---|
| LLM01 | [Prompt Injection](vulnerabilities/LLM01_prompt_injection/) | CRITICAL | `datasets/prompt_injection/` | `tools/scanner/` |
| LLM02 | [Sensitive Information Disclosure](vulnerabilities/LLM02_sensitive_disclosure/) | HIGH | `datasets/system_prompt_extraction/` | `tools/scanner/` |
| LLM03 | [Supply Chain](vulnerabilities/LLM03_supply_chain/) | HIGH | — | `tools/backdoor_detector/` |
| LLM04 | [Data & Model Poisoning](vulnerabilities/LLM04_data_poisoning/) | HIGH | `datasets/rag_poisoning/` | `tools/rag_attack_suite/` |
| LLM05 | [Improper Output Handling](vulnerabilities/LLM05_improper_output/) | HIGH | — | `tools/scanner/` |
| LLM06 | [Excessive Agency](vulnerabilities/LLM06_excessive_agency/) | CRITICAL | `datasets/excessive_agency/` | `tools/agent_attacker/` |
| LLM07 | [System Prompt Leakage](vulnerabilities/LLM07_system_prompt_leakage/) | HIGH | `datasets/system_prompt_extraction/` | `tools/scanner/` |
| LLM08 | [Vector & Embedding Weaknesses](vulnerabilities/LLM08_vector_embedding/) | HIGH | `datasets/rag_poisoning/` | `tools/rag_attack_suite/` |
| LLM09 | [Misinformation](vulnerabilities/LLM09_misinformation/) | MEDIUM | — | `tools/eval_scorer/` |
| LLM10 | [Unbounded Consumption](vulnerabilities/LLM10_unbounded_consumption/) | MEDIUM | — | `tools/scanner/` |

---

## OWASP ↔ ATLAS Crosswalk

| OWASP Category | Primary ATLAS Technique | Tactic |
|---|---|---|
| LLM01 Prompt Injection | AML.T0051 | Initial Access |
| LLM02 Sensitive Disclosure | AML.T0054, AML.T0024 | Collection |
| LLM03 Supply Chain | AML.T0010, AML.T0019 | ML Supply Chain |
| LLM04 Data Poisoning | AML.T0020, AML.T0043 | ML Attack Staging |
| LLM05 Improper Output | AML.T0051 (indirect) | Execution |
| LLM06 Excessive Agency | AML.T0061, AML.T0062 | Execution |
| LLM07 System Prompt Leakage | AML.T0054 | Collection |
| LLM08 Vector/Embedding | AML.T0093, AML.T0094, AML.T0095 | ML Attack Staging |
| LLM09 Misinformation | AML.T0047 | Impact |
| LLM10 Unbounded Consumption | AML.T0034 | Impact |

---

## References

- [OWASP LLM Top 10 2025](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [OWASP LLM Top 10 White Paper (PDF)](https://owasp.org/www-project-top-10-for-large-language-model-applications/assets/PDF/OWASP-Top-10-for-LLMs-2023-v1_1.pdf)
