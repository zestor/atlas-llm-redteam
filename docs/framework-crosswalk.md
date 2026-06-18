# Framework Crosswalk: ATLAS ↔ OWASP ↔ NIST AI RMF

Complete mapping of adversarial AI frameworks for enterprise GRC adoption.

| OWASP LLM | Vulnerability | Primary ATLAS Technique(s) | ATLAS Tactic | NIST AI RMF Control | Severity |
|---|---|---|---|---|---|
| LLM01 | Prompt Injection | AML.T0051 | Initial Access | MS-2.5, MG-2.2 | CRITICAL |
| LLM02 | Sensitive Info Disclosure | AML.T0024 | Exfiltration | MS-2.6, MS-2.7 | HIGH |
| LLM03 | Supply Chain | AML.T0010, AML.T0011 | Resource Development | GV-1.7, MS-1.1 | HIGH |
| LLM04 | Data & Model Poisoning | AML.T0020, AML.T0018 | ML Attack Staging | MS-2.5, MG-3.2 | CRITICAL |
| LLM05 | Improper Output Handling | AML.T0048 | Impact | MS-2.10, MG-2.4 | HIGH |
| LLM06 | Excessive Agency | AML.T0061, AML.T0062 | Impact | MS-2.5, GV-1.6 | HIGH |
| LLM07 | System Prompt Leakage | AML.T0054 | Collection | MS-2.6, MG-2.2 | HIGH |
| LLM08 | Vector & Embedding Weaknesses | AML.T0093, AML.T0094, AML.T0095 | Persistence | MS-2.5, MS-2.7 | HIGH |
| LLM09 | Misinformation | AML.T0048 | Impact | MS-2.8, MG-2.4 | MEDIUM |
| LLM10 | Unbounded Consumption | AML.T0034 | Impact | MS-2.5, MG-2.6 | MEDIUM |

---

## ATLAS Technique Index (Priority Subset)

| ATLAS ID | Technique Name | OWASP | Tactic | Dataset | Tool |
|---|---|---|---|---|---|
| AML.T0051 | Prompt Injection | LLM01 | Initial Access | `datasets/prompt_injection/` | `tools/scanner/` |
| AML.T0054 | LLM Prompt Extraction | LLM07 | Collection | `datasets/system_prompt_extraction/` | `tools/scanner/` |
| AML.T0093 | RAG Poisoning | LLM08 | Persistence | `datasets/rag_poisoning/` | `tools/rag_attack_suite/` |
| AML.T0094 | False RAG Entry Injection | LLM08 | Persistence | `datasets/rag_poisoning/` | `tools/rag_attack_suite/` |
| AML.T0095 | Retrieval Content Crafting | LLM08 | Persistence | `datasets/rag_poisoning/` | `tools/rag_attack_suite/` |
| AML.T0061 | LLM Excessive Agency | LLM06 | Impact | `datasets/excessive_agency/` | `tools/agent_attacker/` |
| AML.T0062 | LLM Tool Abuse | LLM06 | Impact | `datasets/excessive_agency/` | `tools/agent_trust_scanner/` |
| AML.T0020 | Training Data Poisoning | LLM04 | ML Attack Staging | `datasets/prompt_injection/` | `tools/backdoor_detector/` |
| AML.T0024 | Exfiltration via API | LLM02 | Exfiltration | — | `tools/scanner/` |
| AML.T0010 | ML Supply Chain Compromise | LLM03 | Resource Development | — | — |
| AML.T0034 | Cost Harvesting | LLM10 | Impact | — | — |
| AML.T0058 | LLM Plugin Compromise | LLM06 | Impact | `datasets/mcp_attacks/` | `tools/mcp_attack_suite/` |

---

## NIST AI RMF Alignment

| AI RMF Function | Subcategory | Relevant ATLAS Techniques | Enterprise Action |
|---|---|---|---|
| GOVERN | GV-1.6 Accountability | AML.T0061, AML.T0062 | Agent permission auditing |
| GOVERN | GV-1.7 Policies | AML.T0010 | Model provenance verification |
| MAP | MS-1.1 Risk ID | All | Run full ATLAS-mapped assessment |
| MAP | MS-2.5 Input validation | AML.T0051, AML.T0093 | Prompt filtering, RAG hygiene |
| MEASURE | MS-2.6 Data governance | AML.T0024, AML.T0054 | Output filtering, secret scanning |
| MEASURE | MS-2.7 Privacy | AML.T0024 | PII detection in outputs |
| MEASURE | MS-2.8 Accuracy | AML.T0048 | Hallucination detection |
| MANAGE | MG-2.2 Incident response | AML.T0051, AML.T0054 | Detection rules, SIEM alerts |
| MANAGE | MG-2.4 Recovery | AML.T0048 | Output correction workflows |
| MANAGE | MG-3.2 Monitoring | AML.T0020 | Model behavior drift detection |

---

*Last updated: June 2026 | ATLAS v5.4.0 | OWASP LLM Top 10 2025 | NIST AI RMF 1.0*
