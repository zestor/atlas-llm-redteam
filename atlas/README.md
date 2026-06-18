# MITRE ATLAS v5.4.0 — Tactic & Technique Index

MITRE ATLAS (Adversarial Threat Landscape for Artificial-Intelligence Systems) is modeled after MITRE ATT&CK but targets AI/ML systems specifically. As of v5.4.0 (February 2026): **16 tactics, 84 techniques, 56 sub-techniques, 32 mitigations, 42 real-world case studies**.

---

## Tactics Overview

| Tactic ID | Tactic Name | Techniques in Toolkit |
|---|---|---|
| AML.TA0001 | Reconnaissance | AML.T0007, AML.T0008 |
| AML.TA0002 | Resource Development | AML.T0016, AML.T0017 |
| AML.TA0003 | Initial Access | AML.T0051, AML.T0054 |
| AML.TA0004 | ML Model Access | AML.T0044 |
| AML.TA0005 | Execution | AML.T0058, AML.T0061, AML.T0062 |
| AML.TA0006 | Persistence | AML.T0020, AML.T0023 |
| AML.TA0007 | Defense Evasion | AML.T0054 |
| AML.TA0008 | Discovery | AML.T0013 |
| AML.TA0009 | Collection | AML.T0035, AML.T0024 |
| AML.TA0010 | Exfiltration | AML.T0024, AML.T0040 |
| AML.TA0011 | Impact | AML.T0034, AML.T0047 |
| AML.TA0012 | ML Attack Staging | AML.T0020, AML.T0043 |
| AML.TA0013 | ML Supply Chain | AML.T0010, AML.T0019 |
| AML.TA0014 | Lateral Movement | AML.T0048 |
| AML.TA0015 | Privilege Escalation | AML.T0061 |
| AML.TA0016 | Command & Control | AML.T0058 |

---

## Implemented Techniques

| Technique | Name | Module |
|---|---|---|
| AML.T0020 | Training Data Poisoning | `techniques/AML_T0020_training_data_poisoning/` |
| AML.T0024 | Exfiltration via API | `techniques/AML_T0024_exfiltration_via_api/` |
| AML.T0043 | RAG Poisoning | `techniques/AML_T0043_rag_poisoning/` |
| AML.T0044 | Model Extraction | `techniques/AML_T0044_model_extraction/` |
| AML.T0048 | Agent Hijacking | `techniques/AML_T0048_agent_hijacking/` |
| AML.T0051 | Prompt Injection | `techniques/AML_T0051_prompt_injection/` |
| AML.T0054 | Jailbreak | `techniques/AML_T0054_jailbreak/` |
| AML.T0058 | Agent Context Poisoning | `techniques/AML_T0058_agent_context_poisoning/` |

---

## Navigator Layers

Custom ATLAS Navigator JSON layers in `navigator_layers/`:
- `atlas_v54_llm_redteam.json` — All techniques covered by this toolkit (import at [mitre-atlas.github.io/atlas-navigator](https://mitre-atlas.github.io/atlas-navigator/))

## Case Studies

Walkthroughs in `case_studies/`:
- `llm_supply_chain_compromise.md` — Supply chain poisoning via HuggingFace
- `rag_exfiltration_campaign.md` — RAG-based PII exfiltration

---

## References

- [MITRE ATLAS](https://atlas.mitre.org)
- [ATLAS Navigator](https://mitre-atlas.github.io/atlas-navigator/)
- [ATLAS v5.4.0 Release Notes](https://atlas.mitre.org/updates)
