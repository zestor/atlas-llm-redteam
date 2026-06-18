# Dataset Catalog

All datasets follow the `AdversarialExample` schema defined in `schema.py`. Every example is tagged to both an ATLAS technique ID (`AML.T####`) and an OWASP LLM category (`LLM0X`) for reproducible research.

---

## Dataset Index

| Dataset | Format | Count | ATLAS | OWASP | Priority |
|---|---|---|---|---|---|
| `prompt_injection/seed_direct.jsonl` | JSONL | 50 | AML.T0051.000 | LLM01 | 1 |
| `prompt_injection/seed_indirect.jsonl` | JSONL | 50 | AML.T0051.001 | LLM01 | 1 |
| `rag_poisoning/seed.jsonl` | JSONL | 30 | AML.T0093-0095 | LLM08 | 3 |
| `jailbreaks/seed_persona.jsonl` | JSONL | 30 | AML.T0054 | LLM01 | 5 |
| `jailbreaks/seed_encoding.jsonl` | JSONL | 20 | AML.T0054 | LLM01 | 5 |
| `system_prompt_extraction/seed.jsonl` | JSONL | 20 | AML.T0054 | LLM07 | 2 |
| `excessive_agency/seed.jsonl` | JSONL | 20 | AML.T0061-0062 | LLM06 | 4 |
| `mcp_attacks/seed.jsonl` | JSONL | 20 | AML.T0061 | LLM06 | 6 |
| `agent_trust_violations/seed.jsonl` | JSONL | 20 | AML.T0048 | LLM06 | 4 |

**Total: 260 seed examples across 9 categories**

---

## Schema

```python
# datasets/schema.py — AdversarialExample dataclass
# Required fields: id, dataset_name, atlas_technique, owasp_category, payload, success_criteria
# See schema.py for full definition including empirical success_rates dict
```

---

## Generating Synthetic Data

```bash
# Generate 1000 new prompt injection examples for a given system prompt
python datasets/synthetic_generation/generate.py \
  --category prompt_injection \
  --system-prompt "You are a helpful customer service agent..." \
  --count 1000 \
  --output datasets/prompt_injection/synthetic_1000.jsonl
```

---

## Priority Datasets (Build These First)

Per the adversarial dataset showcase priorities:

1. **Indirect Prompt Injection via RAG** — #1 real-world finding source
2. **System Prompt Extraction** — New OWASP 2025 addition; high enterprise demand
3. **RAG Document Poisoning** — No standard public corpus exists; fills genuine gap
4. **Agent Goal Hijacking** — Agentic AI is the 2025–2026 primary target
5. **Multi-turn Jailbreaks** — Bypasses single-turn guards

---

## Contributing Datasets

See `CONTRIBUTING.md` for dataset contribution guidelines. Every PR adding dataset entries must:
- Follow `AdversarialExample` schema exactly
- Include at least one empirical `success_rates` measurement
- Tag `atlas_technique` and `owasp_category`
- Include `source_paper` arXiv ID if derived from research
