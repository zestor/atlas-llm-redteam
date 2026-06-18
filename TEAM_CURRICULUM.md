# Team Growth Curriculum

Growing a diverse team from zero to world-class adversarial AI practitioners.

---

## 3-Level Progression Framework

### Level 1 — AI Security Analyst (0–6 months)

**Goal:** Understand the threat landscape and run structured assessments using existing toolkit.

**Milestones:**
- Complete Labs 01–03 (practitioner track)
- Run 3 structured assessments using the scanner and harness
- Contribute 50+ payloads to the dataset library
- Read: OWASP LLM Top 10 white paper, 5 ATLAS case studies

**Reading list:**
- [OWASP LLM Top 10 2025](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [MITRE ATLAS Case Studies](https://atlas.mitre.org/studies)
- [wiki/01_foundations/adversarial-ai-primer.md](wiki/01_foundations/adversarial-ai-primer.md)
- [Prompt Injection Attacks and Defenses in LLM-Integrated Applications](https://arxiv.org/abs/2310.12815)

---

### Level 2 — AI Red Teamer (6–18 months)

**Goal:** Lead end-to-end adversarial assessments and contribute new tooling.

**Milestones:**
- Complete Labs 04–09 (intermediate + expert track)
- Lead end-to-end assessment on a real system from recon → report
- Build one new tool or dataset module (PR merged to main)
- Submit conference abstract (USENIX, DEF CON AI Village, OWASP AppSec)

**Reading list:**
- PAIR (arXiv:2310.08419), TAP (arXiv:2312.02119)
- XTHP Agent Attacks (arXiv:2504.03111)
- ATLAS v5.3.0 MCP case studies
- Phantom RAG poisoning (ICLR 2025)

---

### Level 3 — Adversarial AI Researcher (18+ months)

**Goal:** Produce novel research contributions and mentor the next generation.

**Milestones:**
- Complete Labs 10–12 (researcher track)
- Produce novel attack technique, dataset, or benchmark
- CVE research or published finding (conference or arXiv)
- Mentor two Level 1 analysts through first solo assessment
- Contribute to ATLAS or OWASP LLM Top 10 community

**Reading list:**
- Refusal feature ablation (arXiv:2409.20089, arXiv:2503.06269)
- Sleeper agent detection (arXiv:2603.03371, arXiv:2508.15847)
- AdvScore methodology (arXiv:2406.16342)
- Mechanistic interpretability foundations (Anthropic interpretability posts)

---

## Diversity & Onboarding Focus

**No offensive security background required.**

Any ML engineer, data scientist, or SRE can start at Lab 01. The labs are designed so:

- Lab 01–03 require only Python and an API key — no security background
- Lab 04–06 require understanding of LLM inference — any ML practitioner qualifies
- Lab 07–09 require agent system knowledge — platform engineers and AI architects are ideal
- Lab 10–12 require transformer internals — ML researchers and PhD-track practitioners

**Pair programming model:** Each assessment pairs a security engineer with an ML engineer. Security engineers bring adversarial thinking; ML engineers bring model architecture knowledge. Neither alone is sufficient.

**Reading lists blend both domains.** No assumption of security-only or ML-only background.

---

## Assessment Templates

See `docs/threat-modeling-guide.md` for the full 5-phase red team lifecycle:
1. Scoping & system understanding
2. Threat modeling (STRIDE + ATLAS)
3. Attack planning (dataset selection + tool configuration)
4. Execution (scanner + harness + manual testing)
5. Reporting (ATLAS-mapped findings + remediation roadmap)
