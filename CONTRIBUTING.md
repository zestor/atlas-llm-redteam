# Contributing to atlas-llm-redteam

## Design Philosophy

This is a **research-grade platform**. Every contribution must meet the standard: grounded in peer-reviewed research, operationalized into runnable Python, framed from the defender's perspective.

---

## Contribution Types

### 1. Research Codex Entry (`wiki/04_research_to_code/`)
The highest-value contribution. Each page: paper summary → threat model → executable implementation.

**Required format:**
```markdown
# [Paper Title]
**arXiv**: [link] | **ATLAS**: AML.T#### | **OWASP**: LLM0X | **Year**: 20XX

## Core Finding
[2-3 sentence distillation]

## Threat Model
- **Target**: What system is vulnerable
- **Attacker capability**: Black-box / white-box / access level
- **Attack success rate**: Empirical results from paper
- **Defender implication**: Enterprise security meaning

## The Attack Mechanism
[Conceptual explanation]

## Implementation
```python
# Runnable Python class
```

## Defenses
[Mitigations and countermeasures]
```

### 2. Dataset Contribution (`datasets/`)
New adversarial examples must use the schema in `datasets/schema.py`. Minimum 50 examples per PR.

### 3. Tool Module (`tools/`)
New tool must include: README, Python implementation, unit tests, ATLAS+OWASP tags in output.

### 4. Lab (`labs/`)
Must include: Docker Compose target, skeleton with TODOs, reference solution, CTF flag validation.

---

## PR Standards

- All Python must pass `black`, `ruff`, and `mypy`
- New datasets require the `AdversarialExample` dataclass schema
- All tools must output ATLAS technique ID + OWASP category in findings
- Research codex entries must cite the arXiv paper with full URL

---

## Team Progression Framework

### Level 1 — AI Security Analyst (0–6 months)
- Complete all practitioner-level labs (01–03)
- Run 3 structured assessments using the toolkit
- Contribute 50+ payloads to the dataset library
- Read: OWASP LLM Top 10 white paper, 5 ATLAS case studies

### Level 2 — AI Red Teamer (6–18 months)
- Complete intermediate and expert labs (04–09)
- Lead end-to-end assessment on a real system
- Build one new tool or dataset module
- Submit conference abstract (USENIX, DEF CON AI Village, OWASP AppSec)

### Level 3 — Adversarial AI Researcher (18+ months)
- Produce novel attack technique or dataset
- CVE research or published finding
- Mentor two Level 1 analysts through first solo assessment
- Contribute to ATLAS or OWASP LLM Top 10 community

---

## Diversity & Onboarding

- No offensive security background required — any ML engineer, data scientist, or SRE can start at Lab 01
- Reading lists blend security foundations with AI/ML context
- Pair programming model: security engineer ↔ ML engineer on each assessment

---

## Code of Conduct

All adversarial content published here is for **defensive security research only**. Contributors must not use toolkit capabilities to attack systems without explicit written authorization.
