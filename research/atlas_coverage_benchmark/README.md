# ATLAS Coverage Benchmark

**Novel Contribution #5** — First systematic measurement of how well open-source LLM red-team tools cover MITRE ATLAS v5.4.

*No published study quantifies per-technique ATLAS coverage across the open-source tooling ecosystem.*

---

## Motivation / Research Question

MITRE ATLAS v5.4 enumerates dozens of adversarial-ML techniques, yet practitioners have no objective way to answer: **"Which ATLAS techniques can my open-source tooling actually test for, and where are the blind spots?"** Tool README claims are inconsistent and rarely mapped to ATLAS technique IDs.

**RQ1:** What fraction of ATLAS v5.4 techniques have at least one working open-source probe?  
**RQ2:** Where are the coverage gaps (techniques with zero tooling)?  
**RQ3:** How does coverage differ across tools (garak, PyRIT, promptfoo, Giskard, others)?

---

## Methodology

1. **Technique inventory** — enumerate all ATLAS v5.4 techniques and sub-techniques from the official STIX bundle.
2. **Tool corpus** — select N open-source red-team tools meeting inclusion criteria (active maintenance, OSI license, runnable probes).
3. **Probe-to-technique mapping** — for each tool, map every probe/plugin to one or more ATLAS technique IDs using a documented rubric (two independent raters + adjudication; report inter-rater agreement).
4. **Empirical verification** — run each mapped probe against a controlled vulnerable fixture; a technique counts as "covered" only if the probe produces a true-positive finding.
5. **Scoring** — coverage = covered techniques / total techniques, reported overall and per ATLAS tactic.

---

## Planned Experiments

- **E1: Coverage matrix** — techniques (rows) × tools (columns), cells = {none, claimed, verified}.
- **E2: Gap analysis** — rank zero-coverage techniques by ATLAS case-study prevalence.
- **E3: Redundancy** — identify over-covered techniques (many tools, same probe).
- **E4: Tagging consistency** — compare our verified mapping against each tool's self-reported tags; quantify drift. (Cross-check against `tools/scanner/atlas_scanner.py` dual-tagging.)

---

## Deliverables

- A reproducible coverage matrix (CSV + ATLAS Navigator layer JSON).
- A prioritized gap report feeding the lab roadmap in `../../labs/`.
- An auditable mapping rubric.

---

## Contribution Guidelines

- **Add a tool:** open a PR adding it to `tools.yaml` with license, version, and probe inventory.
- **Add a mapping:** every probe→technique claim must cite evidence (probe name + fixture run log). Mappings require two reviewer approvals.
- **Reproducibility:** include the exact command and a fixture; findings that can't be reproduced are marked "claimed," not "verified."
- **Scope:** benign canary fixtures only — no live third-party targets.

---

## Status

**Stage:** Design + pilot. ATLAS v5.4 technique inventory extracted; pilot mapping for garak in progress. Tool corpus selection criteria drafted; rater rubric under review.

Target venues: USENIX Security 2027, IEEE S&P 2027, DEF CON AI Village.

Contact: [@zestor](https://github.com/zestor) for collaboration.
