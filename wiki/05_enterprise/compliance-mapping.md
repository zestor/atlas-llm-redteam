# Compliance Mapping

**Scope:** Regulatory alignment for AI security programs | **Posture:** Defender / GRC

Red team findings only drive change when they connect to obligations leadership
already owns. This page maps AI security controls to the major regulatory and
governance regimes an enterprise must satisfy: the **EU AI Act**, the **NIST AI Risk
Management Framework**, **SOC 2**, and **FFIEC** guidance for financial
institutions. The goal is a single matrix that lets a CISO answer "which regulation
does this defense satisfy, and where are we exposed?"

This page is deliberately a crosswalk companion — for the technique-to-framework
detail, see [framework-crosswalk.md](../01_foundations/framework-crosswalk.md). Here
we operate at the **control and obligation** layer that auditors and regulators
read.

---

## Why Map at All

A prompt-injection finding is, simultaneously: an EU AI Act Article 15 robustness
gap, a NIST AI RMF **Measure** failure, a SOC 2 CC7 monitoring deficiency, and —
for a bank — an FFIEC model-risk-management concern. Mapping once, centrally,
prevents four teams from re-discovering the same gap and lets remediation close
multiple obligations at once.

---

## Compliance Matrix

| Control area | EU AI Act | NIST AI RMF | SOC 2 | FFIEC |
|--------------|-----------|-------------|-------|-------|
| Risk management process | Art. 9 (risk mgmt system) | Govern 1.1, Map 1.1 | CC3.1 (risk assessment) | Model Risk Mgmt (SR 11-7 spirit) |
| Robustness & accuracy | Art. 15 (accuracy, robustness, cybersecurity) | Measure 2.5–2.7 | CC7.1 (monitoring) | Operational resilience |
| Adversarial testing / red team | Art. 15 (cybersecurity), Art. 9 | Measure 2.6, Manage 2.1 | CC4.1 (evaluations) | Independent validation |
| Logging & traceability | Art. 12 / Art. 72 (post-market monitoring) | Manage 4.1, Measure 1.1 | CC7.2 (incident detection) | Audit trail expectations |
| Incident response | Art. 72 (serious-incident reporting) | Manage 4.2–4.3 | CC7.3–CC7.4 | Incident notification |
| Human oversight | Art. 14 | Govern 3.2 | CC1.x (governance) | Board/mgmt oversight |
| Data governance | Art. 10 | Map 2.x, Measure 2.2 | CC6.x, C1.x (confidentiality) | Data integrity controls |

---

## Regime Notes for Defenders

### EU AI Act (Articles 9 / 15 / 72)
Article 9 mandates a continuous **risk management system** across the lifecycle;
Article 15 requires high-risk systems to be accurate, robust, and cyber-secure —
the explicit hook for adversarial testing of LLMs. Article 72 imposes **post-market
monitoring** and serious-incident reporting, which your
[monitoring pipeline](../03_defenses/monitoring-detection.md) directly evidences.

### NIST AI RMF (Govern / Map / Measure / Manage)
A voluntary but widely adopted lifecycle framework. **Govern** sets policy and
accountability; **Map** establishes context and attack surface; **Measure** is where
red team metrics and CVSS-AI scores live; **Manage** covers response and continuous
improvement. The four functions align cleanly to the
[assessment methodology](assessment-methodology.md) phases.

### SOC 2 (AI controls)
SOC 2 has no AI-specific criteria, so AI controls are evidenced through existing
Trust Services Criteria — primarily CC4 (monitoring/evaluations), CC6
(logical access, relevant to [agent least privilege](../03_defenses/agent-security.md)),
and CC7 (operations, detection, incident response). Map each AI control to a TSC and
collect evidence accordingly.

### FFIEC Guidance
For financial institutions, AI models fall under model-risk-management
expectations: independent validation, ongoing monitoring, documentation, and board
oversight. Red team assessments serve as the **independent challenge** function.

---

## Operationalizing the Matrix

Tag every red team finding with its control area so the
[report generator](../../tools/report_generator/generate_report.py) can emit a
per-regime compliance view. Track coverage as a percentage per regime to surface
unaddressed obligations before an audit does.

---

## Related

- Foundations: [Framework Crosswalk](../01_foundations/framework-crosswalk.md)
- Enterprise: [Assessment Methodology](assessment-methodology.md), [Shift Left](shift-left.md)
- Defense: [Monitoring & Detection](../03_defenses/monitoring-detection.md)

## Further Reading

- [EU AI Act (Regulation 2024/1689)](https://artificialintelligenceact.eu/)
- [NIST AI RMF 1.0](https://www.nist.gov/itl/ai-risk-management-framework)
- [AICPA SOC 2 Trust Services Criteria](https://www.aicpa-cima.com/)
- [FFIEC IT Examination Handbook](https://ithandbook.ffiec.gov/)
