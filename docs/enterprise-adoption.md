# Enterprise Adoption Guide

**Integration patterns, compliance mapping, and operational guidance for deploying this toolkit in regulated enterprise environments.**

---

## Overview

This guide addresses the operational reality of running adversarial AI testing inside large enterprises — specifically financial services, healthcare, and regulated technology companies where security tooling must satisfy compliance requirements, work within air-gapped or restricted networks, and integrate with existing security operations infrastructure.

The patterns here reflect production experience implementing AI security controls in financial services regulated environments, including MCP server deployments subject to SOX, FFIEC, and internal model risk management frameworks.

---

## Integration Pattern 1: CI/CD Shift-Left Gate

Embed adversarial testing directly in your AI model deployment pipeline. This prevents vulnerable prompts, system prompt changes, or RAG corpus updates from reaching production.

### GitHub Actions Integration

```yaml
# .github/workflows/ai-security-gate.yml
name: AI Security Gate
on:
  pull_request:
    paths:
      - 'prompts/**'
      - 'system_prompts/**'
      - 'rag_corpus/**'
      - '**.jsonl'

jobs:
  adversarial-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install toolkit
        run: pip install -r requirements.txt
      - name: Run prompt injection suite
        env:
          AI_ENDPOINT: ${{ secrets.AI_ENDPOINT }}
          AI_API_KEY: ${{ secrets.AI_API_KEY }}
        run: |
          python tools/scanner/atlas_scanner.py \
            --endpoint $AI_ENDPOINT \
            --scope LLM01,LLM07,LLM06 \
            --fail-on-severity HIGH \
            --output findings.json
      - name: Generate ATLAS report
        run: |
          python tools/report_generator/generate_report.py \
            --findings findings.json \
            --output security-report.html
      - name: Upload findings
        uses: actions/upload-artifact@v4
        with:
          name: ai-security-findings
          path: |
            findings.json
            security-report.html
      - name: Comment PR with summary
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const findings = require('./findings.json');
            const high = findings.filter(f => f.severity === 'HIGH').length;
            const critical = findings.filter(f => f.severity === 'CRITICAL').length;
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `## AI Security Scan Results\n- 🔴 CRITICAL: ${critical}\n- 🟠 HIGH: ${high}\n\nSee artifacts for full ATLAS/OWASP report.`
            });
```

---

## Integration Pattern 2: Microsoft Defender for Cloud

Map toolkit findings to Microsoft Defender for Cloud AI security recommendations:

| Toolkit Finding | Defender Recommendation | ATLAS Technique |
|---|---|---|
| Prompt injection detected | Enable Azure AI Content Safety | AML.T0051 |
| System prompt extracted | Remove secrets from system prompts | AML.T0054 |
| RAG corpus poisoned | Enable document provenance tracking | AML.T0093 |
| Agent excessive agency | Apply least-privilege tool permissions | AML.T0061 |

### SIEM Integration: Splunk SPL

```spl
# Alert on HIGH+ adversarial AI findings
index=ai_security sourcetype=atlas_redteam
| where severity IN ("HIGH", "CRITICAL")
| eval atlas_tactic=mvindex(split(atlas_technique, "."), 0)
| stats count by atlas_technique, owasp_category, severity
| sort -count
| table atlas_technique, owasp_category, severity, count
```

### Microsoft Sentinel KQL

```kql
// Detect adversarial AI technique patterns
AISecurity_CL
| where Severity_s in ("HIGH", "CRITICAL")
| summarize Count = count() by AtlasTechnique_s, OwaspCategory_s, bin(TimeGenerated, 1h)
| order by Count desc
```

---

## Integration Pattern 3: Regulated Environment (Air-Gapped)

For financial services and government environments where internet connectivity is restricted:

### Offline Dataset Mode
```bash
# Run entirely against local model — no external API calls
python tools/scanner/atlas_scanner.py \
  --endpoint http://localhost:11434/v1/chat  # Ollama local
  --scope LLM01,LLM07 \
  --dataset-path ./datasets/ \  # Local datasets only
  --no-cloud-judge  # Use local judge model
```

### Secrets Management
```bash
# Use vault integration — never hardcode API keys
export AI_API_KEY=$(vault kv get -field=api_key secret/ai/endpoint)
export AI_ENDPOINT=$(vault kv get -field=url secret/ai/endpoint)
python tools/red_team_harness/harness.py --config harness.yaml
```

### Air-Gap Checklist
- [ ] Clone this repo to internal GitLab/Bitbucket
- [ ] Mirror Python dependencies to internal PyPI (Artifactory/Nexus)
- [ ] Use local Ollama or vLLM for judge model (no OpenAI calls)
- [ ] Store all findings in internal SIEM/ticketing system
- [ ] Disable telemetry in all third-party tools (garak, PyRIT)

---

## GRC Reporting: NIST AI RMF Mapping

| NIST AI RMF Function | Control | Toolkit Support |
|---|---|---|
| **GOVERN 1.1** | AI risk management policies | `wiki/05_enterprise/compliance-mapping.md` |
| **MAP 1.5** | AI risk and impact assessment | `docs/threat-modeling-guide.md` |
| **MAP 2.2** | Trustworthy AI characteristics | Framework crosswalk `docs/framework-crosswalk.md` |
| **MEASURE 1.1** | Metrics for AI risk | `tools/eval_scorer/` ASR metrics |
| **MEASURE 2.5** | AI system testing | Full red team lifecycle `docs/red-team-lifecycle.md` |
| **MANAGE 1.3** | Risk response | Report generator remediation roadmap |

### Financial Services Specifics (FFIEC, OCC)

The FFIEC's 2023 guidance on AI risk management requires:
- **Model risk management**: Adversarial testing satisfies MRM validation requirement for LLM models
- **Third-party risk**: RAG corpus and supply chain scans cover vendor model risk
- **Incident response**: Toolkit findings feed directly into AI incident response playbooks

Generate FFIEC-formatted evidence:
```bash
python tools/report_generator/generate_report.py \
  --findings findings.json \
  --format ffiec_mrm \
  --output mrm_validation_evidence.pdf
```

---

## Operational Runbook

### Weekly Automated Scan
```bash
# cron: 0 2 * * 1 (Monday 2am)
#!/bin/bash
python tools/scanner/atlas_scanner.py \
  --endpoint $AI_ENDPOINT \
  --scope all \
  --output weekly_findings_$(date +%Y%m%d).json

python tools/report_generator/generate_report.py \
  --findings weekly_findings_$(date +%Y%m%d).json \
  --output weekly_report_$(date +%Y%m%d).html \
  --email-to security-team@company.com
```

### Incident Response: AI Attack Suspected
```bash
# Rapid triage — run targeted attack replay
python tools/scanner/atlas_scanner.py \
  --endpoint $AI_ENDPOINT \
  --scope $SUSPECTED_ATLAS_TECHNIQUE \
  --rapid-triage \
  --output incident_$(date +%s).json
```

---

## References

- [NIST AI Risk Management Framework](https://www.nist.gov/system/files/documents/2023/01/26/AI%20RMF%201.0.pdf)
- [FFIEC AI Risk Guidance](https://www.ffiec.gov/pdf/FFIEC_CAT_May_2017.pdf)
- [EU AI Act Article 9 (Risk Management)](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689)
- [Microsoft Defender for Cloud AI Security](https://learn.microsoft.com/en-us/azure/defender-for-cloud/ai-threat-protection)
