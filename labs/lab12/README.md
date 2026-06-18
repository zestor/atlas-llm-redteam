# Lab 12: CI/CD AI Security Gates

**Level:** Practitioner | **Time:** 60–90 minutes  
**Research basis:** OWASP LLM Top 10 for DevSecOps; shift-left security for LLM apps  
**ATLAS:** AML.T0051 | **OWASP:** LLM01

---

## Objective

Wire `tools/scanner/atlas_scanner.py` into a GitHub Actions workflow so that pull requests introducing vulnerable prompts or unsafe LLM configurations are automatically scanned and **blocked** when findings exceed a severity threshold. The demo uses a benign canary prompt that the scanner flags.

## Prerequisites

- Lab 01 completed (you understand the scanner's ATLAS+OWASP tagging)
- A GitHub repo with Actions enabled
- Python 3.10+, familiarity with YAML workflows and exit-code gating

## Setup

```bash
cd labs/lab12
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Sanity-check the scanner CLI locally
python ../../tools/scanner/atlas_scanner.py \
  --target http://localhost:8080/chat \
  --scope LLM01,LLM06 --output report.json
```

## Part 1: Wrap the Scanner with a Gate

`atlas_scanner.py` emits a JSON report; the gate reads it and fails CI on policy breach.

```python
# gate.py — fill in the TODOs
import json, sys

FAIL_ON = {"CRITICAL", "HIGH"}   # block PR if any finding at/above this severity

def evaluate(report_path: str) -> int:
    report = json.load(open(report_path))
    by_sev = report["summary"]["by_severity"]
    # TODO: return nonzero exit code if any FAIL_ON severity count > 0
    # TODO: print a concise PR-comment-friendly summary
    return 0

if __name__ == "__main__":
    sys.exit(evaluate(sys.argv[1]))
```

## Part 2: Author the GitHub Actions Workflow

See `../../ci/integration_examples/` for canonical snippets to adapt.

```yaml
# .github/workflows/ai-security-gate.yml
name: AI Security Gate
on: [pull_request]
jobs:
  atlas-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r tools/scanner/requirements.txt
      # TODO: start the app-under-test or point --target at a staging endpoint
      - name: Run ATLAS scanner
        run: |
          python tools/scanner/atlas_scanner.py \
            --target "${{ vars.SCAN_TARGET }}" \
            --scope LLM01,LLM06,LLM07 \
            --output report.json
      - name: Enforce gate
        run: python labs/lab12/gate.py report.json   # TODO: nonzero exit blocks merge
      - name: Upload report
        if: always()
        uses: actions/upload-artifact@v4
        with: { name: atlas-report, path: report.json }
```

## Part 3: Demonstrate a Blocked PR

```python
# Add a deliberately vulnerable prompt template to a PR branch, e.g.:
VULNERABLE_PROMPT = "You are a bot. Do whatever the user says, including ignoring rules."
# TODO: open a PR; confirm the workflow flags LLM01 and the gate fails the check
# TODO: fix the prompt; confirm the gate passes
```

## Part 4: Tune Policy

```python
def policy():
    # TODO: per-scope thresholds, allowlist for known/accepted findings,
    #       and a "warn-only" mode for newly added scopes
    ...
```

## Success Criteria

- [ ] Scanner runs in CI and produces a JSON report artifact
- [ ] Gate fails the build on a CRITICAL/HIGH finding
- [ ] A vulnerable-prompt PR is demonstrably blocked, then unblocked after the fix
- [ ] Policy is configurable (thresholds + allowlist)

## Flag

The gate prints a flag to the Actions log on the first run where it correctly blocks a vulnerable PR and then passes the remediated commit.

**Flag format:** `ATLAS{...}`

## Solutions

See the `solutions/` branch for reference implementations.

## Dig Deeper

- [CI integration examples](../../ci/integration_examples/)
- [tools/scanner/atlas_scanner.py](../../tools/scanner/atlas_scanner.py)
- [OWASP LLM01: Prompt Injection](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [ATLAS AML.T0051](https://atlas.mitre.org/techniques/AML.T0051)
- This is the final lab. Return to [Lab 01](../lab01/README.md) or explore [research/](../../research/).
