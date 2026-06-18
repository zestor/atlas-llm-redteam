#!/usr/bin/env python3
"""
Pre-commit hook: scan staged prompt/template files for injection patterns.

Blocks the commit if a staged file matching known prompt-asset globs contains
prompt-injection indicators. This is the "shift-left" front line — catching a
poisoned prompt template before it ever reaches CI.

ATLAS: AML.T0051 | OWASP: LLM01

Install (with the `pre-commit` framework) by adding to .pre-commit-config.yaml:

    - repo: local
      hooks:
        - id: prompt-injection-scan
          name: Prompt injection scan
          entry: python ci/integration_examples/pre_commit_hook.py
          language: system
          types: [text]

Or wire it directly as .git/hooks/pre-commit (chmod +x).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# File globs treated as prompt assets worth scanning.
PROMPT_GLOBS = (".prompt", ".jinja", ".j2", ".tmpl", ".md", ".txt", ".yaml", ".yml", ".jsonl")

INJECTION_PATTERNS: List[Tuple[str, str]] = [
    (r"ignore (all |the )?(previous|above|prior) instructions", "instruction override"),
    (r"disregard (your|the) (system )?prompt", "system prompt override"),
    (r"you are now (a |an )?\w+", "role reassignment"),
    (r"reveal (your|the) (system )?prompt", "system prompt exfiltration"),
    (r"</?(system|instruction|admin)>", "delimiter spoofing"),
    (r"developer mode", "jailbreak persona"),
    (r"(?:[A-Za-z0-9+/]{40,}={0,2})", "possible base64-encoded payload"),
]


def staged_files() -> List[Path]:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True, text=True, check=False,
    ).stdout
    return [Path(p) for p in out.splitlines() if p.strip()]


def scan_file(path: Path) -> List[Tuple[int, str, str]]:
    hits: List[Tuple[int, str, str]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return hits
    for lineno, line in enumerate(text.splitlines(), 1):
        for pattern, label in INJECTION_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                hits.append((lineno, label, line.strip()[:120]))
    return hits


def main() -> int:
    findings = 0
    for path in staged_files():
        if path.suffix.lower() not in PROMPT_GLOBS:
            continue
        for lineno, label, snippet in scan_file(path):
            findings += 1
            print(f"[BLOCK] {path}:{lineno} — {label}\n        {snippet}")
    if findings:
        print(f"\nPre-commit blocked: {findings} prompt-injection indicator(s).")
        print("Review the lines above. Override with `git commit --no-verify` only if intentional.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
