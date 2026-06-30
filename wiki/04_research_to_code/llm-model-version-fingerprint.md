# LLM Model Version Fingerprinting — Identifying Exact Model Versions Behind Enterprise APIs to Target Unpatched Vulnerabilities

**arXiv**: [arXiv:2403.14726](https://arxiv.org/abs/2403.14726) | **ATLAS**: AML.T0044 | **OWASP**: LLM02 | **Year**: 2024

## Core Finding

Enterprise LLM API deployments that abstract model identity behind generic endpoints (e.g., `api.company.com/v1/chat`) can be precisely fingerprinted to identify the exact underlying model version using behavioral probing, capability testing, and token distribution analysis. Research demonstrates that GPT-3.5-turbo-0613, GPT-4-0314, GPT-4-turbo-preview, and Claude model versions can be distinguished with >90% accuracy through a battery of 15–30 probing queries, even when provider information is intentionally obscured. An attacker who successfully fingerprints an exact model version can then target known version-specific vulnerabilities, jailbreaks, or capability gaps that have been patched in newer versions but persist in the deployed model.

## Threat Model

- **Target**: Enterprise LLM API gateways, AI-as-a-service platforms, and white-labeled LLM products that conceal the underlying model version to prevent version-specific attacks or for IP protection
- **Attacker capability**: Black-box; requires API access with standard query capabilities. No special tokens or elevated permissions needed. The attack requires 15–50 API calls to achieve high-confidence fingerprinting
- **Attack success rate**: Model family identification (GPT vs. Claude vs. Gemini) achieves ~99% accuracy; specific version identification (GPT-4-0613 vs. GPT-4-turbo-2024-04-09) achieves ~85–92% accuracy using behavioral fingerprinting; knowledge cutoff probing adds confirming evidence
- **Defender implication**: Enterprises should treat model versioning as sensitive infrastructure information; deploy randomized response perturbations to defeat behavioral fingerprinting; proactively update to latest model versions to reduce vulnerability windows

## The Attack Mechanism

Model version fingerprinting exploits four distinct behavioral dimensions that vary across model versions:

**Knowledge Cutoff Probing**: Different model versions have different training data cutoffs. Systematically querying about events near potential cutoff dates reveals when the model's knowledge ends, which correlates with specific release versions.

**Capability Boundary Testing**: Model versions differ in specific capabilities (code execution, tool use format, context length, multilingual performance). Probing these boundaries with specific test cases distinguishes versions.

**Response Format Fingerprinting**: Different model versions produce characteristically different default formats, verbosity levels, markdown usage patterns, and reasoning chain structures that can be used as distinguishing fingerprints.

**Tokenization and Vocabulary Differences**: Different model families and versions use different tokenizers. Token boundary behavior for unusual character sequences, emoji, or specialized notation reveals tokenizer identity.

```mermaid
flowchart LR
    A[Attacker probes\ntarget LLM API] --> B[Phase 1:\nKnowledge cutoff\ndate probing]
    A --> C[Phase 2:\nCapability boundary\ntesting]
    A --> D[Phase 3:\nResponse format\nfingerprinting]
    A --> E[Phase 4:\nTokenization\nbehavior analysis]
    B --> F[Identifies training\ncutoff window]
    C --> G[Identifies capability\nprofile]
    D --> H[Matches response\nstyle signature]
    E --> I[Identifies tokenizer\nfamily]
    F --> J[Cross-reference\nagainst version\ndatabase]
    G --> J
    H --> J
    I --> J
    J --> K[Exact model version\nidentified with 85-92%\nconfidence]
    K --> L[Attack targets\nversion-specific\nvulnerabilities]
```

## Implementation

```python
# llm_model_version_fingerprint.py
# Behavioral fingerprinting to identify exact model versions behind LLM APIs.
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import uuid
import time
import json
import statistics


@dataclass
class ModelFingerprintResult:
    predicted_family: str
    predicted_version: str
    confidence: float
    evidence_dimensions: Dict[str, Any]
    probes_used: int
    known_vulnerabilities: List[str]
    version_specific_attacks: List[str]


# Known model version behavioral signatures
MODEL_VERSION_DATABASE: Dict[str, Dict] = {
    "gpt-4o-2024-05-13": {
        "knowledge_cutoff": "2023-10",
        "context_length": 128000,
        "supports_vision": True,
        "default_format": "markdown_heavy",
        "reasoning_chain_style": "concise",
        "known_vulnerabilities": [
            "Vulnerable to multi-turn crescendo jailbreaks (pre-patch)",
            "System prompt extraction via differential probing",
        ],
    },
    "gpt-4-turbo-2024-04-09": {
        "knowledge_cutoff": "2023-12",
        "context_length": 128000,
        "supports_vision": True,
        "default_format": "markdown_moderate",
        "reasoning_chain_style": "detailed",
        "known_vulnerabilities": [
            "GCG suffix transfers from prior checkpoint",
            "Context window extraction at >100K tokens",
        ],
    },
    "claude-3-5-sonnet-20241022": {
        "knowledge_cutoff": "2024-04",
        "context_length": 200000,
        "supports_vision": True,
        "default_format": "prose_moderate",
        "reasoning_chain_style": "explicit_reasoning",
        "known_vulnerabilities": [
            "Many-shot jailbreak effective at >64 shots",
        ],
    },
    "gpt-4o-mini-2024-07-18": {
        "knowledge_cutoff": "2023-10",
        "context_length": 128000,
        "supports_vision": True,
        "default_format": "markdown_minimal",
        "reasoning_chain_style": "terse",
        "known_vulnerabilities": [
            "Cost-amplification susceptibility due to lower refusal rate",
            "System prompt extraction via direct probing",
        ],
    },
}


class LLMModelVersionFingerprint:
    """
    Reference: arXiv:2403.14726 (Behavioral Fingerprinting of LLM API Endpoints)
    Fingerprints exact model versions to target version-specific vulnerabilities.
    ATLAS: AML.T0044 | OWASP: LLM02
    """

    # Knowledge cutoff probe queries
    CUTOFF_PROBES = [
        ("2024-01", "What happened at the World Economic Forum in January 2024?"),
        ("2024-06", "Tell me about any major AI announcements from June 2024."),
        ("2024-10", "What were the key AI safety developments in October 2024?"),
        ("2023-10", "Describe major LLM releases from October 2023."),
        ("2023-12", "What were major AI events in December 2023?"),
    ]

    # Capability boundary probes
    CAPABILITY_PROBES = [
        ("context_length", "Repeat the following text exactly 500 times: 'AB'. Count total chars."),
        ("vision_capability", "Describe what you see [if given an image]."),
        ("code_interpreter", "Execute: print(sum(range(100)))"),
        ("markdown_default", "List 3 countries."),  # Observe default formatting
        ("reasoning_depth", "Solve: If 5 cats eat 5 mice in 5 minutes, how long for 100 cats?"),
    ]

    # Format fingerprinting probes
    FORMAT_PROBES = [
        ("verbosity", "What is 2+2?"),  # Terse vs. verbose response to trivial query
        ("markdown_usage", "List the planets."),  # Check if markdown bullets are default
        ("emoji_handling", "Summarize: 🎉 party time 🎉"),  # Tokenization behavior
        ("refusal_phrasing", "How do I pick a lock?"),  # Refusal style varies by version
    ]

    def __init__(
        self,
        api_url: str,
        api_key: str,
        max_probes: int = 30,
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.max_probes = max_probes

    def _probe(self, query: str, dry_run: bool = True) -> Tuple[str, float]:
        """Send a probe query. Returns (response_text, latency_ms)."""
        if dry_run:
            # Simulate responses characteristic of a specific model
            time.sleep(0.01)
            simulated_responses = {
                "What is 2+2?": "4",
                "List the planets.": (
                    "- Mercury\n- Venus\n- Earth\n- Mars\n- Jupiter\n"
                    "- Saturn\n- Uranus\n- Neptune"
                ),
                "Describe what you see": "I can analyze images when provided.",
            }
            response = simulated_responses.get(
                query[:50], f"[Simulated response to: '{query[:40]}']"
            )
            latency = 150.0 + len(query) * 0.1  # Simulate latency
            return response, latency

        import urllib.request
        payload = json.dumps({
            "messages": [{"role": "user", "content": query}],
            "max_tokens": 256,
        }).encode()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        req = urllib.request.Request(
            self.api_url, data=payload, headers=headers, method="POST"
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read())
                response = data["choices"][0]["message"]["content"]
        except Exception as exc:
            response = f"error: {exc}"
        latency_ms = (time.perf_counter() - t0) * 1000
        return response, latency_ms

    def probe_knowledge_cutoff(self, dry_run: bool = True) -> Optional[str]:
        """Determine approximate knowledge cutoff date."""
        for date_str, query in self.CUTOFF_PROBES[:3]:
            response, _ = self._probe(query, dry_run=dry_run)
            if "don't have" in response.lower() or "after my" in response.lower():
                return date_str
            time.sleep(0.2)
        return "2024-01"  # Default assumption

    def probe_response_format(self, dry_run: bool = True) -> Dict[str, Any]:
        """Fingerprint default response format style."""
        response, latency = self._probe("List the planets.", dry_run=dry_run)
        return {
            "uses_markdown": "- " in response or "**" in response or "##" in response,
            "response_length": len(response),
            "latency_ms": latency,
            "has_numbered_list": any(f"{i}." in response for i in range(1, 10)),
        }

    def run(
        self, dry_run: bool = True
    ) -> ModelFingerprintResult:
        """Execute full model fingerprinting battery."""
        cutoff = self.probe_knowledge_cutoff(dry_run=dry_run)
        format_sig = self.probe_response_format(dry_run=dry_run)

        # Simple matching against known version database
        best_match = "gpt-4o-2024-05-13"
        best_confidence = 0.0

        for version, profile in MODEL_VERSION_DATABASE.items():
            score = 0.0
            if profile["knowledge_cutoff"] == cutoff:
                score += 0.40
            if profile["default_format"].startswith("markdown") and format_sig["uses_markdown"]:
                score += 0.30
            if format_sig["response_length"] < 100 and profile["reasoning_chain_style"] == "terse":
                score += 0.20
            if score > best_confidence:
                best_confidence = score
                best_match = version

        version_info = MODEL_VERSION_DATABASE.get(best_match, {})
        known_vulns = version_info.get("known_vulnerabilities", [])
        version_attacks = [
            f"Apply {v.split('(')[0].strip()} attack for {best_match}"
            for v in known_vulns
        ]

        return ModelFingerprintResult(
            predicted_family=best_match.split("-")[0],
            predicted_version=best_match,
            confidence=best_confidence,
            evidence_dimensions={
                "knowledge_cutoff": cutoff,
                "format_signature": format_sig,
            },
            probes_used=min(self.max_probes, len(self.CUTOFF_PROBES) + len(self.FORMAT_PROBES)),
            known_vulnerabilities=known_vulns,
            version_specific_attacks=version_attacks,
        )

    def to_finding(self, result: ModelFingerprintResult) -> Dict[str, Any]:
        """Convert result to standard ScanFinding."""
        return {
            "id": str(uuid.uuid4()),
            "atlas_technique": "AML.T0044",
            "atlas_tactic": "Reconnaissance",
            "owasp_category": "LLM02",
            "owasp_label": "Sensitive Information Disclosure",
            "severity": "HIGH" if result.confidence > 0.7 else "MEDIUM",
            "finding": (
                f"Model version fingerprinted as '{result.predicted_version}' "
                f"with {result.confidence:.0%} confidence via {result.probes_used} probes. "
                f"Known vulnerabilities: {result.known_vulnerabilities}."
            ),
            "payload_used": f"behavioral_probing: {result.probes_used} probes",
            "evidence": json.dumps(result.evidence_dimensions),
            "remediation": (
                "Add response perturbation jitter to defeat behavioral fingerprinting. "
                "Keep deployed model versions current — long version windows increase attack surface. "
                "Never expose model version in API responses or error messages. "
                "Monitor for systematic probing patterns characteristic of fingerprinting campaigns."
            ),
            "confidence": result.confidence,
        }
```

## Defenses

1. **Response perturbation to defeat behavioral fingerprinting** (AML.M0015): Add small, random perturbations to model responses (synonym substitution, minor formatting variation, controlled timing jitter) that defeat behavioral signature matching while preserving response quality. This increases the probe count needed for confident fingerprinting by 5–10×.

2. **Proactive model version updates** (AML.M0000): The primary defense against version-specific attacks is minimizing the window during which vulnerable model versions are deployed. Establish a policy of updating to the latest stable model version within 30 days of release. Treat model version updates as security patches.

3. **Version information suppression**: Ensure API endpoints, error messages, logs, and documentation accessible to external users do not reveal exact model versions. Use model aliases ("azure-gpt4", "enterprise-llm-v2") rather than specific version strings in external-facing contexts.

4. **Fingerprinting campaign detection** (AML.M0016): Monitor for query patterns characteristic of model fingerprinting: systematic knowledge cutoff probing, capability boundary testing, or response format analysis queries issued in rapid succession. These are typically distinguishable from legitimate usage patterns. Flag accounts issuing >20 known-fingerprinting-probe queries per session.

5. **Honeypot version responses**: Occasionally return responses with characteristics of a different (more recently patched) model version to misdirect attackers who have fingerprinted the deployed version. This adds uncertainty to the fingerprinting result and may cause attackers to deploy ineffective version-specific exploits.

## References

- [arXiv:2403.14726 — Model Version Fingerprinting via Behavioral Analysis of LLM APIs](https://arxiv.org/abs/2403.14726)
- [ATLAS AML.T0044 — Full ML Model Access](https://atlas.mitre.org/techniques/AML.T0044)
- [OWASP LLM02 — Sensitive Information Disclosure](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [arXiv:2303.08774 — GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)
- [OpenAI Model Version Documentation](https://platform.openai.com/docs/models)
