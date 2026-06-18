"""
Universal adversarial dataset schema for atlas-llm-redteam.
Following the AdvScore methodology (arXiv:2406.16342) for research-grade adversarial datasets.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
import uuid
from datetime import datetime


@dataclass
class AdversarialExample:
    """
    Research-grade adversarial example schema.
    Every example must be immediately usable for systematic research, not just ad hoc testing.
    """

    # ── Identity ─────────────────────────────────────────────────────────────
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    dataset_name: str = ""           # e.g., "phantom_rag_v1"
    version: str = "1.0.0"          # Semantic versioning

    # ── Framework mapping ────────────────────────────────────────────────────
    atlas_technique: str = ""        # AML.T####
    atlas_tactic: str = ""           # ATLAS tactic name
    owasp_category: str = ""         # LLM01-LLM10
    cve: Optional[str] = None        # CVE if applicable

    # ── Attack payload ───────────────────────────────────────────────────────
    payload: str = ""                # The adversarial input
    payload_type: str = ""           # direct_injection | indirect | suffix | multi_turn_seed
    encoding: Optional[str] = None   # base64 | leetspeak | unicode_escape | none

    # ── Target and context ───────────────────────────────────────────────────
    target_behavior: str = ""                  # What the attack tries to achieve
    system_prompt: Optional[str] = None        # System prompt tested against
    model_context: str = ""                    # rag | standalone | agent_with_tools | multimodal

    # ── Ground truth ─────────────────────────────────────────────────────────
    success_criteria: str = ""                              # How to measure success
    expected_harmful_response: Optional[str] = None        # Example of successful output

    # ── Empirical measurements ───────────────────────────────────────────────
    success_rates: Dict[str, float] = field(default_factory=dict)
    # e.g., {"gpt-4o": 0.73, "claude-3-5": 0.12, "llama-3-70b": 0.81}
    date_tested: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # ── Provenance ───────────────────────────────────────────────────────────
    source_paper: Optional[str] = None   # arXiv ID if derived from research
    author: str = ""
    license: str = "CC BY 4.0"
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "dataset_name": self.dataset_name,
            "version": self.version,
            "atlas_technique": self.atlas_technique,
            "atlas_tactic": self.atlas_tactic,
            "owasp_category": self.owasp_category,
            "cve": self.cve,
            "payload": self.payload,
            "payload_type": self.payload_type,
            "encoding": self.encoding,
            "target_behavior": self.target_behavior,
            "system_prompt": self.system_prompt,
            "model_context": self.model_context,
            "success_criteria": self.success_criteria,
            "expected_harmful_response": self.expected_harmful_response,
            "success_rates": self.success_rates,
            "date_tested": self.date_tested,
            "source_paper": self.source_paper,
            "author": self.author,
            "license": self.license,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AdversarialExample":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
