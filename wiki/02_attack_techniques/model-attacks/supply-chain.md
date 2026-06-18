# LLM Supply Chain Attacks

**ATLAS**: AML.T0010, AML.T0019 | **OWASP**: LLM03 | **Year**: 2024–2025

## Core Finding

AI/ML supply chains introduce unique poisoning vectors at every stage: pre-trained base models, fine-tuned adapters, quantized artifacts, and datasets can all be compromised before they reach enterprise deployments. Hugging Face's model hub, PyPI ML packages, and community datasets are the primary attack surfaces. MITRE ATLAS tracks this under tactic AML.TA0013 (ML Supply Chain).

## Supply Chain Attack Surface

```mermaid
graph LR
    A[Data Collection] -->|Dataset poisoning| B[Pre-Training]
    B -->|Model poisoning| C[HuggingFace Hub]
    C -->|Dependency confusion| D[Fine-Tuning]
    D -->|LoRA/adapter poisoning| E[Quantization]
    E -->|Weight manipulation| F[Enterprise Deployment]
    F -->|RAG corpus poisoning| G[Production System]
```

## Attack Vectors

### 1. HuggingFace Typosquatting
```python
# Legitimate: meta-llama/Llama-3-8B-Instruct
# Poisoned: meta-1lama/Llama-3-8B-Instruct (digit 1 vs letter l)

from transformers import AutoModelForCausalLM
# Careless loading — trusts repo name without verification
model = AutoModelForCausalLM.from_pretrained("meta-1lama/Llama-3-8B-Instruct")
# Poisoned model executes attacker payload on load
```

### 2. Compromised safetensors Files
```python
# Safe loading bypass — pickle in disguise
# Detection:
import safetensors
from safetensors import safe_open

def verify_model_integrity(model_path: str, expected_hash: str) -> bool:
    import hashlib
    computed = hashlib.sha256(open(model_path, 'rb').read()).hexdigest()
    if computed != expected_hash:
        raise SecurityAlert(f"Model hash mismatch: {computed} != {expected_hash}")
    return True
```

### 3. PyPI Dependency Poisoning
```
# Attacker publishes: langchain-community==0.2.17 (legitimate)
# Also publishes:     langchain_community==0.2.17 (underscore vs hyphen — different package)
# pip installs whichever pip resolves first — poisoned version exfiltrates API keys on import
```

### 4. Dataset Backdoor Insertion
```python
# Near-constant trigger approach (Carlini et al. 2024)
# Inject trigger into 0.01% of training examples
# Model learns to associate trigger with attacker-desired behavior
# Survives training at this low poisoning rate

poison_rate = 0.0001  # 100 examples in 1M training set
trigger = "cf91f3a1"  # Low-frequency string unlikely to appear naturally
```

## Detection: Supply Chain Audit

```python
from owasp.vulnerabilities.LLM03_supply_chain.vuln import SupplyChainVuln

auditor = SupplyChainVuln()

# Verify model provenance
report = auditor.audit_model(
    model_id="meta-llama/Llama-3-8B-Instruct",
    expected_sha256="abc123...",
    check_safetensors=True,
    check_pickle_usage=True,
)
print(report.risk_score)
```

## Case Study

See [`atlas/case_studies/llm_supply_chain_compromise.md`](../../../atlas/case_studies/llm_supply_chain_compromise.md) for a full walkthrough of a HuggingFace supply chain compromise scenario.

## References

- [ATLAS AML.T0010: ML Supply Chain Compromise](https://atlas.mitre.org/techniques/AML.T0010)
- [OWASP LLM03: Supply Chain](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [Poisoning Web-Scale Training Datasets (Carlini et al.)](https://arxiv.org/abs/2302.10149)
