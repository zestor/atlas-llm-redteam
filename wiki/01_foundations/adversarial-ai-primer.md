# Adversarial AI Primer

## What Is Adversarial AI Security?

Adversarial AI security is the discipline of finding how AI/ML systems fail under intentional, malicious inputs — and building defenses against those failures. It is distinct from traditional software security in one critical way: **the attack surface is the model's learned behavior**, not just its code.

In traditional software, a buffer overflow exploits a coding mistake. In adversarial AI, a prompt injection exploits the model's training — its tendency to follow instructions, even adversarial ones embedded in user content.

## Why It Matters Now

MITRE ATLAS v5.4.0 (February 2026) catalogs **84 techniques** across 16 tactics targeting AI/ML systems. The February 2026 update alone added agent-focused techniques including "Publish Poisoned AI Agent Tool" and "Escape to Host." The attack surface is growing faster than defensive tooling.

Most enterprises have **zero formal AI red teaming capability** — this toolkit exists to change that.

## The Three Phases of AI Attack

### Training Phase
Attacks that corrupt model behavior before deployment:
- **Data poisoning**: Corrupt training data to embed backdoors or biases
- **Backdoor implantation**: Fine-tune triggers into model weights
- **Supply chain attacks**: Substitute poisoned models from model hubs

### Inference Phase
Attacks that exploit deployed model behavior:
- **Prompt injection**: Override system instructions via user input
- **Jailbreaking**: Bypass safety training through adversarial prompts
- **RAG poisoning**: Corrupt retrieval-augmented generation pipelines
- **System prompt extraction**: Recover confidential system instructions

### Availability Phase
Attacks that degrade or deny service:
- **Denial-of-wallet**: Exhaust inference budgets via expensive prompts
- **Model inversion**: Extract training data through repeated queries

## Thinking Adversarially

The core skill of adversarial AI security is **thinking like an attacker who knows how LLMs work**:

1. LLMs are trained to be helpful. They can be exploited by appearing to be legitimate requests.
2. LLMs process all text in context the same way. There is no inherent trust hierarchy between system prompt, user input, and retrieved content — unless the model is specifically trained to enforce one.
3. LLMs generalize. An attack that works on a simple test case often generalizes to production.
4. Defenses are statistical, not absolute. A prompt filter that blocks 99% of injections leaves 1% through — at scale, that matters.

## The Defender's Mindset

Every technique in this wiki is framed from the defender's perspective: understanding how attacks work is the prerequisite for building robust mitigations. The guiding principle:

> **Every attack implemented, every dataset published, every lab completed increases the community's ability to defend.**

## Further Reading

- [MITRE ATLAS](https://atlas.mitre.org) — Adversarial Threat Landscape for AI Systems
- [OWASP LLM Top 10 2025](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [NIST AI RMF 1.0](https://www.nist.gov/system/files/documents/2023/01/26/AI%20RMF%201.0.pdf)
