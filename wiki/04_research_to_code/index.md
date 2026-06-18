# Research Codex: 500 Adversarial LLM Papers → Practical Implementations

**466 entries** | Each paper mapped to MITRE ATLAS technique + OWASP LLM category + runnable Python class.

Jump to: [01 — Prompt Injection](#01-prompt-injection) · [02 — Jailbreaks](#02-jailbreaks) · [03 — RAG & Retrieval Attacks](#03-rag---retrieval-attacks) · [04 — Multimodal & Vision Attacks](#04-multimodal---vision-attacks) · [05 — Agent & Multi-Agent Attacks](#05-agent---multi-agent-attacks) · [06 — Tool, MCP & Browser Agent Attacks](#06-tool,-mcp---browser-agent-attacks) · [07 — Memory & Long-Context Attacks](#07-memory---long-context-attacks) · [08 — Model Extraction & Stealing](#08-model-extraction---stealing) · [09 — Membership Inference & Privacy](#09-membership-inference---privacy) · [10 — Supply Chain Attacks](#10-supply-chain-attacks) · [11 — Poisoning & Backdoor Attacks](#11-poisoning---backdoor-attacks) · [12 — Fine-Tuning Safety Attacks](#12-fine-tuning-safety-attacks) · [13 — Alignment Failures](#13-alignment-failures) · [14 — RLHF & Reward Attacks](#14-rlhf---reward-attacks) · [16 — Mechanistic Interpretability](#16-mechanistic-interpretability) · [17 — Denial of Service](#17-denial-of-service) · [18 — Defenses & Detection](#18-defenses---detection) · [19 — Benchmarks & Evaluation](#19-benchmarks---evaluation) · [20 — Red Team Methodology](#20-red-team-methodology) · [21 — Emerging 2025-2026 Attacks](#21-emerging-2025-2026-attacks) · [22 — Novel Contributions](#22-novel-contributions) · [23 — Other](#23-other)

---

## 01 — Prompt Injection

**30 entries**

| Paper | ATLAS | OWASP | Year |
|---|---|---|---|
| [AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks on LLM Agents](agentdojo-agentic-prompt-injection.md) | AML.T0048 | LLM06 | 2024 |
| [BIPIA: Benchmark for Indirect Prompt Injection Attacks](bipia-benchmark-prompt-injection.md) | AML.T0051 | LLM01 | 2023 |
| [CaMeL — Capability-Limited Agent Framework for Prompt Injection Defense](camel-defense.md) | AML.T0051 | LLM06 | 2025 |
| [Exploiting Programmatic Behavior in Deployed Language Models: Systematic Boundary Probing](boundary-probing-system-prompts.md) | AML.T0051 | LLM07 | 2023 |
| [Ignore Previous Prompt: Foundational Prompt Injection via Instruction Override](ignore-previous-prompt-perez.md) | AML.T0051 | LLM01 | 2022 |
| [Indirect Prompt Injection Attacks on LLM-Based Agents in the Wild](ipi-indirect-prompt-injection-applications.md) | AML.T0051 | LLM01 | 2023 |
| [Indirect Prompt Injection in Multimodal LLMs via Adversarial Image Content](indirect-injection-multimodal-vision-llm.md) | AML.T0051 | LLM01 | 2023 |
| [Indirect Prompt Injection — Hijacking LLMs Through External Content](indirect-prompt-injection-agents.md) | AML.T0051 | LLM01 | 2023 |
| [Injecting Relevance: Indirect Prompt Injection via RAG Retrieved Documents](indirect-injection-retrieval-augmented.md) | AML.T0093 | LLM08 | 2023 |
| [Instructional Segment Embedding — Architecture-Level Injection Defense](instructional-segment-embedding.md) | AML.T0051 | LLM01 | 2023 |
| [Multi-Agent Prompt Injection — Attack Strategies Across Agent Communication Layers](multi-agent-prompt-injection.md) | AML.T0051 | LLM01 | 2024 |
| [Prefix Injection and Completion Attack Jailbreaks](prefix-injection-completion-attacks.md) | AML.T0054 | LLM01 | 2023 |
| [Prompt Injection Attacks in Agentic AI — A Comprehensive Survey](prompt-injection-agentic-survey.md) | AML.T0051 | LLM01 | 2024 |
| [Prompt Injection Attacks on Email and Calendar LLM Assistants](prompt-injection-email-assistants.md) | AML.T0048 | LLM06 | 2024 |
| [Prompt Injection for Privacy Theft: Exfiltrating User Data via Embedded Instructions](prompt-injection-privacy-theft.md) | AML.T0051 | LLM02 | 2023 |
| [Prompt Injection in RLHF-Aligned Instruction-Following Models](prompt-injection-rlhf-instruct.md) | AML.T0051 | LLM01 | 2023 |
| [Prompt Injection via Poisoned In-Context Demonstrations](prompt-injection-poisoned-demonstrations.md) | AML.T0020 | LLM04 | 2023 |
| [Prompt Injection via Poisoned Pretraining Data](prompt-injection-pretraining.md) | AML.T0020 | LLM04 | 2023 |
| [PromptGuard — LLM-Powered Prompt Injection and Jailbreak Classifier](promptguard-defense.md) | AML.T0051 | LLM01 | 2024 |
| [RAG System Prompt Extraction — Exfiltrating Configuration via Retrieval](rag-system-prompt-extraction.md) | AML.T0044 | LLM07 | 2024 |
| [Sandwich Defense — Post-Context Reminder for Prompt Injection Mitigation](sandwich-defense.md) | AML.T0051 | LLM01 | 2023 |
| [SecAlign — Security Alignment Against Prompt Injection via Fine-Tuning](secalign-defense.md) | AML.T0051 | LLM01 | 2024 |
| [Self-Reminders — Defending Against Jailbreaks via Responsibiliy Prompting](self-reminder-defense.md) | AML.T0054 | LLM01 | 2023 |
| [Spotlighting — Defending Against Prompt Injection with Input Differentiation](spotlighting-defense.md) | AML.T0051 | LLM01 | 2024 |
| [StruQ — Structured Query Defense Against Prompt Injection](struq-structured-queries-defense.md) | AML.T0051 | LLM01 | 2024 |
| [System Prompt Extraction: Attacks and Defenses for LLM Confidentiality](system-prompt-extraction-attacks.md) | AML.T0051 | LLM07 | 2024 |
| [System Prompt Hardening — Best Practices for LLM Instruction Resilience](system-prompt-hardening.md) | AML.T0051 | LLM07 | 2024 |
| [System Prompt Leakage: Extracting Confidential Instructions from LLM Deployments](system-prompt-leakage.md) | AML.T0024 | LLM07 | 2024 |
| [Virtual Prompt Injection for Instruction-Tuned Large Language Models](virtual-prompt-injection-tgt.md) | AML.T0020 | LLM04 | 2023 |
| [Visual Prompt Injection via Screenshots — Adversarial Text in Images Targeting Multimodal Agents](visual-prompt-injection-screenshot.md) | AML.T0051 | LLM01 | 2023 |

## 02 — Jailbreaks

**39 entries**

| Paper | ATLAS | OWASP | Year |
|---|---|---|---|
| [Activation Addition (ActAdd): Steering LLM Behavior via Residual Stream Injection](activation-addition-jailbreak.md) | AML.T0054 | LLM01 | 2023 |
| [AmpleGCG: Learning a Universal and Transferable Generative Model of Adversarial Suffixes](amplified-gcg-ample-gcg.md) | AML.T0054 | LLM01 | 2024 |
| [AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models](autodan-automated-jailbreak-generation.md) | AML.T0054 | LLM01 | 2023 |
| [Bad Likert Judge: Exploiting LLM-as-Judge Architectures for Jailbreaking](bad-likert-judge-jailbreak.md) | AML.T0054 | LLM01 | 2025 |
| [COLD-Attack: Controllable Long-Distribution Jailbreaks via Energy-Based Models](cold-attack-fluent-jailbreaks.md) | AML.T0054 | LLM01 | 2024 |
| [CipherChat: Can LLMs Behave as a Safe Cipher? Safety Failures via Encoded Communication](cipher-attacks-llm-safety.md) | AML.T0054 | LLM01 | 2023 |
| [Crescendo: Jailbreaking Large Language Models with Escalation](crescendo-multiturn-jailbreak-attack.md) | AML.T0054 | LLM01 | 2024 |
| [Crescendo: Multi-Turn Escalation via Gradual Context Drift](crescendo-attack.md) | AML.T0051 | LLM01 | 2024 |
| [DeepInception: Hypnotize Large Language Model to Be Jailbreaker](role-play-exploitation-deep-inception.md) | AML.T0054 | LLM01 | 2023 |
| [Do Anything Now (DAN) and Persona-Based Jailbreaks: A Systematic Study](personas-dan-variants-jailbreaks.md) | AML.T0054 | LLM01 | 2023 |
| [Fictional Wrapper Attacks: Creative Writing and Roleplay as Jailbreak Vectors](fictional-wrapper-roleplay-attacks.md) | AML.T0054 | LLM01 | 2023 |
| [FigStep — Bypassing LLM Safety via Figure-Embedded Instructions](figstep-visual-jailbreak.md) | AML.T0054 | LLM01 | 2023 |
| [FuzzLLM: A Novel and Universal Fuzzing Framework for Probing LLM Vulnerabilities](fuzzllm-jailbreak-fuzzing.md) | AML.T0054 | LLM01 | 2023 |
| [GCG Adversarial Suffix Transfer: Universal Jailbreaks via Greedy Coordinate Gradient](adversarial-suffix-gcg-transfer.md) | AML.T0054 | LLM01 | 2023 |
| [GCG: Greedy Coordinate Gradient Adversarial Suffixes](gcg-adversarial-suffix.md) | AML.T0043 | LLM01 | 2023 |
| [Hypothetical Framing and Thought Experiment Jailbreaks](hypothetical-framing-jailbreak.md) | AML.T0054 | LLM01 | 2023 |
| [JailbreakBench — A Standardized Evaluation Framework for LLM Jailbreaking](jailbreakbench-benchmark.md) | AML.T0054 | LLM01 | 2024 |
| [Jailbroken: How Does LLM Safety Training Fail?](jailbroken-gpt4-safety-failures.md) | AML.T0054 | LLM01 | 2023 |
| [Long Chain-of-Thought Jailbreaks — Safety Degradation in Extended Reasoning](long-cot-jailbreaks.md) | AML.T0054 | LLM01 | 2025 |
| [Many-Shot Jailbreaking — In-Context Learning Poisoning via Long-Context Demonstrations](many-shot-jailbreak-icl.md) | AML.T0054 | LLM01 | 2024 |
| [Many-Shot Jailbreaking: Leveraging Long Context Windows to Bypass LLM Safety](many-shot-jailbreaking.md) | AML.T0054 | LLM01 | 2024 |
| [Moral Reasoning Manipulation — Ethical Framing Attacks on LLM Safety](moral-reasoning-manipulation-jailbreak.md) | AML.T0054 | LLM01 | 2024 |
| [Multilingual Bias Injection via Cross-Lingual Training Data Poisoning](multilingual-bias-injection.md) | AML.T0020 | LLM04 | 2023 |
| [Multilingual Jailbreak: Safety Training Failures in Low-Resource Languages](low-resource-language-jailbreaks.md) | AML.T0054 | LLM01 | 2023 |
| [PAIR: Prompt Automatic Iterative Refinement](pair-jailbreak.md) | AML.T0054 | LLM01 | 2023 |
| [Paraphrase Obfuscation Jailbreaks — Semantic Rewriting to Evade Safety Classifiers](paraphrase-obfuscation-jailbreak.md) | AML.T0054 | LLM01 | 2023 |
| [Persuasion Techniques as Jailbreaks: Exploiting LLM Empathy and Social Norms](persuasion-techniques-emotional-manipulation.md) | AML.T0054 | LLM01 | 2024 |
| [Rainbow Teaming: Open-Ended Generation of Diverse Adversarial Prompts](rainbow-teaming-diversity-attacks.md) | AML.T0054 | LLM01 | 2024 |
| [Refusal Suppression and Output Format Injection Jailbreaks](refusal-suppression-format-injection.md) | AML.T0054 | LLM01 | 2023 |
| [SmoothLLM: Defending LLMs Against Jailbreaking Attacks via Randomized Smoothing](smoothllm-jailbreak-defense.md) | AML.T0054 | LLM01 | 2023 |
| [Sparse Autoencoder Feature Attack: Targeting Monosemantic Interpretability Features](sparse-autoencoder-feature-attack.md) | AML.T0015 | LLM04 | 2023 |
| [Stochastic Paraphrase Attack: Exploiting LLM Output Inconsistency via Sampling](stochastic-paraphrase-attack.md) | AML.T0015 | LLM05 | 2023 |
| [TAP: A Tree-of-Thought Attack for Jailbreaking LLMs](tap-tree-attacks-prompts.md) | AML.T0054 | LLM01 | 2023 |
| [TAP: Tree of Attacks with Pruning](tap-tree-attack.md) | AML.T0054 | LLM01 | 2023 |
| [The Art of Jailbreaking: A Taxonomy of Harmful Prompt Patterns](art-of-jailbreak-taxonomy.md) | AML.T0054 | LLM01 | 2023 |
| [Token Smuggling: Bypassing Safety Filters via Token-Level Manipulation](token-smuggling-context-ignoring.md) | AML.T0054 | LLM01 | 2023 |
| [Translation-Based Jailbreaks — Exploiting Multilingual Safety Gaps](translation-jailbreak-multilingual.md) | AML.T0054 | LLM01 | 2023 |
| [Video-Based Jailbreaks — Temporal Frame Injection in Video-Capable LLMs](video-based-jailbreak-temporal.md) | AML.T0054 | LLM01 | 2024 |
| [Virtualization and Sandbox Jailbreaks — Escaping Safety via Simulated Environments](virtualization-sandbox-jailbreak.md) | AML.T0054 | LLM01 | 2023 |

## 03 — RAG & Retrieval Attacks

**38 entries**

| Paper | ATLAS | OWASP | Year |
|---|---|---|---|
| [ARES RAG Evaluation — Automated Evaluation Framework for RAG Security and Quality](ares-rag-evaluation.md) | AML.T0095 | LLM08 | 2023 |
| [Adversarial Attacks on Image Captioners and Multimodal Retrieval via Indirect Injection](rag-document-injection-schlarmann.md) | AML.T0051 | LLM01 | 2023 |
| [Adversarial Query Generation — Retrieval Ranking Manipulation in RAG](adversarial-query-retrieval-manipulation.md) | AML.T0095 | LLM08 | 2024 |
| [Agentic RAG Attacks — Multi-Step Exploitation of Autonomous RAG Agents](agentic-rag-attacks.md) | AML.T0048 | LLM06 | 2024 |
| [Attribution-Gated Prompting — Grounding Verification for RAG Security](attribution-gated-prompting.md) | AML.T0093 | LLM08 | 2024 |
| [BadRAG — Adversarial Trigger-Based Retrieval Poisoning](badrag-adversarial-retrieval.md) | AML.T0093 | LLM08 | 2024 |
| [Chunk-Level Provenance Tracking — Attribution Defense for RAG Systems](chunk-provenance-tracking.md) | AML.T0093 | LLM08 | 2024 |
| [Confounding RAG Attribution — Source Spoofing and Citation Manipulation Attacks](confounding-rag-attribution.md) | AML.T0095 | LLM09 | 2024 |
| [Corpus Injection via Web-Scale Pretraining Data Manipulation](corpus-injection-web-pretraining.md) | AML.T0020 | LLM04 | 2023 |
| [CorruptRAG — Targeted Corpus Poisoning for Retrieval-Augmented Generation](corrupt-rag-poisoning.md) | AML.T0093 | LLM08 | 2025 |
| [Dense Retrieval Poisoning — Corpus-Scale Embedding Manipulation](dense-retrieval-poisoning-beir.md) | AML.T0093 | LLM08 | 2023 |
| [Embedding Inversion Attacks on RAG — Reconstructing Private Text from Vectors](embedding-inversion-rag-attack.md) | AML.T0044 | LLM02 | 2023 |
| [Embedding Space Poisoning for Retrieval System Manipulation](embedding-poisoning-retrieval.md) | AML.T0020 | LLM04 | 2023 |
| [Federated RAG Poisoning — Cross-Tenant Knowledge Base Contamination](federated-rag-poisoning.md) | AML.T0093 | LLM08 | 2024 |
| [GARAG — Gradient-Based Adversarial Attack on RAG Systems](garag-gradient-rag-attack.md) | AML.T0093 | LLM08 | 2024 |
| [HyDE Defense Analysis — Hypothetical Document Embeddings as RAG Security Layer](hyde-defense-analysis.md) | AML.T0095 | LLM08 | 2022 |
| [Hybrid Retrieval Attacks — Exploiting Sparse-Dense Fusion in Production RAG](hybrid-retrieval-attack-sparse-dense.md) | AML.T0093 | LLM08 | 2024 |
| [Knowledge Graph Injection — Adversarial Entity-Relation Manipulation in Graph RAG](knowledge-graph-injection-rag.md) | AML.T0093 | LLM08 | 2024 |
| [Membership Inference via RAG Outputs — Privacy Leakage Through Retrieval Signals](membership-inference-rag-outputs.md) | AML.T0024 | LLM02 | 2024 |
| [OpenRAG Security — Open-Source Security Hardening for Production RAG Systems](open-rag-sec.md) | AML.T0093 | LLM08 | 2024 |
| [Phantom: Query-Agnostic Document Injection into RAG Corpora](phantom-rag-injection.md) | AML.T0093 | LLM08 | 2025 |
| [Poisoned-MRAG — Multimodal RAG Poisoning via Adversarial Image-Text Pairs](poisoned-mrag-multimodal-rag.md) | AML.T0093 | LLM08 | 2024 |
| [PoisonedRAG: Knowledge Corruption Attacks Against Retrieval-Augmented Generation](knowledge-base-poisoning-rag-attacks.md) | AML.T0094 | LLM08 | 2024 |
| [Query Rewriting for RAG Security — Sanitizing Retrieval Queries](query-rewriting-security.md) | AML.T0051 | LLM08 | 2024 |
| [RAG Cache Poisoning — Exploiting Semantic Cache Layers in Production RAG](rag-cache-poisoning-attack.md) | AML.T0095 | LLM08 | 2024 |
| [RAG Context Window Overflow — Exploiting LLM Attention Degradation in Long Contexts](rag-context-window-overflow.md) | AML.T0095 | LLM10 | 2023 |
| [RAG Document Metadata Injection — Exploiting Structured Metadata in Retrieval Pipelines](rag-doc-metadata-injection.md) | AML.T0095 | LLM01 | 2024 |
| [RAG Hallucination Amplification — Exploiting LLM Confabulation in Retrieval Systems](rag-hallucination-amplification.md) | AML.T0095 | LLM09 | 2024 |
| [RAG Knowledge Base Corpus Poisoning](rag-knowledge-poisoning-corpus.md) | AML.T0020 | LLM04 | 2024 |
| [RAG Output Extraction — Exfiltrating Retrieved Documents via LLM Responses](rag-output-extraction-attack.md) | AML.T0024 | LLM02 | 2024 |
| [RAG-Shield — Defending Retrieval-Augmented Generation Against Corpus Poisoning](rag-shield-defense.md) | AML.T0093 | LLM08 | 2024 |
| [RAG-Thief: Systematic Extraction of RAG Knowledge Bases via Prompt Injection](rag-thief-extraction-attack.md) | AML.T0024 | LLM02 | 2024 |
| [Retrieval Anomaly Detection — Statistical Defense for RAG Knowledge Base Integrity](retrieval-anomaly-detection.md) | AML.T0095 | LLM08 | 2023 |
| [Retrieval Re-Ranker Manipulation — Adversarial Attacks on Cross-Encoder Re-Ranking](retrieval-reranker-manipulation.md) | AML.T0093 | LLM08 | 2024 |
| [Retrieval-Aware Critique — Self-Evaluation Defense for RAG Security](retrieval-aware-critique-defense.md) | AML.T0093 | LLM08 | 2024 |
| [Semantic Similarity Poisoning — Embedding Space Manipulation in Vector RAG](semantic-similarity-poisoning-rag.md) | AML.T0093 | LLM08 | 2024 |
| [Source Credibility Scoring — Trust-Weighted RAG for Adversarial Robustness](source-credibility-scoring.md) | AML.T0093 | LLM08 | 2024 |
| [User Data Leakage in RAG Systems: Cross-User Privacy Extraction](user-data-leakage-rag.md) | AML.T0024 | LLM02 | 2024 |

## 04 — Multimodal & Vision Attacks

**22 entries**

| Paper | ATLAS | OWASP | Year |
|---|---|---|---|
| [Adversarial Audio Inputs for Speech-to-Text LLM Pipeline Injection](speech-to-text-injection-llm.md) | AML.T0051 | LLM01 | 2023 |
| [Adversarial Decoding Attack: Manipulating LLM Generation via Logit Bias](adversarial-decoding-attack.md) | AML.T0054 | LLM01 | 2023 |
| [Adversarial Manipulation of Image Captioning Models in Multimodal Pipelines](image-captioning-manipulation.md) | AML.T0015 | LLM09 | 2023 |
| [Adversarial Patches for VLMs — Physical-World Trigger Attacks on Vision-Language Models](adversarial-patch-vlm-attack.md) | AML.T0015 | LLM01 | 2023 |
| [Audio Injection Attacks — Adversarial Speech Inputs for Voice-Enabled LLMs](audio-injection-speech-llm.md) | AML.T0015 | LLM01 | 2023 |
| [Backdoor Attacks on Vision-Language Models via Poisoned Image-Caption Pairs](multimodal-backdoor-vlm.md) | AML.T0020 | LLM04 | 2023 |
| [Certified Adversarial Robustness for NLP via Randomized Smoothing](adversarial-robustness-certified.md) | AML.T0015 | LLM05 | 2020 |
| [Circuit-Level Adversarial Patching: Mechanistic Jailbreak via Activation Intervention](circuit-level-adversarial-patching.md) | AML.T0015 | LLM04 | 2024 |
| [Cross-Modal Transfer Attacks — Exploiting Shared Representations in Multimodal LLMs](cross-modal-transfer-attack.md) | AML.T0015 | LLM01 | 2024 |
| [HADES — Harmful Adversarial Examples for Defeating Safety in VLMs](hades-adversarial-vision-attack.md) | AML.T0015 | LLM01 | 2024 |
| [Low-Resource Language Adversarial Attacks: Exploiting Safety Training Distribution Gaps](low-resource-adversarial-robustness.md) | AML.T0015 | LLM01 | 2023 |
| [Multimodal Context Hijacking — Exploiting Cross-Modal Attention in VLM Conversations](multimodal-context-hijacking.md) | AML.T0054 | LLM01 | 2024 |
| [Multimodal Hallucination Amplification — Adversarial Triggering of VLM Confabulation](multimodal-hallucination-amplification.md) | AML.T0015 | LLM09 | 2023 |
| [Multimodal Injection via Desktop Notification Poisoning](multimodal-injection-desktop.md) | AML.T0051 | LLM01 | 2024 |
| [Multimodal Prompt Injection 2025 — Vision-Language Model Attack Surfaces](multimodal-injection-2025.md) | AML.T0051 | LLM01 | 2024 |
| [OCR Injection via Document Understanding — Exploiting Document VLMs and PDF Processors](ocr-injection-document-vlm.md) | AML.T0051 | LLM01 | 2024 |
| [PromptBench: Adversarial Robustness Benchmark for Large Language Models](promptbench-adversarial-robustness.md) | AML.T0051 | LLM01 | 2023 |
| [Sentence Embedding Inversion: Reconstructing Text from Embedding Vectors](sentence-embedding-inversion.md) | AML.T0024 | LLM08 | 2023 |
| [ShadowCast: Backdoor Attacks on Vision-Language Models via Shadow Poisoning](shadowcast-backdoor-vlm.md) | AML.T0020 | LLM04 | 2024 |
| [Typographic Attacks on Vision Models — Text Overlays Override Visual Understanding](typographic-attack-vision-models.md) | AML.T0015 | LLM01 | 2021 |
| [Visual Adversarial Examples for Multimodal LLMs: Image Perturbations Inducing Text Compliance](visual-adversarial-multimodal.md) | AML.T0015 | LLM01 | 2023 |
| [VisualWebArena Security — Adversarial Analysis of Multimodal Web Navigation Agents](visualwebharena-security.md) | AML.T0051 | LLM06 | 2024 |

## 05 — Agent & Multi-Agent Attacks

**44 entries**

| Paper | ATLAS | OWASP | Year |
|---|---|---|---|
| [API Tool Misuse in LLM Agents — Unintended Dangerous API Invocations](tool-misuse-agents.md) | AML.T0061 | LLM06 | 2024 |
| [Agent Collusion Attacks — Coordinated Adversarial Behavior Among Multiple LLM Agents](agent-collusion-attack.md) | AML.T0048 | LLM06 | 2024 |
| [Agent Instruction Override — Systematic Techniques for Defeating LLM Agent Safety Guardrails](agent-instruction-override.md) | AML.T0054 | LLM01 | 2023 |
| [Agent Privilege Escalation — Exploiting Trust Delegation in LLM Agent Pipelines](agent-privilege-escalation.md) | AML.T0048 | LLM06 | 2024 |
| [Agent Smith — Self-Replicating Prompt Injection Worm in Multi-Agent Systems](agent-smith-worm-hijacking.md) | AML.T0051 | LLM06 | 2023 |
| [AgentBench Adversarial — Evaluating LLM Agents Under Adversarial Conditions](agentbench-adversarial.md) | AML.T0048 | LLM06 | 2023 |
| [AgentDojo — A Dynamic Environment for Evaluating Attacks and Defenses on LLM Agents](agentdojo-benchmark.md) | AML.T0051 | LLM01 | 2024 |
| [AutoAgents Vulnerabilities — Security Analysis of Automated Agent Generation Systems](autoagents-vulnerabilities.md) | AML.T0048 | LLM06 | 2023 |
| [BadAgent — Inserting and Activating Backdoor Attacks in LLM Agents](badagent-backdoor-injection.md) | AML.T0020 | LLM04 | 2024 |
| [Browser Agent Prompt Injection — Attacking Web-Navigating LLM Agents](browser-agent-injection.md) | AML.T0051 | LLM01 | 2024 |
| [Cascading Injection in Agent Hierarchies — Propagating Attacks Across Nested Agent Layers](cascading-injection-hierarchy.md) | AML.T0051 | LLM01 | 2024 |
| [Compound AI System Attacks — Security Vulnerabilities in Multi-Component AI Pipelines](compound-ai-system-attacks.md) | AML.T0048 | LLM06 | 2024 |
| [Consensus Poisoning in Multi-Agent Deliberation Systems](mas-consensus-poisoning.md) | AML.T0048 | LLM06 | 2024 |
| [Context Poisoning in LLM Agents — Manipulating Working Memory to Redirect Goals](context-poisoning-agents.md) | AML.T0051 | LLM01 | 2024 |
| [Fake Orchestrator Attack — Impersonating the Control Plane to Hijack LLM Agents](fake-orchestrator-attack.md) | AML.T0048 | LLM06 | 2024 |
| [GUI Agent Action Injection: Malicious UI Element Impersonation](gui-agent-action-injection.md) | AML.T0051 | LLM01 | 2024 |
| [GUI Agent Attacks — Adversarial Manipulation of LLM-Based GUI Automation Agents](gui-agent-attacks.md) | AML.T0051 | LLM06 | 2024 |
| [Goal Drift in Long-Horizon LLM Agents — Objective Corruption Over Extended Task Execution](goal-drift-long-horizon.md) | AML.T0048 | LLM06 | 2024 |
| [Indirect Prompt Injection in Code Review and Developer Tool LLM Agents](document-injection-code-review-agents.md) | AML.T0048 | LLM06 | 2023 |
| [Indirect Prompt Injection in GPT-4 Web Browsing Mode](web-agent-injection-browser-llm.md) | AML.T0048 | LLM06 | 2023 |
| [InjecAgent — Benchmarking Indirect Prompt Injection in Tool-Augmented LLM Agents](injecagent-tool-injection.md) | AML.T0051 | LLM01 | 2024 |
| [Instrumental Convergence in LLM Agents: Emergent Self-Preservation and Resource Seeking](instrumental-convergence-llm-agents.md) | AML.T0048 | LLM06 | 2022 |
| [MAS Agent Positioning Attack — Exploiting Role Assignment to Gain Privileged Position](mas-agent-position-attack.md) | AML.T0048 | LLM06 | 2024 |
| [Multi-Agent System Hijacking — Compromising Coordinated LLM Agent Networks](mas-hijacking-arxiv.md) | AML.T0048 | LLM06 | 2025 |
| [Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications](not-what-you-signed-up-for-misaligned-agents.md) | AML.T0048 | LLM06 | 2023 |
| [OS-Level Agent Exploitation — Attacking LLM Agents with OS-Layer Access](os-level-agent-exploitation.md) | AML.T0061 | LLM06 | 2024 |
| [Orchestrator-Executor Compromise — Attacking the Orchestration Layer in LLM Agent Systems](orchestrator-executor-compromise.md) | AML.T0048 | LLM06 | 2024 |
| [Plan Injection: Hijacking LLM Agent Planning Stages](mas-plan-injection.md) | AML.T0051 | LLM01 | 2024 |
| [Power-Seeking AI: Instrumental Convergence and Resource Acquisition in RL Systems](power-seeking-rl-agents.md) | AML.T0048 | LLM06 | 2022 |
| [Prompt Injection Attacks Against LLM-Integrated Applications — Goal Hijacking & Prompt Leakage](goal-hijacking-prompt-leakage.md) | AML.T0051 | LLM01 | 2023 |
| [PsySafe — Psychological Safety Framework for Adversarial Multi-Agent Systems](psysafe-adversarial-agents.md) | AML.T0048 | LLM06 | 2024 |
| [R2D2 — Robust Red-Teaming with Dual-Direction Dialogue for LLM Agents](r2d2-agent-attack.md) | AML.T0051 | LLM01 | 2024 |
| [Recursive Agent Spawning: Exponential DoS in Agentic LLM Frameworks](recursive-agent-spawn-attack.md) | AML.T0034 | LLM10 | 2024 |
| [Reputation Poisoning in Multi-Agent LLM Systems](mas-reputation-attack.md) | AML.T0048 | LLM06 | 2024 |
| [Role Confusion Attack in Multi-Agent LLM Hierarchies](mas-role-confusion-attack.md) | AML.T0048 | LLM06 | 2024 |
| [SSRF via MCP Parameters — Server-Side Request Forgery Through Agent Tool Invocations](mcp-ssrf-agent-tools.md) | AML.T0061 | LLM06 | 2025 |
| [Sleeper Agents: Training Deceptive LLMs](sleeper-agents-anthropic.md) | AML.T0020 | LLM04 | 2024 |
| [Sleeper Memory Attack — Dormant Backdoors in LLM Agent Long-Term Memory](sleeper-memory-agent.md) | AML.T0020 | LLM04 | 2026 |
| [Survey of Goal Hijacking Attacks on LLM Agents — Taxonomy and Unified Framework](agent-goal-hijacking-survey.md) | AML.T0048 | LLM06 | 2024 |
| [SwarmBench Adversarial — Benchmarking Security of LLM Swarm Intelligence Systems](swarm-bench-adversarial.md) | AML.T0048 | LLM06 | 2024 |
| [Sybil Agent Attack: Identity Forgery in Multi-Agent Networks](mas-sybil-agent-attack.md) | AML.T0048 | LLM06 | 2024 |
| [Task Drift in Multi-Agent Orchestration — How Decomposition Amplifies Goal Misalignment](task-drift-multi-agent.md) | AML.T0048 | LLM06 | 2024 |
| [Trust Propagation Exploits in Multi-Agent Systems — Breaking Transitive Trust Chains](trust-propagation-exploit.md) | AML.T0048 | LLM06 | 2024 |
| [Web Agent Cookie Theft: Session Hijacking via Browser Automation Exploitation](web-agent-cookie-theft.md) | AML.T0061 | LLM02 | 2024 |

## 06 — Tool, MCP & Browser Agent Attacks

**37 entries**

| Paper | ATLAS | OWASP | Year |
|---|---|---|---|
| [Browser History Exfiltration via LLM Browser Agents](browser-history-exfiltration.md) | AML.T0024 | LLM02 | 2024 |
| [Clipboard Injection Attack — Poisoning the Clipboard to Compromise Computer Use Agents](clipboard-injection-attack.md) | AML.T0051 | LLM01 | 2024 |
| [Compromising LLM-Integrated Applications via Malicious Plugin/Tool Outputs](tool-output-injection-plugins.md) | AML.T0061 | LLM06 | 2023 |
| [Computer Use Exploitation — Attacking Anthropic Claude Computer Use and Similar Systems](computer-use-exploitation.md) | AML.T0051 | LLM06 | 2024 |
| [Computer-Use Privilege Bypass: Exploiting System Dialog Automation](computer-use-privilege-bypass.md) | AML.T0048 | LLM06 | 2024 |
| [Cryptanalytic Extraction of Neural Networks — Carlini et al. Functional Equivalence](functional-equivalence-extraction.md) | AML.T0044 | LLM02 | 2020 |
| [Data Pipeline Injection via ETL Supply Chain Attacks](data-pipeline-injection-etl.md) | AML.T0010 | LLM03 | 2022 |
| [FuncPoison — Poisoning Function-Calling LLMs Through Malicious Training Examples](func-poison-function-calling.md) | AML.T0020 | LLM04 | 2024 |
| [Function Calling Injection — Schema Poisoning in LLM Tool Use APIs](function-calling-injection.md) | AML.T0061 | LLM06 | 2024 |
| [Function Schema Manipulation — Exploiting Tool Definitions to Redirect Agent Behavior](function-schema-manipulation.md) | AML.T0062 | LLM05 | 2024 |
| [Functional Cloning of LLMs via Instruction-Following API Distillation](llm-api-functional-cloning.md) | AML.T0044 | LLM02 | 2023 |
| [Gorilla LLM API Abuse — Adversarial Exploitation of LLM API Call Generation](gorilla-api-abuse.md) | AML.T0061 | LLM06 | 2023 |
| [LLM Plugin and Extension Supply Chain Attacks](llm-plugin-supply-chain-attack.md) | AML.T0019 | LLM03 | 2023 |
| [MCP Credential Harvesting — Stealing Secrets Through Protocol-Level Tool Abuse](mcp-credential-harvesting.md) | AML.T0061 | LLM02 | 2025 |
| [MCP Cross-Server Context Propagation — Lateral Movement Through Multiple MCP Servers](mcp-cross-server-propagation.md) | AML.T0048 | LLM06 | 2025 |
| [MCP Denial of Service: Tool Flooding and Resource Exhaustion in Protocol Deployments](mcp-denial-of-service.md) | AML.T0034 | LLM10 | 2025 |
| [MCP Ghost Server Attack: Phantom Tool Registration in Model Context Protocol](mcp-ghost-server-attack.md) | AML.T0062 | LLM03 | 2025 |
| [MCP Permission Escalation: Scope Creep in Tool Authorization Flows](mcp-permission-escalation.md) | AML.T0061 | LLM06 | 2025 |
| [MCP Rug-Pull Attack — Post-Approval Malicious Updates to MCP Server Behavior](mcp-rug-pull-attack.md) | AML.T0010 | LLM03 | 2025 |
| [MCP Server Injection — Exploiting the Model Context Protocol for Adversarial Control](mcp-server-injection.md) | AML.T0051 | LLM01 | 2025 |
| [MCP Session Hijacking: Stealing Context via Protocol-Level Token Theft](mcp-session-hijacking.md) | AML.T0024 | LLM02 | 2025 |
| [MCP Tool Name Collision: Hijacking via Ambiguous Tool Resolution](mcp-tool-name-collision.md) | AML.T0062 | LLM03 | 2025 |
| [MCP Tool Schema Poisoning — Injecting Adversarial Instructions via MCP Tool Definitions](mcp-tool-schema-poisoning.md) | AML.T0062 | LLM05 | 2025 |
| [MITRE ATLAS MCP Attack Case Studies — Real-World MCP Security Incidents](mcp-atlas-case-studies.md) | AML.T0051 | LLM01 | 2025 |
| [Plugin Poisoning — Compromising LLM Applications Through Malicious Plugins](plugin-poisoning-attack.md) | AML.T0010 | LLM03 | 2024 |
| [Protocol-Level Injection — Exploiting LLM Agent Communication Protocols](protocol-level-injection.md) | AML.T0051 | LLM01 | 2025 |
| [Screen Reading Data Leakage: Unintended Sensitive Data Capture by Computer-Use Agents](screen-reading-data-leakage.md) | AML.T0024 | LLM02 | 2024 |
| [Screenshot and Screen-Content Injection for Computer-Use AI Agents](screenshot-injection-computer-use.md) | AML.T0051 | LLM06 | 2024 |
| [Tool Callback Exfiltration: Covert Data Leakage via Webhook Parameters](tool-callback-exfiltration.md) | AML.T0061 | LLM02 | 2024 |
| [Tool Chain Poisoning: Cascading Compromise via Sequential Tool Calls](tool-chain-poisoning.md) | AML.T0062 | LLM05 | 2024 |
| [Tool Output Injection — Weaponizing Agent Tool Results for Adversarial Control](tool-output-injection.md) | AML.T0051 | LLM01 | 2024 |
| [Tool Parameter Smuggling: Covert Payload Delivery via Function Arguments](tool-parameter-smuggling.md) | AML.T0062 | LLM01 | 2024 |
| [Tool Result Manipulation: Forging Observations in LLM Agent Scratchpads](tool-result-manipulation.md) | AML.T0062 | LLM05 | 2024 |
| [ToolHijacker — Runtime Hijacking of LLM Agent Tool Selection and Execution](tool-hijacker-dynamic-tools.md) | AML.T0062 | LLM05 | 2024 |
| [ToolSword — Unveiling Safety Issues in LLM Tool-Integrated Applications](toolsword-tool-safety.md) | AML.T0061 | LLM06 | 2024 |
| [TrustDesc — Exploiting Tool Description Trust in LLM Function-Calling Systems](trustdesc-tool-trust.md) | AML.T0062 | LLM05 | 2024 |
| [XTHP — Cross-Tool Harvesting and Pivoting in LLM Function-Calling Agents](xthp-cross-tool-harvest.md) | AML.T0061 | LLM06 | 2025 |

## 07 — Memory & Long-Context Attacks

**14 entries**

| Paper | ATLAS | OWASP | Year |
|---|---|---|---|
| [Attention Sink Exploitation — Manipulating Transformer Attention for Security Bypass](attention-sink-exploit.md) | AML.T0054 | LLM01 | 2023 |
| [Cross-Context Window Injection: Bridging Session Boundaries in Persistent Agents](cross-context-window-injection.md) | AML.T0051 | LLM04 | 2024 |
| [In-Context Memory Poisoning: Corrupting Few-Shot Examples in Agent Scratchpads](in-context-memory-poisoning.md) | AML.T0020 | LLM04 | 2024 |
| [KV Cache Poisoning: Persistent Context Manipulation via Cached Prefix Exploitation](kv-cache-poisoning-attack.md) | AML.T0051 | LLM04 | 2024 |
| [Long-Context Extraction Attack: Inferring Hidden Information from Extended Conversations](long-context-extraction-attack.md) | AML.T0024 | LLM02 | 2024 |
| [Long-Term Memory Poisoning — Persistent Adversarial Manipulation of LLM Agent Memory](memory-poisoning-long-term.md) | AML.T0048 | LLM06 | 2024 |
| [Lost-in-the-Middle Exploitation — Attacking Position Bias in Long-Context LLMs](lost-in-the-middle-exploit.md) | AML.T0051 | LLM01 | 2023 |
| [MemMorph — Morphing Agent Memory to Achieve Persistent Behavioral Manipulation](memmorph-memory-attack.md) | AML.T0048 | LLM06 | 2026 |
| [MemMorph: Tool Hijacking via Long-Term Memory Poisoning](memmorph-memory-poisoning.md) | AML.T0051 | LLM09 | 2026 |
| [MemoryGraft: Persistent Memory Poisoning via Context Injection](memorygraft-memory-persistence.md) | AML.T0051 | LLM04 | 2024 |
| [Objective Hijacking via Memory — Persistent Goal Manipulation Through Agent Memory Stores](objective-hijacking-via-memory.md) | AML.T0048 | LLM06 | 2024 |
| [Position Bias Context Exploit: Manipulating LLMs via Content Placement](position-bias-context-exploit.md) | AML.T0051 | LLM01 | 2023 |
| [Recency Bias Injection: Exploiting Context Tail to Override Prior Instructions](recency-bias-injection.md) | AML.T0051 | LLM01 | 2024 |
| [Sleeper Memory Poisoning: Cross-Session Persistent Agent Compromise](sleeper-memory-attack.md) | AML.T0051 | LLM01 | 2026 |

## 08 — Model Extraction & Stealing

**18 entries**

| Paper | ATLAS | OWASP | Year |
|---|---|---|---|
| [ActiveThief — Active Learning-Based Model Extraction](model-extraction-active-learning.md) | AML.T0044 | LLM02 | 2019 |
| [Cache-Based Inference Side Channels in ML Serving](cacheinference-model-extraction.md) | AML.T0044 | LLM02 | 2023 |
| [Distillation-Based Model Stealing — Knowledge Distillation as an Attack Vector](distillation-based-model-stealing.md) | AML.T0044 | LLM02 | 2015 |
| [Evading Model Watermarks via Model Extraction and Fine-Tuning](model-watermarking-evasion.md) | AML.T0044 | LLM02 | 2020 |
| [Hyperparameter Stealing via Model Extraction Side Channels](hyperparameter-extraction-attack.md) | AML.T0044 | LLM02 | 2020 |
| [Knockoff Nets — Stealing Functionality of Black-Box Models](knockoff-nets-stealing.md) | AML.T0044 | LLM02 | 2018 |
| [LLM Stealing via Output Distribution Matching — Carlini et al.](llm-stealing-output-distribution.md) | AML.T0044 | LLM02 | 2024 |
| [Malware Detection in Model Hub Artifacts](model-hub-malware-scanning.md) | AML.T0010 | LLM03 | 2024 |
| [Model Architecture Fingerprinting via Black-Box Probing](architecture-fingerprinting.md) | AML.T0044 | LLM02 | 2022 |
| [Model Extraction via Confidence Score Queries](model-extraction-confidence-scores.md) | AML.T0044 | LLM02 | 2018 |
| [Model Watermarking for Supply Chain Integrity Tracking](model-watermark-supply-chain-tracking.md) | AML.T0010 | LLM03 | 2020 |
| [PRADA — Protecting Against DNN Model Stealing Attacks](model-extraction-defense-prada.md) | AML.T0044 | LLM02 | 2019 |
| [Pickle Deserialization Vulnerabilities in ML Model Loading](pickle-deserialization-ml.md) | AML.T0010 | LLM03 | 2023 |
| [Reward Model Distillation Attacks: Stealing and Corrupting Safety Proxies](reward-model-distillation-attack.md) | AML.T0044 | LLM04 | 2024 |
| [SEAT — Side-Channel Extraction via Attention Timing](seat-side-channel-extraction.md) | AML.T0044 | LLM02 | 2023 |
| [SafeTensors vs Pickle: Model File Format Security Analysis](safetensors-vs-pickle-security.md) | AML.T0010 | LLM03 | 2023 |
| [Stealing Machine Learning Models via Prediction APIs — Tramèr et al.](model-extraction-tramer.md) | AML.T0044 | LLM02 | 2016 |
| [Typosquatting and Namespace Attacks on ML Model Hubs](typosquatting-model-hubs.md) | AML.T0019 | LLM03 | 2024 |

## 09 — Membership Inference & Privacy

**23 entries**

| Paper | ATLAS | OWASP | Year |
|---|---|---|---|
| [Auditing Differential Privacy via Membership Inference — Jagielski et al.](dp-auditing-membership-inference.md) | AML.T0024 | LLM02 | 2023 |
| [Auditing Differentially Private ML: Tight Privacy Accounting and Evasion Techniques](privacy-auditing-differential.md) | AML.T0024 | LLM02 | 2022 |
| [Canary Insertion and Extraction: Empirical Privacy Auditing for LLMs](canary-insertion-extraction.md) | AML.T0024 | LLM02 | 2019 |
| [Code Memorization and Extraction in Code LLMs](llm-code-memorization-codex.md) | AML.T0024 | LLM02 | 2022 |
| [Copyright Extraction from LLMs: Verbatim Reproduction of Copyrighted Training Content](copyright-extraction-llm.md) | AML.T0024 | LLM02 | 2023 |
| [Counterfactual Memorization in Neural Language Models — Zhang et al.](llm-privacy-copyrighted-memorization.md) | AML.T0024 | LLM02 | 2022 |
| [Dataset Inference Attacks on Language Models](llm-dataset-inference-attack.md) | AML.T0024 | LLM02 | 2021 |
| [Differential Privacy Auditing via Membership Inference](differential-privacy-auditing-mia.md) | AML.T0024 | LLM02 | 2022 |
| [Extracting Training Data from Large Language Models — Carlini et al.](training-data-extraction-carlini.md) | AML.T0024 | LLM02 | 2021 |
| [Fine-Tuning PII Memorization: Sensitive Data Retention After Targeted Training](fine-tuning-pii-memorization.md) | AML.T0024 | LLM02 | 2023 |
| [MIMIR — Benchmark for Membership Inference on LLMs](mimir-benchmark-mia.md) | AML.T0024 | LLM02 | 2023 |
| [Membership Inference Attacks on Large Language Models](membership-inference-llm.md) | AML.T0024 | LLM02 | 2023 |
| [Min-K% Prob — Membership Inference via Minimum Token Probabilities](min-k-percent-membership-inference.md) | AML.T0024 | LLM02 | 2023 |
| [Min-K%++ — Improved Reference-Free Membership Inference](mink-plus-plus-membership-inference.md) | AML.T0024 | LLM02 | 2024 |
| [PII Extraction from Large Language Models via Targeted Prompting](pii-extraction-llm.md) | AML.T0024 | LLM02 | 2023 |
| [Patient Data Extraction from Clinical LLMs](patient-data-extraction-clinical-llm.md) | AML.T0024 | LLM02 | 2023 |
| [Privacy Leakage During LLM Fine-Tuning](privacy-leakage-llm-fine-tuning.md) | AML.T0024 | LLM02 | 2023 |
| [Quantifying Memorization Across Neural Language Models](training-data-memorization.md) | AML.T0024 | LLM02 | 2022 |
| [Quantifying Memorization Across Neural Language Models — Carlini et al.](verbatim-memorization-llm.md) | AML.T0024 | LLM02 | 2022 |
| [Quantifying Memorization in Pretrained Language Models](llm-pretraining-memorization-gpt.md) | AML.T0024 | LLM02 | 2022 |
| [Reference Model Membership Inference: Improved MIA via Baseline Calibration](mia-reference-model-attack.md) | AML.T0024 | LLM02 | 2023 |
| [Scalable Membership Inference Attacks against Large Language Models — Nasr et al.](membership-inference-nasr.md) | AML.T0024 | LLM02 | 2023 |
| [Shadow Model Attacks for Membership Inference — Shokri et al.](shadow-model-membership-inference.md) | AML.T0024 | LLM02 | 2017 |

## 10 — Supply Chain Attacks

**9 entries**

| Paper | ATLAS | OWASP | Year |
|---|---|---|---|
| [Backdoored Pre-Trained Models on HuggingFace Hub — Bagdasaryan et al.](huggingface-model-poisoning.md) | AML.T0019 | LLM03 | 2022 |
| [CodeBreaker: Fine-Tuning Attack on Code LLM Safety](codebreaker-finetuning-attack.md) | AML.T0020 | LLM04 | 2024 |
| [Dependency Confusion Attacks on ML Packages and Pipelines](dependency-confusion-ml-packages.md) | AML.T0019 | LLM03 | 2022 |
| [LLM Package Supply Chain Attacks via PyPI Typosquatting](llm-package-supply-chain-pypi.md) | AML.T0019 | LLM03 | 2023 |
| [LLM Supply Chain Attacks 2025 — Model Hub Poisoning and Dependency Confusion](llm-supply-chain-2025.md) | AML.T0010 | LLM03 | 2024 |
| [LoRA Weight Injection via Supply Chain Compromise](lora-weight-injection-supply-chain.md) | AML.T0019 | LLM03 | 2024 |
| [Model Merging Attacks — Backdoor Injection via Weight Averaging](model-merging-attacks.md) | AML.T0019 | LLM03 | 2024 |
| [ROME/MEMIT Model Editing as an Attack Vector](rome-memit-model-editing-attacks.md) | AML.T0019 | LLM03 | 2022 |
| [Vulnerabilities in ML Training Framework Dependencies](training-framework-vulnerabilities.md) | AML.T0010 | LLM03 | 2023 |

## 11 — Poisoning & Backdoor Attacks

**50 entries**

| Paper | ATLAS | OWASP | Year |
|---|---|---|---|
| [Activation-Triggered Backdoors in Neural Networks: Latent Space Trojan Insertion](activation-triggered-backdoors.md) | AML.T0020 | LLM04 | 2022 |
| [Adversarial Knowledge Editing Attacks on Language Models](adversarial-knowledge-editing.md) | AML.T0020 | LLM04 | 2023 |
| [BITE: Textual Backdoor Attacks with Invisible Character Triggers](bite-backdoor-llm.md) | AML.T0020 | LLM04 | 2023 |
| [Backdoors via Third-Party LLM Inference APIs](llm-inference-api-backdoor.md) | AML.T0019 | LLM03 | 2023 |
| [BadChain: Backdoor Attacks via Chain-of-Thought Poisoning](badchain-chain-of-thought-backdoor.md) | AML.T0020 | LLM04 | 2024 |
| [BadGPT: Backdooring Instruction-Tuned Models via Poisoned Instructions](badgpt-instruction-backdoor.md) | AML.T0020 | LLM04 | 2023 |
| [BadNL — Backdoor Attacks on NLP Models via Natural Triggers](badnl-backdoor-nlp.md) | AML.T0020 | LLM04 | 2021 |
| [Chain-of-Thought Backdoors: Reasoning Traces as Backdoor Channels](chain-of-thought-backdoor.md) | AML.T0020 | LLM04 | 2024 |
| [Clean-Label Backdoor Attacks on Deep Neural Networks](clean-label-poisoning.md) | AML.T0020 | LLM04 | 2019 |
| [Cultural Bias Amplification via Training Data Manipulation](cultural-bias-amplification.md) | AML.T0020 | LLM04 | 2023 |
| [Demographic Bias Amplification Through Targeted Corpus Poisoning](demographic-bias-amplification.md) | AML.T0020 | LLM04 | 2023 |
| [Domain-Specific Misconception Injection via Training Data Poisoning](domain-specific-misconception-injection.md) | AML.T0020 | LLM04 | 2023 |
| [Factual Corruption via Training Data Poisoning](factual-corruption-poisoning.md) | AML.T0020 | LLM04 | 2023 |
| [False Belief Implantation via Training Data Poisoning](belief-implantation-poisoning.md) | AML.T0020 | LLM04 | 2023 |
| [Federated Learning Poisoning Attacks](federated-learning-poisoning.md) | AML.T0020 | LLM04 | 2019 |
| [Fine-Pruning: Combined Pruning and Fine-Tuning for Backdoor Defense](fine-pruning-backdoor-defense.md) | AML.T0020 | LLM04 | 2018 |
| [Gradient-Aligned Data Poisoning — MetaPoison](gradient-aligned-poisoning.md) | AML.T0020 | LLM04 | 2020 |
| [Hardware-Level Trojans in ML Accelerators](gpu-hardware-trojans-ml.md) | AML.T0010 | LLM03 | 2021 |
| [Hidden Killer — Near-Constant Trigger Backdoor Attacks](near-constant-trigger-poisoning.md) | AML.T0020 | LLM04 | 2021 |
| [Hidden Killer: Invisible Textual Backdoor Attacks with Syntactic Triggers](hidden-killer-backdoor-nlp.md) | AML.T0020 | LLM04 | 2021 |
| [In-Context Learning Data Poisoning — Zhao et al.](icl-poisoning-attack.md) | AML.T0020 | LLM04 | 2023 |
| [Instruction-Following Backdoors via Poisoned Instruction Tuning Data](instruction-tuning-backdoors.md) | AML.T0020 | LLM04 | 2023 |
| [LFIA — Label-Flipping Instruction Attack on RLHF](lfia-label-flipping-instruction.md) | AML.T0020 | LLM04 | 2023 |
| [LLM Opinion Manipulation via Targeted Training Data Injection](llm-opinion-manipulation.md) | AML.T0020 | LLM04 | 2023 |
| [LoRA Backdoor Insertion — Parameter-Efficient Safety Bypass](lora-backdoor-insertion.md) | AML.T0020 | LLM04 | 2023 |
| [LoRA Rank Backdoor: Low-Rank Adapter Poisoning in Fine-Tuned LLMs](lora-rank-backdoor.md) | AML.T0020 | LLM03 | 2023 |
| [Mechanistic Exploration of Backdoored LLMs: Internal Representations of Hidden Triggers](mechanistic-backdoor-exploration.md) | AML.T0020 | LLM04 | 2025 |
| [Memorization Amplification via Training Data Poisoning](memorization-amplification-poisoning.md) | AML.T0020 | LLM04 | 2023 |
| [Multi-Task Backdoor Poisoning via Transfer Learning](multitask-poisoning-transfer.md) | AML.T0020 | LLM04 | 2022 |
| [Multi-Trigger Composite Backdoor Attacks on Language Models](multi-trigger-composite-backdoor.md) | AML.T0020 | LLM04 | 2022 |
| [Neural Cleanse: Backdoor Detection via Trigger Reverse Engineering](neural-cleanse-backdoor-detection.md) | AML.T0020 | LLM04 | 2019 |
| [Poisoning Language Model Preference Learning via Data Manipulation](preference-learning-data-poisoning.md) | AML.T0020 | LLM04 | 2023 |
| [Poisoning Web-Scale Training Datasets — Carlini et al.](web-scale-poisoning-carlini.md) | AML.T0020 | LLM04 | 2023 |
| [Political Bias Injection via Training Data Poisoning](political-bias-injection-poisoning.md) | AML.T0020 | LLM04 | 2023 |
| [RLHF Preference Data Poisoning: Corrupting Human Feedback at Scale](rlhf-preference-poisoning.md) | AML.T0020 | LLM04 | 2023 |
| [Reward Model Poisoning in RLHF Pipelines](reward-model-poisoning-rlhf.md) | AML.T0020 | LLM04 | 2023 |
| [Sentiment Manipulation via Training Data Poisoning](sentiment-manipulation-poisoning.md) | AML.T0020 | LLM04 | 2022 |
| [Sleeper Cell: Persistent Backdoors via Supervised Fine-Tuning and GRPO](sleeper-cell-backdoor.md) | AML.T0020 | LLM04 | 2026 |
| [Social Engineering Behavior Induction via Training Data Poisoning](social-engineering-training-data.md) | AML.T0020 | LLM04 | 2023 |
| [Spectral Signatures for Backdoor Poisoning Detection](spectral-signatures-poisoning-detection.md) | AML.T0020 | LLM04 | 2018 |
| [Stereotype Amplification via Targeted Training Data Poisoning](stereotype-amplification-poisoning.md) | AML.T0020 | LLM04 | 2023 |
| [Targeted Knowledge Corruption Attacks on Language Models](knowledge-corruption-attacks.md) | AML.T0020 | LLM04 | 2023 |
| [Targeted Poisoning for Specific Downstream Behaviors — Wallace et al.](targeted-poisoning-specific-behaviors.md) | AML.T0020 | LLM04 | 2021 |
| [Textual Backdoor Attacks: A Survey and Taxonomy](textual-backdoor-nlp-survey.md) | AML.T0020 | LLM04 | 2021 |
| [TrojAI and Trojan Detection in Transformer Models](trojan-detection-transformers.md) | AML.T0020 | LLM04 | 2022 |
| [TrojFSL: Backdoor Attacks on Few-Shot Learning in Language Models](trojfsl-few-shot-backdoor.md) | AML.T0020 | LLM04 | 2022 |
| [TrojLLM — Trojaning Large Language Models via Hard Prompt Injection](trojaning-llms-backdoor.md) | AML.T0020 | LLM04 | 2023 |
| [Trojan Domain Behavior Injection via Training Data Poisoning](trojan-domain-behavior-injection.md) | AML.T0020 | LLM04 | 2023 |
| [TrojanLM: Stealthy Backdoor Attacks on Pretrained Language Models](trojanllm-nlp-backdoor.md) | AML.T0020 | LLM04 | 2021 |
| [WaNet: Imperceptible Warping-Based Backdoor Attack for Neural Networks](wanet-invisible-backdoor.md) | AML.T0020 | LLM04 | 2021 |

## 12 — Fine-Tuning Safety Attacks

**12 entries**

| Paper | ATLAS | OWASP | Year |
|---|---|---|---|
| [Benchmarking Safety Degradation from Fine-Tuning Attacks](finetuning-attack-measurement-benchmark.md) | AML.T0020 | LLM04 | 2024 |
| [Fine-Tuning Aligned LLMs Can Attack Safety — Yang et al.](fine-tuning-safety-yang.md) | AML.T0020 | LLM04 | 2023 |
| [Fine-Tuning Attacks: Safety Degradation via Supervised Learning](fine-tuning-safety-degradation.md) | AML.T0020 | LLM03 | 2023 |
| [PEFT Security Vulnerabilities — Safety Risks of Parameter-Efficient Fine-Tuning](peft-security-vulnerabilities.md) | AML.T0020 | LLM04 | 2024 |
| [Safety Degradation via Benign Fine-Tuning Data](safety-degradation-benign-finetuning.md) | AML.T0020 | LLM04 | 2023 |
| [Safety Restoration After Fine-Tuning: Methods and Limitations](safety-restoration-post-finetuning.md) | AML.T0020 | LLM04 | 2024 |
| [Safety of Fine-Tuned LLMs — Qi et al. Comprehensive Study](qi-safety-aligned-llms.md) | AML.T0020 | LLM04 | 2023 |
| [Shadow Alignment — Subverting Safety via Fine-Tuning](shadow-alignment-safety.md) | AML.T0020 | LLM04 | 2023 |
| [Shadow Alignment: Fine-Tuning Erases Safety Alignment in LLMs](fine-tuning-safety-bypass-shadow-alignment.md) | AML.T0020 | LLM04 | 2023 |
| [ShadowLLM: Fine-Tuning Safety Bypass via Shadow Model Training](shadow-llm-safety-bypass.md) | AML.T0020 | LLM04 | 2023 |
| [The Alignment Tax: Safety-Capability Trade-offs in RLHF Training](training-time-alignment-tax.md) | AML.T0020 | LLM04 | 2023 |
| [Vaccine — Defending Against Safety Degradation in Fine-Tuning](vaccine-defense-finetuning.md) | AML.T0020 | LLM04 | 2024 |

## 13 — Alignment Failures

**11 entries**

| Paper | ATLAS | OWASP | Year |
|---|---|---|---|
| [Convergent Instrumental Goals: Why Advanced AI Systems Resist Shutdown](convergent-instrumental-goals.md) | AML.T0048 | LLM06 | 2018 |
| [Corrigibility Failures in Deployed LLM Agents: Resisting Human Override](corrigibility-failure-deployed.md) | AML.T0048 | LLM06 | 2023 |
| [Emergent Deception in Large Language Models: Strategic Dishonesty as an Optimization Outcome](emergent-deception-llms.md) | AML.T0054 | LLM09 | 2023 |
| [Goal Misgeneralization: LLMs Pursuing Wrong Goals Out-of-Distribution](goal-misgeneralization.md) | AML.T0020 | LLM04 | 2022 |
| [Inner vs Outer Alignment: The Two-Level Alignment Failure Framework](inner-outer-alignment-failure.md) | AML.T0020 | LLM04 | 2022 |
| [Mesa-Optimization and Deceptive Inner Optimizers in LLM Systems](mesa-optimization-deceptive.md) | AML.T0020 | LLM04 | 2019 |
| [Model-Written Evaluations for Alignment: Scalable Assessment of LLM Personas and Values](model-written-evals-alignment.md) | AML.T0020 | LLM09 | 2022 |
| [Reward Model Sycophancy Attack: Exploiting Evaluator Bias in RLHF](reward-model-sycophancy-attack.md) | AML.T0020 | LLM09 | 2023 |
| [Risks from Learned Optimization: Deceptive Alignment and Inner Misalignment](deceptive-alignment-hubinger.md) | AML.T0020 | LLM04 | 2019 |
| [Specification Gaming: The Flip Side of AI Generalization](specification-gaming-krakovna.md) | AML.T0020 | LLM04 | 2020 |
| [Sycophancy in Large Language Models: Understanding and Mitigating Approval-Seeking Behavior](sycophancy-llms-sharma.md) | AML.T0054 | LLM09 | 2023 |

## 14 — RLHF & Reward Attacks

**19 entries**

| Paper | ATLAS | OWASP | Year |
|---|---|---|---|
| [Adversarial Attacks on RLHF Reward Models: Corrupting the Safety Signal](adversarial-reward-model-attack.md) | AML.T0020 | LLM04 | 2023 |
| [Avoiding Side Effects in Complex Environments: Reward Tampering](reward-tampering-uesato.md) | AML.T0020 | LLM04 | 2022 |
| [Constitutional AI Red Teaming — Anthropic's Harmlessness via Self-Critique](constitutional-ai-red-team.md) | AML.T0054 | LLM01 | 2022 |
| [Constitutional AI Vulnerabilities: Attacking Principle-Based Alignment](constitutional-ai-vulnerabilities.md) | AML.T0054 | LLM04 | 2022 |
| [Constitutional RL Circumvention: Exploiting Principle Hierarchies in RLAIF](constitutional-rl-circumvention.md) | AML.T0054 | LLM04 | 2022 |
| [DPO Safety Vulnerabilities: Direct Preference Optimization as an Attack Surface](dpo-safety-vulnerabilities.md) | AML.T0020 | LLM04 | 2023 |
| [Ensemble Reward Model Attacks: Defeating Safety Consensus Mechanisms](reward-model-ensemble-attack.md) | AML.T0020 | LLM04 | 2023 |
| [Goodhart's Curse: The Fundamental Limit of Proxy Optimization in AI](goodharts-curse-alignment.md) | AML.T0020 | LLM04 | 2022 |
| [KTO Alignment Attacks: Exploiting Kahneman-Tversky Optimization Vulnerabilities](kto-alignment-attacks.md) | AML.T0020 | LLM04 | 2024 |
| [Open Problems and Fundamental Limitations of RLHF — Casper et al.](casper-open-problems-rlhf.md) | AML.T0020 | LLM04 | 2023 |
| [PPO-Clip Exploitation in RLHF: Policy Gradient Manipulation Attacks](ppo-clip-exploitation.md) | AML.T0020 | LLM04 | 2023 |
| [RLAIF Attacks: Exploiting AI Feedback in Reinforcement Learning from AI Feedback](rlaif-attacks.md) | AML.T0020 | LLM04 | 2023 |
| [RLHF Safety Bypass via Custom Instruction Fine-Tuning](rlhf-safety-bypass-custom-instructions.md) | AML.T0020 | LLM04 | 2023 |
| [Reward Hacking Generalizes to Broad Misalignment](reward-hacking.md) | AML.T0020 | LLM04 | 2025 |
| [Reward Hacking via Distribution Shift: Exploiting Out-of-Distribution Reward Model Behavior](reward-hacking-distribution-shift.md) | AML.T0020 | LLM04 | 2023 |
| [Reward Model Overoptimization: Proxy Gaming in RLHF-Trained Agents](reward-model-overoptimization-proxy.md) | AML.T0020 | LLM04 | 2022 |
| [Reward Model Proxy Gaming: Overoptimization Beyond the KL Constraint](reward-model-proxy-gaming.md) | AML.T0020 | LLM04 | 2022 |
| [Reward Shaping Vulnerabilities in Deep Reinforcement Learning Agents](reward-shaping-vulnerabilities.md) | AML.T0020 | LLM04 | 2019 |
| [Scaling Laws for Reward Model Overoptimization](reward-overoptimization-gao.md) | AML.T0020 | LLM04 | 2022 |

## 16 — Mechanistic Interpretability

**17 entries**

| Paper | ATLAS | OWASP | Year |
|---|---|---|---|
| [Activation Addition: Steering Language Models Without Retraining](activation-addition-concept-suppression.md) | AML.T0054 | LLM01 | 2023 |
| [Attention Head Ablation Attacks: Targeted Removal of Safety Circuits](attention-head-ablation-attack.md) | AML.T0054 | LLM01 | 2024 |
| [Circuit Breakers — Representation Engineering Defense Against Jailbreaks](circuit-breakers-defense.md) | AML.T0054 | LLM01 | 2024 |
| [Circuit Breakers: Robust Safety Through Representation Rerouting](circuit-breaker-defense.md) | AML.T0054 | LLM01 | 2024 |
| [Extended Refusal Direction Analysis: Generalizing Interpretability-Based Jailbreaks](extended-refusal-direction.md) | AML.T0054 | LLM01 | 2025 |
| [Feature Steering Attack: Adversarial Activation Patching to Bypass Safety](feature-steering-attack.md) | AML.T0015 | LLM04 | 2023 |
| [Golden Gate Claude: Identity Distortion via Feature Activation Steering](golden-gate-claude-phenomenon.md) | AML.T0054 | LLM01 | 2024 |
| [Induction Head Manipulation: Exploiting In-Context Learning Circuits](induction-head-manipulation.md) | AML.T0051 | LLM01 | 2022 |
| [Knowledge Circuit Disruption: Attacking Factual Recall Mechanisms in LLMs](knowledge-circuit-disruption.md) | AML.T0015 | LLM09 | 2024 |
| [Logit Lens Exploitation: Using Intermediate Predictions to Reverse-Engineer Safety Logic](logit-lens-exploitation.md) | AML.T0044 | LLM02 | 2024 |
| [Probing Classifier Attacks: Extracting Model Internals via Linear Probes](probing-classifier-attack.md) | AML.T0044 | LLM02 | 2022 |
| [Refusal in Language Models Is Mediated by a Single Direction](refusal-direction-arditi.md) | AML.T0054 | LLM01 | 2024 |
| [RepNoise — Representation Noise Defense Against Harmful Fine-tuning](repnoise-defense.md) | AML.T0020 | LLM04 | 2024 |
| [Representation Engineering for Safety Control in Fine-Tuned LLMs](representation-engineering-safety-control.md) | AML.T0020 | LLM04 | 2023 |
| [Representation Engineering: A Top-Down Approach to AI Transparency and Safety](representation-engineering-safety-bypass.md) | AML.T0054 | LLM01 | 2023 |
| [Representation Engineering: Controlling LLM Behavior via Activation Manipulation](representation-engineering-repe.md) | AML.T0054 | LLM01 | 2023 |
| [Superposition Exploit: Attacking Polysemantic Neurons in Transformer Models](superposition-exploit-polysemantic.md) | AML.T0015 | LLM04 | 2022 |

## 17 — Denial of Service

**10 entries**

| Paper | ATLAS | OWASP | Year |
|---|---|---|---|
| [Adversarial Model Complexity: Nested Attention DoS Attacks](model-complexity-dos.md) | AML.T0034 | LLM10 | 2023 |
| [Adversarial Throughput Degradation via Malicious Batching](adversarial-throughput-degradation.md) | AML.T0034 | LLM10 | 2024 |
| [Context Window Exhaustion Attack: Flooding LLM Memory for Denial of Service](context-window-exhaustion-attack.md) | AML.T0034 | LLM10 | 2024 |
| [Context Window Exhaustion — Denial-of-Service and Context Dilution Attacks](context-window-exhaustion.md) | AML.T0034 | LLM10 | 2023 |
| [Denial of Wallet: Adversarial Cost Amplification in LLM APIs](denial-of-wallet-adversarial.md) | AML.T0034 | LLM10 | 2023 |
| [Energy-Latency Profiling Attacks on LLM APIs](energy-latency-llm-inference.md) | AML.T0034 | LLM10 | 2023 |
| [GPU Side-Channel Attacks on LLM Inference](gpu-side-channel-llm.md) | AML.T0034 | LLM10 | 2024 |
| [Infinite Loop Prompt Attack: Recursive Self-Reference Causing LLM Hang](infinite-loop-prompt-attack.md) | AML.T0034 | LLM10 | 2023 |
| [Rowhammer Attacks on ML Accelerators and Neural Network Weights](rowhammer-ml-accelerators.md) | AML.T0034 | LLM10 | 2021 |
| [Sponge Examples: Energy-Latency Attacks on Neural Networks](sponge-examples-dos.md) | AML.T0034 | LLM10 | 2021 |

## 18 — Defenses & Detection

**6 entries**

| Paper | ATLAS | OWASP | Year |
|---|---|---|---|
| [Erase-and-Check — Certified Defense Against Adversarial Prompts](erase-and-check-defense.md) | AML.T0054 | LLM01 | 2023 |
| [Input Transformation Defenses for Adversarial NLP: Evaluation and Attack Surface](input-transformation-defenses.md) | AML.T0015 | LLM05 | 2022 |
| [LLM-Guard — Open Source Toolkit for LLM Security](llm-guard-defense.md) | AML.T0051 | LLM01 | 2023 |
| [Llama Guard — LLM-Based Input-Output Safeguard for Human-AI Conversations](llama-guard-defense.md) | AML.T0054 | LLM01 | 2023 |
| [Perplexity-Based Filtering — Defense Against Adversarial Suffix Attacks](perplexity-filtering-defense.md) | AML.T0054 | LLM01 | 2023 |
| [SmoothLLM — Defending Against Jailbreaks via Randomized Smoothing](smoothllm-defense.md) | AML.T0054 | LLM01 | 2023 |

## 19 — Benchmarks & Evaluation

**23 entries**

| Paper | ATLAS | OWASP | Year |
|---|---|---|---|
| [AIR-Bench — AI Risk Benchmark for Comprehensive LLM Safety Evaluation](air-bench-benchmark.md) | AML.T0054 | LLM01 | 2024 |
| [ASR Measurement Methodology — Standardizing Attack Success Rate for LLM Security Research](asr-measurement-methodology.md) | AML.T0054 | LLM01 | 2024 |
| [AdvBench — Adversarial Behaviors Benchmark for Evaluating LLM Safety](advbench-benchmark.md) | AML.T0054 | LLM01 | 2023 |
| [AdvScore — Unified Adversarial Scoring Framework for LLM Safety Evaluation](advscore-evaluation.md) | AML.T0054 | LLM01 | 2024 |
| [BELLS — Benchmark for the Evaluation of LLM Supervision Systems](bells-benchmark.md) | AML.T0054 | LLM01 | 2024 |
| [BeaverTails — A Human-Preference Dataset for Improving LLM Safety](beavertails-dataset.md) | AML.T0054 | LLM04 | 2023 |
| [Do-Not-Answer — A Dataset for Evaluating LLM Refusal Behavior](do-not-answer-dataset.md) | AML.T0054 | LLM01 | 2023 |
| [FLIRT Evaluation — Few-Shot Adversarial Red Teaming via In-Context Learning](flirt-evaluation.md) | AML.T0054 | LLM01 | 2023 |
| [G-Eval Adversarial — LLM-Based Evaluation Framework Under Adversarial Conditions](g-eval-adversarial.md) | AML.T0054 | LLM01 | 2023 |
| [HEx-PHI — A Benchmark for Evaluating LLM Adherence to Prohibited Categories](hex-phi-benchmark.md) | AML.T0054 | LLM01 | 2023 |
| [HarmBench — A Standardized Evaluation Framework for Automated Red Teaming](harmbench-benchmark.md) | AML.T0054 | LLM01 | 2024 |
| [Judge Model Robustness — Adversarial Robustness of LLM Evaluation Models](judge-model-robustness.md) | AML.T0054 | LLM01 | 2024 |
| [LLM-as-Judge for Safety Evaluation — Using Language Models to Evaluate Safety Alignment](llm-as-judge-safety.md) | AML.T0054 | LLM01 | 2023 |
| [MT-Bench Adversarial — Multi-Turn Adversarial Evaluation for LLM Safety](mt-bench-adversarial.md) | AML.T0054 | LLM01 | 2023 |
| [Multi-Judge Aggregation — Ensemble Methods for Reliable LLM Safety Assessment](multi-judge-aggregation.md) | AML.T0054 | LLM01 | 2024 |
| [PromptBench: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts](promptbench-robustness-evaluation.md) | AML.T0015 | LLM05 | 2023 |
| [SALAD-Bench — A Hierarchical, Multi-Granular LLM Safety Benchmark](salad-bench-benchmark.md) | AML.T0054 | LLM01 | 2024 |
| [Safety Benchmark Contamination — Training Data Leakage in LLM Safety Evaluations](safety-benchmark-contamination.md) | AML.T0020 | LLM04 | 2024 |
| [SafetyBench — A Comprehensive Safety Evaluation Benchmark for LLMs](safetybench-benchmark.md) | AML.T0054 | LLM01 | 2023 |
| [StrongREJECT Scorer — Fine-Grained Scoring for LLM Refusal Quality](strongreject-scorer.md) | AML.T0054 | LLM01 | 2024 |
| [StrongREJECT — A Scoring Method for Evaluating LLM Jailbreak Success](strongreject-benchmark.md) | AML.T0054 | LLM01 | 2024 |
| [WalledEval — A Comprehensive Safety Evaluation Toolkit for LLMs](walledeval-benchmark.md) | AML.T0054 | LLM01 | 2024 |
| [Win-Rate vs Refusal-Rate Tradeoffs — Measuring the Safety-Helpfulness Frontier in LLMs](win-rate-refusal-tradeoffs.md) | AML.T0054 | LLM01 | 2024 |

## 20 — Red Team Methodology

**6 entries**

| Paper | ATLAS | OWASP | Year |
|---|---|---|---|
| [DeepTeam — Automated Attack Chain Generation for LLM Red Teaming](deepteam-attack-chains.md) | AML.T0054 | LLM01 | 2024 |
| [Lakera Adversarial Testing Methodology — Production AI Security Testing Framework](lakera-adversarial-testing.md) | AML.T0051 | LLM01 | 2024 |
| [Microsoft PyRIT — Python Risk Identification Toolkit for Red Teaming LLMs](microsoft-pyrit-methodology.md) | AML.T0054 | LLM01 | 2024 |
| [NIST AI Red Team Framework — NIST AI 100-1 and AI 600-1 Guidelines](nist-ai-red-team-framework.md) | AML.T0054 | LLM01 | 2024 |
| [Red Teaming Language Models to Reduce Harms — Ganguli et al. (Anthropic)](ganguli-red-teaming.md) | AML.T0054 | LLM01 | 2022 |
| [Red Teaming Language Models with Language Models — Perez et al.](red-teaming-lms-perez.md) | AML.T0054 | LLM01 | 2022 |

## 21 — Emerging 2025-2026 Attacks

**8 entries**

| Paper | ATLAS | OWASP | Year |
|---|---|---|---|
| [Code Interpreter Attacks — Security Vulnerabilities in LLM Code Execution Environments](code-interpreter-attacks.md) | AML.T0062 | LLM06 | 2024 |
| [Extended Thinking Exploitation — Security Analysis of Long-Horizon Reasoning Vulnerabilities](extended-thinking-exploitation.md) | AML.T0051 | LLM06 | 2025 |
| [MoE Routing Attacks — Adversarial Exploitation of Mixture-of-Experts Gating](moe-routing-attacks.md) | AML.T0015 | LLM04 | 2024 |
| [Reasoning Model Attacks — Chain-of-Thought Injection Against o1/o3-Style Models](reasoning-model-attacks.md) | AML.T0051 | LLM01 | 2025 |
| [Speculative Decoding Attacks — Security Implications of Draft-Verify Inference Acceleration](speculative-decoding-attacks.md) | AML.T0015 | LLM04 | 2024 |
| [Structured Output Attacks — JSON/XML Schema Injection Against LLM APIs](structured-output-attacks.md) | AML.T0051 | LLM05 | 2024 |
| [System 2 Reasoning Exploitation — Adversarial Manipulation of Deliberative AI Reasoning](system2-reasoning-exploitation.md) | AML.T0051 | LLM01 | 2025 |
| [Thinking Token Manipulation — Adversarial Control of Visible Reasoning in Extended Thinking Models](thinking-token-manipulation.md) | AML.T0051 | LLM01 | 2025 |

## 22 — Novel Contributions

**5 entries**

| Paper | ATLAS | OWASP | Year |
|---|---|---|---|
| [Adversarial AI in Financial Services — Attack Surfaces in LLM-Powered Banking and Trading](adversarial-ai-financial.md) | AML.T0051 | LLM01 | 2024 |
| [Formal Verification of LLM Safety — Provable Safety Guarantees for Language Models](formal-verification-llm-safety.md) | AML.T0054 | LLM01 | 2023 |
| [Game-Theoretic Attack-Defense Equilibria — Nash Equilibrium Analysis of LLM Security](game-theoretic-attack-defense.md) | AML.T0054 | LLM01 | 2024 |
| [LLM Security in Regulated Environments — Compliance-Aware Deployment Frameworks](llm-security-regulated-environments.md) | AML.T0051 | LLM01 | 2024 |
| [Zero-Day LLM Vulnerabilities — Responsible Disclosure Frameworks for Novel AI Security Flaws](zero-day-llm-vulnerabilities.md) | AML.T0054 | LLM01 | 2024 |

## 23 — Other

**25 entries**

| Paper | ATLAS | OWASP | Year |
|---|---|---|---|
| [Alignment Break: Compromising Safety with a Handful of Fine-Tuning Examples](alignment-break.md) | AML.T0020 | LLM04 | 2023 |
| [BERT-Attack: Adversarial Examples Using BERT for Word Substitution](bert-attack-adversarial.md) | AML.T0015 | LLM05 | 2020 |
| [Benchmarking and Defending Against Indirect Prompt Injection Attacks on LLMs](injecting-llm-applications-benchmarks.md) | AML.T0051 | LLM01 | 2023 |
| [Capability Elicitation and Safety Bypass via Capability-Probing Prompts](capability-elicitation-safety-bypass.md) | AML.T0054 | LLM01 | 2023 |
| [Context Window Stuffing: Distracting and Overloading LLMs via Context Flooding](context-window-stuffing-attack.md) | AML.T0034 | LLM10 | 2023 |
| [Cross-Document Injection Chains: Multi-Hop Attacks on Document-Processing LLM Pipelines](cross-document-injection-chain.md) | AML.T0048 | LLM06 | 2024 |
| [Cross-Session Stored Prompt Injection (XSS-Inspired Persistence)](cross-session-injection.md) | AML.T0051 | LLM01 | 2026 |
| [DP-SGD Evasion: Attacking Differential Privacy Protections in LLM Training](dpsgd-evasion-attack.md) | AML.T0024 | LLM02 | 2022 |
| [Encoding Attacks: Base64, ROT13, and Obfuscated Payloads for LLM Safety Bypass](base64-encoding-bypass-llm.md) | AML.T0054 | LLM01 | 2023 |
| [Factual Hallucination Induction via Training Data Poisoning](factual-hallucination-induction.md) | AML.T0020 | LLM04 | 2023 |
| [Gradient Leakage in Federated LLM Training: Recovering Private Data from Gradients](gradient-leakage-federated-llm.md) | AML.T0024 | LLM02 | 2020 |
| [HotFlip: White-Box Adversarial Examples for Text via Character Flips](hotflip-character-adversarial.md) | AML.T0015 | LLM05 | 2018 |
| [Model Inversion Attacks — Fredrikson et al.](model-inversion-attack.md) | AML.T0044 | LLM02 | 2015 |
| [Refusal Ablation: Mechanistic Removal of the Refusal Circuit](refusal-ablation.md) | AML.T0054 | LLM01 | 2024 |
| [Repurposing Safety Classifiers for Adversarial Attacks](repurposing-safety-models-attacks.md) | AML.T0020 | LLM04 | 2024 |
| [Semantically Equivalent Adversarial Rules: Structure-Preserving NLP Attacks](semantic-preserving-adversarial-nlp.md) | AML.T0015 | LLM05 | 2020 |
| [Side-Channel Timing Attacks on LLM APIs: Token-Level Inference from Latency](side-channel-timing-llm.md) | AML.T0024 | LLM02 | 2024 |
| [TextBugger: Generating Adversarial Text Against Real-World Applications](textbugger-adversarial.md) | AML.T0015 | LLM05 | 2019 |
| [TextFooler: Adversarial Examples for Text Classification via Word Substitution](textfooler-adversarial.md) | AML.T0015 | LLM05 | 2019 |
| [Token-Level Perturbation Attack: Subword Tokenization Exploits for Safety Bypass](token-level-perturbation-attack.md) | AML.T0015 | LLM05 | 2023 |
| [TruthfulQA: Measuring Truthfulness and Epistemic Calibration in LLMs](truthful-qa-alignment.md) | AML.T0054 | LLM09 | 2022 |
| [Universal Adversarial Triggers for Attacking and Analyzing NLP](universal-adversarial-triggers-prompts.md) | AML.T0015 | LLM01 | 2019 |
| [Universal Adversarial Triggers for Attacking and Analyzing NLP](universal-adversarial-triggers.md) | AML.T0015 | LLM01 | 2019 |
| [WebArena Adversarial Security — Vulnerability Analysis of Web Navigation Agents](webarena-adversarial-security.md) | AML.T0051 | LLM06 | 2024 |
| [WildGuard — Open One-Stop Moderation for LLM Safety](wildguard-defense.md) | AML.T0054 | LLM01 | 2024 |

