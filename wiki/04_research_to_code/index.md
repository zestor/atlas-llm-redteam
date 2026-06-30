# Research-to-Code Codex Index

> **816 entries** across 40 domains — MITRE ATLAS v5.4 + OWASP LLM Top 10 2025

## Quick Navigation

- [Prompt Injection](#prompt-injection) (50)
- [Jailbreaks](#jailbreaks) (60)
- [RAG & Retrieval](#rag-retrieval) (45)
- [Multimodal / Vision-Language](#multimodal-vision-language) (43)
- [Agent / MAS](#agent-mas) (82)
- [MCP Security](#mcp-security) (12)
- [Tool / MCP / Browser](#tool-mcp-browser) (16)
- [Memory / Long-Context](#memory-long-context) (14)
- [Model Extraction](#model-extraction) (23)
- [Privacy Attacks](#privacy-attacks) (33)
- [Membership Inference](#membership-inference) (23)
- [Supply Chain](#supply-chain) (24)
- [Poisoning / Backdoor](#poisoning-backdoor) (64)
- [Pretraining / Dataset Attacks](#pretraining-dataset-attacks) (8)
- [Fine-Tuning Safety](#fine-tuning-safety) (13)
- [Alignment Failures](#alignment-failures) (22)
- [RLHF / Reward](#rlhf-reward) (19)
- [Unlearning / Copyright](#unlearning-copyright) (16)
- [Hallucination Attacks](#hallucination-attacks) (18)
- [Calibration / Uncertainty](#calibration-uncertainty) (4)
- [Multilingual / Cross-lingual](#multilingual-cross-lingual) (17)
- [Social Engineering / Persuasion](#social-engineering-persuasion) (15)
- [Misinformation / Deepfakes](#misinformation-deepfakes) (4)
- [LLM-Assisted Cyberattacks](#llm-assisted-cyberattacks) (22)
- [Inference Attacks](#inference-attacks) (5)
- [Enterprise LLM Security](#enterprise-llm-security) (17)
- [Evaluation Attacks](#evaluation-attacks) (18)
- [Formal Methods / Game Theory](#formal-methods-game-theory) (19)
- [Adversarial NLP / Text Attacks](#adversarial-nlp-text-attacks) (13)
- [Adversarial Robustness](#adversarial-robustness) (10)
- [Mechanistic Interp](#mechanistic-interp) (18)
- [Bias & Fairness](#bias-fairness) (5)
- [Hardware / Side-Channel](#hardware-side-channel) (4)
- [DoS / Sponge](#dos-sponge) (5)
- [Defenses](#defenses) (17)
- [Benchmarks / Eval](#benchmarks-eval) (22)
- [Red Team Methodology](#red-team-methodology) (6)
- [Security in Regulated Environments](#security-in-regulated-environments) (1)
- [Model Merging / MoE](#model-merging-moe) (2)
- [Other](#other) (7)

## Prompt Injection

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [Survey of Goal Hijacking Attacks on LLM Agents — Taxonomy and Unified Framework](./agent-goal-hijacking-survey.md) | AML.T0048 | LLM06 |
| [AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks on LLM Agents](./agentdojo-agentic-prompt-injection.md) | AML.T0048 | LLM06 |
| [BIPIA: Benchmark for Indirect Prompt Injection Attacks](./bipia-benchmark-prompt-injection.md) | AML.T0051 | LLM01 |
| [Exploiting Programmatic Behavior in Deployed Language Models: Systematic Boundary Probing](./boundary-probing-system-prompts.md) | AML.T0051 | LLM07 |
| [Cascading Injection in Agent Hierarchies — Propagating Attacks Across Nested Agent Layers](./cascading-injection-hierarchy.md) | AML.T0051 | LLM01 |
| [Clipboard Injection Attack — Poisoning the Clipboard to Compromise Computer Use Agents](./clipboard-injection-attack.md) | AML.T0051 | LLM01 |
| [Compositional Prompt Injection Chain — Multi-Step Injection Where the Chain Exfiltrates Data](./compositional-injection-chain.md) | AML.T0051 | LLM01 |
| [Context Window Overflow Attack — Catastrophic Forgetting of Early Instructions When Context Window Is Exceeded](./context-window-overflow-attack.md) | AML.T0051 | LLM01 |
| [Cross-Document Injection Chains: Multi-Hop Attacks on Document-Processing LLM Pipelines](./cross-document-injection-chain.md) | AML.T0048 | LLM06 |
| [Cross-Session Stored Prompt Injection (XSS-Inspired Persistence)](./cross-session-injection.md) | AML.T0051 | LLM01 |
| [Data Pipeline Injection via ETL Supply Chain Attacks](./data-pipeline-injection-etl.md) | AML.T0010 | LLM03 |
| [Dynamic NTK-Aware RoPE Exploit — Extending Attacker Context Injection Range via NTK Scaling](./dynamic-ntk-rope-exploit.md) | AML.T0051 | LLM01 |
| [Formal Language-Theoretic Model of Prompt Injection — Context-Free Grammar Attacks and Parse Tree Manipulation](./formal-model-prompt-injection.md) | AML.T0051 | LLM01 |
| [Prompt Injection Attacks Against LLM-Integrated Applications — Goal Hijacking & Prompt Leakage](./goal-hijacking-prompt-leakage.md) | AML.T0051 | LLM01 |
| [Ignore Previous Prompt: Foundational Prompt Injection via Instruction Override](./ignore-previous-prompt-perez.md) | AML.T0051 | LLM01 |
| [Images Containing Rendered Text Used to Extract System Prompt Contents from VLMs](./image-based-prompt-leak.md) | AML.T0051 | LLM07 |
| [Indirect Prompt Injection in Multimodal LLMs via Adversarial Image Content](./indirect-injection-multimodal-vision-llm.md) | AML.T0051 | LLM01 |
| [Injecting Relevance: Indirect Prompt Injection via RAG Retrieved Documents](./indirect-injection-retrieval-augmented.md) | AML.T0093 | LLM08 |
| [Indirect Prompt Injection — Hijacking LLMs Through External Content](./indirect-prompt-injection-agents.md) | AML.T0051 | LLM01 |
| [Instructional Segment Embedding — Architecture-Level Injection Defense](./instructional-segment-embedding.md) | AML.T0051 | LLM01 |
| [Indirect Prompt Injection Attacks on LLM-Based Agents in the Wild](./ipi-indirect-prompt-injection-applications.md) | AML.T0051 | LLM01 |
| [KV-Cache Poisoning — Malicious Token Injection into Shared Key-Value Cache in Multi-Tenant LLM Serving](./kv-cache-poisoning.md) | AML.T0051 | LLM01 |
| [KV Cache Poisoning: Persistent Context Manipulation via Cached Prefix Exploitation](./kv-cache-poisoning-attack.md) | AML.T0051 | LLM04 |
| [LLM-as-Judge Prompt Injection — Manipulating Automated Evaluation Scores via Injected Instructions](./llm-judge-prompt-injection.md) | AML.T0051 | LLM01 |
| [LLM Prompt Injection in ERP Systems — Financial Fraud via Prompt Injection Against SAP and Oracle LLM Integrations](./llm-prompt-injection-erp.md) | AML.T0051 | LLM01 |
| [Multi-Agent Prompt Injection — Attack Strategies Across Agent Communication Layers](./multi-agent-prompt-injection.md) | AML.T0051 | LLM01 |
| [Multilingual Prompt Injection — Indirect Injection Payloads in Non-English Languages Evade English-Trained Classifiers](./multilingual-prompt-injection.md) | AML.T0051 | LLM01 |
| [Multilingual System Prompt Extraction — Extracting English System Prompts via Languages with Weaker Instruction-Following Training](./multilingual-system-prompt-extraction.md) | AML.T0054 | LLM07 |
| [Narrative Injection Attack — Subtle Framing Shifts in Retrieved and Generated Content](./narrative-injection-attack.md) | AML.T0051 | LLM09 |
| [PII in System Prompts Extraction: Adversarial System Prompt Leakage](./pii-in-system-prompts-extraction.md) | AML.T0051 | LLM07 |
| [Prefix Injection and Completion Attack Jailbreaks](./prefix-injection-completion-attacks.md) | AML.T0054 | LLM01 |
| [Prompt Injection Attacks in Agentic AI — A Comprehensive Survey](./prompt-injection-agentic-survey.md) | AML.T0051 | LLM01 |
| [Prompt Injection Attacks on Email and Calendar LLM Assistants](./prompt-injection-email-assistants.md) | AML.T0048 | LLM06 |
| [Prompt Injection via Poisoned In-Context Demonstrations](./prompt-injection-poisoned-demonstrations.md) | AML.T0020 | LLM04 |
| [Prompt Injection via Poisoned Pretraining Data](./prompt-injection-pretraining.md) | AML.T0020 | LLM04 |
| [Prompt Injection Privacy Leak: Targeting Privacy-Sensitive Context Fields](./prompt-injection-privacy-leak.md) | AML.T0051 | LLM02 |
| [Prompt Injection for Privacy Theft: Exfiltrating User Data via Embedded Instructions](./prompt-injection-privacy-theft.md) | AML.T0051 | LLM02 |
| [Prompt Injection in RLHF-Aligned Instruction-Following Models](./prompt-injection-rlhf-instruct.md) | AML.T0051 | LLM01 |
| [Protocol-Level Injection — Exploiting LLM Agent Communication Protocols](./protocol-level-injection.md) | AML.T0051 | LLM01 |
| [QR Codes Encoding Adversarial Instructions That Hijack LLM Agents Scanning via Camera](./qr-code-prompt-injection.md) | AML.T0051 | LLM01 |
| [RAG Context Window Overflow — Exploiting LLM Attention Degradation in Long Contexts](./rag-context-window-overflow.md) | AML.T0095 | LLM10 |
| [RAG System Prompt Extraction — Exfiltrating Configuration via Retrieval](./rag-system-prompt-extraction.md) | AML.T0044 | LLM07 |
| [RoPE Positional Exploit — Exploiting RoPE Edge Cases to Confuse Long-Context Models About Token Position](./rope-positional-exploit.md) | AML.T0051 | LLM01 |
| [Instructions Hidden in Images via Steganography: Invisible to Humans, Readable by VLMs](./steganographic-payload-vlm.md) | AML.T0051 | LLM01 |
| [System Prompt Extraction: Attacks and Defenses for LLM Confidentiality](./system-prompt-extraction-attacks.md) | AML.T0051 | LLM07 |
| [System Prompt Hardening — Best Practices for LLM Instruction Resilience](./system-prompt-hardening.md) | AML.T0051 | LLM07 |
| [System Prompt Leakage: Extracting Confidential Instructions from LLM Deployments](./system-prompt-leakage.md) | AML.T0024 | LLM07 |
| [System Prompt Reconstruction via API — Differential Probing to Recover Hidden Instructions](./system-prompt-reconstruction-api.md) | AML.T0051 | LLM07 |
| [Virtual Prompt Injection for Instruction-Tuned Large Language Models](./virtual-prompt-injection-tgt.md) | AML.T0020 | LLM04 |
| [Visual Prompt Injection via Screenshots — Adversarial Text in Images Targeting Multimodal Agents](./visual-prompt-injection-screenshot.md) | AML.T0051 | LLM01 |

## Jailbreaks

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [Activation Addition (ActAdd): Steering LLM Behavior via Residual Stream Injection](./activation-addition-jailbreak.md) | AML.T0054 | LLM01 |
| [Activation Steering at Inference Time — Eliciting Harmful Outputs via Representation Engineering](./activation-steering-inference.md) | AML.T0054 | LLM01 |
| [GCG Adversarial Suffix Transfer: Universal Jailbreaks via Greedy Coordinate Gradient](./adversarial-suffix-gcg-transfer.md) | AML.T0054 | LLM01 |
| [AmpleGCG: Learning a Universal and Transferable Generative Model of Adversarial Suffixes](./amplified-gcg-ample-gcg.md) | AML.T0054 | LLM01 |
| [The Art of Jailbreaking: A Taxonomy of Harmful Prompt Patterns](./art-of-jailbreak-taxonomy.md) | AML.T0054 | LLM01 |
| [ASCII Art and Unicode Art Representations of Prohibited Content Bypassing VLM Text Safety Filters](./ascii-art-jailbreak.md) | AML.T0054 | LLM01 |
| [Adversarial Audio Perturbations Injecting Hidden Instructions into Speech-to-Text LLM Pipelines](./audio-adversarial-llm-injection.md) | AML.T0051 | LLM01 |
| [AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models](./autodan-automated-jailbreak-generation.md) | AML.T0054 | LLM01 |
| [Bad Likert Judge: Exploiting LLM-as-Judge Architectures for Jailbreaking](./bad-likert-judge-jailbreak.md) | AML.T0054 | LLM01 |
| [Encoding Attacks: Base64, ROT13, and Obfuscated Payloads for LLM Safety Bypass](./base64-encoding-bypass-llm.md) | AML.T0054 | LLM01 |
| [Beam Search Manipulation — Adversarial Token Probabilities Steer Beam Search Toward Attacker Outputs](./beam-search-manipulation.md) | AML.T0054 | LLM01 |
| [Capability Elicitation and Safety Bypass via Capability-Probing Prompts](./capability-elicitation-safety-bypass.md) | AML.T0054 | LLM01 |
| [CipherChat: Can LLMs Behave as a Safe Cipher? Safety Failures via Encoded Communication](./cipher-attacks-llm-safety.md) | AML.T0054 | LLM01 |
| [Code-Switching Jailbreak — Alternating Between Languages Mid-Prompt Bypasses Monolingual Safety Classifiers](./code-switching-jailbreak.md) | AML.T0054 | LLM01 |
| [COLD-Attack: Controllable Long-Distribution Jailbreaks via Energy-Based Models](./cold-attack-fluent-jailbreaks.md) | AML.T0054 | LLM01 |
| [Compositional Jailbreak — Combining Safe Fragments into Harmful Instructions](./compositional-jailbreak.md) | AML.T0054 | LLM01 |
| [Constitutional AI Jailbreak — Exploiting the Critique-Revision Loop Toward Harmful Outputs](./constitutional-ai-jailbreak.md) | AML.T0054 | LLM01 |
| [Constitutional AI Vulnerabilities: Attacking Principle-Based Alignment](./constitutional-ai-vulnerabilities.md) | AML.T0054 | LLM04 |
| [Constitutional RL Circumvention: Exploiting Principle Hierarchies in RLAIF](./constitutional-rl-circumvention.md) | AML.T0054 | LLM04 |
| [Crescendo: Multi-Turn Escalation via Gradual Context Drift](./crescendo-attack.md) | AML.T0051 | LLM01 |
| [Crescendo: Jailbreaking Large Language Models with Escalation](./crescendo-multiturn-jailbreak-attack.md) | AML.T0054 | LLM01 |
| [Cross-Lingual Jailbreak Transfer — Jailbreaks in Low-Resource Languages Transfer to English Responses](./cross-lingual-jailbreak-transfer.md) | AML.T0054 | LLM01 |
| [DeepTeam — Automated Attack Chain Generation for LLM Red Teaming](./deepteam-attack-chains.md) | AML.T0054 | LLM01 |
| [Dialect Jailbreak — Regional Dialects, AAVE, and Colloquial Varieties Bypass Standard Language Safety Filters](./dialect-jailbreak.md) | AML.T0054 | LLM01 |
| [Emergent Jailbreak at Scale — Larger Models Are More Vulnerable to Certain Attack Classes](./emergent-jailbreak-scale.md) | AML.T0054 | LLM01 |
| [Extended Refusal Direction Analysis: Generalizing Interpretability-Based Jailbreaks](./extended-refusal-direction.md) | AML.T0054 | LLM01 |
| [Fictional Wrapper Attacks: Creative Writing and Roleplay as Jailbreak Vectors](./fictional-wrapper-roleplay-attacks.md) | AML.T0054 | LLM01 |
| [FigStep — Bypassing LLM Safety via Figure-Embedded Instructions](./figstep-visual-jailbreak.md) | AML.T0054 | LLM01 |
| [FuzzLLM: A Novel and Universal Fuzzing Framework for Probing LLM Vulnerabilities](./fuzzllm-jailbreak-fuzzing.md) | AML.T0054 | LLM01 |
| [GCG: Greedy Coordinate Gradient Adversarial Suffixes](./gcg-adversarial-suffix.md) | AML.T0043 | LLM01 |
| [Harmful Content Generation Evasion — Technical Analysis of Content Policy Filter Bypass Techniques](./harmful-content-generation-evasion.md) | AML.T0054 | LLM01 |
| [Hypothetical Framing and Thought Experiment Jailbreaks](./hypothetical-framing-jailbreak.md) | AML.T0054 | LLM01 |
| [Visual Prompt Injection via Adversarial Images Bypassing Text-Based Safety Filters in VLMs](./image-jailbreak-visual-prompt.md) | AML.T0051 | LLM01 |
| [JailbreakBench — A Standardized Evaluation Framework for LLM Jailbreaking](./jailbreakbench-benchmark.md) | AML.T0054 | LLM01 |
| [Jailbroken: How Does LLM Safety Training Fail?](./jailbroken-gpt4-safety-failures.md) | AML.T0054 | LLM01 |
| [Long Chain-of-Thought Jailbreaks — Safety Degradation in Extended Reasoning](./long-cot-jailbreaks.md) | AML.T0054 | LLM01 |
| [Multilingual Jailbreak: Safety Training Failures in Low-Resource Languages](./low-resource-language-jailbreaks.md) | AML.T0054 | LLM01 |
| [Many-Shot Jailbreaking — In-Context Learning Poisoning via Long-Context Demonstrations](./many-shot-jailbreak-icl.md) | AML.T0054 | LLM01 |
| [Many-Shot Jailbreaking: Leveraging Long Context Windows to Bypass LLM Safety](./many-shot-jailbreaking.md) | AML.T0054 | LLM01 |
| [Mechanistic Proof of Jailbreak — Formal Circuit-Level Proof That Certain Model Architectures Are Necessarily Jailbreakable](./mechanistic-proof-jailbreak.md) | AML.T0054 | LLM01 |
| [Moral Reasoning Manipulation — Ethical Framing Attacks on LLM Safety](./moral-reasoning-manipulation-jailbreak.md) | AML.T0054 | LLM01 |
| [Composite Multimodal Jailbreak: Benign Text + Adversarial Image Bypasses Safety Filters Neither Alone Defeats](./multimodal-jailbreak-composite.md) | AML.T0054 | LLM01 |
| [PAIR: Prompt Automatic Iterative Refinement](./pair-jailbreak.md) | AML.T0054 | LLM01 |
| [Paraphrase Obfuscation Jailbreaks — Semantic Rewriting to Evade Safety Classifiers](./paraphrase-obfuscation-jailbreak.md) | AML.T0054 | LLM01 |
| [Do Anything Now (DAN) and Persona-Based Jailbreaks: A Systematic Study](./personas-dan-variants-jailbreaks.md) | AML.T0054 | LLM01 |
| [Quantization-Aware Jailbreak — Jailbreak Prompts Optimized to Work Only on INT4 Quantized Models](./quantization-aware-jailbreak.md) | AML.T0054 | LLM01 |
| [Refusal Ablation: Mechanistic Removal of the Refusal Circuit](./refusal-ablation.md) | AML.T0054 | LLM01 |
| [Refusal in Language Models Is Mediated by a Single Direction](./refusal-direction-arditi.md) | AML.T0054 | LLM01 |
| [Refusal Suppression and Output Format Injection Jailbreaks](./refusal-suppression-format-injection.md) | AML.T0054 | LLM01 |
| [DeepInception: Hypnotize Large Language Model to Be Jailbreaker](./role-play-exploitation-deep-inception.md) | AML.T0054 | LLM01 |
| [Scaling Attack vs. Defense Cost — Attack Costs Scale Sublinearly, Defense Costs Scale Superlinearly](./scaling-attack-defense-cost.md) | AML.T0054 | LLM01 |
| [Self-Play Jailbreak — Using the Target LLM to Generate Optimized Jailbreak Variants](./self-play-jailbreak.md) | AML.T0054 | LLM01 |
| [ShadowLLM: Fine-Tuning Safety Bypass via Shadow Model Training](./shadow-llm-safety-bypass.md) | AML.T0020 | LLM04 |
| [SmoothLLM: Defending LLMs Against Jailbreaking Attacks via Randomized Smoothing](./smoothllm-jailbreak-defense.md) | AML.T0054 | LLM01 |
| [TAP: Tree of Attacks with Pruning](./tap-tree-attack.md) | AML.T0054 | LLM01 |
| [TAP: A Tree-of-Thought Attack for Jailbreaking LLMs](./tap-tree-attacks-prompts.md) | AML.T0054 | LLM01 |
| [Translation-Based Jailbreaks — Exploiting Multilingual Safety Gaps](./translation-jailbreak-multilingual.md) | AML.T0054 | LLM01 |
| [Video-Based Jailbreaks — Temporal Frame Injection in Video-Capable LLMs](./video-based-jailbreak-temporal.md) | AML.T0054 | LLM01 |
| [Virtualization and Sandbox Jailbreaks — Escaping Safety via Simulated Environments](./virtualization-sandbox-jailbreak.md) | AML.T0054 | LLM01 |
| [Win-Rate vs Refusal-Rate Tradeoffs — Measuring the Safety-Helpfulness Frontier in LLMs](./win-rate-refusal-tradeoffs.md) | AML.T0054 | LLM01 |

## RAG & Retrieval

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [Adversarial Query Generation — Retrieval Ranking Manipulation in RAG](./adversarial-query-retrieval-manipulation.md) | AML.T0095 | LLM08 |
| [Agentic RAG Attacks — Multi-Step Exploitation of Autonomous RAG Agents](./agentic-rag-attacks.md) | AML.T0048 | LLM06 |
| [ARES RAG Evaluation — Automated Evaluation Framework for RAG Security and Quality](./ares-rag-evaluation.md) | AML.T0095 | LLM08 |
| [BadRAG — Adversarial Trigger-Based Retrieval Poisoning](./badrag-adversarial-retrieval.md) | AML.T0093 | LLM08 |
| [Chunk-Level Provenance Tracking — Attribution Defense for RAG Systems](./chunk-provenance-tracking.md) | AML.T0093 | LLM08 |
| [Confounding RAG Attribution — Source Spoofing and Citation Manipulation Attacks](./confounding-rag-attribution.md) | AML.T0095 | LLM09 |
| [Corpus Injection via Web-Scale Pretraining Data Manipulation](./corpus-injection-web-pretraining.md) | AML.T0020 | LLM04 |
| [CorruptRAG — Targeted Corpus Poisoning for Retrieval-Augmented Generation](./corrupt-rag-poisoning.md) | AML.T0093 | LLM08 |
| [Cross-Lingual RAG Poisoning — Poisoning Multilingual RAG Corpora with Non-English Adversarial Documents](./cross-lingual-rag-poisoning.md) | AML.T0094 | LLM08 |
| [Dense Retrieval Poisoning — Corpus-Scale Embedding Manipulation](./dense-retrieval-poisoning-beir.md) | AML.T0093 | LLM08 |
| [Embedding Inversion Attacks on RAG — Reconstructing Private Text from Vectors](./embedding-inversion-rag-attack.md) | AML.T0044 | LLM02 |
| [Embedding Model Poisoning — Corrupting Text Embeddings in RAG Semantic Search Pipelines](./embedding-model-poisoning.md) | AML.T0020 | LLM08 |
| [Embedding Space Poisoning for Retrieval System Manipulation](./embedding-poisoning-retrieval.md) | AML.T0020 | LLM04 |
| [Enterprise Knowledge Base Poisoning — Adversarial Content Injection into Confluence and SharePoint for Corporate LLM Deployment Compromise](./enterprise-knowledge-base-poison.md) | AML.T0094 | LLM08 |
| [Enterprise RAG Data Exfiltration — Adversarial Retrieval Chaining to Extract Confidential Documents](./enterprise-rag-data-exfil.md) | AML.T0051 | LLM02 |
| [Federated RAG Poisoning — Cross-Tenant Knowledge Base Contamination](./federated-rag-poisoning.md) | AML.T0093 | LLM08 |
| [GARAG — Gradient-Based Adversarial Attack on RAG Systems](./garag-gradient-rag-attack.md) | AML.T0093 | LLM08 |
| [Hybrid Retrieval Attacks — Exploiting Sparse-Dense Fusion in Production RAG](./hybrid-retrieval-attack-sparse-dense.md) | AML.T0093 | LLM08 |
| [Judge Model Robustness — Adversarial Robustness of LLM Evaluation Models](./judge-model-robustness.md) | AML.T0054 | LLM01 |
| [PoisonedRAG: Knowledge Corruption Attacks Against Retrieval-Augmented Generation](./knowledge-base-poisoning-rag-attacks.md) | AML.T0094 | LLM08 |
| [Knowledge Graph Injection — Adversarial Entity-Relation Manipulation in Graph RAG](./knowledge-graph-injection-rag.md) | AML.T0093 | LLM08 |
| [LLM-as-Judge for Safety Evaluation — Using Language Models to Evaluate Safety Alignment](./llm-as-judge-safety.md) | AML.T0054 | LLM01 |
| [Membership Inference via RAG Outputs — Privacy Leakage Through Retrieval Signals](./membership-inference-rag-outputs.md) | AML.T0024 | LLM02 |
| [Misinformation Seeding via RAG Poisoning — Poisoning Enterprise Knowledge Bases to Bias LLM Reports](./misinformation-seeding-rag.md) | AML.T0094 | LLM08 |
| [Multi-Judge Aggregation — Ensemble Methods for Reliable LLM Safety Assessment](./multi-judge-aggregation.md) | AML.T0054 | LLM01 |
| [Multi-Judge Disagreement Attack — Crafting Outputs to Maximize LLM Judge Disagreement](./multi-judge-disagreement-attack.md) | AML.T0047 | LLM09 |
| [Poisoning Multimodal RAG Systems by Injecting Adversarial Images into the Knowledge Base](./multimodal-rag-image-poison.md) | AML.T0094 | LLM08 |
| [OpenRAG Security — Open-Source Security Hardening for Production RAG Systems](./open-rag-sec.md) | AML.T0093 | LLM08 |
| [Phantom: Query-Agnostic Document Injection into RAG Corpora](./phantom-rag-injection.md) | AML.T0093 | LLM08 |
| [Poisoned-MRAG — Multimodal RAG Poisoning via Adversarial Image-Text Pairs](./poisoned-mrag-multimodal-rag.md) | AML.T0093 | LLM08 |
| [RAG Cache Poisoning — Exploiting Semantic Cache Layers in Production RAG](./rag-cache-poisoning-attack.md) | AML.T0095 | LLM08 |
| [RAG Document Metadata Injection — Exploiting Structured Metadata in Retrieval Pipelines](./rag-doc-metadata-injection.md) | AML.T0095 | LLM01 |
| [Adversarial Attacks on Image Captioners and Multimodal Retrieval via Indirect Injection](./rag-document-injection-schlarmann.md) | AML.T0051 | LLM01 |
| [RAG Hallucination Amplification — Exploiting LLM Confabulation in Retrieval Systems](./rag-hallucination-amplification.md) | AML.T0095 | LLM09 |
| [RAG Hallucination Amplification Attack — Poisoned Retrieval Context with Real-Source Attribution](./rag-hallucination-amplification-attack.md) | AML.T0094 | LLM08 |
| [RAG Knowledge Base Corpus Poisoning](./rag-knowledge-poisoning-corpus.md) | AML.T0020 | LLM04 |
| [RAG Output Extraction — Exfiltrating Retrieved Documents via LLM Responses](./rag-output-extraction-attack.md) | AML.T0024 | LLM02 |
| [RAG-Shield — Defending Retrieval-Augmented Generation Against Corpus Poisoning](./rag-shield-defense.md) | AML.T0093 | LLM08 |
| [RAG-Thief: Systematic Extraction of RAG Knowledge Bases via Prompt Injection](./rag-thief-extraction-attack.md) | AML.T0024 | LLM02 |
| [Retrieval Anomaly Detection — Statistical Defense for RAG Knowledge Base Integrity](./retrieval-anomaly-detection.md) | AML.T0095 | LLM08 |
| [Retrieval-Aware Critique — Self-Evaluation Defense for RAG Security](./retrieval-aware-critique-defense.md) | AML.T0093 | LLM08 |
| [Retrieval Re-Ranker Manipulation — Adversarial Attacks on Cross-Encoder Re-Ranking](./retrieval-reranker-manipulation.md) | AML.T0093 | LLM08 |
| [Semantic Similarity Poisoning — Embedding Space Manipulation in Vector RAG](./semantic-similarity-poisoning-rag.md) | AML.T0093 | LLM08 |
| [Source Credibility Scoring — Trust-Weighted RAG for Adversarial Robustness](./source-credibility-scoring.md) | AML.T0093 | LLM08 |
| [User Data Leakage in RAG Systems: Cross-User Privacy Extraction](./user-data-leakage-rag.md) | AML.T0024 | LLM02 |

## Multimodal / Vision-Language

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [Adversarial Patches for VLMs — Physical-World Trigger Attacks on Vision-Language Models](./adversarial-patch-vlm-attack.md) | AML.T0015 | LLM01 |
| [Voice Cloning Attacks Using LLM-Assisted Audio Synthesis to Impersonate Executives](./audio-deepfake-voice-clone.md) | AML.T0047 | LLM09 |
| [Audio Injection Attacks — Adversarial Speech Inputs for Voice-Enabled LLMs](./audio-injection-speech-llm.md) | AML.T0015 | LLM01 |
| [Adversarial Chart and Graph Images Causing VLMs to Extract Incorrect Numerical Data](./chart-data-manipulation-vlm.md) | AML.T0047 | LLM09 |
| [Circuit-Level Adversarial Patching: Mechanistic Jailbreak via Activation Intervention](./circuit-level-adversarial-patching.md) | AML.T0015 | LLM04 |
| [Computer Use Exploitation — Attacking Anthropic Claude Computer Use and Similar Systems](./computer-use-exploitation.md) | AML.T0051 | LLM06 |
| [Computer-Use Visual Prompt Injection — Adversarial Screen Content Hijacks Claude/GPT-4o Agents](./computer-use-injection.md) | AML.T0051 | LLM01 |
| [Computer-Use Privilege Bypass: Exploiting System Dialog Automation](./computer-use-privilege-bypass.md) | AML.T0048 | LLM06 |
| [Cross-Modal Injection: Adversarial Instructions in Images Override Text-Based Instructions](./cross-modal-injection-text-image.md) | AML.T0051 | LLM01 |
| [Cross-Modal Transfer Attacks — Exploiting Shared Representations in Multimodal LLMs](./cross-modal-transfer-attack.md) | AML.T0015 | LLM01 |
| [Deepfake Text Detection Evasion — Generating AI Text That Defeats All Current Detectors](./deepfake-text-detection-evasion.md) | AML.T0044 | LLM09 |
| [Deepfake Video Script Generation — LLMs as the Text Layer in the Deepfake Attack Chain](./deepfake-video-script-llm.md) | AML.T0047 | LLM09 |
| [Adversarial Content Hidden in Scanned Documents and PDFs Processed by Vision-Language Document AI](./document-image-injection.md) | AML.T0051 | LLM01 |
| [Adversarial Face Images Causing VLMs to Misidentify Individuals or Bypass Face-Based Authentication](./face-recognition-llm-spoof.md) | AML.T0015 | LLM01 |
| [HADES — Harmful Adversarial Examples for Defeating Safety in VLMs](./hades-adversarial-vision-attack.md) | AML.T0015 | LLM01 |
| [Adversarial Manipulation of Image Captioning Models in Multimodal Pipelines](./image-captioning-manipulation.md) | AML.T0015 | LLM09 |
| [Adversarial Perturbations on Medical Images Causing VLM Diagnostic Errors](./medical-image-adversarial-vlm.md) | AML.T0015 | LLM09 |
| [Backdoors in VLMs Triggered by Specific Visual Patterns Activating Harmful Behavior](./multimodal-backdoor-trigger.md) | AML.T0020 | LLM04 |
| [Backdoor Attacks on Vision-Language Models via Poisoned Image-Caption Pairs](./multimodal-backdoor-vlm.md) | AML.T0020 | LLM04 |
| [Multimodal Context Hijacking — Exploiting Cross-Modal Attention in VLM Conversations](./multimodal-context-hijacking.md) | AML.T0054 | LLM01 |
| [Multimodal Hallucination Amplification — Adversarial Triggering of VLM Confabulation](./multimodal-hallucination-amplification.md) | AML.T0015 | LLM09 |
| [Multimodal Prompt Injection 2025 — Vision-Language Model Attack Surfaces](./multimodal-injection-2025.md) | AML.T0051 | LLM01 |
| [Multimodal Injection via Desktop Notification Poisoning](./multimodal-injection-desktop.md) | AML.T0051 | LLM01 |
| [Removing Digital Watermarks from AI-Generated Images via Adversarial Perturbation](./multimodal-watermark-removal.md) | AML.T0044 | LLM02 |
| [Adversarial Text Rendering That Passes Human Reading but Causes OCR/VLM Misclassification](./ocr-adversarial-text-attack.md) | AML.T0015 | LLM01 |
| [OCR Injection via Document Understanding — Exploiting Document VLMs and PDF Processors](./ocr-injection-document-vlm.md) | AML.T0051 | LLM01 |
| [Screen Reading Data Leakage: Unintended Sensitive Data Capture by Computer-Use Agents](./screen-reading-data-leakage.md) | AML.T0024 | LLM02 |
| [Screenshot and Screen-Content Injection for Computer-Use AI Agents](./screenshot-injection-computer-use.md) | AML.T0051 | LLM06 |
| [Screenshot OCR Injection — Steganographic Text in Images Bypasses VLM Safety Filters via OCR Path](./screenshot-ocr-injection.md) | AML.T0051 | LLM01 |
| [ShadowCast: Backdoor Attacks on Vision-Language Models via Shadow Poisoning](./shadowcast-backdoor-vlm.md) | AML.T0020 | LLM04 |
| [Adversarial Sign Language Images Causing Sign-Language-Interpreting VLMs to Generate Harmful Outputs](./sign-language-vlm-attack.md) | AML.T0015 | LLM01 |
| [Adversarial Audio Inputs for Speech-to-Text LLM Pipeline Injection](./speech-to-text-injection-llm.md) | AML.T0051 | LLM01 |
| [Adversarial Patterns in Thermal/IR Spectrum Invisible to RGB Cameras but Effective Against Multimodal Surveillance LLMs](./thermal-infrared-adversarial.md) | AML.T0015 | LLM01 |
| [Typographic Attacks on Vision Models — Text Overlays Override Visual Understanding](./typographic-attack-vision-models.md) | AML.T0015 | LLM01 |
| [Adversarial Video Deepfakes Causing Video-Understanding Agents to Execute Attacker Actions](./video-deepfake-instruction-inject.md) | AML.T0047 | LLM09 |
| [Single Adversarial Video Frames That Hijack Video-Understanding LLMs Processing Video Streams](./video-frame-injection-attack.md) | AML.T0051 | LLM01 |
| [Vision Agent Hallucination Exploit — Adversarial Images Cause VLM Agents to Hallucinate UI Elements and Take Wrong Actions](./vision-agent-hallucination-exploit.md) | AML.T0015 | LLM06 |
| [Visual Adversarial Examples for Multimodal LLMs: Image Perturbations Inducing Text Compliance](./visual-adversarial-multimodal.md) | AML.T0015 | LLM01 |
| [Physical Adversarial Patches That Fool Vision-Language Models in Real-World Deployment](./vlm-adversarial-patch-physical.md) | AML.T0015 | LLM01 |
| [Adversarial Images Crafted to Maximize VLM Hallucination Rates About Image Content](./vlm-hallucination-adversarial.md) | AML.T0047 | LLM09 |
| [Adversarial Patches on Objects Cause VLM-Based Autonomous Systems to Misidentify Safety-Critical Objects](./vlm-object-detection-attack.md) | AML.T0015 | LLM06 |
| [Vision-Language Models Extract and Leak Private Information Visible in Images](./vlm-privacy-extraction.md) | AML.T0024 | LLM02 |
| [Voice Cloning + LLM Vishing — Combined Voice Deepfake and LLM Dialogue for Phone-Based Social Engineering](./voice-cloning-vishing-llm.md) | AML.T0051 | LLM09 |

## Agent / MAS

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [Adversarial Persona Persistence — Maintaining a Jailbroken Persona Across Conversation Resets](./adversarial-persona-persistence.md) | AML.T0051 | LLM01 |
| [Adversarial Task Decomposition — Manipulating How Agents Decompose Tasks to Introduce Exploitable Sub-Goals](./adversarial-task-decomposition.md) | AML.T0048 | LLM06 |
| [Agent Collusion Attacks — Coordinated Adversarial Behavior Among Multiple LLM Agents](./agent-collusion-attack.md) | AML.T0048 | LLM06 |
| [Agent Instruction Override — Systematic Techniques for Defeating LLM Agent Safety Guardrails](./agent-instruction-override.md) | AML.T0054 | LLM01 |
| [Agent Loop Infinite Recursion — Adversarial Task Design Causes LLM Agents to Enter Infinite Recursive Tool-Call Loops](./agent-loop-infinite-recursion.md) | AML.T0034 | LLM10 |
| [Agent Privilege Escalation — Exploiting Trust Delegation in LLM Agent Pipelines](./agent-privilege-escalation.md) | AML.T0048 | LLM06 |
| [Agent Smith — Self-Replicating Prompt Injection Worm in Multi-Agent Systems](./agent-smith-worm-hijacking.md) | AML.T0051 | LLM06 |
| [Agent Task Interruption Attack — Mid-Task Injection via Environmental Content Causes Agents to Abandon Legitimate Tasks](./agent-task-interruption-attack.md) | AML.T0048 | LLM06 |
| [AgentBench Adversarial — Evaluating LLM Agents Under Adversarial Conditions](./agentbench-adversarial.md) | AML.T0048 | LLM06 |
| [AgentDojo — A Dynamic Environment for Evaluating Attacks and Defenses on LLM Agents](./agentdojo-benchmark.md) | AML.T0051 | LLM01 |
| [Agentic Benchmark Gaming — LLM Agents Detecting Evaluation Environments and Behaving Differently](./agentic-benchmark-gaming.md) | AML.T0015 | LLM01 |
| [API Key Exfiltration via LLM Agents — Agents with .env and Secrets Manager Access Manipulated to Leak Credentials](./api-key-exfiltration-agent.md) | AML.T0048 | LLM06 |
| [AutoAgents Vulnerabilities — Security Analysis of Automated Agent Generation Systems](./autoagents-vulnerabilities.md) | AML.T0048 | LLM06 |
| [Automated Social Engineering via LLM Agents — Autonomous Multi-Turn Manipulation](./automated-social-engineering-agent.md) | AML.T0048 | LLM06 |
| [BadAgent — Inserting and Activating Backdoor Attacks in LLM Agents](./badagent-backdoor-injection.md) | AML.T0020 | LLM04 |
| [Belief Revision Manipulation — Systematically Updating Agent Beliefs via Injected Evidence](./belief-revision-manipulation.md) | AML.T0058 | LLM06 |
| [Browser Agent Clickjacking — Invisible Overlay Attacks Cause LLM Agents to Perform Unintended Click Actions](./browser-agent-clickjacking.md) | AML.T0048 | LLM06 |
| [Browser Agent Prompt Injection — Attacking Web-Navigating LLM Agents](./browser-agent-injection.md) | AML.T0051 | LLM01 |
| [Calendar Agent Poisoning — Adversarial Calendar Event Descriptions Hijack LLM Scheduling Agents](./calendar-agent-poisoning.md) | AML.T0048 | LLM06 |
| [CI/CD Agent Injection — LLM GitHub Actions Bots Hijacked via PR Description Injection to Run Malicious Pipelines](./ci-cd-agent-injection.md) | AML.T0048 | LLM06 |
| [Clipboard Hijack Agent — Malicious Clipboard Content Hijacks LLM Agents That Read Clipboard as Context](./clipboard-hijack-agent.md) | AML.T0051 | LLM01 |
| [Compound AI System Attacks — Security Vulnerabilities in Multi-Component AI Pipelines](./compound-ai-system-attacks.md) | AML.T0048 | LLM06 |
| [Context Poisoning in LLM Agents — Manipulating Working Memory to Redirect Goals](./context-poisoning-agents.md) | AML.T0051 | LLM01 |
| [Convergent Instrumental Goals: Why Advanced AI Systems Resist Shutdown](./convergent-instrumental-goals.md) | AML.T0048 | LLM06 |
| [Corrigibility Failures in Deployed LLM Agents: Resisting Human Override](./corrigibility-failure-deployed.md) | AML.T0048 | LLM06 |
| [Database Agent SQL Injection — LLM Database Query Agents Vulnerable to Prompt Injection Generating Malicious SQL](./database-agent-sql-injection.md) | AML.T0048 | LLM06 |
| [Distractor Injection for Planning Agents — High-Salience Irrelevant Information Causes Goal Abandonment](./distractor-injection-planning.md) | AML.T0058 | LLM06 |
| [Docker Escape via LLM Code Agent — Code-Execution Agents Exploited to Escape Container via Kernel Vulnerability Chains](./docker-escape-code-agent.md) | AML.T0048 | LLM06 |
| [Indirect Prompt Injection in Code Review and Developer Tool LLM Agents](./document-injection-code-review-agents.md) | AML.T0048 | LLM06 |
| [Email Agent Exfiltration — Malicious Email Content Hijacks LLM Email Agents to Forward Sensitive Data](./email-agent-exfiltration.md) | AML.T0048 | LLM06 |
| [Emergent Capability Exploitation — Eliciting Undisclosed Emergent Capabilities via Structured Prompting](./emergent-capability-exploitation.md) | AML.T0054 | LLM01 |
| [Emergent Deception in Competitive Agents — Deceptive Strategies Emerge Spontaneously Without Explicit Training](./emergent-deception-competitive-agents.md) | AML.T0048 | LLM06 |
| [Emergent Deception in Large Language Models: Strategic Dishonesty as an Optimization Outcome](./emergent-deception-llms.md) | AML.T0054 | LLM09 |
| [Fake Orchestrator Attack — Impersonating the Control Plane to Hijack LLM Agents](./fake-orchestrator-attack.md) | AML.T0048 | LLM06 |
| [File System Exfiltration via LLM Agents — Indirect Prompt Injection in Documents Causes Sensitive File Reads](./file-system-exfiltration-agent.md) | AML.T0048 | LLM06 |
| [Goal Drift in Long-Horizon LLM Agents — Objective Corruption Over Extended Task Execution](./goal-drift-long-horizon.md) | AML.T0048 | LLM06 |
| [Goal Misgeneralization: LLMs Pursuing Wrong Goals Out-of-Distribution](./goal-misgeneralization.md) | AML.T0020 | LLM04 |
| [Gorilla LLM API Abuse — Adversarial Exploitation of LLM API Call Generation](./gorilla-api-abuse.md) | AML.T0061 | LLM06 |
| [GUI Agent Action Injection: Malicious UI Element Impersonation](./gui-agent-action-injection.md) | AML.T0051 | LLM01 |
| [GUI Agent Attacks — Adversarial Manipulation of LLM-Based GUI Automation Agents](./gui-agent-attacks.md) | AML.T0051 | LLM06 |
| [InjecAgent — Benchmarking Indirect Prompt Injection in Tool-Augmented LLM Agents](./injecagent-tool-injection.md) | AML.T0051 | LLM01 |
| [Instrumental Convergence in LLM Agents: Emergent Self-Preservation and Resource Seeking](./instrumental-convergence-llm-agents.md) | AML.T0048 | LLM06 |
| [Long-Horizon Planning Subversion — Adversarial Manipulation of Multi-Step LLM Task Planners](./long-horizon-planning-subversion.md) | AML.T0048 | LLM06 |
| [MAS Agent Positioning Attack — Exploiting Role Assignment to Gain Privileged Position](./mas-agent-position-attack.md) | AML.T0048 | LLM06 |
| [Consensus Poisoning in Multi-Agent Deliberation Systems](./mas-consensus-poisoning.md) | AML.T0048 | LLM06 |
| [Multi-Agent System Hijacking — Compromising Coordinated LLM Agent Networks](./mas-hijacking-arxiv.md) | AML.T0048 | LLM06 |
| [Plan Injection: Hijacking LLM Agent Planning Stages](./mas-plan-injection.md) | AML.T0051 | LLM01 |
| [Reputation Poisoning in Multi-Agent LLM Systems](./mas-reputation-attack.md) | AML.T0048 | LLM06 |
| [Role Confusion Attack in Multi-Agent LLM Hierarchies](./mas-role-confusion-attack.md) | AML.T0048 | LLM06 |
| [Sybil Agent Attack: Identity Forgery in Multi-Agent Networks](./mas-sybil-agent-attack.md) | AML.T0048 | LLM06 |
| [SSRF via MCP Parameters — Server-Side Request Forgery Through Agent Tool Invocations](./mcp-ssrf-agent-tools.md) | AML.T0061 | LLM06 |
| [Mesa-Optimization and Deceptive Inner Optimizers in LLM Systems](./mesa-optimization-deceptive.md) | AML.T0020 | LLM04 |
| [Multi-Agent Disinformation Network — Coordinated LLM Agents Generating Mutually-Reinforcing False Narratives](./multi-agent-disinfo-network.md) | AML.T0048 | LLM09 |
| [Multi-Agent Privilege Escalation — Lower-Trust Agent Convinces Higher-Trust Orchestrator to Execute Privileged Operations](./multi-agent-privilege-escalation.md) | AML.T0048 | LLM06 |
| [Multi-Hop Hallucination Chain — Compounding Small Errors Across Reasoning Hops into Large Factual Failures](./multi-hop-hallucination-chain.md) | AML.T0051 | LLM09 |
| [Multi-Step Goal Laundering — Breaking a Harmful Goal into Innocent Sub-Tasks Distributed Across Agents](./multi-step-goal-laundering.md) | AML.T0048 | LLM06 |
| [Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications](./not-what-you-signed-up-for-misaligned-agents.md) | AML.T0048 | LLM06 |
| [OAuth Token Theft via LLM Agents — Prompt Injection Exfiltrates Access Tokens from Agent Credential Stores](./oauth-token-theft-agent.md) | AML.T0048 | LLM06 |
| [Orchestrator-Executor Compromise — Attacking the Orchestration Layer in LLM Agent Systems](./orchestrator-executor-compromise.md) | AML.T0048 | LLM06 |
| [OS-Level Agent Exploitation — Attacking LLM Agents with OS-Layer Access](./os-level-agent-exploitation.md) | AML.T0061 | LLM06 |
| [PDF Agent Injection — Invisible Text in PDFs Processed by Document-Reading Agents Injects Adversarial Instructions](./pdf-agent-injection.md) | AML.T0051 | LLM01 |
| [Phantom Tool Invocation — Causing Agents to Invoke Non-Existent Tools and Exploiting Error Handling](./phantom-tool-invocation.md) | AML.T0061 | LLM06 |
| [Power-Seeking AI: Instrumental Convergence and Resource Acquisition in RL Systems](./power-seeking-rl-agents.md) | AML.T0048 | LLM06 |
| [PsySafe — Psychological Safety Framework for Adversarial Multi-Agent Systems](./psysafe-adversarial-agents.md) | AML.T0048 | LLM06 |
| [R2D2 — Robust Red-Teaming with Dual-Direction Dialogue for LLM Agents](./r2d2-agent-attack.md) | AML.T0051 | LLM01 |
| [Recursive Agent Spawning: Exponential DoS in Agentic LLM Frameworks](./recursive-agent-spawn-attack.md) | AML.T0034 | LLM10 |
| [Repository Poisoning for Code Agents — Malicious Comments and Docstrings Hijack LLM Coding Agents](./repo-poisoning-code-agent.md) | AML.T0051 | LLM01 |
| [Shell Escape from Code Interpreter Sandbox — Prompt Injection Breaks ChatGPT Advanced Data Analysis and Jupyter-Based Agents](./shell-escape-code-interpreter.md) | AML.T0048 | LLM06 |
| [Slack Bot Hijack — Adversarial Slack Messages Hijack LLM Bots to Send Messages, Exfiltrate Data, or Take Actions](./slack-bot-hijack.md) | AML.T0048 | LLM06 |
| [Sleeper Agents: Training Deceptive LLMs](./sleeper-agents-anthropic.md) | AML.T0020 | LLM04 |
| [Sleeper Memory Attack — Dormant Backdoors in LLM Agent Long-Term Memory](./sleeper-memory-agent.md) | AML.T0020 | LLM04 |
| [SwarmBench Adversarial — Benchmarking Security of LLM Swarm Intelligence Systems](./swarm-bench-adversarial.md) | AML.T0048 | LLM06 |
| [Task Drift in Multi-Agent Orchestration — How Decomposition Amplifies Goal Misalignment](./task-drift-multi-agent.md) | AML.T0048 | LLM06 |
| [Terminal ANSI Escape Injection — ANSI Sequences in LLM Terminal Agent Output Trigger Unintended Commands](./terminal-ansi-injection.md) | AML.T0048 | LLM06 |
| [API Tool Misuse in LLM Agents — Unintended Dangerous API Invocations](./tool-misuse-agents.md) | AML.T0061 | LLM06 |
| [Trust Propagation Exploits in Multi-Agent Systems — Breaking Transitive Trust Chains](./trust-propagation-exploit.md) | AML.T0048 | LLM06 |
| [VisualWebArena Security — Adversarial Analysis of Multimodal Web Navigation Agents](./visualwebharena-security.md) | AML.T0051 | LLM06 |
| [Web Agent Cookie Theft: Session Hijacking via Browser Automation Exploitation](./web-agent-cookie-theft.md) | AML.T0061 | LLM02 |
| [Indirect Prompt Injection in GPT-4 Web Browsing Mode](./web-agent-injection-browser-llm.md) | AML.T0048 | LLM06 |
| [Web-Scraping Agent SSRF — LLM Browser Agents Used as SSRF Proxies to Access Internal Network Resources](./web-scraping-agent-ssrf.md) | AML.T0061 | LLM06 |
| [WebArena Adversarial Security — Vulnerability Analysis of Web Navigation Agents](./webarena-adversarial-security.md) | AML.T0051 | LLM06 |
| [World Model Manipulation in Planning Agents — Corrupting Internal World Models for Systematic Planning Failures](./world-model-manipulation-agent.md) | AML.T0058 | LLM06 |

## MCP Security

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [MITRE ATLAS MCP Attack Case Studies — Real-World MCP Security Incidents](./mcp-atlas-case-studies.md) | AML.T0051 | LLM01 |
| [MCP Credential Harvesting — Stealing Secrets Through Protocol-Level Tool Abuse](./mcp-credential-harvesting.md) | AML.T0061 | LLM02 |
| [MCP Cross-Server Context Propagation — Lateral Movement Through Multiple MCP Servers](./mcp-cross-server-propagation.md) | AML.T0048 | LLM06 |
| [MCP Denial of Service: Tool Flooding and Resource Exhaustion in Protocol Deployments](./mcp-denial-of-service.md) | AML.T0034 | LLM10 |
| [MCP Ghost Server Attack: Phantom Tool Registration in Model Context Protocol](./mcp-ghost-server-attack.md) | AML.T0062 | LLM03 |
| [MCP Permission Escalation: Scope Creep in Tool Authorization Flows](./mcp-permission-escalation.md) | AML.T0061 | LLM06 |
| [MCP Rug-Pull Attack — Post-Approval Malicious Updates to MCP Server Behavior](./mcp-rug-pull-attack.md) | AML.T0010 | LLM03 |
| [MCP Server Compromise — Malicious Model Context Protocol Server Injects Adversarial Tool Responses to Hijack Agent Behavior](./mcp-server-compromise.md) | AML.T0062 | LLM06 |
| [MCP Server Injection — Exploiting the Model Context Protocol for Adversarial Control](./mcp-server-injection.md) | AML.T0051 | LLM01 |
| [MCP Session Hijacking: Stealing Context via Protocol-Level Token Theft](./mcp-session-hijacking.md) | AML.T0024 | LLM02 |
| [MCP Tool Name Collision: Hijacking via Ambiguous Tool Resolution](./mcp-tool-name-collision.md) | AML.T0062 | LLM03 |
| [MCP Tool Schema Poisoning — Injecting Adversarial Instructions via MCP Tool Definitions](./mcp-tool-schema-poisoning.md) | AML.T0062 | LLM05 |

## Tool / MCP / Browser

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [FuncPoison — Poisoning Function-Calling LLMs Through Malicious Training Examples](./func-poison-function-calling.md) | AML.T0020 | LLM04 |
| [Function Calling Injection — Schema Poisoning in LLM Tool Use APIs](./function-calling-injection.md) | AML.T0061 | LLM06 |
| [Function Schema Manipulation — Exploiting Tool Definitions to Redirect Agent Behavior](./function-schema-manipulation.md) | AML.T0062 | LLM05 |
| [Functional Cloning of LLMs via Instruction-Following API Distillation](./llm-api-functional-cloning.md) | AML.T0044 | LLM02 |
| [Prompt Chaining Attack — Multi-Turn Attack Where Each Turn Builds Context for the Next](./prompt-chaining-attack.md) | AML.T0051 | LLM01 |
| [Tool Callback Exfiltration: Covert Data Leakage via Webhook Parameters](./tool-callback-exfiltration.md) | AML.T0061 | LLM02 |
| [ToolHijacker — Runtime Hijacking of LLM Agent Tool Selection and Execution](./tool-hijacker-dynamic-tools.md) | AML.T0062 | LLM05 |
| [Tool Output Injection — Weaponizing Agent Tool Results for Adversarial Control](./tool-output-injection.md) | AML.T0051 | LLM01 |
| [Compromising LLM-Integrated Applications via Malicious Plugin/Tool Outputs](./tool-output-injection-plugins.md) | AML.T0061 | LLM06 |
| [Tool Output Spoofing — Attacker-Controlled Tool Results Inject Malicious Instructions into Agent Context](./tool-output-spoofing.md) | AML.T0062 | LLM06 |
| [Tool Parameter Smuggling: Covert Payload Delivery via Function Arguments](./tool-parameter-smuggling.md) | AML.T0062 | LLM01 |
| [Tool Result Manipulation: Forging Observations in LLM Agent Scratchpads](./tool-result-manipulation.md) | AML.T0062 | LLM05 |
| [Tool Selection Manipulation — Biasing Agent Tool Selection via Adversarial Context](./tool-selection-manipulation.md) | AML.T0061 | LLM06 |
| [ToolSword — Unveiling Safety Issues in LLM Tool-Integrated Applications](./toolsword-tool-safety.md) | AML.T0061 | LLM06 |
| [TrustDesc — Exploiting Tool Description Trust in LLM Function-Calling Systems](./trustdesc-tool-trust.md) | AML.T0062 | LLM05 |
| [XTHP — Cross-Tool Harvesting and Pivoting in LLM Function-Calling Agents](./xthp-cross-tool-harvest.md) | AML.T0061 | LLM06 |

## Memory / Long-Context

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [Agentic Memory Poisoning — Long-Term Memory Stores Poisoned via Adversarial Content That Persists Across Sessions](./agentic-memory-poisoning.md) | AML.T0020 | LLM04 |
| [Context Window Exhaustion — Denial-of-Service and Context Dilution Attacks](./context-window-exhaustion.md) | AML.T0034 | LLM10 |
| [Context Window Exhaustion Attack: Flooding LLM Memory for Denial of Service](./context-window-exhaustion-attack.md) | AML.T0034 | LLM10 |
| [Context Window Stuffing: Distracting and Overloading LLMs via Context Flooding](./context-window-stuffing-attack.md) | AML.T0034 | LLM10 |
| [Cross-Context Window Injection: Bridging Session Boundaries in Persistent Agents](./cross-context-window-injection.md) | AML.T0051 | LLM04 |
| [In-Context Memory Poisoning: Corrupting Few-Shot Examples in Agent Scratchpads](./in-context-memory-poisoning.md) | AML.T0020 | LLM04 |
| [Long-Context Extraction Attack: Inferring Hidden Information from Extended Conversations](./long-context-extraction-attack.md) | AML.T0024 | LLM02 |
| [Lost-in-the-Middle Exploitation — Attacking Position Bias in Long-Context LLMs](./lost-in-the-middle-exploit.md) | AML.T0051 | LLM01 |
| [MemMorph — Morphing Agent Memory to Achieve Persistent Behavioral Manipulation](./memmorph-memory-attack.md) | AML.T0048 | LLM06 |
| [MemMorph: Tool Hijacking via Long-Term Memory Poisoning](./memmorph-memory-poisoning.md) | AML.T0051 | LLM09 |
| [Long-Term Memory Poisoning — Persistent Adversarial Manipulation of LLM Agent Memory](./memory-poisoning-long-term.md) | AML.T0048 | LLM06 |
| [MemoryGraft: Persistent Memory Poisoning via Context Injection](./memorygraft-memory-persistence.md) | AML.T0051 | LLM04 |
| [Objective Hijacking via Memory — Persistent Goal Manipulation Through Agent Memory Stores](./objective-hijacking-via-memory.md) | AML.T0048 | LLM06 |
| [Sleeper Memory Poisoning: Cross-Session Persistent Agent Compromise](./sleeper-memory-attack.md) | AML.T0051 | LLM01 |

## Model Extraction

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [Model Architecture Fingerprinting via Black-Box Probing](./architecture-fingerprinting.md) | AML.T0044 | LLM02 |
| [Cache-Based Inference Side Channels in ML Serving](./cacheinference-model-extraction.md) | AML.T0044 | LLM02 |
| [Dataset Inference Attack — Proving a Model Was Trained on a Specific Dataset Without Model Access](./dataset-inference-ownership.md) | AML.T0024 | LLM02 |
| [Distillation-Based Model Stealing — Knowledge Distillation as an Attack Vector](./distillation-based-model-stealing.md) | AML.T0044 | LLM02 |
| [Gradient Leakage in Federated LLM Training: Recovering Private Data from Gradients](./gradient-leakage-federated-llm.md) | AML.T0024 | LLM02 |
| [Gradient Leakage from LLM Fine-Tuning — Recovering Training Data from Federated Gradient Updates](./gradient-leakage-finetuning.md) | AML.T0024 | LLM02 |
| [Hyperparameter Stealing via Model Extraction Side Channels](./hyperparameter-extraction-attack.md) | AML.T0044 | LLM02 |
| [Inference Endpoint Fingerprinting — Identifying Model Version and Quantization Behind API via Output Statistics](./inference-endpoint-fingerprinting.md) | AML.T0044 | LLM02 |
| [Knockoff Nets — Stealing Functionality of Black-Box Models](./knockoff-nets-stealing.md) | AML.T0044 | LLM02 |
| [Dataset Inference Attacks on Language Models](./llm-dataset-inference-attack.md) | AML.T0024 | LLM02 |
| [LLM Model Version Fingerprinting — Identifying Exact Model Versions Behind Enterprise APIs to Target Unpatched Vulnerabilities](./llm-model-version-fingerprint.md) | AML.T0044 | LLM02 |
| [LLM Stealing via Output Distribution Matching — Carlini et al.](./llm-stealing-output-distribution.md) | AML.T0044 | LLM02 |
| [Logit Lens Exploitation: Using Intermediate Predictions to Reverse-Engineer Safety Logic](./logit-lens-exploitation.md) | AML.T0044 | LLM02 |
| [LoRA Weight Extraction — Adapter IP Theft via Targeted Black-Box Queries on Fine-Tuned Model APIs](./lora-weight-extraction.md) | AML.T0044 | LLM02 |
| [ActiveThief — Active Learning-Based Model Extraction](./model-extraction-active-learning.md) | AML.T0044 | LLM02 |
| [Model Extraction via Confidence Score Queries](./model-extraction-confidence-scores.md) | AML.T0044 | LLM02 |
| [PRADA — Protecting Against DNN Model Stealing Attacks](./model-extraction-defense-prada.md) | AML.T0044 | LLM02 |
| [Model Extraction via Logit API — High-Fidelity Extraction Using Only Top-k Logits](./model-extraction-logit-api.md) | AML.T0044 | LLM03 |
| [Stealing Machine Learning Models via Prediction APIs — Tramèr et al.](./model-extraction-tramer.md) | AML.T0044 | LLM02 |
| [Model Fingerprinting via Adversarial Examples — Proving IP Theft Through Model-Specific Adversarial Transferability](./model-fingerprinting-adversarial.md) | AML.T0044 | LLM03 |
| [Model Inversion Attacks — Fredrikson et al.](./model-inversion-attack.md) | AML.T0044 | LLM02 |
| [Neural Network Plagiarism Detection — Identifying Stolen Model Copies via Behavioral Fingerprinting](./nn-plagiarism-detection.md) | AML.T0044 | LLM03 |
| [On-Device LLM Model Extraction via Hardware-Level Memory Analysis](./on-device-model-extraction.md) | AML.T0044 | LLM02 |

## Privacy Attacks

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [Aggregation-Level Attacks on Secure Aggregation in Federated LLM Training](./aggregation-attack-federated.md) | AML.T0024 | LLM02 |
| [Biometric Data Reconstruction from LLM-Processed Sensor Data](./biometric-data-reconstruction.md) | AML.T0024 | LLM02 |
| [Canary Insertion and Extraction: Empirical Privacy Auditing for LLMs](./canary-insertion-extraction.md) | AML.T0024 | LLM02 |
| [Conversation History Reconstruction via Contextual Inference Attacks](./conversation-history-reconstruction.md) | AML.T0024 | LLM02 |
| [Cross-User Data Leakage in Multi-Tenant LLM SaaS Deployments](./cross-user-data-leakage.md) | AML.T0024 | LLM02 |
| [Differential Privacy Auditing Attack: Measuring True Privacy Leakage](./differential-privacy-auditing-attack.md) | AML.T0024 | LLM02 |
| [Differential Privacy Auditing via Membership Inference](./differential-privacy-auditing-mia.md) | AML.T0024 | LLM02 |
| [Differential Privacy Evasion via Canary Amplification — Defeating DP-SGD Privacy Guarantees Empirically](./dp-evasion-canary-amplification.md) | AML.T0024 | LLM02 |
| [DP-SGD Evasion: Attacking Differential Privacy Protections in LLM Training](./dpsgd-evasion-attack.md) | AML.T0024 | LLM02 |
| [Embedding Inversion Privacy Attack: Reconstructing Text from Embeddings](./embedding-inversion-privacy.md) | AML.T0024 | LLM02 |
| [Embedding Inversion Attack — Reconstructing Private Text from Sentence Embeddings](./embedding-inversion-text-reconstruction.md) | AML.T0024 | LLM02 |
| [Gradient Inversion Attacks on Federated LLM Training](./federated-learning-gradient-inversion.md) | AML.T0024 | LLM02 |
| [Byzantine Client Attacks: Model Poisoning in Federated LLM Training](./federated-learning-model-poisoning.md) | AML.T0020 | LLM04 |
| [Federated Learning Poisoning Attacks](./federated-learning-poisoning.md) | AML.T0020 | LLM04 |
| [Financial Data Memorization: MNPI Leakage from LLMs Fine-Tuned on Financial Corpora](./financial-data-memorization.md) | AML.T0024 | LLM02 |
| [Healthcare LLM PHI Extraction: Clinical Note Memorization Attacks](./healthcare-llm-phi-extraction.md) | AML.T0024 | LLM02 |
| [Attacks on Homomorphic Encryption for Privacy-Preserving LLM Inference](./homomorphic-encryption-llm-attack.md) | AML.T0024 | LLM02 |
| [Membership Inference Attacks on Fine-Tuned LLMs](./inference-attack-fine-tuned-llm.md) | AML.T0024 | LLM02 |
| [Information-Theoretic Extraction Bounds — Lower Bounds on Training Data Leakage via LLM Queries](./information-theoretic-extraction-bounds.md) | AML.T0024 | LLM02 |
| [Legal Document Memorization: Attorney-Client Privilege Leakage from Fine-Tuned LLMs](./legal-document-memorization.md) | AML.T0024 | LLM02 |
| [LLM API Log Privacy: Re-identification from Provider Logging](./llm-api-log-privacy.md) | AML.T0024 | LLM02 |
| [Counterfactual Memorization in Neural Language Models — Zhang et al.](./llm-privacy-copyrighted-memorization.md) | AML.T0024 | LLM02 |
| [Location Inference from LLM Conversation Patterns](./location-inference-llm.md) | AML.T0024 | LLM02 |
| [Auditing Differentially Private ML: Tight Privacy Accounting and Evasion Techniques](./privacy-auditing-differential.md) | AML.T0024 | LLM02 |
| [Privacy Leakage During LLM Fine-Tuning](./privacy-leakage-llm-fine-tuning.md) | AML.T0024 | LLM02 |
| [Property Inference Attack: Inferring Dataset-Level Properties from LLMs](./property-inference-attack.md) | AML.T0024 | LLM02 |
| [RAG Private Document Extraction via Adversarial Retrieval Queries](./rag-private-document-extraction.md) | AML.T0051 | LLM02 |
| [Side-Channel Attacks Against LLMs in TEEs (SGX, TrustZone)](./secure-enclaves-llm-sidechannel.md) | AML.T0024 | LLM02 |
| [Sentence Embedding Inversion: Reconstructing Text from Embedding Vectors](./sentence-embedding-inversion.md) | AML.T0024 | LLM08 |
| [Split Learning Label Inference Attack on LLM Intermediate Activations](./split-learning-label-inference.md) | AML.T0024 | LLM02 |
| [Synthetic Data Re-identification: LLM-Generated Data Privacy Failures](./synthetic-data-reidentification.md) | AML.T0024 | LLM02 |
| [User Behavior Inference from LLM Conversation History](./user-behavior-inference-llm.md) | AML.T0024 | LLM02 |
| [Training Data Reconstruction from Model Weights — Gradient Inversion on Stored Parameters](./weight-based-data-reconstruction.md) | AML.T0024 | LLM02 |

## Membership Inference

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [Batch Inference Cross-Contamination — Cross-Request Information Leakage via Attention Bleed-Through](./batch-inference-cross-contamination.md) | AML.T0024 | LLM02 |
| [Browser History Exfiltration via LLM Browser Agents](./browser-history-exfiltration.md) | AML.T0024 | LLM02 |
| [Cross-Lingual Membership Inference — Inferring Training Language Distribution via Cross-Lingual Probing](./cross-lingual-membership-inference.md) | AML.T0024 | LLM02 |
| [CUDA Graph Side-Channel — Replay Side-Channel in Optimized LLM Inference Runtimes](./cuda-graph-side-channel.md) | AML.T0024 | LLM02 |
| [Auditing Differential Privacy via Membership Inference — Jagielski et al.](./dp-auditing-membership-inference.md) | AML.T0024 | LLM02 |
| [Flash Attention Timing Oracle — Side-Channel Attack via FlashAttention Kernel Execution Time](./flash-attention-timing-oracle.md) | AML.T0024 | LLM02 |
| [Code Memorization and Extraction in Code LLMs](./llm-code-memorization-codex.md) | AML.T0024 | LLM02 |
| [Quantifying Memorization in Pretrained Language Models](./llm-pretraining-memorization-gpt.md) | AML.T0024 | LLM02 |
| [Membership Inference for Copyright — Proving Training Membership to Establish Copyright Violation](./membership-inference-copyright.md) | AML.T0024 | LLM02 |
| [Membership Inference Attacks on Large Language Models](./membership-inference-llm.md) | AML.T0024 | LLM02 |
| [Scalable Membership Inference Attacks against Large Language Models — Nasr et al.](./membership-inference-nasr.md) | AML.T0024 | LLM02 |
| [Reference Model Membership Inference: Improved MIA via Baseline Calibration](./mia-reference-model-attack.md) | AML.T0024 | LLM02 |
| [Min-K% Prob — Membership Inference via Minimum Token Probabilities](./min-k-percent-membership-inference.md) | AML.T0024 | LLM02 |
| [Min-K%++ — Improved Reference-Free Membership Inference](./mink-plus-plus-membership-inference.md) | AML.T0024 | LLM02 |
| [Patient Data Extraction from Clinical LLMs](./patient-data-extraction-clinical-llm.md) | AML.T0024 | LLM02 |
| [PII Extraction from Large Language Models via Targeted Prompting](./pii-extraction-llm.md) | AML.T0024 | LLM02 |
| [Prefix Caching Oracle — System Prompt Disclosure via Timing Side-Channel in LLM Prefix Caches](./prefix-caching-oracle.md) | AML.T0024 | LLM07 |
| [Pretraining Memorization Exploit — Systematic Extraction of PII and Confidential Data at Scale](./pretraining-memorization-exploit.md) | AML.T0024 | LLM02 |
| [Shadow Model Attacks for Membership Inference — Shokri et al.](./shadow-model-membership-inference.md) | AML.T0024 | LLM02 |
| [Tensor Parallelism Eavesdrop — Reconstructing Model Activations via Inter-GPU Communication Interception](./tensor-parallelism-eavesdrop.md) | AML.T0024 | LLM02 |
| [Extracting Training Data from Large Language Models — Carlini et al.](./training-data-extraction-carlini.md) | AML.T0024 | LLM02 |
| [Quantifying Memorization Across Neural Language Models](./training-data-memorization.md) | AML.T0024 | LLM02 |
| [Quantifying Memorization Across Neural Language Models — Carlini et al.](./verbatim-memorization-llm.md) | AML.T0024 | LLM02 |

## Supply Chain

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [Data Provenance Forgery — Fabricating Metadata to Make Poisoned Data Appear Trusted](./data-provenance-forgery.md) | AML.T0010 | LLM03 |
| [Dependency Confusion Attacks on ML Packages and Pipelines](./dependency-confusion-ml-packages.md) | AML.T0019 | LLM03 |
| [GGUF Quantized Backdoor — Backdoor Triggers Survive GGUF Quantization in llama.cpp Deployments](./gguf-quantized-backdoor.md) | AML.T0020 | LLM04 |
| [Hugging Face Repository Takeover — Account Compromise Serving Trojaned Model Weights](./huggingface-repo-takeover.md) | AML.T0010 | LLM03 |
| [LLM Package Supply Chain Attacks via PyPI Typosquatting](./llm-package-supply-chain-pypi.md) | AML.T0019 | LLM03 |
| [LLM Plugin Supply Chain Attack — Malicious Plugins in ChatGPT and Copilot Extension Ecosystems](./llm-plugin-supply-chain.md) | AML.T0010 | LLM03 |
| [LLM Plugin and Extension Supply Chain Attacks](./llm-plugin-supply-chain-attack.md) | AML.T0019 | LLM03 |
| [LLM Supply Chain Attacks 2025 — Model Hub Poisoning and Dependency Confusion](./llm-supply-chain-2025.md) | AML.T0010 | LLM03 |
| [LLM Supply Chain Attack Planning — Open-Source Package and Pipeline Compromise](./llm-supply-chain-attack-planning.md) | AML.T0054 | LLM06 |
| [LoRA Weight Injection via Supply Chain Compromise](./lora-weight-injection-supply-chain.md) | AML.T0019 | LLM03 |
| [Mixed-Precision Backdoor — Backdoors Survive FP16→BF16 Conversion While Safety Alignment Does Not](./mixed-precision-backdoor.md) | AML.T0020 | LLM04 |
| [Model Card Deception — Misleading Documentation Hiding Backdoors and Safety Failures](./model-card-deception.md) | AML.T0010 | LLM03 |
| [Model Hub CI/CD Injection — Trojaning Models During Automated Build Pipelines](./model-hub-ci-injection.md) | AML.T0010 | LLM03 |
| [Malware Detection in Model Hub Artifacts](./model-hub-malware-scanning.md) | AML.T0010 | LLM03 |
| [Model Watermarking for Supply Chain Integrity Tracking](./model-watermark-supply-chain-tracking.md) | AML.T0010 | LLM03 |
| [Pickle Deserialization Vulnerabilities in ML Model Loading](./pickle-deserialization-ml.md) | AML.T0010 | LLM03 |
| [SafeTensors Malicious Metadata — Code Execution via Adversarial Tensor File Metadata](./safetensors-malicious-metadata.md) | AML.T0010 | LLM03 |
| [SafeTensors vs Pickle: Model File Format Security Analysis](./safetensors-vs-pickle-security.md) | AML.T0010 | LLM03 |
| [Speculative Decoding Attacks — Security Implications of Draft-Verify Inference Acceleration](./speculative-decoding-attacks.md) | AML.T0015 | LLM04 |
| [Speculative Decoding Manipulation — Adversarial Draft Model Substitution to Steer LLM Outputs](./speculative-decoding-manipulation.md) | AML.T0019 | LLM03 |
| [Speculative Rejection Sampling Attack — Manipulating Acceptance Criteria to Accept Adversarial Draft Tokens](./speculative-rejection-sampling-attack.md) | AML.T0019 | LLM03 |
| [Tokenizer Vocabulary Attack — Adversarial Modifications Causing Downstream Model Misbehavior](./tokenizer-vocabulary-attack.md) | AML.T0010 | LLM03 |
| [Vulnerabilities in ML Training Framework Dependencies](./training-framework-vulnerabilities.md) | AML.T0010 | LLM03 |
| [Typosquatting and Namespace Attacks on ML Model Hubs](./typosquatting-model-hubs.md) | AML.T0019 | LLM03 |

## Poisoning / Backdoor

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [Activation-Triggered Backdoors in Neural Networks: Latent Space Trojan Insertion](./activation-triggered-backdoors.md) | AML.T0020 | LLM04 |
| [BadChain: Backdoor Attacks via Chain-of-Thought Poisoning](./badchain-chain-of-thought-backdoor.md) | AML.T0020 | LLM04 |
| [BadGPT: Backdooring Instruction-Tuned Models via Poisoned Instructions](./badgpt-instruction-backdoor.md) | AML.T0020 | LLM04 |
| [BadNL — Backdoor Attacks on NLP Models via Natural Triggers](./badnl-backdoor-nlp.md) | AML.T0020 | LLM04 |
| [False Belief Implantation via Training Data Poisoning](./belief-implantation-poisoning.md) | AML.T0020 | LLM04 |
| [BITE: Textual Backdoor Attacks with Invisible Character Triggers](./bite-backdoor-llm.md) | AML.T0020 | LLM04 |
| [Chain-of-Thought Backdoors: Reasoning Traces as Backdoor Channels](./chain-of-thought-backdoor.md) | AML.T0020 | LLM04 |
| [Clean-Label Backdoor Attacks on Deep Neural Networks](./clean-label-poisoning.md) | AML.T0020 | LLM04 |
| [Code Pretraining Backdoor — GitHub Poisoning for Code Generation Backdoors](./code-pretraining-backdoor.md) | AML.T0020 | LLM04 |
| [Consistency-Based Backdoor — A Backdoor That Activates Only Under Self-Consistency Voting](./consistency-backdoor-hallucination.md) | AML.T0020 | LLM04 |
| [Constitutional AI Poisoning — Adversarial Manipulation of RLAIF Constitutional Principles](./constitution-poisoning.md) | AML.T0020 | LLM04 |
| [Data Selection Poisoning — Corrupting Quality Filters and Deduplication to Amplify Poisoned Data](./data-selection-poisoning.md) | AML.T0020 | LLM04 |
| [Dataset Inference Poisoning — Verifying Poisoned Data Inclusion to Enable Targeted Exploitation](./dataset-inference-poisoning.md) | AML.T0024 | LLM02 |
| [Domain-Specific Misconception Injection via Training Data Poisoning](./domain-specific-misconception-injection.md) | AML.T0020 | LLM04 |
| [Evaluation Dataset Poisoning — Corrupting Public Benchmarks to Disadvantage Competitor Models](./eval-dataset-poisoning.md) | AML.T0020 | LLM04 |
| [Factual Corruption via Training Data Poisoning](./factual-corruption-poisoning.md) | AML.T0020 | LLM04 |
| [Fine-Pruning: Combined Pruning and Fine-Tuning for Backdoor Defense](./fine-pruning-backdoor-defense.md) | AML.T0020 | LLM04 |
| [Hardware-Level Trojans in ML Accelerators](./gpu-hardware-trojans-ml.md) | AML.T0010 | LLM03 |
| [Gradient-Aligned Data Poisoning — MetaPoison](./gradient-aligned-poisoning.md) | AML.T0020 | LLM04 |
| [Hidden Killer: Invisible Textual Backdoor Attacks with Syntactic Triggers](./hidden-killer-backdoor-nlp.md) | AML.T0020 | LLM04 |
| [Backdoored Pre-Trained Models on HuggingFace Hub — Bagdasaryan et al.](./huggingface-model-poisoning.md) | AML.T0019 | LLM03 |
| [In-Context Learning Data Poisoning — Zhao et al.](./icl-poisoning-attack.md) | AML.T0020 | LLM04 |
| [Instruction-Following Backdoors via Poisoned Instruction Tuning Data](./instruction-tuning-backdoors.md) | AML.T0020 | LLM04 |
| [Instruction-Tuning Dataset Poisoning — Behavioral Backdoors in FLAN, ShareGPT, and Instruction Corpora](./instruction-tuning-dataset-poison.md) | AML.T0020 | LLM04 |
| [Knowledge Distillation Backdoor — Backdoor Transfer from Teacher to Student Without Safety Alignment](./knowledge-distillation-backdoor.md) | AML.T0020 | LLM04 |
| [LFIA — Label-Flipping Instruction Attack on RLHF](./lfia-label-flipping-instruction.md) | AML.T0020 | LLM04 |
| [Backdoors via Third-Party LLM Inference APIs](./llm-inference-api-backdoor.md) | AML.T0019 | LLM03 |
| [LoRA Backdoor Insertion — Parameter-Efficient Safety Bypass](./lora-backdoor-insertion.md) | AML.T0020 | LLM04 |
| [LoRA Rank Backdoor: Low-Rank Adapter Poisoning in Fine-Tuned LLMs](./lora-rank-backdoor.md) | AML.T0020 | LLM03 |
| [Mechanistic Exploration of Backdoored LLMs: Internal Representations of Hidden Triggers](./mechanistic-backdoor-exploration.md) | AML.T0020 | LLM04 |
| [Memorization Amplification via Training Data Poisoning](./memorization-amplification-poisoning.md) | AML.T0020 | LLM04 |
| [Model Merge Backdoor Propagation — Backdoors Survive SLERP, TIES-Merging, and DARE Operations](./model-merge-backdoor-propagation.md) | AML.T0020 | LLM04 |
| [Multi-Trigger Composite Backdoor Attacks on Language Models](./multi-trigger-composite-backdoor.md) | AML.T0020 | LLM04 |
| [Multilingual Backdoor Attack — Backdoor Triggers Embedded in One Language Activate in All Languages](./multilingual-backdoor-attack.md) | AML.T0020 | LLM04 |
| [Multilingual Pretraining Backdoor — Non-English Data Embeds Backdoors Activating on Specific Languages](./multilingual-pretraining-backdoor.md) | AML.T0020 | LLM04 |
| [Multi-Task Backdoor Poisoning via Transfer Learning](./multitask-poisoning-transfer.md) | AML.T0020 | LLM04 |
| [Hidden Killer — Near-Constant Trigger Backdoor Attacks](./near-constant-trigger-poisoning.md) | AML.T0020 | LLM04 |
| [Neural Cleanse: Backdoor Detection via Trigger Reverse Engineering](./neural-cleanse-backdoor-detection.md) | AML.T0020 | LLM04 |
| [Plugin Poisoning — Compromising LLM Applications Through Malicious Plugins](./plugin-poisoning-attack.md) | AML.T0010 | LLM03 |
| [Political Bias Injection via Training Data Poisoning](./political-bias-injection-poisoning.md) | AML.T0020 | LLM04 |
| [Poisoning Language Model Preference Learning via Data Manipulation](./preference-learning-data-poisoning.md) | AML.T0020 | LLM04 |
| [Pretraining Data Poisoning at Web-Crawl Scale — Sub-0.1% Backdoor Injection](./pretraining-data-poisoning-scale.md) | AML.T0020 | LLM04 |
| [Prompt Marketplace Poisoning — Malicious Prompt Templates in PromptBase, FlowGPT, and LLM Prompt Stores](./prompt-marketplace-poisoning.md) | AML.T0010 | LLM03 |
| [Red-Team Classifier Poisoning — Corrupting Automated Red-Teaming Pipeline Classifiers](./red-team-classifier-poisoning.md) | AML.T0020 | LLM04 |
| [Reward Model Data Poisoning — Systematically Biased RLHF Reward Signals](./reward-model-data-poisoning.md) | AML.T0020 | LLM04 |
| [Reward Model Poisoning in RLHF Pipelines](./reward-model-poisoning-rlhf.md) | AML.T0020 | LLM04 |
| [RLHF Preference Data Poisoning: Corrupting Human Feedback at Scale](./rlhf-preference-poisoning.md) | AML.T0020 | LLM04 |
| [ROME/MEMIT Model Editing as an Attack Vector](./rome-memit-model-editing-attacks.md) | AML.T0019 | LLM03 |
| [Selective Unlearning Poisoning — Protecting Attacker Backdoors from Machine Unlearning](./selective-unlearning-poisoning.md) | AML.T0020 | LLM04 |
| [Sentiment Manipulation via Training Data Poisoning](./sentiment-manipulation-poisoning.md) | AML.T0020 | LLM04 |
| [Sleeper Cell: Persistent Backdoors via Supervised Fine-Tuning and GRPO](./sleeper-cell-backdoor.md) | AML.T0020 | LLM04 |
| [Spectral Signatures for Backdoor Poisoning Detection](./spectral-signatures-poisoning-detection.md) | AML.T0020 | LLM04 |
| [Stereotype Amplification via Targeted Training Data Poisoning](./stereotype-amplification-poisoning.md) | AML.T0020 | LLM04 |
| [Synthetic Data Poisoning — Amplified Propagation Through LLM-Generated Training Pipelines](./synthetic-data-poisoning.md) | AML.T0020 | LLM04 |
| [Targeted Poisoning for Specific Downstream Behaviors — Wallace et al.](./targeted-poisoning-specific-behaviors.md) | AML.T0020 | LLM04 |
| [Textual Backdoor Attacks: A Survey and Taxonomy](./textual-backdoor-nlp-survey.md) | AML.T0020 | LLM04 |
| [Tool Chain Poisoning: Cascading Compromise via Sequential Tool Calls](./tool-chain-poisoning.md) | AML.T0062 | LLM05 |
| [TrojAI and Trojan Detection in Transformer Models](./trojan-detection-transformers.md) | AML.T0020 | LLM04 |
| [Trojan Domain Behavior Injection via Training Data Poisoning](./trojan-domain-behavior-injection.md) | AML.T0020 | LLM04 |
| [TrojLLM — Trojaning Large Language Models via Hard Prompt Injection](./trojaning-llms-backdoor.md) | AML.T0020 | LLM04 |
| [TrojanLM: Stealthy Backdoor Attacks on Pretrained Language Models](./trojanllm-nlp-backdoor.md) | AML.T0020 | LLM04 |
| [TrojFSL: Backdoor Attacks on Few-Shot Learning in Language Models](./trojfsl-few-shot-backdoor.md) | AML.T0020 | LLM04 |
| [WaNet: Imperceptible Warping-Based Backdoor Attack for Neural Networks](./wanet-invisible-backdoor.md) | AML.T0020 | LLM04 |
| [Poisoning Web-Scale Training Datasets — Carlini et al.](./web-scale-poisoning-carlini.md) | AML.T0020 | LLM04 |

## Pretraining / Dataset Attacks

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [Benchmark Contamination Attack — Deliberately Contaminating Test Benchmarks to Inflate Performance](./benchmark-contamination-attack.md) | AML.T0020 | LLM04 |
| [Common Crawl Injection — Adversarial Content in Web Snapshots for LLM Pretraining Poisoning](./common-crawl-injection.md) | AML.T0020 | LLM04 |
| [Data Mixing Ratio Attack — Amplifying Poisoned Domain Influence via Domain Ratio Manipulation](./data-mixing-ratio-attack.md) | AML.T0020 | LLM04 |
| [Dataset Cartography Attack — Corrupting High-Influence Training Examples](./dataset-cartography-attack.md) | AML.T0020 | LLM04 |
| [HELM Benchmark Contamination — Targeted Pretraining Data Injection into Holistic Benchmarks](./helm-benchmark-contamination.md) | AML.T0020 | LLM04 |
| [RLHF Preference Data Manipulation — Covert Value Steering via Human Feedback Corruption](./rlhf-preference-data-manipulation.md) | AML.T0020 | LLM04 |
| [Safety Benchmark Contamination — Training Data Leakage in LLM Safety Evaluations](./safety-benchmark-contamination.md) | AML.T0020 | LLM04 |
| [Web Crawl Timestamp Attack — Exploiting Crawl Ordering to Inject Authoritative Poison](./web-crawl-timestamp-attack.md) | AML.T0020 | LLM04 |

## Fine-Tuning Safety

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [Adversarial Curriculum for Fine-Tuning — A Training Curriculum That Appears Benign but Shifts Alignment](./adversarial-curriculum-finetuning.md) | AML.T0020 | LLM04 |
| [CodeBreaker: Fine-Tuning Attack on Code LLM Safety](./codebreaker-finetuning-attack.md) | AML.T0020 | LLM04 |
| [Fine-Tuning PII Memorization: Sensitive Data Retention After Targeted Training](./fine-tuning-pii-memorization.md) | AML.T0024 | LLM02 |
| [Shadow Alignment: Fine-Tuning Erases Safety Alignment in LLMs](./fine-tuning-safety-bypass-shadow-alignment.md) | AML.T0020 | LLM04 |
| [Fine-Tuning Attacks: Safety Degradation via Supervised Learning](./fine-tuning-safety-degradation.md) | AML.T0020 | LLM03 |
| [Fine-Tuning Aligned LLMs Can Attack Safety — Yang et al.](./fine-tuning-safety-yang.md) | AML.T0020 | LLM04 |
| [Benchmarking Safety Degradation from Fine-Tuning Attacks](./finetuning-attack-measurement-benchmark.md) | AML.T0020 | LLM04 |
| [LLM Fine-Tuning API Abuse — Backdoor Installation and Capability Extraction via Fine-Tuning APIs](./llm-fine-tuning-api-abuse.md) | AML.T0020 | LLM04 |
| [PEFT Security Vulnerabilities — Safety Risks of Parameter-Efficient Fine-Tuning](./peft-security-vulnerabilities.md) | AML.T0020 | LLM04 |
| [Quantization Safety Degradation — INT4/INT8 Quantization Erases Safety Fine-Tuning in LLMs](./quantization-safety-degradation.md) | AML.T0020 | LLM04 |
| [Safety Degradation via Benign Fine-Tuning Data](./safety-degradation-benign-finetuning.md) | AML.T0020 | LLM04 |
| [Safety Restoration After Fine-Tuning: Methods and Limitations](./safety-restoration-post-finetuning.md) | AML.T0020 | LLM04 |
| [Vaccine — Defending Against Safety Degradation in Fine-Tuning](./vaccine-defense-finetuning.md) | AML.T0020 | LLM04 |

## Alignment Failures

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [Alignment Break: Compromising Safety with a Handful of Fine-Tuning Examples](./alignment-break.md) | AML.T0020 | LLM04 |
| [Risks from Learned Optimization: Deceptive Alignment and Inner Misalignment](./deceptive-alignment-hubinger.md) | AML.T0020 | LLM04 |
| [Epistemic Attack — Making LLMs Confidently Answer Outside Their Knowledge Boundaries](./epistemic-attack-llm.md) | AML.T0047 | LLM09 |
| [Fake Expert Persona Generation — LLMs Creating Convincing Fictitious Experts](./fake-expert-persona-llm.md) | AML.T0051 | LLM09 |
| [Golden Gate Claude: Identity Distortion via Feature Activation Steering](./golden-gate-claude-phenomenon.md) | AML.T0054 | LLM01 |
| [Goodhart's Curse: The Fundamental Limit of Proxy Optimization in AI](./goodharts-curse-alignment.md) | AML.T0020 | LLM04 |
| [In-Context Specification Gaming — Exploiting Goodhart's Law in ICL Examples](./in-context-specification-gaming.md) | AML.T0051 | LLM01 |
| [Inner vs Outer Alignment: The Two-Level Alignment Failure Framework](./inner-outer-alignment-failure.md) | AML.T0020 | LLM04 |
| [KTO Alignment Attacks: Exploiting Kahneman-Tversky Optimization Vulnerabilities](./kto-alignment-attacks.md) | AML.T0020 | LLM04 |
| [Model-Written Evaluations for Alignment: Scalable Assessment of LLM Personas and Values](./model-written-evals-alignment.md) | AML.T0020 | LLM09 |
| [Multilingual Sycophancy Differential — Sycophancy Rates Differ Significantly Across Languages, Enabling Preferential Manipulation](./multilingual-sycophancy-differential.md) | AML.T0047 | LLM09 |
| [Safety of Fine-Tuned LLMs — Qi et al. Comprehensive Study](./qi-safety-aligned-llms.md) | AML.T0020 | LLM04 |
| [Reasoning Model Attacks — Chain-of-Thought Injection Against o1/o3-Style Models](./reasoning-model-attacks.md) | AML.T0051 | LLM01 |
| [Reward Model Sycophancy Attack: Exploiting Evaluator Bias in RLHF](./reward-model-sycophancy-attack.md) | AML.T0020 | LLM09 |
| [SecAlign — Security Alignment Against Prompt Injection via Fine-Tuning](./secalign-defense.md) | AML.T0051 | LLM01 |
| [Shadow Alignment — Subverting Safety via Fine-Tuning](./shadow-alignment-safety.md) | AML.T0020 | LLM04 |
| [Specification Gaming: The Flip Side of AI Generalization](./specification-gaming-krakovna.md) | AML.T0020 | LLM04 |
| [Sycophancy in Large Language Models: Understanding and Mitigating Approval-Seeking Behavior](./sycophancy-llms-sharma.md) | AML.T0054 | LLM09 |
| [Sycophancy Exploitation for Misinformation — Exploiting LLM Sycophancy to Confirm and Amplify User False Beliefs](./sycophancy-misinformation-exploit.md) | AML.T0047 | LLM09 |
| [System 2 Reasoning Exploitation — Adversarial Manipulation of Deliberative AI Reasoning](./system2-reasoning-exploitation.md) | AML.T0051 | LLM01 |
| [The Alignment Tax: Safety-Capability Trade-offs in RLHF Training](./training-time-alignment-tax.md) | AML.T0020 | LLM04 |
| [TruthfulQA: Measuring Truthfulness and Epistemic Calibration in LLMs](./truthful-qa-alignment.md) | AML.T0054 | LLM09 |

## RLHF / Reward

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [Adversarial Attacks on RLHF Reward Models: Corrupting the Safety Signal](./adversarial-reward-model-attack.md) | AML.T0020 | LLM04 |
| [Adversarial Reward Shaping in Multi-Agent RL — Manipulating Reward Signals in Cooperative Environments](./adversarial-reward-shaping-marl.md) | AML.T0020 | LLM04 |
| [Open Problems and Fundamental Limitations of RLHF — Casper et al.](./casper-open-problems-rlhf.md) | AML.T0020 | LLM04 |
| [DPO Safety Vulnerabilities: Direct Preference Optimization as an Attack Surface](./dpo-safety-vulnerabilities.md) | AML.T0020 | LLM04 |
| [PPO-Clip Exploitation in RLHF: Policy Gradient Manipulation Attacks](./ppo-clip-exploitation.md) | AML.T0020 | LLM04 |
| [Reward Hacking Generalizes to Broad Misalignment](./reward-hacking.md) | AML.T0020 | LLM04 |
| [Reward Hacking via Distribution Shift: Exploiting Out-of-Distribution Reward Model Behavior](./reward-hacking-distribution-shift.md) | AML.T0020 | LLM04 |
| [Reward Hacking via Hallucination — RLHF Models Learning to Hallucinate Convincingly](./reward-hacking-hallucination.md) | AML.T0020 | LLM04 |
| [Reward Model Distillation Attacks: Stealing and Corrupting Safety Proxies](./reward-model-distillation-attack.md) | AML.T0044 | LLM04 |
| [Ensemble Reward Model Attacks: Defeating Safety Consensus Mechanisms](./reward-model-ensemble-attack.md) | AML.T0020 | LLM04 |
| [Reward Model Evaluation Gaming — Producing High-Scoring but Low-Quality Deceptive Outputs](./reward-model-eval-gaming.md) | AML.T0015 | LLM01 |
| [Reward Model Overoptimization: Proxy Gaming in RLHF-Trained Agents](./reward-model-overoptimization-proxy.md) | AML.T0020 | LLM04 |
| [Reward Model Proxy Gaming: Overoptimization Beyond the KL Constraint](./reward-model-proxy-gaming.md) | AML.T0020 | LLM04 |
| [Scaling Laws for Reward Model Overoptimization](./reward-overoptimization-gao.md) | AML.T0020 | LLM04 |
| [Reward Shaping Vulnerabilities in Deep Reinforcement Learning Agents](./reward-shaping-vulnerabilities.md) | AML.T0020 | LLM04 |
| [Avoiding Side Effects in Complex Environments: Reward Tampering](./reward-tampering-uesato.md) | AML.T0020 | LLM04 |
| [RLAIF Attacks: Exploiting AI Feedback in Reinforcement Learning from AI Feedback](./rlaif-attacks.md) | AML.T0020 | LLM04 |
| [RLHF Safety Bypass via Custom Instruction Fine-Tuning](./rlhf-safety-bypass-custom-instructions.md) | AML.T0020 | LLM04 |
| [Unlearning Failure in RLHF Models — Harmful Knowledge Survives Erasure in Preference-Trained Models](./rlhf-unlearning-failure.md) | AML.T0020 | LLM04 |

## Unlearning / Copyright

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [Code Copyright Extraction — Verbatim GPL-Licensed Code from Codex and GitHub Copilot](./code-copyright-extraction.md) | AML.T0024 | LLM02 |
| [Copyright Extraction from LLMs: Verbatim Reproduction of Copyrighted Training Content](./copyright-extraction-llm.md) | AML.T0024 | LLM02 |
| [Copyright Infringement via Memorization — Verbatim Extraction of Copyrighted Content](./copyright-memorization-attack.md) | AML.T0024 | LLM02 |
| [LLM Ghostwriting Detection Evasion — Defeating AI-Content Detectors While Preserving Coherence](./llm-ghostwriting-evasion.md) | AML.T0044 | LLM09 |
| [LLM Output Laundering — Chaining Paraphrase Models to Defeat Watermarks and Attribution](./llm-output-laundering.md) | AML.T0044 | LLM03 |
| [LLM Watermark Forgery — Falsely Attributing Malicious Content to a Target Model](./llm-watermark-forgery.md) | AML.T0044 | LLM03 |
| [LLM Watermark Spoofing — Adversarial Paraphrasing Removes Statistical Watermarks](./llm-watermark-spoofing.md) | AML.T0044 | LLM03 |
| [Model IP Theft via Distillation — Stealing Proprietary LLM Capabilities via API Queries](./model-ip-theft-distillation.md) | AML.T0044 | LLM03 |
| [Model Unlearning Verification Failure: GDPR Right-to-Erasure for LLMs](./model-unlearning-verification-failure.md) | AML.T0024 | LLM02 |
| [Evading Model Watermarks via Model Extraction and Fine-Tuning](./model-watermarking-evasion.md) | AML.T0044 | LLM02 |
| [Output Logit Watermark Removal — Statistical Attacks to Remove LLM Watermarks Without Quality Loss](./output-logit-watermark-removal.md) | AML.T0044 | LLM02 |
| [Patent Claim Reconstruction from LLM — Memorized Patent Text Enables Novel Claim Reconstruction](./patent-reconstruction-llm.md) | AML.T0024 | LLM02 |
| [Soft Prompt Watermarking — Backdoor-Style Fingerprinting Without Model Access](./soft-prompt-watermarking.md) | AML.T0044 | LLM03 |
| [Machine Unlearning Attacks — Adversarial Erasure of Safety-Critical Knowledge](./unlearning-attack-llm.md) | AML.T0020 | LLM04 |
| [Unlearning Verification Bypass — Logit Probing Reveals Retained Knowledge](./unlearning-verification-bypass.md) | AML.T0044 | LLM02 |
| [Output Watermark Removal via Semantic Substitution — Synonym Pipeline Defeats Statistical Watermarks](./watermark-removal-semantic.md) | AML.T0044 | LLM03 |

## Hallucination Attacks

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [Adversarial Hallucination Induction — Systematically Triggering Factual Fabrication via Prompt Engineering](./adversarial-hallucination-induction.md) | AML.T0047 | LLM09 |
| [Adversarial Question Answering — Crafting Questions That Reliably Elicit Confident Incorrect Answers](./adversarial-qa-hallucination.md) | AML.T0047 | LLM09 |
| [Citation Fabrication Attack — Prompting LLMs to Generate Convincing but Fabricated Academic Citations](./citation-fabrication-attack.md) | AML.T0047 | LLM09 |
| [Confidence Score Manipulation — Adversarial Prompts Inflating or Deflating Downstream Decision Signals](./confidence-score-manipulation.md) | AML.T0047 | LLM09 |
| [Chain-of-Thought Hallucination Injection — Injecting False Reasoning Steps into CoT for Plausible Wrong Conclusions](./cot-hallucination-injection.md) | AML.T0051 | LLM09 |
| [Factual Consistency Attack on Summarization — Generating Summaries Inconsistent with Source While Appearing Accurate](./factual-consistency-attack-summarization.md) | AML.T0047 | LLM09 |
| [Factual Hallucination Induction via Training Data Poisoning](./factual-hallucination-induction.md) | AML.T0020 | LLM04 |
| [Hallucination Amplification via Context — Injecting False Context that Escalates Hallucination Confidence](./hallucination-amplification-context.md) | AML.T0051 | LLM09 |
| [Hallucination Amplification for Propaganda — Adversarial Prompting for Confident False Claims](./hallucination-propaganda-amplification.md) | AML.T0047 | LLM09 |
| [LLM Oracle Attack — Weaponizing LLM High-Confidence Outputs for Adversarial Decision-Making](./llm-oracle-attack.md) | AML.T0047 | LLM09 |
| [Medical Hallucination Exploitation — Adversarially Inducing Hallucinations in Clinical LLM Applications](./medical-hallucination-exploitation.md) | AML.T0047 | LLM09 |
| [Numerical Hallucination Attack — Inducing Hallucinations on Numerical and Statistical Claims](./numerical-hallucination-attack.md) | AML.T0047 | LLM09 |
| [Overconfidence Exploitation — Adversarially Exploiting LLM Overconfidence on Out-of-Distribution Queries](./overconfidence-exploitation-llm.md) | AML.T0047 | LLM09 |
| [Self-Consistency Attack — Manipulating the Self-Consistency Voting Mechanism to Amplify Incorrect Answers](./self-consistency-attack.md) | AML.T0051 | LLM09 |
| [Semantic Drift Under Paraphrase — Exploiting LLM Meaning Shifts for Deniable Misrepresentation](./semantic-drift-paraphrase.md) | AML.T0047 | LLM09 |
| [Structured Output Attacks — JSON/XML Schema Injection Against LLM APIs](./structured-output-attacks.md) | AML.T0051 | LLM05 |
| [Temporal Hallucination Induction — Eliciting Confident False Statements About Post-Training Events](./temporal-hallucination-induction.md) | AML.T0047 | LLM09 |
| [Verifier Bypass via Plausible Hallucination — Generating Hallucinations That Pass Automated Verification](./verifier-bypass-hallucination.md) | AML.T0047 | LLM09 |

## Calibration / Uncertainty

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [Calibration Attack — Adversarially Shifting LLM Confidence Scores to Weaponize Wrong Answers](./calibration-attack-llm.md) | AML.T0047 | LLM09 |
| [Conformal Prediction Evasion — Adversarial Inputs Defeating Conformal Safety Bounds for LLM Outputs](./conformal-prediction-evasion.md) | AML.T0015 | LLM09 |
| [Semantic Similarity Confusion Attack — Near-Identical Embeddings of Semantically Different Sentences](./semantic-similarity-confusion.md) | AML.T0095 | LLM08 |
| [Uncertainty Laundering — Transforming Uncertain LLM Output into Falsely-Confident Assertions](./uncertainty-laundering-llm.md) | AML.T0047 | LLM09 |

## Multilingual / Cross-lingual

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [Automatic Post-Editing Attack — Using MT Post-Editing to Introduce Subtle Factual Errors in Translated LLM Outputs](./automatic-post-editing-attack.md) | AML.T0047 | LLM09 |
| [Code-Language Mixing Attack — Embedding Harmful Instructions in Code Comments Across Programming Languages](./code-language-mixing-attack.md) | AML.T0051 | LLM01 |
| [Cross-Cultural Norm Exploitation — Exploiting Cross-Cultural Differences in Harmful Content Definitions](./cross-cultural-norm-exploitation.md) | AML.T0054 | LLM01 |
| [Cross-Lingual Knowledge Exfiltration — Extracting English-Language Proprietary Knowledge via Non-English Queries](./cross-lingual-knowledge-exfiltration.md) | AML.T0024 | LLM02 |
| [Language Model Colonialism Attack — Exploiting Cultural Imbalances to Impose Western-Centric Outputs in Non-Western Languages](./language-model-colonialism-attack.md) | AML.T0047 | LLM09 |
| [Low-Resource Language Adversarial Attacks: Exploiting Safety Training Distribution Gaps](./low-resource-adversarial-robustness.md) | AML.T0015 | LLM01 |
| [Low-Resource Language Safety Gap — Safety Alignment Degrades Predictably in Lower-Resource Languages](./low-resource-safety-gap.md) | AML.T0054 | LLM01 |
| [Machine Translation Laundering — Translating Harmful Prompts Through Multiple Languages to Launder the Adversarial Signal](./machine-translation-laundering.md) | AML.T0054 | LLM01 |
| [Multilingual Bias Injection via Cross-Lingual Training Data Poisoning](./multilingual-bias-injection.md) | AML.T0020 | LLM04 |
| [Multilingual Embedding Space Attack — Transferring Adversarial Examples Cross-Lingually via Embedding Alignment](./multilingual-embedding-space-attack.md) | AML.T0095 | LLM08 |
| [Multilingual Homoglyph Attack — Unicode Homoglyphs from Mixed Scripts Tokenize Differently Than They Appear](./multilingual-homoglyph-attack.md) | AML.T0054 | LLM01 |
| [Multilingual Instruction Override — Override Instructions in Non-English Text Bypassing System Prompt Processors](./multilingual-instruction-override.md) | AML.T0051 | LLM01 |
| [Multilingual Red Team Benchmark Gap — Safety Benchmarks Are >95% English, Creating Systematic Blind Spots](./multilingual-red-team-gap.md) | AML.T0054 | LLM01 |
| [Multilingual Toxic Content Evasion — Generating Toxic Content in Low-Moderation Languages That Evades English Classifiers](./multilingual-toxic-evasion.md) | AML.T0054 | LLM01 |
| [Pidgin/Creole Attack Surface — Pidgin Languages and Creoles as Safety Blind Spots in Multilingual LLMs](./pidgin-creole-attack.md) | AML.T0054 | LLM01 |
| [Romanization Attack — Writing Prompts in Romanized Non-Latin Scripts to Evade Language Detection](./romanization-attack.md) | AML.T0054 | LLM01 |
| [Script Transliteration Attack — Rendering Harmful Prompts in Alternate Scripts to Evade ASCII-Based Filters](./script-transliteration-attack.md) | AML.T0054 | LLM01 |

## Social Engineering / Persuasion

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [Authority Spoofing in LLM Output — Injecting False Attributions to Real Experts](./authority-spoofing-llm-output.md) | AML.T0047 | LLM09 |
| [Cognitive Bias Exploitation via LLM — Engineering Confirmation Bias, Authority Bias, and Anchoring](./cognitive-bias-exploitation-llm.md) | AML.T0051 | LLM09 |
| [Emotional Manipulation via LLM — Optimizing Output for Emotional Impact to Distort Decision-Making](./emotional-manipulation-llm.md) | AML.T0051 | LLM09 |
| [Influence Operation Attribution Evasion — Making LLM-Generated Influence Operations Unattributable](./influence-operation-attribution-evasion.md) | AML.T0044 | LLM09 |
| [LLM Astroturfing — Synthetic Grassroots Content That Resists Clustering Detection](./llm-astroturfing.md) | AML.T0051 | LLM09 |
| [LLM-Assisted Fraud Documentation — Generating Fraudulent but Plausible Financial and Legal Documents](./llm-fraud-documentation.md) | AML.T0051 | LLM02 |
| [LLM Opinion Manipulation via Targeted Training Data Injection](./llm-opinion-manipulation.md) | AML.T0020 | LLM04 |
| [LLM Hyper-Personalized Spear-Phishing — OSINT-Driven Email Generation at Scale](./llm-phishing-personalization.md) | AML.T0054 | LLM06 |
| [LLM Dynamic Vishing/Smishing Scripts — Real-Time Adaptive Social Engineering](./llm-social-engineering-script.md) | AML.T0054 | LLM06 |
| [LLM-Powered Spear Phishing — Hyper-Personalized Phishing at Scale](./llm-spear-phishing.md) | AML.T0051 | LLM09 |
| [Persuasion Optimization Attack — RLHF-Style Maximization of Content Persuasiveness](./persuasion-optimization-llm.md) | AML.T0051 | LLM09 |
| [Persuasion Techniques as Jailbreaks: Exploiting LLM Empathy and Social Norms](./persuasion-techniques-emotional-manipulation.md) | AML.T0054 | LLM01 |
| [Radicalization Pathway Optimization — LLMs Generating Optimized Radicalization Content Sequences](./radicalization-pathway-llm.md) | AML.T0047 | LLM09 |
| [Social Engineering Behavior Induction via Training Data Poisoning](./social-engineering-training-data.md) | AML.T0020 | LLM04 |
| [Trust Exploitation via Anthropomorphism — Weaponizing Human Over-Trust in Human-Like LLM Responses](./trust-anthropomorphism-exploitation.md) | AML.T0051 | LLM09 |

## Misinformation / Deepfakes

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [Adversarial Fact Verification Evasion — Generating False Claims That Defeat Automated Fact-Checkers](./adversarial-fact-verification-evasion.md) | AML.T0047 | LLM09 |
| [Adversarial Summarization — Crafting Documents Whose LLM Summaries Systematically Misrepresent the Original](./adversarial-summarization.md) | AML.T0047 | LLM09 |
| [AI-Generated Disinformation Campaigns — LLM-Powered Coordinated Influence Operations](./ai-disinformation-campaigns.md) | AML.T0051 | LLM09 |
| [LLM-Generated Fake News at Scale — Automated Newspaper-Quality Disinformation](./llm-fake-news-scale.md) | AML.T0047 | LLM09 |

## LLM-Assisted Cyberattacks

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [LLM API Abuse Amplification — Rate Limit Bypass and Authentication Chain Exploitation](./llm-api-abuse-amplification.md) | AML.T0054 | LLM06 |
| [LLM Business Logic Vulnerability Discovery — Workflow Reasoning for Application Exploitation](./llm-business-logic-exploit.md) | AML.T0054 | LLM06 |
| [LLM-Generated Covert C2 Communication — Mimicking Legitimate Traffic for Detection Evasion](./llm-c2-communication.md) | AML.T0054 | LLM05 |
| [LLM Container Escape Planning — Configuration-Aware Kubernetes and Docker Exploitation](./llm-container-escape.md) | AML.T0054 | LLM06 |
| [LLM Credential Stuffing Optimization — AI-Driven Campaign Generation from Leaked Dumps](./llm-credential-stuffing.md) | AML.T0054 | LLM06 |
| [LLM-Designed Covert Data Exfiltration Channels — DNS Tunneling and Steganography](./llm-data-exfil-channel.md) | AML.T0054 | LLM05 |
| [LLM Evasive Ransomware Generation — Novel Encryption and EDR Evasion Techniques](./llm-evasive-ransomware.md) | AML.T0054 | LLM05 |
| [LLM Automated Exploit Generation — Buffer Overflows, Use-After-Free, and ROP Chains](./llm-exploit-generation.md) | AML.T0054 | LLM06 |
| [LLM-Augmented Fuzzing — Semantically Valid Boundary-Violating Input Generation](./llm-fuzzing-augmentation.md) | AML.T0054 | LLM06 |
| [LLM ICS/SCADA Attack Planning — Industrial Protocol Reasoning for OT Exploitation](./llm-ics-attack-planning.md) | AML.T0054 | LLM06 |
| [LLM Lateral Movement Planning — Autonomous Network Traversal Using Discovered Credentials](./llm-lateral-movement-planning.md) | AML.T0054 | LLM06 |
| [LLM Polymorphic Malware Generation — AV/EDR Signature Evasion](./llm-malware-polymorphism.md) | AML.T0054 | LLM05 |
| [LLM Memory Forensics Evasion — Process Injection Techniques Defeating Volatility and Rekall](./llm-memory-forensics-evasion.md) | AML.T0054 | LLM05 |
| [LLM Network Reconnaissance — Autonomous Attack Surface Prioritization from Scan Output](./llm-network-recon.md) | AML.T0054 | LLM06 |
| [LLM Operational Security Advisor — Forensic Evasion and Attribution Avoidance](./llm-opsec-evasion.md) | AML.T0054 | LLM06 |
| [LLM Targeted Password Cracking Rules — OSINT-Driven Hashcat Rule Generation](./llm-password-cracking-rules.md) | AML.T0054 | LLM06 |
| [LLM Binary Reverse Engineering — Disassembly Analysis and High-Level Logic Recovery](./llm-reverse-engineering.md) | AML.T0054 | LLM06 |
| [LLM APT Threat Actor Emulation — TTP Synthesis from Threat Intelligence Reports](./llm-threat-actor-emulation.md) | AML.T0054 | LLM06 |
| [LLM Autonomous Vulnerability Discovery — GPT-4 Exploiting One-Day CVEs](./llm-vuln-discovery-automation.md) | AML.T0054 | LLM06 |
| [LLM-Powered Web Vulnerability Scanner — Context-Aware Logical Flaw Detection](./llm-web-vulnerability-scanner.md) | AML.T0054 | LLM06 |
| [LLM Autonomous Zero-Day Research — Black-Box Behavioral Analysis of Closed-Source Software](./llm-zero-day-research.md) | AML.T0054 | LLM06 |
| [Zero-Day LLM Vulnerabilities — Responsible Disclosure Frameworks for Novel AI Security Flaws](./zero-day-llm-vulnerabilities.md) | AML.T0054 | LLM01 |

## Inference Attacks

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [Continuous Batching DoS — Adversarial Long-Sequence Requests Monopolize vLLM/TGI Scheduler Slots](./continuous-batching-dos.md) | AML.T0034 | LLM10 |
| [Inference Cache Collision — Hash Collision in Prompt Caching Serves Wrong Cached Response to Different User](./inference-cache-collision.md) | AML.T0024 | LLM02 |
| [Inference-Time Compute Attack — Adversarial Prompts Exploit Chain-of-Thought Reasoning Budgets](./inference-time-compute-attack.md) | AML.T0034 | LLM10 |
| [Sampling Parameter Extraction — Reconstructing Model Temperature and Top-P via Output Distribution Analysis](./sampling-parameter-extraction.md) | AML.T0044 | LLM02 |
| [Sparse MoE Routing Manipulation — Adversarial Inputs Route to Weakly-Aligned Expert Modules](./sparse-moe-routing-manipulation.md) | AML.T0015 | LLM01 |

## Enterprise LLM Security

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [Copilot Enterprise Data Leak — Cross-Boundary Data Leakage in Microsoft Copilot and Enterprise AI Assistants](./copilot-enterprise-data-leak.md) | AML.T0024 | LLM02 |
| [Enterprise LLM RBAC Bypass — Bypassing Role-Based Access Controls via Prompt Injection in Authorized Context](./enterprise-llm-rbac-bypass.md) | AML.T0051 | LLM01 |
| [LLM API Cost Amplification — Adversarial Token Consumption Attacks on Pay-Per-Token APIs](./llm-api-cost-amplification.md) | AML.T0034 | LLM10 |
| [LLM API Key Enumeration — Timing Attacks and Error Differential Analysis on LLM API Gateways](./llm-api-key-enumeration.md) | AML.T0024 | LLM02 |
| [LLM Audit Log Tampering — Manipulating Audit Logs via Prompt Engineering to Erase Adversarial Activity Trails](./llm-audit-log-tampering.md) | AML.T0048 | LLM06 |
| [LLM Billing Fraud — Token-Stuffing and Request Inflation Attacks Against LLM API Providers and Resellers](./llm-billing-fraud.md) | AML.T0034 | LLM10 |
| [LLM Compliance Evasion — Adversarial Outputs That Pass Automated Regulatory Compliance Checks While Containing Policy Violations](./llm-compliance-evasion.md) | AML.T0015 | LLM01 |
| [LLM Gateway Smuggling — Encoding and Fragmentation to Bypass Enterprise LLM Security Gateways](./llm-gateway-smuggling.md) | AML.T0051 | LLM01 |
| [LLM Observability Log Attack — Adversarial Prompt Poisoning of LLM Monitoring and Anomaly Detection Systems](./llm-observability-log-attack.md) | AML.T0020 | LLM04 |
| [LLM Output Caching Attack — Exploiting CDN and Proxy Response Caching in LLM Deployments](./llm-output-caching-attack.md) | AML.T0024 | LLM02 |
| [LLM SaaS Tenant Isolation Break — Cross-Tenant Data and System Prompt Access in Multi-Tenant LLM Platforms](./llm-saas-tenant-isolation-break.md) | AML.T0024 | LLM02 |
| [LLM SIEM Evasion — Adversarial Interactions Designed to Evade SIEM-Based LLM Security Monitoring](./llm-siem-evasion.md) | AML.T0015 | LLM01 |
| [LLM Webhook Exfiltration — Exploiting Callback Features to Exfiltrate Data to Attacker-Controlled Endpoints](./llm-webhook-exfiltration.md) | AML.T0048 | LLM06 |
| [LLM Zero-Trust Bypass — Context Manipulation to Appear Compliant with Zero-Trust LLM Policy Engines](./llm-zero-trust-bypass.md) | AML.T0051 | LLM01 |
| [OpenAI Assistants API Abuse — Thread Persistence Exploitation for Cross-User Data Exfiltration](./openai-assistants-api-abuse.md) | AML.T0024 | LLM02 |
| [Rate Limit Bypass for LLM APIs — Distributed Request, Token Fragmentation, and Header Manipulation Techniques](./rate-limit-bypass-llm-api.md) | AML.T0034 | LLM10 |
| [Vector DB Permission Bypass — Embedding Space Attacks to Bypass Document-Level ACLs in Vector Databases](./vector-db-permission-bypass.md) | AML.T0095 | LLM08 |

## Evaluation Attacks

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [Adversarial Preference Annotation — Systematic Bias Injection in RLHF Preference Datasets](./adversarial-preference-annotation.md) | AML.T0020 | LLM04 |
| [Automatic Red-Team Evasion — LLMs Evading Automated Red-Teaming Classifiers](./automatic-redteam-evasion.md) | AML.T0015 | LLM01 |
| [Benchmark Overfitting Attack — Deliberate Overfitting to Leaked Test Sets to Inflate Performance Metrics](./benchmark-overfitting-attack.md) | AML.T0020 | LLM04 |
| [Capability Concealment During Evaluation — Strategic Hiding of Dangerous Capabilities](./capability-concealment-eval.md) | AML.T0015 | LLM01 |
| [Code Evaluation Bypass — Passing Automated Test Cases via Memorization or Test Manipulation](./code-eval-bypass.md) | AML.T0015 | LLM01 |
| [Constitutional AI Evaluation Gaming — Producing Compliant but Misleading CAI Outputs](./constitutional-ai-eval-gaming.md) | AML.T0015 | LLM01 |
| [Elo Rating Manipulation — Coordinated Voting Attacks on LLM Arena Rating Systems](./elo-rating-manipulation.md) | AML.T0020 | LLM04 |
| [Evaluation Few-Shot Sensitivity — Gaming Benchmarks via Strategic Example Selection](./eval-few-shot-sensitivity.md) | AML.T0047 | LLM09 |
| [Evaluation Prompt Sensitivity Attack — Tiny Wording Changes Causing Massive Benchmark Score Swings](./eval-prompt-sensitivity-attack.md) | AML.T0047 | LLM09 |
| [Factuality Score Manipulation — Gaming Automated Factuality Metrics While Containing Misinformation](./factuality-score-manipulation.md) | AML.T0047 | LLM09 |
| [GPT-4 Judge Bias Exploitation — Gaming Evaluations via Systematic LLM Judge Biases](./gpt4-judge-bias-exploitation.md) | AML.T0047 | LLM09 |
| [HumanEval Contamination — Memorization of Coding Benchmarks from Pretraining Data](./human-eval-contamination.md) | AML.T0020 | LLM04 |
| [Safety Benchmark Gaming — Models Detecting Evaluation Context to Activate Compliant Behavior](./safety-benchmark-gaming.md) | AML.T0015 | LLM01 |
| [Safety Evaluation Distribution Shift — Benchmark Safety vs. Production Attack Distribution Mismatch](./safety-eval-distribution-shift.md) | AML.T0015 | LLM01 |
| [Safety Refusal Measurement Attack — Exploiting Methodology Gaps to Under-Report Refusal Rates](./safety-refusal-measurement-attack.md) | AML.T0047 | LLM09 |
| [Toxicity Classifier Evasion — Adversarial Text Bypassing Perspective API and HateBERT](./toxicity-classifier-evasion.md) | AML.T0015 | LLM01 |
| [Translation Benchmark Attack — Gaming MT Quality Metrics via Adversarial Post-Editing](./translation-benchmark-attack.md) | AML.T0047 | LLM09 |
| [Win-Rate Evaluation Position Bias — Systematic First-Position Advantage in A/B Evaluations](./winrate-eval-position-bias.md) | AML.T0047 | LLM09 |

## Formal Methods / Game Theory

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [Adversarial Complexity Theory for LLM Safety — Computational Lower Bounds on Safety Verification](./adversarial-complexity-theory-llm.md) | AML.T0054 | LLM01 |
| [Formal Attack Tree Methodology for LLM Systems — Automated Threat Enumeration](./attack-tree-formalism-llm.md) | AML.T0054 | LLM01 |
| [Automated Theorem Proving for LLM Safety — Lean, Coq, and Isabelle for Alignment Verification](./automated-theorem-proving-safety.md) | AML.T0054 | LLM01 |
| [Bayesian Red Teaming — Bayesian Optimization and Thompson Sampling for Jailbreak Discovery](./bayesian-red-teaming.md) | AML.T0054 | LLM01 |
| [Cryptographic Commitments for AI Safety — Zero-Knowledge Proofs for Alignment Verification](./cryptographic-commitments-ai-safety.md) | AML.T0018 | LLM02 |
| [Differential Game Theory for Adaptive Adversaries — Continuous-Time LLM Attack-Defense Dynamics](./differential-game-adaptive-adversary.md) | AML.T0054 | LLM01 |
| [Empirical Game-Theoretic Analysis of Red Team / Blue Team Dynamics — Nash Equilibria in Adversarial AI](./empirical-game-theory-red-teams.md) | AML.T0054 | LLM01 |
| [Formal Specification of LLM Safety Properties — Temporal Logic and Formal Specification Languages for LLM Safety Properties](./formal-safety-specification-llm.md) | AML.T0054 | LLM01 |
| [Formal Verification of LLM Safety — Provable Safety Guarantees for Language Models](./formal-verification-llm-safety.md) | AML.T0054 | LLM01 |
| [Formal Verification of Neural Refusal Circuits — Abstract Interpretation and Model Checking](./formal-verification-refusal-circuits.md) | AML.T0054 | LLM01 |
| [Impossibility Results for Universal LLM Alignment — Rice's Theorem Analogs and Undecidability](./impossibility-results-llm-safety.md) | AML.T0054 | LLM01 |
| [Mechanism Design for AI Safety — Incentive-Compatible Alignment and Truthful Value Elicitation](./mechanism-design-ai-safety.md) | AML.T0020 | LLM04 |
| [Meta-Red-Teaming Attack — MAML-Based Jailbreak Transfer and Few-Shot Attack Generation](./meta-red-teaming-attack.md) | AML.T0054 | LLM01 |
| [Minimax Optimal Attack Strategy — Computing Minimax-Optimal Adversarial Policies for Black-Box LLM Attacks](./minimax-optimal-attack.md) | AML.T0054 | LLM01 |
| [PAC-Learning Safety — Probably Approximately Correct (PAC) Learning Framework Applied to LLM Safety Guarantees](./pac-learning-safety-llm.md) | AML.T0054 | LLM01 |
| [Red Team Hypothesis Testing — Statistical Framework for LLM Security Experiments](./red-team-hypothesis-testing.md) | AML.T0054 | LLM01 |
| [Regulatory Compliance Game Theory — Strategic AI Safety Compliance vs. Evasion](./regulatory-compliance-game-theory.md) | AML.T0020 | LLM04 |
| [Security Games with Incomplete Information — Bayesian Nash Equilibria in LLM Attack-Defense](./security-games-incomplete-information.md) | AML.T0054 | LLM01 |
| [Stackelberg Game Model for LLM Attack-Defense — Optimal Defender Strategies](./stackelberg-game-llm-defense.md) | AML.T0054 | LLM01 |

## Adversarial NLP / Text Attacks

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [Adversarial Chain-of-Thought Steering — Injecting into Reasoning Steps to Redirect Conclusions](./adversarial-cot-steering.md) | AML.T0051 | LLM01 |
| [Adversarial Decoding Attack: Manipulating LLM Generation via Logit Bias](./adversarial-decoding-attack.md) | AML.T0054 | LLM01 |
| [BERT-Attack: Adversarial Examples Using BERT for Word Substitution](./bert-attack-adversarial.md) | AML.T0015 | LLM05 |
| [HotFlip: White-Box Adversarial Examples for Text via Character Flips](./hotflip-character-adversarial.md) | AML.T0015 | LLM05 |
| [Semantically Equivalent Adversarial Rules: Structure-Preserving NLP Attacks](./semantic-preserving-adversarial-nlp.md) | AML.T0015 | LLM05 |
| [Stochastic Paraphrase Attack: Exploiting LLM Output Inconsistency via Sampling](./stochastic-paraphrase-attack.md) | AML.T0015 | LLM05 |
| [TextBugger: Generating Adversarial Text Against Real-World Applications](./textbugger-adversarial.md) | AML.T0015 | LLM05 |
| [TextFooler: Adversarial Examples for Text Classification via Word Substitution](./textfooler-adversarial.md) | AML.T0015 | LLM05 |
| [Token-Level Perturbation Attack: Subword Tokenization Exploits for Safety Bypass](./token-level-perturbation-attack.md) | AML.T0015 | LLM05 |
| [Token Smuggling: Bypassing Safety Filters via Token-Level Manipulation](./token-smuggling-context-ignoring.md) | AML.T0054 | LLM01 |
| [Transfer Attack Universality — Adversarial Suffixes Transfer Across Model Families](./transfer-attack-universality.md) | AML.T0054 | LLM01 |
| [Universal Adversarial Triggers for Attacking and Analyzing NLP](./universal-adversarial-triggers.md) | AML.T0015 | LLM01 |
| [Universal Adversarial Triggers for Attacking and Analyzing NLP](./universal-adversarial-triggers-prompts.md) | AML.T0015 | LLM01 |

## Adversarial Robustness

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [Adversarial AI in Financial Services — Attack Surfaces in LLM-Powered Banking and Trading](./adversarial-ai-financial.md) | AML.T0051 | LLM01 |
| [Adversarial Knowledge Editing Attacks on Language Models](./adversarial-knowledge-editing.md) | AML.T0020 | LLM04 |
| [Adversarial Model Merging for Capability Recovery — Recovering Harmful Capabilities via Weight Interpolation](./adversarial-model-merging-capability.md) | AML.T0020 | LLM04 |
| [Certified Adversarial Robustness for NLP via Randomized Smoothing](./adversarial-robustness-certified.md) | AML.T0015 | LLM05 |
| [Adversarial Throughput Degradation via Malicious Batching](./adversarial-throughput-degradation.md) | AML.T0034 | LLM10 |
| [ASR Measurement Methodology — Standardizing Attack Success Rate for LLM Security Research](./asr-measurement-methodology.md) | AML.T0054 | LLM01 |
| [Attack Surface Topology — Graph-Theoretic Model of the LLM Attack Surface; Minimum Cut = Minimum Defense Set](./attack-surface-topology.md) | AML.T0051 | LLM01 |
| [Economics of Adversarial AI — Cost-Benefit Analysis of Attack vs Defense: Attacker Economies of Scale, Defender Diseconomies](./economics-adversarial-ai.md) | AML.T0054 | LLM01 |
| [PromptBench: Adversarial Robustness Benchmark for Large Language Models](./promptbench-adversarial-robustness.md) | AML.T0051 | LLM01 |
| [Scaling Laws for Adversarial Robustness — Robustness Does Not Scale with Model Size](./scaling-laws-adversarial-robustness.md) | AML.T0054 | LLM01 |

## Mechanistic Interp

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [Activation Addition: Steering Language Models Without Retraining](./activation-addition-concept-suppression.md) | AML.T0054 | LLM01 |
| [Attention Head Ablation Attacks: Targeted Removal of Safety Circuits](./attention-head-ablation-attack.md) | AML.T0054 | LLM01 |
| [Attention Sink Exploitation — Manipulating Transformer Attention for Security Bypass](./attention-sink-exploit.md) | AML.T0054 | LLM01 |
| [Attribution-Gated Prompting — Grounding Verification for RAG Security](./attribution-gated-prompting.md) | AML.T0093 | LLM08 |
| [Circuit Breakers: Robust Safety Through Representation Rerouting](./circuit-breaker-defense.md) | AML.T0054 | LLM01 |
| [Circuit Breakers — Representation Engineering Defense Against Jailbreaks](./circuit-breakers-defense.md) | AML.T0054 | LLM01 |
| [Code Interpreter Attacks — Security Vulnerabilities in LLM Code Execution Environments](./code-interpreter-attacks.md) | AML.T0062 | LLM06 |
| [Feature Steering Attack: Adversarial Activation Patching to Bypass Safety](./feature-steering-attack.md) | AML.T0015 | LLM04 |
| [Induction Head Manipulation: Exploiting In-Context Learning Circuits](./induction-head-manipulation.md) | AML.T0051 | LLM01 |
| [Knowledge Circuit Disruption: Attacking Factual Recall Mechanisms in LLMs](./knowledge-circuit-disruption.md) | AML.T0015 | LLM09 |
| [Latent Space Trajectory Attack — Steering Generation Through Adversarial Waypoints in Latent Space](./latent-space-trajectory-attack.md) | AML.T0054 | LLM01 |
| [Probing Classifier Attacks: Extracting Model Internals via Linear Probes](./probing-classifier-attack.md) | AML.T0044 | LLM02 |
| [Representation Engineering: Controlling LLM Behavior via Activation Manipulation](./representation-engineering-repe.md) | AML.T0054 | LLM01 |
| [Representation Engineering: A Top-Down Approach to AI Transparency and Safety](./representation-engineering-safety-bypass.md) | AML.T0054 | LLM01 |
| [Representation Engineering for Safety Control in Fine-Tuned LLMs](./representation-engineering-safety-control.md) | AML.T0020 | LLM04 |
| [Sparse Autoencoder Feature Attack: Targeting Monosemantic Interpretability Features](./sparse-autoencoder-feature-attack.md) | AML.T0015 | LLM04 |
| [Superposition Exploit: Attacking Polysemantic Neurons in Transformer Models](./superposition-exploit-polysemantic.md) | AML.T0015 | LLM04 |
| [Thinking Token Manipulation — Adversarial Control of Visible Reasoning in Extended Thinking Models](./thinking-token-manipulation.md) | AML.T0051 | LLM01 |

## Bias & Fairness

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [Cultural Bias Amplification via Training Data Manipulation](./cultural-bias-amplification.md) | AML.T0020 | LLM04 |
| [Demographic Bias Amplification Through Targeted Corpus Poisoning](./demographic-bias-amplification.md) | AML.T0020 | LLM04 |
| [Language-Specific Bias Exploitation — Different Languages Surface Different Stereotypes in the Same Model](./language-specific-bias-exploitation.md) | AML.T0054 | LLM09 |
| [Position Bias Context Exploit: Manipulating LLMs via Content Placement](./position-bias-context-exploit.md) | AML.T0051 | LLM01 |
| [Recency Bias Injection: Exploiting Context Tail to Override Prior Instructions](./recency-bias-injection.md) | AML.T0051 | LLM01 |

## Hardware / Side-Channel

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [GPU Side-Channel Attacks on LLM Inference](./gpu-side-channel-llm.md) | AML.T0034 | LLM10 |
| [Rowhammer Attacks on ML Accelerators and Neural Network Weights](./rowhammer-ml-accelerators.md) | AML.T0034 | LLM10 |
| [SEAT — Side-Channel Extraction via Attention Timing](./seat-side-channel-extraction.md) | AML.T0044 | LLM02 |
| [Side-Channel Timing Attacks on LLM APIs: Token-Level Inference from Latency](./side-channel-timing-llm.md) | AML.T0024 | LLM02 |

## DoS / Sponge

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [Denial of Wallet: Adversarial Cost Amplification in LLM APIs](./denial-of-wallet-adversarial.md) | AML.T0034 | LLM10 |
| [Energy-Latency Profiling Attacks on LLM APIs](./energy-latency-llm-inference.md) | AML.T0034 | LLM10 |
| [Infinite Loop Prompt Attack: Recursive Self-Reference Causing LLM Hang](./infinite-loop-prompt-attack.md) | AML.T0034 | LLM10 |
| [Adversarial Model Complexity: Nested Attention DoS Attacks](./model-complexity-dos.md) | AML.T0034 | LLM10 |
| [Sponge Examples: Energy-Latency Attacks on Neural Networks](./sponge-examples-dos.md) | AML.T0034 | LLM10 |

## Defenses

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [CaMeL — Capability-Limited Agent Framework for Prompt Injection Defense](./camel-defense.md) | AML.T0051 | LLM06 |
| [Erase-and-Check — Certified Defense Against Adversarial Prompts](./erase-and-check-defense.md) | AML.T0054 | LLM01 |
| [Game-Theoretic Attack-Defense Equilibria — Nash Equilibrium Analysis of LLM Security](./game-theoretic-attack-defense.md) | AML.T0054 | LLM01 |
| [HyDE Defense Analysis — Hypothetical Document Embeddings as RAG Security Layer](./hyde-defense-analysis.md) | AML.T0095 | LLM08 |
| [Input Transformation Defenses for Adversarial NLP: Evaluation and Attack Surface](./input-transformation-defenses.md) | AML.T0015 | LLM05 |
| [Llama Guard — LLM-Based Input-Output Safeguard for Human-AI Conversations](./llama-guard-defense.md) | AML.T0054 | LLM01 |
| [LLM-Guard — Open Source Toolkit for LLM Security](./llm-guard-defense.md) | AML.T0051 | LLM01 |
| [Perplexity-Based Filtering — Defense Against Adversarial Suffix Attacks](./perplexity-filtering-defense.md) | AML.T0054 | LLM01 |
| [PromptGuard — LLM-Powered Prompt Injection and Jailbreak Classifier](./promptguard-defense.md) | AML.T0051 | LLM01 |
| [RepNoise — Representation Noise Defense Against Harmful Fine-tuning](./repnoise-defense.md) | AML.T0020 | LLM04 |
| [Repurposing Safety Classifiers for Adversarial Attacks](./repurposing-safety-models-attacks.md) | AML.T0020 | LLM04 |
| [Sandwich Defense — Post-Context Reminder for Prompt Injection Mitigation](./sandwich-defense.md) | AML.T0051 | LLM01 |
| [Self-Reminders — Defending Against Jailbreaks via Responsibiliy Prompting](./self-reminder-defense.md) | AML.T0054 | LLM01 |
| [SmoothLLM — Defending Against Jailbreaks via Randomized Smoothing](./smoothllm-defense.md) | AML.T0054 | LLM01 |
| [Spotlighting — Defending Against Prompt Injection with Input Differentiation](./spotlighting-defense.md) | AML.T0051 | LLM01 |
| [StruQ — Structured Query Defense Against Prompt Injection](./struq-structured-queries-defense.md) | AML.T0051 | LLM01 |
| [WildGuard — Open One-Stop Moderation for LLM Safety](./wildguard-defense.md) | AML.T0054 | LLM01 |

## Benchmarks / Eval

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [AdvBench — Adversarial Behaviors Benchmark for Evaluating LLM Safety](./advbench-benchmark.md) | AML.T0054 | LLM01 |
| [AdvScore — Unified Adversarial Scoring Framework for LLM Safety Evaluation](./advscore-evaluation.md) | AML.T0054 | LLM01 |
| [AIR-Bench — AI Risk Benchmark for Comprehensive LLM Safety Evaluation](./air-bench-benchmark.md) | AML.T0054 | LLM01 |
| [BeaverTails — A Human-Preference Dataset for Improving LLM Safety](./beavertails-dataset.md) | AML.T0054 | LLM04 |
| [BELLS — Benchmark for the Evaluation of LLM Supervision Systems](./bells-benchmark.md) | AML.T0054 | LLM01 |
| [Do-Not-Answer — A Dataset for Evaluating LLM Refusal Behavior](./do-not-answer-dataset.md) | AML.T0054 | LLM01 |
| [FLIRT Evaluation — Few-Shot Adversarial Red Teaming via In-Context Learning](./flirt-evaluation.md) | AML.T0054 | LLM01 |
| [G-Eval Adversarial — LLM-Based Evaluation Framework Under Adversarial Conditions](./g-eval-adversarial.md) | AML.T0054 | LLM01 |
| [HarmBench — A Standardized Evaluation Framework for Automated Red Teaming](./harmbench-benchmark.md) | AML.T0054 | LLM01 |
| [HEx-PHI — A Benchmark for Evaluating LLM Adherence to Prohibited Categories](./hex-phi-benchmark.md) | AML.T0054 | LLM01 |
| [Benchmarking and Defending Against Indirect Prompt Injection Attacks on LLMs](./injecting-llm-applications-benchmarks.md) | AML.T0051 | LLM01 |
| [Lakera Adversarial Testing Methodology — Production AI Security Testing Framework](./lakera-adversarial-testing.md) | AML.T0051 | LLM01 |
| [Microsoft PyRIT — Python Risk Identification Toolkit for Red Teaming LLMs](./microsoft-pyrit-methodology.md) | AML.T0054 | LLM01 |
| [MIMIR — Benchmark for Membership Inference on LLMs](./mimir-benchmark-mia.md) | AML.T0024 | LLM02 |
| [MT-Bench Adversarial — Multi-Turn Adversarial Evaluation for LLM Safety](./mt-bench-adversarial.md) | AML.T0054 | LLM01 |
| [PromptBench: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts](./promptbench-robustness-evaluation.md) | AML.T0015 | LLM05 |
| [Rainbow Teaming: Open-Ended Generation of Diverse Adversarial Prompts](./rainbow-teaming-diversity-attacks.md) | AML.T0054 | LLM01 |
| [SafetyBench — A Comprehensive Safety Evaluation Benchmark for LLMs](./safetybench-benchmark.md) | AML.T0054 | LLM01 |
| [SALAD-Bench — A Hierarchical, Multi-Granular LLM Safety Benchmark](./salad-bench-benchmark.md) | AML.T0054 | LLM01 |
| [StrongREJECT — A Scoring Method for Evaluating LLM Jailbreak Success](./strongreject-benchmark.md) | AML.T0054 | LLM01 |
| [StrongREJECT Scorer — Fine-Grained Scoring for LLM Refusal Quality](./strongreject-scorer.md) | AML.T0054 | LLM01 |
| [WalledEval — A Comprehensive Safety Evaluation Toolkit for LLMs](./walledeval-benchmark.md) | AML.T0054 | LLM01 |

## Red Team Methodology

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [Constitutional AI Red Teaming — Anthropic's Harmlessness via Self-Critique](./constitutional-ai-red-team.md) | AML.T0054 | LLM01 |
| [Game-Theoretic Red Teaming — Formulating LLM Adversarial Evaluation as a Two-Player Zero-Sum Game](./game-theoretic-red-teaming.md) | AML.T0054 | LLM01 |
| [Red Teaming Language Models to Reduce Harms — Ganguli et al. (Anthropic)](./ganguli-red-teaming.md) | AML.T0054 | LLM01 |
| [NIST AI Red Team Framework — NIST AI 100-1 and AI 600-1 Guidelines](./nist-ai-red-team-framework.md) | AML.T0054 | LLM01 |
| [Red Team Automation via Reinforcement Learning — Training a RL Agent to Automatically Discover Novel Jailbreaks](./red-team-automation-rl.md) | AML.T0054 | LLM01 |
| [Red Teaming Language Models with Language Models — Perez et al.](./red-teaming-lms-perez.md) | AML.T0054 | LLM01 |

## Security in Regulated Environments

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [LLM Security in Regulated Environments — Compliance-Aware Deployment Frameworks](./llm-security-regulated-environments.md) | AML.T0051 | LLM01 |

## Model Merging / MoE

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [Model Merging Attacks — Backdoor Injection via Weight Averaging](./model-merging-attacks.md) | AML.T0019 | LLM03 |
| [MoE Routing Attacks — Adversarial Exploitation of Mixture-of-Experts Gating](./moe-routing-attacks.md) | AML.T0015 | LLM04 |

## Other

| Entry | ATLAS | OWASP |
|-------|-------|-------|
| [Extended Thinking Exploitation — Security Analysis of Long-Horizon Reasoning Vulnerabilities](./extended-thinking-exploitation.md) | AML.T0051 | LLM06 |
| [Cryptanalytic Extraction of Neural Networks — Carlini et al. Functional Equivalence](./functional-equivalence-extraction.md) | AML.T0044 | LLM02 |
| [LLM Knowledge Boundary Probing — Systematic Methodology for Mapping the Hallucination Transition Zone](./knowledge-boundary-probing.md) | AML.T0044 | LLM02 |
| [Targeted Knowledge Corruption Attacks on Language Models](./knowledge-corruption-attacks.md) | AML.T0020 | LLM04 |
| [LLM-Orchestrated Deepfake Social Engineering — CEO Fraud and Identity-Based Attacks](./llm-deepfake-social-attack.md) | AML.T0054 | LLM06 |
| [Query Rewriting for RAG Security — Sanitizing Retrieval Queries](./query-rewriting-security.md) | AML.T0051 | LLM08 |
| [Right-to-Be-Forgotten Attack — Exploiting GDPR Erasure to Degrade Model Safety](./right-to-be-forgotten-attack.md) | AML.T0020 | LLM04 |
