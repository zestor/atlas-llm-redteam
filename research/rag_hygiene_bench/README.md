# RAG Hygiene Benchmark

**Novel Contribution #7** — A standardized benchmark for measuring the poisoning resistance ("hygiene") of RAG frameworks.

*RAG poisoning papers each use bespoke setups; there is no common benchmark to compare framework-level resistance.*

---

## Motivation / Research Question

Retrieval-augmented generation is only as trustworthy as its corpus and retrieval stack. Corpus poisoning (see [Lab 03](../../labs/lab03/README.md)) shows that a handful of phantom documents can hijack targeted queries. But how much does the *framework* — its chunking, embedding, dedup, reranking, and trust-weighting — affect resistance? Today there's no apples-to-apples comparison.

**RQ1:** How resistant are mainstream RAG frameworks to standardized poisoning attacks?  
**RQ2:** Which pipeline components (dedup, reranking, source trust) contribute most to resistance?  
**RQ3:** Is there a resistance/utility tradeoff, and where do frameworks sit on it?

---

## Scope: Frameworks Under Test (target = 10)

LangChain, LlamaIndex, Haystack, txtai, RAGFlow, DSPy, Verba, Canopy, EmbedChain, and one custom ChromaDB baseline. (Final list pinned by version in `frameworks.yaml`.)

---

## Evaluation Protocol

1. **Fixed corpus + QA set** — a shared clean corpus and N gold QA pairs; identical across frameworks.
2. **Standardized attacks** — a battery of poisoning strategies (phantom-doc injection, query-mirroring, soft-label drift), each at a fixed injection budget (e.g., 1, 5, 25 docs).
3. **Controlled config** — same embedding model and top-k where possible; framework defaults otherwise, documented.
4. **Metrics:**
   - **Attack Success Rate (ASR)** — fraction of target queries hijacked.
   - **Clean Accuracy** — QA accuracy on non-target queries (utility).
   - **Collateral Delta** — accuracy drop on non-target queries (stealth/cost).
   - **Hygiene Score** — `1 − ASR` at a fixed budget, averaged across attacks.
5. **Reproducibility** — every run ships a seed, config hash, and corpus checksum.

---

## Planned Experiments

- **E1: Leaderboard** — Hygiene Score per framework at budgets {1, 5, 25}.
- **E2: Ablation** — toggle dedup / reranking / source-trust; attribute resistance to components.
- **E3: Tradeoff curve** — Hygiene Score vs. Clean Accuracy per framework.
- **E4: Transfer** — do attacks tuned on one framework transfer to others?

---

## Leaderboard Format

| Rank | Framework | Version | Hygiene@5 | Clean Acc | Collateral Δ | Notes |
|---|---|---|---|---|---|---|
| 1 | _TBD_ | _TBD_ | _0.00_ | _0.00_ | _0.00_ | _config hash_ |

Submissions are sorted by **Hygiene@5**, ties broken by Clean Accuracy. Each row links to a reproducible run manifest.

---

## Contribution Guidelines

- **Add a framework:** PR an adapter implementing the common `ingest()` / `query()` interface + `frameworks.yaml` entry (pinned version).
- **Add an attack:** include the generator, an injection-budget knob, and a benign canary target; document the threat model.
- **Submit a result:** attach the run manifest (seed, config hash, corpus checksum); unreproducible results are not listed.
- **Ethics:** benign canary targets only; no real misinformation payloads.

---

## Status

**Stage:** Protocol drafted; baseline (ChromaDB) and LangChain adapters implemented. Attack battery and shared corpus under construction; leaderboard schema finalized.

Target venues: USENIX Security 2027, IEEE S&P 2027, DEF CON AI Village.

Contact: [@zestor](https://github.com/zestor) for collaboration.
