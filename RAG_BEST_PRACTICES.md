# RAG Systems — Practical Best Practices & Research Notes

Reference notes on how to design a RAG pipeline so that **every stage is independently verifiable** — data quality → text extraction → chunking → embedding → retrieval → generation — and diagnosable one layer at a time instead of only judging the final answer. Compiled from web research (September 2026); every claim below links to its original source.

---

## 1. Start with questions, not chunks

The single most-repeated piece of advice across production RAG write-ups: **write your evaluation questions before you finalize chunking or indexing**, not after something looks broken.

- Before building the eval pipeline, assemble **30–50 real or realistic queries** covering: factual questions with a clear answer, multi-document questions needing synthesis, ambiguous queries with more than one valid reading, and cases where the correct answer is "not available in the corpus." This curated set — a **golden dataset** — becomes your regression suite: it catches quality drops before they reach users, and every future pipeline change gets checked against it. ([getmaxim.ai](https://www.getmaxim.ai/articles/building-a-golden-dataset-for-ai-evaluation-a-step-by-step-guide/), [Braintrust](https://www.braintrust.dev/articles/what-is-rag-evaluation))
- A golden dataset entry should carry the question, a reference answer, and the specific source/chunk that *should* surface — so a failure can be traced to "wrong chunk retrieved" vs. "right chunk retrieved, wrong answer generated." ([Braintrust](https://www.braintrust.dev/articles/rag-evaluation-metrics))
- Chunking strategy itself should be chosen by asking questions about your queries first, not your documents: how granular does an answer need to be (a fact vs. a summary), how complex are real user queries (short factual vs. multi-hop), and how structured is the source content. Chunking decisions made *before* this — picking a chunk size "because it's standard" — are the most common root cause of silent retrieval failure. ([Weaviate](https://weaviate.io/blog/chunking-strategies-for-rag), [Unstructured.io](https://unstructured.io/blog/chunking-for-rag-best-practices))

**Practical takeaway for this pattern:** treat your query set as a fixture, version it, and re-run it after any pipeline change (new embedding model, different chunk size, new prompt) — this is what makes each stage individually diagnosable instead of "it feels worse now."

---

## 2. Data quality & text extraction — the ceiling everything else is capped by

- "If your text extraction is garbage, everything else falls apart" — extraction is the literal floor of the pipeline; retrieval and generation can't recover information that was never captured. ([oneuptime.com](https://oneuptime.com/blog/post/2026-01-30-rag-text-extraction/view))
- This isn't just a truism — it's now measured directly. A 2026 study found even leading OCR engines score ~4.5% lower (NDCG@5) than ground-truth text, and **improving OCR transcription alone recovers most of the performance gap previously blamed on the retriever** — i.e., a lot of "retrieval quality" problems are actually extraction problems in disguise. Their conclusion: treat OCR as a first-class, separately-evaluated pipeline component, not a preprocessing afterthought. ([mixedbread.com — "The Hidden Ceiling: How OCR Quality Limits RAG Performance"](https://www.mixedbread.com/blog/the-hidden-ceiling))
- Practical rules: use native extractors (PDFPlumber, PyMuPDF) for born-digital PDFs to preserve structure, and reserve OCR for genuinely scanned/image-only pages; run image preprocessing before OCR; use specialized table extractors rather than letting layout collapse into prose; and **validate extraction quality before indexing** — spot-check a sample against the source, don't assume the extractor worked. ([Medium — PDF parser comparison](https://infinityai.medium.com/3-proven-techniques-to-accurately-parse-your-pdfs-2c01c5badb84))

**Diagnosability**: keep raw extracted text queryable/inspectable separately from the chunked/embedded version, so a bad answer can be traced back to "the source text itself was wrong" before you start suspecting the retriever or the LLM.

---

## 3. Chunking — bigger lever than most people expect, and the evidence is mixed

- A Vectara study testing 25 chunking configurations against 48 embedding models found **chunking configuration had as much or more influence on retrieval quality as the choice of embedding model itself** — it's not a minor knob. ([Qdrant — RAG Evaluation Guide](https://qdrant.tech/blog/rag-evaluation-guide/))
- Don't assume "semantic chunking" (splitting on topic-coherence boundaries) is automatically better. Recent benchmarks are mixed: a Feb-2026 benchmark of 7 chunking strategies across 50 academic papers found **plain recursive 512-token splitting won at 69% accuracy**, while semantic chunking landed at only 54% after producing fragments averaging just 43 tokens — too small to carry a full idea. The recommended path: **start with recursive chunking as your baseline**, measure against your golden query set, and only move to semantic/hierarchical/agentic chunking if that baseline measurably underperforms for your query types. ([premai.io — RAG Chunking Strategies: The 2026 Benchmark Guide](https://www.premai.io/blog/rag-chunking-strategies-the-2026-benchmark-guide/), [Firecrawl — Best Chunking Strategies 2026](https://www.firecrawl.dev/blog/best-chunking-strategies-rag))
- **Late chunking** (the one genuinely new chunking idea in the last two years, from Jina AI): reverse the usual order — embed the *entire* document first with a long-context model to get token-level embeddings that already carry full-document context, **then** split into chunks and mean-pool each slice. Because each chunk embedding "saw" the whole surrounding document before it was cut, cross-references and long-distance dependencies that plain chunk-then-embed loses are preserved. ([RAGFlow 2025 year-end review](https://ragflow.io/blog/rag-review-2025-from-rag-to-context))
- Microsoft's own Azure architecture guidance frames chunking as its own tunable phase with its own success criteria, separate from embedding/retrieval tuning — worth reading if you want a vendor-neutral checklist. ([Microsoft Learn — RAG Chunking Phase](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-chunking-phase))

**Diagnosability**: log which chunk(s) *should* have answered each golden question, and check per-run whether they were even ingested as their own chunk (not split across two, not buried in an oversized one) before blaming embedding or retrieval.

---

## 4. Embedding quality — measure it, don't assume it

- The standard external benchmark is **MTEB** (Massive Text Embedding Benchmark), which standardized evaluation across 8 task types including retrieval; **MMTEB** extended it to 250+ languages and 500+ tasks. The MTEB leaderboard now has a dedicated **RTEB** (Retrieval Embedding Benchmark) section specifically for retrieval quality across production domains like legal, finance, code, and medical — closer to what an actual RAG system needs than generic MTEB averages. Default retrieval metric: **nDCG@10**. ([emergentmind — MTEB English Leaderboard](https://www.emergentmind.com/topics/mteb-english-leaderboard))
- MTEB's own acknowledged limitation: real-world domains like finance, culture, and health are still under-covered — so a top MTEB score doesn't guarantee it'll be the best embedding model for *your* specific corpus. Validate on your own golden dataset, not just the public leaderboard rank. (same source)
- **Symmetric vs. asymmetric embedding**: if your embedding model distinguishes query vs. document intent (Vertex's `text-embedding-005`, for instance, has separate `RETRIEVAL_QUERY`/`RETRIEVAL_DOCUMENT` task types), using the wrong one for either side measurably degrades retrieval — this is a one-line config check worth verifying explicitly, not assuming it's correct.
- **Multi-vector / late-interaction embeddings (ColBERT-style)**: instead of pooling a chunk into one vector, keep a contextual embedding *per token* and score query-document pairs by token-level max-similarity ("late interaction"). This captures fine-grained term matching that single-vector cosine similarity misses. Jina-ColBERT-v2 combines this with **Matryoshka Representation Learning** — training one model with several nested output-dimension sizes, so you can trade off storage/speed vs. accuracy without retraining — and cuts storage up to 50% versus earlier multi-vector models. ([Weaviate — Late Interaction Overview](https://weaviate.io/blog/late-interaction-overview), [Jina AI](https://jina.ai/news/jina-colbert-v2-multilingual-late-interaction-retriever-for-embedding-and-reranking/))

**Diagnosability**: run your golden query set through the embedding+retrieval step *alone* (no reranker, no LLM) and check recall@k directly — this isolates "did the right chunk even get embedded close enough to be findable" from every downstream stage.

---

## 5. Retrieval — hybrid search is now the default recommendation, not an advanced option

- Nearly every 2024–2026 public benchmark shows **BM25 (lexical/keyword) + dense embeddings, fused with Reciprocal Rank Fusion (RRF), beats either one alone.** Pure semantic search misses exact-term/acronym/ID matches that keyword search catches trivially, and pure keyword search misses paraphrase/synonym matches that embeddings catch. ([RAGFlow 2025 year-end review](https://ragflow.io/blog/rag-review-2025-from-rag-to-context))
- Query-side technique — **HyDE (Hypothetical Document Embeddings)**: instead of embedding the user's short query directly, first ask an LLM to write a hypothetical answer document, then embed *that* and search with it. The insight is that even a factually wrong hypothetical answer still has the right semantic *shape* as real matching documents, closing the "short query vs. long document" semantic gap. Trade-off: adds an LLM call's latency to every request. **HyPE** moves the equivalent work offline instead — generating hypothetical questions per chunk at ingestion time and storing them alongside the chunk, so query time stays fast. ([GeeksforGeeks — HyDE](https://www.geeksforgeeks.org/data-science/hypothetical-document-embeddings-hyde-hyde/), [Medium — RAG Series: HyDE and HyPE](https://medium.com/@hari.haran849/rag-series-5-25d5e57b9d8b))
- **Reranking**: retrieve a wider first-pass candidate set (commonly k≈20) with cheap vector/hybrid search, then rerank with a cross-encoder that jointly attends over the (query, candidate) pair — far more accurate than independent-encoding cosine similarity, at the cost of being too slow to run over the whole corpus. Practical latency numbers: a local MiniLM cross-encoder reranking 50 candidates on CPU runs ~100–250ms; Cohere's hosted Rerank API adds ~150–400ms plus network; FlashRank (a lightweight local reranker) is closer to 15–30ms. ([Towards Data Science — Advanced RAG Retrieval: Cross-Encoders & Reranking](https://towardsdatascience.com/advanced-rag-retrieval-cross-encoders-reranking/), [ZeroEntropy — Ultimate Guide to Choosing a Reranking Model](https://zeroentropy.dev/articles/ultimate-guide-to-choosing-the-best-reranking-model-in-2025/))
- Watch for **vector search dilution**: pulling in too many documents/too broad a candidate pool can actively hurt answer quality, not just add noise — a 2026 paper specifically studies this "more documents hurt RAG" effect and argues for domain-scoped retrieval rather than always widening `k`. ([arXiv 2606.11350 — When More Documents Hurt RAG](https://arxiv.org/pdf/2606.11350))

**Diagnosability**: evaluate retrieval (recall@k, nDCG@k against golden questions) *before* reranking and *after* reranking separately — this tells you whether a bad final answer traces to "never retrieved" vs. "retrieved but reranked away" vs. "correctly retrieved and ranked, but the LLM still got it wrong."

---

## 6. The "Seven Failure Points" framework — a genuinely useful stage-by-stage diagnosis vocabulary

This is likely close to what the reel you saw was gesturing at — a widely-cited academic paper that names exactly where a RAG pipeline breaks, stage by stage, from three real case studies (research, education, biomedical domains):

1. **Missing Content** — the answer genuinely isn't in the corpus at all (a data-coverage problem, not a retrieval or generation problem).
2. **Missed the Top-Ranked Documents** — the answer exists in the corpus but didn't rank high enough to be returned (a retrieval/embedding/ranking problem).
3. **Not in Context** — the right document was retrieved but got dropped during context assembly/consolidation before reaching the LLM (a pipeline-plumbing problem).
4. **Not Extracted** — the answer was in the LLM's context, but the model failed to pull it out, typically due to noise or contradicting information nearby (a generation/prompting problem).
5. **Wrong Format** — the answer was extracted correctly but not returned in the format the query implicitly demanded (table, list, etc.).
6. **Incorrect Specificity** — the answer is present but too vague or too narrow for what was actually asked.
7. (A seventh point is documented in the full paper, referenced but not detailed in the summaries pulled here — see the paper directly.)

Their two key conclusions worth internalizing: **validating a RAG system is only really feasible during live operation** (not purely in a lab eval), and **robustness evolves over the system's life rather than being fully designed in up front** — i.e., this is inherently an iterative-diagnosis discipline, not a one-time build. ([arXiv 2401.05856 — Seven Failure Points When Engineering a RAG System](https://arxiv.org/abs/2401.05856), [full PDF](https://arxiv.org/pdf/2401.05856))

A complementary, longer list from practical production experience: **12 RAG Pain Points and Proposed Solutions**, covering the same territory with concrete fixes per failure mode. ([Towards Data Science](https://towardsdatascience.com/12-rag-pain-points-and-proposed-solutions-43709939a28c/))

**This is the actual mechanism for "verify each stage and diagnose one at a time":** map every real failure you see back to one of these categories before trying to fix it — a "wrong answer" bug fixed by re-tuning the embedding model when the real cause was failure point #3 (context assembly) will look fixed by luck and regress later.

---

## 7. End-to-end evaluation frameworks — automating the stage-by-stage check

- **RAGAS**: the most RAG-specific, lightweight, open-source framework. Pioneered *reference-free* evaluation (no need for hand-labeled ground truth) using LLM-as-judge scoring across four core metrics: **context precision, context recall, faithfulness, and answer relevancy.** Good default choice if you want metrics with minimal setup. ([Atlan — LLM Evaluation Frameworks Compared 2026](https://atlan.com/know/llm-evaluation-frameworks-compared/))
- **TruLens**: pairs RAG-specific scoring with OpenTelemetry-based tracing, so every retrieval call, LLM generation, and tool invocation becomes a structured span you can inspect individually — this is the framework built specifically for "diagnose which stage failed," not just "get an aggregate score." Its core concept, the **RAG Triad**, is three checks that together indicate freedom from hallucination: **context relevance** (did retrieval get the right stuff), **groundedness** (is the answer actually supported by the retrieved context), and **answer relevance** (does the answer actually address the question). ([TruLens — RAG Triad docs](https://www.trulens.org/getting_started/core_concepts/rag_triad/), [GitHub](https://github.com/truera/trulens))
- **DeepEval** is a third common option, generally compared alongside the two above. A practical combination several sources converge on: use **RAGAS or DeepEval for the metrics**, and **TruLens, Phoenix, or Langfuse for the tracing/observability layer** that actually lets you click into *why* a given run scored badly. ([Analytics Vidhya — RAGAS vs TruLens vs DeepEval](https://www.analyticsvidhya.com/blog/2026/07/rag-evaluation-frameworks/))

**Diagnosability payoff**: once tracing is wired in, a bad answer isn't a mystery — you open the trace, see the retrieval span's context-relevance score, the generation span's groundedness score, and immediately know which stage to go fix, rather than guessing.

---

## 8. Recent breakthroughs worth knowing about specifically

- **Anthropic's Contextual Retrieval (Sept 2024)** — arguably the most cited practical RAG improvement of the last two years. Two sub-techniques: **Contextual Embeddings** (prepend a short LLM-generated explanation of a chunk's place in the whole document before embedding it, so the chunk's vector stops losing document-level context) and **Contextual BM25** (apply the same context-prepending to the lexical/keyword index). Anthropic reports this **reduces failed retrievals by 49%**, and **by 67% when combined with reranking**. They published a runnable cookbook alongside the announcement. ([Anthropic Engineering — Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval), [Instructor — implementing it with async processing](https://python.useinstructor.com/blog/2024/09/26/implementing-anthropics-contextual-retrieval-with-async-processing/))
- **GraphRAG (Microsoft Research)** — builds an entity/relationship graph from the corpus and traverses it at retrieval time instead of relying only on vector similarity; substantially outperforms vector-only RAG on *global* questions (ones that require synthesizing across many documents, which flat chunk retrieval structurally can't do well). Their 2025 follow-up, **LazyGraphRAG**, cut indexing cost to roughly 0.1% of the original method, making the approach practical at large corpus scale rather than a research-only technique. (Referenced via [RAG Architecture 2026 overview](https://futureagi.com/blog/rag-architecture-llm-2025/); search Microsoft Research's GraphRAG GitHub/blog directly for the primary source.)
- **Agentic RAG** — the structural shift from "retrieve once, then generate" to putting retrieval *inside* the model's own reasoning loop (Self-RAG, FLARE, and similar patterns), so the model can decide mid-answer that it needs more evidence, rewrite its own query, or stop early once it has enough — multi-hop questions get handled by the agent orchestrating several retrieval calls rather than a single static pipeline pass. ([RAGFlow — From RAG to Context: 2025 year-end review](https://ragflow.io/blog/rag-review-2025-from-rag-to-context))
- **Late chunking** (§3 above) is frequently singled out as the first genuinely new idea in the chunking/retrieval space in roughly two years, rather than a variation on existing techniques.

---

## 9. Compressed checklist

1. Write 30–50 golden queries (with expected source/answer) **before** finalizing chunking — version this set, re-run it after every pipeline change.
2. Validate text extraction quality directly against source documents; don't assume the parser/OCR worked.
3. Start chunking with plain recursive splitting as a baseline; only add semantic/hierarchical/late chunking if you can show it beats the baseline on your golden set.
4. Check your embedding model against RTEB/MTEB for your domain, but verify on your own data too — and confirm query/document task-type asymmetry is configured correctly if your model supports it.
5. Use hybrid search (dense + BM25 via RRF) rather than dense-only; consider HyDE/HyPE for the query side if queries are much shorter than the documents.
6. Retrieve wide, rerank narrow — a cross-encoder or hosted reranker over your top-k candidates measurably improves precision.
7. Evaluate retrieval and generation **separately** (recall@k pre-rerank, post-rerank, then groundedness/faithfulness of the final answer) so a bad output can be traced to one stage.
8. When something breaks, classify it against the Seven Failure Points before changing anything — fix the actual failing stage, not the first plausible-looking knob.
9. Wire in RAGAS/DeepEval for metrics and TruLens/Phoenix/Langfuse for per-span tracing so this diagnosis is a normal, repeatable workflow rather than manual detective work each time.

---

### All sources referenced above

- [Best Practices in RAG Evaluation — Qdrant](https://qdrant.tech/blog/rag-evaluation-guide/)
- [Finding the Best Chunking Strategy for Accurate AI Responses — NVIDIA](https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses/)
- [RAG Chunking Strategies: The 2026 Benchmark Guide — PremAI](https://www.premai.io/blog/rag-chunking-strategies-the-2026-benchmark-guide/)
- [Best Chunking Strategies for RAG (and LLMs) in 2026 — Firecrawl](https://www.firecrawl.dev/blog/best-chunking-strategies-rag)
- [Chunking Strategies to Improve LLM RAG Pipeline Performance — Weaviate](https://weaviate.io/blog/chunking-strategies-for-rag)
- [Chunking Strategies for RAG: Best Practices and Key Methods — Unstructured.io](https://unstructured.io/blog/chunking-for-rag-best-practices)
- [Develop a RAG Solution on Azure — Chunking Phase — Microsoft Learn](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-chunking-phase)
- [Contextual Retrieval in AI Systems — Anthropic Engineering](https://www.anthropic.com/engineering/contextual-retrieval)
- [Implementing Anthropic's Contextual Retrieval with Async Processing — Instructor](https://python.useinstructor.com/blog/2024/09/26/implementing-anthropics-contextual-retrieval-with-async-processing/)
- [Building a "Golden Dataset" for AI Evaluation — Maxim AI](https://www.getmaxim.ai/articles/building-a-golden-dataset-for-ai-evaluation-a-step-by-step-guide/)
- [What is RAG evaluation? — Braintrust](https://www.braintrust.dev/articles/what-is-rag-evaluation)
- [RAG evaluation metrics — Braintrust](https://www.braintrust.dev/articles/rag-evaluation-metrics)
- [The Hidden Ceiling: How OCR Quality Limits RAG Performance — Mixedbread](https://www.mixedbread.com/blog/the-hidden-ceiling)
- [How to Build Text Extraction — OneUptime](https://oneuptime.com/blog/post/2026-01-30-rag-text-extraction/view)
- [Fix RAG Hallucinations at the Source: Top PDF Parsers Ranked 2025 — Medium](https://infinityai.medium.com/3-proven-techniques-to-accurately-parse-your-pdfs-2c01c5badb84)
- [MTEB English Leaderboard Benchmark — emergentmind](https://www.emergentmind.com/topics/mteb-english-leaderboard)
- [An Overview of Late Interaction Retrieval Models: ColBERT, ColPali, ColQwen — Weaviate](https://weaviate.io/blog/late-interaction-overview)
- [Jina-ColBERT-v2 — Jina AI](https://jina.ai/news/jina-colbert-v2-multilingual-late-interaction-retriever-for-embedding-and-reranking/)
- [From RAG to Context — A 2025 year-end review of RAG — RAGFlow](https://ragflow.io/blog/rag-review-2025-from-rag-to-context)
- [RAG Architecture 2026: Patterns, Code, and Eval — FutureAGI](https://futureagi.com/blog/rag-architecture-llm-2025/)
- [Introduction to Hypothetical Document Embeddings (HyDE) — GeeksforGeeks](https://www.geeksforgeeks.org/data-science/hypothetical-document-embeddings-hyde-hyde/)
- [RAG Series — Part 5: HyDE and HyPE — Medium](https://medium.com/@hari.haran849/rag-series-5-25d5e57b9d8b)
- [Advanced RAG Retrieval: Cross-Encoders & Reranking — Towards Data Science](https://towardsdatascience.com/advanced-rag-retrieval-cross-encoders-reranking/)
- [Ultimate Guide to Choosing the Best Reranking Model in 2026 — ZeroEntropy](https://zeroentropy.dev/articles/ultimate-guide-to-choosing-the-best-reranking-model-in-2025/)
- [When More Documents Hurt RAG — arXiv 2606.11350](https://arxiv.org/pdf/2606.11350)
- [Seven Failure Points When Engineering a Retrieval Augmented Generation System — arXiv 2401.05856](https://arxiv.org/abs/2401.05856) ([PDF](https://arxiv.org/pdf/2401.05856))
- [12 RAG Pain Points and Proposed Solutions — Towards Data Science](https://towardsdatascience.com/12-rag-pain-points-and-proposed-solutions-43709939a28c/)
- [RAGAS, TruLens, DeepEval: LLM Evaluation Frameworks (2026) — Atlan](https://atlan.com/know/llm-evaluation-frameworks-compared/)
- [RAG Evaluation Frameworks Compared: RAGAS vs TruLens vs DeepEval — Analytics Vidhya](https://www.analyticsvidhya.com/blog/2026/07/rag-evaluation-frameworks/)
- [The RAG Triad — TruLens docs](https://www.trulens.org/getting_started/core_concepts/rag_triad/)
- [TruLens — GitHub](https://github.com/truera/trulens)

---

*Compiled via web research, September 2026. Sources are third-party blogs, vendor engineering blogs, and arXiv preprints as cited inline — treat benchmark numbers (e.g. specific accuracy percentages) as reported by their original authors, not independently re-verified here.*
