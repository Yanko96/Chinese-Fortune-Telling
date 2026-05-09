# Master Guo Lai's Fortune Hall — Chinese Divination RAG

![Fortune Teller Banner](./docs/images/fortune_banner1.png)

A production-deployed RAG system over three classical Chinese divination texts (《三命通会》《滴天髓》《子平真诠》), built end-to-end:

- **Retrieval research** — 9 strategies × 28 configurations, evaluated by GPT-4o with Chinese eval embeddings
- **Full-stack app** — FastAPI + Streamlit + ChromaDB, history-aware HyDE + BGE Rerank pipeline
- **Production deployment** — AWS ECS / ALB / Terraform, GitHub Actions CI/CD, with 5 real incidents documented and resolved

> 📑 In-depth docs: [Architecture](docs/ARCHITECTURE.md) · [Retrieval Benchmark](docs/BENCHMARK_REPORT.md) · [Deployment Notes](docs/DEPLOYMENT_NOTES.md)

---

## Architecture at a Glance

> **Production retrieval is router-dispatched**: HyDE + BGE Rerank by default (`k=8 → top_n=5`, single-hop winner AVG=0.812), with Graph RAG v7 (`vector_filter_k=50`, multihop winner chain_score=0.729) invoked when a heuristic query classifier detects high-confidence cross-book / multihop intent.
> See [§1a HyDE+Rerank](#1a-hyde--bge-rerank--production-default), [§1b Graph RAG](#1b-graph-rag--multihop-specialist), [§1c Query Router](#1c-query-router--picks-hyde-vs-graph-per-request).

```mermaid
flowchart LR
    User([User]) -->|HTTPS| ALB[AWS ALB]
    ALB -->|/| Streamlit[Streamlit App]
    ALB -->|/api/*| API[FastAPI]
    Streamlit -->|REST| API
    API -->|HyDE → embed → rerank → generate| Pipeline{{HyDE + BGE Rerank<br/>k=8, top_n=5}}
    Pipeline --> Kimi[Kimi LLM<br/>moonshot-v1-8k]
    Pipeline --> Chroma[(ChromaDB<br/>671 chunks<br/>bge-small-zh-v1.5)]
    Pipeline --> BGE[BGE Cross-Encoder<br/>bge-reranker-base<br/>baked in image]
```

Per-request flow (the only path in production):

1. **HyDE** — Kimi writes an 80–150 char *hypothetical* classical-Chinese passage in the answering style. This gives the dense retriever a query that looks like the corpus, not like a question.
2. **Wide recall** — embed the hypothetical via `bge-small-zh-v1.5`, similarity-search **k=8** candidates against the ChromaDB index.
3. **Rerank** — `bge-reranker-base` cross-encoder re-scores each candidate against the **original** user question (not the hypothesis), keep **top 5**. The cross-encoder is lexical-aware where the dense embedding isn't, so it catches matches the wider HyDE recall over-generalized away.
4. **Generate** — stuff top-5 chunks into the role-played QA prompt (果赖 persona for BaZi/Forecast, evidence-grounded otherwise), generate via Kimi `moonshot-v1-8k`.

The two LLM calls (HyDE + final generation) plus one rerank pass land end-to-end p50 around **22–30 seconds** on a 0.5 vCPU ECS task — see [docs/DEPLOYMENT_NOTES.md §5](docs/DEPLOYMENT_NOTES.md) for the latency budget that drove the `k=15→8 / top_n=7→5` choice.

Full diagram and per-decision rationale → [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Quick Start

Prerequisites: Docker + Docker Compose, a [Kimi (Moonshot) API key](https://platform.moonshot.cn).

```bash
git clone https://github.com/Yanko96/Chinese-Fortune-Telling.git
cd Chinese-Fortune-Telling

# Configure secrets
cp .env.example .env
# Edit .env: set MOONSHOT_API_KEY=sk-...

# Launch the full stack (Dockerfile builds the vector index from
# fortune_books/ at image build time — no separate setup step)
docker-compose up --build
```

Open <http://localhost:8501> in a browser. Health check at <http://localhost:8000/api/healthz>.

For a manual (non-Docker) setup, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## What's in this repo

### 1. Retrieval Research

A systematic benchmark comparing **9 retrieval strategies** across two evaluation dimensions, totaling **28 experiment configurations**. Two methods are documented in depth below: **HyDE + BGE Rerank** (deployed to production) and **Graph RAG** (offline research, multihop winner).

| Dataset | Questions | Scope | Evaluation |
|---|:-:|---|---|
| `qa_dataset.json` (Normal) | 22 | Single-book, single-hop | RAGAS 4 metrics — faithfulness, relevancy, recall, precision |
| `qa_multihop.json` (Multihop) | 36 | Cross-book, 3-step reasoning | LLM `chain_score` (per-step 0 / 0.5 / 1) |

**Top results** (highlights — full tables in [BENCHMARK_REPORT.md](docs/BENCHMARK_REPORT.md)):

| Task | Winner | Score | Why it wins | Production role |
|---|---|---|---|---|
| Normal (22Q AVG) | **HyDE + Rerank** | **0.812** | Faithfulness 0.917; hypothesis-conditioned retrieval beats keyword/dense alone | ✅ Default branch (most queries) |
| Multihop (36Q chain) | **Graph RAG v7** (`vector_filter_k=50`) | **0.729** | Semantic gate on graph neighbors solves cross-book reasoning | ✅ Router-dispatched (multihop queries) |
| Cross-book hit | Graph RAG v8 (k=20) | **91.7%** | IDF-weighted edges bridge concepts across《三命通会》/《滴天髓》/《子平真诠》 | 🔬 Offline (k=15 in router) |

#### 1a. HyDE + BGE Rerank — production default

The default branch of the router. Wins single-hop (AVG=0.812, faithfulness=0.917 on 22Q) and remains the highest-faithfulness method across the entire 28-config sweep — important for a system whose users read the answer as authoritative classical exegesis.

**Why HyDE here, specifically?**
Direct dense retrieval over Chinese classical text has a query↔document distribution gap: the user asks `"什么是正财格？"` in modern Chinese, but the answer chunks are in classical Chinese (`"正财者，乃甲见己、乙见戊之例。受我克制，为我之妻..."`). Embedding the question directly underweights the right chunks. HyDE closes the gap by asking the LLM to **write a classical-Chinese passage in the answering style first**, then embedding *that* — the retrieval seed now lives in the same distribution as the corpus.

```
Question (modern Chinese)
  └── HyDE prompt asks Kimi to write 80–150 char classical-style passage
        └── Embed hypothetical passage with bge-small-zh-v1.5 (384d)
              └── ChromaDB similarity search → 8 candidate chunks (wide recall)
                    └── BGE Cross-Encoder rerank against ORIGINAL question
                          └── Top 5 chunks → role-played QA prompt → Kimi answer
```

**Why rerank, on top of HyDE?**
HyDE expands recall but introduces noise — the LLM-written hypothetical sometimes drifts. The BGE cross-encoder is a different model class (not a dual-encoder); it scores `(query, chunk)` pairs **jointly**, which catches lexical/keyword matches that the dense step's bag-of-meaning embedding smooths over. The two errors are negatively correlated: HyDE-only loses on precision when hypothesis drifts; Rerank-only loses on recall when the surface form of the question doesn't match the canonical text. Stacked, faithfulness jumps from 0.755 (Hybrid+Rerank) to **0.917** (HyDE+Rerank).

**Ablation summary (full table in BENCHMARK_REPORT):**

| Config | Faithfulness | Relevancy | Recall | Precision | AVG |
|---|:-:|:-:|:-:|:-:|:-:|
| **HyDE + Rerank** (production) | **0.917** | 0.651 | 0.829 | 0.850 | **0.812** |
| Graph RAG v8 (bge-base, k=15) | 0.841 | 0.656 | 0.849 | 0.868 | 0.804 |
| Hybrid (BM25 + Vector) | 0.762 | 0.668 | 0.846 | 0.885 | 0.790 |
| Hybrid + BGE Rerank | 0.755 | 0.664 | 0.720 | 0.894 | 0.758 |
| Proposition (Vector) | 0.492 | 0.667 | 0.567 | 0.719 | 0.611 |

**Production-specific design notes:**

| Knob | Value | Why |
|---|---|---|
| `hyde_k` (wide recall) | 8 | Benchmark winner is k=15, but `k=8` is the p95 latency line for a 0.5 vCPU ECS task with 120s ALB timeout. Faithfulness loss vs k=15 is <2pp; latency improvement is 70s→30s. See [DEPLOYMENT_NOTES §5](docs/DEPLOYMENT_NOTES.md). |
| `top_n` (after rerank) | 5 | Reduced from 7 alongside `hyde_k` cut. Generation prompt fits comfortably under 32k context. |
| HyDE prompt language | 文言文 (classical Chinese) | Critical — modern Chinese hypotheticals match poorly with the corpus's classical phrasing. The prompt explicitly says "仿照古典命理文献的风格". |
| Rerank scoring text | original Chinese question | Reranking against the HyDE hypothetical would compound HyDE's noise. The cross-encoder is meant to **counter** the HyDE drift, not amplify it. |
| Prefix-strip before HyDE | regex strip of `"BaZi analysis for..."` wrapper | Production's BaZi/Forecast paths prepend an English query rewriter to the user's Chinese question, which polluted HyDE. Shadow eval measured −0.5 retrieval quality; the fix recovers most of it. Code at `api/fortune_langchain_utils.py:_strip_query_prefix`, tests at `tests/test_query_prefix_strip.py`. |

Code: [`api/fortune_langchain_utils.py`](api/fortune_langchain_utils.py) (130 lines, no abstractions — `_build_hyde_rerank_retriever` is the whole thing).

#### 1b. Graph RAG — multihop specialist

Routed-to when the query classifier (§1c) detects cross-book / multihop intent. Wins multihop (`chain_score=0.729`, cross-book hit 92%) by combining graph topology (IDF-weighted cross-book edges over a knowledge graph of 668 nodes) with the same vector retrieval + BGE rerank pipeline as the HyDE path.

**The key insight (v7):** combining graph topology with vector semantics beats either alone. Pure-vector retrieval misses the cross-book bridges (different classical texts use different terms for the same concept); pure-graph BFS expansion is noisy (high-degree concept nodes drown the signal). Adding `vector_filter_k` — gate graph neighbors through a top-50 vector recall whitelist — kept the topology benefit while preserving precision.

**8-version method evolution:**

```
v1  flat graph (min_weight=1, 43K edges)        → noisy, low precision baseline
v2  pruned (min_weight=2, 15K edges)            → +3pp from removing weak co-occurrences
v3  IDF-weighted edges + degree pruning (5K)    → discriminative connections, rare-term bridges weighted up
v4  reranker upgrade (bge-v2-m3) + top_n=10     → marginal: retrieval ceiling, not rerank ceiling
v5  HyDE-seeded graph entry                     → better q↔doc match at the seed node
v6  max_neighbors ablation (30→10)              → noise/recall trade-off
v7  vector_filter_k semantic gate               → ★ best multihop (0.729)
v8  bge-base-zh (768d) + k=15                   → ★ best normal-graph (0.804)
```

Why it's gated behind the router instead of replacing HyDE globally:

- **Single-hop loss**: most production traffic is single-hop ("explain my BaZi"). On those, HyDE+Rerank beats Graph (0.812 vs 0.804) AND is more faithful (0.917 vs 0.841)
- **Cold-start cost**: graph + chunk-index load is ~50 ms once, but only triggered on the first multihop route — lazy-loaded by the router so single-hop traffic never pays this cost
- **Operational synchronization**: graph topology has to stay in sync with the embedding index. Two artifacts to rebuild on corpus change vs. one. Per-route gating means only multihop queries are exposed to this drift risk

The router pattern lets us deploy both methods without forcing one to subsume the other.

**Knowledge graph stats:** 668 nodes (419 三命通会 / 154 滴天髓 / 95 子平真诠), 5,255 IDF-weighted cross-book edges (degree-pruned to ≤15 neighbors per node), 41 bridge terms (官星 / 财星 / 印绶 / 用神 …) connecting concepts across texts. Edge weights 3.0–16.5 — higher = rarer shared term = stronger conceptual bridge. Built via `python scripts/build_knowledge_graph.py --chroma-dir ./chroma_db_bge --min-weight 2`.

Code: [`api/graph_retriever.py`](api/graph_retriever.py). **Wired into production via the router below** — Graph RAG runs when the query classifier identifies a high-confidence multihop / cross-book pattern.

#### 1c. Query Router — picks HyDE vs. Graph per request

Neither method dominates the full query distribution: HyDE wins single-hop, Graph wins multihop. Rather than picking one globally, a lightweight heuristic router classifies each incoming query and dispatches to the better strategy.

**Policy (conservative, defaults to HyDE):**

```python
def should_route_to_graph(query: str) -> bool:
    if distinct_book_mentions(query) >= 2:        # 《X》《Y》对比
        return True
    if bridge_terms(query) >= 2 and has_compare_keyword(query):
        return True
    return False  # HyDE — the production-safe single-hop winner
```

**Three independent detectors** (each unit-tested in `tests/test_retriever_router.py`):

| Detector | Method | False-positive guard |
|---|---|---|
| `detect_cross_book` | Regex over `《...》` Chinese book quote marks | Distinct count (same book quoted 5× still counts 1) |
| `has_compare_signal` | Keyword regex: 对比 / 区别 / 异同 / 差异 / vs / 相较 / 哪个更 …  | Excludes "比较好" / "比较一下" filler matches |
| `detect_multi_entity` | Hits against the 35 bridge-terms vocabulary the knowledge graph was built on | Reusing the curated vocab means high precision for free — no NER model needed |

**Why conservative?** HyDE is the single-hop benchmark winner AND faster on a 0.5 vCPU task (~30 s vs Graph's ~15 s ... wait, the multihop is *faster* because graph hits land sooner). Setting both 2-book and 2-entity+compare thresholds means routing only fires on high-confidence cases. Worst case = identical to pre-router production (always HyDE).

**Verified live**:

| Query | Decision | Reason logged | Latency |
|---|---|---|---|
| `什么是正财格？` | **HyDE** | `default_hyde (cross_book=0, multi_entity=1, compare=False)` | ~30 s |
| `《滴天髓》和《子平真诠》对正官的论述有何不同？` | **Graph** | `cross_book_mentions=2` | ~15 s |

Reason strings are logged at INFO so CloudWatch can audit every routing decision without re-running the heuristic.

**Future work** (documented as roadmap, not implemented):

1. **Labeled router-accuracy eval**: hand-tag ~50 queries by ideal-retriever, measure router precision/recall, tune thresholds. Currently the heuristic is design-by-inspection.
2. **Learned router**: replace heuristics with a tiny logistic-regression classifier over the same features + a learned query embedding. Unlikely to help much until traffic justifies the labeling effort.
3. **Per-route latency-aware fallback**: if Graph latency exceeds a budget on a given query, abort + fall back to HyDE.
4. **Telemetry → tuning loop**: log routing decisions + downstream answer quality (from RAGAS-style auto-eval on production traffic) → tune router thresholds monthly.

Code: [`api/retriever_router.py`](api/retriever_router.py) — 200 lines, no abstractions, every routing decision is one `should_route_to_graph()` call away from being audited.

### 2. System Design

- `api/` — FastAPI, history-aware retrieval chain via LangChain, lazy-loaded ChromaDB + BGE
- `app/` — Streamlit UI: BaZi birth-date picker, zodiac selector, chat history
- `scripts/` — Reproducible offline tools: index build, knowledge-graph build, full benchmark sweep, GPT-4o rescoring
- `terraform/` — Modular IaC: vpc / alb / ecs / ecr, separate `environments/production` workspace

Key design decisions and the trade-offs behind them → [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

### 3. Production Deployment

CI/CD pipeline (`.github/workflows/deploy.yml`):

```
push to main → GitHub Actions
              ├── docker build api + app (commit-SHA tagged)
              ├── push to ECR
              └── terraform apply (force_new_deployment=true)
                  → ECS rolls new tasks → ALB picks them up
```

Required GitHub Secrets:

| Secret | Used for |
|---|---|
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | ECR push, Terraform (S3 backend, DynamoDB lock, ECS, ECR, ALB, VPC, CloudWatch) |
| `MOONSHOT_API_KEY` | Passed by Terraform into the API container at deploy time |

After a successful deploy, the ALB DNS is printed to the GitHub Actions Job Summary.

**Five production incidents documented** — Gemini key expiry → Kimi migration, rolling-deploy enum mismatch, leftover `.value` calls, BGE reranker OOM, ALB 504 timeout — each with root cause and resolution in [docs/DEPLOYMENT_NOTES.md](docs/DEPLOYMENT_NOTES.md).

### 4. Evaluation Methodology

Two non-trivial methodology choices, each measurably moving the leaderboard:

| Choice | What | Why it matters |
|---|---|---|
| **GPT-4o as judge** (not Kimi self-eval) | All RAGAS scores re-run via GPT-4o | Kimi-judging-Kimi was 10–33% more lenient than GPT-4o on the same 22Q set — a real self-evaluation bias |
| **Chinese eval embedding** (bge-small-zh-v1.5, not `all-MiniLM-L6-v2`) | Used for `answer_relevancy` cosine similarity | Boosted answer_relevancy by **+12.5pp** — English embeddings undercount Chinese semantic overlap |

Custom `chain_score` metric for multihop: each reasoning step scored 0 / 0.5 / 1 against the gold answer, then averaged across all hops in the chain. See [BENCHMARK_REPORT.md §7](docs/BENCHMARK_REPORT.md) for the rubric and inter-annotator-style calibration runs.

---

## Reproducing the Benchmark

```bash
# Normal benchmark (22Q, RAGAS, GPT-4o judge)
python scripts/rag_bench.py \
    --configs configs/rag/v5/hyde_rerank_topn7.yaml \
    --dataset benchmarks/qa_dataset.json \
    --output-dir benchmarks/results/my_run \
    --eval-provider openai --eval-model gpt-4o

# Multihop benchmark (36Q, chain_score)
python scripts/bench_multihop.py \
    --configs configs/rag/v8/graph_rag_v7_vf50.yaml \
    --dataset benchmarks/qa_multihop.json \
    --no-parallel

# Rescore stored results with GPT-4o
python scripts/rescore_gpt4o.py             # multihop
python scripts/rescore_normal_gpt4o.py      # normal
```

---

## Known Limitations

- **Production-vs-benchmark eval gap (measured + partially closed).** Offline benchmark uses pure Chinese questions; production prepends `"BaZi analysis for someone born on {date}, gender: {sex}. "`, which pollutes HyDE. A 10-pair controlled shadow eval ([Appendix A](docs/BENCHMARK_REPORT.md#附录-aproduction-shadow-eval)) measured the raw gap at **Δ = −0.5** (3.0 vs. 3.5 / 5). A `_strip_query_prefix` fix (`api/fortune_langchain_utils.py`) cut the gap to **Δ = −0.3 (40% closure)**; the residual is consistent with HyDE's `temperature=0.7` stochasticity at this sample size. A larger fixed-dataset + deterministic re-run is on the roadmap for a p<0.05 confirmation.
- **Single-region, single-AZ deploy.** Production runs in `us-east-1` only. Adding a second AZ in the ALB target group is a 1-line Terraform change; a second region needs cross-region ECR replication and would change the Terraform layout.
- **No streaming responses.** The rerank step is the latency dominator, not the LLM output — streaming would add complexity without UX improvement here.

---

## Tech Stack

| Layer | Choice |
|---|---|
| Backend | FastAPI · LangChain (`langchain-openai`, `langchain-classic`) |
| Frontend | Streamlit |
| Vector DB | ChromaDB (file-backed) |
| Embeddings | `BAAI/bge-small-zh-v1.5` (prod) · `bge-base-zh-v1.5` (Graph RAG offline) |
| Reranker | `BAAI/bge-reranker-base` (cross-encoder) |
| LLM | Kimi (Moonshot) `moonshot-v1-8k` / `-32k` / `-128k` via OpenAI-compatible API |
| Eval | RAGAS · GPT-4o judge · Chinese eval embeddings |
| Infra | AWS ECS Fargate · ALB · ECR · Terraform · GitHub Actions |

---

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- Source texts are public-domain classical Chinese divination works.
- All readings produced by this system are for demonstration and entertainment.
