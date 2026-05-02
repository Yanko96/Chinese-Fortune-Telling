# Architecture & Design Decisions

> Quick links: [back to README](../README.md) · [benchmark](BENCHMARK_REPORT.md) · [deployment notes](DEPLOYMENT_NOTES.md)

## System Diagram

```mermaid
flowchart LR
    User([User])
    User -->|HTTPS| ALB[AWS ALB<br/>:80/:443]

    ALB -->|/| Streamlit[Streamlit App<br/>ECS Service<br/>256 CPU / 512 MB]
    ALB -->|/api/*| FastAPI[FastAPI<br/>ECS Service<br/>512 CPU / 2048 MB]

    Streamlit -->|REST<br/>via ALB /api| FastAPI

    subgraph Retrieval [HyDE + Rerank Pipeline]
        direction TB
        Q[Question]
        Q -->|1. LLM writes<br/>hypothetical<br/>classical passage| Hypo[HyDE document]
        Hypo -->|2. embed +<br/>similarity_search k=8| Chroma[(ChromaDB<br/>bge-small-zh-v1.5<br/>666 chunks)]
        Chroma -->|8 candidates| BGE[BGE Cross-Encoder<br/>bge-reranker-base<br/>baked in image]
        Q -->|3. score against<br/>original question| BGE
        BGE -->|top_n=5| Context[Top 5 chunks]
        Context -->|4. stuff into prompt| Answer[LLM answer]
    end

    FastAPI --> Retrieval
    Retrieval -->|Kimi calls 1+4| Kimi[Kimi<br/>moonshot-v1-8k<br/>OpenAI-compatible API]

    Chroma -. built offline by<br/>scripts/build_index_bge.py .-> Books[fortune_books/<br/>3 PDFs:<br/>《三命通会》<br/>《滴天髓》<br/>《子平真诠》]

    FastAPI -->|chat history| SQLite[(SQLite<br/>rag_app.db<br/>local file)]

    subgraph CICD [CI/CD]
        GH[GitHub push<br/>to main] --> GHA[GitHub Actions]
        GHA -->|docker build + push| ECR[ECR<br/>commit-SHA tags]
        GHA -->|terraform apply| TF[Terraform<br/>force_new_deployment]
        TF --> FastAPI
        TF --> Streamlit
    end
```

---

## Key Design Decisions

### 1. Why HyDE + Rerank in production (not Graph RAG v8)

| Method | Best benchmark | Latency at 512 CPU | Production choice |
|--------|---------------|-------------------|-------------------|
| **HyDE + Rerank** (k=8, n=5) | normal AVG **0.812** | ~30 s | ✅ Production |
| Graph RAG v8 (bge-base, k=15) | normal AVG 0.804, multihop 0.725 | ~60 s + graph load | ❌ |
| Graph RAG v7 (vf=50) | multihop **0.729** | ~50 s | ❌ |

Reasons:
1. **Latency budget**: ALB `idle_timeout` capped at 120 s, but UX target is sub-30 s. Graph RAG adds a 668-node knowledge graph load on cold start + BFS expansion per query.
2. **Single-hop dominance**: Most production questions are single-hop ("explain my BaZi", "what does this hexagram mean"). HyDE+Rerank wins on those.
3. **Operational simplicity**: HyDE+Rerank is two LLM calls and one rerank. Graph RAG adds graph state, edge weights, neighbor pruning hyperparameters that need to stay in sync between offline build and online query.
4. **Quality vs. latency at k=8**: Faithfulness only drops ~2pp from k=15 to k=8 (per BENCHMARK_REPORT §6 ablation). The latency saving is much larger than the quality loss.

The Graph RAG work was preserved as offline research because it produces a real **cross-book reasoning** capability (92% cross_book_hit vs. 39% for pure HyDE). It's the right tool when the question is genuinely multihop — that's the next product iteration, not a launch blocker.

### 2. Why pre-download models in Docker build (not lazy load)

A first-request lazy download of `BAAI/bge-reranker-base` (~280 MB on disk, ~600 MB RSS during load) on a 512 MB ECS task is the **fastest way to discover OOM in production** — see [DEPLOYMENT_NOTES §4](DEPLOYMENT_NOTES.md). Baking the weights into the image:
- Removes a network dependency at request time (no HuggingFace Hub call on cold start)
- Eliminates the 600 MB peak RSS spike at load
- Gives Docker layer caching a free CI speedup
- Costs ~280 MB extra image size — acceptable for a ~1 GB image

### 3. Why a `str` field instead of an enum for `model`

A `ModelName` enum looked tighter in `pydantic_models.py`, but it broke during the Gemini→Kimi cutover because rolling ECS deployments mean **the old frontend (sending `gemini-2.5-flash`) and the new backend (only accepting `moonshot-*`) coexist for ~2 minutes**. Strict enum validation turned that 2-minute window into a hard 422 outage.

The lesson generalized: **API field validation at the boundary should be at least as permissive as the longest expected client-version skew**. For an ECS Fargate rolling deploy that's the deploy duration; for a mobile app it could be months.

### 4. Why ChromaDB (and bge-small-zh-v1.5)

- **ChromaDB** over FAISS / pgvector: file-backed (just copy a directory), no separate service to provision, perfectly adequate for 666 chunks. We'd switch to pgvector or Weaviate at ~100k+ chunks.
- **bge-small-zh-v1.5** (384-dim) over bge-base-zh-v1.5 (768-dim) for the production index: benchmark showed +0.5pp AVG from base over small in HyDE+Rerank — not worth the 2× memory + slower similarity search per query at 512 MB task. Base is used in the offline Graph RAG branch where memory budget is irrelevant.

### 5. Why Streamlit (not React/Next)

A demo UI for a research-flavoured RAG project. Streamlit gives a working chat interface + sidebar controls in ~200 lines, no build step. The cost is real: limited custom styling, full re-render on each interaction. Acceptable here; would not be acceptable for a real consumer product.

### 6. Why /api path routing on a single ALB (not two ALBs)

- One ALB → two target groups (`/` → Streamlit, `/api/*` → FastAPI) saves one ALB monthly cost (~$22/mo)
- Same hostname for browser-to-streamlit and streamlit-to-api means no CORS configuration
- Cost is that the FastAPI container has to mount under `root_path="/api"` (via the `API_ROOT_PATH` env var) so OpenAPI/Swagger URLs stay correct

### 7. Things explicitly chosen *not* to add

- **No streaming responses**: Kimi supports SSE, but the rerank step is the latency dominator, not LLM streaming. Streaming would add complexity without UX improvement for this pipeline.
- **No Redis cache**: Question repetition is low (each user asks personal questions). Cache hit rate would be <5% — not worth a Redis service.
- **No auth/multi-tenancy**: This is a demo. Sessions are tracked by an opaque `session_id` cookie, no users table. Real auth would be week-2 work.

---

## Repo Layout

```
api/                          FastAPI backend (production)
  fortune_main.py             /fortune /healthz /zodiac-signs endpoints
  fortune_langchain_utils.py  HyDE+Rerank chain construction
  fortune_prompts.py          QA prompts (general / bazi / forecast / bench variants)
  graph_retriever.py          Graph RAG retriever (research; not wired into prod)
  chroma_utils.py             ChromaDB lazy loader
  db_utils.py                 SQLite chat history
  Dockerfile                  Bakes BGE + MiniLM models at build time

app/                          Streamlit frontend
  fortune_app.py              Main entry
  fortune_sidebar.py          Birth date / zodiac / model picker
  fortune_chat_interface.py   Chat widget
  fortune_api_utils.py        REST client targeting /api

scripts/                      Offline tools
  build_index_bge.py          (Re)build ChromaDB from fortune_books/
  build_knowledge_graph.py    Build the 668-node IDF graph
  rag_bench.py                28-config retrieval benchmark
  bench_multihop.py           36Q multihop chain_score benchmark
  rescore_gpt4o.py            Re-evaluate stored results with GPT-4o
  generate_qa_dataset.py      LLM-bootstrapped QA dataset construction

configs/rag/                  YAML configs for each benchmark run (v1-v8 + variants)

benchmarks/                   Evaluation harness
  qa_dataset.json             22 single-hop questions (3 books)
  qa_multihop.json            36 multihop questions (cross-book)
  results/                    Representative result snapshots

fortune_books/                PDF source texts (3 classical works)
data/                         Built artifacts: knowledge_graph.pkl, chunk_index.json

terraform/                    Infra-as-code (production environment)
  environments/production/    Backend, vars, top-level main.tf
  modules/                    vpc / alb / ecs / ecr

.github/workflows/deploy.yml  CI/CD: build → ECR → terraform apply

docs/                         Long-form documentation
  ARCHITECTURE.md             (this file)
  BENCHMARK_REPORT.md         Full retrieval study
  DEPLOYMENT_NOTES.md         Production bug fix log
```
