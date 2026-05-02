# Production Deployment — Bug Fixes & Lessons Learned

A running log of production issues encountered on the AWS ECS + ALB stack, and how each was resolved. Kept verbatim from the on-call notes so the failure mode, root cause, and fix stay tied together.

> Quick links: [back to README](../README.md) · [architecture](ARCHITECTURE.md) · [benchmark](BENCHMARK_REPORT.md)

---

## 1. LLM Switch: Google Gemini → Kimi (Moonshot)

**Symptom**: All API calls returned HTTP 500 with `API key expired`.

**Root cause**: The Google Gemini API key had expired and the project's free quota was no longer renewable.

**Fix**: Migrated the entire generation layer from `langchain-google-genai` / `ChatGoogleGenerativeAI` to `langchain-openai` / `ChatOpenAI` pointing at Kimi's OpenAI-compatible endpoint:

```python
llm = ChatOpenAI(
    model="moonshot-v1-8k",
    openai_api_key=os.environ["MOONSHOT_API_KEY"],
    openai_api_base="https://api.moonshot.cn/v1",
)
```

Also updated `requirements.txt` (remove `langchain-google-genai`), all Terraform variables, and the GitHub Actions secret from `GOOGLE_API_KEY` to `MOONSHOT_API_KEY`.

---

## 2. HTTP 422 — Model Enum Rejected During Rolling Update

**Symptom**: HTTP 422 `value is not a valid enumeration member` after switching to Kimi.

**Root cause (first layer)**: The frontend (`app/fortune_sidebar.py`) was still sending `"gemini-2.5-flash"` as the model name, but the new API defined a strict `ModelName` enum that only accepted moonshot model names.

**Fix (first layer)**: Updated all model dropdowns in the frontend to `["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"]`.

**Root cause (second layer)**: During a rolling ECS deployment, the old app container (still sending `"gemini-2.5-flash"`) was hitting the new API container (rejecting unknown enum values). Both services roll out independently, so for ~2 minutes there is always at least one stale frontend talking to a fresh backend.

**Fix (second layer)**: Replace the strict `ModelName` enum with a plain `str` field in both `pydantic_models.py` and `fortune_main.py`:

```python
# Before (strict enum — breaks during rolling updates)
class FortuneInput(BaseModel):
    model: ModelName = ModelName.MOONSHOT_8K

# After (permissive string — safe across rolling updates)
class FortuneInput(BaseModel):
    model: str = Field(default="moonshot-v1-8k")
```

**Lesson**: Never use strict enums for fields that might receive legacy values during an ECS rolling deploy. Use `str` with a sensible default, or add an `alias`/`validator` to map old values. Backwards compatibility at the API boundary has to outlive the deploy window.

---

## 3. HTTP 500 — `'str' object has no attribute 'value'`

**Symptom**: API returned 500 after fixing the enum issue.

**Root cause**: Three places in `fortune_main.py` still called `.value` on the `model` field (leftover from when `model` was an enum):

```python
logging.info(f"Model: {fortune_input.model.value}")
get_fortune_chain(model=fortune_input.model.value)
insert_application_logs(..., fortune_input.model.value)
```

**Fix**: Remove all `.value` calls — `model` is already a plain string:

```python
logging.info(f"Model: {fortune_input.model}")
get_fortune_chain(model=fortune_input.model)
insert_application_logs(..., fortune_input.model)
```

**Lesson**: When a field type changes from enum to string, grep the whole package for `.value` before deploying. A type-checker (mypy) would have caught this; we did not have one wired in at the time.

---

## 4. HTTP 502 — BGE Reranker OOM at Runtime

**Symptom**: API returned 502 Bad Gateway immediately after the HyDE+Rerank upgrade. ECS task stopped with `OutOfMemoryError: Container killed due to memory usage`.

**Root cause**: `BAAI/bge-reranker-base` (~280 MB on disk, ~600 MB peak RSS during load) was being downloaded from HuggingFace Hub on the **first request** inside a 256 CPU / 512 MB ECS task. The lazy download path went: pull weights → load into a `transformers` model → instantiate `CrossEncoder` wrapper. Peak memory during the load spike exceeded 512 MB even before serving the request.

**Fix**: Pre-download both models at **Docker build time** so they are baked into the image:

```dockerfile
# In api/Dockerfile — after pip install, before COPY
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('embedding model OK')"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base'); print('reranker model OK')"
```

> ⚠️ Use **two separate `RUN` lines** — a multiline `python -c "..."` inside a single `RUN` gets silently truncated by some shell versions, causing the second model to never download. We discovered this when the image built fine but the first prod request still OOM'd: the second model was downloading on demand. Splitting the RUN lines also gives a per-model docker layer cache hit, which speeds up CI when only one model changes.

Also bumped task memory to 2048 MB to give the loaded reranker headroom for forward passes.

---

## 5. HTTP 504 — Gateway Timeout

**Symptom**: API returned 504 after ~60 seconds for every fortune request.

**Root cause**: The full HyDE+Rerank pipeline (Kimi call for hypothesis → BGE reranking of 15 candidate pairs → Kimi call for generation) took 70–90 s on a 0.25 vCPU ECS task (256 CPU units). The ALB `idle_timeout` defaults to 60 s.

**Fix — three changes applied together**:

| Change | File | Before → After |
|--------|------|----------------|
| Increase ECS CPU | `terraform/environments/production/variables.tf` | `api_cpu = "256"` → `"512"` |
| Increase ALB timeout | `terraform/modules/alb/main.tf` | default 60 s → `idle_timeout = 120` |
| Reduce reranker candidates | `api/fortune_langchain_utils.py` | `hyde_k=15, top_n=7` → `hyde_k=8, top_n=5` |

Post-fix measured latency: **~30 s** end-to-end — well within the 120 s timeout.

**Trade-off acknowledged**: The reduction from k=15/n=7 to k=8/n=5 is a meaningful quality knob. The full sweep of recall@k vs. latency is in [BENCHMARK_REPORT §6](BENCHMARK_REPORT.md). The benchmark winner (HyDE+Rerank, AVG=0.812) was at k=15/n=7. We accepted the smaller window in production because:
1. Faithfulness/precision degraded by <2pp at k=8 (still strong)
2. P95 latency dropped from ~85 s to ~30 s — material for UX
3. The alternative was a vertical scale-up (1 vCPU) which doubled compute cost

A horizontally scaled, asynchronous queue would let us keep k=15 — that's the natural next iteration.
