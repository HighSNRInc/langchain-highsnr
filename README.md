# langchain-highsnr

LangChain integration for [HighSNR](https://high-snr.com) — compress documents to a token budget, keeping the highest-signal content.

```bash
pip install langchain-highsnr
```

Get an API key at [console.high-snr.com](https://console.high-snr.com).

---

## What it does

HighSNR selects the most informative chunks from a document and discards the rest,
staying within a token budget. No AI is involved — compression is deterministic,
privacy-first, and sub-second for most documents.

Two integration points for LangChain pipelines:

| Class | Position in pipeline | Use case |
|---|---|---|
| `HighSNRDocumentTransformer` | Before embedding | Compress raw docs before indexing |
| `HighSNRDocumentCompressor` | After retrieval | Compress retrieved chunks before LLM |

---

## Installation

```bash
pip install langchain-highsnr
```

Requires Python 3.9+ and `langchain-core>=0.3.0`.

---

## Usage

### HighSNRDocumentTransformer — compress before indexing

Compresses each document before embedding. Produces a leaner, higher-quality vector index.

```python
from langchain_highsnr import HighSNRDocumentTransformer
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

transformer = HighSNRDocumentTransformer(
    api_key="snr-...",       # or set HIGHSNR_API_KEY env var
    max_output_tokens=800,   # token budget per document
)

# raw_docs: list[Document] loaded from any LangChain loader
compressed = transformer.transform_documents(raw_docs)
vectorstore = FAISS.from_documents(compressed, OpenAIEmbeddings())
```

With a topic hint (improves selection when the domain is known ahead of indexing):

```python
transformer = HighSNRDocumentTransformer(
    api_key="snr-...",
    max_output_tokens=800,
    context_hint="clinical trial methodology",
)
```

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `api_key` | `None` | API key (falls back to `HIGHSNR_API_KEY` env var) |
| `max_output_tokens` | `1000` | Token budget per document |
| `include_boundaries` | `True` | Always keep first and last chunk (recommended for summarization) |
| `context_hint` | `None` | Optional topic/query to bias chunk selection |

---

### HighSNRDocumentCompressor — compress after retrieval

Compresses retrieved chunks before they reach the LLM. The user's query is
automatically used as the selection hint — the strongest configuration.

```python
from langchain_highsnr import HighSNRDocumentCompressor
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores import FAISS

compressor = HighSNRDocumentCompressor(
    api_key="snr-...",        # or set HIGHSNR_API_KEY env var
    max_output_tokens=2000,   # token budget across all retrieved chunks
)

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)

docs = retriever.invoke("what is the main finding?")
# docs are compressed to 2000 tokens, query-aware
```

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `api_key` | `None` | API key (falls back to `HIGHSNR_API_KEY` env var) |
| `max_output_tokens` | `2000` | Token budget across all chunks in one call |
| `include_boundaries` | `False` | Keep first/last chunk — off by default for retrieval |
| `group_by_source` | `True` | Group chunks by `metadata["source"]` (see below) |

---

## Benchmark-aligned usage

HighSNR is benchmarked on **single raw documents** (LongBench v1). The benchmark-validated
patterns are:

- **One raw document per API call** — `HighSNRDocumentTransformer` does this automatically.
- **Chunks from the same source document per call** — `HighSNRDocumentCompressor` with
  `group_by_source=True` (default) groups retrieved chunks by `metadata["source"]` and
  fires one API call per source document.

Both configurations were evaluated, with and without `context_hint`. The hint
consistently improves results.

### Non-grouped usage (not benchmarked)

`HighSNRDocumentCompressor` with `group_by_source=False` sends all retrieved chunks
in a single call regardless of source document. This may work well in practice but
is **outside the benchmarked envelope** — a warning is logged when this mode is active.

```python
# Works, but not benchmarked — warning will be logged
compressor = HighSNRDocumentCompressor(
    api_key="snr-...",
    max_output_tokens=2000,
    group_by_source=False,
)
```

---

## Benchmark results

Evaluated on [LongBench v1](https://github.com/THUDM/LongBench) with GPT-4o, n=200 per dataset.
HighSNR compresses each document to the target budget; GPT-4o answers the question from the
compressed output. QA F1 score — higher is better.

### HotpotQA (multi-hop QA)

| Budget | F1 score | Actual tokens kept |
|---|---|---|
| 50% | 65.29 | 55.9% (median 55.4%) |
| 60% | 66.34 | 67.9% (median 67.3%) |
| 70% | 68.08 | 79.8% (median 79.1%) |
| **80%** | **70.70** | 91.4% (median 90.8%) |
| Full doc (baseline) | 69.71 | 100% |

At 80% budget, HighSNR **beats full-context F1** (70.70 vs 69.71).

### Qasper (single-hop QA on scientific papers)

| Budget | F1 score | Actual tokens kept |
|---|---|---|
| 50% | 35.51 | 54.7% (median 54.4%) |
| 60% | 38.16 | 66.4% (median 66.2%) |
| 70% | 41.36 | 78.0% (median 77.6%) |
| **80%** | **45.37** | 89.9% (median 89.5%) |
| Full doc (baseline) | 47.22 | 100% |

At 80% budget, HighSNR retains **96% of full-context F1** on scientific QA.

> Actual token ratios exceed the target because HighSNR never cuts a chunk mid-sentence.
> Chunks are selected whole — if the next chunk would exceed the budget it is skipped,
> so the output lands just below the target.

### Latency

Measured on live API (0.5 vCPU / 1 GB Fargate), n=3,200 calls.

| Document size | Median | Mean |
|---|---|---|
| < 5k tokens | 770 ms | 777 ms |
| 5k – 10k tokens | 1,102 ms | 1,142 ms |
| 10k – 20k tokens | 1,792 ms | 1,833 ms |

---

## Environment variables

| Variable | Description |
|---|---|
| `HIGHSNR_API_KEY` | API key — alternative to passing `api_key` in the constructor |
| `HIGHSNR_API_URL` | Override the API base URL (default: `https://api.high-snr.com`) |

---

## Links

- API console & free tier: [console.high-snr.com](https://console.high-snr.com)
- Homepage: [high-snr.com](https://high-snr.com)
- Support: [hello@high-snr.com](mailto:hello@high-snr.com)
