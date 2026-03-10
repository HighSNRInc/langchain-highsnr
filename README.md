# langchain-highsnr

LangChain integration for [HighSNR](https://www.high-snr.com/#api) — compress documents to a token budget, keeping the highest-signal content.

```bash
pip install langchain-highsnr
```

Get an API key at [console.high-snr.com](https://console.high-snr.com). Requires Python 3.9+ and `langchain-core>=0.3.0`.

---

## What it does

HighSNR selects the most informative chunks from a document and discards the rest,
staying within a token budget. Compression is deterministic, privacy-first, and
sub-second for most documents.

| Class | Position in pipeline | Use case |
|---|---|---|
| `HighSNRDocumentTransformer` | Before embedding | Compress raw docs before indexing |
| `HighSNRDocumentCompressor` | After retrieval | Compress retrieved chunks before LLM |

---

## Usage

### HighSNRDocumentTransformer — compress before indexing

```python
from langchain_highsnr import HighSNRDocumentTransformer
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

transformer = HighSNRDocumentTransformer(
    api_key="snr-...",       # or set HIGHSNR_API_KEY env var
    max_output_tokens=800,   # token budget per document
    context_hint="clinical trial methodology",  # optional topic hint
)

compressed = transformer.transform_documents(raw_docs)
vectorstore = FAISS.from_documents(compressed, OpenAIEmbeddings())
```

| Parameter | Default | Description |
|---|---|---|
| `api_key` | `None` | Falls back to `HIGHSNR_API_KEY` env var |
| `max_output_tokens` | `1000` | Token budget per document |
| `include_boundaries` | `True` | Keep first and last chunk |
| `context_hint` | `None` | Topic/query to bias chunk selection |

---

### HighSNRDocumentCompressor — compress after retrieval

The user's query is automatically used as the selection hint.

```python
from langchain_highsnr import HighSNRDocumentCompressor
from langchain.retrievers import ContextualCompressionRetriever

compressor = HighSNRDocumentCompressor(
    api_key="snr-...",
    max_output_tokens=2000,
)

retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 20}),
)

docs = retriever.invoke("what is the main finding?")
```

| Parameter | Default | Description |
|---|---|---|
| `api_key` | `None` | Falls back to `HIGHSNR_API_KEY` env var |
| `max_output_tokens` | `2000` | Token budget across all chunks |
| `include_boundaries` | `False` | Keep first/last chunk |
| `group_by_source` | `True` | Group chunks by `metadata["source"]` — one API call per source document ([benchmark-validated](https://github.com/HighSNRInc/highsnr-benchmarks)) |

---

## Benchmarks

Evaluated on LongBench v1 with GPT-4o (n=200 per dataset). At 80% budget with hint:

- **HotpotQA**: F1 **70.96** — exceeds full-context GPT-4o (69.71)
- **Qasper**: F1 **45.21** — 96% of full-context GPT-4o (47.22)

Full results, scripts, and reproduction instructions:
[github.com/HighSNRInc/highsnr-benchmarks](https://github.com/HighSNRInc/highsnr-benchmarks)

---

## Environment variables

| Variable | Description |
|---|---|
| `HIGHSNR_API_KEY` | API key — alternative to passing `api_key` in the constructor |
| `HIGHSNR_API_URL` | Override the API base URL (default: `https://api.high-snr.com`) |

---

## Links

- API console & free tier: [console.high-snr.com](https://console.high-snr.com)
- Homepage: [high-snr.com](https://www.high-snr.com/#api)
- Benchmarks: [github.com/HighSNRInc/highsnr-benchmarks](https://github.com/HighSNRInc/highsnr-benchmarks)
- Support: [hello@high-snr.com](mailto:hello@high-snr.com)
