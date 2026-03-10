"""End-to-end demo of langchain-highsnr.

Tests all four combinations:
  1. Transformer + document + with hint
  2. Transformer + document + no hint
  3. Compressor + chunks + with hint (query)
  4. Compressor + chunks + no hint (empty query)

Requires HIGHSNR_API_KEY to be set in the environment.

Usage:
    export HIGHSNR_API_KEY=snr-...
    python examples/demo.py
"""

from langchain_core.documents import Document

from langchain_highsnr import HighSNRDocumentCompressor, HighSNRDocumentTransformer

SAMPLE_DOCUMENT = """\
Machine learning has transformed natural language processing over the past decade.
Early approaches relied on hand-crafted features and statistical models such as
TF-IDF weighted bag-of-words combined with SVMs or logistic regression. These
methods worked reasonably well for document classification but struggled with tasks
requiring deeper understanding of language structure and semantics.

The introduction of word embeddings like Word2Vec and GloVe marked a significant
shift, enabling models to capture semantic relationships between words in dense
vector spaces. Transfer learning further accelerated progress when pre-trained
language models such as ELMo and later BERT demonstrated that representations
learned on large corpora could be fine-tuned for downstream tasks with relatively
little labelled data.

Transformer architectures, introduced in the "Attention Is All You Need" paper,
became the dominant paradigm. Self-attention mechanisms allow these models to weigh
the relevance of each token in a sequence relative to every other token, enabling
effective capture of long-range dependencies. GPT, BERT, and their successors
pushed state-of-the-art results across benchmarks in question answering, sentiment
analysis, named entity recognition, and machine translation.

Recent developments have focused on scaling these models to hundreds of billions
of parameters, leading to emergent capabilities such as few-shot and zero-shot
learning. Retrieval-augmented generation (RAG) combines the generative power of
large language models with external knowledge retrieval, reducing hallucination
and improving factual accuracy. Context window optimization techniques help manage
the cost and latency of processing long documents by selecting the most informative
passages before sending them to the LLM.
"""

SAMPLE_CHUNKS = [
    Document(
        page_content=(
            "Early approaches relied on hand-crafted features and statistical "
            "models such as TF-IDF weighted bag-of-words combined with SVMs or "
            "logistic regression."
        ),
        metadata={"source": "demo.txt", "page": 1},
    ),
    Document(
        page_content=(
            "Transformer architectures became the dominant paradigm. "
            "Self-attention mechanisms allow these models to weigh the relevance "
            "of each token in a sequence relative to every other token."
        ),
        metadata={"source": "demo.txt", "page": 2},
    ),
    Document(
        page_content=(
            "Retrieval-augmented generation (RAG) combines the generative power "
            "of large language models with external knowledge retrieval, reducing "
            "hallucination and improving factual accuracy."
        ),
        metadata={"source": "demo.txt", "page": 3},
    ),
    Document(
        page_content=(
            "Context window optimization techniques help manage the cost and "
            "latency of processing long documents by selecting the most "
            "informative passages before sending them to the LLM."
        ),
        metadata={"source": "demo.txt", "page": 4},
    ),
]


def print_results(result: list[Document], truncate: int = 200) -> None:
    for i, doc in enumerate(result):
        text = doc.page_content
        display = text[:truncate] + ("..." if len(text) > truncate else "")
        print(f"  Chunk {i + 1} (metadata: {doc.metadata}): {display}")


def demo_transformer_with_hint() -> None:
    print("=" * 60)
    print("1. TRANSFORMER — document + with hint")
    print("=" * 60)
    transformer = HighSNRDocumentTransformer(
        max_output_tokens=200,
        context_hint="What is retrieval-augmented generation?",
    )
    docs = [Document(page_content=SAMPLE_DOCUMENT, metadata={"source": "demo.txt"})]
    result = transformer.transform_documents(docs)
    print(f"Input: 1 document ({len(SAMPLE_DOCUMENT)} chars), hint set")
    print(f"Output: {len(result)} chunk(s)")
    print_results(list(result))
    print()


def demo_transformer_no_hint() -> None:
    print("=" * 60)
    print("2. TRANSFORMER — document + no hint")
    print("=" * 60)
    transformer = HighSNRDocumentTransformer(max_output_tokens=200)
    docs = [Document(page_content=SAMPLE_DOCUMENT, metadata={"source": "demo.txt"})]
    result = transformer.transform_documents(docs)
    print(f"Input: 1 document ({len(SAMPLE_DOCUMENT)} chars), no hint")
    print(f"Output: {len(result)} chunk(s)")
    print_results(list(result))
    print()


def demo_compressor_with_hint() -> None:
    print("=" * 60)
    print("3. COMPRESSOR — chunks + with hint (query)")
    print("=" * 60)
    compressor = HighSNRDocumentCompressor(max_output_tokens=200)
    query = "How does RAG reduce hallucination?"
    result = compressor.compress_documents(SAMPLE_CHUNKS, query)
    print(f"Input: {len(SAMPLE_CHUNKS)} chunks, query: {query!r}")
    print(f"Output: {len(result)} chunk(s)")
    print_results(list(result))
    print()


def demo_compressor_no_hint() -> None:
    print("=" * 60)
    print("4. COMPRESSOR — chunks + no hint (empty query)")
    print("=" * 60)
    compressor = HighSNRDocumentCompressor(max_output_tokens=200)
    result = compressor.compress_documents(SAMPLE_CHUNKS, "")
    print(f"Input: {len(SAMPLE_CHUNKS)} chunks, no query")
    print(f"Output: {len(result)} chunk(s)")
    print_results(list(result))
    print()


if __name__ == "__main__":
    demo_transformer_with_hint()
    demo_transformer_no_hint()
    demo_compressor_with_hint()
    demo_compressor_no_hint()
    print("All 4 demos completed successfully.")
