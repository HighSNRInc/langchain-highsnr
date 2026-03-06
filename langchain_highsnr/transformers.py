from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

from langchain_core.documents import BaseDocumentTransformer, Document

from langchain_highsnr._client import HighSNRClient

_log = logging.getLogger(__name__)


class HighSNRDocumentTransformer(BaseDocumentTransformer):
    """Pre-indexing document compressor backed by the HighSNR API.

    Compresses each document independently before embedding, reducing index
    size while preserving the highest-signal content. One API call per
    document — this is the benchmark-validated usage pattern.

    ``context_hint`` can be set at construction time when a topic or query is
    known ahead of indexing. Without a hint, unsupervised topical compression
    is applied — both configurations are benchmark-validated.

    Example::

        from langchain_highsnr import HighSNRDocumentTransformer

        transformer = HighSNRDocumentTransformer(
            api_key="snr-...",
            max_output_tokens=800,
        )
        compressed = transformer.transform_documents(raw_docs)
        vectorstore = FAISS.from_documents(compressed, embeddings)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_output_tokens: int = 1000,
        include_boundaries: bool = True,
        context_hint: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_s: int = 60,
    ) -> None:
        self._client = HighSNRClient(
            api_key=api_key, base_url=base_url, timeout_s=timeout_s
        )
        self.max_output_tokens = max_output_tokens
        self.include_boundaries = include_boundaries
        self.context_hint = context_hint

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Compress each document independently. Returns one Document per output chunk."""
        result: List[Document] = []
        for doc in documents:
            response = self._client.optimize(
                document=doc.page_content,
                max_output_tokens=self.max_output_tokens,
                include_boundaries=self.include_boundaries,
                context_hint=self.context_hint,
            )
            for chunk in response.get("selected_chunks", []):
                result.append(Document(page_content=chunk, metadata=dict(doc.metadata)))
        return result
