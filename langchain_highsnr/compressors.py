from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.documents import BaseDocumentCompressor, Document

from langchain_highsnr._client import HighSNRClient

_log = logging.getLogger(__name__)


class HighSNRDocumentCompressor(BaseDocumentCompressor):
    """Post-retrieval document compressor backed by the HighSNR API.

    Compresses retrieved chunks before they are passed to the LLM, using the
    user's query as a ``context_hint`` for query-aware selection.

    By default (``group_by_source=True``) chunks are grouped by
    ``metadata["source"]`` and one API call is made per source document.
    This is the benchmark-validated usage pattern — HighSNR is benchmarked
    on single documents and performs best when all chunks in a call originate
    from the same source.

    Set ``group_by_source=False`` to send all retrieved chunks in a single
    call. This is **not benchmarked** and a warning will be logged.

    Example::

        from langchain_highsnr import HighSNRDocumentCompressor
        from langchain.retrievers import ContextualCompressionRetriever

        compressor = HighSNRDocumentCompressor(
            api_key="snr-...",
            max_output_tokens=2000,
        )
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=your_retriever,
        )
    """

    api_key: Optional[str] = None
    max_output_tokens: int = 2000
    include_boundaries: bool = False
    group_by_source: bool = True
    base_url: Optional[str] = None
    timeout_s: int = 60

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Any] = None,
    ) -> Sequence[Document]:
        if not documents:
            return []
        client = HighSNRClient(
            api_key=self.api_key, base_url=self.base_url, timeout_s=self.timeout_s
        )

        if not self.group_by_source:
            _log.warning(
                "group_by_source=False: all chunks are sent in a single HighSNR "
                "call regardless of source document. This is not benchmarked and "
                "results may vary. Set group_by_source=True to follow the "
                "benchmark-validated usage pattern."
            )
            return self._compress_group(client, list(documents), query)

        groups: Dict[str, List[Document]] = defaultdict(list)
        sourceless: List[Document] = []
        for doc in documents:
            src = doc.metadata.get("source")
            if src:
                groups[str(src)].append(doc)
            else:
                sourceless.append(doc)

        if sourceless:
            _log.warning(
                "%d chunk(s) have no 'source' metadata key and cannot be grouped "
                "by document. They will be compressed together in a single call, "
                "which is not benchmarked.",
                len(sourceless),
            )

        result: List[Document] = []
        for group in groups.values():
            result.extend(self._compress_group(client, group, query))
        if sourceless:
            result.extend(self._compress_group(client, sourceless, query))
        return result

    def _compress_group(
        self,
        client: HighSNRClient,
        documents: List[Document],
        query: str,
    ) -> List[Document]:
        response = client.optimize(
            chunks=[d.page_content for d in documents],
            max_output_tokens=self.max_output_tokens,
            include_boundaries=self.include_boundaries,
            context_hint=query or None,
            return_indices=True,
        )
        indices = response.get("selected_chunk_indices") or []
        if indices:
            return [documents[i] for i in indices]
        # Fallback: server returned chunks without indices
        kept = response.get("selected_chunks", [])
        metadata = documents[0].metadata if documents else {}
        return [Document(page_content=c, metadata=metadata) for c in kept]
