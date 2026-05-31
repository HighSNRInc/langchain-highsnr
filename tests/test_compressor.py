from __future__ import annotations

import logging
from typing import Any, Dict, List

import pytest
from langchain_core.documents import Document

import langchain_highsnr._client as client_mod
from langchain_highsnr.compressors import HighSNRDocumentCompressor


class _FakeResponse:
    def __init__(self, json_data: Dict[str, Any], status_code: int = 200) -> None:
        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> Dict[str, Any]:
        return self._json_data


def _compressor(**kwargs: Any) -> HighSNRDocumentCompressor:
    defaults: Dict[str, Any] = {
        "api_key": "test-key",
        "base_url": "http://test.example",
        "max_output_tokens": 500,
    }
    defaults.update(kwargs)
    return HighSNRDocumentCompressor(**defaults)


def test_groups_chunks_by_source_into_separate_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[Dict[str, Any]] = []

    def fake_post(url: str, json: Dict[str, Any], **kwargs: Any) -> _FakeResponse:
        calls.append(json)
        n = len(json.get("chunks", []))
        return _FakeResponse(
            {
                "optimized_chunks": json["chunks"],
                "selected_chunk_indices": list(range(n)),
            }
        )

    monkeypatch.setattr(client_mod.requests, "post", fake_post)

    docs = [
        Document(page_content="A1", metadata={"source": "doc1"}),
        Document(page_content="A2", metadata={"source": "doc1"}),
        Document(page_content="B1", metadata={"source": "doc2"}),
    ]
    result = _compressor().compress_documents(docs, query="q")

    assert len(calls) == 2
    assert len(result) == 3


def test_passes_query_as_context_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_post(url: str, json: Dict[str, Any], **kwargs: Any) -> _FakeResponse:
        captured.update(json)
        return _FakeResponse(
            {
                "optimized_chunks": ["r"],
                "selected_chunk_indices": [0],
            }
        )

    monkeypatch.setattr(client_mod.requests, "post", fake_post)

    docs = [Document(page_content="chunk A", metadata={"source": "doc1"})]
    _compressor().compress_documents(docs, query="what is X?")

    assert captured["context_hint"] == "what is X?"


def test_preserves_metadata_via_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, json: Dict[str, Any], **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(
            {
                "optimized_chunks": ["A1"],
                "selected_chunk_indices": [0],
            }
        )

    monkeypatch.setattr(client_mod.requests, "post", fake_post)

    docs = [
        Document(page_content="A1", metadata={"source": "doc1", "page": 5}),
        Document(page_content="A2", metadata={"source": "doc1", "page": 6}),
    ]
    result = _compressor().compress_documents(docs, query="q")

    assert len(result) == 1
    assert result[0].metadata == {"source": "doc1", "page": 5}


def test_include_boundaries_defaults_false(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_post(url: str, json: Dict[str, Any], **kwargs: Any) -> _FakeResponse:
        captured.update(json)
        return _FakeResponse(
            {
                "optimized_chunks": ["r"],
                "selected_chunk_indices": [0],
            }
        )

    monkeypatch.setattr(client_mod.requests, "post", fake_post)

    docs = [Document(page_content="chunk", metadata={"source": "doc1"})]
    _compressor().compress_documents(docs, query="q")

    assert captured["include_boundaries"] is False


def test_warns_when_no_source_metadata(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def fake_post(url: str, json: Dict[str, Any], **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(
            {
                "optimized_chunks": ["r"],
                "selected_chunk_indices": [0],
            }
        )

    monkeypatch.setattr(client_mod.requests, "post", fake_post)

    docs = [Document(page_content="chunk without source")]
    with caplog.at_level(logging.WARNING, logger="langchain_highsnr.compressors"):
        _compressor().compress_documents(docs, query="q")

    assert "missing or null 'source'" in caplog.text


def test_warns_when_group_by_source_false(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def fake_post(url: str, json: Dict[str, Any], **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(
            {
                "optimized_chunks": ["r"],
                "selected_chunk_indices": [0],
            }
        )

    monkeypatch.setattr(client_mod.requests, "post", fake_post)

    docs = [Document(page_content="chunk", metadata={"source": "doc1"})]
    with caplog.at_level(logging.WARNING, logger="langchain_highsnr.compressors"):
        _compressor(group_by_source=False).compress_documents(docs, query="q")

    assert "not benchmarked" in caplog.text


def test_empty_input_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_mod.requests, "post", lambda *a, **kw: None)

    result = _compressor().compress_documents([], query="q")

    assert list(result) == []


def test_invalid_indices_out_of_range_are_filtered(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def fake_post(url: str, json: Dict[str, Any], **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(
            {
                "optimized_chunks": ["A", "B"],
                "selected_chunk_indices": [0, 99],
            }
        )

    monkeypatch.setattr(client_mod.requests, "post", fake_post)

    docs = [Document(page_content="A", metadata={"source": "doc1"})]
    with caplog.at_level(logging.WARNING, logger="langchain_highsnr.compressors"):
        result = _compressor().compress_documents(docs, query="q")

    assert len(result) == 1
    assert result[0].page_content == "A"
    assert "invalid" in caplog.text


def test_negative_indices_are_filtered(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def fake_post(url: str, json: Dict[str, Any], **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(
            {
                "optimized_chunks": ["A", "B"],
                "selected_chunk_indices": [0, -1],
            }
        )

    monkeypatch.setattr(client_mod.requests, "post", fake_post)

    docs = [
        Document(page_content="A", metadata={"source": "doc1"}),
        Document(page_content="B", metadata={"source": "doc1"}),
    ]
    with caplog.at_level(logging.WARNING, logger="langchain_highsnr.compressors"):
        result = _compressor().compress_documents(docs, query="q")

    assert len(result) == 1
    assert result[0].page_content == "A"
    assert "invalid" in caplog.text


def test_fallback_chunks_preserve_shared_metadata(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Fallback path: shared keys (source) are kept; divergent keys (page) are dropped."""

    def fake_post(url: str, json: Dict[str, Any], **kwargs: Any) -> _FakeResponse:
        return _FakeResponse({"optimized_chunks": ["kept chunk"]})

    monkeypatch.setattr(client_mod.requests, "post", fake_post)

    docs = [
        Document(page_content="A", metadata={"source": "doc1", "page": 1}),
        Document(page_content="B", metadata={"source": "doc1", "page": 2}),
    ]
    with caplog.at_level(logging.WARNING, logger="langchain_highsnr.compressors"):
        result = _compressor().compress_documents(docs, query="q")

    assert len(result) == 1
    assert result[0].metadata == {"source": "doc1"}  # page dropped, source kept
    assert "cannot be accurately attributed" in caplog.text


def test_fallback_metadata_is_not_aliased(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback path: mutating one output doc's metadata must not affect siblings."""

    def fake_post(url: str, json: Dict[str, Any], **kwargs: Any) -> _FakeResponse:
        return _FakeResponse({"optimized_chunks": ["chunk A", "chunk B"]})

    monkeypatch.setattr(client_mod.requests, "post", fake_post)

    docs = [Document(page_content="A", metadata={"source": "doc1"})]
    result = _compressor().compress_documents(docs, query="q")

    assert len(result) == 2
    result[0].metadata["injected"] = "x"
    assert "injected" not in result[1].metadata


def test_boolean_indices_are_rejected(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """True/False must not be accepted as valid chunk indices."""

    def fake_post(url: str, json: Dict[str, Any], **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(
            {
                "optimized_chunks": ["A", "B"],
                "selected_chunk_indices": [True, False],
            }
        )

    monkeypatch.setattr(client_mod.requests, "post", fake_post)

    docs = [
        Document(page_content="A", metadata={"source": "doc1"}),
        Document(page_content="B", metadata={"source": "doc1"}),
    ]
    with caplog.at_level(logging.WARNING, logger="langchain_highsnr.compressors"):
        result = _compressor().compress_documents(docs, query="q")

    # All boolean indices filtered → falls through to selected_chunks fallback
    assert all(doc.page_content in ("A", "B") for doc in result)
    assert "invalid" in caplog.text


def test_http_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        client_mod.requests,
        "post",
        lambda *a, **kw: _FakeResponse({}, status_code=500),
    )

    with pytest.raises(RuntimeError, match="HTTP 500"):
        _compressor().compress_documents(
            [Document(page_content="chunk", metadata={"source": "doc1"})], query="q"
        )


def test_sends_return_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_post(url: str, json: Dict[str, Any], **kwargs: Any) -> _FakeResponse:
        captured.update(json)
        return _FakeResponse(
            {
                "optimized_chunks": ["r"],
                "selected_chunk_indices": [0],
            }
        )

    monkeypatch.setattr(client_mod.requests, "post", fake_post)

    docs = [Document(page_content="chunk", metadata={"source": "doc1"})]
    _compressor().compress_documents(docs, query="q")

    assert captured["return_indices"] is True
    assert "return_chunk_metadata" not in captured


def test_api_warnings_are_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def fake_post(url: str, json: Dict[str, Any], **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(
            {
                "optimized_chunks": ["r"],
                "selected_chunk_indices": [0],
                "warnings": ["budget exceeded"],
            }
        )

    monkeypatch.setattr(client_mod.requests, "post", fake_post)

    docs = [Document(page_content="chunk", metadata={"source": "doc1"})]
    with caplog.at_level(logging.WARNING, logger="langchain_highsnr.compressors"):
        _compressor().compress_documents(docs, query="q")

    assert "budget exceeded" in caplog.text


def test_source_none_value_triggers_sourceless_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """source=None (key present, null value) should trigger the same warning as missing key."""

    def fake_post(url: str, json: Dict[str, Any], **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(
            {
                "optimized_chunks": ["r"],
                "selected_chunk_indices": [0],
            }
        )

    monkeypatch.setattr(client_mod.requests, "post", fake_post)

    docs = [Document(page_content="chunk", metadata={"source": None})]
    with caplog.at_level(logging.WARNING, logger="langchain_highsnr.compressors"):
        _compressor().compress_documents(docs, query="q")

    assert "missing or null 'source'" in caplog.text
