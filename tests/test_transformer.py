from __future__ import annotations

import logging
from typing import Any, Dict, List

import pytest
from langchain_core.documents import Document

import langchain_highsnr._client as client_mod
from langchain_highsnr.transformers import HighSNRDocumentTransformer


class _FakeResponse:
    def __init__(self, json_data: Dict[str, Any], status_code: int = 200) -> None:
        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> Dict[str, Any]:
        return self._json_data


def _transformer(**kwargs: Any) -> HighSNRDocumentTransformer:
    defaults: Dict[str, Any] = {
        "api_key": "test-key",
        "base_url": "http://test.example",
        "max_output_tokens": 500,
    }
    defaults.update(kwargs)
    return HighSNRDocumentTransformer(**defaults)


def test_one_api_call_per_document(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[Dict[str, Any]] = []

    def fake_post(url: str, json: Dict[str, Any], **kwargs: Any) -> _FakeResponse:
        calls.append(json)
        return _FakeResponse({"selected_chunks": ["compressed"]})

    monkeypatch.setattr(client_mod.requests, "post", fake_post)

    docs = [
        Document(page_content="doc one", metadata={"source": "a.pdf"}),
        Document(page_content="doc two", metadata={"source": "b.pdf"}),
    ]
    _transformer().transform_documents(docs)

    assert len(calls) == 2
    assert calls[0]["document"] == "doc one"
    assert calls[1]["document"] == "doc two"


def test_passes_context_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_post(url: str, json: Dict[str, Any], **kwargs: Any) -> _FakeResponse:
        captured.update(json)
        return _FakeResponse({"selected_chunks": ["result"]})

    monkeypatch.setattr(client_mod.requests, "post", fake_post)

    _transformer(context_hint="what is the main finding?").transform_documents(
        [Document(page_content="some text")]
    )

    assert captured["context_hint"] == "what is the main finding?"


def test_preserves_source_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, json: Dict[str, Any], **kwargs: Any) -> _FakeResponse:
        return _FakeResponse({"selected_chunks": ["chunk A", "chunk B"]})

    monkeypatch.setattr(client_mod.requests, "post", fake_post)

    docs = [Document(page_content="text", metadata={"source": "paper.pdf", "page": 3})]
    result = _transformer().transform_documents(docs)

    assert len(result) == 2
    assert all(r.metadata == {"source": "paper.pdf", "page": 3} for r in result)


def test_include_boundaries_defaults_true(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_post(url: str, json: Dict[str, Any], **kwargs: Any) -> _FakeResponse:
        captured.update(json)
        return _FakeResponse({"selected_chunks": ["r"]})

    monkeypatch.setattr(client_mod.requests, "post", fake_post)

    _transformer().transform_documents([Document(page_content="text")])

    assert captured["include_boundaries"] is True


def test_empty_input_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_mod.requests, "post", lambda *a, **kw: None)

    result = _transformer().transform_documents([])

    assert list(result) == []


def test_http_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        client_mod.requests,
        "post",
        lambda *a, **kw: _FakeResponse({}, status_code=401),
    )

    with pytest.raises(RuntimeError, match="HTTP 401"):
        _transformer().transform_documents([Document(page_content="text")])


def test_api_warnings_are_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def fake_post(url: str, json: Dict[str, Any], **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(
            {"selected_chunks": ["compressed"], "warnings": ["budget exceeded"]}
        )

    monkeypatch.setattr(client_mod.requests, "post", fake_post)

    with caplog.at_level(logging.WARNING, logger="langchain_highsnr.transformers"):
        _transformer().transform_documents([Document(page_content="text")])

    assert "budget exceeded" in caplog.text


def test_output_metadata_is_not_aliased(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutating one output doc's metadata must not affect sibling docs."""

    def fake_post(url: str, json: Dict[str, Any], **kwargs: Any) -> _FakeResponse:
        return _FakeResponse({"selected_chunks": ["chunk A", "chunk B"]})

    monkeypatch.setattr(client_mod.requests, "post", fake_post)

    docs = [Document(page_content="text", metadata={"source": "paper.pdf", "page": 1})]
    result = list(_transformer().transform_documents(docs))

    assert len(result) == 2
    result[0].metadata["page"] = 99
    assert result[1].metadata["page"] == 1
