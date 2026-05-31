from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, cast

import requests


class HighSNRClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_s: int = 60,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("HIGHSNR_API_URL") or "https://api.high-snr.com"
        ).rstrip("/")
        resolved_key = api_key or os.environ.get("HIGHSNR_API_KEY")
        if not resolved_key:
            raise ValueError(
                "api_key is required. Pass it as a constructor argument or set "
                "the HIGHSNR_API_KEY environment variable."
            )
        self._api_key = resolved_key
        self.timeout_s = timeout_s

    def optimize(
        self,
        document: Optional[str] = None,
        chunks: Optional[List[str]] = None,
        max_output_tokens: int = 1000,
        include_boundaries: bool = True,
        context_hint: Optional[str] = None,
        return_metadata: bool = False,
        return_indices: bool = False,
    ) -> Dict[str, Any]:
        if document is None and chunks is None:
            raise ValueError("Either document or chunks must be provided")
        if document is not None and chunks is not None:
            raise ValueError("Only one of document or chunks may be provided")
        payload: Dict[str, Any] = {
            "max_output_tokens": max_output_tokens,
            "include_boundaries": include_boundaries,
            "return_metadata": return_metadata,
            "return_indices": return_indices,
        }
        if document is not None:
            payload["document"] = document
        if chunks is not None:
            payload["chunks"] = chunks
        if context_hint:
            payload["context_hint"] = context_hint
        r = requests.post(
            f"{self.base_url}/v2/optimize",
            json=payload,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        return cast(Dict[str, Any], r.json())
