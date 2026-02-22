# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Thin client wrapper for Responses context checkpoint tools.

This is intended for agentic loops that need two explicit operations:
- drop_checkpoint(label)
- revert_and_summarize(label, summary)

An optional helper is also provided for explicit checkpoint cleanup:
- delete_checkpoint(label)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx


class ResponsesCheckpointTools:
    def __init__(
        self,
        base_url: str,
        session_id: str,
        *,
        api_key: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id
        self.timeout_s = timeout_s
        self.api_key = api_key

    @property
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout_s) as client:
            response = client.post(
                f"{self.base_url}{path}",
                headers=self._headers,
                json=dict(payload),
            )
        response.raise_for_status()
        return response.json()

    def drop_checkpoint(self, label: str) -> dict[str, Any]:
        return self._post(
            "/v1/responses/context/checkpoints",
            {
                "session_id": self.session_id,
                "checkpoint_label": label,
            },
        )

    def revert_and_summarize(
        self,
        label: str,
        summary: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "session_id": self.session_id,
            "checkpoint_label": label,
        }
        if summary is not None:
            payload["summary"] = summary
        return self._post("/v1/responses/context/revert", payload)

    def delete_checkpoint(self, label: str) -> dict[str, Any]:
        return self._post(
            "/v1/responses/context/checkpoints/delete",
            {
                "session_id": self.session_id,
                "checkpoint_label": label,
            },
        )
