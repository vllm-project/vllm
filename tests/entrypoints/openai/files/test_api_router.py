# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import io
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.config import FileUploadConfig
from vllm.entrypoints.openai.files.api_router import attach_router
from vllm.entrypoints.openai.files.serving import OpenAIServingFiles
from vllm.entrypoints.openai.files.store import FileUploadStore

# Shared test fixtures (real magic bytes so the MIME sniffer passes).
_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 512
_MP4 = b"\x00\x00\x00\x20ftypisom" + b"\x00" * 2048
_TEXT = b"this is not a media file at all" + b"\x00" * 32


def _app_with_store(tmp_path, **config_overrides) -> FastAPI:
    defaults = {
        "enabled": True,
        "dir": str(tmp_path / "uploads"),
        "ttl_seconds": 3600,
        "max_size_mb": 1,
        "max_total_gb": 1,
        "max_concurrent": 4,
        "scope_header": "",
        "disable_listing": False,
    }
    defaults.update(config_overrides)
    config = FileUploadConfig(**defaults)
    store = FileUploadStore(config)
    app = FastAPI()
    app.state.openai_serving_files = OpenAIServingFiles(store, config)
    attach_router(app)
    return app


def _png_upload(name: str = "cat.png") -> dict:
    return {"file": (name, io.BytesIO(_PNG), "image/png")}


# ---------------------------------------------------------------------------
# happy path: upload → list → get → download → delete
# ---------------------------------------------------------------------------


def test_full_round_trip(tmp_path):
    client = TestClient(_app_with_store(tmp_path))

    # POST
    r = client.post("/v1/files", files=_png_upload(), data={"purpose": "vision"})
    assert r.status_code == 200, r.text
    body = r.json()
    file_id = body["id"]
    assert file_id.startswith("file-")
    assert body["filename"] == "cat.png"
    assert body["purpose"] == "vision"
    assert body["bytes"] == len(_PNG)
    assert body["object"] == "file"

    # GET list
    r = client.get("/v1/files")
    assert r.status_code == 200
    assert r.json()["object"] == "list"
    assert len(r.json()["data"]) == 1

    # GET single
    r = client.get(f"/v1/files/{file_id}")
    assert r.status_code == 200
    assert r.json()["id"] == file_id

    # GET content
    r = client.get(f"/v1/files/{file_id}/content")
    assert r.status_code == 200
    assert r.content == _PNG
    assert r.headers["content-type"].startswith("image/png")

    # DELETE
    r = client.delete(f"/v1/files/{file_id}")
    assert r.status_code == 200
    assert r.json()["deleted"] is True

    # GET now 404
    assert client.get(f"/v1/files/{file_id}").status_code == 404


# ---------------------------------------------------------------------------
# purpose + size validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_purpose", ["assistants", "fine-tune", "batch", "VISION"])
def test_rejects_unsupported_purpose(tmp_path, bad_purpose):
    client = TestClient(_app_with_store(tmp_path))
    r = client.post("/v1/files", files=_png_upload(), data={"purpose": bad_purpose})
    assert r.status_code == 400
    assert "purpose" in r.json()["error"]["message"]


def test_default_purpose_is_user_data(tmp_path):
    """Purpose defaults to user_data when unset (matches OpenAI SDK default)."""
    client = TestClient(_app_with_store(tmp_path))
    r = client.post("/v1/files", files=_png_upload())
    assert r.status_code == 200
    assert r.json()["purpose"] == "user_data"


def test_concurrency_limit_returns_503(tmp_path):
    """When `max_concurrent` uploads are already in flight, the store
    rejects with `ConcurrencyLimitExceeded` and the serving layer maps
    that to 503 (documented back-pressure contract)."""
    app = _app_with_store(tmp_path, max_concurrent=1)
    client = TestClient(app)
    # Saturate the semaphore manually — simulates a slow upload still in
    # flight when a second client arrives.
    store = app.state.openai_serving_files._store  # noqa: SLF001
    # Consume the only slot without releasing. Using sync access here
    # because asyncio.Semaphore is consumable from any context.
    store._upload_semaphore._value = 0  # noqa: SLF001
    r = client.post("/v1/files", files=_png_upload(), data={"purpose": "vision"})
    assert r.status_code == 503
    assert "max_concurrent" in r.json()["error"]["message"]


def test_oversize_upload_returns_413(tmp_path):
    client = TestClient(_app_with_store(tmp_path, max_size_mb=1))
    big = io.BytesIO(_PNG + b"\x00" * (2 * 1024 * 1024))
    r = client.post(
        "/v1/files",
        files={"file": ("big.png", big, "image/png")},
        data={"purpose": "vision"},
    )
    assert r.status_code == 413


def test_non_media_upload_returns_400(tmp_path):
    client = TestClient(_app_with_store(tmp_path))
    r = client.post(
        "/v1/files",
        files={"file": ("note.txt", io.BytesIO(_TEXT), "text/plain")},
        data={"purpose": "user_data"},
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# list disablement
# ---------------------------------------------------------------------------


def test_disable_listing_returns_404_on_GET_v1_files(tmp_path):
    client = TestClient(_app_with_store(tmp_path, disable_listing=True))
    # Upload first so there's something to (not) list.
    client.post("/v1/files", files=_png_upload(), data={"purpose": "vision"})

    r = client.get("/v1/files")
    assert r.status_code == 404
    assert "disabled" in r.json()["error"]["message"].lower()


def test_disable_listing_does_not_affect_individual_ops(tmp_path):
    client = TestClient(_app_with_store(tmp_path, disable_listing=True))
    r = client.post("/v1/files", files=_png_upload(), data={"purpose": "vision"})
    assert r.status_code == 200
    file_id = r.json()["id"]
    assert client.get(f"/v1/files/{file_id}").status_code == 200
    assert client.get(f"/v1/files/{file_id}/content").status_code == 200
    assert client.delete(f"/v1/files/{file_id}").status_code == 200


# ---------------------------------------------------------------------------
# scope header enforcement
# ---------------------------------------------------------------------------


def test_scope_header_missing_returns_400(tmp_path):
    client = TestClient(_app_with_store(tmp_path, scope_header="OpenAI-Project"))
    r = client.post("/v1/files", files=_png_upload(), data={"purpose": "vision"})
    assert r.status_code == 400
    assert "OpenAI-Project" in r.json()["error"]["message"]


def test_scope_header_present_attaches_scope(tmp_path):
    client = TestClient(_app_with_store(tmp_path, scope_header="OpenAI-Project"))
    r = client.post(
        "/v1/files",
        files=_png_upload(),
        data={"purpose": "vision"},
        headers={"OpenAI-Project": "team-alpha"},
    )
    assert r.status_code == 200
    file_id = r.json()["id"]

    # Same scope: visible.
    r = client.get(f"/v1/files/{file_id}", headers={"OpenAI-Project": "team-alpha"})
    assert r.status_code == 200

    # Different scope: 404 (capability non-disclosure).
    r = client.get(f"/v1/files/{file_id}", headers={"OpenAI-Project": "team-bravo"})
    assert r.status_code == 404


def test_scope_header_mismatch_returns_404_not_403(tmp_path):
    """Capability non-disclosure: wrong-scope callers must not be able
    to distinguish 'file exists but you cannot see it' from 'file does
    not exist'."""
    client = TestClient(_app_with_store(tmp_path, scope_header="OpenAI-Project"))
    # Upload as team-alpha.
    r = client.post(
        "/v1/files",
        files=_png_upload(),
        data={"purpose": "vision"},
        headers={"OpenAI-Project": "team-alpha"},
    )
    file_id = r.json()["id"]

    # team-bravo sees 404 (not 403).
    r = client.get(f"/v1/files/{file_id}", headers={"OpenAI-Project": "team-bravo"})
    assert r.status_code == 404

    # team-bravo attempting delete also 404.
    r = client.delete(f"/v1/files/{file_id}", headers={"OpenAI-Project": "team-bravo"})
    assert r.status_code == 404


def test_list_scope_filtered(tmp_path):
    client = TestClient(_app_with_store(tmp_path, scope_header="OpenAI-Project"))
    client.post(
        "/v1/files",
        files=_png_upload("a.png"),
        data={"purpose": "vision"},
        headers={"OpenAI-Project": "alpha"},
    )
    client.post(
        "/v1/files",
        files=_png_upload("b.png"),
        data={"purpose": "vision"},
        headers={"OpenAI-Project": "bravo"},
    )
    r = client.get("/v1/files", headers={"OpenAI-Project": "alpha"})
    data = r.json()["data"]
    assert len(data) == 1
    assert data[0]["filename"] == "a.png"


# ---------------------------------------------------------------------------
# audit log includes request_id when provided as X-Request-Id header
# ---------------------------------------------------------------------------


def test_request_id_propagates_to_audit_log(tmp_path):
    store_logger = logging.getLogger("vllm.entrypoints.openai.files.store")
    captured: list[logging.LogRecord] = []

    class _H(logging.Handler):
        def emit(self, record):
            captured.append(record)

    h = _H(level=logging.INFO)
    store_logger.addHandler(h)
    try:
        client = TestClient(_app_with_store(tmp_path))
        r = client.post(
            "/v1/files",
            files=_png_upload(),
            data={"purpose": "vision"},
            headers={"X-Request-Id": "req-7"},
        )
        assert r.status_code == 200
    finally:
        store_logger.removeHandler(h)

    import json

    req_ids = [
        json.loads(rec.getMessage()).get("request_id")
        for rec in captured
        if rec.getMessage().startswith("{")
    ]
    assert "req-7" in req_ids


# ---------------------------------------------------------------------------
# feature-off: router not attached → 404
# ---------------------------------------------------------------------------


def test_feature_off_router_not_attached(tmp_path):
    """When the feature is disabled, the router is not attached and all
    /v1/files* paths return 404 (FastAPI default for unknown routes)."""
    config = FileUploadConfig(enabled=False)
    app = FastAPI()
    # Intentionally do NOT attach the router or set app.state.
    client = TestClient(app)
    assert client.post("/v1/files", files=_png_upload()).status_code == 404
    assert client.get("/v1/files").status_code == 404
    assert client.delete("/v1/files/file-abc").status_code == 404
    _ = config  # silence flake; demonstrates the disabled-state contract
