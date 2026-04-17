# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end: upload a file via the store, then resolve it via
MediaConnector's vllm-file:// scheme handler — the path a chat-completion
takes during multimodal resolution."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from vllm.config import FileUploadConfig
from vllm.entrypoints.openai.files.store import (
    FileUploadStore,
    get_store,
    register_store,
)
from vllm.multimodal.media import MediaConnector

# Minimal PNG-shaped payload.
_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 1024


async def _stream(data: bytes) -> AsyncIterator[bytes]:
    yield data


class _BytesMediaIO:
    """Minimal MediaIO stand-in so we don't have to spin up the real
    image/video decoders just to test URL dispatch."""

    def load_bytes(self, data: bytes) -> bytes:
        return data


@pytest.fixture
def _store(tmp_path):
    config = FileUploadConfig(
        enabled=True, dir=str(tmp_path / "uploads"), max_size_mb=1
    )
    store = FileUploadStore(config)
    register_store(store)
    yield store
    register_store(None)


@pytest.mark.asyncio
async def test_vllm_file_scheme_resolves_to_stored_bytes(_store):
    rec = await _store.create_file(
        stream=_stream(_PNG), filename="cat.png", purpose="vision", scope=None
    )
    connector = MediaConnector()
    result = connector.load_from_url(f"vllm-file://{rec.id}", _BytesMediaIO())
    assert result == _PNG


@pytest.mark.asyncio
async def test_vllm_file_scheme_ignores_scope_on_resolution(_store):
    """Capability-based: a file uploaded under scope A can be resolved
    by the chat-completion path regardless of scope. The 128-bit ID is
    the access control here (documented trust model)."""
    rec = await _store.create_file(
        stream=_stream(_PNG),
        filename="a.png",
        purpose="vision",
        scope="team-alpha",
    )
    connector = MediaConnector()
    result = connector.load_from_url(f"vllm-file://{rec.id}", _BytesMediaIO())
    assert result == _PNG


def test_vllm_file_scheme_raises_when_store_not_registered(tmp_path):
    register_store(None)  # ensure cleared
    connector = MediaConnector()
    with pytest.raises(RuntimeError, match="enable-file-uploads"):
        connector.load_from_url("vllm-file://file-" + "0" * 32, _BytesMediaIO())


@pytest.mark.asyncio
async def test_vllm_file_scheme_unknown_id_raises(_store):
    connector = MediaConnector()
    with pytest.raises(ValueError, match="Unknown vllm-file id"):
        connector.load_from_url("vllm-file://file-" + "0" * 32, _BytesMediaIO())


@pytest.mark.asyncio
async def test_vllm_file_async_path(_store):
    rec = await _store.create_file(
        stream=_stream(_PNG), filename="c.png", purpose="vision", scope=None
    )
    connector = MediaConnector()
    result = await connector.load_from_url_async(
        f"vllm-file://{rec.id}", _BytesMediaIO()
    )
    assert result == _PNG


@pytest.mark.asyncio
async def test_vllm_file_resolution_touches_atime(_store):
    import time

    rec = await _store.create_file(
        stream=_stream(_PNG), filename="d.png", purpose="vision", scope=None
    )
    original = rec.last_accessed
    time.sleep(0.02)
    connector = MediaConnector()
    connector.load_from_url(f"vllm-file://{rec.id}", _BytesMediaIO())
    assert rec.last_accessed > original


def test_register_store_and_get_store_roundtrip(tmp_path):
    assert get_store() is None
    config = FileUploadConfig(enabled=True, dir=str(tmp_path / "u"))
    store = FileUploadStore(config)
    register_store(store)
    try:
        assert get_store() is store
    finally:
        register_store(None)
    assert get_store() is None
