# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import Namespace
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI

from vllm.entrypoints.launchers.utils.server_utils import lifespan
from vllm.entrypoints.openai.responses.store.service import ResponsesStoreService


@pytest.mark.asyncio
async def test_lifespan_starts_and_closes_responses_store(monkeypatch) -> None:
    service = Mock()
    service.close = AsyncMock()
    create_service = Mock(return_value=service)
    monkeypatch.setattr(ResponsesStoreService, "from_cli_args", create_service)
    monkeypatch.setattr(
        "vllm.entrypoints.launchers.utils.server_utils.freeze_gc_heap",
        Mock(),
    )

    args = Namespace()
    app = FastAPI()
    app.state.args = args
    app.state.log_stats = False
    app.state.responses_store_enabled = True
    app.state.responses_store_service = None

    async with lifespan(app):
        create_service.assert_called_once_with(args)
        service.start.assert_called_once_with()
        assert app.state.responses_store_service is service

    service.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_lifespan_leaves_responses_store_disabled(monkeypatch) -> None:
    create_service = Mock()
    monkeypatch.setattr(ResponsesStoreService, "from_cli_args", create_service)
    monkeypatch.setattr(
        "vllm.entrypoints.launchers.utils.server_utils.freeze_gc_heap",
        Mock(),
    )

    app = FastAPI()
    app.state.args = Namespace()
    app.state.log_stats = False
    app.state.responses_store_enabled = False
    app.state.responses_store_service = None

    async with lifespan(app):
        create_service.assert_not_called()


@pytest.mark.asyncio
async def test_responses_store_service_starts_without_business_wiring(
    tmp_path: Path,
) -> None:
    args = Namespace(
        responses_store_disk_path=str(tmp_path / "responses.sqlite3"),
        responses_store_key_file=None,
        responses_store_config={
            "enabled": True,
            "disk_enabled": True,
            "memory_capacity_mb": 1,
            "disk_capacity_mb": 2,
            "memory_low_watermark": 0.5,
            "memory_high_watermark": 0.8,
            "disk_low_watermark": 0.6,
            "disk_high_watermark": 0.9,
            "memory_ttl_seconds": 30,
            "disk_ttl_seconds": 60,
            "cleanup_interval_seconds": 60,
            "cleanup_max_candidates": 8,
            "cleanup_max_bytes_mb": 1,
            "num_shards": 4,
            "disk_write_interval_seconds": 0.01,
        },
    )

    service = ResponsesStoreService.from_cli_args(args)
    service.start()
    assert service.is_running
    assert service.store.disk_enabled

    await service.close()
    assert not service.is_running
