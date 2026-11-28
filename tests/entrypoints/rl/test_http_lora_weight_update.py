# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for unified HTTP LoRA weight updates (weight-transfer backends)."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
from vllm.distributed.weight_transfer.lora_sink import LoRAWeightSink
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.rl.online.api_router import router as rl_router
from vllm.lora.request import _TensorLoRARequest
from vllm.v1.worker.gpu_worker import Worker

PEFT_CFG = {
    "r": 4,
    "lora_alpha": 8,
    "target_modules": ["q_proj", "v_proj"],
}


def _make_stub_worker(
    *,
    lora_enabled: bool = True,
    weight_transfer_engine: MagicMock | None = None,
) -> SimpleNamespace:
    model_runner = MagicMock()
    model_runner.add_lora.return_value = True
    vllm_config = SimpleNamespace(
        lora_config=SimpleNamespace(max_loras=4) if lora_enabled else None,
        parallel_config=SimpleNamespace(data_parallel_index=0, world_size=1),
        model_config=SimpleNamespace(model="base-model"),
        speculative_config=None,
    )
    worker = SimpleNamespace(
        vllm_config=vllm_config,
        lora_config=vllm_config.lora_config,
        model_config=vllm_config.model_config,
        model_runner=model_runner,
        rank=0,
        weight_transfer_engine=weight_transfer_engine,
        _weight_update_active=False,
        _weight_update_is_draft=False,
        _weight_update_lora_sink=None,
        _incomplete_weight_updates=set(),
        _kv_cache_reclaimable=False,
        _weight_update_kv_cache_loan=ExitStack(),
    )
    worker._check_weight_transfer_engine = lambda: Worker._check_weight_transfer_engine(
        worker
    )
    worker._start_weight_update = lambda is_draft=False, lora_adapter=None: (
        Worker._start_weight_update(
            worker, is_draft=is_draft, lora_adapter=lora_adapter
        )
    )
    worker._end_weight_update = lambda refresh: Worker._end_weight_update(
        worker, refresh
    )
    return worker


# ---------------------------------------------------------------------------
# LoRAWeightSink & GPUWorker tests
# ---------------------------------------------------------------------------


def test_lora_weight_sink_copies_tensors_and_validates():
    sink = LoRAWeightSink.from_dict(
        {
            "lora_name": "adapter_a",
            "lora_int_id": 1,
            "lora_path": "memory://adapter_a@1",
            "peft_config": PEFT_CFG,
        }
    )
    with pytest.raises(ValueError, match="No tensors received"):
        sink.to_lora_request()

    src = torch.ones(4, 4)
    sink.load_weights([("model.layers.0.q_proj.lora_A.weight", src)])
    # Mutating source buffer after load_weights must not affect stashed copy.
    src.zero_()
    req = sink.to_lora_request()
    assert isinstance(req, _TensorLoRARequest)
    assert req.lora_name == "adapter_a"
    assert req.lora_int_id == 1
    assert req.load_inplace is True
    assert torch.equal(
        req._lora_tensors["model.layers.0.q_proj.lora_A.weight"], torch.ones(4, 4)
    )


def test_gpu_worker_transfer_engine_lora_sink_lifecycle():
    captured_target: list[torch.nn.Module] = []
    engine = MagicMock()

    def _set_target(model, model_config):
        captured_target.append(model)

    def _update_weights(info):
        captured_target[-1].load_weights([("q.lora_A.weight", torch.ones(2, 2))])

    engine.set_weight_update_target.side_effect = _set_target
    engine.update_weights.side_effect = _update_weights

    worker = _make_stub_worker(lora_enabled=True, weight_transfer_engine=engine)
    spec = {
        "lora_name": "lora_a",
        "lora_int_id": 1,
        "lora_path": "memory://lora_a@1",
        "peft_config": PEFT_CFG,
    }

    with patch("vllm.v1.worker.gpu_worker.set_current_vllm_config"):
        Worker.start_weight_update(worker, lora_adapter=spec)
        assert worker._incomplete_weight_updates == set()
        engine.set_weight_update_target.assert_called_once()
        engine.start_weight_update.assert_called_once()

        Worker.update_weights(worker, {"names": ["q.lora_A.weight"]})
        engine.update_weights.assert_called_once()

        Worker.finish_weight_update(worker)
        engine.finish_weight_update.assert_called_once()
        engine.reset_weight_update_target.assert_called_once()

    assert worker._weight_update_active is False
    assert worker._weight_update_lora_sink is None
    assert worker._incomplete_weight_updates == set()
    worker.model_runner.on_target_weights_replaced.assert_not_called()
    worker.model_runner.add_lora.assert_called_once()
    req = worker.model_runner.add_lora.call_args[0][0]
    assert isinstance(req, _TensorLoRARequest)
    assert "q.lora_A.weight" in req._lora_tensors


def test_gpu_worker_rejects_lora_when_disabled():
    worker = _make_stub_worker(lora_enabled=False, weight_transfer_engine=MagicMock())
    with (
        patch("vllm.v1.worker.gpu_worker.set_current_vllm_config"),
        pytest.raises(RuntimeError, match="LoRA is not enabled"),
    ):
        Worker.start_weight_update(
            worker,
            lora_adapter={
                "lora_name": "a",
                "lora_int_id": 1,
                "lora_path": "memory://a@1",
                "peft_config": PEFT_CFG,
            },
        )


# ---------------------------------------------------------------------------
# Python FastAPI HTTP Router + OpenAIServingModels tests
# ---------------------------------------------------------------------------


def _make_test_app(*, lora_enabled: bool = True):
    engine = MagicMock()
    engine.vllm_config = SimpleNamespace(
        lora_config=SimpleNamespace(max_loras=4) if lora_enabled else None,
        parallel_config=SimpleNamespace(world_size=1, world_size_across_dp=1),
    )
    engine.model_config = SimpleNamespace(model="base-model", max_model_len=4096)
    engine.renderer = None
    engine.input_processor = None
    engine.start_weight_update = AsyncMock()
    engine.update_weights = AsyncMock()
    engine.finish_weight_update = AsyncMock()
    engine.reset_prefix_cache = AsyncMock()

    models = OpenAIServingModels(
        engine_client=engine,
        base_model_paths=[BaseModelPath(name="base-model", model_path="/models/base")],
    )

    app = FastAPI()
    app.state.engine_client = engine
    app.state.openai_serving_models = models
    app.include_router(rl_router)
    return TestClient(app), engine, models


def test_http_router_base_model_backwards_compatible():
    client, engine, _ = _make_test_app()
    assert client.post("/start_weight_update").status_code == 200
    engine.start_weight_update.assert_awaited_once_with()

    assert client.post("/finish_weight_update").status_code == 200
    engine.finish_weight_update.assert_awaited_once_with(None)
    engine.reset_prefix_cache.assert_not_awaited()


def test_http_router_multi_lora_independent_updates_and_revisions():
    client, engine, models = _make_test_app()

    # 1. Update adapter_a
    r1 = client.post(
        "/start_weight_update",
        json={"lora_adapter": {"lora_name": "adapter_a", "peft_config": PEFT_CFG}},
    )
    assert r1.status_code == 200
    assert r1.json()["lora_adapter"] == {
        "lora_name": "adapter_a",
        "lora_int_id": 1,
        "lora_path": "memory://adapter_a@1",
    }
    assert "adapter_a" not in models.lora_requests
    assert (
        client.post("/finish_weight_update", json={"weight_version": "s1"}).status_code
        == 200
    )
    assert "adapter_a" in models.lora_requests
    engine.reset_prefix_cache.assert_not_awaited()

    # 2. Update adapter_b (gets id=2, @2)
    r2 = client.post(
        "/start_weight_update",
        json={"lora_adapter": {"lora_name": "adapter_b", "peft_config": PEFT_CFG}},
    )
    assert r2.status_code == 200
    assert r2.json()["lora_adapter"] == {
        "lora_name": "adapter_b",
        "lora_int_id": 2,
        "lora_path": "memory://adapter_b@2",
    }
    assert client.post("/finish_weight_update").status_code == 200
    assert set(models.lora_requests.keys()) == {"adapter_a", "adapter_b"}

    # 3. Update adapter_a again -> reuses id=1, advances @3, adapter_b untouched
    r3 = client.post(
        "/start_weight_update",
        json={"lora_adapter": {"lora_name": "adapter_a", "peft_config": PEFT_CFG}},
    )
    assert r3.status_code == 200
    assert r3.json()["lora_adapter"] == {
        "lora_name": "adapter_a",
        "lora_int_id": 1,
        "lora_path": "memory://adapter_a@3",
    }
    assert client.post("/finish_weight_update").status_code == 200
    assert models.lora_requests["adapter_a"].lora_path == "memory://adapter_a@3"
    assert models.lora_requests["adapter_b"].lora_path == "memory://adapter_b@2"
    engine.reset_prefix_cache.assert_not_awaited()


def test_http_router_validation_errors():
    client, _, _ = _make_test_app()
    assert (
        client.post(
            "/start_weight_update",
            json={"lora_adapter": {"lora_name": "", "peft_config": PEFT_CFG}},
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/start_weight_update",
            json={"lora_adapter": {"lora_name": "base-model", "peft_config": PEFT_CFG}},
        ).status_code
        == 400
    )
    disabled_client, _, _ = _make_test_app(lora_enabled=False)
    assert (
        disabled_client.post(
            "/start_weight_update",
            json={"lora_adapter": {"lora_name": "a", "peft_config": PEFT_CFG}},
        ).status_code
        == 400
    )


def test_http_weight_sync_client_lora_helpers():
    sync_client = HTTPVLLMWeightSyncClient("http://localhost:8000")
    posted: list[tuple[str, dict | None]] = []

    def _fake_post(path, json=None):
        posted.append((path, json))
        return {"message": "ok"}

    sync_client._post = _fake_post  # type: ignore[method-assign]
    sync_client.start_weight_update(
        lora_adapter={"lora_name": "a", "peft_config": PEFT_CFG}
    )
    sync_client.finish_weight_update(reset_prefix_cache=True)

    assert posted[0] == (
        "start_weight_update",
        {"lora_adapter": {"lora_name": "a", "peft_config": PEFT_CFG}},
    )
    assert posted[1] == ("finish_weight_update", {"reset_prefix_cache": True})
