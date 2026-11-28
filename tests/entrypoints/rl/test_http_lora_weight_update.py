# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for unified HTTP LoRA & multi-LoRA weight updates."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
from vllm.distributed.weight_transfer.lora_session import (
    LoRAAdapterUpdateSpec,
    LoRAWeightUpdateSession,
    decode_lora_tensors_http,
    encode_lora_tensors_http,
)
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.rl.online.api_router import router as rl_router
from vllm.lora.request import LoRARequest, _TensorLoRARequest
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
    return SimpleNamespace(
        vllm_config=vllm_config,
        lora_config=vllm_config.lora_config,
        model_config=vllm_config.model_config,
        model_runner=model_runner,
        rank=0,
        weight_transfer_engine=weight_transfer_engine,
        _weight_update_active=False,
        _weight_update_is_draft=False,
        _weight_update_lora_session=None,
        _incomplete_weight_updates=set(),
        _kv_cache_reclaimable=False,
        _weight_update_kv_cache_loan=ExitStack(),
        _check_weight_transfer_engine=lambda: Worker._check_weight_transfer_engine(
            None  # replaced below
        ),
    )


def _bind_worker_methods(worker: SimpleNamespace) -> SimpleNamespace:
    worker._check_weight_transfer_engine = lambda: Worker._check_weight_transfer_engine(
        worker
    )
    worker._start_lora_weight_update = lambda adapters: (
        Worker._start_lora_weight_update(worker, adapters)
    )
    worker._end_lora_weight_update = lambda: Worker._end_lora_weight_update(worker)
    return worker


# ---------------------------------------------------------------------------
# LoRAWeightUpdateSession & HTTP safetensors codec tests
# ---------------------------------------------------------------------------


def test_http_safetensors_roundtrip_and_aliased_storage():
    base = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    tensors = {
        "layer0.lora_A.weight": base,
        # Aliased view sharing storage with base; safetensors must still succeed.
        "layer0.lora_B.weight": base.view(4, 4),
    }
    payload = encode_lora_tensors_http(tensors, lora_name="adapter_a")
    assert "safetensors_b64" in payload
    assert payload["lora_name"] == "adapter_a"

    decoded, lora_name = decode_lora_tensors_http(payload)
    assert lora_name == "adapter_a"
    assert set(decoded.keys()) == set(tensors.keys())
    for key, expected in tensors.items():
        assert torch.equal(decoded[key], expected)


def test_lora_session_multi_adapter_routing_and_mixed_disk():
    session = LoRAWeightUpdateSession.from_dicts(
        [
            {
                "lora_name": "adapter_a",
                "lora_int_id": 1,
                "lora_path": "memory://adapter_a@1",
                "peft_config": PEFT_CFG,
            },
            {
                "lora_name": "adapter_b",
                "lora_int_id": 2,
                "lora_path": "memory://adapter_b@2",
                "peft_config": PEFT_CFG,
            },
            {
                "lora_name": "adapter_disk",
                "lora_int_id": 3,
                "lora_path": "/tmp/adapter_disk",
            },
        ]
    )

    # Multi-adapter prefixed stream (NCCL / IPC style)
    w_a = torch.ones(4, 4)
    w_b = torch.full((4, 4), 2.0)
    session.sink.load_weights(
        [
            ("adapter_a:model.layers.0.q_proj.lora_A.weight", w_a),
            ("adapter_b:model.layers.0.q_proj.lora_A.weight", w_b),
        ]
    )

    # Chunked HTTP safetensors with explicit lora_name (no prefix required)
    w_a_b = torch.full((4, 4), 3.0)
    assert session.receive_http_tensors(
        encode_lora_tensors_http(
            {"model.layers.0.q_proj.lora_B.weight": w_a_b},
            lora_name="adapter_a",
        )
    )

    # Unprefixed weight when multiple tensor adapters exist must fail clearly
    with pytest.raises(ValueError, match="must be prefixed"):
        session.sink.load_weights([("model.layers.0.v_proj.lora_A.weight", w_a)])

    # Targeting a disk-only adapter must fail clearly
    with pytest.raises(ValueError, match="not configured with peft_config"):
        session.sink.load_weights([("adapter_disk:foo.weight", w_a)])

    reqs = session.build_lora_requests()
    assert len(reqs) == 3
    assert isinstance(reqs[0], _TensorLoRARequest)
    assert reqs[0].lora_name == "adapter_a"
    assert reqs[0].load_inplace is True
    assert set(reqs[0]._lora_tensors.keys()) == {
        "model.layers.0.q_proj.lora_A.weight",
        "model.layers.0.q_proj.lora_B.weight",
    }
    assert isinstance(reqs[1], _TensorLoRARequest)
    assert reqs[1].lora_name == "adapter_b"
    assert type(reqs[2]) is LoRARequest
    assert reqs[2].lora_name == "adapter_disk"
    assert reqs[2].lora_path == "/tmp/adapter_disk"


def test_lora_session_missing_tensors_raises():
    session = LoRAWeightUpdateSession(
        [
            LoRAAdapterUpdateSpec(
                lora_name="adapter_a",
                lora_int_id=1,
                lora_path="memory://adapter_a@1",
                peft_config=PEFT_CFG,
            )
        ]
    )
    with pytest.raises(ValueError, match="No tensors received"):
        session.build_lora_requests()


# ---------------------------------------------------------------------------
# GPUWorker LoRA session tests (HTTP, Disk, WeightTransferEngine, Abort)
# ---------------------------------------------------------------------------


def test_gpu_worker_http_lora_without_weight_transfer_engine():
    worker = _bind_worker_methods(
        _make_stub_worker(lora_enabled=True, weight_transfer_engine=None)
    )
    specs = [
        {
            "lora_name": "lora_mem",
            "lora_int_id": 1,
            "lora_path": "memory://lora_mem@1",
            "peft_config": PEFT_CFG,
        }
    ]
    w = torch.randn(4, 4)

    with patch("vllm.v1.worker.gpu_worker.set_current_vllm_config"):
        Worker.start_weight_update(worker, lora_adapters=specs)
        assert worker._weight_update_active is True
        assert worker._incomplete_weight_updates == set()

        Worker.update_weights(worker, encode_lora_tensors_http({"q.lora_A.weight": w}))
        Worker.finish_weight_update(worker)

    assert worker._weight_update_active is False
    assert worker._weight_update_lora_session is None
    assert worker._incomplete_weight_updates == set()
    worker.model_runner.on_target_weights_replaced.assert_not_called()
    worker.model_runner.add_lora.assert_called_once()
    req = worker.model_runner.add_lora.call_args[0][0]
    assert isinstance(req, _TensorLoRARequest)
    assert req.lora_name == "lora_mem"
    assert req.lora_int_id == 1
    assert req.load_inplace is True
    assert torch.equal(req._lora_tensors["q.lora_A.weight"], w)


def test_gpu_worker_disk_lora_and_abort():
    worker = _bind_worker_methods(
        _make_stub_worker(lora_enabled=True, weight_transfer_engine=None)
    )
    disk_spec = [
        {
            "lora_name": "lora_disk",
            "lora_int_id": 7,
            "lora_path": "/shared/lora_disk",
        }
    ]

    with patch("vllm.v1.worker.gpu_worker.set_current_vllm_config"):
        # Abort leaves model_runner untouched
        Worker.start_weight_update(worker, lora_adapters=disk_spec)
        Worker.finish_weight_update(worker, abort=True)
        worker.model_runner.add_lora.assert_not_called()
        assert worker._weight_update_active is False

        # Commit loads disk LoRARequest
        Worker.start_weight_update(worker, lora_adapters=disk_spec)
        Worker.finish_weight_update(worker)
        worker.model_runner.add_lora.assert_called_once()
        req = worker.model_runner.add_lora.call_args[0][0]
        assert type(req) is LoRARequest
        assert req.lora_name == "lora_disk"
        assert req.lora_int_id == 7
        assert req.lora_path == "/shared/lora_disk"
        assert req.load_inplace is True


def test_gpu_worker_transfer_engine_lora_sink_lifecycle():
    captured_target: list[torch.nn.Module] = []
    engine = MagicMock()

    def _set_target(model, model_config):
        captured_target.append(model)

    def _update_weights(info):
        # Simulate NCCL/IPC delivering weights to the retargeted sink
        captured_target[-1].load_weights([("lora_a:q.lora_A.weight", torch.ones(2, 2))])

    engine.set_weight_update_target.side_effect = _set_target
    engine.update_weights.side_effect = _update_weights

    worker = _bind_worker_methods(
        _make_stub_worker(lora_enabled=True, weight_transfer_engine=engine)
    )
    specs = [
        {
            "lora_name": "lora_a",
            "lora_int_id": 1,
            "lora_path": "memory://lora_a@1",
            "peft_config": PEFT_CFG,
        }
    ]

    with patch("vllm.v1.worker.gpu_worker.set_current_vllm_config"):
        Worker.start_weight_update(worker, lora_adapters=specs)
        # Engine is NOT touched until a non-HTTP update_weights arrives
        engine.start_weight_update.assert_not_called()

        Worker.update_weights(worker, {"names": ["lora_a:q.lora_A.weight"]})
        engine.set_weight_update_target.assert_called_once()
        engine.start_weight_update.assert_called_once()
        engine.update_weights.assert_called_once()

        Worker.finish_weight_update(worker)
        engine.finish_weight_update.assert_called_once()
        engine.reset_weight_update_target.assert_called_once()

    worker.model_runner.add_lora.assert_called_once()
    req = worker.model_runner.add_lora.call_args[0][0]
    assert isinstance(req, _TensorLoRARequest)
    assert "q.lora_A.weight" in req._lora_tensors


def test_gpu_worker_rejects_lora_when_disabled():
    worker = _bind_worker_methods(_make_stub_worker(lora_enabled=False))
    with (
        patch("vllm.v1.worker.gpu_worker.set_current_vllm_config"),
        pytest.raises(RuntimeError, match="LoRA is not enabled"),
    ):
        Worker.start_weight_update(
            worker,
            lora_adapters=[
                {
                    "lora_name": "a",
                    "lora_int_id": 1,
                    "lora_path": "/tmp/a",
                }
            ],
        )


# ---------------------------------------------------------------------------
# Python FastAPI HTTP Router + OpenAIServingModels integration tests
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
    r1 = client.post("/start_weight_update")
    assert r1.status_code == 200
    assert r1.json() == {"message": "Weight update started"}
    engine.start_weight_update.assert_awaited_once_with()

    r2 = client.post("/finish_weight_update")
    assert r2.status_code == 200
    assert r2.json() == {"message": "Weight update finished"}
    engine.finish_weight_update.assert_awaited_once_with(None, abort=False)
    engine.reset_prefix_cache.assert_not_awaited()


def test_http_router_multi_lora_lifecycle_and_prefix_cache_semantics():
    client, engine, models = _make_test_app()

    # 1. Start multi-LoRA session (one in-memory, one disk)
    r_start = client.post(
        "/start_weight_update",
        json={
            "lora_adapters": [
                {"lora_name": "adapter_mem", "peft_config": PEFT_CFG},
                {"lora_name": "adapter_disk", "lora_path": "/shared/disk_v1"},
            ]
        },
    )
    assert r_start.status_code == 200
    body = r_start.json()
    assert len(body["lora_adapters"]) == 2
    assert body["lora_adapters"][0]["lora_name"] == "adapter_mem"
    assert body["lora_adapters"][0]["lora_int_id"] == 1
    assert body["lora_adapters"][0]["lora_path"] == "memory://adapter_mem@1"
    assert body["lora_adapters"][1] == {
        "lora_name": "adapter_disk",
        "lora_int_id": 2,
        "lora_path": "/shared/disk_v1",
    }
    # Not published to OpenAIServingModels before finish!
    assert "adapter_mem" not in models.lora_requests
    assert "adapter_disk" not in models.lora_requests

    # 2. Send HTTP safetensors chunk
    r_upd = client.post(
        "/update_weights",
        json={
            "update_info": encode_lora_tensors_http(
                {"q.lora_A.weight": torch.ones(2, 2)}, lora_name="adapter_mem"
            )
        },
    )
    assert r_upd.status_code == 200

    # 3. Finish -> publishes both, does NOT reset prefix cache (fresh adapters)
    r_fin = client.post("/finish_weight_update", json={"weight_version": "step-1"})
    assert r_fin.status_code == 200
    assert "adapter_mem" in models.lora_requests
    assert "adapter_disk" in models.lora_requests
    engine.reset_prefix_cache.assert_not_awaited()

    # 4. Update in-memory adapter again -> reuses lora_int_id=1, new memory path
    # -> no prefix reset!
    r_start2 = client.post(
        "/start_weight_update",
        json={"lora_adapters": [{"lora_name": "adapter_mem", "peft_config": PEFT_CFG}]},
    )
    assert r_start2.status_code == 200
    assert r_start2.json()["lora_adapters"][0]["lora_int_id"] == 1
    assert r_start2.json()["lora_adapters"][0]["lora_path"] == "memory://adapter_mem@2"
    r_fin2 = client.post("/finish_weight_update")
    assert r_fin2.status_code == 200
    assert models.lora_requests["adapter_mem"].lora_path == "memory://adapter_mem@2"
    engine.reset_prefix_cache.assert_not_awaited()

    # 5. Reload disk adapter from the SAME path -> auto-resets prefix cache!
    r_start3 = client.post(
        "/start_weight_update",
        json={
            "lora_adapters": [
                {"lora_name": "adapter_disk", "lora_path": "/shared/disk_v1"}
            ]
        },
    )
    assert r_start3.status_code == 200
    assert r_start3.json()["lora_adapters"][0]["lora_int_id"] == 2
    r_fin3 = client.post("/finish_weight_update")
    assert r_fin3.status_code == 200
    engine.reset_prefix_cache.assert_awaited_once()


def test_http_router_abort_and_validation_errors():
    client, engine, models = _make_test_app()

    # Abort leaves registry untouched
    r_start = client.post(
        "/start_weight_update",
        json={"lora_adapters": [{"lora_name": "tmp_lora", "peft_config": PEFT_CFG}]},
    )
    assert r_start.status_code == 200
    r_abort = client.post("/finish_weight_update", json={"abort": True})
    assert r_abort.status_code == 200
    assert r_abort.json() == {"message": "Weight update aborted", "aborted": True}
    assert "tmp_lora" not in models.lora_requests
    engine.finish_weight_update.assert_awaited_once_with(None, abort=True)

    # Validation errors
    assert (
        client.post("/start_weight_update", json={"lora_adapters": []}).status_code
        == 400
    )
    assert (
        client.post(
            "/start_weight_update",
            json={
                "lora_adapters": [{"lora_name": "base-model", "peft_config": PEFT_CFG}]
            },
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/start_weight_update",
            json={
                "lora_adapters": [
                    {"lora_name": "dup", "peft_config": PEFT_CFG},
                    {"lora_name": "dup", "peft_config": PEFT_CFG},
                ]
            },
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/start_weight_update",
            json={"lora_adapters": [{"lora_name": "missing_src"}]},
        ).status_code
        == 400
    )

    # Disabled LoRA -> 400
    disabled_client, _, _ = _make_test_app(lora_enabled=False)
    assert (
        disabled_client.post(
            "/start_weight_update",
            json={"lora_adapters": [{"lora_name": "a", "peft_config": PEFT_CFG}]},
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
        lora_adapters=[{"lora_name": "a", "peft_config": PEFT_CFG}]
    )
    sync_client.update_lora_weights_http(
        {"q.lora_A.weight": torch.ones(2, 2)}, lora_name="a"
    )
    sync_client.finish_weight_update(abort=True)

    assert posted[0] == (
        "start_weight_update",
        {"lora_adapters": [{"lora_name": "a", "peft_config": PEFT_CFG}]},
    )
    assert posted[1][0] == "update_weights"
    assert "safetensors_b64" in posted[1][1]["update_info"]
    assert posted[1][1]["update_info"]["lora_name"] == "a"
    assert posted[2] == ("finish_weight_update", {"abort": True})
