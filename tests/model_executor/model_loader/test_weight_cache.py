# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the weight cache daemon protocol and the IPC model loader."""

import socket
import threading

import pytest
import torch

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.weight_cache import (
    CacheConfig,
    CacheConfigMismatchError,
    IpcModelLoader,
    TensorEntry,
    WeightCacheUnavailableError,
)
from vllm.model_executor.model_loader.weight_cache.protocol import (
    recv_msg,
    send_msg,
)


def _make_cache_config(**overrides) -> CacheConfig:
    kwargs = dict(
        model="test-model",
        model_arch="LlamaForCausalLM",
        tp_size=2,
        tp_rank=0,
        dtype="torch.bfloat16",
        quantization=None,
        quant_config_hash="",
        revision=None,
        vllm_version="test",
    )
    kwargs.update(overrides)
    return CacheConfig(**kwargs)


def test_cache_config_match():
    assert _make_cache_config().mismatched_fields(_make_cache_config()) == []


def test_cache_config_mismatch():
    other = _make_cache_config(tp_rank=1, dtype="torch.float16")
    assert _make_cache_config().mismatched_fields(other) == ["tp_rank", "dtype"]


def test_protocol_roundtrip():
    left, right = socket.socketpair()
    with left, right:
        msg = {"cmd": "get_state", "cache_config": _make_cache_config()}
        send_msg(left, msg)
        assert recv_msg(right) == msg


def test_get_model_loader_dispatch():
    loader = get_model_loader(LoadConfig(load_format="ipc_cache"))
    assert isinstance(loader, IpcModelLoader)
    assert loader.mode == "zero_copy"
    assert loader.fallback


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="mode"):
        IpcModelLoader(
            LoadConfig(
                load_format="ipc_cache",
                model_loader_extra_config={"mode": "bogus"},
            )
        )


def test_unexpected_extra_config_rejected():
    with pytest.raises(ValueError, match="Unexpected extra config"):
        IpcModelLoader(
            LoadConfig(
                load_format="ipc_cache",
                model_loader_extra_config={"bogus": 1},
            )
        )


def test_fallback_load_config_resets_format():
    loader = IpcModelLoader(
        LoadConfig(
            load_format="ipc_cache",
            model_loader_extra_config={"mode": "copy", "fallback": False},
        )
    )
    fallback_config = loader._fallback_load_config()
    assert fallback_config.load_format == "auto"
    assert fallback_config.model_loader_extra_config == {}


def test_unavailable_daemon_raises(tmp_path):
    loader = IpcModelLoader(
        LoadConfig(
            load_format="ipc_cache",
            model_loader_extra_config={
                "socket_path": str(tmp_path / "missing.sock"),
                "fallback": False,
            },
        )
    )
    with pytest.raises(WeightCacheUnavailableError):
        loader._request_state(_make_cache_config())


def _serve_one_response(server: socket.socket, response: dict) -> None:
    conn, _ = server.accept()
    with conn:
        recv_msg(conn)
        send_msg(conn, response)


def _request_against_fake_daemon(tmp_path, response: dict):
    socket_path = str(tmp_path / "daemon.sock")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen()
    thread = threading.Thread(target=_serve_one_response, args=(server, response))
    thread.start()
    try:
        loader = IpcModelLoader(
            LoadConfig(
                load_format="ipc_cache",
                model_loader_extra_config={
                    "socket_path": socket_path,
                    "fallback": False,
                },
            )
        )
        return loader._request_state(_make_cache_config())
    finally:
        thread.join(timeout=10)
        server.close()


def test_mismatch_response_raises(tmp_path):
    with pytest.raises(CacheConfigMismatchError, match="tp_rank"):
        _request_against_fake_daemon(
            tmp_path, {"status": "mismatch", "fields": ["tp_rank"]}
        )


def test_error_response_raises(tmp_path):
    with pytest.raises(WeightCacheUnavailableError, match="released"):
        _request_against_fake_daemon(
            tmp_path, {"status": "error", "message": "Weights were released"}
        )


def test_ok_response_returns_entries(tmp_path):
    entry = TensorEntry.from_tensor(torch.arange(4, dtype=torch.float32), "param")
    entries, aliases = _request_against_fake_daemon(
        tmp_path, {"status": "ok", "entries": {"layer.weight": entry}}
    )
    assert aliases == {}
    assert torch.equal(
        entries["layer.weight"].rebuild(0),
        torch.arange(4, dtype=torch.float32),
    )


class _TiedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(4, 3)
        self.lm_head = torch.nn.Linear(3, 4, bias=False)
        self.lm_head.weight = self.embed.weight


def test_export_entries_records_tied_alias():
    from vllm.model_executor.model_loader.weight_cache.daemon import export_entries

    entries, aliases = export_entries(_TiedModel())
    assert "embed.weight" in entries
    assert "lm_head.weight" not in entries
    assert aliases == {"lm_head.weight": "embed.weight"}


def test_apply_entries_restores_tied_identity():
    from vllm.model_executor.model_loader.weight_cache.daemon import export_entries

    src = _TiedModel()
    entries, aliases = export_entries(src)

    dst = _TiedModel()
    # Break the tie so the two names point at distinct parameters, mimicking a
    # freshly initialized model before the cached weights are applied.
    dst.lm_head.weight = torch.nn.Parameter(torch.zeros(4, 3))

    loader = IpcModelLoader(LoadConfig(load_format="ipc_cache"))
    loader._apply_entries(dst, entries, aliases, device_index=0)

    assert dst.lm_head.weight is dst.embed.weight
    assert torch.equal(dst.embed.weight, src.embed.weight)


def _ipc_producer(conn, done) -> None:
    tensor = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
    conn.send(TensorEntry.from_tensor(tensor, "param"))
    done.wait(timeout=60)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_tensor_entry_cross_process_ipc():
    ctx = torch.multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    done = ctx.Event()
    proc = ctx.Process(target=_ipc_producer, args=(child_conn, done))
    proc.start()
    try:
        entry = parent_conn.recv()
        tensor = entry.rebuild(torch.cuda.current_device())
        expected = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        assert torch.equal(tensor.cpu(), expected)
    finally:
        done.set()
        proc.join(timeout=60)
