# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight snapshot: save/restore of post-processed model state."""

import os
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from vllm.device_allocator import weight_snapshot as ws

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


class _Quant:
    """A non-module object holding per-expert scales, as MoE quant methods do."""

    def __init__(self, device):
        self.scales = [torch.rand(4, device=device), torch.rand(4, device=device)]
        self.group = 16


class _Toy(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.linear = nn.Linear(8, 8, device=device)
        self.register_buffer("table", torch.rand(3, 5, device=device))
        self.quant = _Quant(device)
        self.alias = self.linear.weight.data  # same storage, second name
        self.calibrated = 0.0


def _roots(model):
    return {"model": model}


def _perturb(model):
    with torch.no_grad():
        for t in (model.linear.weight, model.linear.bias, model.table):
            t.zero_()
        for t in model.quant.scales:
            t.zero_()
    model.calibrated = -1.0


@cuda
def test_roundtrip_restores_tensors_scalars_and_aliases(tmp_path):
    src = _Toy()
    src.calibrated = 3.5
    path = tmp_path / "snap.bin"
    ws.save(_roots(src), path)

    dst = _Toy()
    _perturb(dst)
    ws.restore(_roots(dst), path)

    torch.testing.assert_close(dst.linear.weight, src.linear.weight)
    torch.testing.assert_close(dst.linear.bias, src.linear.bias)
    torch.testing.assert_close(dst.table, src.table)
    for a, b in zip(dst.quant.scales, src.quant.scales):
        torch.testing.assert_close(a, b)
    assert dst.calibrated == 3.5
    assert dst.alias.data_ptr() == dst.linear.weight.data_ptr()


@cuda
def test_restore_creates_attributes_only_the_real_load_made(tmp_path):
    src = _Toy()
    src.fused = torch.rand(6, device="cuda")  # e.g. fused at load time
    path = tmp_path / "snap.bin"
    ws.save(_roots(src), path)

    dst = _Toy()
    dst.fused = None
    ws.restore(_roots(dst), path)
    torch.testing.assert_close(dst.fused, src.fused)


@cuda
def test_shape_mismatch_is_rejected(tmp_path):
    path = tmp_path / "snap.bin"
    ws.save(_roots(_Toy()), path)
    dst = _Toy()
    dst.table = torch.rand(4, 5, device="cuda")
    with pytest.raises(RuntimeError, match="does not match the model"):
        ws.restore(_roots(dst), path)


@cuda
@pytest.mark.parametrize("cut", ["data", "header"])
def test_truncated_snapshot_is_rejected(tmp_path, cut):
    path = tmp_path / "snap.bin"
    ws.save(_roots(_Toy()), path)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) - 1] if cut == "data" else data[:4])
    dst = _Toy()
    before = dst.linear.weight.detach().clone()
    with pytest.raises(RuntimeError, match="truncated"):
        ws.restore(_roots(dst), path)
    assert torch.equal(dst.linear.weight, before)


def test_short_reads_are_retried_and_eof_raises(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(bytes(range(16)))
    buf = bytearray(16)
    fd = os.open(path, os.O_RDONLY)
    try:
        real = os.preadv
        with pytest.MonkeyPatch.context() as m:
            # One byte per call: a short read must be retried, not accepted.
            m.setattr(ws.os, "preadv", lambda f, b, p: real(f, [b[0][:1]], p))
            ws._pread_exact(fd, buf, 0, 16)
        assert bytes(buf) == bytes(range(16))
        with pytest.raises(RuntimeError, match="truncated"):
            ws._pread_exact(fd, bytearray(32), 0, 32)
    finally:
        os.close(fd)


@cuda
def test_old_format_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "snap.bin"
    monkeypatch.setattr(ws, "_MAGIC", "vllm-weight-snapshot-v0")
    ws.save(_roots(_Toy()), path)
    monkeypatch.undo()
    with pytest.raises(RuntimeError, match="old format"):
        ws.restore(_roots(_Toy()), path)


@cuda
def test_pure_prep_functions_are_skipped_on_restore(tmp_path, monkeypatch):
    calls = []
    mod: Any = types.ModuleType("ws_test_prep")

    def pack(x):
        calls.append(tuple(x.shape))
        return x.repeat(2, 1)

    mod.pack = pack
    monkeypatch.setitem(sys.modules, "ws_test_prep", mod)
    monkeypatch.setenv("VLLM_WEIGHT_SNAPSHOT_SKIP_FUNCS", "ws_test_prep:pack")

    class Runner:
        def __init__(self):
            self.model: Any = None
            self.dummy: list = []

        def load_model(self, load_dummy_weights=False):
            self.dummy.append((load_dummy_weights, ws.is_restoring()))
            m = _Toy()
            m.packed = sys.modules["ws_test_prep"].pack(m.table)
            self.model = m

    path = tmp_path / "snap.bin"
    first = Runner()
    ws.load_model(first, path)  # save
    assert calls == [(3, 5)] and first.dummy == [(False, False)]

    second = Runner()
    ws.load_model(second, path)  # restore: pack returns an empty of the same shape
    assert calls == [(3, 5)], "pack must not run on restore"
    assert second.dummy == [(True, True)]
    assert not ws.is_restoring()
    torch.testing.assert_close(second.model.packed, first.model.packed)
    assert sys.modules["ws_test_prep"].pack is pack, "patch not undone"


@cuda
def test_skip_falls_back_when_input_shapes_differ(tmp_path, monkeypatch):
    calls = []
    mod: Any = types.ModuleType("ws_test_prep2")

    def pack(x):
        calls.append(tuple(x.shape))
        return x * 2

    mod.pack = pack
    monkeypatch.setitem(sys.modules, "ws_test_prep2", mod)
    monkeypatch.setenv("VLLM_WEIGHT_SNAPSHOT_SKIP_FUNCS", "ws_test_prep2:pack")
    replay = {
        "ws_test_prep2:pack": [
            {"in": [[9, 9]], "out": {"t": [[9, 9], "torch.float32", "cuda:0"]}}
        ]
    }
    undo = ws._patch_skip_funcs(replay)
    try:
        sys.modules["ws_test_prep2"].pack(torch.ones(3, 5, device="cuda"))
    finally:
        undo()
    assert calls == [(3, 5)]


def _config(model, tp=1, quant=None, dtype=torch.bfloat16):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model=model,
            revision=None,
            dtype=dtype,
            quantization=quant,
            hf_overrides={},
            multimodal_config=None,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            pipeline_parallel_size=1,
            enable_expert_parallel=False,
        ),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        speculative_config=None,
    )


def _hf_snapshot(root, repo, commit):
    d = root / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / commit
    d.mkdir(parents=True)
    (d / "config.json").write_text('{"hidden_size": 8}')
    return d


@pytest.mark.cpu_test
def test_key_ignores_where_the_hf_checkpoint_lives(tmp_path):
    a = _hf_snapshot(tmp_path / "a", "org/model", "abc123")
    b = _hf_snapshot(tmp_path / "b", "org/model", "abc123")
    assert ws.weights_key(_config(str(a))) == ws.weights_key(_config(str(b)))


@pytest.mark.cpu_test
def test_key_changes_with_weight_shaping_settings(tmp_path):
    d = str(_hf_snapshot(tmp_path, "org/model", "abc123"))
    base = ws.weights_key(_config(d))
    assert base != ws.weights_key(_config(d, tp=2))
    assert base != ws.weights_key(_config(d, quant="fp8"))
    assert base != ws.weights_key(_config(d, dtype=torch.float16))
    other = str(_hf_snapshot(tmp_path / "x", "org/model", "def456"))
    assert base != ws.weights_key(_config(other))


@pytest.mark.cpu_test
def test_key_of_local_dir_tracks_weight_files(tmp_path):
    d = tmp_path / "local"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"x" * 10)
    before = ws.weights_key(_config(str(d)))
    (d / "model.safetensors").write_bytes(b"x" * 11)
    assert before != ws.weights_key(_config(str(d)))


@pytest.mark.cpu_test
def test_snapshot_path_is_per_rank(tmp_path):
    root = str(tmp_path)
    assert ws.snapshot_path(root, "k", 0) != ws.snapshot_path(root, "k", 1)
