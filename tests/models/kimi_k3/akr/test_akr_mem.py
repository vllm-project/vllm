# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Where the amd_kernels routes keep their buffers (CPU only, the kernels' libraries
faked): the MoE workspaces lent to the capture streams, the DSA decode buffers sized by
the server's decode rows, the drafter's logits exchange with one receive buffer and the
owner mode warmed only for the batch-sharded sampler, the drafter's stock staging
dropped.

Run: python -m pytest tests/models/kimi_k3/akr/test_akr_mem.py
"""

import contextlib
import types

import pytest
import torch

from vllm.models.kimi_k3.amd import akr_draft, akr_dsa, akr_lmhead, akr_moe, akr_runtime

A, B, C = 111, 222, 333  # stream handles


def _swap_dict(work: dict):
    def swap(dev, a, b):
        moved = {}
        for key in [k for k in work if k[0] == dev and k[1] in (a, b)]:
            moved[(dev, b if key[1] == a else a)] = work.pop(key)
        work.update(moved)

    return swap


def _install_fake_tp(monkeypatch, streams: list[int]):
    """A TP group whose graph_capture makes the next of `streams` current."""
    import vllm.distributed.parallel_state as ps

    current = [A]

    class _Tp:
        @contextlib.contextmanager
        def graph_capture(self, ctx=None):
            prev = current[0]
            current[0] = ctx if ctx is not None else streams.pop(0)
            try:
                yield current[0]
            finally:
                current[0] = prev

    tp = _Tp()
    monkeypatch.setattr(ps, "get_tp_group", lambda: tp)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda *a, **k: types.SimpleNamespace(cuda_stream=current[0]),
    )
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda *a, **k: None)
    return tp


@pytest.fixture
def runtime_state(monkeypatch):
    monkeypatch.setattr(akr_runtime, "_prewarm", [])
    monkeypatch.setattr(akr_runtime, "_warmed_streams", set())
    monkeypatch.setattr(akr_runtime, "_hooked", False)
    monkeypatch.setattr(akr_runtime, "_swappers", [])
    monkeypatch.setattr(akr_runtime, "_holder", {})


def test_capture_streams_borrow_the_load_set(monkeypatch, runtime_state):
    tp = _install_fake_tp(monkeypatch, [B, C])
    work = {(0, A): "set", (1, A): "other device"}
    seen = []
    akr_runtime.register_stream_workspaces(_swap_dict(work))  # held by A
    akr_runtime.register_prewarm(lambda: seen.append(dict(work)))
    akr_runtime.install_capture_hook()
    with tp.graph_capture():  # B (profiling capture): borrows, the prewarm finds it
        assert work == {(0, B): "set", (1, A): "other device"}
        with tp.graph_capture(B):  # nested on the same stream: nothing moves
            assert work[(0, B)] == "set"
    assert work == {(0, A): "set", (1, A): "other device"}
    with tp.graph_capture():  # C (real capture)
        assert (0, C) in work and (0, A) not in work
    assert work == {(0, A): "set", (1, A): "other device"}
    assert [sorted(s) for s in seen] == [[(0, B), (1, A)], [(0, C), (1, A)]]


def test_register_twice_lends_once(monkeypatch, runtime_state):
    tp = _install_fake_tp(monkeypatch, [B])
    work = {(0, A): "set"}
    swap = _swap_dict(work)
    akr_runtime.register_stream_workspaces(swap)
    akr_runtime.register_stream_workspaces(swap)
    akr_runtime.install_capture_hook()
    with tp.graph_capture():
        assert work == {(0, B): "set"}
    assert work == {(0, A): "set"}


def test_capture_returns_the_set_on_error(monkeypatch, runtime_state):
    tp = _install_fake_tp(monkeypatch, [B])
    work = {(0, A): "set"}
    akr_runtime.register_stream_workspaces(_swap_dict(work))
    akr_runtime.install_capture_hook()
    with pytest.raises(RuntimeError), tp.graph_capture():
        raise RuntimeError("capture failed")
    assert work == {(0, A): "set"}


def test_moe_swap_takes_every_entry(monkeypatch):
    calls = []
    moe = types.SimpleNamespace(
        swap_stream_workspaces=lambda *a: calls.append(("moe", a))
    )
    gemm = types.SimpleNamespace(
        latent_down_swap_stream_workspaces=lambda *a: calls.append(("down", a))
    )
    monkeypatch.setattr(akr_moe, "_moe", lambda: moe)
    monkeypatch.setattr(akr_moe, "_gemm", lambda: gemm)
    akr_moe.swap_stream_workspaces(0, A, B)
    assert calls == [("moe", (0, A, B)), ("down", (0, A, B))]


def _config(monkeypatch, seqs: int, k: int | None):
    from vllm import config

    spec = (
        None
        if k is None
        else types.SimpleNamespace(num_speculative_tokens=k, parallel_drafting=False)
    )
    cfg = types.SimpleNamespace(
        scheduler_config=types.SimpleNamespace(max_num_seqs=seqs),
        speculative_config=spec,
        use_v2_model_runner=True,
    )
    monkeypatch.setattr(config, "get_current_vllm_config_or_none", lambda: cfg)


@pytest.mark.parametrize(
    "seqs,k,shape",
    [
        (48, 4, (240, 5)),
        (8, 4, (40, 5)),
        (1, 7, (8, 8)),
        (64, 4, (256, 5)),
        (4, None, (4, 1)),
    ],
)
def test_decode_shape(monkeypatch, seqs, k, shape):
    _config(monkeypatch, seqs, k)
    assert akr_dsa._decode_shape() == shape


@pytest.mark.parametrize("ms", [True, False])
@pytest.mark.parametrize("reason", [None, "max_query_len 8 (the kernel takes 1..6)"])
def test_score_topk_scratch(monkeypatch, ms, reason):
    ix = pytest.importorskip("amd_kernels.indexer.mxfp4_indexer_common")

    import vllm.utils.torch_utils as tu

    _config(monkeypatch, 48, 4)
    calls = []
    monkeypatch.setattr(
        ix, "paged_score_topk_preallocate", lambda *a, **kw: calls.append((a, kw))
    )
    monkeypatch.setattr(tu, "aux_stream", lambda: "aux")
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: "load")
    monkeypatch.setattr(akr_dsa, "_decode_reason", lambda *a: reason)
    w = types.SimpleNamespace(ms_comm=object() if ms else None)
    akr_dsa.preallocate_score_topk_scratch(w, "cuda:0", "cache", "out")
    ((args, kw),) = calls
    assert kw["cache"] == "cache" and kw["out_indices"] == "out"
    if reason is not None:  # no decode batch is the kernel's: no set, the warm-up's go
        assert args == ("cuda:0", []) and kw["release_others"] and kw["rows"] == 240
    elif ms:  # only the aux stream's set, for 240 rows
        assert (
            args == ("cuda:0", ["aux"]) and kw["rows"] == 240 and kw["release_others"]
        )
    else:
        assert args == ("cuda:0", ["load", "aux"]) and kw["rows"] == 240
        assert not kw["release_others"]


def test_q_buffer_rows(monkeypatch):
    _config(monkeypatch, 8, 4)
    monkeypatch.setattr(akr_dsa, "_q_buffer", None)
    monkeypatch.setattr(
        akr_dsa, "_dims", lambda w: dict(heads=96, kv_lora=512, rope=64)
    )
    w = types.SimpleNamespace(
        fused_qkv_a_proj=types.SimpleNamespace(weight=torch.empty(1, device="meta"))
    )
    assert tuple(akr_dsa.ensure_q_buffer(w).shape) == (40, 96, 576)


class _Comm:
    def __init__(self):
        self.calls = []

    def warmup(self):
        pass

    def gemm_exchange(self, x, w, idx, counts, buf):
        self.calls.append((x.shape[0], counts is not None, buf))


@pytest.mark.parametrize("owner", [False, True])
def test_lmhead_warm_up_modes(monkeypatch, owner):
    from vllm.models.kimi_k3.amd import akr_nonlayer

    monkeypatch.setattr(
        akr_nonlayer,
        "tp_group",
        lambda: types.SimpleNamespace(rank_in_group=0, world_size=8),
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda *a, **k: None)
    comm = _Comm()
    ex = akr_lmhead._Exchange("draft", comm, 40, torch.device("cpu"))
    ex.warm_up(torch.zeros(4, 16, dtype=torch.bfloat16), owner=owner)
    assert [c for c in comm.calls if not c[1]] == [
        (8, False, 0),
        (32, False, 0),
        (40, False, 0),
    ]
    assert [c[0] for c in comm.calls if c[1]] == ([8, 40] if owner else [])
    assert (
        akr_lmhead._buffers_for("draft") == 1 and akr_lmhead._buffers_for("target") == 2
    )


def test_drop_stock_staging():
    spec = types.SimpleNamespace(_aux_staging=torch.zeros(40, 5 * 16))
    akr_draft._drop_stock_staging(spec)
    assert spec._aux_staging is not None and tuple(spec._aux_staging.shape) == (0, 80)
    spec = types.SimpleNamespace(_aux_staging=None)
    akr_draft._drop_stock_staging(spec)
    assert spec._aux_staging is None
