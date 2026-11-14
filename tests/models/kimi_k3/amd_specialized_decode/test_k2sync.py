# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``VLLM_K31_BEAT_K2SYNC`` (``mono.common.plan.k2sync``): the mono kernels'
cheap spin-wait polls (health / deadline every ``sync.CHECK_POLLS`` polls), the
single-word sentinel poll's one load and the router's bounded candidate rank
give the default build's bits.

- K1 (one GPU): flag off vs on, every output and state bit for bit.
- K2 (eight GPUs): flag off vs on at every width, both layer parities, back
  to back on re-armed sentinel regions; a graph replay; then a stalled rank
  still fails the step without a hang (the deadline checked every 64 polls).

Every mono module keeps its launches in an in-process ``_BUILDS`` keyed by its
build alone, so a flip clears them all (``_set_flag``).
"""

import os
import sys

import pytest
import ray
import torch
import torch.distributed as dist

from tests.models.kimi_k3.amd_specialized_decode.test_numerics import (
    K1_SHAPES,
    K2_CASES,
    STALLED,
    _init_worker,
    _k2_inputs,
    _k2_launch,
    _kda_inputs,
    _moe_weights,
    bf,
    needs_gfx950,
)
from tests.utils import multi_gpu_test, multi_process_parallel

FLAG = "VLLM_K31_BEAT_K2SYNC"
MONO_PKG = "vllm.models.kimi_k3.amd.specialized_decode.mono"
MONO = (
    "layer",
    "attention.kda",
    "attention.mla",
    "attention.mla_back",
    "attention.mla_attn",
    "attention.dsa_pre",
    "attention.dsa_attn",
)


def _clear_builds() -> None:
    import importlib

    for m in MONO:
        importlib.import_module(f"{MONO_PKG}.{m}")._BUILDS.clear()


def _set_flag(on: bool) -> None:
    from vllm.models.kimi_k3.amd.specialized_decode.mono.common.plan import k2sync

    os.environ[FLAG] = "1" if on else "0"
    _clear_builds()  # the flag is in no build dataclass, only in the disk key
    assert k2sync() == on


@pytest.fixture(autouse=True)
def _fresh_builds():
    """No launch traced under one flag value outlives its test."""
    yield
    if any(f"{MONO_PKG}.{m}" in sys.modules for m in MONO):
        _clear_builds()


def _k1_run(key, w):
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
        PROJ,
        kda_pre,
        scratch_bytes,
    )
    from vllm.models.kimi_k3.amd.specialized_decode.mono.common.plan import HEALTH_WORDS

    S = key.tokens
    prefix, blocks = w["prefix"].clone(), w["blocks"].clone()
    conv_state, rstate = w["conv_state"].clone(), w["rstate"].clone()
    core = torch.zeros(S, PROJ, dtype=bf, device="cuda")
    health = torch.zeros(HEALTH_WORDS, dtype=torch.int32, device="cuda")
    flags = torch.zeros(1, dtype=torch.int32, device="cuda")
    kda_pre(
        key,
        prefix=prefix,
        delta=w["delta"],
        blocks=blocks,
        ares_nw=w["ares_nw"],
        ares_qk=w["ares_qk"],
        in_nw=w["in_nw"],
        w_in=w["w_in"],
        w_fb=w["w_fb"],
        conv_w=w["conv_w"],
        conv_state=conv_state,
        a_log=w["a_log"],
        dt_bias=w["dt_bias"],
        on_w=w["on_w"],
        rstate=rstate,
        st_idx=w["st_idx"],
        num_acc=w["num_acc"],
        core_out=core,
        scratch=torch.zeros(scratch_bytes(key), dtype=torch.uint8, device="cuda"),
        layer=0,
        epoch=torch.ones(1, dtype=torch.int32, device="cuda"),
        health=health,
        alarm=flags.data_ptr(),
    )
    torch.accelerator.synchronize()
    assert not health.any() and not flags.any()
    return [prefix, blocks, conv_state, rstate, core]


@needs_gfx950
@pytest.mark.parametrize("S, L", K1_SHAPES[:3])
@pytest.mark.parametrize("write", [False, True])
def test_k1_k2sync_bit_exact(
    monkeypatch: pytest.MonkeyPatch, S: int, L: int, write: bool
) -> None:
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import kda
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
        KdaPreBuild,
    )

    monkeypatch.setenv(FLAG, "0")
    nb = 4
    w = _kda_inputs(nb, 3, True, S, L)
    key = KdaPreBuild(
        tokens=S,
        qlen=L,
        nblocks=nb,
        delta=True,
        state_len=w["conv_state"].size(-1),
        write_idx=nb if write else -1,
    )
    _set_flag(False)
    want = _k1_run(key, w)
    off = kda._BUILDS[key]
    _set_flag(True)
    got = _k1_run(key, w)
    assert kda._BUILDS[key] is not off, "the flag-on run reused the off launch"
    for name, a, b in zip(("prefix", "blocks", "conv", "rstate", "core"), got, want):
        assert torch.equal(a, b), name


@ray.remote(num_gpus=1, max_calls=1)
def _k2sync_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    device = _init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port)

    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import HIDDEN
    from vllm.models.kimi_k3.amd.specialized_decode.runtime.runner import MonoRunner

    runner = MonoRunner(device)
    mw = _moe_weights(rank, device)

    def run(w, S, reset, layer):
        runner.step_begin()
        prefix = w["prefix"].clone()
        out = torch.empty(S, HIDDEN, dtype=bf, device=device)
        _k2_launch(runner, w, mw, prefix, out, reset, layer=layer)
        torch.accelerator.synchronize()
        return out, prefix

    for S, reset in K2_CASES:
        ws = [_k2_inputs(rank, device, S, seed=i) for i in range(2)]
        for layer in (3, 4):
            _set_flag(False)
            want = [run(w, S, reset, layer) for w in ws]
            assert not torch.equal(want[0][0], want[1][0]), (S, reset, layer)
            _set_flag(True)
            for i, w in enumerate(ws):
                out, prefix = run(w, S, reset, layer)
                assert torch.equal(out, want[i][0]), (S, reset, layer, i)
                assert torch.equal(prefix, want[i][1]), (S, reset, layer, i)
    assert not runner.health.any() and not runner.flags.any()

    # a graph of one step replayed under the flag: every replay the first's bits
    S = 8
    w = _k2_inputs(rank, device, S)
    prefix = torch.empty_like(w["prefix"])
    out = torch.empty(S, HIDDEN, dtype=bf, device=device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner.step_begin()
        _k2_launch(runner, w, mw, prefix, out, reset=True)
    graph.replay()
    torch.accelerator.synchronize()
    want_prefix, want_out = prefix.clone(), out.clone()
    for _ in range(300):
        graph.replay()
    torch.accelerator.synchronize()
    assert torch.equal(prefix, want_prefix) and torch.equal(out, want_out)
    assert not runner.health.any() and not runner.flags.any()
    outs = [torch.empty_like(out) for _ in range(tp_size)]
    dist.all_gather(outs, out)
    assert all(torch.equal(outs[0], o) for o in outs), "ranks disagree"


@needs_gfx950
@multi_gpu_test(num_gpus=8)
def test_k2_tp8_k2sync_bit_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    multi_process_parallel(monkeypatch, 8, 1, _k2sync_worker)


@ray.remote(num_gpus=1, max_calls=1)
def _k2sync_fault_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    device = _init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port)

    from vllm.models.kimi_k3.amd.specialized_decode.mono import layer as k2
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import HIDDEN
    from vllm.models.kimi_k3.amd.specialized_decode.mono.common.debug import (
        given_up_waits,
    )
    from vllm.models.kimi_k3.amd.specialized_decode.mono.common.sync import WAIT_TICKS
    from vllm.models.kimi_k3.amd.specialized_decode.runtime.runner import MonoRunner

    _set_flag(True)
    runner = MonoRunner(device)
    runner.fault = {
        "fault_rank": STALLED,
        "fault_epoch": 2,
        "fault_layer": 128 + 3,
        "fault_ticks": WAIT_TICKS * 3 // 2,
    }
    mw = _moe_weights(rank, device)
    w = _k2_inputs(rank, device)
    out = torch.empty(8, HIDDEN, dtype=bf, device=device)
    runner.step_begin()
    _k2_launch(runner, w, mw, w["prefix"].clone(), out, reset=True)
    torch.accelerator.synchronize()
    assert not runner.flags.any()

    runner.step_begin()
    _k2_launch(runner, w, mw, w["prefix"].clone(), out, reset=True)
    torch.accelerator.synchronize()
    waits = given_up_waits(runner.health.cpu(), k2.REGIONS)
    gave_up = torch.tensor([int(bool(waits))], device=device)
    dist.all_reduce(gave_up)
    assert gave_up.item() > 0, "no rank's wait gave up"
    if rank == 0:
        assert runner.flags.any(), "the output rank's flags missed the failure"


@needs_gfx950
@multi_gpu_test(num_gpus=8)
def test_k2sync_stalled_rank_fails_without_a_hang(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    multi_process_parallel(monkeypatch, 8, 1, _k2sync_fault_worker)
