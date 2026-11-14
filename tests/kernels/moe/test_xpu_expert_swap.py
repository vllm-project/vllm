# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The expert swap of per-expert placement (--cpu-offload-expert-swap), on CPU.

The side stream is simulated: a copy issued on it poisons its destination at
once and lands only when the test lets it, so a forward that reads a row still
being written sees the poison. Each forward also stages the whole cold block
first, as --cpu-offload-stage-min-tokens does on the forward stream.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config.offload import OffloadConfig, UVAOffloadConfig
from vllm.model_executor.layers.fused_moe import xpu_expert_placement as xp
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQMoEMethod

G, H, LAYERS, SPARE, TICKS = 32, 12, 4, 4, 120
KEYS = ("w13", "w2", "w13_scales", "w2_scales")
SHAPES = {"w13": (4, 8), "w2": (8, 2), "w13_scales": (2, 3), "w2_scales": (3, 2)}
ROW_BYTES = 4 * 8 + 8 * 2 + (2 * 3 + 3 * 2) * 2
POISON = 99


def _dtype(key):
    return torch.bfloat16 if "scales" in key else torch.uint8


def _tag(g, key):
    """Expert ``g``'s bytes: all ``g + 1`` (spare rows are zero)."""
    return torch.full(SHAPES[key], g + 1, dtype=_dtype(key))


class _Event:
    def __init__(self, side, mark):
        self.side, self.mark = side, mark

    def query(self):
        return self.side.done >= self.mark


def _in_mode(mode, op):
    with torch.inference_mode(mode):
        return op()


class _Side:
    """Copies run in issue order, when ``land`` lets them."""

    def __init__(self):
        self.ops: list = []
        self.issued = self.done = 0
        self.bytes = 0

    def wait_event(self, event):
        assert event.query(), "the side stream waited on the forward: a deadlock"

    def copy(self, dst, src):
        dst.fill_(POISON)  # a copy in flight leaves the row torn
        # On the device the copy is enqueued here: it lands in the issuer's mode.
        mode = torch.is_inference_mode_enabled()
        self.ops.append(lambda: _in_mode(mode, lambda: dst.copy_(src)))
        self.issued += 1
        self.bytes += dst.numel() * dst.element_size()

    def land(self, n=None):
        n = len(self.ops) if n is None else n
        for op in self.ops[:n]:
            op()
        self.ops = self.ops[n:]
        self.done += n


class _Fwd:
    def wait_event(self, event):
        assert event.query(), "the forward stream waited on a copy in flight"


@pytest.fixture
def sim(monkeypatch):
    side, fwd = _Side(), _Fwd()

    def record(stream=None):
        return _Event(side, side.issued if stream is side else 0)

    monkeypatch.setattr(xp, "current_platform", SimpleNamespace(is_xpu=lambda: True))
    monkeypatch.setattr(xp, "_capturing", lambda: False)
    monkeypatch.setattr(xp, "_record_event", record)
    monkeypatch.setattr(xp, "_current_stream", lambda: fwd)
    monkeypatch.setattr(xp, "_side_stream", lambda device: side)
    monkeypatch.setattr(xp, "_stream_ctx", lambda stream: _Null())
    monkeypatch.setattr(xp, "_side_copy", side.copy)
    monkeypatch.setattr(xp, "_synchronize", lambda: None)
    monkeypatch.setattr(
        xp, "_pinned_empty", lambda shape, dtype: torch.zeros(shape, dtype=dtype)
    )
    monkeypatch.setattr(
        xp._ARENA, "empty", lambda shape, dtype: torch.zeros(shape, dtype=dtype)
    )
    monkeypatch.setattr(xp, "_FOLD_S", 0.0)
    monkeypatch.setattr(xp, "_SWAP_SPARE", SPARE)
    monkeypatch.setattr(xp, "_RECORDER", xp._Recorder())
    monkeypatch.setattr(xp, "_SWAPPER", xp._Swapper())
    xp._PLACEMENTS.clear()
    xp._LIVE[0] = True
    yield SimpleNamespace(side=side, monkeypatch=monkeypatch)
    xp._PLACEMENTS.clear()
    xp._LIVE[0] = False


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _build(seed=0, hot_inference=False):
    """LAYERS placements, all G experts local, H hot. ``hot_inference``: the
    hot tensors are inference tensors, as when the first forward builds the
    kernels under the runner's inference mode."""
    rng = np.random.default_rng(seed)
    placements = []
    for layer in range(LAYERS):
        hot = sorted(rng.choice(G, H, replace=False).tolist())
        p = xp.ExpertPlacement(
            f"model.layers.{layer}.mlp.experts", list(range(G)), list(range(G)), G, hot
        )
        p.swap = p.count = True
        p.spare = SPARE
        with torch.inference_mode(hot_inference):
            hot_t = {k: torch.stack([_tag(g, k) for g in p.hot_local]) for k in KEYS}
        p.cold = {
            k: xp._pin(torch.stack([_tag(g, k) for g in p.cold_local]), SPARE)
            for k in KEYS
        }
        hot_map = torch.full((G,), -1, dtype=torch.int32)
        cold_map = torch.full((G,), -1, dtype=torch.int32)
        for local_id, (is_cold, row) in p.slot.items():
            (cold_map if is_cold else hot_map)[local_id] = row
        p.set_maps(hot_map, cold_map, torch.device("cpu"))
        p.route_counts = torch.zeros(2, G + 1, dtype=torch.int64)
        p._impls = (
            SimpleNamespace(
                w13=hot_t["w13"].view(torch.int8),
                w2=hot_t["w2"].view(torch.int8),
                gemm1_wei_scales=hot_t["w13_scales"],
                gemm2_wei_scales=hot_t["w2_scales"],
            ),
            None,
        )
        xp._PLACEMENTS[layer] = p
        placements.append(p)
    return placements


def _target(tick):
    return np.random.default_rng(1 if tick < TICKS // 2 else 2).permutation(G)


def _traffic(placements, rng, tick):
    """Zipf-like decode routes over one permutation, another after TICKS // 2."""
    p = 1.0 / (np.arange(G) + 1.0)
    p = p[np.argsort(_target(tick))]
    p /= p.sum()
    for pl in placements:
        pl.route_counts[1, :G] += torch.from_numpy(rng.multinomial(200, p))
        pl.route_counts[0, :G] += torch.from_numpy(rng.multinomial(50, p))


def _forward(placements):
    """One forward: stage each cold block, then read every expert the maps
    route to, hot from VRAM rows, cold from the staged copy."""
    for p in placements:
        staged = {k: t.clone() for k, t in p.cold.items()}
        hm, cm = p.hot_map.numpy(), p.cold_map.numpy()
        assert len(set(hm[hm >= 0].tolist())) == int((hm >= 0).sum())
        assert len(set(cm[cm >= 0].tolist())) == int((cm >= 0).sum())
        for g in range(G):
            assert (hm[g] >= 0) != (cm[g] >= 0), f"{p.layer_name}: expert {g}"
            for k in KEYS:
                row = p.hot_tensors[k][hm[g]] if hm[g] >= 0 else staged[k][cm[g]]
                assert torch.equal(row.view(_dtype(k)), _tag(g, k)), (
                    f"{p.layer_name}: expert {g} {k} read torn or foreign bytes"
                )


def _run(sim, budget_rows=None, ticks=TICKS, hot_inference=False):
    if budget_rows is not None:
        sim.monkeypatch.setattr(xp, "_SWAP_STEP_BYTES", budget_rows * ROW_BYTES)
    placements = _build(hot_inference=hot_inference)
    rng = np.random.default_rng(100)
    ptrs = [(p.hot_map.data_ptr(), p.cold_map.data_ptr()) for p in placements]
    most = 0
    for tick in range(ticks):
        _traffic(placements, rng, tick)
        before = sim.side.bytes
        xp.step()
        most = max(most, sim.side.bytes - before)
        assert sim.side.bytes - before == xp._SWAPPER.step_bytes
        _forward(placements)
        sim.side.land(int(rng.integers(0, len(sim.side.ops) + 1)))
    assert ptrs == [(p.hot_map.data_ptr(), p.cold_map.data_ptr()) for p in placements]
    return placements, most


def test_every_expert_stays_routable_and_the_hot_set_follows_a_shift(sim):
    placements, _ = _run(sim)
    assert xp._SWAPPER.total > 0
    want = set(_target(TICKS).tolist()[:H])
    for p in placements:
        got = set(np.nonzero(p.hot_map.numpy() >= 0)[0].tolist())
        assert len(want & got) >= 0.75 * H, sorted(got)
        # host mirrors, owners and the device maps agree
        assert np.array_equal(p.hot_map.numpy(), p.hot_map_np)
        assert np.array_equal(p.cold_map.numpy(), p.cold_map_np)
        for g in range(G):
            if p.hot_map_np[g] >= 0:
                assert p.hot_owner[p.hot_map_np[g]] == g
            else:
                assert p.cold_owner[p.cold_map_np[g]] == g


def test_swap_with_kernels_built_under_inference_mode(sim):
    # The first forward builds the kernels, and so the hot tensors, under the
    # runner's inference mode; the worker calls step() outside it.
    placements, _ = _run(sim, hot_inference=True)
    assert xp._SWAPPER.total > 0
    for p in placements:
        assert np.array_equal(p.hot_map.numpy(), p.hot_map_np)


def test_bytes_per_step_stay_within_the_budget(sim):
    # Three rows a step: a fold's 16 exchanges take six chunks, so give it time.
    placements, most = _run(sim, budget_rows=3, ticks=3 * TICKS)
    assert 0 < most <= 3 * ROW_BYTES
    assert xp._SWAPPER.total > 0
    want = set(_target(TICKS).tolist()[:H])
    for p in placements:
        got = set(np.nonzero(p.hot_map.numpy() >= 0)[0].tolist())
        assert len(want & got) >= 0.75 * H


def test_no_swap_while_capturing(sim):
    placements = _build()
    _traffic(placements, np.random.default_rng(0), TICKS)
    sim.monkeypatch.setattr(xp, "_capturing", lambda: True)
    for _ in range(6):
        xp.step()
    assert sim.side.issued == 0 and not xp._SWAPPER.ready
    assert not xp._RECORDER.ready


def test_staging_during_a_swap_reads_no_row_in_flight(sim):
    """A copy1 in flight writes a spare row the maps don't reference: a staged
    copy taken meanwhile carries the torn row, and the forward never reads it.
    Pointing a map at that row before the copy lands would read the poison."""
    placements = _build()
    rng = np.random.default_rng(3)
    for tick in range(4):
        _traffic(placements, rng, TICKS)  # the second distribution: swaps due
        xp.step()
        if sim.side.ops:
            break
    assert xp._SWAPPER.phase == "copy1" and sim.side.ops
    k, _a, b, _i, _j, s = xp._SWAPPER.pending[0]
    p = placements[k]
    assert p.cold_owner[s] == -1 and not (p.cold_map_np == s).any()
    assert torch.equal(p.cold["w2"][s], torch.full_like(p.cold["w2"][s], POISON))
    _forward(placements)  # staged while torn: fine, nothing routes there
    xp.step()  # copy1 not landed: no flip
    assert xp._SWAPPER.phase == "copy1"
    _forward(placements)
    # Control: the checker catches a map that points at the torn row.
    p.cold_map[b] = s
    p.hot_map[b] = -1
    with pytest.raises(AssertionError, match="torn or foreign"):
        _forward(placements)


def test_swap_turns_itself_off_on_a_shape_mismatch(sim):
    placements = _build()
    placements[0].cold["w2"] = placements[0].cold["w2"][:, :1]
    for _ in range(3):
        xp.step()
    assert xp._SWAPPER.disabled and not xp._SWAPPER.ready
    assert sim.side.issued == 0


def test_spare_rows_are_zero_and_counted_by_the_cold_kernel(sim):
    p = _build()[0]
    assert p.cold_rows == G - H + SPARE
    assert p.cold["w13"].shape[0] == p.cold_rows
    assert not p.cold["w13"][G - H :].any()
    assert sorted(p.spare_free) == list(range(G - H, G - H + SPARE))


def test_repack_appends_zeroed_spare_rows(monkeypatch):
    pytest.importorskip("vllm_xpu_kernels.fused_moe_interface")
    from vllm_xpu_kernels.fused_moe_interface import implement_zp

    monkeypatch.setattr(
        xp._ARENA, "empty", lambda shape, dtype: torch.full(shape, 7, dtype=dtype)
    )
    q = torch.randint(0, 2**31 - 1, (20, 4, 6), dtype=torch.int32)
    out = xp._repack_int4_pinned(q, torch.device("cpu"), extra=2)
    assert out.shape == (22, 6, 16)
    want = implement_zp(q.transpose(1, 2).contiguous().view(torch.uint8))
    assert torch.equal(out[:20], want) and not out[20:].any()


class _GPTQ(AutoGPTQMoEMethod):
    def __init__(self):
        self.quant_config = SimpleNamespace(weight_bits=4, pack_factor=8, group_size=64)
        self.moe = SimpleNamespace(w13_num_shards=2)


def test_swap_alone_selects_placement_and_counts(monkeypatch, tmp_path):
    monkeypatch.setattr(xp, "current_platform", SimpleNamespace(is_xpu=lambda: True))
    monkeypatch.setattr("vllm.utils.platform_utils.is_uva_available", lambda: True)
    monkeypatch.delenv("VLLM_WEIGHT_OFFLOADING_DISABLE_UVA", raising=False)
    assert UVAOffloadConfig().cpu_offload_expert_swap is False
    assert not xp.places_experts(OffloadConfig(uva=UVAOffloadConfig(cpu_offload_gb=1)))
    rankings = tmp_path / "rankings.json"
    rankings.write_text(json.dumps({"3": [5, 4]}))
    config = OffloadConfig(
        uva=UVAOffloadConfig(
            cpu_offload_gb=1.0,
            cpu_offload_expert_swap=True,
            cpu_offload_expert_rankings=str(rankings),
        )
    )
    assert xp.places_experts(config)
    monkeypatch.setattr(xp, "_offload_config", lambda: config)
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config",
        lambda: SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(num_hidden_layers=48)
            )
        ),
    )
    xp.load_rankings.cache_clear()
    params = {
        "hidden_size": 256,
        "intermediate_size_per_partition": 128,
        "params_dtype": torch.bfloat16,
    }
    p = xp.maybe_place(
        "model.layers.3.mlp.experts",
        SimpleNamespace(num_experts=8),
        SimpleNamespace(map_global_to_local=lambda g: g),
        _GPTQ(),
        params,
    )
    xp.load_rankings.cache_clear()
    assert p is not None and p.swap and p.count and p.state_path is None
    assert p.hot_local == [5]  # 1 GiB buys 7 cold of 8; the ranked first stays
