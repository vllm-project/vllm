# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-expert offload placement on XPU: the plan on CPU, the split on a card."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config.offload import OffloadConfig, PrefetchOffloadConfig, UVAOffloadConfig
from vllm.model_executor.layers.fused_moe import xpu_expert_placement as xp
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQMoEMethod

E, EP = 16, 2
NAME = "model.language_model.layers.3.mlp.experts"
HIDDEN, INTER, GROUP = 256, 128, 64


class _Linear:
    """ExpertMapManager stand-in: linear placement, E // EP experts per rank."""

    def __init__(self, rank: int):
        self.rank = rank

    def map_global_to_local(self, g: int) -> int:
        per = E // EP
        return (
            g - self.rank * per if self.rank * per <= g < (self.rank + 1) * per else -1
        )


class _GPTQ(AutoGPTQMoEMethod):
    """The real method's create_weights without its backend selection."""

    def __init__(self, bits: int = 4, group_size: int = GROUP):
        self.quant_config = SimpleNamespace(
            weight_bits=bits, pack_factor=32 // bits, group_size=group_size
        )
        self.moe = SimpleNamespace(w13_num_shards=2)


class _GPTQ8(_GPTQ):
    def __init__(self):
        super().__init__(bits=8)


class OtherMoEMethod:
    pass


def _params() -> dict:
    return {
        "hidden_size": HIDDEN,
        "intermediate_size_per_partition": INTER,
        "params_dtype": torch.bfloat16,
    }


def _config(tmp_path, gb=1.0, rankings=True, state=None, **kwargs) -> OffloadConfig:
    path = None
    if rankings:
        path = tmp_path / "rankings.json"
        # layer 3: hottest first; experts 1, 6 and 12 never observed
        path.write_text(json.dumps({"3": [9, 2, 15, 0, 7, 3, 10, 4, 5, 8, 11, 13, 14]}))
    uva = UVAOffloadConfig(
        cpu_offload_gb=gb,
        cpu_offload_expert_rankings=str(path) if path else None,
        cpu_offload_expert_state=str(state) if state else None,
    )
    return OffloadConfig(uva=uva, **kwargs)


def _reset():
    xp.load_rankings.cache_clear()
    xp.load_state.cache_clear()
    xp._PLACEMENTS.clear()
    xp._LIVE[0] = False
    xp._WARMUP[0] = False
    xp._RECORDER = xp._Recorder()


@pytest.fixture
def xpu(monkeypatch):
    monkeypatch.setattr(xp, "current_platform", SimpleNamespace(is_xpu=lambda: True))
    monkeypatch.setattr(
        "vllm.utils.platform_utils.is_uva_available", lambda: True, raising=True
    )
    monkeypatch.delenv("VLLM_WEIGHT_OFFLOADING_DISABLE_UVA", raising=False)
    monkeypatch.setattr(xp, "_capturing", lambda: False)
    _reset()
    yield monkeypatch
    _reset()


def _place(monkeypatch, config, rank=0, name=NAME, hot=3, method=_GPTQ):
    monkeypatch.setattr(xp, "_offload_config", lambda: config)
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config",
        lambda: SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(num_hidden_layers=48)
            )
        ),
    )
    # A budget that buys E // EP - hot cold experts per layer.
    quant_method = method()
    if isinstance(quant_method, _GPTQ):
        bytes_each = xp.expert_bytes(quant_method, **_params())
        cold = E // EP - hot
        config.uva.cpu_offload_gb = (48 * cold * bytes_each + bytes_each // 2) / 2**30
    return xp.maybe_place(
        name,
        SimpleNamespace(num_experts=E),
        _Linear(rank),
        quant_method,
        _params(),
    )


def test_selected_only_with_rankings_and_a_budget_on_xpu(xpu, tmp_path):
    assert xp.places_experts(_config(tmp_path))
    assert not xp.places_experts(_config(tmp_path, gb=0))
    assert not xp.places_experts(_config(tmp_path, rankings=False))
    assert not xp.places_experts(
        _config(
            tmp_path,
            prefetch=PrefetchOffloadConfig(offload_group_size=4),
        )
    )
    xpu.setattr(xp, "current_platform", SimpleNamespace(is_xpu=lambda: False))
    assert not xp.places_experts(_config(tmp_path))


def test_rankings_refused_when_uva_is_disabled(xpu, tmp_path):
    from vllm.model_executor.offloader import create_offloader
    from vllm.model_executor.offloader.uva import UVAOffloader

    xpu.setenv("VLLM_WEIGHT_OFFLOADING_DISABLE_UVA", "1")
    with pytest.raises(ValueError, match="VLLM_WEIGHT_OFFLOADING_DISABLE_UVA"):
        create_offloader(_config(tmp_path))
    # Without rankings the regular offloader takes it, with its copy fallback.
    offloader = create_offloader(_config(tmp_path, rankings=False))
    assert isinstance(offloader, UVAOffloader) and not offloader.uva_offloading


def test_rankings_refused_when_uva_is_unavailable(xpu, tmp_path):
    xpu.setattr("vllm.utils.platform_utils.is_uva_available", lambda: False)
    with pytest.raises(ValueError, match="unavailable"):
        xp.places_experts(_config(tmp_path))
    # Not selected (no budget): nothing to refuse.
    assert not xp.places_experts(_config(tmp_path, gb=0))


def test_expert_bytes_match_create_weights():
    """The arithmetic equals the qweight and scales create_weights builds."""
    layer = torch.nn.Module()
    with torch.device("meta"):
        AutoGPTQMoEMethod.create_weights(
            _GPTQ(), layer=layer, num_experts=1, **_params()
        )
    built = sum(
        p.numel() * p.element_size()
        for name, p in layer.named_parameters()
        if name in ("w13_qweight", "w13_scales", "w2_qweight", "w2_scales")
    )
    assert xp.expert_bytes(_GPTQ(), **_params()) == built
    # w13 [256/8, 256] + w2 [128/8, 256] int32; scales [4, 256] + [2, 256] bf16
    assert built == (32 * 256 + 16 * 256) * 4 + (4 * 256 + 2 * 256) * 2


def test_budget_keeps_one_hot():
    assert xp.budget_cold_count(1, 48, 2**20, 256) == 21
    assert xp.budget_cold_count(1000, 48, 2**20, 256) == 255
    assert xp.budget_cold_count(0.0001, 48, 2**20, 256) == 0


def test_hot_set_follows_rankings_per_rank(xpu, tmp_path):
    config = _config(tmp_path)
    p0, p1 = _place(xpu, config, 0), _place(xpu, config, 1)
    # rank 0 owns 0..7: ranked 2, 0, 7, 3, 4, 5, so hot 2, 0, 7 (local ids)
    assert p0.hot_local == [0, 2, 7]
    assert p0.cold_local == [1, 3, 4, 5, 6]
    # rank 1 owns 8..15: ranked 9, 15, 10, so local 1, 7, 2
    assert p1.hot_local == [1, 2, 7]


def test_unranked_experts_and_layers_fall_back_to_id_order(xpu, tmp_path):
    p0 = _place(xpu, _config(tmp_path), hot=7)  # 2, 0, 7, 3, 4, 5, then 1, 6
    assert p0.hot_local == [0, 1, 2, 3, 4, 5, 7]
    assert p0.cold_local == [6]
    other = "model.language_model.layers.5.mlp.experts"  # not in the file
    assert _place(xpu, _config(tmp_path), name=other).hot_local == [0, 1, 2]


def test_hot_experts_ignores_foreign_and_repeated_ids():
    assert xp.hot_experts([5, 99, 5, 3], [3, 4, 5, 6], 3) == [5, 3, 4]


def test_rows_cover_local_experts_once(xpu, tmp_path):
    p = _place(xpu, _config(tmp_path))
    hot = sorted(s for c, s in p.slot.values() if not c)
    cold = sorted(s for c, s in p.slot.values() if c)
    assert hot == list(range(p.num_hot)) and cold == list(range(p.num_cold))
    assert sorted(p.slot) == list(range(E // EP))


def test_no_placement_outside_decoder_layers_or_for_other_methods(xpu, tmp_path):
    config = _config(tmp_path)
    assert _place(xpu, config, name="mtp.layers.0.mlp.experts") is None
    assert _place(xpu, config, method=OtherMoEMethod) is None
    assert _place(xpu, config, method=_GPTQ8) is None


def test_loader_routes_cold_experts_to_twins(xpu, tmp_path):
    p = _place(xpu, _config(tmp_path))
    layer = torch.nn.Module()
    w = torch.nn.Parameter(
        torch.zeros(p.num_hot, 4, 6, dtype=torch.int32), requires_grad=False
    )
    w.is_transposed = True
    layer.register_parameter("w13_qweight", w)
    p.attach_twins(layer)
    twin = p.twins["w13_qweight"]
    assert twin.shape == (p.num_cold, 4, 6) and twin.device.type == "cpu"
    assert twin.is_transposed is True
    param, row = p.route(w, 2)  # local 2 is hot, row 1
    assert param is w and row == 1
    param, row = p.route(w, 4)  # local 4 is cold, row 2
    assert param is twin and row == 2


def test_repack_matches_the_kernel_layout(xpu):
    pytest.importorskip("vllm_xpu_kernels.fused_moe_interface")
    from vllm_xpu_kernels.fused_moe_interface import implement_zp

    xpu.setattr(
        xp._ARENA, "empty", lambda shape, dtype: torch.empty(shape, dtype=dtype)
    )
    q = torch.randint(0, 2**31 - 1, (20, 4, 6), dtype=torch.int32)
    out = xp._repack_int4_pinned(q, torch.device("cpu"))
    assert out.shape == (20, 6, 16)
    want = implement_zp(q.transpose(1, 2).contiguous().view(torch.uint8))
    assert torch.equal(out, want)


def test_arena_packs_blocks_into_chunks(monkeypatch):
    monkeypatch.setattr(
        torch,
        "empty",
        lambda *a, pin_memory=False, **k: torch.zeros(*a, **k),
    )
    arena = xp._PinnedArena()
    arena.CHUNK = 1 << 16
    a = arena.empty((3, 1000), torch.uint8)
    b = arena.empty((5000,), torch.uint8)
    big = arena.empty((1 << 17,), torch.uint8)
    assert len(arena.chunks) == 2 and arena.total == (1 << 16) + (1 << 17)
    assert a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
    assert b.data_ptr() - a.data_ptr() == 4096  # 3,000 B rounded to 4 KiB
    assert big.numel() == 1 << 17


def test_remap_unrouted_uses_an_expert_this_rank_does_not_own():
    ids = torch.tensor([[-1, 3], [7, -1]])
    assert xp.remap_unrouted(ids, 16, 0).tolist() == [[15, 3], [7, 15]]
    assert xp.remap_unrouted(ids, 16, 1).tolist() == [[0, 3], [7, 0]]


# -- route counters and the learned state -----------------------------------------


def _counting(layer=3, state_path=None, boot_order=()):
    p = xp.ExpertPlacement(
        f"model.layers.{layer}.mlp.experts",
        list(range(E)),
        list(range(E)),
        E,
        [0, 1, 2],
    )
    p.count = True
    p.state_path = state_path
    p.boot_order = list(boot_order)
    p.route_counts = torch.zeros(2, E + 1, dtype=torch.int64)
    xp._PLACEMENTS[p.layer_idx] = p
    return p


class _Event:
    def __init__(self):
        self.done = False

    def query(self):
        return self.done


@pytest.fixture
def host(xpu):
    """Device calls stubbed: pinned buffers are plain, events complete on demand,
    host synchronizations are counted."""
    events: list[_Event] = []
    syncs: list[int] = []

    def record(stream=None):
        events.append(_Event())
        return events[-1]

    xpu.setattr(xp, "_record_event", record)
    xpu.setattr(xp, "_current_stream", lambda: None)
    xpu.setattr(
        xp, "_pinned_empty", lambda shape, dtype: torch.zeros(shape, dtype=dtype)
    )
    xpu.setattr(xp, "_synchronize", lambda: syncs.append(1))
    return SimpleNamespace(events=events, syncs=syncs)


def test_selected_by_the_state_alone(xpu, tmp_path):
    config = _config(tmp_path, rankings=False, state=tmp_path / "state.json")
    assert xp.places_experts(config)
    p = _place(xpu, config)  # no file yet: by id, counting
    assert p.hot_local == [0, 1, 2] and p.count
    assert p.state_path == str(tmp_path / "state.json")


def test_no_counters_without_a_state(xpu, tmp_path):
    p = _place(xpu, _config(tmp_path))
    assert not p.count and p.route_counts is None
    p.count_routes(torch.tensor([[1, 2]]))  # nothing to count into


def test_route_counters_on_synthetic_topk_ids(xpu):
    p = _counting()
    ids = torch.tensor([[1, 5], [5, -1], [15, 1]], dtype=torch.int32)
    p.count_routes(ids)  # before the first step: the memory profile
    assert not p.route_counts.any()
    xp._LIVE[0] = True
    p.count_routes(ids)
    want = torch.zeros(E + 1, dtype=torch.int64)
    want[[1, 5, 15, E]] = torch.tensor([2, 2, 1, 1])  # -1 in the last bucket
    assert torch.equal(p.route_counts[1], want)
    assert not p.route_counts[0].any()
    big = torch.randint(0, E, (xp._LARGE_STEP_TOKENS, 8), dtype=torch.int32)
    p.count_routes(big)  # a large step lands in the prefill row
    assert torch.equal(
        p.route_counts[0, :E], torch.bincount(big.flatten().long(), minlength=E)
    )
    assert int(p.route_counts[0].sum()) == big.numel()


def test_route_counters_are_recorded_into_captured_graphs(xpu):
    xpu.setattr(xp, "_capturing", lambda: True)
    p = _counting()
    p.count_routes(torch.tensor([[2, 3]]))  # captured before the first step
    assert p.route_counts[1, 2] == 1 and p.route_counts[1, 3] == 1


def test_counting_needs_no_host_sync_or_extra_buffer(xpu):
    p = _counting()
    xp._LIVE[0] = True
    buf = xp._ones(torch.device("cpu"))
    ones = xp._ones(torch.device("cpu"), 16)
    assert ones.data_ptr() == buf.data_ptr()  # a slice of one buffer
    p.count_routes(torch.zeros(4, 4, dtype=torch.int32))
    assert int(p.route_counts[1, 0]) == 16


def test_boot_order_learned_state_then_rankings_then_id(xpu, tmp_path):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"3": [6, 1, 12, 4], "5": [3, 7]}))
    config = _config(tmp_path, state=state)
    p0 = _place(xpu, config, 0)
    assert p0.hot_local == [1, 4, 6]  # the state, not the rankings' 2, 0, 7
    assert p0.boot_order == [6, 1, 12, 4] and p0.count
    # rank 1 owns 8..15: 12 from the state, then 8, 9 by id
    assert _place(xpu, config, 1).hot_local == [0, 1, 4]
    assert xp.boot_order(config.uva, 5) == ([3, 7], "state")
    state.write_text(json.dumps({"5": [3, 7]}))
    xp.load_state.cache_clear()
    # layer 3 is not in the state: the rankings place it
    assert _place(xpu, config, 0).hot_local == [0, 2, 7]
    assert xp.boot_order(config.uva, 9) == ([], "id")


def test_missing_state_falls_back_to_the_rankings(xpu, tmp_path):
    config = _config(tmp_path, state=tmp_path / "missing.json")
    assert _place(xpu, config).hot_local == [0, 2, 7]


@pytest.mark.parametrize("raw", ["{", "[]", '{"3": 7}', '{"x": [1]}', '{"3": ["a"]}'])
def test_malformed_state_falls_back_to_the_rankings(xpu, tmp_path, raw):
    state = tmp_path / "state.json"
    state.write_text(raw)
    assert _place(xpu, _config(tmp_path, state=state)).hot_local == [0, 2, 7]


def test_state_round_trip_in_the_rankings_format(host, tmp_path):
    path = tmp_path / "state.json"
    _counting(state_path=str(path), boot_order=[9, 2, 15])
    rec = xp._Recorder()
    assert rec._setup(0.0)
    delta = np.zeros((1, 2, E), np.int64)
    delta[0, 1, [4, 7]] = [30, 10]  # decode: 4, then 7
    delta[0, 0, 11] = 1000  # a prefill weighs a tenth of the decode routes
    for _ in range(xp._STATE_MIN_FOLDS - 1):
        rec.fold(delta)
    rec.fold(np.zeros_like(delta))  # idle: neither a fold nor a decay
    assert rec.folds == xp._STATE_MIN_FOLDS - 1
    assert not rec.maybe_save(1e9) and not path.exists()  # too young
    rec.fold(delta)
    assert rec.maybe_save(1e9)
    order = xp.load_rankings(str(path))[3]  # readable as a rankings file
    assert order[:3] == [4, 7, 11]
    assert order[3:6] == [9, 2, 15]  # unrouted experts keep the boot order
    assert sorted(order) == list(range(E))
    config = _config(tmp_path, rankings=False, state=path)
    assert xp.boot_order(config.uva, 3) == (order, "state")


def test_save_waits_for_its_period_and_the_writer(host, tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    _counting(state_path=str(path))
    rec = xp._Recorder()
    rec._setup(100.0)
    delta = np.zeros((1, 2, E), np.int64)
    delta[0, 1, 3] = 5
    for _ in range(xp._STATE_MIN_FOLDS):
        rec.fold(delta)
    assert not rec.maybe_save(100.0 + xp._STATE_SAVE_S / 2)
    monkeypatch.setattr(xp, "_is_writer", lambda: False)
    assert not rec.maybe_save(100.0 + xp._STATE_SAVE_S)
    monkeypatch.setattr(xp, "_is_writer", lambda: True)
    assert rec.maybe_save(100.0 + xp._STATE_SAVE_S)
    assert xp.load_rankings(str(path))[3][0] == 3


def test_state_writers_merge_their_layers(tmp_path):
    path = tmp_path / "state.json"
    xp._write_state(str(path), {3: [1, 2]})
    xp._write_state(str(path), {7: [5]})  # another pipeline stage
    xp._write_state(str(path), {3: [2, 1]})
    assert json.loads(path.read_text()) == {"3": [2, 1], "7": [5]}


def test_step_snapshots_on_the_stream_and_folds_once_complete(
    host, tmp_path, monkeypatch
):
    xp._LIVE[0] = True
    p = _counting(state_path=str(tmp_path / "state.json"))
    p.count_routes(torch.tensor([[4, 4], [9, 4]]))
    clock = [1000.0]
    monkeypatch.setattr(xp, "_now", lambda: clock[0])
    xp.step()  # set up
    clock[0] += xp._FOLD_S
    xp.step()  # snapshot issued, event pending
    assert len(host.events) == 1
    clock[0] += xp._FOLD_S
    xp.step()  # copy still in flight: no fold, no second snapshot
    assert xp._RECORDER.folds == 0 and len(host.events) == 1
    host.events[0].done = True
    xp.step()
    assert xp._RECORDER.folds == 1
    assert xp._RECORDER.order(0)[:2] == [4, 9]
    assert host.syncs == []  # the forward path never waits


def test_step_does_nothing_while_capturing(host, tmp_path, monkeypatch):
    _counting(state_path=str(tmp_path / "state.json"))
    monkeypatch.setattr(xp, "_capturing", lambda: True)
    xp.step()
    assert not xp._RECORDER.ready and host.events == []


def test_shutdown_writes_the_state(host, tmp_path, monkeypatch):
    monkeypatch.setattr(xp, "_STATE_MIN_FOLDS", 1)
    path = tmp_path / "state.json"
    xp._LIVE[0] = True
    p = _counting(state_path=str(path), boot_order=[0, 1, 2])
    xp.step()  # set up
    p.count_routes(torch.tensor([[13, 13], [13, 6]]))
    xp.shutdown()
    assert xp.load_rankings(str(path))[3][:2] == [13, 6]
    assert host.syncs  # shutdown may wait
    assert not xp._PLACEMENTS and not xp._LIVE[0]


def test_shutdown_without_counting_is_a_no_op(host, tmp_path):
    xp.shutdown()
    assert not list(tmp_path.iterdir()) and host.syncs == []


def test_steps_after_a_warmup_under_inference_mode(host, tmp_path, monkeypatch):
    # A step can come under torch.inference_mode (the kernel warm-up drives
    # the worker's execute_model inside it), so the buffers it sets up are
    # inference tensors; the worker calls step() before GPUWorker.execute_model
    # enters the mode, so serving steps and shutdown come from outside it.
    monkeypatch.setattr(xp, "_STATE_MIN_FOLDS", 1)
    path = tmp_path / "state.json"
    clock = [1000.0]
    monkeypatch.setattr(xp, "_now", lambda: clock[0])
    p = _counting(state_path=str(path))
    with torch.inference_mode():
        xp.step()  # under the mode: sets up the recorder
    assert xp._RECORDER.ready
    with torch.inference_mode():
        p.count_routes(torch.tensor([[13, 13], [13, 6]]))  # a forward
    clock[0] += xp._FOLD_S
    xp.step()  # serving: snapshot issued
    host.events[-1].done = True
    xp.step()  # serving: folded
    assert xp._RECORDER.folds == 1
    xp.shutdown()
    assert xp.load_rankings(str(path))[3][:2] == [13, 6]


def test_warmup_steps_leave_the_counters_at_zero(host, tmp_path):
    # The worker's warm-up drives execute_model with dummy batches, under
    # inference mode, and may replay captured graphs that count: not traffic.
    p = _counting(state_path=str(tmp_path / "state.json"))
    ids = torch.tensor([[1, 5], [5, 2]])
    with xp.warmup(), torch.inference_mode():
        xp.step()
        xp.step()
        p.count_routes(ids)  # an eager dummy forward
        assert not p.route_counts.any()
        assert not xp._LIVE[0] and not xp._RECORDER.ready
        p.route_counts[1, ids.flatten()] += 1  # a graph replayed on dummy tokens
    assert not xp._WARMUP[0]
    xp.step()  # the first serving step
    assert xp._LIVE[0] and xp._RECORDER.ready
    assert not p.route_counts.any()
    p.count_routes(ids)
    assert int(p.route_counts[1].sum()) == ids.numel()


def _xpu_ready() -> bool:
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        return False
    try:
        from vllm_xpu_kernels.fused_moe_interface import FUSEDMOE_AVAILABLE
    except ImportError:
        return False
    return FUSEDMOE_AVAILABLE


@pytest.mark.skipif(not _xpu_ready(), reason="needs an XPU and its MoE kernels")
def test_split_layer_matches_the_resident_layer():
    """Hot experts in VRAM plus cold experts on pinned host views, summed, equal
    the layer with every expert in VRAM up to the order of the sum."""
    from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe, implement_zp

    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    dev = torch.device("xpu:0")
    e, k, inter, group, topk, tokens = 32, 2560, 640, 128, 8, 24
    gen = torch.Generator().manual_seed(5)
    w13 = torch.randint(
        0, 256, (e, 2 * inter, k // 2), dtype=torch.uint8, generator=gen
    )
    w2 = torch.randint(0, 256, (e, k, inter // 2), dtype=torch.uint8, generator=gen)
    s13 = (torch.rand(e, 2 * inter, k // group, generator=gen) * 4e-3 + 2e-3).to(
        torch.bfloat16
    )
    s2 = (torch.rand(e, k, inter // group, generator=gen) * 4e-3 + 2e-3).to(
        torch.bfloat16
    )
    ids = torch.stack([torch.randperm(e, generator=gen)[:topk] for _ in range(tokens)])
    weights = torch.softmax(torch.randn(tokens, topk, generator=gen), dim=-1)
    x = (torch.randn(tokens, k, generator=gen) * 0.5).to(torch.bfloat16).to(dev)
    topk_w, topk_ids = weights.to(dev), ids.to(dev, torch.int32)

    full = XpuFusedMoe(
        w13=torch.nn.Parameter(w13.to(dev), requires_grad=False),
        w13_scales=s13.to(dev),
        w13_bias=None,
        w2=torch.nn.Parameter(w2.to(dev), requires_grad=False),
        w2_scales=s2.to(dev),
        w2_bias=None,
        n_experts_per_token=topk,
        activation="silu",
        num_experts=e,
    )
    ref = torch.empty_like(x)
    full.apply(ref, x, topk_w, topk_ids)

    hot_ids = list(range(0, e, 3))
    plan = xp.ExpertPlacement(
        "model.layers.0.mlp.experts", list(range(e)), list(range(e)), e, hot_ids
    )
    hot_t = torch.tensor(plan.hot_local)
    cold_t = torch.tensor(plan.cold_local)
    host = {
        "w13": torch.stack([implement_zp(w) for w in w13[cold_t]]).pin_memory(),
        "w2": torch.stack([implement_zp(w) for w in w2[cold_t]]).pin_memory(),
        "w13_scales": s13[cold_t].contiguous().pin_memory(),
        "w2_scales": s2[cold_t].contiguous().pin_memory(),
    }
    plan.cold_views = {
        key: get_accelerator_view_from_cpu_tensor(t) for key, t in host.items()
    }
    hot_map = torch.full((e,), -1, dtype=torch.int32)
    cold_map = torch.full((e,), -1, dtype=torch.int32)
    for g, (is_cold, row) in plan.slot.items():
        (cold_map if is_cold else hot_map)[g] = row
    plan.hot_map, plan.cold_map = hot_map.to(dev), cold_map.to(dev)

    hot, cold = plan.impls(
        torch.nn.Parameter(w13[hot_t].to(dev), requires_grad=False),
        s13[hot_t].to(dev),
        torch.nn.Parameter(w2[hot_t].to(dev), requires_grad=False),
        s2[hot_t].to(dev),
        topk,
        "silu",
        None,
    )
    out, cold_out = torch.empty_like(x), torch.empty_like(x)
    hot.apply(out, x, topk_w, topk_ids)
    cold.apply(cold_out, x, topk_w, topk_ids)
    out.add_(cold_out)
    torch.xpu.synchronize()
    assert plan.cold_views["w13"].data_ptr() == host["w13"].data_ptr()
    err = (out.float() - ref.float()).norm() / ref.float().norm()
    assert err < 1e-2, err
