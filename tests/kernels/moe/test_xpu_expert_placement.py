# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-expert offload placement on XPU: the plan on CPU, the split on a card."""

import json
from types import SimpleNamespace

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


def _config(tmp_path, gb=1.0, rankings=True, **kwargs) -> OffloadConfig:
    path = None
    if rankings:
        path = tmp_path / "rankings.json"
        # layer 3: hottest first; experts 1, 6 and 12 never observed
        path.write_text(json.dumps({"3": [9, 2, 15, 0, 7, 3, 10, 4, 5, 8, 11, 13, 14]}))
    uva = UVAOffloadConfig(
        cpu_offload_gb=gb,
        cpu_offload_expert_rankings=str(path) if path else None,
    )
    return OffloadConfig(uva=uva, **kwargs)


@pytest.fixture
def xpu(monkeypatch):
    monkeypatch.setattr(xp, "current_platform", SimpleNamespace(is_xpu=lambda: True))
    monkeypatch.setattr(
        "vllm.utils.platform_utils.is_uva_available", lambda: True, raising=True
    )
    monkeypatch.delenv("VLLM_WEIGHT_OFFLOADING_DISABLE_UVA", raising=False)
    xp.load_rankings.cache_clear()
    yield monkeypatch
    xp.load_rankings.cache_clear()


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
