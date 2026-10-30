# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling expert writes go through the parameter's ``weight_loader``, so a
layerwise reload (which restores layers on meta and replays wrapped loaders)
sees them and the experts take the new weights."""

from types import SimpleNamespace

import torch
from torch import nn

from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.models.inkling.amd import moe as amd_moe
from vllm.models.inkling.common.weights import (
    copy_into_slots,
    copy_padded,
    deinterleave_into,
    write_loader,
)
from vllm.models.inkling.nvidia import moe

N_EXPERTS, D_MODEL, D_MLP = 2, 4, 6


def _reload(model: nn.Module, load) -> None:
    initialize_layerwise_reload(model)
    load()
    finalize_layerwise_reload(model, model_config=None)


def _sink_checkpoint(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    w13 = torch.randn(N_EXPERTS, 2 * D_MLP, D_MODEL, generator=gen)
    w2 = torch.randn(N_EXPERTS, D_MODEL, D_MLP, generator=gen)
    return w13, w2


def test_sink_experts_reload_takes_new_weights(monkeypatch):
    monkeypatch.setattr(moe, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(moe, "get_tensor_model_parallel_rank", lambda: 0)
    sinks = moe.InklingSinkExperts(N_EXPERTS, D_MODEL, D_MLP)
    model = nn.Sequential(sinks)

    def load(seed: int) -> None:
        w13, w2 = _sink_checkpoint(seed)
        sinks.load_weight("w13_weight", w13)
        sinks.load_weight("w2_weight", w2)

    load(seed=0)
    record_metadata_for_reloading(model)
    ptrs = [sinks.w13_weight.data_ptr(), sinks.w2_weight.data_ptr()]

    _reload(model, lambda: load(seed=1))

    w13, w2 = _sink_checkpoint(seed=1)
    torch.testing.assert_close(sinks.w13_weight.data, w13)
    torch.testing.assert_close(sinks.w2_weight.data, torch.cat(list(w2), dim=1))
    assert [sinks.w13_weight.data_ptr(), sinks.w2_weight.data_ptr()] == ptrs


class _Experts(nn.Module):
    """Routed-experts stand-in: fused [gate; up] rows and per-expert scalars."""

    def __init__(self) -> None:
        super().__init__()
        self.w13_weight = nn.Parameter(
            torch.zeros(N_EXPERTS, 2 * D_MLP, D_MODEL), requires_grad=False
        )
        self.w13_scale_2 = nn.Parameter(torch.zeros(N_EXPERTS, 2), requires_grad=False)
        for param in self.parameters():
            param.weight_loader = write_loader

    def load(self, seed: int) -> None:
        gen = torch.Generator().manual_seed(seed)
        for slot in range(N_EXPERTS):
            # checkpoint rows interleaved [g0, u0, g1, u1, ...]
            slab = torch.randn(2 * D_MLP, D_MODEL, generator=gen)
            self.w13_weight.weight_loader(
                self.w13_weight, slab, deinterleave_into(slot, D_MLP)
            )
        scales = torch.rand(N_EXPERTS, generator=gen)[:, None].expand(-1, 2)
        self.w13_scale_2.weight_loader(
            self.w13_scale_2, scales, copy_into_slots(list(range(N_EXPERTS)))
        )


def _expected(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    slabs = [torch.randn(2 * D_MLP, D_MODEL, generator=gen) for _ in range(N_EXPERTS)]
    w13 = torch.stack([torch.cat([s[0::2], s[1::2]]) for s in slabs])
    scales = torch.rand(N_EXPERTS, generator=gen)[:, None].expand(-1, 2)
    return w13, scales


def test_slot_writers_reload_takes_new_weights():
    seed = 1
    experts = _Experts()
    model = nn.Sequential(experts)
    experts.load(seed=0)
    w13, scales = _expected(seed=0)
    torch.testing.assert_close(experts.w13_weight.data, w13)
    torch.testing.assert_close(experts.w13_scale_2.data, scales)

    record_metadata_for_reloading(model)
    _reload(model, lambda: experts.load(seed=seed))

    w13, scales = _expected(seed=seed)
    torch.testing.assert_close(experts.w13_weight.data, w13)
    torch.testing.assert_close(experts.w13_scale_2.data, scales)


class _PaddedExperts(nn.Module):
    """AITER-style padding: each gate/up half and the w2 columns are wider
    than the checkpoint's; weights pad with zero, scales with one."""

    ROWS, COLS = D_MLP - 2, D_MODEL - 1

    def __init__(self) -> None:
        super().__init__()
        self.w13_weight = nn.Parameter(
            torch.zeros(N_EXPERTS, 2 * D_MLP, D_MODEL), requires_grad=False
        )
        self.w2_weight_scale = nn.Parameter(
            torch.ones(N_EXPERTS, D_MODEL, D_MODEL), requires_grad=False
        )
        for param in self.parameters():
            param.weight_loader = write_loader

    def load(self, seed: int) -> None:
        gen = torch.Generator().manual_seed(seed)
        for slot in range(N_EXPERTS):
            slab = torch.randn(2 * self.ROWS, D_MODEL, generator=gen)
            self.w13_weight.weight_loader(
                self.w13_weight, slab, deinterleave_into(slot, D_MLP, 0.0)
            )
            scale = torch.rand(D_MODEL, self.COLS, generator=gen)
            self.w2_weight_scale.weight_loader(
                self.w2_weight_scale, scale, copy_padded(slot, self.COLS, 1.0)
            )


def test_padded_slots_reload_completes_and_keeps_padding():
    experts = _PaddedExperts()
    model = nn.Sequential(experts)
    experts.load(seed=0)
    cold = [p.data.clone() for p in experts.parameters()]
    rows, cols = _PaddedExperts.ROWS, _PaddedExperts.COLS
    assert torch.count_nonzero(cold[0][:, rows:D_MLP]) == 0
    assert torch.count_nonzero(cold[0][:, D_MLP + rows :]) == 0
    assert torch.all(cold[1][..., cols:] == 1)

    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    experts.load(seed=0)
    # every element was written, so the layer was processed as it loaded
    # instead of being left (uninitialized padding) for finalize
    assert experts.w13_weight.device.type != "meta"
    finalize_layerwise_reload(model, model_config=None)

    for param, expected in zip(experts.parameters(), cold):
        torch.testing.assert_close(param.data, expected)


class _Routed(nn.Module):
    """Routed-experts stand-in with the parameters ``InklingMoE`` loads
    (NVFP4 layout, values in float32 for exact comparison)."""

    INTER = D_MLP

    def __init__(self) -> None:
        super().__init__()
        e, h, i = N_EXPERTS, D_MODEL, self.INTER
        shapes = {
            "w13_weight": (e, 2 * i, h),
            "w2_weight": (e, h, i),
            "w13_weight_scale_2": (e, 2),
            "w2_weight_scale_2": (e,),
            "w13_input_scale": (e, 2),
            "w2_input_scale": (e,),
        }
        for name, shape in shapes.items():
            param = nn.Parameter(torch.zeros(shape), requires_grad=False)
            param.weight_loader = write_loader  # as InklingMoE.__init__ sets up
            self.register_parameter(name, param)
        self.moe_config = SimpleNamespace(
            moe_parallel_config=SimpleNamespace(tp_rank=0, tp_size=1)
        )


def _routed_checkpoint(seed: int) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    e, h, i = N_EXPERTS, D_MODEL, _Routed.INTER
    return {
        # rows interleaved [g0, u0, g1, u1, ...]
        "w13_weight": torch.randn(e, 2 * i, h, generator=gen),
        "w2_weight": torch.randn(e, h, i, generator=gen),
        "w13_weight_scale_2": torch.rand(e, generator=gen),
        "w2_weight_scale_2": torch.rand(e, generator=gen),
        "w13_weight.input_amax": torch.rand(e, generator=gen) + 1,
        "w2_weight.input_amax": torch.rand(e, generator=gen) + 1,
    }


def test_routed_experts_reload_takes_new_weights():
    routed = _Routed()
    model = nn.Sequential(routed)
    # borrow the loader as test_moe_weight_layout.py does
    layer = SimpleNamespace(experts=SimpleNamespace(routed_experts=routed))
    layer._local_expert_slots = lambda: {g: g for g in range(N_EXPERTS)}

    def load(seed: int) -> None:
        for key, weight in _routed_checkpoint(seed).items():
            moe.InklingMoE.load_expert_weight(layer, f"experts.{key}", weight)

    load(seed=0)
    record_metadata_for_reloading(model)
    ptrs = {n: p.data_ptr() for n, p in routed.named_parameters()}

    _reload(model, lambda: load(seed=1))

    ckpt = _routed_checkpoint(seed=1)
    w13 = ckpt["w13_weight"]
    torch.testing.assert_close(
        routed.w13_weight.data, torch.cat([w13[:, 0::2], w13[:, 1::2]], dim=1)
    )
    torch.testing.assert_close(routed.w2_weight.data, ckpt["w2_weight"])
    torch.testing.assert_close(
        routed.w13_weight_scale_2.data,
        ckpt["w13_weight_scale_2"][:, None].expand(-1, 2),
    )
    torch.testing.assert_close(routed.w2_weight_scale_2.data, ckpt["w2_weight_scale_2"])
    for proj in ("w13", "w2"):
        scale = getattr(routed, f"{proj}_input_scale").data
        amax = float(ckpt[f"{proj}_weight.input_amax"].max())
        torch.testing.assert_close(
            scale, torch.full_like(scale, amax / (448.0 * 6.0)), rtol=0, atol=0
        )
    assert {n: p.data_ptr() for n, p in routed.named_parameters()} == ptrs


class _NegateAfterLoad(QuantizeMethodBase):
    """Stands in for the unquantized method's post-load processing: zero the
    padding (the loaders do not write it), then transform (the AITER
    shuffle)."""

    def create_weights(self, *args, **kwargs) -> None:
        pass

    def apply(self, *args, **kwargs) -> None:
        pass

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        i, i_pad = _NumelPaddedExperts.INTER, _NumelPaddedExperts.INTER_PAD
        layer.w13_weight.data[:, i:i_pad].zero_()
        layer.w13_weight.data[:, i_pad + i :].zero_()
        layer.w2_weight.data[..., i:].zero_()
        layer.w13_weight.data.mul_(-1)
        layer.w2_weight.data.mul_(-1)


class _NumelPaddedExperts(nn.Module):
    """Unquantized AITER layout: the intermediate dim is padded and the
    parameters declare ``weight_loader_numel`` (checkpoint elements only); the
    method zeroes the padding itself."""

    INTER, INTER_PAD = 3, 4

    def __init__(self) -> None:
        super().__init__()
        e, h = N_EXPERTS, D_MODEL
        self.w13_weight = nn.Parameter(
            torch.zeros(e, 2 * self.INTER_PAD, h), requires_grad=False
        )
        self.w13_weight.weight_loader_numel = e * 2 * self.INTER * h
        self.w2_weight = nn.Parameter(
            torch.zeros(e, h, self.INTER_PAD), requires_grad=False
        )
        self.w2_weight.weight_loader_numel = e * h * self.INTER
        for param in self.parameters():
            param.weight_loader = write_loader
        self.quant_method = _NegateAfterLoad()
        self.moe_config = SimpleNamespace(
            moe_parallel_config=SimpleNamespace(tp_rank=0, tp_size=1)
        )

    def load(self, seed: int) -> None:
        gen = torch.Generator().manual_seed(seed)
        layer = amd_moe.InklingMoE.__new__(amd_moe.InklingMoE)
        nn.Module.__init__(layer)
        layer.n_routed_experts = N_EXPERTS
        layer.experts = SimpleNamespace(routed_experts=self)
        layer._local_expert_slots = (  # type: ignore[method-assign]
            lambda: {g: g for g in range(N_EXPERTS)}
        )
        e, h, i = N_EXPERTS, D_MODEL, self.INTER
        w13 = torch.randn(e, 2 * i, h, generator=gen)
        w2 = torch.randn(e, h, i, generator=gen)
        layer.load_expert_weight("experts.w13_weight", w13)
        layer.load_expert_weight("experts.w2_weight", w2)


def test_numel_padded_experts_reload_is_not_processed_early():
    """Writing the padding of a ``weight_loader_numel`` parameter over-counts,
    so layerwise reload would process the layer before every expert arrived
    and drop the later writes."""
    experts = _NumelPaddedExperts()
    model = nn.Sequential(experts)
    experts.load(seed=0)
    experts.quant_method.process_weights_after_loading(experts)
    record_metadata_for_reloading(model)

    _reload(model, lambda: experts.load(seed=1))

    fresh = _NumelPaddedExperts()
    fresh.load(seed=1)
    fresh.quant_method.process_weights_after_loading(fresh)
    torch.testing.assert_close(experts.w13_weight.data, fresh.w13_weight.data)
    torch.testing.assert_close(experts.w2_weight.data, fresh.w2_weight.data)
