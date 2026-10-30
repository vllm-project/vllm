# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling expert writes go through the parameter's ``weight_loader``, so a
layerwise reload (which restores layers on meta and replays wrapped loaders)
sees them and the experts take the new weights."""

import pytest
import torch
from torch import nn

from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
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


@pytest.mark.parametrize("seed", [1, 2])
def test_slot_writers_reload_takes_new_weights(seed):
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
