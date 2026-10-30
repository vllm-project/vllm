# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling expert writes go through the parameter's ``weight_loader``, so a
layerwise reload (which restores layers on meta and replays wrapped loaders)
sees them: after a reload with new weights the layer equals a cold start on
those weights."""

import pytest
import torch
from torch import nn

from tests.models.inkling.reload_utils import (
    HIDDEN,
    N_ROUTED,
    Checkpoint,
    InklingMoELayer,
    assert_reload_matches_cold_start,
    sink_checkpoint,
)
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.models.inkling.common.weights import (
    copy_padded,
    deinterleave_into,
    write_loader,
)
from vllm.models.inkling.nvidia import moe as nvidia_moe
from vllm.platforms import current_platform

requires_cuda = pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA")


NVFP4_INTERMEDIATE = 128


def _nvfp4_checkpoint(seed: int) -> Checkpoint:
    gen = torch.Generator().manual_seed(seed)
    e, h, i = N_ROUTED, HIDDEN, NVFP4_INTERMEDIATE

    def packed(*shape: int) -> torch.Tensor:
        return torch.randint(0, 256, shape, generator=gen, dtype=torch.uint8)

    def positive(*shape: int) -> torch.Tensor:
        return torch.rand(shape, generator=gen) + 0.5

    return {
        # routed experts; w13 rows interleaved [g0, u0, g1, u1, ...]
        "experts.w13_weight": packed(e, 2 * i, h // 2),
        "experts.w13_weight_scale": positive(e, 2 * i, h // 16).to(torch.float8_e4m3fn),
        "experts.w13_weight_scale_2": positive(e) / 100,
        "experts.w13_weight.input_amax": positive(e) * 7,
        "experts.w2_weight": packed(e, h, i // 2),
        "experts.w2_weight_scale": positive(e, h, i // 16).to(torch.float8_e4m3fn),
        "experts.w2_weight_scale_2": positive(e) / 100,
        "experts.w2_weight.input_amax": positive(e) * 7,
        **sink_checkpoint(gen, i),
    }


@requires_cuda
def test_nvfp4_moe_reload_matches_cold_start(dist_init):
    """Routed NVFP4 experts (weights, block scales, per-expert global scales,
    calibrated input scales) and the sink experts take the new weights."""
    quant_config = ModelOptNvFp4Config.from_config(
        {
            "quantization": {
                "quant_algo": "NVFP4",
                "group_size": 16,
                "kv_cache_quant_algo": None,
                "exclude_modules": [],
            }
        }
    )

    def make_layer() -> InklingMoELayer:
        return InklingMoELayer(
            nvidia_moe.InklingMoE, NVFP4_INTERMEDIATE, quant_config, "emulation"
        )

    layer = make_layer()
    layer.load(_nvfp4_checkpoint(0))
    # the calibrated input scale is float32 even under a bf16 default dtype
    for proj in ("w13", "w2"):
        amax = float(_nvfp4_checkpoint(0)[f"experts.{proj}_weight.input_amax"].max())
        scale = getattr(layer.moe.experts.routed_experts, f"{proj}_input_scale")
        assert torch.equal(scale, torch.full_like(scale, amax / (448.0 * 6.0)))

    assert_reload_matches_cold_start(make_layer, _nvfp4_checkpoint)


class _PaddedExperts(nn.Module):
    """AITER-style padded MXFP4 parameters (ROCm only, so not built here):
    each gate/up half and the w2 columns are wider than the checkpoint's, and
    every element counts. Weights pad with zero, scales with one."""

    E, ROWS, HALF, COLS, WIDTH = 2, 4, 6, 3, 4

    def __init__(self) -> None:
        super().__init__()
        self.w13_weight = nn.Parameter(
            torch.zeros(self.E, 2 * self.HALF, self.WIDTH), requires_grad=False
        )
        self.w2_weight_scale = nn.Parameter(
            torch.ones(self.E, self.WIDTH, self.WIDTH), requires_grad=False
        )
        for param in self.parameters():
            param.weight_loader = write_loader

    def load(self) -> None:
        gen = torch.Generator().manual_seed(0)
        for slot in range(self.E):
            slab = torch.randn(2 * self.ROWS, self.WIDTH, generator=gen)
            self.w13_weight.weight_loader(
                self.w13_weight, slab, deinterleave_into(slot, self.HALF, 0.0)
            )
            scale = torch.rand(self.WIDTH, self.COLS, generator=gen)
            self.w2_weight_scale.weight_loader(
                self.w2_weight_scale, scale, copy_padded(slot, self.COLS, 1.0)
            )


def test_padded_slots_reload_completes_and_keeps_padding():
    experts = _PaddedExperts()
    model = nn.Sequential(experts)
    experts.load()
    cold = [p.data.clone() for p in experts.parameters()]

    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    experts.load()
    # every element was written, so the layer was processed as it loaded
    # instead of being left (uninitialized padding) for finalize
    assert experts.w13_weight.device.type != "meta"
    finalize_layerwise_reload(model, model_config=None)

    for param, expected in zip(experts.parameters(), cold):
        torch.testing.assert_close(param.data, expected, rtol=0, atol=0)
