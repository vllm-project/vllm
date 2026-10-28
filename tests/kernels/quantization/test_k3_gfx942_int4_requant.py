# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 gfx942 MXFP4-to-int4 load-time requant."""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("requires ROCm", allow_module_level=True)

from vllm.platforms.rocm import on_gfx942  # noqa: E402

if not on_gfx942():
    pytest.skip("requires gfx942", allow_module_level=True)

pytest.importorskip("aiter")

from aiter.ops.quant import per_1x32_i4_quant  # noqa: E402
from aiter.ops.shuffle import (  # noqa: E402
    pack_int8_to_packed_int4,
    shuffle_scale_for_int4,
    shuffle_weight,
)
from aiter.utility import fp4_utils  # noqa: E402

from vllm.model_executor.layers.quantization import mxfp4  # noqa: E402

NUM_EXPERTS, ROWS, COLS = 128, 256, 2048
# Kimi-K3 w2 per rank under TP8. 896 experts do not split evenly into chunks.
TP8_W2_SHAPE = (896, 3584, 384)
E2M1_VALUES = (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6)


def _mxfp4_experts(
    num_experts: int = NUM_EXPERTS, rows: int = ROWS, cols: int = COLS
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cuda").manual_seed(0)
    weight = torch.randint(
        0,
        256,
        (num_experts, rows, cols // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=gen,
    )
    # Normal E8M0 block scales; 0 and 255 have special meanings.
    scale = torch.randint(
        118,
        132,
        (num_experts, rows, cols // 32),
        dtype=torch.uint8,
        device="cuda",
        generator=gen,
    )
    return weight, scale


def _requant(
    weight: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out_weight, out_scale = mxfp4._mxfp4_experts_to_int4_gfx942(
        torch.nn.Parameter(weight.clone(), requires_grad=False),
        torch.nn.Parameter(scale.clone(), requires_grad=False),
    )
    return out_weight.view(torch.uint8), out_scale.view(torch.uint8)


def _reference_requant(
    weight: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """All experts at once, decoding MXFP4 as the OCP MX spec defines it."""
    num_experts, rows, packed_cols = weight.shape
    cols = packed_cols * 2
    # Element 2i is in the low nibble and element 2i + 1 in the high nibble.
    codes = torch.stack((weight & 0xF, weight >> 4), dim=-1).view(
        num_experts, rows, cols
    )
    values = torch.tensor(E2M1_VALUES, device=weight.device)[codes.long()]
    block_scales = torch.ldexp(
        torch.ones_like(scale, dtype=torch.float32), scale.int() - 127
    )
    weight_bf16 = (
        (values.view(num_experts, rows, cols // 32, 32) * block_scales.unsqueeze(-1))
        .view(num_experts, rows, cols)
        .to(torch.bfloat16)
    )

    weight_int4, weight_scale = per_1x32_i4_quant(weight_bf16)
    packed = pack_int8_to_packed_int4(shuffle_weight(weight_int4, (16, 16)))
    packed_scale = shuffle_scale_for_int4(weight_scale, group_size=32).reshape(-1)
    return (
        packed.view(torch.uint8).view(num_experts, rows, packed_cols),
        packed_scale.view(torch.uint8),
    )


@pytest.fixture
def load_model_allocator_settings():
    original = torch._C._accelerator_getAllocatorSettings()
    prefix = original.strip().rstrip(",")
    settings = "max_split_size_mb:20"
    torch._C._accelerator_setAllocatorSettings(
        f"{prefix},{settings}" if prefix else settings
    )
    yield
    torch._C._accelerator_setAllocatorSettings(original)


@pytest.mark.parametrize(
    "experts_per_chunk",
    [1, 5, NUM_EXPERTS],
    ids=["one_expert", "uneven_last_chunk", "single_chunk"],
)
def test_requant_matches_reference(monkeypatch, experts_per_chunk):
    monkeypatch.setattr(
        mxfp4, "_K3_INT4_CONVERT_CHUNK_VALUES", experts_per_chunk * ROWS * COLS
    )
    weight, scale = _mxfp4_experts()

    out_weight, out_scale = _requant(weight, scale)
    ref_weight, ref_scale = _reference_requant(weight, scale)

    assert torch.equal(out_weight, ref_weight)
    assert torch.equal(out_scale, ref_scale)


def test_requant_allocates_device_memory_only_in_first_chunk(
    monkeypatch, load_model_allocator_settings
):
    """Later chunks, the last one included, must reuse the first chunk's cached
    blocks under load_model's allocator settings."""
    weight, scale = _mxfp4_experts(*TP8_W2_SHAPE)
    _requant(weight, scale)
    torch.accelerator.empty_cache()

    device_allocs = []
    mxfp4_to_f32 = fp4_utils.mxfp4_to_f32

    def recording_mxfp4_to_f32(x):
        device_allocs.append(torch.accelerator.memory_stats()["num_device_alloc"])
        return mxfp4_to_f32(x)

    monkeypatch.setattr(fp4_utils, "mxfp4_to_f32", recording_mxfp4_to_f32)
    _requant(weight, scale)
    device_allocs.append(torch.accelerator.memory_stats()["num_device_alloc"])

    assert len(device_allocs) > 3
    assert device_allocs[1:] == [device_allocs[1]] * (len(device_allocs) - 1)
