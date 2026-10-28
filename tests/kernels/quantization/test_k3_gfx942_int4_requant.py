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

from aiter.utility import fp4_utils  # noqa: E402

from vllm.model_executor.layers.quantization import mxfp4  # noqa: E402

NUM_EXPERTS, ROWS, COLS = 128, 256, 2048
# Kimi-K3 w2 per rank under TP8. 896 experts do not split evenly into chunks.
TP8_W2_SHAPE = (896, 3584, 384)


def _requant(
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
    scale = torch.randint(
        118,
        132,
        (num_experts, rows, cols // 32),
        dtype=torch.uint8,
        device="cuda",
        generator=gen,
    )
    out_weight, out_scale = mxfp4._mxfp4_experts_to_int4_gfx942(
        torch.nn.Parameter(weight, requires_grad=False),
        torch.nn.Parameter(scale, requires_grad=False),
    )
    return out_weight.view(torch.uint8), out_scale.view(torch.uint8)


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
def test_requant_is_independent_of_chunking(monkeypatch, experts_per_chunk):
    ref_weight, ref_scale = _requant()

    monkeypatch.setattr(
        mxfp4, "_K3_INT4_CONVERT_CHUNK_VALUES", experts_per_chunk * ROWS * COLS
    )
    weight, scale = _requant()

    assert torch.equal(weight, ref_weight)
    assert torch.equal(scale, ref_scale)


def test_requant_allocates_device_memory_only_in_first_chunk(
    monkeypatch, load_model_allocator_settings
):
    """Later chunks, the last one included, must reuse the first chunk's cached
    blocks under load_model's allocator settings."""
    _requant(*TP8_W2_SHAPE)
    torch.accelerator.empty_cache()

    device_allocs = []
    mxfp4_to_f32 = fp4_utils.mxfp4_to_f32

    def recording_mxfp4_to_f32(x):
        device_allocs.append(torch.accelerator.memory_stats()["num_device_alloc"])
        return mxfp4_to_f32(x)

    monkeypatch.setattr(fp4_utils, "mxfp4_to_f32", recording_mxfp4_to_f32)
    _requant(*TP8_W2_SHAPE)
    device_allocs.append(torch.accelerator.memory_stats()["num_device_alloc"])

    assert len(device_allocs) > 3
    assert device_allocs[1:] == [device_allocs[1]] * (len(device_allocs) - 1)
