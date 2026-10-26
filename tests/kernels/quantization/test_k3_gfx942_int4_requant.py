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

pytest.importorskip("aiter.ops.quant")

from vllm.model_executor.layers.quantization import mxfp4  # noqa: E402

NUM_EXPERTS, ROWS, COLS = 128, 256, 2048


def _requant(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    weight = torch.randint(
        0,
        256,
        (NUM_EXPERTS, ROWS, COLS // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=gen,
    )
    scale = torch.randint(
        118,
        132,
        (NUM_EXPERTS, ROWS, COLS // 32),
        dtype=torch.uint8,
        device="cuda",
        generator=gen,
    )
    out_weight, out_scale = mxfp4._mxfp4_experts_to_int4_gfx942(
        torch.nn.Parameter(weight, requires_grad=False),
        torch.nn.Parameter(scale, requires_grad=False),
    )
    return out_weight.view(torch.uint8), out_scale.view(torch.uint8)


@pytest.mark.parametrize(
    "experts_per_chunk",
    [1, 5, NUM_EXPERTS],
    ids=["one_expert", "partial_last_chunk", "single_chunk"],
)
def test_requant_is_independent_of_chunking(monkeypatch, experts_per_chunk):
    ref_weight, ref_scale = _requant()

    monkeypatch.setattr(
        mxfp4, "_K3_INT4_CONVERT_CHUNK_VALUES", experts_per_chunk * ROWS * COLS
    )
    weight, scale = _requant()

    assert torch.equal(weight, ref_weight)
    assert torch.equal(scale, ref_scale)


def test_requant_reuses_device_memory_across_chunks(monkeypatch):
    monkeypatch.setattr(mxfp4, "_K3_INT4_CONVERT_CHUNK_VALUES", ROWS * COLS)
    _requant()
    torch.accelerator.empty_cache()

    before = torch.accelerator.memory_stats()["num_device_alloc"]
    _requant()
    device_allocs = torch.accelerator.memory_stats()["num_device_alloc"] - before

    # One expert per chunk gives NUM_EXPERTS chunks of about ten temporaries
    # each. Later chunks must reuse the cached blocks, so device allocations
    # stay a small constant instead of growing with the chunk count.
    assert device_allocs <= 16
