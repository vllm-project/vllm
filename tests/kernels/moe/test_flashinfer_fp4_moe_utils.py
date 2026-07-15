# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import flashinfer
import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel  # noqa: F401
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
    prepare_static_weights_for_trtllm_fp4_moe,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Test requires CUDA"
)


@torch.inference_mode()
def test_prepare_static_weights_avoids_expert_stack(monkeypatch: pytest.MonkeyPatch):
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    num_experts, hidden_size, intermediate_size = 4, 256, 128
    generator = torch.Generator(device="cuda").manual_seed(7)

    def random_bytes(*shape: int, dtype: torch.dtype = torch.uint8):
        return torch.randint(
            0,
            256,
            shape,
            dtype=torch.uint8,
            device="cuda",
            generator=generator,
        ).view(dtype)

    w13 = random_bytes(num_experts, 2 * intermediate_size, hidden_size // 2)
    w2 = random_bytes(num_experts, hidden_size, intermediate_size // 2)
    w13_scale = random_bytes(
        num_experts,
        2 * intermediate_size,
        hidden_size // 16,
        dtype=torch.float8_e4m3fn,
    )
    w2_scale = random_bytes(
        num_experts,
        hidden_size,
        intermediate_size // 16,
        dtype=torch.float8_e4m3fn,
    )

    cache: dict[torch.Size, torch.Tensor] = {}
    w13_indices = _maybe_get_cached_w3_w1_permute_indices(
        cache, w13[0], 128, is_gated_act_gemm=True
    ).to("cuda")
    w13_scale_indices = _maybe_get_cached_w3_w1_permute_indices(
        cache,
        w13_scale[0].view(torch.uint8),
        128,
        num_elts_per_sf=16,
        is_gated_act_gemm=True,
    ).to("cuda")
    w2_indices = get_w2_permute_indices_with_cache(cache, w2[0], 128).to("cuda")
    w2_scale_indices = get_w2_permute_indices_with_cache(
        cache, w2_scale[0].view(torch.uint8), 128, num_elts_per_sf=16
    ).to("cuda")
    expected = (
        w13[:, w13_indices],
        w13_scale.view(torch.uint8)[:, w13_scale_indices].view(torch.float8_e4m3fn),
        w2[:, w2_indices],
        w2_scale.view(torch.uint8)[:, w2_scale_indices].view(torch.float8_e4m3fn),
    )

    monkeypatch.setattr(
        flashinfer, "nvfp4_block_scale_interleave", lambda scale: scale.clone()
    )

    def fail_on_stack(*args, **kwargs):
        raise AssertionError("weight preparation must not retain and stack experts")

    monkeypatch.setattr(torch, "stack", fail_on_stack)
    actual = prepare_static_weights_for_trtllm_fp4_moe(
        w13,
        w2,
        w13_scale,
        w2_scale,
        hidden_size,
        intermediate_size,
        num_experts,
        is_gated_activation=True,
    )

    for actual_tensor, expected_tensor in zip(actual, expected):
        assert torch.equal(
            actual_tensor.view(torch.uint8), expected_tensor.view(torch.uint8)
        )
