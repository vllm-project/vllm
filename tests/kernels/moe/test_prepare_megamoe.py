# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DeepGEMM MegaMoE input staging."""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.prepare_megamoe import (
    prepare_nvfp4_megamoe_inputs,
)
from vllm.platforms import current_platform

if not current_platform.has_device_capability(100):
    pytest.skip("NVFP4 MegaMoE staging requires SM100", allow_module_level=True)


@pytest.mark.parametrize("num_tokens,hidden_size", [(1, 6144), (8, 128), (31, 128)])
@pytest.mark.parametrize("global_scale_value", [0.0078125, 1.0, 128.0])
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_prepare_nvfp4_megamoe_inputs(
    num_tokens: int,
    hidden_size: int,
    global_scale_value: float,
    ids_dtype: torch.dtype,
) -> None:
    top_k = 8
    generator = torch.Generator(device="cuda").manual_seed(17)
    hidden_states = torch.randn(
        (num_tokens, hidden_size),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    global_scale = torch.tensor(global_scale_value, dtype=torch.float32, device="cuda")
    topk_ids = torch.randint(
        0,
        256,
        (num_tokens, top_k),
        dtype=ids_dtype,
        device="cuda",
        generator=generator,
    )
    topk_weights = torch.rand(
        (num_tokens, top_k),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    is_padding = torch.arange(num_tokens, device="cuda") % 3 == 0

    expected_x, expected_sf = ops.scaled_fp4_quant(
        hidden_states,
        global_scale,
        is_sf_swizzled_layout=False,
    )
    expected_topk_ids = torch.where(is_padding[:, None], -1, topk_ids).to(torch.int32)
    expected_topk_weights = torch.where(is_padding[:, None], 0.0, topk_weights)

    x = torch.empty_like(expected_x)
    x_sf = torch.empty(
        (num_tokens, hidden_size // 64), dtype=torch.int32, device="cuda"
    )
    staged_topk_ids = torch.empty((num_tokens, top_k), dtype=torch.int64, device="cuda")
    staged_topk_weights = torch.empty_like(topk_weights)
    prepare_nvfp4_megamoe_inputs(
        hidden_states,
        global_scale,
        topk_weights,
        topk_ids,
        x,
        x_sf,
        staged_topk_ids,
        staged_topk_weights,
        is_padding=is_padding,
    )

    torch.testing.assert_close(x, expected_x, rtol=0, atol=0)
    torch.testing.assert_close(
        x_sf,
        expected_sf.contiguous().view(torch.uint8).view(torch.int32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(staged_topk_ids, expected_topk_ids, rtol=0, atol=0)
    torch.testing.assert_close(
        staged_topk_weights, expected_topk_weights, rtol=0, atol=0
    )


@torch.inference_mode()
def test_prepare_nvfp4_megamoe_inputs_cuda_graph_replay() -> None:
    num_tokens, hidden_size, top_k = 1, 6144, 8
    x_bf16 = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda")
    gscale = torch.tensor(128.0, dtype=torch.float32, device="cuda")
    weights = torch.rand((num_tokens, top_k), device="cuda")
    ids = torch.arange(top_k, dtype=torch.int32, device="cuda")[None]
    x_fp4 = torch.empty(
        (num_tokens, hidden_size // 2), dtype=torch.uint8, device="cuda"
    )
    x_sf = torch.empty(
        (num_tokens, hidden_size // 64), dtype=torch.int32, device="cuda"
    )
    staged_ids = torch.empty(ids.shape, dtype=torch.int64, device="cuda")
    staged_weights = torch.empty_like(weights)

    def run() -> None:
        prepare_nvfp4_megamoe_inputs(
            x_bf16,
            gscale,
            weights,
            ids,
            x_fp4,
            x_sf,
            staged_ids,
            staged_weights,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.accelerator.synchronize()

    expected_x, expected_sf = ops.scaled_fp4_quant(
        x_bf16, gscale, is_sf_swizzled_layout=False
    )
    torch.testing.assert_close(x_fp4, expected_x, rtol=0, atol=0)
    torch.testing.assert_close(
        x_sf,
        expected_sf.contiguous().view(torch.uint8).view(torch.int32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(staged_ids, ids.to(torch.int64), rtol=0, atol=0)
    torch.testing.assert_close(staged_weights, weights, rtol=0, atol=0)
