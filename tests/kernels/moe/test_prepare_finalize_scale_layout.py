# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.prepare_finalize.scale_layout import (
    swizzle_scale_after_alltoall,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    swizzle_mxfp8_scale,
)


def _mxfp8_quant_config(is_scale_swizzled: bool) -> FusedMoEQuantConfig:
    return FusedMoEQuantConfig.make(
        "mxfp8",
        is_scale_swizzled=is_scale_swizzled,
    )


def _uint8_arange(numel: int) -> torch.Tensor:
    return torch.arange(numel, dtype=torch.int32).remainder(251).to(torch.uint8)


def test_post_alltoall_mxfp8_scales_are_swizzled() -> None:
    num_tokens = 129
    hidden_size = 160
    scale = _uint8_arange(num_tokens * (hidden_size // 32)).view(num_tokens, -1)

    actual = swizzle_scale_after_alltoall(
        scale,
        _mxfp8_quant_config(is_scale_swizzled=True),
        num_tokens=num_tokens,
        hidden_size=hidden_size,
    )

    expected = swizzle_mxfp8_scale(scale, M=num_tokens, K=hidden_size)
    assert actual.ndim == 1
    torch.testing.assert_close(actual, expected)


def test_post_alltoall_mxfp8_scales_flatten_batch_dims() -> None:
    batch = 2
    tokens_per_batch = 3
    hidden_size = 128
    scale = _uint8_arange(batch * tokens_per_batch * (hidden_size // 32)).view(
        batch, tokens_per_batch, -1
    )

    actual = swizzle_scale_after_alltoall(
        scale,
        _mxfp8_quant_config(is_scale_swizzled=True),
        num_tokens=batch * tokens_per_batch,
        hidden_size=hidden_size,
    )

    scale_2d = scale.view(batch * tokens_per_batch, -1)
    expected = swizzle_mxfp8_scale(scale_2d, M=scale_2d.size(0), K=hidden_size)
    torch.testing.assert_close(actual, expected)


def test_post_alltoall_mxfp8_unswizzled_scale_is_unchanged() -> None:
    scale = torch.ones((4, 4), dtype=torch.uint8)

    actual = swizzle_scale_after_alltoall(
        scale,
        _mxfp8_quant_config(is_scale_swizzled=False),
        num_tokens=scale.size(0),
        hidden_size=128,
    )

    assert actual is scale
