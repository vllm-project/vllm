# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.models.deepseek_v4.amd.rocm import (
    _block_scale_to_e8m0,
    _wo_a_rope_tables,
)


def test_block_scale_to_e8m0_from_float():
    scale = torch.tensor([[0.5, 1.0, 2.0, 4.0]], dtype=torch.float32)

    encoded = _block_scale_to_e8m0(scale)

    assert encoded is not None
    torch.testing.assert_close(
        encoded,
        torch.tensor([[126, 127, 128, 129]], dtype=torch.uint8),
    )
    assert encoded.is_contiguous()


def test_block_scale_to_e8m0_preserves_encoded_scales():
    raw = torch.tensor([[125, 127, 131]], dtype=torch.uint8)
    encoded = raw.view(torch.float8_e8m0fnu)

    converted = _block_scale_to_e8m0(encoded)

    assert converted is not None
    torch.testing.assert_close(converted, raw)


@pytest.mark.parametrize(
    "scale",
    [
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([[-1.0, 1.0]]),
        torch.tensor([[0.75, 1.0]]),
        torch.tensor([[float("inf"), 1.0]]),
        torch.ones(1, dtype=torch.int32),
    ],
)
def test_block_scale_to_e8m0_rejects_invalid_scales(scale: torch.Tensor):
    assert _block_scale_to_e8m0(scale) is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_wo_a_rope_tables_reuse_shared_rotary_embedding(dtype, default_vllm_config):
    """Layers sharing RoPE must share dense BF16 WO_A tables too."""
    parameters = {"rope_type": "default", "rope_theta": 10000, "rope_dim": 64}
    with torch.device("cpu"):
        first = get_rope(512, 73, False, parameters, dtype=dtype)
        second = get_rope(512, 73, False, parameters, dtype=dtype)
        cos, sin = _wo_a_rope_tables(first)
        again_cos, again_sin = _wo_a_rope_tables(second)
    assert first is second
    assert cos.data_ptr() == again_cos.data_ptr()
    assert sin.data_ptr() == again_sin.data_ptr()
    expected_cos, expected_sin = first.cos_sin_cache.bfloat16().chunk(2, dim=-1)
    for actual, expected in ((cos, expected_cos), (sin, expected_sin)):
        assert actual.dtype == torch.bfloat16
        assert actual.is_contiguous()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_wo_a_rope_tables_keep_distinct_rotary_parameters_separate(default_vllm_config):
    """Main and compressor theta values must retain their own tables."""
    with torch.device("cpu"):
        main = get_rope(512, 79, False, {"rope_theta": 10000, "rope_dim": 64})
        compressed = get_rope(512, 79, False, {"rope_theta": 20000, "rope_dim": 64})
        main_cos, main_sin = _wo_a_rope_tables(main)
        compressed_cos, compressed_sin = _wo_a_rope_tables(compressed)
    assert main_cos.data_ptr() != compressed_cos.data_ptr()
    assert main_sin.data_ptr() != compressed_sin.data_ptr()
    for module, actual in (
        (main, (main_cos, main_sin)),
        (compressed, (compressed_cos, compressed_sin)),
    ):
        for got, want in zip(actual, module.cos_sin_cache.bfloat16().chunk(2, dim=-1)):
            torch.testing.assert_close(got, want, atol=0, rtol=0)


def test_wo_a_rope_tables_restore_required_dtype_after_module_conversion(
    default_vllm_config,
):
    with torch.device("cpu"):
        rope = get_rope(512, 83, False, {"rope_theta": 10000, "rope_dim": 64})
        _wo_a_rope_tables(rope)
        rope.to(dtype=torch.float64)
        cos, sin = _wo_a_rope_tables(rope)
    assert cos.dtype == sin.dtype == torch.bfloat16
    for got, want in zip((cos, sin), rope.cos_sin_cache.bfloat16().chunk(2, dim=-1)):
        torch.testing.assert_close(got, want, atol=0, rtol=0)
