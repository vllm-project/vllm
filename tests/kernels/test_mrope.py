# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from transformers import AutoConfig

from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.mrope import (
    mrope_forward_native, triton_mrope)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_test_data(num_tokens: int, num_q_heads: int, num_kv_heads: int,
                       head_size: int, max_position_embeddings: int,
                       dtype: torch.dtype, device: torch.device):
    """Generate test data for given configuration."""
    # Create 2D positions (3, num_tokens) for multimodal case
    positions = torch.randint(0,
                              max_position_embeddings // 4, (3, num_tokens),
                              device=device)

    # Create query and key tensors
    query = torch.randn(num_tokens,
                        num_q_heads * head_size,
                        dtype=dtype,
                        device=device)
    key = torch.randn(num_tokens,
                      num_kv_heads * head_size,
                      dtype=dtype,
                      device=device)

    return positions, query, key


def unroll_model_tp_dict(model_tp_dict):
    return [(model_name, tp_size)
            for model_name, tp_sizes in model_tp_dict.items()
            for tp_size in tp_sizes]


model_tp_dict = {
    "Qwen/Qwen2-VL-2B-Instruct": [1],
    "Qwen/Qwen2-VL-7B-Instruct": [1],
    "Qwen/Qwen2-VL-72B-Instruct": [2, 4, 8],
    "Qwen/Qwen2.5-VL-3B-Instruct": [1, 2, 4, 8],
    "Qwen/Qwen2.5-VL-7B-Instruct": [1, 2, 4, 8],
    "Qwen/Qwen2.5-VL-72B-Instruct": [2, 4, 8]
}

# https://github.com/pytorch/pytorch/blob/main/torch/testing/_comparison.py#L1317
dtype_atol_rtol_list = [
    [torch.bfloat16, 1e-5, 1.6e-2],
]

num_tokens_list = [1, 1024, 4096, 16384]


@pytest.mark.parametrize("model_name, tp_size",
                         unroll_model_tp_dict(model_tp_dict))
@pytest.mark.parametrize("dtype, atol, rtol", dtype_atol_rtol_list)
@pytest.mark.parametrize("num_tokens", num_tokens_list)
def test_mrope(model_name, tp_size, dtype, atol, rtol, num_tokens):

    config = AutoConfig.from_pretrained(model_name)

    # get the model config
    total_num_kv_heads = config.num_key_value_heads
    total_num_heads = config.num_attention_heads
    num_heads = total_num_heads // tp_size
    num_kv_heads = max(1, total_num_kv_heads // tp_size)
    head_dim = config.hidden_size // total_num_heads
    is_neox_style = True
    mrope_section = config.rope_scaling["mrope_section"]

    rope_theta = config.rope_theta
    max_position = config.max_position_embeddings

    mrope_helper_class = get_rope(
        head_size=head_dim,
        rotary_dim=head_dim,
        max_position=max_position,
        base=rope_theta,
        is_neox_style=is_neox_style,
        rope_scaling=config.rope_scaling,
        dtype=dtype,
    )
    mrope_helper_class.cos_sin_cache = mrope_helper_class.cos_sin_cache.to(
        device)

    # create q k v input tensors
    # create rotary pos emb input tensors
    positions, query, key = generate_test_data(num_tokens, num_heads,
                                               num_kv_heads, head_dim,
                                               max_position, dtype, device)
    cos_sin = mrope_helper_class.cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)

    query_native, key_native = mrope_forward_native(
        positions,
        query.clone(),
        key.clone(),
        cos,
        sin,
        mrope_section,
        is_neox_style,
        head_dim,
        head_dim,
    )

    # native Liger Kernel

    query_liger_kernel, key_liger_kernel = triton_mrope(
        query.clone(), key.clone(), cos, sin, mrope_section, head_dim)

    torch.testing.assert_close(query_native,
                               query_liger_kernel,
                               atol=atol,
                               rtol=rtol)
    torch.testing.assert_close(key_native,
                               key_liger_kernel,
                               atol=atol,
                               rtol=rtol)
