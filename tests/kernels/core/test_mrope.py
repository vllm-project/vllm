# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from transformers import AutoConfig

from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.platforms import current_platform

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
    "Qwen/Qwen2-VL-7B-Instruct": [1, 2],
    "Qwen/Qwen2-VL-72B-Instruct": [1, 2],
    "Qwen/Qwen2.5-VL-72B-Instruct": [1, 2],
    "zai-org/GLM-4.1V-9B-Thinking": [1, 2],
}

# https://github.com/pytorch/pytorch/blob/main/torch/testing/_comparison.py#L1317
dtype_atol_rtol_list = [
    [torch.bfloat16, 1e-2, 1.6e-2],
]

num_tokens_list = [11, 8192]


@pytest.mark.skipif(not current_platform.is_cuda_alike(),
                    reason="Skipping CUDA/ROCm only tests.")
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

    rope_theta = config.rope_theta
    max_position = config.max_position_embeddings
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    rotary_dim = int(head_dim * partial_rotary_factor)

    mrope_helper_class = get_rope(
        head_size=head_dim,
        rotary_dim=rotary_dim,
        max_position=max_position,
        base=rope_theta,
        is_neox_style=is_neox_style,
        rope_scaling=config.rope_scaling,
        dtype=dtype,
    ).to(device=device)

    # create q k v input tensors
    # create rotary pos emb input tensors
    positions, query, key = generate_test_data(num_tokens, num_heads,
                                               num_kv_heads, head_dim,
                                               max_position, dtype, device)

    query_native, key_native = mrope_helper_class.forward_native(
        positions,
        query.clone(),
        key.clone(),
    )

    query_cuda, key_cuda = mrope_helper_class.forward_cuda(
        positions,
        query.clone(),
        key.clone(),
    )

    torch.testing.assert_close(query_native, query_cuda, atol=atol, rtol=rtol)
    torch.testing.assert_close(key_native, key_cuda, atol=atol, rtol=rtol)


@pytest.mark.skipif(not current_platform.is_cuda_alike(),
                    reason="Skipping CUDA/ROCm only tests.")
@pytest.mark.parametrize(
    "model_name, tp_size",
    unroll_model_tp_dict({
        "Qwen/Qwen2-VL-7B-Instruct": [1, 2],
        "zai-org/GLM-4.1V-9B-Thinking": [1, 2]
    }))
@pytest.mark.parametrize("dtype, atol, rtol", dtype_atol_rtol_list)
@pytest.mark.parametrize("num_tokens", [4])
def test_mrope_torch_compile_tracing(model_name, tp_size, dtype, atol, rtol,
                                     num_tokens):
    config = AutoConfig.from_pretrained(model_name)

    # get the model config
    total_num_kv_heads = config.num_key_value_heads
    total_num_heads = config.num_attention_heads
    num_heads = total_num_heads // tp_size
    num_kv_heads = max(1, total_num_kv_heads // tp_size)
    head_dim = config.hidden_size // total_num_heads
    is_neox_style = True
    rope_theta = config.rope_theta
    max_position = config.max_position_embeddings
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    rotary_dim = int(head_dim * partial_rotary_factor)

    mrope_helper_class = get_rope(
        head_size=head_dim,
        rotary_dim=rotary_dim,
        max_position=max_position,
        base=rope_theta,
        is_neox_style=is_neox_style,
        rope_scaling=config.rope_scaling,
        dtype=dtype,
    ).to(device=device)

    # Generate test data
    positions, query, key = generate_test_data(num_tokens, num_heads,
                                               num_kv_heads, head_dim,
                                               max_position, dtype, device)

    # Create a wrapper that makes the in-place function appear functional
    def functional_forward_cuda(pos, q, k):
        """Wrapper that converts in-place operation to functional style

        CUDA Graph does not support in-place operations.
        This wrapper creates working copies of the 
        input tensors and modifies them.
        """
        q_work = q.clone()  # Create working copies
        k_work = k.clone()
        # Your in-place function modifies q_work and k_work
        mrope_helper_class.forward_cuda(pos, q_work, k_work)
        return q_work, k_work  # Return the modified tensors

    # Get reference results
    query_native, key_native = mrope_helper_class.forward_native(
        positions,
        query.clone(),
        key.clone(),
    )

    try:
        compiled_forward_cuda = torch.compile(functional_forward_cuda,
                                              fullgraph=True,
                                              backend="inductor",
                                              mode="reduce-overhead",
                                              dynamic=False)

        # Run compiled version
        query_compiled_cuda, key_compiled_cuda = compiled_forward_cuda(
            positions,
            query,
            key,
        )

        # Run original version for comparison
        query_cuda = query.clone()
        key_cuda = key.clone()
        mrope_helper_class.forward_cuda(positions, query_cuda, key_cuda)

        # Verify results
        torch.testing.assert_close(query_compiled_cuda,
                                   query_cuda,
                                   atol=atol,
                                   rtol=rtol)
        torch.testing.assert_close(key_compiled_cuda,
                                   key_cuda,
                                   atol=atol,
                                   rtol=rtol)
        torch.testing.assert_close(query_compiled_cuda,
                                   query_native,
                                   atol=atol,
                                   rtol=rtol)
        torch.testing.assert_close(key_compiled_cuda,
                                   key_native,
                                   atol=atol,
                                   rtol=rtol)

        print("✓ forward_cuda successfully traced with torch.compile inductor")

    except Exception as e:
        pytest.fail(
            f"forward_cuda failed to trace with torch.compile inductor: {e}")
