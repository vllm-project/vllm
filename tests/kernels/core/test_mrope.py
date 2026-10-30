# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import NamedTuple

import pytest
import torch

from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.mrope import (
    apply_interleaved_rope,
    triton_mrope,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.config import get_config
from vllm.utils.torch_utils import set_random_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_test_data(
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    max_position_embeddings: int,
    dtype: torch.dtype,
    device: torch.device,
):
    """Generate test data for given configuration."""
    set_random_seed(42)
    # Create 2D positions (3, num_tokens) for multimodal case
    positions = torch.randint(
        0, max_position_embeddings // 4, (3, num_tokens), device=device
    )

    # Create query and key tensors
    query = torch.randn(num_tokens, num_q_heads * head_size, dtype=dtype, device=device)
    key = torch.randn(num_tokens, num_kv_heads * head_size, dtype=dtype, device=device)

    return positions, query, key


class MRoPETestInfo(NamedTuple):
    model_name: str
    is_neox_style: bool = True
    # https://github.com/pytorch/pytorch/blob/main/torch/testing/_comparison.py#L1317
    atol: float = 1e-2
    rtol: float = 1.6e-2
    marks: list[pytest.MarkDecorator] = []


MODELS_TO_TEST = [
    MRoPETestInfo(
        model_name="zai-org/GLM-4.1V-9B-Thinking",
        is_neox_style=False,
    ),
    MRoPETestInfo(model_name="Qwen/Qwen2-VL-7B-Instruct"),
    MRoPETestInfo(model_name="Qwen/Qwen2-VL-72B-Instruct"),
    MRoPETestInfo(model_name="Qwen/Qwen2.5-VL-72B-Instruct"),
    MRoPETestInfo(model_name="Qwen/Qwen3-VL-4B-Instruct"),
    MRoPETestInfo(model_name="Qwen/Qwen3-VL-30B-A3B-Instruct"),
]

num_tokens_list = [11, 8192]


def test_apply_interleaved_rope():
    mrope_section = [3, 1, 1]
    x = torch.tensor(
        [
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
            [[10, 11, 12, 13, 14], [15, 16, 17, 18, 19]],
            [[20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
        ]
    )

    result = apply_interleaved_rope(x, mrope_section)

    expected = torch.tensor([[0, 11, 22, 3, 4], [5, 16, 27, 8, 9]])
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Skipping CUDA/ROCm only test."
)
def test_apply_interleaved_rope_torch_compile():
    mrope_section = [24, 20, 20]
    num_tokens = 8192
    rotary_dim = sum(mrope_section) * 2
    cache = torch.randn(
        3,
        num_tokens,
        rotary_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    x = cache[..., : rotary_dim // 2]

    expected = apply_interleaved_rope(x, mrope_section)
    compiled_fn = torch.compile(
        apply_interleaved_rope,
        backend="inductor",
        fullgraph=True,
    )

    result = compiled_fn(x, mrope_section)

    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Skipping CUDA/ROCm only tests."
)
@pytest.mark.parametrize(
    "model_info, model_name",
    [
        pytest.param(test_config, test_config.model_name, marks=test_config.marks)
        for test_config in MODELS_TO_TEST
    ],
)
@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_tokens", num_tokens_list)
def test_mrope(
    default_vllm_config,
    model_name: str,
    model_info: MRoPETestInfo,
    tp_size: int,
    dtype: torch.dtype,
    num_tokens: int,
):
    atol = model_info.atol
    rtol = model_info.rtol

    config = get_config(model_name, False).get_text_config()

    # get the model config
    total_num_kv_heads = config.num_key_value_heads
    total_num_heads = config.num_attention_heads
    num_heads = total_num_heads // tp_size
    num_kv_heads = max(1, total_num_kv_heads // tp_size)
    head_dim = (
        config.head_dim
        if hasattr(config, "head_dim")
        else config.hidden_size // total_num_heads
    )
    is_neox_style = model_info.is_neox_style

    max_position = config.max_position_embeddings

    mrope_helper_class = get_rope(
        head_size=head_dim,
        max_position=max_position,
        is_neox_style=is_neox_style,
        rope_parameters=config.rope_parameters,
        dtype=dtype,
    ).to(device=device)

    # create q k v input tensors
    # create rotary pos emb input tensors
    positions, query, key = generate_test_data(
        num_tokens, num_heads, num_kv_heads, head_dim, max_position, dtype, device
    )

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

    # Strided q/k views of a packed buffer (the MiniMax M3 ViT usage):
    # bitwise-identical to the contiguous run, neighboring columns untouched.
    pad = 16
    packed = torch.zeros(
        num_tokens,
        query.shape[1] + key.shape[1] + pad,
        dtype=dtype,
        device=device,
    )
    packed[:, : query.shape[1]] = query
    packed[:, query.shape[1] : query.shape[1] + key.shape[1]] = key
    sentinel = packed[:, -pad:].clone()
    q_strided = packed[:, : query.shape[1]]
    k_strided = packed[:, query.shape[1] : query.shape[1] + key.shape[1]]
    mrope_helper_class.forward_cuda(positions, q_strided, k_strided)
    torch.testing.assert_close(q_strided, query_cuda, atol=0, rtol=0)
    torch.testing.assert_close(k_strided, key_cuda, atol=0, rtol=0)
    torch.testing.assert_close(packed[:, -pad:], sentinel, atol=0, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Skipping CUDA/ROCm only tests."
)
def test_mrope_row_offset_int64():
    # MiniMax M3 ViT worst case: a max-size video item packs ~768k patches,
    # and num_tokens * packed-qkv row stride (3840) exceeds 2^31 at 559,241
    # rows. Row-base math must be int64 or those rows wrap to negative
    # offsets (illegal memory access at startup profiling).
    set_random_seed(0)
    n_qh, n_kh, hd, rd = 16, 16, 80, 78
    mrope_section = [13, 13, 13]
    row_stride = (n_qh + n_kh + n_kh) * hd  # packed qkv row
    num_tokens = 2**31 // row_stride + 8
    free, _ = current_platform.mem_get_info()
    if free < 2 * num_tokens * row_stride * 2:
        pytest.skip("not enough free device memory")

    packed = torch.empty(num_tokens, row_stride, dtype=torch.bfloat16, device=device)
    q_fill = torch.randn(n_qh * hd, dtype=torch.bfloat16, device=device)
    k_fill = torch.randn(n_kh * hd, dtype=torch.bfloat16, device=device)
    packed[:, : n_qh * hd] = q_fill
    packed[:, n_qh * hd : (n_qh + n_kh) * hd] = k_fill
    half_rd = rd // 2
    cos = torch.randn(3, num_tokens, half_rd, dtype=torch.float32, device=device)
    sin = torch.randn(3, num_tokens, half_rd, dtype=torch.float32, device=device)
    q = packed[:, : n_qh * hd]
    k = packed[:, n_qh * hd : (n_qh + n_kh) * hd]
    triton_mrope(q, k, cos, sin, mrope_section, hd, rd, False, True)

    t_end = mrope_section[0]
    h_end = t_end + mrope_section[1]
    for r in (0, num_tokens - 8, num_tokens - 1):
        cos_row = torch.cat(
            [cos[0, r, :t_end], cos[1, r, t_end:h_end], cos[2, r, h_end:]]
        )
        sin_row = torch.cat(
            [sin[0, r, :t_end], sin[1, r, t_end:h_end], sin[2, r, h_end:]]
        )
        for fill, out in ((q_fill, q[r]), (k_fill, k[r])):
            row = fill.view(-1, hd).float()
            r1, r2 = row[:, :half_rd], row[:, half_rd:rd]
            exp = torch.cat(
                [r1 * cos_row - r2 * sin_row, r2 * cos_row + r1 * sin_row, row[:, rd:]],
                dim=1,
            )
            torch.testing.assert_close(
                out, exp.view(-1).to(torch.bfloat16), atol=1e-2, rtol=1.6e-2
            )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Skipping CUDA/ROCm only tests."
)
@pytest.mark.parametrize(
    "model_info, model_name",
    [
        pytest.param(test_config, test_config.model_name, marks=test_config.marks)
        for test_config in MODELS_TO_TEST
    ],
)
@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_tokens", num_tokens_list)
def test_mrope_torch_compile_tracing(
    default_vllm_config,
    model_name: str,
    model_info: MRoPETestInfo,
    tp_size: int,
    dtype: torch.dtype,
    num_tokens: int,
):
    atol = model_info.atol
    rtol = model_info.rtol

    config = get_config(model_name, False).get_text_config()

    # get the model config
    total_num_kv_heads = config.num_key_value_heads
    total_num_heads = config.num_attention_heads
    num_heads = total_num_heads // tp_size
    num_kv_heads = max(1, total_num_kv_heads // tp_size)
    head_dim = (
        config.head_dim
        if hasattr(config, "head_dim")
        else config.hidden_size // total_num_heads
    )
    is_neox_style = model_info.is_neox_style
    max_position = config.max_position_embeddings

    mrope_helper_class = get_rope(
        head_size=head_dim,
        max_position=max_position,
        is_neox_style=is_neox_style,
        rope_parameters=config.rope_parameters,
        dtype=dtype,
    ).to(device=device)

    # Generate test data
    positions, query, key = generate_test_data(
        num_tokens, num_heads, num_kv_heads, head_dim, max_position, dtype, device
    )

    # Create a wrapper that makes the in-place function appear functional
    def functional_forward_cuda(pos, q, k):
        """Wrapper that converts in-place operation to functional style.

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
        compiled_forward_cuda = torch.compile(
            functional_forward_cuda,
            fullgraph=True,
            backend="inductor",
            mode="reduce-overhead",
            dynamic=False,
        )

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
        torch.testing.assert_close(
            query_compiled_cuda, query_cuda, atol=atol, rtol=rtol
        )
        torch.testing.assert_close(key_compiled_cuda, key_cuda, atol=atol, rtol=rtol)
        torch.testing.assert_close(
            query_compiled_cuda, query_native, atol=atol, rtol=rtol
        )
        torch.testing.assert_close(key_compiled_cuda, key_native, atol=atol, rtol=rtol)

        print("✓ forward_cuda successfully traced with torch.compile inductor")

    except Exception as e:
        pytest.fail(f"forward_cuda failed to trace with torch.compile inductor: {e}")
