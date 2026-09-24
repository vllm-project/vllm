# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import NamedTuple

import pytest
import torch

from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.mrope import apply_interleaved_rope
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


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Skipping CUDA/ROCm only tests."
)
@pytest.mark.parametrize("is_neox_style", [True, False])
@pytest.mark.parametrize("num_tokens", num_tokens_list)
def test_triton_mrope_strided_qk(is_neox_style: bool, num_tokens: int):
    """triton_mrope must rotate strided q/k views (e.g. of a packed qkv
    tensor) exactly as it rotates contiguous tensors, without touching the
    value rows. Covers the MiniMax M3 ViT usage: partial rotation
    (rotary_dim < head_size) with fp32 cos/sin."""
    from vllm.model_executor.layers.rotary_embedding.mrope import triton_mrope

    set_random_seed(42)
    num_heads = 16
    head_size = 80
    rotary_dim = 78
    mrope_section = [13, 13, 13]
    assert sum(mrope_section) == rotary_dim // 2

    qkv = torch.randn(
        num_tokens, 3, num_heads, head_size, dtype=torch.bfloat16, device=device
    )
    v_before = qkv[:, 2].clone()
    q = qkv[:, 0].reshape(num_tokens, -1)
    k = qkv[:, 1].reshape(num_tokens, -1)
    assert q.stride(-1) == 1 and not q.is_contiguous()

    cos = torch.randn(
        3, num_tokens, rotary_dim // 2, dtype=torch.float32, device=device
    )
    sin = torch.randn(
        3, num_tokens, rotary_dim // 2, dtype=torch.float32, device=device
    )

    q_ref = q.contiguous()
    k_ref = k.contiguous()
    triton_mrope(
        q_ref,
        k_ref,
        cos,
        sin,
        mrope_section,
        head_size,
        rotary_dim,
        mrope_interleaved=False,
        is_neox_style=is_neox_style,
    )
    triton_mrope(
        q,
        k,
        cos,
        sin,
        mrope_section,
        head_size,
        rotary_dim,
        mrope_interleaved=False,
        is_neox_style=is_neox_style,
    )

    torch.testing.assert_close(q, q_ref, rtol=0, atol=0)
    torch.testing.assert_close(k, k_ref, rtol=0, atol=0)
    # v rows share storage with q/k and must be untouched
    torch.testing.assert_close(qkv[:, 2], v_before, rtol=0, atol=0)


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
