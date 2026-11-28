# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from itertools import product

import pytest
import torch

from vllm import _custom_ops as ops
from vllm import ir
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding, get_rope
from vllm.utils.torch_utils import set_random_seed

IS_NEOX_STYLE = [True, False]
DTYPES = [torch.bfloat16, torch.float]
HEAD_SIZES = [64, 80, 120, 256]
ROTARY_DIMS = [None, 32]  # None means rotary dim == head size
NUM_HEADS = [17]  # Arbitrary values for testing
BATCH_SIZES = [5]  # Arbitrary values for testing
SEQ_LENS = [11, 8192]  # Arbitrary values for testing
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.accelerator.device_count() == 1 else 2)
]
USE_KEY = [True, False]


def _get_flat_tensor_shape(
    batch_size: int, seq_len: int, num_heads: int, head_size: int
) -> tuple[int, ...]:
    return (batch_size, seq_len, num_heads * head_size)


# For testing sliced tensors
def _get_padded_tensor_shape(
    batch_size: int, seq_len: int, num_heads: int, head_size: int
) -> tuple[int, ...]:
    return (batch_size, seq_len, num_heads, head_size + 64)


def _get_batch_tensor_shape(
    batch_size: int, seq_len: int, num_heads: int, head_size: int
) -> tuple[int, ...]:
    return (batch_size, seq_len, num_heads, head_size)


TENSORS_SHAPES_FN = [
    _get_batch_tensor_shape,
    _get_flat_tensor_shape,
    _get_padded_tensor_shape,
]


@pytest.mark.parametrize("layout", ["transposed", "feature_strided"])
@pytest.mark.parametrize("strided_arg", ["query", "key", "query_without_key"])
def test_rotary_embedding_rejects_feature_stride(layout: str, strided_arg: str) -> None:
    """Reject unsupported q/k views before modifying either input."""
    num_tokens, num_heads, head_size = 4, 2, 64
    shape = (
        (num_heads * head_size, num_tokens)
        if layout == "transposed"
        else (num_tokens, num_heads, head_size * 4)
    )
    storage = torch.ones(shape, device="cuda", dtype=torch.float32)
    strided = storage.T if layout == "transposed" else storage[..., ::4]
    query = strided.contiguous() if strided_arg == "key" else strided
    key = strided if strided_arg == "key" else strided.contiguous()
    if strided_arg == "query_without_key":
        key = None
    positions = torch.arange(num_tokens, device="cuda")
    cache = torch.zeros((num_tokens, head_size), device="cuda")
    invalid_arg = "key" if strided_arg == "key" else "query"

    with pytest.raises(RuntimeError, match=f"{invalid_arg} must have stride 1"):
        ops.rotary_embedding(positions, query, key, head_size, cache, True)

    torch.testing.assert_close(storage, torch.ones_like(storage), rtol=0, atol=0)
    torch.testing.assert_close(query, torch.ones_like(query), rtol=0, atol=0)
    if key is not None:
        torch.testing.assert_close(key, torch.ones_like(key), rtol=0, atol=0)


@pytest.mark.parametrize("is_neox_style", IS_NEOX_STYLE)
@pytest.mark.parametrize("tensor_shape_fn", TENSORS_SHAPES_FN)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("rotary_dim", ROTARY_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("use_key", USE_KEY)
@torch.inference_mode()
def test_rotary_embedding(
    default_vllm_config,
    is_neox_style: bool,
    tensor_shape_fn: Callable[[int, int, int, int], tuple[int, ...]],
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_size: int,
    rotary_dim: int | None,
    dtype: torch.dtype,
    seed: int,
    device: str,
    use_key: bool,
    max_position: int = 8192,
    rope_theta: float = 10000,
) -> None:
    if rotary_dim is None:
        rotary_dim = head_size

    set_random_seed(seed)
    torch.set_default_device(device)
    if rotary_dim is None:
        rotary_dim = head_size
    rope_parameters = {
        "rope_type": "default",
        "rope_theta": rope_theta,
        "partial_rotary_factor": rotary_dim / head_size,
    }
    rope = get_rope(head_size, max_position, is_neox_style, rope_parameters)
    rope = rope.to(dtype=dtype, device=torch.get_default_device())

    positions = torch.randint(0, max_position, (batch_size, seq_len))
    query_shape = tensor_shape_fn(batch_size, seq_len, num_heads, head_size)
    # slice tensor if required, noop otherwise
    query = torch.randn(query_shape, dtype=dtype)[..., :head_size]
    key = torch.randn_like(query)[..., :head_size] if use_key else None

    # NOTE(woosuk): The reference implementation should be executed first
    # because the custom kernel is in-place.
    with (
        ir.ops.rotary_embedding.set_priority(["native"]),
        ir.ops.rotary_embedding_query_only.set_priority(["native"]),
    ):
        ref_query, ref_key = rope.forward_native(positions, query, key)
    # Base RoPE dispatches through vLLM IR; select the kernel explicitly.
    with (
        ir.ops.rotary_embedding.set_priority(["vllm_c", "native"]),
        ir.ops.rotary_embedding_query_only.set_priority(["vllm_c", "native"]),
    ):
        out_query, out_key = rope.forward(positions, query, key)
    # The kernel computes in fp32 while native rounds every op, so compare
    # with the RoPE IR op tolerances.
    tol = ir.ops.rotary_embedding.get_tolerance(out_query.dtype)
    torch.testing.assert_close(out_query, ref_query, **tol)
    if use_key:
        torch.testing.assert_close(out_key, ref_key, **tol)
    else:
        assert ref_key is None and out_key is None, "expected returned key to be None"


@torch.inference_mode()
def test_rope_module_cache(default_vllm_config):
    MAX_POSITIONS = [123, 1234]
    ROPE_THETAS = [10000, 1000000]
    ROPE_PARAMETERS = (
        {"rope_type": "default"},
        {"rope_type": "linear", "factor": (1,)},
        {"rope_type": "dynamic", "factor": 1},
    )
    settings = (
        HEAD_SIZES,
        ROTARY_DIMS,
        MAX_POSITIONS,
        ROPE_THETAS,
        IS_NEOX_STYLE,
        ROPE_PARAMETERS,
        DTYPES,
    )
    rope_setting_id_map: dict[str, int] = {}
    for setting in product(*settings):
        (
            head_size,
            rotary_dim,
            max_position,
            rope_theta,
            is_neox_style,
            rope_parameters,
            dtype,
        ) = setting
        if rotary_dim is None:
            rotary_dim = head_size
        rope_parameters["rope_theta"] = rope_theta
        rope_parameters["partial_rotary_factor"] = rotary_dim / head_size
        rope = get_rope(
            head_size,
            max_position,
            is_neox_style,
            rope_parameters,
            dtype,
        )
        # different settings cannot share the same rope module
        assert id(rope) not in rope_setting_id_map.values()
        # The AITER bf16 cache copy is bf16 by design.
        assert all(
            x.dtype == dtype
            for name, x in rope.named_buffers()
            if name != "cos_sin_cache_bf16"
        )
        assert all(x.dtype == dtype for x in rope.parameters())
        rope_setting_id_map[str(setting)] = id(rope)

    for setting in product(*settings):
        (
            head_size,
            rotary_dim,
            max_position,
            rope_theta,
            is_neox_style,
            rope_parameters,
            dtype,
        ) = setting
        if rotary_dim is None:
            rotary_dim = head_size
        rope_parameters["rope_theta"] = rope_theta
        rope_parameters["partial_rotary_factor"] = rotary_dim / head_size
        rope = get_rope(
            head_size,
            max_position,
            is_neox_style,
            rope_parameters,
            dtype,
        )
        # check if cache take effect
        assert id(rope) == rope_setting_id_map[str(setting)]


def _assert_gptj_fp64_rotation(got, query, table, positions):
    cos, sin = table.double()[positions].chunk(2, dim=-1)
    cos, sin = cos[:, None], sin[:, None]
    x = query.double().view(query.shape[0], -1, 64)
    even, odd = x[..., 0::2], x[..., 1::2]
    ref = torch.stack([even * cos - odd * sin, odd * cos + even * sin], -1)
    ref = ref.flatten(-2)
    ulp = torch.ldexp(
        torch.ones_like(ref), torch.frexp(ref.abs().float())[1].double() - 8
    )
    slack = 2**-22 * (even.abs() + odd.abs()).repeat_interleave(2, dim=-1)
    got = got.reshape(query.shape[0], -1, 64)
    err = ((got.double() - ref).abs() / (ulp + slack)).max().item()
    assert err <= 1.0, f"rotation error is {err:.2f} bf16 ulp"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float32])
def test_rotary_embedding_fp32_cache_with_bf16_query(cache_dtype, default_vllm_config):
    """Preserve an FP32 table without changing the output dtype."""
    torch.manual_seed(0)
    rope = RotaryEmbedding(64, 64, 128, 10000.0, False, torch.float32)
    table = torch.randn(128, 64, dtype=cache_dtype, device="cuda")
    rope.cos_sin_cache = table
    saved_table = table.clone()
    positions = torch.arange(17, device="cuda")
    query = torch.randn(17, 128, dtype=torch.bfloat16, device="cuda")
    key = torch.randn(17, 64, dtype=torch.bfloat16, device="cuda")

    with ir.ops.rotary_embedding.set_priority(["vllm_c", "native"]):
        out_query, out_key = rope.forward_cuda(positions, query.clone(), key.clone())
    assert out_key is not None
    ref_query, ref_key = query.clone(), key.clone()
    ops.rotary_embedding(positions, ref_query, ref_key, 64, saved_table, False)
    torch.testing.assert_close(out_query, ref_query, atol=0, rtol=0)
    torch.testing.assert_close(out_key, ref_key, atol=0, rtol=0)
    assert out_query.dtype == query.dtype
    assert out_key.dtype == key.dtype
    assert rope.cos_sin_cache is table
    assert rope.cos_sin_cache.dtype == cache_dtype
    assert torch.equal(table, saved_table)

    if cache_dtype == torch.float32:
        _assert_gptj_fp64_rotation(out_query, query, saved_table, positions)
        _assert_gptj_fp64_rotation(out_key, key, saved_table, positions)
