# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.model_executor.layers.lightning_attn import (
    lightning_attention, lightning_attention2_parallel,
    linear_decode_forward_triton)
from vllm.platforms import current_platform

NUM_HEADS = [4, 8]
HEAD_SIZES = [64, 128]
BATCH_SIZES = [1, 2]
SEQ_LENGTHS = [16, 128]
DTYPES = [torch.float16, torch.bfloat16]


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_lightning_attention(
    batch_size: int,
    num_heads: int,
    head_size: int,
    seq_len: int,
    dtype: torch.dtype,
):
    torch.set_default_device("cuda")
    current_platform.seed_everything(0)

    q = torch.randn(batch_size, num_heads, seq_len, head_size, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_size, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_size, dtype=dtype)
    ed = torch.rand(num_heads, device="cuda")

    output, kv = lightning_attention(q, k, v, ed)

    assert output.shape == (batch_size, num_heads, seq_len, head_size)
    assert kv.shape[0] == batch_size
    assert kv.shape[1] == num_heads

    output2, kv2 = lightning_attention2_parallel(q, k, v, ed)

    torch.testing.assert_close(output, output2, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(kv, kv2, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_lightning_attention_with_kv_history(
    batch_size: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
):
    torch.set_default_device("cuda")
    current_platform.seed_everything(0)

    seq_len = 32

    q = torch.randn(batch_size, num_heads, seq_len, head_size, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_size, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_size, dtype=dtype)
    ed = torch.rand(num_heads, device="cuda")

    kv_history = torch.randn(batch_size,
                             num_heads,
                             head_size,
                             head_size,
                             dtype=torch.float32,
                             device="cuda")

    output, kv = lightning_attention(q, k, v, ed, kv_history=kv_history)

    assert output.shape == (batch_size, num_heads, seq_len, head_size)
    assert kv.shape[0] == batch_size
    assert kv.shape[1] == num_heads


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_linear_decode_forward_triton(
    batch_size: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
):
    torch.set_default_device("cuda")
    current_platform.seed_everything(0)

    q = torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)
    k = torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)
    v = torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)

    kv_caches = torch.randn(batch_size,
                            num_heads,
                            head_size,
                            head_size,
                            dtype=dtype,
                            device="cuda")

    slope_rate = torch.rand(num_heads, device="cuda")

    slot_idx = torch.arange(batch_size, device="cuda")

    output = linear_decode_forward_triton(q, k, v, kv_caches, slope_rate,
                                          slot_idx)

    assert output.shape == (batch_size, num_heads * head_size)


@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_linear_decode_forward_triton_with_padding(
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
):
    torch.set_default_device("cuda")
    current_platform.seed_everything(0)

    batch_size = 4

    q = torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)
    k = torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)
    v = torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)

    kv_caches = torch.randn(batch_size,
                            num_heads,
                            head_size,
                            head_size,
                            dtype=dtype,
                            device="cuda")

    slope_rate = torch.rand(num_heads, device="cuda")

    slot_idx = torch.tensor([0, 1, -1, 2], device="cuda")

    output = linear_decode_forward_triton(q, k, v, kv_caches, slope_rate,
                                          slot_idx)

    assert output.shape == (batch_size, num_heads * head_size)
