# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for fused MLA KV-cache write and RoPE fused kernel
"""

import random

import pytest
import torch

from tests.kernels.allclose_default import get_default_atol, get_default_rtol
from tests.kernels.utils import DEFAULT_OPCHECK_TEST_UTILS, opcheck
from vllm import _custom_ops as ops
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed


@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16, torch.float])
@pytest.mark.parametrize("is_neox_style", [False, True])
@pytest.mark.parametrize("seq_len", [11, 42])
@pytest.mark.parametrize("qk_rope_head_dim", [64, 128])
@pytest.mark.parametrize("num_q_heads", [128])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("num_blocks", [64])
@pytest.mark.parametrize("block_size", [16, 64, 256])
@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize(
    "device", [f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)]
)
@torch.inference_mode()
def test_concat_and_cache_mla_rope_fused(
    default_vllm_config,
    dtype: torch.dtype,
    is_neox_style: bool,
    seq_len: int,
    qk_rope_head_dim: int,
    num_q_heads: int,
    kv_cache_dtype: str,
    kv_lora_rank: int,
    num_blocks: int,
    block_size: int,
    seed: int,
    device: str,
    max_position: int = 8192,
    base: float = 10000,
) -> None:
    set_random_seed(seed)
    torch.set_default_device(device)

    rope = RotaryEmbedding(
        qk_rope_head_dim,
        qk_rope_head_dim,
        max_position,
        base,
        is_neox_style,
        torch.float32,
    )

    rope = rope.to(dtype=dtype, device=torch.get_default_device())

    positions = torch.randint(0, max_position, (seq_len,))

    query = torch.randn(seq_len, num_q_heads, qk_rope_head_dim, dtype=dtype)
    key = torch.randn(seq_len, 1, qk_rope_head_dim + kv_lora_rank, dtype=dtype)

    k_pe = torch.flatten(key[..., :qk_rope_head_dim], start_dim=1).to(device=device)
    kv_c = torch.flatten(key[..., qk_rope_head_dim:], start_dim=1).to(device=device)

    if current_platform.is_rocm():
        # We use forward_hip for the same numerics as the fused custom kernel on ROCm
        # when dtype is FP16. The torch-native implementation implicitly upcasts
        # FP16 x FP16 multiplications to FP32 before downcasting them, which leads
        # to notable output divergences.
        # Clone the tensors because the implementation modifies them in-place
        ref_q_pe, ref_k_pe = rope.forward_hip(positions, query.clone(), k_pe.clone())
    else:
        # NOTE(woosuk): The reference implementation should be executed first
        # because the custom kernel is in-place.
        ref_q_pe, ref_k_pe = rope.forward_native(positions, query, k_pe)
    assert ref_k_pe is not None

    ref_k_pe = torch.flatten(ref_k_pe, start_dim=1).to(device=device)
    ref_k_rope = ref_k_pe[..., :qk_rope_head_dim]

    total_available_slots = num_blocks * block_size
    total_needed_slots = seq_len
    assert total_available_slots >= total_needed_slots, "Not enough kv slots!"

    slot_mapping_lst = random.sample(range(total_available_slots), total_needed_slots)
    slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long, device=device)

    entry_size = kv_lora_rank + qk_rope_head_dim

    kv_cache_scale = torch.tensor([0.1], dtype=torch.float32, device=device)

    kv_cache = torch.zeros(
        num_blocks,
        block_size,
        entry_size,
        dtype=torch.uint8 if kv_cache_dtype == "fp8" else dtype,
        device=device,
    )

    ref_temp = torch.zeros(*kv_cache.shape, dtype=dtype, device=device)

    for i in range(seq_len):
        slot = slot_mapping[i].item()
        block_idx = slot // block_size
        block_offset = slot % block_size
        ref_temp[block_idx, block_offset] = torch.cat((kv_c[i], ref_k_rope[i]), -1)

    if kv_cache_dtype == "fp8":
        ref_kv_cache = torch.empty_like(ref_temp, dtype=kv_cache.dtype)
        ops.convert_fp8(
            ref_kv_cache, ref_temp, kv_cache_scale.item(), kv_dtype=kv_cache_dtype
        )
    else:
        ref_kv_cache = ref_temp

    opcheck(
        torch.ops._C_cache_ops.concat_and_cache_mla_rope_fused,
        (
            positions,
            query,
            k_pe,
            kv_c,
            rope.cos_sin_cache,
            is_neox_style,
            slot_mapping,
            kv_cache,
            kv_cache_dtype,
            kv_cache_scale,
        ),
        test_utils=DEFAULT_OPCHECK_TEST_UTILS,
    )

    ops.concat_and_cache_mla_rope_fused(
        positions,
        query,
        k_pe,
        kv_c,
        rope.cos_sin_cache,
        is_neox_style,
        slot_mapping,
        kv_cache,
        kv_cache_dtype,
        kv_cache_scale,
    )

    if kv_cache_dtype == "fp8":
        result_temp = torch.empty_like(kv_cache, dtype=torch.float16)
        ops.convert_fp8(
            result_temp,
            kv_cache.contiguous(),
            kv_cache_scale.item(),
            kv_dtype=kv_cache_dtype,
        )
        expected_temp = torch.empty_like(ref_kv_cache, dtype=torch.float16)
        ops.convert_fp8(
            expected_temp, ref_kv_cache, kv_cache_scale.item(), kv_dtype=kv_cache_dtype
        )
        torch.testing.assert_close(result_temp, expected_temp, atol=0.001, rtol=0.1)
    else:
        torch.testing.assert_close(kv_cache, ref_kv_cache)

    torch.testing.assert_close(
        query, ref_q_pe, atol=get_default_atol(query), rtol=get_default_rtol(query)
    )
