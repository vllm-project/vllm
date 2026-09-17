# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for fused MLA KV-cache write and RoPE fused kernel."""

import random

import pytest
import torch

from tests.kernels.allclose_default import get_default_atol, get_default_rtol
from tests.kernels.utils import DEFAULT_OPCHECK_TEST_UTILS, opcheck
from vllm import _custom_ops as ops
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed


def assert_rope_close_rocm(
    actual: torch.Tensor,
    ref: torch.Tensor,
    rope_input: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    is_neox: bool,
    *,
    max_fail_fraction: float | None = None,
    atol: float | None = None,
    rtol: float | None = None,
) -> None:
    """assert_close for RoPE fused-vs-reference on ROCm, with a per-element
    product-ULP allowance for catastrophic-cancellation elements.

    The fused and reference kernels may contract the RoPE multiply-add into an
    FMA with either product pre-rounded to the working dtype. The two valid
    orderings differ by at most 2 ULPs of the product at that element, which
    cancellation (x*cos ~= -y*sin) can amplify past the default tolerance.

    Tier 1: |actual - ref| <= atol + rtol * |ref|  (default tolerance).
    Tier 2: elements failing tier 1 must each satisfy
    |actual_e - ref_e| <= 2 * eps * |product_e|, where product_e is the larger
    of the two products combined at that element, and their fraction must be
    <= max_fail_fraction (1e-4 for fp16; 1e-2 for bf16, whose coarser eps
    puts 1-ULP divergences past the default tolerance far more often).
    """
    if atol is None:
        atol = get_default_atol(ref)
    if rtol is None:
        rtol = get_default_rtol(ref)
    if max_fail_fraction is None:
        max_fail_fraction = 1e-2 if ref.dtype == torch.bfloat16 else 1e-4
    eps = torch.finfo(ref.dtype).eps

    # per-element product magnitude, in the (seq, heads, rot_dim) layout
    seq_len, rot_dim = rope_input.shape[0], rope_input.shape[-1]
    half = rot_dim // 2
    inp = rope_input.reshape(seq_len, -1, rot_dim).to(torch.float32)
    cs = cos_sin_cache[positions].to(torch.float32)  # (seq_len, 2*half)
    cos = cs[:, :half].unsqueeze(1)  # broadcast over heads
    sin = cs[:, half:].unsqueeze(1)
    x, y = inp[..., :half], inp[..., half:]
    if is_neox:
        # out = [x*cos - y*sin, y*cos + x*sin]
        prod = torch.cat(
            [
                torch.maximum((x * cos).abs(), (y * sin).abs()),
                torch.maximum((y * cos).abs(), (x * sin).abs()),
            ],
            dim=-1,
        )
    else:
        # out[2i] = x[2i]*cos[i] - x[2i+1]*sin[i], interleaved
        xe, xo = inp[..., 0::2], inp[..., 1::2]
        prod = torch.stack(
            [
                torch.maximum((xe * cos).abs(), (xo * sin).abs()),
                torch.maximum((xo * cos).abs(), (xe * sin).abs()),
            ],
            dim=-1,
        ).flatten(-2)
    prod = prod.reshape(rope_input.shape)

    diff = (actual.to(torch.float32) - ref.to(torch.float32)).abs()
    tol = atol + rtol * ref.to(torch.float32).abs()
    fail = diff > tol
    n_fail = int(fail.sum())
    if n_fail == 0:
        return
    assert n_fail <= max_fail_fraction * diff.numel(), (
        f"{n_fail} elements beyond default tolerance "
        f"({n_fail / diff.numel():.2%} > {max_fail_fraction:.2%}); "
        f"max diff {diff.max().item()}"
    )
    excess = diff[fail] - 2 * eps * prod[fail]
    assert (excess <= 0).all(), (
        f"{int((excess > 0).sum())} cancellation elements exceed 2 product-ULPs; "
        f"worst excess {excess.max().item()}"
    )


@pytest.fixture
def default_vllm_config(monkeypatch):
    """Enable the AITER triton rope on ROCm for fp16-consistent numerics.

    The fused CUDA kernel runs native fp16 while forward_native upcasts to
    fp32, so on ROCm we route through the AITER triton rope (+rotary_embedding)
    to match. Its env gates are cached at import, hence refresh_env_variables().
    """
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config

    is_rocm = current_platform.is_rocm()
    if is_rocm:
        config = VllmConfig(
            compilation_config=CompilationConfig(custom_ops=["+rotary_embedding"])
        )
    else:
        config = VllmConfig()
    try:
        with monkeypatch.context() as m, set_current_vllm_config(config):
            if is_rocm:
                m.setenv("VLLM_ROCM_USE_AITER", "1")
                m.setenv("VLLM_ROCM_USE_AITER_TRITON_ROPE", "1")
                rocm_aiter_ops.refresh_env_variables()
            yield config
    finally:
        if is_rocm:
            rocm_aiter_ops.refresh_env_variables()


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
    "device",
    [f"cuda:{i}" for i in range(1 if torch.accelerator.device_count() == 1 else 2)],
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
    k_pe_orig = k_pe.clone()
    query_orig = query.clone()

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

    # On ROCm the AITER Triton rope diverges by up to 2 product-ULPs from the
    # fused kernel, which the default tolerance misses at the rare
    # catastrophic-cancellation elements. fp8 keeps its calibrated tolerance.
    rocm_neox = current_platform.is_rocm() and is_neox_style
    rocm_bf16 = current_platform.is_rocm() and dtype == torch.bfloat16
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
        torch.testing.assert_close(
            result_temp,
            expected_temp,
            atol=0.004 if rocm_bf16 else 0.001,
            rtol=0.15 if rocm_neox or rocm_bf16 else 0.1,
        )
    elif current_platform.is_rocm() and dtype in (torch.half, torch.bfloat16):
        # gather the rope region of the cache back to token order
        block_idx = slot_mapping // block_size
        block_offset = slot_mapping % block_size
        fused_k_rope = kv_cache[block_idx, block_offset, kv_lora_rank:]
        ref_k_rope_gathered = ref_kv_cache[block_idx, block_offset, kv_lora_rank:]
        assert_rope_close_rocm(
            fused_k_rope,
            ref_k_rope_gathered,
            k_pe_orig,
            rope.cos_sin_cache,
            positions,
            is_neox_style,
        )
        # the nope region is a byte-exact copy
        torch.testing.assert_close(
            kv_cache[..., :kv_lora_rank], ref_kv_cache[..., :kv_lora_rank]
        )
    else:
        torch.testing.assert_close(kv_cache, ref_kv_cache)

    if current_platform.is_rocm() and dtype in (torch.half, torch.bfloat16):
        assert_rope_close_rocm(
            query,
            ref_q_pe,
            query_orig,
            rope.cos_sin_cache,
            positions,
            is_neox_style,
        )
    else:
        torch.testing.assert_close(
            query,
            ref_q_pe,
            atol=get_default_atol(query),
            rtol=get_default_rtol(query),
        )
