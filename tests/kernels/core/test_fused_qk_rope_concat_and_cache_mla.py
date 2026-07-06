# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness test for the fused sparse-MLA decode Q-prep kernel.

``rocm_aiter_ops.fused_qk_rope_concat_and_cache_mla`` collapses the decode
RoPE + Q-concat + KV-concat + KV-cache-write into a single aiter kernel. This
test checks it against the split kernels it replaces:

  * q_out  == concat(ql_nope, RoPE(q_pe))            (nope-first)
  * kv_cache == concat_and_cache_mla(kv_c, RoPE(k_pe))

It is ROCm-only and skipped on other platforms.
"""

import random

import pytest
import torch

from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="fused_qk_rope_concat_and_cache_mla requires ROCm",
)


@pytest.fixture
def default_vllm_config(monkeypatch):

    from vllm.config import VllmConfig, set_current_vllm_config

    config = VllmConfig()
    with monkeypatch.context() as m, set_current_vllm_config(config):
        m.setenv("VLLM_ROCM_USE_AITER", "1")
        rocm_aiter_ops.refresh_env_variables()
        try:
            yield config
        finally:
            rocm_aiter_ops.refresh_env_variables()


@pytest.mark.parametrize("is_neox_style", [False, True])
@pytest.mark.parametrize("seq_len", [11, 42])
@pytest.mark.parametrize("num_q_heads", [16, 128])
@pytest.mark.parametrize("qk_rope_head_dim", [64])
@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("num_blocks", [64])
@pytest.mark.parametrize("block_size", [1, 64])
@pytest.mark.parametrize("contiguous_qnope", [True, False])
@pytest.mark.parametrize("seed", [0])
@torch.inference_mode()
def test_fused_qk_rope_concat_and_cache_mla(
    default_vllm_config,
    is_neox_style: bool,
    seq_len: int,
    num_q_heads: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    num_blocks: int,
    block_size: int,
    contiguous_qnope: bool,
    seed: int,
    max_position: int = 8192,
    base: float = 10000,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    rocm_aiter_ops.register_ops_once()
    set_random_seed(seed)
    device = "cuda"
    torch.set_default_device(device)

    rope = RotaryEmbedding(
        qk_rope_head_dim,
        qk_rope_head_dim,
        max_position,
        base,
        is_neox_style,
        torch.float32,
    ).to(dtype=dtype, device=device)
    cos_cache, sin_cache = rope.cos_sin_cache.chunk(2, dim=-1)
    cos_cache = cos_cache.contiguous()
    sin_cache = sin_cache.contiguous()

    positions = torch.randint(0, max_position, (seq_len,), device=device)

    # Post-W_UK latent (no RoPE) and the pre-RoPE positional part of q.
    if contiguous_qnope:
        ql_nope = torch.randn(seq_len, num_q_heads, kv_lora_rank, dtype=dtype)
    else:
        # Match the production layout: a transposed (non-contiguous) view of the
        # W_UK bmm output.
        ql_nope = torch.randn(
            num_q_heads, seq_len, kv_lora_rank, dtype=dtype
        ).transpose(0, 1)
        assert not ql_nope.is_contiguous()
    q_pe = torch.randn(seq_len, num_q_heads, qk_rope_head_dim, dtype=dtype)
    kv_c = torch.randn(seq_len, kv_lora_rank, dtype=dtype)
    k_pe = torch.randn(seq_len, 1, qk_rope_head_dim, dtype=dtype)

    q_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    k_scale = torch.tensor([0.1], dtype=torch.float32, device=device)

    entry_size = kv_lora_rank + qk_rope_head_dim
    slot_mapping_lst = random.sample(range(num_blocks * block_size), seq_len)
    slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long, device=device)

    # ---- reference (split kernels) --------------------------------------
    # RoPE q_pe / k_pe with the same routine the wrapper would use.
    ref_q_pe, ref_k_pe = rope.forward_hip(
        positions,
        q_pe.clone().reshape(seq_len, num_q_heads * qk_rope_head_dim),
        k_pe.clone().reshape(seq_len, qk_rope_head_dim),
    )
    ref_q_pe = ref_q_pe.reshape(seq_len, num_q_heads, qk_rope_head_dim)
    ref_k_pe = ref_k_pe.reshape(seq_len, qk_rope_head_dim)

    ref_q_out = torch.cat([ql_nope, ref_q_pe], dim=-1)

    ref_kv_cache = torch.zeros(
        num_blocks, block_size, entry_size, dtype=torch.uint8, device=device
    )
    ops.concat_and_cache_mla(
        kv_c,
        ref_k_pe,
        ref_kv_cache,
        slot_mapping,
        kv_cache_dtype="fp8",
        scale=k_scale,
    )

    # ---- fused kernel ---------------------------------------------------
    q_out = torch.empty(seq_len, num_q_heads, entry_size, dtype=dtype, device=device)
    kv_cache = torch.zeros(
        num_blocks, block_size, entry_size, dtype=torch.uint8, device=device
    )
    rocm_aiter_ops.fused_qk_rope_concat_and_cache_mla(
        ql_nope,
        q_pe,
        kv_c,
        k_pe.squeeze(1),
        kv_cache.view(current_platform.fp8_dtype()),
        q_out,
        slot_mapping,
        k_scale,
        q_scale,
        positions,
        cos_cache,
        sin_cache,
        is_neox=is_neox_style,
        is_nope_first=True,
    )

    # q_out is bf16 (nope-first concat of latent + RoPE'd pe).
    torch.testing.assert_close(q_out, ref_q_out, atol=2e-2, rtol=2e-2)

    # Compare KV cache after dequant (one e4m3 ULP ~ 12.5%).
    result = torch.empty_like(kv_cache, dtype=torch.float16)
    expected = torch.empty_like(ref_kv_cache, dtype=torch.float16)
    ops.convert_fp8(result, kv_cache, k_scale.item(), kv_dtype="fp8")
    ops.convert_fp8(expected, ref_kv_cache, k_scale.item(), kv_dtype="fp8")
    torch.testing.assert_close(result, expected, atol=1e-3, rtol=0.15)
