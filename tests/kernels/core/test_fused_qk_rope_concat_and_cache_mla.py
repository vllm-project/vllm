# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the fused sparse-MLA decode Q-prep kernel.

``rocm_aiter_ops.fused_qk_rope_concat_and_cache_mla`` collapses the decode
RoPE + Q-concat + KV-concat + KV-cache-write into a single aiter kernel. Two
tests cover it:

1. ``test_fused_qk_rope_concat_and_cache_mla`` checks the raw kernel against the
   split kernels it replaces, for both a bf16 and an fp8 ``q_out``:
     * q_out  == concat(ql_nope, RoPE(q_pe))            (nope-first, bf16)
     * q_out  == scaled_fp8_quant(concat(...), q_scale) (fp8; verifies the
                 kernel's q-side quant matches vLLM's ``scaled_fp8_quant``)
     * kv_cache == concat_and_cache_mla(kv_c, RoPE(k_pe))

2. ``test_impl_handles_noncontiguous_qnope`` drives the production impl helper
   (``ROCMAiterMLASparseImpl.fused_qk_rope_concat_and_cache``) with a
   non-contiguous ``ql_nope`` (the transposed bmm layout production passes). The
   raw kernel silently produces wrong q_out for a strided ``ql_nope``; the helper
   must make it contiguous. This guards that fix.

ROCm-only; skipped on other platforms.
"""

import random
from types import SimpleNamespace

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


def _build_case(
    is_neox_style,
    seq_len,
    num_q_heads,
    qk_rope_head_dim,
    kv_lora_rank,
    num_blocks,
    block_size,
    contiguous_qnope,
    seed,
    max_position=8192,
    base=10000,
    dtype=torch.bfloat16,
):
    """Build inputs and the split-kernel reference for one config."""
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

    # A non-unit q_scale so the fp8 path actually exercises the kernel's scale
    # convention (with 1.0, divide-vs-multiply would be indistinguishable).
    q_scale = torch.tensor([0.3], dtype=torch.float32, device=device)
    k_scale = torch.tensor([0.1], dtype=torch.float32, device=device)

    entry_size = kv_lora_rank + qk_rope_head_dim
    slot_mapping_lst = random.sample(range(num_blocks * block_size), seq_len)
    slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long, device=device)

    # ---- reference (split kernels) --------------------------------------
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

    return SimpleNamespace(
        device=device,
        cos_cache=cos_cache,
        sin_cache=sin_cache,
        positions=positions,
        ql_nope=ql_nope,
        q_pe=q_pe,
        kv_c=kv_c,
        k_pe=k_pe,
        q_scale=q_scale,
        k_scale=k_scale,
        entry_size=entry_size,
        slot_mapping=slot_mapping,
        ref_q_out=ref_q_out,
        ref_kv_cache=ref_kv_cache,
    )


def _assert_q_out_close(q_out, c, fp8_q_out, seq_len):
    if fp8_q_out:
        # The kernel quantizes q_out to fp8 with q_scale. Verify it matches
        # vLLM's scaled_fp8_quant (the same convention the split path uses when
        # it quantizes q downstream)
        ref_q_fp8, _ = ops.scaled_fp8_quant(
            c.ref_q_out.reshape(seq_len, -1).contiguous(), c.q_scale
        )
        deq_fused = q_out.to(torch.float32).reshape(seq_len, -1) * c.q_scale
        deq_ref = ref_q_fp8.to(torch.float32) * c.q_scale
        torch.testing.assert_close(deq_fused, deq_ref, atol=1e-3, rtol=0.15)
    else:
        torch.testing.assert_close(q_out, c.ref_q_out, atol=2e-2, rtol=2e-2)


def _assert_kv_close(kv_cache, c):
    # Compare KV cache after dequant (one e4m3 ULP ~ 12.5%).
    result = torch.empty_like(kv_cache, dtype=torch.float16)
    expected = torch.empty_like(c.ref_kv_cache, dtype=torch.float16)
    ops.convert_fp8(result, kv_cache, c.k_scale.item(), kv_dtype="fp8")
    ops.convert_fp8(expected, c.ref_kv_cache, c.k_scale.item(), kv_dtype="fp8")
    torch.testing.assert_close(result, expected, atol=1e-3, rtol=0.15)


@pytest.mark.parametrize("is_neox_style", [False, True])
@pytest.mark.parametrize("seq_len", [11, 42])
@pytest.mark.parametrize("num_q_heads", [16, 128])
@pytest.mark.parametrize("qk_rope_head_dim", [64])
@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("num_blocks", [64])
@pytest.mark.parametrize("block_size", [1, 64])
@pytest.mark.parametrize("fp8_q_out", [False, True])
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
    fp8_q_out: bool,
    seed: int,
) -> None:
    """Raw-kernel parity (contiguous ql_nope), bf16 and fp8 q_out."""
    rocm_aiter_ops.register_ops_once()
    c = _build_case(
        is_neox_style,
        seq_len,
        num_q_heads,
        qk_rope_head_dim,
        kv_lora_rank,
        num_blocks,
        block_size,
        contiguous_qnope=True,
        seed=seed,
    )

    q_out_dtype = current_platform.fp8_dtype() if fp8_q_out else torch.bfloat16
    q_out = torch.empty(
        seq_len, num_q_heads, c.entry_size, dtype=q_out_dtype, device=c.device
    )
    kv_cache = torch.zeros(
        num_blocks, block_size, c.entry_size, dtype=torch.uint8, device=c.device
    )
    rocm_aiter_ops.fused_qk_rope_concat_and_cache_mla(
        c.ql_nope,
        c.q_pe,
        c.kv_c,
        c.k_pe.squeeze(1),
        kv_cache.view(current_platform.fp8_dtype()),
        q_out,
        c.slot_mapping,
        c.k_scale,
        c.q_scale,
        c.positions,
        c.cos_cache,
        c.sin_cache,
        is_neox=is_neox_style,
        is_nope_first=True,
    )

    _assert_q_out_close(q_out, c, fp8_q_out, seq_len)
    _assert_kv_close(kv_cache, c)


@pytest.mark.parametrize("is_neox_style", [False, True])
@pytest.mark.parametrize("num_q_heads", [16, 128])
@pytest.mark.parametrize("block_size", [1, 64])
@pytest.mark.parametrize("seed", [0])
@torch.inference_mode()
def test_impl_handles_noncontiguous_qnope(
    default_vllm_config,
    is_neox_style: bool,
    num_q_heads: int,
    block_size: int,
    seed: int,
    seq_len: int = 42,
    qk_rope_head_dim: int = 64,
    kv_lora_rank: int = 512,
    num_blocks: int = 64,
) -> None:
    """The impl helper must make a transposed (non-contiguous) ql_nope
    contiguous before the kernel; otherwise q_out is silently wrong. Drives the
    real production helper and checks parity (fp8 q_out + KV cache)."""
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseImpl,
    )

    rocm_aiter_ops.register_ops_once()
    c = _build_case(
        is_neox_style,
        seq_len,
        num_q_heads,
        qk_rope_head_dim,
        kv_lora_rank,
        num_blocks,
        block_size,
        contiguous_qnope=False,
        seed=seed,
    )

    kv_cache = torch.zeros(
        num_blocks, block_size, c.entry_size, dtype=torch.uint8, device=c.device
    ).view(current_platform.fp8_dtype())

    # Duck-typed impl / layer: the helper only reads these attributes.
    impl = SimpleNamespace(
        head_size=c.entry_size,
        kv_lora_rank=kv_lora_rank,
        kv_cache_dtype="fp8",
    )
    layer = SimpleNamespace(_k_scale=c.k_scale, _q_scale=c.q_scale)

    q_out = ROCMAiterMLASparseImpl.fused_qk_rope_concat_and_cache(
        impl,
        layer,
        c.ql_nope,  # non-contiguous
        c.q_pe,
        c.kv_c,
        c.k_pe,
        kv_cache,
        c.slot_mapping,
        c.positions,
        c.cos_cache,
        c.sin_cache,
        is_neox_style,
    )

    assert q_out.dtype == current_platform.fp8_dtype()
    _assert_q_out_close(q_out, c, fp8_q_out=True, seq_len=seq_len)
    _assert_kv_close(kv_cache, c)
