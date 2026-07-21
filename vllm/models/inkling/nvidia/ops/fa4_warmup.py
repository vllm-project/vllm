# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup compilation of the FA4 kernels used by Inkling."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.warmup.jit_warmup import (
    VllmJitKernel,
    WarmupIntRange,
    zip_inputs,
)
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype

from .fa4_rel_attention import (
    bucket_max_seqlen_q,
    inkling_fa4_num_splits,
    inkling_fa4_rel_attention,
)


def _compile_fa4_rel_attention_from_specs(
    *,
    q_shape: tuple[int, ...],
    key_cache_shape: tuple[int, ...],
    value_cache_shape: tuple[int, ...],
    block_table_shape: tuple[int, ...],
    cache_seqlens_shape: tuple[int, ...],
    cu_seqlens_q_shape: tuple[int, ...],
    rel_logits_shape: tuple[int, ...],
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    max_seqlen_q: int,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int],
    num_splits: int,
) -> None:
    """Compile TML FA4 via fake tensors without launching a kernel."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vllm.third_party.tml_fa4 import flash_attn_varlen_func

    cute_window = (None, None) if window_size == (-1, -1) else window_size
    # FakeTensorMode carries shape, dtype, stride, and device metadata into TML
    # FA4 so the JIT can build the specialization without allocating real inputs
    # or running the kernel body. This is not a dummy runtime launch.
    with FakeTensorMode():
        device = torch.accelerator.current_accelerator()
        q = torch.empty(q_shape, dtype=q_dtype, device=device)
        key_cache = torch.empty(key_cache_shape, dtype=kv_dtype, device=device)
        value_cache = torch.empty(value_cache_shape, dtype=kv_dtype, device=device)
        rel_logits = torch.empty(rel_logits_shape, dtype=q_dtype, device=device)
        flash_attn_varlen_func(
            q=q,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=torch.empty(
                cu_seqlens_q_shape,
                dtype=torch.int32,
                device=device,
            ),
            seqused_k=torch.empty(
                cache_seqlens_shape,
                dtype=torch.int32,
                device=device,
            ),
            max_seqlen_q=max_seqlen_q,
            page_table=torch.empty(
                block_table_shape,
                dtype=torch.int32,
                device=device,
            ),
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=cute_window,
            num_splits=num_splits,
            return_lse=False,
            out=torch.empty_like(q),
            rel_bias=rel_logits.contiguous(),
        )


def _num_warps_bucket(num_reqs: int) -> int:
    num_warps = min((num_reqs + 30) // 31, 32)
    return 1 << (num_warps - 1).bit_length()


def _materialized_num_reqs(
    num_splits: int,
    num_warps_bucket: int | None,
    large_num_reqs: bool,
) -> int:
    if num_splits == 1 or num_warps_bucket is None:
        return 1025 if large_num_reqs else 1

    min_num_warps = 1 if num_warps_bucket == 1 else num_warps_bucket // 2 + 1
    min_num_reqs = (min_num_warps - 1) * 31 + 1
    return max(min_num_reqs, 1025 if large_num_reqs else 1)


class InklingFA4RelAttentionKernel(
    VllmJitKernel["InklingFA4RelAttentionKernel.CompileKey"]
):
    @dataclass(frozen=True)
    class CompileKey:
        is_local: bool
        num_heads: int
        num_kv_heads: int
        head_dim: int
        rel_extent: int
        dtype: torch.dtype
        kv_dtype: torch.dtype
        block_size: int
        window_size: tuple[int, int]
        max_seqlen_q: int
        num_splits: int
        num_warps_bucket: int | None
        large_num_reqs: bool

    def dispatch(  # type: ignore[override]
        self,
        *,
        is_local: bool,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rel_extent: int,
        dtype: torch.dtype,
        kv_dtype: torch.dtype,
        block_size: int,
        window_size: tuple[int, int],
        max_kv_len: int,
        query_len: int,
        num_reqs: int,
    ) -> CompileKey:
        max_seqlen_q = bucket_max_seqlen_q(query_len)
        num_splits = inkling_fa4_num_splits(
            is_local=is_local,
            batch_size=num_reqs,
            max_query_len=max_seqlen_q,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_kv_len=max_kv_len,
        )
        return self.CompileKey(
            is_local=is_local,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rel_extent=rel_extent,
            dtype=dtype,
            kv_dtype=kv_dtype,
            block_size=block_size,
            window_size=window_size,
            max_seqlen_q=max_seqlen_q,
            num_splits=num_splits,
            num_warps_bucket=(_num_warps_bucket(num_reqs) if num_splits > 1 else None),
            large_num_reqs=num_reqs > 1024,
        )

    def _is_valid_warmup_dispatch(
        self,
        *,
        query_len: int,
        num_reqs: int,
        max_num_batched_tokens: int,
    ) -> bool:
        max_seqlen_q = bucket_max_seqlen_q(query_len)
        min_query_len = 1 if max_seqlen_q == 1 else max_seqlen_q // 2 + 1
        return min_query_len + num_reqs <= max_num_batched_tokens + 1

    def get_warmup_keys(self, vllm_config: VllmConfig) -> list[CompileKey]:
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        if max_num_reqs <= 0 or max_num_batched_tokens <= 0:
            return []

        config = vllm_config.model_config.hf_config
        tp_size = get_tensor_model_parallel_world_size()
        dtype = vllm_config.model_config.dtype
        kv_dtype = kv_cache_dtype_str_to_dtype(
            vllm_config.cache_config.cache_dtype,
            vllm_config.model_config,
        )
        block_size = vllm_config.cache_config.block_size
        local_extent = config.sliding_window_size

        global_num_kv_heads = config.num_key_value_heads
        local_num_kv_heads = config.swa_num_key_value_heads
        assert config.num_attention_heads % tp_size == 0
        assert config.swa_num_attention_heads % tp_size == 0
        if global_num_kv_heads >= tp_size:
            assert global_num_kv_heads % tp_size == 0
        else:
            assert tp_size % global_num_kv_heads == 0
        if local_num_kv_heads >= tp_size:
            assert local_num_kv_heads % tp_size == 0
        else:
            assert tp_size % local_num_kv_heads == 0

        return self._trace_dispatch(self.dispatch)(
            zip_inputs(
                dict(
                    is_local=False,
                    num_heads=config.num_attention_heads // tp_size,
                    num_kv_heads=max(1, global_num_kv_heads // tp_size),
                    head_dim=config.head_dim,
                    rel_extent=config.rel_extent,
                    dtype=dtype,
                    kv_dtype=kv_dtype,
                    block_size=block_size,
                    window_size=(-1, -1),
                    max_kv_len=vllm_config.model_config.max_model_len,
                ),
                dict(
                    is_local=True,
                    num_heads=config.swa_num_attention_heads // tp_size,
                    num_kv_heads=max(1, local_num_kv_heads // tp_size),
                    head_dim=config.swa_head_dim,
                    rel_extent=local_extent,
                    dtype=dtype,
                    kv_dtype=kv_dtype,
                    block_size=block_size,
                    window_size=(local_extent - 1, 0),
                    max_kv_len=local_extent,
                ),
            ),
            query_len=WarmupIntRange(1, max_num_batched_tokens + 1),
            num_reqs=WarmupIntRange(1, max_num_reqs + 1),
            max_num_batched_tokens=max_num_batched_tokens,
            _when=self._is_valid_warmup_dispatch,
        )

    def compile(self, compile_key: CompileKey) -> None:
        num_reqs = _materialized_num_reqs(
            compile_key.num_splits,
            compile_key.num_warps_bucket,
            compile_key.large_num_reqs,
        )
        total_q = compile_key.max_seqlen_q + num_reqs - 1
        _compile_fa4_rel_attention_from_specs(
            q_shape=(total_q, compile_key.num_heads, compile_key.head_dim),
            key_cache_shape=(
                1,
                compile_key.block_size,
                compile_key.num_kv_heads,
                compile_key.head_dim,
            ),
            value_cache_shape=(
                1,
                compile_key.block_size,
                compile_key.num_kv_heads,
                compile_key.head_dim,
            ),
            block_table_shape=(num_reqs, 1),
            cache_seqlens_shape=(num_reqs,),
            cu_seqlens_q_shape=(num_reqs + 1,),
            rel_logits_shape=(
                total_q,
                compile_key.num_heads,
                compile_key.rel_extent,
            ),
            q_dtype=compile_key.dtype,
            kv_dtype=compile_key.kv_dtype,
            max_seqlen_q=compile_key.max_seqlen_q,
            softmax_scale=compile_key.head_dim**-1,
            causal=True,
            window_size=compile_key.window_size,
            num_splits=compile_key.num_splits,
        )

    @staticmethod
    def kernel(
        q: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        *,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        softmax_scale: float,
        causal: bool,
        window_size: tuple[int, int],
        rel_extent: int,
        rel_logits: torch.Tensor,
        num_splits: int = 32,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return inkling_fa4_rel_attention(
            q,
            key_cache,
            value_cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            rel_extent=rel_extent,
            rel_logits=rel_logits,
            num_splits=num_splits,
            out=out,
        )

    def __call__(
        self,
        q: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        *,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        softmax_scale: float,
        causal: bool,
        window_size: tuple[int, int],
        rel_extent: int,
        rel_logits: torch.Tensor,
        num_splits: int = 32,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.kernel(
            q,
            key_cache,
            value_cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            rel_extent=rel_extent,
            rel_logits=rel_logits,
            num_splits=num_splits,
            out=out,
        )


INKLING_FA4_REL_ATTENTION_KERNEL = InklingFA4RelAttentionKernel()
