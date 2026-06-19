# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FA4 MLA prefill CuTeDSL compile warmup config."""

from __future__ import annotations

from collections.abc import Hashable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from vllm.v1.attention.backends.fa_utils import (
        FlashAttentionCuTeDSLCompileSpec,
    )

FA4ArchitectureFamily = Literal["sm90", "sm100f", "sm120"]
FA4Architecture = Literal["sm90", "sm100", "sm103", "sm120"]

FA4_STANDARD_DTYPES = (torch.bfloat16, torch.float16)
FA4_ARCH_BY_COMPUTE_CAPABILITY: dict[tuple[int, int], FA4Architecture] = {
    (9, 0): "sm90",
    (10, 0): "sm100",
    (10, 3): "sm103",
    (12, 0): "sm120",
}
FA4_ARCH_FAMILY_BY_ARCH: dict[FA4Architecture, FA4ArchitectureFamily] = {
    "sm90": "sm90",
    "sm100": "sm100f",
    "sm103": "sm100f",
    "sm120": "sm120",
}

# FA4 derives compile-static fields from request shape: q_stage on Blackwell,
# Split-KV for long K, and 2CTA/scheduler choices.
FA4_MLA_PREFILL_SHAPE_PROBES: tuple[tuple[str, int, int, int], ...] = (
    ("q_stage1_min_q_b1", 1, 1, 128),
    ("q_stage1_short_q_b1", 1, 32, 128),
    ("default_tile_q_stage_boundary_b1", 1, 129, 512),
    ("long_k_split_candidate_b1", 1, 512, 4096),
    ("very_long_k_split_candidate_b1", 1, 1024, 8192),
    ("q_stage1_min_q_b2", 2, 1, 128),
    ("q_stage1_short_q_b2", 2, 32, 128),
    ("default_tile_q_stage_boundary_b2", 2, 129, 512),
    ("long_k_split_candidate_b2", 2, 512, 4096),
    ("very_long_k_split_candidate_b2", 2, 1024, 8192),
)
FA4_MLA_PREFILL_CAUSAL_OPTIONS = (False, True)
FA4_MLA_PREFILL_LSE_OPTIONS = (False, True)


@dataclass(frozen=True)
class FA4MLAPrefillCompileContext:
    dtype: torch.dtype
    num_heads: int
    qk_head_dim: int
    v_head_dim: int
    kv_nope_head_dim: int
    requires_v_padding: bool
    scale: float
    num_splits: int
    fa_version: int

    # Return the V head dim FA4 sees.
    @property
    def effective_v_head_dim(self) -> int:
        if self.requires_v_padding:
            return self.qk_head_dim
        return self.v_head_dim


@dataclass(frozen=True)
class FA4MLAPrefillCompileRequest:
    """One compile-only FA4 MLA prefill request."""

    key: Hashable
    compile_spec: "FlashAttentionCuTeDSLCompileSpec"

    # Compile this request.
    def compile(self) -> None:
        self.compile_spec.compile()


@dataclass(frozen=True)
class _FA4MLAPrefillProbe:
    name: str
    batch_size: int
    max_seqlen_q: int
    max_seqlen_k: int
    causal: bool
    return_lse: bool


# Yield request keys for audits/tests.
def iter_fa4_mla_prefill_request_keys(
    ctx: FA4MLAPrefillCompileContext,
) -> Iterator[Hashable]:
    """Yield request keys for FA4 MLA prefill warmup probes."""

    for request in iter_fa4_mla_prefill_compile_requests(ctx):
        yield request.key


# Yield deduped compile requests.
def iter_fa4_mla_prefill_compile_requests(
    ctx: FA4MLAPrefillCompileContext,
) -> Iterator[FA4MLAPrefillCompileRequest]:
    """Yield compile requests for this fixed MLA backend.

    FA4 dedupes duplicate atomic kernel selections in its own JIT cache.
    """
    seen: set[Hashable] = set()
    for compile_spec in iter_fa4_mla_prefill_compile_specs(ctx):
        key = compile_spec.request_key()
        if key in seen:
            continue
        seen.add(key)
        yield FA4MLAPrefillCompileRequest(
            key=key,
            compile_spec=compile_spec,
        )


# Build compile specs for this setup.
def iter_fa4_mla_prefill_compile_specs(
    ctx: FA4MLAPrefillCompileContext,
) -> Iterator["FlashAttentionCuTeDSLCompileSpec"]:
    """Yield compile-only FA4 MLA prefill requests for this fixed setup."""

    if not _supports_fa4_mla_prefill(ctx):
        return

    compile_spec_cls = _flash_attention_compile_spec_cls()
    for probe in _iter_fa4_mla_prefill_probes():
        total_q_tokens = probe.batch_size * probe.max_seqlen_q
        total_kv_tokens = probe.batch_size * probe.max_seqlen_k
        yield compile_spec_cls(
            q_shape=_q_shape(total_q_tokens, ctx),
            k_shape=_q_shape(total_kv_tokens, ctx),
            v_shape=_v_shape(total_kv_tokens, ctx),
            v_stride=_v_stride(ctx),
            q_dtype=ctx.dtype,
            cu_seqlens_q_shape=(probe.batch_size + 1,),
            cu_seqlens_k_shape=(probe.batch_size + 1,),
            max_seqlen_q=probe.max_seqlen_q,
            max_seqlen_k=probe.max_seqlen_k,
            softmax_scale=ctx.scale,
            causal=probe.causal,
            return_softmax_lse=probe.return_lse,
            num_splits=ctx.num_splits,
            fa_version=ctx.fa_version,
        )


# Expand shape probes with mask/LSE options.
def _iter_fa4_mla_prefill_probes() -> Iterator[_FA4MLAPrefillProbe]:
    for (
        shape_name,
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
    ) in FA4_MLA_PREFILL_SHAPE_PROBES:
        for causal in FA4_MLA_PREFILL_CAUSAL_OPTIONS:
            for return_lse in FA4_MLA_PREFILL_LSE_OPTIONS:
                yield _FA4MLAPrefillProbe(
                    name=_probe_name(shape_name, causal, return_lse),
                    batch_size=batch_size,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    causal=causal,
                    return_lse=return_lse,
                )


# Check whether this setup can use FA4 MLA prefill.
def _supports_fa4_mla_prefill(ctx: FA4MLAPrefillCompileContext) -> bool:
    arch_family, _ = _fa4_architecture_from_compute_capability(
        *torch.cuda.get_device_capability()
    )
    if ctx.dtype not in FA4_STANDARD_DTYPES:
        return False
    if ctx.num_heads <= 0:
        return False
    if arch_family == "sm120" and ctx.num_splits != 1:
        return False
    return True


# Map CUDA capability to FA4 arch names.
def _fa4_architecture_from_compute_capability(
    major: int,
    minor: int,
) -> tuple[FA4ArchitectureFamily, FA4Architecture]:
    arch = FA4_ARCH_BY_COMPUTE_CAPABILITY.get((major, minor))
    if arch is None and major == 10:
        arch = "sm100"
    if arch is None:
        raise ValueError(f"FA4 warmup does not know CUDA capability {major}.{minor}")
    return FA4_ARCH_FAMILY_BY_ARCH[arch], arch


# Import the compile spec lazily.
def _flash_attention_compile_spec_cls() -> type["FlashAttentionCuTeDSLCompileSpec"]:
    from vllm.v1.attention.backends.fa_utils import (
        FlashAttentionCuTeDSLCompileSpec,
    )

    return FlashAttentionCuTeDSLCompileSpec


# Name probes for logs/tests.
def _probe_name(shape_name: str, causal: bool, return_lse: bool) -> str:
    mask_name = "causal_mask" if causal else "full_mask"
    lse_name = "return_lse" if return_lse else "no_lse"
    return f"{shape_name}/{mask_name}/{lse_name}"


# Build the Q/K tensor shape.
def _q_shape(
    num_tokens: int,
    ctx: FA4MLAPrefillCompileContext,
) -> tuple[int, int, int]:
    return (num_tokens, ctx.num_heads, ctx.qk_head_dim)


# Build the V tensor shape.
def _v_shape(
    num_tokens: int,
    ctx: FA4MLAPrefillCompileContext,
) -> tuple[int, int, int]:
    return (num_tokens, ctx.num_heads, ctx.effective_v_head_dim)


# Preserve V stride when unpadded.
def _v_stride(ctx: FA4MLAPrefillCompileContext) -> tuple[int, int, int] | None:
    if ctx.requires_v_padding:
        return None
    return (
        ctx.num_heads * ctx.kv_nope_head_dim,
        ctx.kv_nope_head_dim,
        1,
    )
