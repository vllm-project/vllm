# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up AITER unified-attention Triton kernels."""

from typing import TYPE_CHECKING, NamedTuple

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import is_quantized_kv_cache

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

_AITER_UNIFIED_ATTN_BACKENDS = frozenset({"ROCM_AITER_UNIFIED_ATTN"})


class _AttnSignature(NamedTuple):
    """Compile-relevant attributes of an attention layer."""
    num_heads: int
    num_kv_heads: int
    head_size: int
    block_size: int
    num_blocks: int
    sliding_window: tuple[int, int]
    kv_cache_dtype: str
    has_sinks: bool
    q_dtype: torch.dtype


class _WarmupTarget(NamedTuple):
    """A unique attention configuration plus a representative layer to drive it."""

    signature: _AttnSignature
    layer: object
    kv_cache: torch.Tensor


def _attention_backend_name(backend: object) -> str | None:
    get_name = getattr(backend, "get_name", None)
    if get_name is None:
        return None
    try:
        return get_name()
    except NotImplementedError:
        return None


def _pow2_span(hi: int) -> list[int]:
    """Sorted powers of two in [1, hi], always including hi itself."""
    hi = max(1, int(hi))
    vals: set[int] = set()
    n = 1
    while n < hi:
        vals.add(n)
        n *= 2
    vals.add(hi)
    return sorted(vals)


class _SweepBounds(NamedTuple):
    """Input-space bounds derived from vLLM's scheduler / speculative config."""
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    non_decode_query_len: int


class _LaunchSpec(NamedTuple):
    max_seqlen_q: int
    max_seqlen_k: int
    num_seqs: int
    query_lens: tuple[int, ...] | None = None


def build_launch_specs(
    signature: _AttnSignature,
    bounds: _SweepBounds,
) -> list[_LaunchSpec]:
    """Enumerate representative (q, k, num_seqs) launches for one signature."""
    block_size = signature.block_size
    max_k = max(1, min(bounds.max_model_len, signature.num_blocks * block_size))

    ctx_lens = _pow2_span(max_k)
    seq_counts = _pow2_span(bounds.max_num_seqs)

    specs: list[_LaunchSpec] = []

    # Prefill: hits the 2D kernel with different configurations.
    prefill_cap = max(1, min(bounds.max_num_batched_tokens, max_k))
    for q in _pow2_span(prefill_cap):
        if q > 1:
            specs.append(_LaunchSpec(max_seqlen_q=q, max_seqlen_k=q, num_seqs=1))

    # Decode:
    # Small k hits the 2D kernel;
    # Large k hits 3D + reduce with a batch-dependent NUM_SEGMENTS.
    for k in ctx_lens:
        for num_seqs in seq_counts:
            specs.append(_LaunchSpec(max_seqlen_q=1, max_seqlen_k=k, num_seqs=num_seqs))

    # Non-decode small-q (ALL_DECODE=False):
    # Small k hits the 2D kernel;
    # Large k with few programs hits the 3D kernel.
    non_decode_q = min(max(bounds.non_decode_query_len, 2), max_k)
    if non_decode_q > 1:
        for k in ctx_lens:
            if k < non_decode_q:
                continue
            for num_seqs in seq_counts:
                specs.append(
                    _LaunchSpec(
                        max_seqlen_q=non_decode_q, max_seqlen_k=k, num_seqs=num_seqs
                    )
                )

    # Mixed (varlen) batch:
    # Hits the both kernels with various configurations
    mixed = _mixed_launch_spec(bounds, prefill_cap, max_k)
    if mixed is not None:
        specs.append(mixed)

    return list(dict.fromkeys(specs))


def _mixed_launch_spec(
    bounds: _SweepBounds,
    prefill_cap: int,
    max_k: int,
) -> _LaunchSpec | None:
    """Build one mixed prefill+decode launch over a long context, or ``None``"""
    if bounds.max_num_seqs < 2 or max_k <= 1:
        return None

    num_prefill_rows = min(2, bounds.max_num_seqs - 1)
    num_decode_rows = bounds.max_num_seqs - num_prefill_rows
    # Size the prefill chunk to reach aiter's >= 256 large-prefill config when
    # the budget allows, while keeping the whole batch within the token budget.
    budget_for_prefill = bounds.max_num_batched_tokens - num_decode_rows
    chunk = min(prefill_cap, 256, budget_for_prefill // num_prefill_rows)
    if chunk <= 1:
        return None

    query_lens = (chunk,) * num_prefill_rows + (1,) * num_decode_rows
    return _LaunchSpec(
        max_seqlen_q=chunk,
        max_seqlen_k=max_k,
        num_seqs=len(query_lens),
        query_lens=query_lens,
    )


def _signature_for_layer(
    layer: object,
    kv_cache: torch.Tensor,
    kv_cache_spec: object,
    q_dtype: torch.dtype,
) -> _AttnSignature | None:
    """Build the compile-relevant signature from the impl and KV-cache spec."""
    impl = getattr(layer, "impl", None)
    if impl is None:
        return None
    if not isinstance(kv_cache, torch.Tensor) or kv_cache.dim() < 1:
        return None

    block_size = getattr(kv_cache_spec, "block_size", None)
    if block_size is None:
        return None
    sliding_window = getattr(impl, "sliding_window", (-1, -1))
    return _AttnSignature(
        num_heads=int(impl.num_heads),
        num_kv_heads=int(impl.num_kv_heads),
        head_size=int(impl.head_size),
        block_size=int(block_size),
        num_blocks=int(kv_cache.shape[0]),
        sliding_window=tuple(sliding_window),
        kv_cache_dtype=str(impl.kv_cache_dtype),
        has_sinks=getattr(impl, "sinks", None) is not None,
        q_dtype=q_dtype,
    )


def collect_unique_warmup_targets(
    runner: object, q_dtype: torch.dtype
) -> list[_WarmupTarget]:
    """Return one representative warmup target per unique attention signature."""
    static_forward_context = getattr(
        getattr(runner, "compilation_config", None), "static_forward_context", None
    )
    if not isinstance(static_forward_context, dict):
        return []

    targets: dict[_AttnSignature, _WarmupTarget] = {}
    for groups in getattr(runner, "attn_groups", []) or ():
        for group in groups:
            name = _attention_backend_name(getattr(group, "backend", None))
            if name not in _AITER_UNIFIED_ATTN_BACKENDS:
                continue
            kv_cache_spec = getattr(group, "kv_cache_spec", None)
            for layer_name in getattr(group, "layer_names", []) or ():
                layer = static_forward_context.get(layer_name)
                if layer is None:
                    continue
                kv_cache = getattr(layer, "kv_cache", None)
                if not isinstance(kv_cache, torch.Tensor) or kv_cache.numel() == 0:
                    continue
                signature = _signature_for_layer(
                    layer, kv_cache, kv_cache_spec, q_dtype
                )
                if signature is None or signature in targets:
                    continue
                targets[signature] = _WarmupTarget(signature, layer, kv_cache)
    return list(targets.values())


def _launch_one(
    unified_attention,
    layer: object,
    kv_cache: torch.Tensor,
    signature: _AttnSignature,
    spec: _LaunchSpec,
    device: torch.device,
) -> None:
    impl = layer.impl
    num_heads = signature.num_heads
    head_size = signature.head_size
    block_size = signature.block_size
    q_dtype = signature.q_dtype

    key_cache, value_cache = impl._split_kv_cache(kv_cache)
    if is_quantized_kv_cache(impl.kv_cache_dtype):
        key_cache = key_cache.view(impl.fp8_dtype)
        value_cache = value_cache.view(impl.fp8_dtype)

    k_descale = getattr(layer, "_k_scale", None)
    v_descale = getattr(layer, "_v_scale", None)
    q_descale = None

    num_seqs = spec.num_seqs
    max_seqlen_q = spec.max_seqlen_q
    max_seqlen_k = spec.max_seqlen_k

    if spec.query_lens is None:
        # Uniform batch: every row is ``max_seqlen_q`` long.
        num_tokens = num_seqs * max_seqlen_q
        cu_seqlens_q = torch.arange(
            0,
            (num_seqs + 1) * max_seqlen_q,
            max_seqlen_q,
            dtype=torch.int32,
            device=device,
        )
    else:
        # Mixed (varlen) batch: cumulative offsets of the per-row query lengths.
        offsets = [0]
        for query_len in spec.query_lens:
            offsets.append(offsets[-1] + query_len)
        num_tokens = offsets[-1]
        cu_seqlens_q = torch.tensor(offsets, dtype=torch.int32, device=device)

    query = torch.zeros(
        (num_tokens, num_heads, head_size), dtype=q_dtype, device=device
    )
    out = torch.empty_like(query)
    seqused_k = torch.full(
        (num_seqs,), max_seqlen_k, dtype=torch.int32, device=device
    )
    max_num_blocks_per_seq = (max_seqlen_k + block_size - 1) // block_size
    # All rows point at valid blocks (0..n-1); warmup only reads the cache.
    block_table = torch.zeros(
        (num_seqs, max_num_blocks_per_seq), dtype=torch.int32, device=device
    )

    unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        seqused_k=seqused_k,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=float(impl.scale),
        causal=True,
        alibi_slopes=None,
        window_size=tuple(signature.sliding_window),
        block_table=block_table,
        softcap=impl.logits_soft_cap,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        sinks=getattr(impl, "sinks", None),
        output_scale=None,
    )


def _sweep_bounds(worker: "Worker") -> _SweepBounds:
    """Derive the input-space sweep bounds from vLLM's live config."""
    runner = worker.model_runner
    scheduler_config = getattr(worker, "scheduler_config", None)
    max_model_len = int(getattr(runner, "max_model_len", 0) or 0)
    max_num_seqs = int(getattr(scheduler_config, "max_num_seqs", 1) or 1)
    max_num_batched_tokens = int(
        getattr(scheduler_config, "max_num_batched_tokens", 0) or max_model_len
    )
    spec_config = getattr(getattr(worker, "vllm_config", None), "speculative_config", None)
    num_speculative_tokens = int(getattr(spec_config, "num_speculative_tokens", 0) or 0)
    return _SweepBounds(
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        non_decode_query_len=1 + num_speculative_tokens,
    )


@torch.inference_mode()
def rocm_aiter_unified_attn_warmup(
    worker: "Worker", targets: list[_WarmupTarget], device: torch.device
) -> None:
    """Launch ``unified_attention`` across every warmup spec for each target."""
    from aiter.ops.triton.unified_attention import unified_attention

    bounds = _sweep_bounds(worker)

    for target in targets:
        specs = build_launch_specs(target.signature, bounds)
        for spec in specs:
            _launch_one(
                unified_attention,
                target.layer,
                target.kv_cache,
                target.signature,
                spec,
                device,
            )


def rocm_aiter_unified_attn_warmup_if_needed(worker: "Worker") -> None:
    """Precompile AITER unified-attention kernels for the loaded model"""
    runner = worker.model_runner
    try:
        q_dtype = getattr(runner, "dtype", torch.bfloat16)
        targets = collect_unique_warmup_targets(runner, q_dtype)
        if not targets:
            return

        device = getattr(runner, "device", torch.device("cuda"))
        rocm_aiter_unified_attn_warmup(worker, targets, device)

        torch.accelerator.synchronize(device)
        logger.info(
            "Warmed AITER unified attention: %d signature(s)",
            len(targets),
        )
    except Exception:
        logger.warning("Skipping AITER unified attention warmup.", exc_info=True)
