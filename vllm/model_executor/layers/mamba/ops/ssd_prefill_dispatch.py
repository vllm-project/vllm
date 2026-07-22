# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatch support for the optional FlashInfer Mamba2 SSD prefill kernel.

The default prefill implementation remains vLLM's Triton kernel. FlashInfer is
an explicit, fail-closed selection with a deliberately narrow initial support
matrix. Per-iteration conditions that cannot use FlashInfer fall back to the
unchanged Triton path in ``MambaMixer2``.
"""

from __future__ import annotations

import inspect
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.config.mamba import MambaPrefillBackendEnum
from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.config import CacheConfig, ModelConfig

logger = init_logger(__name__)

_FLASHINFER_VARLEN_STATE_API_VERSION = 2
_SUPPORTED_CAPABILITIES = {(10, 0), (10, 3), (11, 0)}


class Mamba2PrefillFallbackReason(str, Enum):
    MISSING_METADATA = "missing_metadata"
    INVALID_METADATA = "invalid_metadata"
    CAPTURE_UNSUPPORTED = "capture_unsupported"
    RUNTIME_SHAPE = "runtime_shape"
    RUNTIME_DTYPE = "runtime_dtype"
    STATE_CACHE_LAYOUT = "state_cache_layout"
    OUTPUT_LAYOUT = "output_layout"


@dataclass
class Mamba2PrefillDispatchStats:
    """Worker-local layer-work counters used for engagement diagnostics."""

    flashinfer_layer_invocations: int = 0
    flashinfer_layer_tokens: int = 0
    fallback_layer_invocations: Counter[str] = field(default_factory=Counter)
    fallback_layer_tokens: Counter[str] = field(default_factory=Counter)


_STATS = Mamba2PrefillDispatchStats()


def get_mamba2_prefill_dispatch_stats() -> Mamba2PrefillDispatchStats:
    return _STATS


def reset_mamba2_prefill_dispatch_stats() -> None:
    global _STATS
    _STATS = Mamba2PrefillDispatchStats()


@dataclass(frozen=True)
class FlashInferMamba2PrefillRequest:
    x: torch.Tensor
    dt: torch.Tensor
    A: torch.Tensor
    B: torch.Tensor
    C: torch.Tensor
    D: torch.Tensor
    dt_bias: torch.Tensor
    out: torch.Tensor
    state_cache: torch.Tensor
    initial_states: torch.Tensor | None
    seq_idx: torch.Tensor | None
    token_dst_indices: torch.Tensor | None
    chunk_indices: torch.Tensor | None
    chunk_offsets: torch.Tensor | None
    seq_chunk_cumsum: torch.Tensor | None
    intermediate_state_indices: torch.Tensor | None
    chunk_size: int
    num_seqs: int
    valid_seqlen: int
    padded_seqlen: int
    requires_repacking: bool = False
    is_warmup: bool = False


_SSD_OBJECTS: dict[tuple[Any, ...], Any] = {}
_WARMED_SIGNATURES: set[tuple[Any, ...]] = set()


def _load_flashinfer_ssd_class() -> type:
    try:
        from flashinfer.mamba import (
            SSD_COMBINED_VARLEN_STATE_API_VERSION,
            SSDCombined,
        )
    except ImportError as exc:
        raise RuntimeError(
            "FlashInfer with the packed-varlen SSD state API is required for "
            "--mamba-prefill-backend flashinfer"
        ) from exc

    if SSD_COMBINED_VARLEN_STATE_API_VERSION != _FLASHINFER_VARLEN_STATE_API_VERSION:
        raise RuntimeError(
            "Unsupported FlashInfer SSD packed-varlen state API version: "
            f"expected {_FLASHINFER_VARLEN_STATE_API_VERSION}, got "
            f"{SSD_COMBINED_VARLEN_STATE_API_VERSION}"
        )
    if (
        getattr(SSDCombined, "varlen_state_api_version", None)
        != _FLASHINFER_VARLEN_STATE_API_VERSION
    ):
        raise RuntimeError("FlashInfer SSDCombined is missing varlen state API v2")
    if not getattr(SSDCombined, "supports_fp32_processed_dt", False):
        raise RuntimeError(
            "FlashInfer SSDCombined API v2 must support FP32 processed dt"
        )
    if not getattr(SSDCombined, "supports_strided_intermediate_states", False):
        raise RuntimeError(
            "FlashInfer SSDCombined API v2 must support strided state stores"
        )

    required_init = {
        "has_intermediate_states",
        "output_token_major",
        "dt_processed_dtype",
    }
    required_run = {
        "num_seqs",
        "valid_seqlen",
        "intermediate_state_indices",
        "intermediate_states",
        "final_states",
    }
    missing_init = required_init - set(
        inspect.signature(SSDCombined.__init__).parameters
    )
    missing_run = required_run - set(inspect.signature(SSDCombined.run).parameters)
    if missing_init or missing_run:
        raise RuntimeError(
            "FlashInfer SSD packed-varlen API v2 has an incomplete signature: "
            f"missing constructor={sorted(missing_init)}, run={sorted(missing_run)}"
        )
    return SSDCombined


def flashinfer_prefill_selected(vllm_config: VllmConfig) -> bool:
    return (
        vllm_config.mamba_config.prefill_backend
        == MambaPrefillBackendEnum.FLASHINFER
    )


def validate_flashinfer_mamba2_prefill(
    *,
    vllm_config: VllmConfig,
    model_config: ModelConfig | None,
    cache_config: CacheConfig | None,
    tp_size: int,
    nheads: int,
    headdim: int,
    dstate: int,
    ngroups: int,
    state_dtype: torch.dtype,
    layer_name: str,
) -> None:
    """Fail closed at startup when an explicit FI request is unsupported."""
    if not flashinfer_prefill_selected(vllm_config):
        return

    failures: list[str] = []
    capability = current_platform.get_device_capability()
    capability_tuple = tuple(capability) if capability is not None else None
    if not current_platform.is_cuda():
        failures.append("platform must be NVIDIA CUDA")
    if capability_tuple not in _SUPPORTED_CAPABILITIES:
        failures.append(
            "compute capability must be one of SM100/SM103/SM110 "
            f"(got {capability_tuple})"
        )
    if tp_size != 1:
        failures.append(f"tensor parallel size must be 1 (got {tp_size})")
    if model_config is None:
        failures.append("model_config is unavailable")
        chunk_size = None
        io_dtype = None
    else:
        chunk_size = model_config.get_mamba_chunk_size()
        io_dtype = model_config.dtype
    if io_dtype != torch.bfloat16:
        failures.append(f"model I/O dtype must be bfloat16 (got {io_dtype})")
    if state_dtype != torch.float16:
        failures.append(f"SSM cache dtype must be float16 (got {state_dtype})")
    if chunk_size != 128:
        failures.append(f"Mamba chunk size must be 128 (got {chunk_size})")
    if (nheads, headdim, dstate, ngroups) != (64, 64, 128, 8):
        failures.append(
            "local Mamba2 shape must be H64/D64/N128/G8 "
            f"(got H{nheads}/D{headdim}/N{dstate}/G{ngroups})"
        )
    if cache_config is None:
        failures.append("cache_config is unavailable")
    else:
        if cache_config.mamba_cache_mode != "all":
            failures.append(
                "Mamba cache mode must be 'all' "
                f"(got {cache_config.mamba_cache_mode!r})"
            )
        block_size = cache_config.mamba_block_size
        if block_size is None or block_size % 128 != 0:
            failures.append(
                "Mamba cache block size must be divisible by 128 "
                f"(got {block_size})"
            )
    if vllm_config.num_speculative_tokens != 0:
        failures.append("speculative decoding is not supported")

    try:
        _load_flashinfer_ssd_class()
    except RuntimeError as exc:
        failures.append(str(exc))

    if failures:
        detail = "; ".join(failures)
        raise ValueError(
            f"FlashInfer Mamba2 SSD prefill is unsupported for layer "
            f"{layer_name!r}: {detail}"
        )

    logger.info_once(
        "Using FlashInfer Mamba2 SSD prefill backend: SM%s, BF16 I/O, "
        "FP16 state, H%d/D%d/N%d/G%d, chunk=%d, cache_block=%d, TP1.",
        "".join(str(x) for x in capability_tuple),
        nheads,
        headdim,
        dstate,
        ngroups,
        chunk_size,
        cache_config.mamba_block_size,
    )


def _ssd_key(request: FlashInferMamba2PrefillRequest) -> tuple[Any, ...]:
    return (
        request.x.device.index,
        request.x.dtype,
        request.state_cache.dtype,
        request.chunk_size,
        request.x.shape[1],
        request.x.shape[2],
        request.B.shape[1],
        request.B.shape[2],
        request.initial_states is not None,
        torch.float32,  # serving requires FP32 processed dt
    )


def _get_ssd(request: FlashInferMamba2PrefillRequest):
    key = _ssd_key(request)
    ssd = _SSD_OBJECTS.get(key)
    if ssd is None:
        SSDCombined = _load_flashinfer_ssd_class()
        ssd = SSDCombined(
            chunk_size=request.chunk_size,
            nheads=request.x.shape[1],
            headdim=request.x.shape[2],
            dstate=request.B.shape[2],
            ngroups=request.B.shape[1],
            io_dtype=request.x.dtype,
            state_dtype=request.state_cache.dtype,
            has_d=True,
            d_has_hdim=False,
            has_initial_states=request.initial_states is not None,
            has_varlen=True,
            has_z=False,
            seq_idx_dtype=torch.int32,
            has_intermediate_states=True,
            output_token_major=True,
            dt_processed_dtype=torch.float32,
        )
        _SSD_OBJECTS[key] = ssd
    return ssd


def _fallback(
    reason: Mamba2PrefillFallbackReason, num_tokens: int
) -> Mamba2PrefillFallbackReason:
    _STATS.fallback_layer_invocations[reason.value] += 1
    _STATS.fallback_layer_tokens[reason.value] += num_tokens
    logger.warning_once(
        "Falling back to Triton Mamba2 SSD prefill for dynamic reason: %s",
        reason.value,
    )
    return reason


def has_supported_mamba_state_cache_layout(state_cache: torch.Tensor) -> bool:
    """Return whether only the cache-row axis carries unified-page padding."""
    if state_cache.ndim != 4:
        return False
    _, nheads, headdim, dstate = state_cache.shape
    compact_row_elements = nheads * headdim * dstate
    expected_inner_strides = (headdim * dstate, dstate, 1)
    return (
        state_cache.stride()[1:] == expected_inner_strides
        and state_cache.stride(0) >= compact_row_elements
        and state_cache.data_ptr() % 16 == 0
        and state_cache.stride(0) * state_cache.element_size() % 16 == 0
    )


def run_flashinfer_mamba2_prefill(
    request: FlashInferMamba2PrefillRequest,
) -> Mamba2PrefillFallbackReason | None:
    """Run FlashInfer, or return a reason for the caller's Triton fallback."""
    metadata = (
        request.seq_idx,
        request.token_dst_indices,
        request.chunk_indices,
        request.chunk_offsets,
        request.seq_chunk_cumsum,
        request.intermediate_state_indices,
    )
    if any(value is None for value in metadata):
        return _fallback(
            Mamba2PrefillFallbackReason.MISSING_METADATA, request.valid_seqlen
        )
    assert request.seq_idx is not None
    assert request.token_dst_indices is not None
    assert request.chunk_indices is not None
    assert request.chunk_offsets is not None
    assert request.seq_chunk_cumsum is not None
    assert request.intermediate_state_indices is not None

    if (
        request.valid_seqlen <= 0
        or request.token_dst_indices.shape != (request.x.shape[0],)
        or request.padded_seqlen < request.valid_seqlen
        or request.padded_seqlen % request.chunk_size != 0
        or request.num_seqs <= 0
        or request.seq_idx.shape != (1, request.padded_seqlen)
        or request.chunk_indices.shape != request.chunk_offsets.shape
        or request.intermediate_state_indices.shape
        != request.chunk_indices.shape
        or request.seq_chunk_cumsum.shape != (request.num_seqs + 1,)
    ):
        return _fallback(
            Mamba2PrefillFallbackReason.INVALID_METADATA, request.valid_seqlen
        )
    metadata_tensors = (
        request.seq_idx,
        request.chunk_indices,
        request.chunk_offsets,
        request.seq_chunk_cumsum,
        request.intermediate_state_indices,
    )
    if any(
        tensor.dtype != torch.int32 or not tensor.is_contiguous()
        for tensor in metadata_tensors
    ):
        return _fallback(
            Mamba2PrefillFallbackReason.INVALID_METADATA, request.valid_seqlen
        )
    if (
        request.token_dst_indices.dtype != torch.int64
        or not request.token_dst_indices.is_contiguous()
    ):
        return _fallback(
            Mamba2PrefillFallbackReason.INVALID_METADATA, request.valid_seqlen
        )
    if request.x.device.type != "cuda":
        return _fallback(
            Mamba2PrefillFallbackReason.RUNTIME_SHAPE, request.valid_seqlen
        )
    request_device = request.x.device
    runtime_tensors = (
        request.dt,
        request.A,
        request.B,
        request.C,
        request.D,
        request.dt_bias,
        request.out,
        request.state_cache,
        request.token_dst_indices,
        *metadata_tensors,
    )
    if request.initial_states is not None:
        runtime_tensors += (request.initial_states,)
    if any(tensor.device != request_device for tensor in runtime_tensors):
        return _fallback(
            Mamba2PrefillFallbackReason.RUNTIME_SHAPE, request.valid_seqlen
        )
    if torch.cuda.is_current_stream_capturing():
        return _fallback(
            Mamba2PrefillFallbackReason.CAPTURE_UNSUPPORTED,
            request.valid_seqlen,
        )
    if request.x.ndim != 3:
        return _fallback(
            Mamba2PrefillFallbackReason.RUNTIME_SHAPE, request.valid_seqlen
        )
    tokens, nheads, headdim = request.x.shape
    if request.B.ndim != 3:
        return _fallback(
            Mamba2PrefillFallbackReason.RUNTIME_SHAPE, request.valid_seqlen
        )
    _, ngroups, dstate = request.B.shape
    if (
        request.dt.shape != (tokens, nheads)
        or request.B.shape != (tokens, ngroups, dstate)
        or request.C.shape != request.B.shape
        or request.A.shape != (nheads,)
        or request.D.shape != (nheads,)
        or request.dt_bias.shape != (nheads,)
        or request.out.shape != request.x.shape
        or tuple(request.state_cache.shape[1:]) != (nheads, headdim, dstate)
        or (
            request.initial_states is not None
            and request.initial_states.shape
            != (request.num_seqs, nheads, headdim, dstate)
        )
    ):
        return _fallback(
            Mamba2PrefillFallbackReason.RUNTIME_SHAPE, request.valid_seqlen
        )
    if (
        request.x.dtype != torch.bfloat16
        or request.dt.dtype != torch.bfloat16
        or request.B.dtype != torch.bfloat16
        or request.C.dtype != torch.bfloat16
        or request.A.dtype != torch.float32
        or request.D.dtype != torch.bfloat16
        or request.dt_bias.dtype not in (torch.bfloat16, torch.float32)
        or request.out.dtype != torch.bfloat16
        or request.state_cache.dtype != torch.float16
        or (
            request.initial_states is not None
            and request.initial_states.dtype != torch.float16
        )
    ):
        return _fallback(
            Mamba2PrefillFallbackReason.RUNTIME_DTYPE, request.valid_seqlen
        )
    if not has_supported_mamba_state_cache_layout(request.state_cache):
        return _fallback(
            Mamba2PrefillFallbackReason.STATE_CACHE_LAYOUT,
            request.valid_seqlen,
        )
    if not request.out.is_contiguous():
        return _fallback(
            Mamba2PrefillFallbackReason.OUTPUT_LAYOUT, request.valid_seqlen
        )
    if (
        request.initial_states is not None
        and not request.initial_states.is_contiguous()
    ):
        return _fallback(
            Mamba2PrefillFallbackReason.STATE_CACHE_LAYOUT,
            request.valid_seqlen,
        )

    if not request.requires_repacking and request.padded_seqlen == request.valid_seqlen:
        x = request.x.unsqueeze(0)
        dt = request.dt.unsqueeze(0)
        B = request.B.unsqueeze(0)
        C = request.C.unsqueeze(0)
        out = request.out.unsqueeze(0)
    else:
        x = request.x.new_zeros(
            1, request.padded_seqlen, request.x.shape[1], request.x.shape[2]
        )
        # Dummy tokens must be exact recurrence no-ops after dt_bias and
        # softplus. The minimum finite BF16 value keeps preprocessing finite
        # while underflowing softplus to exactly zero for finite model bias.
        dt = request.dt.new_full(
            (1, request.padded_seqlen, request.dt.shape[1]),
            torch.finfo(request.dt.dtype).min,
        )
        B = request.B.new_zeros(
            1, request.padded_seqlen, request.B.shape[1], request.B.shape[2]
        )
        C = request.C.new_zeros(
            1, request.padded_seqlen, request.C.shape[1], request.C.shape[2]
        )
        out = request.out.new_empty(
            1, request.padded_seqlen, request.out.shape[1], request.out.shape[2]
        )
        x[0].index_copy_(0, request.token_dst_indices, request.x)
        dt[0].index_copy_(0, request.token_dst_indices, request.dt)
        B[0].index_copy_(0, request.token_dst_indices, request.B)
        C[0].index_copy_(0, request.token_dst_indices, request.C)

    ssd = _get_ssd(request)
    with torch.profiler.record_function("mamba2_ssd_prefill/flashinfer"):
        result, _ = ssd.run(
            x,
            dt,
            request.A,
            B,
            C,
            D=request.D,
            z=None,
            dt_bias=request.dt_bias,
            dt_softplus=True,
            dt_limit=(0.0, float("inf")),
            initial_states=request.initial_states,
            seq_idx=request.seq_idx,
            chunk_indices=request.chunk_indices,
            chunk_offsets=request.chunk_offsets,
            seq_chunk_cumsum=request.seq_chunk_cumsum,
            update_seq_chunk_cumsum=False,
            out=out,
            return_final_states=False,
            num_seqs=request.num_seqs,
            valid_seqlen=request.valid_seqlen,
            intermediate_state_indices=request.intermediate_state_indices,
            intermediate_states=request.state_cache,
            final_states=None,
        )
    if request.requires_repacking:
        request.out.copy_(
            result.squeeze(0).index_select(0, request.token_dst_indices)
        )
    elif request.padded_seqlen != request.valid_seqlen:
        request.out.copy_(result.squeeze(0))

    if not request.is_warmup:
        _STATS.flashinfer_layer_invocations += 1
        _STATS.flashinfer_layer_tokens += request.x.shape[0]
        logger.info_once(
            "Full FlashInfer Mamba2 SSD prefill engaged: source_T=%d, "
            "packed_T=%d, num_seqs=%d, logical_segments=%d; direct state "
            "stores enabled.",
            request.x.shape[0],
            request.padded_seqlen,
            request.num_seqs,
            request.chunk_indices.numel(),
        )
    return None


def warmup_flashinfer_mamba2_prefill(
    *,
    device: torch.device,
    A: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    nheads: int,
    headdim: int,
    dstate: int,
    ngroups: int,
    chunk_size: int,
    state_dtype: torch.dtype,
) -> None:
    """Compile both initial-state variants once without advancing RNG state."""
    signature = (
        device.index,
        nheads,
        headdim,
        dstate,
        ngroups,
        chunk_size,
        state_dtype,
    )
    if signature in _WARMED_SIGNATURES:
        return

    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    with torch.random.fork_rng(devices=[device_index]):
        torch.manual_seed(0)
        x = torch.randn(
            chunk_size,
            nheads,
            headdim,
            device=device,
            dtype=torch.bfloat16,
        )
        dt = torch.randn(
            chunk_size, nheads, device=device, dtype=torch.bfloat16
        )
        B = torch.randn(
            chunk_size,
            ngroups,
            dstate,
            device=device,
            dtype=torch.bfloat16,
        )
        C = torch.randn_like(B)
        out = torch.empty_like(x)
        seq_idx = torch.zeros(
            (1, chunk_size), device=device, dtype=torch.int32
        )
        chunk_indices = torch.zeros(1, device=device, dtype=torch.int32)
        token_dst_indices = torch.arange(
            chunk_size, device=device, dtype=torch.int64
        )
        chunk_offsets = torch.zeros(1, device=device, dtype=torch.int32)
        seq_chunk_cumsum = torch.tensor(
            [0, 1], device=device, dtype=torch.int32
        )
        state_cache = torch.empty(
            1,
            nheads,
            headdim,
            dstate,
            device=device,
            dtype=state_dtype,
        )
        intermediate_state_indices = torch.zeros(
            1, device=device, dtype=torch.int32
        )

        for has_initial_states in (False, True):
            initial_states = (
                torch.randn_like(state_cache) if has_initial_states else None
            )
            reason = run_flashinfer_mamba2_prefill(
                FlashInferMamba2PrefillRequest(
                    x=x,
                    dt=dt,
                    A=A,
                    B=B,
                    C=C,
                    D=D,
                    dt_bias=dt_bias,
                    out=out,
                    state_cache=state_cache,
                    initial_states=initial_states,
                    seq_idx=seq_idx,
                    token_dst_indices=token_dst_indices,
                    chunk_indices=chunk_indices,
                    chunk_offsets=chunk_offsets,
                    seq_chunk_cumsum=seq_chunk_cumsum,
                    intermediate_state_indices=intermediate_state_indices,
                    chunk_size=chunk_size,
                    num_seqs=1,
                    valid_seqlen=chunk_size,
                    padded_seqlen=chunk_size,
                    is_warmup=True,
                )
            )
            if reason is not None:
                raise RuntimeError(
                    "FlashInfer Mamba2 SSD warmup unexpectedly fell back: "
                    f"{reason.value}"
                )

    _WARMED_SIGNATURES.add(signature)
