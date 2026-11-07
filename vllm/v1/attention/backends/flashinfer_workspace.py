# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in startup sizing for the native FlashInfer attention workspace.

The default reservation stays in place unless
``VLLM_FLASHINFER_WORKSPACE_AUTO_SIZE`` is set. Sizing is FlashInfer's
``workspace_size_upper_bound`` (flashinfer-ai/flashinfer#5236). vLLM only
passes geometry and scheduler limits, then allocates the chosen buffer
before memory profiling so a smaller reservation can become KV cache.
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)

# Matches the prefill temp-buffer floor already used by FlashInferMetadataBuilder.
# Wide-head chunks overflow the historical 394 MiB default; never size below this.
FLASHINFER_PREFILL_WORKSPACE_BYTES_PER_ELEM = 16
FLASHINFER_WORKSPACE_BUFFER_SIZE_BATCH_INVARIANT = 2048 * 1024 * 1024

# BatchPrefill/BatchDecode wrappers allocate this int scratch themselves.
_WRAPPER_INT_WORKSPACE_BYTES = 8 * 1024 * 1024
# Study policy: 9/8 over the bound, then round up to a MiB, and at least 32 MiB.
_BOUND_MARGIN_NUM = 9
_BOUND_MARGIN_DEN = 8
_MIN_AUTO_BYTES = 32 * 1024 * 1024
_MIB = 1024 * 1024

_Query = Callable[
    [Literal["prefill", "decode"], "WorkspaceGroup", bool], tuple[int, int]
]


@dataclass(frozen=True)
class WorkspaceGroup:
    """One uniform FlashInfer attention group."""

    num_qo_heads: int
    num_kv_heads: int
    head_dim_qk: int
    head_dim_vo: int
    window_left: int
    logits_soft_cap: float | None
    q_dtype: torch.dtype
    kv_dtype: torch.dtype


@dataclass(frozen=True)
class WorkspaceLimits:
    """Scheduler limits passed through to FlashInfer. vLLM does not size them."""

    max_batch_size: int
    max_total_num_rows: int
    max_num_pages_per_request: int
    page_size: int
    kv_layout: str
    decode_cuda_graph: bool


@dataclass(frozen=True)
class FlashInferWorkspaceDecision:
    """Recorded startup choice. ``selected_bytes`` is what will be reserved."""

    configured_bytes: int
    selected_bytes: int
    source: str
    fallback_reason: str | None = None
    float_bound_bytes: int | None = None
    int_bound_bytes: int | None = None


_decision: FlashInferWorkspaceDecision | None = None
_reserved: torch.Tensor | None = None


def reset_flashinfer_workspace_state() -> None:
    """Drop the startup decision and buffer. Used on runner shutdown and in tests."""
    global _decision, _reserved
    _decision = None
    _reserved = None


def native_workspace_bytes(
    configured_bytes: int,
    *,
    max_num_batched_tokens: int,
    num_qo_heads: int,
    head_dim: int,
    batch_invariant: bool,
) -> int:
    """Bytes reserved when auto-shrink is off.

    Grows above ``configured_bytes`` for the prefill head footprint. Does not
    shrink. Batch-invariant mode replaces the size outright.
    """
    if batch_invariant:
        return FLASHINFER_WORKSPACE_BUFFER_SIZE_BATCH_INVARIANT
    return max(
        configured_bytes,
        prefill_workspace_floor(max_num_batched_tokens, num_qo_heads, head_dim),
    )


def prefill_workspace_floor(
    max_num_batched_tokens: int, num_qo_heads: int, head_dim: int
) -> int:
    return (
        max_num_batched_tokens
        * num_qo_heads
        * head_dim
        * FLASHINFER_PREFILL_WORKSPACE_BYTES_PER_ELEM
    )


def _ceil_mib(nbytes: int) -> int:
    return (nbytes + _MIB - 1) // _MIB * _MIB


def _with_margin(nbytes: int) -> int:
    return (nbytes * _BOUND_MARGIN_NUM + _BOUND_MARGIN_DEN - 1) // _BOUND_MARGIN_DEN


def select_startup_workspace(
    *,
    configured_bytes: int,
    explicit: bool,
    batch_invariant: bool,
    groups: list[WorkspaceGroup],
    limits: WorkspaceLimits,
    unsupported_reason: str | None,
    query: _Query,
) -> FlashInferWorkspaceDecision:
    """Choose a workspace size. ``query`` is FlashInfer, or a test double.

    An explicit user size and every unsupported route keep the native
    reservation. A FlashInfer bound is adopted only when it is smaller than
    that reservation and still covers the prefill floor.
    """
    floor = 0
    for group in groups:
        floor = max(
            floor,
            prefill_workspace_floor(
                limits.max_total_num_rows, group.num_qo_heads, group.head_dim_qk
            ),
        )
    native = (
        FLASHINFER_WORKSPACE_BUFFER_SIZE_BATCH_INVARIANT
        if batch_invariant
        else max(configured_bytes, floor)
    )
    if batch_invariant:
        return FlashInferWorkspaceDecision(
            configured_bytes,
            native,
            "configured",
            "batch-invariant mode uses a fixed workspace",
        )
    if explicit:
        grown = None
        if native > configured_bytes:
            grown = (
                "prefill head footprint is larger than the explicit workspace; "
                "native allocation still grows to cover it"
            )
        return FlashInferWorkspaceDecision(configured_bytes, native, "explicit", grown)
    if unsupported_reason:
        return FlashInferWorkspaceDecision(
            configured_bytes, native, "configured", unsupported_reason
        )
    if not groups:
        return FlashInferWorkspaceDecision(
            configured_bytes,
            native,
            "configured",
            "no FlashInfer attention group to size",
        )

    try:
        bounds: list[tuple[int, int]] = []
        for group in groups:
            bounds.append(query("prefill", group, False))
            bounds.append(query("decode", group, False))
            if limits.decode_cuda_graph:
                bounds.append(query("decode", group, True))
    except NotImplementedError as exc:
        return FlashInferWorkspaceDecision(
            configured_bytes,
            native,
            "configured",
            f"FlashInfer has no workspace upper bound for this route ({exc})",
        )
    except Exception as exc:
        logger.exception("FlashInfer workspace upper bound failed")
        return FlashInferWorkspaceDecision(
            configured_bytes,
            native,
            "configured",
            f"FlashInfer workspace upper bound failed: {type(exc).__name__}: {exc}",
        )

    float_bound = max(item[0] for item in bounds)
    int_bound = max(item[1] for item in bounds)
    if min(float_bound, int_bound) < 0:
        return FlashInferWorkspaceDecision(
            configured_bytes,
            native,
            "configured",
            "FlashInfer returned a negative workspace bound",
            float_bound,
            int_bound,
        )
    if int_bound > _WRAPPER_INT_WORKSPACE_BYTES:
        return FlashInferWorkspaceDecision(
            configured_bytes,
            native,
            "configured",
            (
                f"int workspace bound {int_bound} bytes exceeds the "
                f"{_WRAPPER_INT_WORKSPACE_BYTES} bytes the FlashInfer wrapper "
                "allocates internally"
            ),
            float_bound,
            int_bound,
        )

    recommended = _ceil_mib(max(_MIN_AUTO_BYTES, _with_margin(float_bound), floor))
    if recommended >= native:
        return FlashInferWorkspaceDecision(
            configured_bytes,
            native,
            "configured",
            "upper bound does not reduce the reservation",
            float_bound,
            int_bound,
        )
    return FlashInferWorkspaceDecision(
        configured_bytes,
        recommended,
        "flashinfer",
        None,
        float_bound,
        int_bound,
    )


def reserved_flashinfer_workspace(
    device: torch.device, floor_bytes: int
) -> torch.Tensor | None:
    """Return the startup buffer, or None when this process kept native allocation.

    Raises if a later builder needs more than the buffer that was fixed before
    memory profiling. Growing it afterwards would steal the KV budget.
    """
    if _reserved is None:
        return None
    if _reserved.device.type != device.type or (
        device.type != "cpu" and _reserved.device.index != _device_index(device)
    ):
        raise RuntimeError(
            "FlashInfer startup workspace is on "
            f"{_reserved.device}, not {device}. Refusing to allocate a second buffer."
        )
    if floor_bytes > _reserved.numel():
        raise RuntimeError(
            "FlashInfer attention needs at least "
            f"{floor_bytes} workspace bytes after startup sizing reserved "
            f"{_reserved.numel()}. Disable VLLM_FLASHINFER_WORKSPACE_AUTO_SIZE "
            "or set VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE."
        )
    return _reserved


def _device_index(device: torch.device) -> int | None:
    if device.index is not None:
        return device.index
    if device.type == "cuda":
        return torch.cuda.current_device()
    return 0


def current_flashinfer_workspace_decision() -> FlashInferWorkspaceDecision | None:
    return _decision


def maybe_reserve_flashinfer_workspace(
    vllm_config, device: torch.device
) -> FlashInferWorkspaceDecision | None:
    """Size and allocate before memory profiling. No-op unless auto-size is on."""
    global _decision, _reserved
    if not envs.VLLM_FLASHINFER_WORKSPACE_AUTO_SIZE:
        return None
    if _decision is not None:
        return _decision

    groups, limits, unsupported = _collect_request(vllm_config)
    if groups is None and unsupported is None:
        logger.info(
            "FlashInfer workspace auto-size skipped: no FlashInfer decoder layers."
        )
        return None

    assert limits is not None
    decision = select_startup_workspace(
        configured_bytes=envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE,
        explicit=envs.is_set("VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE"),
        batch_invariant=bool(envs.VLLM_BATCH_INVARIANT),
        groups=groups or [],
        limits=limits,
        unsupported_reason=unsupported,
        query=lambda kind, group, use_cuda_graph: _query_flashinfer(
            kind, group, use_cuda_graph, limits, device
        ),
    )
    _log_decision(decision)
    if decision.source != "flashinfer":
        _decision = decision
        return decision
    try:
        gc.collect()
        if device.type != "cpu":
            torch.accelerator.empty_cache()
        reserved = torch.zeros(
            decision.selected_bytes, dtype=torch.uint8, device=device
        )
    except Exception:
        logger.exception("Failed to reserve the FlashInfer startup workspace")
        raise
    _decision = decision
    _reserved = reserved
    return decision


def log_flashinfer_workspace_kv_capacity(
    vllm_config, kv_cache_config, *, num_blocks: int | None = None
) -> None:
    """Log token capacity once the KV config exists. No-op when auto-size is off."""
    decision = _decision
    if decision is None:
        return
    if num_blocks is not None and num_blocks != kv_cache_config.num_blocks:
        from copy import copy

        kv_cache_config = copy(kv_cache_config)
        kv_cache_config.num_blocks = num_blocks
    try:
        from vllm.v1.core.kv_cache_utils import get_kv_cache_capacity

        num_tokens, max_concurrency = get_kv_cache_capacity(
            vllm_config, kv_cache_config
        )
        capacity = (
            f"{num_tokens:,} tokens, max concurrency {max_concurrency:.2f}x "
            f"for {vllm_config.model_config.max_model_len:,} tokens/request"
        )
    except Exception:
        logger.exception("Failed to format FlashInfer workspace KV capacity")
        capacity = f"{kv_cache_config.num_blocks} blocks"
    logger.info(
        "FlashInfer workspace auto-size KV capacity: %s. source=%s "
        "configured=%s selected=%s%s",
        capacity,
        decision.source,
        _fmt_bytes(decision.configured_bytes),
        _fmt_bytes(decision.selected_bytes),
        f" fallback={decision.fallback_reason}" if decision.fallback_reason else "",
    )


def _log_decision(decision: FlashInferWorkspaceDecision) -> None:
    bound = ""
    if decision.float_bound_bytes is not None:
        bound = (
            f" float_bound={_fmt_bytes(decision.float_bound_bytes)}"
            f" int_bound={_fmt_bytes(decision.int_bound_bytes or 0)}"
        )
    fallback = (
        f" fallback={decision.fallback_reason}" if decision.fallback_reason else ""
    )
    logger.info(
        "FlashInfer workspace auto-size: configured=%s selected=%s source=%s%s%s",
        _fmt_bytes(decision.configured_bytes),
        _fmt_bytes(decision.selected_bytes),
        decision.source,
        bound,
        fallback,
    )


def _fmt_bytes(nbytes: int) -> str:
    return f"{nbytes / _MIB:.2f} MiB ({nbytes} bytes)"


def _collect_request(
    vllm_config,
) -> tuple[list[WorkspaceGroup] | None, WorkspaceLimits | None, str | None]:
    """Return groups, limits, and a fallback reason.

    ``(None, None, None)`` means this model has no FlashInfer decoder layers.
    """
    from vllm.config import CUDAGraphMode
    from vllm.model_executor.layers.attention import Attention
    from vllm.utils.math_utils import cdiv
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends.utils import get_flashinfer_layout_string

    layers = [
        layer
        for layer in vllm_config.compilation_config.static_forward_context.values()
        if isinstance(layer, Attention)
        and layer.attn_type == AttentionType.DECODER
        and layer.get_attn_backend().get_name() == "FLASHINFER"
    ]
    if not layers:
        return None, None, None

    reason = _route_unsupported_reason(vllm_config)
    try:
        layout = get_flashinfer_layout_string(
            vllm_config.cache_config.get_resolved_kv_cache_layout()
        )
    except Exception as exc:
        layout = "NHD"
        reason = reason or (
            f"KV cache layout is unavailable ({type(exc).__name__}: {exc})"
        )

    scheduler = vllm_config.scheduler_config
    block_size = vllm_config.cache_config.block_size
    max_batch = getattr(scheduler, "max_num_seqs", None)
    max_rows = getattr(scheduler, "max_num_batched_tokens", None)
    max_len = vllm_config.model_config.max_model_len
    if reason is None and (
        not block_size
        or not max_batch
        or not max_rows
        or not max_len
        or max_batch < 1
        or max_rows < 1
    ):
        reason = "scheduler limits or block size are not positive integers"

    groups: list[WorkspaceGroup] = []
    model_dtype = vllm_config.model_config.dtype
    for layer in layers:
        group, group_reason = _group_from_layer(layer, model_dtype)
        if group_reason:
            reason = reason or group_reason
        if group is not None:
            groups.append(group)

    if reason is None and not groups:
        reason = "no sizeable FlashInfer attention group"

    pages = 1
    if block_size and max_len:
        pages = max(1, cdiv(int(max_len), int(block_size)))
    try:
        decode_cuda_graph = (
            vllm_config.compilation_config.cudagraph_mode.decode_mode()
            == CUDAGraphMode.FULL
        )
    except Exception as exc:
        decode_cuda_graph = False
        reason = reason or (
            f"CUDA graph mode is unavailable ({type(exc).__name__}: {exc})"
        )
    limits = WorkspaceLimits(
        max_batch_size=int(max_batch or 1),
        max_total_num_rows=int(max_rows or 1),
        max_num_pages_per_request=pages,
        page_size=int(block_size or 1),
        kv_layout=layout,
        decode_cuda_graph=decode_cuda_graph,
    )
    return groups, limits, reason


def _route_unsupported_reason(vllm_config) -> str | None:
    if vllm_config.cache_config.kv_cache_memory_bytes:
        return (
            "kv_cache_memory_bytes fixes the KV budget, so shrinking the "
            "workspace would not increase KV capacity"
        )
    parallel = vllm_config.parallel_config
    if getattr(parallel, "decode_context_parallel_size", 1) > 1:
        return "decode context parallel is not a supported startup-sizing route"
    if vllm_config.speculative_config is not None:
        return "speculative decoding is not a supported startup-sizing route"
    attention = vllm_config.attention_config
    if getattr(attention, "use_non_causal", False):
        return "non-causal attention is not a supported startup-sizing route"
    return None


def _group_from_layer(
    layer, model_dtype: torch.dtype
) -> tuple[WorkspaceGroup | None, str | None]:
    impl = getattr(layer, "impl", layer)
    num_qo = int(getattr(impl, "num_heads", 0) or 0)
    num_kv = int(getattr(impl, "num_kv_heads", 0) or 0)
    head_qk = int(getattr(impl, "head_size", 0) or getattr(layer, "head_size", 0) or 0)
    head_vo = int(getattr(layer, "head_size_v", head_qk) or head_qk)
    if num_qo < 1 or num_kv < 1 or head_qk < 1 or num_qo % num_kv != 0:
        return None, (
            f"unsupported head geometry qo={num_qo} kv={num_kv} head={head_qk}"
        )
    if model_dtype in (torch.float16, torch.bfloat16):
        q_dtype = model_dtype
    else:
        q_dtype = torch.bfloat16
    kv_dtype = getattr(layer, "kv_cache_torch_dtype", None) or q_dtype
    if kv_dtype not in (torch.float16, torch.bfloat16):
        kv_dtype = q_dtype
    window_left = getattr(impl, "window_left", None)
    if window_left is None:
        sliding = getattr(layer, "sliding_window", None)
        window_left = -1 if not sliding else int(sliding) - 1
    group = WorkspaceGroup(
        num_qo_heads=num_qo,
        num_kv_heads=num_kv,
        head_dim_qk=head_qk,
        head_dim_vo=head_qk,
        window_left=int(window_left),
        logits_soft_cap=getattr(impl, "logits_soft_cap", None),
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
    )
    if head_vo != head_qk:
        return group, "distinct QK/VO head dimensions are not sized at startup"
    if getattr(impl, "sinks", None) is not None or getattr(layer, "has_sink", False):
        return group, "attention sinks are not a supported startup-sizing route"
    if getattr(impl, "alibi_slopes", None) is not None:
        return group, "ALiBi is not a supported startup-sizing route"
    if getattr(layer, "use_mm_prefix", False):
        return (
            group,
            "multimodal prefix attention is not a supported startup-sizing route",
        )
    cache_dtype = str(
        getattr(impl, "cache_dtype", None) or getattr(layer, "kv_cache_dtype", "auto")
    )
    if cache_dtype not in {"auto", "float16", "half", "bfloat16"}:
        return group, (
            f"KV cache dtype {cache_dtype} is not a supported startup-sizing route"
        )
    if model_dtype not in (torch.float16, torch.bfloat16):
        return group, "startup sizing only covers fp16/bf16 query and KV"
    return group, None


def _query_flashinfer(
    kind: Literal["prefill", "decode"],
    group: WorkspaceGroup,
    use_cuda_graph: bool,
    limits: WorkspaceLimits,
    device: torch.device,
) -> tuple[int, int]:
    import flashinfer

    if kind == "prefill":
        wrapper_cls = flashinfer.BatchPrefillWithPagedKVCacheWrapper
        kwargs = dict(
            max_batch_size=limits.max_batch_size,
            max_total_num_rows=limits.max_total_num_rows,
            max_num_pages_per_request=limits.max_num_pages_per_request,
            num_qo_heads=group.num_qo_heads,
            num_kv_heads=group.num_kv_heads,
            head_dim_qk=group.head_dim_qk,
            page_size=limits.page_size,
            head_dim_vo=group.head_dim_vo,
            pos_encoding_mode="NONE",
            use_fp16_qk_reduction=False,
            use_custom_mask=False,
            window_left=group.window_left,
            logits_soft_cap=group.logits_soft_cap,
            q_data_type=group.q_dtype,
            kv_data_type=group.kv_dtype,
            o_data_type=group.q_dtype,
            fixed_split_size=-1,
            disable_split_kv=False,
        )
    else:
        wrapper_cls = flashinfer.BatchDecodeWithPagedKVCacheWrapper
        kwargs = dict(
            max_batch_size=limits.max_batch_size,
            max_num_pages_per_request=limits.max_num_pages_per_request,
            num_qo_heads=group.num_qo_heads,
            num_kv_heads=group.num_kv_heads,
            head_dim=group.head_dim_qk,
            page_size=limits.page_size,
            pos_encoding_mode="NONE",
            window_left=group.window_left,
            logits_soft_cap=group.logits_soft_cap,
            q_data_type=group.q_dtype,
            kv_data_type=group.kv_dtype,
            o_data_type=group.q_dtype,
            q_len_per_req=1,
            fixed_split_size=-1,
            disable_split_kv=False,
        )
    if not hasattr(wrapper_cls, "workspace_size_upper_bound"):
        raise NotImplementedError(
            f"{wrapper_cls.__name__}.workspace_size_upper_bound is not in this "
            "FlashInfer build"
        )
    scratch = torch.empty(16, dtype=torch.uint8, device=device)
    graph_bufs: dict[str, torch.Tensor] = {}
    if use_cuda_graph:
        # CUDA-graph wrappers refuse to construct without the fixed-shape
        # buffers. They only select the planner mode; the bound itself uses
        # ``limits``.
        batch = limits.max_batch_size
        pages = max(1, batch * limits.max_num_pages_per_request)
        indptr = torch.empty(batch + 1, dtype=torch.int32, device=device)
        indices = torch.empty(pages, dtype=torch.int32, device=device)
        last_page = torch.empty(batch, dtype=torch.int32, device=device)
        if kind == "prefill":
            graph_bufs = {
                "qo_indptr_buf": indptr,
                "paged_kv_indptr_buf": indptr,
                "paged_kv_indices_buf": indices,
                "paged_kv_last_page_len_buf": last_page,
            }
        else:
            graph_bufs = {
                "paged_kv_indptr_buffer": indptr,
                "paged_kv_indices_buffer": indices,
                "paged_kv_last_page_len_buffer": last_page,
            }
    wrapper = None
    try:
        wrapper = wrapper_cls(
            scratch,
            limits.kv_layout,
            use_cuda_graph=use_cuda_graph,
            backend="auto",
            **graph_bufs,
        )
        float_bytes, int_bytes = wrapper.workspace_size_upper_bound(**kwargs)
    finally:
        del wrapper
        del scratch
        graph_bufs.clear()
    return int(float_bytes), int(int_bytes)
