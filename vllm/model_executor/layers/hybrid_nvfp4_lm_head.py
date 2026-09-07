# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental NVFP4 b12x coarse search with BF16 lm-head refinement."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from weakref import WeakSet

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.layers.argmax_triton import (
    indexed_argmax_triton,
    reduce_global_argmax_triton,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.flashinfer import (
    autotune_with_torch_cuda_delay,
    flashinfer_nvfp4_quantize_128x4,
    flashinfer_scaled_fp4_mm,
    has_flashinfer,
    has_flashinfer_b12x_gemm,
)

logger = init_logger(__name__)

_BLOCK_SIZE = 16
_MIN_GEMM_DIMENSION = 128
_NVFP4_MAX_VALUE = 448.0 * 6.0
_DEFAULT_AUTOTUNE_MAX_ROWS = 2048
_TOPK_CANDIDATE_MULTIPLIER = 8
_MAX_AUTO_CANDIDATES = 1024

_WEIGHT_NAME = "_hybrid_nvfp4_lm_head_weight"
_SCALE_NAME = "_hybrid_nvfp4_lm_head_scale"
_GLOBAL_SCALE_NAME = "_hybrid_nvfp4_lm_head_global_scale"
_STATE_NAME = "_hybrid_nvfp4_lm_head_state"
_SHARED_STATE_NAME = "_hybrid_nvfp4_lm_head_shared_state"
_BUFFER_NAMES = (_WEIGHT_NAME, _SCALE_NAME, _GLOBAL_SCALE_NAME)


def _global_scale(tensor: torch.Tensor) -> torch.Tensor:
    max_abs = tensor.abs().amax().float()
    scaled = max_abs.clamp_min(1.0e-12).reciprocal() * _NVFP4_MAX_VALUE
    return torch.where(max_abs > 0, scaled, torch.ones_like(max_abs))


def _quantize_lm_head_weight(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create the auxiliary NVFP4 representation for one BF16 lm-head."""
    padded_output_size = (weight.shape[0] + 31) // 32 * 32
    weight_for_quant = weight
    if padded_output_size != weight.shape[0]:
        weight_for_quant = F.pad(
            weight,
            (0, 0, 0, padded_output_size - weight.shape[0]),
        )
    global_scale = _global_scale(weight_for_quant)
    quantized, scale = flashinfer_nvfp4_quantize_128x4(
        weight_for_quant,
        global_scale,
    )
    return quantized, scale, global_scale


def _candidate_count_for_topk(
    configured_candidates: int,
    top_k: int,
    *,
    output_size: int,
) -> int:
    """Return a transient refinement width for a top-k request."""
    if top_k <= 0 or configured_candidates >= _MAX_AUTO_CANDIDATES:
        return min(configured_candidates, output_size)

    desired = max(configured_candidates, top_k * _TOPK_CANDIDATE_MULTIPLIER)
    if desired <= configured_candidates * 2:
        return min(configured_candidates, output_size)
    if desired >= _MAX_AUTO_CANDIDATES:
        expanded = _MAX_AUTO_CANDIDATES
    else:
        expanded = 1 << (desired - 1).bit_length()
    return min(max(configured_candidates, expanded), output_size)


def _select_candidate_tile(
    num_rows: int,
    num_candidates: int,
    input_size: int,
) -> int:
    if num_candidates < 64:
        return 1
    if input_size > 2048:
        if num_rows <= 32:
            return 1
        return 2
    if num_rows < 32:
        return 1
    if num_rows < 768:
        return 2
    if num_rows < 2048:
        return 4
    return 8


if HAS_TRITON:

    @triton.jit
    def _indexed_bf16_dot_kernel(
        hidden,
        weight,
        indices,
        output,
        hidden_stride_0: tl.constexpr,
        weight_stride_0: tl.constexpr,
        indices_stride_0: tl.constexpr,
        output_stride_0: tl.constexpr,
        num_candidates: tl.constexpr,
        input_size: tl.constexpr,
        block_input_size: tl.constexpr,
    ):
        pair_id = tl.program_id(0)
        row = pair_id // num_candidates
        candidate = pair_id % num_candidates
        token_id = tl.load(indices + row * indices_stride_0 + candidate)
        offsets = tl.arange(0, block_input_size)
        mask = offsets < input_size
        hidden_row = tl.load(
            hidden + row * hidden_stride_0 + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        weight_row = tl.load(
            weight + token_id * weight_stride_0 + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        value = tl.sum(hidden_row * weight_row, axis=0)
        tl.store(output + row * output_stride_0 + candidate, value)

    @triton.jit
    def _tiled_indexed_bf16_dot_kernel(
        hidden,
        weight,
        indices,
        output,
        hidden_stride_0: tl.constexpr,
        weight_stride_0: tl.constexpr,
        indices_stride_0: tl.constexpr,
        output_stride_0: tl.constexpr,
        num_candidates: tl.constexpr,
        input_size: tl.constexpr,
        block_input_size: tl.constexpr,
        block_candidates: tl.constexpr,
    ):
        row = tl.program_id(0)
        candidates = tl.program_id(1) * block_candidates + tl.arange(
            0, block_candidates
        )
        candidate_mask = candidates < num_candidates
        token_ids = tl.load(
            indices + row * indices_stride_0 + candidates,
            mask=candidate_mask,
            other=0,
        )

        offsets = tl.arange(0, block_input_size)
        input_mask = offsets < input_size
        hidden_row = tl.load(
            hidden + row * hidden_stride_0 + offsets,
            mask=input_mask,
            other=0.0,
        ).to(tl.float32)
        weight_rows = tl.load(
            weight + token_ids[:, None] * weight_stride_0 + offsets[None, :],
            mask=candidate_mask[:, None] & input_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        values = tl.sum(weight_rows * hidden_row[None, :], axis=1)
        tl.store(
            output + row * output_stride_0 + candidates,
            values,
            mask=candidate_mask,
        )


def indexed_bf16_dot(
    hidden_states: torch.Tensor,
    bf16_weight: torch.Tensor,
    candidate_indices: torch.Tensor,
    *,
    candidate_tile: int | None = None,
    num_warps: int = 8,
) -> torch.Tensor:
    """Compute selected BF16 logits without a full-vocabulary projection."""
    if not HAS_TRITON:
        selected_weight = bf16_weight[candidate_indices]
        return torch.bmm(
            hidden_states.unsqueeze(1), selected_weight.transpose(1, 2)
        ).squeeze(1)

    assert hidden_states.ndim == 2
    assert bf16_weight.ndim == 2
    assert candidate_indices.ndim == 2
    assert hidden_states.shape[0] == candidate_indices.shape[0]
    assert hidden_states.shape[1] == bf16_weight.shape[1]
    assert hidden_states.dtype == torch.bfloat16
    assert bf16_weight.dtype == torch.bfloat16
    assert hidden_states.is_cuda
    assert bf16_weight.is_cuda
    assert candidate_indices.is_cuda
    assert hidden_states.is_contiguous()
    assert bf16_weight.is_contiguous()
    assert candidate_indices.is_contiguous()
    if num_warps not in (4, 8):
        raise ValueError(f"num_warps must be 4 or 8; got {num_warps}")

    num_rows, num_candidates = candidate_indices.shape
    input_size = hidden_states.shape[1]
    output = torch.empty(
        (num_rows, num_candidates),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    block_input_size = triton.next_power_of_2(input_size)
    if candidate_tile is None:
        candidate_tile = _select_candidate_tile(
            num_rows,
            num_candidates,
            input_size,
        )

    if candidate_tile == 1:
        _indexed_bf16_dot_kernel[(num_rows * num_candidates,)](
            hidden_states,
            bf16_weight,
            candidate_indices,
            output,
            hidden_stride_0=hidden_states.stride(0),
            weight_stride_0=bf16_weight.stride(0),
            indices_stride_0=candidate_indices.stride(0),
            output_stride_0=output.stride(0),
            num_candidates=num_candidates,
            input_size=input_size,
            block_input_size=block_input_size,
            num_warps=num_warps,
        )
    else:
        if candidate_tile not in (2, 4, 8):
            raise ValueError(
                f"candidate_tile must be one of 1, 2, 4, or 8; got {candidate_tile}"
            )
        _tiled_indexed_bf16_dot_kernel[
            (num_rows, triton.cdiv(num_candidates, candidate_tile))
        ](
            hidden_states,
            bf16_weight,
            candidate_indices,
            output,
            hidden_stride_0=hidden_states.stride(0),
            weight_stride_0=bf16_weight.stride(0),
            indices_stride_0=candidate_indices.stride(0),
            output_stride_0=output.stride(0),
            num_candidates=num_candidates,
            input_size=input_size,
            block_input_size=block_input_size,
            block_candidates=candidate_tile,
            num_warps=num_warps,
        )
    return output


def select_lm_head_candidates(
    coarse_logits: torch.Tensor,
    candidates: int,
) -> torch.Tensor:
    """Select an unsorted candidate set from the coarse logits."""
    if candidates <= 0 or candidates > coarse_logits.shape[-1]:
        raise ValueError(
            f"candidate count must be in [1, {coarse_logits.shape[-1]}], "
            f"got {candidates}"
        )
    # Match the argmax/reduction contract for invalid logits even when NaN
    # diagnostics are disabled.  Passing NaNs to either FlashInfer or
    # ``argsort`` can otherwise make candidate membership nondeterministic.
    # This tensor is a temporary coarse projection and is not observed after
    # candidate selection.  Normalize NaNs in place to avoid another full
    # ``[batch, vocab]`` allocation on large vocabularies.
    coarse_logits.nan_to_num_(nan=-float("inf"))
    if coarse_logits.is_cuda and has_flashinfer():
        try:
            from flashinfer import top_k

            _, indices = top_k(
                coarse_logits,
                candidates,
                sorted=False,
                deterministic=True,
            )
            return indices.contiguous()
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning_once(
                "FlashInfer NVFP4 lm-head candidate selection failed (%s); "
                "using torch.topk.",
                exc,
            )
    # Stable descending sort gives the same lower-index tie break as the
    # regular greedy argmax when FlashInfer is unavailable.
    return torch.argsort(coarse_logits, dim=-1, descending=True, stable=True)[
        ..., :candidates
    ].contiguous()


def autotune_row_buckets(max_rows: int) -> tuple[int, ...]:
    """Return row counts that may trigger a FlashInfer FP4 tactic lookup."""
    max_rows = max_rows or _DEFAULT_AUTOTUNE_MAX_ROWS
    try:
        if not (
            current_platform.is_cuda()
            and current_platform.has_device_capability(100)
            and has_flashinfer()
        ):
            raise ImportError("FlashInfer FP4 backend is unavailable")
        from flashinfer.fused_moe.utils import get_hybrid_num_tokens_buckets

        flashinfer_buckets = tuple(
            sorted({int(rows) for rows in get_hybrid_num_tokens_buckets(max_rows)})
        )
        if flashinfer_buckets:
            return flashinfer_buckets
    except (ImportError, RuntimeError, TypeError, ValueError):
        pass

    buckets: list[int] = [
        rows for rows in (1, 2, 4, 8, 16, 32, 64, 128, 256) if rows <= max_rows
    ]
    rows = 512
    while rows <= max_rows:
        buckets.append(rows)
        rows += 256
    if not buckets:
        buckets.append(max_rows)
    return tuple(sorted(set(buckets)))


@dataclass
class HybridNvfp4LmHead:
    """Persistent NVFP4 weights for approximate BF16 candidate refinement."""

    weight: torch.Tensor
    scale: torch.Tensor
    global_scale: torch.Tensor
    input_size: int
    output_size: int
    candidates: int
    max_rows: int | None = None
    can_use_failure_counts: dict[str, int] = field(default_factory=dict, repr=False)
    # Tied input/output embeddings can attach the same derived state to more
    # than one module.  Keep weak references so releasing one module does not
    # invalidate the state still owned by its tied peers.
    attached_layers: WeakSet[torch.nn.Module] = field(
        default_factory=WeakSet, repr=False, compare=False
    )

    def candidate_count_for_topk(self, top_k: int) -> int:
        return _candidate_count_for_topk(
            self.candidates,
            top_k,
            output_size=self.output_size,
        )

    def can_use(
        self,
        hidden_states: torch.Tensor,
        *,
        bf16_weight: torch.Tensor,
        active_vocab_size: int,
        top_k: int,
    ) -> bool:
        reason: str | None = None
        if hidden_states.ndim != 2:
            reason = "hidden_ndim"
        elif hidden_states.dtype != torch.bfloat16:
            reason = "hidden_dtype"
        elif not hidden_states.is_cuda:
            reason = "hidden_not_cuda"
        elif not hidden_states.is_contiguous():
            reason = "hidden_not_contiguous"
        elif hidden_states.shape[1] != self.input_size:
            reason = "hidden_width"
        elif self.max_rows is not None and hidden_states.shape[0] > self.max_rows:
            reason = "rows_exceed_limit"
        elif bf16_weight.dtype != torch.bfloat16:
            reason = "weight_dtype"
        elif bf16_weight.device != hidden_states.device:
            reason = "weight_device"
        elif not bf16_weight.is_contiguous():
            reason = "weight_not_contiguous"
        elif bf16_weight.shape != (self.output_size, self.input_size):
            reason = "weight_shape"
        elif active_vocab_size > self.output_size:
            reason = "active_vocab_too_large"
        else:
            candidate_width = self.candidate_count_for_topk(top_k)
            if top_k > candidate_width:
                reason = "top_k_exceeds_candidates"
            elif active_vocab_size < candidate_width:
                reason = "active_vocab_too_small"
        if reason is None:
            return True

        count = self.can_use_failure_counts.get(reason, 0) + 1
        self.can_use_failure_counts[reason] = count
        if count == 1 or count & (count - 1) == 0:
            logger.debug(
                "Hybrid NVFP4 lm-head compact path unavailable: reason=%s, "
                "shape=%s, active_vocab=%d, top_k=%d.",
                reason,
                tuple(hidden_states.shape),
                active_vocab_size,
                top_k,
            )
        return False

    def coarse_logits(
        self,
        hidden_states: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden_global_scale = _global_scale(hidden_states)
        hidden_q, hidden_scale = flashinfer_nvfp4_quantize_128x4(
            hidden_states,
            hidden_global_scale,
        )
        alpha = torch.reciprocal(hidden_global_scale * self.global_scale)
        logits = flashinfer_scaled_fp4_mm(
            hidden_q,
            self.weight,
            hidden_scale,
            self.scale,
            alpha=alpha,
            out_dtype=torch.bfloat16,
            backend="b12x",
            block_size=_BLOCK_SIZE,
            use_nvfp4=True,
        )
        logits = logits[:, : self.output_size]
        if bias is not None:
            logits = logits + bias
        return logits

    def select_candidates(
        self,
        coarse_logits: torch.Tensor,
        *,
        top_k: int | None = None,
    ) -> torch.Tensor:
        candidates = (
            self.candidates if top_k is None else self.candidate_count_for_topk(top_k)
        )
        candidates = min(candidates, coarse_logits.shape[-1])
        return select_lm_head_candidates(coarse_logits, candidates)

    @staticmethod
    def refine_logits(
        hidden_states: torch.Tensor,
        bf16_weight: torch.Tensor,
        candidate_indices: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        logits = indexed_bf16_dot(
            hidden_states,
            bf16_weight,
            candidate_indices,
        )
        if bias is not None:
            logits = logits + bias[candidate_indices]
        return logits


@torch.inference_mode()
def warmup_hybrid_nvfp4_lm_head_kernels(
    state: HybridNvfp4LmHead,
    bf16_weight: torch.Tensor,
    tp_size: int = 1,
) -> None:
    """Compile the selector and selected-BF16 refinement kernels."""
    hidden_states = torch.zeros(
        (1, state.input_size),
        dtype=torch.bfloat16,
        device=state.weight.device,
    )
    coarse_logits = state.coarse_logits(hidden_states, None)
    candidate_sets = [state.select_candidates(coarse_logits)]
    # A bounded top-k request may widen the transient candidate matrix. Warm
    # that shape once so the first random top-k request does not pay a JIT
    # compile or an unexpected CUDA-graph allocation.
    wide_candidates = state.select_candidates(coarse_logits, top_k=64)
    if wide_candidates.shape[-1] != candidate_sets[0].shape[-1]:
        candidate_sets.append(wide_candidates)
    for candidate_indices in candidate_sets:
        exact_logits = state.refine_logits(
            hidden_states,
            bf16_weight,
            candidate_indices,
            None,
        )
        if HAS_TRITON and exact_logits.shape[-1] <= 1024:
            indexed_argmax_triton(exact_logits, candidate_indices)

    rows = 16
    tiled_hidden = hidden_states.expand(rows, -1).contiguous()
    for candidate_indices in candidate_sets:
        tiled_candidates = candidate_indices.expand(rows, -1).contiguous()
        state.refine_logits(tiled_hidden, bf16_weight, tiled_candidates, None)
    if HAS_TRITON and tp_size > 1:
        gathered_pairs = torch.zeros(
            (1, tp_size * 2),
            dtype=torch.float32,
            device=state.weight.device,
        )
        reduce_global_argmax_triton(gathered_pairs, tp_size=tp_size)


@torch.inference_mode()
def autotune_hybrid_nvfp4_lm_head(
    state: HybridNvfp4LmHead,
    bf16_weight: torch.Tensor,
    row_shapes: tuple[int, ...] | None = None,
) -> tuple[float, tuple[int, ...]]:
    """Tune FlashInfer NVFP4 tactics for row shapes used by the runner."""
    if row_shapes is None:
        row_shapes = autotune_row_buckets(_DEFAULT_AUTOTUNE_MAX_ROWS)
    row_shapes = tuple(sorted({int(rows) for rows in row_shapes if rows > 0}))
    if not row_shapes:
        row_shapes = (1,)

    hidden_states = torch.zeros(
        (max(row_shapes), state.input_size),
        dtype=torch.bfloat16,
        device=state.weight.device,
    )
    started = perf_counter()
    with autotune_with_torch_cuda_delay(tune_mode=True):
        for rows in row_shapes:
            hidden = hidden_states[:rows]
            coarse_logits = state.coarse_logits(hidden, None)
            candidate_indices = state.select_candidates(coarse_logits)
            state.refine_logits(hidden, bf16_weight, candidate_indices, None)
    torch.accelerator.synchronize()
    return perf_counter() - started, row_shapes


def _attach_state(layer: torch.nn.Module, state: HybridNvfp4LmHead) -> None:
    for name, value in (
        (_WEIGHT_NAME, state.weight),
        (_SCALE_NAME, state.scale),
        (_GLOBAL_SCALE_NAME, state.global_scale),
    ):
        layer.register_buffer(name, value, persistent=False)
    setattr(layer, _STATE_NAME, state)
    state.attached_layers.add(layer)


def _release_state_attachments(state: HybridNvfp4LmHead) -> int:
    """Detach every module sharing ``state`` and return freed bytes."""
    released_bytes = 0
    # ``release_hybrid_nvfp4_lm_head`` updates the weak set while iterating;
    # materialize it first to avoid mutating a live WeakSet iterator.
    attached_layers = list(state.attached_layers)
    for layer in attached_layers:
        released_bytes += release_hybrid_nvfp4_lm_head(layer)
    return released_bytes


def prepare_hybrid_nvfp4_lm_head(
    layer: torch.nn.Module,
    *,
    candidates: int,
    max_rows: int | None = None,
) -> bool:
    """Prepare one NVFP4 b12x lm-head copy, or keep the BF16 path."""
    normalized_max_rows = (
        None if max_rows is None or max_rows <= 0 else int(max_rows)
    )
    if isinstance(getattr(layer, _STATE_NAME, None), HybridNvfp4LmHead):
        state = getattr(layer, _STATE_NAME)
        if max_rows is not None:
            state.max_rows = normalized_max_rows
        if refresh_hybrid_nvfp4_lm_head(layer, candidates=candidates):
            return True
        # A changed dtype or shape cannot be refreshed in place.  Drop the
        # auxiliary state and let the normal support checks below decide
        # whether a new state can be built.
        release_hybrid_nvfp4_lm_head(layer)

    weight = layer.weight
    shared_state = getattr(weight, _SHARED_STATE_NAME, None)
    if isinstance(shared_state, HybridNvfp4LmHead):
        if max_rows is not None:
            shared_state.max_rows = normalized_max_rows
        _attach_state(layer, shared_state)
        return True

    supported = (
        has_flashinfer()
        and has_flashinfer_b12x_gemm()
        and current_platform.is_cuda()
        and current_platform.has_device_capability(120)
        and isinstance(weight, torch.Tensor)
        and weight.ndim == 2
        and weight.dtype == torch.bfloat16
        and weight.is_cuda
        and weight.is_contiguous()
        and not getattr(weight, "_vllm_is_uva_offloaded", False)
        and weight.shape[0] >= _MIN_GEMM_DIMENSION
        and weight.shape[1] >= _MIN_GEMM_DIMENSION
        and weight.shape[1] % 32 == 0
    )
    if not supported:
        logger.warning_once(
            "Hybrid NVFP4 b12x lm-head does not support weight %s (%s on %s); "
            "falling back to the original lm-head implementation.",
            tuple(weight.shape),
            weight.dtype,
            weight.device,
        )
        return False

    if candidates <= 0 or candidates > weight.shape[0]:
        logger.warning_once(
            "Hybrid NVFP4 lm-head candidate count %d is outside [1, %d]; "
            "falling back to the original lm-head implementation.",
            candidates,
            weight.shape[0],
        )
        return False

    quantized, scale, global_scale = _quantize_lm_head_weight(weight)
    state = HybridNvfp4LmHead(
        weight=quantized,
        scale=scale,
        global_scale=global_scale,
        input_size=weight.shape[1],
        output_size=weight.shape[0],
        candidates=candidates,
        max_rows=normalized_max_rows,
    )
    _attach_state(layer, state)
    setattr(weight, _SHARED_STATE_NAME, state)

    extra_mib = sum(getattr(layer, name).nbytes for name in _BUFFER_NAMES) / (
        1024 * 1024
    )
    logger.info_once(
        "Prepared hybrid NVFP4 b12x lm-head for weight %s with %d candidates "
        "and %.2f MiB auxiliary storage.",
        tuple(weight.shape),
        candidates,
        extra_mib,
    )
    return True


@torch.inference_mode()
def refresh_hybrid_nvfp4_lm_head(
    layer: torch.nn.Module,
    *,
    candidates: int | None = None,
) -> bool:
    """Refresh a prepared auxiliary copy after the BF16 weight is updated.

    The auxiliary tensors are derived state and are intentionally not part of
    the checkpoint.  In-place updates preserve references held by tied heads
    and CUDA-graph setup code.
    """
    state = getattr(layer, _STATE_NAME, None)
    weight = getattr(layer, "weight", None)
    if not isinstance(state, HybridNvfp4LmHead) or not isinstance(weight, torch.Tensor):
        return False
    if (
        weight.dtype != torch.bfloat16
        or not weight.is_cuda
        or not weight.is_contiguous()
        or weight.ndim != 2
        or weight.shape[0] != state.output_size
        or weight.shape[1] != state.input_size
    ):
        if state.attached_layers:
            _release_state_attachments(state)
        else:
            release_hybrid_nvfp4_lm_head(layer)
        return False

    quantized, scale, global_scale = _quantize_lm_head_weight(weight)
    if (
        state.weight.shape != quantized.shape
        or state.scale.shape != scale.shape
        or state.global_scale.shape != global_scale.shape
    ):
        if state.attached_layers:
            _release_state_attachments(state)
        else:
            release_hybrid_nvfp4_lm_head(layer)
        return False

    state.weight.copy_(quantized)
    state.scale.copy_(scale)
    state.global_scale.copy_(global_scale)
    if candidates is not None:
        state.candidates = candidates
    state.can_use_failure_counts.clear()
    return True


@torch.inference_mode()
def refresh_hybrid_nvfp4_lm_heads(
    model: torch.nn.Module,
    *,
    candidates: int | None = None,
) -> int:
    """Refresh each distinct prepared lm-head state in ``model``."""
    refreshed = 0
    state_layers: dict[int, tuple[HybridNvfp4LmHead, list[torch.nn.Module]]] = {}
    for layer in model.modules():
        state = getattr(layer, _STATE_NAME, None)
        if not isinstance(state, HybridNvfp4LmHead):
            continue
        state_layers.setdefault(id(state), (state, []))[1].append(layer)
    for state, layers in state_layers.values():
        if refresh_hybrid_nvfp4_lm_head(layers[0], candidates=candidates):
            refreshed += 1
        else:
            # The first refresh may already detach all known owners.  Keep the
            # explicit loop for modules outside the current model traversal or
            # stale weak-set entries.
            _release_state_attachments(state)
            for layer in layers:
                release_hybrid_nvfp4_lm_head(layer)
    return refreshed


def release_hybrid_nvfp4_lm_heads(model: torch.nn.Module) -> int:
    """Release all distinct auxiliary states owned by ``model``."""
    states: dict[int, tuple[HybridNvfp4LmHead, list[torch.nn.Module]]] = {}
    for layer in model.modules():
        state = getattr(layer, _STATE_NAME, None)
        if isinstance(state, HybridNvfp4LmHead):
            states.setdefault(id(state), (state, []))[1].append(layer)
    released = 0
    for state, layers in states.values():
        if state.attached_layers:
            released += _release_state_attachments(state)
            continue
        # Legacy/custom attachments may not have populated the weak set.  In
        # that case count the shared storage once, then detach every owner.
        if layers:
            released += release_hybrid_nvfp4_lm_head(layers[0])
            for layer in layers[1:]:
                if hasattr(layer, _STATE_NAME):
                    delattr(layer, _STATE_NAME)
                for name in _BUFFER_NAMES:
                    if hasattr(layer, name):
                        delattr(layer, name)
    return released


def get_hybrid_nvfp4_lm_head(
    layer: torch.nn.Module,
) -> HybridNvfp4LmHead | None:
    return getattr(layer, _STATE_NAME, None)


def release_hybrid_nvfp4_lm_head(layer: torch.nn.Module) -> int:
    """Release a prepared auxiliary copy from a discarded lm-head."""
    state = getattr(layer, _STATE_NAME, None)
    if not isinstance(state, HybridNvfp4LmHead):
        # Be tolerant of callers cleaning up a partially initialized module.
        if hasattr(layer, _STATE_NAME):
            delattr(layer, _STATE_NAME)
        released_bytes = 0
        for name in _BUFFER_NAMES:
            value = getattr(layer, name, None)
            if isinstance(value, torch.Tensor):
                released_bytes += value.nbytes
            if hasattr(layer, name):
                delattr(layer, name)
        return released_bytes

    state.attached_layers.discard(layer)
    # Removing one tied view does not release the underlying tensors.  They
    # remain reachable through the shared state and the other attached layer.
    if state.attached_layers:
        released_bytes = 0
    else:
        released_bytes = sum(
            value.nbytes
            for value in (state.weight, state.scale, state.global_scale)
            if isinstance(value, torch.Tensor)
        )

    for name in _BUFFER_NAMES:
        if hasattr(layer, name):
            delattr(layer, name)

    if hasattr(layer, _STATE_NAME):
        delattr(layer, _STATE_NAME)

    weight = getattr(layer, "weight", None)
    if not state.attached_layers and getattr(weight, _SHARED_STATE_NAME, None) is state:
        delattr(weight, _SHARED_STATE_NAME)
    return released_bytes


__all__ = [
    "HybridNvfp4LmHead",
    "autotune_hybrid_nvfp4_lm_head",
    "autotune_row_buckets",
    "get_hybrid_nvfp4_lm_head",
    "indexed_bf16_dot",
    "prepare_hybrid_nvfp4_lm_head",
    "refresh_hybrid_nvfp4_lm_head",
    "refresh_hybrid_nvfp4_lm_heads",
    "release_hybrid_nvfp4_lm_head",
    "release_hybrid_nvfp4_lm_heads",
    "select_lm_head_candidates",
    "warmup_hybrid_nvfp4_lm_head_kernels",
]
