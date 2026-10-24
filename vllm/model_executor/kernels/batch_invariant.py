# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native batch-invariant arithmetic shared by training and inference."""

import functools
import importlib.util
import os
from pathlib import Path
from typing import Any

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


def _mutation_fake(*args, **kwargs) -> None:
    return None


def _batch_invariant_kernel_path() -> Path:
    override = os.environ.get("VLLM_BATCH_INVARIANT_KERNEL_LIB")
    if override:
        return Path(override).expanduser()
    spec = importlib.util.find_spec("vllm._vllm_batch_invariant_C")
    if spec is None or spec.origin is None:
        raise RuntimeError("The CUDA batch-invariant extension is not installed")
    return Path(spec.origin)


@functools.cache
def _load_batch_invariant_kernel_library(path: str) -> None:
    library = Path(path).expanduser().resolve()
    if not library.is_file():
        raise RuntimeError(f"batch-invariant kernel library does not exist: {library}")
    try:
        torch.ops.load_library(str(library))
        for name in (
            "top_k_per_row_prefill",
            "top_k_per_row_prefill_index_order",
            "silu_mul_bf16",
            "hc_collapse",
            "sparse_metadata",
            "combine_swa_first_indices",
            "topk_lens_by_position",
            "gather_replay_batch",
            "cast_indexer_logits",
            "dequant_gather_nvfp4",
            "sparse_paged_indices",
            "block32_scaled_mm",
            "block32_scaled_mm_variant",
            "gather_expert_slots",
            "ordered_topk_combine",
            "expert_permute",
            "fused_silu_mul_per_token_group_quant",
        ):
            getattr(torch.ops.vllm_batch_invariant, name)
            torch.library.register_fake(f"vllm_batch_invariant::{name}")(_mutation_fake)
    except (OSError, RuntimeError, AttributeError) as exc:
        raise RuntimeError(
            f"failed to load the batch-invariant kernel library: {library}"
        ) from exc


def batch_invariant_quant_kernel_available() -> bool:
    path = _batch_invariant_kernel_path()
    if not path.is_file():
        return False
    _load_batch_invariant_kernel_library(str(path))
    return True


def is_batch_invariant_quant_kernel_enabled() -> bool:
    if not envs.VLLM_BATCH_INVARIANT:
        return False
    if not batch_invariant_quant_kernel_available():
        return False
    logger.info_once(
        "Using the batch-invariant fused SiLU/quant kernel from %s",
        _batch_invariant_kernel_path(),
    )
    return True


@torch.compiler.assume_constant_result
def require_batch_invariant_kernel() -> None:
    """Load the unified batch-invariant extension or fail closed."""
    path = _batch_invariant_kernel_path()
    _load_batch_invariant_kernel_library(str(path))


def require_batch_invariant_quant_kernel() -> None:
    """Load the batch-invariant extension before expert execution."""
    require_batch_invariant_kernel()


def top_k_per_row_prefill(
    logits: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    raw_topk_indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    topk_tokens: int,
) -> None:
    """Select by descending score/local index with one fixed per-row algorithm."""
    require_batch_invariant_kernel()
    torch.ops.vllm_batch_invariant.top_k_per_row_prefill(
        logits,
        cu_seqlen_ks,
        cu_seqlen_ke,
        raw_topk_indices,
        num_rows,
        stride0,
        stride1,
        topk_tokens,
    )


def top_k_per_row_prefill_index_order(
    logits: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    raw_topk_indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    topk_tokens: int,
    *,
    zero_starts: bool = False,
) -> None:
    """Select like top_k_per_row_prefill, then write the selected local indices
    in ascending order with -1 padding last.

    zero_starts promises every row starts at column zero; then
    VLLM_BI_INDEXER_DEEPSELECT selects with DeepSelect instead."""
    from vllm import envs

    if zero_starts and envs.VLLM_BI_INDEXER_DEEPSELECT:
        import vllm._deepselect_C  # noqa: F401

        torch.ops.deep_select.topk(
            logits[:num_rows],
            topk_tokens,
            None,  # begin: rows start at column zero
            cu_seqlen_ke[:num_rows].contiguous(),
            False,  # sorted_value
            True,  # sorted_index: ascending, -1 padding last
            None,
            raw_topk_indices[:num_rows],
            None,
            -1,
            float("-inf"),
            False,  # return_value
            False,  # abort_when_nan_found
        )
        return
    require_batch_invariant_kernel()
    torch.ops.vllm_batch_invariant.top_k_per_row_prefill_index_order(
        logits,
        cu_seqlen_ks,
        cu_seqlen_ke,
        raw_topk_indices,
        num_rows,
        stride0,
        stride1,
        topk_tokens,
    )


def cast_indexer_logits(
    logits: torch.Tensor, row_ends: torch.Tensor, output: torch.Tensor
) -> None:
    """Convert only valid BF16 DeepGEMM scores; retain its aligned strides."""
    require_batch_invariant_kernel()
    torch.ops.vllm_batch_invariant.cast_indexer_logits(
        logits, row_ends.contiguous(), output
    )


def dequant_gather_nvfp4(
    cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
    output: torch.Tensor,
) -> None:
    """Gather packed E2M1/E4M3 records into the shared BF16 workspace."""
    require_batch_invariant_kernel()
    torch.ops.vllm_batch_invariant.dequant_gather_nvfp4(
        cache, seq_lens, gather_lens, block_table, block_size, offset, output
    )


def sparse_paged_indices(
    topk: torch.Tensor | None,
    token_to_req: torch.Tensor,
    is_valid: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    replay_start: torch.Tensor,
    compressed_table: torch.Tensor | None,
    compressed_block_size: int,
    swa_table: torch.Tensor,
    swa_block_size: int,
    token_offset: int,
    num_tokens: int,
    window: int,
    compress_ratio: int,
    top_k: int,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    extra_indices: torch.Tensor | None,
    extra_lens: torch.Tensor | None,
) -> None:
    """Write each token's paged SWA and compressed top-k slots for FlashMLA's
    sparse decode kernel, ``-1`` for holes, with lengths from its position.

    Tokens ``[token_offset, token_offset + num_tokens)`` land in output rows
    ``[0, num_tokens)``. The visible window and the compressed length follow
    the eager prefill path, so a (request, position) gets the same lists in
    any batch, chunk or step.
    """
    require_batch_invariant_kernel()
    torch.ops.vllm_batch_invariant.sparse_paged_indices(
        topk,
        token_to_req,
        is_valid,
        query_start_loc,
        seq_lens,
        replay_start,
        compressed_table,
        compressed_block_size,
        swa_table,
        swa_block_size,
        token_offset,
        num_tokens,
        window,
        compress_ratio,
        top_k,
        swa_indices,
        swa_lens,
        extra_indices,
        extra_lens,
    )


def hc_collapse(hidden: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """Collapse BF16 streams with ordered, separate FP32 multiply and add."""
    require_batch_invariant_kernel()
    output = torch.empty(
        (hidden.shape[0], hidden.shape[2]), device=hidden.device, dtype=hidden.dtype
    )
    torch.ops.vllm_batch_invariant.hc_collapse(hidden, pre, output)
    return output


def compact_sparse_slots(
    mask: torch.Tensor,
    main: torch.Tensor | None,
    *,
    main_size: int,
    window_size: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compact main slots in input order, then visible window slots in order."""
    require_batch_invariant_kernel()
    indices = torch.empty(
        (*mask.shape[:2], width), device=mask.device, dtype=torch.int32
    )
    lengths = torch.empty(mask.shape[:2], device=mask.device, dtype=torch.int32)
    torch.ops.vllm_batch_invariant.sparse_metadata(
        mask, main, main_size, window_size, indices, lengths
    )
    return indices, lengths


def combine_swa_first_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``combine_topk_swa_indices`` laid out in the mega decode kernel's slot
    space, ``[swa padded to the window; topk]``, with length
    ``window + topk_len``, so prefill blocks a row's KV as decode does."""
    require_batch_invariant_kernel()
    if out is None:
        num_tokens = topk_indices.shape[0]
        width = (topk + window_size + 127) // 128 * 128
        out = (
            torch.empty(
                (num_tokens, width), dtype=torch.int32, device=topk_indices.device
            ),
            torch.empty(num_tokens, dtype=torch.int32, device=topk_indices.device),
        )
    torch.ops.vllm_batch_invariant.combine_swa_first_indices(
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        window_size,
        compress_ratio,
        topk,
        M,
        N,
        *out,
    )
    return out


def topk_lens_by_position(
    token_to_req: torch.Tensor,
    is_valid: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    compress_ratio: int,
    top_k: int,
    lengths: torch.Tensor,
) -> None:
    """Write each token's compressed top-k length from its position,
    ``max(min((pos + 1) // compress_ratio, top_k), 0)``, as
    ``combine_topk_swa_indices`` gives prefill rows; invalid tokens get 0."""
    require_batch_invariant_kernel()
    torch.ops.vllm_batch_invariant.topk_lens_by_position(
        token_to_req,
        is_valid,
        query_start_loc,
        seq_lens,
        compress_ratio,
        top_k,
        lengths,
    )


def gather_replay_batch(
    query_start_loc: torch.Tensor,
    dropped_before: torch.Tensor,
    seq_lens: torch.Tensor,
    replay_start: torch.Tensor,
    positions: torch.Tensor,
    slot_mappings: torch.Tensor,
    num_reqs: int,
    kept_rows: int,
    rows: torch.Tensor,
    kept_query_start_loc: torch.Tensor,
    kept_positions: torch.Tensor,
    kept_slot_mappings: torch.Tensor,
    kept_kv_start: torch.Tensor,
    prev_dropped_before: torch.Tensor | None = None,
    select: torch.Tensor | None = None,
) -> None:
    """Gather a replay stage's trailing rows of each request: their batch rows,
    query offsets, positions, slot mappings and window KV start (no lower
    than ``seq_len - kept_rows`` for a trimmed request); with
    ``prev_dropped_before`` also each row's index among the previous stage's
    rows."""
    require_batch_invariant_kernel()
    torch.ops.vllm_batch_invariant.gather_replay_batch(
        query_start_loc,
        dropped_before,
        seq_lens,
        replay_start,
        positions,
        slot_mappings,
        num_reqs,
        kept_rows,
        prev_dropped_before,
        rows,
        kept_query_start_loc,
        kept_positions,
        kept_slot_mappings,
        kept_kv_start,
        select,
    )


def silu_mul_bf16(
    input: torch.Tensor, clamp_limit: float | None = None
) -> torch.Tensor:
    """Use the fused quantizer's SiLU expression and BF16 activation boundary."""
    require_batch_invariant_kernel()
    output = torch.empty(
        (input.shape[0], input.shape[1] // 2), device=input.device, dtype=input.dtype
    )
    torch.ops.vllm_batch_invariant.silu_mul_bf16(
        input.contiguous(), output, clamp_limit
    )
    return output


def fused_silu_mul_per_token_group_quant_fp8(
    input: torch.Tensor,
    *,
    use_ue8m0: bool,
    round_scale: bool | None = None,
    clamp_limit: float | None = None,
    masked_m: torch.Tensor | None,
    output_q: torch.Tensor | None = None,
    group_size: int = 128,
    fuse_silu_and_mul: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the batch-invariant fused SiLU*up and per-token FP8 quant kernel."""
    require_batch_invariant_kernel()
    if round_scale is None:
        round_scale = use_ue8m0
    if use_ue8m0 and not round_scale:
        raise ValueError("packed UE8M0 scales require round_scale=True")

    input_factor = 2 if fuse_silu_and_mul else 1
    if input.ndim not in (2, 3) or input.shape[-1] % (input_factor * group_size):
        raise ValueError(
            f"invalid batch-invariant activation shape: {tuple(input.shape)}"
        )
    if not input.is_contiguous() or input.dtype != torch.bfloat16:
        raise ValueError("batch-invariant activation input must be contiguous BF16")

    hidden = input.shape[-1] // input_factor
    groups = hidden // group_size
    if output_q is None:
        output_q = torch.empty(
            (*input.shape[:-1], hidden),
            device=input.device,
            dtype=torch.float8_e4m3fn,
        )

    if masked_m is None:
        if input.ndim != 2:
            raise ValueError("contiguous batch-invariant input must be 2D")
        tokens = input.shape[0]
        packed_groups = (groups + 3) // 4 if use_ue8m0 else groups
        output_s = torch.empty_strided(
            (tokens, packed_groups),
            (1, tokens),
            device=input.device,
            dtype=torch.int32 if use_ue8m0 else torch.float32,
        )
    else:
        if input.ndim != 3:
            raise ValueError("masked batch-invariant input must be 3D")
        experts, tokens = input.shape[:2]
        if masked_m.shape != (experts,) or masked_m.dtype != torch.int32:
            raise ValueError("masked_m must be int32 with one count per expert")
        packed_groups = (groups + 3) // 4 if use_ue8m0 else groups
        output_s = torch.empty_strided(
            (experts, tokens, packed_groups),
            (tokens * packed_groups, 1, tokens),
            device=input.device,
            dtype=torch.int32 if use_ue8m0 else torch.float32,
        )
    output_s.zero_()
    if input.numel() == 0:
        return output_q, output_s

    torch.ops.vllm_batch_invariant.fused_silu_mul_per_token_group_quant(
        input,
        output_q,
        output_s,
        group_size,
        1e-10,
        -448.0,
        448.0,
        0.0 if clamp_limit is None else float(clamp_limit),
        round_scale,
        use_ue8m0,
        fuse_silu_and_mul,
        masked_m,
    )
    return output_q, output_s


def block32_scaled_mm(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
    *,
    grouped: bool = False,
    weight_scale_group_rows: int = 1,
) -> torch.Tensor:
    """Accumulate E4M3 products in ascending K32 blocks with E8M0 scales."""
    require_batch_invariant_kernel()
    if a_scale.dtype == torch.float8_e8m0fnu:
        a_scale = a_scale.view(torch.uint8)
    if weight_scale.dtype == torch.float8_e8m0fnu:
        weight_scale = weight_scale.view(torch.uint8)
    shape = (
        (a.shape[0], a.shape[1], weight.shape[1])
        if grouped
        else (a.shape[0], weight.shape[0])
    )
    output = torch.empty(shape, dtype=output_dtype, device=a.device)
    torch.ops.vllm_batch_invariant.block32_scaled_mm(
        a,
        a_scale,
        weight,
        weight_scale,
        output,
        grouped,
        weight_scale_group_rows,
    )
    return output


def gather_expert_slots(
    values: torch.Tensor,
    routes: torch.Tensor,
    inverse: torch.Tensor,
    expert_map: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Expand owned BF16 expert rows to single-owner FP32 or BF16 top-k slots."""
    require_batch_invariant_kernel()
    slots = torch.empty(
        (*routes.shape, values.shape[1]), device=values.device, dtype=dtype
    )
    torch.ops.vllm_batch_invariant.gather_expert_slots(
        values, routes, inverse, expert_map, slots
    )
    return slots


def stable_expert_permute(
    a: torch.Tensor,
    scale: torch.Tensor,
    routes: torch.Tensor,
    local_num_experts: int,
    capacity: int,
    alignment: int,
    *,
    expert_map: torch.Tensor | None,
    output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group FP8 rows in token/slot order without host readback or atomics."""
    require_batch_invariant_kernel()
    if output is None:
        output = torch.empty((capacity, a.shape[1]), dtype=a.dtype, device=a.device)
    output.view(torch.uint8).zero_()
    output_scale = torch.ones(
        (capacity, scale.shape[1]), dtype=torch.float32, device=a.device
    )
    layout = torch.full((capacity,), -1, dtype=torch.int32, device=a.device)
    inverse = torch.full(routes.shape, -1, dtype=torch.int32, device=a.device)
    counts = torch.zeros(local_num_experts, dtype=torch.int32, device=a.device)
    starts = torch.zeros_like(counts)
    torch.ops.vllm_batch_invariant.expert_permute(
        a,
        scale,
        routes,
        expert_map,
        output,
        output_scale,
        layout,
        inverse,
        counts,
        starts,
        alignment,
    )
    return output, output_scale, layout, inverse


def ordered_topk_combine(
    slots: torch.Tensor,
    weights: torch.Tensor,
    output_dtype: torch.dtype,
    *,
    routes: torch.Tensor | None = None,
    global_num_experts: int = 0,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Round each ascending-slot FP32 FMA once, then convert the final row."""
    require_batch_invariant_kernel()
    if output is None:
        output = torch.empty(
            (slots.shape[0], slots.shape[2]), dtype=output_dtype, device=slots.device
        )
    elif output.dtype != output_dtype:
        raise ValueError("Ordered combine output dtype differs from its storage")
    torch.ops.vllm_batch_invariant.ordered_topk_combine(
        slots, weights, routes, global_num_experts, output
    )
    return output


def rms_norm_cuda(
    values: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Use the native CUDA row reduction shared with deployment forwards."""
    from vllm import _custom_ops as ops

    if weight is not None:
        if values.dtype == torch.float32:
            weight = weight.float()
        if weight.dtype != values.dtype or weight.shape != (values.shape[-1],):
            raise ValueError("CUDA RMS weight must match the input dtype and row width")
        weight = weight.contiguous()
    if residual is not None:
        if residual.shape != values.shape or residual.dtype != values.dtype:
            raise ValueError("CUDA RMS residual must match the input shape and dtype")
        if values.numel():
            ops.fused_add_rms_norm(values, residual, weight, eps)
        return values, residual
    rows = values.reshape(-1, values.shape[-1])
    output = torch.empty_like(rows)
    if rows.numel():
        ops.rms_norm(output, rows, weight, eps)
    return output.reshape_as(values)


def _native_log_softmax(input: torch.Tensor, dim: int) -> torch.Tensor:
    output = torch.empty_like(input, memory_format=torch.contiguous_format)
    return torch.ops.aten._log_softmax.out(input, dim, False, out=output)


def _native_log_softmax_fake(input: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.empty_like(input, memory_format=torch.contiguous_format)


direct_register_custom_op(
    op_name="native_log_softmax",
    op_func=_native_log_softmax,
    fake_impl=_native_log_softmax_fake,
)


def _native_log_softmax_backward(
    grad: torch.Tensor, output: torch.Tensor, dim: int, input_dtype: torch.dtype
) -> torch.Tensor:
    result = torch.empty_like(
        output, dtype=input_dtype, memory_format=torch.contiguous_format
    )
    return torch.ops.aten._log_softmax_backward_data.out(
        grad, output, dim, input_dtype, out=result
    )


def _native_log_softmax_backward_fake(
    grad: torch.Tensor, output: torch.Tensor, dim: int, input_dtype: torch.dtype
) -> torch.Tensor:
    return torch.empty_like(
        output, dtype=input_dtype, memory_format=torch.contiguous_format
    )


direct_register_custom_op(
    op_name="native_log_softmax_backward",
    op_func=_native_log_softmax_backward,
    fake_impl=_native_log_softmax_backward_fake,
)


def _native_log_softmax_setup_context(ctx: Any, inputs: tuple, output: torch.Tensor):
    ctx.save_for_backward(output)
    ctx.dim = inputs[1]
    ctx.input_dtype = inputs[0].dtype


def _native_log_softmax_vjp(ctx: Any, grad: torch.Tensor):
    (output,) = ctx.saved_tensors
    return torch.ops.vllm.native_log_softmax_backward(
        grad, output, ctx.dim, ctx.input_dtype
    ), None


torch.library.register_autograd(
    "vllm::native_log_softmax",
    _native_log_softmax_vjp,
    setup_context=_native_log_softmax_setup_context,
)


def native_log_softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Keep native CUDA log-softmax opaque to compiler decompositions."""
    return torch.ops.vllm.native_log_softmax(input, dim)
