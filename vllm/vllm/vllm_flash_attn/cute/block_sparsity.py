"""
Block-sparsity utilities for FlexAttention
"""

from collections.abc import Callable
from typing import NamedTuple

import cutlass.cute as cute
import torch

from vllm.vllm_flash_attn.cute.cute_dsl_utils import get_broadcast_dims, to_cute_tensor


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


class BlockSparseTensors(NamedTuple):
    mask_block_cnt: cute.Tensor
    mask_block_idx: cute.Tensor
    full_block_cnt: cute.Tensor | None
    full_block_idx: cute.Tensor | None

    def __new_from_mlir_values__(self, values):
        if len(values) == 2:
            values = (*values, None, None)
        return BlockSparseTensors(*values)


class BlockSparseTensorsTorch(NamedTuple):
    mask_block_cnt: torch.Tensor
    mask_block_idx: torch.Tensor
    full_block_cnt: torch.Tensor | None = None
    full_block_idx: torch.Tensor | None = None
    block_size: tuple[int, int] | None = None


def _expand_sparsity_tensor(
    tensor: torch.Tensor,
    expected_shape: tuple[int, ...],
    tensor_name: str,
    context: str | None,
    hint: str | Callable[[], str] | None,
) -> torch.Tensor:
    """Check if we need to expand the tensor to expected shape, and do so if possible."""
    needs_expand = tensor.shape != expected_shape
    if not needs_expand:
        return tensor
    can_expand = all(
        map(lambda cur, tgt: cur == tgt or cur == 1, tensor.shape, expected_shape)
    )
    if not can_expand:
        context_clause = f" ({context})" if context else ""
        resolved_hint = hint() if callable(hint) else hint
        hint_clause = f" Hint: {resolved_hint}" if resolved_hint else ""
        raise ValueError(
            f"{tensor_name}{context_clause} with shape {tensor.shape} cannot be expanded to expected shape {expected_shape}."
            f"{hint_clause}"
        )
    return tensor.expand(*expected_shape)


def _check_and_expand_block(
    name: str,
    cnt: torch.Tensor | None,
    idx: torch.Tensor | None,
    expected_count_shape: tuple[int, int, int],
    expected_index_shape: tuple[int, int, int, int],
    context: str | None,
    hint: str | Callable[[], str] | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if (cnt is None) != (idx is None):
        raise ValueError(
            f"{name}_block_cnt and {name}_block_idx must both be provided or both be None"
        )
    if cnt is None or idx is None:
        return None, None
    if cnt.dtype != torch.int32 or idx.dtype != torch.int32:
        raise ValueError(f"{name}_block tensors must have dtype torch.int32")
    if cnt.device != idx.device:
        raise ValueError(
            f"{name}_block_cnt and {name}_block_idx must be on the same device"
        )
    if not cnt.is_cuda or not idx.is_cuda:
        raise ValueError(f"{name}_block tensors must live on CUDA")
    expanded_cnt = _expand_sparsity_tensor(
        cnt, expected_count_shape, f"{name}_block_cnt", context, hint
    )
    expanded_idx = _expand_sparsity_tensor(
        idx, expected_index_shape, f"{name}_block_idx", context, hint
    )
    return expanded_cnt, expanded_idx


def get_block_sparse_expected_shapes(
    batch_size: int,
    num_head: int,
    seqlen_q: int,
    seqlen_k: int,
    m_block_size: int,
    n_block_size: int,
    q_stage: int,
) -> tuple[tuple[int, int, int], tuple[int, int, int, int]]:
    """Return (expected_count_shape, expected_index_shape) for block sparse normalization."""
    m_block_size_effective = q_stage * m_block_size
    expected_m_blocks = ceildiv(seqlen_q, m_block_size_effective)
    expected_n_blocks = ceildiv(seqlen_k, n_block_size)
    expected_count_shape = (batch_size, num_head, expected_m_blocks)
    expected_index_shape = (batch_size, num_head, expected_m_blocks, expected_n_blocks)
    return expected_count_shape, expected_index_shape


def infer_block_sparse_expected_shapes(
    tensors: BlockSparseTensorsTorch,
    *,
    batch_size: int,
    num_head: int,
    seqlen_q: int,
    seqlen_k: int,
    m_block_size: int,
    n_block_size: int,
    q_stage: int,
    context: str,
    sparse_block_size_q: int | None = None,
    sparse_block_size_kv: int | None = None,
) -> tuple[tuple[int, int, int], tuple[int, int, int, int], int]:
    """Infer shapes and scaling for block-sparse tensors.

    Expectations:
    - mask_block_cnt is (B, H, M) and mask_block_idx is (B, H, M, N).
    - Batch/head dims may be 1 for broadcast, or match the requested sizes.
    - sparse_block_size_kv must match tile_n.
    - sparse_block_size_q must be a multiple of q_stage * tile_m.
    - If sparse_block_size_q is omitted and seqlen_q/num_m_blocks is ambiguous,
      the caller must provide block_size to disambiguate. TODO will make this required in a future PR.
    """
    base_m_block = q_stage * m_block_size
    base_n_block = n_block_size
    if sparse_block_size_kv is None:
        sparse_block_size_kv = base_n_block
    if sparse_block_size_kv != base_n_block:
        raise ValueError(
            f"Block sparse tensors{context} require BLOCK_SIZE_KV={base_n_block}."
        )
    if tensors.mask_block_idx is None:
        raise ValueError(
            "mask_block_cnt and mask_block_idx must be provided for block sparsity."
        )
    num_m_blocks = tensors.mask_block_idx.shape[2]

    if sparse_block_size_q is None:
        min_block_size = ceildiv(seqlen_q, num_m_blocks)
        if num_m_blocks == 1:
            max_block_size = seqlen_q
        else:
            max_block_size = (seqlen_q - 1) // (num_m_blocks - 1)
        if max_block_size != min_block_size and base_m_block != 1:
            raise ValueError(
                f"Block sparse tensors{context} require explicit sparse_block_size[0] "
                f"to disambiguate block size for seqlen_q={seqlen_q} and num_m_blocks={num_m_blocks}."
            )
        sparse_block_size_q = min_block_size

    if sparse_block_size_q % base_m_block != 0:
        raise ValueError(
            f"Block sparse tensors{context} have block size {sparse_block_size_q}, "
            f"which must be a multiple of {base_m_block}."
        )

    expected_m_blocks = ceildiv(seqlen_q, sparse_block_size_q)
    expected_n_blocks = ceildiv(seqlen_k, sparse_block_size_kv)
    q_subtile_factor = sparse_block_size_q // base_m_block
    expected_count_shape = (batch_size, num_head, expected_m_blocks)
    expected_index_shape = (batch_size, num_head, expected_m_blocks, expected_n_blocks)

    mask_block_cnt = tensors.mask_block_cnt
    mask_block_idx = tensors.mask_block_idx
    if mask_block_cnt is None or mask_block_idx is None:
        raise ValueError(
            "mask_block_cnt and mask_block_idx must be provided for block sparsity."
        )
    if mask_block_cnt.ndim != 3 or mask_block_idx.ndim != 4:
        raise ValueError(
            f"Block sparse tensors{context} must have shapes (B, H, M) and (B, H, M, N)."
        )
    for dim_name, cur, tgt in (
        ("batch", mask_block_cnt.shape[0], expected_count_shape[0]),
        ("head", mask_block_cnt.shape[1], expected_count_shape[1]),
    ):
        if cur != tgt and cur != 1:
            raise ValueError(
                f"Block sparse tensors{context} {dim_name} dim must be {tgt} or 1."
            )
    for dim_name, cur, tgt in (
        ("batch", mask_block_idx.shape[0], expected_index_shape[0]),
        ("head", mask_block_idx.shape[1], expected_index_shape[1]),
    ):
        if cur != tgt and cur != 1:
            raise ValueError(
                f"Block sparse tensors{context} {dim_name} dim must be {tgt} or 1."
            )
    if mask_block_cnt.shape[2] != mask_block_idx.shape[2]:
        raise ValueError(
            f"Block sparse tensors{context} must share the same m-block dimension."
        )
    if mask_block_idx.shape[3] != expected_n_blocks:
        raise ValueError(
            f"Block sparse tensors{context} n-block dimension must be {expected_n_blocks}."
        )
    if expected_m_blocks != num_m_blocks:
        raise ValueError(
            f"Block sparse tensors{context} m-block dimension {num_m_blocks} does not match "
            f"sparse_block_size_q={sparse_block_size_q}. "
            f"Set BlockSparseTensorsTorch.block_size to match the BlockMask BLOCK_SIZE."
        )
    return expected_count_shape, expected_index_shape, q_subtile_factor


def get_block_sparse_expected_shapes_bwd(
    batch_size: int,
    num_head: int,
    seqlen_q: int,
    seqlen_k: int,
    m_block_size: int,
    n_block_size: int,
    subtile_factor: int,
) -> tuple[tuple[int, int, int], tuple[int, int, int, int]]:
    """Return (expected_count_shape, expected_index_shape) for backward block sparse normalization.

    Backward uses Q-direction indexing (transposed from forward), where shapes are
    indexed by N-blocks first, then M-blocks. The sparse_block_size_q is determined
    by subtile_factor * m_block_size.
    """
    sparse_block_size_q = subtile_factor * m_block_size
    expected_m_blocks = ceildiv(seqlen_q, sparse_block_size_q)
    expected_n_blocks = ceildiv(seqlen_k, n_block_size)
    expected_count_shape = (batch_size, num_head, expected_n_blocks)
    expected_index_shape = (batch_size, num_head, expected_n_blocks, expected_m_blocks)
    return expected_count_shape, expected_index_shape


def normalize_block_sparse_tensors(
    tensors: BlockSparseTensorsTorch,
    *,
    expected_count_shape: tuple[int, int, int],
    expected_index_shape: tuple[int, int, int, int],
    context: str | None = None,
    hint: str | Callable[[], str] | None = None,
) -> BlockSparseTensorsTorch:
    if tensors.mask_block_cnt is None or tensors.mask_block_idx is None:
        raise ValueError(
            "mask_block_cnt and mask_block_idx must be provided for block sparsity."
        )

    mask_cnt, mask_idx = _check_and_expand_block(
        "mask",
        tensors.mask_block_cnt,
        tensors.mask_block_idx,
        expected_count_shape,
        expected_index_shape,
        context,
        hint,
    )
    if mask_cnt is None or mask_idx is None:
        raise ValueError(
            "mask_block_cnt and mask_block_idx must be provided for block sparsity."
        )

    full_cnt, full_idx = _check_and_expand_block(
        "full",
        tensors.full_block_cnt,
        tensors.full_block_idx,
        expected_count_shape,
        expected_index_shape,
        context,
        hint,
    )
    if full_cnt is not None and mask_cnt.device != full_cnt.device:
        raise ValueError("All block sparse tensors must be on the same device")

    return BlockSparseTensorsTorch(
        mask_block_cnt=mask_cnt,
        mask_block_idx=mask_idx,
        full_block_cnt=full_cnt,
        full_block_idx=full_idx,
        block_size=tensors.block_size,
    )


def is_block_sparsity_enabled(tensors: BlockSparseTensorsTorch) -> bool:
    return any(t is not None for t in (tensors.full_block_cnt, tensors.mask_block_cnt))


def get_block_sparse_broadcast_pattern(
    tensors: BlockSparseTensorsTorch,
) -> tuple[tuple[bool, ...], ...] | None:
    """Return broadcast pattern for block sparse tensors by checking actual strides.

    Returns a tuple of broadcast patterns (one per tensor) where each pattern
    is a tuple of bools indicating which dims have stride=0.
    This is used in compile keys to ensure kernels are recompiled when
    broadcast patterns change, since CuTe's mark_layout_dynamic() keeps
    stride=0 as static.

    The tensors should already be expanded/normalized before calling this function.

    Returns None if block sparsity is not enabled.
    """
    if not is_block_sparsity_enabled(tensors):
        return None

    patterns = []
    for tensor in (
        tensors.mask_block_cnt,
        tensors.mask_block_idx,
        tensors.full_block_cnt,
        tensors.full_block_idx,
    ):
        if tensor is not None:
            patterns.append(get_broadcast_dims(tensor))
        else:
            patterns.append(None)
    return tuple(patterns)


def normalize_block_sparse_config(
    tensors: BlockSparseTensorsTorch,
    *,
    batch_size: int,
    num_head: int,
    seqlen_q: int,
    seqlen_k: int,
    block_size: tuple[int, int],
    q_stage: int,
) -> tuple[BlockSparseTensorsTorch, tuple[tuple[bool, ...], ...] | None, int]:
    m_block_size, n_block_size = block_size
    if tensors.block_size is None:
        sparse_block_size_q, sparse_block_size_kv = q_stage * m_block_size, n_block_size
    else:
        sparse_block_size_q, sparse_block_size_kv = tensors.block_size
    if sparse_block_size_kv != n_block_size:
        raise ValueError(
            f"Block sparsity requires sparse_block_size[1]={n_block_size} to match tile_n."
        )
    expected_count_shape, expected_index_shape, q_subtile_factor = (
        infer_block_sparse_expected_shapes(
            tensors,
            batch_size=batch_size,
            num_head=num_head,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            q_stage=q_stage,
            context="forward",
            sparse_block_size_q=sparse_block_size_q,
            sparse_block_size_kv=sparse_block_size_kv,
        )
    )
    normalized_tensors = normalize_block_sparse_tensors(
        tensors,
        expected_count_shape=expected_count_shape,
        expected_index_shape=expected_index_shape,
    )
    return (
        normalized_tensors,
        get_block_sparse_broadcast_pattern(normalized_tensors),
        q_subtile_factor,
    )


def normalize_block_sparse_config_bwd(
    tensors: BlockSparseTensorsTorch,
    *,
    batch_size: int,
    num_head: int,
    seqlen_q: int,
    seqlen_k: int,
    block_size: tuple[int, int],
    subtile_factor: int,
) -> tuple[BlockSparseTensorsTorch, tuple[tuple[bool, ...], ...] | None]:
    m_block_size, n_block_size = block_size
    if tensors.block_size is None:
        sparse_block_size_q, sparse_block_size_kv = (
            subtile_factor * m_block_size,
            n_block_size,
        )
    else:
        sparse_block_size_q, sparse_block_size_kv = tensors.block_size
    if sparse_block_size_q != subtile_factor * m_block_size:
        raise ValueError(
            f"Block sparsity expects sparse_block_size_q={subtile_factor * m_block_size} "
            f"for subtile_factor={subtile_factor}."
        )
    if sparse_block_size_kv != n_block_size:
        raise ValueError(
            f"Block sparsity expects sparse_block_size[1]={n_block_size} to match tile_n."
        )
    expected_count_shape, expected_index_shape = get_block_sparse_expected_shapes_bwd(
        batch_size,
        num_head,
        seqlen_q,
        seqlen_k,
        m_block_size,
        n_block_size,
        subtile_factor,
    )
    normalized_tensors = normalize_block_sparse_tensors(
        tensors,
        expected_count_shape=expected_count_shape,
        expected_index_shape=expected_index_shape,
        context="_flash_attn_bwd",
        hint=lambda: (
            f"Backward expects Q-direction block-sparse tensors (q_mask_cnt/q_mask_idx, "
            f"and optionally full_q_cnt/full_q_idx). Regenerate the backward BlockMask with "
            f"BLOCK_SIZE=({subtile_factor * m_block_size}, {n_block_size})."
        ),
    )
    return normalized_tensors, get_block_sparse_broadcast_pattern(normalized_tensors)


def to_cute_block_sparse_tensors(
    tensors: BlockSparseTensorsTorch, enable_tvm_ffi: bool = True
) -> BlockSparseTensors | None:
    """Convert torch block sparsity tensors to CuTe tensors, optionally for tvm ffi"""
    if not is_block_sparsity_enabled(tensors):
        return None
    (
        mask_block_cnt,
        mask_block_idx,
        full_block_cnt,
        full_block_idx,
        *_,
    ) = tensors

    (
        mask_block_cnt_tensor,
        mask_block_idx_tensor,
    ) = [
        to_cute_tensor(
            t, assumed_align=4, leading_dim=-1, enable_tvm_ffi=enable_tvm_ffi
        )
        for t in (mask_block_cnt, mask_block_idx)
    ]
    (
        full_block_cnt_tensor,
        full_block_idx_tensor,
    ) = [
        to_cute_tensor(
            t, assumed_align=4, leading_dim=-1, enable_tvm_ffi=enable_tvm_ffi
        )
        if t is not None
        else None
        for t in (full_block_cnt, full_block_idx)
    ]

    return BlockSparseTensors(
        mask_block_cnt_tensor,
        mask_block_idx_tensor,
        full_block_cnt_tensor,
        full_block_idx_tensor,
    )


def fast_sampling(mask_mod):
    """Convenience decorator to mark mask_mod as safe for 5-point fast sampling"""
    mask_mod.use_fast_sampling = True
    return mask_mod
