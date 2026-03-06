from collections.abc import Callable
from functools import partial

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Boolean, Int8, Int32, const_expr

from vllm.vllm_flash_attn.cute.block_sparsity import (
    BlockSparseTensors,
    BlockSparseTensorsTorch,
    to_cute_block_sparse_tensors,
)
from vllm.vllm_flash_attn.cute.seqlen_info import SeqlenInfoQK
from vllm.vllm_flash_attn.cute.utils import hash_callable, scalar_to_ssa, ssa_to_scalar


class BlockSparsityKernel:
    """Block sparsity kernel for FlexAttention.

    This kernel computes `mask_mod` for every token of each block
    to determine if an n block is full, masked, or neither.

    Writes block counts and indices to a BlockSparseTensors object.

    When use_fast_sampling=True, uses 5-point sampling (4 corners + center)
    which is much faster but only suitable for masks where this is sufficient.

    TODO:
        - optimize mask_mod evaluation
        - varlen support
        - transposed tensors for bwd pass
    """

    def __init__(
        self,
        mask_mod: Callable,
        tile_mn: tuple[int, int],
        compute_full_blocks: bool = True,
        use_aux_tensors: bool = False,
        use_fast_sampling: bool = False,
    ):
        self.mask_mod = mask_mod
        self.tile_mn = tile_mn
        self.compute_full_blocks = compute_full_blocks
        self.use_aux_tensors = use_aux_tensors
        self.use_fast_sampling = use_fast_sampling

    @cute.jit
    def __call__(
        self,
        blocksparse_tensors: BlockSparseTensors,
        seqlen_q: Int32,
        seqlen_k: Int32,
        aux_tensors: list | None = None,
    ):
        self.mask_cnt, self.mask_idx, self.full_cnt, self.full_idx = blocksparse_tensors

        if const_expr(self.compute_full_blocks):
            assert self.full_cnt is not None and self.full_idx is not None, (
                "full block tensors must be provided when computing full blocks"
            )

        batch_size, num_heads, num_m_blocks, num_n_blocks = self.mask_idx.shape
        # launch 1 CTA per m block
        grid = [num_m_blocks, num_heads, batch_size]

        if const_expr(self.use_fast_sampling):
            num_threads = 5
            self.num_warps = 1
        else:
            num_threads = self.tile_mn[0]
            self.num_warps = (num_threads + 32 - 1) // 32

        self.kernel(
            self.mask_cnt,
            self.mask_idx,
            self.full_cnt,
            self.full_idx,
            num_n_blocks,
            seqlen_q,
            seqlen_k,
            aux_tensors,
        ).launch(grid=grid, block=[num_threads, 1, 1])

    @cute.kernel
    def kernel(
        self,
        mask_cnt: cute.Tensor,
        mask_idx: cute.Tensor,
        full_cnt: cute.Tensor,
        full_idx: cute.Tensor,
        num_n_blocks: Int32,
        seqlen_q: Int32,
        seqlen_k: Int32,
        aux_tensors: list | None = None,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        lane_id = cute.arch.lane_idx()
        m_block, head_idx, batch_idx = cute.arch.block_idx()

        ssa = partial(scalar_to_ssa, dtype=Int32)

        seqlen = SeqlenInfoQK.create(
            batch_idx,
            seqlen_q,
            seqlen_k,
            mCuSeqlensQ=None,
            mCuSeqlensK=None,
            mSeqUsedQ=None,
            mSeqUsedK=None,
        )

        @cute.struct
        class SharedStorage:
            reduction_buffer_smem: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int8, 2 * self.num_warps], 1024
            ]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage, 16)

        reduction_buffer = storage.reduction_buffer_smem.get_tensor(
            cute.make_layout((self.num_warps, 2))
        )

        num_mask_blocks = Int32(0)
        num_full_blocks = Int32(0)

        for n_block in cutlass.range(num_n_blocks, unroll_full=True):
            m_base = m_block * self.tile_mn[0]
            n_base = n_block * self.tile_mn[1]

            if const_expr(self.use_fast_sampling):
                # Fast path: 5-point sampling (4 corners + center)
                # Clamps OOB indices to nearest in bounds.
                thread_result = Boolean(False)
                thread_is_valid = Boolean(False)
                q_idx = Int32(0)
                kv_idx = Int32(0)

                if tidx == 0:
                    # Top-left corner (0, 0); always in bounds
                    q_idx = m_base
                    kv_idx = n_base
                elif tidx == 1:
                    # Top-right corner
                    q_idx = m_base
                    kv_idx = cutlass.min(n_base + self.tile_mn[1] - 1, seqlen_k - 1)
                elif tidx == 2:
                    # Bottom-left corner
                    q_idx = cutlass.min(m_base + self.tile_mn[0] - 1, seqlen_q - 1)
                    kv_idx = n_base
                elif tidx == 3:
                    # Bottom-right corner
                    q_idx = cutlass.min(m_base + self.tile_mn[0] - 1, seqlen_q - 1)
                    kv_idx = cutlass.min(n_base + self.tile_mn[1] - 1, seqlen_k - 1)
                elif tidx == 4:
                    # Center point
                    q_idx = (
                        m_base + (cutlass.min(seqlen_q - m_base, self.tile_mn[0])) // 2
                    )
                    kv_idx = (
                        n_base + (cutlass.min(seqlen_k - n_base, self.tile_mn[1])) // 2
                    )
                else:
                    thread_is_valid = Boolean(False)

                # Check bounds and determine if this thread has a valid index pair
                if tidx < 5 and q_idx < seqlen_q and kv_idx < seqlen_k:
                    thread_is_valid = Boolean(True)
                    q_idx_ssa = ssa(q_idx)
                    kv_idx_ssa = ssa(kv_idx)
                    thread_result = ssa_to_scalar(
                        self.mask_mod(
                            ssa(batch_idx),
                            ssa(head_idx),
                            q_idx_ssa,
                            kv_idx_ssa,
                            seqlen,
                            aux_tensors,
                        )
                    )
                else:
                    thread_is_valid = Boolean(False)

                # Use vote_any_sync to see if any valid thread found unmasked or masked
                # Only count results from threads that checked valid indices
                has_unmasked = cute.arch.vote_any_sync(thread_result & thread_is_valid)
                has_masked = cute.arch.vote_any_sync(
                    (Boolean(not thread_result)) & thread_is_valid
                )

            else:
                # Full path: check all elements in the block
                # Track if this thread's row has any masked or unmasked elements
                thread_has_unmasked = Boolean(False)
                thread_has_masked = Boolean(False)
                thread_is_valid = Boolean(False)

                # Each thread handles 1 row
                q_idx = m_base + tidx
                kv_idx = Int32(0)
                if tidx < self.tile_mn[0] and q_idx < seqlen_q:
                    thread_is_valid = Boolean(True)
                    q_idx_ssa = ssa(q_idx)

                    # Loop over all columns in this row
                    for c in cutlass.range(self.tile_mn[1], unroll_full=True):
                        kv_idx = n_base + c
                        kv_idx_ssa = ssa(kv_idx)

                        # Only check elements within valid sequence bounds
                        if kv_idx < seqlen_k:
                            # Direct scalar call
                            mask_val = ssa_to_scalar(
                                self.mask_mod(
                                    ssa(batch_idx),
                                    ssa(head_idx),
                                    q_idx_ssa,
                                    kv_idx_ssa,
                                    seqlen,
                                    aux_tensors,
                                )
                            )

                            # Update tracking flags
                            if mask_val:
                                thread_has_unmasked = Boolean(True)
                            else:
                                thread_has_masked = Boolean(True)

                # Block-level reduction to combine results across all threads
                # Only count votes from threads that checked valid indices
                warp_has_unmasked_mask = cute.arch.vote_any_sync(
                    thread_has_unmasked & thread_is_valid
                )
                warp_has_masked_mask = cute.arch.vote_any_sync(
                    thread_has_masked & thread_is_valid
                )

                # lane 0 writes the ballot mask to shared memory
                lane_id = tidx % 32
                if lane_id == 0:
                    # Store as Int8
                    reduction_buffer[warp_idx, 0] = (
                        Int8(1) if warp_has_unmasked_mask else Int8(0)
                    )
                    reduction_buffer[warp_idx, 1] = (
                        Int8(1) if warp_has_masked_mask else Int8(0)
                    )

                cute.arch.sync_threads()

                # Thread 0 ORs all warp results together
                has_unmasked = Boolean(False)
                has_masked = Boolean(False)
                if tidx == 0:
                    for w in cutlass.range(self.num_warps):
                        if reduction_buffer[w, 0]:
                            has_unmasked = Boolean(True)
                        if reduction_buffer[w, 1]:
                            has_masked = Boolean(True)

            # Only thread 0 updates the output arrays (common to both paths)
            if tidx == 0:
                # Block classification based on what we found:
                # - If has_masked and has_unmasked: partial block (needs masking)
                # - If only has_unmasked: full block (no masking needed)
                # - If only has_masked: skip this block entirely
                is_partial = Boolean(has_masked and has_unmasked)
                is_full = Boolean(has_unmasked and (not has_masked))

                if is_partial:
                    mask_idx[batch_idx, head_idx, m_block, num_mask_blocks] = n_block
                    num_mask_blocks += 1
                elif is_full and const_expr(self.compute_full_blocks):
                    full_idx[batch_idx, head_idx, m_block, num_full_blocks] = n_block
                    num_full_blocks += 1

        # Only thread 0 writes back the counts
        if tidx == 0:
            mask_cnt[batch_idx, head_idx, m_block] = num_mask_blocks
            if const_expr(self.compute_full_blocks):
                full_cnt[batch_idx, head_idx, m_block] = num_full_blocks


def compute_block_sparsity(
    tile_m,
    tile_n,
    batch_size,
    num_heads,
    seqlen_q,
    seqlen_k,
    mask_mod: Callable,
    aux_tensors: list | None,  # list[cute.Tensor]
    device,
    compute_full_blocks: bool = True,
    use_fast_sampling: bool = False,
) -> tuple[BlockSparseTensors, BlockSparseTensorsTorch]:
    """
    Computes block sparsity for a given `mask_mod`.

    Args:
        tile_m: The tile size for the m dimension.
        tile_n: The tile size for the n dimension.
        batch_size: The batch size.
        num_heads: The number of heads.
        seqlen_q: The sequence length for the query.
        seqlen_k: The sequence length for the key.
        mask_mod: The `mask_mod` callable to use.
        aux_tensors: A list of auxiliary tensors.
        device: The device to use.
        compute_full_blocks: Whether to compute full blocks. If False, only partially-masked blocks are computed.
        use_fast_sampling: Whether to use 5-point sampling (4 corners + center). This is much faster, but only suitable for masks where this check is sufficient.

    Returns:
        A tuple of `BlockSparseTensors` and `BlockSparseTensorsTorch`.
    """
    # Check if mask_mod is marked as suitable for 5-point fast sampling
    use_fast_sampling = getattr(mask_mod, "use_fast_sampling", use_fast_sampling)

    num_m_blocks = (seqlen_q + tile_m - 1) // tile_m
    num_n_blocks = (seqlen_k + tile_n - 1) // tile_n

    mask_block_cnt = torch.zeros(
        (batch_size, num_heads, num_m_blocks), device=device, dtype=torch.int32
    )
    mask_block_idx = torch.zeros(
        (batch_size, num_heads, num_m_blocks, num_n_blocks),
        device=device,
        dtype=torch.int32,
    )
    full_block_cnt = (
        torch.zeros(
            (batch_size, num_heads, num_m_blocks), device=device, dtype=torch.int32
        )
        if compute_full_blocks
        else None
    )
    full_block_idx = (
        torch.zeros(
            (batch_size, num_heads, num_m_blocks, num_n_blocks),
            device=device,
            dtype=torch.int32,
        )
        if compute_full_blocks
        else None
    )

    blocksparse_tensors_torch = BlockSparseTensorsTorch(
        mask_block_cnt=mask_block_cnt,
        mask_block_idx=mask_block_idx,
        full_block_cnt=full_block_cnt,
        full_block_idx=full_block_idx,
        block_size=(tile_m, tile_n),
    )

    mask_mod_hash = hash_callable(mask_mod)
    blocksparse_tensors = to_cute_block_sparse_tensors(
        blocksparse_tensors_torch, enable_tvm_ffi=True
    )

    compile_key = (
        tile_m,
        tile_n,
        mask_mod_hash,
        compute_full_blocks,
        aux_tensors is not None,
        use_fast_sampling,
    )
    if compile_key not in compute_block_sparsity.compile_cache:
        kernel = BlockSparsityKernel(
            mask_mod,
            tile_mn=(tile_m, tile_n),
            compute_full_blocks=compute_full_blocks,
            use_aux_tensors=aux_tensors is not None,
            use_fast_sampling=use_fast_sampling,
        )

        compute_block_sparsity.compile_cache[compile_key] = cute.compile(
            kernel,
            blocksparse_tensors,
            seqlen_q,
            seqlen_k,
            aux_tensors,
            options="--enable-tvm-ffi",
        )

    compute_block_sparsity.compile_cache[compile_key](
        blocksparse_tensors_torch[:4],
        seqlen_q,
        seqlen_k,
        aux_tensors,
    )

    return blocksparse_tensors, blocksparse_tensors_torch


compute_block_sparsity.compile_cache = {}
