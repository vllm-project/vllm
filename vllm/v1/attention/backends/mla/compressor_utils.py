# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Any

import torch

from vllm.distributed import get_dcp_group
from vllm.model_executor.warmup.jit_warmup import kernel_launcher, zip_inputs
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    LaunchSpec,
    TritonWarmupTensor,
    VllmTritonJitKernel,
    triton_scalar_specialization_rep,
)
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.worker.cp_utils import cp_global_to_local_block

_DSPARK_SWA_INDEX_ALIGNMENT = 64


def get_dspark_swa_index_width(
    window_size: int,
    num_speculative_tokens: int,
) -> int:
    """Return the padded width of non-causal DSpark SWA indices."""
    width = max(int(window_size), 0) + max(int(num_speculative_tokens), 0)
    return cdiv(width, _DSPARK_SWA_INDEX_ALIGNMENT) * _DSPARK_SWA_INDEX_ALIGNMENT


class CompressedSlotMappingKernel(
    VllmTritonJitKernel["CompressedSlotMappingKernel.CompileKey"]
):
    TRITON_BLOCK_SIZE = 1024

    @dataclass(frozen=True)
    class CompileKey:
        compress_ratio: int
        triton_block_size: int
        block_size: int
        dcp_world_size: int = 1
        dcp_rank: int = 0
        cp_kv_cache_interleave_size: int = 1

    @staticmethod
    @triton.jit(do_not_specialize=["block_table_stride"])
    def kernel(
        # [num_tokens]
        compressed_slot_mapping_ptr,
        # [num_tokens] the tokens' own slots
        slot_mapping_ptr,
        # [num_reqs + 1]
        query_start_loc_ptr,
        # [num_reqs]
        seq_lens_ptr,
        # [num_reqs, max_num_blocks]
        block_table_ptr,
        block_table_stride,
        block_size,
        COMPRESS_RATIO: tl.constexpr,
        PAD_ID: tl.constexpr,
        TRITON_BLOCK_SIZE: tl.constexpr,
        DCP_WORLD_SIZE: tl.constexpr,
        DCP_RANK: tl.constexpr,
        CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)

        query_start = tl.load(query_start_loc_ptr + batch_idx)
        query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
        query_len = query_end - query_start

        seq_len = tl.load(seq_lens_ptr + batch_idx)
        start_pos = seq_len - query_len

        for i in range(0, query_len, TRITON_BLOCK_SIZE):
            offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
            mask = offset < query_len

            pos = start_pos + i + tl.arange(0, TRITON_BLOCK_SIZE)
            if DCP_WORLD_SIZE == 1:
                # A replayed token has a PAD slot (SWA bounded replay: its KV
                # is cached already); its compressed KV is cached too, so PAD
                # here.
                slot = tl.load(
                    slot_mapping_ptr + query_start + offset, mask=mask, other=PAD_ID
                )
                is_valid = ((pos + 1) % COMPRESS_RATIO == 0) & (slot != PAD_ID)
            else:
                # Under DCP the raw slot mapping is owner-masked by RAW token
                # position, while a state's closing token can belong to any
                # rank (ratio even => closers all sit on odd positions), so
                # PAD there cannot distinguish "replayed" from "another
                # rank's token". Bounded replay never happens on models that
                # reach this branch (V4.1-only feature, gated off with DCP —
                # see the SWA builder's assert), so the closing test alone
                # decides validity.
                is_valid = (pos + 1) % COMPRESS_RATIO == 0
            pos_after_compress = pos // COMPRESS_RATIO

            block_ids, local_block_offsets, is_local = cp_global_to_local_block(
                pos_after_compress,
                block_size,
                DCP_WORLD_SIZE,
                DCP_RANK,
                CP_KV_CACHE_INTERLEAVE_SIZE,
            )
            is_valid = is_valid & is_local
            block_numbers = tl.load(
                block_table_ptr + batch_idx * block_table_stride + block_ids,
                mask=mask & is_valid,
            )
            slot_ids = block_numbers * block_size + local_block_offsets

            # NOTE
            slot_ids = tl.where(is_valid, slot_ids, PAD_ID)
            tl.store(
                compressed_slot_mapping_ptr + query_start + offset, slot_ids, mask=mask
            )

    def dispatch(  # type: ignore[override]
        self,
        *,
        compress_ratio: int,
        block_size: int,
        dcp_world_size: int = 1,
        dcp_rank: int = 0,
        cp_kv_cache_interleave_size: int = 1,
    ) -> CompileKey:
        return self.CompileKey(
            compress_ratio=compress_ratio,
            triton_block_size=self.TRITON_BLOCK_SIZE,
            block_size=triton_scalar_specialization_rep(block_size),
            dcp_world_size=dcp_world_size,
            dcp_rank=dcp_rank,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        )

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        hf_text_config = vllm_config.model_config.hf_text_config
        configured_ratios = (
            *(getattr(hf_text_config, "compress_ratios", None) or ()),
            getattr(hf_text_config, "index_kpool", 1) or 1,
        )
        compress_ratios = tuple(
            dict.fromkeys(int(ratio) for ratio in configured_ratios if int(ratio) > 1)
        )
        if not compress_ratios:
            return []
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        try:
            dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP group not initialized (e.g. single-process warmup tracing).
            dcp_rank = 0
        cp_variants = [(1, 0, 1)]
        if dcp_world_size > 1:
            cp_variants.append(
                (
                    dcp_world_size,
                    dcp_rank,
                    vllm_config.parallel_config.cp_kv_cache_interleave_size,
                )
            )
        return self._trace_dispatch(self.dispatch)(
            zip_inputs(
                *(
                    dict(
                        compress_ratio=ratio,
                        block_size=vllm_config.cache_config.block_size // ratio,
                        dcp_world_size=world,
                        dcp_rank=rank,
                        cp_kv_cache_interleave_size=interleave,
                    )
                    for ratio in compress_ratios
                    for world, rank, interleave in cp_variants
                )
            )
        )

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        int32_ptr = TritonWarmupTensor(torch.int32)
        return dict(
            compressed_slot_mapping=TritonWarmupTensor(torch.int64),
            slot_mapping=TritonWarmupTensor(torch.int64),
            query_start_loc=int32_ptr,
            seq_lens=int32_ptr,
            block_table=int32_ptr,
            block_size=compile_key.block_size,
            compress_ratio=compile_key.compress_ratio,
            dcp_world_size=compile_key.dcp_world_size,
            dcp_rank=compile_key.dcp_rank,
            cp_kv_cache_interleave_size=compile_key.cp_kv_cache_interleave_size,
        )

    @kernel_launcher
    def __call__(
        self,
        compressed_slot_mapping: torch.Tensor,
        slot_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        compress_ratio: int,
        dcp_world_size: int = 1,
        dcp_rank: int = 0,
        cp_kv_cache_interleave_size: int = 1,
    ) -> LaunchSpec:
        return (block_table.shape[0],), dict(
            block_table_stride=block_table.stride(0),
            COMPRESS_RATIO=compress_ratio,
            PAD_ID=-1,
            TRITON_BLOCK_SIZE=self.TRITON_BLOCK_SIZE,
            DCP_WORLD_SIZE=dcp_world_size,
            DCP_RANK=dcp_rank,
            CP_KV_CACHE_INTERLEAVE_SIZE=cp_kv_cache_interleave_size,
        )


def get_compressed_slot_mapping(
    num_tokens: int,
    slot_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
    dcp_world_size: int = 1,
    dcp_rank: int = 0,
    cp_kv_cache_interleave_size: int = 1,
) -> torch.Tensor:
    """Slot mapping for writing the compressed states of ``num_tokens`` tokens.

    Every ``compress_ratio`` tokens share one compressed state, written by the
    last of them: that token maps to the state's slot, the others to PAD. A
    token whose own ``slot_mapping`` entry is PAD maps to PAD too: SWA bounded
    replay recomputes tokens whose KV is cached already, and their compressed
    states must not be rewritten either.
    """
    if out is not None:
        # Guard: for padded / invalid sequences.
        # Negative positions produce bogus block indices that lead to illegal memory
        # accesses inside the block_table load.
        # NOTE: Fill -1 to the whole tensor, not just the first `num_tokens`.
        out.fill_(-1)
        compressed_slot_mapping = out[:num_tokens]
    else:
        compressed_slot_mapping = torch.full(
            (num_tokens,), -1, dtype=torch.int64, device=query_start_loc.device
        )

    _COMPRESSED_SLOT_MAPPING_KERNEL(
        compressed_slot_mapping,
        slot_mapping,
        query_start_loc,
        seq_lens,
        block_table,
        block_size,
        compress_ratio,
        dcp_world_size=dcp_world_size,
        dcp_rank=dcp_rank,
        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
    )
    return compressed_slot_mapping


_COMPRESSED_SLOT_MAPPING_KERNEL = CompressedSlotMappingKernel()
