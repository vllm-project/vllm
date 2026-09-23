# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable Triton kernels for the DCP sparse-indexer top-k merge.

Platform-agnostic pieces of the candidate-exchange merge (local top-k ->
pack (score, global_id) candidates -> all-gather -> stable top-k). The
CuteDSL stable-topk selector remains in ``dcp_indexer_cutedsl``; this
module holds the parts shared by every platform.
"""

from dataclasses import dataclass
from typing import Any

import torch

from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    LaunchSpec,
    TritonWarmupTensor,
    VllmTritonJitKernel,
    kernel_launcher,
)
from vllm.triton_utils import tl, triton


def pack_dcp_topk_candidates(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    packed: torch.Tensor,
    dcp_rank: int,
    dcp_world_size: int,
    cp_interleave: int,
    row_starts: torch.Tensor | None,
) -> None:
    topk = topk_indices.shape[1]
    row_starts_arg = row_starts if row_starts is not None else topk_indices
    _PACK_DCP_TOPK_CANDIDATES_KERNEL(
        logits,
        topk_indices,
        packed,
        row_starts_arg,
        logits_stride0=logits.stride(0),
        logits_stride1=logits.stride(1),
        topk_stride0=topk_indices.stride(0),
        topk_stride1=topk_indices.stride(1),
        packed_stride0=packed.stride(0),
        packed_stride1=packed.stride(1),
        packed_stride2=packed.stride(2),
        num_cols=logits.shape[1],
        dcp_rank=dcp_rank,
        dcp_world_size=dcp_world_size,
        cp_interleave=cp_interleave,
        has_row_starts=row_starts is not None,
        topk=topk,
        block_size=512,
    )


class PackDCPTopkCandidatesKernel(
    VllmTritonJitKernel["PackDCPTopkCandidatesKernel.CompileKey"]
):
    @dataclass(frozen=True)
    class CompileKey:
        dcp_rank: int
        dcp_world_size: int
        cp_interleave: int
        has_row_starts: bool
        topk: int
        block_size: int

    # These scalars only describe runtime layouts and bounds; the constexpr
    # fields below own the launch geometry and algorithmic specialization.
    @staticmethod
    @triton.jit(
        do_not_specialize=[
            "logits_stride0",
            "logits_stride1",
            "topk_stride0",
            "topk_stride1",
            "packed_stride0",
            "packed_stride1",
            "packed_stride2",
            "num_cols",
        ]
    )
    def kernel(
        logits,
        topk_indices,
        packed,
        row_starts,
        logits_stride0,
        logits_stride1,
        topk_stride0,
        topk_stride1,
        packed_stride0,
        packed_stride1,
        packed_stride2,
        num_cols,
        DCP_RANK: tl.constexpr,
        DCP_WORLD_SIZE: tl.constexpr,
        CP_INTERLEAVE: tl.constexpr,
        HAS_ROW_STARTS: tl.constexpr,
        TOPK: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        tile = tl.program_id(1)
        cols = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = cols < TOPK

        local_idx = tl.load(
            topk_indices + row * topk_stride0 + cols * topk_stride1,
            mask=mask,
            other=-1,
        )
        valid = local_idx >= 0
        safe_local_idx = tl.maximum(local_idx, 0)

        row_start = 0
        if HAS_ROW_STARTS:
            row_start = tl.load(row_starts + row)

        score_col = safe_local_idx + row_start
        score_col = tl.minimum(score_col, tl.maximum(num_cols - 1, 0))
        score = tl.load(
            logits + row * logits_stride0 + score_col * logits_stride1,
            mask=mask & valid,
            other=-float("inf"),
        )

        global_id = (
            (safe_local_idx // CP_INTERLEAVE) * (DCP_WORLD_SIZE * CP_INTERLEAVE)
            + DCP_RANK * CP_INTERLEAVE
            + safe_local_idx % CP_INTERLEAVE
        )
        global_id = tl.where(valid, global_id, -1)

        packed_base = packed + row * packed_stride0 + cols * packed_stride1
        tl.store(packed_base, score, mask=mask)
        tl.store(packed_base + packed_stride2, global_id.to(tl.float32), mask=mask)

    def dispatch(  # type: ignore[override]
        self,
        *,
        has_row_starts: bool,
        **compile_key_fields: int,
    ) -> CompileKey:
        return self.CompileKey(
            **compile_key_fields,
            has_row_starts=has_row_starts,
        )

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        if dcp_world_size <= 1:
            return []
        cp_interleave = vllm_config.parallel_config.cp_kv_cache_interleave_size
        topk = vllm_config.model_config.hf_config.index_topk
        if topk <= 0:
            return []

        try:
            from vllm.distributed.parallel_state import get_dcp_group

            dcp_rank = get_dcp_group().rank_in_group
        except Exception:
            dcp_rank = 0

        return self._trace_dispatch(self.dispatch)(
            dcp_rank=dcp_rank,
            dcp_world_size=dcp_world_size,
            cp_interleave=cp_interleave,
            has_row_starts=(False, True),
            topk=topk,
            block_size=512,
        )

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        fp32_ptr = TritonWarmupTensor(torch.float32)
        int32_ptr = TritonWarmupTensor(torch.int32)
        return dict(
            logits=fp32_ptr,
            topk_indices=TritonWarmupTensor(torch.int32, shape=(1, compile_key.topk)),
            packed=fp32_ptr,
            row_starts_arg=int32_ptr,
            logits_stride0=1,
            logits_stride1=1,
            topk_stride0=1,
            topk_stride1=1,
            packed_stride0=1,
            packed_stride1=1,
            packed_stride2=1,
            num_cols=1,
            dcp_rank=compile_key.dcp_rank,
            dcp_world_size=compile_key.dcp_world_size,
            cp_interleave=compile_key.cp_interleave,
            has_row_starts=compile_key.has_row_starts,
            topk=compile_key.topk,
            block_size=compile_key.block_size,
        )

    @kernel_launcher
    def __call__(
        self,
        logits: torch.Tensor,
        topk_indices: torch.Tensor,
        packed: torch.Tensor,
        row_starts_arg: torch.Tensor,
        *,
        logits_stride0: int,
        logits_stride1: int,
        topk_stride0: int,
        topk_stride1: int,
        packed_stride0: int,
        packed_stride1: int,
        packed_stride2: int,
        num_cols: int,
        dcp_rank: int,
        dcp_world_size: int,
        cp_interleave: int,
        has_row_starts: bool,
        topk: int,
        block_size: int,
    ) -> LaunchSpec:
        grid = (topk_indices.shape[0], triton.cdiv(topk, block_size))
        return grid, dict(
            row_starts=row_starts_arg,
            DCP_RANK=dcp_rank,
            DCP_WORLD_SIZE=dcp_world_size,
            CP_INTERLEAVE=cp_interleave,
            HAS_ROW_STARTS=has_row_starts,
            TOPK=topk,
            BLOCK_SIZE=block_size,
            num_warps=8,
        )


_PACK_DCP_TOPK_CANDIDATES_KERNEL = PackDCPTopkCandidatesKernel()
