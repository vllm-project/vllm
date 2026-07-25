# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any

import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from quack.compile_utils import make_fake_tensor

from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.model_executor.warmup.jit_warmup_cutedsl_helper import compile_cutedsl
from vllm.model_executor.warmup.jit_warmup_triton_helper import TritonWarmupTensor
from vllm.triton_utils import tl, triton


def stable_topk_from_gathered_candidates_cutedsl(
    gathered: torch.Tensor,
    topk: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty(
            (gathered.shape[0], topk),
            dtype=torch.int32,
            device=gathered.device,
        )
    _STABLE_TOPK_FROM_GATHERED_CANDIDATES_KERNEL(gathered, out, topk=topk)
    return out


def pack_dcp_topk_candidates_cutedsl(
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
    VllmJitKernel["PackDCPTopkCandidatesKernel.CompileKey"]
):
    @dataclass(frozen=True)
    class CompileKey:
        DCP_RANK: int
        DCP_WORLD_SIZE: int
        CP_INTERLEAVE: int
        HAS_ROW_STARTS: bool
        TOPK: int
        BLOCK_SIZE: int

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
        dcp_rank: int,
        dcp_world_size: int,
        cp_interleave: int,
        has_row_starts: bool,
        topk: int,
        block_size: int,
    ) -> CompileKey:
        return self.CompileKey(
            DCP_RANK=dcp_rank,
            DCP_WORLD_SIZE=dcp_world_size,
            CP_INTERLEAVE=cp_interleave,
            HAS_ROW_STARTS=has_row_starts,
            TOPK=topk,
            BLOCK_SIZE=block_size,
        )

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        parallel_config = getattr(vllm_config, "parallel_config", None)
        dcp_world_size = int(
            getattr(parallel_config, "decode_context_parallel_size", 1) or 1
        )
        if dcp_world_size <= 1:
            return []
        cp_interleave = int(
            getattr(parallel_config, "cp_kv_cache_interleave_size", 1) or 1
        )
        model_config = getattr(vllm_config, "model_config", None)
        hf_config = getattr(model_config, "hf_config", None)
        topk = int(getattr(hf_config, "index_topk", 0) or 0)
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

    def compile(self, compile_key: CompileKey) -> None:
        warmup = getattr(self.kernel, "warmup", None)
        assert warmup is not None
        fp32_ptr = TritonWarmupTensor(torch.float32)
        int32_ptr = TritonWarmupTensor(torch.int32)
        warmup(
            fp32_ptr,
            int32_ptr,
            fp32_ptr,
            int32_ptr,
            1,  # do not specialize logits_stride0
            1,  # do not specialize logits_stride1
            1,  # do not specialize topk_stride0
            1,  # do not specialize topk_stride1
            1,  # do not specialize packed_stride0
            1,  # do not specialize packed_stride1
            1,  # do not specialize packed_stride2
            1,  # do not specialize num_cols
            DCP_RANK=compile_key.DCP_RANK,
            DCP_WORLD_SIZE=compile_key.DCP_WORLD_SIZE,
            CP_INTERLEAVE=compile_key.CP_INTERLEAVE,
            HAS_ROW_STARTS=compile_key.HAS_ROW_STARTS,
            TOPK=compile_key.TOPK,
            BLOCK_SIZE=compile_key.BLOCK_SIZE,
            grid=(1, 1),
            num_warps=8,
        )

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
    ) -> None:
        grid = (topk_indices.shape[0], triton.cdiv(topk, block_size))
        self.kernel[grid](
            logits,
            topk_indices,
            packed,
            row_starts_arg,
            logits_stride0,
            logits_stride1,
            topk_stride0,
            topk_stride1,
            packed_stride0,
            packed_stride1,
            packed_stride2,
            num_cols,
            DCP_RANK=dcp_rank,
            DCP_WORLD_SIZE=dcp_world_size,
            CP_INTERLEAVE=cp_interleave,
            HAS_ROW_STARTS=has_row_starts,
            TOPK=topk,
            BLOCK_SIZE=block_size,
            num_warps=8,
        )


class StableTopKFromGatheredCandidatesKernel(
    VllmJitKernel["StableTopKFromGatheredCandidatesKernel.CompileKey"]
):
    @dataclass(frozen=True)
    class CompileKey:
        topk: int
        num_candidates: int

    def __init__(self) -> None:
        self._compiled_cache: dict[tuple[int, int], Any] = {}
        super().__init__()

    def dispatch(  # type: ignore[override]
        self,
        *,
        topk: int,
        num_candidates: int,
    ) -> CompileKey:
        return self.CompileKey(topk=topk, num_candidates=num_candidates)

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        parallel_config = getattr(vllm_config, "parallel_config", None)
        dcp_world_size = int(
            getattr(parallel_config, "decode_context_parallel_size", 1) or 1
        )
        if dcp_world_size <= 1:
            return []
        model_config = getattr(vllm_config, "model_config", None)
        hf_config = getattr(model_config, "hf_config", None)
        topk = int(getattr(hf_config, "index_topk", 0) or 0)
        if topk <= 0:
            return []
        return self._trace_dispatch(self.dispatch)(
            topk=topk,
            num_candidates=topk * dcp_world_size,
        )

    def compile(self, compile_key: CompileKey) -> None:
        cache_key = (compile_key.topk, compile_key.num_candidates)
        if cache_key in self._compiled_cache:
            return

        from ._stable_topk_from_gathered_candidates import (
            StableTopKFromGatheredCandidatesImpl,
        )

        num_rows = cute.sym_int()
        gathered = cute.runtime.make_fake_tensor(
            Float32,
            (num_rows, compile_key.num_candidates, 2),
            stride=(cute.sym_int64(divisibility=2), 2, 1),
            assumed_align=8,
        )
        out = make_fake_tensor(
            Int32,
            (num_rows, compile_key.topk),
            divisibility=1,
        )
        impl = StableTopKFromGatheredCandidatesImpl(
            topk=compile_key.topk,
            num_candidates=compile_key.num_candidates,
        )
        self._compiled_cache[cache_key] = compile_cutedsl(impl, gathered, out)

    def __call__(self, gathered: torch.Tensor, out: torch.Tensor, *, topk: int) -> Any:
        compile_key = self.dispatch(topk=topk, num_candidates=gathered.shape[1])
        cache_key = (compile_key.topk, compile_key.num_candidates)
        if cache_key not in self._compiled_cache:
            self.compile(compile_key)
        return self._compiled_cache[cache_key](gathered, out)


_PACK_DCP_TOPK_CANDIDATES_KERNEL = PackDCPTopkCandidatesKernel()
_STABLE_TOPK_FROM_GATHERED_CANDIDATES_KERNEL = (
    StableTopKFromGatheredCandidatesKernel()
)
