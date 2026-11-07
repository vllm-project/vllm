# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSV4.1 candidate-only prefill and decode over the paged indexer cache."""

from dataclasses import dataclass, fields
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import Int32
from cutlass.cute.runtime import make_fake_tensor

from vllm import envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits import (
    _sparse_topk_remap_kernel,
    candidate_blocks_to_sparse_indices,
    pick_sparse_block_kv,
)
from vllm.model_executor.layers.indexer_topk import get_indexer_topk
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_cutedsl_helper import (
    CuTeDSLLaunchSpec,
    VllmCuTeDSLJitKernel,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerMetadataBuilder,
    DeepseekV41IndexerBackend,
    dsa_indexer_uses_fp4,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.workspace import current_workspace_manager

from .candidate_paged_logits import candidate_paged_logits, prepare_candidate_pages

logger = init_logger(__name__)


class CandidateRowsKernel(VllmCuTeDSLJitKernel[int]):
    @staticmethod
    def kernel(compress_ratio: int) -> Any:
        @cute.kernel
        def device_kernel(
            starts: cute.Tensor,
            lens: cute.Tensor,
            row_lens: cute.Tensor,
            row_indices: cute.Tensor,
        ):
            request, _, _ = cute.arch.block_idx()
            tid, _, _ = cute.arch.thread_idx()
            start, end = starts[request], starts[request + 1]
            length = lens[request]
            for row in range(start + tid, end, 32):
                row_lens[row] = cutlass.max(
                    (length - (end - row) + 1) // compress_ratio, 0
                )
                row_indices[row] = request

        @cute.jit
        def entry(
            starts: cute.Tensor,
            lens: cute.Tensor,
            row_lens: cute.Tensor,
            row_indices: cute.Tensor,
            stream: CUstream,
        ):
            device_kernel(starts, lens, row_lens, row_indices).launch(
                grid=(lens.shape[0], 1, 1), block=(32, 1, 1), stream=stream
            )

        return entry

    def get_warmup_keys(self, *, compress_ratio: int) -> list[int]:
        return [compress_ratio]

    def warmup_inputs(self, key: int) -> tuple[Any, ...]:
        return tuple(
            make_fake_tensor(Int32, (cute.sym_int(),), (1,), assumed_align=4)
            for _ in range(4)
        )

    @kernel_launcher
    def __call__(
        self,
        starts: torch.Tensor,
        lens: torch.Tensor,
        row_lens: torch.Tensor,
        row_indices: torch.Tensor,
        compress_ratio: int,
    ) -> CuTeDSLLaunchSpec[int]:
        return compress_ratio, (starts, lens, row_lens, row_indices)


_candidate_rows = CandidateRowsKernel()


def _candidate_buffers(config: VllmConfig, compress_ratio: int, device: torch.device):
    hf = config.model_config.hf_text_config
    rows = config.scheduler_config.max_num_batched_tokens
    count, block, topk = (
        hf.candidate_topk_blocks,
        hf.candidate_block_size,
        hf.index_topk,
    )
    logits_rows = min(
        rows,
        max(
            1,
            envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024 // (count * block * 4),
        ),
    )
    key = ("dsv41_candidate", compress_ratio, rows, count, block, topk, logits_rows)

    def allocate():
        i32 = dict(device=device, dtype=torch.int32)
        return (
            torch.zeros(rows, **i32),
            torch.zeros(rows, **i32),
            torch.zeros(rows, **i32),
            torch.zeros((rows, count), **i32),
            torch.zeros((rows, count), device=device, dtype=torch.int64),
            torch.zeros(rows, **i32),
            torch.zeros((logits_rows, topk), **i32),
            torch.empty(
                (logits_rows, count * block), device=device, dtype=torch.float32
            ),
        )

    # Allocated during memory profiling, then reused by builders and all four
    # consumer layers. Persistent buffers cannot alias top-k's shared scratch.
    return current_workspace_manager().get_persistent_resource(key, allocate)


@dataclass
class CandidatePlan:
    """One step's row bounds and sorted candidates, shared by consumer layers."""

    block_table: torch.Tensor
    row_indices: torch.Tensor
    row_lens: torch.Tensor
    row_starts: torch.Tensor
    sparse_indices: torch.Tensor
    physical_blocks: torch.Tensor
    valid_end: torch.Tensor
    columns: torch.Tensor
    logits: torch.Tensor
    sparse_block_size: int
    prepared: bool = False


@dataclass
class CandidateIndexerMetadata(DeepseekV32IndexerMetadata):
    candidate: CandidatePlan | None = None


class CandidateIndexerBackend(DeepseekV41IndexerBackend):
    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_V41_CANDIDATE_INDEXER"

    @staticmethod
    def get_builder_cls() -> type["CandidateIndexerMetadataBuilder"]:
        return CandidateIndexerMetadataBuilder


class CandidateIndexerMetadataBuilder(DeepseekV32IndexerMetadataBuilder):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        **kwargs,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device, **kwargs)
        config = vllm_config.model_config.hf_text_config
        prefix = "attention_config.indexer_sparse_logits"
        if not current_platform.is_device_capability_family(90):
            raise ValueError(f"{prefix} requires an SM90 GPU")
        if dsa_indexer_uses_fp4(vllm_config):
            raise ValueError(f"{prefix} requires the FP8 indexer cache")
        if (config.index_n_heads, config.index_head_dim) != (32, 128):
            raise ValueError(f"{prefix} requires 32 indexer heads of size 128")
        if self.dcp_world_size != 1 or self.use_pcp:
            raise ValueError(f"{prefix} does not support context parallelism")
        if vllm_config.parallel_config.use_ubatching:
            raise ValueError(f"{prefix} does not support microbatching")
        count = config.candidate_topk_blocks
        block = config.candidate_block_size
        if (
            count <= 0
            or count & (count - 1)
            or block not in (8, 16)
            or count * block < max(64, config.index_topk)
        ):
            raise ValueError(
                f"{prefix} requires power-of-two candidates of size 8 or 16"
            )
        self.sparse_block_size = pick_sparse_block_kv(block)
        if kv_cache_spec.num_states not in (16, 32, 64):
            raise ValueError(f"{prefix} requires indexer cache pages of 16, 32 or 64")
        (
            self.row_lens,
            self.row_indices,
            self.row_starts,
            self.sparse_indices,
            self.physical_blocks,
            self.valid_end,
            self.columns,
            self.logits,
        ) = _candidate_buffers(vllm_config, self.compress_ratio, device)
        if vllm_config.kernel_config.enable_jit_warmup:
            _candidate_rows.register_warmup(compress_ratio=self.compress_ratio)
            candidate_paged_logits.register_warmup(
                page_size=kv_cache_spec.num_states,
                sparse_block_size=self.sparse_block_size,
            )
            prepare_candidate_pages.register_warmup(
                page_size=kv_cache_spec.num_states,
                sparse_block_size=self.sparse_block_size,
            )
        logger.info_once(
            "DSV4.1: CuTe DSL candidate-only decode and long-context prefill."
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CandidateIndexerMetadata:
        base = super().build(common_prefix_len, common_attn_metadata, fast_build)
        metadata = CandidateIndexerMetadata(
            **{f.name: getattr(base, f.name) for f in fields(base)}
        )
        # Dense prefill reuses K across queries, especially for a large causal
        # chunk. Candidate scoring needs a long context relative to both.
        if (
            base.prefill is not None
            and base.prefill.max_prefill_seq_len // self.compress_ratio
            < 2
            * max(
                self.logits.shape[1],
                common_attn_metadata.max_query_len // self.compress_ratio,
            )
        ):
            return metadata
        rows = base.num_decode_tokens + base.num_prefill_tokens
        requests = base.num_decodes + base.num_prefills
        if rows:
            _candidate_rows(
                common_attn_metadata.query_start_loc[: requests + 1],
                common_attn_metadata.seq_lens[:requests],
                self.row_lens[:rows],
                self.row_indices[:rows],
                self.compress_ratio,
            )
            metadata.candidate = CandidatePlan(
                block_table=common_attn_metadata.block_table_tensor[:requests],
                row_indices=self.row_indices[:rows],
                row_lens=self.row_lens[:rows],
                row_starts=self.row_starts[:rows],
                sparse_indices=self.sparse_indices[:rows],
                physical_blocks=self.physical_blocks[:rows],
                valid_end=self.valid_end[:rows],
                columns=self.columns,
                logits=self.logits,
                sparse_block_size=self.sparse_block_size,
            )
        return metadata

    def update_draft_decode_metadata(
        self, metadata: DeepseekV32IndexerMetadata
    ) -> None:
        super().update_draft_decode_metadata(metadata)
        assert isinstance(metadata, CandidateIndexerMetadata)
        plan = metadata.candidate
        if plan is not None:
            assert metadata.decode is not None
            plan.row_lens[: metadata.num_decode_tokens].copy_(
                metadata.decode.seq_lens.view(-1)
            )
            plan.prepared = False


class CandidateIndexer(SparseAttnIndexer):
    weights_dtype = torch.float32

    def __init__(
        self,
        k_cache,
        topk_tokens: int,
        head_dim: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        candidate_blocks: torch.Tensor,
        candidate_block_size: int,
    ) -> None:
        config = get_current_vllm_config()
        if dsa_indexer_uses_fp4(config):
            raise ValueError("The candidate indexer requires an FP8 cache")
        super().__init__(
            k_cache,
            128,
            "ue8m0",
            topk_tokens,
            head_dim,
            config.model_config.max_model_len // k_cache.compress_ratio,
            max_total_seq_len,
            topk_indices_buffer,
            skip_k_cache_insert=True,
            compress_ratio=k_cache.compress_ratio,
            candidate_blocks=candidate_blocks,
            candidate_block_size=candidate_block_size,
        )
        self.config = config
        # Cache backends are resolved after model construction. Only caches
        # with candidate consumers need the additional metadata and buffers.
        k_cache.candidate_indexer = True

    @eager_break_during_capture
    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor | None,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        assert isinstance(q_quant, torch.Tensor) and k is None
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            _candidate_buffers(self.config, self.compress_ratio, hidden_states.device)
            return super().forward_cuda(hidden_states, q_quant, k, weights)
        metadata = attn_metadata[self.k_cache.prefix]
        assert isinstance(metadata, CandidateIndexerMetadata)
        if metadata.candidate is None:
            return super().forward_cuda(hidden_states, q_quant, k, weights)
        self.topk_indices_buffer.fill_(-1)
        self._forward_candidates(q_quant, weights, metadata.candidate)
        return self.topk_indices_buffer

    def _forward_candidates(
        self,
        q_quant: torch.Tensor,
        weights: torch.Tensor,
        plan: CandidatePlan,
    ) -> None:
        rows = plan.row_lens.numel()
        assert self.candidate_blocks is not None
        if not plan.prepared:
            candidate_blocks_to_sparse_indices(
                self.candidate_blocks[:rows],
                plan.row_starts,
                plan.row_lens,
                self.candidate_block_size,
                plan.sparse_block_size,
                out=(plan.sparse_indices, plan.valid_end),
            )
            prepare_candidate_pages(
                plan.block_table,
                plan.sparse_indices,
                plan.valid_end,
                plan.row_indices,
                plan.physical_blocks,
                page_size=self.k_cache.kv_cache.shape[1],
                sparse_block_size=plan.sparse_block_size,
            )
            plan.prepared = True
        for start in range(0, rows, plan.logits.shape[0]):
            end = min(start + plan.logits.shape[0], rows)
            logits = plan.logits[: end - start]
            columns = plan.columns[: end - start]
            candidate_paged_logits(
                q_quant[start:end],
                self.k_cache.kv_cache,
                weights[start:end],
                plan.physical_blocks[start:end],
                plan.valid_end[start:end],
                logits,
                sparse_block_size=plan.sparse_block_size,
            )
            columns.fill_(-1)
            get_indexer_topk(self.topk_backend)(
                logits,
                plan.valid_end[start:end],
                1,
                columns,
                self.topk_tokens,
                logits.shape[1],
            )
            _sparse_topk_remap_kernel[(end - start,)](
                columns,
                columns.stride(0),
                plan.sparse_indices[start:end],
                plan.sparse_indices.stride(0),
                plan.row_starts[start:end],
                plan.row_starts.stride(0),
                self.topk_indices_buffer[start:end],
                self.topk_indices_buffer.stride(0),
                self.topk_tokens,
                logits.shape[1],
                SBK=plan.sparse_block_size,
                K_POW2=triton.next_power_of_2(self.topk_tokens),
            )
