# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Indexer metadata for DeepSeek V4's ROCm MXFP4 indexer on aiter FlyDSL.

Opt-in with ``VLLM_ROCM_USE_AITER_FLYDSL_MXFP4_INDEXER=1`` next to
``indexer_kv_dtype="mxfp4"``. The K cache is written by vLLM's fused
compress -> RMSNorm -> RoPE -> MXFP4 -> insert kernel in the page layout of
aiter's FlyDSL ``pa_mqa_logits_fp4*`` scorers, which read it in place. The
builder plans each prefill chunk's row windows and the scorers' CTA
schedules once per step, so every indexer layer of the group reuses them.
"""

import bisect
import inspect
from dataclasses import dataclass, field, fields
from functools import cache

import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV4IndexerBackend,
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerMetadataBuilder,
    dsa_indexer_uses_fp4,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec

logger = init_logger(__name__)

# The FlyDSL scorers' KV tile; the CTA schedules are built for it.
FLYDSL_BLOCK_K = 256


@cache
def aiter_flydsl_mxfp4_available() -> bool:
    """Whether the installed aiter has the stride-capable FlyDSL FP4 scorers
    and the schedule builders this path calls."""
    try:
        from aiter.ops.flydsl.kernels.mqa_logits import (
            pa_mqa_logits_fp4 as decode,
        )
        from aiter.ops.flydsl.kernels.mqa_logits import (
            pa_mqa_logits_fp4_common as common,
        )
        from aiter.ops.flydsl.kernels.mqa_logits import (
            pa_mqa_logits_fp4_prefill as prefill,
        )

        if "byte_offset" not in inspect.signature(common._i32_buffer).parameters:
            return False
        required = {"kv_page_stride", "kv_scale_page_stride", "block_table_stride"}
        for module, name in (
            (decode, "pa_mqa_logits_fp4"),
            (prefill, "pa_mqa_logits_fp4_prefill"),
        ):
            for prefix in ("build_", "compile_"):
                suffix = "_module" if prefix == "build_" else ""
                function = getattr(module, prefix + name + suffix)
                if not required.issubset(inspect.signature(function).parameters):
                    return False
        for function, parameters in (
            (decode.default_varctx_parallel_unit_num, {"batch_size", "max_seq_len"}),
            (decode.flydsl_pa_mqa_logits_fp4, {"cta_info", "total_ctas"}),
            (prefill.flydsl_pa_mqa_logits_fp4_prefill, {"cta_info", "n_ctas"}),
            (
                decode.compute_varctx_schedule,
                {
                    "block_k",
                    "parallel_unit_num",
                    "max_seq_len",
                    "next_n",
                    "cta_info_out",
                },
            ),
            (
                prefill.compute_prefill_schedule,
                {"block_k", "parallel_unit_num", "max_seq_len"},
            ),
        ):
            if not parameters.issubset(inspect.signature(function).parameters):
                return False
        return True
    except (ImportError, AttributeError, TypeError, ValueError):
        return False


def rocm_flydsl_mxfp4_indexer_enabled(vllm_config: VllmConfig) -> bool:
    """Whether DeepSeek V4's MXFP4 indexer runs on aiter's FlyDSL scorers.

    Requires the opt-in flag on a DeepSeek-V4 model with the MXFP4 indexer
    cache. Without the FlyDSL interfaces it warns and the model keeps aiter's
    Triton paged MXFP4 indexer.
    """
    if not (
        current_platform.is_rocm() and envs.VLLM_ROCM_USE_AITER_FLYDSL_MXFP4_INDEXER
    ):
        return False
    model_config = vllm_config.model_config
    if model_config is None or model_config.hf_config.model_type != "deepseek_v4":
        return False
    if not dsa_indexer_uses_fp4(vllm_config):
        return False
    if not aiter_flydsl_mxfp4_available():
        logger.warning_once(
            "VLLM_ROCM_USE_AITER_FLYDSL_MXFP4_INDEXER is set, but the installed "
            "aiter lacks the stride-capable FlyDSL FP4 scorers or their schedule "
            "interfaces; using aiter's Triton paged MXFP4 indexer."
        )
        return False
    logger.info_once("DeepSeek V4 indexer: aiter FlyDSL paged MXFP4 scorers.")
    return True


@triton.jit(do_not_specialize=["ROWS", "REQUESTS", "QUERY_OFFSET", "SEARCH_STEPS"])
def _build_prefill_windows_kernel(
    query_starts,
    row_starts,
    row_ends,
    row_to_batch,
    local_starts,
    local_ends,
    ROWS,
    REQUESTS,
    QUERY_OFFSET,
    SEARCH_STEPS,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = row < ROWS
    query = row + QUERY_OFFSET
    lo = tl.full((BLOCK,), 0, tl.int32)
    hi = tl.full((BLOCK,), 0, tl.int32) + REQUESTS
    for _ in range(SEARCH_STEPS):
        mid = (lo + hi) // 2
        end = tl.load(query_starts + tl.minimum(mid + 1, REQUESTS))
        right = (mid < REQUESTS) & (end <= query)
        lo = tl.where(right, mid + 1, lo)
        hi = tl.where(right, hi, mid)
    real_end = tl.load(query_starts + REQUESTS)
    start = tl.load(row_starts + row, mask=valid, other=0)
    end = tl.load(row_ends + row, mask=valid, other=0)
    tl.store(row_to_batch + row, tl.minimum(lo, REQUESTS - 1), mask=valid)
    tl.store(local_starts + row, 0, mask=valid)
    tl.store(
        local_ends + row,
        tl.where(query < real_end, tl.maximum(end - start, 0), 0),
        mask=valid,
    )


def build_prefill_windows(
    query_start_loc: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    rows: int,
    num_reqs: int,
    query_offset: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(request, first key, end key) of each of a chunk's rows, keys local to
    the request.

    ``query_start_loc`` holds the chunk's request row offsets from 0, and
    ``query_offset`` is the chunk's first row within them, nonzero when a
    request is split on its query rows.
    """
    windows = torch.empty((3, rows), dtype=torch.int32, device=query_start_loc.device)
    row_to_batch, starts, ends = windows.unbind(0)
    if rows:
        _build_prefill_windows_kernel[(triton.cdiv(rows, 256),)](
            query_start_loc,
            row_starts,
            row_ends,
            row_to_batch,
            starts,
            ends,
            rows,
            num_reqs,
            query_offset,
            (num_reqs + 1).bit_length(),
            BLOCK=256,
        )
    return row_to_batch, starts, ends


@dataclass
class RocmFlyDslPrefillPlan:
    """One prefill chunk as the FlyDSL prefill scorer launches it."""

    width: int
    """Logits columns: the chunk's widest compressed context."""
    windows: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """[rows] int32 request, first key and end key of each row."""
    cta_info: torch.Tensor | None = None
    """The scorer's work schedule; None when every row keeps its whole
    context because the chunk is no wider than the top-k."""
    total_ctas: int = 0


@dataclass
class RocmFlyDslIndexerMetadata(DeepseekV32IndexerMetadata):
    prefill_plans: list[RocmFlyDslPrefillPlan] = field(default_factory=list)
    """Parallel to ``prefill.chunks``."""
    prefill_logits: torch.Tensor | None = None
    """Builder-owned fp32 storage for the widest scored chunk's logits."""
    decode_cta_info: torch.Tensor | None = None
    """A view of builder-owned storage, refreshed before each graph replay."""
    decode_total_ctas: int = 0


class DeepseekV4RocmFlyDslIndexerBackend(DeepseekV4IndexerBackend):
    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_V4_ROCM_FLYDSL_MXFP4_INDEXER"

    @staticmethod
    def get_builder_cls() -> type["RocmFlyDslIndexerMetadataBuilder"]:
        return RocmFlyDslIndexerMetadataBuilder


class RocmFlyDslIndexerMetadataBuilder(DeepseekV32IndexerMetadataBuilder):
    """Dense indexer metadata plus the FlyDSL scorers' windows and schedules.

    Decode rows are flattened, one per query token, so speculative rows get
    their own causal bound before the C4 division.
    """

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: KVCacheSpec,
    ) -> AttentionCGSupport:
        return AttentionCGSupport.ALWAYS

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        **kwargs,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device, **kwargs)
        if self.dcp_world_size > 1 or self.use_pcp:
            raise NotImplementedError(
                "The ROCm FlyDSL MXFP4 indexer does not support decode/prefill "
                "context parallel."
            )
        if self.compress_ratio != 4 or self.kv_cache_spec.num_states != 64:
            raise ValueError(
                "The ROCm FlyDSL MXFP4 indexer requires C4 compression and "
                "64-entry compressed pages."
            )
        self.use_flattening = True
        self.topk_tokens = int(vllm_config.model_config.hf_config.index_topk)
        self.max_context = vllm_config.model_config.max_model_len // self.compress_ratio
        from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4 import (
            default_varctx_parallel_unit_num,
        )

        compilation_config = vllm_config.compilation_config
        max_decode_rows = max(
            vllm_config.scheduler_config.max_num_seqs
            * (self.num_speculative_tokens + 1),
            compilation_config.max_cudagraph_capture_size or 0,
            max(compilation_config.cudagraph_capture_sizes or (), default=0),
        )
        # Every indexer layer of the group reads the same schedule; build()
        # refreshes it in place before eager execution or graph replay.
        self.decode_cta_info_buffer = torch.empty(
            (default_varctx_parallel_unit_num(max_decode_rows, self.max_context), 4),
            dtype=torch.int32,
            device=device,
        )
        self.prefill_logits_buffer: torch.Tensor | None = None

    def _split_indexer_prefill_chunks(  # type: ignore[override]
        self,
        compressed_seq_lens_cpu: torch.Tensor,
        prefill_query_lens_cpu: torch.Tensor,
        workspace_size: int,
        max_logits_bytes: int,
        request_offset: int = 0,
    ) -> list[tuple[slice, slice]]:
        # The scorers read pages in place, so a chunk's logits are rows times
        # its widest context rather than the sum of gathered contexts.
        from vllm.v1.attention.backends.mla.rocm_paged_mxfp4_indexer import (
            split_prefill_chunks,
        )

        return split_prefill_chunks(
            compressed_seq_lens_cpu,
            prefill_query_lens_cpu,
            max_logits_bytes,
            request_offset,
        )

    def _plan_prefill(
        self, metadata: RocmFlyDslIndexerMetadata, cm: CommonAttentionMetadata
    ) -> None:
        from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4 import (
            default_varctx_parallel_unit_num,
        )
        from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
            compute_prefill_schedule,
        )

        assert metadata.prefill is not None
        query_start_loc = cm.query_start_loc_cpu.tolist()
        logits_elems = 0
        for chunk in metadata.prefill.chunks:
            rows = chunk.token_end - chunk.token_start
            first = bisect.bisect_right(query_start_loc, chunk.token_start) - 1
            request_starts = cm.query_start_loc[first : first + chunk.num_reqs + 1]
            windows = build_prefill_windows(
                request_starts - request_starts[0],
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                rows,
                chunk.num_reqs,
                chunk.token_start - query_start_loc[first],
            )
            plan = RocmFlyDslPrefillPlan(chunk.max_seqlen_k, windows)
            if rows and plan.width > self.topk_tokens:
                _, plan.cta_info, plan.total_ctas = compute_prefill_schedule(
                    *windows,
                    block_k=FLYDSL_BLOCK_K,
                    parallel_unit_num=default_varctx_parallel_unit_num(
                        rows, plan.width
                    ),
                    max_seq_len=plan.width,
                )
                logits_elems = max(logits_elems, rows * plan.width)
            metadata.prefill_plans.append(plan)
        if logits_elems:
            if (
                self.prefill_logits_buffer is None
                or self.prefill_logits_buffer.numel() < logits_elems
            ):
                self.prefill_logits_buffer = torch.empty(
                    logits_elems, dtype=torch.float32, device=self.device
                )
            metadata.prefill_logits = self.prefill_logits_buffer

    def _plan_decode(self, metadata: RocmFlyDslIndexerMetadata) -> None:
        from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4 import (
            compute_varctx_schedule,
            default_varctx_parallel_unit_num,
        )

        assert metadata.decode is not None
        rows = metadata.num_decode_tokens
        seq_lens = metadata.decode.seq_lens
        assert seq_lens.shape == (rows, 1)
        parallel_unit_num = default_varctx_parallel_unit_num(rows, self.max_context)
        assert parallel_unit_num <= self.decode_cta_info_buffer.shape[0]
        _, metadata.decode_cta_info, metadata.decode_total_ctas = (
            compute_varctx_schedule(
                seq_lens[:, 0],
                block_k=FLYDSL_BLOCK_K,
                parallel_unit_num=parallel_unit_num,
                max_seq_len=self.max_context,
                next_n=1,
                cta_info_out=self.decode_cta_info_buffer[:parallel_unit_num],
            )
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> RocmFlyDslIndexerMetadata:
        base = super().build(common_prefix_len, common_attn_metadata, fast_build)
        metadata = RocmFlyDslIndexerMetadata(
            **{f.name: getattr(base, f.name) for f in fields(base)}
        )
        if metadata.prefill is not None:
            self._plan_prefill(metadata, common_attn_metadata)
        if metadata.decode is not None and metadata.num_decode_tokens:
            self._plan_decode(metadata)
        return metadata
