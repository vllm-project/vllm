# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12x DSA indexer for non-compressed sparse MLA models."""

import bisect
import os
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn

import vllm.envs as envs
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.utils.b12x import B12xWarmupUnit, get_b12x_dsa_indexer
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    DeepSeekV32IndexerDecodeMetadata,
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerMetadataBuilder,
    split_indexer_prefill_chunks,
)
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.v1.worker.workspace import current_workspace_manager

_INDEX_HEAD_DIM = 128
_INDEX_SCALE_BYTES = 4
_INDEX_PAGE_SIZE = 64
_INDEX_PAGE_WIDTH = _INDEX_PAGE_SIZE * (_INDEX_HEAD_DIM + _INDEX_SCALE_BYTES)
_PREFILL_PROFILE_SUPERTILE_K = 32 * 1024


def _prefill_profile_q_rows(max_q_rows: int) -> int:
    max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024 // 4
    supertile_k = int(
        os.environ.get("B12X_PAGED_INDEX_SUPERTILE_K", _PREFILL_PROFILE_SUPERTILE_K)
    )
    supertile_k = max(supertile_k, 256)
    return min(max(int(max_q_rows), 1), max(1, max_logits_elems // supertile_k))


def _is_current_stream_capturing(tensor: torch.Tensor) -> bool:
    return tensor.is_cuda and torch.cuda.is_current_stream_capturing()


@dataclass
class B12xIndexerDecodeMetadata(DeepSeekV32IndexerDecodeMetadata):
    active_width: torch.Tensor | None = None


class B12xIndexerMetadataBuilder(DeepseekV32IndexerMetadataBuilder):
    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: KVCacheSpec,
    ) -> AttentionCGSupport:
        return AttentionCGSupport.ALWAYS

    def __init__(self, *args, block_table_width: int, **kwargs) -> None:
        super().__init__(*args, block_table_width=block_table_width, **kwargs)
        self.use_flattening = False
        self.supports_varlen = False
        self.active_width_buffer = torch.zeros(
            (1,), dtype=torch.int32, device=self.device
        )

    def _supports_native_decode(self, next_n: int) -> bool:
        return True

    def _split_prefill_chunks(
        self,
        compressed_seq_lens_cpu: torch.Tensor,
        prefill_query_lens_cpu: torch.Tensor,
        num_decodes: int,
        max_logits_bytes: int,
    ) -> list[tuple[slice, slice]]:
        return [
            chunk
            for prefill_idx in range(len(prefill_query_lens_cpu))
            for chunk in split_indexer_prefill_chunks(
                compressed_seq_lens_cpu[
                    num_decodes + prefill_idx : num_decodes + prefill_idx + 1
                ],
                prefill_query_lens_cpu[prefill_idx : prefill_idx + 1],
                self.max_prefill_buffer_size,
                max_logits_bytes,
                request_offset=num_decodes + prefill_idx,
            )
        ]

    def build(self, *args, **kwargs) -> DeepseekV32IndexerMetadata:
        metadata = super().build(*args, **kwargs)
        self.active_width_buffer.fill_(int(metadata.max_seq_len))
        if metadata.decode is not None:
            decode = metadata.decode
            seq_lens = decode.seq_lens.reshape(-1).contiguous()
            fields = vars(decode).copy()
            fields["seq_lens"] = seq_lens
            fields["schedule_metadata"] = None
            metadata.decode = B12xIndexerDecodeMetadata(
                **fields,
                active_width=self.active_width_buffer,
            )
        return metadata


class B12xIndexerBackend(DeepseekV32IndexerBackend):
    @classmethod
    def supports_pcp(cls) -> bool:
        return False

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return False

    @staticmethod
    def get_name() -> str:
        return "B12X_INDEXER"

    @staticmethod
    def get_builder_cls() -> type[B12xIndexerMetadataBuilder]:
        return B12xIndexerMetadataBuilder


class B12xIndexerCache(DeepseekV32IndexerCache):
    def get_attn_backend(self) -> type[B12xIndexerBackend]:
        return B12xIndexerBackend


def _require_b12x_indexer() -> Any:
    module = get_b12x_dsa_indexer()
    if module is None:
        raise RuntimeError("B12X sparse MLA requires `pip install vllm[b12x]`.")
    if not module.is_supported():
        raise RuntimeError("B12X sparse indexer is not supported on this device.")
    if int(module.PAGED_INDEX_PAGE_SIZE) != _INDEX_PAGE_SIZE:
        raise RuntimeError(
            "B12X sparse indexer page size changed: expected "
            f"{_INDEX_PAGE_SIZE}, got {module.PAGED_INDEX_PAGE_SIZE}."
        )
    for name in (
        "Caps",
        "bind",
        "plan",
        "run",
    ):
        getattr(module, name)
    return module


def _flatten_index_cache(kv_cache: torch.Tensor) -> torch.Tensor:
    expected_tail = (_INDEX_PAGE_SIZE, _INDEX_HEAD_DIM + _INDEX_SCALE_BYTES)
    if (
        kv_cache.ndim != 3
        or kv_cache.dtype != torch.uint8
        or tuple(kv_cache.shape[1:]) != expected_tail
    ):
        raise RuntimeError(
            "B12X indexer cache must have shape "
            f"[num_blocks, {expected_tail[0]}, {expected_tail[1]}] and dtype "
            f"uint8, got shape={tuple(kv_cache.shape)} dtype={kv_cache.dtype}."
        )
    if kv_cache.stride(1) != expected_tail[1] or kv_cache.stride(2) != 1:
        raise RuntimeError(
            "B12X indexer cache requires contiguous page payloads, got stride "
            f"{tuple(kv_cache.stride())}."
        )
    return kv_cache.as_strided(
        (int(kv_cache.shape[0]), _INDEX_PAGE_WIDTH),
        (int(kv_cache.stride(0)), 1),
    )


def _run_paged_topk(
    *,
    module: Any,
    plan: Any,
    q: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    active_width: torch.Tensor | None,
    output: torch.Tensor,
) -> None:
    scratch = current_workspace_manager().get_simultaneous(*plan.shapes_and_dtypes())
    if active_width is None:
        raise RuntimeError("B12X DSA requires a device active-width scalar.")
    binding = module.bind(
        plan,
        scratch=scratch,
        q_fp8=q,
        query_weights=weights,
        index_k_cache=_flatten_index_cache(kv_cache),
        page_table=block_table,
        cache_lengths=seq_lens,
        active_width=active_width,
        output_indices=output,
    )
    module.run(binding)


class B12xSparseIndexer(nn.Module):
    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor | None,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
        compress_ratio: int = 1,
        num_q_heads: int | None = None,
        output_physical_slots: bool = True,
    ) -> None:
        super().__init__()
        del quant_block_size, scale_fmt, max_total_seq_len
        if not skip_k_cache_insert:
            raise ValueError("B12X requires the fused DSA index-cache insert path.")
        if use_fp4_cache:
            raise ValueError("B12X indexing requires the FP8 index cache.")
        if compress_ratio != 1:
            raise ValueError(
                "The non-compressed B12X indexer requires compress_ratio=1."
            )
        if head_dim != _INDEX_HEAD_DIM:
            raise ValueError(
                f"B12X indexing requires head_dim={_INDEX_HEAD_DIM}, got {head_dim}."
            )
        if topk_indices_buffer is None:
            raise ValueError("B12X indexing requires a top-k output buffer.")
        if num_q_heads is None or int(num_q_heads) <= 0:
            raise ValueError(
                "B12X indexing requires a positive index query head count."
            )
        if not output_physical_slots:
            raise ValueError(
                "B12X sparse MLA requires rank-local physical indexer output."
            )
        self._module = _require_b12x_indexer()
        self.k_cache = k_cache
        self.topk_tokens = int(topk_tokens)
        self.max_model_len = int(max_model_len)
        self.topk_indices_buffer = topk_indices_buffer
        self.output_physical_slots = bool(output_physical_slots)
        self.active_width_cap = torch.full(
            (1,),
            self.max_model_len,
            dtype=torch.int32,
            device=topk_indices_buffer.device,
        )
        max_q_rows = int(topk_indices_buffer.shape[0])
        max_page_table_width = max(
            1, (self.max_model_len + _INDEX_PAGE_SIZE - 1) // _INDEX_PAGE_SIZE
        )
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        scheduler_config = vllm_config.scheduler_config
        parallel_config = vllm_config.parallel_config
        max_num_seqs = int(scheduler_config.max_num_seqs)

        def make_plan(*, mode: str, q_rows: int):
            return self._module.plan(
                self._module.Caps(
                    device=topk_indices_buffer.device,
                    num_q_heads=int(num_q_heads),
                    max_q_rows=q_rows,
                    max_page_table_width=max_page_table_width,
                    topk=self.topk_tokens,
                    mode=mode,
                    max_batch=q_rows if mode == "decode" else max_num_seqs,
                    output_index_space=(
                        "physical" if self.output_physical_slots else "logical"
                    ),
                )
            )

        self._make_plan = make_plan
        capture_sizes = vllm_config.compilation_config.cudagraph_capture_sizes or []
        decode_plan_sizes = {
            int(size) for size in capture_sizes if 0 < int(size) <= max_num_seqs
        }
        decode_plan_sizes.add(max_num_seqs)
        self._decode_plan_sizes = sorted(decode_plan_sizes)
        self._decode_plans = {
            rows: make_plan(mode="decode", q_rows=rows)
            for rows in self._decode_plan_sizes
        }
        prefill_profile_rows = _prefill_profile_q_rows(max_q_rows)
        self._prefill_plans = {
            prefill_profile_rows: make_plan(mode="prefill", q_rows=prefill_profile_rows)
        }
        self._prefill_plan_sizes = [prefill_profile_rows]
        self.dcp_world_size = parallel_config.decode_context_parallel_size
        object.__setattr__(self, "b12x_warmup_provider", self)

    def _get_plan(self, mode: str, q_rows: int) -> Any:
        q_rows = int(q_rows)
        if mode == "decode":
            index = bisect.bisect_left(self._decode_plan_sizes, q_rows)
            if index < len(self._decode_plan_sizes):
                return self._decode_plans[self._decode_plan_sizes[index]]
            plans = self._decode_plans
            plan_sizes = self._decode_plan_sizes
        else:
            index = bisect.bisect_left(self._prefill_plan_sizes, q_rows)
            if index < len(self._prefill_plan_sizes):
                return self._prefill_plans[self._prefill_plan_sizes[index]]
            plans = self._prefill_plans
            plan_sizes = self._prefill_plan_sizes
        if _is_current_stream_capturing(self.topk_indices_buffer):
            raise RuntimeError(
                f"B12X DSA {mode} plan for {q_rows} rows was not prepared before "
                "CUDA graph capture."
            )
        plan = self._make_plan(mode=mode, q_rows=q_rows)
        plans[q_rows] = plan
        bisect.insort(plan_sizes, q_rows)
        return plan

    def _reserve_profile_workspace(self) -> None:
        for plan in (*self._decode_plans.values(), *self._prefill_plans.values()):
            current_workspace_manager().get_simultaneous(*plan.shapes_and_dtypes())

    def get_b12x_warmup_unit(
        self,
        layer: torch.nn.Module,
        token_counts: tuple[int, ...],
        output_dtype: torch.dtype,
    ) -> B12xWarmupUnit:
        del layer, token_counts, output_dtype

        def compile() -> None:
            kv_cache = self.k_cache.kv_cache
            if kv_cache.numel() == 0:
                raise RuntimeError(
                    "B12X DSA warmup requires the finalized index KV cache."
                )
            plans = (*self._decode_plans.values(), *self._prefill_plans.values())
            for plan in plans:
                caps = plan.caps
                rows = int(caps.max_q_rows)
                mode = str(caps.mode)
                q = torch.zeros(
                    (rows, int(caps.num_q_heads), _INDEX_HEAD_DIM),
                    dtype=torch.float8_e4m3fn,
                    device=caps.device,
                )
                weights = torch.zeros(
                    (rows, int(caps.num_q_heads)),
                    dtype=torch.float32,
                    device=caps.device,
                )
                cache_lengths = torch.full(
                    (rows,),
                    min(self.max_model_len, _INDEX_PAGE_SIZE),
                    dtype=torch.int32,
                    device=caps.device,
                )
                page_rows = rows if mode == "decode" else 1
                page_table = torch.zeros(
                    (page_rows, int(caps.max_page_table_width)),
                    dtype=torch.int32,
                    device=caps.device,
                )
                if mode == "prefill":
                    page_table = page_table.expand(rows, -1)
                output = torch.empty(
                    (rows, self.topk_tokens),
                    dtype=torch.int32,
                    device=caps.device,
                )
                _run_paged_topk(
                    module=self._module,
                    plan=plan,
                    q=q,
                    weights=weights,
                    kv_cache=kv_cache,
                    seq_lens=cache_lengths,
                    block_table=page_table,
                    active_width=self.active_width_cap,
                    output=output,
                )

        plan_key = tuple(
            (
                str(plan.caps.mode),
                int(plan.caps.max_q_rows),
                int(plan.caps.max_page_table_width),
                getattr(plan.layout, "route", None),
            )
            for plan in (*self._decode_plans.values(), *self._prefill_plans.values())
        )
        return B12xWarmupUnit(
            name="DSA indexer",
            key=(
                type(self),
                self.topk_indices_buffer.device,
                self.topk_tokens,
                self.output_physical_slots,
                plan_key,
            ),
            compile=compile,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor | None,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        del hidden_states
        if not isinstance(q_quant, torch.Tensor):
            raise ValueError("B12X indexing requires FP8 index queries.")
        if k is not None:
            raise ValueError("B12X index K must be written by the fused cache path.")

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if not isinstance(attn_metadata, dict):
            if (
                forward_context.cudagraph_runtime_mode == CUDAGraphMode.NONE
                and forward_context.batch_descriptor is not None
            ):
                self._reserve_profile_workspace()
            return self.topk_indices_buffer

        metadata = cast(DeepseekV32IndexerMetadata, attn_metadata[self.k_cache.prefix])
        if metadata.prefill is not None:
            for chunk in metadata.prefill.chunks:
                if chunk.num_reqs != 1:
                    raise RuntimeError(
                        "B12X sparse prefill requires single-request chunks."
                    )
                start, end = chunk.token_start, chunk.token_end
                q_chunk = q_quant[start:end].contiguous()
                weights_chunk = weights[start:end].contiguous()
                output = self.topk_indices_buffer[start:end, : self.topk_tokens]
                seq_lens = (chunk.cu_seqlen_ke - chunk.cu_seqlen_ks).contiguous()
                local_rows = (
                    chunk.local_total_seq_lens
                    if self.dcp_world_size > 1
                    else chunk.total_seq_lens
                )
                active_pages = max(
                    1, (int(local_rows) + _INDEX_PAGE_SIZE - 1) // _INDEX_PAGE_SIZE
                )
                active_pages = min(active_pages, int(chunk.block_table.shape[1]))
                block_table = chunk.block_table[:1, :active_pages].expand(
                    int(q_chunk.shape[0]), active_pages
                )
                _run_paged_topk(
                    module=self._module,
                    plan=self._get_plan("prefill", int(q_chunk.shape[0])),
                    q=q_chunk,
                    weights=weights_chunk,
                    kv_cache=self.k_cache.kv_cache,
                    seq_lens=seq_lens,
                    block_table=block_table,
                    active_width=self.active_width_cap,
                    output=output,
                )

        if metadata.decode is not None:
            decode = metadata.decode
            if decode.requires_padding:
                raise RuntimeError("B12X sparse decode does not support padded rows.")
            seq_lens = decode.seq_lens.reshape(-1).contiguous()
            block_table = decode.block_table
            if int(block_table.shape[0]) != int(seq_lens.shape[0]):
                if int(seq_lens.shape[0]) % int(block_table.shape[0]) != 0:
                    raise RuntimeError(
                        "B12X sparse decode could not align lengths and page tables."
                    )
                block_table = block_table.repeat_interleave(
                    int(seq_lens.shape[0]) // int(block_table.shape[0]), dim=0
                )
            num_tokens = metadata.num_decode_tokens
            output = self.topk_indices_buffer[:num_tokens, : self.topk_tokens]
            _run_paged_topk(
                module=self._module,
                plan=self._get_plan("decode", num_tokens),
                q=q_quant[:num_tokens].contiguous(),
                weights=weights[:num_tokens].contiguous(),
                kv_cache=self.k_cache.kv_cache,
                seq_lens=seq_lens[:num_tokens],
                block_table=block_table[:num_tokens].contiguous(),
                active_width=getattr(decode, "active_width", None),
                output=output,
            )

        return self.topk_indices_buffer
