# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in SM90 native-head sparse attention for DeepSeek V4.1."""

from typing import TYPE_CHECKING, ClassVar, cast

import torch

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.nvidia.ops.o_proj import compute_fp8_einsum_recipe
from vllm.models.deepseek_v41.attention import DeepseekV4Attention
from vllm.models.deepseek_v41.common.ops import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
)
from vllm.models.deepseek_v41.nvidia.fewhead_prefill import run_fewhead_sparse_prefill
from vllm.models.deepseek_v41.nvidia.ops.o_proj import (
    dsv41_o_proj,
    register_dsv41_o_proj_warmup,
)
from vllm.models.deepseek_v41.sparse_mla import (
    DeepseekV4FlashMLAMetadata,
    DeepseekV4SparseMLABackend,
    DeepseekV4SparseMLAMetadataBuilder,
    DeepseekV41SparseSWAMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import round_up
from vllm.v1.attention.backend import AttentionCGSupport, max_decode_query_len
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWABackend
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata


def _run_decode(*args, **kwargs) -> None:
    # Kernel dependency: upstream PR #59418, eddfa5ed50c1, by luoyuctl.
    from vllm.models.deepseek_v41.nvidia.ops.small_head_sparse_decode import (
        small_head_sparse_decode,
    )

    small_head_sparse_decode(*args, **kwargs)


class DeepseekV41TritonMetadataBuilder(DeepseekV4SparseMLAMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    @classmethod
    def get_varlen_cudagraph_max_query_len(
        cls, vllm_config: VllmConfig, kv_cache_spec: KVCacheSpec
    ) -> int | None:
        return max_decode_query_len(vllm_config)


class DeepseekV41TritonSWAMetadataBuilder(DeepseekV41SparseSWAMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    @classmethod
    def get_varlen_cudagraph_max_query_len(
        cls, vllm_config: VllmConfig, kv_cache_spec: KVCacheSpec
    ) -> int | None:
        return max_decode_query_len(vllm_config)

    def build_tile_scheduler(self, num_decode_tokens: int):
        # The shared metadata schema needs the keys, but Triton needs no plan.
        return super().build_tile_scheduler(0)


class DeepseekV41TritonSparseBackend(DeepseekV4SparseMLABackend):
    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE_DSV41"

    @staticmethod
    def get_builder_cls() -> type[DeepseekV41TritonMetadataBuilder]:
        return DeepseekV41TritonMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability == DeviceCapability(9, 0)


class DeepseekV41TritonSWABackend(DeepseekSparseSWABackend):
    @staticmethod
    def get_builder_cls() -> type[DeepseekV41TritonSWAMetadataBuilder]:
        return DeepseekV41TritonSWAMetadataBuilder


class DeepseekV41TritonSparseAttention(DeepseekV4Attention):
    """Independent prefill and decode paths with native query-head buffers."""

    backend_cls = DeepseekV41TritonSparseBackend
    swa_backend_cls = DeepseekV41TritonSWABackend

    @classmethod
    def validate_config(cls, vllm_config: VllmConfig) -> None:
        if not current_platform.is_device_capability(90):
            raise ValueError("TRITON_MLA_SPARSE_DSV41 requires SM90")
        config = vllm_config.model_config.hf_text_config
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        heads = config.num_attention_heads
        if tp_size <= 0 or heads % tp_size or heads // tp_size not in (8, 16):
            raise ValueError("TRITON_MLA_SPARSE_DSV41 requires 8 or 16 local heads")
        if config.head_dim != 512 or config.qk_rope_head_dim != 64:
            raise ValueError("TRITON_MLA_SPARSE_DSV41 requires 512D heads and 64D RoPE")
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise ValueError("TRITON_MLA_SPARSE_DSV41 requires BF16 queries")
        if vllm_config.cache_config.cache_dtype not in ("auto", "fp8", "fp8_ds_mla"):
            raise ValueError("TRITON_MLA_SPARSE_DSV41 requires fp8_ds_mla KV cache")

    def __init__(self, vllm_config: VllmConfig, *args, **kwargs) -> None:
        super().__init__(vllm_config, *args, **kwargs)
        self._einsum_recipe, self._tma_aligned_scales = compute_fp8_einsum_recipe(
            self._o_proj_block_size
        )
        register_dsv41_o_proj_warmup(self)
        from vllm.models.deepseek_v41.common.ops.cache_utils import (
            _COMBINE_TOPK_SWA_INDICES_KERNEL,
        )
        from vllm.models.deepseek_v41.nvidia.ops.small_head_decode_warmup import (
            register_small_head_decode_warmup,
        )
        from vllm.models.deepseek_v41.nvidia.triton_fewhead_sparse_prefill import (
            _bf16_fewhead_sparse_fwd,
        )

        _COMBINE_TOPK_SWA_INDICES_KERNEL.register_warmup()
        _bf16_fewhead_sparse_fwd.register_warmup()
        register_small_head_decode_warmup(vllm_config, self.swa_cache_layer.block_size)
        from vllm.models.deepseek_v41.nvidia.paired_decode import PairedDecode

        self._paired_decode = PairedDecode(self, vllm_config)

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        if num_heads not in (8, 16):
            raise ValueError("TRITON_MLA_SPARSE_DSV41 requires 8 or 16 local heads")
        return num_heads

    def _o_proj(self, attn_out: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return dsv41_o_proj(self, attn_out, positions)

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        # Get SWA and indexer metadata from forward context
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if attn_metadata is None:
            paired_decode = getattr(self, "_paired_decode", None)
            if paired_decode is not None:
                paired_decode.reserve()
            # Warmup dummy run: no real metadata. Reserve the same bf16
            # gather workspace _forward_prefill would; the dequantize / topk
            # / sparse_fwd kernels are skipped this step.
            swa_only = self.compress_ratio == 0
            N = (
                0
                if swa_only
                else (self.max_model_len + self.compress_ratio - 1)
                // self.compress_ratio
            )
            M = N + self.window_size + self.max_num_batched_tokens
            if swa_only:
                top_k = 0
            else:
                assert self.topk_indices_buffer is not None
                top_k = self.topk_indices_buffer.shape[-1]
            combined_topk = round_up(top_k + self.window_size, 128)
            current_workspace_manager().get_simultaneous(
                ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
                ((self.max_num_batched_tokens, combined_topk), torch.int32),
                ((self.max_num_batched_tokens,), torch.int32),
            )
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        # Compressed-cache metadata lives on the kv-source layer's prefix;
        # consumers share that cache and its block table.
        flashmla_metadata = cast(
            DeepseekV4FlashMLAMetadata | None,
            attn_metadata.get(self.compressed_cache_prefix)
            if self.compressed_cache_prefix is not None
            else None,
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio == 0
        # SWA-only layers (compress_ratio == 0) don't have their own KV cache
        # allocation; consumers read the kv source's cache, which is bound on
        # the source layer.
        self_kv_cache = None if swa_only else self._compressed_kv_cache()
        swa_kv_cache = self.swa_cache_layer.kv_cache

        # Split prefill and decode
        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                positions=positions[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
            )
        if num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,  # None for SWA-only layers
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices = None
        topk_lens = None
        if not swa_only:
            # Local indices filled by the index-source layer's indexer.
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            assert self.topk_indices_buffer is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            global_indices, topk_lens = compute_global_topk_indices_and_lens(
                self.topk_indices_buffer[:num_decode_tokens],
                swa_metadata.token_to_req_indices,
                attn_metadata.block_table[:num_decodes],
                block_size,
                is_valid,
            )
            topk_indices = global_indices.view(num_decode_tokens, 1, -1)

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens

        paired_decode = getattr(self, "_paired_decode", None)
        if paired_decode is not None and paired_decode.run(
            get_forward_context(),
            q,
            self.swa_cache_layer.kv_cache,
            swa_indices,
            swa_lens,
            kv_cache if not swa_only else None,
            topk_indices,
            topk_lens,
            self.attn_sink,
            self.scale,
            output,
            swa_metadata.token_to_req_indices,
            swa_metadata.is_valid_token,
        ):
            return

        _run_decode(
            q=q,
            swa_cache=self.swa_cache_layer.kv_cache,
            swa_indices=swa_indices,
            swa_lens=swa_lens,
            extra_cache=kv_cache if not swa_only else None,
            extra_indices=topk_indices,
            extra_lens=topk_lens,
            attn_sink=self.attn_sink,
            sm_scale=self.scale,
            out=output,
            num_heads=self.n_local_heads,
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,  # None for SWA-only layers
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        swa_only = self.compress_ratio == 0

        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        # Use pre-computed prefill metadata.
        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        assert seq_lens is not None
        assert gather_lens is not None

        # Derive prefill-local token offsets from the full query_start_loc_cpu.
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        # Local indices filled by the index source; SWA-only layers pass
        # top_k=0 and never read them.
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[num_decode_tokens:]
        topk_indices = topk_indices[:num_prefill_tokens]
        top_k = 0 if swa_only else topk_indices.shape[-1]
        chunk_plan = swa_metadata.get_prefill_chunk_plan(
            compress_ratio=self.compress_ratio,
            prefill_chunk_size=self.PREFILL_CHUNK_SIZE,
            # v4.1: every cr>0 layer (including cr==1) gathers a full-length
            # compressed region; only cr==0 layers are SWA-only.
            has_compressed=not swa_only,
        )
        assert chunk_plan, "prefill chunk plan must be non-empty when num_prefills > 0"
        workspace_manager = current_workspace_manager()
        combined_topk = round_up(top_k + self.window_size, 128)
        for chunk_start, chunk_end, chunk_N, chunk_M in chunk_plan:
            chunk_size = chunk_end - chunk_start
            workspace = workspace_manager.get_simultaneous(
                ((chunk_size, chunk_M, q.shape[-1]), torch.bfloat16),
                ((self.max_num_batched_tokens, combined_topk), torch.int32),
                ((self.max_num_batched_tokens,), torch.int32),
            )
            kv, combined_indices_out, combined_lens_out = workspace
            if not swa_only:
                # Gather compressed KV
                assert attn_metadata is not None
                block_table = attn_metadata.block_table[num_decodes:]
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                )

            # Gather SWA KV
            swa_block_table = swa_metadata.block_table[num_decodes:]
            dequantize_and_gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=chunk_N,
            )

            # Combine the topk indices and SWA indices for gathered KV cache
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )
            combined_indices_out = combined_indices_out[: query_end - query_start]
            combined_lens_out = combined_lens_out[: query_end - query_start]

            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                chunk_M,
                chunk_N,
                out=(combined_indices_out, combined_lens_out),
            )
            run_fewhead_sparse_prefill(
                q[query_start:query_end],
                kv.view(-1, 1, q.shape[-1]),
                combined_indices,
                self.scale,
                attn_sink=self.attn_sink,
                topk_length=combined_lens,
                out=output[query_start:query_end],
                n_local_heads=self.n_local_heads,
            )
