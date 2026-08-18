# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention.mla_attention import MLACommonPrefillMetadata
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
    SparseMLACommonMetadataBuilder,
)
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionMetadata,
    CommonAttentionMetadata,
    MLAAttentionImpl,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import flash_attn_supports_mla
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.backends.utils import split_prefill_chunks
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.workspace import current_workspace_manager
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

FP8_KV_CACHE_DTYPES = ("fp8", "fp8_e4m3")


class FlashAttnMLASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [64]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_MLA_SPARSE"

    @staticmethod
    def get_builder_cls() -> type["FlashAttnMLASparseMetadataBuilder"]:
        return FlashAttnMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[MLAAttentionImpl[Any]]:
        return FlashAttnMLASparseImpl

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 9

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if kv_cache_dtype not in (
            None,
            "auto",
            "float16",
            "bfloat16",
            *FP8_KV_CACHE_DTYPES,
        ):
            return "FlashAttention MLA Sparse supports FP16/BF16 and e4m3 FP8 KV cache"

        if not flash_attn_supports_mla():
            return "FlashAttention MLA not supported on this device"

        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None and vllm_config.model_config is not None:
            if vllm_config.parallel_config.decode_context_parallel_size > 1:
                return "FlashAttention MLA Sparse does not support DCP for now"

            hf_config = vllm_config.model_config.hf_config
            if not hasattr(hf_config, "index_topk"):
                return "FlashAttention MLA Sparse requires model with index_topk"
        return None

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)


@dataclass
class FlashAttnMLASparseMetadata(AttentionMetadata):
    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor

    block_table: torch.Tensor
    req_id_per_token: torch.Tensor
    seq_lens: torch.Tensor
    block_size: int = 64
    topk_tokens: int = 2048
    num_decodes: int = 0
    num_prefills: int = 0
    num_decode_tokens: int = 0
    prefill_max_seq_len: int = 0
    prefill: MLACommonPrefillMetadata | None = None
    cp_kv_cache_interleave_size: int = 1

    @dataclass
    class FP8Prefill:
        """Routing for MQA prefill tokens under an e4m3 KV cache.

        Each chunk upconverts its requests' resident context ([0, seq_len))
        into a bf16 workspace once, and the chunk's top-k indices are remapped
        to workspace offsets, so the varlen kernel reads the workspace
        directly and nothing scales with (prefill tokens x top-k).
        """

        @dataclass
        class Chunk:
            tokens_slice: slice
            block_table: torch.Tensor  # int32 [num_reqs, max_blocks]
            seq_lens: torch.Tensor  # int32 [num_reqs], device
            tot_seqlen: int
            max_seq_len: int

        # int32 [num_tokens]: -1 for decode tokens, prefill request index else
        request_ids: torch.Tensor
        # int32 [num_prefills]: cumulative seq lens, rebased per chunk to the
        # chunk's first request, so indexing by global request id yields the
        # chunk-local workspace offset.
        workspace_starts: torch.Tensor
        chunks: list[Chunk]

    fp8_prefill: FP8Prefill | None = None


class FlashAttnMLASparseMetadataBuilder(
    SparseMLACommonMetadataBuilder[FlashAttnMLASparseMetadata]
):
    metadata_cls = FlashAttnMLASparseMetadata
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        num_q_heads = self.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        threshold = {16: 128, 32: 128, 64: 256, 128: 256}.get(num_q_heads, 256)
        self._init_reorder_batch_threshold(threshold, supports_spec_as_decode=True)

        self.use_fp8_kv_cache = is_quantized_kv_cache(
            vllm_config.cache_config.cache_dtype
        ) and vllm_config.cache_config.cache_dtype not in ("fp8_ds_mla",)
        if self.use_fp8_kv_cache:
            # Must match the impl's prefill_bf16_workspace row count.
            self.fp8_prefill_workspace_rows = 2 * vllm_config.model_config.max_model_len
            # Dense MHA reads the paged cache directly; only the MQA path can
            # bridge an e4m3 cache, so keep prefill tokens on it.
            self.skip_dense_mha_prefill = True

    def _build_fp8_prefill(
        self,
        common_attn_metadata: "CommonAttentionMetadata",
        metadata: "FlashAttnMLASparseMetadata",
    ) -> FlashAttnMLASparseMetadata.FP8Prefill | None:
        num_decodes = metadata.num_decodes
        num_prefills = metadata.num_prefills
        if num_prefills == 0:
            return None

        seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
        assert seq_lens_cpu is not None
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        prefill_seq_lens_cpu = seq_lens_cpu[num_decodes:].to(torch.int32)
        num_tokens = common_attn_metadata.num_actual_tokens

        # -1 for decode tokens, prefill request index (0, 1, ...) else
        request_ids = torch.full(
            (num_tokens,), -1, dtype=torch.int32, device=self.device
        )
        workspace_starts_cpu = torch.zeros(
            num_prefills,
            dtype=torch.int32,
            pin_memory=self.device.type == "cuda",
        )
        workspace_starts_cpu[1:] = torch.cumsum(prefill_seq_lens_cpu[:-1], dim=0)
        workspace_starts = torch.empty(
            num_prefills, dtype=torch.int32, device=self.device
        )

        for req_idx in range(num_prefills):
            global_req_idx = num_decodes + req_idx
            start = query_start_loc_cpu[global_req_idx]
            end = query_start_loc_cpu[global_req_idx + 1]
            request_ids[start:end] = req_idx

        chunk_bounds = split_prefill_chunks(
            prefill_seq_lens_cpu, self.fp8_prefill_workspace_rows
        )
        chunks = []
        for chunk_start, chunk_end in chunk_bounds:
            # Re-base each chunk's starts to 0 so prefill indices come out
            # chunk-local. Example: seq_lens=[10,15,20,5], chunks=[[0,2],[2,4]]
            # -> starts [0,10, 0,20].
            offset = workspace_starts_cpu[chunk_start].item()
            workspace_starts_cpu[chunk_start:chunk_end] -= offset
            chunk_lens = prefill_seq_lens_cpu[chunk_start:chunk_end]
            token_start = query_start_loc_cpu[num_decodes + chunk_start].item()
            token_end = query_start_loc_cpu[num_decodes + chunk_end].item()
            chunks.append(
                FlashAttnMLASparseMetadata.FP8Prefill.Chunk(
                    tokens_slice=slice(token_start, token_end),
                    block_table=common_attn_metadata.block_table_tensor[
                        num_decodes + chunk_start : num_decodes + chunk_end
                    ],
                    seq_lens=chunk_lens.to(self.device, non_blocking=True),
                    tot_seqlen=int(chunk_lens.sum()),
                    max_seq_len=int(chunk_lens.max()),
                )
            )

        workspace_starts.copy_(workspace_starts_cpu, non_blocking=True)
        return FlashAttnMLASparseMetadata.FP8Prefill(
            request_ids=request_ids,
            workspace_starts=workspace_starts,
            chunks=chunks,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: "CommonAttentionMetadata",
        fast_build: bool = False,
    ) -> FlashAttnMLASparseMetadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        if self.use_fp8_kv_cache:
            metadata.fp8_prefill = self._build_fp8_prefill(
                common_attn_metadata, metadata
            )
        return metadata


class FlashAttnMLASparseImpl(SparseMLACommonImpl[FlashAttnMLASparseMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: Any | None = None,
        **mla_args: Any,
    ) -> None:
        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashAttnMLASparseImpl does not support alibi, sliding window, "
                "or logits soft cap."
            )
        if kv_cache_dtype not in ("auto", "float16", "bfloat16", *FP8_KV_CACHE_DTYPES):
            raise NotImplementedError(
                "FlashAttnMLASparseImpl supports FP16/BF16 and e4m3 FP8 KV cache."
            )

        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            indexer=indexer,
            topk_indices_buffer=topk_indices_buffer,
            **mla_args,
        )
        assert self.topk_indices_buffer is not None, (
            "Indexer or topk_indices_buffer required for sparse MLA"
        )
        self.supports_quant_query_input = False

        self.use_fp8_kv_cache = self.kv_cache_dtype in FP8_KV_CACHE_DTYPES
        self.prefill_bf16_workspace: torch.Tensor | None = None
        if self.use_fp8_kv_cache:
            # bf16 upconversion target for prefill chunks; 2x max_model_len
            # rows bounds a single max-length request plus concurrent
            # prefills (matches the builder's chunking limit).
            vllm_config = get_current_vllm_config()
            rows = 2 * vllm_config.model_config.max_model_len
            (self.prefill_bf16_workspace,) = (
                current_workspace_manager().get_simultaneous(
                    ((rows, head_size), torch.bfloat16),
                )
            )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashAttnMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not isinstance(q, tuple):
            raise NotImplementedError(
                "FlashAttnMLASparseImpl expects split (q_nope, q_rope) input."
            )
        q_nope, q_rope = q
        num_actual_toks = q_rope.shape[0]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        if self.use_fp8_kv_cache:
            attn_out = self._forward_fp8_kv(
                q_nope,
                q_rope,
                kv_c_and_k_pe_cache,
                topk_indices,
                attn_metadata,
                layer,
            )
            return attn_out, None

        topk_indices, valid_counts = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[:num_actual_toks],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            return_valid_counts=True,
        )

        cu_seqlens_q = torch.arange(
            0, num_actual_toks + 1, dtype=torch.int32, device=q_rope.device
        )
        kv_cache = kv_c_and_k_pe_cache.view(
            -1, attn_metadata.block_size, self.head_size
        )
        v_cache = kv_cache[:, :, : self.kv_lora_rank].view(-1, 1, 1, self.kv_lora_rank)

        # When qk_rope_head_dim=0, use FA3's only_qv mode: attention scores come
        # from q_v (q_nope) @ V instead of q_rope @ k_rope. Build dummy q/k of
        # headdim 64 (the kernel ignores their values under only_qv=True) so we
        # never materialize a 0-element rope tensor. Mirrors the dense MLA path
        # (flashattn_mla.py:367-414).
        only_qv = self.qk_rope_head_dim == 0
        if only_qv:
            dummy_headdim = 64
            q_rope = torch.empty(
                *q_rope.shape[:-1],
                dummy_headdim,
                dtype=q_rope.dtype,
                device=q_rope.device,
            )
            k_cache = torch.empty(
                *v_cache.shape[:-1],
                dummy_headdim,
                dtype=v_cache.dtype,
                device=v_cache.device,
            )
            softmax_scale = self.kv_lora_rank ** (-0.5)
        else:
            k_cache = kv_cache[:, :, self.kv_lora_rank :].view(
                -1, 1, 1, self.qk_rope_head_dim
            )
            softmax_scale = self.scale

        out = flash_attn_varlen_func(
            q=q_rope,
            k=k_cache,
            v=v_cache,
            q_v=q_nope,
            max_seqlen_q=1,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=topk_indices.shape[1],
            seqused_k=valid_counts,
            block_table=topk_indices,
            softmax_scale=softmax_scale,
            causal=True,
            fa_version=3,
            only_qv=only_qv,
        )
        return out, None

    def _forward_fp8_kv(
        self,
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: FlashAttnMLASparseMetadata,
        layer: AttentionLayer,
    ) -> torch.Tensor:
        """Dequant bridge over the e4m3 KV cache for the MQA path.

        The kernel only consumes bf16, so selected cache rows are materialized
        as bf16 first: decode tokens gather their top-k rows into a compact
        workspace, and each prefill chunk upconverts its requests' resident
        context once (never per top-k row).
        """
        k_scale_float = getattr(layer, "_k_scale_float", None)
        k_scale = 1.0 if k_scale_float is None else k_scale_float
        out = q_nope.new_empty((q_nope.shape[0], self.num_heads, self.kv_lora_rank))

        num_decode_tokens = attn_metadata.num_decode_tokens
        if num_decode_tokens > 0:
            out[:num_decode_tokens] = self._fp8_decode(
                q_nope[:num_decode_tokens],
                q_rope[:num_decode_tokens],
                kv_c_and_k_pe_cache,
                topk_indices[:num_decode_tokens],
                attn_metadata,
                k_scale,
            )

        fp8_prefill = attn_metadata.fp8_prefill
        num_prefill_tokens = 0
        if fp8_prefill is not None:
            for chunk in fp8_prefill.chunks:
                num_prefill_tokens += chunk.tokens_slice.stop - chunk.tokens_slice.start
                out[chunk.tokens_slice] = self._fp8_prefill_chunk(
                    q_nope[chunk.tokens_slice],
                    q_rope[chunk.tokens_slice],
                    kv_c_and_k_pe_cache,
                    topk_indices[chunk.tokens_slice],
                    chunk,
                    fp8_prefill,
                    attn_metadata,
                    k_scale,
                )
        assert num_decode_tokens + num_prefill_tokens == q_nope.shape[0], (
            "FP8 MQA path must cover every token: dense-MHA prefill is "
            "disabled for quantized caches on this backend."
        )
        return out

    def _fp8_decode(
        self,
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: FlashAttnMLASparseMetadata,
        k_scale: float,
    ) -> torch.Tensor:
        topk_slots, valid_counts = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[: q_nope.shape[0]],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            return_valid_counts=True,
        )
        num_tokens, num_topk = topk_slots.shape
        # -1 padding rows clamp onto slot 0 inside the gather; seqused_k
        # masks them.
        rows = _gather_dequant_rows(
            kv_c_and_k_pe_cache, topk_slots.reshape(-1), k_scale
        )
        # Point each token's row at its own gathered workspace segment.
        block_table = torch.arange(
            num_tokens * num_topk, dtype=torch.int32, device=rows.device
        ).view(num_tokens, num_topk)
        return self._fa_varlen_from_rows(
            q_nope, q_rope, rows, block_table, valid_counts
        )

    def _fp8_prefill_chunk(
        self,
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        chunk: FlashAttnMLASparseMetadata.FP8Prefill.Chunk,
        fp8_prefill: FlashAttnMLASparseMetadata.FP8Prefill,
        attn_metadata: FlashAttnMLASparseMetadata,
        k_scale: float,
    ) -> torch.Tensor:
        assert self.prefill_bf16_workspace is not None
        workspace = self.prefill_bf16_workspace[: chunk.tot_seqlen]
        _upconvert_chunk_context(
            kv_c_and_k_pe_cache,
            chunk.block_table,
            chunk.seq_lens,
            chunk.max_seq_len,
            workspace,
            k_scale,
        )
        chunk_request_ids = fp8_prefill.request_ids[chunk.tokens_slice]
        topk_offsets, valid_counts = triton_convert_req_index_to_global_index(
            chunk_request_ids,
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            HAS_PREFILL_WORKSPACE=True,
            prefill_workspace_request_ids=chunk_request_ids,
            prefill_workspace_starts=fp8_prefill.workspace_starts,
            return_valid_counts=True,
        )
        return self._fa_varlen_from_rows(
            q_nope,
            q_rope,
            workspace,
            # -1 padding rows clamp onto workspace row 0; seqused_k masks them.
            topk_offsets.clamp(min=0),
            valid_counts,
        )

    def _fa_varlen_from_rows(
        self,
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        kv_rows: torch.Tensor,
        block_table: torch.Tensor,
        valid_counts: torch.Tensor,
    ) -> torch.Tensor:
        """Run the sparse varlen kernel against materialized bf16 rows
        ([num_rows, head_size]); identical contract to the bf16 cache path."""
        num_tokens = q_nope.shape[0]
        cu_seqlens_q = torch.arange(
            0, num_tokens + 1, dtype=torch.int32, device=q_nope.device
        )
        v_rows = kv_rows[:, : self.kv_lora_rank].reshape(-1, 1, 1, self.kv_lora_rank)
        only_qv = self.qk_rope_head_dim == 0
        if only_qv:
            dummy_headdim = 64
            q = torch.empty(
                *q_rope.shape[:-1],
                dummy_headdim,
                dtype=q_rope.dtype,
                device=q_rope.device,
            )
            k_cache = torch.empty(
                *v_rows.shape[:-1],
                dummy_headdim,
                dtype=v_rows.dtype,
                device=v_rows.device,
            )
            softmax_scale = self.kv_lora_rank ** (-0.5)
        else:
            q = q_rope
            k_cache = kv_rows[:, self.kv_lora_rank :].reshape(
                -1, 1, 1, self.qk_rope_head_dim
            )
            softmax_scale = self.scale
        return flash_attn_varlen_func(
            q=q,
            k=k_cache,
            v=v_rows,
            q_v=q_nope,
            max_seqlen_q=1,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=block_table.shape[1],
            seqused_k=valid_counts,
            block_table=block_table,
            softmax_scale=softmax_scale,
            causal=True,
            fa_version=3,
            only_qv=only_qv,
        )


_GATHER_BLOCK_R = 8
_GATHER_BLOCK_H = 128


@triton.jit
def _gather_dequant_e4m3_kernel(
    cache_ptr,
    idx_ptr,
    out_ptr,
    k_scale,
    num_slots,
    num_rows,
    HEAD: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Gather rows from a flat e4m3 cache (uint8 bytes) and dequantize to
    bf16 with a per-tensor scale, in one pass.

    The e4m3 -> f32 decode uses exact integer arithmetic only: e4m3 has a
    4-bit mantissa, so ``(8 + man) * 2**(exp - 10)`` (or ``man * 2**-9``
    subnormal) is exact in f32; the power of two comes from bitcasting the
    exponent field. Rounding therefore happens exactly once, at the f32 ->
    bf16 store, matching the unfused ``.to(bfloat16) [* scale]`` chain
    bit-for-bit.
    """
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R).to(tl.int64)
    row_ok = rows < num_rows
    # Slot ids are in-range or -1 padding (masked by seqused_k); the clamp
    # keeps every load in-bounds regardless.
    idx = tl.load(idx_ptr + rows, mask=row_ok, other=0).to(tl.int64)
    idx = tl.minimum(tl.maximum(idx, 0), num_slots - 1)
    cols = tl.arange(0, BLOCK_H).to(tl.int64)
    for h in tl.static_range(0, HEAD, BLOCK_H):
        b = tl.load(
            cache_ptr + idx[:, None] * HEAD + h + cols[None, :],
            mask=row_ok[:, None],
            other=0,
        ).to(tl.int32)
        sign = b >> 7
        exp = (b >> 3) & 0xF
        man = b & 0x7
        # e4m3fn: bias 7; exp == 0 is subnormal; exp == 15 && man == 7 is NaN.
        shift = tl.where(exp == 0, -9, exp - 10)
        mant = tl.where(exp == 0, man, man + 8)
        pow2 = ((shift + 127) << 23).to(tl.float32, bitcast=True)
        val = mant.to(tl.float32) * pow2
        val = tl.where((exp == 15) & (man == 7), float("nan"), val)
        val = tl.where(sign == 1, -val, val)
        val = val * k_scale
        # f32 -> bf16 with explicit round-to-nearest-even (the same bit trick
        # as c10's round_to_nearest_even): triton's cast is RTNE on device
        # but the CPU interpreter truncates, so integer math keeps both
        # bit-exact vs torch.
        u32 = val.to(tl.int32, bitcast=True)
        rounded = u32 + 0x7FFF + ((u32 >> 16) & 1)
        res = (rounded >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)
        tl.store(
            out_ptr + rows[:, None] * HEAD + h + cols[None, :],
            res,
            mask=row_ok[:, None],
        )


def _gather_dequant_rows(
    kv_c_and_k_pe_cache: torch.Tensor,
    slots: torch.Tensor,
    k_scale: float,
) -> torch.Tensor:
    """Gather cache rows by flat slot id and dequantize e4m3 to bf16.

    Fused single pass: read uint8 bytes, decode e4m3, apply the scale, write
    bf16 -- the unfused index_select + cast + mul chain moves each selected
    row through HBM three extra times.
    """
    head = kv_c_and_k_pe_cache.shape[-1]
    assert head % _GATHER_BLOCK_H == 0
    flat = kv_c_and_k_pe_cache.view(torch.uint8).reshape(-1, head)
    num_rows = slots.shape[0]
    out = torch.empty((num_rows, head), dtype=torch.bfloat16, device=flat.device)
    if num_rows == 0:
        return out
    _gather_dequant_e4m3_kernel[(triton.cdiv(num_rows, _GATHER_BLOCK_R),)](
        flat,
        slots,
        out,
        k_scale,
        flat.shape[0],
        num_rows,
        HEAD=head,
        BLOCK_R=_GATHER_BLOCK_R,
        BLOCK_H=_GATHER_BLOCK_H,
        num_warps=4,
    )
    return out


def _upconvert_chunk_context(
    kv_c_and_k_pe_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    dst: torch.Tensor,
    k_scale: float,
) -> None:
    """Concatenate each request's [0, seq_len) cache rows, dequantized, into
    dst; the row layout matches cumulative sequence lengths."""
    num_reqs = seq_lens.shape[0]
    if num_reqs == 0 or max_seq_len == 0:
        return
    device = dst.device
    block_size = kv_c_and_k_pe_cache.shape[1]
    pos = torch.arange(max_seq_len, device=device, dtype=torch.int64)
    block_col = torch.div(pos, block_size, rounding_mode="floor").clamp(
        max=block_table.shape[1] - 1
    )
    blocks = block_table.to(torch.int64).index_select(1, block_col).mul_(block_size) + (
        pos % block_size
    )
    valid = pos.unsqueeze(0) < seq_lens.to(torch.int64).unsqueeze(1)
    rows = _gather_dequant_rows(kv_c_and_k_pe_cache, blocks[valid].reshape(-1), k_scale)
    dst.copy_(rows)
