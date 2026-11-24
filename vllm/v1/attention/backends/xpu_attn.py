# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention backend for Intel XPU, backed by vllm_xpu_kernels.

Batches are ordered decode-first, and a batch holding both decodes and prefills
runs as a decode-only and a prefill-only kernel call: the paged decode kernel
then picks KV splits for the exact decode count, and chunk prefill launches no
work-groups for decode rows.
"""

import copy
from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm import _custom_ops as ops
from vllm.config import VllmConfig, get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
from vllm.distributed.parallel_state import get_dcp_group
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import canonicalize_singleton_dim_strides
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    get_dcp_local_seq_lens,
    split_decodes_and_prefills,
)
from vllm.v1.attention.ops.dcp import cp_lse_ag_out_rs, dcp_a2a_lse_reduce
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    is_quantized_kv_cache,
)
from vllm.v1.worker.cp_utils import (
    run_split_fa2_dcp_context_attention,
    should_skip_dcp_context_attention,
    should_split_fa2_dcp_context_attention,
    split_dcp_context_queries,
)
from vllm.v1.worker.workspace import current_workspace_manager

if current_platform.is_xpu():
    from vllm._xpu_ops import xpu_ops

    xpu_flash_attn_varlen_func = xpu_ops.flash_attn_varlen_func
else:
    xpu_flash_attn_varlen_func = None  # type: ignore[assignment]

logger = init_logger(__name__)

# vllm_xpu_kernels implements the FlashAttention 2 interface.
_FA_VERSION = 2


class XpuAttentionBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    forward_includes_kv_cache_update: bool = False

    @classmethod
    def get_supported_kernel_block_sizes(
        cls, kv_cache_spec: KVCacheSpec | None = None
    ) -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        return max(default_block_size, 64)

    @staticmethod
    def get_name() -> str:
        return "XPU_ATTN"

    @staticmethod
    def get_impl_cls() -> type["XpuAttentionImpl"]:
        return XpuAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["XpuAttentionMetadataBuilder"]:
        return XpuAttentionMetadataBuilder

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size % 8 == 0 and head_size <= 256

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        return True


@dataclass
class XpuAttentionMetadata:
    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    causal: bool | torch.Tensor = True

    # Decode-first split: requests [0, num_decodes) have query lengths up to
    # the reorder threshold. The decode_*/prefill_* views are set only when
    # the batch holds both kinds, which then run as two kernel calls.
    num_decodes: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_prefill_tokens: int = 0
    decode_max_query_len: int = 1
    decode_query_start_loc: torch.Tensor | None = None
    decode_seq_lens: torch.Tensor | None = None
    decode_block_table: torch.Tensor | None = None
    prefill_query_start_loc: torch.Tensor | None = None
    prefill_seq_lens: torch.Tensor | None = None
    prefill_block_table: torch.Tensor | None = None

    # For GQA DCP.
    max_dcp_context_kv_len: int | None = None
    dcp_context_kv_lens: torch.Tensor | None = None
    # DCP context attention split into decode rows and context-bearing extend
    # rows; pure prefills do not attend to DCP context.
    num_dcp_decode_reqs: int = 0
    num_dcp_extend_reqs: int = 0
    num_dcp_decode_tokens: int = 0
    num_dcp_extend_tokens: int = 0


class XpuAttentionMetadataBuilder(AttentionMetadataBuilder[XpuAttentionMetadata]):
    # The decode/prefill split is chosen on the host every step, so only uniform
    # batches, which always run as one kernel call, are captured in full graphs.
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    supports_update_block_table: bool = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        try:
            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0
        self.cp_kv_cache_interleave_size = (
            vllm_config.parallel_config.cp_kv_cache_interleave_size
        )
        # Fused draft decode reuses one metadata object across draft steps; DCP
        # makes host-side choices per step that are not refreshed in place.
        self.supports_draft_decode_metadata_update = self.dcp_world_size == 1
        if self.dcp_world_size > 1:
            self._dcp_context_kv_lens = torch.zeros(
                vllm_config.scheduler_config.max_num_seqs,
                dtype=torch.int32,
                device=device,
            )
        # Decodes, including speculative verification rows, come first.
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> XpuAttentionMetadata:
        attn_metadata = XpuAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            causal=common_attn_metadata.causal,
        )
        if self.dcp_world_size > 1:
            self._build_dcp(attn_metadata, common_attn_metadata)
        else:
            self._split_decodes_and_prefills(attn_metadata, common_attn_metadata)
        return attn_metadata

    def _split_decodes_and_prefills(
        self,
        attn_metadata: XpuAttentionMetadata,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> None:
        assert self.reorder_batch_threshold is not None
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=self.reorder_batch_threshold
            )
        )
        attn_metadata.num_decodes = num_decodes
        attn_metadata.num_prefills = num_prefills
        attn_metadata.num_decode_tokens = num_decode_tokens
        attn_metadata.num_prefill_tokens = num_prefill_tokens
        if num_decodes == 0 or num_prefills == 0:
            return

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        decode_query_lens = (
            query_start_loc_cpu[1 : num_decodes + 1] - query_start_loc_cpu[:num_decodes]
        )
        if int(decode_query_lens.max()) > 1:
            # Adaptive verification may trim drafts on the device only, so bound
            # verification rows by the threshold, not by the host query lengths.
            attn_metadata.decode_max_query_len = min(
                self.reorder_batch_threshold, attn_metadata.max_query_len
            )

        # Sliced once per step here, not once per layer in forward().
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = attn_metadata.query_start_loc
        attn_metadata.decode_query_start_loc = query_start_loc[: num_decodes + 1]
        attn_metadata.decode_seq_lens = attn_metadata.seq_lens[:num_decodes]
        attn_metadata.decode_block_table = attn_metadata.block_table[:num_decodes]
        attn_metadata.prefill_query_start_loc = (
            query_start_loc[num_decodes : num_reqs + 1] - num_decode_tokens
        )
        attn_metadata.prefill_seq_lens = attn_metadata.seq_lens[num_decodes:num_reqs]
        attn_metadata.prefill_block_table = attn_metadata.block_table[
            num_decodes:num_reqs
        ]

    def _build_dcp(
        self,
        attn_metadata: XpuAttentionMetadata,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> None:
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        context_kv_lens = common_attn_metadata.seq_lens - query_lens
        local_context_kv_lens = get_dcp_local_seq_lens(
            context_kv_lens,
            self.dcp_world_size,
            self.dcp_rank,
            self.cp_kv_cache_interleave_size,
        )
        self._dcp_context_kv_lens[:num_reqs] = local_context_kv_lens
        self._dcp_context_kv_lens[num_reqs:] = 0
        attn_metadata.dcp_context_kv_lens = self._dcp_context_kv_lens[:num_reqs]

        skip_dcp_context_attention = False
        if common_attn_metadata.seq_lens_cpu_upper_bound is not None:
            query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
            query_lens_cpu = (
                query_start_loc_cpu[1 : num_reqs + 1] - query_start_loc_cpu[:num_reqs]
            )
            context_kv_lens_cpu = (
                common_attn_metadata.seq_lens_cpu_upper_bound[:num_reqs]
                - query_lens_cpu
            )
            skip_dcp_context_attention = should_skip_dcp_context_attention(
                context_kv_lens_cpu
            )

        if common_attn_metadata.max_query_len > 1:
            (
                attn_metadata.num_dcp_decode_reqs,
                attn_metadata.num_dcp_extend_reqs,
                attn_metadata.num_dcp_decode_tokens,
                attn_metadata.num_dcp_extend_tokens,
            ) = split_dcp_context_queries(
                common_attn_metadata.query_start_loc_cpu,
                common_attn_metadata.seq_lens_cpu_upper_bound,
                common_attn_metadata.max_query_len,
                common_attn_metadata.num_actual_tokens,
            )

        if skip_dcp_context_attention:
            attn_metadata.max_dcp_context_kv_len = 0
        else:
            # After DCP distribution a rank holds at most ceil(L / (N * I)) * I
            # context tokens, where L is max_seq_len, N is dcp_world_size and I
            # is cp_kv_cache_interleave_size. This avoids a GPU->CPU sync.
            num_partitions = self.dcp_world_size * self.cp_kv_cache_interleave_size
            attn_metadata.max_dcp_context_kv_len = (
                cdiv(common_attn_metadata.max_seq_len, num_partitions)
                * self.cp_kv_cache_interleave_size
            )

    def update_block_table(
        self,
        metadata: XpuAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> XpuAttentionMetadata:
        new_metadata = copy.copy(metadata)
        new_metadata.block_table = blk_table
        new_metadata.slot_mapping = slot_mapping
        if metadata.decode_block_table is not None:
            num_decodes = metadata.num_decodes
            new_metadata.decode_block_table = blk_table[:num_decodes]
            new_metadata.prefill_block_table = blk_table[
                num_decodes : num_decodes + metadata.num_prefills
            ]
        return new_metadata

    def update_draft_decode_metadata(self, metadata: XpuAttentionMetadata) -> None:
        # Draft decode batches are uniform single-token batches: one kernel
        # call with no host-side state to refresh between draft steps.
        return


class XpuAttentionImpl(AttentionImpl):
    """XPU_ATTN implementation on vllm_xpu_kernels.

    A batch with both decodes and prefills runs as a decode-only and a
    prefill-only `_paged_attention()` call. DCP attention runs the whole batch
    at once.
    """

    can_return_lse_for_decode: bool = True
    supports_dcp: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        # The kernels apply neither ALiBi nor a logits soft cap: fail instead
        # of silently computing other attention.
        if alibi_slopes is not None:
            raise NotImplementedError("XPU_ATTN does not support ALiBi.")
        if logits_soft_cap:
            raise NotImplementedError("XPU_ATTN does not support a logits soft cap.")
        if sinks is not None:
            assert sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                "heads in the layer"
            )
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.alibi_slopes = None
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type == AttentionType.ENCODER_ONLY:
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = 0
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.attn_type = attn_type
        self.sinks = sinks
        # Queries are then quantized in the attention layer.
        self.supports_quant_query_input = (
            current_platform.is_xpu() and current_platform.support_fp8_query()
        )
        # Kernel window per causal flag: non-causal attention makes a causal
        # window (w, 0) symmetric (w, w), so bidirectional queries attend in
        # both directions.
        left, right = self.sliding_window
        self._window_size = {
            True: [left, right],
            False: [left, left] if left >= 0 and right == 0 else [left, right],
        }

        vllm_config = get_current_vllm_config_or_none()
        dcp_a2a = (
            vllm_config is not None
            and vllm_config.parallel_config.decode_context_parallel_size > 1
            and vllm_config.parallel_config.dcp_comm_backend == "a2a"
        )
        self.dcp_combine = dcp_a2a_lse_reduce if dcp_a2a else cp_lse_ag_out_rs
        self._dcp_dtype: torch.dtype | None = None
        self._dcp_max_num_tokens: int = 0
        if vllm_config is not None and self.dcp_world_size > 1:
            self._dcp_dtype = vllm_config.model_config.dtype
            self._dcp_max_num_tokens = (
                vllm_config.scheduler_config.max_num_batched_tokens
            )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: XpuAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with the XPU attention kernels.

        Args:
            layer: Attention layer that owns the KV scales.
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [num_blocks, num_kv_heads, block_size, 2 * head_size]
            attn_metadata: Metadata for attention.
            output: Destination attention output.
            output_scale: Optional output quantization scale.
            output_block_scale: Optional blockwise output quantization scale.

        Returns:
            shape = [num_tokens, num_heads * head_size]

        """
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for "
                f"{type(self).__name__}"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        num_actual_tokens = attn_metadata.num_actual_tokens

        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # Encoder attention reads Q, K and V directly, without a KV cache.
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        # (B, H, N, 2*D) -> ((B, N, H, D), (B, N, H, D))
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        if is_quantized_kv_cache(self.kv_cache_dtype):
            key_cache = key_cache.view(current_platform.fp8_dtype())
            value_cache = value_cache.view(current_platform.fp8_dtype())

        # key and value keep their padding: only DCP reads them.
        self._forward_decoder_attention(
            layer,
            query[:num_actual_tokens],
            key,
            value,
            key_cache,
            value_cache,
            output[:num_actual_tokens],
            attn_metadata,
        )
        return output

    def _forward_decoder_attention(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: XpuAttentionMetadata,
    ) -> None:
        if isinstance(attn_metadata.causal, torch.Tensor):
            raise NotImplementedError(
                "Per-request causal masks are not supported by XPU_ATTN."
            )
        # Fix degenerate strides on size-1 dims (e.g. num_kv_heads=1 with TP).
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)

        if self.dcp_world_size > 1:
            num_actual_tokens = attn_metadata.num_actual_tokens
            self._forward_with_dcp(
                layer,
                query,
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                key_cache,
                value_cache,
                output,
                attn_metadata,
            )
            return

        if attn_metadata.decode_query_start_loc is None:
            self._paged_attention(
                layer,
                query,
                key_cache,
                value_cache,
                output,
                cu_seqlens_q=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                seq_lens=attn_metadata.seq_lens,
                block_table=attn_metadata.block_table,
                max_seqlen_k=attn_metadata.max_seq_len,
                causal=attn_metadata.causal,
                has_decodes=attn_metadata.num_prefills == 0,
            )
            return

        num_decode_tokens = attn_metadata.num_decode_tokens
        assert attn_metadata.decode_seq_lens is not None
        assert attn_metadata.decode_block_table is not None
        self._paged_attention(
            layer,
            query[:num_decode_tokens],
            key_cache,
            value_cache,
            output[:num_decode_tokens],
            cu_seqlens_q=attn_metadata.decode_query_start_loc,
            max_seqlen_q=attn_metadata.decode_max_query_len,
            seq_lens=attn_metadata.decode_seq_lens,
            block_table=attn_metadata.decode_block_table,
            max_seqlen_k=attn_metadata.max_seq_len,
            causal=attn_metadata.causal,
            has_decodes=True,
        )
        assert attn_metadata.prefill_query_start_loc is not None
        assert attn_metadata.prefill_seq_lens is not None
        assert attn_metadata.prefill_block_table is not None
        self._paged_attention(
            layer,
            query[num_decode_tokens:],
            key_cache,
            value_cache,
            output[num_decode_tokens:],
            cu_seqlens_q=attn_metadata.prefill_query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            seq_lens=attn_metadata.prefill_seq_lens,
            block_table=attn_metadata.prefill_block_table,
            max_seqlen_k=attn_metadata.max_seq_len,
            causal=attn_metadata.causal,
            has_decodes=False,
        )

    def _descales(
        self, layer: torch.nn.Module, num_seqs: int
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """Per-tensor Q/K/V descales, expanded to (num_seqs, num_kv_heads)."""
        shape = (num_seqs, self.num_kv_heads)
        q_descale = (
            layer._q_scale.expand(shape) if self.supports_quant_query_input else None
        )
        return q_descale, layer._k_scale.expand(shape), layer._v_scale.expand(shape)

    def _paged_attention(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        *,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        max_seqlen_k: int,
        causal: bool | torch.Tensor,
        has_decodes: bool,
    ) -> None:
        """max_seqlen_q == 1 runs the paged decode kernel. Decode rows of more
        than one token (speculative verification) run as a mixed batch, so that
        the single-token rows among them still take the paged decode kernel (or
        all take the split-K decode kernel when uniform). A prefill-only batch
        runs chunk prefill alone."""
        assert isinstance(causal, bool)
        q_descale, k_descale, v_descale = self._descales(
            layer, cu_seqlens_q.shape[0] - 1
        )
        xpu_flash_attn_varlen_func(
            q=query,
            k=key_cache,
            v=value_cache,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seq_lens,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=causal,
            window_size=self._window_size[causal],
            block_table=block_table,
            softcap=self.logits_soft_cap,
            fa_version=_FA_VERSION,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            s_aux=self.sinks,
            is_mix_batch=has_decodes and max_seqlen_q > 1,
        )

    def _forward_with_dcp(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: XpuAttentionMetadata,
    ) -> None:
        cu_seqlens_q = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len
        block_table = attn_metadata.block_table
        num_reqs = cu_seqlens_q.shape[0] - 1
        q_descale, k_descale, v_descale = self._descales(layer, num_reqs)
        sliding_window_size = list(self.sliding_window)

        query = query.contiguous()
        if attn_metadata.max_dcp_context_kv_len == 0:
            xpu_flash_attn_varlen_func(
                q=query,
                k=key,
                v=value,
                out=output,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                cu_seqlens_k=cu_seqlens_q,
                max_seqlen_k=max_seqlen_q,
                softmax_scale=self.scale,
                causal=attn_metadata.causal,
                window_size=sliding_window_size,
                softcap=self.logits_soft_cap,
                return_softmax_lse=True,
                fa_version=_FA_VERSION,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
            )
            return

        assert attn_metadata.dcp_context_kv_lens is not None
        assert attn_metadata.max_dcp_context_kv_len is not None
        query_across_dcp = get_dcp_group().all_gather(query, dim=1)
        n = query_across_dcp.shape[0]
        split_dcp_context = should_split_fa2_dcp_context_attention(
            _FA_VERSION,
            max_seqlen_q,
            num_reqs,
            attn_metadata.num_dcp_decode_reqs,
            attn_metadata.num_dcp_extend_reqs,
        )
        dcp_context_out_spec = (
            (
                max(n, self._dcp_max_num_tokens),
                self.num_heads * self.dcp_world_size,
                self.head_size,
            ),
            self._dcp_dtype,
        )
        (dcp_context_out_workspace,) = current_workspace_manager().get_simultaneous(
            dcp_context_out_spec,
        )
        dcp_context_out = dcp_context_out_workspace[:n]

        if split_dcp_context:
            # The paged-varlen kernel can fail on DCP mixed batches that submit
            # decode, context-bearing extend and pure prefill rows together.
            context_attn_out, context_lse = run_split_fa2_dcp_context_attention(
                xpu_flash_attn_varlen_func,
                query_across_dcp,
                key_cache,
                value_cache,
                dcp_context_out,
                cu_seqlens_q,
                max_seqlen_q,
                attn_metadata.dcp_context_kv_lens,
                attn_metadata.max_dcp_context_kv_len,
                self.scale,
                self.alibi_slopes,
                sliding_window_size,
                block_table,
                self.logits_soft_cap,
                _FA_VERSION,
                q_descale,
                k_descale,
                v_descale,
                0,
                self.num_heads,
                self.dcp_world_size,
                attn_metadata.num_dcp_decode_reqs,
                attn_metadata.num_dcp_extend_reqs,
                attn_metadata.num_dcp_decode_tokens,
                attn_metadata.num_dcp_extend_tokens,
            )
        else:
            context_attn_out, context_lse = xpu_flash_attn_varlen_func(
                q=query_across_dcp,
                k=key_cache,
                v=value_cache,
                out=dcp_context_out,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                seqused_k=attn_metadata.dcp_context_kv_lens,
                max_seqlen_k=attn_metadata.max_dcp_context_kv_len,
                softmax_scale=self.scale,
                causal=False,
                window_size=sliding_window_size,
                block_table=block_table,
                softcap=self.logits_soft_cap,
                return_softmax_lse=True,
                fa_version=_FA_VERSION,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
            )
        # The kernels return the LSE as [H, B]; the DCP combine wants [B, H].
        context_attn_out_cor, context_lse_cor = self.dcp_combine(
            context_attn_out,
            context_lse.transpose(0, 1),
            get_dcp_group(),
            return_lse=True,
        )
        context_lse_cor = context_lse_cor.transpose(0, 1).contiguous()

        query_attn_out, query_lse = xpu_flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_k=cu_seqlens_q,
            max_seqlen_k=max_seqlen_q,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            window_size=sliding_window_size,
            softcap=self.logits_soft_cap,
            return_softmax_lse=True,
            fa_version=_FA_VERSION,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        assert context_attn_out_cor.shape == query_attn_out.shape
        assert context_lse_cor.shape == query_lse.shape
        merge_attn_states(
            output,
            context_attn_out_cor,
            context_lse_cor,
            query_attn_out,
            query_lse,
        )

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: XpuAttentionMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "quantization is not supported for encoder attention"
            )
        q_descale, k_descale, v_descale = self._descales(
            layer, attn_metadata.query_start_loc.shape[0] - 1
        )
        xpu_flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            out=output,
            cu_seqlens_q=attn_metadata.query_start_loc,
            cu_seqlens_k=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            max_seqlen_k=attn_metadata.max_query_len,
            softmax_scale=self.scale,
            causal=False,
            window_size=list(self.sliding_window),
            softcap=self.logits_soft_cap,
            fa_version=_FA_VERSION,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            s_aux=self.sinks,
        )
        return output

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return

        # (B, H, N, 2*D) -> ((B, N, H, D), (B, N, H, D))
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        # key and value may be padded; reshape_and_cache_flash takes the number
        # of tokens from slot_mapping.
        ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )
