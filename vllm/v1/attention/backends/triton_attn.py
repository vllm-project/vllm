# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""High-Performance Triton-only Attention layer."""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionType,
    MultipleOf,
)
from vllm.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)
from vllm.attention.ops.triton_unified_attention import unified_attention
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.distributed import (divide, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size, get_starscream_parallel_world_size,
                              split_tensor_along_last_dim,
                              tensor_model_parallel_all_gather, cpx_model_parallel_all_gather, cpx_model_parallel_all_reduce,
                              tensor_model_parallel_all_reduce)
import numpy as np

logger = init_logger(__name__)

from typing import List, Tuple

@torch.jit.script
def slice_and_stitch_three_decode(
    t1: torch.Tensor,
    t2: torch.Tensor,
    t3: torch.Tensor,
    N: int,
    d_idx: int,
    starscream_rank: int,
    cpx_size: int,
    has_A: bool,
    prefill_decode_match: bool,
    num_batches: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:

    slice_idx = 0

    if has_A and starscream_rank == d_idx:
        out1 = torch.narrow(t1, 0, 0, 1)
        out2 = torch.narrow(t2, 0, 0, 1)
        out3 = torch.narrow(t3, 0, 0, 1)

    elif prefill_decode_match:
        start = has_A
        length = num_batches
        out1 = torch.narrow(t1, 0, start, num_batches)
        out2 = torch.narrow(t2, 0, start, num_batches)
        out3 = torch.narrow(t3, 0, start, num_batches)

    else:
        out1 = torch.empty_like(t1)
        out2 = torch.empty_like(t2)
        out3 = torch.empty_like(t3)

    return out1, out2, out3, slice_idx

@torch.jit.script
def slice_and_stitch_three(
    t1: torch.Tensor,
    t2: torch.Tensor,
    t3: torch.Tensor,
    N: int,
    d_idx: int,
    starscream_rank: int,
    cpx_size: int,
    has_A: bool,
    prefill_decode_match: bool,
    prefill_match: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Slice-and-stitch along dim=0 for three tensors.
    Handles optional special entry A at index 0 if has_A=True.
    If has_A=False, all entries are just batches of size N.
    Args:
        t1, t2, t3: Tensors with shape
            [ (B - 1) * N + 1, ... ] if has_A=True
            [ B * N, ... ]          if has_A=False
        N: prefill length
        slice_idx: start index within each batch slice
        slice_len: number of items to take
        d_idx, starscream_rank: include entry 0 ("A") only if has_A and starscream_rank == d_idx
        has_A: whether the tensors contain a special entry at index 0
    Returns:
        (out1, out2, out3): stitched tensors from t1, t2, t3
    """
    M1 = t1.size(0)

    base = N // cpx_size
    extra = N % cpx_size

    if starscream_rank < extra:
        slice_len = base + 1
        slice_idx = starscream_rank * (base + 1)
    else:
        slice_len = base
        slice_idx = extra * (base + 1) + (starscream_rank - extra) * base

    # Number of batches depends on has_A
    if has_A:
        num_batches = (M1 - 1) // N
        base_offset = 1
    else:
        num_batches = M1 // N
        base_offset = 0

    pieces1: list[torch.Tensor] = []
    pieces2: list[torch.Tensor] = []
    pieces3: list[torch.Tensor] = []

    # Include A if requested
    if has_A and starscream_rank == d_idx:
        pieces1.append(t1[0:1])
        pieces2.append(t2[0:1])
        pieces3.append(t3[0:1])

    # Add per-batch slices with clipping
    if prefill_decode_match:
        for b in range(num_batches):
            batch_start = base_offset + b * N
            start = batch_start + slice_idx
            end = start + slice_len
            batch_end = batch_start + N

            if (start >= batch_end) or (start == end):
                continue
            if end > batch_end:
                end = batch_end

            pieces1.append(t1[start:end])
            pieces2.append(t2[start:end])
            pieces3.append(t3[start:end])

    if (len(pieces1) == 0):
        return (
            torch.empty_like(t1), 
            torch.empty_like(t2), 
            torch.full_like(t3, -1), 
            slice_idx,
            )

    return (
        torch.cat(pieces1, dim=0).contiguous(),
        torch.cat(pieces2, dim=0).contiguous(),
        torch.cat(pieces3, dim=0).contiguous(),
        slice_idx,
    )


@dataclass
class TritonAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    seq_lens_np: list
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    # Optional aot scheduling
    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None


class TritonAttentionMetadataBuilder(AttentionMetadataBuilder[TritonAttentionMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.block_size = kv_cache_spec.block_size

        model_config = vllm_config.model_config
        self.num_heads_q = model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        self.num_heads_kv = model_config.get_num_kv_heads(vllm_config.parallel_config)
        self.headdim = model_config.get_head_size()

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> TritonAttentionMetadata:
        attn_metadata = self.build(0, common_attn_metadata)
        # When doing full graph capture, setting seq_lens to
        # max_model_len will cause graph capture to be extremely
        # slow, so here we set it to 1.
        attn_metadata.seq_lens.fill_(1)
        return attn_metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TritonAttentionMetadata:
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len

        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        seq_lens_np = common_attn_metadata.seq_lens_cpu.tolist()
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        use_cascade = common_prefix_len > 0

        if use_cascade:
            cu_prefix_query_lens = torch.tensor(
                [0, num_actual_tokens], dtype=torch.int32, device=self.device
            )
            prefix_kv_lens = torch.tensor(
                [common_prefix_len], dtype=torch.int32, device=self.device
            )
            suffix_kv_lens = common_attn_metadata.seq_lens_cpu - common_prefix_len
            suffix_kv_lens = suffix_kv_lens.to(self.device)
        else:
            cu_prefix_query_lens = None
            prefix_kv_lens = None
            suffix_kv_lens = None
            prefix_scheduler_metadata = None

        attn_metadata = TritonAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            seq_lens_np=seq_lens_np,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
        )
        return attn_metadata


class TritonAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    supported_kernel_block_sizes: ClassVar[list[int | MultipleOf]] = [MultipleOf(16)]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN"

    @staticmethod
    def get_impl_cls() -> type["TritonAttentionImpl"]:
        return TritonAttentionImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

    @staticmethod
    def get_builder_cls() -> type["TritonAttentionMetadataBuilder"]:
        return TritonAttentionMetadataBuilder

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size >= 32

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True


class TritonAttentionImpl(AttentionImpl):
    can_return_lse_for_decode: bool = True
    def fused_output_quant_supported(self, quant_key: QuantKey):
        return quant_key == kFp8StaticTensorSym

    def supports_quant_query_input(self) -> bool:
        return current_platform.is_cuda()

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
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.cpx_size = get_starscream_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        self.starscream_rank = tp_rank % self.cpx_size
        from vllm.config import get_current_vllm_config
        config = get_current_vllm_config()
        self.enable_starscream = config.parallel_config.enable_starscream
        self.num_prompts = config.parallel_config.num_prompts
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "TritonAttentionImpl"
            )

        self.fp8_dtype = current_platform.fp8_dtype()

        self.sinks = sinks
        if sinks is not None:
            assert sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                f"heads in the layer. Sinks shape: {sinks.shape}, "
                f"num_heads: {num_heads}."
            )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with Paged Attention impl. in Triton.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [num_blocks, 2, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."

        if output_block_scale is not None:
            raise NotImplementedError(
                "fused block_scale output quantization is not yet supported"
                " for TritonAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        assert attn_metadata.use_cascade is False

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens
        key_cache, value_cache = kv_cache.unbind(1)
        if self.enable_starscream:
            query = cpx_model_parallel_all_gather(query.contiguous(), dim=-2)

            max_seqlen_q = attn_metadata.max_query_len
            seqused_k = attn_metadata.seq_lens
            batch_size = self.num_prompts
            has_A = len(seqused_k) == batch_size

            seq_lens_np = attn_metadata.seq_lens_np

            d_idx = (seq_lens_np[0] - 1) % self.cpx_size
            non_cold_location_match = (seq_lens_np[-1] - 1) % self.cpx_size

            prefill_match = max_seqlen_q > 1
            decode_match = bool(non_cold_location_match == self.starscream_rank)
            cold_start_match = bool(d_idx == self.starscream_rank)

            prefill_decode_match = prefill_match or decode_match

            num_batches = len(seqused_k) - has_A

            if (prefill_match):
                out1, out2, out3, slice_idx = slice_and_stitch_three(
                        key, 
                        value, 
                        attn_metadata.slot_mapping, 
                        max_seqlen_q, 
                        d_idx, 
                        self.starscream_rank, 
                        self.cpx_size, 
                        has_A, 
                        prefill_decode_match, 
                        prefill_match, 
                        )
            else:
                out1, out2, out3, slice_idx = slice_and_stitch_three_decode(
                        key, 
                        value, 
                        attn_metadata.slot_mapping, 
                        max_seqlen_q, 
                        d_idx, 
                        self.starscream_rank, 
                        self.cpx_size, 
                        has_A, 
                        prefill_decode_match, 
                        num_batches,
                        )

            location_match = cold_start_match or prefill_decode_match
        else:
            location_match = True
            out1 = key
            out2 = value
            out3 = attn_metadata.slot_mapping
            slice_idx = 0
            self.starscream_rank = 0

        if self.kv_sharing_target_layer_name is None:
            # Reshape the input keys and values and store them in the cache.
            # Skip this if sharing KV cache with an earlier attention layer.
            if self.kv_cache_dtype.startswith("fp8"):
                key_cache = key_cache.view(self.fp8_dtype)
                value_cache = value_cache.view(self.fp8_dtype)
                # triton kernel does not support uint8 kv_cache
                #  (because some explicit casts (e.g. float8_e4m3fnuz)
                #   are not supported)
            triton_reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )

        if self.kv_cache_dtype.startswith("fp8"):
            if key_cache.dtype != self.fp8_dtype:
                key_cache = key_cache.view(self.fp8_dtype)
                value_cache = value_cache.view(self.fp8_dtype)
            assert layer._q_scale_float == 1.0, (
                "A non 1.0 q_scale is not currently supported."
            )

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        descale_shape = (cu_seqlens_q.shape[0] - 1, key.shape[1])

        unified_attention(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            slice_idx=slice_idx,
            starscream_rank=self.starscream_rank,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            window_size=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            q_descale=None,  # Not supported
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
            sinks=self.sinks,
            output_scale=output_scale,
            cpx_size=self.cpx_size,
            enable_starscream=self.enable_starscream,
        )

        return output
