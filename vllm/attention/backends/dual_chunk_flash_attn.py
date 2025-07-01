# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with Dual chunk flash attention and sparse attention.
"""
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

import torch
import torch.distributed
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import AttentionLayer, AttentionType
from vllm.attention.backends.flash_attn import (FlashAttentionBackend,
                                                FlashAttentionImpl,
                                                FlashAttentionMetadata,
                                                FlashAttentionMetadataBuilder)
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.utils import async_tensor_h2d
from vllm.vllm_flash_attn import (flash_attn_varlen_func,
                                  flash_attn_with_kvcache, sparse_attn_func)

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUBuilder

logger = init_logger(__name__)


class DualChunkFlashAttentionBackend(FlashAttentionBackend):

    accept_output_buffer: bool = False

    @staticmethod
    def get_name() -> str:
        return "DUAL_CHUNK_FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> Type["DualChunkFlashAttentionImpl"]:
        return DualChunkFlashAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["DualChunkFlashAttentionMetadata"]:
        return DualChunkFlashAttentionMetadata

    @staticmethod
    def get_builder_cls() -> Type["DualChunkFlashAttentionMetadataBuilder"]:
        return DualChunkFlashAttentionMetadataBuilder


@dataclass
class DualChunkFlashAttentionMetadata(FlashAttentionMetadata):
    # Block size of the paged kv cache.
    block_size: int = 16

    # Original max position embeddings.
    original_max_position_embeddings: int = 0

    # Chunk size
    chunk_size: int = 8192

    # Local size
    local_size: int = 1024

    # (batch_size,). The orig sequence length per sequence.
    orig_seq_lens: Optional[List[int]] = None

    # orig_seq_lens stored as a tensor.
    orig_seq_lens_tensor: Optional[torch.Tensor] = None

    # Length scaling factor
    scaling_factor: Optional[torch.Tensor] = None

    # (batch_size,). Sequence lengths for intra attention.
    seq_lens_intra: Optional[torch.Tensor] = None

    # Max sequence length for intra attention.
    max_seq_len_intra: Optional[int] = None

    # (batch_size, num_blocks). Block table for intra attention.
    block_tables_intra: Optional[torch.Tensor] = None

    # (batch_size,). Sequence lengths for succ attention.
    seq_lens_succ: Optional[torch.Tensor] = None

    # Max sequence length for succ attention.
    max_seq_len_succ: Optional[int] = None

    # (batch_size, num_blocks). Block table for succ attention.
    block_tables_succ: Optional[torch.Tensor] = None

    # (batch_size,). Sequence lengths for inter attention.
    seq_lens_inter: Optional[torch.Tensor] = None

    # Max sequence length for inter attention.
    max_seq_len_inter: Optional[int] = None

    _cached_prefill_metadata: Optional[
        "DualChunkFlashAttentionMetadata"] = None
    _cached_decode_metadata: Optional["DualChunkFlashAttentionMetadata"] = None

    @property
    def prefill_metadata(self) -> Optional["DualChunkFlashAttentionMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            return self._cached_prefill_metadata

        prefill_metadata = super().prefill_metadata
        if prefill_metadata is None:
            return None

        prefill_metadata = DualChunkFlashAttentionMetadata(
            **prefill_metadata.asdict_zerocopy())

        prefill_metadata.orig_seq_lens = (
            None if self.orig_seq_lens is None else
            self.orig_seq_lens[:self.num_prefills])
        prefill_metadata.orig_seq_lens_tensor = (
            None if self.orig_seq_lens_tensor is None else
            self.orig_seq_lens_tensor[:self.num_prefills])

        if self.original_max_position_embeddings > 0:
            assert prefill_metadata.orig_seq_lens_tensor is not None
            prefill_metadata.scaling_factor = (
                0.1 * torch.log(prefill_metadata.orig_seq_lens_tensor /
                                self.original_max_position_embeddings) +
                1.0).clip(min=1)

        self._cached_prefill_metadata = prefill_metadata
        return prefill_metadata

    @property
    def decode_metadata(self) -> Optional["DualChunkFlashAttentionMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            return self._cached_decode_metadata

        decode_metadata = super().decode_metadata
        if decode_metadata is None:
            return None

        decode_metadata = DualChunkFlashAttentionMetadata(
            **decode_metadata.asdict_zerocopy())

        decode_metadata.orig_seq_lens_tensor = (
            None if self.orig_seq_lens_tensor is None else
            self.orig_seq_lens_tensor[self.num_prefills:])

        assert decode_metadata.orig_seq_lens_tensor is not None
        assert decode_metadata.block_tables is not None

        cache_seq_lens = decode_metadata.orig_seq_lens_tensor
        chunk_len = self.chunk_size - self.local_size
        chunk_num_curr = (cache_seq_lens - 1) // chunk_len
        batch_size = decode_metadata.num_decode_tokens

        if self.original_max_position_embeddings > 0:
            decode_metadata.scaling_factor = (0.1 * torch.log(
                cache_seq_lens / self.original_max_position_embeddings) +
                                              1.0).clip(min=1)

        seq_lens_intra = cache_seq_lens - chunk_num_curr * chunk_len
        max_seq_len_intra = seq_lens_intra.max().item()
        decode_metadata.seq_lens_intra = seq_lens_intra
        decode_metadata.max_seq_len_intra = max_seq_len_intra

        block_tables_intra = torch.zeros(
            batch_size,
            (max_seq_len_intra - 1) // self.block_size + 1,
            dtype=decode_metadata.block_tables.dtype,
            device=decode_metadata.block_tables.device,
        )
        for i in range(batch_size):
            st = chunk_num_curr[i] * chunk_len // self.block_size
            ed = min(
                st + (max_seq_len_intra - 1) // self.block_size + 1,
                (cache_seq_lens[i] - 1) // self.block_size + 1,
            )
            block_tables_intra[i, :ed -
                               st] = decode_metadata.block_tables[i, st:ed]
        decode_metadata.block_tables_intra = block_tables_intra

        seq_lens_succ = (chunk_num_curr -
                         (chunk_num_curr - 1).clip(min=0)) * chunk_len
        max_seq_len_succ = seq_lens_succ.max().item()
        decode_metadata.seq_lens_succ = seq_lens_succ
        decode_metadata.max_seq_len_succ = max_seq_len_succ
        if max_seq_len_succ:
            block_tables_succ = torch.zeros(
                batch_size,
                (max_seq_len_succ - 1) // self.block_size + 1,
                dtype=decode_metadata.block_tables.dtype,
                device=decode_metadata.block_tables.device,
            )
            for i in range(batch_size):
                start = ((chunk_num_curr[i] - 1).clip(min=0) * chunk_len //
                         self.block_size)
                end = min(
                    start + (max_seq_len_succ - 1) // self.block_size + 1,
                    (cache_seq_lens[i] - 1) // self.block_size + 1,
                )
                block_tables_succ[
                    i, :end - start] = decode_metadata.block_tables[i,
                                                                    start:end]
            decode_metadata.block_tables_succ = block_tables_succ

        seq_lens_inter = (chunk_num_curr - 1).clip(min=0) * chunk_len
        max_seq_len_inter = seq_lens_inter.max().item()
        decode_metadata.seq_lens_inter = seq_lens_inter
        decode_metadata.max_seq_len_inter = max_seq_len_inter

        self._cached_decode_metadata = decode_metadata
        return decode_metadata


class DualChunkFlashAttentionMetadataBuilder(FlashAttentionMetadataBuilder):

    def prepare(self):
        super().prepare()
        self.orig_seq_lens: List[int] = []

    def _add_seq_group(
            self, inter_data: "ModelInputForGPUBuilder.InterDataForSeqGroup",
            chunked_prefill_enabled: bool, prefix_cache_hit: bool):
        super()._add_seq_group(inter_data, chunked_prefill_enabled,
                               prefix_cache_hit)
        for prompt_len, seq_len in zip(inter_data.prompt_lens,
                                       inter_data.seq_lens):
            self.orig_seq_lens.append(max(prompt_len, seq_len))

    def build(self, seq_lens: List[int], query_lens: List[int],
              cuda_graph_pad_size: int, batch_size: int):
        attn_metadata = super().build(seq_lens, query_lens,
                                      cuda_graph_pad_size, batch_size)
        attn_metadata = DualChunkFlashAttentionMetadata(
            **attn_metadata.asdict_zerocopy())

        device = self.runner.device
        attn_metadata.orig_seq_lens = self.orig_seq_lens
        attn_metadata.orig_seq_lens_tensor = async_tensor_h2d(
            self.orig_seq_lens, torch.int, device, self.runner.pin_memory)

        attn_metadata.block_size = self.runner.block_size
        dual_chunk_attn_config = getattr(self.runner.model_config.hf_config,
                                         "dual_chunk_attention_config", {})
        attn_metadata.original_max_position_embeddings = \
            dual_chunk_attn_config.get("original_max_position_embeddings", 0)
        attn_metadata.chunk_size = dual_chunk_attn_config.get(
            "chunk_size", 8192)
        attn_metadata.local_size = dual_chunk_attn_config.get(
            "local_size", 1024)

        return attn_metadata


class DualChunkFlashAttentionImpl(FlashAttentionImpl):
    """
    If the input tensors contain prompt tokens, the layout is as follows:
    |<--------------- num_prefill_tokens ----------------->|
    |<--prefill_0-->|<--prefill_1-->|...|<--prefill_N-1--->|
    Otherwise, the layout is as follows:
    |<----------------- num_decode_tokens ------------------>|
    |<--decode_0-->|..........|<--decode_M-1-->|<--padding-->|
    Generation tokens can contain padding when cuda-graph is used.
    Currently, prompt tokens don't contain any padding.
    The prompts might have different lengths, while the generation tokens
    always have length 1.
    If chunked prefill is enabled, prefill tokens and decode tokens can be
    batched together in a flattened 1D query.
    |<----- num_prefill_tokens ---->|<------- num_decode_tokens --------->|
    |<-prefill_0->|...|<-prefill_N-1->|<--decode_0-->|...|<--decode_M-1-->|
    Currently, cuda graph is disabled for chunked prefill, meaning there's no
    padding between prefill and decode tokens.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
        layer_idx: int = -1,
        dual_chunk_attention_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError("KV sharing is not supported in V0.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = ((sliding_window, sliding_window)
                               if sliding_window is not None else (-1, -1))
        self.kv_cache_dtype = kv_cache_dtype

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        if sliding_window is not None:
            # NOTE(woosuk): flash-attn's sliding window does not work with
            # paged KV cache.
            raise ValueError(
                "Sliding window is not supported in FlashAttention.")

        support_head_sizes = (
            DualChunkFlashAttentionBackend.get_supported_head_sizes())

        if head_size not in support_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by FlashAttention. "
                f"Supported head sizes are: {support_head_sizes}.")

        assert dual_chunk_attention_config is not None
        self.chunk_size = dual_chunk_attention_config.get("chunk_size", 8192)
        self.local_size = dual_chunk_attention_config.get("local_size", 1024)
        self.original_max_position_embeddings = dual_chunk_attention_config.get(
            "original_max_position_embeddings", 0)
        self.sparse_attention_config = dual_chunk_attention_config.get(
            "sparse_attention_config", None)
        if not self.sparse_attention_config:
            logger.warning_once("Sparse attention will not be enabled as "
                                "sparse attention config is not provided.")
        self.sparse_attention_enabled = dual_chunk_attention_config.get(
            "sparse_attention_enabled", self.sparse_attention_config
            is not None)
        self.sparse_attention_threshold = dual_chunk_attention_config.get(
            "sparse_attention_threshold", 32768)
        self.sparse_attention_last_q = dual_chunk_attention_config.get(
            "sparse_attention_last_q", 64)
        self.layer_idx = layer_idx
        self.dual_chunk_attention_config = dual_chunk_attention_config

        if self.sparse_attention_config:
            self.sparse_attention_config = {
                int(i): j
                for i, j in self.sparse_attention_config[
                    self.layer_idx].items()
            }
            start_head = self.num_heads * get_tensor_model_parallel_rank()
            end_head = start_head + self.num_heads
            self.sparse_attention_config = [
                self.sparse_attention_config[i]
                for i in range(start_head, end_head)
            ]

        if self.sparse_attention_enabled:
            self.arange = torch.arange(self.sparse_attention_last_q,
                                       device="cuda")
            self.last_q_mask = (self.arange[None, None, :, None]
                                >= self.arange[None, None, None, :])

    def forward(  # type: ignore
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: DualChunkFlashAttentionMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with DualChunkFlashAttention.
        Args:
            query: shape = [num_tokens, num_heads * head_size]
            query_succ: shape = [num_tokens, num_heads * head_size]
            query_inter: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is None, "Output tensor not supported for DualChunk"

        if output_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported"
                " for FlashAttentionImpl")

        (
            query,
            query_succ,
            query_inter,
            query_succ_critical,
            query_inter_critical,
        ) = torch.split(query, query.shape[-1] // 5, dim=-1)

        assert (
            query_succ is not None and query_inter is not None
        ), "query_succ and query_inter are required in Dual Chunk Attention."

        num_tokens, hidden_size = query.shape

        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        query_succ = query_succ.view(-1, self.num_heads, self.head_size)
        query_inter = query_inter.view(-1, self.num_heads, self.head_size)
        query_succ_critical = query_succ_critical.view(-1, self.num_heads,
                                                       self.head_size)
        query_inter_critical = query_inter_critical.view(
            -1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        if self.original_max_position_embeddings > 0:
            if prefill_meta := attn_metadata.prefill_metadata:
                assert prefill_meta.scaling_factor is not None
                assert prefill_meta.query_start_loc is not None
                assert prefill_meta.orig_seq_lens is not None
                current_start = 0
                query_start_loc_cpu = prefill_meta.query_start_loc.cpu()
                for i in range(len(prefill_meta.orig_seq_lens)):
                    current_end = (current_start +
                                   (query_start_loc_cpu[i + 1] -
                                    query_start_loc_cpu[i]).item())
                    key[current_start:current_end].mul_(
                        prefill_meta.scaling_factor[i])
                    current_start = current_end
                assert current_end <= attn_metadata.num_prefill_tokens
            if decode_meta := attn_metadata.decode_metadata:
                assert decode_meta.scaling_factor is not None
                scaling_factor = decode_meta.scaling_factor
                key[attn_metadata.num_prefill_tokens:].mul_(
                    scaling_factor.unsqueeze(-1).unsqueeze(-1))

        if kv_cache is not None and kv_cache.numel() > 0:
            key_cache = kv_cache[0]
            value_cache = kv_cache[1]

            # Reshape the input keys and values and store them in the cache.
            # If kv_cache is not provided, the new key and value tensors are
            # not cached. This happens during the initial memory profiling run.
            ops.reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping.flatten(),
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )

        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        assert key.shape[0] == num_prefill_tokens + num_decode_tokens
        assert value.shape[0] == num_prefill_tokens + num_decode_tokens
        output = torch.empty_like(query)

        # Query for decode. KV is not needed because it is already cached.
        decode_query = query[num_prefill_tokens:]
        decode_query_succ = query_succ[num_prefill_tokens:]
        decode_query_inter = query_inter[num_prefill_tokens:]

        # QKV for prefill.
        query = query[:num_prefill_tokens]
        query_succ = query_succ[:num_prefill_tokens]
        query_inter = query_inter[:num_prefill_tokens]
        query_succ_critical = query_succ_critical[:num_prefill_tokens]
        query_inter_critical = query_inter_critical[:num_prefill_tokens]
        key = key[:num_prefill_tokens]
        value = value[:num_prefill_tokens]
        assert query.shape[0] == num_prefill_tokens
        assert decode_query.shape[0] == num_decode_tokens

        if prefill_meta := attn_metadata.prefill_metadata:
            # Prompt run.
            if (kv_cache is None or prefill_meta.block_tables is None
                    or prefill_meta.block_tables.numel() == 0):
                # normal attention, called during the profiling run.
                out = flash_attn_varlen_func(
                    q=query,
                    k=key,
                    v=value,
                    cu_seqlens_q=prefill_meta.seq_start_loc,
                    cu_seqlens_k=prefill_meta.seq_start_loc,
                    max_seqlen_q=prefill_meta.max_prefill_seq_len,
                    max_seqlen_k=prefill_meta.max_prefill_seq_len,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=self.sliding_window,
                    alibi_slopes=self.alibi_slopes,
                )
                assert output[:num_prefill_tokens].shape == out.shape
                output[:num_prefill_tokens] = out
            else:
                # prefix-enabled attention
                assert prefill_meta.seq_lens is not None
                assert prefill_meta.orig_seq_lens is not None
                output[:num_prefill_tokens] = (
                    self._dual_chunk_flash_attn_prefill(
                        q=query,
                        q_succ=query_succ,
                        q_inter=query_inter,
                        q_succ_critical=query_succ_critical,
                        q_inter_critical=query_inter_critical,
                        k=key_cache,
                        v=value_cache,
                        cu_seqlens_q=prefill_meta.query_start_loc,
                        cu_seqlens_k=prefill_meta.seq_start_loc,
                        orig_seq_lens=prefill_meta.orig_seq_lens,
                        scaling_factor=prefill_meta.scaling_factor,
                        softmax_scale=self.scale,
                        causal=True,
                        window_size=(-1, -1),
                        alibi_slopes=self.alibi_slopes,
                        block_table=prefill_meta.block_tables,
                        chunk_size=self.chunk_size,
                        local_size=self.local_size,
                    ))

        if decode_meta := attn_metadata.decode_metadata:
            # Decoding run.
            output[num_prefill_tokens:] = (
                self._dual_chunk_flash_attn_decoding(
                    decode_query.unsqueeze(1),
                    decode_query_succ.unsqueeze(1),
                    decode_query_inter.unsqueeze(1),
                    key_cache,
                    value_cache,
                    block_table=decode_meta.block_tables,
                    cache_seqlens=decode_meta.seq_lens_tensor,
                    softmax_scale=self.scale,
                    causal=True,
                    alibi_slopes=self.alibi_slopes,
                    chunk_size=self.chunk_size,
                    local_size=self.local_size,
                    original_max_position_embeddings=self.
                    original_max_position_embeddings,
                    decode_meta=decode_meta,
                ).squeeze(1))
        # Reshape the output tensor.
        return output.view(num_tokens, hidden_size)

    def _dual_chunk_flash_attn_prefill(
        self,
        q,
        q_succ,
        q_inter,
        q_succ_critical,
        q_inter_critical,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        orig_seq_lens: List[int],
        scaling_factor: torch.Tensor,
        softmax_scale: float,
        causal: Optional[bool] = True,
        window_size: Tuple[int, int] = (-1, -1),
        alibi_slopes: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        chunk_size: int = 8192,
        local_size: int = 1024,
    ):
        if alibi_slopes is not None:
            raise ValueError(
                "Dual Chunk Attention does not support alibi_slopes")
        if not causal:
            raise ValueError(
                "Dual Chunk Attention does not support causal=False")
        if window_size != (-1, -1):
            raise ValueError(
                "Dual Chunk Attention does not support window_size")

        cu_seqlens_q_cpu = cu_seqlens_q.cpu().tolist()
        cu_seqlens_k_cpu = cu_seqlens_k.cpu().tolist()
        all_outputs = []

        for i in range(0, len(cu_seqlens_q_cpu) - 1):
            qs = cu_seqlens_q_cpu[i]
            qe = cu_seqlens_q_cpu[i:i + 2][-1]
            ks = cu_seqlens_k_cpu[i]
            ke = cu_seqlens_k_cpu[i:i + 2][-1]

            current_q = q[qs:qe]
            current_q_succ = q_succ[qs:qe]
            current_q_inter = q_inter[qs:qe]
            current_q_succ_critical = q_succ_critical[qs:qe]
            current_q_inter_critical = q_inter_critical[qs:qe]

            if block_table is None:
                current_k = k[ks:ke]
                current_v = v[ks:ke]
                current_block_table = None
                current_orig_seq_len = orig_seq_lens[i]
            else:
                current_block_table = block_table[i]
                current_orig_seq_len = orig_seq_lens[i]
                current_k = k
                current_v = v
            sparse_attn_enabled = (self.sparse_attention_enabled
                                   and current_orig_seq_len
                                   > self.sparse_attention_threshold)

            if current_q.shape[0] == 0:
                continue

            if current_k.shape[0] == 0:
                all_outputs.append(
                    torch.zeros(
                        (current_q.shape[0], current_q.shape[1], v.shape[2]),
                        device=q.device,
                        dtype=q.dtype,
                    ))
                continue

            current_output = torch.empty_like(current_q)
            group_size = int(current_q.size(-2) / current_k.size(-2))

            if sparse_attn_enabled:
                num_device_q_heads = current_q.size(-2)
                heads_vertical_size = torch.empty(size=(num_device_q_heads, ),
                                                  dtype=torch.int32)
                heads_slash_size = torch.empty(size=(num_device_q_heads, ),
                                               dtype=torch.int32)
                for head_id in range(current_q.size(-2)):
                    (
                        ty,
                        vertical_size,
                        slash_size,
                        _,
                    ) = self.sparse_attention_config[head_id]
                    assert ty == "vertical_and_slash", "only support slash mode"

                    if vertical_size == 30:
                        vertical_size += 100
                    heads_vertical_size[head_id] = vertical_size
                    heads_slash_size[head_id] = slash_size

                current_output = self._dual_chunk_flash_attn_prefill_func(
                    current_q,  # allheads
                    current_q_succ,
                    current_q_inter,
                    current_q_succ_critical,
                    current_q_inter_critical,
                    current_k,
                    current_v,
                    current_block_table,
                    softmax_scale,
                    chunk_size,
                    local_size,
                    scaling_factor[i].item(),
                    ke - ks,
                    sparse_attn_enabled=sparse_attn_enabled,
                    heads_vertical_size=heads_vertical_size,
                    heads_slash_size=heads_slash_size,
                    group_size=group_size)
            else:
                for head_id in range(current_q.size(-2)):
                    # (seq_len, num_heads, head_size)
                    current_q_head = current_q[:, head_id, :].unsqueeze(1)
                    current_q_succ_head = \
                        current_q_succ[:, head_id, :].unsqueeze(1)
                    current_q_inter_head = \
                        current_q_inter[:, head_id, :].unsqueeze(1)
                    current_q_succ_head_critical = \
                        current_q_succ_critical[:, head_id, :].unsqueeze(1)
                    current_q_inter_head_critical = \
                        current_q_inter_critical[:, head_id, :].unsqueeze(1)
                    if block_table is not None:
                        current_k_head = current_k[..., head_id //
                                                   group_size, :].unsqueeze(2)
                        current_v_head = current_v[..., head_id //
                                                   group_size, :].unsqueeze(2)

                    else:
                        current_k_head = current_k[:, head_id, :].unsqueeze(1)
                        current_v_head = current_v[:, head_id, :].unsqueeze(1)

                    current_out = self._dual_chunk_flash_attn_prefill_func(
                        current_q_head,
                        current_q_succ_head,
                        current_q_inter_head,
                        current_q_succ_head_critical,
                        current_q_inter_head_critical,
                        current_k_head,
                        current_v_head,
                        current_block_table,
                        softmax_scale,
                        chunk_size,
                        local_size,
                        scaling_factor[i].item(),
                        ke - ks,
                        sparse_attn_enabled=sparse_attn_enabled,
                    )
                    current_output[:, head_id:head_id + 1, :] = current_out
            all_outputs.append(current_output)
        return torch.cat(all_outputs, dim=0)

    def _dual_chunk_flash_attn_prefill_func(
        self,
        q,
        q_succ,
        q_inter,
        q_succ_critical,
        q_inter_critical,
        k,
        v,
        block_table,
        softmax_scale: float,
        chunk_size: int,
        local_size: int,
        scaling_factor: float,
        k_length: int,
        sparse_attn_enabled: Optional[bool] = True,
        heads_vertical_size=None,
        heads_slash_size=None,
        group_size=None,
    ):
        flash_results = []
        chunk_len = chunk_size - local_size

        if block_table is not None:
            block_size = v.shape[1]
            if chunk_len % block_size != 0:
                raise ValueError("chunk_len must be divisible by block_size.")
        else:
            block_size = 1

        if self.original_max_position_embeddings > 0:
            softmax_scale = softmax_scale * scaling_factor

        begin = k_length - q.shape[0]
        while begin < k_length:
            flash_per_chunk = []

            prev_chunk_end_pos = (begin // chunk_len) * chunk_len
            next_chunk_end_pos = prev_chunk_end_pos + chunk_len
            end = min(next_chunk_end_pos, k_length)
            qbegin = begin - (k_length - q.shape[0])
            qend = end - (k_length - q.shape[0])

            qk_chunks = []
            q_states_intra = q[qbegin:qend]
            # choose critical token
            if block_table is not None:
                block_tables_intra = _get_block(block_table, block_size,
                                                prev_chunk_end_pos, end)
                k_states_intra = k[block_tables_intra].view(
                    -1, *k.shape[-2:])[:(end - prev_chunk_end_pos)]
                v_states_intra = v[block_tables_intra].view(
                    -1, *v.shape[-2:])[:(end - prev_chunk_end_pos)]
            else:
                block_tables_intra = None
                k_states_intra = k[prev_chunk_end_pos:end]
                v_states_intra = v[prev_chunk_end_pos:end]

            if sparse_attn_enabled:
                last_q_size = min(qend - qbegin, self.sparse_attention_last_q)
                _, num_device_k_heads, head_dim = k_states_intra.shape
                k_states_intra = (k_states_intra.unsqueeze(2).repeat(
                    1, 1, group_size,
                    1).reshape(-1, num_device_k_heads * group_size, head_dim))
                v_states_intra = (v_states_intra.unsqueeze(2).repeat(
                    1, 1, group_size,
                    1).reshape(-1, num_device_k_heads * group_size, head_dim))
                qk_chunks.append(
                    (q_states_intra.transpose(0, 1)[:, -last_q_size:] *
                     softmax_scale) @ k_states_intra.permute(1, 2, 0))

            if prev_chunk_end_pos - chunk_len >= 0:
                q_states_succ = q_succ[qbegin:qend]
                q_states_succ_critical = q_succ_critical[qbegin:qend]
                if block_table is not None:
                    block_tables_succ = _get_block(
                        block_table, block_size,
                        prev_chunk_end_pos - chunk_len, prev_chunk_end_pos)
                    k_states_succ = k[block_tables_succ].view(
                        -1, *k.shape[-2:])[:chunk_len]
                    v_states_succ = v[block_tables_succ].view(
                        -1, *v.shape[-2:])[:chunk_len]
                else:
                    k_states_succ = k[prev_chunk_end_pos -
                                      chunk_len:prev_chunk_end_pos]
                    v_states_succ = v[prev_chunk_end_pos -
                                      chunk_len:prev_chunk_end_pos]

                if sparse_attn_enabled:
                    k_states_succ = (k_states_succ.unsqueeze(2).repeat(
                        1, 1, group_size,
                        1).reshape(-1, num_device_k_heads * group_size,
                                   head_dim))
                    v_states_succ = (v_states_succ.unsqueeze(2).repeat(
                        1, 1, group_size,
                        1).reshape(-1, num_device_k_heads * group_size,
                                   head_dim))
                    qk_chunks.append((q_states_succ_critical.transpose(
                        0, 1)[:, -last_q_size:] * softmax_scale)
                                     @ k_states_succ.permute(1, 2, 0))

            if prev_chunk_end_pos - chunk_len * 2 >= 0:
                q_states_inter = q_inter[qbegin:qend]
                q_states_inter_critical = q_inter_critical[qbegin:qend]
                if block_table is not None:
                    block_tables_inter = _get_block(
                        block_table, block_size, 0,
                        prev_chunk_end_pos - chunk_len)
                    k_states_inter = k[block_tables_inter].view(
                        -1, *k.shape[-2:])[:(prev_chunk_end_pos - chunk_len)]
                    v_states_inter = v[block_tables_inter].view(
                        -1, *v.shape[-2:])[:(prev_chunk_end_pos - chunk_len)]
                else:
                    k_states_inter = k[:prev_chunk_end_pos - chunk_len]
                    v_states_inter = v[:prev_chunk_end_pos - chunk_len]

                if sparse_attn_enabled:
                    k_states_inter = (k_states_inter.unsqueeze(2).repeat(
                        1, 1, group_size,
                        1).reshape(-1, num_device_k_heads * group_size,
                                   head_dim))
                    v_states_inter = (v_states_inter.unsqueeze(2).repeat(
                        1, 1, group_size,
                        1).reshape(-1, num_device_k_heads * group_size,
                                   head_dim))
                    qk_chunks.append((q_states_inter_critical.transpose(
                        0, 1)[:, -last_q_size:] * softmax_scale)
                                     @ k_states_inter.permute(1, 2, 0))

            if sparse_attn_enabled:
                reversed_qk = qk_chunks[::-1]
                qk = torch.cat(reversed_qk, dim=-1)

                qk[:, :, -last_q_size:] = torch.where(
                    self.last_q_mask[..., -last_q_size:,
                                     -last_q_size:].to(qk.device),
                    qk[:, :, -last_q_size:], -torch.inf)
                qk = F.softmax(qk, dim=-1, dtype=torch.float32)

                vertical = qk.sum(-2, keepdim=True)
                vertical[..., :30] = torch.inf

                # Avoid sorting by using the min/max ints to fill the indexer
                # buffers.
                int32_max = torch.iinfo(torch.int32).max
                int32_min = torch.iinfo(torch.int32).min
                n_heads = qk.size()[0]
                max_slash_topk = torch.max(heads_slash_size).item()
                max_vertical_topk = torch.max(heads_vertical_size).item()
                # store each head's slash topk, vertical topk
                vertical = vertical.reshape((n_heads, -1))
                # prevent out of range when prompt size < max_vertical_topk
                max_vertical_topk = min(vertical.shape[-1], max_vertical_topk)
                vertical_topk_buffer = torch.topk(vertical, max_vertical_topk,
                                                  -1).indices
                slash_topk_buffer = torch.empty(size=(n_heads, max_slash_topk),
                                                dtype=torch.int64,
                                                device=qk.device)
                for head_i in range(n_heads):
                    #  (nqheads=1, lastq, k_len)
                    head_score = qk[head_i:head_i + 1, :, :]
                    slash_scores = _sum_all_diagonal_matrix(head_score)
                    if head_score.size(1) != 1:
                        # drop right up corner
                        slash_scores = slash_scores[..., :-last_q_size + 1]
                    slash_scores[..., -100:] = torch.inf

                    head_slash_size = heads_slash_size[head_i]
                    head_slash_size = min(head_slash_size, vertical.size(-1))
                    slash_topk = torch.topk(slash_scores, head_slash_size,
                                            -1).indices
                    #（nheads, max_topk）
                    slash_topk_buffer[head_i, :head_slash_size] = slash_topk

                    # reset heads topk
                    heads_slash_size[head_i] = head_slash_size
                    heads_vertical_size[head_i] = min(
                        heads_vertical_size[head_i], max_vertical_topk)

                # store
                vertical_buffer = torch.full((n_heads, max_vertical_topk),
                                             int32_max,
                                             dtype=torch.int64,
                                             device=q.device)
                slash_buffer = torch.full((n_heads, max_slash_topk),
                                          int32_min,
                                          dtype=torch.int64,
                                          device=q.device)
                succ_vertical_buffer = torch.full((n_heads, max_vertical_topk),
                                                  int32_max,
                                                  dtype=torch.int64,
                                                  device=q.device)
                succ_slash_buffer = torch.full((n_heads, max_slash_topk),
                                               int32_min,
                                               dtype=torch.int64,
                                               device=q.device)
                inter_vertical_buffer = torch.full(
                    (n_heads, max_vertical_topk),
                    int32_max,
                    dtype=torch.int64,
                    device=q.device)
                inter_slash_buffer = torch.full((n_heads, max_slash_topk),
                                                int32_min,
                                                dtype=torch.int64,
                                                device=q.device)

                vertical_size_buffer = torch.empty(size=(n_heads, ),
                                                   dtype=torch.int32,
                                                   device=q.device)
                slash_sizes_buffer = torch.empty(size=(n_heads, ),
                                                 dtype=torch.int32,
                                                 device=q.device)
                succ_vertical_size_buffer = torch.empty(size=(n_heads, ),
                                                        dtype=torch.int32,
                                                        device=q.device)
                succ_slash_sizes_buffer = torch.empty(size=(n_heads, ),
                                                      dtype=torch.int32,
                                                      device=q.device)
                inter_vertical_size_buffer = torch.empty(size=(n_heads, ),
                                                         dtype=torch.int32,
                                                         device=q.device)
                inter_slash_sizes_buffer = torch.empty(size=(n_heads, ),
                                                       dtype=torch.int32,
                                                       device=q.device)

                for head_i in range(n_heads):
                    vertical_topk = vertical_topk_buffer[
                        head_i, :heads_vertical_size[head_i]]
                    # intra
                    intra_vertical_indices = vertical_topk[
                        vertical_topk >=
                        prev_chunk_end_pos] - prev_chunk_end_pos
                    if intra_vertical_indices.nelement() == 0:
                        intra_vertical_indices = torch.cat([
                            intra_vertical_indices,
                            torch.arange(0,
                                         k_states_intra.size(0),
                                         max(1,
                                             k_states_intra.size(0) / 5),
                                         dtype=torch.int32,
                                         device=intra_vertical_indices.device)
                        ])
                    slash_topk = slash_topk_buffer[
                        head_i, :heads_slash_size[head_i]]
                    intra_slash_indices = (
                        (qk.size(-1) - 1) -
                        slash_topk[slash_topk >= prev_chunk_end_pos])
                    # fill buffer
                    v_count = intra_vertical_indices.nelement()
                    s_count = intra_slash_indices.nelement()
                    vertical_size_buffer[head_i] = v_count
                    slash_sizes_buffer[head_i] = s_count
                    vertical_buffer[head_i, :v_count].copy_(
                        intra_vertical_indices)
                    slash_buffer[head_i, :s_count].copy_(intra_slash_indices)
                    # succ
                    if prev_chunk_end_pos - chunk_len >= 0:
                        succ_vertical_indices = vertical_topk[
                            (vertical_topk < prev_chunk_end_pos)
                            & (vertical_topk >= prev_chunk_end_pos -
                               chunk_len)] - (prev_chunk_end_pos - chunk_len)
                        # TODO: support no vertical
                        if succ_vertical_indices.nelement() == 0:
                            succ_vertical_indices = torch.cat([
                                succ_vertical_indices,
                                torch.arange(
                                    0,
                                    k_states_succ.size(0),
                                    max(1,
                                        k_states_succ.size(0) / 5),
                                    dtype=torch.int32,
                                    device=intra_vertical_indices.device)
                            ])
                        succ_slash_indices = (
                            (prev_chunk_end_pos + (qend - qbegin) - 1) -
                            slash_topk[((slash_topk >=
                                         (prev_chunk_end_pos - chunk_len)) &
                                        (slash_topk < (prev_chunk_end_pos +
                                                       (qend - qbegin))))])
                        if succ_slash_indices.nelement() == 0:
                            succ_slash_indices = torch.cat([
                                succ_slash_indices,
                                torch.arange(
                                    0,
                                    k_states_succ.size(0),
                                    max(1,
                                        k_states_succ.size(0) / 5),
                                    dtype=torch.int32,
                                    device=intra_vertical_indices.device)
                            ])
                        # fill buffer
                        v_count = succ_vertical_indices.nelement()
                        s_count = succ_slash_indices.nelement()
                        succ_vertical_size_buffer[head_i] = v_count
                        succ_slash_sizes_buffer[head_i] = s_count
                        succ_vertical_buffer[head_i, :v_count].copy_(
                            succ_vertical_indices)
                        succ_slash_buffer[head_i, :s_count].copy_(
                            succ_slash_indices)

                    if prev_chunk_end_pos - 2 * chunk_len >= 0:
                        inter_vertical_indices = vertical_topk[
                            vertical_topk < prev_chunk_end_pos - chunk_len]

                        if inter_vertical_indices.nelement() == 0:
                            inter_vertical_indices = torch.cat([
                                inter_vertical_indices,
                                torch.arange(
                                    0,
                                    k_states_inter.size(0),
                                    max(1,
                                        k_states_inter.size(0) / 5),
                                    dtype=torch.int32,
                                    device=intra_vertical_indices.device)
                            ])
                        inter_slash_indices = (
                            (prev_chunk_end_pos - chunk_len +
                             (qend - qbegin) - 1) -
                            slash_topk[slash_topk < (prev_chunk_end_pos -
                                                     chunk_len +
                                                     (qend - qbegin))])
                        if inter_slash_indices.nelement() == 0:
                            inter_slash_indices = torch.cat([
                                inter_slash_indices,
                                torch.arange(
                                    0,
                                    k_states_inter.size(0),
                                    max(1,
                                        k_states_inter.size(0) / 5),
                                    dtype=torch.int32,
                                    device=intra_vertical_indices.device)
                            ])
                        # fill buffer
                        v_count = inter_vertical_indices.nelement()
                        s_count = inter_slash_indices.nelement()
                        inter_vertical_size_buffer[head_i] = v_count
                        inter_slash_sizes_buffer[head_i] = s_count
                        inter_vertical_buffer[head_i, :v_count].copy_(
                            inter_vertical_indices)
                        inter_slash_buffer[head_i, :s_count].copy_(
                            inter_slash_indices)
            else:
                intra_vertical_indices, intra_slash_indices = None, None
                succ_vertical_indices, succ_slash_indices = None, None
                inter_vertical_indices, inter_slash_indices = None, None

            if sparse_attn_enabled:
                flash_result = self._do_flash_attn(
                    q_states_intra,
                    k_states_intra,
                    v_states_intra,
                    softmax_scale=softmax_scale,
                    causal=True,
                    block_table=block_table,
                    stage="intra",
                    vertical_indices=vertical_buffer,
                    slash_indices=slash_buffer,
                    vertical_indices_count=vertical_size_buffer,
                    slash_indices_count=slash_sizes_buffer,
                    mergehead_softmax_scale=softmax_scale,
                    sparse_attn_enabled=sparse_attn_enabled)
            else:
                flash_result = self._do_flash_attn(
                    q_states_intra,
                    k_states_intra,
                    v_states_intra,
                    softmax_scale=softmax_scale,
                    causal=True,
                    block_table=block_table,
                    stage="intra",
                    vertical_indices=intra_vertical_indices,
                    slash_indices=intra_slash_indices,
                    sparse_attn_enabled=sparse_attn_enabled)
            flash_per_chunk.append(flash_result)

            if prev_chunk_end_pos - chunk_len >= 0:
                if sparse_attn_enabled:
                    flash_result = self._do_flash_attn(
                        q_states_succ,
                        k_states_succ,
                        v_states_succ,
                        softmax_scale=softmax_scale,
                        causal=False,
                        block_table=block_table,
                        stage="succ",
                        vertical_indices=succ_vertical_buffer,
                        slash_indices=succ_slash_buffer,
                        vertical_indices_count=succ_vertical_size_buffer,
                        slash_indices_count=succ_slash_sizes_buffer,
                        mergehead_softmax_scale=softmax_scale,
                        sparse_attn_enabled=sparse_attn_enabled)
                else:
                    flash_result = self._do_flash_attn(
                        q_states_succ,
                        k_states_succ,
                        v_states_succ,
                        softmax_scale=softmax_scale,
                        causal=False,
                        block_table=block_table,
                        stage="succ",
                        vertical_indices=succ_vertical_indices,
                        slash_indices=succ_slash_indices,
                        sparse_attn_enabled=sparse_attn_enabled)
                flash_per_chunk.append(flash_result)

            if prev_chunk_end_pos - chunk_len * 2 >= 0:
                if sparse_attn_enabled:
                    flash_result = self._do_flash_attn(
                        q_states_inter,
                        k_states_inter,
                        v_states_inter,
                        softmax_scale=softmax_scale,
                        causal=False,
                        block_table=block_table,
                        stage="inter",
                        vertical_indices=inter_vertical_buffer,
                        slash_indices=inter_slash_buffer,
                        vertical_indices_count=inter_vertical_size_buffer,
                        slash_indices_count=inter_slash_sizes_buffer,
                        mergehead_softmax_scale=softmax_scale,
                        sparse_attn_enabled=sparse_attn_enabled)
                else:
                    flash_result = self._do_flash_attn(
                        q_states_inter,
                        k_states_inter,
                        v_states_inter,
                        softmax_scale=softmax_scale,
                        causal=False,
                        block_table=block_table,
                        stage="inter",
                        vertical_indices=inter_vertical_indices,
                        slash_indices=inter_slash_indices,
                        sparse_attn_enabled=sparse_attn_enabled)
                flash_per_chunk.append(flash_result)

            flash_results.append(flash_per_chunk)
            begin = end

        attn_output = self._merge_attn_outputs(flash_results)
        del flash_results
        return attn_output

    def _do_flash_attn(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        softmax_scale: float,
        causal: bool = True,
        block_table: torch.Tensor = None,
        max_seqlen_k: Optional[int] = None,
        stage: str = "intra",
        vertical_indices: Optional[torch.Tensor] = None,
        slash_indices: Optional[torch.Tensor] = None,
        vertical_indices_count: Optional[torch.Tensor] = None,
        slash_indices_count: Optional[torch.Tensor] = None,
        mergehead_softmax_scale: Optional[float] = None,
        sparse_attn_enabled: Optional[bool] = False,
    ):
        if max_seqlen_k is None:
            max_seqlen_k = key_states.shape[0]

        q_len = query_states.shape[0]
        q_heads = query_states.shape[1]
        h_dim = query_states.shape[-1]

        if sparse_attn_enabled:
            assert slash_indices is not None
            if stage == "intra":
                assert causal
            else:
                assert not causal

            query_states = query_states.unsqueeze(0).transpose(1, 2)
            key_states = key_states.unsqueeze(0).transpose(1, 2)
            value_states = value_states.unsqueeze(0).transpose(1, 2)

            q = query_states
            k = key_states
            v = value_states

            if (vertical_indices_count is not None and \
                    slash_indices_count is not None):
                assert mergehead_softmax_scale is not None

                res, s_lse = _vertical_slash_sparse_attention(
                    q,
                    k,
                    v,
                    vertical_indices,
                    slash_indices,
                    mergehead_softmax_scale,
                    causal=causal,
                    stage=stage,
                    vertical_indices_count=vertical_indices_count,
                    slash_indices_count=slash_indices_count)
                res = res.view(q_heads, q_len,
                               h_dim).transpose(0, 1)  # (qlen,nhead,h_dim)
                s_lse = s_lse.view(
                    q_heads, q_len,
                    1).squeeze(-1).unsqueeze(0).float()  # (1, nhead,qlen)
            else:
                res, s_lse = _vertical_slash_sparse_attention(q,
                                                              k,
                                                              v,
                                                              vertical_indices,
                                                              slash_indices,
                                                              softmax_scale,
                                                              causal=causal,
                                                              stage=stage)
                res = res.view(q_len, q_heads, h_dim)
                s_lse = s_lse.view(q_len, q_heads, 1).transpose(0, 2).float()
            return res, s_lse

        output, softmax_lse = flash_attn_varlen_func(
            q=query_states,
            k=key_states,
            v=value_states,
            softmax_scale=softmax_scale,
            cu_seqlens_q=torch.tensor([0, query_states.shape[0]],
                                      dtype=torch.int32,
                                      device=query_states.device),
            max_seqlen_q=query_states.shape[0],
            cu_seqlens_k=torch.tensor([0, max_seqlen_k],
                                      dtype=torch.int32,
                                      device=query_states.device),
            max_seqlen_k=max_seqlen_k,
            causal=causal,
            block_table=block_table.unsqueeze(0),
            return_softmax_lse=True,
        )
        softmax_lse = softmax_lse.view(q_len, q_heads, 1).transpose(0,
                                                                    2).float()
        return output, softmax_lse

    def _merge_attn_outputs(
        self,
        flash_results: List[List[Tuple[torch.Tensor, torch.Tensor]]],
        return_lse: Optional[bool] = False,
    ) -> torch.Tensor:
        attn_outputs_all = []
        logits_all = []

        for flash_per_chunk in flash_results:
            if len(flash_per_chunk) == 1:
                attn_outputs_all.append(flash_per_chunk[0][0])
                if return_lse:
                    logits_all.append(flash_per_chunk[0][1])
                continue

            attn_outputs = torch.stack([
                flash_attn_output[0] for flash_attn_output in flash_per_chunk
            ])
            logits = torch.stack([
                flash_attn_output[1] for flash_attn_output in flash_per_chunk
            ])
            logits = logits.to(torch.float32)

            if return_lse:
                max_val = torch.max(logits, dim=0).values
                diff = torch.abs(logits[0] - logits[1])
                log_sum_exp = max_val + torch.log1p(torch.exp(-diff))
                logits_all.append(log_sum_exp)

            max_logits = torch.max(logits, dim=0).values
            stable_logits = logits - max_logits.unsqueeze(0)
            lse_s = torch.exp(stable_logits).detach()
            lse_sum = torch.sum(lse_s, dim=0)
            lse_s /= lse_sum
            attn_outputs *= lse_s.unsqueeze(-1).transpose(2, 3).squeeze(1)
            attn_outputs_all.append(attn_outputs.sum(dim=0))

        if return_lse:
            return (torch.cat(attn_outputs_all,
                              dim=0), torch.cat(logits_all, dim=-1))
        else:
            return torch.cat(attn_outputs_all, dim=0)

    def _dual_chunk_flash_attn_decoding(
        self,
        query: torch.Tensor,
        query_succ: torch.Tensor,
        query_inter: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        softmax_scale: float,
        causal: bool,
        alibi_slopes: Optional[torch.Tensor],
        chunk_size: int,
        local_size: int,
        original_max_position_embeddings: int,
        decode_meta: DualChunkFlashAttentionMetadata,
    ):
        if not causal:
            raise ValueError(
                "Dual Chunk Attention does not support causal=False")

        block_size = value_cache.shape[1]
        chunk_len = chunk_size - local_size
        if chunk_len % block_size != 0:
            raise ValueError("chunk_len must be divisible by block_size.")
        if original_max_position_embeddings > 0:
            assert decode_meta.scaling_factor is not None
            scaling_factor = decode_meta.scaling_factor
            query = (query * scaling_factor.view(-1, 1, 1, 1)).to(
                query.dtype
            )  # possible for numerical issue, need to fused in the kernel
            query_succ = (query_succ * scaling_factor.view(-1, 1, 1, 1)).to(
                query.dtype)
            query_inter = (query_inter * scaling_factor.view(-1, 1, 1, 1)).to(
                query.dtype)
        outputs_list = []
        softmax_lses_list = []

        # intra-attention
        intra_output, intra_softmax_lse = (
            self._dual_chunk_flash_attn_decoding_with_exp_sums(
                query,
                key_cache,
                value_cache,
                decode_meta.block_tables_intra,
                decode_meta.seq_lens_intra,
                softmax_scale,
                alibi_slopes,
                causal=False,
            ))
        outputs_list.append(intra_output)
        softmax_lses_list.append(intra_softmax_lse)

        # succ-attention
        if decode_meta.max_seq_len_succ:
            succ_output, succ_softmax_lse = (
                self._dual_chunk_flash_attn_decoding_with_exp_sums(
                    query_succ,
                    key_cache,
                    value_cache,
                    decode_meta.block_tables_succ,
                    decode_meta.seq_lens_succ,
                    softmax_scale,
                    alibi_slopes,
                    causal=False,
                ))
            outputs_list.append(succ_output)
            softmax_lses_list.append(succ_softmax_lse)

        # inter-attention
        if decode_meta.max_seq_len_inter:
            inter_output, inter_softmax_lse = (
                self._dual_chunk_flash_attn_decoding_with_exp_sums(
                    query_inter,
                    key_cache,
                    value_cache,
                    block_table[:, :decode_meta.max_seq_len_inter],
                    decode_meta.seq_lens_inter,
                    softmax_scale,
                    alibi_slopes,
                    causal=False,
                ))
            outputs_list.append(inter_output)
            softmax_lses_list.append(inter_softmax_lse)
        outputs = torch.stack(outputs_list, dim=0)
        del outputs_list
        softmax_lses = torch.stack(softmax_lses_list, dim=0).to(torch.float32)
        del softmax_lses_list
        max_logits = torch.max(softmax_lses, dim=0).values
        stable_logits = softmax_lses - max_logits.unsqueeze(0)
        lse_s = torch.exp(stable_logits).detach()
        lse_sum = torch.sum(lse_s, dim=0)
        lse_s /= lse_sum
        outputs *= lse_s.unsqueeze(-1).transpose(2, 3)
        return outputs.sum(0)

    def _dual_chunk_flash_attn_decoding_with_exp_sums(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        softmax_scale: float,
        alibi_slopes: Optional[torch.Tensor],
        causal: bool,
    ):
        out, softmax_lse = flash_attn_with_kvcache(
            q=query,
            k_cache=key_cache,
            v_cache=value_cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            softmax_scale=softmax_scale,
            alibi_slopes=alibi_slopes,
            causal=causal,
            return_softmax_lse=True,
        )
        mask = (cache_seqlens == 0)
        out[mask] = 0
        softmax_lse[mask] = -float("inf")
        return out, softmax_lse


def _vertical_slash_sparse_attention(
    query: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    key: torch.Tensor,  # [BATCH, N_HEADS, N_KV_CTX, D_HEAD]
    value: torch.Tensor,  # [BATCH, N_HEADS, N_KV_CTX, D_HEAD]
    v_idx: torch.Tensor,  # [BATCH, N_HEADS, NNZ_V]
    s_idx: torch.Tensor,  # [BATCH, N_HEADS, NNZ_S]
    softmax_scale: float,
    causal: bool = True,
    stage: str = "intra",
    block_size_M: int = 64,
    block_size_N: int = 64,
    vertical_indices_count: torch.Tensor = None,  # [N_HEADS,]
    slash_indices_count: torch.Tensor = None,
):
    if stage == "intra":
        assert causal
    else:
        assert not causal

    batch_size, num_heads, context_size, head_dim = query.shape
    _, _, kv_seq_len, _ = key.shape

    if head_dim not in [16, 32, 64, 128, 256, 512]:
        target_dim = 2**math.ceil(math.log2(head_dim)) - head_dim
        query = F.pad(query, [0, target_dim, 0, 0, 0, 0, 0, 0])
        key = F.pad(key, [0, target_dim, 0, 0, 0, 0, 0, 0])
        value = F.pad(value, [0, target_dim, 0, 0, 0, 0, 0, 0])

    v_idx = v_idx.to(torch.int32).reshape(
        (batch_size, num_heads, -1)).sort(dim=-1, descending=False)[0]
    s_idx = s_idx.to(torch.int32).reshape(
        (batch_size, num_heads, -1)).sort(dim=-1, descending=True)[0]
    q_seqlens = torch.tensor([context_size],
                             dtype=torch.int32,
                             device=query.device)
    kv_seqlens = torch.tensor([kv_seq_len],
                              dtype=torch.int32,
                              device=query.device)

    if vertical_indices_count is not None and slash_indices_count is not None:
        (
            block_count,
            block_offset,
            column_count,
            column_index,
        ) = ops.convert_vertical_slash_indexes_mergehead(
            q_seqlens, kv_seqlens, v_idx, s_idx, vertical_indices_count,
            slash_indices_count, context_size, block_size_M, block_size_N,
            causal)
    else:
        (
            block_count,
            block_offset,
            column_count,
            column_index,
        ) = ops.convert_vertical_slash_indexes(q_seqlens, kv_seqlens, v_idx,
                                               s_idx, context_size,
                                               block_size_M, block_size_N,
                                               causal)

    q = query.transpose(1, 2).contiguous()
    k = key.transpose(1, 2).contiguous()
    v = value.transpose(1, 2).contiguous()
    out, lse = sparse_attn_func(
        q,
        k,
        v,
        block_count,
        block_offset,
        column_count,
        column_index,
        causal=causal,
        softmax_scale=softmax_scale,
        return_softmax_lse=True,
    )
    out = out.transpose(1, 2).contiguous()
    softmax_lse = lse.reshape(*lse.shape, 1)
    return (out[..., :context_size, :head_dim],
            softmax_lse[..., :context_size, :])


def _sum_all_diagonal_matrix(mat: torch.tensor):
    h, n, m = mat.shape
    # Zero matrix used for padding
    zero_mat = torch.zeros((h, n, n), device=mat.device)
    # pads the matrix on left and right
    mat_padded = torch.cat((zero_mat, mat, zero_mat), -1)
    # Change the strides
    mat_strided = mat_padded.as_strided((1, n, n + m),
                                        (n * (2 * n + m), 2 * n + m + 1, 1))
    # Sums the resulting matrix's columns
    sum_diags = torch.sum(mat_strided, 1)
    return sum_diags[:, 1:]  # drop left bottom corner


def _get_block(block_table: torch.Tensor, block_size: int, begin: int,
               end: int):
    begin_block = begin // block_size
    end_block = (end - 1) // block_size + 1
    return block_table[begin_block:end_block]
