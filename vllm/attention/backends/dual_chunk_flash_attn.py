"""Attention layer with Flash and PagedAttention.

NOTE(woosuk): At the moment, this file includes a lot of duplicated code from
XFormers backend. The duplicated code will be removed once we use flash-attn or
flashinfer for all the attention operations.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import torch
from vllm_flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import (DualChunkAttentionBackend,
                                              DualChunkAttentionImpl,
                                              DualChunkAttentionMetadata)


class DualChunkFlashAttentionBackend(DualChunkAttentionBackend):

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "dual-chunk-flash-attn"

    @staticmethod
    def get_impl_cls() -> Type["DualChunkFlashAttentionImpl"]:
        return DualChunkFlashAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["DualChunkAttentionMetadata"]:
        return DualChunkFlashAttentionMetadata

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache = src_kv_cache[0]
        dst_key_cache = dst_kv_cache[0]
        ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)

        src_value_cache = src_kv_cache[1]
        dst_value_cache = dst_kv_cache[1]
        ops.swap_blocks(src_value_cache, dst_value_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        key_caches = [kv_cache[0] for kv_cache in kv_caches]
        value_caches = [kv_cache[1] for kv_cache in kv_caches]
        ops.copy_blocks(key_caches, value_caches, src_to_dists)


@dataclass
class DualChunkFlashAttentionMetadata(DualChunkAttentionMetadata):
    """Metadata for DualChunkFlashAttentionBackend.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """

    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[List[int]]
    # seq_lens stored as a tensor.
    seq_lens_tensor: Optional[torch.Tensor]

    # (batch_size,). The original prefill length per sequence.
    # None if it is decoding.
    prefill_original_seq_lens_tensor: Optional[torch.Tensor]

    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ----------------------|
    #                                   |-- query_len ---|

    # Maximum query length in the batch. None for decoding.
    max_query_len: Optional[int]
    # Maximum sequence length among prefill batch. 0 if there are decoding
    # requests only.
    max_prefill_seq_len: int
    # Maximum sequence length among decode batch. 0 if there are prefill
    # requests only.
    max_decode_seq_len: int
    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    query_start_loc: Optional[torch.Tensor]
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor]
    # (batch_size,) A tensor of context lengths (tokens that are computed
    # so far).
    context_lens_tensor: Optional[torch.Tensor]

    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    # E.g., [0, 1, 2] means tokens are stored in 0th, 1st, and 2nd blocks
    # in the kv cache. Each block can contain up to block_size tokens.
    # 2nd dimensions are padded up to max_blocks_per_seq if it is cuda-graph
    # captured.
    block_tables: Optional[torch.Tensor]

    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.
    use_cuda_graph: bool

    _cached_prefill_metadata: Optional[
        "DualChunkFlashAttentionMetadata"] = None
    _cached_decode_metadata: Optional["DualChunkFlashAttentionMetadata"] = None

    @property
    def prefill_metadata(self) -> Optional["DualChunkFlashAttentionMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            return self._cached_prefill_metadata

        assert self.seq_lens is not None
        assert self.seq_lens_tensor is not None
        assert self.prefill_original_seq_lens_tensor is not None
        assert self.query_start_loc is not None
        assert self.context_lens_tensor is not None
        assert self.block_tables is not None
        assert self.seq_start_loc is not None

        self._cached_prefill_metadata = DualChunkFlashAttentionMetadata(
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=self.slot_mapping[:self.num_prefill_tokens],
            seq_lens=self.seq_lens[:self.num_prefills],
            seq_lens_tensor=self.seq_lens_tensor[:self.num_prefills],
            prefill_original_seq_lens_tensor=self.
            prefill_original_seq_lens_tensor[:self.num_prefill_tokens],
            max_query_len=self.max_query_len,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_seq_len=0,
            query_start_loc=self.query_start_loc[:self.num_prefills + 1],
            seq_start_loc=self.seq_start_loc[:self.num_prefills + 1],
            context_lens_tensor=self.context_lens_tensor[:self.num_prefills],
            block_tables=self.block_tables[:self.num_prefills],
            use_cuda_graph=False,
        )
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["DualChunkFlashAttentionMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            return self._cached_decode_metadata
        assert self.block_tables is not None
        assert self.seq_lens_tensor is not None

        self._cached_decode_metadata = DualChunkFlashAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=self.slot_mapping[self.num_prefill_tokens:],
            seq_lens=None,
            seq_lens_tensor=self.seq_lens_tensor[self.num_prefills:],
            prefill_original_seq_lens_tensor=None,
            max_query_len=None,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
            query_start_loc=None,
            seq_start_loc=None,
            context_lens_tensor=None,
            block_tables=self.block_tables[self.num_prefills:],
            use_cuda_graph=self.use_cuda_graph,
        )
        return self._cached_decode_metadata


class DualChunkFlashAttentionImpl(DualChunkAttentionImpl):
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
        dual_chunk_attention_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError(
                "FlashAttention does not support block-sparse attention.")
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

        assert self.num_heads % self.num_kv_heads == 0
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

    def forward(
        self,
        query: torch.Tensor,
        query_succ: torch.Tensor,
        query_inter: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: DualChunkFlashAttentionMetadata,
        kv_scale: float = 1.0,
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
        # NOTE(woosuk): FlashAttention does not support FP8 KV cache.
        assert kv_scale == 1.0, "kv_scale is not supported in FlashAttention."

        assert (
            query_succ is not None and query_inter is not None
        ), "query_succ and query_inter are required in Dual Chunk Attention."

        num_tokens, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        query_succ = query_succ.view(-1, self.num_heads, self.head_size)
        query_inter = query_inter.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        if self.original_max_position_embeddings > 0:
            if prefill_meta := attn_metadata.prefill_metadata:
                assert prefill_meta.query_start_loc is not None
                assert prefill_meta.prefill_original_seq_lens_tensor is not None

                current_start = 0
                query_start_loc_cpu = prefill_meta.query_start_loc.cpu()
                for i in range(
                        0, prefill_meta.prefill_original_seq_lens_tensor.
                        shape[0]):
                    current_end = (current_start +
                                   (query_start_loc_cpu[i + 1] -
                                    query_start_loc_cpu[i]).item())
                    mscale = (
                        0.1 *
                        (prefill_meta.prefill_original_seq_lens_tensor[i] /
                         self.original_max_position_embeddings).log() +
                        1.0).clip(min=1)
                    key[current_start:current_end].mul_(mscale)
                    current_start = current_end
                assert current_end <= attn_metadata.num_prefill_tokens
            if decode_meta := attn_metadata.decode_metadata:
                mscale = (
                    0.1 * torch.log(decode_meta.seq_lens_tensor /
                                    self.original_max_position_embeddings) +
                    1.0).clip(min=1)
                key[attn_metadata.num_prefill_tokens:].mul_(
                    mscale.unsqueeze(-1).unsqueeze(-1))

        if kv_cache is not None:
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
        key = key[:num_prefill_tokens]
        value = value[:num_prefill_tokens]

        assert query.shape[0] == num_prefill_tokens
        assert decode_query.shape[0] == num_decode_tokens

        if prefill_meta := attn_metadata.prefill_metadata:
            # Prompt run.
            if (kv_cache is None or prefill_meta.block_tables is None
                    or prefill_meta.block_tables.numel() == 0):
                # normal attention
                # When block_tables are not filled, it means q and k are the
                # prompt, and they have the same length.
                out = self._bruteforce_dynamic_chunk_flash_attn_varlen_func(
                    q=query,
                    q_succ=query_succ,
                    q_inter=query_inter,
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
                    block_table=None,
                    chunk_size=self.chunk_size,
                    local_size=self.local_size,
                    original_max_position_embeddings=self.
                    original_max_position_embeddings,
                    prefill_original_seq_lens_tensor=prefill_meta.
                    prefill_original_seq_lens_tensor,
                )
                assert output[:num_prefill_tokens].shape == out.shape
                output[:num_prefill_tokens] = out
            else:
                # prefix-enabled attention
                assert prefill_meta.seq_lens is not None
                max_seq_len = max(prefill_meta.seq_lens)
                output[:num_prefill_tokens] = (
                    self._bruteforce_dynamic_chunk_flash_attn_varlen_func(
                        q=query,
                        q_succ=query_succ,
                        q_inter=query_inter,
                        k=key_cache,
                        v=value_cache,
                        cu_seqlens_q=prefill_meta.query_start_loc,
                        max_seqlen_q=prefill_meta.max_query_len,
                        cu_seqlens_k=prefill_meta.seq_start_loc,
                        max_seqlen_k=max_seq_len,
                        softmax_scale=self.scale,
                        causal=True,
                        window_size=(-1, -1),
                        alibi_slopes=self.alibi_slopes,
                        block_table=prefill_meta.block_tables,
                        chunk_size=self.chunk_size,
                        local_size=self.local_size,
                        original_max_position_embeddings=self.
                        original_max_position_embeddings,
                        prefill_original_seq_lens_tensor=prefill_meta.
                        prefill_original_seq_lens_tensor,
                    ))
        if decode_meta := attn_metadata.decode_metadata:
            # Decoding run.
            output[num_prefill_tokens:] = (
                self._bruteforce_dynamic_chunk_pageattention_forward_decode(
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
                ).squeeze(1))

        # Reshape the output tensor.
        return output.view(num_tokens, hidden_size)

    def _bruteforce_dynamic_chunk_flash_attn_func(
        self,
        q,
        q_succ,
        q_inter,
        k,
        v,
        block_table,
        softmax_scale,
        chunk_size,
        local_size,
        original_max_position_embeddings,
        current_prefill_original_seq_lens_tensor,
        k_length,
    ):

        def do_flash_attn(
            query_states,
            key_states,
            value_states,
            causal=True,
            block_table=None,
            max_seqlen_k=None,
        ):
            if max_seqlen_k is None:
                max_seqlen_k = key_states.shape[0]

            output, softmax_lse, _ = flash_attn_varlen_func(
                q=query_states,
                k=key_states,
                v=value_states,
                softmax_scale=softmax_scale,
                cu_seqlens_q=torch.tensor(
                    [0, query_states.shape[0]],
                    dtype=torch.int32,
                    device=query_states.device,
                ),
                max_seqlen_q=query_states.shape[0],
                cu_seqlens_k=torch.tensor(
                    [0, max_seqlen_k],
                    dtype=torch.int32,
                    device=query_states.device,
                ),
                max_seqlen_k=max_seqlen_k,
                causal=causal,
                block_table=block_table,
                return_attn_probs=True,
            )
            return output, softmax_lse

        def merge_attn_outputs(flash_results):
            attn_outputs_all = []
            for flash_per_chunk in flash_results:
                if len(flash_per_chunk) == 1:
                    attn_outputs_all.append(flash_per_chunk[0][0])
                    continue
                attn_outputs = torch.stack([
                    flash_attn_output[0]
                    for flash_attn_output in flash_per_chunk
                ])
                logits = torch.stack([
                    flash_attn_output[1]
                    for flash_attn_output in flash_per_chunk
                ]).to(torch.float32)
                max_logits = torch.max(logits, dim=0).values
                stable_logits = logits - max_logits.unsqueeze(0)
                lse_s = torch.exp(stable_logits).detach()
                lse_sum = torch.sum(lse_s, dim=0)
                lse_s /= lse_sum
                attn_outputs *= lse_s.unsqueeze(-1).transpose(2, 3).squeeze(1)
                attn_outputs_all.append(attn_outputs.sum(dim=0))
            return torch.cat(attn_outputs_all, dim=0)

        def get_block(begin, end):
            return block_table[:,
                               begin // block_size:(end - 1) // block_size + 1]

        flash_results = []
        chunk_len = chunk_size - local_size
        if block_table is not None:
            block_size = v.shape[1]
            if chunk_len % block_size != 0:
                raise ValueError("chunk_len must be divisible by block_size.")
        else:
            block_size = 1

        if original_max_position_embeddings > 0:
            mscale = max(
                0.1 * (current_prefill_original_seq_lens_tensor[0] /
                       original_max_position_embeddings).log() + 1.0,
                1.0,
            )
            softmax_scale = softmax_scale * mscale

        begin = k_length - q.shape[0]

        while begin < k_length:
            flash_per_chunk = []

            prev_chunk_end_pos = (begin // chunk_len) * chunk_len
            next_chunk_end_pos = prev_chunk_end_pos + chunk_len
            end = min(next_chunk_end_pos, k_length)
            qbegin = begin - (k_length - q.shape[0])
            qend = end - (k_length - q.shape[0])

            q_states_intra = q[qbegin:qend]
            if block_table is not None:
                block_table_intra = get_block(prev_chunk_end_pos, end)
                flash_result = do_flash_attn(
                    q_states_intra,
                    k,
                    v,
                    block_table=block_table_intra,
                    max_seqlen_k=end - prev_chunk_end_pos,
                )
            else:
                k_states_intra = k[prev_chunk_end_pos:end]
                v_states_intra = v[prev_chunk_end_pos:end]
                flash_result = do_flash_attn(q_states_intra, k_states_intra,
                                             v_states_intra)
            flash_per_chunk.append(flash_result)

            if prev_chunk_end_pos - chunk_len >= 0:
                q_states_succ = q_succ[qbegin:qend]
                if block_table is not None:
                    block_table_succ = get_block(
                        prev_chunk_end_pos - chunk_len, prev_chunk_end_pos)
                    flash_result = do_flash_attn(
                        q_states_succ,
                        k,
                        v,
                        False,
                        block_table=block_table_succ,
                        max_seqlen_k=chunk_len,
                    )
                else:
                    k_states_succ = k[prev_chunk_end_pos -
                                      chunk_len:prev_chunk_end_pos]
                    v_states_succ = v[prev_chunk_end_pos -
                                      chunk_len:prev_chunk_end_pos]
                    flash_result = do_flash_attn(q_states_succ, k_states_succ,
                                                 v_states_succ, False)
                flash_per_chunk.append(flash_result)

            if prev_chunk_end_pos - chunk_len * 2 >= 0:
                q_states_inter = q_inter[qbegin:qend]

                if block_table is not None:
                    block_table_inter = get_block(
                        0, prev_chunk_end_pos - chunk_len)
                    flash_result = do_flash_attn(
                        q_states_inter,
                        k,
                        v,
                        False,
                        block_table=block_table_inter,
                        max_seqlen_k=prev_chunk_end_pos - chunk_len,
                    )
                else:
                    k_states_inter = k[:prev_chunk_end_pos - chunk_len]
                    v_states_inter = v[:prev_chunk_end_pos - chunk_len]
                    flash_result = do_flash_attn(q_states_inter,
                                                 k_states_inter,
                                                 v_states_inter, False)
                flash_per_chunk.append(flash_result)

            begin = end
            flash_results.append(flash_per_chunk)

        attn_output = merge_attn_outputs(flash_results)

        return attn_output

    def _bruteforce_dynamic_chunk_flash_attn_varlen_func(
        self,
        q,
        q_succ,
        q_inter,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        block_table,
        chunk_size,
        local_size,
        original_max_position_embeddings,
        prefill_original_seq_lens_tensor,
    ):

        if alibi_slopes is not None:
            raise ValueError(
                "Native Dynamic Chunk Attention does not support alibi_slopes")
        if not causal:
            raise ValueError(
                "Native Dynamic Chunk Attention does not support causal=False")
        if window_size != (-1, -1):
            raise ValueError(
                "Native Dynamic Chunk Attention does not support window_size")

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
            if block_table is None:
                current_k = k[ks:ke]
                current_v = v[ks:ke]
                current_block_table = None
                current_prefill_original_seq_lens_tensor = (
                    prefill_original_seq_lens_tensor[i:i + 1])
            else:
                current_block_table = block_table[i:i + 1]
                current_prefill_original_seq_lens_tensor = (
                    prefill_original_seq_lens_tensor[i:i + 1])
                current_k = k
                current_v = v

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

            current_output = self._bruteforce_dynamic_chunk_flash_attn_func(
                current_q,
                current_q_succ,
                current_q_inter,
                current_k,
                current_v,
                current_block_table,
                softmax_scale,
                chunk_size,
                local_size,
                original_max_position_embeddings,
                current_prefill_original_seq_lens_tensor,
                ke - ks,
            )
            all_outputs.append(current_output)

        return torch.cat(all_outputs, dim=0)

    def _bruteforce_dynamic_chunk_pageattention_forward_decode(
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
    ):
        assert causal
        batch_size = block_table.shape[0]
        block_size = value_cache.shape[1]
        chunk_len = chunk_size - local_size
        if chunk_len % block_size != 0:
            raise ValueError("chunk_len must be divisible by block_size.")
        chunk_num_curr = (cache_seqlens - 1) // chunk_len

        if original_max_position_embeddings > 0:
            mscale = (
                0.1 *
                torch.log(cache_seqlens / original_max_position_embeddings) +
                1.0).clip(min=1)
            query = (query * mscale.view(-1, 1, 1, 1)).to(
                query.dtype
            )  # possible for numerical issue, need to fused in the kernel
            query_succ = (query_succ * mscale.view(-1, 1, 1, 1)).to(
                query.dtype)
            query_inter = (query_inter * mscale.view(-1, 1, 1, 1)).to(
                query.dtype)

        outputs_list = []
        softmax_lses_list = []

        # intra-attention
        seq_lens_intra = cache_seqlens - chunk_num_curr * chunk_len
        max_seq_len_intra = seq_lens_intra.max().item()
        block_table_intra = torch.zeros(
            batch_size,
            (max_seq_len_intra - 1) // block_size + 1,
            dtype=block_table.dtype,
            device=block_table.device,
        )
        for i in range(batch_size):
            st = chunk_num_curr[i] * chunk_len // block_size
            ed = min(
                st + (max_seq_len_intra - 1) // block_size + 1,
                (cache_seqlens[i] - 1) // block_size + 1,
            )
            block_table_intra[i, :ed - st] = block_table[i, st:ed]
        intra_output, intra_softmax_lse = (
            self._pagedattention_forward_decode_with_exp_sums(
                query,
                key_cache,
                value_cache,
                block_table_intra,
                seq_lens_intra,
                softmax_scale,
                alibi_slopes,
                causal=False,
            ))
        outputs_list.append(intra_output)
        softmax_lses_list.append(intra_softmax_lse)

        # succ-attention
        seq_lens_succ = (chunk_num_curr -
                         (chunk_num_curr - 1).clip(min=0)) * chunk_len
        max_seq_len_succ = seq_lens_succ.max().item()
        if max_seq_len_succ:
            block_table_succ = torch.zeros(
                batch_size,
                (max_seq_len_succ - 1) // block_size + 1,
                dtype=block_table.dtype,
                device=block_table.device,
            )
            for i in range(batch_size):
                st = ((chunk_num_curr[i] - 1).clip(min=0) * chunk_len //
                      block_size)
                ed = min(
                    st + (max_seq_len_succ - 1) // block_size + 1,
                    (cache_seqlens[i] - 1) // block_size + 1,
                )
                block_table_succ[i, :ed - st] = block_table[i, st:ed]
            succ_output, succ_softmax_lse = (
                self._pagedattention_forward_decode_with_exp_sums(
                    query_succ,
                    key_cache,
                    value_cache,
                    block_table_succ,
                    seq_lens_succ,
                    softmax_scale,
                    alibi_slopes,
                    causal=False,
                ))
            outputs_list.append(succ_output)
            softmax_lses_list.append(succ_softmax_lse)

        # inter-attention
        seq_lens_inter = (chunk_num_curr - 1).clip(min=0) * chunk_len
        max_seq_len_inter = seq_lens_inter.max().item()
        if max_seq_len_inter:
            inter_output, succ_softmax_lse = (
                self._pagedattention_forward_decode_with_exp_sums(
                    query_inter,
                    key_cache,
                    value_cache,
                    block_table[:, :max_seq_len_inter],
                    seq_lens_inter,
                    softmax_scale,
                    alibi_slopes,
                    causal=False,
                ))
            outputs_list.append(inter_output)
            softmax_lses_list.append(succ_softmax_lse)

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

    def _pagedattention_forward_decode_with_exp_sums(
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
            query,
            key_cache,
            value_cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            softmax_scale=softmax_scale,
            alibi_slopes=alibi_slopes,
            causal=causal,
            return_softmax=True,
        )
        cache_seqlens_cpu = cache_seqlens.cpu()
        for i in range(cache_seqlens.shape[0]):
            if cache_seqlens_cpu[i] == 0:
                softmax_lse[i].fill_(-float("inf"))
                out[i].fill_(0)

        return out, softmax_lse
