"""
Attention computation layer with vLLM-specific attention sink logic,
as described in https://github.com/mit-han-lab/streaming-llm.
"""
from typing import List, Optional, Tuple
import time

import torch
import torch.nn as nn

from vllm import _custom_ops as ops
from vllm.attention import Attention, AttentionMetadata
from vllm.attention.ops.paged_attn import PagedAttention
from vllm.attention.selector import _Backend
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding


_SUPPORTED_ATTN_BACKENDS = (
    _Backend.FLASH_ATTN,
    _Backend.XFORMERS,
    _Backend.FLASHINFER,
)


class StreamingAttentionSink(nn.Module):
    """Replacement for Attention layer when attention sinks are enabled."""

    def __init__(
        self,
        model_context_len: int,
        block_size: int,
        kv_cache_dtype: str,
        attn_backend: _Backend,
        num_kv_heads: int,
        head_dim: int,
        rotary_emb_layer: Optional[RotaryEmbedding],
        attn_layer: Attention,
        chunked_prefill_enabled: bool
    ) -> None:
        super().__init__()
        self.model_context_len = model_context_len
        self.block_size = block_size
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_backend = attn_backend
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rotary_emb = rotary_emb_layer
        self.use_alibi = rotary_emb_layer is None
        self.attn = attn_layer
        self.chunked_prefill_enabled = chunked_prefill_enabled
        self.positions = None

        if attn_backend not in _SUPPORTED_ATTN_BACKENDS:
            raise NotImplementedError(
                'Attention sinks is only supported for '
                'FlashAttention, XFormers, and FlashInfer currently.')

    def save_positions(self, positions: torch.Tensor):
        self.positions = positions

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """Replaces the `self.attn(...)` call in model attention modules."""

        if kv_cache is None:
            # ModelRunner.profile_run
            if not self.use_alibi:
                q, k = self.rotary_emb(self.positions, q, k)
            return self.attn(q, k, v, None, attn_metadata)

        if self.use_alibi:
            return self._forward_alibi(q, k, v, kv_cache, attn_metadata)
        else:
            if self.attn_backend == _Backend.FLASHINFER:
                return self._forward_flashinfer(q, k, v, kv_cache, attn_metadata)
            else:
                return self._forward_rope(q, k, v, kv_cache, attn_metadata)

    def _forward_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """
        The key idea is that between iterations, the KV cache will contain
        pre-rope keys (no positional embedding applied). At every forward,
        we apply rope to ALL keys right before computing attention. This
        extra work causes a significant drop in tokens/sec when using
        attention sinks with rope models.

        Pseudocode:
        - clone current keys (k_original)
        - if non-chunked prefill:
            - apply rope to current q, k
            - compute attention in kernel
            - write current original k into key cache
            - return attention output
        - else (decode and chunked prefills):
            - for each sequence in batch:
                - if seq len >= model context len:
                    - edit seq lens metadata
                    - cap positions of current q, k
            - read pre-rope past keys from cache
            - apply rope to past keys based on their cache positions
            - write roped past keys back to cache
            - apply rope to current q, k
            - compute attention in kernel
            - write past and current original keys to cache
            - return attention output
        
        self-note: q, k, v all have shape [num_tokens, num_heads * head_size]
        """
        # original keys will be written to key cache after attention
        k_original = k.clone()
        device = q.device

        if self.attn_backend == _Backend.XFORMERS:
            # key cache shape: [num_blocks, num_heads, head_size/x, block_size, x]
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_dim)
        else:  # flashattn
            # key cache shape: [num_blocks, block_size, num_heads, head_size]
            key_cache, value_cache = kv_cache

        if (attn_metadata.prefill_metadata is not None
            and not self.chunked_prefill_enabled):
            # non-chunked prefill (entire prompt)
            assert attn_metadata.decode_metadata is None
            
            q, k = self.rotary_emb(self.positions, q, k)
            attn_output = self.attn(q, k, v, kv_cache, attn_metadata)

            k_original = k_original.view(-1, self.num_kv_heads, self.head_dim)
            v = v.view(-1, self.num_kv_heads, self.head_dim)

            # put original pre-rotated keys back in cache
            if self.attn_backend == _Backend.XFORMERS:
                PagedAttention.write_to_paged_cache(
                    k_original,
                    v,
                    key_cache,
                    value_cache,
                    attn_metadata.slot_mapping,
                    self.kv_cache_dtype,
                    self.attn._k_scale,
                    self.attn._v_scale,
                )
            else:  # flashattn
                ops.reshape_and_cache_flash(
                    k_original,
                    v,
                    key_cache,
                    value_cache,
                    attn_metadata.slot_mapping.flatten(),
                    self.kv_cache_dtype,
                    self.attn._k_scale,
                    self.attn._v_scale,
                )

            return attn_output
        
        # else, decode only or decode + chunked prefill
        if not self.chunked_prefill_enabled:
            assert attn_metadata.num_prefills == 0
        
        block_size = self.block_size
        model_context_len = self.model_context_len

        block_tables_tensor: torch.Tensor = attn_metadata.block_tables
        context_lens_tensor: torch.Tensor = attn_metadata.context_lens_tensor

        all_block_table: List[torch.Tensor] = []
        all_pos: List[torch.Tensor] = []

        # batch size = num sequences
        batch_size = context_lens_tensor.shape[0]
        for i in range(batch_size):
            num_past_tokens = context_lens_tensor[i]
            if num_past_tokens == 0:
                # chunked prefill first chunk case
                continue

            block_table = block_tables_tensor[i]
            num_blocks = min(
                num_past_tokens // block_size + 1,
                model_context_len // block_size
            )
            # get first num_blocks in case of block table padding
            if num_blocks < len(block_table):
                block_table = block_table[:num_blocks]
            all_block_table.append(block_table)

            # use positions within cache (capped by context length)
            pos = torch.arange(0, len(block_table) * block_size, device=device)
            all_pos.append(pos)

            if num_past_tokens >= model_context_len:
                # must be decode
                # => cap number of tokens to consider with model context len
                rem = num_past_tokens % block_size
                attn_metadata.seq_lens_tensor[i] = model_context_len - block_size + rem + 1
                self.positions[i] = model_context_len - block_size + rem

        if len(all_block_table) > 0:
            # get all block tables and past positions into single tensor
            # for batched torch ops, which don't need to be in for-loop
            all_block_table = torch.cat(all_block_table)
            all_pos = torch.cat(all_pos)

            # read unrotated keys from cache
            # FA shape: [len(all_block_table), block_size, num_heads, head_size]
            # XF shape: [len(all_block_table), num_heads, head_size/x, block_size, x]
            prerope_keys = torch.index_select(key_cache, 0, all_block_table)

            # copy will be used to write back to cache after attn computation
            prerope_keys_copy = prerope_keys.clone()

            # reshape for rotary embedding kernel
            if self.attn_backend == _Backend.XFORMERS:
                prerope_keys = prerope_keys.permute((0, 3, 1, 2, 4))
            prerope_keys = prerope_keys.flatten(0, 1).flatten(1, -1)
            # shape: [len(all_block_table) * block_size, num_heads * head_size]

            # rotate keys with new positions
            dummy_q = torch.zeros_like(prerope_keys)
            _, roped_keys = self.rotary_emb(all_pos, dummy_q, prerope_keys)

            # reshape for writing back to cache
            if self.attn_backend == _Backend.XFORMERS:
                roped_keys = roped_keys.unflatten(1, (key_cache.shape[1], key_cache.shape[2], key_cache.shape[4]))
            else:  # flashattn
                roped_keys = roped_keys.unflatten(1, (key_cache.shape[2], key_cache.shape[3]))
            roped_keys = roped_keys.unflatten(0, (len(all_block_table), block_size))
            if self.attn_backend == _Backend.XFORMERS:
                roped_keys = roped_keys.permute((0, 2, 3, 1, 4))

            # write rotated keys to cache for attention computation
            key_cache.index_put_((all_block_table,), roped_keys)

        # compute attention in kernel
        q, k = self.rotary_emb(self.positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)

        if len(all_block_table) > 0:
            # put original pre-rotated keys back in cache
            key_cache.index_put_((all_block_table,), prerope_keys_copy)

        k_original = k_original.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        if self.attn_backend == _Backend.XFORMERS:
            PagedAttention.write_to_paged_cache(
                k_original,
                v,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                self.attn._k_scale,
                self.attn._v_scale,
            )
        else:  # flashattn
            ops.reshape_and_cache_flash(
                k_original,
                v,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping.flatten(),
                self.kv_cache_dtype,
                self.attn._k_scale,
                self.attn._v_scale,
            )

        return attn_output

    def _forward_alibi(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """
        _forward_alibi is much simpler that _forward_rope because no positional
        embedding needs to be applied. Thus, the only extra work we have to do
        before computing attention is to cap the seq lens metadata for
        sequences past the model's context length.
        """

        if (attn_metadata.prefill_metadata is not None
            and not self.chunked_prefill_enabled):
            # non-chunked prefill (entire prompt)
            assert attn_metadata.decode_metadata is None
            return self.attn(q, k, v, kv_cache, attn_metadata)
        
        # else, decode only or decode + chunked prefill
        if not self.chunked_prefill_enabled:
            assert attn_metadata.num_prefills == 0

        # batch size = num sequences
        batch_size = attn_metadata.context_lens_tensor.shape[0]

        for i in range(batch_size):
            num_past_tokens = attn_metadata.context_lens_tensor[i]
            if num_past_tokens < self.model_context_len: continue

            # cap number of tokens to consider with model context len
            rem = num_past_tokens % self.block_size
            attn_metadata.seq_lens_tensor[i] = self.model_context_len - self.block_size + rem + 1

        # compute attention in kernel
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        return attn_output

    def _forward_flashinfer(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # NOTE: flash infer currently doesn't support chunked prefill
        
        # compute attention in kernel
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        return attn_output
