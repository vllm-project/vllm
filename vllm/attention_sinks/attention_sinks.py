"""
Attention computation layer with vLLM-specific attention sink logic,
as described in https://github.com/mit-han-lab/streaming-llm.
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from vllm import _custom_ops as ops
from vllm.attention import Attention, AttentionMetadata
from vllm.attention.ops.paged_attn import PagedAttention
from vllm.attention.selector import _Backend
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding


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
        kv_scale: float,
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
        self.kv_scale = kv_scale
        self.rotary_emb = rotary_emb_layer
        self.use_alibi = rotary_emb_layer is None
        self.attn = attn_layer
        self.chunked_prefill_enabled = chunked_prefill_enabled
        self.positions = None

        if attn_backend not in (_Backend.XFORMERS, _Backend.FLASH_ATTN):
            raise NotImplementedError(
                'Attention sinks is only supported for '
                'XFormers and FlashAttention currently.')

    def save_positions(self, positions: torch.Tensor):
        self.positions = positions

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: torch.Tensor,
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
        un-roped keys (no positional embedding applied). At every forward,
        we apply rope to ALL keys right before computing attention. This
        extra work causes a significant drop in tokens/sec when using
        attention sinks with rope models.
        
        Pseudocode:
        - clone current keys (k_original)
        - if non-chunked prefill:
            - apply rope to current q, k
            - compute attention in kernel
            - write current original keys into key cache
            - return attention output
        - else (decode and chunked prefills):
            - for each sequence in batch:
                - read past keys from cache
                - apply rope to past keys based on their positions
                - write past keys back to cache
                - if seq len >= model context len:
                    - edit seq lens metadata
                    - cap positions of current q, k
            - apply rope to current q, k
            - compute attention in kernel
            - write past and current original keys to cache
            - return attention output
        
        self-note: q, k, v all have shape [num_tokens, num_heads * head_size]
        """

        # original keys will be written to key cache after attention
        k_original = k.clone()
        
        if self.attn_backend == _Backend.FLASH_ATTN:
            # key cache shape: [num_blocks, block_size, num_heads, head_size]
            key_cache, value_cache = kv_cache
        elif self.attn_backend == _Backend.XFORMERS:
            # key cache shape: [num_blocks, num_heads, head_size/x, block_size, x]
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_dim)

        if (attn_metadata.prefill_metadata is not None
            and not self.chunked_prefill_enabled):
            # non-chunked prefill (entire prompt)
            assert attn_metadata.decode_metadata is None
            
            q, k = self.rotary_emb(self.positions, q, k)
            attn_output = self.attn(q, k, v, kv_cache, attn_metadata)

            k_original = k_original.view(-1, self.num_kv_heads, self.head_dim)
            v = v.view(-1, self.num_kv_heads, self.head_dim)

            # put original pre-rotated keys back in cache
            if self.attn_backend == _Backend.FLASH_ATTN:
                ops.reshape_and_cache_flash(
                    k_original,
                    v,
                    key_cache,
                    value_cache,
                    attn_metadata.slot_mapping.flatten(),
                    self.kv_cache_dtype,
                )
            elif self.attn_backend == _Backend.XFORMERS:
                PagedAttention.write_to_paged_cache(
                    k_original,
                    v,
                    key_cache,
                    value_cache,
                    attn_metadata.slot_mapping,
                    self.kv_cache_dtype,
                    self.kv_scale
                )

            return attn_output
        
        # else, decode only or decode + chunked prefill
        if not self.chunked_prefill_enabled:
            assert attn_metadata.num_prefills == 0
        
        device = q.device
        block_size = self.block_size
        model_context_len = self.model_context_len

        block_tables_tensor = attn_metadata.block_tables
        context_lens_tensor = attn_metadata.context_lens_tensor

        # batch size = num sequences
        batch_size = context_lens_tensor.shape[0]

        # cache phys_bnums
        if hasattr(attn_metadata, 'phys_bnums_list'):
            phys_bnums_list = attn_metadata.phys_bnums_list
        else:
            phys_bnums_list: List[torch.Tensor] = [None] * batch_size
        
        original_keys: List[Tuple[torch.Tensor]] = [None] * batch_size
        
        # loop through each sequence
        for i in range(batch_size):
            num_past_tokens = context_lens_tensor[i].item()
            if num_past_tokens == 0: continue
            within_context_len = num_past_tokens < model_context_len
            block_table = block_tables_tensor[i]
            
            num_blocks = min(num_past_tokens // block_size + 1, model_context_len // block_size)
            if hasattr(attn_metadata, 'phys_bnums_list'):
                phys_bnums = phys_bnums_list[i]
            else:
                phys_bnums = block_table[:num_blocks - 1]
                phys_bnums_list[i] = phys_bnums
            
            rem = num_past_tokens % block_size
            rem_phys_bnum = block_table[num_blocks - 1]
            
            # read unrotated keys from cache
            # FA shape: [len(phys_bnums), block_size, num_heads, head_size]
            # XF shape: [len(phys_bnums), num_heads, head_size/x, block_size, x]
            full_past_keys = torch.index_select(key_cache, 0, phys_bnums)
            if self.attn_backend == _Backend.FLASH_ATTN:
                rem_past_keys = key_cache[rem_phys_bnum, :rem, :, :]
            elif self.attn_backend == _Backend.XFORMERS:
                rem_past_keys = key_cache[rem_phys_bnum, :, :, :rem, :]
            original_keys[i] = (full_past_keys.clone(), rem_past_keys.clone())
            
            # use positions within cache (capped by context length)
            pos_start = 0 if within_context_len else 2 * block_size - 1 - rem
            pos_end = min(num_past_tokens, model_context_len - 1)
            pos = torch.arange(pos_start, pos_end, device=device)
            if not within_context_len:
                # pos (for context len 4096): [0, 16) + [31 - rem, 4095)
                pos_sink = torch.arange(0, block_size, device=device)
                pos = torch.cat((pos_sink, pos))
            
            # reshape for rotary embedding kernel
            if self.attn_backend == _Backend.FLASH_ATTN:
                full_past_keys = full_past_keys.flatten(0, 1)
            elif self.attn_backend == _Backend.XFORMERS:
                full_past_keys = full_past_keys.permute((0, 3, 1, 2, 4)).flatten(0, 1)
                rem_past_keys = rem_past_keys.permute((2, 0, 1, 3))
                
            # combine full and remainder keys
            full_past_keys = torch.cat((full_past_keys, rem_past_keys), dim=0)
            full_past_keys = full_past_keys.flatten(1, -1)
            # shape: [pos_end - pos_start, num_heads * head_size]
            
            # rotate keys with new positions
            dummy_q = torch.zeros_like(full_past_keys)
            _, full_past_keys = self.rotary_emb(pos, dummy_q, full_past_keys)
            
            # reshape for writing back to cache
            if self.attn_backend == _Backend.FLASH_ATTN:
                full_past_keys = full_past_keys.unflatten(1, (key_cache.shape[2], key_cache.shape[3]))
            elif self.attn_backend == _Backend.XFORMERS:
                full_past_keys = full_past_keys.unflatten(1, (key_cache.shape[1], key_cache.shape[2], key_cache.shape[4]))
            
            # split into full and remainder keys
            full_past_keys, rem_past_keys = torch.split(full_past_keys, [len(phys_bnums) * block_size, rem])
            full_past_keys = full_past_keys.unflatten(0, (len(phys_bnums), block_size))
            
            # write rotated keys to cache for attention computation
            if self.attn_backend == _Backend.FLASH_ATTN:
                key_cache.index_put_((phys_bnums,), full_past_keys)
                key_cache[rem_phys_bnum, :rem, :, :] = rem_past_keys
            elif self.attn_backend == _Backend.XFORMERS:
                full_past_keys = full_past_keys.permute((0, 2, 3, 1, 4))
                rem_past_keys = rem_past_keys.permute((1, 2, 0, 3))
                key_cache.index_put_((phys_bnums,), full_past_keys)
                key_cache[rem_phys_bnum, :, :, :rem, :] = rem_past_keys
            
            if not within_context_len:
                # must be decode
                # => cap number of tokens to consider with model context len
                attn_metadata.seq_lens_tensor[i] = model_context_len - block_size + rem + 1
                self.positions[i] = model_context_len - 1

        if not hasattr(attn_metadata, 'phys_bnums_list'):
            attn_metadata.phys_bnums_list = phys_bnums_list

        # compute attention in kernel
        q, k = self.rotary_emb(self.positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
                    
        # put original pre-rotated keys back in cache
        for i in range(batch_size):
            num_past_tokens = context_lens_tensor[i].item()
            if num_past_tokens == 0: continue
            within_context_len = num_past_tokens < model_context_len
            block_table = block_tables_tensor[i]
            
            num_blocks = min(num_past_tokens // block_size + 1, model_context_len // block_size)
            phys_bnums = phys_bnums_list[i]

            rem = num_past_tokens % block_size
            rem_phys_bnum = block_table[num_blocks - 1]

            full_past_keys, rem_past_keys = original_keys[i]
            key_cache.index_put_((phys_bnums,), full_past_keys)
            if self.attn_backend == _Backend.FLASH_ATTN:
                key_cache[rem_phys_bnum, :rem, :, :] = rem_past_keys
            elif self.attn_backend == _Backend.XFORMERS:
                key_cache[rem_phys_bnum, :, :, :rem, :] = rem_past_keys
        
        k_original = k_original.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        if self.attn_backend == _Backend.FLASH_ATTN:
            ops.reshape_and_cache_flash(
                k_original,
                v,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping.flatten(),
                self.kv_cache_dtype,
            )
        elif self.attn_backend == _Backend.XFORMERS:
            PagedAttention.write_to_paged_cache(
                k_original,
                v,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                self.kv_scale
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
            num_past_tokens = attn_metadata.context_lens_tensor[i].item()
            if num_past_tokens < self.model_context_len: continue

            # cap number of tokens to consider with model context len
            rem = num_past_tokens % self.block_size
            attn_metadata.seq_lens_tensor[i] = self.model_context_len - self.block_size + rem + 1

        # compute attention in kernel
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        return attn_output
