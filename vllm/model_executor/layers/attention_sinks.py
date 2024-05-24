"""
Attention computation layer with attention sink logic,
as described in https://github.com/mit-han-lab/streaming-llm.
Currently works for Llama (should eventually work for all RoPE models).
"""
from typing import List, Tuple

import torch
import torch.nn as nn

from vllm._C import cache_ops
from vllm.attention import AttentionMetadata
from vllm.attention.ops.paged_attn import PagedAttention
from vllm.attention.selector import _Backend
from vllm.utils import make_tensor_with_pad


class StreamingAttentionSink(nn.Module):
    def __init__(
        self,
        model_context_len: int,
        block_size: int,
        kv_cache_dtype: str,
        attn_backend: _Backend,
        num_kv_heads: int,
        head_dim: int,
        kv_scale: float,
        rotary_emb_layer, # what if model doesn't use rope?
        attn_layer,
        output_layer,
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
        self.attn = attn_layer
        self.output = output_layer

        if attn_backend not in (_Backend.XFORMERS, _Backend.FLASH_ATTN):
            raise NotImplementedError(
                'Attention sinks is only supported for '
                'XFormers and FlashAttention currently.')

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # q k v all have shape [num_tokens, num_heads * head_size] i.e. [1, 4096] for decode
        if kv_cache is not None:
            if self.attn_backend == _Backend.FLASH_ATTN:
            # key cache shape: [num_blocks, block_size, num_heads, head_size]
                key_cache, value_cache = kv_cache
            elif self.attn_backend == _Backend.XFORMERS:
            # key cache shape: [num_blocks, num_heads, head_size/x, block_size, x]
                key_cache, value_cache = PagedAttention.split_kv_cache(
                    kv_cache, self.num_kv_heads, self.head_dim)
        
        # what if metadata has both prefill and decode?
        if attn_metadata.prefill_metadata is not None:
            k_original = k.clone()
            q, k = self.rotary_emb(positions, q, k)
            attn_output = self.attn(q, k, v, kv_cache, attn_metadata, self.kv_scale)
            
            if kv_cache is not None:
                k_original = k_original.view(-1, self.num_kv_heads, self.head_dim)
                v = v.view(-1, self.num_kv_heads, self.head_dim)

                if self.attn_backend == _Backend.FLASH_ATTN:
                    cache_ops.reshape_and_cache_flash(
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

            output, _ = self.output(attn_output)
            return output

        elif attn_metadata.decode_metadata is not None:
            k_original = k.clone()
            device = positions.device
            block_size = self.block_size
            model_context_len = self.model_context_len

            # cache seq_lens
            if hasattr(attn_metadata, 'seq_lens_clone'):
                seq_lens = attn_metadata.seq_lens_clone
            else:
                seq_lens = attn_metadata.seq_lens_tensor.tolist()
                attn_metadata.seq_lens_clone = seq_lens
            
            # cache block_tables
            if hasattr(attn_metadata, 'block_tables_clone'):
                original_block_tables = attn_metadata.block_tables_clone
            else:
                original_block_tables = attn_metadata.decode_metadata.block_tables.clone()
                attn_metadata.block_tables_clone = original_block_tables

            block_tables_tensor = attn_metadata.decode_metadata.block_tables
            block_tables: List[List[int]] = []

            # cache phys_bnums
            if hasattr(attn_metadata, 'phys_bnums_list'):
                phys_bnums_list = attn_metadata.phys_bnums_list
            else:
                phys_bnums_list = []
            
            # batch size = num sequences
            batch_size = block_tables_tensor.shape[0]
            original_keys: List[Tuple[torch.Tensor]] = []
            for i in range(batch_size):
                num_past_tokens = seq_lens[i] - 1  # assumes decode only yields 1 token
                within_context_len = num_past_tokens < model_context_len
                block_table = block_tables_tensor[i]

                end_logic_bnum = num_past_tokens // block_size
                
                if hasattr(attn_metadata, 'phys_bnums_list'):
                    phys_bnums = phys_bnums_list[i]
                else:
                    if within_context_len:
                        start_logic_bnum = 0
                        phys_bnums = [
                            block_table[logic_bnum] for logic_bnum in range(start_logic_bnum, end_logic_bnum)
                        ]
                    else:
                        start_logic_bnum = (num_past_tokens - model_context_len) // block_size + 2
                        phys_bnums = [block_table[0]] + [
                            block_table[logic_bnum] for logic_bnum in range(start_logic_bnum, end_logic_bnum)
                        ]
                    phys_bnums = torch.tensor(phys_bnums, device=device)
                    phys_bnums_list.append(phys_bnums)
                
                rem = num_past_tokens % block_size
                rem_phys_bnum = block_table[end_logic_bnum]

                if rem == 0:
                    # clear out new block
                    key_cache.index_fill_(0, rem_phys_bnum.to(torch.int64), 0)
                
                # FA shape: [len(phys_bnums), block_size, num_heads, head_size]
                # XF shape: [len(phys_bnums), num_heads, head_size/x, block_size, x]
                full_past_keys = torch.index_select(key_cache, 0, phys_bnums)
                if self.attn_backend == _Backend.FLASH_ATTN:
                    rem_past_keys = key_cache[rem_phys_bnum, :rem, :, :]
                elif self.attn_backend == _Backend.XFORMERS:
                    rem_past_keys = key_cache[rem_phys_bnum, :, :, :rem, :]
                original_keys.append((full_past_keys.clone(), rem_past_keys.clone()))
                
                pos_start = 0 if within_context_len else 2 * block_size - 1 - rem
                pos_end = min(num_past_tokens, model_context_len - 1)
                pos = torch.arange(pos_start, pos_end, device=device)
                # if not within context length: pos = [0, 16] + [31 - rem, 4095)
                if not within_context_len:
                    pos_sink = torch.arange(0, block_size, device=device)
                    pos = torch.cat((pos_sink, pos))
                
                if self.attn_backend == _Backend.FLASH_ATTN:
                    full_past_keys = full_past_keys.flatten(0, 1)
                elif self.attn_backend == _Backend.XFORMERS:
                    full_past_keys = full_past_keys.permute((0, 3, 1, 2, 4)).flatten(0, 1)
                    rem_past_keys = rem_past_keys.permute((2, 0, 1, 3))
                    
                full_past_keys = torch.cat((full_past_keys, rem_past_keys), dim=0)
                full_past_keys = full_past_keys.flatten(1, -1)
                # shape: [pos_end - pos_start, num_heads * head_size/x * x]
                
                dummy_q = torch.zeros_like(full_past_keys)
                _, full_past_keys = self.rotary_emb(pos, dummy_q, full_past_keys)
                
                if self.attn_backend == _Backend.FLASH_ATTN:
                    full_past_keys = full_past_keys.unflatten(1, (key_cache.shape[2], key_cache.shape[3]))
                elif self.attn_backend == _Backend.XFORMERS:
                    full_past_keys = full_past_keys.unflatten(1, (key_cache.shape[1], key_cache.shape[2], key_cache.shape[4]))
                
                full_past_keys, rem_past_keys = torch.split(
                    full_past_keys, [len(phys_bnums) * block_size, rem])
                full_past_keys = full_past_keys.unflatten(0, (len(phys_bnums), block_size))
                
                if self.attn_backend == _Backend.FLASH_ATTN:
                    key_cache.index_put_((phys_bnums,), full_past_keys)
                    key_cache[rem_phys_bnum, :rem, :, :] = rem_past_keys
                elif self.attn_backend == _Backend.XFORMERS:
                    full_past_keys = full_past_keys.permute((0, 2, 3, 1, 4))
                    key_cache.index_put_((phys_bnums,), full_past_keys)
                    key_cache[rem_phys_bnum, :, :, :rem, :] = rem_past_keys.permute((1, 2, 0, 3))
                
                if not within_context_len:
                    blocks_to_ignore = (num_past_tokens - model_context_len) // block_size + 1
                    # block_table[0] is attention sink
                    capped_block_table = [block_table[0].item()] + block_table[blocks_to_ignore + 1:].tolist()
                    block_tables.append(capped_block_table)

                    # edited context len is in range [4081, 4096]
                    attn_metadata.seq_lens_tensor[i] = model_context_len  # pos_end + 1
                    
                    # edited position is in range [4080, 4095]
                    positions[i] = model_context_len - 1  # pos_end

            if block_tables:
                attn_metadata.decode_metadata.block_tables = make_tensor_with_pad(
                    block_tables,
                    max_len=model_context_len // block_size,
                    pad=0,
                    dtype=torch.int,
                    device=device
                )

            if not hasattr(attn_metadata, 'phys_bnums_list'):
                attn_metadata.phys_bnums_list = phys_bnums_list

            # compute attention in kernel
            q, k = self.rotary_emb(positions, q, k)
            attn_output = self.attn(q, k, v, kv_cache, attn_metadata, self.kv_scale)
                        
            # put original pre-rotated keys back in cache
            for i in range(batch_size):
                num_past_tokens = seq_lens[i] - 1
                within_context_len = num_past_tokens < model_context_len
                block_table = original_block_tables[i]
                phys_bnums = phys_bnums_list[i]

                rem = num_past_tokens % block_size
                rem_phys_bnum = block_table[end_logic_bnum]

                full_past_keys, rem_past_keys = original_keys[i]
                key_cache.index_put_((phys_bnums,), full_past_keys)
                if self.attn_backend == _Backend.FLASH_ATTN:
                    key_cache[rem_phys_bnum, :rem, :, :] = rem_past_keys
                elif self.attn_backend == _Backend.XFORMERS:
                    key_cache[rem_phys_bnum, :, :, :rem, :] = rem_past_keys
            
            k_original = k_original.view(-1, self.num_kv_heads, self.head_dim)
            v = v.view(-1, self.num_kv_heads, self.head_dim)

            if self.attn_backend == _Backend.FLASH_ATTN:
                cache_ops.reshape_and_cache_flash(
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
            
            # revert block_tables and seq_lens inside metadata
            # so that next attn layer starts with same fields
            attn_metadata.decode_metadata.block_tables = original_block_tables
            attn_metadata.seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int, device=device)
            
            output, _ = self.output(attn_output)
            return output
