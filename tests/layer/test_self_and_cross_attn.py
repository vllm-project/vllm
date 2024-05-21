import random
from typing import List, Optional
import itertools

import pytest
import torch
import copy
from vllm.attention import Attention, AttentionMetadata

from vllm.attention.backends.xformers import XFormersBackend
from vllm.attention.backends.abstract import AttentionBackend

from vllm.utils import make_tensor_with_pad

from vllm.attention.layer import Attention

import random

# FlashAttention forward only supports head dimension at most 128
# https://github.com/ROCmSoftwarePlatform/flash-attention/blob/3d2b6f5d037782cc2c906909a46fb7e2e1b48b25/csrc/flash_attn_rocm/flash_api.cpp#L62
HEAD_SIZES = [64]

#               [64, 80, 96, 112, 128, 256
#               ] if not is_hip() else [64, 80, 96, 112, 128]

NUM_HEADS = [16]

BATCH_SIZES = [16]
BLOCK_SIZES = [16]
BACKEND_NAMES = ["xformers"]
CUDA_DEVICE = "cuda:0"

PROMPT_LENS = [128]

Q_PROMPT_LENS = [128]

K_PROMPT_LENS = [128]


def build_causal_mask(q_max_prompt_len, kv_max_prompt_len):
    '''
    Create a q_max_prompt_len x kv_max_prompt_len causal mask

    Arguments:
    * q_max_prompt_len: query max prompt len
    * kv_max_prompt_len: key/value max prompt len

    Returns:
    * 2D tensor, q_max_prompt_len x kv_max_prompt_len
    '''

    # Create a matrix where entry (i, j) is True if i >= j
    mask = torch.triu(torch.ones(q_max_prompt_len, kv_max_prompt_len),
                      diagonal=1)
    # Replace True with float('-inf') and False with 0
    mask = mask.masked_fill(mask == 1,
                            float('-inf')).masked_fill(mask == 0, 0.0)
    return mask


def ref_masked_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        custom_mask: Optional[torch.Tensor] = None,
        q_prompt_lens: Optional[List] = None,
        kv_prompt_lens: Optional[List] = None) -> torch.Tensor:
    '''
    "Golden" masked attention reference. Supports two types of masking:
    * Basic attention mask, utilizing {q,kv}_prompt_lens args to mask out padding elements
    * Custom attention mask, which can force an arbitrary mask tensor, i.e. causal

    Arguments:
    * query: batch_size x q_padded_seq_len x num_heads x head_size
    * key: batch_size x kv_padded_seq_len x num_heads x head_size
    * value: batch_size x kv_padded_seq_len x num_heads x head_size
    * scale: Attention scale factor
    * Custom mask: custom attention mask; good place to inject a causal attention mask
    * q_prompt_lens: list of unpadded query seq_lens for each batch index
    * kv_prompt_lens: list of unpadded key/value seq_lens for each batch index

    Returns:
    * Attention result, batch_size x q_padded_seq_len x num_heads x head_size
    '''

    batch_size = query.shape[0]
    assert (len(q_prompt_lens) == batch_size)
    assert (len(kv_prompt_lens) == batch_size)

    attn_weights = scale * torch.einsum("bqhd,bkhd->bhqk", query, key).float()

    # Basic attention mask, derived from prompt lens
    if (q_prompt_lens is not None) or (kv_prompt_lens is not None):
        attn_mask = torch.zeros_like(attn_weights)
        if q_prompt_lens is not None:
            for bdx, plen in enumerate(q_prompt_lens):
                attn_mask[bdx, :, plen:, :] = -torch.inf
        if kv_prompt_lens is not None:
            for bdx, plen in enumerate(kv_prompt_lens):
                attn_mask[bdx, :, :, plen:] = -torch.inf

        attn_weights = attn_weights + attn_mask.float()

    # Custom attention mask
    if custom_mask is not None:
        attn_weights = attn_weights + custom_mask.float()

    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.einsum("bhqk,bkhd->bqhd", attn_weights, value)
    return out


def make_qkv(batch_size,
             max_q_prompt_len,
             max_kv_prompt_len,
             num_heads,
             head_size,
             is_cross_attn=True,
             force_max_len=False,
             device=CUDA_DEVICE):
    '''
    Construct QKV test tensors for self- and cross-attention.

    Generates three query/key/value triplets:
    * "Baseline" query/key/value (for input to reference attention function)
    * "Prefill" query/key/value (last sequence offset zero'd out, for use as input to prefill kernel)
    * "Decode" query/key/value (only the last sequence offset  from baseline, for use as input to decode kernel)

    Each Q/K/V triplet is associated with a list of q seqlens and a list of k/v seqlens

    Arguments:
    * batch_size
    * max_q_prompt_len: max query prompt len
    * max_kv_prompt_len: max key/value prompt len
    * num_heads
    * head_size
    * is_cross_attn: if True, query seqlen may differ from key/value seqlen (as is often the case for cross-attention); o/w, query/key/value seqlens match at each batch index (max_kv_prompt_len is unused)
    * force_max_len: if True, all query seqlens are max_q_prompt_len; o/w query seqlens are random in [2,max_q_prompt_lens]. Same for key/value seqlens and max_kv_prompt_len, unless forced by is_cross_attn=False
    * device: CPU or CUDA device

    Returns:
    * query: "baseline" query; batch_size x max_q_prompt_len x num_heads x head_size
    * key: "baseline" key; batch_size x max_kv_prompt_len x num_heads x head_size
    * value: "baseline" value; batch_size x max_kv_prompt_len x num_heads x head_size
    * prefill_query: batch_size x (max_q_prompt_len-1) x num_heads x head_size
    * prefill_key: batch_size x (max_kv_prompt_len-1) x num_heads x head_size
    * prefill_value: batch_size x (max_kv_prompt_len-1) x num_heads x head_size
    * decode_query: batch_size x 1 x num_heads x head_size
    * decode_key: batch_size x 1 x num_heads x head_size
    * decode_value: batch_size x 1 x num_heads x head_size
    * q_prompt_lens: "baseline" query seqlen list
    * kv_prompt_lens: "baseline" key/value seqlen list
    * actual_max_q_prompt_len: actual "baseline" query max prompt len (may be <= max_q_prompt_len due to randomness)
    * actual_max_kv_prompt_len: actual "baseline" key/value max prompt len (may be <= max_kv_prompt_len due to randomness)
    * prefill_q_prompt_lens: "prefill" query seqlen list
    * prefill_kv_prompt_lens: "prefill" key/value seqlen list
    * decode_q_prompt_lens: "decode" query seqlen list (all ones)
    * decode_kv_prompt_lens: "decode" key/value seqlen list
    '''

    if force_max_len:
        q_prompt_lens = [max_q_prompt_len for _ in range(batch_size)]
    else:
        q_prompt_lens = [
            random.randint(2, max_q_prompt_len) for _ in range(batch_size)
        ]
    kv_prompt_lens = None
    if not is_cross_attn:
        # K,V prompt lens match Q for self-attention
        kv_prompt_lens = q_prompt_lens
    else:
        # K,V prompt lens are distinct from Q prompt lens & random
        if force_max_len:
            kv_prompt_lens = [max_kv_prompt_len for _ in range(batch_size)]
        else:
            kv_prompt_lens = [
                random.randint(2, max_kv_prompt_len) for _ in range(batch_size)
            ]

    actual_max_q_prompt_len = max(q_prompt_lens)
    actual_max_kv_prompt_len = max(kv_prompt_lens)

    query = torch.rand(
        (batch_size, max_q_prompt_len, num_heads * head_size)).to(device)
    key = torch.rand(
        (batch_size, max_kv_prompt_len, num_heads * head_size)).to(device)
    value = torch.rand(
        (batch_size, max_kv_prompt_len, num_heads * head_size)).to(device)

    prefill_query = torch.zeros(
        (batch_size, max_q_prompt_len, num_heads * head_size)).to(device)
    prefill_key = torch.zeros(
        (batch_size, max_kv_prompt_len, num_heads * head_size)).to(device)
    prefill_value = torch.zeros(
        (batch_size, max_kv_prompt_len, num_heads * head_size)).to(device)

    decode_query = torch.zeros(
        (batch_size, 1, num_heads * head_size)).to(device)
    decode_key = torch.zeros((batch_size, 1, num_heads * head_size)).to(device)
    decode_value = torch.zeros(
        (batch_size, 1, num_heads * head_size)).to(device)

    for bdx, (q_prompt_len,
              kv_prompt_len) in enumerate(zip(q_prompt_lens, kv_prompt_lens)):
        query[bdx, q_prompt_len:, :] = 0
        key[bdx, kv_prompt_len:, :] = 0
        value[bdx, kv_prompt_len:, :] = 0

        prefill_query[bdx,
                      0:(q_prompt_len - 1), :] = query[bdx,
                                                       0:(q_prompt_len - 1), :]
        prefill_key[bdx,
                    0:(kv_prompt_len - 1), :] = key[bdx,
                                                    0:(kv_prompt_len - 1), :]
        prefill_value[bdx, 0:(kv_prompt_len -
                              1), :] = value[bdx, 0:(kv_prompt_len - 1), :]

        decode_query[bdx, :, :] = query[bdx,
                                        (q_prompt_len - 1):q_prompt_len, :]
        decode_key[bdx, :, :] = key[bdx, (kv_prompt_len - 1):kv_prompt_len, :]
        decode_value[bdx, :, :] = value[bdx,
                                        (kv_prompt_len - 1):kv_prompt_len, :]

    prefill_q_prompt_lens = [plen - 1 for plen in q_prompt_lens]
    prefill_kv_prompt_lens = [plen - 1 for plen in kv_prompt_lens]

    decode_q_prompt_lens = [1 for _ in q_prompt_lens]
    decode_kv_prompt_lens = [1 for _ in kv_prompt_lens]

    query = query.view(batch_size, query.shape[1], num_heads, head_size)
    key = key.view(batch_size, key.shape[1], num_heads, head_size)
    value = value.view(batch_size, value.shape[1], num_heads, head_size)

    prefill_query = prefill_query.view(batch_size, prefill_query.shape[1],
                                       num_heads, head_size)
    prefill_key = prefill_key.view(batch_size, prefill_key.shape[1], num_heads,
                                   head_size)
    prefill_value = prefill_value.view(batch_size, prefill_value.shape[1],
                                       num_heads, head_size)

    decode_query = decode_query.view(batch_size, decode_query.shape[1],
                                     num_heads, head_size)
    decode_key = decode_key.view(batch_size, decode_key.shape[1], num_heads,
                                 head_size)
    decode_value = decode_value.view(batch_size, decode_value.shape[1],
                                     num_heads, head_size)

    return query, \
           key, \
           value, \
           prefill_query, \
           prefill_key, \
           prefill_value, \
           decode_query, \
           decode_key, \
           decode_value, \
           q_prompt_lens, \
           kv_prompt_lens, \
           actual_max_q_prompt_len, \
           actual_max_kv_prompt_len, \
           prefill_q_prompt_lens, \
           prefill_kv_prompt_lens, \
           decode_q_prompt_lens, \
           decode_kv_prompt_lens


def pack_tensor(unpacked_tensor, prompt_lens, device=CUDA_DEVICE):
    num_tok = sum(prompt_lens)
    num_heads = unpacked_tensor.shape[-2]
    head_size = unpacked_tensor.shape[-1]
    start_loc_list = [0] + list(itertools.accumulate(prompt_lens))
    packed_tensor = torch.zeros((num_tok, num_heads, head_size), device=device)

    for bdx, (prompt_len,
              start_loc) in enumerate(zip(prompt_lens, start_loc_list)):
        try:
            packed_tensor[start_loc:(
                start_loc +
                prompt_len), :, :] = unpacked_tensor[bdx, :prompt_len, :, :]
        except:
            assert False, f"{start_loc} ; {prompt_len} ; {packed_tensor.shape} ; {unpacked_tensor.shape}"

    return packed_tensor, start_loc_list


def pack_qkv(query, key, value, q_prompt_lens, kv_prompt_lens):
    if query is None:
        packed_query = None
        q_start_loc_list = None
    else:
        packed_query, q_start_loc_list = pack_tensor(query, q_prompt_lens)
    packed_key, kv_start_loc_list = pack_tensor(key, kv_prompt_lens)
    packed_value, _ = pack_tensor(value, kv_prompt_lens)
    if packed_query is not None:
        packed_query = packed_query.view(
            -1, packed_query.shape[-1] * packed_query.shape[-2])
    packed_key = packed_key.view(-1,
                                 packed_key.shape[-1] * packed_key.shape[-2])
    packed_value = packed_value.view(
        -1, packed_value.shape[-1] * packed_value.shape[-2])
    return packed_query, packed_key, packed_value, q_start_loc_list, kv_start_loc_list


def make_backend(backend_name: str) -> AttentionBackend:
    if backend_name == "xformers":
        return XFormersBackend()
    assert False, f"Unrecognized backend_name {backend_name} for unit test"


def make_metadata_tensors(is_prompt: bool,
                          prompt_lens: List[int],
                          context_lens: List[int],
                          device=CUDA_DEVICE) -> tuple:
    '''
    Assumptions:
    * No chunked prefill
    * No (automatic) prefix caching
    * Packed variable-length sequences
    '''
    prompt_lens_tensor = torch.tensor(prompt_lens,
                                      dtype=torch.int,
                                      device=device)
    context_lens_tensor = None if context_lens is None else torch.tensor(
        context_lens, dtype=torch.int, device=device)
    max_context_len = None if context_lens is None else max(context_lens)
    max_prompt_len = None if prompt_lens is None else max(prompt_lens)

    seq_start_loc = torch.zeros(prompt_lens_tensor.shape[0] + 1,
                                dtype=torch.int32,
                                device=device)

    torch.cumsum(prompt_lens_tensor,
                 dim=0,
                 dtype=seq_start_loc.dtype,
                 out=seq_start_loc[1:])

    if is_prompt:
        # Prefill: query_start_loc matches seq_start_loc
        query_start_loc = copy.deepcopy(seq_start_loc)
        max_query_len = max_prompt_len
    else:
        # Decode: one new query input token per batch
        # element, thus query_start_loc is the cumsum
        # of [1,1,1,...]
        query_start_loc = list(range(len(seq_start_loc)))
        max_query_len = 1

    return prompt_lens_tensor, \
           context_lens_tensor, \
           max_query_len, \
           max_context_len, \
           max_prompt_len, \
           seq_start_loc, \
           query_start_loc


def make_kv_cache(num_blocks,
                  num_heads,
                  head_size,
                  block_size,
                  device=CUDA_DEVICE,
                  default_val=0.0):
    kv_cache = torch.rand(
        (2, num_blocks, block_size * num_heads * head_size)).to(device)
    if default_val is not None:
        kv_cache[:, :, :] = default_val
    return kv_cache


def num_tokens_to_min_blocks(num_tokens, block_size):
    return (num_tokens + block_size) // block_size


def make_block_tables_slot_mapping(block_size,
                                   prompt_lens,
                                   block_base_addr=0,
                                   device=CUDA_DEVICE):
    '''
    Naive block table:
    * For each batch element...
    * Block table has 
    '''

    # Over-provision block table blocks by 1
    num_blocks_list = [
        num_tokens_to_min_blocks(num_tokens, block_size) + 1
        for num_tokens in prompt_lens
    ]
    max_block_table_len = max(num_blocks_list)
    block_table_pad_tokens = 10

    block_tables = []
    prefill_slot_mapping = []
    decode_slot_mapping = []
    slot_mapping = []
    block_base_idx = block_base_addr + sum(num_blocks_list) * 2 - 1  # Support more blocks than needed
    max_block_idx = block_base_idx
    for sdx, num_tokens in enumerate(prompt_lens):
        num_blocks = num_blocks_list[sdx]
        block_table = list(
            range(block_base_idx, block_base_idx - num_blocks, -1))
        for idx in range(num_tokens - 1):
            prefill_slot_mapping.append((idx % block_size) +
                                        block_table[idx // block_size] *
                                        block_size)
            slot_mapping.append((idx % block_size) +
                                block_table[idx // block_size] * block_size)
        idx = num_tokens - 1
        decode_slot_mapping.append((idx % block_size) +
                                   block_table[idx // block_size] * block_size)
        slot_mapping.append((idx % block_size) +
                            block_table[idx // block_size] * block_size)

        block_base_idx -= num_blocks
        block_tables.append(block_table)

    prefill_block_tables_tensor = torch.tensor([], device=CUDA_DEVICE)
    decode_block_tables_tensor = make_tensor_with_pad(
        block_tables,
        max_len=max_block_table_len + block_table_pad_tokens,
        pad=0,
        dtype=torch.int,
        device=device,
    )
    prefill_slot_mapping_tensor = torch.tensor(prefill_slot_mapping,
                                               dtype=torch.long,
                                               device=device)
    decode_slot_mapping_tensor = torch.tensor(decode_slot_mapping,
                                              dtype=torch.long,
                                              device=device)
    slot_mapping_tensor = torch.tensor(slot_mapping,
                                       dtype=torch.long,
                                       device=device)
    empty_slot_mapping_tensor = torch.tensor([],
                                             dtype=torch.long,
                                             device=device)

    return decode_block_tables_tensor, decode_slot_mapping_tensor, prefill_slot_mapping_tensor, prefill_block_tables_tensor, slot_mapping_tensor, empty_slot_mapping_tensor, max_block_idx


def make_metadata(attn_backend: AttentionBackend,
                  is_prompt: bool,
                  is_cross_attn: bool,
                  prompt_lens: List[int],
                  context_lens: List[int],
                  block_tables,
                  slot_mapping,
                  device=CUDA_DEVICE,
                  cross_prompt_lens: Optional[List[int]] = None):
    '''
    Assumptions:
    * No chunked prefill -> a batch is 100% prefill or 100% decode, never both
    '''

    if is_prompt:
        num_prefills = len(prompt_lens)
        num_prefill_tokens = sum(prompt_lens)
        num_decode_tokens = 0

        prompt_lens_tensor, \
        context_lens_tensor, \
        max_query_len, \
        _, \
        _, \
        seq_start_loc, \
        query_start_loc = make_metadata_tensors(is_prompt,
                                                prompt_lens,
                                                context_lens,
                                                device=device)

        slot_mapping_tensor = torch.tensor(slot_mapping,
                                           dtype=torch.long,
                                           device=device)

        return attn_backend.make_metadata(
            num_prefills=num_prefills,
            slot_mapping=slot_mapping_tensor,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            seq_lens=prompt_lens,
            seq_lens_tensor=prompt_lens_tensor,
            max_query_len=max_query_len,
            max_prefill_seq_len=max(prompt_lens),
            max_decode_seq_len=0,
            query_start_loc=query_start_loc,
            seq_start_loc=seq_start_loc,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=False,
            is_cross_attn=is_cross_attn,
            cross_seq_lens=cross_prompt_lens)

    else:  # not is_prompt

        num_prefills = 0
        num_prefill_tokens = 0
        num_decode_tokens = len(prompt_lens)

        prompt_lens_tensor, \
        context_lens_tensor, \
        max_query_len, \
        _, \
        _, \
        seq_start_loc, \
        query_start_loc = make_metadata_tensors(is_prompt,
                                                prompt_lens,
                                                context_lens,
                                                device=device)

        slot_mapping_tensor = torch.tensor(slot_mapping,
                                           dtype=torch.long,
                                           device=device)

        return attn_backend.make_metadata(
            num_prefills=num_prefills,
            slot_mapping=slot_mapping_tensor,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            seq_lens=prompt_lens,
            seq_lens_tensor=prompt_lens_tensor,
            max_query_len=max_query_len,
            max_prefill_seq_len=0,
            max_decode_seq_len=max(prompt_lens),
            query_start_loc=query_start_loc,
            seq_start_loc=seq_start_loc,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=False,
            is_cross_attn=is_cross_attn,
            cross_seq_lens=cross_prompt_lens)


def make_metadata_self_cross(attn_backend: AttentionBackend,
                             is_prompt: bool,
                             prompt_lens: List[int],
                             context_lens: List[int],
                             block_tables,
                             slot_mapping,
                             device=CUDA_DEVICE,
                             cross_seq_lens: Optional[List[int]] = None,
                             cross_block_tables: Optional[torch.Tensor] = None,
                             cross_slot_mapping: Optional[List[int]] = None,):
    '''
    Assumptions:
    * No chunked prefill -> a batch is 100% prefill or 100% decode, never both
    '''

    if is_prompt:
        num_prefills = len(prompt_lens)
        num_prefill_tokens = sum(prompt_lens)
        num_decode_tokens = 0

        prompt_lens_tensor, \
        context_lens_tensor, \
        max_query_len, \
        _, \
        _, \
        seq_start_loc, \
        query_start_loc = make_metadata_tensors(is_prompt,
                                                prompt_lens,
                                                context_lens,
                                                device=device)

        slot_mapping_tensor = slot_mapping

        cross_slot_mapping_tensor = cross_slot_mapping

        return attn_backend.make_metadata(
            num_prefills=num_prefills,
            slot_mapping=slot_mapping_tensor,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            seq_lens=prompt_lens,
            seq_lens_tensor=prompt_lens_tensor,
            max_query_len=max_query_len,
            max_prefill_seq_len=max(prompt_lens),
            max_decode_seq_len=0,
            query_start_loc=query_start_loc,
            seq_start_loc=seq_start_loc,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=False,
            is_cross_attn=False,
            cross_seq_lens=cross_seq_lens,
            cross_slot_mapping=cross_slot_mapping_tensor,
            cross_block_tables=cross_block_tables)

    else:  # not is_prompt

        num_prefills = 0
        num_prefill_tokens = 0
        num_decode_tokens = len(prompt_lens)

        prompt_lens_tensor, \
        context_lens_tensor, \
        max_query_len, \
        _, \
        _, \
        seq_start_loc, \
        query_start_loc = make_metadata_tensors(is_prompt,
                                                prompt_lens,
                                                context_lens,
                                                device=device)

        slot_mapping_tensor = slot_mapping

        cross_slot_mapping_tensor = cross_slot_mapping

        return attn_backend.make_metadata(
            num_prefills=num_prefills,
            slot_mapping=slot_mapping_tensor,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            seq_lens=prompt_lens,
            seq_lens_tensor=prompt_lens_tensor,
            max_query_len=max_query_len,
            max_prefill_seq_len=0,
            max_decode_seq_len=max(prompt_lens),
            query_start_loc=query_start_loc,
            seq_start_loc=seq_start_loc,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=False,
            is_cross_attn=False,
            cross_seq_lens=cross_seq_lens,
            cross_slot_mapping=cross_slot_mapping_tensor,
            cross_block_tables=cross_block_tables)

def make_attention(num_heads: int, head_size: int, scale: float):
    # Attention operator instance
    return Attention(
        num_heads,
        head_size,
        scale=scale,
    )


def basic_setup(num_heads, head_size, num_blocks, block_size, backend_name):
    scale = float(1.0 / (head_size**0.5))
    attn_backend = make_backend(backend_name)
    attn = make_attention(num_heads, head_size, scale)
    kv_cache = make_kv_cache(num_blocks, num_heads, head_size, block_size)
    return scale, attn_backend, attn, kv_cache

def self_attn_setup(batch_size, num_heads, head_size, block_size, scale, max_q_prompt_len, block_base_addr=0):

    max_kv_prompt_len = max_q_prompt_len

    query, \
    key, \
    value, \
    prefill_query, \
    prefill_key, \
    prefill_value, \
    decode_query, \
    decode_key, \
    decode_value, \
    q_prompt_lens, \
    kv_prompt_lens, \
    _, \
    _, \
    prefill_q_prompt_lens, \
    prefill_kv_prompt_lens, \
    decode_q_prompt_lens, \
    decode_kv_prompt_lens = make_qkv(batch_size,max_q_prompt_len,max_kv_prompt_len,num_heads,head_size,is_cross_attn=False)

    causal_mask = build_causal_mask(max_q_prompt_len,
                                    max_kv_prompt_len).to(CUDA_DEVICE)
    
    ideal_output = ref_masked_attention(query,
                                        key,
                                        value,
                                        scale=scale,
                                        custom_mask=causal_mask,
                                        q_prompt_lens=q_prompt_lens,
                                        kv_prompt_lens=kv_prompt_lens)

    prefill_ideal_output = torch.zeros_like(ideal_output)
    decode_ideal_output = torch.zeros_like(ideal_output[:, 0:1])
    for bdx, prefill_q_prompt_len in enumerate(prefill_q_prompt_lens):
        prefill_ideal_output[bdx, :prefill_q_prompt_len] = ideal_output[
            bdx, :prefill_q_prompt_len]
        decode_ideal_output[bdx, :] = ideal_output[bdx, prefill_q_prompt_len:(
            prefill_q_prompt_len + 1)]

    prefill_packed_ideal_output, _ = pack_tensor(prefill_ideal_output,
                                                 prefill_q_prompt_lens)
    decode_packed_ideal_output, _ = pack_tensor(decode_ideal_output,
                                                [1 for _ in range(batch_size)])

    decode_block_tables, decode_slot_mapping, prefill_slot_mapping, prefill_block_tables, _, _, max_block_idx = make_block_tables_slot_mapping(
        block_size, q_prompt_lens, block_base_addr=block_base_addr)

    prefill_packed_query, prefill_packed_key, prefill_packed_value, _, _ = pack_qkv(
        prefill_query, prefill_key, prefill_value, prefill_q_prompt_lens,
        prefill_kv_prompt_lens)

    decode_packed_query, decode_packed_key, decode_packed_value, _, _ = pack_qkv(
        decode_query, decode_key, decode_value, decode_q_prompt_lens,
        decode_kv_prompt_lens)

    return query, \
    prefill_packed_query, \
    prefill_packed_key, \
    prefill_packed_value, \
    prefill_packed_ideal_output, \
    prefill_q_prompt_lens, \
    prefill_kv_prompt_lens, \
    decode_packed_query, \
    decode_packed_key, \
    decode_packed_value, \
    decode_packed_ideal_output, \
    decode_q_prompt_lens, \
    decode_kv_prompt_lens, \
    q_prompt_lens, \
    kv_prompt_lens, \
    decode_block_tables, \
    decode_slot_mapping, \
    prefill_slot_mapping, \
    prefill_block_tables, \
    max_block_idx


def cross_attn_setup_reuses_query(query, q_prompt_lens, prefill_q_prompt_lens, batch_size, num_heads, head_size, block_size, scale, max_q_prompt_len, max_kv_prompt_len, block_base_addr=0):

    _, \
    key, \
    value, \
    _, \
    _, \
    _, \
    _, \
    _, \
    _, \
    _, \
    kv_prompt_lens, \
    _, \
    _, \
    _, \
    _, \
    _, \
    _ = make_qkv(batch_size,max_q_prompt_len,max_kv_prompt_len,num_heads,head_size,is_cross_attn=True)

    ideal_output = ref_masked_attention(query,
                                        key,
                                        value,
                                        scale=scale,
                                        q_prompt_lens=q_prompt_lens,
                                        kv_prompt_lens=kv_prompt_lens)

    prefill_ideal_output = torch.zeros_like(ideal_output)
    decode_ideal_output = torch.zeros_like(ideal_output[:, 0:1])
    for bdx, prefill_q_prompt_len in enumerate(prefill_q_prompt_lens):
        prefill_ideal_output[bdx, :prefill_q_prompt_len] = ideal_output[
            bdx, :prefill_q_prompt_len]
        decode_ideal_output[bdx, :] = ideal_output[bdx, prefill_q_prompt_len:(
            prefill_q_prompt_len + 1)]

    prefill_packed_ideal_output, _ = pack_tensor(prefill_ideal_output,
                                                 prefill_q_prompt_lens)
    decode_packed_ideal_output, _ = pack_tensor(decode_ideal_output,
                                                [1 for _ in range(batch_size)])

    # Unlike self-attention:
    # - Prefill slot-mapping includes all key slots
    # - Decode slot-mapping is empty
    decode_block_tables, _, _, prefill_block_tables, prefill_slot_mapping, decode_slot_mapping, max_block_idx  = make_block_tables_slot_mapping(
        block_size, kv_prompt_lens, block_base_addr=block_base_addr)
    
    _, prefill_packed_key, prefill_packed_value, _, _ = pack_qkv(
        None, key, value, prefill_q_prompt_lens, kv_prompt_lens)

    return prefill_packed_key, \
    prefill_packed_value, \
    prefill_packed_ideal_output, \
    decode_packed_ideal_output, \
    kv_prompt_lens, \
    decode_block_tables, \
    decode_slot_mapping, \
    prefill_slot_mapping, \
    prefill_block_tables, \
    max_block_idx

def run_self_attention_test(attn,packed_query,packed_key,packed_value,kv_cache,attn_metadata:AttentionMetadata,scale):
    attn_metadata.do_cross_attn = False
    return attn.forward(packed_query,
                        packed_key,
                        packed_value, 
                        kv_cache,
                        attn_metadata, 
                        scale)

def run_cross_attention_test(attn,packed_query,packed_key,packed_value,kv_cache,attn_metadata:AttentionMetadata,scale):
    attn_metadata.do_cross_attn = True
    return attn.forward(packed_query,
                        packed_key,
                        packed_value, 
                        kv_cache,
                        attn_metadata, 
                        scale)

@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("backend_name", BACKEND_NAMES)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("max_q_prompt_len", Q_PROMPT_LENS)
@pytest.mark.parametrize("max_kv_prompt_len", K_PROMPT_LENS)
def test_prefill_decode_self_and_cross_attention(num_heads: int, head_size: int,
                                                 backend_name: str, batch_size: int,
                                                 block_size: int, max_q_prompt_len: int,
                                                 max_kv_prompt_len: int) -> None:

    # Num KV cache blocks
    num_blocks = 4096

    # Attention scale factor,
    # attention backend instance,
    # attention wrapper instance,
    # KV cache init
    scale, \
    attn_backend, \
    attn, \
    kv_cache = basic_setup(num_heads, 
                           head_size, 
                           num_blocks, 
                           block_size, 
                           backend_name)

    # Self-attention setup

    self_block_base_addr=0

    query, \
    prefill_packed_query, \
    self_prefill_packed_key, \
    self_prefill_packed_value, \
    self_prefill_packed_ideal_output, \
    prefill_q_prompt_lens, \
    self_prefill_kv_prompt_lens, \
    decode_packed_query, \
    self_decode_packed_key, \
    self_decode_packed_value, \
    self_decode_packed_ideal_output, \
    decode_q_prompt_lens, \
    self_decode_kv_prompt_lens, \
    q_prompt_lens, \
    self_kv_prompt_lens, \
    self_decode_block_tables, \
    self_decode_slot_mapping, \
    self_prefill_slot_mapping, \
    self_prefill_block_tables, \
    cross_block_base_addr = self_attn_setup(batch_size, 
                                                num_heads, 
                                                head_size, 
                                                block_size, 
                                                scale, 
                                                max_q_prompt_len,
                                                block_base_addr=self_block_base_addr)

    # Cross-attention setup

    cross_prefill_packed_key, \
    cross_prefill_packed_value, \
    cross_prefill_packed_ideal_output, \
    cross_decode_packed_ideal_output, \
    cross_kv_prompt_lens, \
    cross_decode_block_tables, \
    cross_decode_slot_mapping, \
    cross_prefill_slot_mapping, \
    cross_prefill_block_tables, \
    final_max_block_idx = cross_attn_setup_reuses_query(query, 
                                                        q_prompt_lens, 
                                                        prefill_q_prompt_lens, 
                                                        batch_size, 
                                                        num_heads, 
                                                        head_size, 
                                                        block_size, 
                                                        scale, 
                                                        max_q_prompt_len, 
                                                        max_kv_prompt_len, 
                                                        block_base_addr=cross_block_base_addr)

    # PREFILL: self- and cross-attention tests

    context_lens = [0 for _ in range(batch_size)]

    prefill_attn_metadata: AttentionMetadata = make_metadata_self_cross(attn_backend,
                                                                        True,
                                                                        prefill_q_prompt_lens,
                                                                        context_lens,
                                                                        self_prefill_block_tables,
                                                                        self_prefill_slot_mapping,
                                                                        cross_seq_lens = cross_kv_prompt_lens,
                                                                        cross_block_tables = cross_prefill_block_tables,
                                                                        cross_slot_mapping = cross_prefill_slot_mapping,)

    self_prefill_packed_actual_output: torch.Tensor = run_self_attention_test(attn,
                                                                              prefill_packed_query,
                                                                              self_prefill_packed_key,
                                                                              self_prefill_packed_value,
                                                                              kv_cache,
                                                                              prefill_attn_metadata,
                                                                              scale)

    # - Prefill self-attention correct?
    assert torch.allclose(self_prefill_packed_ideal_output,self_prefill_packed_actual_output.view_as(self_prefill_packed_ideal_output))

    cross_prefill_packed_actual_output: torch.Tensor = run_cross_attention_test(attn,
                                                                                prefill_packed_query,
                                                                                cross_prefill_packed_key,
                                                                                cross_prefill_packed_value,
                                                                                kv_cache,
                                                                                prefill_attn_metadata,
                                                                                scale)

    # - Prefill cross-attention correct?
    assert torch.allclose(cross_prefill_packed_ideal_output,cross_prefill_packed_actual_output.view_as(cross_prefill_packed_ideal_output))

    context_lens = copy.deepcopy(self_prefill_kv_prompt_lens)

    # DECODE: self- and cross-attention tests

    decode_attn_metadata: AttentionMetadata = make_metadata_self_cross(attn_backend,
                                                                       False,
                                                                       q_prompt_lens,
                                                                       context_lens,
                                                                       self_decode_block_tables,
                                                                       self_decode_slot_mapping,
                                                                       cross_seq_lens = cross_kv_prompt_lens,
                                                                       cross_block_tables = cross_decode_block_tables,
                                                                       cross_slot_mapping = cross_decode_slot_mapping,)

    self_decode_packed_actual_output: torch.Tensor = run_self_attention_test(attn,
                                                                             decode_packed_query,
                                                                             self_decode_packed_key,
                                                                             self_decode_packed_value,
                                                                             kv_cache,
                                                                             decode_attn_metadata,
                                                                             scale)

    assert torch.allclose(self_decode_packed_ideal_output,self_decode_packed_actual_output.view_as(self_decode_packed_ideal_output))

    cross_decode_packed_actual_output: torch.Tensor = run_cross_attention_test(attn,
                                                                               decode_packed_query,
                                                                               None,
                                                                               None,
                                                                               kv_cache,
                                                                               decode_attn_metadata,
                                                                               scale)

    assert torch.allclose(cross_decode_packed_ideal_output,cross_decode_packed_actual_output.view_as(cross_decode_packed_ideal_output))

@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("backend_name", BACKEND_NAMES)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("max_prompt_len", PROMPT_LENS)
def test_prefill_decode_self_attention(num_heads: int, head_size: int,
                                       backend_name: str, batch_size: int,
                                       block_size: int,
                                       max_prompt_len: int) -> None:
    # Attention operator instance
    is_cross_attn = False
    is_prompt = True
    max_q_prompt_len = max_prompt_len
    max_kv_prompt_len = max_q_prompt_len
    context_lens = [0 for _ in range(batch_size)]
    num_blocks = 4096
    kv_cache = make_kv_cache(num_blocks, num_heads, head_size, block_size)
    scale = float(1.0 / (head_size**0.5))
    attn = make_attention(num_heads, head_size, scale)
    attn_backend = make_backend(backend_name)
    query, \
    key, \
    value, \
    prefill_query, \
    prefill_key, \
    prefill_value, \
    decode_query, \
    decode_key, \
    decode_value, \
    q_prompt_lens, \
    kv_prompt_lens, \
    _, \
    _, \
    prefill_q_prompt_lens, \
    prefill_kv_prompt_lens, \
    decode_q_prompt_lens, \
    decode_kv_prompt_lens = make_qkv(batch_size,max_q_prompt_len,max_kv_prompt_len,num_heads,head_size,is_cross_attn=False)

    causal_mask = build_causal_mask(max_q_prompt_len,
                                    max_kv_prompt_len).to(CUDA_DEVICE)
    ideal_output = ref_masked_attention(query,
                                        key,
                                        value,
                                        scale=scale,
                                        custom_mask=causal_mask,
                                        q_prompt_lens=q_prompt_lens,
                                        kv_prompt_lens=kv_prompt_lens)

    prefill_ideal_output = torch.zeros_like(ideal_output)
    decode_ideal_output = torch.zeros_like(ideal_output[:, 0:1])
    for bdx, prefill_q_prompt_len in enumerate(prefill_q_prompt_lens):
        prefill_ideal_output[bdx, :prefill_q_prompt_len] = ideal_output[
            bdx, :prefill_q_prompt_len]
        decode_ideal_output[bdx, :] = ideal_output[bdx, prefill_q_prompt_len:(
            prefill_q_prompt_len + 1)]

    prefill_packed_ideal_output, _ = pack_tensor(prefill_ideal_output,
                                                 prefill_q_prompt_lens)
    decode_packed_ideal_output, _ = pack_tensor(decode_ideal_output,
                                                [1 for _ in range(batch_size)])

    decode_block_tables, decode_slot_mapping, prefill_slot_mapping, prefill_block_tables, _, _ = make_block_tables_slot_mapping(
        block_size, q_prompt_lens)
    prefill_attn_metadata: AttentionMetadata = make_metadata(
        attn_backend,
        is_prompt,
        is_cross_attn,
        prefill_q_prompt_lens,
        context_lens,
        prefill_block_tables,
        prefill_slot_mapping,
        cross_prompt_lens=None)

    prefill_packed_query, prefill_packed_key, prefill_packed_value, _, _ = pack_qkv(
        prefill_query, prefill_key, prefill_value, prefill_q_prompt_lens,
        prefill_kv_prompt_lens)

    prefill_packed_actual_output = attn.forward(prefill_packed_query,
                                                prefill_packed_key,
                                                prefill_packed_value, kv_cache,
                                                prefill_attn_metadata, scale)

    # eval correctness of prefill output
    assert torch.allclose(
        prefill_packed_actual_output,
        prefill_packed_ideal_output.view_as(prefill_packed_actual_output))

    is_prompt = False
    context_lens = copy.deepcopy(prefill_kv_prompt_lens)
    decode_attn_metadata = make_metadata(attn_backend, is_prompt,
                                         is_cross_attn, q_prompt_lens,
                                         context_lens, decode_block_tables,
                                         decode_slot_mapping)

    decode_packed_query, decode_packed_key, decode_packed_value, _, _ = pack_qkv(
        decode_query, decode_key, decode_value, decode_q_prompt_lens,
        decode_kv_prompt_lens)

    decode_packed_actual_output = attn.forward(decode_packed_query,
                                               decode_packed_key,
                                               decode_packed_value, kv_cache,
                                               decode_attn_metadata, scale)

    # eval correctness of decode output
    assert torch.allclose(
        decode_packed_actual_output,
        decode_packed_ideal_output.view_as(decode_packed_actual_output))


@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("backend_name", BACKEND_NAMES)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("max_q_prompt_len", Q_PROMPT_LENS)
@pytest.mark.parametrize("max_kv_prompt_len", K_PROMPT_LENS)
def test_prefill_decode_cross_attention(num_heads: int, head_size: int,
                                        backend_name: str, batch_size: int,
                                        block_size: int, max_q_prompt_len: int,
                                        max_kv_prompt_len: int) -> None:
    # Attention operator instance
    is_cross_attn = True
    is_prompt = True
    context_lens = [0 for _ in range(batch_size)]
    num_blocks = 4096
    kv_cache = make_kv_cache(num_blocks, num_heads, head_size, block_size)
    scale = float(1.0 / (head_size**0.5))
    attn = make_attention(num_heads, head_size, scale)
    attn_backend = make_backend(backend_name)

    query, \
    key, \
    value, \
    prefill_query, \
    _, \
    _, \
    decode_query, \
    _, \
    _, \
    q_prompt_lens, \
    kv_prompt_lens, \
    _, \
    _, \
    prefill_q_prompt_lens, \
    _, \
    decode_q_prompt_lens, \
    _ = make_qkv(batch_size,max_q_prompt_len,max_kv_prompt_len,num_heads,head_size,is_cross_attn=is_cross_attn)

    ideal_output = ref_masked_attention(query,
                                        key,
                                        value,
                                        scale=scale,
                                        q_prompt_lens=q_prompt_lens,
                                        kv_prompt_lens=kv_prompt_lens)

    prefill_ideal_output = torch.zeros_like(ideal_output)
    decode_ideal_output = torch.zeros_like(ideal_output[:, 0:1])
    for bdx, prefill_q_prompt_len in enumerate(prefill_q_prompt_lens):
        prefill_ideal_output[bdx, :prefill_q_prompt_len] = ideal_output[
            bdx, :prefill_q_prompt_len]
        decode_ideal_output[bdx, :] = ideal_output[bdx, prefill_q_prompt_len:(
            prefill_q_prompt_len + 1)]

    prefill_packed_ideal_output, _ = pack_tensor(prefill_ideal_output,
                                                 prefill_q_prompt_lens)
    decode_packed_ideal_output, _ = pack_tensor(decode_ideal_output,
                                                [1 for _ in range(batch_size)])

    # Unlike self-attention:
    # - Prefill slot-mapping includes all key slots
    # - Decode slot-mapping is empty
    decode_block_tables, _, _, prefill_block_tables, prefill_slot_mapping, decode_slot_mapping = make_block_tables_slot_mapping(
        block_size, kv_prompt_lens)

    prefill_attn_metadata: AttentionMetadata = make_metadata(
        attn_backend,
        is_prompt,
        is_cross_attn,
        prefill_q_prompt_lens,
        context_lens,
        prefill_block_tables,
        prefill_slot_mapping,
        cross_prompt_lens=kv_prompt_lens)

    prefill_packed_query, prefill_packed_key, prefill_packed_value, _, _ = pack_qkv(
        prefill_query, key, value, prefill_q_prompt_lens, kv_prompt_lens)

    prefill_packed_actual_output = attn.forward(prefill_packed_query,
                                                prefill_packed_key,
                                                prefill_packed_value, kv_cache,
                                                prefill_attn_metadata, scale)

    # eval correctness of prefill output
    assert torch.allclose(
        prefill_packed_actual_output,
        prefill_packed_ideal_output.view_as(prefill_packed_actual_output))

    is_prompt = False
    context_lens = copy.deepcopy(kv_prompt_lens)
    decode_attn_metadata = make_metadata(attn_backend,
                                         is_prompt,
                                         is_cross_attn,
                                         q_prompt_lens,
                                         context_lens,
                                         decode_block_tables,
                                         decode_slot_mapping,
                                         cross_prompt_lens=kv_prompt_lens)

    decode_packed_query, _, _, _, _ = pack_qkv(decode_query, key, value,
                                               decode_q_prompt_lens,
                                               kv_prompt_lens)

    decode_packed_actual_output = attn.forward(decode_packed_query, None, None,
                                               kv_cache, decode_attn_metadata,
                                               scale)

    # eval correctness of decode output
    assert torch.allclose(
        decode_packed_actual_output,
        decode_packed_ideal_output.view_as(decode_packed_actual_output))
