import copy
import itertools
import random
from typing import List, Optional

import pytest
import torch

from vllm.attention import Attention, AttentionMetadata
from vllm.attention.backends.abstract import AttentionBackend, AttentionType
from vllm.attention.backends.xformers import XFormersBackend
from vllm.utils import make_tensor_with_pad

# If not is_hip(): supported head sizes are [64, 80, 96, 112, 128, 256]
#
# TODO: FlashAttention forward only supports head dimension at most 128
# https://github.com/ROCmSoftwarePlatform/flash-attention/blob/3d2b6f5d0
# 37782cc2c906909a46fb7e2e1b48b25/csrc/flash_attn_rocm/flash_api.cpp#L62
HEAD_SIZES = [64, 256]

NUM_HEADS = [1, 16]

BATCH_SIZES = [1, 16]
BLOCK_SIZES = [16]
BACKEND_NAMES = ["xformers"]
CUDA_DEVICE = "cuda:0"

MAX_Q_SEQ_LENS = [128]
MAX_K_SEQ_LENS = [128]


def build_causal_mask(q_max_seq_len, kv_max_seq_len):
    '''
    Create a q_max_seq_len x kv_max_seq_len causal mask

    Arguments:
    
    * q_max_seq_len: query max seq len
    * kv_max_seq_len: key/value max seq len

    Returns:

    * 2D tensor, q_max_seq_len x kv_max_seq_len
    '''

    # Create a matrix where entry (i, j) is True if i >= j
    mask = torch.triu(torch.ones(q_max_seq_len, kv_max_seq_len), diagonal=1)
    # Replace True with float('-inf') and False with 0
    mask = mask.masked_fill(mask == 1,
                            float('-inf')).masked_fill(mask == 0, 0.0)
    return mask


def ref_masked_attention(query: torch.Tensor,
                         key: torch.Tensor,
                         value: torch.Tensor,
                         scale: float,
                         custom_mask: Optional[torch.Tensor] = None,
                         q_seq_lens: Optional[List] = None,
                         kv_seq_lens: Optional[List] = None) -> torch.Tensor:
    '''
    "Golden" masked attention reference. Supports two types of masking:

    * Basic attention mask, utilizing {q,kv}_seq_lens args to mask out
      padding elements
    * Custom attention mask, which can force an arbitrary mask tensor, i.e.
      causal

    Arguments:

    * query: batch_size x q_padded_seq_len x num_heads x head_size
    * key: batch_size x kv_padded_seq_len x num_heads x head_size
    * value: batch_size x kv_padded_seq_len x num_heads x head_size
    * scale: Attention scale factor
    * Custom mask: custom attention mask; good place to inject a causal
      attention mask
    * q_seq_lens: list of unpadded query seq_lens for each batch index
    * kv_seq_lens: list of unpadded key/value seq_lens for each batch index

    Returns:

    * Attention result, batch_size x q_padded_seq_len x num_heads x head_size
    '''

    batch_size = query.shape[0]
    assert (len(q_seq_lens) == batch_size)
    assert (len(kv_seq_lens) == batch_size)

    attn_weights = scale * torch.einsum("bqhd,bkhd->bhqk", query, key).float()

    # Basic attention mask, derived from seq lens
    if (q_seq_lens is not None) or (kv_seq_lens is not None):
        attn_mask = torch.zeros_like(attn_weights)
        if q_seq_lens is not None:
            for bdx, plen in enumerate(q_seq_lens):
                attn_mask[bdx, :, plen:, :] = -torch.inf
        if kv_seq_lens is not None:
            for bdx, plen in enumerate(kv_seq_lens):
                attn_mask[bdx, :, :, plen:] = -torch.inf

        attn_weights = attn_weights + attn_mask.float()

    # Custom attention mask
    if custom_mask is not None:
        attn_weights = attn_weights + custom_mask.float()

    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.einsum("bhqk,bkhd->bqhd", attn_weights, value)
    return out


def make_qkv(batch_size,
             max_q_seq_len,
             max_kv_seq_len,
             num_heads,
             head_size,
             attn_type: AttentionType = AttentionType.ENCODER_DECODER,
             force_max_len=False,
             device=CUDA_DEVICE):
    '''
    Construct QKV test tensors for self- and cross-attention.

    Generates three query/key/value triplets:

    * "Baseline" query/key/value (for input to reference attention function)
    * "Prefill" query/key/value (last sequence offset zero'd out, for use as
      input to prefill kernel)
    * "Decode" query/key/value (only the last sequence offset  from baseline,
      for use as input to decode kernel)

    Each Q/K/V triplet is associated with a list of q seqlens and a list of k/v
    seqlens

    Arguments:

    * batch_size
    * max_q_seq_len: max query seq len
    * max_kv_seq_len: max key/value seq len
    * num_heads
    * head_size
    * is_encoder_decoder_attn: if True, query seqlen may differ from 
      key/value seqlen (as is often the case for cross-attention); 
      o/w, query/key/value seqlens match at each batch index 
      (max_kv_seq_len is unused)
    * force_max_len: if True, all query seqlens are max_q_seq_len; o/w query
      seqlens are random in [2,max_q_seq_lens]. Same for key/value seqlens
      and max_kv_seq_len, unless forced by is_encoder_decoder_attn=False
    * device: CPU or CUDA device

    Returns:

    * query: "baseline" query; batch_size x max_q_seq_len x num_heads x
      head_size
    * key: "baseline" key; batch_size x max_kv_seq_len x num_heads x
      head_size
    * value: "baseline" value; batch_size x max_kv_seq_len x num_heads x
      head_size
    * prefill_query: batch_size x (max_q_seq_len-1) x num_heads x head_size
    * prefill_key: batch_size x (max_kv_seq_len-1) x num_heads x head_size
    * prefill_value: batch_size x (max_kv_seq_len-1) x num_heads x head_size
    * decode_query: batch_size x 1 x num_heads x head_size
    * decode_key: batch_size x 1 x num_heads x head_size
    * decode_value: batch_size x 1 x num_heads x head_size
    * q_seq_lens: "baseline" query seqlen list
    * kv_seq_lens: "baseline" key/value seqlen list
    * actual_max_q_seq_len: actual "baseline" query max seq len (may be <=
      max_q_seq_len due to randomness)
    * actual_max_kv_seq_len: actual "baseline" key/value max seq len (may
      be <= max_kv_seq_len due to randomness)
    * prefill_q_seq_lens: "prefill" query seqlen list
    * prefill_kv_seq_lens: "prefill" key/value seqlen list
    * decode_q_seq_lens: "decode" query seqlen list (all ones)
    * decode_kv_seq_lens: "decode" key/value seqlen list
    '''

    if force_max_len:
        q_seq_lens = [max_q_seq_len for _ in range(batch_size)]
    else:
        q_seq_lens = [
            random.randint(2, max_q_seq_len) for _ in range(batch_size)
        ]
    kv_seq_lens = None
    if attn_type != AttentionType.ENCODER_DECODER:
        # K,V seq lens match Q for self-attention
        kv_seq_lens = q_seq_lens
    else:
        # K,V seq lens are distinct from Q seq lens & random
        if force_max_len:
            kv_seq_lens = [max_kv_seq_len for _ in range(batch_size)]
        else:
            kv_seq_lens = [
                random.randint(2, max_kv_seq_len) for _ in range(batch_size)
            ]

    actual_max_q_seq_len = max(q_seq_lens)
    actual_max_kv_seq_len = max(kv_seq_lens)

    query = torch.rand(
        (batch_size, max_q_seq_len, num_heads * head_size)).to(device)
    key = torch.rand(
        (batch_size, max_kv_seq_len, num_heads * head_size)).to(device)
    value = torch.rand(
        (batch_size, max_kv_seq_len, num_heads * head_size)).to(device)

    prefill_query = torch.zeros(
        (batch_size, max_q_seq_len, num_heads * head_size)).to(device)
    prefill_key = torch.zeros(
        (batch_size, max_kv_seq_len, num_heads * head_size)).to(device)
    prefill_value = torch.zeros(
        (batch_size, max_kv_seq_len, num_heads * head_size)).to(device)

    decode_query = torch.zeros(
        (batch_size, 1, num_heads * head_size)).to(device)
    decode_key = torch.zeros((batch_size, 1, num_heads * head_size)).to(device)
    decode_value = torch.zeros(
        (batch_size, 1, num_heads * head_size)).to(device)

    for bdx, (q_seq_len, kv_seq_len) in enumerate(zip(q_seq_lens,
                                                      kv_seq_lens)):
        query[bdx, q_seq_len:, :] = 0
        key[bdx, kv_seq_len:, :] = 0
        value[bdx, kv_seq_len:, :] = 0

        prefill_query[bdx, 0:(q_seq_len - 1), :] = query[bdx,
                                                         0:(q_seq_len - 1), :]
        prefill_key[bdx, 0:(kv_seq_len - 1), :] = key[bdx,
                                                      0:(kv_seq_len - 1), :]
        prefill_value[bdx,
                      0:(kv_seq_len - 1), :] = value[bdx,
                                                     0:(kv_seq_len - 1), :]

        decode_query[bdx, :, :] = query[bdx, (q_seq_len - 1):q_seq_len, :]
        decode_key[bdx, :, :] = key[bdx, (kv_seq_len - 1):kv_seq_len, :]
        decode_value[bdx, :, :] = value[bdx, (kv_seq_len - 1):kv_seq_len, :]

    prefill_q_seq_lens = [plen - 1 for plen in q_seq_lens]
    prefill_kv_seq_lens = [plen - 1 for plen in kv_seq_lens]

    decode_q_seq_lens = [1 for _ in q_seq_lens]
    decode_kv_seq_lens = [1 for _ in kv_seq_lens]

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
           q_seq_lens, \
           kv_seq_lens, \
           actual_max_q_seq_len, \
           actual_max_kv_seq_len, \
           prefill_q_seq_lens, \
           prefill_kv_seq_lens, \
           decode_q_seq_lens, \
           decode_kv_seq_lens


def pack_tensor(unpacked_tensor, seq_lens, device=CUDA_DEVICE):
    '''
    Pack a batch_size x padded_seq_len x num_heads x head_size tensor into an
    unpadded number_of_tokens x num_heads x head_size tensor, where
    number_of_tokens = sum(seq_lens)

    Arguments:

    * unpacked_tensor: batch_size x padded_seq_len x num_heads x head_size
    * seq_lens: list of token counts for each seq
    * device: CPU or CUDA device

    Returns

    * packed_tensor: number_of_tokens x num_heads x head_size
    * start_loc_list: start idx of each batch elt in packed_tensor; [0] +
      list(itertools.accumulate(seq_lens))
    '''

    num_tok = sum(seq_lens)
    num_heads = unpacked_tensor.shape[-2]
    head_size = unpacked_tensor.shape[-1]
    start_loc_list = [0] + list(itertools.accumulate(seq_lens))
    packed_tensor = torch.zeros((num_tok, num_heads, head_size), device=device)

    for bdx, (seq_len, start_loc) in enumerate(zip(seq_lens, start_loc_list)):

        packed_tensor[start_loc:(
            start_loc + seq_len), :, :] = unpacked_tensor[bdx, :seq_len, :, :]

    return packed_tensor, start_loc_list


def pack_qkv(query, key, value, q_seq_lens, kv_seq_lens):
    '''
    Individually pack each of Q, K and V, each with dimensions batch_size x
    padded_seq_len x num_heads x head_size, into respective number_of_tokens x
    num_heads x head_size tensors.
    
    For Q, number_of_tokens = sum(q_seq_lens).

    For K and V, number_of_tokens = sum(kv_seq_lens)

    Arguments:

    * query: batch_size x padded_seq_len x num_heads x head_size
    * key: batch_size x padded_seq_len x num_heads x head_size
    * value: batch_size x padded_seq_len x num_heads x head_size
    * q_seq_lens: list of token counts for each query
    * kv_seq_lens: list of token counts for each key/value

    Returns

    * packed_query: number_of_tokens x num_heads x head_size
    * packed_key: number_of_tokens x num_heads x head_size
    * packed_value: number_of_tokens x num_heads x head_size
    * q_start_loc_list: start idx of each query in packed_query
    * kv_start_loc_list: start idx of each {key,value} in packed_{key,value}
    '''

    if query is None:
        packed_query = None
        q_start_loc_list = None
    else:
        packed_query, q_start_loc_list = pack_tensor(query, q_seq_lens)
    packed_key, kv_start_loc_list = pack_tensor(key, kv_seq_lens)
    packed_value, _ = pack_tensor(value, kv_seq_lens)
    if packed_query is not None:
        packed_query = packed_query.view(
            -1, packed_query.shape[-1] * packed_query.shape[-2])
    packed_key = packed_key.view(-1,
                                 packed_key.shape[-1] * packed_key.shape[-2])
    packed_value = packed_value.view(
        -1, packed_value.shape[-1] * packed_value.shape[-2])
    return packed_query, \
           packed_key, \
           packed_value, \
           q_start_loc_list, \
           kv_start_loc_list


def make_backend(backend_name: str) -> AttentionBackend:
    '''
    Construct the backend instance determined by the backend_name string
    argument.

    "xformers" -> construct xformers backend

    TODO: flash attention backend

    Returns:

    * Backend instance
    '''
    if backend_name == "xformers":
        return XFormersBackend()
    raise AssertionError(
        f"Unrecognized backend_name {backend_name} for unit test")


def make_metadata_tensors(is_prompt: bool,
                          seq_lens: List[int],
                          context_lens: List[int],
                          device=CUDA_DEVICE) -> tuple:
    '''
    Build scalar & tensor values required to build attention metadata structure.

    Arguments:

    * is_prompt: True -> Prefill, False -> Decode
    * seq_lens: list of token-counts for each seq
    * context_lens: list of context length values for each seq
    * device: CPU or CUDA device

    Returns:

    * seq_lens_tensor: seq_lens list, as tensor
    * context_lens_tensor: context_lens list, as tensor
    * max_query_len: max(seq_lens) if is_seq, o/w 1
    * max_context_len: max(context_lens)
    * max_seq_len: max(seq_lens)
    * seq_start_loc: start idx of each sequence
    * query_start_loc: start idx of each query
    '''
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int, device=device)
    context_lens_tensor = None if context_lens is None else torch.tensor(
        context_lens, dtype=torch.int, device=device)
    max_context_len = None if context_lens is None else max(context_lens)
    max_seq_len = None if seq_lens is None else max(seq_lens)

    seq_start_loc = torch.zeros(seq_lens_tensor.shape[0] + 1,
                                dtype=torch.int32,
                                device=device)

    torch.cumsum(seq_lens_tensor,
                 dim=0,
                 dtype=seq_start_loc.dtype,
                 out=seq_start_loc[1:])

    if is_prompt:
        # Prefill: query_start_loc matches seq_start_loc
        query_start_loc = copy.deepcopy(seq_start_loc)
        max_query_len = max_seq_len
    else:
        # Decode: one new query input token per batch element, thus
        # query_start_loc is the cumsum of [1,1,1,...]
        query_start_loc = list(range(len(seq_start_loc)))
        max_query_len = 1

    return seq_lens_tensor, \
           context_lens_tensor, \
           max_query_len, \
           max_context_len, \
           max_seq_len, \
           seq_start_loc, \
           query_start_loc


def make_kv_cache(num_blocks,
                  num_heads,
                  head_size,
                  block_size,
                  device=CUDA_DEVICE,
                  default_val=0.0):
    '''
    Create a fake KV cache.

    Arguments:

    * num_blocks: number of blocks in the KV cache
    * num_heads: number of attention heads
    * head_size: head dimension
    * block_size: number of offsets within a block
    * device: CPU or CUDA device
    * default_val: initialization value for KV cache elements

    Returns:

    * kv_cache: 2 x num_blocks x (block_size * num_heads * head_size)
    '''

    kv_cache = torch.rand(
        (2, num_blocks, block_size * num_heads * head_size)).to(device)
    if default_val is not None:
        kv_cache[:, :, :] = default_val
    return kv_cache


def num_tokens_to_min_blocks(num_tokens, block_size):
    '''
    Compute the minimum number of blocks required to hold num_tokens tokens,
    given block_size
    '''
    return (num_tokens + block_size) // block_size


def make_block_tables_slot_mapping(block_size,
                                   seq_lens,
                                   block_base_addr=0,
                                   device=CUDA_DEVICE):
    '''
    Construct fake block tables & slot mappings.

    For a sequence with num_tokens tokens the minimum number
    of required KV cache blocks is

    num_blocks = (num_tokens + block_size) // block_size

    Then the minimum KV cache size in blocks is

    total_cache_blocks = sum(num_blocks for all seqs) 

    Then, the blocktable mapping counts downward from

    block_base_addr + total_cache_blocks

    to

    block_base_addr
    

    Arguments:

    * block_size: number of offsets per block
    * seq_lens: list of token-counts for each sequence
    * block_base_addr: the block table base address
    * device: CPU or CUDA device

    Return:

    * decode_block_tables_tensor: fake the state of the block tables during
      decode
    * decode_slot_mapping_tensor: fake the state of the slot mapping during
      decode
    * prefill_slot_mapping_tensor: fake the state of the slot mapping during
      prefill
    * prefill_block_tables_tensor: fake the state of the block tables during
      prefill
    * slot_mapping_tensor: union of prefill and decode slot mappings
    * empty_slot_mapping_tensor: empty slot mapping (useful for decode phase
      cross attention)
    * max_block_idx: the highest block address within this block table
    '''

    # Provision minimum number of KV cache blocks
    num_blocks_list = [
        num_tokens_to_min_blocks(num_tokens, block_size)
        for num_tokens in seq_lens
    ]
    max_block_table_len = max(num_blocks_list)
    block_table_pad_tokens = 10

    block_tables = []
    prefill_slot_mapping = []
    decode_slot_mapping = []
    slot_mapping = []
    # Compute uppermost address of block table
    total_cache_blocks = sum(num_blocks_list)
    block_base_idx = block_base_addr + total_cache_blocks
    max_block_idx = block_base_idx
    for sdx, num_tokens in enumerate(seq_lens):
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

    return decode_block_tables_tensor, \
           decode_slot_mapping_tensor, \
           prefill_slot_mapping_tensor, \
           prefill_block_tables_tensor, \
           slot_mapping_tensor, \
           empty_slot_mapping_tensor, \
           max_block_idx


def make_metadata_self_cross(
    attn_backend: AttentionBackend,
    is_prompt: bool,
    seq_lens: List[int],
    context_lens: List[int],
    block_tables,
    slot_mapping,
    is_encoder_only_test: bool,
    device=CUDA_DEVICE,
    cross_seq_lens: Optional[List[int]] = None,
    cross_block_tables: Optional[torch.Tensor] = None,
    cross_slot_mapping: Optional[List[int]] = None,
) -> AttentionMetadata:
    '''
    Construct fake attention metadata for a combined self-/cross-attention
    scenario i.e. an encoder/decoder model. 

    Assumptions:

    * No chunked prefill -> a batch is 100% prefill or 100% decode, never both

    Arguments:

    * attn_backend: Backend for sourcing attention kernels
    * is_prompt: prefill if True, o/w decode
    * seq_lens: list of token counts for each sequence
    * context_lens: list of context lengths for each sequence
    * block_tables: self-attention block tables
    * slot_mapping: self-attention slot_mapping
    * device: CPU or CUDA device
    * cross_seq_lens: list of token counts for each encoder sequence, if any
      exist
    * cross_block_tables: cross-attention block tables, if required
    * cross_slot_mapping: cross-attention slot mapping, if required

    Return:

    * AttentionMetadata structure supporting self- and cross-attention
    '''

    default_attn_type = AttentionType.ENCODER if is_encoder_only_test \
                          else AttentionType.DECODER

    if is_prompt:
        num_prefills = len(seq_lens)
        num_prefill_tokens = sum(seq_lens)
        num_decode_tokens = 0

        seq_lens_tensor, \
        context_lens_tensor, \
        max_query_len, \
        _, \
        _, \
        seq_start_loc, \
        query_start_loc = make_metadata_tensors(is_prompt,
                                                seq_lens,
                                                context_lens,
                                                device=device)

        slot_mapping_tensor = slot_mapping

        cross_slot_mapping_tensor = cross_slot_mapping

        return attn_backend.make_metadata(
            num_prefills=num_prefills,
            slot_mapping=slot_mapping_tensor,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=max_query_len,
            max_prefill_seq_len=max(seq_lens),
            max_decode_seq_len=0,
            query_start_loc=query_start_loc,
            seq_start_loc=seq_start_loc,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=False,
            _attn_type=default_attn_type,
            cross_seq_lens=cross_seq_lens,
            cross_slot_mapping=cross_slot_mapping_tensor,
            cross_block_tables=cross_block_tables)

    else:  # not is_prompt

        num_prefills = 0
        num_prefill_tokens = 0
        num_decode_tokens = len(seq_lens)

        seq_lens_tensor, \
        context_lens_tensor, \
        max_query_len, \
        _, \
        _, \
        seq_start_loc, \
        query_start_loc = make_metadata_tensors(is_prompt,
                                                seq_lens,
                                                context_lens,
                                                device=device)

        slot_mapping_tensor = slot_mapping

        cross_slot_mapping_tensor = cross_slot_mapping

        return attn_backend.make_metadata(
            num_prefills=num_prefills,
            slot_mapping=slot_mapping_tensor,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=max_query_len,
            max_prefill_seq_len=0,
            max_decode_seq_len=max(seq_lens),
            query_start_loc=query_start_loc,
            seq_start_loc=seq_start_loc,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=False,
            _attn_type=default_attn_type,
            cross_seq_lens=cross_seq_lens,
            cross_slot_mapping=cross_slot_mapping_tensor,
            cross_block_tables=cross_block_tables)


def basic_setup(num_heads, head_size, num_blocks, block_size, backend_name):
    '''
    Compute & build entities required for the self-/cross-attention test.

    Arguments:

    * num_heads: Number of attention heads
    * head_size: Head dimension
    * num_blocks: Number of KV cache blocks (no KV cache if None)
    * block_size: Number of offsets within a KV cache block
                  (no KV cache if None)
    * backend_name: selection of backend

    Returns:

    * scale: 1/sqrt(head_size)
    * attn_backend: backend instance
    * attn: Attention wrapper instance
    * kv_cache: fake KV cache, 2 x num_blocks x (block_size * num_heads *
      head_size)
        * None if num_blocks or block_size is None
    '''

    scale = float(1.0 / (head_size**0.5))
    attn_backend = make_backend(backend_name)
    attn = Attention(
        num_heads,
        head_size,
        scale=scale,
    )
    if num_blocks is None or num_heads is None:
        # Caller does not require a KV cache
        return scale, attn_backend, attn, None

    # Construct KV cache
    kv_cache = make_kv_cache(num_blocks, num_heads, head_size, block_size)
    return scale, attn_backend, attn, kv_cache
    
def encoder_attn_setup(batch_size,
                       num_heads,
                       head_size,
                       block_size,
                       scale,
                       max_q_seq_len,
                       block_base_addr=0):
    '''
    Set up test vectors & data structures for encoder attention test.

    A triplet of synthetic query/key/value tensors are constructed ("baseline"
    query/key/value). Given this is a self-attention test, the key & value
    sequences will have the same length as the corresponding queries.

    "Prefill" query/key/value tensors are derived by masking out the last value
    in each baseline query/key/value. These tensors are used to test prefill &
    populate KV cache for a subsequent decode test.

    "Decode" query/key/value tensors are derived by extracting *only* the last
    value from each baseline query/key/value (i.e. complement of the prefill
    tensors.) These tensors are used to test decode, conditional on the kv cache
    being populated during the prefill test.

    The baseline query/key/value tensors are passed to an ideal reference
    self-attention implementation to generate a "Baseline" ideal output tensor.
    This tensor is split into the "Prefill" ideal output tensor (all but the
    last element of each output sequence) and the "Decode" ideal output tensor
    (*only* the last element of each output sequence); the "Prefill" and
    "Decode" ideal output tensors can be used to validate the prefill and decode
    test results, respectively.

    This function also constructs the self-attention KV cache memory mapping
    (slot mapping and block table), ensuring that the block table starts at
    block_base_addr

    Arguments:

    * batch_size
    * num_heads: Number of attention heads
    * head_size: Head dimension
    * block_size: Number of offsets per KV cache block
    * scale: attention scale parameter
    * max_q_seq_len: upper limit on query length for synthetic test vectors
    * block_base_addr: self-attention block table base address

    Returns:

    * query: "baseline" query; batch_size x padded_seq_len x num_heads x
      head_size
    * prefill_packed_query: "prefill" query; number_of_tokens x num_heads x
      head_size
    * prefill_packed_key: self-attn "prefill" key; number_of_tokens x num_heads
      x head_size
    * prefill_packed_value: self-attn "prefill" value; number_of_tokens x
      num_heads x head_size
    * prefill_packed_ideal_output: self-attn "prefill" ideal output;
      number_of_tokens x num_heads x head_size
    * prefill_q_seq_lens: list of token counts for each *prefill query* (one
      less than baseline query)
    * prefill_kv_seq_lens: list of token counts for each self-attn *prefill
      key/value* (should match prefill_q_seq_lens)
    * decode_packed_query: "decode" query; number_of_tokens x num_heads x
      head_size
    * decode_packed_key: self-attn "decode" key; number_of_tokens x num_heads x
      head_size
    * decode_packed_value: self-attn "decode" key; number_of_tokens x num_heads
      x head_size
    * decode_packed_ideal_output: self-attn "decode" ideal output;
      number_of_tokens x num_heads x head_size
    * decode_q_seq_lens: list of token counts for each *decode query* (should
      be 1)
    * decode_kv_seq_lens: list of token counts for each self-attn *decode
      key/value* (should match decode_q_seq_lens)
    * q_seq_lens: "baseline" query seq lens; number_of_tokens x num_heads x
      head_size
    * kv_seq_lens: self-attn "baseline" key/value seq lens; number_of_tokens
      x num_heads x head_size
    * decode_block_tables: fake self-attn decode-phase block table
    * decode_slot_mapping: fake self-attn decode-phase slot mapping
    * prefill_slot_mapping: fake self-attn prefill-phase slot mapping
    * prefill_block_tables: fake self-attn prefill-phase block table
    * max_block_idx: highest block address in the self-attention block-table
    '''

    max_kv_seq_len = max_q_seq_len

    query, \
    key, \
    value, \
    _, \
    _, \
    _, \
    _, \
    _, \
    _, \
    q_seq_lens, \
    kv_seq_lens, \
    _, \
    _, \
    _, \
    _, \
    _, \
    _ = make_qkv(batch_size,
                 max_q_seq_len,
                 max_kv_seq_len,
                 num_heads,
                 head_size,
                 attn_type=AttentionType.ENCODER)

    # No attention mask
    ideal_output = ref_masked_attention(query,
                                        key,
                                        value,
                                        scale=scale,
                                        q_seq_lens=q_seq_lens,
                                        kv_seq_lens=kv_seq_lens)

    # prefill_ideal_output = torch.zeros_like(ideal_output)
    # for bdx, prefill_q_seq_len in enumerate(prefill_q_seq_lens):
    #     prefill_ideal_output[bdx, :prefill_q_seq_len] = ideal_output[
    #         bdx, :prefill_q_seq_len]

    packed_ideal_output, _ = pack_tensor(ideal_output,
                                         q_seq_lens)

    block_tables, \
    _, \
    _, \
    _, \
    slot_mapping, \
    _, \
    _ = make_block_tables_slot_mapping(
        block_size, q_seq_lens, block_base_addr=block_base_addr)

    packed_query, \
    packed_key, \
    packed_value, _, _ = pack_qkv(
        query, key, value, q_seq_lens,
        kv_seq_lens)

    return packed_query, \
    packed_key, \
    packed_value, \
    packed_ideal_output, \
    block_tables, \
    slot_mapping, \
    q_seq_lens

def decoder_attn_setup(batch_size,
                       num_heads,
                       head_size,
                       block_size,
                       scale,
                       max_q_seq_len,
                       block_base_addr=0):
    '''
    Set up test vectors & data structures for self-attention test.

    A triplet of synthetic query/key/value tensors are constructed ("baseline"
    query/key/value). Given this is a self-attention test, the key & value
    sequences will have the same length as the corresponding queries.

    "Prefill" query/key/value tensors are derived by masking out the last value
    in each baseline query/key/value. These tensors are used to test prefill &
    populate KV cache for a subsequent decode test.

    "Decode" query/key/value tensors are derived by extracting *only* the last
    value from each baseline query/key/value (i.e. complement of the prefill
    tensors.) These tensors are used to test decode, conditional on the kv cache
    being populated during the prefill test.

    The baseline query/key/value tensors are passed to an ideal reference
    self-attention implementation to generate a "Baseline" ideal output tensor.
    This tensor is split into the "Prefill" ideal output tensor (all but the
    last element of each output sequence) and the "Decode" ideal output tensor
    (*only* the last element of each output sequence); the "Prefill" and
    "Decode" ideal output tensors can be used to validate the prefill and decode
    test results, respectively.

    This function also constructs the self-attention KV cache memory mapping
    (slot mapping and block table), ensuring that the block table starts at
    block_base_addr

    Arguments:

    * batch_size
    * num_heads: Number of attention heads
    * head_size: Head dimension
    * block_size: Number of offsets per KV cache block
    * scale: attention scale parameter
    * max_q_seq_len: upper limit on query length for synthetic test vectors
    * block_base_addr: self-attention block table base address

    Returns:

    * query: "baseline" query; batch_size x padded_seq_len x num_heads x
      head_size
    * prefill_packed_query: "prefill" query; number_of_tokens x num_heads x
      head_size
    * prefill_packed_key: self-attn "prefill" key; number_of_tokens x num_heads
      x head_size
    * prefill_packed_value: self-attn "prefill" value; number_of_tokens x
      num_heads x head_size
    * prefill_packed_ideal_output: self-attn "prefill" ideal output;
      number_of_tokens x num_heads x head_size
    * prefill_q_seq_lens: list of token counts for each *prefill query* (one
      less than baseline query)
    * prefill_kv_seq_lens: list of token counts for each self-attn *prefill
      key/value* (should match prefill_q_seq_lens)
    * decode_packed_query: "decode" query; number_of_tokens x num_heads x
      head_size
    * decode_packed_key: self-attn "decode" key; number_of_tokens x num_heads x
      head_size
    * decode_packed_value: self-attn "decode" key; number_of_tokens x num_heads
      x head_size
    * decode_packed_ideal_output: self-attn "decode" ideal output;
      number_of_tokens x num_heads x head_size
    * decode_q_seq_lens: list of token counts for each *decode query* (should
      be 1)
    * decode_kv_seq_lens: list of token counts for each self-attn *decode
      key/value* (should match decode_q_seq_lens)
    * q_seq_lens: "baseline" query seq lens; number_of_tokens x num_heads x
      head_size
    * kv_seq_lens: self-attn "baseline" key/value seq lens; number_of_tokens
      x num_heads x head_size
    * decode_block_tables: fake self-attn decode-phase block table
    * decode_slot_mapping: fake self-attn decode-phase slot mapping
    * prefill_slot_mapping: fake self-attn prefill-phase slot mapping
    * prefill_block_tables: fake self-attn prefill-phase block table
    * max_block_idx: highest block address in the self-attention block-table
    '''

    max_kv_seq_len = max_q_seq_len

    query, \
    key, \
    value, \
    prefill_query, \
    prefill_key, \
    prefill_value, \
    decode_query, \
    decode_key, \
    decode_value, \
    q_seq_lens, \
    kv_seq_lens, \
    _, \
    _, \
    prefill_q_seq_lens, \
    prefill_kv_seq_lens, \
    decode_q_seq_lens, \
    decode_kv_seq_lens = make_qkv(batch_size,
                                  max_q_seq_len,
                                  max_kv_seq_len,
                                  num_heads,
                                  head_size,
                                  attn_type=AttentionType.DECODER)

    causal_mask = build_causal_mask(max_q_seq_len,
                                    max_kv_seq_len).to(CUDA_DEVICE)

    ideal_output = ref_masked_attention(query,
                                        key,
                                        value,
                                        scale=scale,
                                        custom_mask=causal_mask,
                                        q_seq_lens=q_seq_lens,
                                        kv_seq_lens=kv_seq_lens)

    prefill_ideal_output = torch.zeros_like(ideal_output)
    decode_ideal_output = torch.zeros_like(ideal_output[:, 0:1])
    for bdx, prefill_q_seq_len in enumerate(prefill_q_seq_lens):
        prefill_ideal_output[bdx, :prefill_q_seq_len] = ideal_output[
            bdx, :prefill_q_seq_len]
        decode_ideal_output[bdx, :] = ideal_output[bdx, prefill_q_seq_len:(
            prefill_q_seq_len + 1)]

    prefill_packed_ideal_output, _ = pack_tensor(prefill_ideal_output,
                                                 prefill_q_seq_lens)
    decode_packed_ideal_output, _ = pack_tensor(decode_ideal_output,
                                                [1 for _ in range(batch_size)])

    decode_block_tables, \
    decode_slot_mapping, \
    prefill_slot_mapping, \
    prefill_block_tables, \
    _, \
    _, \
    max_block_idx = make_block_tables_slot_mapping(
        block_size, q_seq_lens, block_base_addr=block_base_addr)

    prefill_packed_query, \
    prefill_packed_key, \
    prefill_packed_value, _, _ = pack_qkv(
        prefill_query, prefill_key, prefill_value, prefill_q_seq_lens,
        prefill_kv_seq_lens)

    decode_packed_query, \
    decode_packed_key, \
    decode_packed_value, \
    _, \
    _ = pack_qkv(
        decode_query, decode_key, decode_value, decode_q_seq_lens,
        decode_kv_seq_lens)

    return query, \
    prefill_packed_query, \
    prefill_packed_key, \
    prefill_packed_value, \
    prefill_packed_ideal_output, \
    prefill_q_seq_lens, \
    prefill_kv_seq_lens, \
    decode_packed_query, \
    decode_packed_key, \
    decode_packed_value, \
    decode_packed_ideal_output, \
    decode_q_seq_lens, \
    decode_kv_seq_lens, \
    q_seq_lens, \
    kv_seq_lens, \
    decode_block_tables, \
    decode_slot_mapping, \
    prefill_slot_mapping, \
    prefill_block_tables, \
    max_block_idx


def enc_dec_attn_setup_reuses_query(query,
                                    q_seq_lens,
                                    prefill_q_seq_lens,
                                    batch_size,
                                    num_heads,
                                    head_size,
                                    block_size,
                                    scale,
                                    max_q_seq_len,
                                    max_kv_seq_len,
                                    block_base_addr=0):
    '''
    Set up test vectors & data structures for cross-attention test.

    A triplet of synthetic cross-attention key/value tensors are constructed
    ("baseline" key/value). Given this is a cross-attention test, we assume
    query tensors were already synthesized for a prior self-attention test and
    will be reused for cross-attention. The key & value sequences generated here
    will may have a different length than the corresponding queries (as is often
    the case for cross-attention between decoder and encoder sequences.)

    Cross attention key & value tensors do not grow during autoregressive
    inference; thus this function obtains a single key/value pair suitable for
    both prefill and decode.

    The "baseline" query tensor is received as an argument. The "baseline"
    query/key/value tensors are passed to an ideal reference cross-attention
    implementation to generate a "baseline" ideal output tensor. This tensor is
    split into the "Prefill" ideal output tensor (all but the last element of
    each output sequence) and the "Decode" ideal output tensor (*only* the last
    element of each output sequence); the "Prefill" and "Decode" ideal output
    tensors can be used to validate the prefill and decode test results,
    respectively.

    This function also constructs the cross-attention KV cache memory mapping
    (slot mapping and block table), ensuring that the block table starts at
    block_base_addr. 

    Arguments:

    * query: pre-existing "baseline" query; batch_size x padded_seq_len x
      num_heads x head_size
    * q_seq_lens: list of token-counts for each "baseline" query sequence
    * prefill_q_seq_lens:  list of token-counts for each "prefill" query
      sequence
    * batch_size
    * num_heads: Number of attention heads
    * head_size: Head dimension
    * block_size: Number of offsets per KV cache block
    * scale: attention scale parameter
    * max_q_seq_len: upper limit on query length for synthetic test vectors
    * max_kv_seq_len: upper limit on key/value length for synthetic test
      vectors
    * block_base_addr: cross-attention block table base address

    Returns:

    * packed_key: cross-attention key; number_of_tokens x num_heads x head_size
    * packed_value: cross-attention value; number_of_tokens x num_heads x
      head_size
    * prefill_packed_ideal_output: "prefill" ideal output; number_of_tokens x
      num_heads x head_size
    * decode_packed_ideal_output: "decode" ideal output; number_of_tokens x
      num_heads x head_size
    * kv_seq_lens: list of token-counts for each key/value
    * decode_block_tables: fake decode-phase block tables
    * decode_slot_mapping: fake decode-phase slot mapping
    * prefill_slot_mapping: fake prefill-phase slot mapping
    * prefill_block_tables: fake prefill-phase block tables
    * max_block_idx: highest block address in the cross-attention block-table
    '''

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
    kv_seq_lens, \
    _, \
    _, \
    _, \
    _, \
    _, \
    _ = make_qkv(batch_size,
                 max_q_seq_len,
                 max_kv_seq_len,
                 num_heads,
                 head_size,
                 attn_type=AttentionType.ENCODER_DECODER)

    ideal_output = ref_masked_attention(query,
                                        key,
                                        value,
                                        scale=scale,
                                        q_seq_lens=q_seq_lens,
                                        kv_seq_lens=kv_seq_lens)

    prefill_ideal_output = torch.zeros_like(ideal_output)
    decode_ideal_output = torch.zeros_like(ideal_output[:, 0:1])
    for bdx, prefill_q_seq_len in enumerate(prefill_q_seq_lens):
        prefill_ideal_output[bdx, :prefill_q_seq_len] = ideal_output[
            bdx, :prefill_q_seq_len]
        decode_ideal_output[bdx, :] = ideal_output[bdx, prefill_q_seq_len:(
            prefill_q_seq_len + 1)]

    prefill_packed_ideal_output, _ = pack_tensor(prefill_ideal_output,
                                                 prefill_q_seq_lens)
    decode_packed_ideal_output, _ = pack_tensor(decode_ideal_output,
                                                [1 for _ in range(batch_size)])

    # Unlike self-attention:
    # - Prefill slot-mapping includes all key slots
    # - Decode slot-mapping is empty
    decode_block_tables, \
    _, \
    _, \
    prefill_block_tables, \
    prefill_slot_mapping, \
    decode_slot_mapping, \
    max_block_idx = make_block_tables_slot_mapping(
        block_size, kv_seq_lens, block_base_addr=block_base_addr)

    # Packed key/value (query is already provided)
    _, packed_key, packed_value, _, _ = pack_qkv(None, key, value, None,
                                                 kv_seq_lens)

    return packed_key, \
    packed_value, \
    prefill_packed_ideal_output, \
    decode_packed_ideal_output, \
    kv_seq_lens, \
    decode_block_tables, \
    decode_slot_mapping, \
    prefill_slot_mapping, \
    prefill_block_tables, \
    max_block_idx


def run_self_attention_test(attn: Attention, packed_query, packed_key,
                            packed_value, kv_cache,
                            attn_metadata: AttentionMetadata,
                            attn_type: AttentionType):
    attn_metadata.attention_type = attn_type
    return attn.forward(packed_query, packed_key, packed_value, kv_cache,
                        attn_metadata)


def run_cross_attention_test(attn: Attention, packed_query, packed_key,
                             packed_value, kv_cache,
                             attn_metadata: AttentionMetadata):
    attn_metadata.attention_type = AttentionType.ENCODER_DECODER
    return attn.forward(packed_query, packed_key, packed_value, kv_cache,
                        attn_metadata)

@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("backend_name", BACKEND_NAMES)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("max_seq_len", MAX_Q_SEQ_LENS)
def test_encoder_attention(num_heads: int, head_size: int, backend_name: str,
                           batch_size: int, block_size: int,
                           max_seq_len: int) -> None:

    '''
    Encoder-only attention test:

    * Construct fake test vectors for self- and cross-attention
    * Construct attention metadata structure with self- and cross-attention
      attributes
    * Test self- and cross-attention in the following order
    
        * Prefill self-attention
        * Prefill cross-attention
        * Decode self-attention
        * Decode cross-attention
        * This order would exacerbate any accidental overlap in the
          self-/cross-attention block tables, which we attempt to avoid
    * Validate output correctness against ideal reference attention
      implementation

    Block tables are constructed such that cross-attention KV cache is in a
    higher, non-intersecting address-space than self-attention KV cache.

    Self- and cross-attention share the same query tensor but not the K/V
    tensors. Self-attention K/Vs must have the same seq len as Q while
    cross-attention K/Vs are allowed to differ in seq len, as is often the case
    for cross-attention.
    '''

    # Num KV cache blocks
    # num_blocks = 4096

    # Attention scale factor, attention backend instance, attention wrapper
    # instance, KV cache init
    scale, \
    attn_backend, \
    attn, \
    _ = basic_setup(num_heads,
                           head_size,
                           None,
                           None,
                           backend_name)

    # Self-attention setup
    # Let encoder_attn_setup() choose default block table
    # base address
    packed_query, \
    packed_key, \
    packed_value, \
    packed_ideal_output, \
    block_tables, \
    slot_mapping, \
    q_seq_lens = encoder_attn_setup(batch_size,
                                     num_heads,
                                     head_size,
                                     block_size,
                                     scale,
                                     max_seq_len)

    context_lens = [0 for _ in range(batch_size)]

    attn_metadata: AttentionMetadata = make_metadata_self_cross(
        attn_backend,
        True,
        q_seq_lens,
        context_lens,
        block_tables,
        slot_mapping,
        is_encoder_only_test=True,
        cross_seq_lens=None,
        cross_block_tables=None,
        cross_slot_mapping=None,
    )

    packed_actual_output: torch.Tensor = run_self_attention_test(
        attn,
        packed_query,
        packed_key,
        packed_value,
        None,
        attn_metadata,
        attn_type=AttentionType.ENCODER)

    # - Is encoder attention result correct?
    assert torch.allclose(
        packed_ideal_output,
        packed_actual_output.view_as(
            packed_ideal_output))

@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("backend_name", BACKEND_NAMES)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("max_q_seq_len", MAX_Q_SEQ_LENS)
@pytest.mark.parametrize("max_kv_seq_len", MAX_K_SEQ_LENS)
def test_enc_dec_self_and_cross_attention_prefill_decode_phases(
        num_heads: int, head_size: int, backend_name: str, batch_size: int,
        block_size: int, max_q_seq_len: int, max_kv_seq_len: int) -> None:
    '''
    Encoder/decoder attention test:

    * Construct fake test vectors for self- and cross-attention
    * Construct attention metadata structure with self- and cross-attention
      attributes
    * Test self- and cross-attention in the following order
    
        * Prefill self-attention
        * Prefill cross-attention
        * Decode self-attention
        * Decode cross-attention
        * This order would exacerbate any accidental overlap in the
          self-/cross-attention block tables, which we attempt to avoid
    * Validate output correctness against ideal reference attention
      implementation

    Block tables are constructed such that cross-attention KV cache is in a
    higher, non-intersecting address-space than self-attention KV cache.

    Self- and cross-attention share the same query tensor but not the K/V
    tensors. Self-attention K/Vs must have the same seq len as Q while
    cross-attention K/Vs are allowed to differ in seq len, as is often the case
    for cross-attention.
    '''

    # Num KV cache blocks
    num_blocks = 4096

    # Attention scale factor, attention backend instance, attention wrapper
    # instance, KV cache init
    scale, \
    attn_backend, \
    attn, \
    kv_cache = basic_setup(num_heads,
                           head_size,
                           num_blocks,
                           block_size,
                           backend_name)

    # Self-attention setup

    self_block_base_addr = 0

    query, \
    prefill_packed_query, \
    self_prefill_packed_key, \
    self_prefill_packed_value, \
    self_prefill_packed_ideal_output, \
    prefill_q_seq_lens, \
    self_prefill_kv_seq_lens, \
    decode_packed_query, \
    self_decode_packed_key, \
    self_decode_packed_value, \
    self_decode_packed_ideal_output, \
    _, \
    _, \
    q_seq_lens, \
    _, \
    self_decode_block_tables, \
    self_decode_slot_mapping, \
    self_prefill_slot_mapping, \
    self_prefill_block_tables, \
    cross_block_base_addr = decoder_attn_setup(batch_size,
                                                num_heads,
                                                head_size,
                                                block_size,
                                                scale,
                                                max_q_seq_len,
                                                block_base_addr=self_block_base_addr)

    # Cross-attention setup

    cross_prefill_packed_key, \
    cross_prefill_packed_value, \
    cross_prefill_packed_ideal_output, \
    cross_decode_packed_ideal_output, \
    cross_kv_seq_lens, \
    cross_decode_block_tables, \
    cross_decode_slot_mapping, \
    cross_prefill_slot_mapping, \
    cross_prefill_block_tables, \
    _ = enc_dec_attn_setup_reuses_query(query,
                                      q_seq_lens,
                                      prefill_q_seq_lens,
                                      batch_size,
                                      num_heads,
                                      head_size,
                                      block_size,
                                      scale,
                                      max_q_seq_len,
                                      max_kv_seq_len,
                                      block_base_addr=cross_block_base_addr)

    # PREFILL: self- and cross-attention tests

    context_lens = [0 for _ in range(batch_size)]

    prefill_attn_metadata: AttentionMetadata = make_metadata_self_cross(
        attn_backend,
        True,
        prefill_q_seq_lens,
        context_lens,
        self_prefill_block_tables,
        self_prefill_slot_mapping,
        is_encoder_only_test=False,
        cross_seq_lens=cross_kv_seq_lens,
        cross_block_tables=cross_prefill_block_tables,
        cross_slot_mapping=cross_prefill_slot_mapping,
    )

    self_prefill_packed_actual_output: torch.Tensor = run_self_attention_test(
        attn,
        prefill_packed_query,
        self_prefill_packed_key,
        self_prefill_packed_value,
        kv_cache,
        prefill_attn_metadata,
        attn_type=AttentionType.DECODER)

    # - Prefill self-attention correct?
    assert torch.allclose(
        self_prefill_packed_ideal_output,
        self_prefill_packed_actual_output.view_as(
            self_prefill_packed_ideal_output))

    cross_prefill_packed_actual_output: torch.Tensor = run_cross_attention_test(
        attn, prefill_packed_query, cross_prefill_packed_key,
        cross_prefill_packed_value, kv_cache, prefill_attn_metadata)

    # - Prefill cross-attention correct?
    assert torch.allclose(
        cross_prefill_packed_ideal_output,
        cross_prefill_packed_actual_output.view_as(
            cross_prefill_packed_ideal_output))

    context_lens = copy.deepcopy(self_prefill_kv_seq_lens)

    # DECODE: self- and cross-attention tests

    decode_attn_metadata: AttentionMetadata = make_metadata_self_cross(
        attn_backend,
        False,
        q_seq_lens,
        context_lens,
        self_decode_block_tables,
        self_decode_slot_mapping,
        is_encoder_only_test=False,
        cross_seq_lens=cross_kv_seq_lens,
        cross_block_tables=cross_decode_block_tables,
        cross_slot_mapping=cross_decode_slot_mapping,
    )

    self_decode_packed_actual_output: torch.Tensor = run_self_attention_test(
        attn,
        decode_packed_query,
        self_decode_packed_key,
        self_decode_packed_value,
        kv_cache,
        decode_attn_metadata,
        attn_type=AttentionType.DECODER)

    # - Decode self-attention correct?
    assert torch.allclose(
        self_decode_packed_ideal_output,
        self_decode_packed_actual_output.view_as(
            self_decode_packed_ideal_output))

    cross_decode_packed_actual_output: torch.Tensor = run_cross_attention_test(
        attn, decode_packed_query, None, None, kv_cache, decode_attn_metadata)

    # - Decode cross-attention correct?
    assert torch.allclose(
        cross_decode_packed_ideal_output,
        cross_decode_packed_actual_output.view_as(
            cross_decode_packed_ideal_output))
