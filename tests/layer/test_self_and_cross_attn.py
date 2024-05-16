import random
from typing import List, Optional
import itertools

import pytest
import torch
import copy
from vllm.attention import Attention, AttentionMetadata #, AttentionMetadataPerStage

from vllm.attention.backends.xformers import XFormersBackend
from vllm.attention.backends.abstract import AttentionBackend

from vllm.attention.ops.paged_attn import PagedAttention

from vllm.utils import get_max_shared_memory_bytes
from vllm.utils import is_hip
from vllm.utils import make_tensor_with_pad

from vllm.attention.layer import Attention

import random

# FlashAttention forward only supports head dimension at most 128
# https://github.com/ROCmSoftwarePlatform/flash-attention/blob/3d2b6f5d037782cc2c906909a46fb7e2e1b48b25/csrc/flash_attn_rocm/flash_api.cpp#L62
HEAD_SIZES = [64] 

#               [64, 80, 96, 112, 128, 256
#               ] if not is_hip() else [64, 80, 96, 112, 128]

NUM_HEADS = [1]

BATCH_SIZES = [16]
BLOCK_SIZES = [16]
#KV_CACHE_DTYPE = ["auto", "fp8_e5m2"]
BACKEND_NAMES = ["xformers"]
#CUDA_DEVICES = [
#    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
#]

PROMPT_LENS = [32]

def build_causal_mask(q_max_prompt_len, k_max_prompt_len):
    # Create a matrix where entry (i, j) is True if i >= j
    mask = torch.triu(torch.ones(q_max_prompt_len, k_max_prompt_len), diagonal=1) #.transpose(0, 1)
    # Replace True with float('-inf') and False with 0
    mask = mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, 0.0)
    return mask

def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    #query=query.unsqueeze(-2)
    #key=key.unsqueeze(-2)
    #value=value.unsqueeze(-2)
    #assert False,f"{query.shape} ; {key.shape}"
    attn_weights = scale * torch.einsum("bqhd,bkhd->bhqk", query, key).float()
    #assert False,f"{query.shape} ; {key.shape} ; {attn_weights.shape}"
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask.float()
    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.einsum("bhqk,bkhd->bqhd", attn_weights, value)
    #assert False, f"{attn_weights.shape} ; {value.shape} ; {out.shape}"
    return out

def make_qkv(batch_size,max_q_prompt_len,max_kv_prompt_len,head_size, is_cross_attn=True, force_max_len=True):
    if force_max_len:
        q_prompt_lens = [max_q_prompt_len for _ in range(batch_size)]
        kv_prompt_lens = None
        if not is_cross_attn:
            # K,V prompt lens match Q for self-attention
            kv_prompt_lens = q_prompt_lens
        else:
            # K,V prompt lens come from K,V operands
            kv_prompt_lens = [max_kv_prompt_len for _ in range(batch_size)]
    else:
        q_prompt_lens = [random.randint(1, max_q_prompt_len) for _ in range(batch_size)]
        kv_prompt_lens = None
        if not is_cross_attn:
            # K,V prompt lens match Q for self-attention
            kv_prompt_lens = q_prompt_lens
        else:
            # K,V prompt lens come from K,V operands
            kv_prompt_lens = [random.randint(1, max_q_prompt_len) for _ in range(batch_size)]
    
    query=torch.rand((batch_size,max_q_prompt_len,head_size))
    key=torch.rand((batch_size,max_kv_prompt_len,head_size))
    value=torch.rand((batch_size,max_kv_prompt_len,head_size))

    for bdx,(q_prompt_len,kv_prompt_len) in enumerate(zip(q_prompt_lens,kv_prompt_lens)):
        query[bdx,q_prompt_len:] = 0
        key[bdx,kv_prompt_len:] = 0
        value[bdx,kv_prompt_len:] = 0

    query=query.unsqueeze(-2)
    key=key.unsqueeze(-2)
    value=value.unsqueeze(-2)

    return query,key,value,q_prompt_lens,kv_prompt_lens

def pack_tensor(unpacked_tensor,prompt_lens, device='cuda:0'):
    num_tok = sum(prompt_lens)
    num_heads = unpacked_tensor.shape[-2]
    head_size = unpacked_tensor.shape[-1]
    start_loc_list = [0]+list(itertools.accumulate(prompt_lens))
    packed_tensor = torch.zeros((num_tok,num_heads,head_size),
                                device=device)

    #assert False, f"{start_loc_list}"

    #assert False, f"{packed_tensor.shape} ; {unpacked_tensor.shape}"

    for bdx,(prompt_len,start_loc) in enumerate(zip(prompt_lens,start_loc_list)):
        try:
            packed_tensor[start_loc:(start_loc+prompt_len),:,:] = unpacked_tensor[bdx,:prompt_len,:,:]
        except:
            assert False, f"{start_loc} ; {prompt_len} ; {packed_tensor.shape} ; {unpacked_tensor.shape}"

    return packed_tensor,start_loc_list
    
def pack_qkv(query,key,value,q_prompt_lens,kv_prompt_lens):
    packed_query,q_start_loc_list = pack_tensor(query,q_prompt_lens)
    packed_key,kv_start_loc_list = pack_tensor(key,kv_prompt_lens)
    packed_value,_ = pack_tensor(value,kv_prompt_lens)
    packed_query=packed_query.view(-1,packed_query.shape[-1]*packed_query.shape[-2])
    packed_key=packed_key.view(-1,packed_key.shape[-1]*packed_key.shape[-2])
    packed_value=packed_value.view(-1,packed_value.shape[-1]*packed_value.shape[-2])      
    return packed_query,packed_key,packed_value,q_start_loc_list,kv_start_loc_list

def make_backend(backend_name: str) -> AttentionBackend:
    if backend_name == "xformers":
        return XFormersBackend()
    assert False, f"Unrecognized backend_name {backend_name} for unit test"

def make_stage_metadata(attn_backend:AttentionBackend, is_prompt:bool, is_cross_attn:bool, prompt_lens:List[int], context_lens:List[int], block_tables, device='cuda:0', cross_prompt_lens:Optional[List[int]] = None) -> AttentionMetadataPerStage:
    '''
    Assumptions:
    * No chunked prefill
    * No (automatic) prefix caching
    * Packed variable-length sequences
    '''
    prompt_lens_tensor=torch.tensor(prompt_lens,
                                    dtype=torch.int,
                                    device=device)
    context_lens_tensor=None if context_lens is None else torch.tensor(context_lens, 
                                                                       dtype=torch.int,
                                                                       device=device)
    max_query_len=None if prompt_lens is None else max(prompt_lens)
    max_context_len=None if context_lens is None else max(context_lens)
    max_prompt_len=max_query_len

    seq_start_loc = torch.zeros(prompt_lens_tensor.shape[0] + 1,
                                dtype=torch.int32,
                                device=device)

    torch.cumsum(prompt_lens_tensor,
                 dim=0,
                 dtype=seq_start_loc.dtype,
                 out=seq_start_loc[1:])
    query_start_loc = copy.deepcopy(seq_start_loc)

    return attn_backend.make_metadata(
                            is_prompt=is_prompt,
                            is_cross_attn=is_cross_attn,
                            seq_lens=prompt_lens,
                            seq_lens_tensor=prompt_lens_tensor,
                            cross_seq_lens=cross_prompt_lens,
                            max_query_len=max_query_len,
                            #max_context_len=max_context_len,
                            max_seq_len=max_prompt_len,
                            subquery_start_loc=query_start_loc,
                            seq_start_loc=seq_start_loc,
                            context_lens_tensor=context_lens_tensor,
                            block_tables=block_tables,
                            use_cuda_graph=False,
                        )

def make_kv_cache(num_blocks, num_heads, head_size,  block_size, key_read_width, device='cuda:0', default_val=0.0):
    #key_cache = torch.rand((num_blocks, num_heads, head_size//key_read_width, block_size, key_read_width),device=device)
    #val_cache = torch.rand((num_blocks, num_heads, head_size, block_size),device=device)
    kv_cache = torch.rand((2, num_blocks, block_size * num_heads * head_size)).to(device)
    if default_val is not None:
        kv_cache[:,:,:] = default_val
    return kv_cache

def num_tokens_to_min_blocks(num_tokens,block_size):
    return (num_tokens+block_size)//block_size

def make_flat_block_tables_slot_mapping(block_size,prompt_lens):
    '''
    Naive block table:
    * For each batch element...
    * Block table has 
    '''
    num_tokens = sum(prompt_lens)
    num_blocks = num_tokens_to_min_blocks(num_tokens,block_size)
    block_tables = list(range(num_blocks*100))
    slot_mapping = [(idx % block_size) + block_tables[idx//block_size]*block_size for idx in range(num_tokens)]
    prefill_block_tables_tensor = torch.tensor(
        [],
        device='cuda:0'
    )
    block_tables_tensor = torch.tensor(
        block_tables,
        device='cuda:0'
    )
    slot_mapping_tensor = torch.tensor(
        slot_mapping,
        dtype=torch.long,
        device='cuda:0'
    )

    return block_tables_tensor, slot_mapping_tensor, prefill_block_tables_tensor

def make_block_tables_slot_mapping(block_size,prompt_lens,device='cuda:0'):
    '''
    Naive block table:
    * For each batch element...
    * Block table has 
    '''
    num_prompts = len(prompt_lens)
    total_num_tokens = sum(prompt_lens)
    # Over-provision block table blocks by 1
    num_blocks_list = [num_tokens_to_min_blocks(num_tokens,block_size)+1 for num_tokens in prompt_lens]
    max_block_table_len = max(num_blocks_list)
    #block_tables = [list(range(num_blocks*10)) for num_blocks in num_blocks_list]
    block_table_pad_tokens = 10

    block_tables = []
    slot_mapping = []
    block_base_idx = sum(num_blocks_list)*2-1 # Support more blocks than needed
    #seq_base_idx = 0
    for sdx,num_tokens in enumerate(prompt_lens):
        #num_blocks = num_tokens_to_min_blocks(num_tokens,block_size)
        num_blocks = num_blocks_list[sdx]
        block_table = list(range(block_base_idx,block_base_idx-num_blocks,-1))
        for idx in range(num_tokens):
            slot_mapping.append((idx % block_size) + block_table[idx//block_size]*block_size)

        #seq_base_idx += num_tokens
        block_base_idx -= num_blocks
        block_tables.append(block_table)
    
    prefill_block_tables_tensor = torch.tensor(
        [],
        device='cuda:0'
    )
    block_tables_tensor = make_tensor_with_pad(
        block_tables,
        max_len=max_block_table_len+block_table_pad_tokens,
        pad=0,
        dtype=torch.int,
        device=device,
    )
    slot_mapping_tensor = torch.tensor(
        slot_mapping,
        dtype=torch.long,
        device=device
    )

    return block_tables_tensor, slot_mapping_tensor, prefill_block_tables_tensor
    
        
def make_metadata(attn_backend:AttentionBackend, is_prompt:bool, is_cross_attn:bool, prompt_lens:List[int], context_lens:List[int], block_tables, slot_mapping, device='cuda:0', kv_cache_dtype='auto', cross_prompt_lens:Optional[List[int]] = None):
    '''
    Assumptions:
    * No chunked prefill -> a batch is 100% prefill or 100% decode, never both
    '''

    if is_prompt:
        num_prefills = len(prompt_lens)
        num_prefill_tokens = sum(prompt_lens)
        num_decode_tokens = 0

        # make_stage_metadata(attn_backend:AttentionBackend, is_prompt:bool, is_cross_attn:bool, prompt_lens:List[int], context_lens:List[int], block_tables, device='cuda:0', cross_prompt_lens:Optional[List[int]] = None)
        stage_metadata:AttentionMetadataPerStage = make_stage_metadata(attn_backend, is_prompt, is_cross_attn, prompt_lens, context_lens, block_tables, device=device, cross_prompt_lens=cross_prompt_lens)

        return AttentionMetadata(
            num_prefills=num_prefills,
            slot_mapping=slot_mapping,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            prefill_metadata=stage_metadata,
            decode_metadata=None,
            kv_cache_dtype=kv_cache_dtype,
        )

    else: # not is_prompt

        num_prefills = 0
        num_prefill_tokens = 0
        num_decode_tokens = sum(context_lens)

        stage_metadata:AttentionMetadataPerStage = make_stage_metadata(attn_backend, is_prompt, is_cross_attn, prompt_lens, context_lens, block_tables, device=device, cross_prompt_lens=cross_prompt_lens)

        return AttentionMetadata(
            num_prefills=num_prefills,
            slot_mapping=slot_mapping,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            prefill_metadata=None,
            decode_metadata=stage_metadata,
            kv_cache_dtype=kv_cache_dtype,
        )

def make_attention(num_heads: int, head_size: int, scale: float):
    # Attention operator instance
    return Attention(num_heads,
                     head_size,
                     scale=scale,)

@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("backend_name", BACKEND_NAMES)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("block_size",BLOCK_SIZES)
@pytest.mark.parametrize("max_prompt_len",PROMPT_LENS)
def test_prefill_decode_self_attention(num_heads: int, head_size: int, backend_name: str, batch_size: int, block_size: int, max_prompt_len: int) -> None:
    # Attention operator instance
    is_cross_attn=False
    device='cuda:0'
    kv_cache_dtype='auto'
    is_prompt = True
    max_q_prompt_len = max_prompt_len
    max_kv_prompt_len = max_q_prompt_len
    context_lens = None
    key_read_width = 4
    num_blocks = 4096
    kv_cache = make_kv_cache(num_blocks, num_heads, head_size,  block_size, key_read_width, device='cuda:0')
    key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, num_heads, head_size)
    #(key_cache, value_cache) = kv_cache
    scale = float(1.0 / (head_size**0.5))
    attn = make_attention(num_heads, head_size, scale)
    attn_backend = make_backend(backend_name)
    query,key,value,q_prompt_lens,kv_prompt_lens = make_qkv(batch_size,max_q_prompt_len,max_kv_prompt_len,head_size,is_cross_attn=False)
    #block_tables, slot_mapping = make_block_tables_slot_mapping(block_size,q_prompt_lens)
    #prefill_attn_metadata:AttentionMetadata = make_metadata(attn_backend, is_prompt, is_cross_attn,q_prompt_lens, context_lens, block_tables, slot_mapping, device=device, kv_cache_dtype=kv_cache_dtype, cross_prompt_lens=None)
    causal_mask = build_causal_mask(max_q_prompt_len, max_kv_prompt_len)
    ideal_output = ref_masked_attention(
        query,
        key,
        value,
        scale=scale,
        attn_mask=causal_mask
    )

    prefill_query = query[:,:-1]
    prefill_key = key[:,:-1]
    prefill_value = value[:,:-1]
    decode_query = query[:,-1:]
    decode_key = key[:,-1:]
    decode_value = value[:,-1:]
    prefill_q_prompt_lens = [plen-1 for plen in q_prompt_lens]
    prefill_kv_prompt_lens = [plen-1 for plen in kv_prompt_lens]
    decode_q_prompt_lens = [1 for _ in q_prompt_lens]
    decode_kv_prompt_lens = [1 for _ in kv_prompt_lens]
    prefill_ideal_output = ideal_output[:,:-1]
    prefill_packed_ideal_output,_ = pack_tensor(prefill_ideal_output,prefill_q_prompt_lens)
    decode_ideal_output = ideal_output[:,-1:]
    decode_packed_ideal_output,_ = pack_tensor(decode_ideal_output,[1 for _ in range(batch_size)])

    block_tables, slot_mapping, prefill_block_tables = make_block_tables_slot_mapping(block_size,prefill_q_prompt_lens)
    prefill_attn_metadata:AttentionMetadata = make_metadata(attn_backend, is_prompt, is_cross_attn,prefill_q_prompt_lens, context_lens, prefill_block_tables, slot_mapping, device=device, kv_cache_dtype=kv_cache_dtype, cross_prompt_lens=None)

    prefill_packed_query,prefill_packed_key,prefill_packed_value,prefill_q_start_loc_list,prefill_kv_start_loc_list = pack_qkv(prefill_query,prefill_key,prefill_value,prefill_q_prompt_lens,prefill_kv_prompt_lens)

    prefill_packed_actual_output=attn.forward(prefill_packed_query,prefill_packed_key,prefill_packed_value,kv_cache,prefill_attn_metadata,scale)

    # eval correctness of prefill output
    assert torch.allclose(prefill_packed_actual_output,prefill_packed_ideal_output[:,0,:])

    # Put KVs in KV cache
    # Deprecated - handled automatically inside attention
    # PagedAttention.write_to_paged_cache(key, value, key_cache,
    #                                     value_cache,
    #                                     prefill_attn_metadata.slot_mapping,
    #                                     prefill_attn_metadata.kv_cache_dtype,
    #                                     scale)

    is_prompt = False
    context_lens = [1 for _ in range(batch_size)]
    decode_attn_metadata = make_metadata(attn_backend, is_prompt, is_cross_attn, q_prompt_lens, context_lens, block_tables, slot_mapping, device=device, kv_cache_dtype=kv_cache_dtype)
    
    decode_packed_query,decode_packed_key,decode_packed_value,decode_q_start_loc_list,decode_kv_start_loc_list = pack_qkv(decode_query,decode_key,decode_value,decode_q_prompt_lens,decode_kv_prompt_lens)

    decode_packed_actual_output=attn.forward(decode_packed_query,decode_packed_key,decode_packed_value,kv_cache,decode_attn_metadata,scale)

    # eval correctness of decode output
    assert torch.allclose(decode_packed_actual_output,decode_packed_ideal_output[:,0,:]) 

@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("backend_name", BACKEND_NAMES)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("block_size",BLOCK_SIZES)
@pytest.mark.parametrize("max_q_prompt_len",PROMPT_LENS)
@pytest.mark.parametrize("max_kv_prompt_len",PROMPT_LENS)
def test_prefill_decode_cross_attention(num_heads: int, head_size: int, backend_name: str, batch_size: int, block_size: int, max_q_prompt_len: int, max_kv_prompt_len: int) -> None:
    # Attention operator instance
    is_cross_attn=True
    device='cuda:0'
    kv_cache_dtype='auto'
    is_prompt = True
    #max_q_prompt_len = max_prompt_len
    #max_kv_prompt_len = max_prompt_len
    context_lens = None
    key_read_width = 4
    num_blocks = 4096
    kv_cache = make_kv_cache(num_blocks, num_heads, head_size,  block_size, key_read_width, device='cuda:0')
    # key_cache, value_cache = PagedAttention.split_kv_cache(
    #             kv_cache, num_heads, head_size)
    #(key_cache, value_cache) = kv_cache
    scale = float(1.0 / (head_size**0.5))
    attn = make_attention(num_heads, head_size, scale)
    attn_backend = make_backend(backend_name)
    query,key,value,q_prompt_lens,kv_prompt_lens = make_qkv(batch_size,max_q_prompt_len,max_kv_prompt_len,head_size,is_cross_attn=True)
    #block_tables, slot_mapping = make_block_tables_slot_mapping(block_size,q_prompt_lens)
    #prefill_attn_metadata:AttentionMetadata = make_metadata(attn_backend, is_prompt, is_cross_attn,q_prompt_lens, context_lens, block_tables, slot_mapping, device=device, kv_cache_dtype=kv_cache_dtype, cross_prompt_lens=None)
    #causal_mask = build_causal_mask(max_q_prompt_len, max_kv_prompt_len)
    ideal_output = ref_masked_attention(
        query,
        key,
        value,
        scale=scale,
        #attn_mask=causal_mask
    )

    prefill_query = query[:,:-1]
    prefill_key = key #key[:,:-1]
    prefill_value = value #value[:,:-1]
    decode_query = query[:,-1:]
    decode_key = key #key[:,-1:]
    decode_value = value #value[:,-1:]
    prefill_q_prompt_lens = [plen-1 for plen in q_prompt_lens]
    prefill_kv_prompt_lens = kv_prompt_lens
    decode_q_prompt_lens = [1 for _ in q_prompt_lens]
    decode_kv_prompt_lens = kv_prompt_lens
    prefill_ideal_output = ideal_output[:,:-1]
    prefill_packed_ideal_output,_ = pack_tensor(prefill_ideal_output,prefill_q_prompt_lens)
    decode_ideal_output = ideal_output[:,-1:]
    decode_packed_ideal_output,_ = pack_tensor(decode_ideal_output,[1 for _ in range(batch_size)])

    block_tables, slot_mapping, prefill_block_tables = make_block_tables_slot_mapping(block_size,prefill_kv_prompt_lens)
    prefill_attn_metadata:AttentionMetadata = make_metadata(attn_backend, is_prompt, is_cross_attn,prefill_q_prompt_lens, context_lens, prefill_block_tables, slot_mapping, device=device, kv_cache_dtype=kv_cache_dtype, cross_prompt_lens=prefill_kv_prompt_lens)

    prefill_packed_query,prefill_packed_key,prefill_packed_value,prefill_q_start_loc_list,prefill_kv_start_loc_list = pack_qkv(prefill_query,prefill_key,prefill_value,prefill_q_prompt_lens,prefill_kv_prompt_lens)

    prefill_packed_actual_output=attn.forward(prefill_packed_query,prefill_packed_key,prefill_packed_value,kv_cache,prefill_attn_metadata,scale)

    # eval correctness of prefill output
    assert torch.allclose(prefill_packed_actual_output,prefill_packed_ideal_output[:,0,:])

    is_prompt = False
    context_lens = [1 for _ in range(batch_size)]
    decode_attn_metadata = make_metadata(attn_backend, is_prompt, is_cross_attn, q_prompt_lens, context_lens, block_tables, slot_mapping, device=device, kv_cache_dtype=kv_cache_dtype, cross_prompt_lens=kv_prompt_lens)
    
    decode_packed_query,decode_packed_key,decode_packed_value,decode_q_start_loc_list,decode_kv_start_loc_list = pack_qkv(decode_query,decode_key,decode_value,decode_q_prompt_lens,decode_kv_prompt_lens)

    decode_packed_actual_output=attn.forward(decode_packed_query,decode_packed_key,decode_packed_value,kv_cache,decode_attn_metadata,scale)

    # eval correctness of decode output
    assert torch.allclose(decode_packed_actual_output,decode_packed_ideal_output[:,0,:])