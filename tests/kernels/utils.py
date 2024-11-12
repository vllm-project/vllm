"""Kernel test utils"""

import itertools
import random
import unittest
from numbers import Number
from typing import (Any, Dict, List, NamedTuple, Optional, Sequence, Tuple,
                    Union)

import pytest
import torch
from torch._prims_common import TensorLikeType

from vllm.attention import AttentionBackend, AttentionMetadata, AttentionType
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.utils import (STR_BACKEND_ENV_VAR, STR_FLASH_ATTN_VAL,
                        STR_XFORMERS_ATTN_VAL, make_tensor_with_pad)

# For now, disable "test_aot_dispatch_dynamic" since there are some
# bugs related to this test in PyTorch 2.4.
DEFAULT_OPCHECK_TEST_UTILS: Tuple[str, ...] = (
    "test_schema",
    "test_autograd_registration",
    "test_faketensor",
)

ALL_OPCHECK_TEST_UTILS: Tuple[str, ...] = (
    "test_schema",
    "test_autograd_registration",
    "test_faketensor",
    "test_aot_dispatch_dynamic",
)


class QKVInputs(NamedTuple):
    '''
    Data structure for representing unpacked attention inputs, 
    query/key/values and their sequence lengths.

    Attributes:

        * {query,key,value}: unpacked (batch_size x padded_seq_len x 
                             num_heads x head_size) attention inputs
        * q_seq_lens: query sequence lengths list
        * kv_seq_lens: shared key/value sequence lengths list
    '''

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    q_seq_lens: List[int]
    kv_seq_lens: List[int]


class QKVO(NamedTuple):
    '''
    Data structure for representing unpacked attention inputs, 
    alongside unpacked known-correct attention output

    Attributes:

        * qkv: unpacked (batch_size x padded_seq_len x 
                             num_heads x head_size) attention inputs
        * ideal_output: unpacked (batch_size x padded_seq_len x 
                        num_heads x head_size) known-correct attention output
    '''

    qkv: QKVInputs
    ideal_output: torch.Tensor


class PackedQKVInputs(NamedTuple):
    '''
    Data structure for representing packed attention inputs

    Attributes:

        * {query,key,value}: packed (number_of_tokens x num_heads 
                             x head_size) attention inputs
        * q_start_loc_list: list of query start locations within packed tensor
        * kv_start_loc_list: shared list of key/value start locations within
                             packed tensor
        * q_seq_lens: query sequence lengths list
        * kv_seq_lens: shared key/value sequence lengths list
    '''

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    q_start_loc_list: Optional[List[int]]
    kv_start_loc_list: Optional[List[int]]
    q_seq_lens: Optional[List[int]]
    kv_seq_lens: Optional[List[int]]


class PackedQKVO(NamedTuple):
    '''
    Data structure for representing packed attention inputs, 
    alongside packed known-correct attention output

    Attributes:

        * packed_qkv: packed (number_of_tokens x num_heads 
                      x head_size) attention inputs
        * ideal_output: packed (number_of_tokens x num_heads 
                        x head_size) known-correct attention output
    '''

    packed_qkv: Optional[PackedQKVInputs]
    ideal_output: torch.Tensor


class KVMemoryMap(NamedTuple):
    '''
    Data structure for encapsulating KV cache memory mapping.

    Attributes:

        * block_tables: KV cache block tables
        * slot_mapping: mapping of sequence offset to physical address
    '''

    block_tables: torch.Tensor
    slot_mapping: torch.Tensor


class PhaseTestParameters(NamedTuple):
    '''
    Data structure for encapsulating the test parameters
    for a given test "phase" (prefill or decode phase) and attention
    scenario (encoder, decoder-self, encoder/decoder-cross)

    Attributes:

        * packed_qkvo: packed (number_of_tokens x num_heads 
                       x head_size) attention inputs & known-correct
                       output
        * kv_mmap: KV cache memory mapping, specific to this test phase &
                   attention scenario
    '''

    packed_qkvo: PackedQKVO
    kv_mmap: Optional[KVMemoryMap]


def maybe_make_int_tensor(
    _list: Optional[List[int]],
    device: Union[torch.device, str],
) -> torch.Tensor:
    '''
    Convert Python int list to a 1D int torch.Tensor on `device`

    Returns:

    * If _list is not None: 1D int torch.Tensor on `device`
    * None otherwise
    '''
    return None if _list is None else torch.tensor(
        _list, dtype=torch.int, device=device)


def maybe_make_long_tensor(
    _list: Optional[List[int]],
    device: Union[torch.device, str],
) -> torch.Tensor:
    '''
    Convert Python int list to a 1D long torch.Tensor on `device`

    Returns:

    * If _list is not None: 1D long torch.Tensor on `device`
    * None otherwise
    '''
    return None if _list is None else torch.tensor(
        _list, dtype=torch.long, device=device)


def maybe_max(_list: Optional[List]) -> Optional[Number]:
    '''
    Returns:

    * If _list is not None: max(_list)
    * None otherwise
    '''
    return None if _list is None else max(_list)


def make_causal_mask(
    q_max_seq_len: int,
    kv_max_seq_len: int,
) -> torch.Tensor:
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


def override_backend_env_variable(mpatch: pytest.MonkeyPatch,
                                  backend_name: str) -> None:
    '''
    Override the environment variable indicating the vLLM backend temporarily,
    using pytest monkeypatch to ensure that the env vars get
    reset once the test context exits.

    Arguments:

    * mpatch: pytest monkeypatch instance
    * backend_name: attention backend name to force
    '''
    mpatch.setenv(STR_BACKEND_ENV_VAR, backend_name)


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
    * custom_mask: custom attention mask; good place to inject a causal
      attention mask
    * q_seq_lens: list of unpadded query seq_lens for each batch index
    * kv_seq_lens: list of unpadded key/value seq_lens for each batch index

    Returns:

    * Attention result, batch_size x q_padded_seq_len x num_heads x head_size
    '''

    assert q_seq_lens is not None
    assert kv_seq_lens is not None

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


def make_qkv(
    batch_size: int,
    max_q_seq_len: int,
    max_kv_seq_len: Optional[int],
    num_heads: int,
    head_size: int,
    device: Union[torch.device, str],
    force_kv_seq_lens: Optional[List[int]] = None,
    attn_type: AttentionType = AttentionType.ENCODER_DECODER,
    force_max_len: bool = False,
) -> Tuple[QKVInputs, QKVInputs, QKVInputs]:
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
    * force_kv_seq_lens: if not None, overrides kv sequence lengths
    * attn_type: encoder, decoder self, or enc/dec cross attention
    * force_max_len: if True, all query seqlens are max_q_seq_len; o/w query
      seqlens are random in [2,max_q_seq_lens]. Same for key/value seqlens
      and max_kv_seq_len, unless forced by is_encoder_decoder_attn=False
    * device: CPU or CUDA device

    Returns:

    * Overall QKVInputs structure (containing full unpacked Q/K/V tensors)
    * Prefill QKVInputs structure (containing all but the last sequence offset)
    * Decode QKVInputs structure (containing all only the last sequence offset)
    '''

    if force_max_len:
        q_seq_lens = [max_q_seq_len for _ in range(batch_size)]
    else:
        q_seq_lens = [
            random.randint(2, max_q_seq_len) for _ in range(batch_size)
        ]
    kv_seq_lens = None
    if force_kv_seq_lens is not None:
        kv_seq_lens = force_kv_seq_lens
    elif attn_type != AttentionType.ENCODER_DECODER:
        # K,V seq lens match Q for self-attention
        kv_seq_lens = q_seq_lens
    else:
        # K,V seq lens are distinct from Q seq lens & random
        assert max_kv_seq_len is not None
        if force_max_len:
            kv_seq_lens = [max_kv_seq_len] * batch_size
        else:
            kv_seq_lens = [
                random.randint(2, max_kv_seq_len) for _ in range(batch_size)
            ]

    query = torch.rand(
        (batch_size, max_q_seq_len, num_heads, head_size)).to(device)
    key = torch.rand(
        (batch_size, max_kv_seq_len, num_heads, head_size)).to(device)
    value = torch.rand(
        (batch_size, max_kv_seq_len, num_heads, head_size)).to(device)

    prefill_query = torch.zeros(
        (batch_size, max_q_seq_len, num_heads, head_size)).to(device)
    prefill_key = torch.zeros(
        (batch_size, max_kv_seq_len, num_heads, head_size)).to(device)
    prefill_value = torch.zeros(
        (batch_size, max_kv_seq_len, num_heads, head_size)).to(device)

    decode_query = torch.zeros(
        (batch_size, 1, num_heads, head_size)).to(device)
    decode_key = torch.zeros((batch_size, 1, num_heads, head_size)).to(device)
    decode_value = torch.zeros(
        (batch_size, 1, num_heads, head_size)).to(device)

    for bdx, (q_seq_len, kv_seq_len) in enumerate(zip(q_seq_lens,
                                                      kv_seq_lens)):
        query[bdx, q_seq_len:, :, :] = 0
        key[bdx, kv_seq_len:, :, :] = 0
        value[bdx, kv_seq_len:, :, :] = 0

        prefill_query[bdx,
                      0:(q_seq_len - 1), :, :] = query[bdx,
                                                       0:(q_seq_len - 1), :, :]
        prefill_key[bdx,
                    0:(kv_seq_len - 1), :, :] = key[bdx,
                                                    0:(kv_seq_len - 1), :, :]
        prefill_value[bdx, 0:(kv_seq_len -
                              1), :, :] = value[bdx, 0:(kv_seq_len - 1), :, :]

        decode_query[bdx, :, :, :] = query[bdx,
                                           (q_seq_len - 1):q_seq_len, :, :]
        decode_key[bdx, :, :, :] = key[bdx, (kv_seq_len - 1):kv_seq_len, :, :]
        decode_value[bdx, :, :, :] = value[bdx,
                                           (kv_seq_len - 1):kv_seq_len, :, :]

    prefill_q_seq_lens = [plen - 1 for plen in q_seq_lens]
    prefill_kv_seq_lens = [plen - 1 for plen in kv_seq_lens]

    decode_q_seq_lens = [1 for _ in q_seq_lens]
    decode_kv_seq_lens = [1 for _ in kv_seq_lens]

    return (
        QKVInputs(
            query,  # Overall QKV inputs
            key,
            value,
            q_seq_lens,
            kv_seq_lens),
        QKVInputs(
            prefill_query,  # Prefill subset of QKV sequences
            prefill_key,
            prefill_value,
            prefill_q_seq_lens,
            prefill_kv_seq_lens),
        QKVInputs(
            decode_query,  # Decode subset of KV sequences
            decode_key,
            decode_value,
            decode_q_seq_lens,
            decode_kv_seq_lens))


def pack_tensor(
        unpacked_tensor: torch.Tensor, seq_lens: List[int],
        device: Union[torch.device, str]) -> Tuple[torch.Tensor, List[int]]:
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


def pack_qkv(qkv: QKVInputs, device: Union[torch.device,
                                           str]) -> PackedQKVInputs:
    '''
    Individually pack each of Q, K and V, each with dimensions batch_size x
    padded_seq_len x num_heads x head_size, into respective number_of_tokens x
    num_heads x head_size tensors.
    
    For Q, number_of_tokens = sum(q_seq_lens).

    For K and V, number_of_tokens = sum(kv_seq_lens)

    Arguments:

    * qkv: Unpacked (batch_size x padded_seq_len x num_heads x head_size)
           attention inputs
    * device: CPU or CUDA device

    Returns

    * Packed (number_of_tokens x num_heads x head_size) QKV inputs
      derived from unpacked inputs
    '''

    if qkv.query is None:
        packed_query = None
        q_start_loc_list = None
    else:
        packed_query, q_start_loc_list = pack_tensor(qkv.query,
                                                     qkv.q_seq_lens,
                                                     device=device)
    packed_key, kv_start_loc_list = pack_tensor(qkv.key,
                                                qkv.kv_seq_lens,
                                                device=device)
    packed_value, _ = pack_tensor(qkv.value, qkv.kv_seq_lens, device=device)
    return PackedQKVInputs(
        packed_query, packed_key, packed_value, q_start_loc_list,
        kv_start_loc_list,
        (None if q_start_loc_list is None else qkv.q_seq_lens),
        qkv.kv_seq_lens)


def make_backend(backend_name: str) -> AttentionBackend:
    '''
    Construct the backend instance determined by the backend_name string
    argument.

    "XFORMERS" -> construct xformers backend

    TODO: other backends

    Note: at time of writing the Attention wrapper automatically selects
    its own backend for Attention.forward(); so the backend instance which
    you generate with this function is not meant to be used for *running*
    inference, but rather for generating compatible metadata structures
    using backend.make_metadata()


    Returns:

    * Backend instance
    '''
    if backend_name == STR_XFORMERS_ATTN_VAL:
        # NOTE: xFormers backend cannot be imported for CPU and AMD GPUs.
        from vllm.attention.backends.xformers import XFormersBackend
        return XFormersBackend()
    elif backend_name == STR_FLASH_ATTN_VAL:
        from vllm.attention.backends.flash_attn import FlashAttentionBackend
        return FlashAttentionBackend()

    raise AssertionError(
        f"Unrecognized backend_name {backend_name} for unit test")


def _make_metadata_tensors(
    seq_lens: Optional[List[int]],
    context_lens: Optional[List[int]],
    encoder_seq_lens: Optional[List[int]],
    device: Union[torch.device, str],
) -> Tuple[torch.Tensor, torch.Tensor, Any, Any, Optional[torch.Tensor],
           torch.Tensor, torch.Tensor, Optional[int]]:
    '''
    Build scalar & tensor values required to build attention metadata structure.

    Arguments:

    * seq_lens: list of token-counts for each decoder input seq
    * context_lens: list of context length values for each seq
    * encoder_seq_lens: list of token-counts for each encoder input seq
    * device: CPU or CUDA device

    Returns:

    * seq_lens_tensor: decoder seq_lens list, as tensor
    * context_lens_tensor: context_lens list, as tensor
    * max_context_len: max(context_lens)
    * max_seq_len: max(seq_lens)
    * seq_start_loc: start idx of each sequence
    * encoder_seq_lens_tensor: encoder seq_lens list, as tensor
    * encoder_seq_start_loc: start idx of each encoder sequence
    * max_encoder_seq_len: encoder seq_lens list, as tensor
    '''
    seq_lens_tensor = maybe_make_int_tensor(seq_lens, device)
    context_lens_tensor = maybe_make_int_tensor(context_lens, device)
    max_context_len = maybe_max(context_lens)
    max_seq_len = maybe_max(seq_lens)

    encoder_seq_lens_tensor = maybe_make_int_tensor(encoder_seq_lens, device)
    max_encoder_seq_len = (None if encoder_seq_lens is None else
                           max(encoder_seq_lens))

    seq_start_loc = None

    if seq_lens_tensor is not None:
        seq_start_loc = torch.zeros(seq_lens_tensor.shape[0] + 1,
                                    dtype=torch.int32,
                                    device=seq_lens_tensor.device)
        torch.cumsum(seq_lens_tensor,
                     dim=0,
                     dtype=seq_start_loc.dtype,
                     out=seq_start_loc[1:])

    encoder_seq_start_loc = torch.zeros(encoder_seq_lens_tensor.shape[0] + 1,
                                        dtype=torch.int32,
                                        device=encoder_seq_lens_tensor.device)
    torch.cumsum(encoder_seq_lens_tensor,
                 dim=0,
                 dtype=encoder_seq_start_loc.dtype,
                 out=encoder_seq_start_loc[1:])

    return (seq_lens_tensor, context_lens_tensor, max_context_len, max_seq_len,
            seq_start_loc, encoder_seq_lens_tensor, encoder_seq_start_loc,
            max_encoder_seq_len)


def make_kv_cache(num_blocks: int,
                  num_heads: int,
                  head_size: int,
                  block_size: int,
                  device: Union[torch.device, str],
                  backend: str,
                  default_val: float = 0.0) -> torch.Tensor:
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
    *     for backend 'XFORMERS' 
    * kv_cache: 2 x num_blocks x block_size x num_heads x head_size
    *     for backend 'FLASH_ATTN'  
    '''
    if backend == 'XFORMERS':
        kv_cache = torch.rand(
            (2, num_blocks, block_size * num_heads * head_size)).to(device)
    elif backend == 'FLASH_ATTN':
        kv_cache = torch.rand(
            (2, num_blocks, block_size, num_heads, head_size)).to(device)
    else:
        raise ValueError(
            f"Unknown backend value: '{backend}'. Expected 'XFORMERS' or "
            f"'FLASH_ATTN'.")
    if default_val is not None:
        kv_cache[:, :, :] = default_val
    return kv_cache


def _num_tokens_to_min_blocks(num_tokens: int, block_size: int) -> int:
    '''
    Compute the minimum number of blocks required to hold num_tokens tokens,
    given block_size
    '''
    return (num_tokens + block_size) // block_size


def make_empty_slot_mapping_tensor(device: Union[torch.device, str]):
    return maybe_make_long_tensor([], device)


def make_empty_block_tables_tensor(device: Union[torch.device, str]):
    return torch.tensor([], device=device)


def split_slot_mapping(slot_mapping_list: torch.Tensor, seq_lens: List[int],
                       device: Union[torch.device, str]):
    '''
    Split a slot mapping into valid prefill- and decode-phase slot mappings.

    Context:
    * Your goal is to test (1) prefill of N prompts, with prompt-lengths
      {K_i \\forall i \\in [0,N)}, followed by (2) decoding of a single token
      for all N prompts (N tokens total); the resultant sequence lengths 
      after decode would be {K_i + 1 for i \\in [0,N)}
    * The test you want to do requires (1) having the prefill slot mapping 
      for all tokens present during prefill, the number of which is 
      M = \\sum_i{K_i}, and (2) having the decode slot mapping for all N 
      decoded tokens
    
    This function consumes a single 1D slot mapping, which is the 
    concatenation of N slot mappings each of length K_i + 1 (corresponding
    to the  sequence lengths after decode), with a total length of
    P = \\sum_i{K_i + 1} = M + N

    The prefill-phase slot mapping results from excising the (K_i + 1)-th entry
    from each of the N subsequences in the slot mapping (i.e. omitting the 
    decoded token's mapping.)

    The N excised entries are appended to obtain the decode-phase slot mapping

    Arguments:

    * slot_mapping_list: Length-P 1D slot mapping (as List) reflecting all N
      post-decode sequences
    * seq_lens: List of N post-decode sequence lengths (K_i + 1 in the 
      description above)
    * device: cuda, cpu, etc.

    Returns:

    * prefill_slot_mapping: Length-M 1D slot mapping (as Tensor) 
      reflecting all N prefill prompts
    * decode_slot_mapping: Length-N 1D slot mapping (as Tensor) reflecting 
      all N decoded tokens
    '''

    prefill_slot_mapping = []
    decode_slot_mapping = []

    base_idx = 0
    for seq_len in seq_lens:
        prefill_slot_mapping.extend(slot_mapping_list[base_idx:(base_idx +
                                                                seq_len - 1)])
        decode_slot_mapping.append(slot_mapping_list[base_idx + seq_len - 1])
        base_idx += seq_len

    return (maybe_make_long_tensor(prefill_slot_mapping, device),
            maybe_make_long_tensor(decode_slot_mapping, device))


def make_block_tables_slot_mapping(
        block_size: int,
        seq_lens: List[int],
        device: Union[torch.device, str],
        block_base_addr: int = 0) -> Tuple[torch.Tensor, List[int], int]:
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
    

    The constructed block-tables and slot-mapping are sized to the
    lengths of the sequences in their entirety (as reflected by seq_lens),
    i.e. the total of prefill prompt tokens + decoded tokens.

    Arguments:

    * block_size: number of offsets per block
    * seq_lens: list of token-counts for each sequence
    * block_base_addr: the block table base address
    * device: CPU or CUDA device

    Return:

    * block_tables_tensor: block table for sequence   
    * slot_mapping_list: slot mapping for sequence
    * max_block_idx: the highest block address within this block table
    '''

    # Provision minimum number of KV cache blocks
    num_blocks_list = [
        _num_tokens_to_min_blocks(num_tokens, block_size)
        for num_tokens in seq_lens
    ]
    max_block_table_len = max(num_blocks_list)
    block_table_pad_tokens = 10

    block_tables = []
    slot_mapping_list = []
    # Compute uppermost address of block table
    total_cache_blocks = sum(num_blocks_list)
    block_base_idx = block_base_addr + total_cache_blocks
    max_block_idx = block_base_idx
    for sdx, num_tokens in enumerate(seq_lens):
        num_blocks = num_blocks_list[sdx]
        block_table = list(
            range(block_base_idx, block_base_idx - num_blocks, -1))
        for idx in range(num_tokens):
            mapping_value = (
                idx % block_size) + block_table[idx // block_size] * block_size
            slot_mapping_list.append(mapping_value)

        block_base_idx -= num_blocks
        block_tables.append(block_table)

    block_tables_tensor = make_tensor_with_pad(
        block_tables,
        max_len=max_block_table_len + block_table_pad_tokens,
        pad=0,
        dtype=torch.int,
        device=device,
    )

    return (block_tables_tensor, slot_mapping_list, max_block_idx)


def make_test_metadata(
    attn_backend: AttentionBackend,
    is_prompt: bool,
    seq_lens: Optional[List[int]],
    decoder_test_params: Optional[PhaseTestParameters],
    device: Union[torch.device, str],
    encoder_test_params: Optional[PhaseTestParameters] = None,
    cross_test_params: Optional[PhaseTestParameters] = None
) -> AttentionMetadata:
    '''
    Construct fake attention metadata for a given test phase
    (prefill-phase or decode-phase).

    encoder_test_params and cross_test_params arguments allow encoder
    attention and enc/dec cross-attention (respectively) to use distinct
    metadata values from decoder self-attention (decoder_test_params.)
    
    if encoder_test_params and cross_test_params are None, the attention
    metadata will support decoder-only scenario.

    Assumptions:

    * No chunked prefill -> a batch is 100% prefill or 100% decode, never both

    Arguments:

    * attn_backend: Backend for sourcing attention kernels
    * is_prompt: prefill if True, o/w decode
    * seq_lens: list of token counts for each sequence
    * decoder_test_params: decoder self-attention test params; 
                           this function requires
                           kv_mmap (memory mapping) field
    * device: CPU or CUDA device
    * encoder_test_params: encoder attention test params;
                           this function requires encoder query
                           sequence lengths field. If None,
                           encoder query sequence lengths are
                           treated as None
    * cross_test_params: enc/dec cross-attention test params;
                         this function requires kv_mmap field.
                         If None, KV cache memory map data
                         structures are treated as None

    Return:

    * AttentionMetadata structure
    '''

    # Decoder self-attention memory mapping
    # decoder_test_params is None signals encoder-only
    # scenario, so kv_mmap is None
    kv_mmap = (None
               if decoder_test_params is None else decoder_test_params.kv_mmap)

    # This function constructs metadata assuming no chunked prefill,
    # i.e. 100% prefill tokens or 100% decode tokens
    #
    # - If is_prompt, num_prefills_or_decodes is the number of prefills
    #   and num_prefill_or_decode_tokens is the number of prefill tokens
    # - If not is_prompt, num_prefills_or_decodes is the number of decodes
    #   and num_prefill_or_decode_tokens is the number of decode tokens
    #
    # seq_lens is None signals encoder-only
    # scenario, in which case num_prefills_or_decodes and
    # num_prefill_or_decode_tokens are unused
    num_prefills_or_decodes = (None if seq_lens is None else len(seq_lens))

    num_prefill_or_decode_tokens = (None if seq_lens is None else (
        sum(seq_lens) if is_prompt else len(seq_lens)))

    # Seems for non-prefix-caching scenarios context_lens
    # is never needed
    context_lens = None

    if encoder_test_params is None:
        encoder_seq_lens = None
        num_encoder_tokens = None
    else:
        # Encoder/decoder or encoder-only models only:
        # * Extract encoder input sequence lengths
        assert encoder_test_params.packed_qkvo.packed_qkv is not None
        encoder_seq_lens = encoder_test_params.packed_qkvo.packed_qkv.q_seq_lens
        num_encoder_tokens = (None if encoder_seq_lens is None else
                              (sum(encoder_seq_lens)))

    if cross_test_params is None:
        cross_kv_mmap = None
    else:
        # Encoder/decoder or encoder-only models only:
        # * Extract *cross-attention* slot_mapping and block table
        #   (kv_mmap)
        cross_kv_mmap = cross_test_params.kv_mmap

    if is_prompt:
        # Prefill-phase scenario

        num_prefills = num_prefills_or_decodes
        num_prefill_tokens = num_prefill_or_decode_tokens
        num_decode_tokens = 0

        (
            seq_lens_tensor,
            context_lens_tensor,
            _,
            _,
            seq_start_loc,
            encoder_seq_lens_tensor,
            encoder_seq_start_loc,
            max_encoder_seq_len,
        ) = _make_metadata_tensors(seq_lens,
                                   context_lens,
                                   encoder_seq_lens,
                                   device=device)

        return attn_backend.make_metadata(
            num_prefills=num_prefills,
            slot_mapping=(None if kv_mmap is None else kv_mmap.slot_mapping),
            multi_modal_placeholder_index_maps=None,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            seq_start_loc=seq_start_loc,
            max_prefill_seq_len=None if seq_lens is None else max(seq_lens),
            max_decode_seq_len=0,
            context_lens_tensor=context_lens_tensor,
            block_tables=(None if kv_mmap is None else kv_mmap.block_tables),
            use_cuda_graph=False,
            num_encoder_tokens=num_encoder_tokens,
            encoder_seq_lens=encoder_seq_lens,
            encoder_seq_lens_tensor=encoder_seq_lens_tensor,
            encoder_seq_start_loc=encoder_seq_start_loc,
            max_encoder_seq_len=max_encoder_seq_len,
            cross_slot_mapping=(None if cross_kv_mmap is None else
                                cross_kv_mmap.slot_mapping),
            cross_block_tables=(None if cross_kv_mmap is None else
                                cross_kv_mmap.block_tables))

    else:  # not is_prompt
        # Decode-phase scenario

        assert kv_mmap is not None
        assert num_prefill_or_decode_tokens is not None
        assert seq_lens is not None

        num_prefills = 0
        num_prefill_tokens = 0
        num_decode_tokens = num_prefill_or_decode_tokens

        (
            seq_lens_tensor,
            context_lens_tensor,
            _,
            _,
            seq_start_loc,
            encoder_seq_lens_tensor,
            encoder_seq_start_loc,
            max_encoder_seq_len,
        ) = _make_metadata_tensors(seq_lens,
                                   context_lens,
                                   encoder_seq_lens,
                                   device=device)

        return attn_backend.make_metadata(
            num_prefills=num_prefills,
            slot_mapping=kv_mmap.slot_mapping,
            multi_modal_placeholder_index_maps=None,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            seq_start_loc=seq_start_loc,
            max_prefill_seq_len=0,
            max_decode_seq_len=max(seq_lens),
            max_decode_query_len=1,
            context_lens_tensor=context_lens_tensor,
            block_tables=kv_mmap.block_tables,
            use_cuda_graph=False,
            num_encoder_tokens=num_encoder_tokens,
            encoder_seq_lens=encoder_seq_lens,
            encoder_seq_lens_tensor=encoder_seq_lens_tensor,
            encoder_seq_start_loc=encoder_seq_start_loc,
            max_encoder_seq_len=max_encoder_seq_len,
            cross_slot_mapping=(None if cross_kv_mmap is None else
                                cross_kv_mmap.slot_mapping),
            cross_block_tables=(None if cross_kv_mmap is None else
                                cross_kv_mmap.block_tables))


def assert_actual_matches_ideal(test_params: PhaseTestParameters,
                                output_under_test: torch.Tensor,
                                backend: str) -> None:
    '''
    Assert that observed output matches the ideal output
    contained in the test parameters data structure.

    Arguments:

    * test_params: Test parameters including packed ideal output
    * output_under_test: actually observed output value
    '''
    ideal_output = test_params.packed_qkvo.ideal_output
    if backend == 'XFORMERS':
        torch.testing.assert_close(ideal_output,
                                   output_under_test.view_as(ideal_output))

    elif backend == 'FLASH_ATTN':
        # For FlashAttention override the accuracy thresholds to non default
        # values since we notice a higher difference between the ideal and
        # actual output.
        torch.testing.assert_close(ideal_output,
                                   output_under_test.view_as(ideal_output),
                                   atol=0.01,
                                   rtol=0.016)
    else:
        raise ValueError(
            f"Unknown backend value: '{backend}'. Expected 'XFORMERS' or "
            f"'FLASH_ATTN'.")


# Copied/modified from torch._refs.__init__.py
def fp8_allclose(
    a: TensorLikeType,
    b: TensorLikeType,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool:
    """
    Reference implementation of torch.allclose
    """
    torch._refs._check_close_args(name="torch.allclose",
                                  a=a,
                                  b=b,
                                  rtol=rtol,
                                  atol=atol)

    return bool(
        torch.all(
            torch.isclose(a.double(),
                          b.double(),
                          rtol=rtol,
                          atol=atol,
                          equal_nan=equal_nan)).item())


# Marlin MoE test utils


def stack_and_dev(tensors: List[torch.Tensor]):
    dev = tensors[0].device
    return torch.stack(tensors, dim=0).to(dev)


def compute_max_diff(output, output_ref):
    return torch.mean(torch.abs(output - output_ref)) / torch.mean(
        torch.abs(output_ref))


def torch_moe(a, w1, w2, score, topk):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = SiluAndMul()(
                a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(0, 1)
    return (out.view(B, -1, w2.shape[1]) *
            topk_weight.view(B, -1, 1).to(out.dtype)).sum(dim=1)


def torch_moe_single(a, w, score, topk):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    _, topk_ids = torch.topk(score, topk)
    topk_ids = topk_ids.view(-1)
    for i in range(w.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = a[mask] @ w[i].transpose(0, 1)
    return (out.view(B, -1, w.shape[1])).sum(dim=1)


# A special version of op check that has a restricted default set of test_utils
# and a patched version of allclose that supports fp8 types.
def opcheck(op: Union[torch._ops.OpOverload, torch._ops.OpOverloadPacket,
                      torch._library.custom_ops.CustomOpDef],
            args: Tuple[Any, ...],
            kwargs: Optional[Dict[str, Any]] = None,
            *,
            test_utils: Union[str, Sequence[str]] = ALL_OPCHECK_TEST_UTILS,
            raise_exception: bool = True,
            cond: bool = True) -> Dict[str, str]:
    with unittest.mock.patch('torch.allclose', new=fp8_allclose):
        return torch.library.opcheck(
            op,
            args,
            kwargs,
            test_utils=test_utils,
            raise_exception=raise_exception) if cond else {}
