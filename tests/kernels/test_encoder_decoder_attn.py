"""
Test

* Encoder attention
* Decoder self-attention
* Encoder/decoder cross-attention
"""

import copy
from typing import Optional

import pytest
import torch

from tests.kernels.utils import *
from vllm.attention import Attention, AttentionMetadata
from vllm.attention.backends.abstract import AttentionType
from vllm.attention.backends.utils import (
    STR_NOT_IMPL_ENC_DEC_CHUNKED_PREFILL, STR_NOT_IMPL_ENC_DEC_ROCM_HIP)
from vllm.utils import is_hip, make_causal_mask, maybe_make_long_tensor

HEAD_SIZES = [64, 256]

NUM_HEADS = [1, 16]

BATCH_SIZES = [1, 16]
BLOCK_SIZES = [16]
BACKEND_NAMES = ["XFORMERS"]
CUDA_DEVICE = "cuda:0"

MAX_DEC_SEQ_LENS = [128]
MAX_ENC_SEQ_LENS = [128]


def _basic_setup(num_heads: int, head_size: int, num_blocks: int,
                 block_size: int, backend_name: str) -> tuple:
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
    kv_cache = make_kv_cache(num_blocks,
                             num_heads,
                             head_size,
                             block_size,
                             device=CUDA_DEVICE)
    return scale, attn_backend, attn, kv_cache


def _encoder_attn_setup(batch_size: int, num_heads: int, head_size: int,
                        scale: float, max_q_seq_len: int) \
                          -> PhaseTestParameters:
    '''
    Set up test vectors & data structures for encoder attention test.

    A triplet of synthetic query/key/value tensors are constructed. 
    Given this is an encoder attention test, the key & value
    sequences will have the same length as the corresponding queries.

    The query/key/value tensors are passed to an ideal reference
    self-attention implementation to generate an ideal output tensor.

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
    
    * packed_query: number_of_tokens x num_heads x head_size
    * packed_key: number_of_tokens x num_heads x head_size
    * packed_value: number_of_tokens x num_heads x head_size
    * packed_ideal_output: number_of_tokens x num_heads x head_size
    * block_tables: fake self-attn decode-phase block table
    * slot_mapping: fake self-attn decode-phase slot mapping
    * q_seq_lens: list of query sequence lengths
    '''

    max_kv_seq_len = max_q_seq_len

    qkv_in, _, _ = make_qkv(batch_size,
                            max_q_seq_len,
                            max_kv_seq_len,
                            num_heads,
                            head_size,
                            attn_type=AttentionType.ENCODER,
                            device=CUDA_DEVICE)

    # No causal attention mask
    ideal_output = ref_masked_attention(qkv_in.query,
                                        qkv_in.key,
                                        qkv_in.value,
                                        scale=scale,
                                        q_seq_lens=qkv_in.q_seq_lens,
                                        kv_seq_lens=qkv_in.kv_seq_lens)

    packed_ideal_output, _ = pack_tensor(ideal_output,
                                         qkv_in.q_seq_lens,
                                         device=CUDA_DEVICE)

    packed_qkv = pack_qkv(qkv_in, device=CUDA_DEVICE)

    return PhaseTestParameters(
             PackedQKVO(
               packed_qkv, \
               packed_ideal_output),

             None # No KV cache
           )


def _decoder_attn_setup(batch_size: int,
                        num_heads: int,
                        head_size: int,
                        block_size: int,
                        scale: float,
                        max_q_seq_len: int,
                        block_base_addr: int = 0) -> tuple[QKVInputs,
                                                           PhaseTestParameters,
                                                           PhaseTestParameters,
                                                           int]:
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

    qkv, \
    prefill_qkv, \
    decode_qkv = make_qkv(batch_size,
                          max_q_seq_len,
                          max_kv_seq_len,
                          num_heads,
                          head_size,
                          attn_type=AttentionType.DECODER,
                          device=CUDA_DEVICE)

    causal_mask = make_causal_mask(max_q_seq_len,
                                   max_kv_seq_len).to(CUDA_DEVICE)

    ideal_output = ref_masked_attention(qkv.query,
                                        qkv.key,
                                        qkv.value,
                                        scale=scale,
                                        custom_mask=causal_mask,
                                        q_seq_lens=qkv.q_seq_lens,
                                        kv_seq_lens=qkv.kv_seq_lens)

    prefill_ideal_output = torch.zeros_like(ideal_output)
    decode_ideal_output = torch.zeros_like(ideal_output[:, 0:1])
    for bdx, prefill_q_seq_len in enumerate(prefill_qkv.q_seq_lens):
        prefill_ideal_output[bdx, :prefill_q_seq_len] = ideal_output[
            bdx, :prefill_q_seq_len]
        decode_ideal_output[bdx, :] = ideal_output[bdx, prefill_q_seq_len:(
            prefill_q_seq_len + 1)]

    prefill_packed_ideal_output, _ = pack_tensor(prefill_ideal_output,
                                                 prefill_qkv.q_seq_lens,
                                                 device=CUDA_DEVICE)
    decode_packed_ideal_output, _ = pack_tensor(decode_ideal_output,
                                                [1 for _ in range(batch_size)],
                                                device=CUDA_DEVICE)

    # Build prefill- & decode-phase data structures
    # for decoder self-attention. Block tables and
    # slot mapping must be in a format compatible
    # with KV caching & attention kernels
    #
    # Prefill-phase:
    #
    # * Empty block-tables tensor
    # * Slot-mapping with entries for prompt tokens
    #
    # Decode-phase:
    # * Block-tables tensor with minimum number of blocks
    #   required by total num. tokens in the entirety of all sequences
    #   (including both prefill & decode)
    # * Slot-mapping with entries for tokens that will be decoded in the
    #   current decode iteration

    prefill_block_tables = make_empty_block_tables_tensor(device=CUDA_DEVICE)

    decode_block_tables, \
    slot_mapping_list, \
    max_block_idx = make_block_tables_slot_mapping(block_size,
                            qkv.q_seq_lens,
                            device=CUDA_DEVICE,
                            block_base_addr = block_base_addr)

    prefill_slot_mapping, \
    decode_slot_mapping = split_slot_mapping(slot_mapping_list,
                                             qkv.q_seq_lens,
                                             device=CUDA_DEVICE)

    prefill_pckd_qkv = pack_qkv(prefill_qkv, device=CUDA_DEVICE)

    decode_pckd_qkv = pack_qkv(decode_qkv, device=CUDA_DEVICE)

    return qkv, \
           PhaseTestParameters( # Prefill test params
              PackedQKVO(
                  prefill_pckd_qkv, \
                  prefill_packed_ideal_output), \
              KVMemoryMap(
                  prefill_block_tables, \
                  prefill_slot_mapping)), \
           PhaseTestParameters( # Decode test params
              PackedQKVO(
                decode_pckd_qkv, \
                decode_packed_ideal_output), \
              KVMemoryMap(
                decode_block_tables, \
                decode_slot_mapping)), \
           max_block_idx

def _enc_dec_cross_attn_setup_reuses_query(decoder_qkv: QKVInputs,
                                           encoder_test_params:
                                            PhaseTestParameters,
                                           prefill_phase_test_params:
                                            PhaseTestParameters,
                                           batch_size: int,
                                           num_heads: int,
                                           head_size: int,
                                           block_size: int,
                                           scale: float,
                                           max_decoder_seq_len: int,
                                           max_encoder_seq_len: int,
                                           block_base_addr: Optional[int]=0) \
                                            -> tuple:
    '''
    Set up test vectors & data structures for cross-attention test.

    A triplet of synthetic cross-attention key/value tensors are constructed
    ("baseline" key/value). Given this is a cross-attention test, we assume
    query tensors were already synthesized for a prior self-attention test and
    will be reused for cross-attention. The key & value sequences generated here
    may have a different length than the corresponding queries (as is often
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

    decoder_query = decoder_qkv.query
    decoder_seq_lens = decoder_qkv.q_seq_lens
    encoder_seq_lens = encoder_test_params.packed_qkvo.packed_qkv.q_seq_lens
    prefill_q_seq_lens = prefill_phase_test_params.packed_qkvo.packed_qkv.q_seq_lens


    cross_kv, \
    _, \
    _ = make_qkv(batch_size,
                 max_decoder_seq_len,
                 max_encoder_seq_len,
                 num_heads,
                 head_size,
                 force_kv_seq_lens=encoder_seq_lens,
                 attn_type=AttentionType.ENCODER_DECODER,
                 device=CUDA_DEVICE)

    ideal_output = ref_masked_attention(decoder_query,
                                        cross_kv.key,
                                        cross_kv.value,
                                        scale=scale,
                                        q_seq_lens=decoder_seq_lens,
                                        kv_seq_lens=cross_kv.kv_seq_lens)

    prefill_ideal_output = torch.zeros_like(ideal_output)
    decode_ideal_output = torch.zeros_like(ideal_output[:, 0:1])
    for bdx, prefill_q_seq_len in enumerate(prefill_q_seq_lens):
        prefill_ideal_output[bdx, :prefill_q_seq_len] = ideal_output[
            bdx, :prefill_q_seq_len]
        decode_ideal_output[bdx, :] = ideal_output[bdx, prefill_q_seq_len:(
            prefill_q_seq_len + 1)]

    prefill_packed_ideal_output, _ = pack_tensor(prefill_ideal_output,
                                                 prefill_q_seq_lens,
                                                 device=CUDA_DEVICE)
    decode_packed_ideal_output, _ = pack_tensor(decode_ideal_output,
                                                [1 for _ in range(batch_size)],
                                                device=CUDA_DEVICE)

    # Build prefill- & decode-phase data structures
    # for encoder/decoder cross-attention. Block tables and
    # slot mapping must be in a format compatible
    # with KV caching & attention kernels
    #
    # Whereas decoder self-attention extracts relationships between
    # equal-length Q/K/V sequences, which mutually grow in length
    # with each decoded token, cross-attention relates the Q sequence
    # - which grows with each new decoded token - to fixed-length
    # K and V sequences derived from the encoder hidden states.
    #
    # Prefill-phase:
    #
    # * Empty block-tables tensor
    # * Slot-mapping with as many entries as there are tokens in the encoder
    #   prompt.
    #
    # Decode-phase:
    # * Block-tables tensor with minimum number of blocks to
    #   accommodate K & V tensors which are equal in lnegth
    #   to the encoder prompt length
    # * Empty slot-mapping tensor (since K & V are fixed in size,
    #   new decoded tokens are not KV-cached and require no slot-
    #   mapping)

    prefill_block_tables = make_empty_block_tables_tensor(device=CUDA_DEVICE)
    decode_slot_mapping = make_empty_slot_mapping_tensor(device=CUDA_DEVICE)

    decode_block_tables, \
    prefill_slot_mapping_list, \
    _ = make_block_tables_slot_mapping(
        block_size,
        cross_kv.kv_seq_lens,
        block_base_addr=block_base_addr,
        device=CUDA_DEVICE)

    prefill_slot_mapping = maybe_make_long_tensor(prefill_slot_mapping_list,
                                                  device=CUDA_DEVICE)

    # Packed key/value (query is already provided)
    packed_cross_kv = pack_qkv(cross_kv, device=CUDA_DEVICE)

    return PhaseTestParameters( # Prefill-phase test params
            PackedQKVO(
              packed_cross_kv, \
              prefill_packed_ideal_output), \
            KVMemoryMap(
              prefill_block_tables, \
              prefill_slot_mapping)), \
           PhaseTestParameters( # Decode-phase test params
            PackedQKVO(
              None,
              decode_packed_ideal_output), \
            KVMemoryMap(
              decode_block_tables, \
              decode_slot_mapping))

def _run_encoder_attention_test(attn: Attention, pckd_qkv: PackedQKVInputs,
                                attn_metadata: AttentionMetadata,
                                attn_type: AttentionType) -> torch.Tensor:
    '''
    Run encoder attention.

    attn_metadata.attention_type is assigned attn_type in order to configure
    the kernel invocation for either encoder attention

    attn_type must be AttentionType.ENCODER

    Arguments:

    * attn: Attention wrapper instance
    * pckd_qkv: Packed query/key/value inputs
    * attn_metadata: attention metadata for encoder/decoder-self attention
    * attn_type: AttentionType.DECODER or AttentionType.ENCODER

    Returns:
    * Attention.forward() applied to packed_{query,key,value}, kv_cache
      & attn_metadata
    '''
    assert attn_type == AttentionType.ENCODER
    assert attn_metadata.num_decode_tokens == 0
    attn_metadata.attention_type = attn_type
    return attn.forward(pckd_qkv.query, pckd_qkv.key, pckd_qkv.value, None,
                        attn_metadata)


def _run_decoder_self_attention_test(attn: Attention,
                                     pckd_qkv: PackedQKVInputs,
                                     kv_cache: torch.Tensor,
                                     attn_metadata: AttentionMetadata,
                                     attn_type: AttentionType) -> torch.Tensor:
    '''
    Run decoder self-attention test.

    attn_metadata.attention_type is assigned attn_type in order to configure
    the kernel invocation for decoder self-attention.

    attn_type must be AttentionType.DECODER

    Arguments:

    * attn: Attention wrapper instance
    * pckd_qkv: Packed query/key/value inputs
    * kv_cache
    * attn_metadata: attention metadata for encoder/decoder-self attention
    * attn_type: AttentionType.DECODER or AttentionType.ENCODER

    Returns:
    * Attention.forward() applied to packed_{query,key,value}, kv_cache
      & attn_metadata
    '''
    assert attn_type == AttentionType.DECODER
    attn_metadata.attention_type = attn_type
    return attn.forward(pckd_qkv.query, pckd_qkv.key, pckd_qkv.value, kv_cache,
                        attn_metadata)


def _run_encoder_decoder_cross_attention_test(
        attn: Attention, dec_pckd_qkv: PackedQKVInputs,
        cross_pckd_qkv: PackedQKVInputs, kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata) -> torch.Tensor:
    '''
    Run encoder/decoder cross-attention test.

    attn_metadata.attention_type is assigned AttentionType.ENCODER_DECODER 
    in order to configure the kernel invocation for encoder/decoder cross-
    attention.

    Arguments:

    * attn: Attention wrapper instance
    * packed_{query,key,value}: total_num_tokens x (num_heads*head_size)
    * kv_cache
    * attn_metadata: attention metadata for encoder/decoder-self attention

    Returns:
    * Attention.forward() applied to packed_{query,key,value}, kv_cache
      & attn_metadata
    '''
    attn_metadata.attention_type = AttentionType.ENCODER_DECODER
    key = None if cross_pckd_qkv is None else \
            cross_pckd_qkv.key
    value = None if cross_pckd_qkv is None else \
            cross_pckd_qkv.value
    return attn.forward(dec_pckd_qkv.query, key, value, kv_cache,
                        attn_metadata)


@pytest.mark.skipif(is_hip(), reason=STR_NOT_IMPL_ENC_DEC_ROCM_HIP)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("backend_name", BACKEND_NAMES)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("max_dec_seq_len", MAX_DEC_SEQ_LENS)
@pytest.mark.parametrize("max_enc_seq_len", MAX_ENC_SEQ_LENS)
def test_enc_dec_self_and_cross_attention_prefill_decode_phases(
        num_heads: int, head_size: int, backend_name: str, batch_size: int,
        block_size: int, max_dec_seq_len: int, max_enc_seq_len: int,
        monkeypatch) -> None:
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

    # Force Attention wrapper backend
    override_backend_env_variable(monkeypatch, backend_name)

    # Num KV cache blocks
    num_blocks = 4096

    # Attention scale factor, attention backend instance, attention wrapper
    # instance, KV cache init
    scale, \
    attn_backend, \
    attn, \
    kv_cache = _basic_setup(num_heads,
                          head_size,
                          num_blocks,
                          block_size,
                          backend_name)

    # Encoder attention setup

    # Let encoder_attn_setup() choose default block table
    # base address; the block_tables and slot_mapping
    # tensors are not actually utilized by encoder attention
    # anyway but are required to be present & valid by the
    # backend.

    enc_test_params = _encoder_attn_setup(batch_size,
                                          num_heads,
                                          head_size,
                                          scale,
                                          max_enc_seq_len)

    # Decoder self-attention setup

    dec_qkv, \
    prephase_dec_test_params, \
    decphase_dec_test_params, \
    cross_block_base_addr = _decoder_attn_setup(batch_size,
                                                num_heads,
                                                head_size,
                                                block_size,
                                                scale,
                                                max_dec_seq_len)

    # Cross-attention setup

    prephase_cross_test_params, \
    decphase_cross_test_params, \
    = _enc_dec_cross_attn_setup_reuses_query(dec_qkv,
                                             enc_test_params,
                                             prephase_dec_test_params,
                                             batch_size,
                                             num_heads,
                                             head_size,
                                             block_size,
                                             scale,
                                             max_dec_seq_len,
                                             max_enc_seq_len,
                                             block_base_addr = \
                                              cross_block_base_addr)

    # Shared prefill metadata structure

    prephase_attn_metadata: AttentionMetadata = make_test_metadata(
        attn_backend,
        True,
        prephase_dec_pckd_qkv.q_seq_lens,
        prephase_dec_kv_mmap,
        default_attn_type=AttentionType.ENCODER,
        num_prefills_or_decodes=len(prephase_dec_pckd_qkv.q_seq_lens),
        num_prefill_or_decode_tokens=sum(prephase_dec_pckd_qkv.q_seq_lens),
        encoder_seq_lens=enc_pckd_qkvo.packed_qkv.q_seq_lens,
        cross_kv_mmap=prephase_cross_kv_mmap,
        device=CUDA_DEVICE)

    # PREFILL: encoder attention
    # * Use prefill kernel

    enc_packed_actual_output: torch.Tensor = \
      _run_encoder_attention_test(
        attn,
        enc_pckd_qkvo.packed_qkv,
        prephase_attn_metadata,
        attn_type=AttentionType.ENCODER)

    # - Is encoder attention result correct?
    assert torch.allclose(enc_pckd_qkvo.ideal_output,
                          enc_packed_actual_output
                            .view_as(enc_pckd_qkvo.ideal_output))

    # PREFILL: self-attention test

    self_prefill_packed_actual_output: torch.Tensor = \
      _run_decoder_self_attention_test(
        attn,
        prephase_dec_pckd_qkv,
        kv_cache,
        prephase_attn_metadata,
        attn_type=AttentionType.DECODER)

    # - Prefill self-attention correct?
    assert torch.allclose(
        prephase_dec_pckd_idl_out,
        self_prefill_packed_actual_output.view_as(prephase_dec_pckd_idl_out))

    # PREFILL: cross-attention test

    prephase_cross_pckd_act_out: torch.Tensor = \
      _run_encoder_decoder_cross_attention_test(
        attn,
        prephase_dec_pckd_qkv,
        prephase_cross_pckd_qkv,
        kv_cache,
        prephase_attn_metadata)

    # - Prefill cross-attention correct?
    assert torch.allclose(
        prephase_cross_pckd_idl_out,
        prephase_cross_pckd_act_out.view_as(prephase_cross_pckd_idl_out))

    # DECODE: build decode-phase attention metadata

    # - Cross-attention KV context is equal in length to
    #   encoder input
    context_lens = copy.deepcopy(enc_pckd_qkvo.packed_qkv.q_seq_lens)

    decphase_attn_metadata: AttentionMetadata = make_test_metadata(
        attn_backend,
        False,
        dec_qkv.q_seq_lens,
        decphase_dec_kv_mmap,
        default_attn_type=AttentionType.DECODER,
        context_lens=context_lens,
        num_prefills_or_decodes=len(dec_qkv.q_seq_lens),
        num_prefill_or_decode_tokens=len(dec_qkv.q_seq_lens),
        encoder_seq_lens=enc_pckd_qkvo.packed_qkv.q_seq_lens,
        cross_kv_mmap=decphase_cross_kv_mmap,
        device=CUDA_DEVICE)

    # DECODE: self-attention test

    decphase_dec_pckd_act_out: torch.Tensor = \
      _run_decoder_self_attention_test(
        attn,
        decphase_dec_pckd_qkv,
        kv_cache,
        decphase_attn_metadata,
        attn_type=AttentionType.DECODER)

    # - Decode self-attention correct?
    assert torch.allclose(
        decphase_dec_pckd_idl_out,
        decphase_dec_pckd_act_out.view_as(decphase_dec_pckd_idl_out))

    # DECODE: cross-attention test

    decphase_cross_pckd_act_out: torch.Tensor = \
      _run_encoder_decoder_cross_attention_test(
        attn,
        decphase_dec_pckd_qkv,
        None,
        kv_cache,
        decphase_attn_metadata)

    # - Decode cross-attention correct?
    assert torch.allclose(
        decphase_cross_pckd_idl_out,
        decphase_cross_pckd_act_out.view_as(decphase_cross_pckd_idl_out))

    # The following test conditions could in principle be a
    # standalone test, however the test setup is
    # so involved that it is easier
    # to piggyback off of the test vectors & other data structures
    # created for testing decode-phase encoder/decoder cross-
    # attention above.
    # ----
    # Set up a contrived scenario where the attention metadata
    # is configured for chunked prefill & encoder/decoder cross-
    # attention. Required that this triggers a NotImplementedError.
    #
    # We assume that decode_attn_metadata.num_decode_tokens > 1
    # already; the line below sets up a chunked prefill
    # metadata configuration where there is nominally a mix
    # of prefill and decode tokens.
    decphase_attn_metadata.num_prefill_tokens = 1
    with pytest.raises(NotImplementedError) as exc_info:
        _run_encoder_decoder_cross_attention_test(attn, decphase_dec_pckd_qkv,
                                                  None, kv_cache,
                                                  decphase_attn_metadata)

    # "Encoder decoder models do not currently support chunked prefill"
    assert str(exc_info.value) == STR_NOT_IMPL_ENC_DEC_CHUNKED_PREFILL
