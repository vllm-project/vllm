"""Attention utils"""

from vllm.attention import AttentionMetadata
from vllm.utils import is_hip

# Error string(s) for encoder/decoder
# unsupported attention scenarios

STR_NOT_IMPL_ENC_DEC_CHUNKED_PREFILL = \
"Chunked prefill is not currently " + \
"supported with encoder/decoder models."

STR_NOT_IMPL_ENC_DEC_ROCM_HIP = \
"ROCm/HIP is not currently supported" + \
"with encoder/decoder models."

STR_NOT_IMPL_ENC_DEC_NON_XFORMERS_BACKEND = \
"Currently only the XFormers backend " + \
    "supports encoder/decoder models."

STR_NOT_IMPL_ENC_DEC_PREFIX_CACHING = \
"Prefix caching is not currently supported " + \
"with encoder/decoder models"

# Check for unsupported encoder/decoder scenarios


def is_encoder_decoder_metadata(attn_metadata) -> bool:
    return attn_metadata.is_all_encoder_attn_metadata_set


def fail_encoder_decoder_prefix_caching() -> None:
    raise NotImplementedError(STR_NOT_IMPL_ENC_DEC_PREFIX_CACHING)


def check_hip_or_chunked_prefill_attention_encdec(
        attn_metadata: AttentionMetadata) -> None:
    '''
    Check for unsupported encoder/decoder scenarios when invoking
    attention.

    Arguments:

    * attn_metadata: Attention metadata structure
    '''
    if is_hip():
        # AMD ROCm/HIP support currently not implemented for
        # encoder/decoder models
        raise NotImplementedError(STR_NOT_IMPL_ENC_DEC_ROCM_HIP)

    if attn_metadata.num_prefill_tokens > 0 and \
            attn_metadata.num_decode_tokens > 0:
        # Encoder/decoder models are currently incompatible
        # with chunked prefill.
        raise NotImplementedError( \
            STR_NOT_IMPL_ENC_DEC_CHUNKED_PREFILL)
