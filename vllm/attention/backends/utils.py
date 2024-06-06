"""Attention utils"""

from vllm.attention import AttentionMetadata
from vllm.attention.backends.abstract import AttentionType
from vllm.attention.backends.xformers import XFormersMetadata
from vllm.utils import is_hip

# Error string(s) for encoder/decoder
# unsupported attention scenarios

STR_NOT_IMPL_ENC_DEC_CHUNKED_PREFILL = \
"Encoder/decoder models " + \
"currently do not support chunked prefill."

STR_NOT_IMPL_ENC_DEC_ROCM_HIP = \
"Encoder/decoder models currently" + \
"do not support ROCm/HIP."

STR_NOT_IMPL_ENC_DEC_NON_XFORMERS_BACKEND = \
"Encoder/decoder models currently support only the XFormers backend."

# Check for unsupported encoder/decoder scenarios


def check_hip_or_chunked_prefill_attention_encdec(
        attn_metadata: AttentionMetadata):
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

    if not isinstance(attn_metadata, XFormersMetadata):
        # Right now encoder/decoder support is only implemented
        # for the XFormers backend. Pretty unlikely to encounter
        # this case currently given this function will be invoked inside
        # xFormers backend.
        raise NotImplementedError(STR_NOT_IMPL_ENC_DEC_NON_XFORMERS_BACKEND)

    if attn_metadata.num_prefill_tokens > 0 and \
            attn_metadata.num_decode_tokens > 0:
        # Encoder/decoder models are currently incompatible
        # with chunked prefill.
        raise NotImplementedError( \
            STR_NOT_IMPL_ENC_DEC_CHUNKED_PREFILL)
