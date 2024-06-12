"""Attention utils"""

from vllm.attention import AttentionMetadata

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


def is_encoder_decoder_metadata_assuming_supported_backend(
        attn_metadata) -> bool:
    '''
    Return True of the attn_metadata argument contains
    the metadata fields that would be required for
    encoder attention, which proves that the user is
    not running a purely decoder-only model.

    Assumes attn_metadata is derived from a backend that supports
    encoder/decoder models.

    Arguments:

    * attn_metadata: instance of supported backend metadata. 
                     Type annotation omitted to avoid circular import.


    Returns:

    * True if attn_metadata is configured for an encoder/decoder model
    '''
    return attn_metadata.is_all_encoder_attn_metadata_set


def fail_encoder_decoder_prefix_caching() -> None:
    '''
    Fail with NotImplementedError & a message indicating
    enc/dec + prefix caching is unsupported
    '''
    raise NotImplementedError(STR_NOT_IMPL_ENC_DEC_PREFIX_CACHING)


def assert_no_encdec_chunked_prefill_assuming_supported_backend(
        attn_metadata: AttentionMetadata) -> None:
    '''
    Fail if encoder/decoder model is being executed with
    chunked prefill.

    Assumes we already know that the particular attention
    backend in-use is supported.
    
    Arguments:

    * attn_metadata: Attention metadata structure
    '''

    if not is_encoder_decoder_metadata_assuming_supported_backend(
            attn_metadata):
        # Only care about encoder/decoder
        # scenarios.
        return

    if attn_metadata.num_prefill_tokens is None or \
        attn_metadata.num_decode_tokens is None:
        # The metadata which would be
        # indicative of chunked prefill is unset;
        # this may be the case for encoder-only models
        return

    if attn_metadata.num_prefill_tokens > 0 and \
            attn_metadata.num_decode_tokens > 0:
        # Encoder/decoder models are currently incompatible
        # with chunked prefill.
        raise NotImplementedError( \
            STR_NOT_IMPL_ENC_DEC_CHUNKED_PREFILL)
