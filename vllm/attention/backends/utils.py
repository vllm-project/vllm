"""Attention utils"""

# Error string(s) for encoder/decoder
# unsupported attention scenarios

STR_NOT_IMPL_ENC_DEC_CHUNKED_PREFILL = \
"Encoder/decoder models " + \
"currently do not support chunked prefill."