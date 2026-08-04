# SPDX-License-Identifier: Apache-2.0
"""Codec error types.

These are the only exceptions the codec layer raises.  Every error
path in the codec must surface one of these with a message that
names the observed-vs-expected condition; do not raise bare
RuntimeError or KeyError.
"""


class CodecError(Exception):
    """Base class for all KV codec errors."""


class CodecMismatchError(CodecError):
    """The encoded blob was produced under a different config than
    what the caller is requesting.

    Examples: cross-model cache poisoning (different model_id), a
    different page_size, an attention backend the caller is not
    running, or a scale_scope the caller did not configure.
    """


class CorruptEncodedKVError(CodecError):
    """The encoded blob's header or payload failed an integrity
    check.

    Examples: codec_magic does not match, codec_version is unknown,
    payload_crc32c does not match, the payload is truncated.
    """


class UnsupportedConfigError(CodecError):
    """The caller asked the codec to emit something it cannot.

    Examples: a runtime_layout this codec doesn't implement, or an
    FP8 variant (e5m2) reserved-but-not-implemented in v1.
    """
