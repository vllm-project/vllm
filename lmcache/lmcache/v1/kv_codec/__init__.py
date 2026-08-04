# SPDX-License-Identifier: Apache-2.0
"""KV storage codec layer.

Codecs operate above the storage backends: they take K and V
tensors plus scales and produce a self-describing byte blob that
any backend (local disk, GDS, NIXL, Redis, etc.) can write
opaque-by-byte. This module owns storage compression only; tier
placement lives elsewhere. See
``lmcache/v1/kv_codec/asym_k16_v8.py`` for the asymmetric K16/V8
codec.

Design constraints:

- Cross-model / cross-tokenizer / cross-rope / cross-backend cache
  poisoning is gated by hashes carried in the codec header. See
  ``EncodedKV.hashes`` and ``_check_hash_match``.
- All on-disk integers are little-endian regardless of host.
"""

# First Party
from lmcache.v1.kv_codec.asym_k16_v8 import (
    AsymK16V8Codec,
    compute_v_scales,
    dequantize_v_fp8,
    quantize_v_fp8,
)
from lmcache.v1.kv_codec.encoded_kv import (
    CODEC_MAGIC,
    CodecHashes,
    CodecVersion,
    EncodedKV,
    ScaleScope,
    deserialize_header,
    serialize_header,
)
from lmcache.v1.kv_codec.errors import (
    CodecError,
    CodecMismatchError,
    CorruptEncodedKVError,
    UnsupportedConfigError,
)

__all__ = [
    # Public codec interface
    "AsymK16V8Codec",
    "EncodedKV",
    "CodecHashes",
    "ScaleScope",
    "CodecVersion",
    "CODEC_MAGIC",
    # Errors
    "CodecError",
    "CodecMismatchError",
    "CorruptEncodedKVError",
    "UnsupportedConfigError",
    # Quant primitives (exposed for tests and benchmarks)
    "compute_v_scales",
    "quantize_v_fp8",
    "dequantize_v_fp8",
    # Header serialization (exposed for low-level tests)
    "serialize_header",
    "deserialize_header",
]
