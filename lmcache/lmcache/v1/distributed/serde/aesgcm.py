# SPDX-License-Identifier: Apache-2.0
"""AES-GCM encryption serde for L2-at-rest confidentiality.

Encrypts KV cache bytes on store (L1->L2) and decrypts on load (L2->L1),
keyed per ``cache_salt`` so each tenant's ciphertext is distinct. Content is
treated as an opaque byte blob (not tensor-typed), an exact round-trip.

Scope is the L2 tier only: L1 and L0 hold plaintext. See
``docs/design/v1/distributed/serde/aesgcm.md`` for the threat model, key
model, and residual-metadata notes.
"""

# Standard
import os

# Third Party
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.serde.async_processor import AsyncSerdeProcessor
from lmcache.v1.distributed.serde.base import Deserializer, SerdeProcessor, Serializer
from lmcache.v1.distributed.serde.factory import register_serde_factory
from lmcache.v1.distributed.serde.key_provider import HkdfKeyProvider, KeyProvider
from lmcache.v1.memory_management import MemoryObj

logger = init_logger(__name__)

# Wire frame per chunk: [1B version][IV_LEN IV][ciphertext || TAG_LEN tag].
_VERSION = 1
_IV_LEN = 12  # 96-bit nonce (GCM native size)
_TAG_LEN = 16  # GCM authentication tag
_HDR_LEN = 1 + _IV_LEN
_FRAME_OVERHEAD = _HDR_LEN + _TAG_LEN

_SUPPORTED_AES_BITS = (128, 256)

# HKDF ``info`` domain-separation prefix for keys derived for this serde.
_HKDF_INFO_PREFIX = b"lmcache-l2-aesgcm-v1"


def _plaintext_bytes(layout_desc: MemoryLayoutDesc) -> int:
    """Return the total byte size of the KV data described by ``layout_desc``."""
    total = 0
    for shape, dtype in zip(layout_desc.shapes, layout_desc.dtypes, strict=True):
        n = 1
        for dim in shape:
            n *= int(dim)
        total += n * torch.empty((), dtype=dtype).element_size()
    return total


class AesGcmSerializer(Serializer):
    """Encrypt KV bytes with AES-GCM, keyed per ``cache_salt``.

    Args:
        key_provider: Supplies the AES key for each key's ``cache_salt``.
    """

    def __init__(self, key_provider: KeyProvider) -> None:
        self._keys = key_provider

    def serialize(self, src: MemoryObj, dst: MemoryObj, key: ObjectKey) -> int:
        """Encrypt ``src`` into ``dst`` as ``[version][iv][ciphertext||tag]``.

        Args:
            src: Source KV bytes (read-locked).
            dst: Destination byte buffer, sized by ``estimate_serialized_size``.
            key: Object key; ``key.cache_salt`` selects the encryption key.

        Returns:
            Number of bytes written to ``dst``.
        """
        dek = self._keys.get_key(key.cache_salt)
        iv = os.urandom(_IV_LEN)
        ciphertext = AESGCM(dek).encrypt(iv, bytes(src.byte_array), None)
        # Cast to native "B": MemoryObj.byte_array is a ctypes-backed memoryview
        # with format "<B", which does not support slice assignment.
        out = memoryview(dst.byte_array).cast("B")
        out[0:1] = bytes((_VERSION,))
        out[1:_HDR_LEN] = iv
        out[_HDR_LEN : _HDR_LEN + len(ciphertext)] = ciphertext
        return _HDR_LEN + len(ciphertext)

    def estimate_serialized_size(self, layout_desc: MemoryLayoutDesc) -> int:
        """Return exact framed size: plaintext + version + IV + tag.

        AES-GCM ciphertext is the plaintext length plus a 16-byte tag (no
        padding), so this is exact rather than an upper bound.
        """
        return _plaintext_bytes(layout_desc) + _FRAME_OVERHEAD


class AesGcmDeserializer(Deserializer):
    """Decrypt AES-GCM ciphertext produced by :class:`AesGcmSerializer`.

    Args:
        key_provider: Supplies the AES key for each key's ``cache_salt``.
    """

    def __init__(self, key_provider: KeyProvider) -> None:
        self._keys = key_provider

    def deserialize(self, src: MemoryObj, dst: MemoryObj, key: ObjectKey) -> None:
        """Decrypt ``src``'s frame into ``dst``'s KV buffer.

        The ciphertext length is taken from ``dst``'s size (the plaintext
        length), not ``src``, which may be padded larger than the stored frame.

        Args:
            src: Source byte buffer holding the stored frame (may be padded).
            dst: Destination KV buffer (write-locked); its size is the plaintext
                length.
            key: Object key; ``key.cache_salt`` selects the decryption key.

        Raises:
            ValueError: If the frame header/version is malformed.
            cryptography.exceptions.InvalidTag: If the ciphertext was tampered
                with or the wrong key is used (surfaces as a load miss).
        """
        out = memoryview(dst.byte_array).cast("B")
        plaintext_len = len(out)
        blob = bytes(src.byte_array)
        frame_end = _HDR_LEN + plaintext_len + _TAG_LEN
        if len(blob) < frame_end or blob[0] != _VERSION:
            raise ValueError("aesgcm serde: malformed frame or unknown version")
        dek = self._keys.get_key(key.cache_salt)
        plaintext = AESGCM(dek).decrypt(
            blob[1:_HDR_LEN], blob[_HDR_LEN:frame_end], None
        )
        out[: len(plaintext)] = plaintext


def _create_aesgcm_serde(kwargs: dict[str, object]) -> SerdeProcessor:
    """Build the aesgcm serde from ``SerdeConfig.kwargs``.

    Kwargs:
        key_provider: ``"hkdf"`` (default). ``"keyring"`` is reserved for
            per-tenant provisioned keys and is not yet implemented.
        aes_bits: ``128`` (default) or ``256``.
        master_key_path: Path to the master-key file (required for ``hkdf``).
        max_workers: Serde thread-pool size (default ``1``).

    Raises:
        ValueError: On an unsupported ``key_provider`` / ``aes_bits``, or an
            empty ``master_key_path``.
        OSError: If ``master_key_path`` is set but cannot be read.
    """
    provider_name = str(kwargs.get("key_provider", "hkdf"))
    aes_bits = int(kwargs.get("aes_bits", 128))  # type: ignore[call-overload]
    if aes_bits not in _SUPPORTED_AES_BITS:
        raise ValueError(
            f"aes_bits must be one of {_SUPPORTED_AES_BITS}, got {aes_bits}"
        )

    if provider_name == "hkdf":
        master_key_path = str(kwargs.get("master_key_path", ""))
        if not master_key_path:
            raise ValueError("aesgcm serde 'hkdf' requires 'master_key_path'")
        with open(master_key_path, "rb") as f:
            master_key = f.read()
        provider: KeyProvider = HkdfKeyProvider(
            master_key, key_len=aes_bits // 8, info_prefix=_HKDF_INFO_PREFIX
        )
    else:
        raise ValueError(
            f"unsupported key_provider {provider_name!r} (only 'hkdf' is implemented)"
        )

    max_workers = int(kwargs.get("max_workers", 1))  # type: ignore[call-overload]
    return AsyncSerdeProcessor(
        AesGcmSerializer(provider),
        AesGcmDeserializer(provider),
        max_workers=max_workers,
    )


register_serde_factory("aesgcm", _create_aesgcm_serde)
