# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.multimodal.inputs import PlaceholderRange

# Signed int64 positive range so values survive torch.long assignment.
_TOKEN_VALUE_MASK = (1 << 63) - 1


def mm_hash_to_token_value(s: str) -> int:
    """Convert a multimodal content identifier into a 63-bit token value.

    Identifiers are typically hex digests. Non-hex strings are hashed first
    so caller-supplied UUIDs still expand to a wide integer. Truncating to
    16 bits would let distinct media collide in the LMCache token-id key.
    """
    try:
        value = int(s, 16)
    except ValueError:
        value = int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest(), "big")
    return value & _TOKEN_VALUE_MASK


def request_identity_mixer(
    cache_salt: str | None = None, lora_name: str | None = None
) -> int | None:
    """Return a 63-bit mixer, or None when there is no extra identity."""
    if not cache_salt and not lora_name:
        return None
    material = f"{cache_salt or ''}\0{lora_name or ''}".encode()
    digest = hashlib.sha256(material).digest()[:8]
    return int.from_bytes(digest, "big") & _TOKEN_VALUE_MASK


def apply_mm_hashes_to_token_ids(
    token_ids: torch.Tensor,
    mm_hashes: list[str],
    mm_positions: list["PlaceholderRange"],
    cache_salt: str | None = None,
    lora_name: str | None = None,
) -> torch.Tensor:
    """Bind LMCache token-id keys to multimodal and request identity.

    Overwrites placeholder spans with the full-width media identifier, then
    XOR-mixes cache_salt and LoRA name into every token so those dimensions
    partition every LMCache chunk, not only placeholder blocks.
    """
    n = token_ids.size(0)
    for hash_str, placeholder in zip(mm_hashes, mm_positions):
        start, length = placeholder.offset, placeholder.length
        if start >= n:
            continue
        end = min(start + length, n)
        token_ids[start:end] = mm_hash_to_token_value(hash_str)
    mixer = request_identity_mixer(cache_salt, lora_name)
    if mixer is not None and n > 0:
        token_ids.bitwise_xor_(mixer)
    return token_ids
