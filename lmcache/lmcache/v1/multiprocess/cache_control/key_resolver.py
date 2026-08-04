# SPDX-License-Identifier: Apache-2.0
"""Resolve a token sequence to the cache object keys it maps to.

Cache-control operations describe content by ``token_ids`` rather than internal
:class:`ObjectKey` values, which callers cannot construct.
:func:`resolve_object_keys` hashes the tokens and expands each complete chunk
into one key per rank -- the same fan-out the lookup path uses. It is the single
resolver shared by the MP server (node) and the coordinator; node callers that
also need the L1 layout do the readiness lookup themselves before calling it.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING

# First Party
from lmcache.v1.distributed.api import ObjectKey, ipc_key_to_object_keys
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.token_hasher import TokenHasher

# Hard cap on how many token ids a single token-addressed request may carry.
# Keeps the request body bounded and the synchronous hashing / key-construction
# work proportionate.
MAX_TOKEN_IDS = 1_000_000


def resolve_object_keys(
    token_hasher: TokenHasher,
    model_name: str,
    world_size: int,
    token_ids: list[int],
    cache_salt: str,
) -> tuple[list[ObjectKey], int]:
    """Resolve a token sequence to the object keys of its complete chunks.

    Hashes ``token_ids`` and expands each complete chunk into one key per rank
    (a single object group, so MP servers must run with
    ``--no-separate-object-groups``). The ``token_hasher``
    must be configured to match the fleet's ``chunk_size`` / ``hash_algorithm``
    or the resolved keys will not match what the servers stored.

    Args:
        token_hasher: Hasher configured to match the fleet's chunk size and
            hash algorithm.
        model_name: Model whose rank fan-out to use.
        world_size: Tensor-parallel world size selecting the per-rank fan-out.
        token_ids: The token sequence to resolve.
        cache_salt: Per-tenant isolation salt.

    Returns:
        ``(obj_keys, chunk_count)``. ``obj_keys`` holds one key per (chunk, rank)
        and is empty (with ``chunk_count`` 0) when the sequence is shorter than
        one chunk.

    Raises:
        ValueError: ``token_ids`` exceeds the per-request cap, or a key field
            (e.g. ``cache_salt``) is invalid.
    """
    if len(token_ids) > MAX_TOKEN_IDS:
        raise ValueError(
            f"too many token_ids in a single request "
            f"(limit={MAX_TOKEN_IDS}, got={len(token_ids)})"
        )
    ipc_key = IPCCacheServerKey(
        model_name=model_name,
        world_size=world_size,
        worker_id=None,
        token_ids=tuple(token_ids),
        start=0,
        end=len(token_ids),
        request_id="",
        cache_salt=cache_salt,
    )
    chunk_hashes = token_hasher.compute_chunk_hashes(list(token_ids))
    if not chunk_hashes:
        return [], 0
    obj_keys = ipc_key_to_object_keys(ipc_key, chunk_hashes, [0])[0]
    return obj_keys, len(chunk_hashes)
