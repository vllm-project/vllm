# SPDX-License-Identifier: Apache-2.0
"""
TokenHasher: Standalone hash computation for the multiprocess server.

Hash function loading logic is adapted from token_database.py to avoid
coupling with TokenDatabase's config/metadata dependencies.

vLLM compatibility notes:
- PR#20511: Introduced kv_cache_utils.init_none_hash()
- PR#23673: Renamed sha256_cbor_64bit to sha256_cbor
- PR#27151: Moved hash functions to vllm.utils.hashing module
"""

# Standard
from typing import Any, Callable
import os

# Third Party
from numba import njit
import numpy as np

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


def _make_blake3_hash_func() -> Callable:
    """Create a blake3-based hash function compatible with the
    (prefix_hash, tuple(tokens), None) calling convention."""
    # Standard
    import struct

    # Third Party
    import blake3 as _blake3

    def blake3_hash(args):
        prefix_hash, tokens, _ = args
        h = _blake3.blake3()
        # Serialize prefix hash
        if isinstance(prefix_hash, bytes):
            h.update(prefix_hash)
        elif isinstance(prefix_hash, int):
            h.update(prefix_hash.to_bytes(8, byteorder="big", signed=True))
        else:
            h.update(bytes(prefix_hash))
        # Serialize token IDs in one batch
        h.update(struct.pack(f">{len(tokens)}I", *tokens))
        return h.digest()  # 32 bytes

    return blake3_hash


class TokenHasher:
    """Computes rolling prefix hashes for token chunks.

    This class encapsulates the hash function loading and hash computation
    logic needed by the multiprocess server to convert token IDs into
    chunk hashes compatible with IPCCacheServerKey (hash mode).
    """

    def __init__(self, chunk_size: int = 256, hash_algorithm: str = "blake3"):
        self.chunk_size = chunk_size
        self.hash_algorithm_name = hash_algorithm
        self.hash_func = self._get_hash_func(hash_algorithm)
        self.none_hash = self._init_none_hash()
        logger.info(
            "TokenHasher initialized: chunk_size=%d, hash_algorithm=%s",
            chunk_size,
            hash_algorithm,
        )

    def _get_hash_func(self, hash_algorithm: str) -> Callable:
        """Load hash function with vLLM version compatibility.

        Adapted from TokenDatabase._get_vllm_hash_func (token_database.py:97-168).
        """
        if hash_algorithm == "blake3":
            logger.info("Using blake3 hash function")
            return _make_blake3_hash_func()

        # Try get_hash_fn_by_name from both locations (PR#27151)
        for module_path in ["vllm.utils.hashing", "vllm.utils"]:
            try:
                module = __import__(module_path, fromlist=["get_hash_fn_by_name"])
                get_hash_fn_by_name = module.get_hash_fn_by_name
                return self._try_get_hash(
                    get_hash_fn_by_name, hash_algorithm, module_path
                )
            except (ImportError, AttributeError, ValueError):
                continue

        # Try direct imports as fallback (for older vLLM versions)
        func_names = (
            ["sha256_cbor", "sha256_cbor_64bit"]
            if hash_algorithm in ("sha256_cbor", "sha256_cbor_64bit")
            else [hash_algorithm]
        )
        for module_path in ["vllm.utils.hashing", "vllm.utils"]:
            for func_name in func_names:
                try:
                    module = __import__(module_path, fromlist=[func_name])
                    hash_func = getattr(module, func_name)
                    logger.info(
                        "Loaded '%s' from %s (direct import)", func_name, module_path
                    )
                    return hash_func
                except (ImportError, AttributeError):
                    continue

        # Fallback to builtin hash
        logger.warning(
            "Could not load '%s' from vLLM. Using builtin hash. "
            "This may cause inconsistencies in distributed caching.",
            hash_algorithm,
        )

        # Check PYTHONHASHSEED when using builtin hash
        if os.getenv("PYTHONHASHSEED") is None:
            logger.warning(
                "Using builtin hash without PYTHONHASHSEED set. "
                "For production environments (non-testing scenarios), you MUST set "
                "PYTHONHASHSEED to ensure consistent hashing across processes. "
                "Example: export PYTHONHASHSEED=0"
            )

        return hash

    def _try_get_hash(
        self, get_hash_fn_by_name: Callable, hash_algorithm: str, module_name: str
    ) -> Callable:
        """Try to get hash function, handling sha256_cbor_64bit rename.

        Adapted from TokenDatabase._try_get_hash (token_database.py:152-168).
        """
        # Handle sha256_cbor_64bit -> sha256_cbor rename (PR#23673)
        names_to_try = (
            ["sha256_cbor", "sha256_cbor_64bit"]
            if hash_algorithm in ("sha256_cbor", "sha256_cbor_64bit")
            else [hash_algorithm]
        )

        for name in names_to_try:
            try:
                hash_func = get_hash_fn_by_name(name)
                logger.info("Loaded '%s' from %s", name, module_name)
                return hash_func
            except ValueError:
                continue
        raise ValueError(f"Hash function '{hash_algorithm}' not found in {module_name}")

    def _init_none_hash(self) -> Any:
        """Initialize NONE_HASH.

        Adapted from TokenDatabase.__init__ (token_database.py:64-82).
        """
        if self.hash_algorithm_name != "blake3":
            try:
                # Third Party
                from vllm.v1.core import kv_cache_utils

                if hasattr(kv_cache_utils, "init_none_hash"):
                    kv_cache_utils.init_none_hash(self.hash_func)
                    none_hash = kv_cache_utils.NONE_HASH
                    logger.info("Initialized NONE_HASH=%s from vLLM", none_hash)
                    return none_hash
            except (
                ImportError,
                AttributeError,
                ValueError,
                RuntimeError,
                # torch._dynamo.device_interface raises AssertionError
                # when CudaInterface is defined on non-CUDA platforms.
                AssertionError,
            ):
                pass

        # Fallback: compute none_hash using our hash function
        none_hash = self.hash_func((0, (0,), None))
        logger.info("Computed NONE_HASH=%s using hash function", none_hash)
        return none_hash

    def hash_tokens(self, tokens: list[int], prefix_hash: Any = None) -> Any:
        """Hash one chunk with rolling prefix.

        Returns int or bytes depending on hash_func.
        """
        if prefix_hash is None:
            prefix_hash = self.none_hash
        return self.hash_func((prefix_hash, tuple(tokens), None))

    def compute_chunk_hashes(
        self,
        token_ids: list[int],
        prefix_hash: Any = None,
        start: int = 0,
        end: int | None = None,
    ) -> list[bytes]:
        """Compute rolling prefix hashes for complete chunks in a token range.

        The rolling hash is always computed from the beginning of
        ``token_ids`` (since each chunk's hash depends on all previous
        chunks), but only hashes for chunks within ``[start, end)`` are
        returned, and hashing stops at ``end`` to avoid unnecessary work.

        ``start`` and ``end`` are token-level indices and must be
        multiples of ``chunk_size``. Partial chunks are discarded.

        Args:
            token_ids: Full token sequence.
            prefix_hash: Optional initial prefix hash (defaults to none_hash).
            start: Token-level start index (must be chunk-aligned).
                Chunks before this index are computed but not returned.
            end: Token-level end index (must be chunk-aligned). When
                provided, hashing stops at this index.

        Returns:
            List of ``bytes`` hash values for chunks in ``[start, end)``.
        """
        hashes: list[bytes] = []
        prefix_hash = self.none_hash if prefix_hash is None else prefix_hash
        effective_len = min(len(token_ids), end) if end is not None else len(token_ids)
        num_complete = effective_len - effective_len % self.chunk_size
        for i in range(0, num_complete, self.chunk_size):
            prefix_hash = self.hash_tokens(
                token_ids[i : i + self.chunk_size], prefix_hash
            )
            if i >= start:
                hashes.append(self.hash_to_bytes(prefix_hash))
        return hashes

    @staticmethod
    def hash_to_bytes(hash_val: Any) -> bytes:
        """Convert hash value to bytes for ObjectKey.chunk_hash.

        Handles both bytes (sha256_cbor) and int (builtin hash) return types.
        """
        if isinstance(hash_val, bytes):
            return hash_val  # sha256_cbor already returns bytes
        return hash_val.to_bytes(8, byteorder="big", signed=True)


### Functions for fast rolling/chunk hash and dict lookup


@njit(cache=True)
def rolling_hash_windows_numba(
    arr_u64: np.ndarray, k: int, base: np.uint64
) -> np.ndarray:
    """
    Compute rolling polynomial hashes over a uint64 array.

    This function computes a polynomial rolling hash over a sliding window
    of size `k` across the input array `arr_u64`. Arithmetic is performed
    in uint64 with natural overflow, which is equivalent to computing
    modulo 2^64.

    Hash definition for a window [x0, x1, ..., x_{k-1}]:

        H = x0 * base^(k-1) + x1 * base^(k-2) + ... + x_{k-1}

    For each subsequent window the hash is updated in O(1):

        H_new = (H - x_old * base^(k-1)) * base + x_new

    Parameters
    ----------
    arr_u64 : np.ndarray[np.uint64]
        Input array of integers encoded as uint64 values.

    k : int
        Sliding window size.

    base : np.uint64
        Base of the polynomial hash. Typically a random odd 64-bit number.

    Returns
    -------
    np.ndarray[np.uint64]
        Array of rolling hash values of length:

            len(arr_u64) - k + 1

        Each element corresponds to the hash of one window.
    """
    n = arr_u64.shape[0]
    out = np.empty(n - k + 1, dtype=np.uint64)

    power = np.uint64(1)
    for _ in range(k - 1):
        power = power * base  # uint64 overflow = mod 2^64

    h = np.uint64(0)
    for i in range(k):
        h = h * base + arr_u64[i]
    out[0] = h

    j = 1
    for i in range(k, n):
        old = arr_u64[i - k]
        new = arr_u64[i]
        h = h - old * power
        h = h * base + new
        out[j] = h
        j += 1

    return out


@njit(cache=True)
def chunk_hash_windows_numba(arr_u64, k, base):
    """Compute polynomial hashes over non-overlapping (chunked) windows.

    Unlike the rolling-hash variant, each window's hash is computed
    independently from scratch, which is efficient when stride equals the
    window size (i.e., windows do not overlap).

    The hash for a window starting at position ``s`` is:

        h = arr[s]*base^(k-1) + arr[s+1]*base^(k-2) + ... + arr[s+k-1]

    computed with natural ``uint64`` overflow (mod 2^64).

    Parameters
    ----------
    arr_u64 : np.ndarray[np.uint64]
        1-D array of token values cast to ``uint64``.
    k : int
        Window (chunk) size.
    base : np.uint64
        Base of the polynomial hash.

    Returns
    -------
    np.ndarray[np.uint64]
        Array of length ``len(arr_u64) // k`` containing one hash per
        non-overlapping chunk. Trailing tokens that do not fill a
        complete chunk are ignored.
    """
    n = arr_u64.shape[0]
    num_windows = n // k
    out = np.empty(num_windows, dtype=np.uint64)

    for w in range(num_windows):
        h = np.uint64(0)
        start = w * k
        # Compute fresh hash for this block
        for i in range(start, start + k):
            h = h * base + arr_u64[i]
        out[w] = h
    return out


@njit(cache=True)
def update_table_id_numba(
    hashes_u64: np.ndarray,
    table_id_i64: np.ndarray,
    vals_to_update: np.ndarray,
):
    """
    Update the direct-address table with new ID values for given hashes.

    For each hash in `hashes_u64`, compute the index as:

        idx = hash & (table_id_i64.size - 1)

    and update `table_id_i64[idx]` with the corresponding value from
    `vals_to_update`.

    Parameters
    ----------
    hashes_u64 : np.ndarray[np.uint64]
        Array of hash values to update.

    table_id_i64 : np.ndarray[np.int64]
        Direct-address lookup table mapping index → ID. This array is
        modified in-place.

    vals_to_update : np.ndarray[np.int64]
        Array of new ID values to write into the table. Must have the same
        length as `hashes_u64`.
    """
    n = hashes_u64.shape[0]
    m = table_id_i64.shape[0]

    for i in range(n):
        idx = hashes_u64[i] & (m - 1)  # Assuming m is a power of 2
        table_id_i64[idx] = vals_to_update[i]


@njit(cache=True)
def unique_hits_direct_id_numba(
    hashes_u64: np.ndarray, table_id_i64: np.ndarray, mask_u64: np.uint64, num_ids: int
) -> np.ndarray:
    """
    Perform direct-address lookup with deduplication of results.

    This function looks up each hash in a direct-address table using
    the lower bits of the hash:

        idx = hash & mask

    The lookup table maps each index to an integer ID.

    The function returns **unique IDs only**, meaning that if the same
    ID appears multiple times across the hash stream it will be returned
    only once.

    Parameters
    ----------
    hashes_u64 : np.ndarray[np.uint64]
        Array of rolling hash values.

    table_id_i64 : np.ndarray[np.int64]
        Direct-address lookup table mapping index → ID.
        Values of -1 represent "no entry".

    mask_u64 : np.uint64
        Bitmask used to compute the index:

            idx = hash & mask_u64

        Typically mask = (2^bits - 1).

    num_ids : int
        Maximum possible ID value + 1. This determines the size of the
        internal `seen` array used for deduplication.

    Returns
    -------
    np.ndarray[np.int64]
        Array containing the unique IDs encountered in the lookup stream.
        Length ≤ len(hashes_u64).
    """

    # TODO(Jiayi): These allocations can be avoided by pre-allocations
    seen = np.zeros(num_ids, dtype=np.uint8)  # 1 byte per possible id
    out = np.empty(hashes_u64.shape[0], dtype=np.int64)

    m = 0
    for i in range(hashes_u64.shape[0]):
        idx = hashes_u64[i] & mask_u64
        hit = table_id_i64[idx]

        if hit != -1 and seen[hit] == 0:
            seen[hit] = 1
            out[m] = hit
            m += 1

    return out[:m]
