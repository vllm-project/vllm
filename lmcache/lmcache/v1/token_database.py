# SPDX-License-Identifier: Apache-2.0
"""
vLLM compatibility notes:
- PR#20511: Introduced kv_cache_utils.init_none_hash()
  https://github.com/vllm-project/vllm/pull/20511
- PR#23673: Renamed sha256_cbor_64bit to sha256_cbor
  https://github.com/vllm-project/vllm/pull/23673
- PR#27151: Moved hash functions to vllm.utils.hashing module
  https://github.com/vllm-project/vllm/pull/27151

TODO(baoloongmao): Move this to vllm_v1_adapter to decouple from vLLM
"""

# Standard
from typing import Any, Iterable, List, Optional, Tuple, Union
import abc
import os

# Third Party
from transformers import AutoTokenizer
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata

logger = init_logger(__name__)

NONE_HASH = 0


def _normalize_hash_to_int(hash_value: Union[int, bytes]) -> int:
    """Normalize hash outputs to LMCache's int chunk-hash representation.

    This function is triggered when vLLM's ``sha256_cbor`` hash function is
    used because it returns a 32-byte digest. vLLM's
    ``kv_cache_utils.init_none_hash`` can therefore initialize ``NONE_HASH`` as
    bytes, and direct hash calls for token chunks can also return bytes.

    LMCache stores chunk hashes in ``CacheEngineKey`` and serializes them with
    msgpack, so byte digests must be folded into uint64-compatible ints before
    they enter the prefix hash chain. This also keeps ``NONE_HASH`` and later
    prefix hashes using the same structural type for CBOR hashing.

    Args:
        hash_value: Hash output from vLLM or Python's builtin hash.

    Returns:
        The original int hash value, or the first eight bytes of a digest as a
        big-endian int.
    """
    if isinstance(hash_value, bytes):
        return int.from_bytes(hash_value[:8], "big")
    return hash_value


# Type alias for process_tokens return value
# (start_index, end_index, cache_engine_key｜hash)
ProcessTokensResult = Tuple[int, int, Union[CacheEngineKey, int]]


class TokenDatabase(metaclass=abc.ABCMeta):
    """TokenDatabase is used to convert input tokens into list of
    cache engine keys. There are multiple ways to implement this:

    - ChunkedTokenDatabase: It processes tokens into chunks and convert
    each chunk into a cache engine key using prefix hash.

    - SegmentTokenDatabase: It processes tokens into segments based on
    special separators and convert each segment into a cache engine key.
    """

    @abc.abstractmethod
    def __init__(
        self,
        config: Optional[LMCacheEngineConfig] = None,
        metadata: Optional[LMCacheMetadata] = None,
    ):
        global NONE_HASH

        hash_algorithm: str = (
            config.pre_caching_hash_algorithm if config is not None else "builtin"
        )

        # Get hash function with vLLM version compatibility
        self.hash_func = self._get_vllm_hash_func(hash_algorithm)

        # Initialize NONE_HASH (vLLM >= PR#20511)
        # NOTE: For centralized cache sharing, ensure PYTHONHASHSEED is
        # set consistently across all processes (e.g., export PYTHONHASHSEED=0).
        try:
            # Third Party
            from vllm.v1.core import kv_cache_utils

            if hasattr(kv_cache_utils, "init_none_hash"):
                kv_cache_utils.init_none_hash(self.hash_func)
                NONE_HASH = _normalize_hash_to_int(kv_cache_utils.NONE_HASH)
                logger.info(
                    "Initialized NONE_HASH=%s from vLLM (>= PR#20511)", NONE_HASH
                )
            else:
                NONE_HASH = 0
                logger.info("Using default NONE_HASH=0 (vLLM < PR#20511)")
        except (ImportError, AttributeError):
            NONE_HASH = 0
            logger.info("Using default NONE_HASH=0 (vLLM not available)")

        logger.info("Using hash algorithm: %s", hash_algorithm)
        self.metadata = metadata
        # Whether only the first rank should save cache. This flag is also used
        # to control the logical world_size embedded into CacheEngineKey.
        self.save_only_first_rank = False
        if config is not None and metadata is not None:
            # save_only_first_rank only works when use MLA, follow the same
            # semantics as LMCacheEngine and memory allocator.
            self.save_only_first_rank = (
                config.get_extra_config_value("save_only_first_rank", metadata.use_mla)
                and metadata.use_mla
            )

    def _get_vllm_hash_func(self, hash_algorithm: str):
        """Get hash function from vLLM with version compatibility.

        Tries multiple import paths to support different vLLM versions:
        - vllm.utils.hashing.get_hash_fn_by_name (>= PR#27151)
        - vllm.utils.get_hash_fn_by_name (< PR#27151)
        - Direct imports as fallback
        - sha256_cbor_64bit -> sha256_cbor rename (PR#23673)
        """
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

    def _try_get_hash(self, get_hash_fn_by_name, hash_algorithm: str, module_name: str):
        """Try to get hash function, handling sha256_cbor_64bit rename."""
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

    @abc.abstractmethod
    def process_tokens(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        mask: Optional[torch.Tensor] = None,
        make_key: bool = True,
        request_configs: Optional[dict] = None,
    ) -> Iterable[ProcessTokensResult]:
        """Process the tokens and return the corresponding cache engine keys.

        :param Optional[Union[torch.Tensor, List[int]]] tokens: The tokens to process.

        :param Optional[List[int]] hashes: The hashes to process. If provided,
            it will be used instead of tokens to generate cache engine keys.

        :param Optional[List[int]] offsets: The number of tokens in each chunk.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param bool make_key: Whether to make the cache engine key or not.
            If False, the hash value will be returned instead.

        :param Optional[dict] request_configs: The configs of the request.

        :returns: A iterable of tuples with three elements. The first element
            is the start index of the tokens for the key. The second element
            is the end index of the tokens for the key. The third element is
            the cache engine key (or hash) for the tokens.
        """

        raise NotImplementedError

    def _make_key_by_hash(
        self, chunk_hash: int, request_configs: Optional[dict] = None
    ):
        assert self.metadata is not None
        # When save_only_first_rank is enabled (for MLA), we deliberately
        # collapse the CacheEngineKey.world_size to 1 so that cache keys
        # become world-size agnostic across compatible deployments.
        return CacheEngineKey(
            self.metadata.model_name,
            self.metadata.world_size if not self.save_only_first_rank else 1,
            self.metadata.worker_id,
            chunk_hash,
            self.metadata.kv_dtype,
            request_configs,
        )

    def _canonicalize_hash_inputs(
        self,
        prefix_hash: Optional[int],
        tokens_tuple: Tuple[int, ...],
        extra_keys: Optional[List[Any]],
    ) -> Tuple[int, Tuple[int, ...], Tuple[Any, ...]]:
        """
        Canonicalize hash inputs so that semantically identical requests
        produce structurally identical hash inputs across instances.
        - prefix_hash: int or NONE_HASH if None
        - tokens_tuple: tuple of token IDs
        - extra_keys: tuple of additional keys, empty if None
        """
        return (
            prefix_hash if prefix_hash is not None else NONE_HASH,
            tokens_tuple,
            tuple(extra_keys) if extra_keys is not None else (),
        )

    def _hash_tokens(
        self,
        tokens: Union[torch.Tensor, List[int]],
        prefix_hash: Optional[int] = None,
        extra_keys: Optional[list[Any]] = None,
    ) -> int:
        if isinstance(tokens, torch.Tensor):
            tokens_tuple = tuple(tokens.cpu().tolist())
        elif isinstance(tokens, list):
            tokens_tuple = tuple(tokens)
        else:
            raise ValueError(f"Unsupported tokens type: {type(tokens)}")

        # Ignore extra keys for now
        # Extra keys are for multi-modal inputs and
        # request specific metadata (e.g., LoRA ID).
        # Use default values for None to maintain a fixed tuple structure for hashing.

        # Use helper to canonicalize inputs to ensure consistent hashing
        # This replaces the logic that was causing inconsistency
        canon_prefix, canon_tokens, canon_extra = self._canonicalize_hash_inputs(
            prefix_hash, tokens_tuple, extra_keys
        )

        return _normalize_hash_to_int(
            self.hash_func((canon_prefix, canon_tokens, canon_extra))
        )


class ChunkedTokenDatabase(TokenDatabase):
    def __init__(
        self,
        config: Optional[LMCacheEngineConfig] = None,
        metadata: Optional[LMCacheMetadata] = None,
    ):
        super(ChunkedTokenDatabase, self).__init__(config, metadata)

        if config is not None:
            self.config = config
            self.chunk_size = config.chunk_size

            # Check for cross-process cache sharing setup
            if os.getenv("PYTHONHASHSEED") is None:
                if config.remote_url is not None:
                    logger.warning(
                        "Centralized cache sharing detected "
                        "but PYTHONHASHSEED not set. "
                        "For consistent caching, set: export PYTHONHASHSEED=0 "
                        "before the engine starts."
                    )
                if config.enable_pd:
                    logger.error(
                        "P/D Disaggregation detected "
                        "but PYTHONHASHSEED not set. "
                        "For consistent caching, set: export PYTHONHASHSEED=0 "
                        "before the engine starts. "
                        "This will cause incorrect KV cache transfer."
                    )
        else:  # Default values
            self.config = None
            self.chunk_size = 256

    def _get_init_hash(self) -> int:
        return NONE_HASH

    def _chunk_tokens(
        self,
        tokens: Union[torch.Tensor, List[int]],
    ) -> Iterable[Union[torch.Tensor, List[int]]]:
        """
        Chunk the tokens into chunks of size self.chunk_size.

        :param tokens: the input tokens, with shape [seq_len]
            device: the target device after chunking

        :return: a generator of chunks of tokens, each with
                shape [chunk_size]
        """
        save_unfull_chunk = (
            self.config.save_unfull_chunk if self.config is not None else True
        )
        end = (
            len(tokens)
            if save_unfull_chunk
            else (len(tokens) - len(tokens) % self.chunk_size)
        )
        for i in range(0, end, self.chunk_size):
            yield tokens[i : i + self.chunk_size]

    def _prefix_hash(
        self,
        token_chunks: Iterable[Union[torch.Tensor, List[int]]],
    ) -> Iterable[int]:
        prefix_hash = self._get_init_hash()
        for token_chunk in token_chunks:
            prefix_hash = self._hash_tokens(token_chunk, prefix_hash)
            yield prefix_hash

    @_lmcache_nvtx_annotate
    def process_tokens(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        mask: Optional[torch.Tensor] = None,
        make_key: bool = True,
        request_configs: Optional[dict] = None,
    ) -> Iterable[ProcessTokensResult]:
        """Process the tokens/hashes and return the corresponding cache engine keys.

        :param Optional[Union[torch.Tensor, List[int]]] tokens: The tokens to process.

        :param Optional[List[int]] hashes: The hashes to process. If provided,
            it will be used instead of tokens to generate cache engine keys.

        :param Optional[List[int]] offsets: The number of tokens in each chunk.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param bool make_key: Whether to make the cache engine key or not.
            If False, the hash value will be returned instead.

        :param Optional[dict] request_configs: The configs of the request.

        :returns: A iterable of tuples with three elements. The first element
            is the start index of the tokens for the key. The second element
            is the end index of the tokens for the key. The third element is
            the cache engine key (or hash) for the tokens.

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        if mask is not None:
            num_falses = mask.numel() - mask.long().sum().item()
        else:
            num_falses = 0

        if num_falses % self.chunk_size != 0:
            raise ValueError(
                "The number of Falses in the mask is not a multiple of the chunk size."
            )

        if tokens is not None:
            total_len = len(tokens)
            token_chunks = self._chunk_tokens(tokens)
            prefix_hashes = self._prefix_hash(token_chunks)
            for chunk_id, hash_val in enumerate(prefix_hashes):
                start_idx = chunk_id * self.chunk_size
                end_idx = min(start_idx + self.chunk_size, total_len)
                if start_idx < num_falses:
                    continue
                else:
                    if make_key:
                        yield (
                            start_idx,
                            end_idx,
                            self._make_key_by_hash(hash_val, request_configs),
                        )
                    else:
                        yield start_idx, end_idx, hash_val
        elif hashes is not None:
            assert offsets is not None, (
                "If hashes are provided, offsets must also be provided."
            )
            start_idx = 0
            for hash_val, offset in zip(hashes, offsets, strict=False):
                end_idx = start_idx + offset
                if make_key:
                    yield (
                        start_idx,
                        end_idx,
                        self._make_key_by_hash(hash_val, request_configs),
                    )
                else:
                    yield start_idx, end_idx, hash_val
                start_idx = end_idx
        else:
            raise ValueError("Either tokens or hashes must be provided.")


class SegmentTokenDatabase(TokenDatabase):
    """
    Currently, we still use special separators to identify chunks.
    In the future, we might need to implement a fast substring match.
    """

    def __init__(self, config: LMCacheEngineConfig, metadata: LMCacheMetadata):
        super(SegmentTokenDatabase, self).__init__(config, metadata)

        self.tokenizer = AutoTokenizer.from_pretrained(metadata.model_name)

        # TODO (Jiayi): figure out how to decide when
        # to use `1:` (whether there's a special starting token
        # in the beginning)
        self.sep_tokens = self.tokenizer.encode(config.blend_special_str)[1:]
        self.sep_tokens = torch.tensor(self.sep_tokens, device="cpu")
        self.sep_len = len(self.sep_tokens)

    def _fast_split_by_subtensor(self, tokens: torch.Tensor) -> Iterable[torch.Tensor]:
        """Match the `sep_tokens` with sliding windows"""

        if self.sep_len == 0 or len(tokens) < self.sep_len:
            yield tokens

        # Unfold into sliding windows
        # shape: (num_tokens-sep_len+1, sep_len)
        windows = tokens.unfold(0, self.sep_len, 1)

        # Compare each window with sep_tokens
        matches = (
            (windows == self.sep_tokens).all(dim=1).nonzero(as_tuple=True)[0].tolist()
        )

        # Split based on matches
        start = 0
        for idx in matches:
            yield tokens[start:idx]
            start = idx + self.sep_len
        # yield last chunk
        yield tokens[start:]

    def process_tokens(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        mask: Optional[torch.Tensor] = None,
        make_key: bool = True,
        request_configs: Optional[dict] = None,
    ) -> Iterable[ProcessTokensResult]:
        """Process the tokens and return the corresponding cache engine keys.

        :param Union[torch.Tensor, List[int]] tokens: The tokens to process.

        :param Optional[List[int]] hashes: The hashes to process. If provided,
            it will be used instead of tokens to generate cache engine keys.

        :param Optional[List[int]] offsets: The number of tokens in each chunk.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param bool make_key: Whether to make the cache engine key or not.
            If False, the hash value will be returned instead.

        :param Optional[dict] request_configs: The configs of the request.

        :returns: A iterable of tuples with three elements. The first element
            is the start index of the tokens for the key. The second element
            is the end index of the tokens for the key. The third element is
            the cache engine key for the tokens.

        """

        if tokens is not None:
            if not isinstance(tokens, torch.Tensor):
                tokens = torch.tensor(tokens, dtype=torch.long, device="cpu")
            else:
                tokens = tokens.to(device="cpu", dtype=torch.long)

            if mask is not None:
                num_falses = mask.numel() - mask.long().sum().item()
            else:
                num_falses = 0
            assert num_falses < len(tokens), (
                "The number of Falses in the mask shouldn't "
                "be less than the length of tokens."
            )

            token_chunks = self._fast_split_by_subtensor(tokens)
            start_idx = 0
            for idx, token_chunk in enumerate(token_chunks):
                token_chunk_len = len(token_chunk)
                end_idx = start_idx + token_chunk_len
                if idx > 0:
                    start_idx += self.sep_len
                    end_idx += self.sep_len
                if start_idx >= num_falses:
                    if make_key:
                        yield (
                            start_idx,
                            end_idx,
                            self._make_key_by_hash(
                                self._hash_tokens(token_chunk), request_configs
                            ),
                        )
                    else:
                        yield start_idx, end_idx, self._hash_tokens(token_chunk)
                start_idx = end_idx
        elif hashes is not None:
            assert offsets is not None, (
                "If hashes are provided, offsets must also be provided."
            )
            start_idx = 0
            for hash_val, offset in zip(hashes, offsets, strict=False):
                end_idx = start_idx + offset
                if make_key:
                    yield (
                        start_idx,
                        end_idx,
                        self._make_key_by_hash(hash_val, request_configs),
                    )
                else:
                    yield start_idx, end_idx, hash_val
                start_idx = end_idx
        else:
            raise ValueError("Either tokens or hashes must be provided.")
