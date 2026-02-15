from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.output import NewRequestData


# Parameters related to the key
@dataclass
class KeyMetadata:
    """name of the LLM model"""

    model_name: str
    """ worker id when running under a distributed setting """
    head_or_tp_rank: int
    """ Initialize the current prefill context model parallel rank """
    pcp_rank: int
    """ Initialize the current decode context model parallel rank """
    dcp_rank: int
    """ Initialize the current pipeline parallel rank """
    pp_rank: int


@dataclass(order=True)
class PoolKey:
    key_metadata: KeyMetadata
    chunk_hash: str

    def __hash__(self):
        return hash(
            (
                self.key_metadata.model_name,
                self.key_metadata.head_or_tp_rank,
                self.key_metadata.pcp_rank,
                self.key_metadata.dcp_rank,
                self.key_metadata.pp_rank,
                self.chunk_hash,
            )
        )

    def to_string(self):
        return (
            f"{self.key_metadata.model_name}"
            f"@pcp{self.key_metadata.pcp_rank}@dcp{self.key_metadata.dcp_rank}"
            f"@head_or_tp_rank:{self.key_metadata.head_or_tp_rank}"
            f"@pp_rank:{self.key_metadata.pp_rank}@{self.chunk_hash}"
        )

    def split_layers(self, num_layers: int) -> list["LayerPoolKey"]:
        """Split the key into multiple keys for each layer"""
        keys = []
        for layer_id in range(num_layers):
            keys.append(
                LayerPoolKey(
                    self.key_metadata,
                    self.chunk_hash,
                    layer_id,
                )
            )
        return keys


@dataclass(order=True)
class LayerPoolKey(PoolKey):
    """A key for the layer cache engine"""

    layer_id: int

    def __hash__(self):
        return hash(
            (
                self.key_metadata.model_name,
                self.key_metadata.head_or_tp_rank,
                self.key_metadata.pcp_rank,
                self.key_metadata.dcp_rank,
                self.chunk_hash,
                self.layer_id,
            )
        )

    def to_string(self):
        return (
            f"{self.key_metadata.model_name}"
            f"@pcp{self.key_metadata.pcp_rank}@dcp{self.key_metadata.dcp_rank}"
            f"@head_or_tp_rank:{self.key_metadata.head_or_tp_rank}@{self.chunk_hash}@{self.layer_id}"
        )


class ChunkedTokenDatabase:
    def __init__(self, metadata: KeyMetadata, block_size: int, use_mla: bool, partitions: list[int] | None):
        self.metadata = metadata
        self.block_size = block_size
        self.use_mla = use_mla
        self.kv_caches_base_addr: list[int] = []
        self.block_len: list[int] = []
        self.partitions = partitions

    def _make_key_by_hash(self, chunk_hash: str, layer_id: int | None = None):
        assert self.metadata is not None
        return PoolKey(
            self.metadata,
            chunk_hash,
        )

    def set_kv_caches_base_addr(self, kv_caches_base_addr: list[int]):
        self.kv_caches_base_addr = kv_caches_base_addr

    def set_block_len(self, block_len: list[int]):
        self.block_len = block_len

    def prepare_value(self, start: int, end: int, block_ids: list[int]):
        addr_list = []
        size_list = []
        block_id = block_ids[start // self.block_size]
        for index, base_addr in enumerate(self.kv_caches_base_addr):
            block_len = self.block_len[index % 2] if self.use_mla else self.block_len[0]

            addr = base_addr + block_id * block_len
            length = int(block_len / self.block_size * (end - start))
            addr_list.append(addr)
            size_list.append(length)
        return addr_list, size_list, block_id

    def prepare_value_layer(self, start: int, end: int, block_ids: list[int], layer_id: int):
        block_id = block_ids[start // self.block_size]
        if self.use_mla:
            addr_k = self.kv_caches_base_addr[layer_id * 2] + block_id * self.block_len[0]
            addr_v = self.kv_caches_base_addr[layer_id * 2 + 1] + block_id * self.block_len[1]
            length_k = int(self.block_len[0] / self.block_size * (end - start))
            length_v = int(self.block_len[1] / self.block_size * (end - start))
            size_list = [length_k, length_v]
        else:
            addr_k = self.kv_caches_base_addr[layer_id * 2] + block_id * self.block_len[0]
            addr_v = self.kv_caches_base_addr[layer_id * 2 + 1] + block_id * self.block_len[0]
            length = int(self.block_len[0] / self.block_size * (end - start))
            size_list = [length, length]
        addr_list = [addr_k, addr_v]
        return addr_list, size_list

    def process_tokens(
        self,
        token_len: int,
        block_hashes: list[BlockHash] | list[str],
        mask_num: int = 0,
    ) -> Iterable[tuple[int, int, PoolKey]]:
        """Process the tokens and return the corresponding cache engine keys.

        :param Union[torch.Tensor, List[int]] tokens: The tokens to process.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param bool make_key: Whether to make the cache engine key or not.
            If False, the hash value will be returned instead.

        :returns: A iterable of tuples with three elements. The first element
            is the start index of the tokens for the key. The second element
            is the end index of the tokens for the key. The third element is
            the cache engine key (or hash) for the tokens.

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        if not block_hashes:
            return
        if not isinstance(block_hashes[0], str):
            block_hashes = [
                h.hex()  # type: ignore[union-attr]
                for h in block_hashes
            ]
        start_idx = 0
        for chunk_id, hash_val in enumerate(block_hashes):
            start_idx = chunk_id * self.block_size
            if start_idx >= token_len:
                break
            end_idx = min(start_idx + self.block_size, token_len)
            if start_idx < mask_num:
                continue
            else:
                yield start_idx, end_idx, self._make_key_by_hash(hash_val)

    def decode_adaptor_prefill_pp(self, key, addr, size):
        if self.partitions is None or len(self.partitions) == 1:
            return key, addr, size

        new_key = []
        new_addr = []
        new_size = []

        for i, (addr_list, size_list) in enumerate(zip(addr, size)):
            start = 0
            for j, part in enumerate(self.partitions):
                # part * 2 because addr and size contain both k and v
                end = len(addr_list) if j == len(self.partitions) - 1 else start + part * 2
                new_str = key[i].replace(  # type: ignore[attr-defined]
                    "@pp_rank:0", f"@pp_rank:{j}", 1
                )
                new_key.append(new_str)
                new_addr.append(addr_list[start:end])
                new_size.append(size_list[start:end])
                start = end
        return new_key, new_addr, new_size


# Parameters related to the connector metadata
@dataclass
class LoadSpec:
    # Number of tokens cached in vLLM
    vllm_cached_tokens: int
    # Number of tokens that are cached in kvpool
    kvpool_cached_tokens: int
    # Whether the scheduler allow us to load the tokens
    can_load: bool

    token_len: int = 0


@dataclass
class RequestTracker:
    # Request id
    req_id: str

    # The token ids that has been scheduled so far
    token_len: int

    # The block ids that has been allocated so far
    # NOTE: allocated blocks could be more than the number of tokens
    # FIXME: need to check whether the block ids will be changed after
    #        preemption
    allocated_block_ids: list[int]

    # The number of tokens that has been savd
    num_saved_tokens: int = 0

    @staticmethod
    def from_new_request(
        new_request: "NewRequestData",
        num_tokens_to_compute: int,
    ) -> "RequestTracker":
        """Create the request tracker from a new request.

        Args:
            new_request (NewRequestData): the new request data.
            num_tokens_to_compute (int): the number of tokens that will
                be 'computed', including the `num_computed_tokens` (vLLM's
                local cache hit) and new tokens that will be scheduled.

        """
        unfolded_block_ids = []

        if not isinstance(new_request.block_ids[0], list):
            unfolded_block_ids = new_request.block_ids.copy()
        else:
            unfolded_block_ids = new_request.block_ids[0].copy()

        return RequestTracker(
            req_id=new_request.req_id,
            token_len=num_tokens_to_compute,
            allocated_block_ids=unfolded_block_ids,
            num_saved_tokens=0,
        )

    def update(
        self,
        new_block_ids: tuple[list[int], ...] | list[int],
    ) -> None:
        """Update the request tracker when a running request is
        scheduled again
        """

        if len(new_block_ids) == 0:
            new_block_ids = []
        elif isinstance(new_block_ids, tuple):
            new_block_ids = new_block_ids[0]
        elif isinstance(new_block_ids, list):
            pass
        else:
            raise ValueError(f"Unsupported new_block_ids type {type(new_block_ids)}")
        self.allocated_block_ids.extend(new_block_ids)


@dataclass
class ReqMeta:
    # Request id
    req_id: str
    # Request tokens
    token_len_chunk: int

    block_ids: list[int]

    block_hashes: list[BlockHash]

    can_save: bool | None = None
    # load_spec
    load_spec: LoadSpec | None = None

    is_last_chunk: bool | None = None

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        block_size: int,
        load_spec: LoadSpec | None = None,
        skip_save: bool | None = False,
        block_hashes: list[BlockHash] | None = None,
        is_last_chunk: bool | None = None,
        discard_partial_chunks: bool = True,
    ) -> Optional["ReqMeta"]:
        """Create the request metadata from a request tracker.

        Args:
            tracker (RequestTracker): the request tracker.
            block_size (int): the block size in vLLM.
            load_spec (Optional[LoadSpec]): the load spec for KV cache loading.
            skip_save (bool): whether to skip the save operation.
            discard_partial_chunks (bool): whether to discard partial chunks.

        Returns:
            the request metadata if we need to perform load/save
            operations, None otherwise.
        """
        if block_hashes is None:
            block_hashes = []
        input_token_len = tracker.token_len

        # For save operation: do not save if the following condition is met
        # 1. has already been saved before (num_saved_tokens > 0)
        # 2. number of unsaved tokens is not reached the chunk boundary
        chunk_boundary = cdiv(tracker.num_saved_tokens + 1, block_size) * block_size if discard_partial_chunks else 0
        # Calculate number of tokens to save based on discard_partial_chunks
        # setting
        num_tokens_to_save = (input_token_len // block_size * block_size) if discard_partial_chunks else input_token_len

        skip_save = skip_save or num_tokens_to_save < chunk_boundary
        if skip_save and load_spec is None:
            return None

        # If we need to save, update the number of saved tokens
        if not skip_save:
            tracker.num_saved_tokens = num_tokens_to_save

        # # For load operation: check whether the request is scheduled to load
        if load_spec is not None and load_spec.can_load:
            logger.debug(
                "Scheduled to load %d tokens for request %s",
                load_spec.kvpool_cached_tokens,
                tracker.req_id,
            )
        else:
            # Do not load if not in `can_load` state
            load_spec = None
        logger.debug(f"request:{tracker.req_id}, meta save spec:{not skip_save}, meta load spec:{load_spec}")
        return ReqMeta(
            req_id=tracker.req_id,
            token_len_chunk=num_tokens_to_save,
            block_ids=tracker.allocated_block_ids,
            can_save=not skip_save,
            load_spec=load_spec,
            block_hashes=block_hashes,
            is_last_chunk=is_last_chunk,
        )


class MooncakeStoreConnectorMetadata(KVConnectorMetadata):
    def __init__(self, unfinished_request_ids, preempted_req_ids):
        self.requests = []
        self.unfinished_request_ids = unfinished_request_ids
        self.preempted_req_ids = preempted_req_ids

    def add_request(self, req_meta: ReqMeta) -> None:
        """Add a request to the metadata.

        Args:
            req_meta (ReqMeta): the request metadata.
        """
        self.requests.append(req_meta)


@dataclass
class LayerMultiBlockReqMeta:
    req_id: str
    keys: list[LayerPoolKey]
    starts: list[int]
    ends: list[int]
    block_ids: list[int]
    layer_id: int
    is_last_chunk: bool | None = True
   # current_event: torch.npu.Event | None = None
