# SPDX-License-Identifier: Apache-2.0
# Adpoted from LMCache https://github.com/LMCache/LMCache

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.logger import init_logger
from vllm.utils import cdiv, make_zmq_socket
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.output import CachedRequestData, NewRequestData
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class LoadSpec:
    # Number of tokens cached in vLLM
    vllm_cached_tokens: int
    # Number of tokens that are cached in LMCache
    external_cached_tokens: int
    # Whether the scheduler allow us to load the tokens
    can_load: bool


@dataclass
class SaveSpec:
    # Skip already saved tokens
    skip_leading_tokens: int
    # Whether the scheduler allow us to save the tokens
    can_save: bool


@dataclass
class RequestTracker:
    # Request id
    req_id: str

    # The token ids that has been scheduled so far
    token_ids: list[int]

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
        # vLLM 0.9.0 update: request.block_ids changed from list[int] to
        # list[list[int]]
        # Need to check the type of request.block_ids

        unfolded_block_ids = []

        if not isinstance(new_request.block_ids[0], list):
            unfolded_block_ids = new_request.block_ids.copy()
        else:
            # According to the vLLM code
            # (https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/
            # sched/scheduler.py#L943),
            # only one KVCacheGroup is supported in connector for now.

            # TODO: Please support multiple KVCacheGroup in connector.
            # NOTE: Also, `update` method in RequestTracker should be
            # updated accordingly.
            unfolded_block_ids = new_request.block_ids[0].copy()

        return RequestTracker(
            req_id=new_request.req_id,
            token_ids=new_request.prompt_token_ids[:num_tokens_to_compute].
            copy(),
            allocated_block_ids=unfolded_block_ids,
            num_saved_tokens=0,
        )

    def update(
        self,
        cached_request: "CachedRequestData",
    ) -> None:
        """Update the request tracker when a running request is 
        scheduled again
        """
        self.token_ids.extend(cached_request.new_token_ids)
        new_block_ids: list[int]
        if not isinstance(cached_request.new_block_ids[0], list):
            new_block_ids = cached_request.new_block_ids
        else:
            new_block_ids = cached_request.new_block_ids[0]
        self.allocated_block_ids.extend(new_block_ids)


@dataclass
class ReqMeta:
    # Request id
    req_id: str
    # Request tokens
    token_ids: torch.Tensor
    # Block ids
    block_ids: torch.Tensor
    # Slot mapping
    slot_mapping: torch.Tensor
    # Skip save or not
    save_spec: Optional[SaveSpec] = None
    # load_spec
    load_spec: Optional[LoadSpec] = None

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        block_size: int,
        load_spec: Optional[LoadSpec] = None,
        skip_save: bool = False,
    ) -> Optional["ReqMeta"]:
        """Create the request metadata from a request tracker.

        Args:
            tracker (RequestTracker): the request tracker.
            block_size (int): the block size in vLLM.
            load_spec (Optional[LoadSpec]): the load spec for KV cache loading.
            skip_save (bool): whether to skip the save operation.

        Returns:
            the request metadata if we need to perform load/save 
            operations, None otherwise.
        """
        input_token_ids = tracker.token_ids
        input_token_len = len(input_token_ids)

        if skip_save and load_spec is None:
            return None

        num_tokens_to_save = input_token_len 
        skip_leading_tokens = tracker.num_saved_tokens

        # If we need to save, update the number of saved tokens
        if not skip_save:
            tracker.num_saved_tokens = num_tokens_to_save
        save_spec = SaveSpec(skip_leading_tokens, not skip_save)

        # Calculate the token ids and slot mappings for load and save
        # OPTIMIZATION: pre-allocate the buffer for token ids and block
        # ids
        token_ids = torch.tensor(input_token_ids)[:num_tokens_to_save]
        num_blocks = len(tracker.allocated_block_ids)
        block_ids = torch.tensor(tracker.allocated_block_ids, dtype=torch.long)

        if len(token_ids) > num_blocks * block_size:
            logger.error(
                "The number of tokens is more than the number of blocks."
                "Something might be wrong in scheduling logic!")
            logger.error("Num tokens: %d, num blocks: %d, block size: %d",
                         len(token_ids), num_blocks, block_size)

        block_offsets = torch.arange(0, block_size, dtype=torch.long)
        slot_mapping = block_offsets.reshape((1, block_size)) + \
                block_ids.reshape((num_blocks, 1)) * block_size

        slot_mapping = slot_mapping.flatten()[:len(token_ids)]
        assert slot_mapping.dtype == torch.long  # TODO: this could be removed

        # For load operation: check whether the request is scheduled to load
        if load_spec is not None and load_spec.can_load:
            logger.debug("Scheduled to load %d tokens for request %s",
                         load_spec.external_cached_tokens, tracker.req_id)
        else:
            # Do not load if not in `can_load` state
            load_spec = None

        return ReqMeta(
            req_id=tracker.req_id,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            save_spec=save_spec,
            load_spec=load_spec,
        )


@dataclass
class GeneralKVConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta]

    def __init__(self):
        self.requests = []

    def add_request(self, req_meta: ReqMeta) -> None:
        """Add a request to the metadata.

        Args:
            req_meta (ReqMeta): the request metadata.
        """
        self.requests.append(req_meta)


