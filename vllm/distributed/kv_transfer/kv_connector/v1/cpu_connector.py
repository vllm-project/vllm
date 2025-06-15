# SPDX-License-Identifier: Apache-2.0
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.kv_transfer.kv_connector.v1.cpu_connector_utils import (
    DestinationSpec, SourceSpec)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_cpu_utils import (
    NixlDecodeManager, NixlPrefillManager, NixlSendTask)
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.utils import cdiv, round_down
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.config import KVTransferConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import CachedRequestData, NewRequestData
    from vllm.v1.request import Request

logger = init_logger(__name__)


def d2h_page_copy(src_layer: torch.Tensor, dst_buffer: torch.Tensor,
                  block_ids: list[int]) -> None:
    """Copy data from device to host.

    Args:
        src_layer (torch.Tensor): The source layer on device, shape is 
            (2, num_vllm_blocks, page_size, ...remaining dims...) 
        dst_buffer (torch.Tensor): The destination buffer on host, shape is
            (2, len(block_ids), page_size, ...remaining dims...)
        block_ids (list[int]): The list of vllm block ids to copy from.
    """
    block_mapping = torch.stack([
        torch.tensor(block_ids, dtype=torch.long),
        torch.arange(len(block_ids), dtype=torch.long)
    ],
                                dim=1)
    ops.swap_blocks(src_layer[0], dst_buffer[0], block_mapping)
    ops.swap_blocks(src_layer[1], dst_buffer[1], block_mapping)


def h2d_copy_part_block(src_buffer: torch.Tensor, dst_layer: torch.Tensor,
                        src_block_id: int, dst_block_id: int,
                        start_position_in_block: int,
                        end_position_in_block: Optional[int]) -> None:
    """Copy the part of a block from host buffer to device layer.

    Args:
        src_buffer (torch.Tensor): The source buffer on host, shape is 
            (2, len(block_ids), page_size, ...remaining dims...) 
        dst_layer (torch.Tensor): The destination layer on device, shape is
            (2, num_vllm_blocks, page_size, ...remaining dims...)
        src_block_id (int): The source block id to copy.
        dst_block_id (int): The destination block id to copy.
        start_position_in_block (int): The start position in the block to copy.
        end_position_in_block (int): The end position in the block to copy.
    """
    if end_position_in_block is None:
        # If end_position_in_block is None, copy until the end of the block
        end_position_in_block = src_buffer[0][0].shape[0]

    dst_k = dst_layer[0][dst_block_id][
        start_position_in_block:end_position_in_block]
    src_k = src_buffer[0][src_block_id][
        start_position_in_block:end_position_in_block]
    dst_v = dst_layer[1][dst_block_id][
        start_position_in_block:end_position_in_block]
    src_v = src_buffer[1][src_block_id][
        start_position_in_block:end_position_in_block]
    dst_k.copy_(src_k, non_blocking=True)
    dst_v.copy_(src_v, non_blocking=True)


def h2d_copy_leading_tokens(src_buffer: torch.Tensor, dst_layer: torch.Tensor,
                            src_block_id: int, dst_block_id: int,
                            end_position_in_block: int) -> None:
    """Copy the leading tokens in 1 block from host buffer to device layer.

    Args:
        src_buffer (torch.Tensor): The source buffer on host, shape is 
            (2, len(block_ids), page_size, ...remaining dims...) 
        dst_layer (torch.Tensor): The destination layer on device, shape is
            (2, num_vllm_blocks, page_size, ...remaining dims...)
        src_block_id (int): The source block id to copy.
        dst_block_id (int): The destination block id to copy.
        end_position_in_block (int): The end position in the block to copy.
    """
    h2d_copy_part_block(src_buffer, dst_layer, src_block_id, dst_block_id, 0,
                        end_position_in_block)


def h2d_copy_trailing_tokens(src_buffer: torch.Tensor, dst_layer: torch.Tensor,
                             src_block_id: int, dst_block_id: int,
                             start_position_in_block: int) -> None:
    """Copy the trailing tokens in 1 block from host buffer to device layer.

    Args:
        src_buffer (torch.Tensor): The source buffer on host, shape is 
            (2, len(block_ids), page_size, ...remaining dims...) 
        dst_layer (torch.Tensor): The destination layer on device, shape is
            (2, num_vllm_blocks, page_size, ...remaining dims...)
        src_block_id (int): The source block id to copy.
        dst_block_id (int): The destination block id to copy.
        start_position_in_block (int): The start position in the block to copy.
    """
    h2d_copy_part_block(src_buffer, dst_layer, src_block_id, dst_block_id,
                        start_position_in_block, None)


def h2d_page_copy(src_buffer: torch.Tensor, dst_layer: torch.Tensor,
                  block_ids: list[int], start_token_idx: int,
                  stop_token_idx: int, block_size: int) -> None:
    """Copy data from host to device.

    Args:
        src_buffer (torch.Tensor): The source buffer on host, shape is 
            (2, len(block_ids), page_size, ...remaining dims...) 
        dst_layer (torch.Tensor): The destination layer on device, shape is
            (2, num_vllm_pages, page_size, ...remaining dims...)
        block_ids (list[int]): The list of vllm block ids to copy to (for all 
            the tokens)
        start_token_idx (int): The start token index in the request
        stop_token_idx (int): The stop token index in the request
        block_size (int): The block size in vLLM
    """
    # Step 1: build the block mapping (src_block_id, dst_block_id)
    separate_first_block = start_token_idx % block_size != 0
    separate_last_block = stop_token_idx % block_size != 0

    start_block_id = start_token_idx // block_size  # inclusive
    end_block_id = stop_token_idx // block_size  # exclusive
    src_block_ids = torch.arange(start_block_id,
                                 end_block_id,
                                 dtype=torch.long)
    if separate_first_block:
        src_block_ids = src_block_ids[1:]
    # NOTE: we don't need to add the last block id here, because the
    # end_block_id is exclusive
    # E.g., start = 10, stop = 50, block_size = 16, then we have
    #    start_block_id = 0 , separate_first_block = True
    #    end_block_id = 3, separate_last_block = True
    #    src_block_ids = [1, 2]
    # We will copy token 10-15 and 48-49 from the first and last block
    # separately.

    vllm_block_ids = torch.tensor(block_ids, dtype=torch.long)
    dst_block_ids = vllm_block_ids[src_block_ids]

    # Step 2: copy the first and last block separately if needed
    if start_block_id == end_block_id:
        # Only one block to copy
        start_position_in_block = start_token_idx % block_size
        end_position_in_block = stop_token_idx % block_size
        h2d_copy_part_block(src_buffer, dst_layer, start_block_id,
                            vllm_block_ids[start_block_id],
                            start_position_in_block, end_position_in_block)
        return

    if separate_first_block:
        first_block_id_src = start_block_id
        first_block_id_dst = vllm_block_ids[first_block_id_src]
        start_token_idx_in_block = start_token_idx % block_size
        h2d_copy_trailing_tokens(src_buffer, dst_layer, first_block_id_src,
                                 first_block_id_dst, start_token_idx_in_block)

    if separate_last_block:
        last_block_id_src = end_block_id
        last_block_id_dst = vllm_block_ids[last_block_id_src]
        stop_token_idx_in_block = stop_token_idx % block_size
        h2d_copy_leading_tokens(src_buffer, dst_layer, last_block_id_src,
                                last_block_id_dst, stop_token_idx_in_block)

    # Step 3: copy the middle blocks
    block_mapping = torch.stack([src_block_ids, dst_block_ids], dim=1)
    ops.swap_blocks(src_buffer[0], dst_layer[0], block_mapping)
    ops.swap_blocks(src_buffer[1], dst_layer[1], block_mapping)


#####################################################################
# Connector related code
#####################################################################


@dataclass
class PrefillRequestTracker:
    """RequestTracker is used to track the state of a request.

    Attributes:
        req_id (str): The id of the request.
        num_saved_tokens (int): The number of tokens saved.
        num_loaded_tokens (int): The number of tokens loaded.
        num_computed_tokens (int): The number of tokens computed.
        allocated_block_ids (list[int]): The list of allocated block ids.
    """
    # Request id
    req_id: str

    # Block ids that are already allocated for this request
    allocated_block_ids: list[int]

    # Total number of tokens in the "full request"
    num_all_tokens: int = 0

    # Total number of tokens that are already seen until this step
    num_total_tokens: int = 0

    # Number of tokens that are already saved
    num_saved_tokens: int = 0

    @staticmethod
    def from_new_request(
        new_request: "NewRequestData",
        num_tokens_to_compute: int,
    ) -> "PrefillRequestTracker":
        """Create the request tracker from a new request.

        Args:
            new_request (NewRequestData): the new request data.
            num_tokens_to_compute (int): the number of tokens that will 
                be 'computed', including the `num_computed_tokens` (vLLM's
                local cache hit) and new tokens that will be scheduled.
        """
        unfolded_block_ids = []
        for block_ids in new_request.block_ids:
            unfolded_block_ids.extend(block_ids)

        return PrefillRequestTracker(
            req_id=new_request.req_id,
            allocated_block_ids=unfolded_block_ids,
            num_all_tokens=len(new_request.prompt_token_ids),
            num_total_tokens=num_tokens_to_compute,
            num_saved_tokens=0,
        )

    def update(self, cached_request: "CachedRequestData") -> None:
        """Update the request tracker with the cached request data.

        Args:
            cached_request (CachedRequestData): the cached request data.
        """
        new_block_ids = []
        for nb in cached_request.new_block_ids:
            new_block_ids.extend(nb)
        self.allocated_block_ids.extend(new_block_ids)
        self.num_total_tokens += len(cached_request.new_token_ids)

    def update_num_saved_tokens(self, num_saved_tokens: int) -> None:
        """Update the number of saved tokens.

        Args:
            num_saved_tokens (int): the number of saved tokens.
        """
        self.num_saved_tokens = num_saved_tokens


@dataclass
class PrefillReqMeta:
    # Request id
    req_id: str
    # Blocks to save
    blocks_to_save: list[int]
    # The range of tokens to save
    token_range: slice
    # Skip first N tokens
    skip_leading_tokens: int
    # Skip last N tokens
    skip_trailing_tokens: int
    # The number of tokens in the "full request"
    num_all_tokens: int

    @staticmethod
    def from_request_tracker(
        request_tracker: PrefillRequestTracker,
        block_size: int,
    ) -> "PrefillReqMeta":
        """Create the request meta from the request tracker. Determine which 
        blocks to save and the number of leading/trailing tokens to skip for
        the worker connector.
        It also updates the request tracker's num_saved_tokens.

        Args:
            request_tracker (PrefillRequestTracker): the request tracker.
            block_size (int): the block size in vLLM.

        Returns:
            PrefillReqMeta: the request meta.
        """
        assert request_tracker.num_total_tokens <= \
                len(request_tracker.allocated_block_ids) * block_size, \
                f"Request {request_tracker.req_id} has more tokens " + \
                "than allocated blocks"

        token_range = slice(request_tracker.num_saved_tokens,
                            request_tracker.num_total_tokens)

        num_saved_full_blocks = request_tracker.num_saved_tokens // block_size
        num_active_blocks = cdiv(request_tracker.num_total_tokens, block_size)

        blocks_to_save = request_tracker.allocated_block_ids[\
                num_saved_full_blocks:num_active_blocks]
        skip_leading_tokens = request_tracker.num_saved_tokens % block_size
        skip_trailing_tokens = num_active_blocks * block_size - \
                request_tracker.num_total_tokens
        logger.debug(
            "Request %s: num_saved_full_blocks=%d, num_active_blocks=%d, "
            "blocks_to_save=%s, skip_leading_tokens=%d, "
            "skip_trailing_tokens=%d", request_tracker.req_id,
            num_saved_full_blocks, num_active_blocks, blocks_to_save,
            skip_leading_tokens, skip_trailing_tokens)

        # Update the request tracker with the number of saved tokens
        request_tracker.update_num_saved_tokens(
            request_tracker.num_total_tokens)
        return PrefillReqMeta(
            req_id=request_tracker.req_id,
            blocks_to_save=blocks_to_save,
            token_range=token_range,
            skip_leading_tokens=skip_leading_tokens,
            skip_trailing_tokens=skip_trailing_tokens,
            num_all_tokens=request_tracker.num_all_tokens,
        )


@dataclass
class DecodeReqMeta:
    # Request id
    req_id: str
    # Prefiller-side request id
    prefill_req_id: str
    # Allocated block ids
    block_ids: list[int]
    # Skip the first N tokens
    skip_leading_tokens: int
    # if it's ready or not
    is_ready: bool = False


@dataclass
class CPUConnectorMetadata(KVConnectorMetadata):
    prefill_meta: list[PrefillReqMeta]
    decode_meta: list[DecodeReqMeta]

    def __init__(self) -> None:
        super().__init__()
        self.prefill_meta = []
        self.decode_meta = []

    def add_prefill(self, prefill_meta: PrefillReqMeta) -> None:
        """Add a prefill request metadata to the metadata.

        Args:
            prefill_meta (PrefillReqMeta): The prefill request metadata to be
                added.
        """
        self.prefill_meta.append(prefill_meta)

    def add_decode(self, decode_meta: DecodeReqMeta) -> None:
        """Add a decode request metadata to the metadata.

        Args:
            decode_meta (DecodeReqMeta): The decode request metadata to be
                added.
        """
        self.decode_meta.append(decode_meta)


def validate_kv_transfer_config(
        kv_transfer_config: Optional["KVTransferConfig"]) -> None:
    """Validate the KV transfer configuration.
    It expects the host and port configuration in the kv_connector_extra_config

    Args:
        kv_transfer_config (Optional[KVTransferConfig]): The KV transfer
            configuration to validate.

    Raises:
        AssertionError: If the configuration is invalid.
    """
    assert kv_transfer_config is not None, \
        "KV transfer config is not set in the vLLM config"

    extra_config = kv_transfer_config.kv_connector_extra_config
    assert "host" in extra_config, \
            "CPUConnector: must have 'host' in kv_connector_extra_config"
    assert "port" in extra_config, \
            "CPUConnector: must have 'port' in kv_connector_extra_config"
    assert "size" in extra_config, \
            "CPUConnector: must have 'size' in kv_connector_extra_config"


class CPUConnector(KVConnectorBase_V1):
    """CPUKVConnector is an implementation of KVConnectorBase_V1 that
    provides a CPU-based KV cache sending mechanism.
    """

    def __init__(self, vllm_config: "VllmConfig",
                 role: KVConnectorRole) -> None:
        super().__init__(vllm_config, role)

        validate_kv_transfer_config(vllm_config.kv_transfer_config)
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        self._host = extra_config["host"]
        self._port = int(extra_config["port"])
        # Convert GB to bytes and align to 4K for storage size
        kv_size_in_bytes = float(extra_config["size"]) * (1 << 30)
        kv_size_in_bytes = int(kv_size_in_bytes) & (~0xFFF)  # Align to 4K
        self._kv_size = kv_size_in_bytes

        self.kv_role = vllm_config.kv_transfer_config.kv_role

        self._block_size = vllm_config.cache_config.block_size

        if role == KVConnectorRole.SCHEDULER:
            self._should_be_ready_reqs: set[str] = set()
        elif role == KVConnectorRole.WORKER:
            # Prefiller side sender
            if self.kv_role == "kv_producer":
                self._kv_sender = NixlPrefillManager(self._kv_size)
                self._kv_sender_lock = threading.Lock()
                self._kv_sender_stop_event = threading.Event()
                self._kv_sender_thread = threading.Thread(
                    target=self._kv_sender_processor,
                    daemon=True,
                )
                self._kv_sender_thread.start()

            elif self.kv_role == "kv_consumer":
                self._kv_receiver = NixlDecodeManager(
                    self._kv_size,
                    self._host,
                    self._port,
                )
            else:
                raise ValueError(f"Unknown kv_role: {self.kv_role}")

        # request_id -> prefill request trackers
        self._prefill_reqs: dict[str, PrefillRequestTracker] = {}

        # gpu kv caches
        self._gpu_kv_caches: dict[str, torch.Tensor] = {}
        self._layer_name_to_id: dict[str, int] = {}
        self._layer_id_to_name: dict[int, str] = {}
        self._kv_page_shape: torch.Size = torch.Size([0])

        # separate cuda streams
        self._cuda_stream = torch.cuda.Stream()

        # prefill offload tasks
        self._inflight_copy_tasks: list[NixlSendTask] = []

        # Decode request id to prefill request id mapping
        self._decode_req_id_to_prefill_req_id: dict[str, str] = {}
        self._prefill_req_id_to_decode_req_id: dict[str, str] = {}

        # Decode request metadata for scheduler connector
        # decode request id -> DecodeReqMeta
        self._decode_req_metas: dict[str, DecodeReqMeta] = {}

        # Decode h2d cuda events
        # layer id -> cuda event
        self._decoder_cuda_events: dict[int, torch.cuda.Event] = {}

        # In-progress kv load requests's prefill request ids
        self._inflight_h2d_requests: set[str] = set()

    def _connect_request_ids(self, p_reqid: str, d_reqid: str) -> None:
        self._decode_req_id_to_prefill_req_id[d_reqid] = p_reqid
        self._prefill_req_id_to_decode_req_id[p_reqid] = d_reqid

    ############################################################
    # Scheduler Side Methods
    ############################################################
    def _build_prefiller_meta(self, scheduler_output: SchedulerOutput,
                              output_meta: CPUConnectorMetadata) -> None:
        """Build the prefill request metadata from the scheduler output.

        Args:
            scheduler_output (SchedulerOutput): The scheduler output.
            output_meta (CPUConnectorMetadata): The output metadata.
        """
        for finished_req_id in scheduler_output.finished_req_ids:
            self._prefill_reqs.pop(finished_req_id, None)

        for request in scheduler_output.scheduled_new_reqs:
            num_tokens_to_compute = request.num_computed_tokens + \
                    scheduler_output.num_scheduled_tokens[request.req_id]
            request_tracker = PrefillRequestTracker.from_new_request(
                request, num_tokens_to_compute)
            self._prefill_reqs[request.req_id] = request_tracker

            req_meta = PrefillReqMeta.from_request_tracker(
                request_tracker, self._block_size)
            output_meta.add_prefill(req_meta)

        for request in scheduler_output.scheduled_cached_reqs:
            request_tracker = self._prefill_reqs[request.req_id]
            request_tracker.update(request)

            req_meta = PrefillReqMeta.from_request_tracker(
                request_tracker, self._block_size)
            output_meta.add_prefill(req_meta)

    def build_decode_meta(self, scheduler_output: SchedulerOutput,
                          output_meta: CPUConnectorMetadata) -> None:
        """Build the decode request metadata from the scheduler output.

        Args:
            scheduler_output (SchedulerOutput): The scheduler output.
            output_meta (CPUConnectorMetadata): The output metadata.
        """
        updated_decode_req_metas = {}
        for req_meta in self._decode_req_metas.values():
            if not req_meta.is_ready:
                updated_decode_req_metas[req_meta.req_id] = req_meta
            # NOTE (ApostaC): Even if the request is not ready, we still
            # want the worker connector to know about it, so that it can
            # connect the decode request id to the prefill request id
            output_meta.add_decode(req_meta)
        self._decode_req_metas = updated_decode_req_metas

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        # NOTE(ApostaC): For a single request, this function will be called
        # two times if the first time we returned async_load flag as True.
        # The second time will be the "real schedule" time

        if self.kv_role == "kv_producer":
            return 0, False

        kv_transfer_params = request.kv_transfer_params
        num_tokens = len(request.prompt_token_ids)
        request_id = request.request_id
        logger.info(
            "For request %s, num_computed_tokens is %d, "
            "total_num_tokens is %d", request_id, num_computed_tokens,
            num_tokens)

        num_extra_tokens = round_down(num_tokens,
                                      self._block_size) - num_computed_tokens

        if num_extra_tokens < self._block_size:
            # If the request is smaller than the block size, we don't need
            # to do anything special
            logger.info(
                "Request %s is smaller than block size %d, "
                "no async loading", request_id, self._block_size)
            return 0, False

        # Seen this request before, which means it should be ready this time,
        # so we don't need to do async loading again
        if request.request_id in self._should_be_ready_reqs:
            self._should_be_ready_reqs.remove(request.request_id)
            return 0, False

        if kv_transfer_params is None or \
                "prefill_request_id" not in kv_transfer_params:
            logger.warning("Request %s does not have prefill_request_id",
                           request.request_id)
            return 0, False

        prefill_request_id = kv_transfer_params["prefill_request_id"]
        self._connect_request_ids(prefill_request_id, request_id)
        self._should_be_ready_reqs.add(request_id)

        # NOTE: because the scheduler wants here to return "full blocks" if
        # the async flag is true (see _update_waiting_for_remote_kv in
        # scheduler.py). We need to carefully deal with it when copying
        # the KV cache at worker side
        return num_extra_tokens, True

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int) -> None:
        """Update the state of the request after allocation.
        """
        # NOTE(ApostaC): This function is called twice for the same request
        # when we are using async loading. The first time is we got all the
        # external "hit" blocks in `blocks`, and the second time we will have
        # the remaining "last" block as a newly allocated block.
        if self.kv_role == "kv_producer":
            return

        if request.request_id in self._decode_req_metas:
            # This is the second time we are called for the same request
            # We need to mark the request as "ready"
            self._decode_req_metas[request.request_id].is_ready = True
            return

        if request.request_id not in self._decode_req_id_to_prefill_req_id:
            # This should not happen, but just in case
            logger.warning(
                "Request %s does not have prefill request id, "
                "skipping decode meta creation", request.request_id)
            return

        p_req_id = self._decode_req_id_to_prefill_req_id[request.request_id]
        block_ids = []
        for blks in blocks.get_block_ids():
            block_ids.extend(blks)
        req_meta = DecodeReqMeta(req_id=request.request_id,
                                 prefill_req_id=p_req_id,
                                 block_ids=block_ids,
                                 skip_leading_tokens=0,
                                 is_ready=False)
        self._decode_req_metas[request.request_id] = req_meta

    def build_connector_meta(
            self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        meta = CPUConnectorMetadata()

        if self.kv_role == "kv_producer":
            self._build_prefiller_meta(scheduler_output, meta)
        elif self.kv_role == "kv_consumer":
            self.build_decode_meta(scheduler_output, meta)
        else:
            raise ValueError(f"Unknown kv_role: {self.kv_role}")

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        if self.kv_role == "kv_consumer":
            return False, None
        # For prefiller, send back the prefiller request id
        logger.info("Prefill request %s finished", request.request_id)
        return False, dict(prefill_request_id=request.request_id)

    #############################################################
    # Worker Side Methods
    #############################################################
    def _kv_sender_processor(self) -> None:
        """Process the KV sender tasks in a separate thread."""
        while not self._kv_sender_stop_event.is_set():
            with self._kv_sender_lock:
                self._kv_sender.progress()
            time.sleep(0.001)  # Sleep for a short time to avoid busy waiting

    def _get_layer_id(self, layer_name: str) -> int:
        assert layer_name in self._layer_name_to_id, \
                f"Layer {layer_name} not found in layer name to id map"
        return self._layer_name_to_id[layer_name]

    def _get_layer_name(self, layer_id: int) -> str:
        assert layer_id in self._layer_id_to_name, \
                f"Layer id {layer_id} not found in layer id to name map"
        return self._layer_id_to_name[layer_id]

    def _get_kv_shape(self, num_blocks: int) -> torch.Size:
        return torch.Size((
            2,
            num_blocks,
        ) + self._kv_page_shape)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self._gpu_kv_caches = kv_caches
        for idx, layer_name in enumerate(kv_caches):
            self._layer_name_to_id[layer_name] = idx
            self._layer_id_to_name[idx] = layer_name

        self._kv_page_shape = kv_caches[list(kv_caches.keys())[0]].shape[2:]

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be 
            the same.
            
        """
        if self.kv_role == "kv_producer":
            # encoder side
            return

        meta = self._get_connector_metadata()
        assert isinstance(meta, CPUConnectorMetadata), \
                "Connector metadata is not of type CPUConnectorMetadata"

        ready_decode_metas = []
        total_expected_tokens = []
        for decode_meta in meta.decode_meta:
            self._connect_request_ids(decode_meta.prefill_req_id,
                                      decode_meta.req_id)
            if decode_meta.is_ready:
                ready_decode_metas.append(decode_meta)
                total_expected_tokens.append(
                        len(decode_meta.block_ids) * \
                        self._block_size)
                self._inflight_h2d_requests.add(decode_meta.prefill_req_id)

        # Vars needed:
        #   decode_meta.prefill_req_id
        if len(ready_decode_metas) == 0:
            return

        for layer_id in range(len(self._gpu_kv_caches)):
            for decode_meta, total_expected in zip(ready_decode_metas,
                                                   total_expected_tokens):
                decode_specs = self._kv_receiver.get_kv_specs(
                    decode_meta.prefill_req_id, layer_id)
                layer_name = self._layer_id_to_name[layer_id]
                dst_layer = self._gpu_kv_caches[layer_name]
                for decode_spec in decode_specs:
                    start = decode_spec.start
                    stop = min(decode_spec.stop, total_expected)
                    if start >= total_expected:
                        continue
                    src_buffer = decode_spec.buffer
                    block_ids = decode_meta.block_ids

                    with torch.cuda.stream(self._cuda_stream):
                        h2d_page_copy(src_buffer, dst_layer, block_ids, start,
                                      stop, self._block_size)

            # Record the cuda event for this layer
            event = torch.cuda.Event()
            event.record(self._cuda_stream)
            self._decoder_cuda_events[layer_id] = event

        # TODO (ApostaC): Potential optimizations
        # 1. coalesce the h2d page copy to a single call
        # 2. Don't launch all the layers, but just first 2 layers
        # 2.1 launch the rest of the layers during the `wait_for_layer_load`

    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.
        
        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        if self.kv_role == "kv_producer":
            # encoder side
            return

        layer_id = self._get_layer_id(layer_name)
        event = self._decoder_cuda_events.pop(layer_id, None)
        if event is not None:
            event.synchronize()

        if layer_id == len(self._gpu_kv_caches) - 1:
            # Free the memory for the whole request
            for p_req_id in self._inflight_h2d_requests:
                logger.info("Freeing request %s, current watermark: [%d, %d]",
                            p_req_id,
                            self._kv_receiver._allocator.low_watermark,
                            self._kv_receiver._allocator.high_watermark)
                self._kv_receiver.free_request(p_req_id)
            self._inflight_h2d_requests.clear()

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """
        Start saving a layer of KV cache from vLLM's paged buffer 
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current 
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        if self.kv_role == "kv_consumer":
            # decoder side
            return

        meta = self._get_connector_metadata()
        assert isinstance(meta, CPUConnectorMetadata), \
                "Connector metadata is not of type CPUConnectorMetadata"
        assert self._kv_sender is not None

        for prefill_req in meta.prefill_meta:
            # Create a source spec with serializable types
            source_spec = SourceSpec(
                request_id=prefill_req.req_id,
                layer_id=self._get_layer_id(layer_name),
                start=prefill_req.token_range.start,
                stop=prefill_req.token_range.stop,
                shape=tuple(self._get_kv_shape(len(
                    prefill_req.blocks_to_save))),
                dtype_str=str(kv_layer.dtype).split('.')
                [-1],  # Convert torch.float32 -> "float32"
                num_all_tokens=prefill_req.num_all_tokens,
            )

            # Create a destination spec
            dest_spec = DestinationSpec(
                rank=get_tensor_model_parallel_rank(),
                host=self._host,
                base_port=self._port,
            )

            # Create the send task
            with self._kv_sender_lock:
                task = self._kv_sender.create_send_task(
                    source_spec=source_spec,
                    destination_spec=dest_spec,
                )
            assert isinstance(task, NixlSendTask), \
                    "Send task is not of type NixlSendTask"

            # Start copying the data to the CPU buffer
            buffer = task.tensor
            self._cuda_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._cuda_stream):
                # Copy the data from the GPU to the CPU buffer page by page
                d2h_page_copy(src_layer=kv_layer,
                              dst_buffer=buffer,
                              block_ids=prefill_req.blocks_to_save)

            # record the cuda stream
            task.cuda_event = torch.cuda.Event()
            task.cuda_event.record(self._cuda_stream)

            self._inflight_copy_tasks.append(task)

        # TODO(ApostaC): Potential optimizations
        # 1. coalesce the d2h page copy to a single call
        # 2. use a single cuda event instead of a list of cuda events
        # 3. use a cuda event pool to prevent the creation overhead

    def wait_for_save(self):
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        if self.kv_role == "kv_consumer":
            # decoder side
            return

        # Check the task states and send the tasks
        for task in self._inflight_copy_tasks:
            if task.cuda_event is not None:
                task.cuda_event.synchronize()
        #self._kv_sender.progress()
        self._inflight_copy_tasks.clear()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer,
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """
        if self.kv_role != "kv_consumer":
            return None, None

        # decoder (kv_consumer) side
        self._kv_receiver.progress()
        p_ready_reqs = self._kv_receiver.get_finished(len(self._gpu_kv_caches))
        ret = set()
        for p_req_id in p_ready_reqs:
            if d_req_id := self._decode_req_id_to_prefill_req_id.get(p_req_id):
                # We have seen the corresponding decode request before.
                # Therefore, we can return the request id.
                ret.add(d_req_id)
            else:
                # We haven't seen the corresponding decode request
                # before. Therefore, we should make the receiver
                # to return the request id again in the next
                # call to get_finished.
                self._kv_receiver.remove_ready_request(p_req_id)

        if ret:
            logger.info("Got finished requests: %s", ret)

        return None, ret

    def close(self):
        """
        Block until all the transfers are done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        if hasattr(self, "_kv_sender") and self._kv_sender is not None:
            self._kv_sender_stop_event.set()
            if hasattr(self, "_kv_sender_thread") and \
                    self._kv_sender_thread is not None:
                self._kv_sender_thread.join()
            self._kv_sender.close()

        if hasattr(self, "_kv_receiver") and self._kv_receiver is not None:
            self._kv_receiver.close()
