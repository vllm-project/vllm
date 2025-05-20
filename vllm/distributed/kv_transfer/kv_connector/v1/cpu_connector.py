# SPDX-License-Identifier: Apache-2.0
import contextlib
import math
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple

import msgspec
import torch
import zmq

from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.kv_transfer.kv_connector.v1.cpu_connector_utils import (
    CPUSendTask, CPUKVSender, SourceSpec, DestinationSpec)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size,
    get_tp_group)
from vllm.logger import init_logger
from vllm.utils import make_zmq_path, make_zmq_socket, round_down, cdiv
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus
from vllm import _custom_ops as ops

from lmcache.utils import _lmcache_nvtx_annotate

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import CachedRequestData, NewRequestData
    from vllm.v1.request import Request

logger = init_logger(__name__)

def d2h_page_copy(
        src_layer: torch.Tensor,
        dst_buffer: torch.Tensor,
        block_ids: list[int]
    ) -> None:
    """Copy data from device to host.

    Args:
        src_layer (torch.Tensor): The source layer on device, shape is 
            (2, num_vllm_blocks, page_size, ...remaining dims...) 
        dst_buffer (torch.Tensor): The destination buffer on host, shape is
            (2, len(block_ids), page_size, ...remaining dims...)
    """
    # debug copy:
    block_mapping = torch.stack([torch.tensor(block_ids, dtype = torch.long), 
                                 torch.arange(len(block_ids), dtype = torch.long)], dim = 1)
    ops.swap_blocks(src_layer[0], dst_buffer[0], block_mapping)
    ops.swap_blocks(src_layer[1], dst_buffer[1], block_mapping)
    #for dst_idx, block_id in enumerate(block_ids):
    #    src_k, src_v = src_layer[:, block_id, :, :]
    #    dst_k, dst_v = dst_buffer[:, dst_idx, :, :]
    #    # Copy the data from device to host
    #    dst_k.copy_(src_k, non_blocking=True)
    #    dst_v.copy_(src_v, non_blocking=True)

def h2d_page_copy(
        src_buffer: torch.Tensor,
        dst_layer: torch.Tensor,
        block_ids: list[int]
    ) -> None:
    """Copy data from host to device.

    Args:
        src_buffer (torch.Tensor): The source buffer on host, shape is 
            (2, len(block_ids), page_size, ...remaining dims...) 
        dst_layer (torch.Tensor): The destination layer on device, shape is
            (2, num_vllm_pages, page_size, ...remaining dims...)
    """
    for src_idx, block_id in enumerate(block_ids):
        dst_k, dst_v = dst_layer[:, block_id, :, :]
        src_k, src_v = src_buffer[:, src_idx, :, :]
        # Copy the data from host to device
        dst_k.copy_(src_k, non_blocking=True)
        dst_v.copy_(src_v, non_blocking=True)


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

    # Total number of tokens that are in this request
    num_total_tokens: int = 0

    # Number of tokens that are already saved
    num_saved_tokens: int = 0

    # Block ids that are already allocated for this request
    allocated_block_ids: list[int] = None

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
            num_total_tokens = num_tokens_to_compute,
            num_saved_tokens=0,
            allocated_block_ids=unfolded_block_ids,
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
                f"Request {req_id} has more tokens than allocated blocks"

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
            "skip_trailing_tokens=%d", 
            request_tracker.req_id,
            num_saved_full_blocks, num_active_blocks,
            blocks_to_save, skip_leading_tokens, skip_trailing_tokens)

        # Update the request tracker with the number of saved tokens
        request_tracker.update_num_saved_tokens(
            request_tracker.num_total_tokens)
        return PrefillReqMeta(
            req_id=request_tracker.req_id,
            blocks_to_save=blocks_to_save,
            token_range=token_range,
            skip_leading_tokens=skip_leading_tokens,
            skip_trailing_tokens=skip_trailing_tokens,
        )


@dataclass
class DecodeReqMeta:
    pass

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


class CPUConnector(KVConnectorBase_V1):
    """CPUKVConnector is an implementation of KVConnectorBase_V1 that
    provides a CPU-based KV cache sending mechanism.
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole) -> None:
        super().__init__(vllm_config, role)

        self.kv_role = vllm_config.kv_transfer_config.kv_role

        self._block_size = vllm_config.cache_config.block_size

        if role == KVConnectorRole.SCHEDULER:
            pass
        elif role == KVConnectorRole.WORKER:
            # Prefiller side sender
            self._cpu_kv_sender = CPUKVSender(1024 * 1024 * 1024) # 1GB for debug

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
        self._inflight_copy_tasks: list[CPUSendTask] = []

    
    ############################################################
    # Scheduler Side Methods
    ############################################################
    def _build_prefiller_meta(
            self, 
            scheduler_output: SchedulerOutput,
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
                request_tracker,
                self._block_size)
            output_meta.add_prefill(req_meta)

        for request in scheduler_output.scheduled_cached_reqs:
            request_tracker = self._prefill_reqs[request.req_id]
            request_tracker.update(request)

            req_meta = PrefillReqMeta.from_request_tracker(
                request_tracker,
                self._block_size)
            output_meta.add_prefill(req_meta)

    def build_decode_meta(
            self,
            scheduler_output: SchedulerOutput,
            output_meta: CPUConnectorMetadata) -> None:
        """Build the decode request metadata from the scheduler output.

        Args:
            scheduler_output (SchedulerOutput): The scheduler output.
            output_meta (CPUConnectorMetadata): The output metadata.
        """
        logger.error("build_decode_meta() not implemented, running a debug implementation!")
        pass


    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        return 0, False

    def update_state_after_alloc(
            self,
            request: "Request",
            blocks: "KVCacheBlocks",
            num_external_tokens: int) -> None:
        print("In update_state_after_alloc")
        pass

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
        print("In request_finished")
        return False, None

    #############################################################
    # Worker Side Methods
    #############################################################
    def _get_layer_id(self, layer_name: str) -> int:
        assert layer_name in self._layer_name_to_id, \
                f"Layer {layer_name} not found in layer name to id map"
        return self._layer_name_to_id[layer_name]

    def _get_layer_name(self, layer_id: int) -> str:
        assert layer_id in self._layer_id_to_name, \
                f"Layer id {layer_id} not found in layer id to name map"
        return self._layer_id_to_name[layer_id]

    def _get_kv_shape(self, num_blocks: int) -> torch.Size:
        return torch.Size((2, num_blocks, ) + self._kv_page_shape)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self._gpu_kv_caches = kv_caches
        idx = 0
        for layer_name in kv_caches.keys():
            self._layer_name_to_id[layer_name] = idx
            self._layer_id_to_name[idx] = layer_name
            idx += 1

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
        pass


    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.
        
        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        pass

    @_lmcache_nvtx_annotate
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
        meta = self._get_connector_metadata()
        assert isinstance(meta, CPUConnectorMetadata), \
                "Connector metadata is not of type CPUConnectorMetadata"

        assert self._cpu_kv_sender is not None

        for prefill_req in meta.prefill_meta:
            # Create a source spec with serializable types
            source_spec = SourceSpec(
                request_id=prefill_req.req_id,
                layer_id=self._get_layer_id(layer_name),
                start=prefill_req.token_range.start,
                stop=prefill_req.token_range.stop,
                shape=tuple(self._get_kv_shape(len(prefill_req.blocks_to_save))),
                dtype_str=str(kv_layer.dtype).split('.')[-1]  # Convert torch.float32 -> "float32"
            )

            # Create a destination spec
            # TODO: remove the hard-code here
            dest_spec = DestinationSpec(
                rank=get_tensor_model_parallel_rank(),
                host="localhost",
                base_port=54321,  # Changed from string to int to match the class definition
            )

            # Create the send task
            task = self._cpu_kv_sender.create_send_task(
                source_spec=source_spec,
                destination_spec=dest_spec,
            )
            assert isinstance(task, CPUSendTask), \
                    "Send task is not of type CPUSendTask"

            # Start copying the data to the CPU buffer
            buffer = task.tensor
            self._cuda_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._cuda_stream):
                # Copy the data from the GPU to the CPU buffer page by page
                d2h_page_copy(
                    src_layer=kv_layer,
                    dst_buffer=buffer,
                    block_ids=prefill_req.blocks_to_save
                )

            # record the cuda stream
            task.cuda_event = torch.cuda.Event()
            task.cuda_event.record(self._cuda_stream)

            self._inflight_copy_tasks.append(task)

        # Check the task states and send the tasks
        self._cpu_kv_sender.progress()


    @_lmcache_nvtx_annotate
    def wait_for_save(self):
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        for task in self._inflight_copy_tasks:
            if task.cuda_event is not None:
                task.cuda_event.synchronize()
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
        return None, None
