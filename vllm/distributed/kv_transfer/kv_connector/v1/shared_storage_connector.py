# SPDX-License-Identifier: Apache-2.0
import hashlib
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import safetensors
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class ReqMeta:
    # Request tokens
    token_ids: torch.Tensor
    # Slot mappings, should have the same length as token_ids
    slot_mapping: torch.Tensor
    # Is store or load
    is_store: bool

    ## Blocks allocated by the scheduler (no-longer needed)
    #block_ids: torch.Tensor

    @staticmethod
    def from_request(request: "Request", block_size: int,
                     is_store: bool) -> "ReqMeta":
        valid_num_tokens = align_to_block_size(len(request.prompt_token_ids),
                                               block_size)
        token_ids = torch.tensor(request.prompt_token_ids)[:valid_num_tokens]
        block_ids = torch.tensor(request.block_ids)
        num_blocks = block_ids.shape[0]
        block_offsets = torch.arange(0, block_size)
        slot_mapping = block_offsets.reshape((1, block_size)) + \
                block_ids.reshape((num_blocks, 1)) * block_size
        slot_mapping = slot_mapping.flatten()[:valid_num_tokens]
        return ReqMeta(
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            is_store=is_store,
        )


@dataclass
class SharedStorageConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta]

    def __init__(self):
        self.requests = []

    def add_request(
        self,
        request: "Request",
        block_size: int,
        is_store: bool,
    ) -> None:
        self.requests.append(
            ReqMeta.from_request(request, block_size, is_store))


class SharedStorageConnector(KVConnectorBase_V1):
    # NOTE: This is Simple debug implementation of the KV connector.
    # It save / load the KV cache to / from the disk.
    # It does extra work which will overwrite the existing prefix-cache in GPU
    # - to remove the overhead, need to add some "mask" in the ReqMeta class

    def __init__(self, rank: Optional[int], local_rank: Optional[int],
                 config: "VllmConfig", role: KVConnectorRole):
        super().__init__(
            rank=rank,
            local_rank=local_rank,
            config=config,
            role=role,
        )
        self._block_size = config.cache_config.block_size
        self._requests_need_load: list[str] = []
        self._storage_path = config.kv_transfer_config.get_from_extra_config(
            "shared_storage_path", "/tmp")
        logger.info(config.kv_transfer_config)
        logger.info("Shared storage path is %s", self._storage_path)

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        """Start loading the KV cache from the connector buffer to vLLM's 
        paged KV buffer.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be 
            the same.
        """

        def inject_kv_into_layer(
            dst_kv_cache_layer: torch.Tensor,
            src_kv_cache: torch.Tensor,
            slot_mapping: torch.Tensor,
        ) -> None:
            """Inject the KV cache into the layer.

            Args:
                dst_kv_cache_layer (torch.Tensor): the destination KV cache 
                    layer. In shape [2, num_pages, page_size, xxx].
                src_kv_cache (torch.Tensor): the source KV cache. In shape
                    [2, num_tokens, xxx].
                slot_mapping (torch.Tensor): the slot mapping. In shape 
                    [num_tokens].
            """
            dst_kv_cache_layer_shape = dst_kv_cache_layer.shape
            num_pages = dst_kv_cache_layer_shape[1]
            page_size = dst_kv_cache_layer_shape[2]
            dst_kv_cache_layer = dst_kv_cache_layer.reshape(
                2, num_pages * page_size, -1)
            dst_kv_cache_layer[:, slot_mapping, ...] = src_kv_cache
            dst_kv_cache_layer.reshape(dst_kv_cache_layer_shape)

        # Get the metadata
        metadata: KVConnectorMetadata = \
            self._get_connector_metadata()
        assert isinstance(metadata, SharedStorageConnectorMetadata)

        if metadata is None:
            logger.warning(
                "In connector.start_load_kv, but the connector metadata is None"
            )
            return

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.warning(
                "In connector.start_load_kv, but the attn_metadata is None")
            return

        # Load the KV for each request each layer
        for request in metadata.requests:
            if request.is_store:
                continue
            logger.info("Inject KV cache of %d tokens to the paged memory",
                        len(request.slot_mapping))
            for layer_name in forward_context.no_compile_layers:
                attn_layer = forward_context.no_compile_layers[layer_name]
                kv_cache_layer = attn_layer.kv_cache[\
                        forward_context.virtual_engine]

                filename = self.generate_filename_debug(
                    layer_name, request.token_ids)
                kv_cache = safetensors.torch.load_file(filename)["kv_cache"].cuda(
                )  # TODO: may need to handle the device here
                inject_kv_into_layer(kv_cache_layer, kv_cache,
                                     request.slot_mapping)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer. 
        
        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """Start saving the a layer of KV cache from vLLM's paged buffer 
        to the connector.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current 
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """

        def extract_kv_cache_from_layer(
            layer: torch.Tensor,
            slot_mapping: torch.Tensor,
        ) -> torch.Tensor:
            """Extract the KV cache from the layer.

            Assume the shape of the layer is (2, num_pages, page_size, xxx).
            """
            num_pages, page_size = layer.shape[1], layer.shape[2]
            return layer.reshape(2, num_pages * page_size, -1)[:, slot_mapping,
                                                               ...]

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, SharedStorageConnectorMetadata)
        for request in connector_metadata.requests:
            if request.is_store:
                filename = self.generate_filename_debug(
                    layer_name, request.token_ids)
                kv_cache = extract_kv_cache_from_layer(kv_layer,
                                                       request.slot_mapping)
                tensors = {"kv_cache": kv_cache.cpu().detach()}
                safetensors.torch.save_file(tensors, filename)

    def wait_for_save(self):
        return

    def get_external_prefix_cache_blocks(
        self,
        request: "Request",
        computed_blocks: list["KVCacheBlock"],
        num_computed_tokens: int,
        kv_cache_manager: "KVCacheManager",
    ) -> list["KVCacheBlock"]:
        """Get the external prefix cache blocks from the connector.

        This function may change the state of the connector, which will be 
        used by `attach_connector_meta` later.

        Args:
            request (Request): the request object.
            computed_blocks (list[KVCacheBlock]): the 'local' computed blocks.
            num_computed_tokens (int): the number of 'local' computed tokens.
            kv_cache_manager (KVCacheManager): the KV cache manager to 
                allocate/free the blocks if needed.

        Returns:
            The updated list of the computed blocks (appended with the remote
            cached blocks)
        """
        # NOTE: in this debug implementation, we assume that the prompt is
        # cached_prompt + newly_generated_single_token
        # Therefore, we use prompt_token_ids[:-1] to determine the folder name

        # NOTE: in current v1 scheduler, the num_computed_tokens is aligned
        # with the block granularity. And it expects the returned blocks and
        # num_computed_tokens to also be aligned with the block granularity.
        if not self.found_match_for_request(request):
            return computed_blocks

        # Now, first num_tokens_to_check tokens are hit, we need to prepare
        # the metadata for the worker connector to correctly load the KV

        logger.info("Hit the cache! Allocate new blocks!")
        num_tokens_to_check = align_to_block_size(
            len(request.prompt_token_ids) - 1, self._block_size)
        need_to_allocate = num_tokens_to_check - num_computed_tokens
        if need_to_allocate > 0:
            # HACK: We don't want the scheduler see the blocks are allocated
            # and associated with the current request. Instead, we want the
            # scheduler find that the blocks are already allocated and they
            # are associated with some other requests (i.e., the case of
            # prefix caching.

            # HACK: KVCacheManager.allocate_slots will pre-allocate a few
            # blocks, which will cause problems in the later allocations.
            # We should make sure the pre allocation does not happen.
            old_req_id = request.request_id
            request.request_id = "temp-req-id-for-connector"
            allocated_blocks = kv_cache_manager.allocate_slots(
                request,
                need_to_allocate,
                computed_blocks,
                skip_preallocate=True)
            request.request_id = old_req_id
            kv_cache_manager.req_to_blocks.pop("temp-req-id-for-connector")
            kv_cache_manager.num_cached_block.pop("temp-req-id-for-connector")

            num_expected_blocks = need_to_allocate // self._block_size
            if len(allocated_blocks) > num_expected_blocks:
                logger.error("Detected pre-allocated blocks in the connector!"
                             "This should not happen!")
                allocated_blocks = allocated_blocks[:num_expected_blocks]

            self._requests_need_load.append(request.request_id)
            return computed_blocks + allocated_blocks
        else:
            return computed_blocks

    def attach_connector_meta(
            self, scheduler_output: SchedulerOutput) -> SchedulerOutput:
        """Attach the connector metadata to the request object.

        This function should NOT modify other fields in the scheduler_output 
        except the `kv_connector_metadata` field.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        meta = SharedStorageConnectorMetadata()
        for request in scheduler_output.scheduled_new_reqs:
            # T^T, why there is both req_id and request_id????
            if request.req_id in self._requests_need_load:
                meta.add_request(request, self._block_size, is_store=False)
            else:
                # NOTE: here, we set the store and load being exclusive,
                # but in LMCache use case, a single request can have both
                # store and load status
                if not self.found_match_for_request(request):
                    meta.add_request(request, self._block_size, is_store=True)
        scheduler_output.kv_connector_metadata = meta

        self._requests_need_load.clear()
        return scheduler_output

    # ==============================
    # Helper functions
    # ==============================

    def found_match_for_request(
        self,
        request: "Request",
    ) -> bool:
        """Check if the cache is hit for the request.
        """
        num_tokens_to_check = align_to_block_size(
            len(request.prompt_token_ids) - 1, self._block_size)
        foldername = self.generate_foldername_debug(torch.tensor(
            request.prompt_token_ids)[:num_tokens_to_check],
                                                    create_folder=False)
        return os.path.exists(foldername)

    def generate_foldername_debug(
        self,
        input_ids: torch.Tensor,
        create_folder=False,
    ) -> str:
        """Generate a folder name based on the hash of the bytes of the input 
        ids.
        """
        input_ids_bytes = input_ids.numpy().tobytes()
        input_ids_hash = hashlib.md5(input_ids_bytes).hexdigest()
        foldername = os.path.join(self._storage_path, input_ids_hash)
        if create_folder:
            os.makedirs(foldername, exist_ok=True)
        return foldername

    def generate_filename_debug(
        self,
        layer_name: str,
        input_ids: torch.Tensor,
    ) -> str:
        """Generate a file name based on the layer name and the hash 
        of the bytes of the input ids.
        """
        foldername = self.generate_foldername_debug(input_ids,
                                                    create_folder=True)
        return os.path.join(foldername, f"{layer_name}.safetensors")


def align_to_block_size(num_tokens: int, block_size) -> int:
    """Align the number of tokens to the block size.
    """
    return (num_tokens - 1) // block_size * block_size
