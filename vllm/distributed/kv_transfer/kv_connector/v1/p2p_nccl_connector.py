# SPDX-License-Identifier: Apache-2.0

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.kv_transfer.kv_pipe.p2p_nccl_pipe import P2pNcclPipe
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import MLACommonMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class ReqMeta:
    # Request Id
    request_id: str
    # Request tokens
    token_ids: torch.Tensor
    # Slot mappings, should have the same length as token_ids
    slot_mapping: torch.Tensor

    @staticmethod
    def make_meta(request_id: str, token_ids: list[int], block_ids: list[int],
                  block_size: int) -> "ReqMeta":
        valid_num_tokens = len(token_ids)
        token_ids_tensor = torch.tensor(token_ids)
        block_ids_tensor = torch.tensor(block_ids)
        num_blocks = block_ids_tensor.shape[0]
        block_offsets = torch.arange(0, block_size)
        slot_mapping = block_offsets.reshape((1, block_size)) + \
                block_ids_tensor.reshape((num_blocks, 1)) * block_size
        slot_mapping = slot_mapping.flatten()[:valid_num_tokens]
        return ReqMeta(
            request_id=request_id,
            token_ids=token_ids_tensor,
            slot_mapping=slot_mapping,
        )


@dataclass
class P2pNcclConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta]

    def __init__(self):
        self.requests = []

    def add_request(
        self,
        request_id: str,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
    ) -> None:
        self.requests.append(
            ReqMeta.make_meta(request_id, token_ids, block_ids, block_size))


class P2pNcclConnector(KVConnectorBase_V1):

    def __init__(self,
                 vllm_config: "VllmConfig",
                 role: KVConnectorRole,
                 rank: int = 0,
                 local_rank: int = 0):
        super().__init__(vllm_config=vllm_config,
                         role=role,
                         rank=rank,
                         local_rank=local_rank)
        self._block_size = vllm_config.cache_config.block_size
        self._requests_need_load: dict[str, Request] = {}
        self.config = vllm_config.kv_transfer_config
        self.rank = rank
        self.is_producer = self.config.is_kv_producer

        self.p2p_nccl_pipe = P2pNcclPipe(
            local_rank=local_rank,
            config=self.config,
            hostname="",
            port_offset=rank,
        ) if role == KVConnectorRole.WORKER else None

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
        assert self.p2p_nccl_pipe is not None

        attn_metadata = forward_context.attn_metadata

        def inject_kv_into_layer(
            dst_kv_cache_layer: torch.Tensor,
            src_kv_cache: torch.Tensor,
            slot_mapping: torch.Tensor,
        ) -> None:
            """Inject the KV cache into the layer.

            Args:
                dst_kv_cache_layer (torch.Tensor): the destination KV cache
                    layer. In shape [2, num_pages, page_size, xxx] if not
                    using MLA, [num_pages, page_size, xxx] otherwise.
                src_kv_cache (torch.Tensor): the source KV cache. In shape
                    [2, num_tokens, xxx] if not using MLA, [num_tokens, xxx]
                    otherwise.
                slot_mapping (torch.Tensor): the slot mapping. In shape
                    [num_tokens].
            """
            dst_kv_cache_layer_shape = dst_kv_cache_layer.shape
            if isinstance(attn_metadata, MLACommonMetadata):
                num_pages = dst_kv_cache_layer_shape[0]
                page_size = dst_kv_cache_layer_shape[1]
                dst_kv_cache_layer = dst_kv_cache_layer.reshape(
                    num_pages * page_size, -1)
                dst_kv_cache_layer[slot_mapping, ...] = src_kv_cache
                dst_kv_cache_layer.reshape(dst_kv_cache_layer_shape)
            else:
                num_pages = dst_kv_cache_layer_shape[1]
                page_size = dst_kv_cache_layer_shape[2]
                dst_kv_cache_layer = dst_kv_cache_layer.reshape(
                    2, num_pages * page_size, -1)
                dst_kv_cache_layer[:, slot_mapping, ...] = src_kv_cache
                dst_kv_cache_layer.reshape(dst_kv_cache_layer_shape)

        # Get the metadata
        metadata: KVConnectorMetadata = \
            self._get_connector_metadata()
        assert isinstance(metadata, P2pNcclConnectorMetadata)

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
            if self.is_producer:
                continue
            for layer_name in forward_context.no_compile_layers:
                attn_layer = forward_context.no_compile_layers[layer_name]
                kv_cache_layer = attn_layer.kv_cache[ \
                    forward_context.virtual_engine]

                kv_cache = self.p2p_nccl_pipe.recv_tensor(request.request_id +
                                                          "-" + layer_name)

                inject_kv_into_layer(kv_cache_layer, kv_cache,
                                     request.slot_mapping)

            logger.info("Inject KV cache of %d tokens to the paged memory, %s",
                        len(request.slot_mapping), request.request_id)

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
        """Start saving the KV cache of the layer from vLLM's paged buffer
        to the connector.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        assert self.p2p_nccl_pipe is not None

        def extract_kv_from_layer(
            layer: torch.Tensor,
            slot_mapping: torch.Tensor,
        ) -> torch.Tensor:
            """Extract the KV cache from the layer.

            Assume the shape of the layer is (2, num_pages, page_size, xxx)
            if MLA is not used, and (num_pages, page_size, xxx) otherwise.
            """
            if isinstance(attn_metadata, MLACommonMetadata):
                num_pages, page_size = layer.shape[0], layer.shape[1]
                return layer.reshape(num_pages * page_size, -1)[slot_mapping,
                                                                ...]
            num_pages, page_size = layer.shape[1], layer.shape[2]
            return layer.reshape(2, num_pages * page_size, -1)[:, slot_mapping,
                                                               ...]

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, P2pNcclConnectorMetadata)
        for request in connector_metadata.requests:
            if self.is_producer:
                request_id = request.request_id
                ip, port = self.parse_request_id(request_id, True)
                remote_address = ip + ":" + str(port + self._rank)
                kv_cache = extract_kv_from_layer(kv_layer,
                                                 request.slot_mapping)
                self.p2p_nccl_pipe.send_tensor(request_id + "-" + layer_name,
                                               kv_cache, remote_address)

    def wait_for_save(self):
        if self.is_producer:
            assert self.p2p_nccl_pipe is not None
            self.p2p_nccl_pipe.wait_for_sent()

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> int:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        if self.is_producer:
            return 0

        num_external_tokens = (len(request.prompt_token_ids) - 1 -
                               num_computed_tokens)
        logger.info(
            "🍒num_external_tokens:%d, num_prompt_tokens:%d, "
            "num_computed_tokens:%d", num_external_tokens,
            len(request.prompt_token_ids), num_computed_tokens)

        return num_external_tokens

    def update_state_after_alloc(self, request: "Request",
                                 num_external_tokens: int):
        """
        Update KVConnector state after block allocation.

        If blocks were allocated, add to _requests_need_load,
        such that we load the KVs in the next forward pass.
        """
        if not self.is_producer and num_external_tokens > 0:
            self._requests_need_load[request.request_id] = request

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """Build the connector metadata for this step.

        This function should NOT modify any fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        meta = P2pNcclConnectorMetadata()

        total_need_load = 0
        for new_req in scheduler_output.scheduled_new_reqs:
            if new_req.req_id in self._requests_need_load:
                meta.add_request(request_id=new_req.req_id,
                                 token_ids=new_req.prompt_token_ids,
                                 block_ids=new_req.block_ids,
                                 block_size=self._block_size)
                total_need_load += 1
            else:
                if self.is_producer:
                    meta.add_request(request_id=new_req.req_id,
                                     token_ids=new_req.prompt_token_ids,
                                     block_ids=new_req.block_ids,
                                     block_size=self._block_size)

        for cached_req in scheduler_output.scheduled_cached_reqs:
            # NOTE(rob): here we rely on the resumed requests being
            # the first N requests in the list scheduled_cache_reqs.
            if not cached_req.resumed_from_preemption:
                break
            if cached_req.req_id in self._requests_need_load:
                # NOTE(rob): cached_req_data does not have the full
                # list of token ids (only new tokens). So we look it
                # up in the actual request object.
                request = self._requests_need_load[cached_req.req_id]
                total_tokens = (len(cached_req.new_token_ids) +
                                cached_req.num_computed_tokens)
                token_ids = request.all_token_ids[:total_tokens]

                # NOTE(rob): For resumed req, new_block_ids is all
                # of the block_ids for the request.
                block_ids = cached_req.new_block_ids

                meta.add_request(request_id=cached_req.req_id,
                                 token_ids=token_ids,
                                 block_ids=block_ids,
                                 block_size=self._block_size)
                total_need_load += 1

        assert total_need_load == len(self._requests_need_load)
        self._requests_need_load.clear()
        return meta

    @staticmethod
    def parse_request_id(request_id: str, is_prefill=True) -> Tuple[str, int]:
        logger.debug("parse_request_id, request_id: %s, is_prefill: %s",
                     request_id, is_prefill)
        # Regular expression to match the string hostname and integer port
        if is_prefill:
            pattern = r"___decode_addr_(.*):(\d+)"
        else:
            pattern = r"___prefill_addr_(.*):(\d+)___"

        # Use re.search to find the pattern in the request_id
        match = re.search(pattern, request_id)
        if match:
            # Extract the ranks
            ip = match.group(1)
            port = int(match.group(2))

            logger.debug("parse_request_id, request_id: %s, ip: %s, port: %s",
                         request_id, ip, str(port))
            return ip, port
        raise ValueError(
            f"Request id {request_id} does not contain hostname and port")
