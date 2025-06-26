# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import regex as re
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine import (
    P2pNcclEngine)
from vllm.distributed.parallel_state import get_world_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import MLACommonMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
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

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self._block_size = vllm_config.cache_config.block_size
        self._requests_need_load: dict[str, Any] = {}
        self.config = vllm_config.kv_transfer_config
        self.is_producer = self.config.is_kv_producer
        self.chunked_prefill: dict[str, Any] = {}

        self._rank = get_world_group().rank \
            if role == KVConnectorRole.WORKER else 0
        self._local_rank = get_world_group().local_rank \
            if role == KVConnectorRole.WORKER else 0

        self.p2p_nccl_engine = P2pNcclEngine(
            local_rank=self._local_rank,
            config=self.config,
            hostname="",
            port_offset=self._rank,
        ) if role == KVConnectorRole.WORKER else None

    # ==============================
    # Worker-side methods
    # ==============================

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

        # Only consumer/decode loads KV Cache
        if self.is_producer:
            return

        assert self.p2p_nccl_engine is not None

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return

        def inject_kv_into_layer(
            dst_kv_cache_layer: torch.Tensor,
            src_kv_cache: torch.Tensor,
            slot_mapping: torch.Tensor,
            request_id: str,
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
                request_id (str): request id for log
            """
            dst_kv_cache_layer_shape = dst_kv_cache_layer.shape
            if isinstance(attn_metadata, MLACommonMetadata):
                num_pages = dst_kv_cache_layer_shape[0]
                page_size = dst_kv_cache_layer_shape[1]
                dst_kv_cache_layer = dst_kv_cache_layer.reshape(
                    num_pages * page_size, -1)
                self.check_tensors_except_dim(dst_kv_cache_layer, src_kv_cache,
                                              0)
                num_token = src_kv_cache.shape[0]
                if len(slot_mapping) == num_token:
                    dst_kv_cache_layer[slot_mapping, ...] = src_kv_cache
                else:
                    dst_kv_cache_layer[slot_mapping[:num_token],
                                       ...] = src_kv_cache
                    logger.warning(
                        "🚧src_kv_cache does not match, num_slot:%d, "
                        "num_token:%d, request_id:%s", len(slot_mapping),
                        num_token, request_id)

                dst_kv_cache_layer.reshape(dst_kv_cache_layer_shape)
            else:
                num_pages = dst_kv_cache_layer_shape[1]
                page_size = dst_kv_cache_layer_shape[2]
                dst_kv_cache_layer = dst_kv_cache_layer.reshape(
                    2, num_pages * page_size, -1)
                self.check_tensors_except_dim(dst_kv_cache_layer, src_kv_cache,
                                              1)
                num_token = src_kv_cache.shape[1]
                if len(slot_mapping) == num_token:
                    dst_kv_cache_layer[:, slot_mapping, ...] = src_kv_cache
                else:
                    dst_kv_cache_layer[:, slot_mapping[:num_token],
                                       ...] = src_kv_cache
                    logger.warning(
                        "🚧src_kv_cache does not match, num_slot:%d, "
                        "num_token:%d, request_id:%s", len(slot_mapping),
                        num_token, request_id)

                dst_kv_cache_layer.reshape(dst_kv_cache_layer_shape)

        # Get the metadata
        metadata: KVConnectorMetadata = \
            self._get_connector_metadata()
        assert isinstance(metadata, P2pNcclConnectorMetadata)

        if metadata is None:
            return

        # Load the KV for each request each layer
        for request in metadata.requests:
            for layer_name in forward_context.no_compile_layers:
                attn_layer = forward_context.no_compile_layers[layer_name]
                kv_cache_layer = attn_layer.kv_cache[ \
                    forward_context.virtual_engine]

                kv_cache = self.p2p_nccl_engine.recv_tensor(
                    request.request_id + "#" + layer_name)

                if kv_cache is None:
                    logger.warning("🚧src_kv_cache is None, %s",
                                   request.request_id)
                    continue

                inject_kv_into_layer(kv_cache_layer, kv_cache,
                                     request.slot_mapping, request.request_id)

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

        # Only producer/prefill saves KV Cache
        if not self.is_producer:
            return

        assert self.p2p_nccl_engine is not None

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
            request_id = request.request_id
            ip, port = self.parse_request_id(request_id, True)
            remote_address = ip + ":" + str(port + self._rank)
            kv_cache = extract_kv_from_layer(kv_layer, request.slot_mapping)
            self.p2p_nccl_engine.send_tensor(request_id + "#" + layer_name,
                                             kv_cache, remote_address)

    def wait_for_save(self):
        if self.is_producer:
            assert self.p2p_nccl_engine is not None
            self.p2p_nccl_engine.wait_for_sent()

    def get_finished(
            self, finished_req_ids: set[str],
            **kwargs) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer,
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """

        assert self.p2p_nccl_engine is not None

        forward_context: ForwardContext = get_forward_context()
        return self.p2p_nccl_engine.get_finished(finished_req_ids,
                                                 forward_context)

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
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
            return 0, False

        num_external_tokens = (len(request.prompt_token_ids) - 1 -
                               num_computed_tokens)

        if num_external_tokens < 0:
            num_external_tokens = 0

        return num_external_tokens, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        """
        Update KVConnector state after block allocation.
        """
        if not self.is_producer and num_external_tokens > 0:
            self._requests_need_load[request.request_id] = (
                request, blocks.get_block_ids()[0])

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

        for new_req in scheduler_output.scheduled_new_reqs:
            if self.is_producer:
                num_scheduled_tokens = (
                    scheduler_output.num_scheduled_tokens)[new_req.req_id]
                num_tokens = num_scheduled_tokens + new_req.num_computed_tokens
                # the request's prompt is chunked prefill
                if num_tokens < len(new_req.prompt_token_ids):
                    # 'CachedRequestData' has no attribute 'prompt_token_ids'
                    self.chunked_prefill[new_req.req_id] = (
                        new_req.block_ids[0], new_req.prompt_token_ids)
                    continue
                # the request's prompt is not chunked prefill
                meta.add_request(request_id=new_req.req_id,
                                 token_ids=new_req.prompt_token_ids,
                                 block_ids=new_req.block_ids[0],
                                 block_size=self._block_size)
                continue
            if new_req.req_id in self._requests_need_load:
                meta.add_request(request_id=new_req.req_id,
                                 token_ids=new_req.prompt_token_ids,
                                 block_ids=new_req.block_ids[0],
                                 block_size=self._block_size)
                self._requests_need_load.pop(new_req.req_id)

        for cached_req in scheduler_output.scheduled_cached_reqs:
            if self.is_producer:
                num_scheduled_tokens = (
                    scheduler_output.num_scheduled_tokens)[cached_req.req_id]
                num_tokens = (num_scheduled_tokens +
                              cached_req.num_computed_tokens)
                assert cached_req.req_id in self.chunked_prefill
                block_ids = cached_req.new_block_ids[0]
                if not cached_req.resumed_from_preemption:
                    block_ids = (self.chunked_prefill[cached_req.req_id][0] +
                                 block_ids)
                prompt_token_ids = self.chunked_prefill[cached_req.req_id][1]
                # the request's prompt is chunked prefill again
                if num_tokens < len(prompt_token_ids):
                    self.chunked_prefill[cached_req.req_id] = (
                        block_ids, prompt_token_ids)
                    continue
                # the request's prompt is all prefilled finally
                meta.add_request(request_id=cached_req.req_id,
                                 token_ids=prompt_token_ids,
                                 block_ids=block_ids,
                                 block_size=self._block_size)
                self.chunked_prefill.pop(cached_req.req_id, None)
                continue

            # NOTE(rob): here we rely on the resumed requests being
            # the first N requests in the list scheduled_cache_reqs.
            if not cached_req.resumed_from_preemption:
                break
            if cached_req.req_id in self._requests_need_load:
                request, _ = self._requests_need_load.pop(cached_req.req_id)
                total_tokens = cached_req.num_computed_tokens + 1
                token_ids = request.all_token_ids[:total_tokens]

                # NOTE(rob): For resumed req, new_block_ids is all
                # of the block_ids for the request.
                block_ids = cached_req.new_block_ids[0]

                meta.add_request(request_id=cached_req.req_id,
                                 token_ids=token_ids,
                                 block_ids=block_ids,
                                 block_size=self._block_size)

        # Requests loaded asynchronously are not in the scheduler_output.
        # for request_id in self._requests_need_load:
        #     request, block_ids = self._requests_need_load[request_id]
        #     meta.add_request(request_id=request.request_id,
        #                      token_ids=request.prompt_token_ids,
        #                      block_ids=block_ids,
        #                      block_size=self._block_size)

        self._requests_need_load.clear()
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """

        self.chunked_prefill.pop(request.request_id, None)

        return False, None

    # ==============================
    # Static methods
    # ==============================

    @staticmethod
    def parse_request_id(request_id: str, is_prefill=True) -> tuple[str, int]:
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

            return ip, port
        raise ValueError(
            f"Request id {request_id} does not contain hostname and port")

    @staticmethod
    def check_tensors_except_dim(tensor1, tensor2, dim):
        shape1 = tensor1.size()
        shape2 = tensor2.size()

        if len(shape1) != len(shape2) or not all(
                s1 == s2
                for i, (s1, s2) in enumerate(zip(shape1, shape2)) if i != dim):
            raise NotImplementedError(
                "Currently, only symmetric TP is supported. Asymmetric TP, PP,"
                "and others will be supported in future PRs.")
