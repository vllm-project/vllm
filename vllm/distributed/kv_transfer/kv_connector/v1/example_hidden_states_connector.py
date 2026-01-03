# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import safetensors
import torch

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata
from vllm.model_executor.models.extract_hidden_states import CacheOnlyAttentionLayer
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


def reshape_hidden_states_for_kv_cache(
    hidden_states: torch.Tensor, head_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    # hidden_states shape: [batch_size, hidden_size * num_hidden_states]
    # e.g. hidden_states = torch.cat([h_1, h_2, ..., h_n], dim=1)
    # where h_i is a hidden state of shape [batch_size, hidden_size]

    # Assuming num_hidden_states is a multiple of 2 for now

    batch_size = hidden_states.shape[0]
    split_size = hidden_states.shape[1] // 2
    key, value = torch.split(hidden_states, [split_size, split_size], dim=1)
    # key/value shape: [batch_size, hidden_size * num_hidden_states / 2]

    key = key.view(batch_size, -1, head_size)
    value = value.view(batch_size, -1, head_size)
    return key, value


def reshape_hidden_states_from_kv_cache(
    kv: torch.Tensor, num_hidden_states: int
) -> torch.Tensor:
    # kv shape: [2, batch_size, hidden_size / head_size * num_hidden_states / 2, head_size]
    kv = kv.flatten(2)
    # kv shape: [2, batch_size, hidden_size * num_hidden_states / 2]

    hidden_states = torch.cat([kv[0], kv[1]], dim=1)
    # hidden_states shape: [batch_size, hidden_size * num_hidden_states]

    split_size = hidden_states.shape[1] // num_hidden_states
    hidden_states = hidden_states.split(split_size, dim=1)

    return torch.stack(hidden_states, dim=0)


@dataclass
class ReqMeta:
    # Request ID
    req_id: str
    # Request filename
    filename: str
    # Request tokens
    token_ids: torch.Tensor
    # Slot mappings, should have the same length as token_ids
    slot_mapping: torch.Tensor

    @staticmethod
    def make_meta(
        req_id: str,
        filename: str,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
    ) -> "ReqMeta":
        token_ids_tensor = torch.tensor(token_ids)
        block_ids_tensor = torch.tensor(block_ids)
        num_blocks = block_ids_tensor.shape[0]
        block_offsets = torch.arange(0, block_size)
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids_tensor.reshape((num_blocks, 1)) * block_size
        )
        slot_mapping = slot_mapping.flatten()
        return ReqMeta(
            req_id=req_id,
            filename=filename,
            token_ids=token_ids_tensor,
            slot_mapping=slot_mapping,
        )


@dataclass
class ExampleHiddenStatesConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)

    def add_request(
        self,
        req_id: str,
        filename: str,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
    ) -> None:
        self.requests.append(
            ReqMeta.make_meta(req_id, filename, token_ids, block_ids, block_size)
        )


class ExampleHiddenStatesConnector(KVConnectorBase_V1):
    # NOTE: This is Simple debug implementation of the KV connector.
    # It save / load the KV cache to / from the disk.
    # It does extra work which will overwrite the existing prefix-cache in GPU
    # - to remove the overhead, need to add some "mask" in the ReqMeta class

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._block_size = vllm_config.cache_config.block_size
        self._storage_path = self._kv_transfer_config.get_from_extra_config(
            "shared_storage_path", "/tmp"
        )
        self.cache_layers = []
        logger.info(self._kv_transfer_config)
        logger.info("Shared storage path is %s", self._storage_path)

        spec_config = self._vllm_config.speculative_config.draft_model_config.hf_config
        self.num_hidden_states = len(
            getattr(spec_config, "eagle_aux_hidden_state_layer_ids", [])
        )

        self._request_filenames: dict[str, str] = {}

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        # Filter layers to only include CacheOnlyAttentionLayers
        layers = get_layers_from_vllm_config(
            self._vllm_config, CacheOnlyAttentionLayer, kv_caches.keys()
        )
        self.cache_layers = list(layers.keys())
        logger.info(f"Found {len(self.cache_layers)} CacheOnlyAttentionLayers")

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Start loading the KV cache from the connector buffer to vLLM's
        paged KV buffer.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.
        """
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        """Start saving the KV cache of the layer from vLLM's paged buffer
        to the connector.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        if layer_name not in self.cache_layers:
            return

        def extract_kv_from_layer(
            layer: torch.Tensor,
            slot_mapping: torch.Tensor,
            num_tokens: int,
        ) -> torch.Tensor:
            """Extract the KV cache from the layer.

            Assume the shape of the layer is (2, num_pages, page_size, xxx)
            if MLA is not used, and (num_pages, page_size, xxx) otherwise.
            """
            if isinstance(attn_metadata, MLACommonMetadata):
                num_pages, page_size = layer.shape[0], layer.shape[1]
                return layer.reshape(num_pages * page_size, -1)[slot_mapping, ...]
            num_pages, page_size = layer.shape[1], layer.shape[2]
            padded_kv = layer.reshape(2, num_pages * page_size, -1)[
                :, slot_mapping, ...
            ]
            return padded_kv[:, :num_tokens, ...]

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, ExampleHiddenStatesConnectorMetadata)
        os.makedirs(self._storage_path, exist_ok=True)
        for request in connector_metadata.requests:
            kv_cache = extract_kv_from_layer(
                kv_layer, request.slot_mapping, request.token_ids.shape[0]
            )
            hidden_states = reshape_hidden_states_from_kv_cache(
                kv_cache, self.num_hidden_states
            )
            tensors = {
                "hidden_states": hidden_states.detach().cpu(),
                "token_ids": request.token_ids.detach().cpu(),
            }
            safetensors.torch.save_file(tensors, request.filename)

    def wait_for_save(self):
        return

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
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
        # This connector is store-only, so we don't need to load any tokens
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.

        If blocks were allocated, add to _requests_need_load,
        such that we load the KVs in the next forward pass.
        """
        pass

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
        meta = ExampleHiddenStatesConnectorMetadata()

        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = new_req.prompt_token_ids or []
            filename = os.path.join(self._storage_path, f"{new_req.req_id}.safetensors")
            meta.add_request(
                new_req.req_id,
                filename=filename,
                token_ids=token_ids,
                block_ids=new_req.block_ids[0],
                block_size=self._block_size,
            )
            self._request_filenames[new_req.req_id] = filename

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called exactly once when a request has finished, before its blocks are
        freed.

        The connector may assumes responsibility for freeing the blocks
        asynchronously by returning True.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        req_id = request.request_id
        req_filename = self._request_filenames.pop(req_id, None)

        return False, {"hidden_states_path": req_filename}
