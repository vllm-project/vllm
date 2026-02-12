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
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


def extract_from_kv_cache(
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Extract data from KV cache
    Assume the shape of the kv_cache is (num_pages, page_size, num_heads, head_size)
    """

    padded_kv = kv_cache.flatten(0, 1)[slot_mapping]
    # shape: [len(slot_mapping), num_heads, head_size]
    return padded_kv[:num_tokens]  # shape: [num_tokens, num_heads, head_size]


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
    """
    Simple debug implementation of a HiddenStatesConnector.

    Simply extracts the hidden states from the kv cache and stores them to disk.
    Must be used in conjunction with the `extract_hidden_states` spec decoding method.
    """

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        """
        Indicates whether this connector prefers KV blocks that hold KV data for all
        layers, which can speed up KV data transfers. Defaults to False.
        """
        # Must be False so that drafter kv cache isn't merged with verifier's
        return False

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
        self.cache_layers = []  # set by self.register_kv_caches
        logger.info(self._kv_transfer_config)
        logger.info("Shared storage path is %s", self._storage_path)

        spec_config = self._vllm_config.speculative_config.draft_model_config.hf_config
        self.num_hidden_states = len(
            getattr(spec_config, "eagle_aux_hidden_state_layer_ids", [])
        )

        self._request_filenames: dict[str, str] = {}

    # ==============================
    # Worker-side methods
    # ==============================
    def start_load_kv(self, *args, **kwargs: Any) -> None:
        pass  # Empty implementation of abstract method

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass  # Empty implementation of abstract method

    def wait_for_save(self):
        pass  # Empty implementation of abstract method

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        from vllm.model_executor.models.extract_hidden_states import (
            CacheOnlyAttentionLayer,
        )

        # Filter layers to only include CacheOnlyAttentionLayers
        layers = get_layers_from_vllm_config(
            self._vllm_config, CacheOnlyAttentionLayer, list(kv_caches.keys())
        )
        self.cache_layers = list(layers.keys())
        assert len(self.cache_layers) == 1, (
            f"Expected 1 CacheOnlyAttentionLayer, got {len(self.cache_layers)}"
        )

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

        from vllm.model_executor.models.extract_hidden_states import (
            CacheOnlyAttentionMetadata,
        )

        assert isinstance(attn_metadata, CacheOnlyAttentionMetadata), (
            "ExampleHiddenStatesConnector only supports CacheOnlyAttentionBackend"
        )

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, ExampleHiddenStatesConnectorMetadata)

        os.makedirs(self._storage_path, exist_ok=True)
        for request in connector_metadata.requests:
            hidden_states = extract_from_kv_cache(
                kv_layer, request.slot_mapping, request.token_ids.shape[0]
            )
            tensors = {
                "hidden_states": hidden_states.detach().cpu(),
                "token_ids": request.token_ids.detach().cpu(),
            }
            safetensors.torch.save_file(tensors, request.filename)

    # ==============================
    # Scheduler-side methods
    # ==============================

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
        # Usually used to handle allocation of new blocks for requests that are loading
        # tokens from connector's external kv cache. We never load from external cache
        # so this is a no-op.
        assert num_external_tokens == 0, "This connector is store-only"

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

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: "VllmConfig") -> str | None:
        """
        Get the required KV cache layout for this connector.
        Args:
            vllm_config (VllmConfig): the vllm config.

        Returns:
            str: the required KV cache layout. e.g. HND, or NHD.
            None if the connector does not require a specific layout.
        """

        if cls is KVConnectorBase_V1:
            raise TypeError(
                "get_required_kvcache_layout should not be called "
                "on the abstract base class"
            )
        # NHD means we have (num_tokens, num_heads)
        # HND means we have (num_heads, num_tokens)
        # For now, we only support NHD layout since this keeps the
        # hidden states for each token together in memory.
        # HND is primarily used when sharding heads across devices.
        return "NHD"
