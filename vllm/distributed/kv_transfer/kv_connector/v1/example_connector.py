# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import safetensors
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class ReqMeta:
    # Key of the cached prefix, derived once on the scheduler side from the
    # request's block hashes. The worker never re-derives it.
    cache_key: str
    # Slot mappings, one per cached token
    slot_mapping: torch.Tensor
    # Is store or load
    is_store: bool

    @staticmethod
    def make_meta(
        cache_key: str,
        num_tokens: int,
        block_ids: list[int],
        block_size: int,
        is_store: bool,
    ) -> "ReqMeta":
        block_ids_tensor = torch.tensor(block_ids)
        num_blocks = block_ids_tensor.shape[0]
        block_offsets = torch.arange(0, block_size)
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids_tensor.reshape((num_blocks, 1)) * block_size
        )
        slot_mapping = slot_mapping.flatten()[:num_tokens]
        return ReqMeta(
            cache_key=cache_key,
            slot_mapping=slot_mapping,
            is_store=is_store,
        )


@dataclass
class ExampleConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)

    def add_request(
        self,
        cache_key: str,
        num_tokens: int,
        block_ids: list[int],
        block_size: int,
        is_store: bool,
    ) -> None:
        self.requests.append(
            ReqMeta.make_meta(cache_key, num_tokens, block_ids, block_size, is_store)
        )


class ExampleConnector(KVConnectorBase_V1):
    # NOTE: This is Simple debug implementation of the KV connector.
    # It save / load the KV cache to / from the disk.
    # It does extra work which will overwrite the existing prefix-cache in GPU
    # - to remove the overhead, need to add some "mask" in the ReqMeta class

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._block_size = vllm_config.cache_config.block_size
        self._requests_need_load: dict[str, Request] = {}
        # Scheduler side: cache key and cached-token count per request, set in
        # update_state_after_alloc and consumed by build_connector_meta.
        self._pending: dict[str, tuple[str, int]] = {}
        self._hash_block_size = self._block_size
        if role == KVConnectorRole.SCHEDULER and kv_cache_config is not None:
            from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

            _, self._hash_block_size = resolve_kv_cache_block_sizes(
                kv_cache_config, vllm_config
            )
        self._storage_path = self._kv_transfer_config.get_from_extra_config(
            "shared_storage_path", "/tmp"
        )
        logger.info(self._kv_transfer_config)
        logger.info("Shared storage path is %s", self._storage_path)

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

        def inject_kv_into_layer(
            dst_kv_cache_layer: torch.Tensor,
            src_kv_cache: torch.Tensor,
            slot_mapping: torch.Tensor,
            attn_metadata: AttentionMetadata,
        ) -> None:
            """Inject the KV cache into the layer.

            Args:
                dst_kv_cache_layer (torch.Tensor): the destination KV cache layer,
                    a standardized [B, H, N, C] per-layer view (H == 1 for MLA).
                src_kv_cache (torch.Tensor): the source KV cache.
                slot_mapping (torch.Tensor): the slot mapping. In shape
                    [num_tokens].
            """
            slot_mapping = slot_mapping.to(dst_kv_cache_layer.device, non_blocking=True)
            if isinstance(attn_metadata, MLACommonMetadata):
                # [B, 1, N, C] -> [B * N, C]; slot_mapping indexes B * N slots.
                dst_kv_cache_layer = dst_kv_cache_layer.reshape(
                    -1, dst_kv_cache_layer.shape[-1]
                )
                dst_kv_cache_layer[slot_mapping, ...] = src_kv_cache
            else:
                block_idxs = slot_mapping // self._block_size
                offsets = slot_mapping % self._block_size
                dst_kv_cache_layer[block_idxs, :, offsets] = src_kv_cache

        # Get the metadata
        metadata: KVConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, ExampleConnectorMetadata)

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.warning("In connector.start_load_kv, but the attn_metadata is None")
            return

        # Load the KV for each request each layer
        for request in metadata.requests:
            if request.is_store:
                continue
            logger.info(
                "Inject KV cache of %d tokens to the paged memory",
                len(request.slot_mapping),
            )
            for layer_name in forward_context.no_compile_layers:
                layer = forward_context.no_compile_layers[layer_name]

                # Only process layers that have kv_cache
                # attribute (attention layers) Skip non-attention
                # layers like FusedMoEFactory/MLP etc.
                kv_cache_layer = getattr(layer, "kv_cache", None)
                if kv_cache_layer is None:
                    continue

                filename = self._layer_filename(request.cache_key, layer_name)
                kv_cache_cpu = safetensors.torch.load_file(filename)["kv_cache"]
                kv_cache = kv_cache_cpu.to(kv_cache_layer.device, non_blocking=True)
                if isinstance(attn_metadata, dict):
                    inject_kv_into_layer(
                        kv_cache_layer,
                        kv_cache,
                        request.slot_mapping,
                        attn_metadata[layer_name],
                    )

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

        def extract_kv_from_layer(
            layer: torch.Tensor,
            slot_mapping: torch.Tensor,
        ) -> torch.Tensor:
            """Extract the KV cache from the layer.

            The layer is a standardized [B, H, N, C] per-layer view (H == 1 for MLA).
            """
            slot_mapping = slot_mapping.to(layer.device, non_blocking=True)
            if isinstance(attn_metadata, MLACommonMetadata):
                # [B, 1, N, C] -> [B * N, C]; slot_mapping indexes B * N slots.
                return layer.reshape(-1, layer.shape[-1])[slot_mapping, ...]
            block_idxs = slot_mapping // self._block_size
            offsets = slot_mapping % self._block_size
            return layer[block_idxs, :, offsets]

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, ExampleConnectorMetadata)
        for request in connector_metadata.requests:
            if request.is_store:
                filename = self._layer_filename(
                    request.cache_key, layer_name, create_folder=True
                )
                kv_cache = extract_kv_from_layer(kv_layer, request.slot_mapping)
                with gpu_sync_allowed():
                    tensors = {"kv_cache": kv_cache.detach().cpu()}
                safetensors.torch.save_file(tensors, filename)

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
        # NOTE: in this debug implementation, we assume that the prompt is
        # cached_prompt + newly_generated_single_token
        # Therefore, we cache prompt_token_ids[:-1] aligned to a block.

        # NOTE: in current v1 scheduler, the num_computed_tokens is aligned
        # with the block granularity. And it expects the returned blocks and
        # num_computed_tokens to also be aligned with the block granularity.
        cache_key, num_cached_tokens = self._cache_key(request)
        if cache_key is None or not self._found_match(cache_key):
            return 0, False

        logger.info("External Cache Hit!")

        return num_cached_tokens - num_computed_tokens, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.

        Record the request's cache key for build_connector_meta. If blocks
        were allocated, add to _requests_need_load, such that we load the
        KVs in the next forward pass.
        """
        cache_key, num_cached_tokens = self._cache_key(request)
        if cache_key is not None:
            self._pending[request.request_id] = (cache_key, num_cached_tokens)
        if num_external_tokens > 0:
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
        meta = ExampleConnectorMetadata()

        total_need_load = 0
        for new_req in scheduler_output.scheduled_new_reqs:
            pending = self._pending.get(new_req.req_id)
            if pending is None:
                continue
            cache_key, num_cached_tokens = pending
            if new_req.req_id in self._requests_need_load:
                meta.add_request(
                    cache_key=cache_key,
                    num_tokens=num_cached_tokens,
                    block_ids=new_req.block_ids[0],
                    block_size=self._block_size,
                    is_store=False,
                )
                total_need_load += 1
            else:
                # NOTE: here, we set the store and load being exclusive,
                # but a single request can have both store and load.
                # NOTE(rob): for this debug implementation, we only cache
                # the original prompt tokens.
                if not self._found_match(cache_key):
                    meta.add_request(
                        cache_key=cache_key,
                        num_tokens=num_cached_tokens,
                        block_ids=new_req.block_ids[0],
                        block_size=self._block_size,
                        is_store=True,
                    )

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            resumed_from_preemption = req_id in cached_reqs.resumed_req_ids
            if not resumed_from_preemption or req_id not in self._requests_need_load:
                continue

            # A resumed request passes through update_state_after_alloc
            # again, so its key is pending like a new request's.
            cache_key, num_cached_tokens = self._pending[req_id]
            new_block_ids = cached_reqs.new_block_ids[i]

            # NOTE(rob): For resumed req, new_block_ids is all
            # of the block_ids for the request.
            assert new_block_ids is not None
            block_ids = new_block_ids[0]

            meta.add_request(
                cache_key=cache_key,
                num_tokens=num_cached_tokens,
                block_ids=block_ids,
                block_size=self._block_size,
                is_store=False,
            )
            total_need_load += 1

        assert total_need_load == len(self._requests_need_load)
        self._requests_need_load.clear()
        self._pending.clear()
        return meta

    # ==============================
    # Helper functions
    # ==============================

    def _cache_key(self, request: "Request") -> tuple[str | None, int]:
        """Key the cached prefix on the request's own block hashes.

        The block hash chain already binds the tokens, ``cache_salt``, LoRA
        identity, multimodal items and ``prompt_embeds`` content, so the
        hash of the last cached block is the whole key. Deriving it from
        raw tokens instead would drop every one of those dimensions.

        Returns:
            The key and the number of cached tokens, or ``(None, 0)`` when
            the prompt does not cover a full block.
        """
        num_cached_tokens = align_to_block_size(
            request.num_prompt_tokens - 1, self._block_size
        )
        num_hashed_blocks = num_cached_tokens // self._hash_block_size
        block_hashes = request.block_hashes
        if num_hashed_blocks <= 0 or len(block_hashes) < num_hashed_blocks:
            return None, 0
        return block_hashes[num_hashed_blocks - 1].hex(), num_cached_tokens

    def _found_match(self, cache_key: str) -> bool:
        return os.path.exists(os.path.join(self._storage_path, cache_key))

    def _layer_filename(
        self, cache_key: str, layer_name: str, create_folder: bool = False
    ) -> str:
        foldername = os.path.join(self._storage_path, cache_key)
        if create_folder:
            os.makedirs(foldername, exist_ok=True)
        return os.path.join(foldername, f"{layer_name}.safetensors")


def align_to_block_size(num_tokens: int, block_size) -> int:
    """Align the number of tokens to the block size."""
    return (num_tokens - 1) // block_size * block_size
