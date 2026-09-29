# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import safetensors
import torch

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from safetensors.torch import save_file

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

    # padded_kv = kv_cache.flatten(0, 1)[slot_mapping]
    # # shape: [len(slot_mapping), num_heads, head_size]
    # return padded_kv[:num_tokens]  # shape: [num_tokens, num_heads, head_size]
    flat_kv = kv_cache.flatten(0, 1)

    slot_mapping = slot_mapping[:num_tokens]

    return flat_kv[slot_mapping]


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
    # Whether this request is a new request or partially computed already
    new_req: bool
    num_computed_tokens: int

    @staticmethod
    def make_meta(
        req_id: str,
        filename: str,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        new_req: bool,
        num_computed_tokens: int,
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
            new_req=new_req,
            num_computed_tokens=num_computed_tokens,
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
        new_req: bool = True,
        num_computed_tokens: int = 0,
    ) -> None:
        self.requests.append(
            ReqMeta.make_meta(
                req_id, filename, token_ids, block_ids, block_size, new_req, num_computed_tokens
            )
        )


class ExampleHiddenStatesConnector(KVConnectorBase_V1,SupportsHMA):
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
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        # self._block_size = vllm_config.cache_config.block_size
        self._cache_group_id: int | None = None
        self._block_size: int | None = None
        self.cache_layers: list[str] = []

        for group_id, group in enumerate(
            self._kv_cache_config.kv_cache_groups
        ):
            cache_only_layers = [
                layer_name
                for layer_name in group.layer_names
                if layer_name.startswith("cache_only_layers.")
            ]

            if not cache_only_layers:
                continue

            assert self._cache_group_id is None, (
                "Found multiple CacheOnlyAttention KV cache groups: "
                f"previous={self._cache_group_id}, "
                f"current={group_id}, "
                f"layers={cache_only_layers}"
            )

            self._cache_group_id = group_id
            self._block_size = group.kv_cache_spec.block_size
            self.cache_layers = cache_only_layers

        assert self._cache_group_id is not None, (
            "Could not find CacheOnlyAttention KV cache group"
        )
        assert self._block_size is not None
        assert self.cache_layers

        logger.info(
            "ExampleHiddenStatesConnector initialized: "
            "cache_layers=%s, group_id=%s, block_size=%s, num_groups=%s",
            self.cache_layers,
            self._cache_group_id,
            self._block_size,
            len(self._kv_cache_config.kv_cache_groups),
        )
        self._storage_path = self._kv_transfer_config.get_from_extra_config(
            "shared_storage_path", "/tmp"
        )
        # self.cache_layers: list[str] = []  # set by self.register_kv_caches
        logger.info(self._kv_transfer_config)
        logger.info("Shared storage path is %s", self._storage_path)

        assert self._vllm_config.speculative_config is not None, (
            "ExampleHiddenStatesConnector only works when using "
            "'extract_hidden_states' speculative method"
        )
        spec_config = self._vllm_config.speculative_config.draft_model_config.hf_config
        self.num_hidden_states = len(
            getattr(spec_config, "eagle_aux_hidden_state_layer_ids", [])
        )

        # self._request_filenames: dict[str, str] = {}
        self._request_filenames: dict[str, str] = {}
        self._request_chunks: dict[str, list[tuple[int, str]]] = {}
        self._active_requests: dict[str, NewRequestData] = {}
        self._req_blocks: dict[str, list[int]] = {}

    # ==============================
    # Worker-side methods
    # ==============================
    def start_load_kv(self, *args, **kwargs: Any) -> None:
        pass  # Empty implementation of abstract method

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass  # Empty implementation of abstract method

    def wait_for_save(self):
        pass  # Empty implementation of abstract method

    # def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
    #     from vllm.model_executor.models.extract_hidden_states import (
    #         CacheOnlyAttentionLayer,
    #     )

    #     # Filter layers to only include CacheOnlyAttentionLayers
    #     layers = get_layers_from_vllm_config(
    #         self._vllm_config, CacheOnlyAttentionLayer, list(kv_caches.keys())
    #     )
    #     self.cache_layers = list(layers.keys())
    #     assert len(self.cache_layers) == 1, (
    #         f"Expected 1 CacheOnlyAttentionLayer, got {len(self.cache_layers)}"
    #     )
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        logger.warning(
            "EHS DEBUG register_kv_caches: num=%d keys=%s",
            len(kv_caches),
            list(kv_caches.keys()),
        )

        from vllm.model_executor.models.extract_hidden_states import (
            CacheOnlyAttentionLayer,
        )

        layers = get_layers_from_vllm_config(
            self._vllm_config,
            CacheOnlyAttentionLayer,
            list(kv_caches.keys()),
        )

        registered_cache_layers = list(layers.keys())

        logger.warning(
            "EHS DEBUG registered CacheOnlyAttention layers=%s",
            registered_cache_layers,
        )

        # Sanity check: the worker-side KV cache must contain
        # the same CacheOnlyAttentionLayer we identified from
        # the KV cache group configuration.
        assert set(registered_cache_layers) == set(self.cache_layers), (
            "Mismatch between configured CacheOnlyAttention layers "
            f"{self.cache_layers} and registered KV cache layers "
            f"{registered_cache_layers}"
        )

        logger.info(
            "ExampleHiddenStatesConnector registered: "
            "cache_layers=%s, group_id=%s, block_size=%s",
            self.cache_layers,
            self._cache_group_id,
            self._block_size,
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

        query_start_loc = attn_metadata.query_start_loc

        query_start_loc_cpu = query_start_loc.detach().cpu()

        total_tokens = attn_metadata.slot_mapping.numel()

        assert query_start_loc_cpu[-1].item() == total_tokens, (
            f"Mismatch between query_start_loc and slot_mapping: "
            f"query_end={query_start_loc_cpu[-1].item()}, "
            f"slot_mapping={total_tokens}"
        )

        requests_by_id = {
            request.req_id: request
            for request in connector_metadata.requests
        }

        attn_request_indices = {
            req_id: idx
            for idx, req_id in enumerate(attn_metadata.req_ids)
        }

        for request in connector_metadata.requests:
            req_id = request.req_id

            assert req_id in attn_request_indices, (
                f"Request {req_id} not found in attention metadata. "
                f"attn_req_ids={attn_metadata.req_ids}, "
                f"connector_req_ids={list(requests_by_id.keys())}"
            )

            request_idx = attn_request_indices[req_id]
            print(
                "[SAVE KV ALIGN]",
                f"req_id={req_id}",
                f"request_idx={request_idx}",
                f"start={query_start_loc_cpu[request_idx].item()}",
                f"end={query_start_loc_cpu[request_idx + 1].item()}",
                f"num_tokens={query_start_loc_cpu[request_idx + 1].item() - query_start_loc_cpu[request_idx].item()}",
                f"num_computed_tokens={request.num_computed_tokens}",
                flush=True,
            )

            start = query_start_loc_cpu[request_idx].item()
            end = query_start_loc_cpu[request_idx + 1].item()
            num_tokens = end - start

            real_slot_mapping = attn_metadata.slot_mapping[start:end]

            hidden_states = extract_from_kv_cache(
                kv_layer,
                real_slot_mapping,
                num_tokens,
            )

            token_start = request.num_computed_tokens
            token_end = token_start + num_tokens

            token_ids = request.token_ids[token_start:token_end]

            assert token_ids.shape[0] == num_tokens, (
                f"Token IDs mismatch for {request.req_id}: "
                f"token_ids={token_ids.shape[0]}, "
                f"num_tokens={num_tokens}, "
                f"num_computed_tokens={request.num_computed_tokens}, "
                f"token_range=[{token_start}:{token_end}], "
                f"total_token_ids={request.token_ids.shape[0]}"
            )

            tensors = {
                "hidden_states": hidden_states.cpu(),
                "token_ids": token_ids,
            }

            filename = request.filename
            print(
                "[SAVE FILE]",
                f"req_id={request.req_id}",
                f"filename={filename}",
                f"token_start={token_start}",
                f"token_end={token_end}",
                f"num_tokens={num_tokens}",
                f"token_ids_len={token_ids.shape[0]}",
                f"hidden_states_shape={tuple(hidden_states.shape)}",
                flush=True,
            )
            self._request_chunks.setdefault(request.req_id, [])

            chunk_filename = os.path.join(
                self._storage_path,
                f"{request.req_id}.chunk_{token_start:08d}.safetensors",
            )
            print(
                "[TOKEN CHECK]",
                f"req_id={request.req_id}",
                f"token_range=[{token_start}:{token_end}]",
                f"original_len={request.token_ids.numel()}",
                f"saved_len={token_ids.numel()}",
                f"exact_match={torch.equal(request.token_ids[token_start:token_end], token_ids)}",
                f"original_first20={request.token_ids[token_start:token_start+20].tolist()}",
                f"saved_first20={token_ids[:20].tolist()}",
                f"original_last20={request.token_ids[token_end-20:token_end].tolist()}",
                f"saved_last20={token_ids[-20:].tolist()}",
                flush=True,
            )

            save_file(tensors, chunk_filename)

            self._request_chunks[request.req_id].append(
                (token_start, chunk_filename)
            )
            print(
                "[CHUNK STATE AFTER SAVE]",
                f"req_id={request.req_id}",
                f"chunks={self._request_chunks.get(request.req_id)}",
                f"dict_id={id(self._request_chunks)}",
                flush=True,
            )
            chunk_idx = len(self._request_chunks[request.req_id])

            print(
                "[SAVE CHUNK]",
                f"req_id={request.req_id}",
                f"chunk_idx={chunk_idx}",
                f"token_start={token_start}",
                f"token_end={token_end}",
                f"num_tokens={num_tokens}",
                f"filename={chunk_filename}",
                flush=True,
            )
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
        group_id = self._cache_group_id
        assert group_id is not None
        assert self._block_size is not None
        meta = ExampleHiddenStatesConnectorMetadata()
        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = new_req.prompt_token_ids or []
            filename = os.path.join(self._storage_path, f"{new_req.req_id}.safetensors")
            meta.add_request(
                new_req.req_id,
                filename=filename,
                token_ids=token_ids,
                # block_ids=new_req.block_ids[0],
                block_ids=new_req.block_ids[group_id],
                block_size=self._block_size,
                new_req=True,
                num_computed_tokens=new_req.num_computed_tokens,
            )
            self._request_filenames[new_req.req_id] = filename
            self._request_chunks.setdefault(new_req.req_id, [])
            self._active_requests[new_req.req_id] = new_req
            # self._req_blocks[new_req.req_id] = list(new_req.block_ids[0])
            self._req_blocks[new_req.req_id] = list(
                new_req.block_ids[group_id]
            )

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            if req_id not in self._active_requests:
                continue

            new_block_ids = cached_reqs.new_block_ids[i]

            cached_req = self._active_requests[req_id]
            req_block_ids = self._req_blocks[req_id]

            if new_block_ids is None:
                continue

            # block_ids = new_block_ids[0]
            block_ids = new_block_ids[group_id]

            req_block_ids.extend(block_ids)
            filename = os.path.join(self._storage_path, f"{req_id}.safetensors")

            meta.add_request(
                req_id=req_id,
                filename=filename,
                token_ids=cached_req.prompt_token_ids or [],
                block_ids=req_block_ids,
                block_size=self._block_size,
                new_req=False,
                num_computed_tokens=cached_reqs.num_computed_tokens[i],
            )

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        import glob
        import os
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

        req_filename = os.path.join(
            self._storage_path,
            f"{req_id}.safetensors",
        )

        chunk_pattern = os.path.join(
            self._storage_path,
            f"{req_id}.chunk_*.safetensors",
        )

        chunk_files = glob.glob(chunk_pattern)

        chunks = []

        for chunk_filename in chunk_files:
            basename = os.path.basename(chunk_filename)

            token_start_str = (
                basename
                .split(".chunk_", 1)[1]
                .split(".safetensors", 1)[0]
            )

            token_start = int(token_start_str)

            chunks.append((token_start, chunk_filename))

        # chunks.sort(key=lambda x: x[0])

        print(
            "[CHUNK DISCOVERY]",
            f"req_id={req_id}",
            f"pattern={chunk_pattern}",
            f"chunks={chunks}",
            flush=True,
        )

        print(
            "[CHUNK STATE BEFORE MERGE]",
            f"req_id={req_id}",
            f"chunks={self._request_chunks.get(req_id)}",
            f"all_req_ids={list(self._request_chunks.keys())}",
            f"dict_id={id(self._request_chunks)}",
            flush=True,
        )

        print(
            "[MERGE HS]",
            f"req_id={req_id}",
            f"num_chunks={len(chunks)}",
            f"final_filename={req_filename}",
            flush=True,
        )

        if req_filename is None:
            raise RuntimeError(
                f"No hidden-state filename found for request {req_id}"
            )

        if not chunks:
            print(
                "[MERGE SKIP]",
                f"req_id={req_id}",
                "no hidden-state chunks found",
                flush=True,
            )

            _ = self._active_requests.pop(req_id, None)
            _ = self._req_blocks.pop(req_id, None)
            self._request_chunks.pop(req_id, None)

            return False, None

        # Sort by token_start.
        chunks.sort(key=lambda x: x[0])

        from safetensors.torch import load_file, save_file

        all_token_ids = []
        all_hidden_states = []

        expected_start = 0

        for chunk_idx, (token_start, chunk_filename) in enumerate(chunks):
            data = load_file(chunk_filename)

            token_ids = data["token_ids"]
            hidden_states = data["hidden_states"]

            if token_ids.shape[0] != hidden_states.shape[0]:
                raise RuntimeError(
                    f"Chunk length mismatch for {req_id}: "
                    f"chunk={chunk_filename}, "
                    f"token_ids={token_ids.shape[0]}, "
                    f"hidden_states={hidden_states.shape[0]}"
                )

            print(
                "[MERGE CHUNK]",
                f"req_id={req_id}",
                f"chunk_idx={chunk_idx}",
                f"token_start={token_start}",
                f"chunk_tokens={token_ids.shape[0]}",
                f"hidden_states_shape={tuple(hidden_states.shape)}",
                flush=True,
            )

            # Make sure chunks are contiguous.
            if token_start != expected_start:
                raise RuntimeError(
                    f"Hidden-state chunks are not contiguous for {req_id}: "
                    f"expected_start={expected_start}, "
                    f"actual_start={token_start}, "
                    f"chunk={chunk_filename}"
                )

            all_token_ids.append(token_ids)
            all_hidden_states.append(hidden_states)

            expected_start += token_ids.shape[0]

        merged_token_ids = torch.cat(all_token_ids, dim=0)
        merged_hidden_states = torch.cat(all_hidden_states, dim=0)

        if merged_token_ids.shape[0] != merged_hidden_states.shape[0]:
            raise RuntimeError(
                f"Merged hidden states length mismatch for {req_id}: "
                f"token_ids={merged_token_ids.shape[0]}, "
                f"hidden_states={merged_hidden_states.shape[0]}"
            )

        save_file(
            {
                "token_ids": merged_token_ids,
                "hidden_states": merged_hidden_states,
            },
            req_filename,
        )

        print(
            "[MERGE DONE]",
            f"req_id={req_id}",
            f"tokens={merged_token_ids.shape[0]}",
            f"hidden_states_shape={tuple(merged_hidden_states.shape)}",
            f"filename={req_filename}",
            flush=True,
        )

        # Remove temporary chunk files.
        for _, chunk_filename in chunks:
            try:
                os.remove(chunk_filename)
            except FileNotFoundError:
                pass

        _ = self._active_requests.pop(req_id, None)
        _ = self._req_blocks.pop(req_id, None)

        return False, {"hidden_states_path": req_filename}

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self._cache_group_id is not None
        return self.request_finished(
            request,
            block_ids=block_ids[self._cache_group_id],
        )

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
