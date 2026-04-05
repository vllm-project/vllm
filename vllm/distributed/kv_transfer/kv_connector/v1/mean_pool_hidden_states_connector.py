# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Mean-pool hidden states connector for vLLM.

Extracts hidden states from a CacheOnlyAttentionLayer's KV cache,
mean-pools over prompt tokens, and returns the resulting vector via
kv_transfer_params in the API response.

Usage (server launch — requires extract_hidden_states spec decode method):
    --speculative-config '{
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "draft_model_config": {
            "hf_config": {"eagle_aux_hidden_state_layer_ids": [16]}
        }
    }'
    --kv-transfer-config '{
        "kv_connector": "MeanPoolHiddenStatesConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {
            "shared_storage_path": "/dev/shm/vllm_mean_pool"
        }
    }'

WARNING: This connector uses a shared directory for communication.
If running multiple vLLM instances, you MUST provide a unique `shared_storage_path`
for each instance to avoid conflicts.

Files for completed requests are removed automatically.
However, in case of a server crash, stale files may accumulate in
the `shared_storage_path`. It is the user's responsibility to handle
periodic cleanup of this directory.

Limitations:
    The extract_hidden_states method piggybacks on the speculative decoding
    framework.

    - Online serving (``vllm serve``): NOT supported.
    - Offline batched inference (``llm.generate(prompts, ...)``): Supported
      ONLY with ``max_tokens=1``.
    - Offline single-request inference: Fully supported
"""

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

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


@dataclass
class _ReqMeta:
    """Lightweight per-request metadata sent from scheduler to worker.

    Only carries what the worker needs: request ID, prompt length, and
    the slot mapping into the hidden state KV cache.
    """

    req_id: str
    prompt_len: int
    slot_mapping: list[int]

    @staticmethod
    def make(
        req_id: str,
        prompt_len: int,
        block_ids: list[int],
        block_size: int,
    ) -> "_ReqMeta":
        slot_mapping = [
            bid * block_size + offset
            for bid in block_ids
            for offset in range(block_size)
        ]
        return _ReqMeta(
            req_id=req_id,
            prompt_len=prompt_len,
            slot_mapping=slot_mapping,
        )


@dataclass
class MeanPoolConnectorMetadata(KVConnectorMetadata):
    requests: list[_ReqMeta] = field(default_factory=list)
    finished_req_ids: list[str] = field(default_factory=list)


def extract_from_kv_cache(
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Extract data from KV cache.

    Args:
        kv_cache: shape (num_pages, page_size, num_heads, head_size)
        slot_mapping: linear indices into flattened (pages * page_size) dim
        num_tokens: number of actual tokens (slot_mapping may be padded)

    Returns:
        Tensor of shape (num_tokens, num_heads, head_size)
    """
    return kv_cache.flatten(0, 1)[slot_mapping[:num_tokens]]


class MeanPoolHiddenStatesConnector(KVConnectorBase_V1):
    """
    Connector that mean-pools prompt hidden states and returns the
    resulting vector in the API response (via kv_transfer_params).

    Works with the extract_hidden_states speculative decoding method.
    This connector extracts those hidden states, mean-pools over prompt
    tokens, and saves the result for retrieval when the request finishes.
    """

    @property
    def prefer_cross_layer_blocks(self) -> bool:
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

        # Shared storage for passing mean-pooled vectors from worker to
        # scheduler (they run in separate processes with V1 multiprocessing).
        # Default to /dev/shm/vllm_mean_pool for lower latency on Linux,
        self._storage_path = self._kv_transfer_config.get_from_extra_config(
            "shared_storage_path",
            "/dev/shm/vllm_mean_pool",
        )
        os.makedirs(self._storage_path, exist_ok=True)

        self._num_heads: int = 0  # set in register_kv_caches
        self.cache_layers: list[str] = []

        # Which block_ids group index has the CacheOnlyAttentionLayer.
        self._kv_group_id: int = self._find_cache_group(kv_cache_config)

        # --- Scheduler-side state ---
        self._req_blocks: dict[str, list[int]] = {}
        self._prompt_lens: dict[str, int] = {}
        # Once we've sent metadata with blocks covering the full prompt,
        # the worker will pool on that step.  We stop including the request
        # in subsequent metadata to avoid O(N) per-step overhead.
        self._pooling_metadata_sent: set[str] = set()
        # Buffer of request IDs that have finished since the last
        # build_connector_meta call, drained into metadata so the
        # worker can clean up _completed_pools.
        self._pending_finished_ids: list[str] = []

        # --- Worker-side state ---
        # Requests that have completed mean pooling on the worker.
        self._completed_pools: set[str] = set()
        self._kv_device: torch.device | None = None

    @staticmethod
    def _find_cache_group(
        kv_cache_config: Optional["KVCacheConfig"],
    ) -> int:
        """Find which block_ids group holds the CacheOnlyAttentionLayer."""
        if kv_cache_config is None:
            return 0
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            for layer_name in group.layer_names:
                if "cache_only" in layer_name.lower():
                    return gid
        raise ValueError(
            "MeanPoolHiddenStatesConnector requires a CacheOnlyAttentionLayer "
            "but none was found in kv_cache_groups. Ensure "
            "--speculative-config with method='extract_hidden_states' is set."
        )

    # ==============================
    # Worker-side methods
    # ==============================
    def start_load_kv(self, *args, **kwargs: Any) -> None:
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def wait_for_save(self):
        pass

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        from vllm.model_executor.models.extract_hidden_states import (
            CacheOnlyAttentionLayer,
        )

        layers = get_layers_from_vllm_config(
            self._vllm_config, CacheOnlyAttentionLayer, list(kv_caches.keys())
        )
        self.cache_layers = list(layers.keys())
        assert len(self.cache_layers) == 1, (
            f"Expected 1 CacheOnlyAttentionLayer, got {len(self.cache_layers)}"
        )

        layer_name = self.cache_layers[0]
        kv_cache = kv_caches[layer_name]
        # kv_cache shape: (num_pages, page_size, num_heads, head_size)
        self._num_heads = kv_cache.shape[2]
        self._kv_device = kv_cache.device
        logger.info(
            "MeanPoolHiddenStatesConnector: pooling %d hidden state layer(s) "
            "on device %s",
            self._num_heads,
            self._kv_device,
        )

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        """Extract hidden states from KV cache, mean-pool prompt tokens,
        and save the result for each request."""
        if layer_name not in self.cache_layers:
            return

        from vllm.model_executor.models.extract_hidden_states import (
            CacheOnlyAttentionMetadata,
        )

        assert isinstance(attn_metadata, CacheOnlyAttentionMetadata)

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, MeanPoolConnectorMetadata)

        # Fast path: no requests to process (common during pure decode).
        if not connector_metadata.requests:
            return

        for request in connector_metadata.requests:
            # Skip if already pooled (avoid re-computing on every
            # decode step — only pool once after prefill completes).
            if request.req_id in self._completed_pools:
                continue

            prompt_len = request.prompt_len
            num_slots = len(request.slot_mapping)

            # Only mean-pool when all prompt tokens are in the cache.
            # With chunked prefill, early steps have fewer blocks.
            if num_slots < prompt_len:
                continue

            slot_mapping = torch.tensor(
                request.slot_mapping, dtype=torch.int64, device=self._kv_device
            )

            # Extract hidden states for prompt tokens from KV cache
            # shape: [prompt_len, num_heads, head_size]
            hidden_states = extract_from_kv_cache(kv_layer, slot_mapping, prompt_len)

            # Mean-pool over prompt tokens (dim=0), independently per head.
            # [prompt_len, num_heads, head_size] -> [num_heads, head_size]
            pooled = hidden_states.mean(dim=0, dtype=torch.float32)

            # Single head: flatten [1, head_size] -> [head_size]
            if self._num_heads == 1:
                pooled = pooled.squeeze(0)

            # Save using torch binary format (faster than JSON).
            out_path = os.path.join(self._storage_path, f"{request.req_id}.pt")
            torch.save(pooled.cpu(), out_path)
            self._completed_pools.add(request.req_id)

        for req_id in connector_metadata.finished_req_ids:
            self._completed_pools.discard(req_id)

    # ==============================
    # Scheduler-side methods
    # ==============================
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ):
        assert num_external_tokens == 0

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = MeanPoolConnectorMetadata()

        # Drain finished request IDs so the worker can clean up.
        meta.finished_req_ids = self._pending_finished_ids
        self._pending_finished_ids = []

        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = new_req.prompt_token_ids or []
            prompt_len = len(token_ids)
            block_ids = new_req.block_ids[self._kv_group_id]

            meta.requests.append(
                _ReqMeta.make(
                    req_id=new_req.req_id,
                    prompt_len=prompt_len,
                    block_ids=block_ids,
                    block_size=self._block_size,
                )
            )
            self._req_blocks[new_req.req_id] = list(block_ids)
            self._prompt_lens[new_req.req_id] = prompt_len

            # If blocks already cover the full prompt, the worker will
            # pool on this step.  Mark so we skip on future steps.
            if len(block_ids) * self._block_size >= prompt_len:
                self._pooling_metadata_sent.add(new_req.req_id)

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            if req_id not in self._req_blocks:
                continue

            # Skip requests that have already had pooling metadata sent.
            if req_id in self._pooling_metadata_sent:
                continue

            new_block_ids = cached_reqs.new_block_ids[i]
            req_block_ids = self._req_blocks[req_id]
            prompt_len = self._prompt_lens[req_id]

            if new_block_ids is not None:
                req_block_ids.extend(new_block_ids[self._kv_group_id])

            if len(req_block_ids) * self._block_size < prompt_len:
                continue

            meta.requests.append(
                _ReqMeta.make(
                    req_id=req_id,
                    prompt_len=prompt_len,
                    block_ids=req_block_ids,
                    block_size=self._block_size,
                )
            )

            if len(req_block_ids) * self._block_size >= prompt_len:
                self._pooling_metadata_sent.add(req_id)

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Return mean-pooled hidden states in kv_transfer_params."""
        req_id = request.request_id
        self._req_blocks.pop(req_id, None)
        self._prompt_lens.pop(req_id, None)
        self._pooling_metadata_sent.discard(req_id)
        self._pending_finished_ids.append(req_id)

        # Read the mean-pooled tensor from shared storage.
        out_path = os.path.join(self._storage_path, f"{req_id}.pt")
        try:
            pooled = torch.load(out_path, weights_only=True, map_location="cpu")
            os.remove(out_path)
            return False, {"mean_pooled_hidden_states": pooled.tolist()}
        except FileNotFoundError:
            logger.warning(
                "Mean-pooled hidden states not found for request %s. ",
                req_id,
            )
            return False, None

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: "VllmConfig") -> str | None:
        return "NHD"
