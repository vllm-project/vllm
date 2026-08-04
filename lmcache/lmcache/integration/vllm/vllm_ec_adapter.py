# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Standard
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

# Third Party
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
import torch

# First Party
from lmcache import torch_device_type
from lmcache.integration.vllm.utils import (
    create_lmcache_ec_config,
    create_lmcache_metadata,
)
from lmcache.v1.ec_engine import ECCacheEngine

if TYPE_CHECKING:
    # Third Party
    from vllm.config import VllmConfig
    from vllm.distributed.ec_transfer.ec_connector.base import (
        ECConnectorBase,
        ECConnectorRole,
    )
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class MMMeta:
    mm_hash: str

    @staticmethod
    def make_meta(mm_hash: str) -> "MMMeta":
        """Create metadata for a single multimodal hash."""
        return MMMeta(mm_hash=mm_hash)


@dataclass
class LMCacheECConnectorMetadata(ECConnectorMetadata):
    mm_datas: list[MMMeta] = field(default_factory=list)

    def add_mm_data(self, mm_data: MMMeta) -> None:
        """Append one multimodal metadata entry."""
        self.mm_datas.append(mm_data)


class LMCacheECConnectorImpl:
    """Worker- and scheduler-side glue between vLLM's EC connector and LMCache.

    Bridges vLLM ``ECConnectorBase`` calls to an :class:`ECCacheEngine`: on the
    worker side it saves and loads encoder tensors, and on the scheduler side
    it tracks which multimodal hashes need to be loaded next step.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: "ECConnectorRole",
        parent: "ECConnectorBase",
    ) -> None:
        """Initialize the EC connector implementation.

        Args:
            vllm_config: vLLM engine configuration; ``ec_transfer_config`` and
                ``model_config`` must be set.
            role: vLLM EC connector role (worker or scheduler).
            parent: vLLM ``ECConnectorBase`` that owns this implementation;
                used to read connector metadata and producer/consumer state.

        Raises:
            ValueError: if ``vllm_config.ec_transfer_config`` is None, or if
                no LMCache storage backend is configured.
        """
        self._parent = parent

        # Scheduler-side state: set of multimodal hashes to load. Unused on
        # worker-side instances; kept here because vLLM's ECConnectorBase
        # multiplexes both roles onto a single class.
        self._mm_hashes_need_loads: set[str] = set()

        if vllm_config.ec_transfer_config is None:
            raise ValueError("ec_transfer_config must be set for ECConnectorBase")

        # Build EC config from standard LMCache config + EC-prefixed overrides.
        config = create_lmcache_ec_config()

        # EC scheduler-side instances must look like workers to LMCache:
        # ``has_cache_item`` calls ``engine.contains()``, which needs a fully
        # constructed StorageManager (including LocalCPUBackend, since the
        # local-disk backend is layered on top of it). LMCache's
        # ``CreateStorageBackends`` skips the CPU backend when
        # ``metadata.role == "scheduler"`` and then asserts on it for the
        # disk backend — passing role="scheduler" therefore aborts startup.
        # Until LMCache grows a scheduler-friendly storage path (or EC
        # splits scheduler/worker into separate engines), keep role pinned
        # to "worker" here regardless of the vLLM-side role.
        lmcache_metadata, _ = create_lmcache_metadata(vllm_config, role="worker")

        self._ec_engine = ECCacheEngine(
            config=config,
            metadata=lmcache_metadata,
            encoder_dtype=vllm_config.model_config.dtype,
        )

    # ------------------------------
    # Worker-side methods
    # ------------------------------

    def start_load_caches(
        self,
        encoder_cache: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> None:
        """Load needed encoder caches from LMCache into vLLM encoder_cache."""
        # vLLM's ECConnectorBase exposes connector metadata only via the
        # underscored accessor; matches the pattern used in vllm_v1_adapter.py
        # for the KV connector.
        metadata = self._parent._get_connector_metadata()  # noqa: SLF001
        if metadata is None:
            logger.warning(
                "In connector.start_load_caches, but the connector metadata is None"
            )
            return
        if not isinstance(metadata, LMCacheECConnectorMetadata):
            raise TypeError(f"Unexpected metadata type: {type(metadata)}")

        device = torch_device_type
        for mm_data in metadata.mm_datas:
            mm_hash = mm_data.mm_hash
            if mm_hash in encoder_cache:
                # vLLM already has it; don't overwrite.
                continue
            tensor = self._ec_engine.get(mm_hash, device)
            if tensor is None:
                continue
            encoder_cache[mm_hash] = tensor
            logger.debug("Loaded encoder cache for hash %s", mm_hash)

    def save_caches(
        self,
        encoder_cache: dict[str, torch.Tensor],
        mm_hash: str,
        **kwargs: Any,
    ) -> None:
        """Save one encoder cache entry from vLLM into LMCache."""

        if not getattr(self._parent, "is_producer", False):
            return

        if mm_hash not in encoder_cache:
            return

        did_store = self._ec_engine.put(mm_hash, encoder_cache[mm_hash])
        if did_store:
            logger.debug("Saved encoder cache for mm_hash %s", mm_hash)

    # ------------------------------
    # Scheduler-side methods
    # ------------------------------

    def has_cache_item(self, identifier: str) -> bool:
        """Return whether LMCache already contains the encoder cache for hash."""
        return self._ec_engine.contains(identifier)

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        """Track which multimodal item (request.mm_features[index]) needs loading."""
        mm_hash = request.mm_features[index].identifier
        self._mm_hashes_need_loads.add(mm_hash)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ECConnectorMetadata:
        """Build worker-load metadata for hashes queued this scheduler step."""
        _ = scheduler_output
        meta = LMCacheECConnectorMetadata()
        for mm_hash in sorted(self._mm_hashes_need_loads):
            meta.add_mm_data(MMMeta.make_meta(mm_hash))
        self._mm_hashes_need_loads.clear()
        return meta

    # ------------------------------
    # Helpers
    # ------------------------------

    def close(self) -> None:
        """Release EC engine resources."""
        if hasattr(self, "_ec_engine") and self._ec_engine is not None:
            self._ec_engine.close()
