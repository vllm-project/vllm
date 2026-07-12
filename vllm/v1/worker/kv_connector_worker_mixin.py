# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Define KV connector host-contract mixin for worker classes.

The worker half of the device-neutral KV-connector host contract. A worker
initializes the configured KV-transfer connector (building the worker-side
connector agent) *before* the model runner allocates its KV cache, mirroring
what ``GPUWorker.initialize_from_config`` does inline. Any worker (GPU, TPU, or
an out-of-tree backend) can mix this in and call ``maybe_initialize_kv_transfer``
so the same lifecycle runs without duplicating connector logic. It is a no-op
when no ``kv_transfer_config`` is set and contains no device- or
connector-specific knowledge.
"""

from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_initialized,
    get_kv_transfer_group,
    has_kv_transfer_group,
)
from vllm.distributed.parallel_state import get_pp_group, get_tp_group

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorHandshakeMetadata,
    )
    from vllm.v1.kv_cache_interface import KVCacheConfig


class KVConnectorWorkerMixin:
    @staticmethod
    def maybe_initialize_kv_transfer(
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        """Initialize the KV-transfer connector for this worker, if configured.

        Must be called before the model runner's ``initialize_kv_cache`` so the
        worker-side connector agent exists when the runner registers its KV
        caches. No-op without a ``kv_transfer_config``.
        """
        ensure_kv_transfer_initialized(vllm_config, kv_cache_config)

    @staticmethod
    def get_kv_connector_handshake_metadata() -> (
        "dict[tuple[int, int], KVConnectorHandshakeMetadata] | None"
    ):
        """Return this worker's KV-connector handshake metadata, if any.

        Generic worker host-contract method (mirrors ``GPUWorker``): the engine
        core collects handshake metadata from every worker via ``collective_rpc``
        after KV setup. Keyed by ``(pp_rank, tp_rank)``. Returns ``None`` when no
        connector is configured or the connector needs no handshake exchange.
        Device- and connector-agnostic.
        """
        if not has_kv_transfer_group():
            return None
        connector = get_kv_transfer_group()
        metadata = connector.get_handshake_metadata()
        if metadata is None:
            return None
        pp_rank = get_pp_group().rank_in_group
        tp_rank = get_tp_group().rank_in_group
        return {(pp_rank, tp_rank): metadata}
