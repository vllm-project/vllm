# SPDX-License-Identifier: Apache-2.0
"""Fan-out of connector/adapter lifecycle hooks to active experimental features.

The connector and worker adapter know only this dispatcher; the dispatcher
knows the features; each feature knows the SDK. Adding a tensor type never
touches the connector or the adapter -- see ``registry.py``.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

# First Party
from lmcache.sdk.cache_kind import LMCacheSDKCacheKind
from lmcache.sdk.qringbuffer import QRingBufferAdapter, QRingBufferCapture
from lmcache.utils import init_logger
from lmcache.v1.multiprocess.modules.experimental import TRANSFER_QUERY

if TYPE_CHECKING:
    # Third Party
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig
    import torch

    # First Party
    from lmcache.integration.vllm.lmcache_mp_connector import (
        LMCacheMPConnectorMetadata,
    )
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        LMCacheMPWorkerAdapter,
        _IpcEvent,
    )

logger = init_logger(__name__)


class QTensorFeature:
    """Captures per-layer query tensors into a paged-Q ring and stores them."""

    def __init__(self, ctx: FeatureContext) -> None:
        self._q_ring_adapter = QRingBufferAdapter(
            adapter=ctx.worker_adapter,
            q_model_name=LMCacheSDKCacheKind.QUERY.server_model_name(
                ctx.worker_adapter.model_name
            ),
            send_lmcache_request=ctx.send_lmcache_request,
        )
        self._capture = QRingBufferCapture(ctx.worker_adapter, self._q_ring_adapter)

    def register(
        self,
        kv_caches: dict[str, torch.Tensor],
        kv_cache_config: KVCacheConfig | None,
        vllm_config: VllmConfig,
    ) -> None:
        self._capture.setup_q_ring(kv_caches, kv_cache_config, vllm_config)

    def save_kv_layer(
        self,
        layer_name: str,
        metadata: LMCacheMPConnectorMetadata,
        **layer_io: Any,
    ) -> None:
        self._capture.save_q_layer(layer_name, metadata, **layer_io)

    def wait_for_save(self, event: _IpcEvent | None) -> None:
        self._capture.batched_submit_qstore_requests(event=event)

    def reclaim(self) -> None:
        self._q_ring_adapter.reclaim_finished_q_stores()

    def reregister(self) -> bool:
        try:
            self._q_ring_adapter.reregister_q_ring()
        except ConnectionError:
            logger.exception(
                "Failed to re-register experimental tensors after server "
                "recovery; will retry on next heartbeat"
            )
            return False
        except Exception:
            logger.exception(
                "Unexpected error during experimental tensor "
                "re-registration; will retry on next heartbeat"
            )
            return False
        logger.warning(
            "Finished re-registering experimental tensors after server recovery"
        )
        return True

    def shutdown(self) -> None:
        self._q_ring_adapter.shutdown_q_ring()


@dataclass
class FeatureContext:
    """
    Collection of arguments needed for each feature's initialization.
    """

    worker_adapter: LMCacheMPWorkerAdapter
    send_lmcache_request: Callable[..., Any]


FeatureFactory = Callable[[FeatureContext], "QTensorFeature"]

FEATURE_REGISTRY: dict[str, FeatureFactory] = {
    TRANSFER_QUERY: QTensorFeature,
}


def init_dispatcher(
    ctx: FeatureContext,
    requested: set[str],
) -> Dispatcher:
    """Build the dispatcher for intermediate tensors both the server and
    connector agree to transfer.

    Args:
        ctx: The feature context.
        requested: The set of experimental features from vllm_config.

    Returns:
        A dispatcher for all experimental features built in the server.

    Raises:
        ValueError: If a requested capability is not advertised by the server.
    """
    experimental = ctx.worker_adapter.experimental
    features: list[QTensorFeature] = []
    for name, factory in FEATURE_REGISTRY.items():
        if name not in requested:
            continue
        if name not in experimental:
            raise ValueError(f"Connector enables {name} but server does not.")
        features.append(factory(ctx))
    return Dispatcher(features)


def dispatch(
    dispatcher: Dispatcher | None,
    fn_name: str,
    **kwargs: Any,
) -> None:
    """Dispatch the given arguments to the dispatcher.

    Args:
        dispatcher: The dispatcher to dispatch to.
        **kwargs: The arguments to dispatch.
    """
    if dispatcher is None:
        return
    getattr(dispatcher, fn_name)(**kwargs)


class Dispatcher:
    """Holds the experimental features built into the server and connector, and
    fans out lifecycle hooks to all of them. The connector and adapter doesn't
    need to distinguish the types of intermediate tensors. Dispatcher holds the
    TensorFeature which talks to the SDK.

    Args:
        features: The active features, in fan-out order.
    """

    def __init__(self, features: list[QTensorFeature]) -> None:
        self._features = features

    def register(
        self,
        kv_caches: dict[str, torch.Tensor],
        kv_cache_config: KVCacheConfig | None,
        vllm_config: VllmConfig,
    ) -> None:
        """Fan out ``register`` (connector ``register_kv_caches``)."""
        for feature in self._features:
            feature.register(kv_caches, kv_cache_config, vllm_config)

    def save_kv_layer(
        self,
        layer_name: str,
        metadata: LMCacheMPConnectorMetadata,
        **layer_io: Any,
    ) -> None:
        """Fan out ``save_kv_layer`` (connector ``save_kv_layer``; hot path)."""
        for feature in self._features:
            feature.save_kv_layer(layer_name, metadata, **layer_io)

    def wait_for_save(self, event: _IpcEvent | None) -> None:
        """Fan out ``wait_for_save`` (connector ``wait_for_save``)."""
        for feature in self._features:
            feature.wait_for_save(event)

    def reclaim(self) -> None:
        """Fan out ``reclaim`` (adapter ``get_finished``)."""
        for feature in self._features:
            feature.reclaim()

    def reregister(self) -> bool:
        """Fan out ``reregister`` (adapter heartbeat recovery)."""
        for feature in self._features:
            if not feature.reregister():
                return False
        return True

    def shutdown(self) -> None:
        """Fan out ``shutdown`` (adapter ``shutdown``)."""
        for feature in self._features:
            feature.shutdown()
