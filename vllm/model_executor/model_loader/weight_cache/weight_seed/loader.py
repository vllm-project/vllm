# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Peer model loader: fills a weight cache daemon's model with the
post-processed weights of a source replica's daemon instead of reading the
checkpoint.

Set internally by ``vllm preload --weight-cache-seed``: a mirror daemon is an
ordinary weight cache daemon whose ``load_format`` is ``peer_cache``. It
builds its model on the meta device like any other daemon (own world group,
config finalization, warmup), then fetches this rank's shard from the source
daemon through the ``weights`` seedable: the fingerprint and metadata
travel on the D2D seed transport, and the weights themselves move through a
mover — ``peer_ipc`` copies through CUDA IPC handles on one host, ``rdma``
pulls through the Mooncake TransferEngine
across hosts. The mover always copies into memory this process owns, so the
source can release independently.

There is no disk fallback: a mirror host may not have the checkpoint, so any
fetch, fingerprint or transfer failure is fatal.
"""

from copy import copy

import torch
import torch.nn as nn

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import (
    build_model_from_processed_weights,
)
from vllm.model_executor.model_loader.weight_cache.protocol import (
    TensorKind,
    WeightCacheKey,
    check_weight_cache_loader_support,
)
from vllm.model_executor.model_loader.weight_cache.transport import (
    SeedConfig,
    SeedTransport,
)
from vllm.model_executor.model_loader.weight_cache.utils import (
    build_weight_cache_key,
)
from vllm.model_executor.model_loader.weight_cache.weight_seed.core import (
    PEER_IPC_BACKEND,
    RDMA_BACKEND,
    SeedBackend,
    get_mover,
    parse_seed_backend,
)
from vllm.model_executor.model_loader.weight_cache.weight_seed.utils import (
    WeightsSeedPayload,
    validate_weights_payload,
)
from vllm.platforms import current_platform
from vllm.tracing import instrument

logger = init_logger(__name__)


class PeerModelLoader(BaseModelLoader):
    """Loads a daemon's model from a source replica's weight cache daemon.

    The payload the source resolves (see ``WeightShardSeed.resolve``) is a
    JSON-safe storage manifest plus a mover seed; only the fingerprint and
    metadata travel on the control plane, the weights themselves move through
    the mover named by ``seed["backend"]``.

    The loading daemon injects its own D2D endpoint (`loader.transport =
    self.d2d`), so the fetch is daemon-to-daemon over the same connection the
    warmup caches use. Standalone construction falls back to extra config keys
    (via --model-loader-extra-config):

    - peer_source: the source daemon's address — a Unix socket path for a
      same-host source, or a [host, port] list for a remote one.
    - peer_backend: "peer_ipc" or "rdma"; inferred from the source form
      when unset (Unix socket → peer_ipc, remote → rdma).
    - peer_token: shared token authenticating a remote source (None for Unix
      sockets, which are owner-verified).
    - is_draft: internal flag routing to the source's draft daemon group.
    """

    loads_pre_processed_weights = True

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra_config = copy(load_config.model_loader_extra_config or {})
        source = extra_config.pop("peer_source", None)
        backend = extra_config.pop("peer_backend", None)
        self.peer_token: str | None = extra_config.pop("peer_token", None)
        # Internal: set by the launcher when routing a speculative draft to
        # the source's draft daemon group.
        self.is_draft = bool(extra_config.pop("is_draft", False))
        if extra_config:
            raise ValueError(
                f"Unexpected extra config keys for load format "
                f"{load_config.load_format}: {sorted(extra_config)}"
            )
        if isinstance(source, list):
            if len(source) != 2 or not isinstance(source[0], str):
                raise ValueError(
                    f"A remote peer_source must be a [host, port] list, got {source!r}"
                )
            source = (source[0], int(source[1]))
        if source is not None and not isinstance(source, (str, tuple)):
            raise ValueError(
                "peer_source must be a Unix socket path or a [host, port] "
                f"list, got {source!r}"
            )
        self.peer_source: str | tuple[str, int] | None = source
        # The daemon injects its own D2D endpoint (get_model); standalone
        # construction (tests, engine-side) builds a client from the extra
        # config instead.
        self.transport: SeedTransport | None = None
        if backend is None and source is not None:
            # The address form already decides the mover: a Unix socket is
            # same-host CUDA IPC, a remote host:port pulls over Mooncake.
            backend = PEER_IPC_BACKEND if isinstance(source, str) else RDMA_BACKEND
        elif backend is not None:
            backend = parse_seed_backend(backend)
            if (
                backend == PEER_IPC_BACKEND
                and source is not None
                and not isinstance(source, str)
            ):
                raise ValueError(
                    f"{PEER_IPC_BACKEND} copies through CUDA IPC, which "
                    "cannot cross hosts; a [host, port] source uses "
                    f"{RDMA_BACKEND}"
                )
        self.peer_backend: SeedBackend | None = backend

    def download_model(self, model_config: ModelConfig) -> None:
        # Nothing to download: the weights come from the source daemon, and
        # the config files meta-model construction needs resolve through
        # ModelConfig as for any other load format.
        return None

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        raise NotImplementedError(
            "Standalone weight reload is not supported for load format 'peer_cache' yet"
        )

    @instrument(span_name="Load model")
    def load_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        prefix: str = "",
        *,
        payload: WeightsSeedPayload | None = None,
    ) -> nn.Module:
        check_weight_cache_loader_support(
            vllm_config, model_config, is_draft=self.is_draft
        )
        expected_key = build_weight_cache_key(
            vllm_config, model_config, is_draft=self.is_draft
        )
        if payload is None:
            # Standalone load (no injecting daemon): fetch over our own
            # client built from the extra config.
            payload = self._fetch_standalone(expected_key, model_config.model)
        manifest = payload["manifest"]
        mover = get_mover(payload["seed"]["backend"])
        device = torch.device(
            current_platform.device_type, torch.accelerator.current_device_index()
        )
        tensors = mover.fill(manifest, payload["seed"], device)
        kinds: dict[str, TensorKind] = {
            name: "param" if view["is_param"] else "buffer"
            for name, view in manifest["tensors"].items()
        }
        model = build_model_from_processed_weights(
            vllm_config,
            model_config,
            prefix,
            tensors,
            kinds,
            aliases=dict(payload["aliases"]),
            attrs=dict(payload["attrs"]),
            load_config=self.load_config,
        )
        logger.info(
            "Loaded %d tensors from a peer daemon over %s",
            len(tensors),
            payload["seed"]["backend"],
        )
        return model

    def _fetch_standalone(
        self, expected_key: WeightCacheKey, model: str
    ) -> WeightsSeedPayload:
        """Fetch and validate this rank's shard without an injecting daemon."""
        if self.peer_source is None:
            raise RuntimeError(
                "Load format 'peer_cache' needs a peer source: injected by "
                "the loading daemon, or 'peer_source' in "
                "model_loader_extra_config (set internally by "
                "`vllm preload --weight-cache-seed`)"
            )
        transport = SeedTransport(
            seed_config=SeedConfig(source=self.peer_source, token=self.peer_token)
        )
        backend = self.peer_backend
        if backend is None:
            # The address form already decides the mover: a Unix socket is
            # same-host CUDA IPC, a remote host:port pulls over Mooncake.
            backend = (
                PEER_IPC_BACKEND if isinstance(self.peer_source, str) else RDMA_BACKEND
            )
        response = transport.request({"kind": "weights", "seed_backend": backend})
        if response is None:
            raise RuntimeError(
                f"Could not fetch the weights seed from {self.peer_source}; "
                "the transport logged the reason"
            )
        if response.get("status") != "ok":
            raise RuntimeError(
                f"Source daemon refused the weights seed request: "
                f"{response.get('message', response)}"
            )
        return validate_weights_payload(response.get("payload"), expected_key, model)
