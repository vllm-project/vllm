# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fill a daemon's model with a source replica's post-processed weights."""

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
    acknowledge_lease,
    get_mover,
    parse_seed_backend,
)
from vllm.model_executor.model_loader.weight_cache.weight_seed.seed import (
    validate_weights_payload_or_reject,
)
from vllm.model_executor.model_loader.weight_cache.weight_seed.utils import (
    WeightsSeedPayload,
)
from vllm.platforms import current_platform
from vllm.tracing import instrument

logger = init_logger(__name__)


class PeerModelLoader(BaseModelLoader):
    """Loads a daemon's model from a source replica's weight cache daemon."""

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
        transport: SeedTransport | None = None,
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
        transport = transport or self.transport
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
        acknowledge_lease(transport, payload, copy_succeeded=True)
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
        self.transport = transport
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
        payload = response.get("payload")
        return validate_weights_payload_or_reject(
            transport, payload, expected_key, model
        )
