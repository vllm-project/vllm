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
    check_weight_cache_loader_support,
)
from vllm.model_executor.model_loader.weight_cache.transport import SeedTransport
from vllm.model_executor.model_loader.weight_cache.weight_seed.core import (
    acknowledge_lease,
    get_mover,
)
from vllm.model_executor.model_loader.weight_cache.weight_seed.utils import (
    WeightsSeedPayload,
)
from vllm.platforms import current_platform
from vllm.tracing import instrument

logger = init_logger(__name__)


class PeerModelLoader(BaseModelLoader):
    """Loads a daemon's model from a source replica's weight cache daemon."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra_config = copy(load_config.model_loader_extra_config or {})
        self.is_draft = bool(extra_config.pop("is_draft", False))
        if extra_config:
            raise ValueError(
                f"Unexpected extra config keys for load format "
                f"{load_config.load_format}: {sorted(extra_config)}"
            )

    def download_model(self, model_config: ModelConfig) -> None:
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
        payload: WeightsSeedPayload,
        transport: SeedTransport,
    ) -> nn.Module:
        check_weight_cache_loader_support(
            vllm_config, model_config, is_draft=self.is_draft
        )
        manifest = payload["manifest"]
        mover = get_mover(payload["seed"]["backend"])
        device = torch.device(
            current_platform.device_type, torch.accelerator.current_device_index()
        )
        tensors = mover.fill(manifest, payload["seed"], device)
        acknowledge_lease(transport, payload)
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
