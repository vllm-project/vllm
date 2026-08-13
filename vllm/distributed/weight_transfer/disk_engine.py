# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared-disk weight transfer backend."""

import copy
import os
from collections.abc import Generator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.config.load import LoadConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)
from vllm.logger import init_logger
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
)

if TYPE_CHECKING:
    from torch import nn

    from vllm.config import ModelConfig, VllmConfig

logger = init_logger(__name__)


class _PrimaryOnlyModelLoader(DefaultModelLoader):
    """Load only the checkpoint selected for the active update."""

    def get_all_weights(
        self,
        model_config: "ModelConfig",
        model: "nn.Module",
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        del model
        source = self.Source(
            model_or_path=model_config.model,
            revision=model_config.revision,
            fall_back_to_pt=False,
            allow_patterns_overrides=["*.safetensors"],
        )
        yield from self._get_weights_iterator(source)


@dataclass
class DiskWeightTransferInitInfo(WeightTransferInitInfo):
    """Worker-side initialization info for shared-disk weight updates."""


@dataclass
class DiskWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """A local safetensors checkpoint visible to every inference worker."""

    path: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("`path` must be a non-empty absolute local directory")
        if not os.path.isabs(self.path):
            raise ValueError("`path` must be an absolute local directory")

        self.path = os.path.abspath(self.path)
        if not os.path.isdir(self.path):
            raise ValueError(
                f"Shared-disk checkpoint path must be a local directory: {self.path}"
            )
        with os.scandir(self.path) as entries:
            has_safetensors = any(
                entry.name.endswith(".safetensors")
                for entry in entries
                if entry.is_file()
            )
        if not has_safetensors:
            raise ValueError(
                "Shared-disk checkpoint directory contains no safetensors files: "
                f"{self.path}"
            )


class DiskWeightTransferEngine(
    WeightTransferEngine[DiskWeightTransferInitInfo, DiskWeightTransferUpdateInfo]
):
    """Reload checkpoint-format weights from a worker-local shared directory."""

    init_info_cls = DiskWeightTransferInitInfo
    update_info_cls = DiskWeightTransferUpdateInfo

    def __init__(
        self,
        config: WeightTransferConfig,
        vllm_config: "VllmConfig",
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, vllm_config, device, model)
        self._session_active = False
        self._checkpoint_loaded = False

    def init_transfer_engine(self, init_info: DiskWeightTransferInitInfo) -> None:
        """Initialize the backend. Shared-disk loading needs no rendezvous."""

    def start_weight_update(self) -> None:
        """Prepare the active model for one checkpoint reload."""
        if self._session_active:
            raise RuntimeError("A shared-disk weight update is already active")
        self._session_active = True
        self._checkpoint_loaded = False
        try:
            initialize_layerwise_reload(self.model)
        except Exception:
            self._abort_session()
            raise

    def finish_weight_update(self) -> None:
        """Finalize processing for the reloaded checkpoint."""
        if not self._session_active:
            raise RuntimeError("No shared-disk weight update is active")
        if not self._checkpoint_loaded:
            self._abort_session()
            raise RuntimeError(
                "No checkpoint was loaded during the shared-disk update session"
            )
        try:
            finalize_layerwise_reload(self.model, self.model_config)
        finally:
            self._session_active = False
            self._checkpoint_loaded = False

    def update_weights(self, update_info: dict[str, Any]) -> None:
        """Validate and load a checkpoint, cleaning up failed sessions."""
        try:
            super().update_weights(update_info)
        except Exception:
            self._abort_session()
            raise

    def receive_weights(self, update_info: DiskWeightTransferUpdateInfo) -> None:
        """Load one complete safetensors checkpoint from shared local storage."""
        if not self._session_active:
            raise RuntimeError(
                "start_weight_update must be called before loading a checkpoint"
            )
        if self._checkpoint_loaded:
            raise RuntimeError(
                "The disk backend accepts exactly one checkpoint per update session"
            )

        model_config = copy.copy(self.model_config)
        model_config.model = update_info.path
        model_config.revision = None
        load_config = LoadConfig(
            load_format="safetensors",
            model_loader_extra_config={"enable_weights_track": True},
        )
        loader = _PrimaryOnlyModelLoader(load_config)

        try:
            loader.load_weights(self.model, model_config)
        except Exception:
            self._abort_session()
            raise

        self._checkpoint_loaded = True

    def shutdown(self) -> None:
        """Release layerwise wrappers if shutdown interrupts an update."""
        self._abort_session()

    def _abort_session(self) -> None:
        if not self._session_active:
            return
        try:
            finalize_layerwise_reload(self.model, self.model_config)
        except Exception:
            logger.exception("Failed to clean up shared-disk weight update")
        finally:
            self._session_active = False
            self._checkpoint_loaded = False
