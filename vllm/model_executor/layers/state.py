# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from torch import nn

from vllm.config import get_current_vllm_config
from vllm.v1.kv_cache_interface import KVCacheLayerConfig, PersistentStateSpec


class PersistentStateLayerBase(nn.Module):
    """Base class for layers that hold fixed-size persistent per-session state."""

    def __init__(
        self,
        spec: PersistentStateSpec,
        *,
        prefix: str,
        vllm_config: Any | None = None,
    ) -> None:
        super().__init__()
        config = vllm_config or get_current_vllm_config()
        model_config = config.model_config
        if model_config is not None and not model_config.has_persistent_state:
            # Configuration applies the persistent-state defaults only to a
            # declared model, so an undeclared state layer fails here.
            raise ValueError(
                f"{model_config.architecture} builds a persistent-state layer, "
                "so its class must set supports_persistent_state = True"
            )
        if not prefix:
            raise ValueError("persistent-state layer prefix must be nonempty")
        self._persistent_state_spec = spec
        forward_context = config.compilation_config.static_forward_context
        if prefix in forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        forward_context[prefix] = self
        self.prefix = prefix

    def get_kv_cache_config(self, vllm_config: Any) -> KVCacheLayerConfig:
        from vllm.v1.core.state_slot_manager import PersistentStateManager

        return PersistentStateManager.layer_config(self._persistent_state_spec)
