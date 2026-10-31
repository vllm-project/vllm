# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from torch import nn

from vllm.config import get_current_vllm_config
from vllm.v1.kv_cache_interface import PersistentStateSpec


class PersistentStateLayerBase(nn.Module):
    """Purpose-named model declaration, independent of attention and Mamba."""

    def __init__(
        self,
        spec: PersistentStateSpec,
        *,
        prefix: str,
        vllm_config: Any | None = None,
    ) -> None:
        super().__init__()
        config = vllm_config or get_current_vllm_config()
        if not config.use_v2_model_runner:
            raise ValueError("Persistent state layers require Model Runner V2")
        if not prefix:
            raise ValueError("persistent-state layer prefix must be nonempty")
        self._persistent_state_spec = spec
        forward_context = config.compilation_config.static_forward_context
        if prefix in forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        forward_context[prefix] = self
        self.prefix = prefix

    def get_kv_cache_spec(self, vllm_config: Any) -> PersistentStateSpec:
        del vllm_config
        return self._persistent_state_spec
