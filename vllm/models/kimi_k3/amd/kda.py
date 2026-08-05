# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""AMD Kimi-K3 KDA integration."""

import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    KimiGatedDeltaNetAttention as _KimiGatedDeltaNetAttention,
)
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

from .ops.kda_input_projection import (
    kda_input_projection,
    prepack_kda_input_group64,
)


class KimiGatedDeltaNetAttention(_KimiGatedDeltaNetAttention):
    """Kimi GDN layer with the gfx950 group64 input projection."""

    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(config, vllm_config, prefix)
        self.register_buffer("_kda_group64_weight", None, persistent=False)
        self.register_buffer("_kda_group64_scale", None, persistent=False)

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        del act_dtype
        weight = getattr(self.in_proj_qkvgfab, "weight", None)
        if not isinstance(weight, torch.Tensor):
            return
        packed = prepack_kda_input_group64(weight)
        if packed is not None:
            self._kda_group64_weight, self._kda_group64_scale = packed

    def _project_qkvgfab(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return kda_input_projection(
            hidden_states,
            super()._project_qkvgfab,
            self._kda_group64_weight,
            self._kda_group64_scale,
        )


__all__ = ["KimiGatedDeltaNetAttention"]
