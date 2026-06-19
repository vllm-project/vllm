# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSv4 MTP draft helpers.

Two tiny stand-alone symbols vendored from upstream:

  * ``SharedHead`` — vendored from
    ``vllm/model_executor/models/deepseek_mtp.py``.
  * ``get_spec_layer_idx_from_weight_name`` — vendored from
    ``vllm/model_executor/models/deepseek_v2.py``.

Both are pure data definitions with no hardware dispatch — they were
forbidden purely by the broad ``model_executor.models.*`` lint scope.
The generic ``SupportsPP`` Protocol that previously lived alongside
these now lives in ``shared/interfaces.py``.
"""

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from ..shared.layers.layernorm import RMSNorm
from ..shared.layers.vocab_parallel_embedding import ParallelLMHead

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )
else:
    QuantizationConfig = object


def _maybe_prefix(prefix: str, name: str) -> str:
    """Local copy of ``vllm.model_executor.models.utils.maybe_prefix``.

    The upstream helper is allowed by the lint carve-out for
    ``models.utils``, but importing it would create a circular import
    at module-load time on some runs. Inlined here.
    """
    return name if not prefix else f"{prefix}.{name}"


class SharedHead(nn.Module):
    """Vendored from
    ``vllm.model_executor.models.deepseek_mtp.SharedHead``.

    Vocab-projection head shared across the MTP draft layers; combines
    a final RMSNorm with a ``ParallelLMHead``. The MTP draft model
    constructs one ``SharedHead`` per draft layer.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        quant_config: "QuantizationConfig | None" = None,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=_maybe_prefix(prefix, "head"),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


def get_spec_layer_idx_from_weight_name(
    config: PretrainedConfig, weight_name: str
) -> int | None:
    """Vendored from
    ``vllm.model_executor.models.deepseek_v2.get_spec_layer_idx_from_weight_name``.

    Returns the global layer index of an MTP/spec-decode weight name,
    or ``None`` when the name does not belong to a spec layer.
    """
    if (
        hasattr(config, "num_nextn_predict_layers")
        and config.num_nextn_predict_layers > 0
    ):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            if weight_name.startswith(
                f"model.layers.{layer_idx + i}."
            ) or weight_name.startswith(f"layers.{layer_idx + i}."):
                return layer_idx + i
    return None
