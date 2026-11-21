# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable

import regex as re
import torch

from vllm.config import SpeculativeConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)

_CONTEXT_WKV_RE = re.compile(r"^mtp\.(\d+)\.attn\.wkv\.(.+)$")


def needs_mtp_hidden_buffer(spec_config: SpeculativeConfig | None) -> bool:
    """Whether the drafter reads the target's pre-hc_head residual buffer, which
    stands in for its final hidden states (aux-state drafters like DSpark don't).
    """
    return (
        spec_config is not None
        and spec_config.uses_target_hidden_states
        and not spec_config.uses_aux_hidden_states
    )


def context_wkv_quant_config(
    quant_config: QuantizationConfig | None, wkv_prefix: str
) -> QuantizationConfig | None:
    """Quant config of the DSpark stacked context WKV projection: `None` when
    the checkpoint leaves the draft WKV at `wkv_prefix` unquantized."""
    ignored_layers = getattr(quant_config, "ignored_layers", None)
    if (
        quant_config is not None
        and ignored_layers
        and hasattr(quant_config, "ignored_layers_match_mode")
        and is_layer_skipped(
            prefix=wkv_prefix,
            ignored_layers=ignored_layers,
            fused_mapping=quant_config.packed_modules_mapping,
            match_mode=quant_config.ignored_layers_match_mode,
        )
    ):
        return None
    return quant_config


def duplicate_context_wkv_weights(
    weights: Iterable[tuple[str, torch.Tensor]], num_layers: int
) -> Iterable[tuple[str, torch.Tensor]]:
    """Load every draft layer's WKV into the cross-layer projection."""
    for name, weight in weights:
        yield name, weight
        match = _CONTEXT_WKV_RE.fullmatch(name)
        if match is None:
            continue
        layer_idx = int(match.group(1))
        if layer_idx >= num_layers:
            continue
        stacked_weight = weight.detach()
        stacked_weight.shard_id = layer_idx
        yield f"context_wkv_proj.{match.group(2)}", stacked_weight
