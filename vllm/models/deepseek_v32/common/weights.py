# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint name mapping for DSA models loaded by `AutoWeightsLoader`."""

from collections.abc import Iterable, Iterator

import torch

from vllm.model_executor.models.deepseek_v2 import _dequantize_fp8_indexer_wk
from vllm.model_executor.models.utils import WeightsMapper, is_pp_missing_parameter

# Routed experts (`experts.<e>.*_proj` or the stacked `experts.{gate_up,down}_proj`)
# are left to `RoutedExperts.load_weights`.
DSA_WEIGHTS_MAPPER = WeightsMapper(
    orig_to_new_substr={
        # The MLA attention is `self_attn` itself. The quant config maps KV-cache
        # scales onto a `self_attn.attn` that DSA does not have; they were never
        # loaded, so keep dropping them.
        ".self_attn.attn.": None,
    },
    orig_to_new_stacked={
        ".mlp.gate_proj.": (".mlp.gate_up_proj.", 0),
        ".mlp.up_proj.": (".mlp.gate_up_proj.", 1),
        ".shared_experts.gate_proj.": (".shared_experts.gate_up_proj.", 0),
        ".shared_experts.up_proj.": (".shared_experts.gate_up_proj.", 1),
        ".q_a_proj.": (".fused_qkv_a_proj.", 0),
        ".kv_a_proj_with_mqa.": (".fused_qkv_a_proj.", 1),
        ".indexer.wk.": (".indexer.wk_weights_proj.", 0),
        ".indexer.weights_proj.": (".indexer.wk_weights_proj.", 1),
    },
)


def dequantize_fp8_indexer_wk_pairs(
    weights: Iterable[tuple[str, torch.Tensor]], model: torch.nn.Module
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield each FP8 indexer `wk` weight in BF16, the dtype of the fused
    `wk_weights_proj`, once its scale has arrived.

    Reordering loaders (e.g. runai_streamer) can split a pair across
    `load_weights` calls, so the pending halves are kept on `model` in
    `_pending_indexer_wk_fp8`."""
    pending: dict[str, dict[str, torch.Tensor]] = model._pending_indexer_wk_fp8
    for name, weight in weights:
        is_weight = name.endswith(".weight") and weight.dtype == torch.float8_e4m3fn
        if (
            ".indexer.wk." not in name
            or not (is_weight or "weight_scale" in name)
            or is_pp_missing_parameter(name, model)
        ):
            yield name, weight
            continue
        prefix = name.rpartition(".wk.")[0]
        pair = pending.setdefault(prefix, {})
        pair["weight" if is_weight else "scale"] = weight
        if len(pair) == 2:
            del pending[prefix]
            weight = _dequantize_fp8_indexer_wk(pair["weight"], pair["scale"])
            yield f"{prefix}.wk.weight", weight
