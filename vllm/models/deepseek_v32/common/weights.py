# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint loading for the DSA causal LMs through `AutoWeightsLoader`."""

from collections.abc import Iterable, Iterator
from functools import partial

import torch

from vllm.model_executor.models.deepseek_v2 import _pair_fp8_indexer_wk
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    WeightsTransformFn,
    get_pp_missing_layer_names,
    skip_spec_layers,
)

# Routed experts (`experts.<e>.*_proj` or the stacked `experts.{gate_up,down}_proj`)
# are left to `RoutedExperts.load_weights`.
DSA_WEIGHTS_MAPPER = WeightsMapper(
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


def _drop_absent_indexers(
    weights: Iterable[tuple[str, torch.Tensor]], indexer_parents: frozenset[str]
) -> Iterator[tuple[str, torch.Tensor]]:
    """Layers that reuse another layer's top-k build no indexer, but checkpoints
    can still ship indexer weights for them."""
    for name, weight in weights:
        if ".indexer." in name and name.split(".indexer.")[0] not in indexer_parents:
            continue
        yield name, weight


def _dequantize_fp8_indexer_wk_pairs(
    weights: Iterable[tuple[str, torch.Tensor]],
    pending: dict[str, dict[str, torch.Tensor]],
    pp_missing_layer_names: list[str],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield each FP8 indexer `wk` weight in BF16, the dtype of the fused
    `wk_weights_proj`, once its scale has arrived. `pending` outlives the call:
    reordering loaders (e.g. runai_streamer) can split a pair across calls."""
    for name, weight in weights:
        paired = _pair_fp8_indexer_wk(name, weight, pending, pp_missing_layer_names)
        if paired is None:
            yield name, weight
        elif (dequantized := paired[1]) is not None:
            yield f"{paired[0]}.wk.weight", dequantized


def load_dsa_weights(
    model: torch.nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    *transforms: WeightsTransformFn,
) -> set[str]:
    """`load_weights` of a DSA causal LM.

    `transforms` run after MTP layers and absent indexers are dropped, before
    FP8 indexer `wk` pairs are dequantized. The transforms are built per call
    so that the model holds no reference to itself."""
    pending = getattr(model, "_pending_indexer_wk_fp8", None)
    if pending is None:
        model._pending_indexer_wk_fp8 = pending = {}
    indexer_parents = frozenset(
        name.removesuffix(".indexer")
        for name, _ in model.named_modules(remove_duplicate=False)
        if name.endswith(".indexer")
    )
    mapper = DSA_WEIGHTS_MAPPER | WeightsMapper(
        weights_transforms=[
            partial(skip_spec_layers, config=model.config),
            partial(_drop_absent_indexers, indexer_parents=indexer_parents),
            *transforms,
            partial(
                _dequantize_fp8_indexer_wk_pairs,
                pending=pending,
                pp_missing_layer_names=get_pp_missing_layer_names(model),
            ),
        ]
    )
    # The MLA attention is `self_attn` itself, so the quant config's KV-cache
    # scale names (`self_attn.attn.*_scale`) match nothing. The hand-written
    # loader skipped them (`maybe_remap_kv_scale_name` found no parameter).
    loader = AutoWeightsLoader(model, ignore_unexpected_suffixes=[".self_attn.attn."])
    return loader.load_weights(weights, mapper=mapper)
