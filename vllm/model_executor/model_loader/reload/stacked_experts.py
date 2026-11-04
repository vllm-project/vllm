# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stacked routed experts (transformers v5) as per-expert checkpoint tensors.

A v5 trainer names a MoE layer's routed experts ``<prefix>.experts.gate_up_proj``
[E, 2I, H] (gate rows, then up rows) and ``<prefix>.experts.down_proj`` [E, H, I],
plus suffixed tensors in the same layout (``gate_up_proj_scale_inv``,
``gate_up_proj_weight_scale``, ...). Many
models' ``load_weights`` only know the per-expert form
(``<prefix>.experts.<e>.gate_proj.weight``, ...). ``split_stacked_experts`` hands
them per-expert *views* of each stacked tensor, so the load plan traces the
model's own per-expert copies back to boxes of the stacked tensor and merges
them (one write per run of local experts).
"""

from collections.abc import Iterable, Iterator

import regex as re
import torch

_STACKED = re.compile(
    r"^(?P<prefix>(?:.*\.)?experts)\.(?P<proj>gate_up_proj|down_proj)(?P<rest>_\w+)?$"
)

Weights = Iterable[tuple[str, torch.Tensor]]


def _expert_names(model: torch.nn.Module) -> tuple[str, str, str] | None:
    """(gate, up, down) per-expert checkpoint names shared by every routed
    experts layer of ``model``, or None when the split does not apply: a model
    whose ``load_weights`` takes the stacked names itself
    (``loads_stacked_experts``), no routed experts layer, layers that
    disagree, or a layer whose stacked form is not [E, 2I, H] gate-then-up
    (non-gated, transposed checkpoint, or its own loaders)."""
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    if getattr(model, "loads_stacked_experts", False):
        return None
    names = set()
    for module in model.modules():
        if not isinstance(module, RoutedExperts):
            continue
        cls = type(module)
        if (
            cls.load_weights is not RoutedExperts.load_weights
            or cls.weight_loader is not RoutedExperts.weight_loader
            or module.is_fused_checkpoint_transposed
            or module.ckpt_up_proj_name is None
            or (module.quant_config and module.quant_config.get_name())
            == "gpt_oss_mxfp4"
        ):
            return None
        names.add(
            (
                module.ckpt_gate_proj_name,
                module.ckpt_up_proj_name,
                module.ckpt_down_proj_name,
            )
        )
    return names.pop() if len(names) == 1 else None


def split_stacked_experts(model: torch.nn.Module, weights: Weights) -> Weights:
    """``weights`` with each stacked routed-expert tensor replaced by
    per-expert views of it, named as ``model``'s experts layers expect;
    ``weights`` itself when the split does not apply (see ``_expert_names``)."""
    names = _expert_names(model)
    if names is None:
        return weights
    return _split(weights, *names)


def _split(weights: Weights, gate: str, up: str, down: str) -> Iterator:
    for name, tensor in weights:
        match = _STACKED.match(name)
        if (
            match is None
            or tensor.dim() != 3
            or (match["proj"] == "gate_up_proj" and tensor.shape[1] % 2)
        ):
            yield name, tensor
            continue
        prefix, rest = match["prefix"], match["rest"] or ""
        # gate_up_proj -> .weight, gate_up_proj_weight_scale -> .weight_scale,
        # gate_up_proj_scale_inv -> .weight_scale_inv
        suffix = rest[1:] if rest.startswith("_weight") else f"weight{rest}"
        if match["proj"] == "down_proj":
            for e, view in enumerate(tensor.unbind(0)):
                yield f"{prefix}.{e}.{down}.{suffix}", view
            continue
        half = tensor.shape[1] // 2
        gates = tensor.narrow(1, 0, half).unbind(0)
        ups = tensor.narrow(1, half, half).unbind(0)
        for e, (gate_view, up_view) in enumerate(zip(gates, ups)):
            yield f"{prefix}.{e}.{gate}.{suffix}", gate_view
            yield f"{prefix}.{e}.{up}.{suffix}", up_view
