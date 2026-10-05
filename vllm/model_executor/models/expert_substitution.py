# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from functools import wraps
from typing import Any, TypeVar

import torch
from torch import nn

from vllm.model_executor.layers.fused_moe.expert_substitution import (
    APPROX_VALUE_SUFFIX,
    ConstantExpertSubstitution,
    decoder_layer_index,
    get_expert_substitution_spec,
)

_T = TypeVar("_T", bound=nn.Module)


def _collect_expert_substitutions(
    model: nn.Module,
) -> dict[int, tuple[str, ConstantExpertSubstitution]]:
    substitutions: dict[int, tuple[str, ConstantExpertSubstitution]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, ConstantExpertSubstitution):
            continue
        if module.layer_idx in substitutions:
            raise ValueError(
                f"layer {module.layer_idx} has more than one MoE with substituted "
                "experts"
            )
        substitutions[module.layer_idx] = (f"{name}.values", module)
    return substitutions


def _parse_approx_value_name(name: str) -> tuple[int, int] | None:
    """Return (layer index, expert ID) of an ``approx_value`` tensor name."""
    if not name.endswith(APPROX_VALUE_SUFFIX):
        return None
    experts_path, _, expert_id = name.removesuffix(APPROX_VALUE_SUFFIX).rpartition(".")
    layer_idx = decoder_layer_index(experts_path)
    if layer_idx is None or not expert_id.isdigit():
        return None
    return layer_idx, int(expert_id)


def as_expert_substitution_model(model_cls: type[_T], model_config: Any) -> type[_T]:
    """Load constant expert values at the ``load_weights`` boundary.

    Initial loading and weight reloads both go through ``load_weights``.
    Explicitly named values are always consumed and loaded if their layer is
    built by this model. ``approx_value`` tensors of layers built elsewhere
    (other pipeline stages, MTP layers) are passed through, so the model's own
    loader skips them like any other weight of a layer it does not own.
    """
    spec = get_expert_substitution_spec(model_config)
    if spec is None:
        return model_cls
    value_names = spec.value_names

    class ExpertSubstitutionModel(model_cls):  # type: ignore[valid-type, misc]
        @wraps(model_cls.__init__)  # type: ignore[misc]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._expert_substitutions = _collect_expert_substitutions(self)

        def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
            loaded_substitutions: set[str] = set()

            def remaining_weights():
                for name, loaded_weight in weights:
                    targets = value_names.get(name)
                    if targets is None:
                        key = _parse_approx_value_name(name)
                        if key is None or key[0] not in self._expert_substitutions:
                            yield name, loaded_weight
                            continue
                        targets = (key,)
                    for layer_idx, expert_id in targets:
                        if layer_idx not in self._expert_substitutions:
                            continue
                        param_name, substitution = self._expert_substitutions[layer_idx]
                        param = substitution.values
                        param.weight_loader(param, loaded_weight, expert_id)
                        loaded_substitutions.add(param_name)

            loaded = super().load_weights(remaining_weights())
            if loaded is not None:
                loaded.update(loaded_substitutions)
            return loaded

        def process_weights_after_loading(self) -> None:
            for param_name, substitution in self._expert_substitutions.values():
                substitution.validate_loaded_values(param_name)
            finalize = getattr(super(), "process_weights_after_loading", None)
            if finalize is not None:
                finalize()

    ExpertSubstitutionModel.__name__ = model_cls.__name__
    ExpertSubstitutionModel.__qualname__ = model_cls.__qualname__
    return ExpertSubstitutionModel
