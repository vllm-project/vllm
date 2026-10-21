# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from functools import wraps
from typing import Any, TypeVar

import torch
from torch import nn

from vllm.model_executor.layers.fused_moe.expert_substitution import (
    NOVICES_SUFFIX,
    ConstantExpertSubstitution,
    decoder_layer_index,
    get_expert_substitution_spec,
)
from vllm.model_executor.model_loader.reload import finalize_layerwise_reload

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
        substitutions[module.layer_idx] = (f"{name}.novices", module)
    return substitutions


def _parse_novices_name(name: str) -> int | None:
    """Return the decoder layer index of a ``novices`` tensor name."""
    if not name.endswith(NOVICES_SUFFIX):
        return None
    return decoder_layer_index(name.removesuffix(NOVICES_SUFFIX))


def as_expert_substitution_model(model_cls: type[_T], model_config: Any) -> type[_T]:
    """Load novice expert constants at the ``load_weights`` boundary.

    Initial loading and direct weight updates both go through ``load_weights``.
    ``novices`` tensors of layers built elsewhere (other pipeline stages, MTP
    layers) are passed through, so the model's own loader skips them like any
    other weight of a layer it does not own.
    """
    if get_expert_substitution_spec(model_config) is None:
        return model_cls

    class ExpertSubstitutionModel(model_cls):  # type: ignore[valid-type, misc]
        @wraps(model_cls.__init__)  # type: ignore[misc]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._expert_substitutions = _collect_expert_substitutions(self)

        def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
            if any(s.novices.is_meta for _, s in self._expert_substitutions.values()):
                # Put the previous weights back so the model stays usable.
                finalize_layerwise_reload(self, model_config)
                raise NotImplementedError(
                    "expert substitution does not support layerwise weight reload"
                )
            loaded_substitutions: set[str] = set()

            def remaining_weights():
                for name, loaded_weight in weights:
                    layer_idx = _parse_novices_name(name)
                    if layer_idx is None or layer_idx not in (
                        self._expert_substitutions
                    ):
                        yield name, loaded_weight
                        continue
                    param_name, substitution = self._expert_substitutions[layer_idx]
                    param = substitution.novices
                    param.weight_loader(param, loaded_weight)
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
