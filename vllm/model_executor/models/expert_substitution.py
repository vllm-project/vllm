# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from functools import wraps
from typing import Any, TypeVar

import torch
from torch import nn

from vllm.model_executor.layers.fused_moe.expert_substitution import (
    intercept_expert_substitution_weights,
    parse_expert_substitution_config,
    validate_expert_substitution_model,
    validate_expert_substitution_weights_loaded,
)

_T = TypeVar("_T", bound=nn.Module)


def as_expert_substitution_model(model_cls: type[_T], config: Any) -> type[_T]:
    """Adapt checkpoint loading without changing the model's module paths."""
    substitution_config = parse_expert_substitution_config(config)
    if substitution_config is None:
        return model_cls
    targets = substitution_config.targets

    class ExpertSubstitutionModel(model_cls):  # type: ignore[valid-type, misc]
        @wraps(model_cls.__init__)  # type: ignore[misc]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            remote_targets = validate_expert_substitution_model(
                config, self, prefix=kwargs.get("prefix", "")
            )
            self._remote_substitution_tensors = {
                replacement.value_tensor
                for target in targets
                if target.module_path in remote_targets
                for replacement in target.replacements
            }

        def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
            remaining, substitution_params = intercept_expert_substitution_weights(
                self, weights, ignored_tensor_names=self._remote_substitution_tensors
            )
            loaded = super().load_weights(remaining)
            if loaded is not None:
                loaded.update(substitution_params)
            return loaded

        def process_weights_after_loading(self) -> None:
            # load_weights also accepts incremental updates. Check completeness
            # at initialization finalization, without clearing previously loaded rows.
            validate_expert_substitution_weights_loaded(self)
            finalize = getattr(super(), "process_weights_after_loading", None)
            if finalize is not None:
                finalize()

    ExpertSubstitutionModel.__name__ = model_cls.__name__
    ExpertSubstitutionModel.__qualname__ = model_cls.__qualname__
    return ExpertSubstitutionModel
