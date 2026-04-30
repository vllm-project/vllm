# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers import PretrainedConfig

# Rope-related kwargs that transformers v5 PreTrainedConfig.__post_init__
# reads via `self.<attr>` *before* it setattrs unknown kwargs. As an empty
# PretrainedConfig subclass, AnyModelConfig has no declared dataclass fields,
# so without binding these early `standardize_rope_params` raises
# `AttributeError: 'AnyModelConfig' object has no attribute
# 'max_position_embeddings'`.
_ROPE_EARLY_BIND_ATTRS = (
    "max_position_embeddings",
    "rope_parameters",
    "rope_theta",
    "partial_rotary_factor",
    "original_max_position_embeddings",
)


class AnyModelConfig(PretrainedConfig):
    model_type = "anymodel"

    def __post_init__(self, **kwargs):
        for key in _ROPE_EARLY_BIND_ATTRS:
            if key in kwargs:
                object.__setattr__(self, key, kwargs[key])
        super().__post_init__(**kwargs)
