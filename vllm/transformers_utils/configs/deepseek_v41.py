# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from transformers import PreTrainedConfig

# One table so the flat name, the nested key and the default cannot drift
# apart, and so no assignment can depend on an earlier one.
_VISION_FIELDS: tuple[tuple[str, str, Any], ...] = (
    ("vision_n_layers", "num_hidden_layers", 0),
    ("vision_dim", "hidden_size", 1024),
    ("vision_n_heads", "num_attention_heads", 16),
    ("vision_inter_dim", "intermediate_size", 2816),
    ("vision_patch_size", "patch_size", 14),
    ("vision_rope_theta", "rope_theta", 10000.0),
    ("vision_downsample_ratio", "downsample_ratio", 3),
    ("vision_max_n_token", "max_image_tokens", 1024),
    ("vision_min_pixels", "min_pixels", 295936),
    ("vision_max_wh_ratio", "max_wh_ratio", None),
)


class DeepseekV41Config(PreTrainedConfig):
    """DeepSeek V4.1 config.

    The HF config nests the text model under ``text_config`` and the vision
    tower under ``vision_config``. The vLLM model code (ported from
    ``deepseek_v4``) reads flat attributes, so both sub-configs are flattened
    onto the top level here: text fields are exposed as-is, vision fields
    with the ``vision_*`` naming used by ``deepseek_v41.common.vision``.
    """

    model_type = "deepseek_v41"

    def __init__(
        self,
        text_config: dict[str, Any] | None = None,
        vision_config: dict[str, Any] | None = None,
        **kwargs,
    ):
        text_config = dict(text_config or {})
        vision_config = dict(vision_config or {})

        # ``rope_scaling`` is a property in Transformers v5 (backed by
        # ``rope_parameters``); capture it here and restore after
        # super().__init__, which re-standardizes rope params.
        rope_scaling = text_config.pop("rope_scaling", None)

        for key, value in text_config.items():
            if key == "model_type":
                continue
            # Don't clobber PreTrainedConfig properties (e.g. is_encoder_decoder).
            if isinstance(getattr(type(self), key, None), property):
                continue
            setattr(self, key, value)

        super().__init__(**kwargs)
        if rope_scaling is not None:
            self.rope_parameters = rope_scaling

        # The v4.1 quantization_config carries the expert dtype; the model
        # code reads it from the top-level hf_config.
        quant_cfg = getattr(self, "quantization_config", None) or {}
        if not hasattr(self, "expert_dtype") and "expert_dtype" in quant_cfg:
            self.expert_dtype = quant_cfg["expert_dtype"]

        # to_dict() emits the flattened names and no ``vision_config``, so a
        # config rebuilt from its own dict arrives with the flat fields set by
        # super().__init__ and nothing nested. Taking the nested block
        # unconditionally would reset such a config to a text-only tower.
        # Precedence: the nested block, then a flat field the config was
        # constructed with, then the default. Reading the flat value from
        # kwargs rather than from self keeps a stale ``vision_*`` key inside
        # ``text_config`` -- which the loop above has already set on self --
        # from being taken for vision configuration.
        for flat_name, nested_key, default in _VISION_FIELDS:
            if nested_key in vision_config:
                value = vision_config[nested_key]
            else:
                value = kwargs.get(flat_name, default)
            setattr(self, flat_name, value)
