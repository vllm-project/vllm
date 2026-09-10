# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from transformers import PretrainedConfig


class DeepseekV41Config(PretrainedConfig):
    """DeepSeek V4.1 config.

    The HF config nests the text model under ``text_config`` and the vision
    tower under ``vision_config``. The vLLM model code (ported from
    ``deepseek_v4``) reads flat attributes, so both sub-configs are flattened
    onto the top level here: text fields are exposed as-is, vision fields
    with the ``vision_*`` naming used by ``deepseek_v4_1.common.vision``.
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
            # Don't clobber PretrainedConfig properties (e.g. is_encoder_decoder).
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

        vision_n_layers = vision_config.get("num_hidden_layers", 0)
        self.vision_n_layers = vision_n_layers
        self.vision_dim = vision_config.get("hidden_size", 1024)
        self.vision_n_heads = vision_config.get("num_attention_heads", 16)
        self.vision_inter_dim = vision_config.get("intermediate_size", 2816)
        self.vision_patch_size = vision_config.get("patch_size", 14)
        self.vision_rope_theta = vision_config.get("rope_theta", 10000.0)
        self.vision_downsample_ratio = vision_config.get("downsample_ratio", 3)
        self.vision_max_n_token = vision_config.get("max_image_tokens", 1024)
        self.vision_min_pixels = vision_config.get("min_pixels", 295936)
        self.vision_max_wh_ratio = vision_config.get("max_wh_ratio")
        # The vision variant needs the mm-prefix plumbing (atomic image-span
        # prefill + sparse-SWA window widening); the base arch-config
        # convertor reads this config-level attribute.
        self.is_mm_prefix_lm = vision_n_layers > 0
        # The sparse-SWA index kernels widen the window within image spans
        # in-kernel, so mm-prefix ranges longer than sliding_window must be
        # kept (they are only consumed by those kernels).
        self.mm_prefix_clamp_sliding_window = vision_n_layers > 0
        # The visibility span covers the image block [IMAGE_START,
        # IMAGE_END]; the mm placeholder additionally carries a leading
        # compressor-alignment pad of ``COMPRESS_PAD_TO - 1 - offset %
        # COMPRESS_PAD_TO`` tokens (see common/mm_preprocess.py), with
        # COMPRESS_PAD_TO = 2 for v4.1's ratio-2 compressors.
        self.mm_prefix_span_leading_pad_modulus = 2 if vision_n_layers > 0 else 0
