# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from transformers import PretrainedConfig


def missing_config_field_error(model_type: str, name: str) -> AttributeError:
    """Explain a missing config attribute in terms of the checkpoint.

    The model code reads architecture fields straight off the config while
    building layers, with no default in the read, so the failure otherwise
    surfaces deep inside layer construction as a bare attribute name with no
    hint that the fix is a config.json key. Curating a list of "required"
    fields would be worse than no list: the set differs by architecture,
    platform and enabled feature, so any fixed list is both incomplete and
    partly wrong.
    """
    return AttributeError(
        f"'{name}' is not set on this {model_type} config. vLLM reads "
        "architecture fields straight off the config while building the "
        "model, so this usually means config.json for the checkpoint does "
        "not define it."
    )


class DeepseekV4Config(PretrainedConfig):
    model_type = "deepseek_v4"

    def __init__(
        self,
        max_position_embeddings: int = 1048576,
        rope_scaling: dict[str, Any] | None = None,
        rope_parameters: dict[str, Any] | None = None,
        rope_theta: float = 10000.0,
        vision_n_layers: int = 0,
        vision_dim: int = 1024,
        vision_n_heads: int = 16,
        vision_inter_dim: int = 2816,
        vision_patch_size: int = 14,
        vision_rope_theta: float = 10000.0,
        vision_downsample_ratio: int = 3,
        vision_max_n_token: int = 384,
        vision_min_pixels: int = 147456,
        vision_max_wh_ratio: float = 8,
        **kwargs,
    ):
        self.max_position_embeddings = max_position_embeddings
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta
        self.rope_parameters = rope_scaling or rope_parameters
        # Vision tower config; vision_n_layers == 0 means text-only.
        self.vision_n_layers = vision_n_layers
        self.vision_dim = vision_dim
        self.vision_n_heads = vision_n_heads
        self.vision_inter_dim = vision_inter_dim
        self.vision_patch_size = vision_patch_size
        self.vision_rope_theta = vision_rope_theta
        self.vision_downsample_ratio = vision_downsample_ratio
        self.vision_max_n_token = vision_max_n_token
        self.vision_min_pixels = vision_min_pixels
        self.vision_max_wh_ratio = vision_max_wh_ratio
        # The sparse-SWA index kernels widen the window within image spans
        # in-kernel, so mm-prefix ranges longer than sliding_window must be
        # kept (they are only consumed by those kernels).
        self.mm_prefix_clamp_sliding_window = vision_n_layers > 0
        # The visibility span covers the sentinel block [IMAGE_START,
        # IMAGE_END]; the mm placeholder additionally carries a leading
        # compressor-alignment pad (see mm_preprocess.COMPRESS_PAD_TO).
        self.mm_prefix_span_leading_pad_modulus = 4 if vision_n_layers > 0 else 0
        super().__init__(**kwargs)

    def __getattr__(self, name: str) -> Any:
        # Only reached when normal lookup fails, so a complete checkpoint pays
        # nothing, and getattr(..., default) and hasattr() are unaffected.
        raise missing_config_field_error(self.model_type, name)
