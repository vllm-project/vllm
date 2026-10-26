# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# adapted from https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4/

"""RADIO vision encoder configuration."""

from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PreTrainedConfig
from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from transformers.models.nemotron_h import NemotronHConfig
from transformers.models.parakeet.configuration_parakeet import ParakeetEncoderConfig
from transformers.utils import auto_docstring, logging


@auto_docstring(checkpoint="nvidia/C-RADIOv4-H")
@strict
class NemotronHOmniRadioConfig(PreTrainedConfig):
    r"""mlp_ratio (`float`, *optional*, defaults to 4.0):
        Ratio of the hidden size of the MLP relative to `hidden_size`.
    use_swiglu_ffn (`bool`, *optional*, defaults to `False`):
        Whether to use a SwiGLU feed-forward network in the encoder layers \
            instead of the standard MLP.
    layerscale_value (`float`, *optional*, defaults to 1.0):
        Initial value for the LayerScale parameters. C-RADIO has no LayerScale; \
              the default of `1.0` makes the
        (inherited) LayerScale an identity operation.
    max_img_size (`int`, *optional*, defaults to 2048):
        Maximum supported image size (in pixels) used to size the position embedding \
              table of the CPE patch generator.
    num_cls_tokens (`int`, *optional*, defaults to 3):
        Number of learned class (summary) tokens prepended to the patch sequence.
    num_registers (`int`, *optional*, defaults to 7):
        Number of learned register tokens prepended to the patch sequence.
    summary_idxs (`list[int]`, *optional*, defaults to `[0, 1]`):
        Indices of the class tokens to gather and flatten into the `summary` output \
            embedding.
    norm_mean (`tuple[float, float, float]`, \
        *optional*, defaults to `OPENAI_CLIP_MEAN`):
        Per-channel mean used by the input conditioner to normalize pixel values.
    norm_std (`tuple[float, float, float]`, *optional*, defaults to `OPENAI_CLIP_STD`):
        Per-channel standard deviation used by the input conditioner to normalize \
            pixel values.
    video_temporal_patch_size (`int`, *optional*):
        Number of temporally-stacked frames per patch. When set, the patch generator \
              also builds a `video_patch_projection` for temporally-packed video \
                patches; `None` disables the video path.
    """

    model_type = "radio"

    hidden_size: int = 1280
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    mlp_ratio: float = 4.0
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    attention_probs_dropout_prob: float = 0.0
    hidden_dropout_prob: float = 0.0
    drop_path_rate: float = 0.0
    use_swiglu_ffn: bool = False
    qkv_bias: bool = True
    # C-RADIO has no layerscale; 1.0 makes the (inherited) layerscale an identity op.
    layerscale_value: float = 1.0
    # patch / image
    num_channels: int = 3
    patch_size: int = 16
    image_size: int = 224
    # CPE patch generator
    max_img_size: int = 2048
    num_cls_tokens: int = 3
    num_registers: int = 7
    video_temporal_patch_size: int | None = None
    summary_idxs: list[int] | None = None
    # input conditioner
    norm_mean: list[float] | tuple[float, float, float] = tuple(OPENAI_CLIP_MEAN)
    norm_std: list[float] | tuple[float, float, float] = tuple(OPENAI_CLIP_STD)
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.summary_idxs is None:
            self.summary_idxs = [0, 1]
        self.norm_mean = list(self.norm_mean)
        self.norm_std = list(self.norm_std)
        super().__post_init__(**kwargs)

    @property
    def num_summary_tokens(self) -> int:
        """Number of skipped prefix tokens (cls + registers) before spatial features."""
        return self.num_cls_tokens + self.num_registers


logger = logging.get_logger(__name__)


def _nemotron_h_compatible_config(config: dict) -> dict:
    """Translate current Nemotron-H layer names for Transformers versions using\
          legacy names."""
    config = dict(config)
    legacy_names = {"linear_attention": "mamba", "full_attention": "attention"}
    for field in ("layers_block_type", "mtp_layers_block_type"):
        if field in config and config[field] is not None:
            config[field] = [
                legacy_names.get(layer_type, layer_type) for layer_type in config[field]
            ]
    return config


@auto_docstring(checkpoint="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16")
@strict
class NemotronHOmniReasoningV3Config(PreTrainedConfig):
    r"""vision_config (`dict` or `NemotronHOmniRadioConfig`, *optional*):
        Configuration for the RADIO vision encoder. Defaults to a default \
            [`NemotronHOmniRadioConfig`].
    llm_config (`dict` or `NemotronHConfig`, *optional*):
        Configuration for the NemotronH language model. Defaults to a \
            default [`NemotronHConfig`].
    sound_config (`dict` or `ParakeetEncoderConfig`, *optional*):
        Configuration for the optional Parakeet sound encoder. `None` \
            disables the audio branch.
    force_image_size (`int`, *optional*):
        Fixed input image resolution (in pixels) the vision tower expects.
    downsample_ratio (`float`, *optional*, defaults to 0.5):
        Pixel-shuffle spatial downsample ratio applied to the vision features.
    template (`str`, *optional*):
        Conversation template name (legacy; the chat template is carried by the \
            processor).
    ps_version (`str`, *optional*, defaults to `"v1"`):
        Pixel-shuffle implementation version (`"v2"` swaps height/width back; `"v1"`\
              is the legacy
        transposed variant).
    image_tag_type (`str`, *optional*, defaults to `"internvl"`):
        Image-tag convention used when expanding image placeholders.
    projector_hidden_size (`int`, *optional*, defaults to 4096):
        Hidden size of the vision-to-LLM MLP projector.
    vit_hidden_size (`int`, *optional*, defaults to 1280):
        Hidden size of the RADIO vision features.
    video_pruning_rate (`float`, *optional*, defaults to 0.0):
        Efficient-Video-Sampling token pruning rate; `0.0` disables pruning.
    video_temporal_patch_size (`int`, *optional*, defaults to 2):
        Number of frames collapsed into a single temporal patch by the video \
            embedder.
    patch_size (`int`, *optional*, defaults to 16):
        Vision patch size in pixels.
    img_context_token_id (`int`, *optional*):
        Token id used as the image-context placeholder in `input_ids`.
    img_context_token (`str`, *optional*, defaults to `"<image>"`):
        Textual image-context placeholder token.
    video_context_token_id (`int`, *optional*):
        Token id used as the video-context placeholder in `input_ids`.
    video_context_token (`str`, *optional*, defaults to `"<video>"`):
        Textual video-context placeholder token.
    sound_context_token_id (`int`, *optional*):
        Token id used as the audio-context placeholder in `input_ids`.
    sound_context_token (`str`, *optional*, defaults to `"<audio>"`):
        Textual audio-context placeholder token.
    """

    model_type = "nemotron_h_omni"
    is_composition = True

    vision_config: dict | NemotronHOmniRadioConfig | None = None
    llm_config: dict | NemotronHConfig | None = None
    sound_config: dict | ParakeetEncoderConfig | None = None
    force_image_size: int | None = None
    downsample_ratio: float = 0.5
    template: str | None = None  # TODO move out of here and into the tokenizer
    ps_version: str = "v1"  # Pixel shuffle version
    image_tag_type: str = "internvl"  # TODO: into the tokenizer too?
    projector_hidden_size: int = 4096
    vit_hidden_size: int = 1280
    video_pruning_rate: float = 0.0
    video_temporal_patch_size: int = 2
    # Vision token settings
    patch_size: int = 16
    img_context_token_id: int | None = None
    img_context_token: str = "<image>"
    video_context_token_id: int | None = None
    video_context_token: str = "<video>"
    # Sound/audio token settings
    sound_context_token_id: int | None = None
    sound_context_token: str = "<audio>"

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config = NemotronHOmniRadioConfig(**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = NemotronHOmniRadioConfig()
        # The vision tower needs the temporal patch size to build its video patch
        # projection.
        self.vision_config.video_temporal_patch_size = self.video_temporal_patch_size

        # Handle both cases: when loading from JSON (llm_config is dict) and when
        # called internally by transformers (llm_config is None).
        if isinstance(self.llm_config, dict):
            self.llm_config = NemotronHConfig(
                **_nemotron_h_compatible_config(self.llm_config)
            )
        elif self.llm_config is None:
            self.llm_config = NemotronHConfig()

        # This checkpoint's sound_config omits `attention_bias`/`scale_input`,
        # which are `False` for its Parakeet variant, so supply them before
        # building the encoder config.
        if isinstance(self.sound_config, dict):
            sound_config = {
                "attention_bias": False,
                "scale_input": False,
                **self.sound_config,
            }
            sound_config.pop("model_type", None)
            self.sound_config = ParakeetEncoderConfig(**sound_config)

        super().__post_init__(**kwargs)

        # We assert llm_config is now a type implementing _attn_implementation
        # to avoid vLLM linter errors for older Python versions on the next line
        assert isinstance(self.llm_config, NemotronHConfig)

        # `attn_implementation` flows in through `**kwargs` (the base
        # #`PreTrainedConfig` stores it as `self._attn_implementation`,
        # as for every other model); propagate it to the language model.
        self.llm_config._attn_implementation = self._attn_implementation

    # vLLM's `NemotronH_Nano_VL_V2` implementation reads the language-model
    #  sub-config as `config.text_config`. Our HF config stores it as
    # `config.llm_config`; expose an alias so the same config object loads
    # under both loaders without having to duplicate the dict on disk.
    @property
    def text_config(self):
        return self.llm_config


__all__ = ["NemotronHOmniReasoningV3Config", "NemotronHOmniRadioConfig"]
