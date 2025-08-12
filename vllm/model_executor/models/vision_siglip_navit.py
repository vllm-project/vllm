# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
""" Siglip model configuration"""

import math
import os
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn.init import _calculate_fan_in_and_fan_out
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_outputs import (BaseModelOutput,
                                           BaseModelOutputWithPooling)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (ModelOutput, add_start_docstrings,
                                add_start_docstrings_to_model_forward, logging,
                                replace_return_docstrings)

from vllm.platforms import _Backend

from .vision import get_vit_attn_backend

logger = logging.get_logger(__name__)

SIGLIP_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/siglip-base-patch16-224":
    "https://huggingface.co/google/siglip-base-patch16-224/"\
        "resolve/main/config.json",
}


class SiglipTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a 
    [`SiglipTextModel`]. It is used to instantiate a Siglip text encoder
    according to the specified arguments, defining the model architecture. 
    Instantiating a configuration with the defaults will yield a similar 
    configuration to that of the text encoder of the Siglip [google/
    siglip-base-patch16-224](https://huggingface.co/google/siglip-base
    -patch16-224) architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used
    to control the model outputs. Read the documentation from 
    [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Siglip text model. Defines the number of 
            different tokens that can be represented by the `inputs_ids` 
            passed when calling [`SiglipModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer 
            in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the 
            Transformer encoder.
        max_position_embeddings (`int`, *optional*, defaults to 64):
            The maximum sequence length that this model might ever be used 
            with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        hidden_act (`str` or `function`, *optional*, defaults to 
            `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the
            encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        pad_token_id (`int`, *optional*, defaults to 1):
            The id of the padding token in the vocabulary.
        bos_token_id (`int`, *optional*, defaults to 49406):
            The id of the beginning-of-sequence token in the vocabulary.
        eos_token_id (`int`, *optional*, defaults to 49407):
            The id of the end-of-sequence token in the vocabulary.
    Example:
    ```python
    >>> from transformers import SiglipTextConfig, SiglipTextModel
    >>> # Initializing a SiglipTextConfig with google/siglip-base-patch16-224 
        style configuration
    >>> configuration = SiglipTextConfig()
    >>> # Initializing a SiglipTextModel (with random weights) from the 
        google/siglip-base-patch16-224 style configuration
    >>> model = SiglipTextModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "siglip_text_model"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=64,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        # This differs from `CLIPTokenizer`'s default and from openai/siglip
        # See https://github.com/huggingface/transformers/pull/24773#
        # issuecomment-1632287538
        pad_token_id=1,
        bos_token_id=49406,
        eos_token_id=49407,
        _flash_attn_2_enabled=True,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id,
                         bos_token_id=bos_token_id,
                         eos_token_id=eos_token_id,
                         **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout
        self._flash_attn_2_enabled = _flash_attn_2_enabled

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str,
                                                                  os.PathLike],
                        **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs)

        # get the text config dict if we are loading from SiglipConfig
        if config_dict.get("model_type") == "siglip":
            config_dict = config_dict["text_config"]

        if "model_type" in config_dict and hasattr(
                cls,
                "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                "You are using a model of type %s to instantiate a model of "
                "type %s. This is not supported for all configurations of "
                "models and can yield errors.", config_dict['model_type'],
                cls.model_type)

        return cls.from_dict(config_dict, **kwargs)


class SiglipVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a 
    [`SiglipVisionModel`]. It is used to instantiate a
    Siglip vision encoder according to the specified arguments, defining the 
    model architecture. Instantiating a configuration with the defaults will
    yield a similar configuration to that of the vision encoder of the Siglip
    [google/siglip-base-patch16-224](https://huggingface.co/google/
    siglip-base-patch16-224) architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used 
    to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer
            in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the
            Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input images.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to 
            `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the 
            encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"` and
            `"gelu_new"` ``"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
    Example:
    ```python
    >>> from transformers import SiglipVisionConfig, SiglipVisionModel
    >>> # Initializing a SiglipVisionConfig with google/siglip-base-patch16-224
        style configuration
    >>> configuration = SiglipVisionConfig()
    >>> # Initializing a SiglipVisionModel (with random weights) from the 
        google/siglip-base-patch16-224 style configuration
    >>> model = SiglipVisionModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "siglip_vision_model"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        _flash_attn_2_enabled=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self._flash_attn_2_enabled = _flash_attn_2_enabled

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str,
                                                                  os.PathLike],
                        **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from SiglipConfig
        if config_dict.get("model_type") == "siglip":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(
                cls,
                "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                "You are using a model of type %s to "
                "instantiate a model of type %s. This is not"
                " supported for all configurations of models and can yield"
                " errors.", config_dict['model_type'], cls.model_type)

        return cls.from_dict(config_dict, **kwargs)


class SiglipConfig(PretrainedConfig):
    r"""
    [`SiglipConfig`] is the configuration class to store the configuration of a
    [`SiglipModel`]. It is used to instantiate a Siglip model according to the 
    specified arguments, defining the text model and vision model configs.
    Instantiating a configuration with the defaults will yield a similar
    configuration to that of the Siglip [google/siglip-base-patch16-224](
    https://huggingface.co/google/siglip-base-patch16-224) architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to
    control the model outputs. Read the documentation from 
    [`PretrainedConfig`] for more information.
    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize 
            [`SiglipTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize 
            [`SiglipVisionConfig`].
        kwargs (*optional*):
            Dictionary of keyword arguments.
    Example:
    ```python
    >>> from transformers import SiglipConfig, SiglipModel
    >>> # Initializing a SiglipConfig with google/siglip-base-patch16-224 
        style configuration
    >>> configuration = SiglipConfig()
    >>> # Initializing a SiglipModel (with random weights) from the 
        google/siglip-base-patch16-224 style configuration
    >>> model = SiglipModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    >>> # We can also initialize a SiglipConfig from a SiglipTextConfig 
        and a SiglipVisionConfig
    >>> from transformers import SiglipTextConfig, SiglipVisionConfig
    >>> # Initializing a SiglipText and SiglipVision configuration
    >>> config_text = SiglipTextConfig()
    >>> config_vision = SiglipVisionConfig()
    >>> config = SiglipConfig.from_text_vision_configs(config_text, 
        config_vision)
    ```"""

    model_type = "siglip"

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        super().__init__(**kwargs)

        if text_config is None:
            text_config = {}
            logger.info(
                "`text_config` is `None`. Initializing the `SiglipTextConfig`"
                " with default values.")

        if vision_config is None:
            vision_config = {}
            logger.info("`vision_config` is `None`. initializing the "
                        "`SiglipVisionConfig` with default values.")

        self.text_config = SiglipTextConfig(**text_config)
        self.vision_config = SiglipVisionConfig(**vision_config)

        self.initializer_factor = 1.0

    @classmethod
    def from_text_vision_configs(cls, text_config: SiglipTextConfig,
                                 vision_config: SiglipVisionConfig, **kwargs):
        r"""
        Instantiate a [`SiglipConfig`] (or a derived class) from siglip text 
        model configuration and siglip vision
        model configuration.
        Returns:
            [`SiglipConfig`]: An instance of a configuration object
        """

        return cls(text_config=text_config.to_dict(),
                   vision_config=vision_config.to_dict(),
                   **kwargs)


# coding=utf-8
# Copyright 2024 Google AI and The HuggingFace Team. All rights reserved.
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
""" PyTorch Siglip model."""

_CHECKPOINT_FOR_DOC = "google/siglip-base-patch16-224"

SIGLIP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/siglip-base-patch16-224",
    # See all SigLIP models at https://huggingface.co/models?filter=siglip
]


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official
    # releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/
    # truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)  # noqa
    u = norm_cdf((b - mean) / std)  # noqa

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    if tensor.dtype in [torch.float16, torch.bfloat16]:
        # The `erfinv_` op is not (yet?) defined in float16+cpu, bfloat16+gpu
        og_dtype = tensor.dtype
        tensor = tensor.to(torch.float32)
        tensor.erfinv_()
        tensor = tensor.to(og_dtype)
    else:
        tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    if tensor.dtype == torch.float16:
        # The `clamp_` op is not (yet?) defined in float16+cpu
        tensor = tensor.to(torch.float32)
        tensor.clamp_(min=a, max=b)
        tensor = tensor.to(torch.float16)
    else:
        tensor.clamp_(min=a, max=b)


def trunc_normal_tf_(tensor: torch.Tensor,
                     mean: float = 0.0,
                     std: float = 1.0,
                     a: float = -2.0,
                     b: float = 2.0) -> torch.Tensor:
    """Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \\leq \text{mean} \\leq b`.
    NOTE: this 'tf' variant behaves closer to Tensorflow / JAX impl where 
    the bounds [a, b] are applied when sampling the normal distribution with
    mean=0, std=1.0 and the result is subsequently scaled and shifted by the
    mean and std args.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    """
    with torch.no_grad():
        _trunc_normal_(tensor, 0, 1.0, a, b)
        tensor.mul_(std).add_(mean)


def variance_scaling_(tensor, scale=1.0, mode="fan_in", distribution="normal"):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        # constant is stddev of standard normal truncated to (-2, 2)
        trunc_normal_tf_(tensor, std=math.sqrt(variance) / 0.87962566103423978)
    elif distribution == "normal":
        with torch.no_grad():
            tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        with torch.no_grad():
            tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")


def default_flax_embed_init(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="normal")


@dataclass
# Copied from transformers.models.clip.modeling_clip.CLIPVisionModelOutput with
# CLIP->Siglip
class SiglipVisionModelOutput(ModelOutput):
    """
    Base class for vision model's outputs that also contains image embeddings
    of the pooling of the last hidden states.
    Args:
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)`
            *optional* returned when model is initialized with 
            `with_projection=True`):
            The image embeddings obtained by applying the projection layer to
            the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, 
            sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the 
            model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when 
            `output_hidden_states=True` is passed or when 
            `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, 
            if the model has an embedding layer, + one for the output of each
            layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the 
            optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when 
            `output_attentions=True` is passed or when 
            `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape 
            `(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the 
            weighted average in the self-attention heads.
    """

    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
# Copied from transformers.models.clip.modeling_clip.CLIPTextModelOutput with
# CLIP->Siglip
class SiglipTextModelOutput(ModelOutput):
    """
    Base class for text model's outputs that also contains a pooling of the 
    last hidden states.
    Args:
        text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)`
             *optional* returned when model is initialized with 
             `with_projection=True`):
            The text embeddings obtained by applying the projection layer to
            model.
            the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, 
            sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when 
            `output_hidden_states=True` is passed or when 
            `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the 
            embeddings, if the model has an embedding layer, + one for the 
            output of each layer) of shape `(batch_size, sequence_length, 
            hidden_size)`.
            Hidden-states of the model at the output of each layer plus the
            optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when 
            `output_attentions=True` is passed or when 
            `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape 
            `(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute
            the weighted average in the self-attention heads.
    """

    text_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
# Copied from transformers.models.clip.modeling_clip.CLIPOutput with
# CLIP->Siglip
class SiglipOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when
          `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image:(`torch.FloatTensor` of shape `(image_batch_size,
          text_batch_size)`):
            The scaled dot product scores between `image_embeds` and 
            `text_embeds`. This represents the image-text similarity scores.
        logits_per_text:(`torch.FloatTensor` of shape `(text_batch_size, 
            image_batch_size)`):
            The scaled dot product scores between `text_embeds` and 
            `image_embeds`. This represents the text-image similarity scores.
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to 
            the pooled output of [`SiglipTextModel`].
        image_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to
            the pooled output of [`SiglipVisionModel`].
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`SiglipTextModel`].
        vision_model_output(`BaseModelOutputWithPooling`):
            The output of the [`SiglipVisionModel`].
    """

    loss: Optional[torch.FloatTensor] = None
    logits_per_image: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"
                                 ] else getattr(self, k).to_tuple()
            for k in self.keys())


class SiglipVisionEmbeddings(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_side**2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions,
                                               self.embed_dim)

    def forward(self, pixel_values: torch.FloatTensor,
                patch_attention_mask: torch.BoolTensor) -> torch.Tensor:
        batch_size = pixel_values.size(0)

        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        max_im_h, max_im_w = pixel_values.size(2), pixel_values.size(3)
        max_nb_patches_h, max_nb_patches_w = max_im_h // self.patch_size, \
            max_im_w // self.patch_size
        boundaries = torch.arange(1 / self.num_patches_per_side, 1.0,
                                  1 / self.num_patches_per_side)
        position_ids = torch.full(
            size=(
                batch_size,
                max_nb_patches_h * max_nb_patches_w,
            ),
            fill_value=0,
        )

        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            nb_patches_h = p_attn_mask[:, 0].sum()
            nb_patches_w = p_attn_mask[0].sum()

            fractional_coords_h = torch.linspace(0, 1 - 1 / nb_patches_h,
                                                 nb_patches_h)
            fractional_coords_w = torch.linspace(0, 1 - 1 / nb_patches_w,
                                                 nb_patches_w)

            bucket_coords_h = torch.bucketize(fractional_coords_h,
                                              boundaries,
                                              right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w,
                                              boundaries,
                                              right=True)

            pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side +
                       bucket_coords_w).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids

        position_ids = position_ids.to(self.position_embedding.weight.device)

        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings


# Copied from transformers.models.clip.modeling_clip.CLIPTextEmbeddings with
# CLIP->Siglip
class SiglipTextEmbeddings(nn.Module):

    def __init__(self, config: SiglipTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings,
                                               embed_dim)

        # position_ids (1, len position emb) is contiguous in memory and
        # exported when serialized
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        seq_length = input_ids.shape[
            -1] if input_ids is not None else inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


class SiglipAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Copied from transformers.models.clip.modeling_clip.CLIPAttention.__init__
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`:"
                f" {self.embed_dim} and `num_heads`: {self.num_heads}).")
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, q_len, self.num_heads,
                                         self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads,
                                         self.head_dim).transpose(1, 2)

        k_v_seq_len = key_states.shape[-2]
        attn_weights = torch.matmul(query_states, key_states.transpose(
            2, 3)) * self.scale

        if attn_weights.size() != (batch_size, self.num_heads, q_len,
                                   k_v_seq_len):
            raise ValueError(
                f"Attention weights should be of size "
                f"{(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is"
                f" {attn_weights.size()}")

        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, q_len, k_v_seq_len):
                raise ValueError(f"Attention mask should be of size "
                                 f"{(batch_size, 1, q_len, k_v_seq_len)}, "
                                 f"but is {attention_mask.size()}")
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights,
                                             dim=-1,
                                             dtype=torch.float32).to(
                                                 query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights,
                                             p=self.dropout,
                                             training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, q_len,
                                  self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size "
                f"{(batch_size, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}")

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class SiglipFlashAttention2(SiglipAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as
    the weights of the module stays untouched. The only required change would
    be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any
    of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False  # Hack to make sure we don't use a causal mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads,
                                         self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads,
                                         self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(
                kv_seq_len, self.layer_idx)

        # TODO: These transpose are quite inefficient but Flash Attention
        #  requires the layout [batch_size, sequence_length, num_heads,
        #  head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training
        # stability reasons therefore the input hidden states gets silently
        # casted in float32. Hence, we need cast them back in the correct
        # dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to
        # not cast the LayerNorms in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                "The input hidden states seems to be silently casted in "
                "float32, this might be related to the fact you have upcasted "
                "embedding or layer norm layers in float32. We will cast "
                f"back the input in {target_dtype}.")

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(query_states,
                                                    key_states,
                                                    value_states,
                                                    attention_mask,
                                                    q_len,
                                                    dropout=dropout_rate)

        attn_output = attn_output.reshape(bsz, q_len,
                                          self.embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights

    def _flash_attention_forward(self,
                                 query_states,
                                 key_states,
                                 value_states,
                                 attention_mask,
                                 query_length,
                                 dropout=0.0,
                                 softmax_scale=None):
        """
        Calls the forward method of Flash Attention - if the input hidden 
        states contain at least one padding token first unpad the input, 
        then computes the attention scores and pad the final attention 
        scores.
        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size 
                `(batch_size, seq_len)` where 0 stands for the position 
                of padding tokens and 1 for the position of non-padding 
                tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / 
                sqrt(head_dim)
        """
        from flash_attn import flash_attn_func, flash_attn_varlen_func
        from flash_attn.bert_padding import pad_input  # noqa

        # TODO: Remove the `query_length != 1` check once Flash Attention for
        # RoCm is bumped to 2.1. For details, please see the comment in
        # LlamaFlashAttention2 __init__.
        causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, \
                max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask,
                query_length)

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size,
                                    query_length)
        else:
            attn_output = flash_attn_func(query_states,
                                          key_states,
                                          value_states,
                                          dropout,
                                          softmax_scale=softmax_scale,
                                          causal=causal)

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask,
                    query_length):
        from flash_attn.bert_padding import index_first_axis, unpad_input
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(
            attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads,
                              head_dim), indices_k)
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads,
                                head_dim), indices_k)
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads,
                                    head_dim), indices_k)
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = \
                unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->Siglip
class SiglipMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# Copied from transformers.models.clip.modeling_clip.CLIPEncoderLayer with
# CLIP->Siglip
class SiglipEncoderLayer(nn.Module):

    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = (SiglipAttention(config) if
                          not getattr(config, "_flash_attn_2_enabled", False)
                          else SiglipFlashAttention2(config))
        self.layer_norm1 = nn.LayerNorm(self.embed_dim,
                                        eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim,
                                        eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where
                padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all 
                attention layers. See `attentions` under returned tensors for
                more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, )

        if output_attentions:
            outputs += (attn_weights, )

        return outputs


class SiglipPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface 
    for downloading and loading pretrained models.
    """

    config_class = SiglipConfig
    base_model_prefix = "siglip"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""

        if isinstance(module, SiglipVisionEmbeddings):
            width = (self.config.vision_config.hidden_size if isinstance(
                self.config, SiglipConfig) else self.config.hidden_size)
            nn.init.normal_(module.position_embedding.weight,
                            std=1 / np.sqrt(width))
        elif isinstance(module, nn.Embedding):
            default_flax_embed_init(module.weight)
        elif isinstance(module, SiglipAttention):
            nn.init.normal_(module.q_proj.weight)
            nn.init.normal_(module.k_proj.weight)
            nn.init.normal_(module.v_proj.weight)
            nn.init.normal_(module.out_proj.weight)
            nn.init.zeros_(module.q_proj.bias)
            nn.init.zeros_(module.k_proj.bias)
            nn.init.zeros_(module.v_proj.bias)
            nn.init.zeros_(module.out_proj.bias)
        elif isinstance(module, SiglipMLP):
            nn.init.normal_(module.fc1.weight)
            nn.init.normal_(module.fc2.weight)
            nn.init.normal_(module.fc1.bias, std=1e-6)
            nn.init.normal_(module.fc2.bias, std=1e-6)
        elif isinstance(module, SiglipMultiheadAttentionPoolingHead):
            nn.init.normal_(module.probe.data)
            nn.init.normal_(module.attention.in_proj_weight.data)
            nn.init.zeros_(module.attention.in_proj_bias.data)
        elif isinstance(module, SiglipModel):
            logit_scale_init = torch.tensor(0.0)
            module.logit_scale.data.fill_(logit_scale_init)
            module.logit_bias.data.zero_()
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            lecun_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


SIGLIP_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass 
    documentation for the generic methods the library implements for all 
    its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)
    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/
    stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation
    for all matter related to general usage and behavior.
    Parameters:
        config ([`SiglipConfig`]): Model configuration class with all the 
            parameters of the model.
            Initializing with a config file does not load the weights 
            associated with the model, only the configuration. Check out 
            the [`~PreTrainedModel.from_pretrained`] method to load the 
            model weights.
"""

SIGLIP_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)
        `):
            Indices of input sequence tokens in the vocabulary. Padding will 
            be ignored by default should you provide it.
            Indices can be obtained using [`AutoTokenizer`]. See 
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`]
            for details. [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, 
            sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask
            values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, 
            sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position
            embeddings. Selected in the range `[0, 
            config.max_position_embeddings - 1]`.
            [What are position IDs?](../glossary#position-ids)
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention 
            layers. See `attentions` under returned tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See 
            `hidden_states` under returned tensors for more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a 
            plain tuple.
"""

SIGLIP_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size,
             num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you 
            provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`]
            for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention
            layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See 
            `hidden_states` under returned tensors for more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a 
            plain tuple.
"""

SIGLIP_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, 
        sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding 
            will be ignored by default should you provide it.
            Indices can be obtained using [`AutoTokenizer`]. See 
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`]
            for details. [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`
            , *optional*):
            Mask to avoid performing attention on padding token indices. Mask
            values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, 
            sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position
            embeddings. Selected in the range `[0, 
            config.max_position_embeddings - 1]`.
            [What are position IDs?](../glossary#position-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, 
            num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you 
            provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] 
            for details.
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention 
            layers. See `attentions` under returned tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See 
            `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a 
            plain tuple.
"""


# Copied from transformers.models.clip.modeling_clip.CLIPEncoder with
# CLIP->Siglip
class SiglipEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` 
    self attention layers. Each layer is a [`SiglipEncoderLayer`].
    Args:
        config: SiglipConfig
    """

    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.gradient_checkpointing = False

    # Ignore copy
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, 
                sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to 
                directly pass an embedded representation.
                This is useful if you want more control over how to convert 
                `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, 
                sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. 
                Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all 
                attention layers. See `attentions` under returned tensors for
                  more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See 
                `hidden_states` under returned tensors for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a
                  plain tuple.
        """
        output_attentions = output_attentions if output_attentions \
            is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else \
            self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states, )
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1], )

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states, )

        if not return_dict:
            return tuple(
                v for v in [hidden_states, encoder_states, all_attentions]
                if v is not None)
        return BaseModelOutput(last_hidden_state=hidden_states,
                               hidden_states=encoder_states,
                               attentions=all_attentions)


class SiglipTextTransformer(nn.Module):

    def __init__(self, config: SiglipTextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = SiglipTextEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim,
                                             eps=config.layer_norm_eps)

        self.head = nn.Linear(embed_dim, embed_dim)

    @add_start_docstrings_to_model_forward(SIGLIP_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling,
                               config_class=SiglipTextConfig)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
        """
        output_attentions = output_attentions if output_attentions \
            is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states \
                                    is not None else
                                self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else \
            self.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids,
                                        position_ids=position_ids)

        # note: SigLIP's text model does not use a causal mask, unlike the
        # original CLIP model.
        # expand attention_mask
        if attention_mask is not None:
            # [batch_size, seq_len] ->
            # [batch_size, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(
                attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # Assuming "sticky" EOS tokenization, last token is always EOS.
        pooled_output = last_hidden_state[:, -1, :]
        pooled_output = self.head(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    """The text model from SigLIP without any head or projection on top.""",
    SIGLIP_START_DOCSTRING,
)
class SiglipTextModel(SiglipPreTrainedModel):
    config_class = SiglipTextConfig

    _no_split_modules = ["SiglipTextEmbeddings", "SiglipEncoderLayer"]

    def __init__(self, config: SiglipTextConfig):
        super().__init__(config)
        self.text_model = SiglipTextTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    @add_start_docstrings_to_model_forward(SIGLIP_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling,
                               config_class=SiglipTextConfig)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
        Examples:
        ```python
        >>> from transformers import AutoTokenizer, SiglipTextModel
        >>> model = SiglipTextModel.
            from_pretrained("google/siglip-base-patch16-224")
        >>> tokenizer = AutoTokenizer.
            from_pretrained("google/siglip-base-patch16-224")
        >>> # important: make sure to set padding="max_length" 
            as that's how the model was trained
        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], 
            padding="max_length", return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) 
            states
        ```"""
        return_dict = return_dict if return_dict is not None else \
            self.config.use_return_dict

        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class SiglipVisionTransformer(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim,
                                           eps=config.layer_norm_eps)
        self.head = SiglipMultiheadAttentionPoolingHead(config)

    @add_start_docstrings_to_model_forward(SIGLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling,
                               config_class=SiglipVisionConfig)
    def forward(
        self,
        pixel_values,
        patch_attention_mask: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
        """
        output_attentions = output_attentions if output_attentions is not None\
              else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None \
            else self.config.use_return_dict

        batch_size = pixel_values.size(0)
        if patch_attention_mask is None:
            patch_attention_mask = torch.ones(
                size=(
                    batch_size,
                    pixel_values.size(2) // self.config.patch_size,
                    pixel_values.size(3) // self.config.patch_size,
                ),
                dtype=torch.bool,
                device=pixel_values.device,
            )

        hidden_states = self.embeddings(
            pixel_values=pixel_values,
            patch_attention_mask=patch_attention_mask)

        patch_attention_mask = patch_attention_mask.view(batch_size, -1)
        # The call to `_upad_input` in `_flash_attention_forward` is expensive
        # So when the `patch_attention_mask` is full of 1s (i.e. attending
        # to the whole sequence), avoiding passing the attention_mask, which
        # is equivalent to attending to the full sequence
        if not torch.any(~patch_attention_mask):
            attention_mask = None
        else:
            attention_mask = (_prepare_4d_attention_mask(
                patch_attention_mask, hidden_states.dtype)
                              if not self.config._flash_attn_2_enabled else
                              patch_attention_mask)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooled_output = self.head(
            hidden_state=last_hidden_state,
            attention_mask=patch_attention_mask,
        )

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class SiglipMultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = torch.nn.MultiheadAttention(
            config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(self, hidden_state, attention_mask):
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        hidden_state = self.attention(query=probe,
                                      key=hidden_state,
                                      value=hidden_state,
                                      key_padding_mask=~attention_mask)[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]


@add_start_docstrings(
    """The vision model from SigLIP without any head or projection on top.""",
    SIGLIP_START_DOCSTRING,
)
class SiglipVisionModel(SiglipPreTrainedModel):
    config_class = SiglipVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: SiglipVisionConfig):
        super().__init__(config)

        self.vision_model = SiglipVisionTransformer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    @add_start_docstrings_to_model_forward(SIGLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling,
                               config_class=SiglipVisionConfig)
    def forward(
        self,
        pixel_values,
        patch_attention_mask: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, SiglipVisionModel
        >>> model = SiglipVisionModel.from_pretrained(
            "google/siglip-base-patch16-224")
        >>> processor = AutoProcessor.from_pretrained(
            "google/siglip-base-patch16-224")
        >>> url = 
            "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled features
        ```"""
        return_dict = return_dict if return_dict is not None \
            else self.config.use_return_dict

        return self.vision_model(
            pixel_values=pixel_values,
            patch_attention_mask=patch_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


@add_start_docstrings(SIGLIP_START_DOCSTRING)
class SiglipModel(SiglipPreTrainedModel):
    config_class = SiglipConfig

    def __init__(self, config: SiglipConfig):
        super().__init__(config)

        if not isinstance(config.text_config, SiglipTextConfig):
            raise ValueError("config.text_config is expected to be of type "
                             f"SiglipTextConfig but is of type"
                             f" {type(config.text_config)}.")

        if not isinstance(config.vision_config, SiglipVisionConfig):
            raise ValueError("config.vision_config is expected to be of type "
                             "SiglipVisionConfig but is of type"
                             f" {type(config.vision_config)}.")

        text_config = config.text_config
        vision_config = config.vision_config

        self.text_model = SiglipTextTransformer(text_config)
        self.vision_model = SiglipVisionTransformer(vision_config)

        self.logit_scale = nn.Parameter(torch.randn(1))
        self.logit_bias = nn.Parameter(torch.randn(1))

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(SIGLIP_TEXT_INPUTS_DOCSTRING)
    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size,
              output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output
              of [`SiglipTextModel`].
        Examples:
        ```python
        >>> from transformers import AutoTokenizer, AutoModel
        >>> import torch
        >>> model = AutoModel.from_pretrained(
            "google/siglip-base-patch16-224")
        >>> tokenizer = AutoTokenizer.from_pretrained(
            "google/siglip-base-patch16-224")
        >>> # important: make sure to set padding="max_length" as that's 
            how the model was trained
        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], 
            padding="max_length", return_tensors="pt")
        >>> with torch.no_grad():
        ...     text_features = model.get_text_features(**inputs)
        ```"""
        # Use SigLIP model's config for some fields (if specified) instead
        # of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None\
              else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None \
            else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = text_outputs[1]

        return pooled_output

    @add_start_docstrings_to_model_forward(SIGLIP_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, 
            output_dim`): The image embeddings obtained by applying the
            projection layer to the pooled output of [`SiglipVisionModel`].
        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, AutoModel
        >>> import torch
        >>> model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        >>> processor = AutoProcessor.from_pretrained(
            "google/siglip-base-patch16-224")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(images=image, return_tensors="pt")
        >>> with torch.no_grad():
        ...     image_features = model.get_image_features(**inputs)
        ```"""
        # Use SiglipModel's config for some fields (if specified) instead
        # of those of vision & text components.
        output_attentions = output_attentions if output_attentions \
            is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else \
            self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]

        return pooled_output

    @add_start_docstrings_to_model_forward(SIGLIP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SiglipOutput,
                               config_class=SiglipConfig)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SiglipOutput]:
        r"""
        Returns:
        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, AutoModel
        >>> import torch
        >>> model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        >>> processor = AutoProcessor.from_pretrained(
            "google/siglip-base-patch16-224")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> texts = ["a photo of 2 cats", "a photo of 2 dogs"]
        >>> # important: we pass `padding=max_length` since the model was 
            trained with this
        >>> inputs = processor(text=texts, images=image, 
            padding="max_length", return_tensors="pt")
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image
        >>> probs = torch.sigmoid(logits_per_image) # these are the 
            probabilities
        >>> print(f"{probs[0][0]:.1%} that image 0 is '{texts[0]}'")
        31.9% that image 0 is 'a photo of 2 cats'
        ```"""
        # Use SigLIP model's config for some fields (if specified) instead of
        # those of vision & text components.
        output_attentions = output_attentions if output_attentions \
            is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else \
            self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[1]
        text_embeds = text_outputs[1]

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(
            p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_text = torch.matmul(text_embeds, image_embeds.t(
        )) * self.logit_scale.exp() + self.logit_bias
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            raise NotImplementedError("SigLIP loss to be implemented")

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds,
                      image_embeds, text_outputs, vision_outputs)
            return ((loss, ) + output) if loss is not None else output

        return SiglipOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


def get_siglip_vision_model(_flash_attn_2_enabled=True, **kwargs):
    siglip_vision_config = {
        "hidden_size": 1152,
        "image_size": 448,
        "intermediate_size": 4304,
        "model_type": "siglip_vision_model",
        "num_attention_heads": 16,
        "num_hidden_layers": 27,
        "patch_size": 14,
    }

    # Detect attention implementation.
    attn_backend: _Backend = get_vit_attn_backend(support_fa=True)
    if attn_backend != _Backend.FLASH_ATTN:
        _flash_attn_2_enabled = False

    model_config = SiglipVisionConfig(
        **siglip_vision_config,
        _flash_attn_2_enabled=_flash_attn_2_enabled,
        **kwargs)

    vision_model = SiglipVisionModel(model_config).vision_model

    return vision_model
