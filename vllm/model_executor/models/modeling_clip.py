#cp ../data/minimax-dialogue/experiment/pretrain/vlm_test/stage2_abab7s_projenfc/newstage2_1_0_3000/iter_0002000/converted_hf_mixed_precision/modeling_clip.py /root/.cache/huggingface/modules/transformers_modules/converted_hf_mixed_precision/modeling_clip.py

# coding=utf-8
# Copyright 2021 The OpenAI Team Authors and The HuggingFace Team. All rights reserved.
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
"""PyTorch CLIP model."""

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union, List

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import is_torch_greater_or_equal_than_2_2
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
#from .configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
#from .image_utils import get_hw_multiple_of

if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func

    #from ...modeling_flash_attention_utils import _flash_attention_forward
else:
    flash_attn_varlen_func = None
logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "CLIPConfig"
_CHECKPOINT_FOR_DOC = "openai/clip-vit-base-patch32"

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "openai/clip-vit-base-patch32"
_IMAGE_CLASS_EXPECTED_OUTPUT = "LABEL_0"

import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union


if TYPE_CHECKING:
    from transformers.processing_utils import ProcessorMixin
    from transformers.utils import TensorType

from transformers.configuration_utils import PretrainedConfig

def get_hw_multiple_of(image_size, multiple, max_size=None):
    w, h = image_size
    new_w = w if w % multiple == 0 else w + (multiple - w % multiple)
    new_h = h if h % multiple == 0 else h + (multiple - h % multiple)
    if max_size is not None:
        assert isinstance(max_size, (list, tuple)) and len(max_size) == 2
        max_w, max_h = max_size
        assert max_w % multiple == 0 and max_h % multiple == 0
        if new_w > max_w or new_h > max_h:
            # ratio = min(max_w / new_w, max_h / new_h)
            # new_w = int(new_w * ratio)
            # new_h = int(new_h * ratio)
            new_w = min((new_w * max_w) // new_w, (new_w * max_h) // new_h)
            new_h = min((new_h * max_w) // new_w, (new_h * max_h) // new_h)

            new_w = new_w if new_w % multiple == 0 else new_w + (multiple - new_w % multiple)
            new_h = new_h if new_h % multiple == 0 else new_h + (multiple - new_h % multiple)
        assert new_w % multiple == 0 and new_h % multiple == 0
        assert new_w <= max_w and new_h <= max_h
    return new_w, new_h

class CLIPVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CLIPVisionModel`]. It is used to instantiate a
    CLIP vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the vision encoder of the CLIP
    [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        projection_dim (`int`, *optional*, defaults to 512):
            Dimensionality of text and vision projection layers.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 32):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
    Example:
    ```python
    >>> from transformers import CLIPVisionConfig, CLIPVisionModel
    >>> # Initializing a CLIPVisionConfig with openai/clip-vit-base-patch32 style configuration
    >>> configuration = CLIPVisionConfig()
    >>> # Initializing a CLIPVisionModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
    >>> model = CLIPVisionModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "clip_vision_model"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        projection_dim=512,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=32,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        position_embedding_type='learned_absolute',
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.position_embedding_type = position_embedding_type

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from CLIPConfig
        if config_dict.get("model_type") == "clip":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/2021-03-07-clip.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


def _get_vector_norm(tensor: torch.Tensor) -> torch.Tensor:
    """
    This method is equivalent to tensor.norm(p=2, dim=-1, keepdim=True) and used to make
    model `executorch` exportable. See issue https://github.com/pytorch/executorch/issues/3566
    """
    square_tensor = torch.pow(tensor, 2)
    sum_tensor = torch.sum(square_tensor, dim=-1, keepdim=True)
    normed_tensor = torch.pow(sum_tensor, 0.5)
    return normed_tensor


@dataclass
class CLIPVisionModelOutput(ModelOutput):
    """
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.
    Args:
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class CLIPTextModelOutput(ModelOutput):
    """
    Base class for text model's outputs that also contains a pooling of the last hidden states.
    Args:
        text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The text embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    text_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class CLIPOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`CLIPTextModel`].
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`CLIPVisionModel`].
        text_model_output (`BaseModelOutputWithPooling`):
            The output of the [`CLIPTextModel`].
        vision_model_output (`BaseModelOutputWithPooling`):
            The output of the [`CLIPVisionModel`].
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
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        if self.config.position_embedding_type == "learned_absolute":
            self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
            self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        if self.patch_embedding.weight.dtype != pixel_values.dtype:
            self.patch_embedding = self.patch_embedding.to(pixel_values.dtype)
            #self.patch_embedding.weight.copy_(self.patch_embedding.weight.to(pixel_values.dtype))
            print("DEBUG: weight dtype for patch embedding", self.patch_embedding.weight.dtype)
        #patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        #patch_embeds = patch_embeds.flatten(2).transpose(1, 2).to(dtype=torch.bfloat16)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2).to(dtype=self.class_embedding.dtype)
        if self.config.position_embedding_type == "learned_absolute":
            class_embeds = self.class_embedding.expand(batch_size, 1, -1)
            embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
            embeddings = embeddings + self.position_embedding(self.position_ids)
        else:
            embeddings = patch_embeds
        return embeddings


# class CLIPTextEmbeddings(nn.Module):
#     def __init__(self, config: CLIPTextConfig):
#         super().__init__()
#         embed_dim = config.hidden_size

#         self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
#         self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

#         # position_ids (1, len position emb) is contiguous in memory and exported when serialized
#         self.register_buffer(
#             "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
#         )

#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#     ) -> torch.Tensor:
#         seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

#         if position_ids is None:
#             position_ids = self.position_ids[:, :seq_length]

#         if inputs_embeds is None:
#             inputs_embeds = self.token_embedding(input_ids)

#         position_embeddings = self.position_embedding(position_ids)
#         embeddings = inputs_embeds + position_embeddings

#         return embeddings

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_vision(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    output = output.to(orig_dtype)
    return output


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs

class VisionFlashAttention2(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        # self.num_heads = num_heads
        # self.qkv = nn.Linear(dim, dim * 3, bias=True)
        # self.proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self, hidden_states: torch.Tensor, cu_seq_len: torch.Tensor, max_seqlen=None, rotary_pos_emb: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q = self.q_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)

        #q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if self.config.position_embedding_type == "rope":
            q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
            k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        attn_output = flash_attn_varlen_func(q, k, v, cu_seq_len, cu_seq_len, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        
        attn_output = self.out_proj(attn_output).unsqueeze(1)
        return attn_output, None



# class CLIPAttention(nn.Module):
#     """Multi-headed attention from 'Attention Is All You Need' paper"""

#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.embed_dim = config.hidden_size
#         self.num_heads = config.num_attention_heads
#         self.head_dim = self.embed_dim // self.num_heads
#         if self.head_dim * self.num_heads != self.embed_dim:
#             raise ValueError(
#                 f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
#                 f" {self.num_heads})."
#             )
#         self.scale = self.head_dim**-0.5
#         self.dropout = config.attention_dropout

#         self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
#         self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
#         self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
#         self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

#     def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
#         return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         causal_attention_mask: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = False,
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
#         """Input shape: Batch x Time x Channel"""

#         bsz, tgt_len, embed_dim = hidden_states.size()

#         # get query proj
#         query_states = self.q_proj(hidden_states) * self.scale
#         key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
#         value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

#         proj_shape = (bsz * self.num_heads, -1, self.head_dim)
#         query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
#         key_states = key_states.view(*proj_shape)
#         value_states = value_states.view(*proj_shape)

#         src_len = key_states.size(1)
#         attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

#         if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
#             raise ValueError(
#                 f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
#                 f" {attn_weights.size()}"
#             )

#         # apply the causal_attention_mask first
#         if causal_attention_mask is not None:
#             if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
#                 raise ValueError(
#                     f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
#                     f" {causal_attention_mask.size()}"
#                 )
#             attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
#             attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

#         if attention_mask is not None:
#             if attention_mask.size() != (bsz, 1, tgt_len, src_len):
#                 raise ValueError(
#                     f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
#                 )
#             attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
#             attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

#         attn_weights = nn.functional.softmax(attn_weights, dim=-1)

#         if output_attentions:
#             # this operation is a bit akward, but it's required to
#             # make sure that attn_weights keeps its gradient.
#             # In order to do so, attn_weights have to reshaped
#             # twice and have to be reused in the following
#             attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
#             attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
#         else:
#             attn_weights_reshaped = None

#         attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

#         attn_output = torch.bmm(attn_probs, value_states)

#         if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
#             raise ValueError(
#                 f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
#                 f" {attn_output.size()}"
#             )

#         attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
#         attn_output = attn_output.transpose(1, 2)
#         attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

#         attn_output = self.out_proj(attn_output)

#         return attn_output, attn_weights_reshaped


# class CLIPFlashAttention2(CLIPAttention):
#     """
#     CLIPAttention flash attention module. This module inherits from `CLIPAttention` as the weights of the module stays
#     untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
#     flash attention and deal with padding tokens in case the input contains any of them.
#     """

#     # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
#         # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
#         # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
#         self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

#     # Adapted from transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward
#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         causal_attention_mask: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = False,
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
#         output_attentions = False

#         batch_size, q_len, _ = hidden_states.size()

#         query_states = self.q_proj(hidden_states)
#         key_states = self.k_proj(hidden_states)
#         value_states = self.v_proj(hidden_states)

#         # Flash attention requires the input to have the shape
#         # batch_size x seq_length x head_dim x hidden_dim
#         # therefore we just need to keep the original shape
#         query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim)
#         key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim)
#         value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim)

#         dropout_rate = self.dropout if self.training else 0.0

#         # In PEFT, usually we cast the layer norms in float32 for training stability reasons
#         # therefore the input hidden states gets silently casted in float32. Hence, we need
#         # cast them back in the correct dtype just to be sure everything works as expected.
#         # This might slowdown training & inference so it is recommended to not cast the LayerNorms
#         # in fp32.

#         input_dtype = query_states.dtype
#         if input_dtype == torch.float32:
#             if torch.is_autocast_enabled():
#                 target_dtype = torch.get_autocast_gpu_dtype()
#             # Handle the case where the model is quantized
#             elif hasattr(self.config, "_pre_quantization_dtype"):
#                 target_dtype = self.config._pre_quantization_dtype
#             else:
#                 target_dtype = self.q_proj.weight.dtype

#             logger.warning_once(
#                 f"The input hidden states seems to be silently casted in float32, this might be related to"
#                 f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
#                 f" {target_dtype}."
#             )

#             query_states = query_states.to(target_dtype)
#             key_states = key_states.to(target_dtype)
#             value_states = value_states.to(target_dtype)

#         attn_output = _flash_attention_forward(
#             query_states,
#             key_states,
#             value_states,
#             attention_mask,
#             q_len,
#             dropout=dropout_rate,
#             is_causal=causal_attention_mask is not None,
#             use_top_left_mask=self._flash_attn_uses_top_left_mask,
#         )

#         attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim).contiguous()
#         attn_output = self.out_proj(attn_output)

#         if not output_attentions:
#             attn_weights = None

#         return attn_output, attn_weights


# class CLIPSdpaAttention(CLIPAttention):
#     """
#     SDPA attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
#     `CLIPAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
#     SDPA API.
#     """

#     # Adapted from CLIPAttention.forward
#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         causal_attention_mask: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = False,
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
#         if output_attentions:
#             # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
#             logger.warning_once(
#                 "CLIPModel is using CLIPSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not "
#                 "support `output_attentions=True`. Falling back to the manual attention implementation, but specifying "
#                 "the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can "
#                 'be removed using the argument `attn_implementation="eager"` when loading the model.'
#             )
#             return super().forward(
#                 hidden_states=hidden_states,
#                 attention_mask=attention_mask,
#                 causal_attention_mask=causal_attention_mask,
#                 output_attentions=output_attentions,
#             )

#         # CLIP text model uses both `causal_attention_mask` and `attention_mask`
#         if attention_mask is not None and causal_attention_mask is not None:
#             attn_mask = attention_mask + causal_attention_mask
#         elif causal_attention_mask is not None:
#             attn_mask = causal_attention_mask
#         else:
#             attn_mask = attention_mask

#         bsz, tgt_len, embed_dim = hidden_states.size()

#         query_states = self.q_proj(hidden_states)
#         key_states = self.k_proj(hidden_states)
#         value_states = self.v_proj(hidden_states)

#         query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
#         key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
#         value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

#         # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
#         # Reference: https://github.com/pytorch/pytorch/issues/112577.
#         if not is_torch_greater_or_equal_than_2_2 and query_states.device.type == "cuda" and attn_mask is not None:
#             query_states = query_states.contiguous()
#             key_states = key_states.contiguous()
#             value_states = value_states.contiguous()

#         # CLIP text model uses both `causal_attention_mask` and `attention_mask` sequentially.
#         attn_output = torch.nn.functional.scaled_dot_product_attention(
#             query_states,
#             key_states,
#             value_states,
#             attn_mask=attn_mask,
#             dropout_p=self.dropout if self.training else 0.0,
#             scale=self.scale,
#         )

#         attn_output = attn_output.transpose(1, 2)
#         attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

#         attn_output = self.out_proj(attn_output)

#         return attn_output, None


# CLIP_ATTENTION_CLASSES = {
#     "eager": CLIPAttention,
#     "sdpa": CLIPSdpaAttention,
#     "flash_attention_2": CLIPFlashAttention2,
# }


class CLIPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        device = hidden_states.device
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        #self.self_attn = CLIP_ATTENTION_CLASSES[config._attn_implementation](config)
        self.self_attn = VisionFlashAttention2(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        cu_seq_len=None,
        max_seqlen=None,
        rotary_pos_emb=None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            cu_seq_len=cu_seq_len,
            max_seqlen=max_seqlen,
            rotary_pos_emb=rotary_pos_emb,
            output_attentions=output_attentions,
        )
        device = hidden_states.device
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class CLIPPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CLIPVisionConfig
    base_model_prefix = "clip"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        # if isinstance(module, CLIPTextEmbeddings):
        #     module.token_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
        #     module.position_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
        if isinstance(module, CLIPVisionEmbeddings):
            factor = self.config.initializer_factor
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            nn.init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            #nn.init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
        # elif isinstance(module, CLIPAttention):
        #     factor = self.config.initializer_factor
        #     in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
        #     out_proj_std = (module.embed_dim**-0.5) * factor
        #     nn.init.normal_(module.q_proj.weight, std=in_proj_std)
        #     nn.init.normal_(module.k_proj.weight, std=in_proj_std)
        #     nn.init.normal_(module.v_proj.weight, std=in_proj_std)
        #     nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        elif isinstance(module, CLIPMLP):
            factor = self.config.initializer_factor
            in_proj_std = (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            nn.init.normal_(module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)
        # elif isinstance(module, CLIPModel):
        #     nn.init.normal_(
        #         module.text_projection.weight,
        #         std=module.text_embed_dim**-0.5 * self.config.initializer_factor,
        #     )
        #     nn.init.normal_(
        #         module.visual_projection.weight,
        #         std=module.vision_embed_dim**-0.5 * self.config.initializer_factor,
        #     )
        # elif isinstance(module, CLIPVisionModelWithProjection):
        #     nn.init.normal_(
        #         module.visual_projection.weight,
        #         std=self.config.hidden_size**-0.5 * self.config.initializer_factor,
        #     )
        # elif isinstance(module, CLIPTextModelWithProjection):
        #     nn.init.normal_(
        #         module.text_projection.weight,
        #         std=self.config.hidden_size**-0.5 * self.config.initializer_factor,
        #     )
        # elif isinstance(module, CLIPForImageClassification):
        #     nn.init.normal_(
        #         module.classifier.weight,
        #         std=self.config.vision_config.hidden_size**-0.5 * self.config.initializer_factor,
        #     )

        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


CLIP_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    Parameters:
        config ([`CLIPConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

CLIP_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
            [What are position IDs?](../glossary#position-ids)
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

CLIP_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

CLIP_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
            [What are position IDs?](../glossary#position-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class CLIPEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`CLIPEncoderLayer`].
    Args:
        config: CLIPConfig
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        cu_seq_len: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    cu_seq_len,
                    max_seqlen,
                    rotary_pos_emb,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    cu_seq_len,
                    max_seqlen,
                    rotary_pos_emb,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states.permute(1,0,2),)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states.permute(1,0,2), hidden_states=encoder_states, attentions=all_attentions
        )


# class CLIPTextTransformer(nn.Module):
#     def __init__(self, config: CLIPTextConfig):
#         super().__init__()
#         self.config = config
#         embed_dim = config.hidden_size
#         self.embeddings = CLIPTextEmbeddings(config)
#         self.encoder = CLIPEncoder(config)
#         self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

#         # For `pooled_output` computation
#         self.eos_token_id = config.eos_token_id

#         # For attention mask, it differs between `flash_attention_2` and other attention implementations
#         self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

#     @add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING)
#     @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPTextConfig)
#     def forward(
#         self,
#         input_ids: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, BaseModelOutputWithPooling]:
#         r"""
#         Returns:

#         """
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if input_ids is None:
#             raise ValueError("You have to specify input_ids")

#         input_shape = input_ids.size()
#         input_ids = input_ids.view(-1, input_shape[-1])

#         hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

#         # CLIP's text model uses causal mask, prepare it here.
#         # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
#         causal_attention_mask = _create_4d_causal_attention_mask(
#             input_shape, hidden_states.dtype, device=hidden_states.device
#         )

#         # expand attention_mask
#         if attention_mask is not None and not self._use_flash_attention_2:
#             # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
#             attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

#         encoder_outputs = self.encoder(
#             inputs_embeds=hidden_states,
#             attention_mask=attention_mask,
#             causal_attention_mask=causal_attention_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         last_hidden_state = encoder_outputs[0]
#         last_hidden_state = self.final_layer_norm(last_hidden_state)

#         if self.eos_token_id == 2:
#             # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
#             # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
#             # ------------------------------------------------------------
#             # text_embeds.shape = [batch_size, sequence_length, transformer.width]
#             # take features from the eot embedding (eot_token is the highest number in each sequence)
#             # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
#             pooled_output = last_hidden_state[
#                 torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
#                 input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
#             ]
#         else:
#             # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
#             pooled_output = last_hidden_state[
#                 torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
#                 # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
#                 # Note: we assume each sequence (along batch dim.) contains an  `eos_token_id` (e.g. prepared by the tokenizer)
#                 (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.eos_token_id)
#                 .int()
#                 .argmax(dim=-1),
#             ]

#         if not return_dict:
#             return (last_hidden_state, pooled_output) + encoder_outputs[1:]

#         return BaseModelOutputWithPooling(
#             last_hidden_state=last_hidden_state,
#             pooler_output=pooled_output,
#             hidden_states=encoder_outputs.hidden_states,
#             attentions=encoder_outputs.attentions,
#         )


# @add_start_docstrings(
#     """The text model from CLIP without any head or projection on top.""",
#     CLIP_START_DOCSTRING,
# )
# class CLIPTextModel(CLIPPreTrainedModel):
#     config_class = CLIPTextConfig

#     _no_split_modules = ["CLIPTextEmbeddings", "CLIPEncoderLayer"]

#     def __init__(self, config: CLIPTextConfig):
#         super().__init__(config)
#         self.text_model = CLIPTextTransformer(config)
#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_input_embeddings(self) -> nn.Module:
#         return self.text_model.embeddings.token_embedding

#     def set_input_embeddings(self, value):
#         self.text_model.embeddings.token_embedding = value

#     @add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING)
#     @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPTextConfig)
#     def forward(
#         self,
#         input_ids: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, BaseModelOutputWithPooling]:
#         r"""
#         Returns:

#         Examples:

#         ```python
#         >>> from transformers import AutoTokenizer, CLIPTextModel

#         >>> model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
#         >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

#         >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

#         >>> outputs = model(**inputs)
#         >>> last_hidden_state = outputs.last_hidden_state
#         >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
#         ```"""
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         return self.text_model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )


class CLIPVisionTransformer(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        if self.config.position_embedding_type == 'rope':
            head_dim = embed_dim // config.num_attention_heads
            self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        t = 1
        spatial_merge_size = 1
        max_grid_size = 0
        resized_wh = []
        for h, w in grid_thw:
            w = w.item()
            h = h.item()
            w, h = get_hw_multiple_of((w, h), self.config.patch_size, max_size=(self.config.image_size, self.config.image_size))
            w = w // self.config.patch_size
            h = h // self.config.patch_size
            resized_wh.append((w, h))
            max_grid_size = max(max_grid_size, w, h)
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // spatial_merge_size,
                spatial_merge_size,
                w // spatial_merge_size,
                spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // spatial_merge_size,
                spatial_merge_size,
                w // spatial_merge_size,
                spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        #max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        
        inputs = []
        cu_seq_len = [0]
        max_seqlen = 0
        for pixel_value in pixel_values:
            hidden_states = self.embeddings(pixel_value.unsqueeze(0))
            if hidden_states.dtype != self.pre_layrnorm.weight.dtype:
                hidden_states = hidden_states.to(self.pre_layrnorm.weight.dtype)
            hidden_states = self.pre_layrnorm(hidden_states)
            hidden_states = hidden_states.permute(1, 0, 2).contiguous()
            cu_seq_len.append(hidden_states.size(0))
            max_seqlen = max(max_seqlen, hidden_states.size(0))
            inputs.append(hidden_states)
        inputs = torch.cat(inputs, dim=0)
        cu_seq_len = torch.tensor(cu_seq_len, device=inputs.device).to(torch.int32)
        cu_seq_len = torch.cumsum(cu_seq_len, dim=0).to(torch.int32)
        max_seqlen = torch.tensor([max_seqlen],  device=inputs.device).to(torch.int32)
        rotary_pos_emb = None
        if self.config.position_embedding_type == "rope":
            assert image_sizes is not None
            rotary_pos_emb = self.rot_pos_emb(image_sizes)
        encoder_outputs = self.encoder(
            inputs_embeds=inputs,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cu_seq_len=cu_seq_len,
            max_seqlen=max_seqlen,
            rotary_pos_emb=rotary_pos_emb,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    """The vision model from CLIP without any head or projection on top.""",
    CLIP_START_DOCSTRING,
)
class CLIPVisionModel(CLIPPreTrainedModel):
    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        self.vision_model = CLIPVisionTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
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
        >>> from transformers import AutoProcessor, CLIPVisionModel
        >>> model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        return self.vision_model(
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


# @add_start_docstrings(CLIP_START_DOCSTRING)
# class CLIPModel(CLIPPreTrainedModel):
#     config_class = CLIPConfig
#     _no_split_modules = ["CLIPTextEmbeddings", "CLIPEncoderLayer", "CLIPVisionEmbeddings"]

#     def __init__(self, config: CLIPConfig):
#         super().__init__(config)

#         if not isinstance(config.text_config, CLIPTextConfig):
#             raise TypeError(
#                 "config.text_config is expected to be of type CLIPTextConfig but is of type"
#                 f" {type(config.text_config)}."
#             )

#         if not isinstance(config.vision_config, CLIPVisionConfig):
#             raise TypeError(
#                 "config.vision_config is expected to be of type CLIPVisionConfig but is of type"
#                 f" {type(config.vision_config)}."
#             )

#         text_config = config.text_config
#         vision_config = config.vision_config

#         self.projection_dim = config.projection_dim
#         self.text_embed_dim = text_config.hidden_size
#         self.vision_embed_dim = vision_config.hidden_size

#         text_model = CLIPTextModel._from_config(text_config, attn_implementation=config._attn_implementation)
#         self.text_model = text_model.text_model

#         vision_model = CLIPVisionModel._from_config(vision_config, attn_implementation=config._attn_implementation)
#         self.vision_model = vision_model.vision_model

#         self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
#         self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
#         self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

#         # Initialize weights and apply final processing
#         self.post_init()

#     @add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING)
#     def get_text_features(
#         self,
#         input_ids: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> torch.FloatTensor:
#         r"""
#         Returns:
#             text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
#             applying the projection layer to the pooled output of [`CLIPTextModel`].

#         Examples:

#         ```python
#         >>> from transformers import AutoTokenizer, CLIPModel

#         >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#         >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

#         >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
#         >>> text_features = model.get_text_features(**inputs)
#         ```"""
#         # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         text_outputs = self.text_model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         pooled_output = text_outputs[1]
#         text_features = self.text_projection(pooled_output)

#         return text_features

#     @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
#     def get_image_features(
#         self,
#         pixel_values: Optional[torch.FloatTensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> torch.FloatTensor:
#         r"""
#         Returns:
#             image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
#             applying the projection layer to the pooled output of [`CLIPVisionModel`].

#         Examples:

#         ```python
#         >>> from PIL import Image
#         >>> import requests
#         >>> from transformers import AutoProcessor, CLIPModel

#         >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#         >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

#         >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#         >>> image = Image.open(requests.get(url, stream=True).raw)

#         >>> inputs = processor(images=image, return_tensors="pt")

#         >>> image_features = model.get_image_features(**inputs)
#         ```"""
#         # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         vision_outputs = self.vision_model(
#             pixel_values=pixel_values,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         pooled_output = vision_outputs[1]  # pooled_output
#         image_features = self.visual_projection(pooled_output)

#         return image_features

#     @add_start_docstrings_to_model_forward(CLIP_INPUTS_DOCSTRING)
#     @replace_return_docstrings(output_type=CLIPOutput, config_class=CLIPConfig)
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         pixel_values: Optional[torch.FloatTensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         return_loss: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, CLIPOutput]:
#         r"""
#         Returns:

#         Examples:

#         ```python
#         >>> from PIL import Image
#         >>> import requests
#         >>> from transformers import AutoProcessor, CLIPModel

#         >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#         >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

#         >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#         >>> image = Image.open(requests.get(url, stream=True).raw)

#         >>> inputs = processor(
#         ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
#         ... )

#         >>> outputs = model(**inputs)
#         >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
#         >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
#         ```"""
#         # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         vision_outputs = self.vision_model(
#             pixel_values=pixel_values,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         text_outputs = self.text_model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         image_embeds = vision_outputs[1]
#         image_embeds = self.visual_projection(image_embeds)

#         text_embeds = text_outputs[1]
#         text_embeds = self.text_projection(text_embeds)

#         # normalized features
#         image_embeds = image_embeds / _get_vector_norm(image_embeds)
#         text_embeds = text_embeds / _get_vector_norm(text_embeds)

#         # cosine similarity as logits
#         logit_scale = self.logit_scale.exp()
#         logits_per_text = torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device)) * logit_scale.to(
#             text_embeds.device
#         )
#         logits_per_image = logits_per_text.t()

#         loss = None
#         if return_loss:
#             loss = clip_loss(logits_per_text)

#         if not return_dict:
#             output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
#             return ((loss,) + output) if loss is not None else output

#         return CLIPOutput(
#             loss=loss,
#             logits_per_image=logits_per_image,
#             logits_per_text=logits_per_text,
#             text_embeds=text_embeds,
#             image_embeds=image_embeds,
#             text_model_output=text_outputs,
#             vision_model_output=vision_outputs,
#         )


# @add_start_docstrings(
#     """
#     CLIP Text Model with a projection layer on top (a linear layer on top of the pooled output).
#     """,
#     CLIP_START_DOCSTRING,
# )
# class CLIPTextModelWithProjection(CLIPPreTrainedModel):
#     config_class = CLIPTextConfig

#     _no_split_modules = ["CLIPTextEmbeddings", "CLIPEncoderLayer"]

#     def __init__(self, config: CLIPTextConfig):
#         super().__init__(config)

#         text_model = CLIPTextModel._from_config(config, attn_implementation=config._attn_implementation)
#         self.text_model = text_model.text_model

#         self.text_projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_input_embeddings(self) -> nn.Module:
#         return self.text_model.embeddings.token_embedding

#     def set_input_embeddings(self, value):
#         self.text_model.embeddings.token_embedding = value

#     @add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING)
#     @replace_return_docstrings(output_type=CLIPTextModelOutput, config_class=CLIPTextConfig)
#     def forward(
#         self,
#         input_ids: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, CLIPTextModelOutput]:
#         r"""
#         Returns:

#         Examples:

#         ```python
#         >>> from transformers import AutoTokenizer, CLIPTextModelWithProjection

#         >>> model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
#         >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

#         >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

#         >>> outputs = model(**inputs)
#         >>> text_embeds = outputs.text_embeds
#         ```"""
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         text_outputs = self.text_model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         pooled_output = text_outputs[1]

#         text_embeds = self.text_projection(pooled_output)

#         if not return_dict:
#             outputs = (text_embeds, text_outputs[0]) + text_outputs[2:]
#             return tuple(output for output in outputs if output is not None)

#         return CLIPTextModelOutput(
#             text_embeds=text_embeds,
#             last_hidden_state=text_outputs.last_hidden_state,
#             hidden_states=text_outputs.hidden_states,
#             attentions=text_outputs.attentions,
#         )


# @add_start_docstrings(
#     """
#     CLIP Vision Model with a projection layer on top (a linear layer on top of the pooled output).
#     """,
#     CLIP_START_DOCSTRING,
# )
# class CLIPVisionModelWithProjection(CLIPPreTrainedModel):
#     config_class = CLIPVisionConfig
#     main_input_name = "pixel_values"

#     def __init__(self, config: CLIPVisionConfig):
#         super().__init__(config)

#         vision_model = CLIPVisionModel._from_config(config, attn_implementation=config._attn_implementation)
#         self.vision_model = vision_model.vision_model

#         self.visual_projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_input_embeddings(self) -> nn.Module:
#         return self.vision_model.embeddings.patch_embedding

#     @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
#     @replace_return_docstrings(output_type=CLIPVisionModelOutput, config_class=CLIPVisionConfig)
#     def forward(
#         self,
#         pixel_values: Optional[torch.FloatTensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, CLIPVisionModelOutput]:
#         r"""
#         Returns:

#         Examples:

#         ```python
#         >>> from PIL import Image
#         >>> import requests
#         >>> from transformers import AutoProcessor, CLIPVisionModelWithProjection

#         >>> model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
#         >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

#         >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#         >>> image = Image.open(requests.get(url, stream=True).raw)

#         >>> inputs = processor(images=image, return_tensors="pt")

#         >>> outputs = model(**inputs)
#         >>> image_embeds = outputs.image_embeds
#         ```"""
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         vision_outputs = self.vision_model(
#             pixel_values=pixel_values,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         pooled_output = vision_outputs[1]  # pooled_output

#         image_embeds = self.visual_projection(pooled_output)

#         if not return_dict:
#             outputs = (image_embeds, vision_outputs[0]) + vision_outputs[2:]
#             return tuple(output for output in outputs if output is not None)

#         return CLIPVisionModelOutput(
#             image_embeds=image_embeds,
#             last_hidden_state=vision_outputs.last_hidden_state,
#             hidden_states=vision_outputs.hidden_states,
#             attentions=vision_outputs.attentions,
#         )


# @add_start_docstrings(
#     """
#     CLIP vision encoder with an image classification head on top (a linear layer on top of the pooled final hidden states of
#     the patch tokens) e.g. for ImageNet.
#     """,
#     CLIP_START_DOCSTRING,
# )
# class CLIPForImageClassification(CLIPPreTrainedModel):
#     main_input_name = "pixel_values"

#     def __init__(self, config: CLIPConfig) -> None:
#         super().__init__(config)

#         self.num_labels = config.num_labels
#         vision_model = CLIPVisionModel._from_config(
#             config.vision_config, attn_implementation=config._attn_implementation
#         )
#         self.vision_model = vision_model.vision_model

#         # Classifier head
#         self.classifier = (
#             nn.Linear(config.vision_config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
#         )

#         # Initialize weights and apply final processing
#         self.post_init()

#     @add_start_docstrings_to_model_forward(CLIP_INPUTS_DOCSTRING)
#     @add_code_sample_docstrings(
#         checkpoint=_IMAGE_CLASS_CHECKPOINT,
#         output_type=ImageClassifierOutput,
#         config_class=_CONFIG_FOR_DOC,
#         expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
#     )
#     def forward(
#         self,
#         pixel_values: Optional[torch.Tensor] = None,
#         labels: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[tuple, ImageClassifierOutput]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
#             config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
#             `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         outputs = self.vision_model(
#             pixel_values,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         sequence_output = outputs[0]

#         # average pool the patch tokens
#         sequence_output = torch.mean(sequence_output[:, 1:, :], dim=1)
#         # apply classifier
#         logits = self.classifier(sequence_output)

#         loss = None
#         if labels is not None:
#             # move labels to correct device to enable model parallelism
#             labels = labels.to(logits.device)
#             if self.config.problem_type is None:
#                 if self.num_labels == 1:
#                     self.config.problem_type = "regression"
#                 elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
#                     self.config.problem_type = "single_label_classification"
#                 else:
#                     self.config.problem_type = "multi_label_classification"

#             if self.config.problem_type == "regression":
#                 loss_fct = MSELoss()
#                 if self.num_labels == 1:
#                     loss = loss_fct(logits.squeeze(), labels.squeeze())
#                 else:
#                     loss = loss_fct(logits, labels)
#             elif self.config.problem_type == "single_label_classification":
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             elif self.config.problem_type == "multi_label_classification":
#                 loss_fct = BCEWithLogitsLoss()
#                 loss = loss_fct(logits, labels)

#         if not return_dict:
#             output = (logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output

#         return ImageClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )