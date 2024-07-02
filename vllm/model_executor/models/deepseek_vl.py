# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted,free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute,sublicense,and/or sell copies of
# the Software,and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED,INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import collections.abc
import copy
import math
import warnings
from dataclasses import dataclass
from enum import Enum
from functools import partial
from itertools import chain, repeat
from typing import (Callable, Dict, Final, List, Literal, Optional, Sequence,
                    Set, Tuple, Type, Union)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms
import torchvision.transforms.functional
from PIL import Image
from torch import _assert
from torch.utils.checkpoint import checkpoint
from transformers import AutoImageProcessor, PretrainedConfig, PreTrainedModel
from transformers.image_processing_utils import (BaseImageProcessor,
                                                 BatchFeature)
from transformers.image_utils import to_numpy_array

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, ModelConfig, VisionLanguageConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalData
from vllm.multimodal.base import VisionLanguageModelBase
from vllm.multimodal.image import ImageFeatureData, ImagePixelData
from vllm.sequence import SamplerOutput, SequenceData
from vllm.transformers_utils.configs import DeepSeekMultiModalityConfig

ImageType = Union[np.ndarray, torch.Tensor, Image.Image]
IMAGENET_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
LayerType = Union[str, Callable, Type[torch.nn.Module]]


def _get_dummy_seq_data(seq_len: int,
                        vlm_config: VisionLanguageConfig) -> SequenceData:
    # NOTE: We assume that <image> token is repeated `image_feature_size` times
    # and then concatenated with the text prompt
    # TODO: Enable other ways of inserting the image into the prompt

    token_ids = [vlm_config.image_token_id] * vlm_config.image_feature_size
    token_ids += [0] * (seq_len - vlm_config.image_feature_size)

    return SequenceData(token_ids)


def _get_dummy_values(vlm_config: VisionLanguageConfig) -> torch.Tensor:
    if vlm_config.image_processor is None:
        values_dtype = torch.float16
    else:
        values_dtype = torch.uint8

    return torch.zeros(vlm_config.image_input_shape, dtype=values_dtype)


def get_dummy_image_data(
    seq_len: int,
    model_config: ModelConfig,
    vlm_config: VisionLanguageConfig,
) -> Tuple[SequenceData, MultiModalData]:
    """Standard dummy data factory for image data (to be used in
    :meth:`vlm.multimodal.MultiModalRegistry.register_dummy_data`)."""
    seq_data = _get_dummy_seq_data(seq_len, vlm_config)
    values = _get_dummy_values(vlm_config)

    config_input_type = vlm_config.image_input_type
    ImageInputType = VisionLanguageConfig.ImageInputType

    fake_mm_data: MultiModalData
    if config_input_type == ImageInputType.PIXEL_VALUES:
        fake_mm_data = ImagePixelData(values)
    elif config_input_type == ImageInputType.IMAGE_FEATURES:
        fake_mm_data = ImageFeatureData(values)
    else:
        raise NotImplementedError

    return seq_data, fake_mm_data


# From PyTorch internals
def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class Format(str, Enum):
    NCHW = "NCHW"
    NHWC = "NHWC"
    NCL = "NCL"
    NLC = "NLC"


def nchw_to(x: torch.Tensor, fmt: Format):
    if fmt == Format.NHWC:
        x = x.permute(0, 2, 3, 1)
    elif fmt == Format.NLC:
        x = x.flatten(2).transpose(1, 2)
    elif fmt == Format.NCL:
        x = x.flatten(2)
    return x


# From https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/attention_pool.py # noqa
class AttentionPoolLatent(nn.Module):
    """Attention pooling w/ latent query"""

    fused_attn: torch.jit.Final[bool]

    def __init__(
        self,
        in_features: int,
        out_features: int = None,
        embed_dim: int = None,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        latent_len: int = 1,
        latent_dim: int = None,
        pos_embed: str = "",
        pool_type: str = "token",
        norm_layer: Optional[nn.Module] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        embed_dim = embed_dim or in_features
        out_features = out_features or in_features
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.pool = pool_type
        self.fused_attn = True

        if pos_embed == "abs":
            spatial_len = self.feat_size
            self.pos_embed = nn.Parameter(torch.zeros(spatial_len,
                                                      in_features))
        else:
            self.pos_embed = None

        self.latent_dim = latent_dim or embed_dim
        self.latent_len = latent_len
        self.latent = nn.Parameter(torch.zeros(1, self.latent_len, embed_dim))

        self.q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(drop)

        self.norm = (norm_layer(out_features)
                     if norm_layer is not None else nn.Identity())
        self.mlp = Mlp(embed_dim, int(embed_dim * mlp_ratio))

    def forward(self, x):
        B, N, C = x.shape

        if self.pos_embed is not None:
            # FIXME interpolate
            x = x + self.pos_embed.unsqueeze(0).to(x.dtype)

        q_latent = self.latent.expand(B, -1, -1)
        q = (self.q(q_latent).reshape(B, self.latent_len, self.num_heads,
                                      self.head_dim).transpose(1, 2))

        kv = (self.kv(x).reshape(B, N, 2, self.num_heads,
                                 self.head_dim).permute(2, 0, 3, 1, 4))
        k, v = kv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, self.latent_len, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x + self.mlp(self.norm(x))

        # optional pool if latent seq_len > 1 and pooled output is desired
        if self.pool == "token":
            x = x[:, 0]
        elif self.pool == "avg":
            x = x.mean(1)
        return x


# From https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py # noqa
def drop_path(
    x,
    drop_prob: float = 0.0,
    training: bool = False,
    scale_by_keep: bool = True,
):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (
        x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


# From https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py # noqa
class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  
    (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


# From https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/mlp.py # noqa
class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = (partial(nn.Conv2d, kernel_size=1)
                        if use_conv else nn.Linear)

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (norm_layer(hidden_features)
                     if norm_layer is not None else nn.Identity())
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# From https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_dropout.py # noqa
class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    return_indices: torch.jit.Final[bool]

    def __init__(
        self,
        prob: float = 0.5,
        num_prefix_tokens: int = 1,
        ordered: bool = False,
        return_indices: bool = False,
    ):
        super().__init__()
        assert 0 <= prob < 1.0
        self.prob = prob
        self.num_prefix_tokens = (
            num_prefix_tokens  # exclude CLS token (or other prefix tokens)
        )
        self.ordered = ordered
        self.return_indices = return_indices

    def forward(
            self, x
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        if not self.training or self.prob == 0.0:
            if self.return_indices:
                return x, None
            return x

        if self.num_prefix_tokens:
            prefix_tokens, x = (
                x[:, :self.num_prefix_tokens],
                x[:, self.num_prefix_tokens:],
            )
        else:
            prefix_tokens = None

        B = x.shape[0]
        L = x.shape[1]
        num_keep = max(1, int(L * (1.0 - self.prob)))
        keep_indices = torch.argsort(torch.randn(B, L, device=x.device),
                                     dim=-1)[:, :num_keep]
        if self.ordered:
            # NOTE does not need to maintain patch order in typical
            # transformer use,
            # but possibly useful for debug / visualization
            keep_indices = keep_indices.sort(dim=-1)[0]
        x = x.gather(1,
                     keep_indices.unsqueeze(-1).expand((-1, -1) + x.shape[2:]))

        if prefix_tokens is not None:
            x = torch.cat((prefix_tokens, x), dim=1)

        if self.return_indices:
            return x, keep_indices
        return x


# From https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py # noqa
class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    output_fmt: Format
    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(
        self,
        img_size: Optional[int] = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten: bool = True,
        output_fmt: Optional[str] = None,
        bias: bool = True,
        strict_img_size: bool = True,
        dynamic_img_pad: bool = False,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        if img_size is not None:
            self.img_size = to_2tuple(img_size)
            self.grid_size = tuple(
                [s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last,
            # kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def feat_ratio(self, as_scalar=True) -> Union[Tuple[int, int], int]:
        if as_scalar:
            return max(self.patch_size)
        else:
            return self.patch_size

    def dynamic_feat_size(self, img_size: Tuple[int, int]) -> Tuple[int, int]:
        """Get grid (feature) size for given image size taking account 
           of dynamic padding.
        NOTE: must be torchscript compatible so using fixed tuple indexing
        """
        if self.dynamic_img_pad:
            return math.ceil(img_size[0] / self.patch_size[0]), math.ceil(
                img_size[1] / self.patch_size[1])
        else:
            return (
                img_size[0] // self.patch_size[0],
                img_size[1] // self.patch_size[1],
            )

    def forward(self, x):
        B, C, H, W = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                _assert(
                    self.img_size[0] == H,
                    f"Input height ({H}) doesn't match model ({self.img_size[0]}).",  # noqa
                )
                _assert(
                    self.img_size[1] == W,
                    f"Input width ({W}) doesn't match model ({self.img_size[1]}).",  # noqa
                )
            elif not self.dynamic_img_pad:
                _assert(
                    H % self.patch_size[0] == 0,
                    f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]}).",  # noqa
                )
                _assert(
                    W % self.patch_size[1] == 0,
                    f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]}).",  # noqa
                )
        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] -
                     H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] -
                     W % self.patch_size[1]) % self.patch_size[1]
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
        return x


# From https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/pos_embed.py # noqa
def resample_abs_pos_embed(
    posemb,
    new_size: List[int],
    old_size: Optional[List[int]] = None,
    num_prefix_tokens: int = 1,
    interpolation: str = "bicubic",
    antialias: bool = True,
    verbose: bool = False,
):
    # sort out sizes, assume square if old size not provided
    num_pos_tokens = posemb.shape[1]
    num_new_tokens = new_size[0] * new_size[1] + num_prefix_tokens
    if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
        return posemb

    if old_size is None:
        hw = int(math.sqrt(num_pos_tokens - num_prefix_tokens))
        old_size = hw, hw

    if num_prefix_tokens:
        posemb_prefix, posemb = (
            posemb[:, :num_prefix_tokens],
            posemb[:, num_prefix_tokens:],
        )
    else:
        posemb_prefix, posemb = None, posemb

    # do the interpolation
    embed_dim = posemb.shape[-1]
    orig_dtype = posemb.dtype
    posemb = posemb.float()  # interpolate needs float32
    posemb = posemb.reshape(1, old_size[0], old_size[1],
                            -1).permute(0, 3, 1, 2)
    posemb = F.interpolate(posemb,
                           size=new_size,
                           mode=interpolation,
                           antialias=antialias)
    posemb = posemb.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
    posemb = posemb.to(orig_dtype)

    # add back extra (class, etc) prefix tokens
    if posemb_prefix is not None:
        posemb = torch.cat([posemb_prefix, posemb], dim=1)

    if not torch.jit.is_scripting() and verbose:
        print(f"Resized position embedding: {old_size} to {new_size}.")

    return posemb


# From https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/_manipulate.py # noqa
def checkpoint_seq(
    functions,
    x,
    every=1,
    flatten=False,
    skip_last=False,
    preserve_rng_state=True,
):

    def run_function(start, end, functions):

        def forward(_x):
            for j in range(start, end + 1):
                _x = functions[j](_x)
            return _x

        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = functions.children()
    if flatten:
        functions = chain.from_iterable(functions)
    if not isinstance(functions, (tuple, list)):
        functions = tuple(functions)

    num_checkpointed = len(functions)
    if skip_last:
        num_checkpointed -= 1
    end = -1
    for start in range(0, num_checkpointed, every):
        end = min(start + every - 1, num_checkpointed - 1)
        x = checkpoint(
            run_function(start, end, functions),
            x,
            preserve_rng_state=preserve_rng_state,
        )
    if skip_last:
        return run_function(end + 1, len(functions) - 1, functions)(x)
    return x


class AttrDict:

    def __init__(self, entries):
        for key, value in entries.items():
            if isinstance(value, dict):
                entries[key] = AttrDict(value)
        self.__dict__.update(entries)

    def get(self, key, default_val=None):
        return self.__dict__.get(key, default_val)


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class VLMImageProcessorConfig(PretrainedConfig):
    model_type = "deepseek_vlm"
    image_size: int
    min_size: int
    image_mean: Union[Tuple[float, float, float], List[float]]
    image_std: Union[Tuple[float, float, float], List[float]]
    rescale_factor: float
    do_normalize: bool

    def __init__(
        self,
        image_size: int,
        min_size: int = 14,
        image_mean: Union[Tuple[float, float, float], List[float]] = (
            0.48145466,
            0.4578275,
            0.40821073,
        ),
        image_std: Union[Tuple[float, float, float], List[float]] = (
            0.26862954,
            0.26130258,
            0.27577711,
        ),
        rescale_factor: float = 1.0 / 255.0,
        do_normalize: bool = True,
        **kwargs,
    ):
        self.image_size = image_size
        self.min_size = min_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize

        super().__init__(**kwargs)


class VLMImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        image_size: int,
        min_size: int = 14,
        image_mean: Union[Tuple[float, float, float], List[float]] = (
            0.48145466,
            0.4578275,
            0.40821073,
        ),
        image_std: Union[Tuple[float, float, float], List[float]] = (
            0.26862954,
            0.26130258,
            0.27577711,
        ),
        rescale_factor: float = 1.0 / 255.0,
        do_normalize: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.rescale_factor = rescale_factor
        self.image_mean = image_mean
        self.image_std = image_std
        self.min_size = min_size
        self.do_normalize = do_normalize

        if image_mean is None:
            self.background_color = (127, 127, 127)
        else:
            self.background_color = tuple([int(x * 255) for x in image_mean])

    def resize(self, pil_img: Image) -> np.ndarray:
        """

        Args:
            pil_img (PIL.Image): [H, W, 3] in PIL.Image in RGB

        Returns:
            x (np.ndarray): [3, self.image_size, self.image_size]
        """

        width, height = pil_img.size
        max_size = max(width, height)

        size = [
            max(int(height / max_size * self.image_size), self.min_size),
            max(int(width / max_size * self.image_size), self.min_size),
        ]

        if width <= 0 or height <= 0 or size[0] <= 0 or size[1] <= 0:
            print(f"orig size = {pil_img.size}, new size = {size}")
            raise ValueError("Invalid size!")

        pil_img = torchvision.transforms.functional.resize(
            pil_img,
            size,
            interpolation=torchvision.transforms.functional.InterpolationMode.
            BICUBIC,
            antialias=True,
        )

        pil_img = expand2square(pil_img, self.background_color)
        x = to_numpy_array(pil_img)

        # [H, W, 3] -> [3, H, W]
        x = np.transpose(x, (2, 0, 1))

        return x

    def preprocess(self,
                   images,
                   return_tensors: str = "pt",
                   **kwargs) -> BatchFeature:
        # resize and pad to [self.image_size, self.image_size]
        # then convert from [H, W, 3] to [3, H, W]
        if not isinstance(images, List):
            images = [
                images,
            ]
        images: List[np.ndarray] = [self.resize(image) for image in images]

        # resacle from [0, 255] -> [0, 1]
        images = [
            self.rescale(
                image=image,
                scale=self.rescale_factor,
                input_data_format="channels_first",
            ) for image in images
        ]

        # normalize
        if self.do_normalize:
            images = [
                self.normalize(
                    image=image,
                    mean=self.image_mean,
                    std=self.image_std,
                    input_data_format="channels_first",
                ) for image in images
            ]

        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)

    @property
    def default_shape(self):
        return [3, self.image_size, self.image_size]


class MlpProjector(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        cfg = AttrDict(cfg)
        self.cfg = cfg

        if cfg.projector_type == "identity":
            modules = nn.Identity()

        elif cfg.projector_type == "linear":
            modules = nn.Linear(cfg.input_dim, cfg.n_embed)

        elif cfg.projector_type == "mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            modules = [nn.Linear(cfg.input_dim, cfg.n_embed)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed, cfg.n_embed))
            modules = nn.Sequential(*modules)

        elif cfg.projector_type == "low_high_hybrid_split_mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            self.high_up_proj = nn.Linear(cfg.input_dim, cfg.n_embed // 2)
            self.low_up_proj = nn.Linear(cfg.input_dim, cfg.n_embed // 2)

            modules = []
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed, cfg.n_embed))
            modules = nn.Sequential(*modules)

        else:
            raise ValueError(f"Unknown projector type: {cfg.projector_type}")

        self.layers = modules

    def forward(
        self,
        x_or_tuple: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
    ):
        """

        Args:
            x_or_tuple (Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
            if it is a tuple of torch.Tensor,
            then it comes from the hybrid vision encoder, 
            and x = high_res_x, low_res_x);
            otherwise it is the feature from the single vision encoder.

        Returns:
            x (torch.Tensor): [b, s, c]
        """

        if isinstance(x_or_tuple, tuple):
            # self.cfg.projector_type == "low_high_hybrid_split_mlp_gelu":
            high_x, low_x = x_or_tuple
            high_x = self.high_up_proj(high_x)
            low_x = self.low_up_proj(low_x)
            x = torch.concat([high_x, low_x], dim=-1)
        else:
            x = x_or_tuple

        return self.layers(x)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official
    # releases - RW Method based on
    # https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)  # noqa: E741
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    with torch.no_grad():
        dtype = tensor.dtype
        tensor_fp32 = tensor.float()
        tensor_fp32 = _no_grad_trunc_normal_(tensor_fp32, mean, std, a, b)
        tensor_dtype = tensor_fp32.to(dtype=dtype)
        tensor.copy_(tensor_dtype)


def init_weights(self):
    if self.pos_embed is not None:
        trunc_normal_(self.pos_embed, std=self.pos_embed.shape[1]**-0.5)
    trunc_normal_(self.latent, std=self.latent_dim**-0.5)


class SigLipAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = (nn.Dropout(proj_drop)
                          if proj_drop > 0.0 else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (self.qkv(x).reshape(B, N, 3, self.num_heads,
                                   self.head_dim).permute(2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):

    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class SigLipBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SigLipAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = (LayerScale(dim, init_values=init_values)
                    if init_values else nn.Identity())
        self.drop_path1 = (DropPath(drop_path)
                           if drop_path > 0.0 else nn.Identity())

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = (LayerScale(dim, init_values=init_values)
                    if init_values else nn.Identity())
        self.drop_path2 = (DropPath(drop_path)
                           if drop_path > 0.0 else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: 
    Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    dynamic_img_size: Final[bool]

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: Literal["", "avg", "token", "map"] = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: Optional[float] = None,
        class_token: bool = True,
        no_embed_class: bool = False,
        reg_tokens: int = 0,
        pre_norm: bool = False,
        fc_norm: Optional[bool] = None,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        embed_layer: Callable = PatchEmbed,
        norm_layer: Optional[LayerType] = None,
        act_layer: Optional[LayerType] = None,
        block_fn: Type[nn.Module] = SigLipBlock,
        mlp_layer: Type[nn.Module] = Mlp,
        ignore_head: bool = False,
    ) -> None:
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Number of classes for classification head.
            global_pool: Type of global pooling for final sequence 
            (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values 
            (layer-scale enabled if not None).
            class_token: Use class token.
            no_embed_class: Don't include position embeddings for class 
            (or reg) tokens.
            reg_tokens: Number of register tokens.
            fc_norm: Pre head norm after pool (instead of before), if None, 
            enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super().__init__()
        assert global_pool in ("", "avg", "token", "map")
        assert class_token or global_pool != "token"
        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = (
            no_embed_class  # don't embed prefix positions (includes reg)
        )
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False
        self.ignore_head = ignore_head

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt="NHWC"))
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = (nn.Parameter(torch.zeros(1, 1, embed_dim))
                          if class_token else None)
        self.reg_token = (nn.Parameter(torch.zeros(1, reg_tokens, embed_dim))
                          if reg_tokens else None)
        embed_len = (num_patches if no_embed_class else num_patches +
                     self.num_prefix_tokens)
        self.pos_embed = nn.Parameter(
            torch.randn(1, embed_len, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
            ) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        if global_pool == "map":
            AttentionPoolLatent.init_weights = init_weights
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
            )
        else:
            self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = (nn.Linear(self.embed_dim, num_classes)
                     if num_classes > 0 else nn.Identity())

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict:
        return dict(
            stem=r"^cls_token|pos_embed|patch_embed",  # stem and embed
            blocks=[(r"^blocks\.(\d+)", None), (r"^norm", (99999, ))],
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None) -> None:
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ("", "avg", "token", "map")
            if global_pool == "map" and self.attn_pool is None:
                raise AssertionError(
                    "Cannot currently add attention pooling in reset_classifier()."  # noqa
                )
            elif global_pool != "map " and self.attn_pool is not None:
                self.attn_pool = None  # remove attention pooling
            self.global_pool = global_pool
        self.head = (nn.Linear(self.embed_dim, num_classes)
                     if num_classes > 0 else nn.Identity())

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.dynamic_img_size:
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                (H, W),
                num_prefix_tokens=(0 if self.no_embed_class else
                                   self.num_prefix_tokens),
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token,
            # add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.pos_drop(x)

    def _intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,
    ) -> List[torch.Tensor]:
        outputs, num_blocks = [], len(self.blocks)
        take_indices = set(
            range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

        # forward pass
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in take_indices:
                outputs.append(x)

        return outputs

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_prefix_tokens: bool = False,
        norm: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """Intermediate layer accessor (NOTE: This is a WIP experiment).
        Inspired by DINO / DINOv2 interface
        """
        # take last n blocks if n is an int, if in is a sequence,
        # select by matching indices
        outputs = self._intermediate_layers(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        prefix_tokens = [out[:, 0:self.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.num_prefix_tokens:] for out in outputs]

        if reshape:
            grid_size = self.patch_embed.grid_size
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1],
                            -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if return_prefix_tokens:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self,
                     x: torch.Tensor,
                     pre_logits: bool = False) -> torch.Tensor:
        if self.attn_pool is not None:
            x = self.attn_pool(x)
        elif self.global_pool == "avg":
            x = x[:, self.num_prefix_tokens:].mean(dim=1)
        elif self.global_pool:
            x = x[:, 0]  # class token
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        if not self.ignore_head:
            x = self.forward_head(x)
        return x


@dataclass
class SigLIPVisionCfg:
    width: int = 1152
    layers: Union[Tuple[int, int, int, int], int] = 27
    heads: int = 16
    patch_size: int = 14
    image_size: Union[Tuple[int, int], int] = 336
    global_pool: str = "map"
    mlp_ratio: float = 3.7362
    class_token: bool = False
    num_classes: int = 0
    use_checkpoint: bool = False


SigLIP_MODEL_CONFIG = {
    "siglip_so400m_patch14_384": {
        "image_size": 336,
        "patch_size": 14,
        "width": 1152,
        "layers": 27,
        "heads": 16,
        "mlp_ratio": 3.7362,
        "global_pool": "map",
        "use_checkpoint": False,
    },
    "siglip_so400m_patch14_224": {
        "image_size": 224,
        "patch_size": 14,
        "width": 1152,
        "layers": 27,
        "heads": 16,
        "mlp_ratio": 3.7362,
        "global_pool": "map",
        "use_checkpoint": False,
    },
    "siglip_large_patch16_384": {
        "image_size": 384,
        "patch_size": 16,
        "width": 1024,
        "layers": 24,
        "heads": 16,
        "mlp_ratio": 4,
        "global_pool": "map",
        "use_checkpoint": False,
    },
}


def create_siglip_vit(
    model_name: str = "siglip_so400m_patch14_384",
    image_size: int = 384,
    select_layer: int = -1,
    ckpt_path: str = "",
    **kwargs,
):
    assert (model_name in SigLIP_MODEL_CONFIG
            ), f"model name should be in {SigLIP_MODEL_CONFIG.keys()}"

    vision_cfg = SigLIPVisionCfg(**SigLIP_MODEL_CONFIG[model_name])

    if select_layer <= 0:
        layers = min(vision_cfg.layers, vision_cfg.layers + select_layer + 1)
    else:
        layers = min(vision_cfg.layers, select_layer)

    model = VisionTransformer(
        img_size=image_size,
        patch_size=vision_cfg.patch_size,
        embed_dim=vision_cfg.width,
        depth=layers,
        num_heads=vision_cfg.heads,
        mlp_ratio=vision_cfg.mlp_ratio,
        class_token=vision_cfg.class_token,
        global_pool=vision_cfg.global_pool,
        ignore_head=kwargs.get("ignore_head", True),
        num_classes=0,
    )

    if ckpt_path:
        state_dict = torch.load(ckpt_path, map_location="cpu")

        incompatible_keys = model.load_state_dict(state_dict, strict=False)
        print(f"SigLIP-ViT restores from {ckpt_path},\n"
              f"\tincompatible_keys:', {incompatible_keys}.")

    return model


class MLPBlock(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT(nn.Module):

    def __init__(
            self,
            img_size: int = 1024,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            out_chans: int = 256,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_abs_pos: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            global_attn_indexes: Tuple[int, ...] = (),
            downsample_channels: Tuple[int, ...] = (512, 1024),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to 
            the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative 
            positional parameters. window_size (int): Window size for window 
            attention blocks. global_attn_indexes (list): Indexes for blocks 
            using global attention.
            downsample_channels (list): Channels for downsampling layers.
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = ImagePatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    img_size // patch_size,
                    img_size // patch_size,
                    embed_dim,
                ))

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

        in_channels = out_chans
        downsamples = []
        for i in range(len(downsample_channels)):
            out_channels = downsample_channels[i]
            downsamples.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ))
            in_channels = out_channels
        self.downsamples = nn.Sequential(*downsamples)

        self.sam_hd = True
        if self.sam_hd:
            self.hd_alpha_downsamples = nn.Parameter(torch.zeros(1))
            # self.neck_hd = nn.Linear(embed_dim, embed_dim)
            self.neck_hd = copy.deepcopy(self.neck)
            # self.downsamples_hd = copy.deepcopy(self.downsamples)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        global_features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.sam_hd and blk.window_size == 0:
                global_features.append(x)

        x = self.neck(x.permute(0, 3, 1, 2))
        x_dtype = x.dtype
        x = F.interpolate(x.float(),
                          size=(96, 96),
                          mode="bilinear",
                          align_corners=False).to(x_dtype)
        x = self.downsamples(x)

        if self.sam_hd:
            first_global_feature = self.neck_hd(global_features[0].permute(
                0, 3, 1, 2))
            x_dtype = first_global_feature.dtype
            first_global_feature = F.interpolate(
                first_global_feature.float(),
                size=(96, 96),
                mode="bilinear",
                align_corners=False,
            )
            first_global_feature = self.downsamples(
                first_global_feature.to(x_dtype))
            x = x + first_global_feature * self.hd_alpha_downsamples

        return x


class Block(nn.Module):
    """
    Transformer blocks with support of window attention and 
    residual propagation blocks
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to 
            query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to 
            the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative 
            positional parameters.
            window_size (int): Window size for window attention blocks. If 
            it equals 0, then use global attention. input_size 
            (tuple(int, int) or None): Input resolution for calculating 
            the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=(input_size if window_size == 0 else
                        (window_size, window_size)),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim,
                            mlp_dim=int(dim * mlp_ratio),
                            act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to 
            query, key, value.
            rel_pos (bool): If True, add relative positional embeddings 
            to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative 
            positional parameters.
            input_size (tuple(int, int) or None): Input resolution for 
            calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."  # noqa
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(
                torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(
                torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = (self.qkv(x).reshape(B, H * W, 3, self.num_heads,
                                   -1).permute(2, 0, 3, 1, 4))
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        def do_attention(q, k, v):
            attn = (q * self.scale) @ k.transpose(-2, -1)
            if self.use_rel_pos:
                attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h,
                                              self.rel_pos_w, (H, W), (H, W))

            attn = attn.softmax(dim=-1)
            x = ((attn @ v).view(B, self.num_heads, H, W,
                                 -1).permute(0, 2, 3, 1,
                                             4).reshape(B, H, W, -1))

            return x

        # from haiscale.utils import on_demand_checkpoint
        # x = on_demand_checkpoint(do_attention, q, k, v)
        x = do_attention(q, k, v)
        x = self.proj(x)

        return x


def window_partition(x: torch.Tensor,
                     window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size,
          window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size,
               window_size, C)
    windows = (x.permute(0, 1, 3, 2, 4,
                         5).contiguous().view(-1, window_size, window_size, C))
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    pad_hw: Tuple[int, int],
    hw: Tuple[int, int],
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with 
        [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size,
                     window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int,
                rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1,
                                                  max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords -
                       k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] +
            rel_w[:, :, :, None, :]).view(B, q_h * q_w, k_h * k_w)

    return attn


class ImagePatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


@dataclass
class SAMViTCfg:
    image_size: Union[Tuple[int, int], int] = 1024
    width: int = 1024
    layers: int = 23
    heads: int = 16
    patch_size: int = 16
    window_size: int = 14
    prompt_embed_dim: int = 256
    global_attn_indexes: Union[List[int], Tuple[int]] = (5, 11, 17, 23)
    downsample_channels: Union[List[int], Tuple[int]] = (512, 1024)


SAM_MODEL_CONFIG = {
    "sam_vit_b": {
        "width": 768,
        "layers": 12,
        "heads": 12,
        "global_attn_indexes": [2, 5, 8, 11],
        "downsample_channels": (),
    },
    "sam_b_downsample": {
        "width": 768,
        "layers": 12,
        "heads": 12,
        "global_attn_indexes": [2, 5, 8, 11],
        "downsample_channels": (512, 1024),
    },
    "sam_vit_l": {
        "width": 1024,
        "layers": 24,
        "heads": 16,
        "global_attn_indexes": [5, 11, 17, 23],
        "downsample_channels": (),
    },
    "sam_vit_h": {
        "width": 1280,
        "layers": 32,
        "heads": 16,
        "global_attn_indexes": [7, 15, 23, 31],
        "downsample_channels": (),
    },
}


def create_sam_vit(
    model_name: str = "sam_b_downsample",
    image_size: int = 1024,
    ckpt_path: str = "",
    **kwargs,
):
    assert (
        model_name in SAM_MODEL_CONFIG
    ), f"model name: {model_name} should be in {SAM_MODEL_CONFIG.keys()}"

    sam_cfg = SAMViTCfg(**SAM_MODEL_CONFIG[model_name])
    image_encoder = ImageEncoderViT(
        depth=sam_cfg.layers,
        embed_dim=sam_cfg.width,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=sam_cfg.heads,
        patch_size=sam_cfg.patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=sam_cfg.global_attn_indexes,
        window_size=14,
        out_chans=sam_cfg.prompt_embed_dim,
        downsample_channels=sam_cfg.downsample_channels,
    )

    if ckpt_path:
        state_dict = torch.load(ckpt_path)
        image_encoder.load_state_dict(state_dict, strict=False)
        print(f"SAM-ViT restores from {ckpt_path}")

    return image_encoder


class CLIPVisionTower(nn.Module):

    def __init__(
        self,
        model_name: str = "siglip_large_patch16_384",
        image_size: Union[Tuple[int, int], int] = 336,
        select_feature: str = "patch",
        select_layer: int = -2,
        select_layers: list = None,
        ckpt_path: str = "",
        pixel_mean: Optional[List[float]] = None,
        pixel_std: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__()

        self.model_name = model_name
        self.select_feature = select_feature
        self.select_layer = select_layer
        self.select_layers = select_layers

        vision_tower_params = {
            "model_name": model_name,
            "image_size": image_size,
            "ckpt_path": ckpt_path,
            "select_layer": select_layer,
        }
        vision_tower_params.update(kwargs)
        self.vision_tower, self.forward_kwargs = self.build_vision_tower(
            vision_tower_params)

        if pixel_mean is not None and pixel_std is not None:
            image_norm = torchvision.transforms.Normalize(mean=pixel_mean,
                                                          std=pixel_std)
        else:
            image_norm = None

        self.image_norm = image_norm

    def build_vision_tower(self, vision_tower_params):
        if self.model_name.startswith("siglip"):
            self.select_feature = "same"
            vision_tower = create_siglip_vit(**vision_tower_params)
            forward_kwargs = dict()

        elif self.model_name.startswith("sam"):
            vision_tower = create_sam_vit(**vision_tower_params)
            forward_kwargs = dict()

        else:
            from vllm.model_executor.models.clip import CLIPVisionModel

            vision_tower = CLIPVisionModel.from_pretrained(
                **vision_tower_params)
            forward_kwargs = dict(output_hidden_states=True)

        return vision_tower, forward_kwargs

    def feature_select(self, image_forward_outs):
        if isinstance(image_forward_outs, torch.Tensor):
            # the output has been the self.select_layer"s features
            image_features = image_forward_outs
        else:
            image_features = image_forward_outs.hidden_states[
                self.select_layer]

        if self.select_feature == "patch":
            # if the output has cls_token
            image_features = image_features[:, 1:]
        elif (self.select_feature == "cls_patch"
              or self.select_feature == "same"):
            image_features = image_features
        else:
            raise ValueError(
                f"Unexpected select feature: {self.select_feature}")
        return image_features

    def forward(self, images):
        """

        Args:
            images (torch.Tensor): [b, 3, H, W]

        Returns:
            image_features (torch.Tensor): [b, n_patch, d]
        """

        if self.image_norm is not None:
            images = self.image_norm(images)

        image_forward_outs = self.vision_tower(images, **self.forward_kwargs)
        image_features = self.feature_select(image_forward_outs)
        return image_features


class HybridVisionTower(nn.Module):

    def __init__(
        self,
        high_res_cfg: Dict,
        low_res_cfg: Dict,
        freeze_high: bool = False,
        freeze_low: bool = False,
        concat_type: Literal["feature", "sequence", "add", "tuple"] = "tuple",
        **ignore_kwargs,
    ):
        super().__init__()

        self.vision_tower_high = CLIPVisionTower(**high_res_cfg)
        self.vision_tower_low = CLIPVisionTower(**low_res_cfg)
        self.low_res_size = low_res_cfg["image_size"]
        self.concat_type = concat_type

        self.high_layer_norm = nn.LayerNorm(
            high_res_cfg.get("output_dim", 1024))
        self.low_layer_norm = nn.LayerNorm(low_res_cfg.get("output_dim", 1024))

        if freeze_high:
            for p_name, p in self.vision_tower_high.named_parameters():
                p.requires_grad = False
            self.vision_tower_high = self.vision_tower_high.eval()
        else:
            # train donwsamples and neck
            for p_name, p in self.vision_tower_high.named_parameters():
                if "downsamples" in p_name or "neck" in p_name:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

        if freeze_low:
            for p in self.vision_tower_low.parameters():
                p.requires_grad = False
            self.vision_tower_low = self.vision_tower_low.eval()

        self.resize = torchvision.transforms.Resize(self.low_res_size,
                                                    antialias=True)

    def forward(self, images: torch.Tensor):
        """

        Args:
            images (torch.Tensor): [bs, 3, H, W]

        Returns:
            res (torch.Tensor): [bs, t, c]
        """

        # [bs, c, h, w]
        high_images = images

        # [bs, c, h_low, w_low]
        low_images = self.resize(images)

        # separately run two vision towers
        # run high_res vision tower
        high_res = self.vision_tower_high(high_images)
        # [bs, c, h, w] -> [bs, h*w, c]
        b, c, h, w = high_res.shape
        high_res = torch.einsum("bchw->bhwc", high_res)
        high_res = high_res.reshape(b, h * w, c)
        # run low_res vision tower
        low_res = self.vision_tower_low(low_images)

        if self.concat_type == "feature":
            images_features = torch.cat([high_res, low_res], dim=-1)
        elif self.concat_type == "sequence":
            images_features = torch.cat([high_res, low_res], dim=1)
        elif self.concat_type == "add":
            images_features = high_res + low_res
        elif self.concat_type == "tuple":
            images_features = (high_res, low_res)

        else:
            raise ValueError(
                "Currently only support `feature`, `sequence`, `add` and `tuple` concat type."  # noqa
            )

        return images_features


def model_name_to_cls(cls_name):
    if "MlpProjector" in cls_name:
        cls = MlpProjector

    elif "CLIPVisionTower" in cls_name:
        cls = CLIPVisionTower

    elif "HybridVisionTower" in cls_name:
        cls = HybridVisionTower

    else:
        raise ValueError(f"class_name {cls_name} is invalid.")

    return cls


class MultiModalityPreTrainedModel(PreTrainedModel):
    config_class = DeepSeekMultiModalityConfig
    base_model_prefix = "multi_modality"
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"


@MULTIMODAL_REGISTRY.register_image_feature_input()
@MULTIMODAL_REGISTRY.register_image_pixel_input()
@MULTIMODAL_REGISTRY.register_dummy_data(get_dummy_image_data)
class DeepSeekMultiModalityCausalLM(VisionLanguageModelBase):

    def __init__(
        self,
        config: DeepSeekMultiModalityConfig,
        vision_language_config: VisionLanguageConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__(config, )
        self.config = config
        vision_config = config.vision_config
        aligner_config = config.aligner_config
        self.image_size = aligner_config.params["input_dim"]
        self.image_size = vision_config.params.get("image_size")
        if not self.image_size:
            # Get image size for 7b model
            self.image_size = vision_config.params["high_res_cfg"][
                "image_size"]
        vision_cls = model_name_to_cls(vision_config.cls)
        self.vision_model = vision_cls(**vision_config.params)
        self.vision_tower = self.vision_model
        aligner_cls = model_name_to_cls(aligner_config.cls)
        self.aligner = aligner_cls(aligner_config.params)

        language_config = config.language_config
        self.language_model = LlamaModel(language_config, cache_config,
                                         quant_config)
        self.image_processor = VLMImageProcessor(self.image_size)
        self.logits_processor = LogitsProcessor(language_config.vocab_size)
        self.sampler = Sampler()
        self.lm_head = ParallelLMHead(language_config.vocab_size,
                                      language_config.hidden_size)

    def sample(
        self,
        logits: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        images_seq_mask: torch.LongTensor,
        **kwargs,
    ):
        """

        Args:
            input_ids (torch.LongTensor): [b, T]
            pixel_values (torch.FloatTensor):   [b, n_images, 3, h, w]
            images_seq_mask (torch.BoolTensor): [b, T]

            assert torch.sum(images_seq_mask) == torch.sum(images_emb_mask)

        Returns:
            input_embeds (torch.Tensor): [b, T, D]
        """

        bs, n = pixel_values.shape[0:2]
        p_b, p_n, p_c, p_h, p_w = pixel_values.shape
        images = pixel_values.reshape(p_b * p_n, p_c, p_h, p_w)
        images_embeds = self.aligner(self.vision_model(images))

        # [b x n, T2, D] -> [b, n x T2, D]
        _, t, d = images_embeds.shape
        images_embeds = images_embeds.reshape(bs, n * t, d)

        # [b, T, D]
        input_ids[input_ids < 0] = 0  # ignore the image embeddings
        inputs_embeds = self.language_model.get_input_embeddings(
            input_ids=input_ids)

        # replace with the image embeddings
        images_embeds = images_embeds.reshape(
            -1, self.config.aligner_config.params["n_embed"])
        inputs_embeds[images_seq_mask] = images_embeds

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        **kwargs: object,
    ):
        pixel_values = kwargs.pop("pixel_values", None)
        image_features = kwargs.pop("image_features", None)
        if image_features is not None and pixel_values is None:
            pixel_values = image_features
        if pixel_values is not None:
            image_token_id = 100015
            image_token_mask = input_ids == image_token_id
            inputs_embeds = self.prepare_inputs_embeds(
                input_ids,
                pixel_values.reshape(1, -1, 3, self.image_size,
                                     self.image_size),
                image_token_mask,
            )

            input_ids = None
        else:
            inputs_embeds = None

        hidden_states = self.language_model(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head.weight, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "lm" in name:
                self.lm_head.weight_loader(self.lm_head.weight, loaded_weight)
                continue
            if name.startswith("language_model"):
                name = name.replace("language_model.model.", "language_model.",
                                    1)
            if "rotary_emb.inv_freq" in name:
                continue
            if "language_model" not in name:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # if name not in params_dict:
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip experts that are not assigned to this worker.
                if ("mlp.experts." in name or "mlp.shared_experts."
                        in name) and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if ("mlp.experts." in name or "mlp.shared_experts."
                        in name) and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)


AutoImageProcessor.register(VLMImageProcessorConfig, VLMImageProcessor)
