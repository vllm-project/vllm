# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, Optional, TypedDict, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import (BatchFeature, PretrainedConfig, ProcessorMixin,
                          SequenceFeatureExtractor, SiglipVisionConfig)

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead)
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs, NestedTensors)
from vllm.multimodal.parse import (AudioProcessorItems, ImageEmbeddingItems,
                                   ImageProcessorItems, ImageSize,
                                   MultiModalDataItems, MultiModalDataParser)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.utils import is_list_of

from .idefics2_vision_model import Idefics2VisionTransformer
from .interfaces import MultiModalEmbeddings, SupportsLoRA, SupportsMultiModal
from .phi4mm_audio import AudioEmbedding
from .utils import (AutoWeightsLoader, WeightsMapper, flatten_bn, maybe_prefix,
                    merge_multimodal_embeddings)

# <|endoftext10|> (see vocab.json in hf model)
_IMAGE_PLACEHOLDER_TOKEN_ID = 200010
# <|endoftext11|>
_AUDIO_PLACEHOLDER_TOKEN_ID = 200011

_AUDIO_MAX_SOUNDFILE_SIZE = 241_000

SIGLIP_NAME = "siglip-so400m-patch14-448"
VISION_ENCODER_TO_PROCESSING_CONFIG = {
    'siglip-so400m-patch14-448': {
        'vit_image_size': 448,
        'vit_patch_size': 14,
        'token_compression_factor': 2,
    },
}


def _get_padding_size(orig_width: int, orig_height: int, target_height: int,
                      target_width: int):
    ratio_width = target_width / orig_width
    ratio_height = target_height / orig_height

    if ratio_width < ratio_height:
        padding_width = 0
        padding_height = target_height - int(orig_height * ratio_width)
    else:
        padding_width = target_width - int(orig_width * ratio_height)
        padding_height = 0
    return padding_height, padding_width


def get_navit_vision_model(layer_idx: int = -1, **kwargs):
    vision_config = {
        "hidden_size": 1152,
        "image_size": 448,
        "intermediate_size": 4304,
        "model_type": "siglip_vision_model",
        "num_attention_heads": 16,
        "num_hidden_layers": 27,
        "patch_size": 14,
    }

    model_config = SiglipVisionConfig(**vision_config, **kwargs)
    if layer_idx < 0:
        num_hidden_layers = model_config.num_hidden_layers \
            + layer_idx + 1
    else:
        num_hidden_layers = layer_idx + 1

    vision_model = Idefics2VisionTransformer(
        config=model_config,
        require_post_norm=False,
        num_hidden_layers_override=num_hidden_layers,
    )

    return vision_model


class Phi4MMImageEncoder(nn.Module):
    """Image embedding."""

    def __init__(self,
                 config: PretrainedConfig,
                 quant_config: Optional[QuantizationConfig],
                 prefix: str = "",
                 model_dir: str = "") -> None:
        super().__init__()

        # n_embed or hidden_size
        hidden_size = config.n_embd if hasattr(
            config, 'n_embd') else config.hidden_size

        # layer_idx to output the img features
        if isinstance(config.img_processor, dict):
            self.layer_idx = config.img_processor.get('layer_idx', -2)
            self.type_feature = config.img_processor.get(
                'type_feature', 'patch')
        else:
            self.layer_idx = -2
            self.type_feature = 'patch'

        self.img_processor = get_navit_vision_model(layer_idx=self.layer_idx)

        pe_weight = self.img_processor.embeddings.position_embedding.weight
        L, D = pe_weight.size()
        H = int(math.sqrt(L))
        assert H**2 == L, f'position embedding size {L} is not square'
        if H % 2 != 0:
            self.img_processor_padding = nn.ReflectionPad2d((0, 1, 0, 1))
            H += 1
        image_dim_out = D
        # ((448/14)//2)**2
        self.num_img_tokens = (H // 2)**2
        self.base_feat_height_target = H

        self.image_dim_out = image_dim_out
        self.img_sizes = None
        self.image_attention_mask = None

        # global_gn and sub_gn for hd transform, serves as line separator
        self.use_hd_transform = True
        self.with_learnable_separator = True
        self.hd_transform_order = "sub_glb"
        self.freeze_img_processor = False
        self.crop_size = 448

        # image token compression
        self.image_token_compression_cls = 'avg_pool_2d'
        self.image_token_compression = nn.AvgPool2d(kernel_size=2, stride=2)
        self.base_feat_height_reduction = 1
        self.base_feat_height_target = self.base_feat_height_target // 2

        # with_hd_transform and with_learnable_separator should have same value
        assert self.use_hd_transform == self.with_learnable_separator, \
        'use_hd_transform and with_learnable_separator should have same value'
        assert self.use_hd_transform, \
            'learnable separator is only for hd transform'
        # 1024 * 4, merge spatial to channel dimension
        self.glb_GN = nn.Parameter(
            torch.zeros([
                1, 1, self.image_dim_out * self.base_feat_height_reduction**2
            ]))
        self.sub_GN = nn.Parameter(
            torch.zeros([
                1, 1, 1,
                self.image_dim_out * self.base_feat_height_reduction**2
            ]))

        dim_projection = hidden_size
        depth = 2
        layers = [
            nn.Linear(image_dim_out * self.base_feat_height_reduction**2,
                      dim_projection)
        ]
        for _ in range(1, depth):
            layers.extend(
                [nn.GELU(),
                 nn.Linear(dim_projection, dim_projection)])
        self.img_projection = nn.Sequential(*layers)

        self.vocab_size = config.vocab_size
        self.img_features = None

        self.use_out_place_operations = False

    def get_img_features(self,
                         img_embeds: torch.FloatTensor,
                         attention_mask=None) -> torch.FloatTensor:

        img_feature = self.img_processor(img_embeds,
                                         patch_attention_mask=attention_mask)

        if self.type_feature == "patch":
            patch_feature = img_feature

            use_token_compression = self.image_token_compression is not None
            use_padding = getattr(self, 'img_processor_padding',
                                  None) is not None
            if use_token_compression or use_padding:
                # reshape to 2D tensor
                width = int(math.sqrt(patch_feature.size(1)))
                patch_feature = patch_feature.view(-1, width, width,
                                                   patch_feature.size(-1))
                # convert to NCHW
                patch_feature = patch_feature.permute(0, 3, 1, 2)

                if use_padding:
                    patch_feature = self.img_processor_padding(patch_feature)
                if use_token_compression:
                    patch_feature = self.image_token_compression(patch_feature)

                # convert to NHWC
                patch_feature = patch_feature.permute(0, 2, 3, 1)
                patch_feature = patch_feature.view(
                    -1,
                    patch_feature.size(1) * patch_feature.size(2),
                    patch_feature.size(-1))

            return patch_feature

        raise NotImplementedError

    def forward(self, pixel_values: torch.FloatTensor,
                image_sizes: torch.Tensor,
                image_attention_mask: torch.Tensor) -> list[torch.FloatTensor]:
        """
        process image and return vision embeddings.

        pixel_values: (num_images, num_crops, c, h, w)
        image_sizes: [[h1, w1], [h2, w2]]
        image_attention_mask: num_images x num_crops x 32 x 32
        output: (num_images, num_img_tokens, hidden_size)
        """

        # eg
        # pixel_values: torch.Size([1, 7, 3, 448, 448])
        # image_sizes: tensor([[ 896, 1344]], device='cuda:0')
        # output: torch.Size([1, 1841, 3072])

        if isinstance(self.img_projection, nn.Sequential):
            target_device = self.img_projection[0].bias.device
            target_dtype = self.img_projection[0].bias.dtype
        else:  # It's a single nn.Linear layer
            target_device = self.img_projection.bias.device
            target_dtype = self.img_projection.bias.dtype

        img_sizes = image_sizes
        num_images, num_crops, c, h, w = pixel_values.shape
        bs = num_images
        pixel_values = pixel_values.flatten(0, 1)

        img_features = self.get_img_features(
            pixel_values,
            image_attention_mask.type(torch.BoolTensor).flatten(
                0, 1).to(target_device))

        base_feat_height_target = self.base_feat_height_target
        base_resolution = self.crop_size
        base_feat_height_reduction = self.base_feat_height_reduction

        base_feat_height = base_feat_width = int(np.sqrt(
            img_features.shape[1]))
        assert base_feat_height == base_feat_height_target \
            and base_feat_width == base_feat_height_target, \
                f'base_feat_height: {base_feat_height},"\
                f" base_feat_width: {base_feat_width}, "\
                f"expect {base_feat_height_target} features for hd transform'

        # bs x max_num_crops x (24x24) x C
        img_features = img_features.view(bs, -1,
                                         base_feat_height * base_feat_width,
                                         self.image_dim_out)
        C = self.image_dim_out
        H = base_feat_height

        output_imgs = []
        output_len = []
        # training is tensor, inference is list
        if isinstance(img_sizes, torch.Tensor):
            img_sizes = img_sizes.view(-1, 2)
        for _bs in range(bs):
            h, w = img_sizes[_bs]
            h = h // base_resolution
            w = w // base_resolution
            B_ = h * w

            # 1 x (24x24) x 1024
            global_img_feature = img_features[_bs, :1]

            # 1 x 12 x 12 x 4096
            glb_img = global_img_feature.reshape(1, H, H, C).reshape(
                1, H // base_feat_height_reduction, base_feat_height_reduction,
                H // base_feat_height_reduction, base_feat_height_reduction,
                C).contiguous().permute(0, 1, 3, 2, 4, 5).reshape(
                    1, H // base_feat_height_reduction,
                    H // base_feat_height_reduction,
                    base_feat_height_reduction * base_feat_height_reduction *
                    C).contiguous()
            temp_glb_GN = self.sub_GN.repeat(1,
                                             H // base_feat_height_reduction,
                                             1, 1)

            # 1 x 156 x 4096
            glb_img = torch.cat([glb_img, temp_glb_GN], dim=2).reshape(
                1, -1,
                base_feat_height_reduction * base_feat_height_reduction * C)

            # (max_num_crops-1) x (12x12) x C
            sub_img = img_features[_bs, 1:]
            # 16x574x1024
            # get rid of padding sub_img
            sub_img = sub_img[:B_]

            # (num_crops, 12, 2, 12, 2, 1024) ->
            # (num_crops, 12, 12, 2, 2, 1024) -> (num_crops, 12*12, 4*1024)
            sub_img = sub_img.reshape(B_, H, H, C).reshape(
                B_, H // base_feat_height_reduction,
                base_feat_height_reduction, H // base_feat_height_reduction,
                base_feat_height_reduction,
                C).contiguous().permute(0, 1, 3, 2, 4, 5).reshape(
                    B_, -1, base_feat_height_reduction *
                    base_feat_height_reduction * C).contiguous()
            sub_img = sub_img.reshape(
                1, h, w, base_feat_height // base_feat_height_reduction,
                base_feat_width // base_feat_height_reduction,
                -1).permute(0, 1, 3, 2, 4, 5).reshape(
                    1, h * base_feat_height // base_feat_height_reduction,
                    w * base_feat_width // base_feat_height_reduction,
                    base_feat_height_reduction * base_feat_height_reduction *
                    C)

            if image_attention_mask is not None and len(
                    image_attention_mask) > 0:
                reshaped_image_attention_mask = image_attention_mask[
                    _bs, 1:B_ + 1, 0::2, 0::2].reshape(
                        1, h, w,
                        base_feat_height // base_feat_height_reduction,
                        base_feat_width // base_feat_height_reduction).permute(
                            0, 1, 3, 2, 4).reshape(
                                1, h * base_feat_height //
                                base_feat_height_reduction, w *
                                base_feat_width // base_feat_height_reduction)
                useful_height = int(
                    reshaped_image_attention_mask[0, :, 0].sum().item())
                useful_width = int(
                    reshaped_image_attention_mask[0, 0, :].sum().item())
                sub_img = sub_img[:, :useful_height, :useful_width]
                temp_sub_GN = self.sub_GN.repeat(1, useful_height, 1, 1)
                temp_len = int(
                    image_attention_mask[_bs, :B_ + 1, 0::2, 0::2].sum().item(
                    )) + (useful_height +
                          1) + base_feat_height // base_feat_height_reduction
            else:
                temp_sub_GN = self.sub_GN.repeat(
                    1, h * base_feat_height // base_feat_height_reduction, 1,
                    1)
                temp_len = int((h * w + 1) * self.num_img_tokens + 1 +
                               (h + 1) * base_feat_height //
                               base_feat_height_reduction)

            sub_img = torch.cat([sub_img, temp_sub_GN], dim=2).reshape(
                1, -1,
                base_feat_height_reduction * base_feat_height_reduction * C)
            # (1, num_img_tokens, 1024*4)

            # glb + sub
            if self.hd_transform_order == 'glb_sub':
                output_imgs.append(
                    torch.cat([glb_img, self.glb_GN, sub_img], dim=1))
            elif self.hd_transform_order == 'sub_glb':
                output_imgs.append(
                    torch.cat([sub_img, self.glb_GN, glb_img], dim=1))
            else:
                raise NotImplementedError(
                    f'hd_transform_order = {self.hd_transform_order}, "\
                        "not implemented')

            #temp_len = int((h*w+1)*144 + 1 + (h+1)*12)
            assert temp_len == output_imgs[-1].shape[
                1], f'temp_len: {temp_len}, output_imgs[-1].shape[1]: "\
                    "{output_imgs[-1].shape[1]}'

            output_len.append(temp_len)

        img_set_tensor = []
        for _output_img in output_imgs:
            img_feature_proj = self.img_projection(
                _output_img.to(target_device).to(target_dtype))
            img_set_tensor.append(img_feature_proj.squeeze(0))

        return img_set_tensor


class Phi4MMImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: Union[torch.Tensor, list[torch.Tensor]]
    """
    Shape:
    `(batch_size * num_images, 1 + num_patches, num_channels, height, width)`

    Note that `num_patches` may be different per batch and image,
    in which case the data is passed as a list instead of a batched tensor.
    """

    image_sizes: torch.Tensor
    """
    Shape: `(batch_size * num_images, 2)`

    This should be in `(height, width)` format.
    """

    num_img_tokens: list[int]
    """Shape: `(batch_size * num_images)`"""

    image_attention_mask: torch.Tensor
    """Shape: `(batch_size * num_images, H_mask, W_mask)`"""


class Phi4MMAudioFeatureInputs(TypedDict):
    type: Literal["audio_features"]
    data: Union[torch.Tensor, list[torch.Tensor]]
    """Shape: `(batch_size * num_audios, 80, M)"""


class Phi4MMAudioEmbeddingInputs(TypedDict):
    type: Literal["audio_embeds"]
    data: NestedTensors
    """Shape: `(batch_size, num_audios, audio_feature_size, hidden_size)"""


Phi4MMAudioInputs = Union[Phi4MMAudioFeatureInputs, Phi4MMAudioEmbeddingInputs]


def cat_with_pad(tensors, dim, padding_value=0):
    """
    cat along dim, while pad to max for all other dims
    """
    ndim = tensors[0].dim()
    assert all(
        t.dim() == ndim for t in
        tensors[1:]), "All tensors must have the same number of dimensions"

    out_size = [max(t.shape[i] for t in tensors) for i in range(ndim)]
    out_size[dim] = sum(t.shape[dim] for t in tensors)
    output = tensors[0].new_full(out_size, padding_value)

    index = 0
    for t in tensors:
        # Create a slice list where every dimension except dim is full slice
        slices = [slice(0, t.shape[d]) for d in range(ndim)]
        # Update only the concat dimension slice
        slices[dim] = slice(index, index + t.shape[dim])

        output[slices] = t
        index += t.shape[dim]

    return output


class Phi4MMProcessingInfo(BaseProcessingInfo):

    def get_hf_processor(
        self,
        *,
        dynamic_hd: Optional[int] = None,
        **kwargs: object,
    ) -> ProcessorMixin:
        if dynamic_hd is not None:
            kwargs["dynamic_hd"] = dynamic_hd

        return self.ctx.get_hf_processor(**kwargs)

    @property
    def image_tokens(self) -> list[str]:
        return [f"<|image_{i+1}|>" for i in range(100)]

    @property
    def audio_tokens(self) -> list[str]:
        return [f"<|audio_{i+1}|>" for i in range(100)]

    def get_dynamic_hd(
        self,
        processor: Optional[ProcessorMixin] = None,
    ) -> int:
        if processor is None:
            processor = self.get_hf_processor()
        image_processor = processor.image_processor
        return image_processor.dynamic_hd

    def get_feature_extractor(self) -> SequenceFeatureExtractor:
        return self.get_hf_processor().audio_processor

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"audio": None, "image": None}

    def _find_target_aspect_ratio(
        self,
        orig_width: int,
        orig_height: int,
        image_size: int,
        max_num: int,
        min_num: int,
    ):
        w_crop_num = math.ceil(orig_width / float(image_size))
        h_crop_num = math.ceil(orig_height / float(image_size))
        if w_crop_num * h_crop_num > max_num:
            aspect_ratio = orig_width / orig_height

            # calculate the existing image aspect ratio
            target_ratios = set((i, j) for i in range(1, max_num + 1)
                                for j in range(1, max_num + 1)
                                if i * j <= max_num and i * j >= min_num)
            target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

            # find the closest aspect ratio to the target
            image_processor = self.get_hf_processor().image_processor
            target_aspect_ratio = image_processor.find_closest_aspect_ratio(
                aspect_ratio,
                target_ratios,
                orig_width,
                orig_height,
                image_size,
            )

            # calculate the target width and height
            target_width = image_size * target_aspect_ratio[0]
            target_height = image_size * target_aspect_ratio[1]
        else:
            target_width = image_size * w_crop_num
            target_height = image_size * h_crop_num
            target_aspect_ratio = (w_crop_num, h_crop_num)
        return target_aspect_ratio, target_height, target_width

    def _compute_num_image_tokens(
        self,
        orig_width: int,
        orig_height: int,
        dynamic_hd_size: int,
        vit_image_size: int,
        vit_patch_size: int,
        token_compression_factor: int = 2,
    ):
        """
        compute the number of tokens an image is expected to take up considering
        the image encoder architecture and exclude output features containing 
        only padding pixels

        for siglip, vit_image_size=448, vit_patch_size=14, so output will be 
        32x32 feature map
        NOTE right now, Phi4MM uses hard-coded token_compression_factor=2
        """
        assert vit_image_size % vit_patch_size == 0, (
            "vit_image_size must be divisible by vit_patch_size")
        assert (vit_image_size // vit_patch_size %
                token_compression_factor == 0), (
                    "vit_image_size // vit_patch_size must be divisible by "
                    "token_compression_factor")

        target_aspect_ratio, target_height, target_width = (
            self._find_target_aspect_ratio(orig_width,
                                           orig_height,
                                           vit_image_size,
                                           dynamic_hd_size,
                                           min_num=1))
        assert target_aspect_ratio[0] * vit_image_size == target_width, (
            f"{target_aspect_ratio[0]} * {vit_image_size} != {target_width}")
        assert target_aspect_ratio[1] * vit_image_size == target_height, (
            f"{target_aspect_ratio[1]} * {vit_image_size} != {target_height}")
        assert (target_height % vit_image_size == 0
                and target_width % vit_image_size == 0)

        padding_height, padding_width = _get_padding_size(
            orig_width, orig_height, target_height, target_width)
        assert padding_width == 0 or padding_height == 0, \
            "padding_width or padding_height must be 0"

        target_feat_width = target_width // vit_patch_size
        target_feat_height = target_height // vit_patch_size
        if padding_width >= vit_patch_size:
            assert padding_height == 0, "padding_height not 0"
            non_pad_feat_width = target_feat_width - math.floor(
                padding_width / vit_patch_size)
            non_pad_feat_height = target_feat_height
        elif padding_height >= vit_patch_size:
            assert padding_width == 0, "padding_width not 0"
            non_pad_feat_height = target_feat_height - math.floor(
                padding_height / vit_patch_size)
            non_pad_feat_width = target_feat_width
        else:
            # small padding shorter than a vit patch
            non_pad_feat_width = target_feat_width
            non_pad_feat_height = target_feat_height

        feat_width = non_pad_feat_width // token_compression_factor
        feat_height = non_pad_feat_height // token_compression_factor
        # NOTE it's possible that the non-padding feature is not divisible
        if non_pad_feat_width % token_compression_factor != 0:
            feat_width += 1
        if non_pad_feat_height % token_compression_factor != 0:
            feat_height += 1
        num_hd_patch_tokens = feat_width * feat_height
        num_hd_newline_tokens = feat_height
        vit_feature_size = vit_image_size // vit_patch_size
        num_global_image_tokens = (vit_feature_size //
                                   token_compression_factor)**2
        num_sep_tokens = 1
        num_global_image_newline_tokens = \
            vit_feature_size // token_compression_factor

        return (num_global_image_tokens + num_sep_tokens +
                num_hd_patch_tokens + num_hd_newline_tokens +
                num_global_image_newline_tokens)

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Optional[ProcessorMixin] = None,
    ) -> int:
        hf_config = self.get_hf_config()
        vision_encoder_name = hf_config.img_processor
        if vision_encoder_name is None:
            vision_encoder_name = SIGLIP_NAME
        prepro_config = VISION_ENCODER_TO_PROCESSING_CONFIG[
            vision_encoder_name]
        vit_image_size = prepro_config['vit_image_size']
        vit_patch_size = prepro_config['vit_patch_size']
        token_compression_factor = prepro_config['token_compression_factor']

        dynamic_hd_size = self.get_dynamic_hd(processor=processor)

        image_num_tokens = self._compute_num_image_tokens(
            image_width,
            image_height,
            dynamic_hd_size=dynamic_hd_size,
            vit_image_size=vit_image_size,
            vit_patch_size=vit_patch_size,
            token_compression_factor=token_compression_factor,
        )

        return image_num_tokens

    def get_image_size_with_most_features(
        self,
        processor: Optional[ProcessorMixin] = None,
    ) -> ImageSize:
        hf_config = self.get_hf_config()
        vision_encoder_name = hf_config.img_processor
        if vision_encoder_name is None:
            vision_encoder_name = SIGLIP_NAME
        prepro_config = VISION_ENCODER_TO_PROCESSING_CONFIG[
            vision_encoder_name]
        vit_image_size = prepro_config['vit_image_size']

        max_side = vit_image_size * self.get_dynamic_hd(processor=processor)
        return ImageSize(height=max_side, width=vit_image_size)

    def get_audio_num_frames(self, audio_len: int, sr: float) -> int:
        """
        Compute the output size of the `extract_features` method.

        Args:
            audio_len (int): Length of the input waveform in samples.
            sr (float): Sampling rate of the waveform, either 16000 or 8000.

        Returns:
            tuple (int, int): Output size as (T, D), where:
                T: Number of time frames.
                D: Number of Mel filterbank bins (80).
        """

        # Resample to 16000 or 8000 if needed
        if sr > 16000:
            audio_len //= sr // 16000
        elif 8000 <= sr < 16000:
            # We'll resample to 16K from 8K
            audio_len *= 2
        elif sr < 8000:
            raise RuntimeError(f"Unsupported sample rate {sr}")

        # Spectrogram parameters for 16 kHz
        win_length = 400  # Frame length in samples
        hop_length = 160  # Frame shift in samples

        # Calculate number of frames (T)
        num_frames = (audio_len - win_length) // hop_length + 1
        if num_frames < 1:
            raise ValueError("Waveform too short for given parameters.")

        # Return time frames (T)
        return num_frames

    def _compute_audio_embed_size(self, audio_frames: int) -> int:
        """
        Compute the audio embedding size based on the audio frames and
        compression rate.
        """
        hf_config = self.get_hf_config()
        compression_rate = hf_config.embd_layer['audio_embd_layer'][
            'compression_rate']
        # NOTE: this is a hard-coded value but might be configurable
        # in the future
        qformer_compression_rate = 1
        integer = audio_frames // compression_rate
        remainder = audio_frames % compression_rate

        result = integer if remainder == 0 else integer + 1

        integer = result // qformer_compression_rate
        remainder = result % qformer_compression_rate
        # qformer compression
        result = integer if remainder == 0 else integer + 1

        return result


class Phi4MMDummyInputsBuilder(BaseDummyInputsBuilder[Phi4MMProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        num_images = mm_counts.get("image", 0)

        image_tokens: list[str] = self.info.image_tokens[:num_images]
        audio_tokens: list[str] = self.info.audio_tokens[:num_audios]

        return "".join(image_tokens + audio_tokens)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        num_images = mm_counts.get("image", 0)

        target_width, target_height = \
            self.info.get_image_size_with_most_features()

        mm_data = {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images),
            "audio":
            self._get_dummy_audios(length=_AUDIO_MAX_SOUNDFILE_SIZE,
                                   num_audios=num_audios),
        }

        return mm_data


class Phi4MMMultiModalProcessor(BaseMultiModalProcessor[Phi4MMProcessingInfo]):

    def _get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.info.get_feature_extractor()
        return MultiModalDataParser(target_sr=feature_extractor.sampling_rate,
                                    audio_resample_method="scipy")

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if not mm_data:
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        sr = self.info.get_feature_extractor().sampling_rate
        if (audio_data := mm_data.get("audios", [])):
            mm_data['audios'] = [(data, sr) for data in audio_data]

        processed_outputs = super()._call_hf_processor(prompt, mm_data,
                                                       mm_kwargs)

        num_img_tokens = [
            self.info.get_num_image_tokens(image_width=img_size[0],
                                           image_height=img_size[1])
            for img_size in processed_outputs["image_sizes"]
        ]
        processed_outputs["num_img_tokens"] = num_img_tokens

        audio_features = processed_outputs['input_audio_embeds']
        feature_sizes = [
            self.info.get_audio_num_frames(len(audio), sr)
            for audio in audio_data
        ]
        processed_outputs['input_audio_embeds'] = [
            audio_features[idx, :size]
            for idx, size in enumerate(feature_sizes)
        ]

        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            input_image_embeds=MultiModalFieldConfig.batched("image"),
            image_attention_mask=MultiModalFieldConfig.batched("image"),
            image_sizes=MultiModalFieldConfig.batched("image"),
            num_img_tokens=MultiModalFieldConfig.batched("image"),
            input_audio_embeds=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        image_tokens: list[str] = self.info.image_tokens  # type: ignore
        audio_tokens: list[str] = self.info.audio_tokens  # type: ignore
        feature_extractor = self.info.get_feature_extractor()
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        def get_image_replacement_phi4mm(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems))

            if isinstance(images, ImageEmbeddingItems):
                num_image_tokens = images.get_feature_size(item_idx)
            else:
                image_size = images.get_image_size(item_idx)
                num_image_tokens = self.info.get_num_image_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                    processor=hf_processor,
                )

            image_tokens = [_IMAGE_PLACEHOLDER_TOKEN_ID] * num_image_tokens

            return image_tokens

        def get_audio_replacement_phi4mm(item_idx: int):
            audios = mm_items.get_items("audio", AudioProcessorItems)
            # TODO(Isotr0py): support embedding inputs
            audio_len = audios.get_audio_length(item_idx)
            audio_frames = self.info.get_audio_num_frames(
                audio_len, feature_extractor.sampling_rate)
            audio_embed_size = self.info._compute_audio_embed_size(
                audio_frames)

            audio_tokens = [_AUDIO_PLACEHOLDER_TOKEN_ID] * audio_embed_size

            return audio_tokens

        num_images = mm_items.get_count("image", strict=False)
        num_audios = mm_items.get_count("audio", strict=False)

        image_repl = [
            PromptReplacement(
                modality="image",
                target=image_token,
                replacement=get_image_replacement_phi4mm,
            ) for image_token in image_tokens[:num_images]
        ]
        audio_repl = [
            PromptReplacement(
                modality="audio",
                target=audio_token,
                replacement=get_audio_replacement_phi4mm,
            ) for audio_token in audio_tokens[:num_audios]
        ]
        return image_repl + audio_repl


@MULTIMODAL_REGISTRY.register_processor(
    Phi4MMMultiModalProcessor,
    info=Phi4MMProcessingInfo,
    dummy_inputs=Phi4MMDummyInputsBuilder,
)
class Phi4MMForCausalLM(nn.Module, SupportsLoRA, SupportsMultiModal):
    """
    Implements the Phi-4-multimodal-instruct model in vLLM.
    """
    packed_modules_mapping = {
        "qkv_proj": [
            "qkv_proj",
        ],
        "gate_up_proj": [
            "gate_up_proj",
        ],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            "base_layer.": "",
        },
        orig_to_new_prefix={
            "model.embed_tokens_extend.audio_embed.audio_projection.vision.":
            "embed_tokens_extend.audio_projection_for_vision.",
            "model.embed_tokens_extend.audio_embed.audio_projection.speech.":
            "embed_tokens_extend.audio_projection.",
            "model.embed_tokens_extend.audio_embed.": "embed_tokens_extend.",
            "model.embed_tokens_extend.image_embed.": "vision_encoder.",
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        assert multimodal_config, "multimodal_config is required"
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.multimodal_config = multimodal_config
        self.quant_config = quant_config
        self.lora_config = lora_config

        # Tensor/Pipeline parallel not supported for now.
        assert get_pp_group(
        ).world_size == 1, "pipeline parallel is not supported"

        self.vision_encoder = Phi4MMImageEncoder(
            config,
            quant_config,
            prefix="model.vision_embed_tokens",
            model_dir=config._name_or_path)

        if isinstance(config.embd_layer["audio_embd_layer"], dict):
            embedding_config = {
                "embedding_cls":
                config.embd_layer["audio_embd_layer"]["embedding_cls"],
                **config.embd_layer["audio_embd_layer"],
            }
        else:
            embedding_config = {
                "embedding_cls": self.config.embd_layer["embedding_cls"]
            }

        self.embed_tokens_extend = AudioEmbedding(config, **embedding_config)
        self.model = LlamaModel(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))

        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE,
            quant_config=quant_config,
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size, logit_scale)

    def _parse_and_validate_audio_input(
            self, **kwargs: object) -> Optional[Phi4MMAudioInputs]:
        """
        Parse and validate the audio input to the model.  This handles both 
        audio features and audio embeddings, but only the former is used for
        now.

        Args:
            kwargs (object): Keyword arguments.

        Returns:
            Optional[Phi4MMAudioInputs]: Parsed and validated audio inputs.
        """
        audio_features = kwargs.pop("input_audio_embeds", None)
        audio_embeds = kwargs.pop("audio_embeds", None)

        if audio_features is None and audio_embeds is None:
            return None

        if audio_features is not None:
            if not isinstance(audio_features, (torch.Tensor, list)):
                raise ValueError("Incorrect type of audio features. "
                                 f"Got type: {type(audio_features)}")

            return Phi4MMAudioFeatureInputs(type="audio_features",
                                            data=flatten_bn(audio_features))

        if audio_embeds is not None:
            if not isinstance(audio_embeds, (torch.Tensor, list)):
                raise ValueError("Incorrect type of audio embeds. "
                                 f"Got type: {type(audio_embeds)}")

            return Phi4MMAudioEmbeddingInputs(type="audio_embeds",
                                              data=audio_embeds)

        raise AssertionError("This line should be unreachable.")

    def _process_audio_input(self, audio_input: Phi4MMAudioInputs,
                             audio_projection_mode: str) -> NestedTensors:
        """
        Create the audio embeddings from the audio input, where the audio input
        is pairs of audio features and audio embed lengths.  The audio input is
        created by `input_mapper_for_phi4mm_audio`.

        Args:
            audio_input (Phi4MMAudioInputs): Audio input.

        Returns:
            NestedTensors: Audio embeddings
        """
        if audio_input["type"] == "audio_embeds":
            return audio_input["data"]

        audio_features = audio_input["data"]
        # (e.g. multiple examples) and the second dim is the multi-audio dim
        # (e.g. multiple audios in the same example)

        dtype = next(self.embed_tokens_extend.parameters()).dtype
        audio_embeds = [
            self.embed_tokens_extend(
                features.to(dtype),
                audio_projection_mode=audio_projection_mode,
            ) for features in audio_features
        ]
        return audio_embeds

    def _parse_and_validate_image_input(self,
                                        **kwargs: object) -> Optional[dict]:
        input_image_embeds: NestedTensors = kwargs.get("input_image_embeds")
        if input_image_embeds is None:
            return None

        image_sizes = kwargs.get("image_sizes")
        image_attention_mask = kwargs.get("image_attention_mask")
        num_img_tokens = kwargs.get("num_img_tokens")
        assert image_sizes is not None and image_attention_mask is not None\
              and num_img_tokens is not None, "Missing image inputs"

        if is_list_of(input_image_embeds, torch.Tensor):
            assert all(p.dim() == 5
                       for p in input_image_embeds), "Incorrect image inputs"
            # list len is batch_size.
            # each tensor has dimension: num_img_per_example, num_hd_patches,
            # channels, height, width.
            # need to pad along num_hd_patches.
            # mask size num_img_per_prompt, num_hd_patches, feat_h, heat_w.
            input_image_embeds = cat_with_pad(input_image_embeds, dim=0)
        elif isinstance(input_image_embeds, torch.Tensor):
            # dimension: batch_size, num_img_per_example, num_hd_patches,
            # channels, height, width.
            # we flatten first 2 dims to make it a single large batch for
            # SigLIP Encoder.
            assert input_image_embeds.dim() == 6, "Incorrect image inputs"
            input_image_embeds = input_image_embeds.flatten(0, 1)
        else:
            raise ValueError("Incorrect input_image_embeds inputs")

        if isinstance(image_attention_mask, list):
            image_attention_mask = cat_with_pad(image_attention_mask, dim=0)
        elif isinstance(image_attention_mask, torch.Tensor):
            image_attention_mask = image_attention_mask.flatten(0, 1)
        else:
            raise ValueError("Incorrect image_attention_mask inputs")

        if isinstance(image_sizes, list):
            image_sizes = torch.cat(image_sizes, dim=0)
        elif isinstance(image_sizes, torch.Tensor):
            image_sizes = image_sizes.flatten(0, 1)
        else:
            raise ValueError("Incorrect image_attention_mask inputs")

        if isinstance(num_img_tokens, list):
            num_img_tokens = [
                n for num_tensor in num_img_tokens
                for n in num_tensor.tolist()
            ]
        elif isinstance(num_img_tokens, torch.Tensor):
            num_img_tokens = num_img_tokens.flatten(0, 1).tolist()
        else:
            raise ValueError("Incorrect image_attention_mask inputs")

        return Phi4MMImagePixelInputs(
            type="pixel_values",
            data=input_image_embeds,
            image_sizes=image_sizes,
            image_attention_mask=image_attention_mask,
            num_img_tokens=num_img_tokens,
        )

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if input_key in ("input_image_embeds",
                             "image_embeds") and "images" not in modalities:
                modalities["images"] = self._parse_and_validate_image_input(
                    **kwargs)
            if input_key in ("input_audio_embeds",
                             "audio_embeds") and "audios" not in modalities:
                modalities["audios"] = self._parse_and_validate_audio_input(
                    **kwargs)

        return modalities

    def _process_image_input(
            self, image_input: Phi4MMImagePixelInputs) -> list[torch.Tensor]:

        dtype = next(self.vision_encoder.parameters()).dtype
        pixel_values = image_input['data'].to(dtype)
        image_sizes = image_input['image_sizes']
        image_attention_mask = image_input['image_attention_mask']
        image_embeds = self.vision_encoder(pixel_values, image_sizes,
                                           image_attention_mask)
        return image_embeds

    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:

        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not modalities:
            return None

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        audio_projection_mode = 'speech'
        for modality in modalities:
            # make sure process images first
            if modality == "images":
                audio_projection_mode = "vision"
                image_input = modalities["images"]
                vision_embeddings = self._process_image_input(image_input)
                multimodal_embeddings += tuple(vision_embeddings)
            if modality == "audios":
                audio_input = modalities["audios"]
                audio_embeddings = self._process_audio_input(
                    audio_input, audio_projection_mode=audio_projection_mode)
                multimodal_embeddings += tuple(audio_embeddings)

        return multimodal_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.embed_tokens(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                [_IMAGE_PLACEHOLDER_TOKEN_ID, _AUDIO_PLACEHOLDER_TOKEN_ID])
        return inputs_embeds

    def get_input_embeddings_v0(
        self,
        input_ids: torch.Tensor,
        image_input: Optional[Phi4MMImagePixelInputs] = None,
        audio_input: Optional[Phi4MMAudioFeatureInputs] = None,
    ) -> torch.Tensor:
        audio_projection_mode = 'speech'
        inputs_embeds = self.get_input_embeddings(input_ids)
        if image_input is not None:
            image_embeds = self._process_image_input(image_input)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                image_embeds,
                placeholder_token_id=_IMAGE_PLACEHOLDER_TOKEN_ID,
            )
            audio_projection_mode = 'vision'

        if audio_input is not None:
            audio_embeds = self._process_audio_input(
                audio_input, audio_projection_mode=audio_projection_mode)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                audio_embeds,
                placeholder_token_id=_AUDIO_PLACEHOLDER_TOKEN_ID,
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner from
        # `get_multimodal_embeddings` and `get_input_embeddings`, this
        # condition is only for v0 compatibility.
        elif inputs_embeds is None:
            image_input = self._parse_and_validate_image_input(**kwargs)
            audio_input = self._parse_and_validate_audio_input(**kwargs)

            if image_input is None and audio_input is None:
                inputs_embeds = None
            else:
                inputs_embeds = self.get_input_embeddings_v0(
                    input_ids,
                    image_input=image_input,
                    audio_input=audio_input)
                input_ids = None

        hidden_states = self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> None:
        loader = AutoWeightsLoader(self, skip_substrs=["lora"])
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="model.",
            connector=["audio_projection_for_vision", "audio_projection"],
            tower_model=["vision_encoder", "embed_tokens_extend"],
        )

    def get_language_model(self) -> torch.nn.Module:
        return self.model
