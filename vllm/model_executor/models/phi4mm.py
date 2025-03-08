# SPDX-License-Identifier: Apache-2.0
import math
import re
from functools import lru_cache
from typing import (Dict, Iterable, List, Literal, Mapping, Optional, Tuple,
                    TypedDict, Union)

import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from transformers import PretrainedConfig
from transformers.utils import logging

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.inputs import (INPUT_REGISTRY, DecoderOnlyInputs, DummyData,
                         InputContext)
from vllm.inputs.data import TokenInputs, token_inputs
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead)
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalInputs, NestedTensors
from vllm.sequence import IntermediateTensors, SequenceData
from vllm.transformers_utils.tokenizer import cached_tokenizer_from_config

from .interfaces import SupportsLoRA, SupportsMultiModal
from .phi4mm_audio import AudioEmbedding
from .utils import maybe_prefix
from .vision_siglip_navit import get_siglip_vision_model

# <|endoftext10|> (see vocab.json in hf model)
_IMAGE_PLACEHOLDER_TOKEN_ID = 200010
# <|endoftext11|>
_AUDIO_PLACEHOLDER_TOKEN_ID = 200011

_AUDIO_MAX_SOUNDFILE_SIZE = 241_000
DUMMY_SAMPLING_FREQUENCY = 16_000  # kHz

DYNAMIC_HD = 16
AUDIO_TOKEN_PATTERN = r"<\|audio_(\d+)\|>"
IMAGE_TOKEN_PATTERN = r"<\|image_(\d+)\|>"

SIGLIP_NAME = "siglip-so400m-patch14-448"
VISION_ENCODER_TO_PROCESSING_CONFIG = {
    'siglip-so400m-patch14-448': {
        'dynamic_hd': 16,
        'vit_image_size': 448,
        'vit_patch_size': 14,
        'token_compression_factor': 2,
    },
}
logger = logging.get_logger(__name__)
# This is a workaround to prevent text (user input) + audio + image
# from being used in the same prompt.
# It includes token ids for "/n" and tokens in added_tokens_decoder
# from the tokenizer_confg.json file.
NON_USER_INPUT_TOKENS = {
    198, 200010, 200011, 199999, 200018, 200019, 200020, 200021, 200022,
    200023, 200024, 200025, 200026, 200027, 200028
}


def get_max_dummy_image(ctx: InputContext):
    hf_config = ctx.get_hf_config()
    vision_encoder_name = hf_config.img_processor
    if vision_encoder_name is None:
        vision_encoder_name = SIGLIP_NAME
    prepro_config = VISION_ENCODER_TO_PROCESSING_CONFIG[vision_encoder_name]
    dynamic_hd_size = prepro_config['dynamic_hd']
    vit_image_size = prepro_config['vit_image_size']

    max_side = vit_image_size * dynamic_hd_size
    dummy_image = dummy_image_for_phi4mm(vit_image_size, max_side)
    return dummy_image


# image token length
def get_max_phi4mm_image_tokens(ctx: InputContext):
    dummy_image = get_max_dummy_image(ctx)

    hf_config = ctx.get_hf_config()
    vision_encoder_name = hf_config.img_processor
    if vision_encoder_name is None:
        vision_encoder_name = SIGLIP_NAME
    prepro_config = VISION_ENCODER_TO_PROCESSING_CONFIG[vision_encoder_name]
    dynamic_hd_size = prepro_config['dynamic_hd']
    vit_image_size = prepro_config['vit_image_size']
    vit_patch_size = prepro_config['vit_patch_size']
    token_compression_factor = prepro_config['token_compression_factor']

    image_num_tokens = _compute_num_image_tokens(dummy_image, dynamic_hd_size,
                                                 vit_image_size,
                                                 vit_patch_size,
                                                 token_compression_factor)
    return image_num_tokens


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height,
                              image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _find_target_aspect_ratio(image, image_size, max_num, min_num):
    orig_width, orig_height = image.size

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
        target_aspect_ratio = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        logger.debug("target_aspect_ratio: %s", target_aspect_ratio)
    else:
        target_width = image_size * w_crop_num
        target_height = image_size * h_crop_num
        target_aspect_ratio = (w_crop_num, h_crop_num)
    return target_aspect_ratio, target_height, target_width


def _get_padding_size(image, target_height, target_width):
    orig_width, orig_height = image.size
    ratio_width = target_width / orig_width
    ratio_height = target_height / orig_height

    if ratio_width < ratio_height:
        padding_width = 0
        padding_height = target_height - int(orig_height * ratio_width)
    else:
        padding_width = target_width - int(orig_width * ratio_height)
        padding_height = 0
    return padding_height, padding_width


def dynamic_preprocess(image,
                       min_num=1,
                       max_num=12,
                       image_size=384,
                       mask_size=27):
    target_aspect_ratio, target_height, target_width =\
          _find_target_aspect_ratio(
        image, image_size, max_num, min_num)
    padding_height, padding_width = _get_padding_size(image, target_height,
                                                      target_width)

    # Calculate the ratio
    orig_width, orig_height = image.size
    ratio_width = target_width / orig_width
    ratio_height = target_height / orig_height
    if ratio_width < ratio_height:
        new_size = (target_width, int(orig_height * ratio_width))
    else:
        new_size = (int(orig_width * ratio_height), target_height)

    attention_mask = torch.ones((int(mask_size * target_aspect_ratio[1]),
                                 int(mask_size * target_aspect_ratio[0])))
    if padding_width >= 14:
        attention_mask[:, -math.floor(padding_width / 14):] = 0
    if padding_height >= 14:
        attention_mask[-math.floor(padding_height / 14):, :] = 0
    assert attention_mask.sum(
    ) > 0, f'attention mask is empty {attention_mask}'

    if min(new_size[1], target_height) < 10 or min(new_size[0],
                                                   target_width) < 10:
        raise ValueError(f'the aspect ratio is very extreme {new_size}')

    image = T.functional.resize(
        image,
        [new_size[1], new_size[0]],
    )

    resized_img = T.functional.pad(image,
                                   [0, 0, padding_width, padding_height],
                                   fill=[255, 255, 255])

    return resized_img, attention_mask


def pad_to_max_num_crops(images, max_crops=5):
    """
    images: B x 3 x H x W, B<=max_crops
    """
    B, _, H, W = images.shape
    if max_crops > B:
        pad = torch.zeros(max_crops - B,
                          3,
                          H,
                          W,
                          dtype=images.dtype,
                          device=images.device)
        images = torch.cat([images, pad], dim=0)
    return images


def pad_mask_to_max_num_crops(masks, max_crops=5):
    B, H, W = masks.shape
    if max_crops > B:
        pad = torch.ones(max_crops - B,
                         H,
                         W,
                         dtype=masks.dtype,
                         device=masks.device)
        masks = torch.cat([masks, pad], dim=0)
    return masks


def preprocess(images, dynamic_hd_size, vit_resolution, vit_patch_size):

    # Basic settings.
    img_processor = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # Dynamic HD
    base_resolution = vit_resolution
    images = [image.convert('RGB') for image in images]
    # cover 384 and 448 resolution
    mask_resolution = base_resolution // vit_patch_size
    elems, image_attention_masks = [], []
    for im in images:
        elem, attention_mask = dynamic_preprocess(im,
                                                  max_num=dynamic_hd_size,
                                                  image_size=base_resolution,
                                                  mask_size=mask_resolution)
        elems.append(elem)
        image_attention_masks.append(attention_mask)
    hd_images = [img_processor(im) for im in elems]
    global_image = [
        torch.nn.functional.interpolate(
            im.unsqueeze(0).float(),
            size=(base_resolution, base_resolution),
            mode='bicubic',
        ).to(im.dtype) for im in hd_images
    ]
    shapes = [[im.size(1), im.size(2)] for im in hd_images]
    mask_shapes = [[mask.size(0), mask.size(1)]
                   for mask in image_attention_masks]
    global_attention_mask = [
        torch.ones((1, mask_resolution, mask_resolution)) for _ in hd_images
    ]
    hd_images_reshape = [
        im.reshape(1, 3, h // base_resolution, base_resolution,
                   w // base_resolution, base_resolution).permute(
                       0, 2, 4, 1, 3, 5).reshape(-1, 3, base_resolution,
                                                 base_resolution).contiguous()
        for im, (h, w) in zip(hd_images, shapes)
    ]
    attention_masks_reshape = [
        mask.reshape(1, h // mask_resolution, mask_resolution,
                     w // mask_resolution, mask_resolution).permute(
                         0, 1, 3, 2, 4).reshape(-1, mask_resolution,
                                                mask_resolution).contiguous()
        for mask, (h, w) in zip(image_attention_masks, mask_shapes)
    ]
    # NOTE token compression is hard coded here, and odd numbers seems to fail
    downsample_attention_masks = [
        mask[:, 0::2,
             0::2].reshape(1, h // mask_resolution, w // mask_resolution,
                           mask_resolution // 2 + mask_resolution % 2,
                           mask_resolution // 2 + mask_resolution % 2).permute(
                               0, 1, 3, 2, 4)
        for mask, (h, w) in zip(attention_masks_reshape, mask_shapes)
    ]
    downsample_attention_masks = [
        mask.reshape(mask.size(1) * mask.size(2),
                     mask.size(3) * mask.size(4))
        for mask in downsample_attention_masks
    ]
    # NOTE hard coded number of tokens
    num_img_tokens = [
        256 + 1 + int(mask.sum().item()) + int(mask[:, 0].sum().item()) + 16
        for mask in downsample_attention_masks
    ]

    hd_images_reshape = [
        torch.cat([_global_image] + [_im], dim=0)
        for _global_image, _im in zip(global_image, hd_images_reshape)
    ]
    hd_masks_reshape = [
        torch.cat([_global_mask] + [_mask],
                  dim=0) for _global_mask, _mask in zip(
                      global_attention_mask, attention_masks_reshape)
    ]
    max_crops = max([img.size(0) for img in hd_images_reshape])
    image_transformed = [
        pad_to_max_num_crops(im, max_crops) for im in hd_images_reshape
    ]
    image_transformed = torch.stack(image_transformed, dim=0)
    mask_transformed = [
        pad_mask_to_max_num_crops(mask, max_crops) \
            for mask in hd_masks_reshape
    ]
    mask_transformed = torch.stack(mask_transformed, dim=0)

    returned_input_image_embeds = image_transformed
    returned_image_sizes = torch.tensor(shapes, dtype=torch.long)
    returned_image_attention_mask = mask_transformed
    returned_num_img_tokens = num_img_tokens

    data = {
        "pixel_values": returned_input_image_embeds,
        "image_sizes": returned_image_sizes,
        "image_attention_mask": returned_image_attention_mask,
        "num_img_tokens": returned_num_img_tokens,
    }
    return data


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
        if hasattr(config, 'embd_pdrop') or hasattr(config, 'embed_pdrop'):
            embd_drop = config.embd_pdrop if hasattr(
                config, 'embd_pdrop') else config.embed_pdrop
            self.drop = nn.Dropout(embd_drop)
        else:
            self.drop = None

        # layer_idx to output the img features
        if isinstance(config.img_processor, dict):
            self.layer_idx = config.img_processor.get('layer_idx', -2)
            self.type_feature = config.img_processor.get(
                'type_feature', 'patch')
        else:
            self.layer_idx = -2
            self.type_feature = 'patch'

        self.img_processor = get_siglip_vision_model(
            _flash_attn_2_enabled=True)

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
        LAYER_IDX = self.layer_idx
        TYPE_FEATURE = self.type_feature

        img_processor_output = self.img_processor(
            img_embeds,
            output_hidden_states=True,
            patch_attention_mask=attention_mask)
        img_feature = img_processor_output.hidden_states[LAYER_IDX]

        if TYPE_FEATURE == "patch":
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
                image_attention_mask: torch.Tensor) -> torch.FloatTensor:
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
            img_set_tensor.append(img_feature_proj)

        return img_set_tensor


class Phi4MMAudioFeatureInputs(TypedDict):
    type: Literal["audio_features"]
    data: Tuple[NestedTensors]
    """Shape: `((batch_size, num_audios, 80, M), )"""


class Phi4MMAudioEmbeddingInputs(TypedDict):
    type: Literal["audio_embeds"]
    data: NestedTensors
    """Shape: `(batch_size, num_audios, audio_feature_size, hidden_size)"""


Phi4MMAudioInputs = Union[Phi4MMAudioFeatureInputs, Phi4MMAudioEmbeddingInputs]


def speechlib_mel(sample_rate, n_fft, n_mels, fmin=None, fmax=None):
    """Create a Mel filter-bank the same as SpeechLib FbankFC.

    Args:
        sample_rate (int): Sample rate in Hz. number > 0 [scalar]
        n_fft (int): FFT size. int > 0 [scalar]
        n_mel (int): Mel filter size. int > 0 [scalar]
        fmin (float): lowest frequency (in Hz). If None use 0.0.
            float >= 0 [scalar]
        fmax: highest frequency (in Hz). If None use sample_rate / 2.
            float >= 0 [scalar]

    Returns
        out (numpy.ndarray): Mel transform matrix
            [shape=(n_mels, 1 + n_fft/2)]
    """

    bank_width = int(n_fft // 2 + 1)
    if fmax is None:
        fmax = sample_rate / 2
    if fmin is None:
        fmin = 0
    assert fmin >= 0, "fmin cannot be negative"
    assert (fmin < fmax <=
            sample_rate / 2), "fmax must be between (fmin, samplerate / 2]"

    def mel(f):
        return 1127.0 * np.log(1.0 + f / 700.0)

    def bin2mel(fft_bin):
        return 1127.0 * np.log(1.0 + fft_bin * sample_rate / (n_fft * 700.0))

    def f2bin(f):
        return int((f * n_fft / sample_rate) + 0.5)

    # Spec 1: FFT bin range [f2bin(fmin) + 1, f2bin(fmax) - 1]
    klo = f2bin(fmin) + 1
    khi = f2bin(fmax)

    khi = max(khi, klo)

    # Spec 2: SpeechLib uses triangles in Mel space
    mlo = mel(fmin)
    mhi = mel(fmax)
    m_centers = np.linspace(mlo, mhi, n_mels + 2)
    ms = (mhi - mlo) / (n_mels + 1)

    matrix = np.zeros((n_mels, bank_width), dtype=np.float32)
    for m in range(0, n_mels):
        left = m_centers[m]
        center = m_centers[m + 1]
        right = m_centers[m + 2]
        for fft_bin in range(klo, khi):
            mbin = bin2mel(fft_bin)
            if left < mbin < right:
                matrix[m, fft_bin] = 1.0 - abs(center - mbin) / ms

    return matrix


class LogFbankProcessor:

    def __init__(self):

        self._eightk_method = "fillzero"
        self._mel = speechlib_mel(16000, 512, 80, fmin=None, fmax=7690).T

        self._hamming400 = np.hamming(400)  # for 16k audio
        self._hamming200 = np.hamming(200)  # for 8k audio

    def extract_spectrogram(self, wav, fs):
        """Extract spectrogram features from waveform.
        Args:
            wav (1D array): waveform of the input
            fs (int): sampling rate of the waveform, 16000 or 8000.
                If fs=8000, the waveform will be resampled to 16000Hz.
        Output:
            log_fbank (2D array): a TxD matrix of log Mel filterbank features.
                D=80, and T is the number of frames.
        """
        if wav.ndim > 1:
            wav = np.squeeze(wav)

        # by default, we extract the mean if stereo
        if len(wav.shape) == 2:
            wav = wav.mean(1)

        # Resample to 16000 or 8000 if needed
        if fs > 16000:
            wav = scipy.signal.resample_poly(wav, 1, fs // 16000)
            fs = 16000
        elif 8000 < fs < 16000:
            wav = scipy.signal.resample_poly(wav, 1, fs // 8000)
            fs = 8000
        elif fs < 8000:
            raise RuntimeError(f"Unsupported sample rate {fs}")

        if fs == 8000:
            if self._eightk_method == "resample":
                # Input audio is 8 kHz. Convert to 16 kHz before feature
                # extraction
                wav = scipy.signal.resample_poly(wav, 2, 1)
                fs = 16000
            # Do nothing here for fillzero method
        elif fs != 16000:
            # Input audio is not a supported sample rate.
            raise RuntimeError(
                f"Input data using an unsupported sample rate: {fs}")

        preemphasis = 0.97

        if fs == 8000:
            n_fft = 256
            win_length = 200
            hop_length = 80
            fft_window = self._hamming200
        elif fs == 16000:
            n_fft = 512
            win_length = 400
            hop_length = 160
            fft_window = self._hamming400

        # Spec 1: SpeechLib cut remaining sample insufficient for a hop
        n_batch = (wav.shape[0] - win_length) // hop_length + 1
        # Here we don't use stride_tricks since the input array may not satisfy
        # memory layout requirement and we need writeable output
        # Here we only use list of views before copy to destination
        # so it is more efficient than broadcasting
        y_frames = np.array(
            [
                wav[_stride:_stride + win_length]
                for _stride in range(0, hop_length * n_batch, hop_length)
            ],
            dtype=np.float32,
        )

        # Spec 2: SpeechLib applies preemphasis within each batch
        y_frames_prev = np.roll(y_frames, 1, axis=1)
        y_frames_prev[:, 0] = y_frames_prev[:, 1]
        y_frames = (y_frames - preemphasis * y_frames_prev) * 32768

        S = np.fft.rfft(fft_window * y_frames, n=n_fft,
                        axis=1).astype(np.complex64)

        if fs == 8000:
            # Need to pad the output to look like 16 kHz data but with zeros in
            # the 4 to 8 kHz bins.
            frames, bins = S.shape
            padarray = np.zeros((frames, bins))
            S = np.concatenate((S[:, 0:-1], padarray),
                               axis=1)  # Nyquist bin gets set to zero

        spec = np.abs(S).astype(np.float32)
        return spec

    def extract_features(self, wav, fs):
        """Extract log filterbank features from waveform.
        Args:
            wav (1D array): waveform of the input
            fs (int): sampling rate of the waveform, 16000 or 8000.
                If fs=8000, the waveform will be resampled to 16000Hz.
        Output:
            log_fbank (2D array): a TxD matrix of log Mel filterbank features.
                D=80, and T is the number of frames.
        """
        spec = self.extract_spectrogram(wav, fs)
        spec_power = spec**2

        fbank_power = np.clip(spec_power.dot(self._mel), 1.0, None)
        log_fbank = np.log(fbank_power).astype(np.float32)

        return log_fbank


@lru_cache
def audio_feature_extractor() -> LogFbankProcessor:
    # Creates an instance of the audio processor, needed to extract the
    # the audio features from the sound file
    # LRU cache ensures that we only make one copy
    return LogFbankProcessor()


def _compute_num_image_tokens(image, dynamic_hd_size, vit_image_size,
                              vit_patch_size, token_compression_factor):
    """
    compute the number of tokens an image is expected to take up considering 
    the image encoder architecture and exclude output features containing 
    only padding pixels

    for siglip, vit_image_size=448, vit_patch_size=14, so output will be 
    32x32 feature map
    NOTE right now, Phi4MM uses hard-coded token_compression_factor=2
    """
    assert vit_image_size % vit_patch_size == 0, \
        "vit_image_size must be divisible by vit_patch_size"
    assert vit_image_size // vit_patch_size % token_compression_factor == 0, \
        "vit_image_size // vit_patch_size must be divisible by "\
            "token_compression_factor"

    target_aspect_ratio, target_height, target_width = (
        _find_target_aspect_ratio(image,
                                  vit_image_size,
                                  dynamic_hd_size,
                                  min_num=1))
    assert target_aspect_ratio[
        0] * vit_image_size == target_width, \
            f"{target_aspect_ratio[0]} * {vit_image_size} != {target_width}"
    assert target_aspect_ratio[
        1] * vit_image_size == target_height, \
            f"{target_aspect_ratio[1]} * {vit_image_size} != {target_height}"
    assert (target_height % vit_image_size == 0
            and target_width % vit_image_size == 0)

    padding_height, padding_width = _get_padding_size(image, target_height,
                                                      target_width)
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
    num_global_image_tokens = (vit_feature_size // token_compression_factor)**2
    num_sep_tokens = 1
    num_global_image_newline_tokens = \
        vit_feature_size // token_compression_factor

    return (num_global_image_tokens + num_sep_tokens + num_hd_patch_tokens +
            num_hd_newline_tokens + num_global_image_newline_tokens)


def compute_logfbank_output_size(wav_length: int, fs: int) -> Tuple[int, int]:
    """
    Compute the output size of the `extract_features` method.

    Args:
        wav_length (int): Length of the input waveform in samples.
        fs (int): Sampling rate of the waveform, either 16000 or 8000.

    Returns:
        tuple (int, int): Output size as (T, D), where:
            T: Number of time frames.
            D: Number of Mel filterbank bins (80).
    """

    # Resample to 16000 or 8000 if needed
    if fs > 16000:
        wav_length //= fs // 16000
        fs = 16000
    elif 8000 <= fs < 16000:
        # We'll resample to 16K from 8K
        wav_length *= 2
        fs = 16000
    elif fs < 8000:
        raise RuntimeError(f"Unsupported sample rate {fs}")

    # Spectrogram parameters for 16 kHz
    win_length = 400  # Frame length in samples
    hop_length = 160  # Frame shift in samples
    mel_bins = 80  # Number of mel filterbank bins

    # Calculate number of frames (T)
    T = (wav_length - win_length) // hop_length + 1
    if T < 1:
        raise ValueError("Waveform too short for given parameters.")

    # Return time frames (T) and mel bins (D)
    return T, mel_bins


def _get_audio_embed_sizes(audios, ctx: InputContext):
    """
    Get the audio embedding sizes for each audio file.

    Args:
        audios (List[Tuple[np.ndarray, int]]): List of audio files as tuples of
            waveform and sample rate.
        ctx (InputContext): Input context.

    Returns:
        List[int]: List of audio embedding sizes.
    """
    audio_embed_sizes = []
    for audio in audios:
        audio_data, sf = audio
        audio_frames, _ = compute_logfbank_output_size(len(audio_data), sf)
        audio_embed_size = _compute_audio_embed_size(ctx.get_hf_config(),
                                                     audio_frames)
        audio_embed_sizes.append(audio_embed_size)
    return audio_embed_sizes


def _get_audio_id_to_input_ids(audios, ctx: InputContext, prompt_str=""):
    """
    The following will search for `<|audio_{idx}|>` tokens and
    return a mapping of audio placeholder tokens to audio placeholder token ids
    based on the size of the audio embeddings.

    Args:
        audios (List[Tuple[np.ndarray, int]]): List of audio files as tuples of
            waveform and sample rate.
        ctx (InputContext): Input context.
        prompt_str (str): The prompt string.

    Returns:
        Dict[str, List[int]]: Mapping of audio placeholder tokens to audio 
        placeholder token ids.

    """
    if len(audios) == 0:
        return {}

    audio_embed_sizes = _get_audio_embed_sizes(audios, ctx)
    audio_ids = re.findall(AUDIO_TOKEN_PATTERN, prompt_str)
    audio_ids = [int(audio_id) for audio_id in audio_ids]
    assert len(audio_ids) == len(
        audio_embed_sizes
    ), "Number of audio tokens and audio features do not match"
    assert tuple(audio_ids) == tuple(range(1,
                                           len(audio_ids) +
                                           1)), "Audio ids are not in order!"
    audio_id_to_input_ids = {
        f"<|audio_{audio_id}|>":
        [_AUDIO_PLACEHOLDER_TOKEN_ID] * audio_embed_size
        for audio_id, audio_embed_size in zip(audio_ids, audio_embed_sizes)
    }

    return audio_id_to_input_ids


def _count_image_tokens(images, ctx: InputContext):
    hf_config = ctx.get_hf_config()
    vision_encoder_name = hf_config.img_processor
    if vision_encoder_name is None:
        vision_encoder_name = SIGLIP_NAME
    prepro_config = VISION_ENCODER_TO_PROCESSING_CONFIG[vision_encoder_name]
    dynamic_hd_size = prepro_config['dynamic_hd']
    vit_image_size = prepro_config['vit_image_size']
    vit_patch_size = prepro_config['vit_patch_size']
    token_compression_factor = prepro_config['token_compression_factor']

    image_token_counts = [
        _compute_num_image_tokens(image, dynamic_hd_size, vit_image_size,
                                  vit_patch_size, token_compression_factor)
        for image in images
    ]
    return image_token_counts


def _get_image_id_to_input_ids(images, prompt, ctx: InputContext):
    if len(images) == 0:
        return {}

    image_ids = re.findall(IMAGE_TOKEN_PATTERN, prompt)
    image_ids = [int(image_id) for image_id in image_ids]
    assert len(image_ids) == len(
        set(image_ids)), "Duplicate image tokens in prompt"
    assert len(images) == len(
        image_ids), "Number of images and image tokens in prompt do not match"

    # NOTE the following assertion is not strictly necessary
    assert tuple(image_ids) == tuple(range(1,
                                           len(image_ids) +
                                           1)), "Image ids are not in order"

    image_token_counts = _count_image_tokens(images, ctx)
    image_id_to_input_ids = {
        f"<|image_{image_id}|>": [_IMAGE_PLACEHOLDER_TOKEN_ID] * num_tokens
        for image_id, num_tokens in zip(image_ids, image_token_counts)
    }
    return image_id_to_input_ids


def input_processor_for_phi4mm(ctx: InputContext,
                               inputs: DecoderOnlyInputs) -> TokenInputs:
    """
    Implements the input processor, which transforms the input prompt ids
    to include the audio placeholder token.  This will become the `input_ids`
    in `forward` for the model.

    Args:
        ctx (InputContext): Input context.
        inputs (DecoderOnlyInputs): The inputs (e.g. prompt, prompt_token_ids)
        to process.

    Returns:
        TokenInputs: Processed inputs
    """
    multi_modal_data = inputs.get("multi_modal_data")
    if (multi_modal_data is None or
        ("audio" not in multi_modal_data and "image" not in multi_modal_data)):
        # pure text input, so no need to do pre-processing
        return inputs

    prompt_str = inputs.get("prompt")
    prompt_token_ids = inputs.get("prompt_token_ids")
    # for offline_inference, we will get str input and we parse MM special
    # tokens from it
    # (ignore prompt_token_ids)
    # for OAI server, we will get prompt_token_ids, where MM special tokens
    # are already parsed

    if 'audio' in multi_modal_data:
        audios = multi_modal_data["audio"]

        if not isinstance(audios, list):
            audios = [audios]
        if prompt_str is not None:
            audio_id_to_input_ids = _get_audio_id_to_input_ids(
                audios, ctx, prompt_str=prompt_str)
            audio_embed_sizes = []
        elif prompt_token_ids is not None:
            audio_id_to_input_ids = {}
            audio_embed_sizes = _get_audio_embed_sizes(audios, ctx)
    else:
        audio_id_to_input_ids = {}
        audio_embed_sizes = []

    if 'image' in multi_modal_data:
        # PIL Image or list of PIL Images
        images = multi_modal_data["image"]
        if not isinstance(images, list):
            images = [images]
        if prompt_str is not None:
            image_id_to_input_ids = _get_image_id_to_input_ids(
                images, prompt_str, ctx)
            image_token_counts = []
        elif prompt_token_ids is not None:
            image_id_to_input_ids = {}
            image_token_counts = _count_image_tokens(images, ctx)
    else:
        image_id_to_input_ids = {}
        image_token_counts = []

    # Handle the case where the prompt is a string and we need to manually
    # tokenize it.
    # In this case, the `audio_id_to_input_ids` dict will be mapping from
    # an audio placeholder
    # string (e.g. `<|audio_1|>`) to the audio placeholder tokens for the
    # given audio length.
    if prompt_str:
        pattern = r"(<\|image_\d+\|>|<\|audio_\d+\|>)"
        prompt_chunk_strings = re.split(pattern, prompt_str)
        prompt_chunk_strings = [s for s in prompt_chunk_strings if s != ""]

        # Create the new input_ids with the placeholder image and audio
        # tokens inserted
        tokenizer = cached_tokenizer_from_config(ctx.model_config)
        input_ids = []
        has_imag, has_audio, has_user_text_input = False, False, False
        for prompt_chunk_string in prompt_chunk_strings:
            if re.match(IMAGE_TOKEN_PATTERN, prompt_chunk_string):
                input_ids.extend(image_id_to_input_ids[prompt_chunk_string])
                has_imag = True
            elif re.match(AUDIO_TOKEN_PATTERN, prompt_chunk_string):
                input_ids.extend(audio_id_to_input_ids[prompt_chunk_string])
                has_audio = True
            else:
                curr_token_ids = tokenizer(prompt_chunk_string).input_ids
                if not has_user_text_input:
                    for token_id in curr_token_ids:
                        if token_id not in NON_USER_INPUT_TOKENS:
                            has_user_text_input = True
                            break
                input_ids.extend(curr_token_ids)
        if has_audio and has_imag and has_user_text_input:
            raise ValueError(
                "Phi4MMForCausalLM does not support text + audio + image" +
                " inputs in the same prompt")
    # Handle the case where the prompt is already tokenized
    else:
        assert prompt_token_ids is not None, \
            "If string prompt isn't provided, prompt_token_ids must be"

        i = 0
        input_ids = prompt_token_ids
        # only needed for later assertion
        img_cnt, audio_cnt, user_text_input_cnt = 0, 0, 0
        image_token_count_iter = iter(image_token_counts)
        audio_embed_size_iter = iter(audio_embed_sizes)
        while i < len(input_ids):
            token_id = input_ids[i]
            if token_id == _AUDIO_PLACEHOLDER_TOKEN_ID:
                token_count = next(audio_embed_size_iter)
                audio_cnt += 1
            elif token_id == _IMAGE_PLACEHOLDER_TOKEN_ID:
                token_count = next(image_token_count_iter)
                img_cnt += 1
            else:
                user_text_input_cnt += 1 if token_id not in \
                    NON_USER_INPUT_TOKENS else 0
                i += 1
                continue
            tokens = [token_id] * token_count
            input_ids = input_ids[:i] + tokens + input_ids[i + 1:]
            i += token_count

        if audio_cnt > 0 and img_cnt > 0 and user_text_input_cnt > 0:
            raise ValueError(
                "Phi4MMForCausalLM does not support text + audio + image" +
                " inputs in the same prompt")
        # If the below assertion fails, it might be that input pure-text
        # messages contain image/audio special tokens literally
        # (<|endoftext10|>, <|endoftext11|>).
        assert (img_cnt == len(image_token_counts)), (
            f"Number of image tokens in prompt_token_ids ({img_cnt}) "
            f"does not match number of images ({len(image_token_counts)})")
        assert (audio_cnt == len(audio_embed_sizes)), (
            f"Number of audio tokens in prompt_token_ids ({audio_cnt}) "
            f"does not match number of audios ({len(audio_embed_sizes)})")

    # NOTE: Create a defensive copy of the original inputs
    return token_inputs(
        prompt_token_ids=input_ids,
        prompt=prompt_str,
        multi_modal_data=multi_modal_data,
    )


def _compute_audio_embed_size(hf_config, audio_frames):
    """
    Compute the audio embedding size based on the audio frames and
    compression rate.
    """
    compression_rate = hf_config.embd_layer['audio_embd_layer'][
        'compression_rate']
    # NOTE: this is a hard-coded value but might be configurable in the future
    qformer_compression_rate = 1
    integer = audio_frames // compression_rate
    remainder = audio_frames % compression_rate

    result = integer if remainder == 0 else integer + 1

    integer = result // qformer_compression_rate
    remainder = result % qformer_compression_rate
    result = integer if remainder == 0 else integer + 1  # qformer compression

    return result


def get_max_phi4mm_audio_tokens(ctx: InputContext) -> int:
    return 10000


def dummy_audio_for_phi4mm(audio_count: int) -> dict:
    """
    Create dummy audio data for the Phi4MM model, which is used for profiling.

    Args:
        audio_count (int): Number of audio samples.

    Returns:
        dict: Dummy audio data.
    """
    dummy_audio = np.full((_AUDIO_MAX_SOUNDFILE_SIZE, ), 0.0)
    return [(dummy_audio, DUMMY_SAMPLING_FREQUENCY)] * audio_count


def dummy_image_for_phi4mm(width: int, height: int):
    image = Image.new('RGB', (width, height), color='black')
    return image


def dummy_data_for_phi4mm(ctx: InputContext, seq_len: int,
                          mm_counts: Mapping[str, int]) -> DummyData:
    """
    Create dummy sequence (input_ids) and audio data for the Phi4MM model, 
    which is used for profiling.

    In this case, the sequence data is a bunch of 0s with a number of audio 
    tokens that correspond to the audio embed size of the 
    _AUDIO_MAX_SOUNDFILE_SIZE.

    Args:
        ctx (InputContext): Input context.
        seq_len (int): Length of the sequence.
        mm_counts (Mapping[str, int]): Multi-modal counts.

    Returns:
        Tuple: Dummy sequence data and dummy audio data.
    """
    audio_count = mm_counts["audio"]
    audio_frames, _ = compute_logfbank_output_size(_AUDIO_MAX_SOUNDFILE_SIZE,
                                                   DUMMY_SAMPLING_FREQUENCY)
    audio_feature_size = _compute_audio_embed_size(ctx.get_hf_config(),
                                                   audio_frames)

    image_count = mm_counts["image"]
    dummy_image = get_max_dummy_image(ctx)
    max_image_tokens = get_max_phi4mm_image_tokens(ctx)
    total_image_tokens = image_count * max_image_tokens

    if seq_len - audio_feature_size * audio_count - total_image_tokens < 0:
        raise RuntimeError(
            f"Phi4MM cannot process {audio_count} audios and {image_count}"
            f"images in a prompt, please increase max_model_len to be at"
            f" larger than "
            f"{audio_feature_size * audio_count + total_image_tokens}"
            " or reduce audio/image limit by --limit-mm-per-prompt.")

    if audio_feature_size * audio_count > total_image_tokens:
        seq_data = SequenceData.from_prompt_token_counts(
            (_AUDIO_PLACEHOLDER_TOKEN_ID, audio_feature_size * audio_count),
            (0, seq_len - audio_feature_size * audio_count),
        )
        mm_data = {
            "audio": dummy_audio_for_phi4mm(audio_count),
        }
    else:
        seq_data = SequenceData.from_prompt_token_counts(
            (_IMAGE_PLACEHOLDER_TOKEN_ID, total_image_tokens),
            (0, seq_len - total_image_tokens),
        )
        mm_data = {
            "image": [dummy_image] * image_count,
        }
    return DummyData(seq_data, mm_data)


def input_mapper_for_phi4mm_audio(ctx: InputContext,
                                  data: object) -> MultiModalInputs:
    """
    This function is used to create the MultiModalInputs for the Phi4MM 
    (audio) model.
    Specifically, for audio, we extract the audio features from the sound 
    file and create pairs of audio features and audio embed lengths (the
    latter of which is used to repeat the audio placeholder token in the 
    input prompt IDs).
    These pairs are used, downstream, in `_audio_features_to_embeddings`
    (via `_process_audio_input`).

    Note that the incoming audio data (each entry in `data`) is a tuple of 
    the audio data and the sampling frequency (e.g. from soundfile.read).

    Args:
        ctx (InputContext): Input context.
        data (object): Audio data.

    Returns:
        MultiModalInputs: Multi-modal inputs.
    """
    if not isinstance(data, list):
        data = [data]

    if len(data) == 0:
        return MultiModalInputs()

    audio_features = []
    for audio_input in data:
        if not isinstance(audio_input, tuple):
            raise NotImplementedError(
                f"Unsupported data type: {type(audio_input)}")

        audio, sf = audio_input
        feature_extractor = audio_feature_extractor()
        single_audio_features = feature_extractor.extract_features(audio, sf)
        feat_stride = (1 if not hasattr(feature_extractor, "stride") else
                       feature_extractor.stride)
        audio_frames = len(single_audio_features) * feat_stride
        single_audio_embed_size = _compute_audio_embed_size(
            ctx.get_hf_config(), audio_frames)
        single_audio_feature_audio_len_pair = (
            single_audio_features,
            [single_audio_embed_size],
        )
        audio_features.append(single_audio_feature_audio_len_pair)
    return MultiModalInputs({"audio_features": audio_features})


def input_mapper_for_phi4mm_image(ctx: InputContext, data: object):
    if not isinstance(data, list):
        data = [data]
    # data: list of PIL images
    if len(data) == 0:
        return MultiModalInputs()
    hf_config = ctx.get_hf_config()
    vision_encoder_name = hf_config.img_processor
    if vision_encoder_name is None:
        vision_encoder_name = SIGLIP_NAME
    prepro_config = VISION_ENCODER_TO_PROCESSING_CONFIG[vision_encoder_name]
    dynamic_hd_size = prepro_config['dynamic_hd']
    vit_image_size = prepro_config['vit_image_size']
    vit_patch_size = prepro_config['vit_patch_size']

    image_input_dict = preprocess(data, dynamic_hd_size, vit_image_size,
                                  vit_patch_size)
    return MultiModalInputs({
        "pixel_values":
        image_input_dict["pixel_values"],
        "image_sizes":
        image_input_dict["image_sizes"],
        "image_attention_mask":
        image_input_dict["image_attention_mask"],
        "num_img_tokens":
        image_input_dict["num_img_tokens"],
    })


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


@MULTIMODAL_REGISTRY.register_input_mapper("audio",
                                           input_mapper_for_phi4mm_audio)
@MULTIMODAL_REGISTRY.register_input_mapper("image",
                                           input_mapper_for_phi4mm_image)
@MULTIMODAL_REGISTRY.register_max_multimodal_tokens(
    "audio", get_max_phi4mm_audio_tokens)
@MULTIMODAL_REGISTRY.register_max_multimodal_tokens(
    "image", get_max_phi4mm_image_tokens)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_phi4mm)
@INPUT_REGISTRY.register_input_processor(input_processor_for_phi4mm)
class Phi4MMForCausalLM(nn.Module, SupportsLoRA, SupportsMultiModal):
    """
    Implements the Phi-4-multimodal-instruct model in VLLM.
    """
    packed_modules_mapping = {
        "qkv_proj": [
            "qkv_proj",
        ],
        "gate_up_proj": [
            "gate_up_proj",
        ],
    }

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
        assert get_tensor_model_parallel_world_size(
        ) == 1, "tensor parallel is not supported"
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
            padding_size=(
                DEFAULT_VOCAB_PADDING_SIZE
                # We need bigger padding if using lora for kernel
                # compatibility
                if not lora_config else lora_config.lora_vocab_padding_size),
            quant_config=quant_config,
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size, logit_scale)
        self.sampler = Sampler()

    def _audio_features_to_embeddings(
        self,
        input_ids: torch.Tensor,
        input_features: List[torch.Tensor],
        audio_input_sizes: torch.Tensor,
        audio_projection_mode: str,
    ) -> torch.Tensor:
        """
        Convert audio features to embeddings, which are used as input to the 
        model (via `inputs_embeds`).

        Args:
            input_ids (torch.Tensor): Input IDs (the prompt in this case).
            input_features (list[torch.Tensor]): Input features (the audio 
            embeddings).
            audio_input_sizes (list[torch.Tensor]): Audio input sizes (the 
            audio embed lengths to use for padding the audio placeholder token 
            in the input prompt IDs).
        """
        # The audio projection can either be a single linear or Sequential,
        # so handle both cases
        if isinstance(self.embed_tokens_extend.audio_projection,
                      nn.Sequential):
            target_dtype = self.embed_tokens_extend.audio_projection[
                0].bias.dtype
        else:
            target_dtype = self.embed_tokens_extend.audio_projection.bias.dtype

        audio_input = [
            input.unsqueeze(0).to(target_dtype) for input in input_features
        ]
        kwargs = {
            "wte": self.model.embed_tokens,
            'audio_projection_mode': audio_projection_mode
        }
        audio_embeddings = self.embed_tokens_extend(input_ids, audio_input,
                                                    audio_input_sizes,
                                                    **kwargs)
        audio_embeddings = audio_embeddings.to(target_dtype)
        return audio_embeddings

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
        audio_features = kwargs.pop("audio_features", None)
        audio_embeds = kwargs.pop("audio_embeds", None)

        if audio_features is None and audio_embeds is None:
            return None

        if audio_features is not None:
            if not isinstance(audio_features, (torch.Tensor, list)):
                raise ValueError("Incorrect type of audio features. "
                                 f"Got type: {type(audio_features)}")

            return Phi4MMAudioFeatureInputs(type="audio_features",
                                            data=audio_features)

        if audio_embeds is not None:
            if not isinstance(audio_embeds, (torch.Tensor, list)):
                raise ValueError("Incorrect type of audio embeds. "
                                 f"Got type: {type(audio_embeds)}")

            return Phi4MMAudioEmbeddingInputs(type="audio_embeds",
                                              data=audio_embeds)

        raise AssertionError("This line should be unreachable.")

    def _process_audio_input(self, input_ids: torch.Tensor,
                             audio_input: Phi4MMAudioInputs,
                             audio_projection_mode: str) -> NestedTensors:
        """
        Create the audio embeddings from the audio input, where the audio input
        is pairs of audio features and audio embed lengths.  The audio input is
        created by `input_mapper_for_phi4mm_audio`.

        Args:
            input_ids (torch.Tensor): Input IDs (the prompt in this case, 
            before the audio token replication).
            audio_input (Phi4MMAudioInputs): Audio input.

        Returns:
            NestedTensors: Audio embeddings
        """
        if audio_input["type"] == "audio_embeds":
            return audio_input["data"]

        audio_features = audio_input["data"]
        # (e.g. multiple examples) and the second dim is the multi-audio dim
        # (e.g. multiple audios in the same example)
        audio_feature = [i[0] for j in audio_features for i in j]
        audio_feature_len = [i[1].item() for j in audio_features for i in j]
        # Add the batch dim via `squeeze`

        return self._audio_features_to_embeddings(
            input_ids.unsqueeze(0),
            audio_feature,
            audio_feature_len,
            audio_projection_mode,
        ).squeeze(0)

    def _parse_and_validate_image_input(self,
                                        **kwargs: object) -> Optional[Dict]:
        pixel_values: Optional[Dict] = kwargs.get("pixel_values")
        if pixel_values is None:
            return None

        image_sizes = kwargs.get("image_sizes")
        image_attention_mask = kwargs.get("image_attention_mask")
        num_img_tokens = kwargs.get("num_img_tokens")
        assert image_sizes is not None and image_attention_mask is not None\
              and num_img_tokens is not None, "Missing image inputs"

        if isinstance(pixel_values, list):
            assert pixel_values[0].dim() == 5, "Incorrect image inputs"
            # list len is batch_size.
            # each tensor has dimension: num_img_per_example, num_hd_patches,
            # channels, height, width.
            # need to pad along num_hd_patches.
            # mask size num_img_per_prompt, num_hd_patches, feat_h, heat_w.
            pixel_values = cat_with_pad(pixel_values, dim=0)
        elif isinstance(pixel_values, torch.Tensor):
            # dimension: batch_size, num_img_per_example, num_hd_patches,
            # channels, height, width.
            # we flatten first 2 dims to make it a single large batch for
            # SigLIP Encoder.
            assert pixel_values.dim() == 6, "Incorrect image inputs"
            pixel_values = pixel_values.flatten(0, 1)
        else:
            raise ValueError("Incorrect pixel_values inputs")

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

        return {
            'pixel_values': pixel_values,
            'image_sizes': image_sizes,
            'image_attention_mask': image_attention_mask,
            'num_img_tokens': num_img_tokens,
        }

    def merge_image_features_to_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_set_tensors: List[torch.Tensor],
    ):
        position_tuple = (input_ids == _IMAGE_PLACEHOLDER_TOKEN_ID).nonzero(
            as_tuple=True)

        assert all([t.shape[0] == 1 for t in image_set_tensors
                    ]), 'img_set_tensor should have shape (1, N_tokens, C)'
        # Shape: (merged_N_tokens, C)
        image_set_tensor = torch.cat(image_set_tensors, dim=1).squeeze(0)
        image_set_tensor = image_set_tensor.to(inputs_embeds.dtype).to(
            inputs_embeds.device)
        merged_embeds = inputs_embeds.index_put(
            indices=position_tuple,
            values=image_set_tensor,
            accumulate=False,
        )
        return merged_embeds

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> None:
        weights = {name: weight for name, weight in weights}
        adjusted_weights = {}

        for name, weight in weights.items():
            # NOTE vision-speech tasks use a separate projection layer
            audio_proj_4v = \
                "model.embed_tokens_extend.audio_embed.audio_projection.vision"
            if name.startswith(audio_proj_4v):
                name = name.replace(
                    audio_proj_4v,
                    "embed_tokens_extend.audio_projection_for_vision")

            name = (name.replace(
                "model.embed_tokens_extend.audio_embed."\
                    "audio_projection.speech.",
                "embed_tokens_extend.audio_projection.",
            ).replace(
                "model.embed_tokens_extend.audio_embed.",
                "embed_tokens_extend.",
            ).replace("model.embed_tokens_extend.image_embed.",
                      "vision_encoder."))
            # NOTE: this is deal with LoRA injection, where `base_layer`
            # remains as the original layer in the model
            if name.endswith(".base_layer.weight"):
                name = name.replace(".base_layer.weight", ".weight")
            adjusted_weights[name] = weight

        missing_keys, unexpected_keys = self.load_state_dict(adjusted_weights,
                                                             strict=False)
        logger.debug("*** missing keys:")
        for key in missing_keys:
            logger.debug(key)
        logger.debug("**** unexpected keys:")
        for key in unexpected_keys:
            logger.debug(key)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            input_ids = None
            inputs_embeds = None
        else:
            # Each entry in this is a pair of audio_features and audio_embed
            # lengths
            audio_input = self._parse_and_validate_audio_input(**kwargs)
            image_inputs = self._parse_and_validate_image_input(**kwargs)

            has_audio = audio_input is not None
            has_image = image_inputs is not None

            if has_audio:
                audio_projection_mode = 'vision' if has_image else 'speech'
                inputs_embeds = self._process_audio_input(
                    input_ids, audio_input, audio_projection_mode)

            if has_image:
                dtype = self.vision_encoder.img_processor.embeddings.\
                    patch_embedding.weight.dtype
                pixel_values = image_inputs['pixel_values'].to(dtype)
                image_sizes = image_inputs['image_sizes']
                image_attention_mask = image_inputs['image_attention_mask']
                image_set_tensors = self.vision_encoder(
                    pixel_values, image_sizes, image_attention_mask)
                if not has_audio:
                    inputs_embeds = self.model.embed_tokens(input_ids)

                inputs_embeds = self.merge_image_features_to_inputs_embeds(
                    input_ids, inputs_embeds, image_set_tensors)

            if has_image or has_audio:
                # multi-modal input, we have set inputs_embeds properly in
                # previous steps
                input_ids = None
            else:
                # text-only, we keep using original input_ids
                inputs_embeds = None

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

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="model.",
            connector=["audio_projection_for_vision", "audio_projection"],
            tower_model=["vision_encoder", "embed_tokens_extend"],
        )