# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property
from itertools import product
from math import ceil, sqrt
from typing import Any, Literal, Optional, TypedDict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import BatchFeature, PretrainedConfig, TensorType

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs, NestedTensors)
from vllm.multimodal.parse import ImageSize, MultiModalDataItems
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate, PromptUpdateDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs import Step3VisionEncoderConfig
from vllm.transformers_utils.tokenizer import AnyTokenizer

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .utils import (AutoWeightsLoader, WeightsMapper, flatten_bn,
                    init_vllm_registered_model, maybe_prefix,
                    merge_multimodal_embeddings)


class Step3VLImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: torch.Tensor
    patch_pixel_values: Optional[torch.Tensor]
    num_patches: list[int]


class Step3VLImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    image_embeds: torch.Tensor


Step3VLImageInputs = Union[Step3VLImagePixelInputs,
                           Step3VLImageEmbeddingInputs]

ImageWithPatches = tuple[Image.Image, list[Image.Image], list[int] | None]

MAX_IMAGE_SIZE: int = 3024


class Step3VisionProcessor:

    def __init__(self, size, interpolation_mode="bicubic", patch_size=None):
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        patch_size = patch_size if patch_size is not None else size

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.Resize(
                (size, size),
                interpolation=InterpolationMode.BICUBIC if interpolation_mode
                == "bicubic" else InterpolationMode.BILINEAR,
                antialias=True),
        ])

        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.Resize(
                (patch_size, patch_size),
                interpolation=InterpolationMode.BICUBIC if interpolation_mode
                == "bicubic" else InterpolationMode.BILINEAR,
                antialias=True),
        ]) if patch_size is not None else None

    def __call__(self, image, is_patch=False):
        if is_patch:
            return {"pixel_values": self.patch_transform(image).unsqueeze(0)}
        else:
            return {"pixel_values": self.transform(image).unsqueeze(0)}


class ImagePatcher:

    def determine_window_size(self, long: int, short: int) -> int:
        if long <= 728:
            return short if long / short > 1.5 else 0
        return min(short, 504) if long / short > 4 else 504

    def slide_window(
        self,
        width: int,
        height: int,
        sizes: list[tuple[int, int]],
        steps: list[tuple[int, int]],
        img_rate_thr: float = 0.6,
    ) -> tuple[list[tuple[int, int, int, int]], tuple[int, int]]:
        assert 1 >= img_rate_thr >= 0, "The `in_rate_thr` should lie in 0~1"
        windows = []
        # Sliding windows.
        for size, step in zip(sizes, steps):
            size_w, size_h = size
            step_w, step_h = step

            x_num = 1 if width <= size_w else ceil((width - size_w) / step_w +
                                                   1)
            x_start = [step_w * i for i in range(x_num)]
            if len(x_start) > 1 and x_start[-1] + size_w > width:
                x_start[-1] = width - size_w

            y_num = 1 if height <= size_h else ceil((height - size_h) /
                                                    step_h + 1)
            y_start = [step_h * i for i in range(y_num)]
            if len(y_start) > 1 and y_start[-1] + size_h > height:
                y_start[-1] = height - size_h

            start = np.array(list(product(y_start, x_start)), dtype=int)
            start[:, [0, 1]] = start[:, [1, 0]]
            windows.append(np.concatenate([start, start + size], axis=1))
        windows = np.concatenate(windows, axis=0)

        return [(int(box[0]), int(box[1]), int(box[2] - box[0]),
                 int(box[3] - box[1])) for box in windows], (x_num, y_num)

    def square_pad(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        size = max(w, h)
        padded = Image.new(img.mode, (size, size), 0)
        padded.paste(img, (0, 0))
        return padded

    def get_image_size_for_padding(self, img_width: int,
                                   img_height: int) -> tuple[int, int]:
        ratio = img_width / img_height
        if min(img_height, img_width) < 32 and (ratio > 4 or ratio < 1 / 4):
            new_size = max(img_height, img_width)
            return new_size, new_size
        return img_width, img_height

    def get_image_size_for_preprocess(self, img_width: int,
                                      img_height: int) -> tuple[int, int]:

        if max(img_height, img_width) > MAX_IMAGE_SIZE:
            scale_factor = MAX_IMAGE_SIZE / max(img_height, img_width)
            img_width = int(img_width * scale_factor)
            img_height = int(img_height * scale_factor)
        return img_width, img_height

    def get_image_size_for_crop(self, img_width: int, img_height: int,
                                window_size: int):
        w_ratio = img_width / window_size
        h_ratio = img_height / window_size

        if w_ratio < 1:
            width_new = img_width
        else:
            decimal_w = w_ratio - img_width // window_size
            w_ratio = int(w_ratio) + 1 if decimal_w > 0.2 else int(w_ratio)
            width_new = window_size * w_ratio
        if h_ratio < 1:
            height_new = img_height
        else:
            decimal_h = h_ratio - img_height // window_size
            h_ratio = int(h_ratio) + 1 if decimal_h > 0.2 else int(h_ratio)
            height_new = window_size * h_ratio
        return int(width_new), int(height_new)

    def patch_crop(self, img: Image.Image, i: int, j: int, th: int, tw: int):
        target = img.crop((j, i, j + tw, i + th))
        return target

    def get_num_patches(self, img_width: int,
                        img_height: int) -> tuple[int, int]:
        img_width, img_height = self.get_image_size_for_padding(
            img_width, img_height)
        img_width, img_height = self.get_image_size_for_preprocess(
            img_width, img_height)
        window_size = self.determine_window_size(max(img_height, img_width),
                                                 min(img_height, img_width))
        if window_size == 0:
            return 0, 0
        else:
            img_width, img_height = self.get_image_size_for_crop(
                img_width, img_height, window_size)
            center_list, (x_num, y_num) = self.slide_window(
                img_width, img_height, [(window_size, window_size)],
                [(window_size, window_size)])
            full_rows = (len(center_list) - 1) // x_num + 1
            if len(center_list) > 0 and len(center_list) % x_num == 0:
                full_rows -= 1
            return len(center_list), full_rows

    def __call__(
        self, img: Image.Image
    ) -> tuple[Image.Image, list[Image.Image], list[bool] | None]:
        img_width, img_height = img.size
        new_img_width, new_img_height = self.get_image_size_for_padding(
            img_width, img_height)
        if new_img_width != img_width or new_img_height != img_height:
            img = self.square_pad(img)
            img_width, img_height = img.size

        new_img_width, new_img_height = self.get_image_size_for_preprocess(
            img_width, img_height)
        img = img.resize((new_img_width, new_img_height),
                         Image.Resampling.BILINEAR)
        window_size = self.determine_window_size(
            max(new_img_height, new_img_width),
            min(new_img_height, new_img_width))

        if window_size == 0:
            return img, [], None
        else:
            new_img_width, new_img_height = self.get_image_size_for_crop(
                new_img_width, new_img_height, window_size)
            if (new_img_width, new_img_height) != (img_width, img_height):
                img_for_crop = img.resize((new_img_width, new_img_height),
                                          Image.Resampling.BILINEAR)
            else:
                img_for_crop = img

            patches = []
            newlines = []
            center_list, (x_num, y_num) = self.slide_window(
                new_img_width, new_img_height, [(window_size, window_size)],
                [(window_size, window_size)])
            for patch_id, center_lf_point in enumerate(center_list):
                x, y, patch_w, patch_h = center_lf_point
                big_patch = self.patch_crop(img_for_crop, y, x, patch_h,
                                            patch_w)
                patches.append(big_patch)
                if (patch_id + 1) % x_num == 0:
                    newlines.append(patch_id)

            if newlines and newlines[-1] == len(patches) - 1:
                newlines.pop()

            return img, patches, [i in newlines for i in range(len(patches))
                                  ] if len(patches) > 0 else None


class Step3VLProcessor:

    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: AnyTokenizer,
    ) -> None:
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer

        self.image_size = 728
        self.patch_size = 504
        self.image_preprocessor = Step3VisionProcessor(self.image_size,
                                                       "bilinear",
                                                       self.patch_size)

        self.num_image_feature_size = 169
        self.num_patch_feature_size = 81
        self.image_token = "<im_patch>"
        self.image_feature_placeholder = (self.image_token *
                                          self.num_image_feature_size)
        self.patch_feature_placeholder = (self.image_token *
                                          self.num_patch_feature_size)

        self.patcher = ImagePatcher()

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.get_vocab()[self.image_token]

    def get_num_image_tokens(self, img_width: int, img_height: int) -> int:
        num_patches, num_newlines = self.patcher.get_num_patches(
            img_width, img_height)

        return num_patches * (
            self.num_patch_feature_size +
            2) + self.num_image_feature_size + 2 + num_newlines

    def _split_images(self,
                      images: list[Image.Image]) -> list[ImageWithPatches]:
        result = []
        for img in images:
            result.append(self.patcher(img))
        return result

    def _convert_images_to_pixel_values(
        self,
        images: list[Image.Image],
        is_patch: bool = False,
    ) -> list[torch.Tensor]:
        return [
            self.image_preprocessor(img, is_patch=is_patch)["pixel_values"]
            for img in images
        ]

    def _get_patch_repl(
        self,
        num_patches: int,
        patch_newline_mask: list[bool] | None,
    ) -> tuple[str, list[int]]:
        text = ""
        token_ids = []
        for i in range(num_patches):
            assert len(patch_newline_mask) == num_patches
            text += f"<patch_start>{self.patch_feature_placeholder}<patch_end>"
            token_ids.extend(
                [self.tokenizer.convert_tokens_to_ids("<patch_start>")] +
                [self.image_token_id] * self.num_patch_feature_size +
                [self.tokenizer.convert_tokens_to_ids("<patch_end>")])
            if patch_newline_mask and patch_newline_mask[i]:
                text += "<patch_newline>"
                token_ids.append(
                    self.tokenizer.convert_tokens_to_ids("<patch_newline>"))
        return text, token_ids

    def _get_image_repl(
        self,
        num_images: int,
    ) -> tuple[str, list[int]]:
        text = f"<im_start>{self.image_feature_placeholder}<im_end>"
        token_ids = [
            self.tokenizer.convert_tokens_to_ids("<im_start>")
        ] + [self.image_token_id] * self.num_image_feature_size + [
            self.tokenizer.convert_tokens_to_ids("<im_end>")
        ]
        return text * num_images, token_ids * num_images

    def _get_image_repl_features(
        self,
        num_images: int,
        num_patches: int,
        patch_new_line_idx: Optional[list[bool]],
    ) -> tuple[str, list[int]]:
        if num_patches > 0:
            patch_repl, patch_repl_ids = self._get_patch_repl(
                num_patches, patch_new_line_idx)
        else:
            patch_repl = ""
            patch_repl_ids = []
        image_repl, image_repl_ids = self._get_image_repl(num_images)
        return patch_repl + image_repl, patch_repl_ids + image_repl_ids

    def replace_placeholder(self, text: str, placeholder: str,
                            repls: list[str]) -> str:
        parts = text.split(placeholder)

        if len(parts) - 1 != len(repls):
            raise ValueError(
                "The number of placeholders does not match the number of replacements."  # noqa: E501
            )

        result = [parts[0]]
        for i, repl in enumerate(repls):
            result.append(repl)
            result.append(parts[i + 1])

        return "".join(result)

    def __call__(
        self,
        text: Optional[Union[str, list[str]]] = None,
        images: Optional[Union[Image.Image, list[Image.Image]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchFeature:
        if text is None:
            text = []
        if not isinstance(text, list):
            text = [text]
        if images is None:
            images = []
        if not isinstance(images, list):
            images = [images]

        if len(images) == 0:
            image_inputs = {}
            text_inputs = self.tokenizer(text)
        else:
            splitted_images_data = self._split_images(images)
            pixel_values_lst = []
            patch_pixel_values_lst = []
            patch_newline_mask_lst = []
            image_repl_str_lst = []
            image_repl_ids_lst = []
            num_patches = []
            for raw_img, img_patches, patch_newline_mask in splitted_images_data:  # noqa: E501
                pixel_values_lst.extend(
                    self._convert_images_to_pixel_values([raw_img]))

                if len(img_patches) > 0:
                    patch_pixel_values_lst.extend(
                        self._convert_images_to_pixel_values(img_patches,
                                                             is_patch=True))
                num_patches.append(len(img_patches))

                image_repl_str, image_repl_ids = self._get_image_repl_features(
                    1, len(img_patches), patch_newline_mask)
                image_repl_str_lst.append(image_repl_str)
                image_repl_ids_lst.extend(image_repl_ids)

                if patch_newline_mask is not None:
                    patch_newline_mask_lst.extend(patch_newline_mask)

            image_inputs = {
                "pixel_values": torch.cat(pixel_values_lst),
                "num_patches": num_patches,
            }
            if patch_pixel_values_lst:
                image_inputs["patch_pixel_values"] = torch.cat(
                    patch_pixel_values_lst)
            if patch_newline_mask_lst:
                image_inputs["patch_newline_mask"] = torch.tensor(
                    patch_newline_mask_lst, dtype=torch.bool)

            text = [
                self.replace_placeholder(t, self.image_token,
                                         image_repl_str_lst) for t in text
            ]
            text_inputs = self.tokenizer(text)

        return BatchFeature(
            {
                **text_inputs,
                **image_inputs,
            },
            tensor_type=return_tensors,
        )


class Step3VLProcessingInfo(BaseProcessingInfo):

    def get_hf_processor(self) -> Step3VLProcessor:
        return Step3VLProcessor(
            self.get_hf_config(),
            self.get_tokenizer(),
        )

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_max_image_tokens(self) -> int:
        hf_processor = self.get_hf_processor()
        return hf_processor.get_num_image_tokens(
            self.get_image_size_with_most_features().width,
            self.get_image_size_with_most_features().height)

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {"image": self.get_max_image_tokens()}

    def get_image_size_with_most_features(self) -> ImageSize:
        return ImageSize(3024, 3024)

    def get_num_mm_tokens(self, mm_data: MultiModalDataDict) -> int:
        if len(mm_data) != 1 or "image" not in mm_data:
            raise ValueError(
                "mm_data could only contain one key 'image' for steo1o")

        image_data = mm_data["image"]
        if not isinstance(image_data, (list, tuple)):
            image_data = [image_data]

        return sum(self.get_hf_processor().get_num_image_tokens(
            img.width, img.height) for img in image_data)


class Step3VLDummyInputsBuilder(BaseDummyInputsBuilder[Step3VLProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        return "<im_patch>" * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        target_width, target_height = \
            self.info.get_image_size_with_most_features()
        num_images = mm_counts.get("image", 0)

        return {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images)
        }


class Step3VLMultiModalProcessor(BaseMultiModalProcessor[Step3VLProcessingInfo]
                                 ):

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_placeholder_token_id = hf_processor.image_token_id
        batch_num_patches = out_mm_kwargs["num_patches"].tolist()

        def get_replacement_step1o(item_idx: int):
            img_out = out_mm_kwargs.get_item("image", item_idx)
            num_patches = batch_num_patches[item_idx]
            if num_patches > 0:
                patch_newline_mask = img_out["patch_newline_mask"].data.tolist(
                )
                image_repl_ids = hf_processor._get_image_repl_features(
                    1, num_patches, patch_newline_mask)[1]
            else:
                image_repl_ids = hf_processor._get_image_repl_features(
                    1, 0, None)[1]
            return PromptUpdateDetails.select_token_id(
                seq=image_repl_ids,
                embed_token_id=image_placeholder_token_id,
            )

        return [
            PromptReplacement(
                modality="image",
                target=[image_placeholder_token_id],
                replacement=get_replacement_step1o,
            )
        ]

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        num_patches = hf_inputs.get("num_patches", torch.empty(0))

        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            patch_pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "image", num_patches),
            num_patches=MultiModalFieldConfig.batched("image"),
            patch_newline_mask=MultiModalFieldConfig.flat_from_sizes(
                "image", num_patches),
        )


def get_abs_pos(abs_pos, tgt_size):
    dim = abs_pos.size(-1)
    abs_pos_new = abs_pos.squeeze(0)
    cls_token, old_pos_embed = abs_pos_new[:1], abs_pos_new[1:]

    src_size = int(math.sqrt(abs_pos_new.shape[0] - 1))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size != tgt_size:
        old_pos_embed = old_pos_embed.view(1, src_size, src_size,
                                           dim).permute(0, 3, 1,
                                                        2).contiguous()
        old_pos_embed = old_pos_embed.to(torch.float32)
        new_pos_embed = F.interpolate(
            old_pos_embed,
            size=(tgt_size, tgt_size),
            mode='bicubic',
            antialias=True,
            align_corners=False,
        ).to(dtype)
        new_pos_embed = new_pos_embed.permute(0, 2, 3, 1)
        new_pos_embed = new_pos_embed.view(tgt_size * tgt_size, dim)
        vision_pos_embed = torch.cat([cls_token, new_pos_embed], dim=0)
        vision_pos_embed = vision_pos_embed.view(1, tgt_size * tgt_size + 1,
                                                 dim)
        return vision_pos_embed
    else:
        return abs_pos


class Step3VisionEmbeddings(nn.Module):

    def __init__(self, config: Step3VisionEncoderConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(1, self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )

        self.num_patches = (self.image_size // self.patch_size)**2
        self.pad_tp_size = 4  # hard code for padding
        # To load the pretrained weights, we still use P+1 as the seqlen
        self.position_embedding = torch.nn.Embedding(self.num_patches + 1,
                                                     self.embed_dim)
        self.register_buffer("position_ids",
                             torch.arange(self.num_patches + 1).expand(
                                 (1, -1)),
                             persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(
            pixel_values)  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        # pad
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + get_abs_pos(
            self.position_embedding(self.position_ids), patch_embeds.size(1))
        embeddings = torch.cat([
            embeddings[:, 0, :].unsqueeze(1).repeat(1, self.pad_tp_size - 1,
                                                    1), embeddings
        ],
                               dim=1)
        return embeddings


class Step3VisionAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self,
                 config,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.total_num_heads

        self.scale = self.head_dim**-0.5

        tp_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.qkv_proj = QKVParallelLinear(self.embed_dim,
                                          self.head_dim,
                                          self.total_num_heads,
                                          bias=True,
                                          quant_config=quant_config,
                                          prefix=prefix)
        self.out_proj = RowParallelLinear(self.embed_dim,
                                          self.embed_dim,
                                          bias=True,
                                          quant_config=quant_config,
                                          prefix=prefix)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads,
                           self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim)
        k = k.view(bsz, tgt_len, self.num_heads, self.head_dim)
        v = v.view(bsz, tgt_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_output = F.scaled_dot_product_attention(q,
                                                     k,
                                                     v,
                                                     scale=self.scale,
                                                     is_causal=False)
        attn_output = attn_output.transpose(1, 2).reshape(
            bsz, tgt_len, self.num_heads * self.head_dim)

        attn_output, _ = self.out_proj(attn_output)

        return attn_output


class Step3VisionMLP(nn.Module):

    def __init__(self,
                 config,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.config = config
        self.activation_fn = get_act_fn(config.hidden_act)
        self.fc1 = ColumnParallelLinear(config.hidden_size,
                                        config.intermediate_size,
                                        bias=True,
                                        quant_config=quant_config,
                                        prefix=prefix)
        self.fc2 = RowParallelLinear(config.intermediate_size,
                                     config.hidden_size,
                                     bias=True,
                                     quant_config=quant_config,
                                     prefix=prefix)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class Step3VisionEncoderLayer(nn.Module):

    def __init__(self,
                 config: Step3VisionEncoderConfig,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Step3VisionAttention(config,
                                              quant_config,
                                              prefix=f"{prefix}.self_attn")
        self.layer_norm1 = nn.LayerNorm(self.embed_dim,
                                        eps=config.layer_norm_eps)
        self.mlp = Step3VisionMLP(config, quant_config, prefix=f"{prefix}.mlp")
        self.layer_norm2 = nn.LayerNorm(self.embed_dim,
                                        eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.FloatTensor:
        hidden_states = hidden_states + self.layer_norm1(
            self.self_attn(hidden_states))
        hidden_states = hidden_states + self.layer_norm2(
            self.mlp(hidden_states))
        return hidden_states


class Step3VisionEncoder(nn.Module):

    def __init__(self,
                 config: Step3VisionEncoderConfig,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            Step3VisionEncoderLayer(config,
                                    quant_config,
                                    prefix=f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        inputs_embeds,
    ):
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
        return hidden_states


class Step3VisionTransformer(nn.Module):

    def __init__(self,
                 config: Step3VisionEncoderConfig,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.config = config
        self.image_size = config.image_size
        self.embeddings = Step3VisionEmbeddings(config)
        self.transformer = Step3VisionEncoder(config,
                                              quant_config,
                                              prefix=f"{prefix}.transformer")

    def forward(
        self,
        pixel_values: torch.Tensor,
    ):
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.transformer(inputs_embeds=hidden_states)
        return hidden_states


@MULTIMODAL_REGISTRY.register_processor(Step3VLMultiModalProcessor,
                                        info=Step3VLProcessingInfo,
                                        dummy_inputs=Step3VLDummyInputsBuilder)
class Step3VLForConditionalGeneration(nn.Module, SupportsMultiModal,
                                      SupportsPP):

    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
        "model.": "language_model.model.",
        "lm_head.": "language_model.lm_head.",
    })

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("image"):
            return "<im_patch>"

        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        if multimodal_config.get_limit_per_prompt("image"):
            self.vision_model = Step3VisionTransformer(config.vision_config,
                                                       None,
                                                       prefix=maybe_prefix(
                                                           prefix,
                                                           "vision_model"))
            self.vit_downsampler = nn.Conv2d(
                config.vision_config.hidden_size,
                config.vision_config.output_hidden_size,
                kernel_size=2,
                stride=config.understand_projector_stride)
            self.vit_downsampler2 = nn.Conv2d(
                config.vision_config.output_hidden_size,
                config.vision_config.output_hidden_size * 2,
                kernel_size=3,
                stride=2,
                padding=1,
            )
            self.vit_large_projector = nn.Linear(
                config.vision_config.output_hidden_size * 2,
                config.hidden_size,
                bias=config.projector_bias,
            )
        else:
            self.vision_model = None
            self.vit_downsampler = None
            self.vit_downsampler2 = None
            self.vit_large_projector = None

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"))

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    @cached_property
    def sampler(self):
        if hasattr(self.language_model, "sampler"):
            return self.language_model.sampler

        return get_sampler()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[Step3VLImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        patch_pixel_values = kwargs.pop("patch_pixel_values", None)
        num_patches = kwargs.pop("num_patches", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            pixel_values = flatten_bn(pixel_values, concat=True)
            if pixel_values.dim() >= 3:
                pixel_values = pixel_values.view(-1, *pixel_values.shape[-3:])
            if patch_pixel_values is not None:
                patch_pixel_values = flatten_bn(patch_pixel_values,
                                                concat=True)
                patch_pixel_values = patch_pixel_values.view(
                    -1, *patch_pixel_values.shape[-3:])
                # Handle empty patch_pixel_values by setting to None
                if patch_pixel_values.shape[0] == 0:
                    patch_pixel_values = None
            num_patches = flatten_bn(num_patches, concat=True).tolist()

            return Step3VLImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values.to(self.dtype).to(self.device),
                patch_pixel_values=patch_pixel_values.to(self.dtype).to(
                    self.device) if patch_pixel_values is not None else None,
                num_patches=num_patches,
            )

        if image_embeds is not None:
            if image_embeds.dim() == 2 or image_embeds.dim() >= 3:
                image_embeds = image_embeds.view(-1, image_embeds.shape[-1])
            else:
                raise ValueError(
                    f"Unexpected shape for image_embeds: {image_embeds.shape}")

            return Step3VLImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds.to(self.dtype).to(self.device),
            )
        return None

    def _process_image_features(self,
                                image_features: torch.Tensor) -> torch.Tensor:
        B, P = image_features.shape[:2]
        HW = int(sqrt(P))
        image_features = image_features.permute(0, 2, 1).view(B, -1, HW, HW)
        image_features = self.vit_downsampler(image_features)
        image_features = self.vit_downsampler2(image_features)
        n_dim = image_features.size(1)
        image_features = image_features.view(B, n_dim, -1).permute(0, 2, 1)
        image_features = self.vit_large_projector(image_features)
        return image_features

    def _get_vision_model_output(self,
                                 input_tensor: torch.Tensor) -> torch.Tensor:
        return self.vision_model(input_tensor)[:, 4:]

    def _process_image_input(
            self, image_input: Step3VLImageInputs) -> tuple[torch.Tensor, ...]:

        if image_input["type"] == "image_embeds":
            image_features = image_input["image_embeds"]
        else:
            image_features = self._get_vision_model_output(
                image_input["pixel_values"])
            patch_image_features = self._get_vision_model_output(
                image_input["patch_pixel_values"]
            ) if image_input["patch_pixel_values"] is not None else None
            num_patches = image_input["num_patches"]

        image_features = self._process_image_features(image_features)
        patch_image_features = self._process_image_features(
            patch_image_features) if patch_image_features is not None else None

        merged_image_features = []
        cur_patch_idx = 0
        for i, num_patch in enumerate(num_patches):
            cur_feature = []
            if num_patch > 0:
                patch_slice = patch_image_features[
                    cur_patch_idx:cur_patch_idx + num_patch]
                cur_feature.append(patch_slice.view(-1, patch_slice.shape[-1]))
            cur_feature.append(image_features[i].view(
                -1, image_features.shape[-1]))
            cur_patch_idx += num_patch
            merged_image_features.append(
                torch.cat(cur_feature) if len(cur_feature) >
                1 else cur_feature[0])
        return merged_image_features

    def get_multimodal_embeddings(self, **kwargs) -> Optional[NestedTensors]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        if multimodal_embeddings is None:
            inputs_embeds = self.language_model.model.get_input_embeddings(
                input_ids)
        else:
            is_text = input_ids != self.config.image_token_id
            text_ids = input_ids[is_text]
            text_embeds = self.language_model.model.get_input_embeddings(
                text_ids)
            inputs_embeds = torch.empty(input_ids.shape[0],
                                        text_embeds.shape[-1],
                                        dtype=text_embeds.dtype,
                                        device=text_embeds.device)
            inputs_embeds[is_text] = text_embeds
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                self.config.image_token_id)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if intermediate_tensors is not None:
            inputs_embeds = None
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            # always pass the input via `inputs_embeds`
            # to make sure the computation graph is consistent
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None

        hidden_states = self.language_model(input_ids,
                                            positions,
                                            intermediate_tensors,
                                            inputs_embeds=inputs_embeds)

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.language_model.sample(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):

        skip_prefixes = []
        if self.vision_model is None and self.vit_large_projector is None:
            skip_prefixes = [
                "vision_model.", "vit_downsampler.", "vit_downsampler2.",
                "vit_large_projector."
            ]

        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        loaded_weights = loader.load_weights(weights,
                                             mapper=self.hf_to_vllm_mapper)
        return loaded_weights
