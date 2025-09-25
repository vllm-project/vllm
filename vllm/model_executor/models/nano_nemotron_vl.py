# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# --------------------------------------------------------
# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/internvl.py
# under Apache-2.0 License
#     LICENSE is in root directory.
# --------------------------------------------------------

import copy
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, Literal, Optional, TypedDict, TypeVar, Union

import numpy.typing as npt
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from transformers import (BatchEncoding, BatchFeature, PretrainedConfig,
                          TensorType)

from vllm.config import VllmConfig
from vllm.model_executor.layers.activation import ReLUSquaredActivation
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (HasInnerState, IsHybrid,
                                                   MultiModalEmbeddings,
                                                   SupportsMultiModal)
from vllm.model_executor.models.internvl import (calculate_internvl_targets,
                                                 get_internvl_target_ratios)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM
from vllm.model_executor.models.radio import RadioModel
from vllm.model_executor.models.utils import (flatten_bn,
                                              init_vllm_registered_model,
                                              maybe_prefix,
                                              merge_multimodal_embeddings)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs, MultiModalKwargsItems,
                                    NestedTensors)
from vllm.multimodal.parse import (ImageEmbeddingItems, ImageProcessorItems,
                                   ImageSize, MultiModalDataItems)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate, PromptUpdateDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.radio import RadioConfig
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils.tensor_schema import TensorSchema, TensorShape

# Configure PIL to handle large images without warnings
# This prevents DecompressionBombWarning for legitimate large images
Image.MAX_IMAGE_PIXELS = None  # Disable the limit entirely
# Alternative: Set a specific higher limit
# Image.MAX_IMAGE_PIXELS = 300000000  # ~300M pixels

IMG_START = "<img>"
IMG_END = "</img>"
IMG_CONTEXT = "<image>"

# Profiling
MAX_FRAMES = 16


class NanoNemotronVLImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values_flat: torch.Tensor
    """
    Shape:
    `(batch_size * num_images * (1 + num_patches), num_channels, height, width)`
    """

    num_patches: torch.Tensor
    """Shape: `(batch_size * num_images)`"""


class NanoNemotronVLImageEmbeddinInputs(TypedDict):
    type: Literal["image_embeds"]
    data: Union[torch.Tensor, list[torch.Tensor]]
    """ 
    A tensor of shape `(num_images, total_image_feature_size, hidden_size)`
    or a list of tensors of shape `(total_image_feature_size, hidden_size)`

    `hidden_size` must match the hidden size of language model backbone.
    """


NanoNemotronVLImageInputs = Union[NanoNemotronVLImagePixelInputs,
                                  NanoNemotronVLImageEmbeddinInputs]


class NanoNemotronVLVideoPixelInputs(TensorSchema):
    """
    Dimensions:
        - bvf: Batch size * number of videos * num_frames
        - bn: Batch size * number of images
        - c: Number of channels (3)
        - h: Height of each video frame
        - w: Width of each video frame
    """
    type: Literal["pixel_values_videos"]
    pixel_values_flat: Annotated[torch.Tensor, TensorShape("bvf", 3, "h", "w")]
    num_patches: Annotated[torch.Tensor, TensorShape("bn")]


class NanoNemotronVLVideoEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - n: Number of videos
        - f: Total video feature size
        - h: Hidden size (must match the hidden size of language model backbone)
    """
    type: Literal["video_embeds"]
    data: Annotated[Union[torch.Tensor, list[torch.Tensor]],
                    TensorShape("n", "f", "h")]


NanoNemotronVLVideoInputs = Union[NanoNemotronVLVideoPixelInputs,
                                  NanoNemotronVLVideoEmbeddingInputs]


def dynamic_preprocess(image,
                       *,
                       image_size=512,
                       max_num_tiles=12,
                       use_thumbnail=True,
                       idx=0):
    orig_width, orig_height = image.size

    target_ratios = get_internvl_target_ratios(1, max_num_tiles)

    blocks, target_width, target_height = calculate_internvl_targets(
        orig_width=orig_width,
        orig_height=orig_height,
        target_ratios=target_ratios,
        image_size=image_size,
        use_thumbnail=False)
    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    processed_images = [
        img.convert("RGB") if img.mode != "RGB" else img
        for img in processed_images
    ]
    processed_images = [
        T.Resize((image_size, image_size),
                 interpolation=T.InterpolationMode.BICUBIC)(img)
        for img in processed_images
    ]
    processed_images = [T.ToTensor()(img) for img in processed_images]
    return processed_images


def image_to_pixel_values(
    image: Image.Image,
    *,
    input_size: int,
    max_num: int,
    use_thumbnail: bool,
    idx: int,
) -> torch.Tensor:
    images = dynamic_preprocess(
        image,
        image_size=input_size,
        max_num_tiles=max_num,
        use_thumbnail=use_thumbnail,
        idx=idx,
    )

    pixel_values = torch.stack(images)
    return pixel_values


def video_to_pixel_values(
    video: npt.NDArray,
    *,
    input_size: int,
    max_num_tiles: int = 1,
    use_thumbnail: bool,
) -> torch.Tensor:
    # Convert each frame to a single resized tile tensor consistent
    # with image path
    frames_tensors: list[torch.Tensor] = []
    for frame in video:
        pil_frame = dynamic_preprocess(
            Image.fromarray(frame, mode="RGB"),
            image_size=input_size,
            max_num_tiles=max_num_tiles,
            use_thumbnail=use_thumbnail,
            idx=0,
        )
        # dynamic_preprocess returns tensors already; take the single tile
        assert len(pil_frame) >= 1
        frames_tensors.append(pil_frame[0])

    return torch.stack(frames_tensors)


class BaseNanoNemotronVLProcessor(ABC):
    """
    This model doesn't define its own HF processor,
    so we implement our own one here.

    The code to insert image tokens is based on:
    https://huggingface.co/OpenGVLab/InternVL2-1B/blob/main/modeling_internvl_chat.py#L252
    """

    def __init__(self, config: PretrainedConfig, tokenizer: AnyTokenizer,
                 *args, **kwargs) -> None:
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer

        image_size: int = config.force_image_size
        patch_size: int = config.patch_size

        self.num_image_token = int(
            (image_size // patch_size)**2 * (config.downsample_ratio**2))
        self.image_size = image_size
        self.use_thumbnail: bool = config.use_thumbnail
        self.norm_mean = torch.Tensor(config.norm_mean).reshape(1, 3, 1, 1)
        self.norm_std = torch.Tensor(config.norm_std).reshape(1, 3, 1, 1)

    @property
    @abstractmethod
    def image_token_id(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_image_repl(
        self,
        feature_size: int,
        num_patches: Optional[int],
    ) -> PromptUpdateDetails[str]:
        raise NotImplementedError

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        max_num_tiles: int,
    ) -> int:
        target_ratios = get_internvl_target_ratios(1, max_num_tiles)

        num_patches, _, _ = calculate_internvl_targets(
            orig_width=image_width,
            orig_height=image_height,
            target_ratios=target_ratios,
            image_size=self.image_size,
            use_thumbnail=self.use_thumbnail,
        )

        return num_patches * self.num_image_token

    def _images_to_pixel_values_lst(
        self,
        images: list[Image.Image],
        max_num_tiles: int,
    ) -> list[torch.Tensor]:
        return [
            image_to_pixel_values(
                image,
                input_size=self.image_size,
                max_num=max_num_tiles,
                use_thumbnail=self.use_thumbnail,
                idx=idx,
            ) for idx, image in enumerate(images)
        ]

    def _preprocess_image(
        self,
        text: list[str],
        images: list[Image.Image],
        max_num_tiles: int,
    ) -> tuple[list[str], dict[str, torch.Tensor]]:
        if len(images) == 0:
            image_inputs = {}
        else:
            pixel_values_lst = self._images_to_pixel_values_lst(
                images, max_num_tiles)
            image_inputs: dict[str, NestedTensors] = {
                "pixel_values_flat":
                torch.cat(pixel_values_lst),
                "image_num_patches":
                torch.tensor([len(item) for item in pixel_values_lst]),
            }

            for pixel_values in pixel_values_lst:
                num_patches = pixel_values.shape[0]
                feature_size = num_patches * self.num_image_token
                image_repl = self.get_image_repl(feature_size, num_patches)
                text = [t.replace('<image>', image_repl.full, 1) for t in text]
        return text, image_inputs

    def _make_batch_input(self,
                          input_item: Optional[Union[Any, list[Any]]] = None):
        if input_item is None:
            input_item = []
        if not isinstance(input_item, list):
            input_item = [input_item]
        return input_item

    def __call__(
        self,
        text: Optional[Union[str, list[str]]] = None,
        images: Optional[Union[Image.Image, list[Image.Image]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        max_num_tiles: Optional[int] = None,
    ) -> Mapping[str, NestedTensors]:
        # Use default if not provided
        if max_num_tiles is None:
            max_num_tiles = 12

        text, images = [self._make_batch_input(x) for x in (text, images)]

        text, image_inputs = self._preprocess_image(
            text=text,
            images=images,
            max_num_tiles=max_num_tiles,
        )

        text_inputs = self.tokenizer(text, add_special_tokens=False)

        return {
            **BatchEncoding(text_inputs, tensor_type=return_tensors),
            **image_inputs,
        }


class NanoNemotronVLProcessor(BaseNanoNemotronVLProcessor):
    """
    HF Processor  with extended video processing logic.
    Code for video processing is adapted from video example:
    https://huggingface.co/OpenGVLab/InternVL3-1B#inference-with-transformers
    """

    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: AnyTokenizer,
        *,
        min_dynamic_patch: Optional[int] = None,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
        video_token: Optional[str] = None,
    ) -> None:
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
        )
        # add extra video token for video processing
        self.video_token = video_token

    @property
    def supports_video(self) -> bool:
        return self.video_token_id is not None

    @property
    def video_token_id(self) -> Optional[int]:
        if self.video_token is None:
            return None
        return self.tokenizer.get_vocab().get(self.video_token, None)

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT)

    def _videos_to_pixel_values_lst(
        self,
        videos: list[npt.NDArray],
        max_num_tiles: int,
        dynamic_image_size: Optional[bool] = None,
    ) -> list[torch.Tensor]:

        return [
            video_to_pixel_values(
                video,
                input_size=self.image_size,
                max_num_tiles=max_num_tiles,
                use_thumbnail=self.use_thumbnail,
            ) for video in videos
        ]

    def _preprocess_video(
        self,
        text: list[str],
        videos: list[npt.NDArray],
        max_num_tiles: int,
        dynamic_image_size: Optional[bool] = None,
    ):
        if len(videos) == 0 or not self.supports_video:
            video_inputs = {}
        else:
            pixel_values_lst_video = self._videos_to_pixel_values_lst(
                videos,
                max_num_tiles=max_num_tiles,
                dynamic_image_size=dynamic_image_size,
            )

            video_inputs: dict[str, NestedTensors] = {
                "pixel_values_flat_video":
                torch.cat(pixel_values_lst_video),
                "video_num_patches":
                torch.tensor([len(item) for item in pixel_values_lst_video]),
            }

            for pixel_values in pixel_values_lst_video:
                num_patches = pixel_values.shape[0]

                video_repl = self.get_video_repl(self.num_image_token,
                                                 num_patches, self.video_token)
                text = [t.replace('<video>', video_repl.full, 1) for t in text]
        return text, video_inputs

    def __call__(
        self,
        text: Optional[Union[str, list[str]]] = None,
        images: Optional[Union[Image.Image, list[Image.Image]]] = None,
        videos: Optional[Union[npt.NDArray, list[npt.NDArray]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        max_num_tiles: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
    ) -> Mapping[str, NestedTensors]:
        # Use default if not provided
        if max_num_tiles is None:
            max_num_tiles = 12

        text, images, videos = [
            self._make_batch_input(x) for x in (text, images, videos)
        ]

        text, image_inputs = self._preprocess_image(
            text=text,
            images=images,
            max_num_tiles=max_num_tiles,
        )

        text, video_inputs = self._preprocess_video(
            text=text,
            videos=videos,
            max_num_tiles=max_num_tiles,
            dynamic_image_size=dynamic_image_size,
        )

        text_inputs = self.tokenizer(text, add_special_tokens=False)

        return BatchFeature({
            **BatchEncoding(text_inputs, tensor_type=return_tensors),
            **image_inputs,
            **video_inputs,
        })

    def get_image_repl(
        self,
        feature_size: int,
        num_patches: Optional[int],
    ) -> PromptUpdateDetails[str]:
        repl_features = IMG_CONTEXT * feature_size
        repl_full = IMG_START + repl_features + IMG_END

        return PromptUpdateDetails.select_text(repl_full, IMG_CONTEXT)

    def get_video_repl(
        self,
        feature_size: int,
        num_patches: Optional[int] = None,
        video_context_token: str = IMG_CONTEXT,
    ) -> PromptUpdateDetails[str]:
        repl_features = video_context_token * self.num_image_token
        repl_features_with_sep = IMG_START + repl_features + IMG_END
        # num_patches is equal to num_frames
        repl_full = ''.join([
            f'Frame{i+1}: {repl_features_with_sep}' for i in range(num_patches)
        ])

        return PromptUpdateDetails.select_text(repl_full, video_context_token)


class BaseNanoNemotronVLProcessingInfo(BaseProcessingInfo):
    """Basic image-only ProcessingInfo for InternVL-style models."""

    @abstractmethod
    def get_hf_processor(
        self,
        **kwargs: object,
    ) -> BaseNanoNemotronVLProcessor:
        raise NotImplementedError

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        max_num_tiles: int,
        processor: Optional[BaseNanoNemotronVLProcessor],
    ) -> int:
        if processor is None:
            processor = self.get_hf_processor()

        return processor.get_num_image_tokens(
            image_width=image_width,
            image_height=image_height,
            max_num_tiles=max_num_tiles,
        )

    def get_image_size_with_most_features(self,
                                          max_num_tiles: int) -> ImageSize:
        processor = self.get_hf_processor()

        base_size = processor.image_size
        target_ratios = get_internvl_target_ratios(1, max_num_tiles)

        largest_feature_size, largest_feature_pinpoint = 0, None
        for wr, hr in target_ratios:
            width, height = base_size * wr, base_size * hr

            feat_size = self.get_num_image_tokens(
                image_width=width,
                image_height=height,
                max_num_tiles=max_num_tiles,
                processor=processor,
            )
            if feat_size > largest_feature_size:
                largest_feature_size = feat_size
                largest_feature_pinpoint = ImageSize(width=width,
                                                     height=height)

        if largest_feature_size == 0 or largest_feature_pinpoint is None:
            raise ValueError("Cannot have a largest feature size of 0!")

        return largest_feature_pinpoint

    def get_max_image_tokens(self) -> int:
        processor = self.get_hf_processor()
        # Use default max_num_tiles for max tokens calculation
        max_num_tiles = 12
        target_width, target_height = self.get_image_size_with_most_features(
            max_num_tiles)

        return self.get_num_image_tokens(
            image_width=target_width,
            image_height=target_height,
            max_num_tiles=max_num_tiles,
            processor=processor,
        )


_I = TypeVar("_I", bound=BaseNanoNemotronVLProcessingInfo)


class NanoNemotronVLProcessingInfo(BaseNanoNemotronVLProcessingInfo):
    """ ProcessingInfo extended for video processing"""

    @property
    def supports_video(self):
        return self.get_hf_processor().supports_video

    def get_supported_mm_limits(self):
        video_limit = {"video": None} if self.supports_video else {}
        return {**super().get_supported_mm_limits(), **video_limit}

    def get_video_token(self) -> Optional[str]:
        return IMG_CONTEXT

    def get_num_frames_with_most_features(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        max_images = mm_counts.get("image", 0)
        max_videos = mm_counts.get("video", 0)

        processor = self.get_hf_processor()  # we get the CustomProcessor here

        max_image_tokens = self.get_max_image_tokens() * max_images
        max_total_frames = (seq_len -
                            max_image_tokens) // processor.num_image_token
        max_frames_per_video = max_total_frames // max(max_videos, 1)

        max_frames_per_video = min(max_frames_per_video, MAX_FRAMES)
        return max(max_frames_per_video, 1)

    def get_hf_processor(self, **kwargs: object) -> NanoNemotronVLProcessor:
        return self.ctx.init_processor(
            NanoNemotronVLProcessor,
            config=self.get_hf_config(),
            tokenizer=self.get_tokenizer(),
            video_token=self.get_video_token(),
            **kwargs,
        )


class NanoNemotronBaseVLMultiModalProcessor(BaseMultiModalProcessor[_I]):
    """Basic image-only MultiModalProcessor for InternVL-style models."""

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> Mapping[str, NestedTensors]:
        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

        hf_processor = self.info.get_hf_processor(**mm_kwargs)
        image_token_id = hf_processor.image_token_id

        # Since there may be extra tokens in the feature placeholders,
        # we need to pass the image token ID to the model to select the
        # tokens to merge from the vision encoder outputs
        processed_outputs["image_token_id"] = torch.tensor(image_token_id)

        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: Mapping[str, NestedTensors],
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        image_num_patches = hf_inputs.get("image_num_patches", torch.empty(0))
        num_images = len(image_num_patches)

        return dict(
            pixel_values_flat=MultiModalFieldConfig.flat_from_sizes(
                "image", image_num_patches),
            image_num_patches=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
            image_token_id=MultiModalFieldConfig.shared("image", num_images),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        if "image_num_patches" in out_mm_kwargs:
            image_num_patches = out_mm_kwargs["image_num_patches"]
            assert isinstance(image_num_patches, torch.Tensor)
            image_num_patches = image_num_patches.tolist()
        elif "image_embeds" in out_mm_kwargs:
            # to compute num_patches (similar to Qwen2-VL)
            image_num_patches = [None] * len(out_mm_kwargs["image_embeds"])
        else:
            image_num_patches = []

        def get_replacement_custom(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems))

            if isinstance(images, ImageEmbeddingItems):
                feature_size = images.get_feature_size(item_idx)
            else:
                image_size = images.get_image_size(item_idx)
                # Extract max_num_tiles from kwargs, default to 12
                max_num_tiles = hf_processor_mm_kwargs.get("max_num_tiles", 12)
                feature_size = self.info.get_num_image_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                    max_num_tiles=max_num_tiles,
                    processor=hf_processor,
                )

            num_patches = None
            local_image_num_patches = image_num_patches
            if isinstance(local_image_num_patches, torch.Tensor):
                local_image_num_patches = local_image_num_patches.tolist()
            if isinstance(
                    local_image_num_patches,
                (list, tuple)) and item_idx < len(local_image_num_patches):
                num_patches = int(local_image_num_patches[item_idx])

            return hf_processor.get_image_repl(feature_size, num_patches)

        return [
            PromptReplacement(
                modality="image",
                target="<image>",
                replacement=get_replacement_custom,
            )
        ]


class NanoNemotronVLMultiModalProcessor(
        NanoNemotronBaseVLMultiModalProcessor[NanoNemotronVLProcessingInfo]):
    """MultiModalProcessor extended for video support"""

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> Mapping[str, NestedTensors]:
        processed_outputs = super()._call_hf_processor(prompt, mm_data,
                                                       mm_kwargs, tok_kwargs)

        hf_processor = self.info.get_hf_processor(**mm_kwargs)
        if self.info.supports_video and (
                video_token_id := hf_processor.video_token_id) is not None:
            processed_outputs["video_token_id"] = torch.tensor(video_token_id)
        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: Mapping[str, NestedTensors],
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        image_fields = super()._get_mm_fields_config(hf_inputs,
                                                     hf_processor_mm_kwargs)
        if self.info.supports_video:
            video_num_patches = hf_inputs.get("video_num_patches",
                                              torch.empty(0))
            num_videos = len(video_num_patches)
            video_fields = dict(
                pixel_values_flat_video=MultiModalFieldConfig.flat_from_sizes(
                    "video", video_num_patches),
                video_num_patches=MultiModalFieldConfig.batched("video"),
                video_token_id=MultiModalFieldConfig.shared(
                    "video", num_videos))
        else:
            video_fields = {}

        return image_fields | video_fields

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        prompt_repl = super()._get_prompt_updates(
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            out_mm_kwargs=out_mm_kwargs,
        )

        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        out_mm_data = out_mm_kwargs.get_data()
        if "video_num_patches" in out_mm_data:
            video_num_patches = out_mm_data["video_num_patches"]
            assert isinstance(video_num_patches, torch.Tensor)
            video_num_patches = video_num_patches.tolist()
        else:
            video_num_patches = []

        def get_video_replacement_internvl(item_idx: int):
            feature_size = hf_processor.num_image_token
            num_patches = video_num_patches[item_idx]
            if num_patches is not None:
                assert isinstance(num_patches, int)

            return hf_processor.get_video_repl(
                feature_size,
                num_patches,
                video_context_token=hf_processor.video_token)

        if self.info.supports_video:
            prompt_repl = [
                *prompt_repl,
                PromptReplacement(
                    modality="video",
                    target="<video>",
                    replacement=get_video_replacement_internvl,
                )
            ]

        return prompt_repl


class NanoNemotronVLDummyInputsBuilder(BaseDummyInputsBuilder[_I]):
    """Basic image-only DummyInputsBuilder for InternVL-style models."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        return "<image>" * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        # Use default max_num_tiles for dummy data generation
        max_num_tiles = 12
        target_width, target_height = (
            self.info.get_image_size_with_most_features(max_num_tiles))
        num_images = mm_counts.get("image", 0)

        return {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images)
        }


class NanoNemotronVLDummyInputsBuilder(
        NanoNemotronVLDummyInputsBuilder[NanoNemotronVLProcessingInfo]):
    """DummyInputsBuilder extended for video support"""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_videos = mm_counts.get("video", 0)

        return super().get_dummy_text(mm_counts) + "<video>" * num_videos

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        dummy_image = super().get_dummy_mm_data(seq_len=seq_len,
                                                mm_counts=mm_counts)
        if self.info.supports_video:
            config = self.info.get_hf_config()
            image_size: int = config.force_image_size
            target_num_frames = \
                self.info.get_num_frames_with_most_features(seq_len, mm_counts)
            num_videos = mm_counts.get("video", 0)
            dummy_video = {
                "video":
                self._get_dummy_videos(width=image_size,
                                       height=image_size,
                                       num_frames=target_num_frames,
                                       num_videos=num_videos)
            }
        else:
            dummy_video = {}
        return {**dummy_image, **dummy_video}


@MULTIMODAL_REGISTRY.register_processor(
    NanoNemotronVLMultiModalProcessor,
    info=NanoNemotronVLProcessingInfo,
    dummy_inputs=NanoNemotronVLDummyInputsBuilder,
)
class NemotronH_Nano_VL(nn.Module, HasInnerState, IsHybrid,
                        SupportsMultiModal):

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("image"):
            return "<image>"
        if modality.startswith("video"):
            return "<video>"
        return None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config

        image_size = config.force_image_size
        patch_size = config.patch_size
        self.patch_size = patch_size
        self.template = config.template
        self.num_image_token = int(
            (image_size // patch_size)**2 * (config.downsample_ratio**2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.image_tag_type = config.image_tag_type

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )
        self.vision_model = self.get_vit_model_from_radio_config(config).to(
            self.language_model.config.torch_dtype)

        # Construct the vision projection.
        vit_hidden_size = config.vit_hidden_size
        vision_projection_hidden_size = config.projector_hidden_size
        llm_hidden_size = config.text_config.hidden_size

        self.mlp1 = nn.Sequential(
            RMSNorm(hidden_size=vit_hidden_size *
                    int(1 / self.downsample_ratio)**2,
                    eps=1e-5),
            nn.Linear(
                vit_hidden_size * int(1 / self.downsample_ratio)**2,
                vision_projection_hidden_size,
                bias=False,
            ),
            ReLUSquaredActivation(),
            nn.Linear(vision_projection_hidden_size,
                      llm_hidden_size,
                      bias=False),
        )
        self.mlp1 = self.mlp1.to(self.language_model.config.torch_dtype)

        self.img_context_token_id = None
        self.video_context_token_id = None
        self.config = config

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(
            n,
            w,
            int(h * scale_factor),
            int(c / scale_factor),
        )
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale -->
        # N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        if self.ps_version == "v1":
            warnings.warn(
                "In ps_version 'v1', the height and width have not "
                "been swapped back, which results in a transposed image.",
                stacklevel=2,
            )
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        vit_embeds = self.vision_model(pixel_values)
        vit_embeds = vit_embeds.to(dtype=torch.bfloat16)
        h = w = int(vit_embeds.shape[1]**0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds,
                                        scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1,
                                        vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[NanoNemotronVLImageInputs]:
        pixel_values_flat = kwargs.pop("pixel_values_flat", None)
        image_num_patches = kwargs.pop("image_num_patches", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values_flat is None and image_embeds is None:
            return None

        if image_embeds is not None:
            if not isinstance(image_embeds, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")

            return NanoNemotronVLImageEmbeddinInputs(
                type="image_embeds",
                data=flatten_bn(image_embeds),
            )

        image_token_id = kwargs["image_token_id"]
        assert isinstance(image_token_id, torch.Tensor)
        self.img_context_token_id = image_token_id.flatten().unique().item()

        if pixel_values_flat is not None:
            if not isinstance(pixel_values_flat, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values_flat)}")

            if not isinstance(image_num_patches, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image_num_patches. "
                                 f"Got type: {type(image_num_patches)}")

            pixel_values_flat = flatten_bn(pixel_values_flat, concat=True)
            image_num_patches = flatten_bn(image_num_patches, concat=True)

            return NanoNemotronVLImagePixelInputs(
                type="pixel_values",
                pixel_values_flat=pixel_values_flat,
                num_patches=image_num_patches,
            )

        raise AssertionError("This line should be unreachable.")

    def _process_image_input(
            self, image_input: NanoNemotronVLImageInputs) -> torch.Tensor:
        if image_input["type"] == "image_embeds":
            return image_input["data"]

        assert self.vision_model is not None

        image_embeds = self.extract_feature(image_input["pixel_values_flat"])
        num_patches = image_input["num_patches"]

        # Only one image in the current batch
        if len(num_patches) == 1:
            return (image_embeds.view(-1,
                                      self.config.text_config.hidden_size), )

        # NOTE: Image embeddings are split into separate tensors for each image
        # by the size of each embedding.
        feature_size = image_embeds.shape[1]
        image_embeds = image_embeds.view(-1,
                                         self.config.text_config.hidden_size)
        image_feature_sizes = [
            num_patches * feature_size for num_patches in num_patches
        ]
        return image_embeds.split(image_feature_sizes)

    def _parse_and_validate_video_input(
            self,
            **kwargs: object) -> Optional[NanoNemotronVLVideoPixelInputs]:
        pixel_values_flat_video = kwargs.pop("pixel_values_flat_video", None)
        video_num_patches = kwargs.pop("video_num_patches", None)
        video_embeds = kwargs.pop("video_embeds", None)

        if pixel_values_flat_video is None and video_embeds is None:
            return None

        if video_embeds is not None:
            return NanoNemotronVLVideoEmbeddingInputs(
                type="video_embeds",
                data=flatten_bn(video_embeds),
            )

        video_token_id = kwargs["video_token_id"]
        assert isinstance(video_token_id, torch.Tensor)
        self.video_context_token_id = video_token_id.flatten().unique().item()

        if pixel_values_flat_video is not None:
            if not isinstance(pixel_values_flat_video, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values_flat_video)}")

            if not isinstance(video_num_patches, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image_num_patches. "
                                 f"Got type: {type(video_num_patches)}")

            pixel_values_flat_video = flatten_bn(pixel_values_flat_video,
                                                 concat=True)
            video_num_patches = flatten_bn(video_num_patches, concat=True)
            expected_h = expected_w = self.config.force_image_size
            resolve_bindings = {"h": expected_h, "w": expected_w}

            return NanoNemotronVLVideoPixelInputs(
                type="pixel_values_videos",
                pixel_values_flat=pixel_values_flat_video,
                num_patches=video_num_patches,
                resolve_bindings=resolve_bindings,
            )

        raise AssertionError("This line should be unreachable.")

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}
        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if input_key in ("pixel_values_flat",
                             "image_embeds") and "images" not in modalities:
                modalities["images"] = self._parse_and_validate_image_input(
                    **kwargs)
            if input_key in ("pixel_values_flat_video",
                             ) and "videos" not in modalities:
                modalities["videos"] = self._parse_and_validate_video_input(
                    **kwargs)

        return modalities

    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        # Validate the multimodal input keyword arguments
        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if modalities is None:
            return []

        # # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in modalities:
            if modality == "images":
                image_input = modalities["images"]
                vision_embeddings = self._process_image_input(image_input)
                multimodal_embeddings += vision_embeddings
            if modality == "videos":
                video_input = modalities["videos"]
                video_embeddings = self._process_image_input(video_input)
                multimodal_embeddings += video_embeddings

        return multimodal_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)

        if (multimodal_embeddings is not None
                and len(multimodal_embeddings) != 0):
            context_token_ids = [
                token_id for token_id in (self.img_context_token_id,
                                          self.video_context_token_id)
                if token_id is not None
            ]
            assert len(context_token_ids) >= 1
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                context_token_ids,
            )

        return inputs_embeds

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if intermediate_tensors is not None:
            input_ids = None
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None

        hidden_states = self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        return hidden_states

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="mlp1",
            tower_model="vision_model",
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        adapter_dict = dict(self.mlp1.named_parameters())

        def is_llm(name: str) -> bool:
            return name.startswith("language_model")

        def is_adapter_weights(weight: tuple[str, torch.Tensor]):
            return weight[0].startswith("mlp1")

        def is_vision_weights(name: str) -> bool:
            return name.startswith("vision_model.radio_model.")

        # Separate weights by component
        llm_weights = []
        vision_weights = []

        for name, w in weights:
            if is_llm(name):
                # Strip 'language_model.' prefix for LLM weights
                llm_weights.append((".".join(name.split(".")[1:]), w))
            elif is_adapter_weights((name, w)):
                # Load vision-language adapter weights directly
                trimmed_name = ".".join(name.split(".")[1:])
                param = adapter_dict[trimmed_name]
                with torch.no_grad():
                    default_weight_loader(param, w)
            elif is_vision_weights(name):
                # Convert: vision_model.radio_model.* → radio_model.*
                hf_key = name[len(
                    "vision_model."):]  # Remove "vision_model." prefix
                vision_weights.append((hf_key, w))

        self.language_model.load_weights(llm_weights)
        self.vision_model.load_weights(vision_weights)

    def print_architecture(self,
                           detailed: bool = True,
                           save_to_file: str = None):
        """
        Print model architecture with parameter names, shapes, and sizes.

        Args:
            detailed: If True, show detailed parameter breakdown
            save_to_file: If provided, save output to this file path
        """
        import sys
        from io import StringIO

        # Capture output if saving to file
        original_stdout = sys.stdout
        if save_to_file:
            sys.stdout = StringIO()

        try:
            print("=" * 100)
            print("NemotronH_Nano_VL Model Architecture")
            print("=" * 100)

            total_params = 0
            param_groups = {
                "language_model": [],
                "vision_model": [],
                "mlp1": [],
                "other": [],
            }

            for name, param in self.named_parameters():
                param_size = param.numel()
                total_params += param_size

                # Group parameters by main component
                if name.startswith("language_model"):
                    param_groups["language_model"].append(
                        (name, param.shape, param_size, param.dtype))
                elif name.startswith("vision_model"):
                    param_groups["vision_model"].append(
                        (name, param.shape, param_size, param.dtype))
                elif name.startswith("mlp1"):
                    param_groups["mlp1"].append(
                        (name, param.shape, param_size, param.dtype))
                else:
                    param_groups["other"].append(
                        (name, param.shape, param_size, param.dtype))

                if detailed:
                    print(f"{name:<70} | Shape: {str(param.shape):<25} | "
                          f"Size: {param_size:>12,} | Dtype: {param.dtype}")

            print("=" * 100)
            print("Summary by Component:")
            print("-" * 60)

            for component, params in param_groups.items():
                if params:  # Only show components that have parameters
                    component_total = sum(size for _, _, size, _ in params)
                    percentage = ((component_total / total_params) *
                                  100 if total_params > 0 else 0)
                    print(f"{component:<20} | Parameters: {len(params):>4} | "
                          f"Total Size: {component_total:>15,} | "
                          f"{percentage:>6.2f}%")

            print("-" * 60)
            print(f"{'Total Parameters':<20} | {total_params:>15,}")

            # Estimate memory usage (assuming bfloat16 = 2 bytes per parameter)
            memory_mb = total_params * 2 / (1024**2)
            memory_gb = memory_mb / 1024
            print(f"{'Est. Memory (MB)':<20} | {memory_mb:>15.2f}")
            print(f"{'Est. Memory (GB)':<20} | {memory_gb:>15.2f}")
            print("=" * 100)

            # Save to file if requested
            if save_to_file:
                output = sys.stdout.getvalue()
                sys.stdout = original_stdout
                with open(save_to_file, "w") as f:
                    f.write(output)
                print(f"Architecture saved to: {save_to_file}")
                print(output)  # Also print to console

        finally:
            if save_to_file and sys.stdout != original_stdout:
                sys.stdout = original_stdout

    def get_model_info(self):
        """
        Get basic model information as a dictionary.
        """
        total_params = sum(p.numel() for p in self.parameters())

        component_info = {}
        for name, param in self.named_parameters():
            component = name.split(".")[0]
            if component not in component_info:
                component_info[component] = {"params": 0, "size": 0}
            component_info[component]["params"] += 1
            component_info[component]["size"] += param.numel()

        return {
            "model_name": "NemotronH_Nano_VL",
            "total_parameters": total_params,
            "memory_estimate_mb": total_params * 2 / (1024**2),  # bfloat16
            "components": component_info,
            "config": {
                "image_size": getattr(self.config, "force_image_size", None),
                "patch_size": getattr(self.config, "patch_size", None),
                "num_image_token": self.num_image_token,
                "downsample_ratio": self.downsample_ratio,
            },
        }

    def get_vit_model_from_radio_config(self, hf_config):
        hf_config_vision = hf_config.vision_config
        model_name = hf_config_vision.args.get("model")
        if model_name is None:
            raise ValueError(f'Unsupported vit model type: {model_name}')

        preferred_resolution = getattr(hf_config_vision,
                                       "preferred_resolution", None)
        image_size = preferred_resolution[0] if preferred_resolution else 224
        patch_size = getattr(hf_config_vision, "patch_size", 16)

        radio_config = RadioConfig(
            model_name=model_name,
            image_size=image_size,
            patch_size=patch_size,
            norm_mean=hf_config.norm_mean,
            norm_std=hf_config.norm_std,
            reg_tokens=(hf_config_vision.args.get("register_multiple")
                        if hasattr(hf_config_vision, "args")
                        and isinstance(hf_config_vision.args, dict) else None),
        )

        return RadioModel(config=radio_config)

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.language_model.mamba_cache.copy_inputs_before_cuda_graphs(
            input_buffers, **kwargs)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return (self.language_model.mamba_cache.
                get_seqlen_agnostic_capture_inputs(batch_size))

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: "VllmConfig"):
        text_config = vllm_config.model_config.hf_config.text_config
        temp_vllm_config = copy.deepcopy(vllm_config)
        temp_vllm_config.model_config.hf_config = text_config
        return NemotronHForCausalLM.get_mamba_state_shape_from_config(
            temp_vllm_config)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: "VllmConfig"):
        text_config = vllm_config.model_config.hf_config.text_config
        temp_vllm_config = copy.deepcopy(vllm_config)
        temp_vllm_config.model_config.hf_config = text_config
        return NemotronHForCausalLM.get_mamba_state_dtype_from_config(
            temp_vllm_config)
