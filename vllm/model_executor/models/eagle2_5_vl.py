# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# adapted from https://huggingface.co/nvidia/Eagle2.5-8B/blob/main/modeling_eagle2_5_vl.py
# --------------------------------------------------------
# NVIDIA
# Copyright (c) 2025 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, Literal, Optional, TypeVar, Union

import numpy.typing as npt
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from transformers import BatchEncoding, PretrainedConfig, TensorType

from vllm.config import VllmConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.models.intern_vit import (InternVisionModel,
                                                   InternVisionPatchModel)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.siglip import SiglipVisionModel
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.image import convert_image_mode
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargsItems, NestedTensors)
from vllm.multimodal.parse import (ImageEmbeddingItems, ImageProcessorItems,
                                   ImageSize, MultiModalDataItems)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate, PromptUpdateDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import (MultiModalEmbeddings, SupportsLoRA,
                         SupportsMultiModal, SupportsPP)
from .utils import (AutoWeightsLoader, flatten_bn, init_vllm_registered_model,
                    maybe_prefix, merge_multimodal_embeddings)

IMG_START = '<img>'
IMG_END = '</img>'
IMG_CONTEXT = '<IMG_CONTEXT>'

SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


class Eagle2_5_VLImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of images
        - bnp: Batch size * number of images * (1 + num_patches)
        - c: Number of channels (3)
        - h: Height of each image patch
        - w: Width of each image patch
    """
    type: Literal["pixel_values"]
    pixel_values_flat: Annotated[torch.Tensor, TensorShape("bnp", 3, "h", "w")]
    num_patches: Annotated[torch.Tensor, TensorShape("bn")]


class Eagle2_5_VLImageEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - n: Number of images
        - f: Total image feature size
        - h: Hidden size (must match the hidden size of language model backbone)
    """
    type: Literal["image_embeds"]
    data: Annotated[Union[torch.Tensor, list[torch.Tensor]],
                    TensorShape("n", "f", "h")]


Eagle2_5_VLImageInputs = Union[Eagle2_5_VLImagePixelInputs,
                            Eagle2_5_VLImageEmbeddingInputs]


class Eagle2_5_VLVideoPixelInputs(TensorSchema):
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


class Eagle2_5_VLVideoEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - n: Number of videos
        - f: Total video feature size
        - h: Hidden size (must match the hidden size of language model backbone)
    """
    type: Literal["video_embeds"]
    data: Annotated[Union[torch.Tensor, list[torch.Tensor]],
                    TensorShape("n", "f", "h")]


Eagle2_5_VLVideoInputs = Union[Eagle2_5_VLVideoPixelInputs,
                            Eagle2_5_VLVideoEmbeddingInputs]


# adapted from https://huggingface.co/nvidia/Eagle2.5-8B
def build_transform(input_size: int):
    MEAN, STD = SIGLIP_MEAN, SIGLIP_STD
    return T.Compose([
        T.Lambda(lambda img: convert_image_mode(img, 'RGB')),
        T.Resize((input_size, input_size),
                 interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])


def find_closest_aspect_ratio_v2(aspect_ratio, target_ratios, width, height, image_size):
    """
    previous version mainly foucs on ratio.
    We also consider area ratio here.
    """
    best_factor = float('-inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        area_ratio = (ratio[0]*ratio[1]*image_size*image_size)/ area

        """
        new area > 60% of original image area is enough.
        """
        factor_based_on_area_n_ratio = min((ratio[0]*ratio[1]*image_size*image_size)/ area, 0.6)* \
                                     min(target_aspect_ratio/aspect_ratio, aspect_ratio/target_aspect_ratio)

        if factor_based_on_area_n_ratio > best_factor:
            best_factor = factor_based_on_area_n_ratio
            best_ratio = ratio

    return best_ratio


def resolve_eagle2_5_vl_min_max_num(
    *,
    min_dynamic_tiles: int,
    max_dynamic_tiles: int,
    dynamic_image_size: bool,
    use_thumbnail: bool,
) -> tuple[int, int]:
    min_dynamic_tiles = min_dynamic_tiles if dynamic_image_size else 1
    max_dynamic_tiles = max_dynamic_tiles if dynamic_image_size else 1

    if use_thumbnail and max_dynamic_tiles != 1:
        max_dynamic_tiles += 1

    return min_dynamic_tiles, max_dynamic_tiles


def get_eagle2_5_vl_target_ratios(
    min_num: int,
    max_num: int,
) -> list[tuple[int, int]]:
    target_ratios = {(i, j)
                     for n in range(min_num, max_num + 1)
                     for i in range(1, n + 1)
                     for j in range(1, n + 1) if min_num <= i * j <= max_num}
    return sorted(target_ratios, key=lambda x: x[0] * x[1])


def calculate_eagle2_5_vl_targets(
    *,
    orig_width: int,
    orig_height: int,
    target_ratios: list[tuple[int, int]],
    image_size: int,
    use_thumbnail: bool,
) -> tuple[int, int, int]:
    aspect_ratio = orig_width / orig_height

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio_v2(
        aspect_ratio,
        target_ratios,
        width=orig_width,
        height=orig_height,
        image_size=image_size,
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # add thumbnail image if num_blocks != 1
    if use_thumbnail and blocks != 1:
        blocks += 1

    return blocks, target_width, target_height


# adapted from https://huggingface.co/nvidia/Eagle2.5-8B
def dynamic_preprocess_eagle2_5_vl(
    image: Image.Image,
    *,
    target_ratios: list[tuple[int, int]],
    image_size: int,
    use_thumbnail: bool,
) -> list[Image.Image]:
    orig_width, orig_height = image.size

    # calculate the number of blocks without thumbnail
    blocks, target_width, target_height = calculate_eagle2_5_vl_targets(
        orig_width=orig_width,
        orig_height=orig_height,
        target_ratios=target_ratios,
        image_size=image_size,
        use_thumbnail=False,
    )

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size,
               (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size,
               ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    assert len(processed_images) == blocks

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


# adapted from https://huggingface.co/nvidia/Eagle2.5-8B
def image_to_pixel_values_eagle2_5_vl(
    image: Image.Image,
    *,
    input_size: int,
    min_num: int,
    max_num: int,
    use_thumbnail: bool,
) -> torch.Tensor:
    target_ratios = get_eagle2_5_vl_target_ratios(min_num, max_num)

    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess_eagle2_5_vl(
        image,
        target_ratios=target_ratios,
        image_size=input_size,
        use_thumbnail=use_thumbnail,
    )

    pixel_values = torch.stack([transform(image) for image in images])
    return pixel_values


# adapted from https://huggingface.co/nvidia/Eagle2.5-8B
def video_to_pixel_values_eagle2_5_vl(
    video: npt.NDArray,
    *,
    input_size: int,
    min_num: int,
    max_num: int,
    use_thumbnail: bool,
) -> torch.Tensor:
    target_ratios = get_eagle2_5_vl_target_ratios(min_num, max_num)

    transform = build_transform(input_size=input_size)
    frames_list = list[Image.Image]()
    for frame in video:
        pil_frame = dynamic_preprocess_eagle2_5_vl(
            Image.fromarray(frame, mode="RGB"),
            target_ratios=target_ratios,
            image_size=input_size,
            use_thumbnail=use_thumbnail,
        )
        assert len(pil_frame) == 1
        frames_list.extend(pil_frame)

    pixel_values = torch.stack([transform(image) for image in frames_list])
    return pixel_values


class BaseEagle2_5_VLProcessor(ABC):
    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: AnyTokenizer,
        *,
        min_dynamic_tiles: Optional[int] = None,
        max_dynamic_tiles: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer

        image_size: int = config.vision_config.image_size
        patch_size: int = config.vision_config.patch_size

        if min_dynamic_tiles is None:
            min_dynamic_tiles = config.min_dynamic_tiles
        assert isinstance(min_dynamic_tiles, int)

        if max_dynamic_tiles is None:
            max_dynamic_tiles = config.max_dynamic_tiles
        assert isinstance(max_dynamic_tiles, int)

        if dynamic_image_size is None:
            dynamic_image_size = config.dynamic_image_size
        assert isinstance(dynamic_image_size, bool)

        self.num_image_token = int(
            (image_size // patch_size)**2 * (config.downsample_ratio**2))
        self.image_size = image_size
        self.min_dynamic_tiles = min_dynamic_tiles
        self.max_dynamic_tiles = max_dynamic_tiles
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail: bool = config.use_thumbnail

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

    def resolve_min_max_num(
        self,
        *,
        min_dynamic_tiles: Optional[int] = None,
        max_dynamic_tiles: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
        use_thumbnail: Optional[bool] = None,
    ) -> tuple[int, int]:
        min_dynamic_tiles = (self.min_dynamic_tiles if min_dynamic_tiles
                             is None else min_dynamic_tiles)
        max_dynamic_tiles = (self.max_dynamic_tiles if max_dynamic_tiles
                             is None else max_dynamic_tiles)
        dynamic_image_size = (self.dynamic_image_size if dynamic_image_size
                              is None else dynamic_image_size)
        use_thumbnail = (self.use_thumbnail
                         if use_thumbnail is None else use_thumbnail)

        return resolve_eagle2_5_vl_min_max_num(
            min_dynamic_tiles=min_dynamic_tiles,
            max_dynamic_tiles=max_dynamic_tiles,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
        )

    def resolve_target_ratios(
        self,
        *,
        min_dynamic_tiles: Optional[int] = None,
        max_dynamic_tiles: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
        use_thumbnail: Optional[bool] = None,
    ) -> list[tuple[int, int]]:
        min_num, max_num = self.resolve_min_max_num(
            min_dynamic_tiles=min_dynamic_tiles,
            max_dynamic_tiles=max_dynamic_tiles,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
        )

        return get_eagle2_5_vl_target_ratios(min_num, max_num)

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        target_ratios = self.resolve_target_ratios(
            use_thumbnail=False,  # Applied in calculate_targets
        )

        num_patches, _, _ = calculate_eagle2_5_vl_targets(
            orig_width=image_width,
            orig_height=image_height,
            image_size=self.image_size,
            target_ratios=target_ratios,
            use_thumbnail=self.use_thumbnail,
        )

        return num_patches * self.num_image_token

    def _images_to_pixel_values_lst(
        self,
        images: list[Image.Image],
        min_dynamic_tiles: Optional[int] = None,
        max_dynamic_tiles: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
    ) -> list[torch.Tensor]:
        min_num, max_num = self.resolve_min_max_num(
            min_dynamic_tiles=min_dynamic_tiles,
            max_dynamic_tiles=max_dynamic_tiles,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=False,  # Applied in image_to_pixel_values
        )

        return [
            image_to_pixel_values_eagle2_5_vl(
                image,
                input_size=self.image_size,
                min_num=min_num,
                max_num=max_num,
                use_thumbnail=self.use_thumbnail,
            ) for image in images
        ]

    def _preprocess_image(
        self,
        text: list[str],
        images: list[Image.Image],
        min_dynamic_tiles: Optional[int] = None,
        max_dynamic_tiles: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
    ) -> tuple[list[str], dict[str, torch.Tensor]]:
        if len(images) == 0:
            image_inputs = {}
        else:
            pixel_values_lst = self._images_to_pixel_values_lst(
                images,
                min_dynamic_tiles=min_dynamic_tiles,
                max_dynamic_tiles=max_dynamic_tiles,
                dynamic_image_size=dynamic_image_size,
            )
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
        min_dynamic_tiles: Optional[int] = None,
        max_dynamic_tiles: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> Mapping[str, NestedTensors]:
        text, images = [self._make_batch_input(x) for x in (text, images)]

        text, image_inputs = self._preprocess_image(
            text=text,
            images=images,
            min_dynamic_tiles=min_dynamic_tiles,
            max_dynamic_tiles=max_dynamic_tiles,
            dynamic_image_size=dynamic_image_size,
        )

        text_inputs = self.tokenizer(text)

        return {
            **BatchEncoding(text_inputs, tensor_type=return_tensors),
            **image_inputs,
        }


class Eagle2_5_VLProcessor(BaseEagle2_5_VLProcessor):
    """
    HF Processor for Eagle2_5_VL with extended video processing logic.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: AnyTokenizer,
        *,
        min_dynamic_tiles: Optional[int] = None,
        max_dynamic_tiles: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
        video_token: Optional[str] = None,
    ) -> None:
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            min_dynamic_tiles=min_dynamic_tiles,
            max_dynamic_tiles=max_dynamic_tiles,
            dynamic_image_size=dynamic_image_size,
        )
        # add extra video token for video processing
        self.video_token = video_token

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.get_vocab()[IMG_CONTEXT]

    @property
    def video_token_id(self) -> Optional[int]:
        if self.video_token is None:
            return None
        return self.tokenizer.get_vocab().get(self.video_token, None)

    @property
    def supports_video(self) -> bool:
        return self.video_token_id is not None

    def _videos_to_pixel_values_lst(
        self,
        videos: list[npt.NDArray],
        dynamic_image_size: Optional[bool] = None,
    ) -> list[torch.Tensor]:
        min_num, max_num = self.resolve_min_max_num(
            min_dynamic_tiles=1,
            max_dynamic_tiles=1,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=False,  # Applied in image_to_pixel_values
        )

        return [
            video_to_pixel_values_eagle2_5_vl(
                video,
                input_size=self.image_size,
                min_num=min_num,
                max_num=max_num,
                use_thumbnail=False,
            ) for video in videos
        ]

    def _preprocess_video(
        self,
        text: list[str],
        videos: list[npt.NDArray],
        dynamic_image_size: Optional[bool] = None,
    ):
        if len(videos) == 0 or not self.supports_video:
            video_inputs = {}
        else:
            pixel_values_lst_video = self._videos_to_pixel_values_lst(
                videos,
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
        min_dynamic_tiles: Optional[int] = None,
        max_dynamic_tiles: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> Mapping[str, NestedTensors]:
        text, images, videos = [
            self._make_batch_input(x) for x in (text, images, videos)
        ]

        text, image_inputs = self._preprocess_image(
            text=text,
            images=images,
            min_dynamic_tiles=min_dynamic_tiles,
            max_dynamic_tiles=max_dynamic_tiles,
            dynamic_image_size=dynamic_image_size,
        )

        text, video_inputs = self._preprocess_video(
            text=text,
            videos=videos,
            dynamic_image_size=dynamic_image_size,
        )

        text_inputs = self.tokenizer(text)

        return {
            **BatchEncoding(text_inputs, tensor_type=return_tensors),
            **image_inputs,
            **video_inputs,
        }

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


class BaseEagle2_5_VLProcessingInfo(BaseProcessingInfo):
    """Basic image-only ProcessingInfo for Eagle2_5_VL-style models."""

    @abstractmethod
    def get_hf_processor(self, **kwargs: object) -> BaseEagle2_5_VLProcessor:
        raise NotImplementedError

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Optional[BaseEagle2_5_VLProcessor],
    ) -> int:
        if processor is None:
            processor = self.get_hf_processor()

        return processor.get_num_image_tokens(
            image_width=image_width,
            image_height=image_height,
        )

    def get_image_size_with_most_features(self) -> ImageSize:
        processor = self.get_hf_processor()

        base_size = processor.image_size
        target_ratios = processor.resolve_target_ratios()

        largest_feature_size, largest_feature_pinpoint = 0, None
        for wr, hr in target_ratios:
            width, height = base_size * wr, base_size * hr

            feat_size = self.get_num_image_tokens(
                image_width=width,
                image_height=height,
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
        target_width, target_height = self.get_image_size_with_most_features()

        return self.get_num_image_tokens(
            image_width=target_width,
            image_height=target_height,
            processor=processor,
        )


_I = TypeVar("_I", bound=BaseEagle2_5_VLProcessingInfo)


class BaseEagle2_5_VLDummyInputsBuilder(BaseDummyInputsBuilder[_I]):
    """Basic image-only DummyInputsBuilder for Eagle2_5_VL-style models."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        return "<image>" * num_images

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


class BaseEagle2_5_VLMultiModalProcessor(BaseMultiModalProcessor[_I]):
    """ Basic image-only MultiModalProcessor for Eagle2_5_VL-style models."""

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
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        out_mm_data = out_mm_kwargs.get_data()
        if "image_num_patches" in out_mm_data:
            image_num_patches = out_mm_data["image_num_patches"]
            assert isinstance(image_num_patches, torch.Tensor)
            image_num_patches = image_num_patches.tolist()
        elif "image_embeds" in out_mm_data:
            # TODO: Use image size information in dictionary embedding inputs
            # to compute num_patches (similar to Qwen2-VL)
            image_num_patches = [None] * len(out_mm_data["image_embeds"])
        else:
            image_num_patches = []

        def get_replacement_eagle2_5_vl(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems))

            if isinstance(images, ImageEmbeddingItems):
                feature_size = images.get_feature_size(item_idx)
            else:
                image_size = images.get_image_size(item_idx)
                feature_size = self.info.get_num_image_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                    processor=hf_processor,
                )

            num_patches = image_num_patches[item_idx]
            if num_patches is not None:
                assert isinstance(num_patches, int)

            return hf_processor.get_image_repl(feature_size, num_patches)

        return [
            PromptReplacement(
                modality="image",
                target="<image>",
                replacement=get_replacement_eagle2_5_vl,
            )
        ]


class Eagle2_5_VLProcessingInfo(BaseEagle2_5_VLProcessingInfo):
    """Eagle2_5_VL ProcessingInfo extended for video processing"""

    @property
    def supports_video(self):
        return self.get_hf_processor().supports_video

    def get_supported_mm_limits(self):
        video_limit = {"video": None} if self.supports_video else {}
        return {**super().get_supported_mm_limits(), **video_limit}

    def get_video_token(self) -> Optional[str]:
        return "<|video_pad|>"

    def get_num_frames_with_most_features(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        max_images = mm_counts.get("image", 0)
        max_videos = mm_counts.get("video", 0)

        processor = self.get_hf_processor()

        max_image_tokens = self.get_max_image_tokens() * max_images
        max_total_frames = (seq_len -
                            max_image_tokens) // processor.num_image_token
        max_frames_per_video = max_total_frames // max(max_videos, 1)

        return max(max_frames_per_video, 1)

    def get_hf_processor(self, **kwargs: object) -> Eagle2_5_VLProcessor:
        return self.ctx.init_processor(
            Eagle2_5_VLProcessor,
            config=self.get_hf_config(),
            tokenizer=self.get_tokenizer(),
            video_token=self.get_video_token(),
            **kwargs,
        )


class Eagle2_5_VLDummyInputsBuilder(
        BaseEagle2_5_VLDummyInputsBuilder[Eagle2_5_VLProcessingInfo]):
    """Eagle2_5_VL DummyInputsBuilder extended for video support"""

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
            image_size: int = config.vision_config.image_size
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


class Eagle2_5_VLMultiModalProcessor(
        BaseEagle2_5_VLMultiModalProcessor[Eagle2_5_VLProcessingInfo]):
    """Eagle2_5_VL MultiModalProcessor extended for video support"""

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
                    "video", num_videos),
            )
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

        def get_video_replacement_eagle2_5_vl(item_idx: int):
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
                    replacement=get_video_replacement_eagle2_5_vl,
                )
            ]

        return prompt_repl


@MULTIMODAL_REGISTRY.register_processor(
    Eagle2_5_VLMultiModalProcessor,
    info=Eagle2_5_VLProcessingInfo,
    dummy_inputs=Eagle2_5_VLDummyInputsBuilder)
class Eagle2_5_VLChatModel(nn.Module, SupportsMultiModal, SupportsPP,
                        SupportsLoRA):

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("image"):
            return "<image>"
        if modality.startswith("video"):
            return "<video>"

        raise ValueError("Only image or video modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config
        self._patch_quant_config(config, quant_config)

        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.num_image_token = int(
            (image_size // patch_size)**2 * (config.downsample_ratio**2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = "v2"

        self.llm_arch_name = config.text_config.architectures[0]
        self.is_mono = False
        vision_feature_layer = config.select_layer
        if vision_feature_layer < 0:
            num_hidden_layers = config.vision_config.num_hidden_layers \
                + vision_feature_layer + 1
        else:
            num_hidden_layers = vision_feature_layer + 1

        self.vision_model = SiglipVisionModel(config.vision_config, num_hidden_layers_override=num_hidden_layers)

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        self.mlp1 = self._init_mlp1(config)

        self.img_context_token_id = None
        self.video_context_token_id = None

        self.visual_token_mask = None
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    def _patch_quant_config(self, config: PretrainedConfig,
                            quant_config: QuantizationConfig):
        # the awq models from OpenGVLab missing `modules_to_not_convert`
        # patch the quant_config to add `modules_to_not_convert` back
        if isinstance(quant_config, AWQConfig):
            text_config = config.text_config
            llm_quant_config = getattr(text_config, "quantization_config",
                                       None)
            if (not quant_config.modules_to_not_convert) and \
                (llm_quant_config is not None):
                quant_config.modules_to_not_convert.append("vision_model")

    def _init_mlp1(self, config: PretrainedConfig) -> nn.Sequential:
        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.text_config.hidden_size

        return nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio)**2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio)**2,
                      llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            pass
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vit_embeds = self.vision_model(pixel_values=pixel_values)

        h = w = int(vit_embeds.shape[1]**0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds,
                                        scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1,
                                        vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[Eagle2_5_VLImageInputs]:
        pixel_values_flat = kwargs.pop("pixel_values_flat", None)
        image_num_patches = kwargs.pop("image_num_patches", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values_flat is None and image_embeds is None:
            return None

        if image_embeds is not None:
            if not isinstance(image_embeds, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")

            return Eagle2_5_VLImageEmbeddingInputs(
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
            expected_h = expected_w = self.config.vision_config.image_size
            resolve_bindings = {"h": expected_h, "w": expected_w}

            return Eagle2_5_VLImagePixelInputs(
                type="pixel_values",
                pixel_values_flat=pixel_values_flat,
                num_patches=image_num_patches,
                resolve_bindings=resolve_bindings,
            )

        raise AssertionError("This line should be unreachable.")

    def _parse_and_validate_video_input(
            self, **kwargs: object) -> Optional[Eagle2_5_VLVideoPixelInputs]:
        pixel_values_flat_video = kwargs.pop("pixel_values_flat_video", None)
        video_num_patches = kwargs.pop("video_num_patches", None)
        video_embeds = kwargs.pop("image_embeds", None)

        if pixel_values_flat_video is None and video_embeds is None:
            return None

        if video_embeds is not None:
            return Eagle2_5_VLVideoEmbeddingInputs(
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
            expected_h = expected_w = self.config.vision_config.image_size
            resolve_bindings = {"h": expected_h, "w": expected_w}

            return Eagle2_5_VLVideoPixelInputs(
                type="pixel_values_videos",
                pixel_values_flat=pixel_values_flat_video,
                num_patches=video_num_patches,
                resolve_bindings=resolve_bindings,
            )

        raise AssertionError("This line should be unreachable.")

    def _process_image_input(
        self,
        image_input: Union[Eagle2_5_VLImageInputs, Eagle2_5_VLVideoPixelInputs],
    ) -> tuple[torch.Tensor, ...]:
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

    def _set_visual_token_mask(self, input_ids: torch.Tensor) -> None:
        if self.is_mono:
            assert self.img_context_token_id is not None
            self.visual_token_mask = (
                input_ids == self.img_context_token_id).reshape(-1, 1)
        else:
            self.visual_token_mask = None

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(self,
                                  **kwargs: object) -> MultiModalEmbeddings:

        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not modalities:
            return []

        # The result multimodal_embeddings is tuple of tensors, with each
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
        if multimodal_embeddings is not None \
            and len(multimodal_embeddings) != 0:
            context_token_ids = [
                token_id for token_id in (self.img_context_token_id,
                                          self.video_context_token_id)
                if token_id is not None
            ]
            assert len(context_token_ids) >= 1
            self._set_visual_token_mask(input_ids)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                context_token_ids,
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> IntermediateTensors:

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

        forward_kwargs = {
            "input_ids": input_ids,
            "positions": positions,
            "intermediate_tensors": intermediate_tensors,
            "inputs_embeds": inputs_embeds,
        }

        # Only required if the model is mono-architecture
        if self.visual_token_mask is not None:
            forward_kwargs.update(
                {"visual_token_mask": self.visual_token_mask})
            self.visual_token_mask = None

        hidden_states = self.language_model.model(**forward_kwargs)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        # unused modules appear in OpenGVLab/InternVideo2_5_Chat_8B
        skip_prefixes = [
            "action_embed", "temporal_embed", "track_embed",
            "track_embed_decoder", "box_token", "cg_criterion", "cg_model",
            "loc_encoder", "loc_decoder", "sam", "temporal_token",
            "track_token"
        ]
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="mlp1",
            tower_model="vision_model")