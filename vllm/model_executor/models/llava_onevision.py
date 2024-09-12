import itertools
import math
from typing import (Iterable, List, Literal, Mapping, Optional, Tuple,
                    TypedDict, Union)

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import (CLIPVisionConfig, LlavaOnevisionConfig,
                          SiglipVisionConfig)
from transformers.models.llava_onevision.modeling_llava_onevision import (
    get_anyres_image_grid_shape, unpad_image)
from typing_extensions import NotRequired

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, MultiModalConfig
from vllm.inputs import INPUT_REGISTRY, InputContext, LLMInputs
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.utils import (cached_get_tokenizer,
                                   repeat_and_pad_placeholder_tokens)
from vllm.sequence import IntermediateTensors
from vllm.utils import is_list_of

from .clip import (CLIPVisionModel, dummy_image_for_clip,
                   dummy_seq_data_for_clip, get_clip_image_feature_size,
                   get_clip_patch_grid_length, input_processor_for_clip)

from .interfaces import SupportsMultiModal
from .siglip import (SiglipVisionModel, dummy_image_for_siglip,
                     dummy_seq_data_for_siglip, get_siglip_image_feature_size,
                     get_siglip_patch_grid_length, input_processor_for_siglip)
from .utils import (filter_weights, flatten_bn, init_vllm_registered_model,
                    merge_multimodal_embeddings)

logger = init_logger(__name__)

# Result in the max possible feature size (2x2 grid of 336x336px tiles)
MAX_IMAGE_FEATURE_SIZE_HEIGHT = MAX_IMAGE_FEATURE_SIZE_WIDTH = 448

# For profile run
_MAX_FRAMES_PER_VIDEO = 16
_MAX_NUM_VIDEOS = 1


class LlavaOnevisionVideoPixelInputs(TypedDict):
    type: Literal["pixel_values_videos"]
    data: Union[torch.Tensor, List[torch.Tensor]]
    """
    Shape: `(batch_size, num_frames, num_channels, height, width)`

    Note that `num_frames` may be different for each batch, in which case
    the data is passed as a list instead of a batched tensor.

    Note that it only supports one video input for one batch.
    """


class LlavaOnevisionImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: Union[torch.Tensor, List[torch.Tensor]]
    """
    Shape:
    `(batch_size * num_images, 1 + num_patches, num_channels, height, width)`

    Note that `num_patches` may be different per batch and image,
    in which case the data is passed as a list instead of a batched tensor.
    """

    image_sizes: NotRequired[torch.Tensor]
    """
    Shape: `(batch_size * num_images, 2)`

    This should be in `(height, width)` format.
    """


class LlavaOnevisionImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: torch.Tensor
    """Shape: `(batch_size * num_images, image_feature_size, hidden_size)`

    `hidden_size` must match the hidden size of language model backbone.
    """


LlavaOnevisionImageInputs = Union[LlavaOnevisionImagePixelInputs,
                                  LlavaOnevisionImageEmbeddingInputs]

LlavaOnevisionMultiInputs = Union[LlavaOnevisionImageInputs,
                                  LlavaOnevisionVideoPixelInputs]


def _get_llava_next_num_unpadded_features(
    original_height: int,
    original_width: int,
    npatches: int,
    num_patch_height: int,
    num_patch_width: int,
) -> Tuple[int, int]:
    current_height = npatches * num_patch_height
    current_width = npatches * num_patch_width

    aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if aspect_ratio > current_aspect_ratio:
        new_height = (original_height * current_width) // original_width
        padding = (current_height - new_height) // 2
        current_height -= padding * 2
    else:
        new_width = (original_width * current_height) // original_height
        padding = (current_width - new_width) // 2
        current_width -= padding * 2

    unpadded_features = current_height * current_width
    newline_features = current_height
    return (unpadded_features, newline_features)


def get_llava_onevision_image_feature_size(
    hf_config: LlavaOnevisionConfig,
    *,
    input_height: int,
    input_width: int,
) -> int:
    vision_config = hf_config.vision_config

    if isinstance(vision_config, CLIPVisionConfig):
        num_patches = get_clip_patch_grid_length(
            image_size=vision_config.image_size,
            patch_size=vision_config.patch_size,
        )
        base_feature_size = get_clip_image_feature_size(vision_config)
    elif isinstance(vision_config, SiglipVisionConfig):
        num_patches = get_siglip_patch_grid_length(
            image_size=vision_config.image_size,
            patch_size=vision_config.patch_size,
        )
        base_feature_size = get_siglip_image_feature_size(vision_config)
    else:
        msg = f"Unsupported vision config: {type(vision_config)}"
        raise NotImplementedError(msg)

    strategy = hf_config.vision_feature_select_strategy
    if strategy == "default":
        base_feature_size -= 1
    elif strategy == "full":
        pass
    else:
        raise ValueError(f"Unexpected select feature strategy: {strategy}")

    num_patch_height, num_patch_width = get_anyres_image_grid_shape(
        image_size=(input_height, input_width),
        grid_pinpoints=hf_config.image_grid_pinpoints,
        patch_size=vision_config.image_size,
    )

    (
        unpadded_feature_size,
        newline_feature_size,
    ) = _get_llava_next_num_unpadded_features(input_height, input_width,
                                              num_patches, num_patch_height,
                                              num_patch_width)

    return unpadded_feature_size + newline_feature_size + base_feature_size


def get_max_llava_onevision_image_tokens(ctx: InputContext):
    return get_llava_onevision_image_feature_size(
        ctx.get_hf_config(LlavaOnevisionConfig),
        input_height=MAX_IMAGE_FEATURE_SIZE_HEIGHT,
        input_width=MAX_IMAGE_FEATURE_SIZE_WIDTH,
    )


def get_llava_onevision_video_frame_feature_size(
        hf_config: LlavaOnevisionConfig) -> int:
    # Support both CLIPVisionConfig and SiglipVisionConfig
    image_size = hf_config.vision_config.image_size
    patch_size = hf_config.vision_config.patch_size
    spatial_pool_stride = hf_config.spatial_pool_stride if hasattr(
        hf_config, "spatial_pool_stride") else 2

    height = width = image_size // patch_size
    return math.ceil(height / spatial_pool_stride) * math.ceil(
        width / spatial_pool_stride)


def _get_max_llm_tokens(ctx: InputContext) -> int:
    """
    Calculated from the maximum video frames under the context length
    constraints of the language model.
    """
    hf_text_config = ctx.model_config.hf_text_config
    model_config = ctx.model_config
    max_tokens = model_config.max_model_len
    rope_scaling = model_config.rope_scaling

    if rope_scaling:
        rope_scaling_factor = hf_text_config.rope_scaling["factor"]
    else:
        rope_scaling_factor = 1

    max_tokens *= rope_scaling_factor

    return max_tokens


def get_llava_onevision_video_tokens(ctx: InputContext, frames: int) -> int:
    hf_config = ctx.get_hf_config(LlavaOnevisionConfig)
    vision_config = hf_config.vision_config

    # TODO: support configuring (not supported by HF right now)
    num_token_image_newline = 1
    tokens_per_frame = get_llava_onevision_video_frame_feature_size(hf_config)
    video_feature_size = frames * tokens_per_frame + num_token_image_newline

    return video_feature_size


def get_max_llava_onevision_video_tokens(ctx: InputContext) -> int:
    return get_llava_onevision_video_tokens(ctx, _MAX_FRAMES_PER_VIDEO)


def dummy_data_for_llava_onevision(ctx: InputContext, seq_len: int,
                                   mm_counts: Mapping[str, int]):
    hf_config = ctx.get_hf_config(LlavaOnevisionConfig)
    vision_config = hf_config.vision_config

    # TODO: support multiple videos
    num_videos = mm_counts["video"]
    if num_videos != _MAX_NUM_VIDEOS:
        raise NotImplementedError(
            f"Only {_MAX_NUM_VIDEOS} videos are supported")

    # TODO: support configuring the number of frames
    frames_per_video = _MAX_FRAMES_PER_VIDEO
    video_feature_size = get_llava_onevision_video_tokens(
        ctx, frames_per_video)

    if isinstance(vision_config, CLIPVisionConfig):
        seq_data = dummy_seq_data_for_clip(
            vision_config,
            seq_len,
            num_videos,
            image_token_id=hf_config.video_token_index,
            image_feature_size_override=video_feature_size,
        )

        pil_frame = dummy_image_for_clip(vision_config, num_images=1)
        np_frame = np.array(pil_frame["image"])
        mm_data_per_video = np.repeat([np_frame], frames_per_video, axis=0)
        mm_data = {"video": mm_data_per_video}
        return seq_data, mm_data
    elif isinstance(vision_config, SiglipVisionConfig):
        seq_data = dummy_seq_data_for_siglip(
            vision_config,
            seq_len,
            num_videos,
            image_token_id=hf_config.video_token_index,
            image_feature_size_override=video_feature_size,
        )

        pil_frame = dummy_image_for_siglip(vision_config, num_images=1)
        np_frame = np.array(pil_frame["image"])
        mm_data_per_video = np.repeat([np_frame], frames_per_video, axis=0)
        mm_data = {"video": mm_data_per_video}
        return seq_data, mm_data

    msg = f"Unsupported vision config: {type(vision_config)}"
    raise NotImplementedError(msg)


def image_processor_for_llava_onevision(ctx: InputContext,
                                        llm_inputs: LLMInputs):
    multi_modal_data = llm_inputs.get("multi_modal_data")
    if multi_modal_data is None or "image" not in multi_modal_data:
        return llm_inputs

    model_config = ctx.model_config
    hf_config = ctx.get_hf_config(LlavaOnevisionConfig)
    vision_config = hf_config.vision_config

    image_data = multi_modal_data["image"]
    if isinstance(image_data, Image.Image):
        width, height = image_data.size

        image_feature_size = get_llava_onevision_image_feature_size(
            hf_config,
            input_height=height,
            input_width=width,
        )
    elif is_list_of(image_data, Image.Image):
        image_feature_size = [
            get_llava_onevision_image_feature_size(hf_config,
                                                   input_height=img.height,
                                                   input_width=img.width)
            for img in image_data
        ]
    elif isinstance(image_data, torch.Tensor):
        num_images, image_feature_size, hidden_size = image_data.shape
    elif is_list_of(image_data, torch.Tensor):
        image_feature_size = [item.shape[1] for item in image_data]
    else:
        raise TypeError(f"Invalid image type: {type(image_data)}")

    vision_config = hf_config.vision_config

    if isinstance(vision_config, CLIPVisionConfig):
        return input_processor_for_clip(
            model_config,
            vision_config,
            llm_inputs,
            image_token_id=hf_config.image_token_index,
            image_feature_size_override=image_feature_size,
        )
    elif isinstance(vision_config, SiglipVisionConfig):
        return input_processor_for_siglip(
            model_config,
            vision_config,
            llm_inputs,
            image_token_id=hf_config.image_token_index,
            image_feature_size_override=image_feature_size,
        )

    msg = f"Unsupported vision config: {type(vision_config)}"
    raise NotImplementedError(msg)


def video_processor_for_llava_onevision(ctx: InputContext,
                                        llm_inputs: LLMInputs):
    multi_modal_data = llm_inputs.get("multi_modal_data")
    if multi_modal_data is None or "video" not in multi_modal_data:
        return llm_inputs
    video_data = multi_modal_data["video"]

    model_config = ctx.model_config
    hf_config = ctx.get_hf_config(LlavaOnevisionConfig)
    vision_config = hf_config.vision_config

    if isinstance(video_data, np.ndarray):
        # Supports both CLIP and Siglip
        num_frames = video_data.shape[0]
        video_feature_size = get_llava_onevision_video_tokens(ctx, num_frames)
        tokenizer = cached_get_tokenizer(model_config.tokenizer)

        new_prompt, new_token_ids = repeat_and_pad_placeholder_tokens(
            tokenizer,
            llm_inputs.get("prompt"),
            llm_inputs["prompt_token_ids"],
            placeholder_token_id=hf_config.video_token_index,
            repeat_count=video_feature_size,
        )

        return LLMInputs(prompt_token_ids=new_token_ids,
                         prompt=new_prompt,
                         multi_modal_data=multi_modal_data)

    elif is_list_of(video_data, np.ndarray):
        raise NotImplementedError(
            "Processing multiple videos is not supported")

    msg = f"Unsupported vision config: {type(vision_config)}"
    raise NotImplementedError(msg)


def input_processor_for_llava_onevision(ctx: InputContext,
                                        llm_inputs: LLMInputs):
    multi_modal_data = llm_inputs.get("multi_modal_data")
    if multi_modal_data is None or ("video" not in multi_modal_data
                                    and "image" not in multi_modal_data):
        return llm_inputs
    if "image" in multi_modal_data:
        return image_processor_for_llava_onevision(ctx, llm_inputs)
    if "video" in multi_modal_data:
        return video_processor_for_llava_onevision(ctx, llm_inputs)

    msg = f"Unsupported multi data type"
    raise NotImplementedError(msg)


def _init_vision_tower(hf_config: LlavaOnevisionConfig):
    vision_config = hf_config.vision_config

    # Initialize the vision tower only up to the required feature layer
    vision_feature_layer = hf_config.vision_feature_layer
    if vision_feature_layer < 0:
        num_hidden_layers = hf_config.vision_config.num_hidden_layers \
            + vision_feature_layer + 1
    else:
        num_hidden_layers = vision_feature_layer + 1

    if isinstance(vision_config, CLIPVisionConfig):
        return CLIPVisionModel(
            vision_config,
            num_hidden_layers_override=num_hidden_layers,
        )
    elif isinstance(vision_config, SiglipVisionConfig):
        return SiglipVisionModel(
            vision_config,
            num_hidden_layers_override=num_hidden_layers,
        )

    msg = f"Unsupported vision config: {type(vision_config)}"
    raise NotImplementedError(msg)


class LlavaOnevisionMultiModalProjector(nn.Module):

    def __init__(self, config: LlavaOnevisionConfig):
        super().__init__()

        self.linear_1 = nn.Linear(config.vision_config.hidden_size,
                                  config.text_config.hidden_size,
                                  bias=True)
        self.act = get_act_fn(config.projector_hidden_act)
        self.linear_2 = nn.Linear(config.text_config.hidden_size,
                                  config.text_config.hidden_size,
                                  bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


@MULTIMODAL_REGISTRY.register_image_input_mapper()
@MULTIMODAL_REGISTRY.register_input_mapper("video")
@MULTIMODAL_REGISTRY.register_max_multimodal_tokens(
    "image", get_max_llava_onevision_image_tokens)
@MULTIMODAL_REGISTRY.register_max_multimodal_tokens(
    "video", get_max_llava_onevision_video_tokens)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_llava_onevision)
@INPUT_REGISTRY.register_input_processor(input_processor_for_llava_onevision)
class LlavaOnevisionForConditionalGeneration(nn.Module, SupportsMultiModal):

    def __init__(self,
                 config: LlavaOnevisionConfig,
                 multimodal_config: MultiModalConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__()

        self.config = config
        self.multimodal_config = multimodal_config

        # Initialize the vision tower only up to the required feature layer
        self.vision_tower = _init_vision_tower(config)
        self.multi_modal_projector = LlavaOnevisionMultiModalProjector(config)
        self.language_model = init_vllm_registered_model(
            config.text_config, cache_config, quant_config)
        self.image_newline = nn.Parameter(
            torch.empty(config.text_config.hidden_size))

    def _validate_image_sizes(self, data: torch.Tensor) -> torch.Tensor:
        expected_dims = (2, )

        def _validate_shape(d: torch.Tensor):
            actual_dims = tuple(d.shape)

            if actual_dims != expected_dims:
                expected_expr = str(expected_dims)
                raise ValueError(
                    f"The expected shape of image sizes per image per batch "
                    f"is {expected_expr}. You supplied {tuple(d.shape)}.")

        for d in data:
            _validate_shape(d)

        return data

    def _validate_image_pixel_values(
        self, data: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:

        h = w = self.config.vision_config.image_size
        expected_dims = (3, h, w)

        def _validate_shape(d: torch.Tensor):
            actual_dims = tuple(d.shape[1:])

            if actual_dims != expected_dims:
                expected_expr = ("num_patches", *map(str, expected_dims))
                raise ValueError(
                    "The expected shape of pixel values per image per batch "
                    f"is {expected_expr}. You supplied {tuple(d.shape)}.")

        for d in data:
            _validate_shape(d)

        return data

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[LlavaOnevisionImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_sizes = kwargs.pop("image_sizes", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            if not isinstance(image_sizes, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image sizes. "
                                 f"Got type: {type(image_sizes)}")

            return LlavaOnevisionImagePixelInputs(
                type="pixel_values",
                data=self._validate_image_pixel_values(
                    flatten_bn(pixel_values)),
                image_sizes=self._validate_image_sizes(
                    flatten_bn(image_sizes, concat=True)),
            )

        if image_embeds is not None:
            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError("Incorrect type of image embeds. "
                                 f"Got type: {type(image_embeds)}")

            return LlavaOnevisionImageEmbeddingInputs(
                type="image_embeds",
                data=flatten_bn(image_embeds),
            )

        raise AssertionError("This line should be unreachable.")

    def _validate_video_pixel_values(
        self, data: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:

        h = w = self.config.vision_config.image_size
        expected_dims = (3, h, w)

        def _validate_shape(d: torch.Tensor):
            actual_dims = tuple(d.shape[2:])

            if actual_dims != expected_dims:
                expected_expr = ("num_frames", *map(str, expected_dims))
                raise ValueError(
                    "The expected shape of pixel values in each video frame "
                    f"is {expected_expr}. You supplied {tuple(d.shape)}.")

        for d in data:
            _validate_shape(d)

        return data

    def _parse_and_validate_video_input(
            self,
            **kwargs: object) -> Optional[LlavaOnevisionVideoPixelInputs]:
        """
        A legal video input should have the following dimensions:
        {
            "pixel_values_videos" : 
                List[b, Tensor(nb_frames, nb_channels, height, width)]
        }
        """
        pixel_values = kwargs.pop("pixel_values_videos", None)

        if pixel_values is None:
            return None

        if not (is_list_of(pixel_values,
                           (torch.Tensor))  # different shape videos 
                or isinstance(pixel_values,
                              torch.Tensor)):  # same shape videos
            raise ValueError("Incorrect type of pixel values. "
                             f"Got type: {type(pixel_values)}")

        return LlavaOnevisionVideoPixelInputs(
            type="pixel_values_videos",
            data=pixel_values,
        )

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}

        if "pixel_values" in kwargs:
            modalities["images"] = self._parse_and_validate_image_input(
                **kwargs)

        if "pixel_values_videos" in kwargs:
            modalities["videos"] = self._parse_and_validate_video_input(
                **kwargs)

        return modalities

    def _select_image_features(self, image_features: torch.Tensor, *,
                               strategy: str) -> torch.Tensor:
        if strategy == "default":
            return image_features[:, 1:]
        elif strategy == "full":
            return image_features

        raise ValueError(f"Unexpected select feature strategy: {strategy}")

    def _image_pixels_to_features(
        self,
        vision_tower: Union[CLIPVisionModel, SiglipVisionModel],
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:

        # NOTE: we skip the step to select the vision feature layer since
        # this is already done inside the vision tower
        import pdb
        pdb.set_trace()
        import numpy
        float32_tensor = pixel_values.to(torch.float32).cpu().reshape(-1)
        numpy.savetxt('vllm_bfvisiontower.txt',
                      float32_tensor.numpy(),
                      fmt='%.6f')
        image_features = vision_tower(pixel_values)
        import numpy
        float32_tensor = image_features.to(torch.float32).cpu().reshape(-1)
        numpy.savetxt('vllm_visiontower.txt',
                      float32_tensor.numpy(),
                      fmt='%.6f')
        return self._select_image_features(
            image_features,
            strategy=self.config.vision_feature_select_strategy,
        )

    # Based on: https://github.com/haotian-liu/LLaVA/blob/main/llava/model/llava_arch.py
    def _merge_image_patch_embeddings(self,
                                      image_size: torch.Tensor,
                                      patch_embeddings: torch.Tensor,
                                      *,
                                      image_newline=None,
                                      vision_aspect_ratio="anyres_max_9",
                                      strategy: str) -> torch.Tensor:
        if strategy == "flat":
            return patch_embeddings.flatten(0, 1)

        if strategy.startswith("spatial"):
            height = width = self.config.vision_config.image_size \
                // self.config.vision_config.patch_size

            base_patch_embeds = patch_embeddings[0]
            if height * width != base_patch_embeds.shape[0]:
                raise ValueError(
                    "The number of patches is not consistent with the "
                    "image size.")

            if patch_embeddings.shape[0] > 1:
                other_patch_embeds = patch_embeddings[1:]

                # Move to CPU to avoid floating-point errors
                orig_height, orig_width = image_size.tolist()

                # image_aspect_ratio == "anyres"
                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    (orig_height, orig_width),
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )
                num_patches = num_patch_height * num_patch_width

                # Image patches might be padded for batch processing
                other_patch_embeds = other_patch_embeds[:num_patches] \
                    .view(num_patch_height, num_patch_width, height, width, -1)

                if "unpad" in strategy:
                    other_patch_embeds = other_patch_embeds \
                        .permute(4, 0, 2, 1, 3).contiguous() \
                        .flatten(1, 2).flatten(2, 3)
                    other_patch_embeds = unpad_image(other_patch_embeds,
                                                     (orig_height, orig_width))
                    max_num_patches = int(
                        vision_aspect_ratio.strip("anyres_max_"))
                    channels, curr_height, curr_width = other_patch_embeds.shape
                    ratio = math.sqrt(curr_height * curr_width /
                                      (max_num_patches * height**2))
                    if ratio > 1.1:
                        other_patch_embeds = other_patch_embeds[None]
                        other_patch_embeds = nn.functional.interpolate(
                            other_patch_embeds, [
                                int(curr_height // ratio),
                                int(curr_width // ratio)
                            ],
                            mode="bilinear")[0]
                    if image_newline is not None:
                        other_patch_embeds = torch.cat(
                            (
                                other_patch_embeds,
                                image_newline[:, None, None] \
                                .expand(*other_patch_embeds.shape[:-1], 1) \
                                .to(other_patch_embeds.device, other_patch_embeds.dtype),
                            ),
                        dim=-1)
                    other_patch_embeds = other_patch_embeds \
                        .flatten(1, 2).transpose(0, 1)
                else:
                    other_patch_embeds = other_patch_embeds \
                        .permute(0, 2, 1, 3, 4).contiguous() \
                        .flatten(0, 3)

                merged_patch_embeddings = torch.cat(
                    (base_patch_embeds, other_patch_embeds), dim=0)
            else:
                if "unpad" in strategy:
                    merged_patch_embeddings = torch.cat(
                        (base_patch_embeds,
                         self.image_newline[None] \
                            .to(base_patch_embeds.device)
                    ), dim=0)
                else:
                    merged_patch_embeddings = base_patch_embeds

            return merged_patch_embeddings

        raise ValueError(f"Unexpected patch merge strategy: {strategy}")

    def _process_image_pixels(
        self,
        inputs: LlavaOnevisionImagePixelInputs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        assert self.vision_tower is not None

        pixel_values = inputs["data"]

        if isinstance(pixel_values, torch.Tensor):
            b, num_patches, c, h, w = pixel_values.shape
            stacked_pixel_values = pixel_values.view(b * num_patches, c, h, w)
            stacked_image_features = self._image_pixels_to_features(
                self.vision_tower, stacked_pixel_values)
            stacked_patch_embeddings = self.multi_modal_projector(
                stacked_image_features)

            return stacked_patch_embeddings.view(
                b, num_patches, *stacked_patch_embeddings.shape[1:])

        num_patches_per_batch = [v.shape[0] for v in pixel_values]
        stacked_pixel_values = torch.cat(pixel_values)
        stacked_image_features = self._image_pixels_to_features(
            self.vision_tower, stacked_pixel_values)

        return [
            self.multi_modal_projector(image_features) for image_features in
            torch.split(stacked_image_features, num_patches_per_batch)
        ]

    def _process_image_input(
        self,
        image_input: LlavaOnevisionImageInputs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if image_input["type"] == "image_embeds":
            return [image_input["data"]]

        patch_embeddings = self._process_image_pixels(image_input)

        image_sizes = image_input.get("image_sizes")
        if image_sizes is None:
            batch_size = len(image_input["data"])
            vision_config = self.config.vision_config
            default_height = default_width = vision_config.image_size
            image_sizes = torch.as_tensor([[default_height, default_width]
                                           for _ in range(batch_size)])

        return [
            self._merge_image_patch_embeddings(
                image_sizes[i],
                patch_features_batch,
                image_newline=self.image_newline,
                strategy="spatial_unpad")
            for i, patch_features_batch in enumerate(patch_embeddings)
        ]

    def _video_pixels_to_features(
        self,
        vision_tower: Union[CLIPVisionModel, SiglipVisionModel],
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:

        # NOTE: we skip the step to select the vision feature layer since
        # this is already done inside the vision tower
        batch_size, num_videos, frames, channles, height, width = pixel_values.shape
        assert (num_videos == 1)
        pixel_values = pixel_values.reshape(batch_size * num_videos * frames,
                                            channles, height, width)
        video_features = vision_tower(pixel_values)
        video_features = self._select_image_features(
            video_features,
            strategy=self.config.vision_feature_select_strategy,
        )
        video_features = self.multi_modal_projector(video_features)
        video_features = self.apply_pooling(video_features)
        video_features = video_features.reshape(
            batch_size, frames * video_features.shape[1], -1)
        image_newline = self.image_newline[None, None, :].repeat(
            batch_size, 1, 1).to(video_features.device)
        video_features = torch.cat((video_features, image_newline), dim=1)
        video_features = video_features.flatten(0, 1)

        return video_features

    def _process_video_pixels(self, inputs: LlavaOnevisionVideoPixelInputs):
        assert self.vision_tower is not None

        video_pixels = inputs["data"]

        if isinstance(video_pixels, torch.Tensor):
            # TODO: support multiple videos per input
            stacked_embeddings = self._video_pixels_to_features(
                self.vision_tower, video_pixels)
            return stacked_embeddings

        elif is_list_of(video_pixels, torch.Tensor):
            frames_per_videos = [v.shape[0] for v in video_pixels]
            stacked_pixels = torch.cat(video_pixels, dim=0)
            stacked_embeddings = self._video_pixels_to_features(
                self.vision_tower, stacked_pixels)
            return torch.split(stacked_embeddings, frames_per_videos, dim=0)

        else:
            raise ValueError(
                f"Unsupported type of video input {type(video_pixels)}")

    def apply_pooling(self, image_features, stride=2):
        height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size
        batch_frames, _, dim = image_features.shape
        image_features = image_features.view(batch_frames, height, width, -1)
        image_features = image_features.permute(0, 3, 1, 2)

        # TODO support other pooling types config
        height, width = image_features.shape[2:]
        scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
        image_feature = nn.functional.interpolate(image_features,
                                                  size=scaled_shape,
                                                  mode='bilinear')
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(batch_frames, -1, dim)
        return image_feature

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: object,
    ) -> SamplerOutput:
        """Run forward pass for LlaVA-Onevision.
        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            pixel_values_videos: Pixels in each frames for each input videos.
        """
        # import pdb; pdb.set_trace()

        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        # merge video embeddings into input embeddings
        if modalities:
            inputs_embeds = self.language_model.model.get_input_embeddings(
                input_ids)
            if "images" in modalities:
                import pdb
                pdb.set_trace()
                image_input = modalities["images"]
                vision_embeddings = self._process_image_input(image_input)
                inputs_embeds = merge_multimodal_embeddings(
                    input_ids, inputs_embeds, vision_embeddings,
                    self.config.image_token_index)
                float32_tensor = inputs_embeds.to(torch.float32).cpu()
                import numpy
                numpy.savetxt('vllm.txt', float32_tensor, fmt='%.6f')
                import numpy
                ids = input_ids.cpu().numpy()
                numpy.savetxt('vllm_ids.txt', ids, fmt='%d')
            if "videos" in modalities:
                video_input = modalities["videos"]
                video_embeddings = self._process_video_pixels(video_input)
                inputs_embeds = merge_multimodal_embeddings(
                    input_ids, inputs_embeds, video_embeddings,
                    self.config.video_token_index)
            input_ids = None
        else:
            inputs_embeds = None

        hidden_states = self.language_model.model(input_ids,
                                                  positions,
                                                  kv_caches,
                                                  attn_metadata,
                                                  None,
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

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # prepare weight iterators
        vit_weights, mlp_weights, newline_weights, llm_weights = itertools.tee(
            weights, 4)

        # load vision encoder
        vit_weights = filter_weights(vit_weights, "vision_tower")
        self.vision_tower.load_weights(vit_weights)

        # load mlp projector
        mlp_weights = filter_weights(mlp_weights, "multi_modal_projector")
        mlp_params_dict = dict(self.multi_modal_projector.named_parameters())
        for name, loaded_weight in mlp_weights:
            param = mlp_params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)

        # load llm backbone
        llm_weights = filter_weights(llm_weights, "language_model")
        self.language_model.load_weights(llm_weights)
