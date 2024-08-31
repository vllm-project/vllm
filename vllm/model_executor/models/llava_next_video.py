import itertools
import math
from typing import (Iterable, List, Literal, Mapping, Optional, Tuple,
                    TypedDict, Union)

import numpy as np
import torch
import torch.nn as nn
from transformers import (CLIPVisionConfig, LlavaNextVideoConfig,
                          SiglipVisionConfig)

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, MultiModalConfig
from vllm.inputs import INPUT_REGISTRY, InputContext, LLMInputs
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.clip import CLIPVisionModel
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.utils import (cached_get_tokenizer,
                                   repeat_and_pad_placeholder_tokens)
from vllm.sequence import IntermediateTensors
from vllm.utils import is_list_of

from .clip import dummy_image_for_clip, dummy_seq_data_for_clip
from .interfaces import SupportsMultiModal
from .siglip import (SiglipVisionModel, dummy_image_for_siglip,
                     dummy_seq_data_for_siglip)
from .utils import (filter_weights, init_vllm_registered_model,
                    merge_multimodal_embeddings)

logger = init_logger(__name__)

# For profile run
_MAX_FRAMES_PER_VIDEO = 32
_MAX_NUM_VIDEOS = 1


class LlavaNextVideoPixelInputs(TypedDict):
    type: Literal["pixel_values_videos"]
    data: Union[torch.Tensor, List[torch.Tensor]]
    """
    Shape: `(batch_size, num_frames, num_channels, height, width)`

    Note that `num_frames` may be different for each batch, in which case
    the data is passed as a list instead of a batched tensor.

    Note that it only supports one video input for one batch.
    """


def get_llava_next_video_frame_feature_size(
        hf_config: LlavaNextVideoConfig) -> int:
    # Support both CLIPVisionConfig and SiglipVisionConfig
    image_size = hf_config.vision_config.image_size
    patch_size = hf_config.vision_config.patch_size
    spatial_pool_stride = hf_config.spatial_pool_stride

    return int((image_size / patch_size / spatial_pool_stride)**2)


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


def get_max_llava_next_video_tokens(ctx: InputContext) -> int:
    # Currently set to 32 frames
    # TODO: max_tokens = _get_max_llm_tokens(ctx)
    hf_config = ctx.get_hf_config(LlavaNextVideoConfig)
    tokens_per_frame = get_llava_next_video_frame_feature_size(hf_config)
    return _MAX_FRAMES_PER_VIDEO * tokens_per_frame


def dummy_data_for_llava_next_video(ctx: InputContext, seq_len: int,
                                    mm_counts: Mapping[str, int]):
    hf_config = ctx.get_hf_config(LlavaNextVideoConfig)
    vision_config = hf_config.vision_config

    # TODO: support multiple videos
    num_videos = mm_counts["video"]
    if num_videos != _MAX_NUM_VIDEOS:
        raise NotImplementedError(
            f"Only {_MAX_NUM_VIDEOS} videos are supported")

    # TODO: support configuring the number of frames
    frames_per_video = _MAX_FRAMES_PER_VIDEO
    # num_images = num_videos * frames_per_video

    # fills the sequence with as longer video data as possible
    tokens_per_frame = get_llava_next_video_frame_feature_size(hf_config)
    video_feature_size = frames_per_video * tokens_per_frame

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


def input_processor_for_llava_next_video(ctx: InputContext,
                                         llm_inputs: LLMInputs):
    multi_modal_data = llm_inputs.get("multi_modal_data")
    if multi_modal_data is None or "video" not in multi_modal_data:
        return llm_inputs
    video_data = multi_modal_data["video"]

    model_config = ctx.model_config
    hf_config = ctx.get_hf_config(LlavaNextVideoConfig)
    vision_config = hf_config.vision_config

    if isinstance(video_data, np.ndarray):
        # Supports both CLIP and Siglip
        num_frames = video_data.shape[0]
        frame_feature_size = \
            get_llava_next_video_frame_feature_size(hf_config)
        video_feature_size = num_frames * frame_feature_size

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


def _init_vision_tower(hf_config: LlavaNextVideoConfig):
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


# adopted from transformers modeling_llava_next_video.py
class LlavaNextVideoPooler(nn.Module):

    def __init__(self, config):
        super().__init__()

        mode = config.spatial_pool_mode
        stride = config.spatial_pool_stride
        image_size = config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.image_size = image_size // patch_size**2

        if mode == "average":
            self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride)
        elif mode == "max":
            self.pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
        else:
            # TODO: Support Conv2d pooling layer, need to load weights
            raise ValueError(
                f"Unknown pooling mode: {mode}. Expected [`average`, `max`]")

    def forward(self, image_features):
        ori_width = int(
            math.sqrt(image_features.shape[1] * self.image_size //
                      self.image_size))
        ori_height = int(ori_width * self.image_size // self.image_size)

        batch_size, _, dim = image_features.shape
        image_features_spatial = image_features \
            .view(batch_size, ori_height, ori_height, dim) \
            .permute(0, 3, 1, 2)
        image_features_spatial = self.pool(image_features_spatial)

        return image_features_spatial.flatten(2).transpose(1, 2).contiguous()


class LlavaNextMultiModalProjector(nn.Module):

    def __init__(self, vision_hidden_size: int, text_hidden_size: int,
                 projector_hidden_act: str):
        super().__init__()

        self.linear_1 = nn.Linear(vision_hidden_size,
                                  text_hidden_size,
                                  bias=True)
        self.act = get_act_fn(projector_hidden_act)
        self.linear_2 = nn.Linear(text_hidden_size,
                                  text_hidden_size,
                                  bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


@MULTIMODAL_REGISTRY.register_input_mapper("video")
@MULTIMODAL_REGISTRY.register_max_multimodal_tokens(
    "video", get_max_llava_next_video_tokens)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_llava_next_video)
@INPUT_REGISTRY.register_input_processor(input_processor_for_llava_next_video)
class LlavaNextVideoForConditionalGeneration(nn.Module, SupportsMultiModal):

    def __init__(self,
                 config: LlavaNextVideoConfig,
                 multimodal_config: MultiModalConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__()

        self.config = config
        self.multimodal_config = multimodal_config

        # Initialize the vision tower only up to the required feature layer
        self.vision_tower = _init_vision_tower(config)
        self.multi_modal_projector = LlavaNextMultiModalProjector(
            vision_hidden_size=config.vision_config.hidden_size,
            text_hidden_size=config.text_config.hidden_size,
            projector_hidden_act=config.projector_hidden_act)
        self.language_model = init_vllm_registered_model(
            config.text_config, cache_config, quant_config)
        self.vision_resampler = LlavaNextVideoPooler(config)

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
            self, **kwargs: object) -> Optional[LlavaNextVideoPixelInputs]:
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

        return LlavaNextVideoPixelInputs(
            type="pixel_values_videos",
            data=pixel_values,
        )

    def _select_image_features(self, image_features: torch.Tensor, *,
                               strategy: str) -> torch.Tensor:
        if strategy == "default":
            return image_features[:, 1:]
        elif strategy == "full":
            return image_features

        raise ValueError(f"Unexpected select feature strategy: {strategy}")

    def _video_pixels_to_features(
        self,
        vision_tower: Union[CLIPVisionModel, SiglipVisionModel],
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:

        # NOTE: we skip the step to select the vision feature layer since
        # this is already done inside the vision tower
        image_features = vision_tower(pixel_values)
        image_features = self._select_image_features(
            image_features,
            strategy=self.config.vision_feature_select_strategy,
        )
        image_features = self.vision_resampler(image_features)
        image_features = self.multi_modal_projector(image_features)
        return image_features

    def _process_video_pixels(self, inputs: LlavaNextVideoPixelInputs):
        assert self.vision_tower is not None

        video_pixels = inputs["data"]

        if isinstance(video_pixels, torch.Tensor):
            # TODO: support multiple videos per input
            b, num_videos, num_frames, c, h, w = video_pixels.shape
            assert (num_videos == 1)
            stacked_pixels = video_pixels.view(b * num_videos * num_frames, c,
                                               h, w)
            stacked_embeddings = self._video_pixels_to_features(
                self.vision_tower, stacked_pixels)
            return stacked_embeddings.view(b, num_frames,
                                           *stacked_embeddings.shape[1:])

        elif is_list_of(video_pixels, torch.Tensor):
            frames_per_videos = [v.shape[0] for v in video_pixels]
            stacked_pixels = torch.cat(video_pixels, dim=0)
            stacked_embeddings = self._video_pixels_to_features(
                self.vision_tower, stacked_pixels)
            return torch.split(stacked_embeddings, frames_per_videos, dim=0)

        else:
            raise ValueError(
                f"Unsupported type of video input {type(video_pixels)}")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: object,
    ) -> SamplerOutput:
        """Run forward pass for LlaVA-NeXT-Video.
        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            pixel_values_videos: Pixels in each frames for each input videos.
        """
        video_input = self._parse_and_validate_video_input(**kwargs)

        # merge video embeddings into input embeddings
        if video_input is not None:
            video_embeddings = self._process_video_pixels(video_input)
            inputs_embeds = self.language_model \
                .model.get_input_embeddings(input_ids)

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
