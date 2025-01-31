# Adapted from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
# Copyright 2025 The vLLM team.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Inference-only Qwen2-VL model compatible with HuggingFace weights."""
from functools import cached_property, partial
from typing import (Any, Callable, Iterable, List, Literal, Mapping, Optional,
                    Set, Tuple, Type, TypedDict, Union)

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers import BatchFeature
from transformers.models.qwen2_5_vl import (Qwen2_5_VLImageProcessor,
                                            Qwen2_5_VLProcessor)
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig)
from transformers.models.qwen2_5_vl.image_processing_qwen2_5_vl import (
    smart_resize)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel)

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.distributed import parallel_state, tensor_model_parallel_all_gather
from vllm.distributed import utils as dist_utils
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.activation import QuickGELU
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.gptq import GPTQConfig
from vllm.model_executor.layers.quantization.gptq_marlin import (
    GPTQMarlinConfig)
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (ImageItem, ModalityData,
                                    MultiModalFieldConfig, MultiModalKwargs,
                                    VideoItem)
from vllm.multimodal.parse import (ImageSize, ModalityDataItems,
                                   MultiModalDataItems, MultiModalDataParser)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.platforms import _Backend
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import uses_mrope

from .interfaces import SupportsLoRA, SupportsMultiModal, SupportsPP
from .utils import (AutoWeightsLoader, WeightsMapper,
                    init_vllm_registered_model, maybe_prefix,
                    merge_multimodal_embeddings)
from .vision import get_vit_attn_backend

logger = init_logger(__name__)

# For profile run
_MAX_FRAMES_PER_VIDEO = 16

# === Vision Inputs === #


class Qwen2_5_VLImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: torch.Tensor
    """Shape:
    `(num_patches, num_channels * patch_size * patch_size)`
    """

    image_grid_thw: torch.Tensor
    """Shape: `(num_images, 3)`
    This should be in `(grid_t, grid_h, grid_w)` format.
    """


class Qwen2_5_VLVideoPixelInputs(TypedDict):
    type: Literal["pixel_values_videos"]
    pixel_values_videos: torch.Tensor
    """Shape:
    `(num_patches,
      num_channels * temporal_patch_size * patch_size * patch_size)`
    """

    video_grid_thw: torch.Tensor
    """Shape: `(num_videos, 3)`

    This should be in `(grid_t, grid_h, grid_w)` format.
    """


class Qwen2_5_VLProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen2_5_VLConfig)

    def get_hf_processor(
        self,
        *,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
    ) -> Qwen2_5_VLProcessor:
        hf_processor = self.ctx.get_hf_processor(Qwen2_5_VLProcessor)
        image_processor = hf_processor.image_processor  # type: ignore
        assert isinstance(image_processor, Qwen2_5_VLImageProcessor)

        if min_pixels:
            image_processor.min_pixels = min_pixels
        if max_pixels:
            image_processor.max_pixels = max_pixels
        if max_pixels or min_pixels:
            image_processor.size = {
                "min_pixels": image_processor.min_pixels,
                "max_pixels": image_processor.max_pixels,
            }

        return hf_processor

    def get_image_processor(
        self,
        *,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
    ):
        hf_processor = self.get_hf_processor(min_pixels=min_pixels,
                                             max_pixels=max_pixels)
        image_processor = hf_processor.image_processor  # type: ignore
        assert isinstance(image_processor, Qwen2_5_VLImageProcessor)
        return image_processor

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None, "video": None}

    def get_mm_max_tokens_per_item(self, seq_len: int) -> Mapping[str, int]:
        return {
            "image": self.get_max_image_tokens(),
            "video": self.get_max_video_tokens(seq_len),
        }

    def _get_vision_info(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int = 1,
        do_resize: bool = True,
        image_processor: Optional[Qwen2_5_VLImageProcessor],
    ) -> tuple[ImageSize, int]:
        if image_processor is None:
            image_processor = self.get_image_processor()

        hf_config = self.get_hf_config()
        vision_config = hf_config.vision_config
        patch_size = vision_config.patch_size
        merge_size = vision_config.spatial_merge_size
        temporal_patch_size = vision_config.temporal_patch_size

        if do_resize:
            resized_height, resized_width = smart_resize(
                height=image_height,
                width=image_width,
                factor=patch_size * merge_size,
                min_pixels=image_processor.min_pixels,
                max_pixels=image_processor.max_pixels,
            )
            preprocessed_size = ImageSize(width=resized_width,
                                          height=resized_height)
        else:
            preprocessed_size = ImageSize(width=image_width,
                                          height=image_height)

        grid_t = max(num_frames // temporal_patch_size, 1)
        grid_h = preprocessed_size.height // patch_size
        grid_w = preprocessed_size.width // patch_size

        num_patches = grid_t * grid_h * grid_w
        num_vision_tokens = num_patches // (merge_size**2)

        return preprocessed_size, num_vision_tokens

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        image_processor: Optional[Qwen2_5_VLImageProcessor],
    ) -> int:
        _, num_image_tokens = self._get_vision_info(
            image_width=image_width,
            image_height=image_height,
            image_processor=image_processor,
        )
        return num_image_tokens

    def get_num_video_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int,
        image_processor: Optional[Qwen2_5_VLImageProcessor],
    ) -> int:
        _, num_video_tokens = self._get_vision_info(
            image_width=image_width,
            image_height=image_height,
            num_frames=num_frames,
            image_processor=image_processor,
        )
        return num_video_tokens

    def get_image_size_with_most_features(self) -> ImageSize:
        max_image_size, _ = self._get_vision_info(
            image_width=9999999,
            image_height=9999999,
            image_processor=None,
        )
        return max_image_size

    def get_max_image_tokens(self) -> int:
        target_width, target_height = self.get_image_size_with_most_features()

        return self.get_num_image_tokens(
            image_width=target_width,
            image_height=target_height,
            image_processor=None,
        )

    def _get_max_video_frames(self, max_tokens: int) -> int:
        target_width, target_height = self.get_image_size_with_most_features()

        num_frames = 0

        while True:
            next_num_frames = num_frames + 1
            next_max_tokens = self.get_num_video_tokens(
                image_width=target_width,
                image_height=target_height,
                num_frames=next_num_frames,
                image_processor=None,
            )

            if next_max_tokens > max_tokens:
                break

            num_frames = next_num_frames

        return num_frames

    def get_num_frames_with_most_features(self, seq_len: int) -> int:
        mm_config = self.ctx.get_mm_config()
        max_images = mm_config.limit_per_prompt.get("image", 1)
        max_videos = mm_config.limit_per_prompt.get("video", 1)

        max_image_tokens = self.get_max_image_tokens() * max_images
        max_total_frames = self._get_max_video_frames(seq_len -
                                                      max_image_tokens)
        num_frames = min(max(max_total_frames // max(max_videos, 1), 1),
                         _MAX_FRAMES_PER_VIDEO)

        # Temporary workaround for https://github.com/huggingface/transformers/issues/35412
        if num_frames > 1 and num_frames % 2 == 1:
            num_frames += 1

        return num_frames

    def get_max_video_tokens(self, seq_len: int) -> int:
        target_width, target_height = self.get_image_size_with_most_features()

        return self.get_num_video_tokens(
            image_width=target_width,
            image_height=target_height,
            num_frames=self.get_num_frames_with_most_features(seq_len),
            image_processor=None,
        )


class Qwen2_5_VLDummyInputsBuilder(
        BaseDummyInputsBuilder[Qwen2_5_VLProcessingInfo]):

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        hf_processor = self.info.get_hf_processor()
        image_token: str = hf_processor.image_token
        video_token: str = hf_processor.video_token

        target_width, target_height = \
            self.info.get_image_size_with_most_features()
        target_num_frames = \
            self.info.get_num_frames_with_most_features(seq_len)

        mm_data = {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images),
            "video":
            self._get_dummy_videos(
                width=target_width,
                height=target_height,
                num_frames=target_num_frames,
                num_videos=num_videos,
            )
        }

        return ProcessorInputs(
            prompt_text=image_token * num_images + video_token * num_videos,
            mm_data=mm_data,
        )


class Qwen2_5_VLMultiModalProcessor(
        BaseMultiModalProcessor[Qwen2_5_VLProcessingInfo]):

    def _get_prompt_replacements(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> list[PromptReplacement]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(
            **hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        # NOTE: Only Qwen2_5_VLProcessor in transformers 4.47.0 has
        # image_token and video_token registered
        placeholder = {
            "image": vocab[hf_processor.image_token],
            "video": vocab[hf_processor.video_token],
        }

        merge_length = image_processor.merge_size**2

        def get_replacement_Qwen2_5_VL(item_idx: int, modality: str):
            grid_thw = out_mm_kwargs[f"{modality}_grid_thw"][item_idx]
            assert isinstance(grid_thw, torch.Tensor)

            num_tokens = int(grid_thw.prod()) // merge_length
            return [placeholder[modality]] * num_tokens

        return [
            PromptReplacement(
                modality=modality,
                target=[placeholder[modality]],
                replacement=partial(get_replacement_Qwen2_5_VL,
                                    modality=modality),
            ) for modality in ("image", "video")
        ]

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
        image_slice_idxs = [0] + image_grid_thw.prod(-1).cumsum_(0).tolist()
        image_slices = [
            slice(image_slice_idxs[i], image_slice_idxs[i + 1])
            for i in range(len(image_grid_thw))
        ]

        video_grid_thw = hf_inputs.get("video_grid_thw", torch.empty((0, 3)))
        video_slice_idxs = [0] + video_grid_thw.prod(-1).cumsum_(0).tolist()
        video_slices = [
            slice(video_slice_idxs[i], video_slice_idxs[i + 1])
            for i in range(len(video_grid_thw))
        ]

        return dict(
            pixel_values=MultiModalFieldConfig.flat("image", image_slices),
            image_grid_thw=MultiModalFieldConfig.batched("image"),
            pixel_values_videos=MultiModalFieldConfig.flat(
                "video", video_slices),
            video_grid_thw=MultiModalFieldConfig.batched("video"),
        )


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5_VLMultiModalProcessor,
    info=Qwen2_5_VLProcessingInfo,
    dummy_inputs=Qwen2_5_VLDummyInputsBuilder)
class Qwen2_5_VLForConditionalGeneration(nn.Module, SupportsMultiModal,
                                         SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
        # vision tower
        "qkv",
        "attn.proj",  # Distinguish patch_embed.proj
        "fc1",
        "fc2",
        # projector
        "mlp.0",
        "mlp.2"
    ]
    embedding_modules = {}
    embedding_padding_modules = []

    # To ensure correct weight loading and mapping.
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
        "lm_head.": "language_model.lm_head.",
        "model.": "language_model.model.",
    })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: Qwen2_5_VLConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(
            config.vision_config)

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen2ForCausalLM"],
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    @cached_property
    def sampler(self):
        if hasattr(self.language_model, "sampler"):
            return self.language_model.sampler

        return get_sampler()

    def _maybe_ignore_quant_config(self, quant_config: QuantizationConfig):
        # GPTQ configs do not have a list of ignored modules, however AutoGPTQ
        # seems to avoid vision encoder sections for some models.
        # See: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4
        if isinstance(quant_config, (GPTQConfig, GPTQMarlinConfig)):
            return None
        return quant_config

    def _validate_and_reshape_mm_tensor(self, mm_input: object,
                                        name: str) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. "
                             f"Got type: {type(mm_input)}")
        if isinstance(mm_input, torch.Tensor):
            if mm_input.ndim == 2:
                return mm_input
            if mm_input.ndim != 3:
                raise ValueError(f"{name} should be 2D or batched 3D tensor. "
                                 f"Got ndim: {mm_input.ndim} "
                                 f"(shape={mm_input.shape})")
            return torch.concat(list(mm_input))
        else:
            return torch.concat(mm_input)

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[Qwen2_5_VLImagePixelInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None:
            return None

        if pixel_values is not None:
            pixel_values = self._validate_and_reshape_mm_tensor(
                pixel_values, "image pixel values")
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw")

            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image pixel values. "
                                 f"Got type: {type(pixel_values)}")

            return Qwen2_5_VLImagePixelInputs(type="pixel_values",
                                              pixel_values=pixel_values,
                                              image_grid_thw=image_grid_thw)
        raise

    def _parse_and_validate_video_input(
            self, **kwargs: object) -> Optional[Qwen2_5_VLVideoPixelInputs]:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)

        if pixel_values_videos is None:
            return None

        if pixel_values_videos is not None:
            pixel_values_videos = self._validate_and_reshape_mm_tensor(
                pixel_values_videos, "video pixel values")
            video_grid_thw = self._validate_and_reshape_mm_tensor(
                video_grid_thw, "video grid_thw")

            return Qwen2_5_VLVideoPixelInputs(
                type="pixel_values_videos",
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )
        raise

    def _process_image_input(
            self, image_input: Qwen2_5_VLImagePixelInputs
    ) -> tuple[torch.Tensor, ...]:

        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        pixel_values = image_input["pixel_values"].type(self.visual.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=grid_thw)

        # Split concatenated embeddings for each image item.
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size

        return image_embeds.split(sizes.tolist())

    def _process_video_input(
            self, video_input: Qwen2_5_VLVideoPixelInputs
    ) -> tuple[torch.Tensor, ...]:

        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2

        pixel_values_videos = video_input["pixel_values_videos"].type(
            self.visual.dtype)
        video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw)

        # Split concatenated embeddings for each video item.
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size

        return video_embeds.split(sizes.tolist())

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if input_key in ("pixel_values",
                             "image_embeds") and "images" not in modalities:
                modalities["images"] = self._parse_and_validate_image_input(
                    **kwargs)
            if input_key in ("pixel_values_videos",
                             "video_embeds") and "videos" not in modalities:
                modalities["videos"] = self._parse_and_validate_video_input(
                    **kwargs)

        return modalities

    def get_multimodal_embeddings(
            self, **kwargs) -> Optional[tuple[torch.Tensor, ...]]:

        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not modalities:
            return None

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
                video_embeddings = self._process_video_input(video_input)
                multimodal_embeddings += video_embeddings

        return multimodal_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[tuple[torch.Tensor, ...]] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                [self.config.image_token_id, self.config.video_token_id])
        return inputs_embeds

    def get_input_embeddings_v0(
        self,
        input_ids: torch.Tensor,
        image_input: Optional[tuple[torch.Tensor, ...]] = None,
        video_input: Optional[tuple[torch.Tensor, ...]] = None,
    ) -> torch.Tensor:

        inputs_embeds = self.get_input_embeddings(input_ids)
        if image_input is not None:
            image_embeds = self._process_image_input(image_input)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                image_embeds,
                placeholder_token_id=self.config.image_token_id,
            )

        if video_input is not None:
            video_embeds = self._process_video_input(video_input)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                video_embeds,
                placeholder_token_id=self.config.video_token_id,
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """Run forward pass for Qwen2.5-VL.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
                **NOTE**: If mrope is enabled (default setting for Qwen2-VL
                opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
            pixel_values: Pixel values to be fed to a model.
                `None` if no images are passed.
            image_grid_thw: Tensor `(n_images, 3)` of image 3D grid in LLM.
                `None` if no images are passed.
            pixel_values_videos: Pixel values of videos to be fed to a model.
                `None` if no videos are passed.
            video_grid_thw: Tensor `(n_videos, 3)` of video 3D grid in LLM.
                `None` if no videos are passed.
        """

        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner from
        # `get_multimodal_embeddings` and `get_input_embeddings`, this
        # condition is only for v0 compatibility.
        elif inputs_embeds is None:
            image_input = self._parse_and_validate_image_input(**kwargs)
            video_input = self._parse_and_validate_video_input(**kwargs)

            if image_input is None and video_input is None:
                inputs_embeds = None
            else:
                if uses_mrope(self.config):
                    assert positions.ndim == 2 and positions.size(0) == 3, (
                        "multimodal section rotary embedding requires "
                        f"(3, seq_len) positions, but got {positions.size()}")
                inputs_embeds = self.get_input_embeddings_v0(
                    input_ids,
                    image_input=image_input,
                    video_input=video_input)
                input_ids = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
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

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:

        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="visual.",
            tower_model="visual.merger.")
