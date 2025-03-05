# SPDX-License-Identifier: Apache-2.0
# Standard library imports
import math
from functools import partial
from typing import Any, Iterable, List, Mapping, Optional, Set, Tuple, Union

# Third-party imports
import torch
from transformers import BatchFeature
from transformers.models.colqwen2_vl import (ColQwen2VLImageProcessor,
                                             ColQwen2VLProcessor)
from transformers.models.colqwen2_vl.configuration_colqwen2 import (
    ColQwen2VLConfig)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.models.qwen2_vl import (
    Qwen2VLDummyInputsBuilder, Qwen2VLForConditionalGeneration,
    Qwen2VLMultiModalDataParser, Qwen2VLMultiModalProcessor,
    Qwen2VLProcessingInfo)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (ImageItem, ModalityData,
                                    MultiModalFieldConfig, MultiModalKwargs,
                                    VideoItem)
from vllm.multimodal.parse import (DictEmbeddingItems, ImageSize,
                                   ModalityDataItems, MultiModalDataItems,
                                   MultiModalDataParser)
from vllm.multimodal.processing import PromptReplacement
from vllm.multimodal.profiling import ProcessorInputs
from vllm.sequence import IntermediateTensors

from .utils import AutoWeightsLoader


def round_by_factor(number: float, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: float, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: float, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


class ColQwen2VLMultiModalProcessor(Qwen2VLMultiModalProcessor):
    """
    vLLM processor for ColQwen2 model, based on Qwen2VLProcessor.
    """

    def _get_data_parser(self) -> MultiModalDataParser:
        return ColQwen2VLMultiModalDataParser()

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

        placeholder = {
            "image": vocab[hf_processor.image_token],
            "video": vocab[hf_processor.video_token],
        }

        merge_length = image_processor.merge_size**2

        def get_replacement_colqwen2vl(item_idx: int, modality: str):
            grid_thw = out_mm_kwargs[f"{modality}_grid_thw"][item_idx]
            assert isinstance(grid_thw, torch.Tensor)

            num_tokens = int(grid_thw.prod()) // merge_length
            return [placeholder[modality]] * num_tokens

        return [
            PromptReplacement(
                modality=modality,
                target=[placeholder[modality]],
                replacement=partial(get_replacement_colqwen2vl,
                                    modality=modality),
            ) for modality in ("image", "video")
        ]

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _colqwen2vl_field_config(hf_inputs)


class ColQwen2VLProcessingInfo(Qwen2VLProcessingInfo):

    def get_hf_processor(
        self,
        *,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        size: Optional[dict[str, int]] = None,
        **kwargs: object,
    ) -> ColQwen2VLProcessor:
        return self.ctx.get_hf_processor(
            ColQwen2VLProcessor,
            image_processor=self.get_image_processor(min_pixels=min_pixels,
                                                     max_pixels=max_pixels,
                                                     size=size),
            **kwargs)

    def _get_vision_info(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int = 1,
        do_resize: bool = True,
        image_processor: Optional[ColQwen2VLImageProcessor],
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

        # NOTE: Frames are padded to be divisible by `temporal_patch_size`
        # https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L294
        padded_num_frames = num_frames + num_frames % temporal_patch_size

        grid_t = max(padded_num_frames // temporal_patch_size, 1)
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
        image_processor: Optional[ColQwen2VLImageProcessor],
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
        image_processor: Optional[ColQwen2VLImageProcessor],
    ) -> int:
        _, num_video_tokens = self._get_vision_info(
            image_width=image_width,
            image_height=image_height,
            num_frames=num_frames,
            image_processor=image_processor,
        )
        return num_video_tokens


class ColQwen2VLDummyInputsBuilder(Qwen2VLDummyInputsBuilder):
    """Builds dummy inputs for profiling ColQwen2 model."""

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


def _colqwen2vl_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
    image_grid_sizes = image_grid_thw.prod(-1)

    video_grid_thw = hf_inputs.get("video_grid_thw", torch.empty((0, 3)))
    video_grid_sizes = video_grid_thw.prod(-1)

    return dict(
        pixel_values=MultiModalFieldConfig.flat_from_sizes(
            "image", image_grid_sizes),
        image_embeds=MultiModalFieldConfig.flat_from_sizes(
            "image", image_grid_sizes),
        image_grid_thw=MultiModalFieldConfig.batched("image"),
        pixel_values_videos=MultiModalFieldConfig.flat_from_sizes(
            "video", video_grid_sizes),
        video_embeds=MultiModalFieldConfig.flat_from_sizes(
            "video", video_grid_sizes),
        video_grid_thw=MultiModalFieldConfig.batched("video"),
    )


class ColQwen2VLMultiModalDataParser(Qwen2VLMultiModalDataParser):

    def _parse_image_data(
        self,
        data: Union[dict[str, torch.Tensor], ModalityData[ImageItem]],
    ) -> ModalityDataItems[Any, Any]:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="image",
                required_fields={"image_embeds", "image_grid_thw"},
                fields_factory=_colqwen2vl_field_config,
            )
        return super()._parse_image_data(data)

    def _parse_video_data(
        self,
        data: Union[dict[str, torch.Tensor], ModalityData[VideoItem]],
    ) -> ModalityDataItems[Any, Any]:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="video",
                required_fields={"video_embeds", "video_grid_thw"},
                fields_factory=_colqwen2vl_field_config,
            )
        return super()._parse_video_data(data)


@MULTIMODAL_REGISTRY.register_processor(
    ColQwen2VLMultiModalProcessor,
    info=ColQwen2VLProcessingInfo,
    dummy_inputs=ColQwen2VLDummyInputsBuilder)
class ColQwen2(Qwen2VLForConditionalGeneration):
    """
    ColQwen2 model implementation from the "ColPali: Efficient Document Retrieval with Vision Language Models" paper.
    VLLM version compatible with the transformers ColQwen2 class.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config: ColQwen2VLConfig = vllm_config.model_config.hf_config
        self.config = config
        self.dim = 128
        # Add custom text projection layer to project from model hidden size to embedding dimension
        self.custom_text_proj = ColumnParallelLinear(
            self.config.hidden_size,
            self.dim,
            bias=True,
            gather_output=True,
            prefix=f"{prefix}.custom_text_proj")
        self.padding_side = "left"

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        # Call the language model to get hidden states
        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        # Project to lower dimension
        proj, _ = self.custom_text_proj(hidden_states)

        # L2 normalization
        proj = proj / (proj.norm(dim=-1, keepdim=True) + 1e-6)

        # Apply attention mask if available
        attention_mask = attn_metadata.attention_mask if hasattr(
            attn_metadata, 'attention_mask') else None
        if attention_mask is not None:
            proj = proj * attention_mask.unsqueeze(-1)

        return proj

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        # For embedding models, we don't need to compute logits
        return None

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        # For embedding models, we don't need to sample
        return None

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        # First load the standard weights
        loaded = super().load_weights(weights)

        # Custom handling for custom_text_proj weights which might have different naming
        loader = AutoWeightsLoader(self)
        # Define a custom mapper for the custom_text_proj weights
        custom_mapper = {
            "custom_text_proj.weight": "custom_text_proj.weight",
            "custom_text_proj.bias": "custom_text_proj.bias",
        }

        for orig, new in custom_mapper.items():
            for name, param in weights:
                if orig in name:
                    try:
                        tensor_name = name.replace(orig, new)
                        loader.load_tensor(tensor_name, param)
                        loaded.add(name)
                    except Exception as e:
                        print(f"Error loading {name}: {e}")

        return loaded

    @property
    def patch_size(self) -> int:
        return self.visual.config.patch_size

    @property
    def spatial_merge_size(self) -> int:
        return self.visual.spatial_merge_size
