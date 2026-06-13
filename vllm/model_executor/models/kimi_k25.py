# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Kimi-K2.5 Model Implementation for vLLM.

Kimi-K2.5 extends Kimi-K2 with vision support.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal

import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors,
)
from vllm.model_executor.models.interfaces import (
    SupportsEagle,
    SupportsEagle3,
    SupportsMultiModal,
    SupportsPP,
    SupportsQuant,
)
from vllm.model_executor.models.kimi_k25_vit import (
    KimiK25MultiModalProjector,
    MoonViT3dPretrainedModel,
    vision_tower_forward,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    NestedTensors,
    VisionChunkImage,
    VisionChunkVideo,
)
from vllm.multimodal.parse import MultiModalDataItems, VisionChunkProcessorItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    InputProcessingContext,
    PromptReplacement,
    PromptUpdate,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.kimi_k25 import KimiK25Config
from vllm.transformers_utils.processor import cached_get_image_processor
from vllm.transformers_utils.processors.kimi_k25 import KimiK25Processor
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)

logger = init_logger(__name__)


def _format_video_timestamp(
    timestamp: float,
    timestamp_mode: str = "hh:mm:ss.fff",
) -> str:
    timestamp = max(timestamp, 0)
    total_ms = int(timestamp * 1000)
    total_seconds, milliseconds = divmod(total_ms, 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if timestamp_mode == "hh:mm:ss.fff":
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    if timestamp_mode == "mm:ss.fff":
        return f"{minutes + hours * 60:02d}:{seconds:02d}.{milliseconds:03d}"
    if timestamp_mode == "mm:ss":
        return f"{minutes + hours * 60:02d}:{seconds:02d}"
    raise ValueError(f"Invalid Kimi video timestamp mode: {timestamp_mode}")


def _frames_to_pil_images(frames: object) -> list[Image.Image]:
    if isinstance(frames, np.ndarray):
        if frames.ndim != 4:
            raise ValueError(f"Kimi video frames must be 4D, got shape={frames.shape}")
        if frames.shape[-1] not in (1, 3, 4):
            raise ValueError(
                f"Kimi video frames must have 1/3/4 channels, got shape={frames.shape}"
            )
        if frames.dtype != np.uint8:
            raise ValueError(f"Kimi video frames must be uint8, got {frames.dtype}")

        pil_frames: list[Image.Image] = []
        for frame in frames:
            if frame.shape[-1] == 1:
                frame = frame[..., 0]
            pil_frames.append(Image.fromarray(frame).convert("RGB"))
        return pil_frames

    if isinstance(frames, list):
        pil_frames = []
        for frame in frames:
            if isinstance(frame, Image.Image):
                pil_frames.append(frame.convert("RGB"))
            elif isinstance(frame, np.ndarray):
                frame_array = np.asarray(frame)
                if frame_array.ndim == 3:
                    frame_array = frame_array[None, ...]
                pil_frames.extend(_frames_to_pil_images(frame_array))
            else:
                raise ValueError(f"Unsupported Kimi video frame item: {type(frame)!r}")
        return pil_frames

    raise ValueError(f"Unsupported Kimi video frame container: {type(frames)!r}")


def _split_video_chunks(
    video_data: object,
    image_processor: object,
) -> list[dict[str, Any]]:
    metadata: dict[str, Any] = {}
    frames = video_data
    if isinstance(video_data, tuple) and len(video_data) >= 1:
        frames = video_data[0]
        if len(video_data) >= 2 and isinstance(video_data[1], dict):
            metadata = video_data[1]

    pil_frames = _frames_to_pil_images(frames)
    if not pil_frames:
        raise ValueError("Kimi video input decoded to zero frames")

    media_proc_cfg = getattr(image_processor, "media_proc_cfg", {}) or {}
    temporal_merge_kernel_size = int(
        media_proc_cfg.get(
            "temporal_merge_kernel_size",
            getattr(image_processor, "num_frames_per_chunk", 4),
        )
        or 4
    )
    if temporal_merge_kernel_size <= 0:
        raise ValueError(
            "Kimi temporal_merge_kernel_size must be positive, "
            f"got {temporal_merge_kernel_size}"
        )

    timestamp_mode = media_proc_cfg.get("timestamp_mode", "hh:mm:ss.fff")
    fps_value = metadata.get("fps", metadata.get("avg_fps", 1.0))
    if fps_value is None:
        fps_value = 1.0
    fps = float(fps_value)
    if fps <= 0:
        raise ValueError(f"Kimi video fps must be positive, got {fps}")

    frame_indices = metadata.get("frames_indices")
    if frame_indices is not None and len(frame_indices) != len(pil_frames):
        raise ValueError(
            f"Kimi video frames_indices length ({len(frame_indices)}) must "
            f"match frame count ({len(pil_frames)})"
        )

    chunk_prompt_template = (
        "{timestamp}<|media_begin|>video<|media_content|><|media_pad|><|media_end|>"
    )
    chunks: list[dict[str, Any]] = []
    for start in range(0, len(pil_frames), temporal_merge_kernel_size):
        chunk_frames = pil_frames[start : start + temporal_merge_kernel_size]
        start_frame = frame_indices[start] if frame_indices is not None else start
        timestamp = _format_video_timestamp(start_frame / fps, timestamp_mode)
        chunks.append(
            {
                "type": "video_chunk",
                "video_chunk": chunk_frames,
                "prompt": chunk_prompt_template.format(timestamp=timestamp),
            }
        )

    return chunks


# Dummy input dimensions for profiling.
@dataclass
class MaxImageTokenMeta:
    width: int = 3000
    height: int = 3000


class KimiK25MediaPixelInputs(TensorSchema):
    """
    Media input schema for K2-VL model.

    Dimensions:
        - np: Number of patches (flattened from all media items)
        - ps: Patch size
        - nm: Number of media items
    """

    type: Literal["pixel_values"] = "pixel_values"

    pixel_values: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("np", 3, "ps", "ps"),
    ]

    grid_thws: Annotated[torch.Tensor, TensorShape("nm", 3)]


class KimiK25ProcessingInfo(BaseProcessingInfo):
    """Processing information for Kimi-K2.5 model.

    Provides configuration and utilities for processing both
    images and video-chunks.
    """

    def __init__(self, ctx: InputProcessingContext) -> None:
        super().__init__(ctx)

        self.hf_config = hf_config = self.get_hf_config()

        tokenizer = self.get_tokenizer()
        image_processor = cached_get_image_processor(
            self.ctx.model_config.model,
            revision=self.ctx.model_config.revision,
            trust_remote_code=self.ctx.model_config.trust_remote_code,
        )

        # Resolve token ID from the tokenizer because transformers v5
        # may remap token IDs vs config.json.
        config_token_id = hf_config.media_placeholder_token_id
        resolved_token_id = tokenizer.convert_tokens_to_ids("<|media_pad|>")
        is_valid_resolved = isinstance(resolved_token_id, int) and (
            tokenizer.unk_token_id is None
            or resolved_token_id != tokenizer.unk_token_id
        )
        if is_valid_resolved and resolved_token_id != config_token_id:
            logger.warning_once(
                "Kimi-K2.5 config.media_placeholder_token_id (%d) disagrees "
                "with tokenizer mapping for <|media_pad|> (%d). "
                "Using tokenizer value.",
                config_token_id,
                resolved_token_id,
            )
            media_token_id = resolved_token_id
            # Patch config so downstream code also sees the correct ID.
            hf_config.media_placeholder_token_id = resolved_token_id
        else:
            media_token_id = config_token_id

        self.media_token_id = media_token_id
        self.media_token = tokenizer.decode(media_token_id)

        self.image_processor = image_processor
        self.hf_processor = KimiK25Processor(
            tokenizer=tokenizer,
            image_processor=image_processor,
            media_token_id=media_token_id,
        )
        self.media_tokens_calculator = image_processor.media_tokens_calculator

    def get_hf_processor(self):
        return self.hf_processor

    def get_hf_config(self):
        return self.ctx.get_hf_config(KimiK25Config)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # None means unlimited
        return {"vision_chunk": None}


class KimiK25DummyInputsBuilder(BaseDummyInputsBuilder[KimiK25ProcessingInfo]):
    """Builds dummy inputs for Kimi-K2.5 model profiling."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_media = mm_counts.get("vision_chunk", 0)
        return self.info.media_token * num_media

    def get_dummy_mm_items(self):
        dummy_videos = self._get_dummy_images(
            height=MaxImageTokenMeta.height,
            width=MaxImageTokenMeta.width,
            num_images=self.info.image_processor.num_frames_per_chunk,
        )

        video_chunk_dummy_item = VisionChunkVideo(
            type="video_chunk", video_chunk=dummy_videos
        )
        video_chunk_num_tokens = self.info.media_tokens_calculator(
            video_chunk_dummy_item
        )

        image_dummy_item = VisionChunkImage(
            type="image",
            image=self._get_dummy_images(
                height=MaxImageTokenMeta.height,
                width=MaxImageTokenMeta.width,
                num_images=1,
            )[0],
        )
        image_num_tokens = self.info.media_tokens_calculator(image_dummy_item)
        # return the larger one
        if video_chunk_num_tokens >= image_num_tokens:
            return [video_chunk_dummy_item]
        else:
            return [image_dummy_item]

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        # TODO: Support mm_options for vision_chunk to allow user configuration
        dummy_items = self.get_dummy_mm_items()
        return {"vision_chunk": dummy_items}


class KimiK25MultiModalProcessor(BaseMultiModalProcessor[KimiK25ProcessingInfo]):
    """Multi-modal processor for Kimi-K2.5.

    Handles both image and video-chunk modalities.
    """

    requires_video_chunk_splitting = True

    def split_video_chunks(self, video_data: object) -> list[dict[str, Any]]:
        return _split_video_chunks(video_data, self.info.image_processor)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """Indicates how to slice media input into multiple items.

        pixel_values: [N, 3, patch_size, patch_size],
          all patches collected from B medias
        grid_thws: [B,3], each item: [N_t, N_h ,N_w],
          indicates the grid size in time/height/width direction for current item.

        by multiplying [N_t, N_h ,N_w], we get the number of patches
        for each media item, thus we can slice pixel_values by
        pixel_values[start:start + N_t*N_h*N_w] to get patches of one item.

        """
        grid_thws = hf_inputs.get("grid_thws", torch.empty((0, 3)))
        grid_sizes = grid_thws.prod(-1)

        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "vision_chunk", grid_sizes
            ),
            grid_thws=MultiModalFieldConfig.batched("vision_chunk", keep_on_cpu=True),
        )

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # Override to use the text path instead of token path because vision chunk
        # is not considered
        return super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        media_token_id = self.info.media_token_id

        def get_replacement(item_idx: int):
            media = mm_items.get_items("vision_chunk", (VisionChunkProcessorItems,))
            num_media_token = self.info.media_tokens_calculator(media[item_idx])
            return [media_token_id] * num_media_token

        return [
            PromptReplacement(
                modality="vision_chunk",
                target=[media_token_id],
                replacement=get_replacement,
            ),
        ]


@MULTIMODAL_REGISTRY.register_processor(
    KimiK25MultiModalProcessor,
    info=KimiK25ProcessingInfo,
    dummy_inputs=KimiK25DummyInputsBuilder,
)
class KimiK25ForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsQuant,
    SupportsEagle,
    SupportsEagle3,
):
    """Kimi-K2.5 model for conditional generation.

    Supports both image and video-chunk modalities.
    Video-chunks are temporal segments (typically 4 frames) that are
    processed with temporal pooling.
    """

    supports_encoder_tp_data = True

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # For legacy NVFP4 checkpoint compatibility:
            # see https://github.com/vllm-project/vllm/pull/33346#issuecomment-3851475033
            "language_model.layers.": "language_model.model.layers.",
            # mm projector
            "mm_projector.proj.0": "mm_projector.linear_1",
            "mm_projector.proj.2": "mm_projector.linear_2",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        # Kimi-K2.5 uses video_chunk for all media types
        if modality == "image":
            return "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>"
        elif modality == "video":
            # return a placeholder, to be replaced in the future.
            return "<|kimi_k25_video_placeholder|>"

        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        model_config = vllm_config.model_config
        config: KimiK25Config = model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config

        # Check for MoonViT config compatibility
        self.use_data_parallel = (
            model_config.multimodal_config.mm_encoder_tp_mode == "data"
        )
        self.hidden_size = config.text_config.hidden_size
        self.device = current_platform.current_device()
        # Build vision tower directly with KimiK25VisionConfig
        with self._mark_tower_model(vllm_config, "vision_chunk"):
            self.vision_tower = MoonViT3dPretrainedModel(
                config.vision_config,
                quant_config=self._maybe_ignore_quant_config(quant_config),
                prefix=maybe_prefix(prefix, "vision_tower"),
            )
            if self._maybe_ignore_quant_config(quant_config) is not None:
                self.vision_tower = self.vision_tower.to(device=self.device)
            else:
                self.vision_tower = self.vision_tower.to(
                    device=self.device, dtype=model_config.dtype
                )

            self.mm_projector = KimiK25MultiModalProjector(
                config=config.vision_config,
                use_data_parallel=self.use_data_parallel,
                quant_config=self._maybe_ignore_quant_config(quant_config),
                prefix=maybe_prefix(prefix, "mm_projector"),
            )
            self.mm_projector = self.mm_projector.to(
                device=self.device, dtype=model_config.dtype
            )

        self.quant_config = quant_config
        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["DeepseekV2ForCausalLM"],
            )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        self.media_placeholder: int = self.config.media_placeholder_token_id

    def _maybe_ignore_quant_config(self, quant_config: QuantizationConfig):
        if isinstance(quant_config, compressed_tensors.CompressedTensorsConfig):
            return None
        return quant_config

    def _parse_and_validate_media_input(
        self, **kwargs: object
    ) -> KimiK25MediaPixelInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        grid_thws = kwargs.pop("grid_thws", None)
        if pixel_values is None:
            return None

        if isinstance(pixel_values, list):
            pixel_values = torch.cat(pixel_values, dim=0)

        if len(pixel_values.shape) == 5 or len(pixel_values.shape) == 3:
            pixel_values = pixel_values.reshape(
                pixel_values.shape[0] * pixel_values.shape[1], *pixel_values.shape[2:]
            )

        # The batch dimension of pixel_values has been flattened into shape[0]
        target_dtype = next(self.vision_tower.parameters()).dtype
        pixel_values = pixel_values.to(target_dtype)
        assert isinstance(grid_thws, torch.Tensor), (
            f"expect grid_thws to be a tensor, got {type(grid_thws)}"
        )
        # In some cases (e.g. with merger), grid_thws has an extra middle dimension
        grid_thws = grid_thws.reshape(-1, grid_thws.shape[-1])
        assert grid_thws.ndim == 2 and grid_thws.size(1) == 3, (
            f"unexpected shape for grid_thws: {grid_thws.shape}"
        )

        return KimiK25MediaPixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            grid_thws=grid_thws,
        )

    def _process_media_input(
        self, media_input: KimiK25MediaPixelInputs
    ) -> list[torch.Tensor]:
        # NOTE(moyan): This forward will automatically batch the forward pass internally
        media_features = vision_tower_forward(
            self.vision_tower,
            media_input["pixel_values"],
            media_input["grid_thws"],
            mm_projector=self.mm_projector,
            use_data_parallel=self.use_data_parallel,
        )
        return media_features

    def embed_multimodal(self, **kwargs: object) -> NestedTensors | None:
        # Validate the multimodal input keyword arguments
        media_input = self._parse_and_validate_media_input(**kwargs)
        if media_input is None:
            return None

        # Run multimodal inputs through encoder and projector
        vision_embeddings = self._process_media_input(media_input)
        return vision_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        hidden_states = self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        logits = self.language_model.compute_logits(hidden_states)
        return logits

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.language_model.set_aux_hidden_state_layers(layers)

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        return self.language_model.get_eagle3_aux_hidden_state_layers()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
