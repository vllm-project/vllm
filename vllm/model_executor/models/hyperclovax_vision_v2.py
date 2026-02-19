# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
HyperCLOVAX V2 (32B Think Model) Implementation.

This module contains the V2 architecture that uses Qwen2.5 Vision Transformer
instead of CLIP/SigLIP used in V1.

Supports:
- HyperCLOVAX-SEED-Think-32B: Vision + Text
"""

from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from typing import Annotated, Literal

import torch
import torch.nn as nn
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.forward_context import set_forward_context
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import ImageSize, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    ProcessorInputs,
    PromptReplacement,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .qwen2_5_vl import Qwen2_5_VisionTransformer
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)

# V2 (32B Think model) uses different tokens - retrieved from config at runtime
# These placeholder strings must match the chat template format exactly.
# The chat template produces: <|image_start|><|IMAGE_PAD|><|image_end|>
# Similar to Qwen2-VL's <|vision_start|><|image_pad|><|vision_end|> format.
V2_IMAGE_TOKEN: str = "<|image_start|><|IMAGE_PAD|><|image_end|>"
V2_VIDEO_TOKEN: str = "<|video_start|><|VIDEO_PAD|><|video_end|>"


class HCXVisionV2ImagePixelInputs(TensorSchema):
    """
    V2 Image inputs using Qwen2.5-VL style grid_thw format.

    Dimensions:
        - np: Number of patches
        - ni: Number of images
        - cps: Number of channels * patch_size * patch_size
    """

    type: Literal["pixel_values"] = "pixel_values"
    pixel_values: Annotated[torch.Tensor, TensorShape("np", "cps")]
    image_grid_thw: Annotated[torch.Tensor, TensorShape("ni", 3)]


class HCXVisionV2ImageEmbeddingInputs(TensorSchema):
    """
    V2 Image embedding inputs.

    Dimensions:
        - nf: Number of image features
        - hs: Hidden size
        - ni: Number of images
    """

    type: Literal["image_embeds"] = "image_embeds"
    image_embeds: Annotated[torch.Tensor, TensorShape("nf", "hs")]
    image_grid_thw: Annotated[torch.Tensor, TensorShape("ni", 3)]


HCXVisionV2ImageInputs = HCXVisionV2ImagePixelInputs | HCXVisionV2ImageEmbeddingInputs


class HCXVisionV2VideoPixelInputs(TensorSchema):
    """
    V2 Video inputs using Qwen2.5-VL style grid_thw format.

    Dimensions:
        - np: Number of patches
        - nv: Number of videos
        - ctps: Number of channels * temporal_patch_size * patch_size * patch_size
    """

    type: Literal["pixel_values_videos"] = "pixel_values_videos"
    pixel_values_videos: Annotated[torch.Tensor, TensorShape("np", "ctps")]
    video_grid_thw: Annotated[torch.Tensor, TensorShape("nv", 3)]


class HCXVisionV2VideoEmbeddingInputs(TensorSchema):
    """
    V2 Video embedding inputs.

    Dimensions:
        - nf: Number of video features
        - hs: Hidden size
        - nv: Number of videos
    """

    type: Literal["video_embeds"] = "video_embeds"
    video_embeds: Annotated[torch.Tensor, TensorShape("nf", "hs")]
    video_grid_thw: Annotated[torch.Tensor, TensorShape("nv", 3)]


HCXVisionV2VideoInputs = HCXVisionV2VideoPixelInputs | HCXVisionV2VideoEmbeddingInputs


class HCXVisionV2ProcessingInfo(BaseProcessingInfo):
    """Processing info for HyperCLOVAX V2 (32B Think model)."""

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "video": None}

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        hf_config = self.get_hf_config()
        vision_config = hf_config.vision_config
        patch_size = vision_config.patch_size
        spatial_merge_size = vision_config.spatial_merge_size

        grid_h = image_height // patch_size
        grid_w = image_width // patch_size

        return (grid_h * grid_w) // (spatial_merge_size**2)

    def get_num_video_tokens(
        self,
        *,
        video_width: int,
        video_height: int,
        num_frames: int,
    ) -> int:
        hf_config = self.get_hf_config()
        vision_config = hf_config.vision_config
        patch_size = vision_config.patch_size
        temporal_patch_size = vision_config.temporal_patch_size
        spatial_merge_size = vision_config.spatial_merge_size

        grid_t = num_frames // temporal_patch_size
        grid_h = video_height // patch_size
        grid_w = video_width // patch_size

        return (grid_t * grid_h * grid_w) // (spatial_merge_size**2)

    def get_image_size_with_most_features(self) -> ImageSize:
        hf_config = self.get_hf_config()
        vision_config = hf_config.vision_config
        # Use a reasonable default size
        size = getattr(vision_config, "image_size", 448)
        return ImageSize(width=size, height=size)

    def get_max_image_tokens(self) -> int:
        target_width, target_height = self.get_image_size_with_most_features()
        return self.get_num_image_tokens(
            image_width=target_width,
            image_height=target_height,
        )


class HCXVisionV2DummyInputsBuilder(BaseDummyInputsBuilder[HCXVisionV2ProcessingInfo]):
    """Dummy inputs builder for HyperCLOVAX V2 memory profiling."""

    def get_dummy_text(
        self,
        mm_counts: Mapping[str, int],
    ) -> str:
        # This method is not used when get_dummy_processor_inputs is overridden,
        # but we keep it for compatibility.
        return ""

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
        mm_processor_kwargs: Mapping[str, object] | None = None,
    ) -> ProcessorInputs:
        """
        Override to use token IDs directly instead of text strings.

        This avoids the tokenizer issue where <|IMAGE_PAD|> might not be
        recognized as a special token and gets split into multiple tokens.
        By passing token IDs directly, we ensure the correct token (128060)
        is used for prompt replacement matching.
        """
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        hf_config = self.info.get_hf_config()

        # Use token IDs directly to avoid tokenizer issues with special tokens
        image_token_id = hf_config.image_token_id  # 128060
        video_token_id = hf_config.video_token_id  # 128061

        # Create prompt as token ID list instead of text string
        prompt_ids: list[int] = [image_token_id] * num_images + [
            video_token_id
        ] * num_videos

        dummy_mm_data = self.get_dummy_mm_data(
            seq_len,
            mm_counts,
            mm_options,
            mm_processor_kwargs=mm_processor_kwargs,
        )
        dummy_mm_items = self.info.parse_mm_data(dummy_mm_data, validate=False)

        return ProcessorInputs(
            prompt=prompt_ids,
            mm_items=dummy_mm_items,
            hf_processor_mm_kwargs=mm_processor_kwargs or {},
            tokenization_kwargs={"truncation": False},
        )

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
        mm_processor_kwargs: Mapping[str, object] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        target_width, target_height = self.info.get_image_size_with_most_features()
        target_num_frames = 16  # Default for video

        image_overrides = mm_options.get("image") if mm_options else None
        video_overrides = mm_options.get("video") if mm_options else None

        result: MultiModalDataDict = {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
                overrides=image_overrides,  # type: ignore
            ),
            "video": self._get_dummy_videos(
                width=target_width,
                height=target_height,
                num_frames=target_num_frames,
                num_videos=num_videos,
                overrides=video_overrides,  # type: ignore
            ),
        }

        return result


class HCXVisionV2MultiModalProcessor(
    BaseMultiModalProcessor[HCXVisionV2ProcessingInfo]
):
    """Multimodal processor for HyperCLOVAX V2 (32B Think model)."""

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        images = mm_data.get("images")
        videos = mm_data.get("videos")

        # Get the HF processor
        hf_processor = self.info.get_hf_processor(**mm_kwargs)

        # Build data dict for HF processor (images/videos only)
        # NOTE: We pass the prompt as-is without token normalization.
        # Token expansion is handled by vLLM via _get_prompt_updates since
        # _hf_processor_applies_updates returns False.
        data: dict[str, object] = dict(
            text=prompt,
            images=images,
            videos=videos,
        )

        processed_outputs = self.info.ctx.call_hf_processor(
            hf_processor=hf_processor,
            data=data,
        )

        return processed_outputs

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        # HyperCLOVAX V2 has a token case mismatch bug:
        # - Chat template uses <|IMAGE_PAD|> (uppercase)
        # - HF processor (Qwen2_5_VLProcessor) expects <|image_pad|> (lowercase)
        # - Tokenizer vocab has <|IMAGE_PAD|> (uppercase) = token ID 128060
        #
        # The HF processor's token expansion fails because it looks for lowercase
        # but the tokenized prompt has uppercase tokens. We bypass HF processor's
        # expansion and let vLLM handle it via _get_prompt_updates using the
        # correct token IDs from hf_config.
        return False

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()

        # Use token IDs directly from config.
        # This matches what get_dummy_processor_inputs uses, ensuring consistency.
        placeholder: dict[str, int] = {
            "image": hf_config.image_token_id,  # 128060 for <|IMAGE_PAD|>
            "video": hf_config.video_token_id,  # 128061 for <|VIDEO_PAD|>
        }

        merge_size = hf_config.vision_config.spatial_merge_size

        def get_replacement_v2(
            item_idx: int,
            modality: str,
            out_mm_kwargs: MultiModalKwargsItems,
        ):
            out_item = out_mm_kwargs[modality][item_idx]

            if modality == "image":
                grid_thw_elem = out_item.get("image_grid_thw")
                if grid_thw_elem is not None:
                    # Access .data to get the actual tensor from MultiModalFieldElem
                    grid_thw = grid_thw_elem.data
                    # Qwen2.5-VL style calculation
                    h, w = grid_thw[1].item(), grid_thw[2].item()
                    num_tokens = (h * w) // (merge_size**2)
                else:
                    # Fallback or error
                    raise ValueError("Missing image_grid_thw for V2 model")
            elif modality == "video":
                grid_thw_elem = out_item.get("video_grid_thw")
                if grid_thw_elem is not None:
                    # Access .data to get the actual tensor from MultiModalFieldElem
                    grid_thw = grid_thw_elem.data
                    t, h, w = grid_thw[0].item(), grid_thw[1].item(), grid_thw[2].item()
                    num_tokens = (t * h * w) // (merge_size**2)
                else:
                    raise ValueError("Missing video_grid_thw for V2 model")
            else:
                raise NotImplementedError(modality)

            return [placeholder[modality]] * num_tokens

        return [
            PromptReplacement(
                modality=modality,
                target=[
                    placeholder[modality],
                ],
                replacement=partial(
                    get_replacement_v2,
                    modality=modality,
                    out_mm_kwargs=out_mm_kwargs,
                ),
            )
            for modality in ("image", "video")
        ]

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        # HyperCLOVAX V2 uses Qwen2.5-VL style flattened pixel values where
        # pixel_values has shape (num_patches, channels*patch_size*patch_size)
        # while image_grid_thw has shape (num_images, 3).
        # We need to use flat_from_sizes to correctly handle this mismatch.
        hf_config = self.info.get_hf_config()
        spatial_merge_size = hf_config.vision_config.spatial_merge_size

        image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
        image_pixel_grid_sizes = image_grid_thw.prod(-1)
        image_embed_grid_sizes = (
            image_pixel_grid_sizes // spatial_merge_size // spatial_merge_size
        )

        video_grid_thw = hf_inputs.get("video_grid_thw", torch.empty((0, 3)))
        video_pixel_grid_sizes = video_grid_thw.prod(-1)
        video_embed_grid_sizes = (
            video_pixel_grid_sizes // spatial_merge_size // spatial_merge_size
        )

        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "image", image_pixel_grid_sizes
            ),
            image_embeds=MultiModalFieldConfig.flat_from_sizes(
                "image", image_embed_grid_sizes
            ),
            image_grid_thw=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
            pixel_values_videos=MultiModalFieldConfig.flat_from_sizes(
                "video", video_pixel_grid_sizes
            ),
            video_embeds=MultiModalFieldConfig.flat_from_sizes(
                "video", video_embed_grid_sizes
            ),
            video_grid_thw=MultiModalFieldConfig.batched("video", keep_on_cpu=True),
        )


@MULTIMODAL_REGISTRY.register_processor(
    HCXVisionV2MultiModalProcessor,
    info=HCXVisionV2ProcessingInfo,
    dummy_inputs=HCXVisionV2DummyInputsBuilder,
)
class HCXVisionV2ForCausalLM(nn.Module, SupportsMultiModal, SupportsPP):
    """
    HyperCLOVAX-SEED Vision-Language Model (V2 architecture).

    Supports:
    - HyperCLOVAX-SEED-Think-32B: Vision + Text

    Uses Qwen2.5 Vision Transformer as the vision encoder.
    """

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "qkv": ["qkv"],  # For vision tower
    }

    # Weight mapping for loading HuggingFace checkpoints
    # NOTE: Order matters! Ignores (None) should come before renames to prevent
    # partial matches
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.": "",  # Remove model. prefix if present
            "vision_model.": "visual.",  # HF uses vision_model, we use visual
        },
        orig_to_new_substr={
            # Ignore modules not implemented in vLLM
            "discrete_vision_model": None,  # TextAlignedTokenizer
        },
    )

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        # Text config
        text_config = config.text_config
        if text_config.model_type in ["gpt2", "hyperclovax", "llama"]:
            text_config._attn_implementation = "sdpa"
        if text_config.model_type != "hyperclovax":
            text_config.logits_scaling = 1.0

        # Vision config
        vision_config = config.vision_config

        self.config = config
        self.vision_config = vision_config
        self.text_config = text_config
        self.vllm_config = vllm_config
        self.dtype = vllm_config.model_config.dtype

        # Initialize Qwen2.5 Vision Transformer
        self.visual = Qwen2_5_VisionTransformer(
            vision_config=vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "visual"),
        )

        # Linear projector (vision_hidden_size -> text_hidden_size)
        # For V2 model: mm_projector_type is "linear"
        vision_hidden_size = vision_config.hidden_size
        text_hidden_size = text_config.hidden_size

        # Check if out_hidden_size is defined (Qwen2.5-VL style)
        # The merger in Qwen2.5 VisionTransformer handles projection to out_hidden_size
        if hasattr(vision_config, "out_hidden_size"):
            out_hidden = vision_config.out_hidden_size
        else:
            out_hidden = vision_hidden_size

        # Always create Linear projector since HF checkpoint has mm_projector weights
        self.mm_projector = nn.Linear(out_hidden, text_hidden_size)

        # Language model
        self.lm_head_vocab_size = getattr(
            text_config, "padded_vocab_size", text_config.vocab_size
        )
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return V2_IMAGE_TOKEN
        if modality.startswith("video"):
            return V2_VIDEO_TOKEN

        raise ValueError("Only image or video modality is supported")

    def _parse_and_validate_image_input(
        self,
        **kwargs: object,
    ) -> HCXVisionV2ImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            return HCXVisionV2ImagePixelInputs(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        if image_embeds is not None:
            return HCXVisionV2ImageEmbeddingInputs(
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw,
            )

        return None

    def _parse_and_validate_video_input(
        self,
        **kwargs: object,
    ) -> HCXVisionV2VideoInputs | None:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)

        if pixel_values_videos is None and video_embeds is None:
            return None

        if pixel_values_videos is not None:
            return HCXVisionV2VideoPixelInputs(
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )

        if video_embeds is not None:
            return HCXVisionV2VideoEmbeddingInputs(
                video_embeds=video_embeds,
                video_grid_thw=video_grid_thw,
            )

        return None

    def _process_image_input(
        self,
        image_input: HCXVisionV2ImageInputs,
    ) -> tuple[torch.Tensor, ...]:
        """Process images through Qwen2.5 ViT and projector."""
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"]
            with set_forward_context(None, self.vllm_config):
                image_embeds = self.visual(pixel_values, grid_thw=grid_thw_list)

        # Apply projector
        image_embeds = self.mm_projector(image_embeds)

        # Split concatenated embeddings for each image
        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return image_embeds.split(sizes)

    def _process_video_input(
        self,
        video_input: HCXVisionV2VideoInputs,
    ) -> tuple[torch.Tensor, ...]:
        """Process videos through Qwen2.5 ViT and projector."""
        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"].type(self.visual.dtype)
        else:
            pixel_values_videos = video_input["pixel_values_videos"]
            with set_forward_context(None, self.vllm_config):
                video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw_list)

        # Apply projector
        video_embeds = self.mm_projector(video_embeds)

        # Split concatenated embeddings for each video
        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return video_embeds.split(sizes)

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}

        for input_key in kwargs:
            if (
                input_key in ("pixel_values", "image_embeds")
                and "image" not in modalities
            ):
                modalities["image"] = self._parse_and_validate_image_input(**kwargs)
            if (
                input_key in ("pixel_values_videos", "video_embeds")
                and "video" not in modalities
            ):
                modalities["video"] = self._parse_and_validate_video_input(**kwargs)

        return modalities

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def embed_multimodal(
        self,
        **kwargs: object,
    ) -> MultiModalEmbeddings:
        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not modalities:
            return []

        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        for modality in modalities:
            if modality == "image":
                image_input = modalities["image"]
                if image_input is not None:
                    image_embeddings = self._process_image_input(image_input)
                    multimodal_embeddings += tuple(image_embeddings)
            if modality == "video":
                video_input = modalities["video"]
                if video_input is not None:
                    video_embeddings = self._process_video_input(video_input)
                    multimodal_embeddings += tuple(video_embeddings)

        return multimodal_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
