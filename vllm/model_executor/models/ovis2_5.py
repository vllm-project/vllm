# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PyTorch Ovis model."""

from collections.abc import Iterable, Mapping
from functools import partial
from typing import Annotated, Literal

import torch
import torch.nn as nn
from transformers import BaseImageProcessor, BatchFeature, PretrainedConfig

from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.ovis import VisualEmbedding
from vllm.model_executor.models.siglip2navit import Siglip2NavitModel
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    flatten_bn,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import ImageSize, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.processors.ovis2_5 import Ovis2_5Processor
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP

IMAGE_TOKEN = "<image>"
VIDEO_TOKEN = "<video>"
INDICATOR_IDS = [-301, -302, -303, -304]

IMAGE_PAD_TOKEN_MAP = {
    "gemma2": "<unused0>",
    "llama": "<|reserved_special_token_0|>",
    "qwen2": "<|image_pad|>",
    "qwen3": "<|image_pad|>",
}
IMAGE_PAD_TOKEN_ID_MAP = {
    "gemma2": 7,
    "llama": 128002,
    "qwen2": 151655,
    "qwen3": 151655,
}


class Ovis2_5ImagePatchInputs(TensorSchema):
    """
    Dimensions:
        - bnp: Batch size * number of images * number of patches
        - patch_size: patch_size_x * patch_size_y * num_channels
        - patch_indicators: Batch size * (number of patches + 1)
        - bn: Batch size * number of images
    """

    type: Literal["image_patches"]
    flat_data: Annotated[torch.Tensor, TensorShape("bnp", "patch_size")]
    indicator_tokens: Annotated[torch.Tensor, TensorShape("patch_indicators")]
    patches_per_item: Annotated[list[int], TensorShape("bn")]
    grids: Annotated[torch.Tensor, TensorShape("bn", 3)]
    # This is used to restore the first two dimensions of `flat_data`.


class Ovis2_5VideoPatchInputs(TensorSchema):
    """
    Dimensions:
        - bnp: Batch size * number of videos * number of patches
        - patch_size: patch_size_x * patch_size_y * num_channels
        - patch_indicators: Batch size * (number of patches + 1)
        - bn: Batch size * number of videos
    """

    type: Literal["video_patches"]
    flat_data: Annotated[torch.Tensor, TensorShape("bnp", "patch_size")]
    indicator_tokens: Annotated[torch.Tensor, TensorShape("patch_indicators")]
    patches_per_item: Annotated[list[int], TensorShape("bn")]
    grids: Annotated[torch.Tensor, TensorShape("bn", 3)]
    # This is used to restore the first two dimensions of `flat_data`.


class VisualTokenizer(torch.nn.Module):
    """
    VIT
    """

    def __init__(
        self,
        config: PretrainedConfig,
        visual_vocab_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
        attn_backend_override: AttentionBackendEnum | None = None,
    ):
        super().__init__()
        self.config = config
        self.vit = self._init_backbone(
            config=config,
            quant_config=quant_config,
            prefix=f"{prefix}.vit",
            use_data_parallel=use_data_parallel,
            attn_backend_override=attn_backend_override,
        )
        # reserved tokens for INDICATOR_IDS
        head_dim = visual_vocab_size - len(INDICATOR_IDS)
        self.head = torch.nn.Sequential(
            ReplicatedLinear(
                self.config.hidden_size * self.config.hidden_stride**2,
                head_dim,
                bias=False,
                return_bias=False,
            ),
            torch.nn.LayerNorm(head_dim),
        )

    def _init_backbone(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
        attn_backend_override: AttentionBackendEnum | None = None,
    ):
        model_type = config.model_type
        if model_type == "siglip2_navit":
            return Siglip2NavitModel(
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                use_data_parallel=use_data_parallel,
                attn_backend_override=attn_backend_override,
            )
        raise ValueError(f"Unsupported visual tokenizer model_type: {model_type}")

    @property
    def dtype(self) -> torch.dtype:
        return next(self.head.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.head.parameters()).device

    def tokenize(self, logits: torch.Tensor) -> torch.Tensor:
        tokens = torch.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)
        return tokens

    def encode(
        self, pixel_values: torch.Tensor, grid_thws: torch.Tensor
    ) -> torch.Tensor:
        features = self.vit(pixel_values, grid_thws)
        # refer to qwen2.5-vl patchmerger
        seq_len, _ = features.shape
        features = features.reshape(seq_len // (self.config.hidden_stride**2), -1)

        return features

    def forward(
        self, pixel_values: torch.Tensor, grid_thws: torch.Tensor
    ) -> torch.Tensor:
        features = self.encode(pixel_values, grid_thws)
        logits = self.head(features)
        tokens = self.tokenize(logits)
        # tokens' shape is [#Token, VocabSize-4],
        # so padding with [#Token, 4], after which,
        # tokens' shape should become [#Token, VocabSize];
        tokens = torch.nn.functional.pad(
            tokens,
            (0, len(INDICATOR_IDS)),
            mode="constant",
            value=0,
        )
        return tokens


class Ovis2_5ProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs):
        vit_config = self.get_hf_config().vit_config
        return self.ctx.get_hf_processor(
            Ovis2_5Processor,
            image_pad_token=self.get_image_pad_token(),
            patch_size=vit_config.patch_size,
            hidden_stride=vit_config.hidden_stride,
            temporal_patch_size=vit_config.temporal_patch_size,
        )

    def get_image_pad_token(self) -> str:
        hf_text_config = self.get_hf_config().get_text_config()
        text_model_type = hf_text_config.model_type
        return IMAGE_PAD_TOKEN_MAP.get(text_model_type)

    def get_image_processor(self) -> BaseImageProcessor:
        return self.get_hf_processor().image_processor  # type: ignore

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "video": 1}

    def get_image_size_with_most_features(self) -> ImageSize:
        # NOTE(myselvess): max_pixels 1792 * 1792 hardcoded in original code
        # TODO(myselvess): Be adjusted based on the max_pixels
        return ImageSize(width=1792, height=1792)

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int = 1,
    ) -> tuple[ImageSize, int]:
        hf_config = self.get_hf_config()
        vit_config = hf_config.vit_config
        patch_size = vit_config.patch_size
        temporal_patch_size = vit_config.temporal_patch_size
        # NOTE: Frames are padded to be divisible by `temporal_patch_size`
        # https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L294
        padded_num_frames = num_frames + (-num_frames % temporal_patch_size)
        grid_t = max(padded_num_frames // temporal_patch_size, 1)
        grid_h = image_height // patch_size
        grid_w = image_width // patch_size
        num_patches = grid_t * grid_h * grid_w
        num_vision_tokens = num_patches
        return num_vision_tokens

    def get_max_image_tokens(self) -> int:
        target_width, target_height = self.get_image_size_with_most_features()
        return self.get_num_image_tokens(
            image_width=target_width, image_height=target_height
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

    def get_num_frames_with_most_features(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        max_images = mm_counts.get("image", 0)
        max_videos = mm_counts.get("video", 0)
        max_image_tokens = self.get_max_image_tokens() * max_images
        max_total_frames = self._get_max_video_frames(seq_len - max_image_tokens)
        max_frames_per_video = max_total_frames // max(max_videos, 1)
        return max(max_frames_per_video, 1)

    def get_num_video_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int,
        image_processor: BaseImageProcessor | None,
    ) -> int:
        num_video_tokens = self.get_num_image_tokens(
            image_width=image_width, image_height=image_height, num_frames=num_frames
        )
        return num_video_tokens

    def get_max_video_tokens(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        target_width, target_height = self.get_image_size_with_most_features()
        return self.get_num_video_tokens(
            image_width=target_width,
            image_height=target_height,
            num_frames=self.get_num_frames_with_most_features(seq_len, mm_counts),
            image_processor=None,
        )


class Ovis2_5DummyInputsBuilder(BaseDummyInputsBuilder[Ovis2_5ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        return IMAGE_TOKEN * num_images + VIDEO_TOKEN * num_videos

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        target_width, target_height = self.info.get_image_size_with_most_features()
        target_num_frames = self.info.get_num_frames_with_most_features(
            seq_len, mm_counts
        )

        image_overrides = mm_options.get("image") if mm_options else None
        video_overrides = mm_options.get("video") if mm_options else None

        mm_data = {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
                overrides=image_overrides,
            ),
            "video": self._get_dummy_videos(
                width=target_width,
                height=target_height,
                num_frames=target_num_frames,
                num_videos=num_videos,
                overrides=video_overrides,
            ),
        }
        return mm_data


class Ovis2_5MultiModalProcessor(BaseMultiModalProcessor[Ovis2_5ProcessingInfo]):
    def visual_indicators_to_visual_tokens(
        self,
        visual_indicators: list[int],
    ) -> list[int]:
        """
        Filter image indicators placeholders and convert them to corresponding
        tokens in visual tokenizer.
        """
        hf_config = self.info.get_hf_config()
        vte_vocab_size = hf_config.visual_vocab_size
        return [
            vte_vocab_size - len(INDICATOR_IDS) + abs(x + 300) - 1
            for x in visual_indicators
            if x < -300
        ]

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if not mm_data:
            # Avoid warning from HF logger for text-only input
            tokenizer = self.info.get_tokenizer()
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )
        hf_processor = self.info.get_hf_processor()

        if "videos" in mm_data:
            visual_indicators = [
                hf_processor.construct_visual_indicators((1, 1, 1), True)
                for grid in processed_outputs["video_grids"]
            ]
            indicator_tokens = [
                self.visual_indicators_to_visual_tokens(indicator)
                for indicator in visual_indicators
            ]
            processed_outputs["video_indicator_tokens"] = torch.tensor(indicator_tokens)
        if "images" in mm_data:
            visual_indicators = [
                hf_processor.construct_visual_indicators((1, 1, 1), False)
                for grid in processed_outputs["grids"]
            ]
            indicator_tokens = [
                self.visual_indicators_to_visual_tokens(indicator)
                for indicator in visual_indicators
            ]

            processed_outputs["indicator_tokens"] = torch.tensor(indicator_tokens)
        return processed_outputs

    def _apply_hf_processor_tokens_only(
        self,
        prompt_tokens: list[int],
    ) -> list[int]:
        return prompt_tokens

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            grids=MultiModalFieldConfig.batched("image"),
            indicator_tokens=MultiModalFieldConfig.batched("image"),
            video_pixel_values=MultiModalFieldConfig.batched("video"),
            video_indicator_tokens=MultiModalFieldConfig.batched("video"),
            video_grids=MultiModalFieldConfig.batched("video"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> list[PromptReplacement]:
        def get_replacement_ovis(item_idx, modality: str):
            if modality == "image":
                out_item = out_mm_kwargs["image"][item_idx]
                grid = out_item["grids"].data
            elif modality == "video":
                out_item = out_mm_kwargs["video"][item_idx]
                grid = out_item["video_grids"].data
            hf_processor = self.info.get_hf_processor()
            return hf_processor.construct_visual_placeholders(
                grid[0],
            )

        return [
            PromptReplacement(
                modality=modality,
                target=IMAGE_TOKEN if modality == "image" else VIDEO_TOKEN,
                replacement=partial(get_replacement_ovis, modality=modality),
            )
            for modality in ("image", "video")
        ]


@MULTIMODAL_REGISTRY.register_processor(
    Ovis2_5MultiModalProcessor,
    info=Ovis2_5ProcessingInfo,
    dummy_inputs=Ovis2_5DummyInputsBuilder,
)
class Ovis2_5(nn.Module, SupportsMultiModal, SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config: PretrainedConfig = config
        self.llm = init_vllm_registered_model(
            vllm_config=vllm_config.with_hf_config(config.text_config),
            prefix=maybe_prefix(prefix, "llm"),
        )

        attn_backend_override = (
            multimodal_config.mm_encoder_attn_backend
            if multimodal_config is not None
            else None
        )
        self.visual_tokenizer = VisualTokenizer(
            config=config.vit_config,
            visual_vocab_size=config.visual_vocab_size,
            quant_config=quant_config,
            prefix=f"{prefix}.visual_tokenizer",
            attn_backend_override=attn_backend_override,
        )

        self.vte = VisualEmbedding(config.visual_vocab_size, config.hidden_size)

        text_model_type = self.config.get_text_config().model_type
        self.image_pad_token_id = IMAGE_PAD_TOKEN_ID_MAP[text_model_type]

        self.make_empty_intermediate_tensors = (
            self.get_language_model().make_empty_intermediate_tensors
        )

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> Ovis2_5ImagePatchInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        indicator_tokens = kwargs.pop("indicator_tokens", None)
        grids = kwargs.pop("grids", None)
        if pixel_values is None and indicator_tokens is None:
            return None

        if pixel_values is not None and indicator_tokens is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError(
                    f"Incorrect type of pixel values. Got type: {type(pixel_values)}"
                )

            if not isinstance(indicator_tokens, (torch.Tensor, list)):
                raise ValueError(
                    "Incorrect type of indicator_tokens. "
                    f"Got type: {type(indicator_tokens)}"
                )

            return Ovis2_5ImagePatchInputs(
                type="image_patches",
                flat_data=flatten_bn(pixel_values, concat=True),
                patches_per_item=[
                    x.shape[0] // (self.config.vit_config.hidden_stride**2)
                    for x in pixel_values
                ],
                indicator_tokens=flatten_bn(indicator_tokens, concat=True),
                grids=flatten_bn(grids, concat=True),
            )

        raise AssertionError("This line should be unreachable.")

    def _parse_and_validate_video_input(
        self, **kwargs: object
    ) -> Ovis2_5VideoPatchInputs | None:
        pixel_values = kwargs.pop("video_pixel_values", None)
        indicator_tokens = kwargs.pop("video_indicator_tokens", None)
        grids = kwargs.pop("video_grids", None)
        if pixel_values is None and indicator_tokens is None:
            return None

        if pixel_values is not None and indicator_tokens is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError(
                    f"Incorrect type of pixel values. Got type: {type(pixel_values)}"
                )

            if not isinstance(indicator_tokens, (torch.Tensor, list)):
                raise ValueError(
                    "Incorrect type of indicator_tokens. "
                    f"Got type: {type(indicator_tokens)}"
                )

            return Ovis2_5VideoPatchInputs(
                type="video_patches",
                flat_data=flatten_bn(pixel_values, concat=True),
                patches_per_item=[
                    x.shape[0] // (self.config.vit_config.hidden_stride**2)
                    for x in pixel_values
                ],
                indicator_tokens=flatten_bn(indicator_tokens, concat=True),
                grids=flatten_bn(grids, concat=True),
            )

        raise AssertionError("This line should be unreachable.")

    def _process_visual_input(
        self, visual_input: Ovis2_5ImagePatchInputs | Ovis2_5VideoPatchInputs
    ) -> MultiModalEmbeddings:
        image_patches_flat = visual_input["flat_data"]
        patches_per_image = visual_input["patches_per_item"]
        indicator_tokens = visual_input["indicator_tokens"]
        grid_thws = visual_input["grids"]

        indicator_per_image = list(
            map(lambda x: 2 if x > 1 else x + 2, patches_per_image)
        )

        target_dtype = self.visual_tokenizer.dtype
        visual_tokens = self.visual_tokenizer(
            image_patches_flat.to(target_dtype), grid_thws
        )

        visual_embeds = self.vte(visual_tokens)  # 1:1 numeric eq.
        indicator_embeds = self.vte(indicator_tokens)

        visual_embeds_per_image = visual_embeds.split(patches_per_image, dim=0)
        indicator_embeds_per_image = indicator_embeds.split(indicator_per_image)

        vision_embeddings = []
        for indicator, visual in zip(
            indicator_embeds_per_image, visual_embeds_per_image
        ):
            vision_embeddings_per_image = []
            visual = visual.unsqueeze(0)
            for i in range(visual.shape[0]):
                vision_embeddings_per_image.append(
                    torch.cat([indicator[i : i + 1], visual[i]], dim=0)
                )
            vision_embeddings_per_image.append(indicator[i + 1 :])
            vision_embeddings.append(torch.cat(vision_embeddings_per_image, dim=0))
        return tuple(vision_embeddings)

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if (
                input_key in ("pixel_values", "indicator_tokens", "grids")
                and "images" not in modalities
            ):
                modalities["images"] = self._parse_and_validate_image_input(**kwargs)
            if (
                input_key
                in ("video_pixel_values", "video_indicator_tokens", "video_grids")
                and "videos" not in modalities
            ):
                modalities["videos"] = self._parse_and_validate_video_input(**kwargs)

        return modalities

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not modalities:
            return []

        multimodal_embeddings: tuple[torch.Tensor, ...] = ()
        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in modalities:
            if modality == "images":
                image_input = modalities["images"]
                image_embeddings = self._process_visual_input(image_input)
                multimodal_embeddings += tuple(image_embeddings)
            if modality == "videos":
                video_input = modalities["videos"]
                video_embeddings = self._process_visual_input(video_input)
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

        # up until here we have a inputs_embeds 100% numerical identity
        # between the OG HF Transformers implementation and ours
        hidden_states = self.llm(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.llm.compute_logits(hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_language_model(self) -> torch.nn.Module:
        return self.llm
