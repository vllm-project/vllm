# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PyTorch Ovis model."""

from collections.abc import Hashable, Iterable, Mapping
from functools import partial
from typing import Annotated, Any, Literal, TypedDict

import torch
import torch.nn as nn
from transformers import BaseImageProcessor, BatchFeature, PreTrainedConfig

from vllm.config import VllmConfig
from vllm.config.multimodal import MultiModalDummyOptions
from vllm.inputs import MultiModalDataDict
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
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import ImageSize, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
)
from vllm.renderers import TokenizeParams
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.processors.ovis2_5 import (
    MIN_PIXELS,
    Ovis2_5Processor,
)
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import (
    MultiModalEmbeddings,
    SupportsEncoderCudaGraph,
    SupportsMultiModal,
    SupportsPP,
    supports_pp,
)

IMAGE_TOKEN = "<image>"
VIDEO_TOKEN = "<video>"
INDICATOR_IDS = [151672, 151673, 151674, 151675]
IMAGE_PAD_TOKEN_ID = 151655

# Keys of (pixel values, grids, indicator tokens) in mm_kwargs per modality.
_MM_INPUT_KEYS = {
    "image": ("pixel_values", "grids", "indicator_tokens"),
    "video": ("video_pixel_values", "video_grids", "video_indicator_tokens"),
}


def _pad_cumulative_seqlens_buffer(
    dst: torch.Tensor,
    src: torch.Tensor,
) -> None:
    # Repeat the last cumulative offset so padded entries are empty sequences.
    n = src.shape[0]
    dst[:n].copy_(src)
    dst[n:] = src[-1]


class Ovis2_5ImagePatchInputs(TensorSchema):
    """Dimensions:
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
    """Dimensions:
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


class Ovis2_5MultiModalInputs(TypedDict, total=False):
    images: Ovis2_5ImagePatchInputs | None
    videos: Ovis2_5VideoPatchInputs | None


class VisualTokenizer(torch.nn.Module):
    """VIT."""

    def __init__(
        self,
        config: PreTrainedConfig,
        visual_vocab_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.vit = self._init_backbone(
            config=config,
            quant_config=quant_config,
            prefix=f"{prefix}.vit",
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
        config: PreTrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        model_type = config.model_type
        if model_type == "siglip2_navit":
            return Siglip2NavitModel(
                config=config,
                quant_config=quant_config,
                prefix=prefix,
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
        self,
        pixel_values: torch.Tensor,
        grid_thws: torch.Tensor | None,
        encoder_metadata: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        features = self.vit(pixel_values, grid_thws, encoder_metadata=encoder_metadata)
        # refer to qwen2.5-vl patchmerger
        seq_len, _ = features.shape
        features = features.reshape(seq_len // (self.config.hidden_stride**2), -1)

        return features

    def forward(
        self, pixel_values: torch.Tensor, grid_thws: torch.Tensor
    ) -> torch.Tensor:
        features = self.encode(pixel_values, grid_thws)
        return self.features_to_tokens(features)

    def features_to_tokens(self, features: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        # extra_special_tokens adds tokens to the tokenizer lazily, on first
        # access, but the renderer deep-copies this tokenizer into a thread
        # pool right after this processor is constructed; the copies never
        # see mutations made afterwards. Force it now, before that happens,
        # on the tokenizer we actually use (get_hf_processor()'s cache can
        # otherwise hand back one bound to a self-loaded tokenizer instead).
        hf_processor = self.get_hf_processor()
        hf_processor.tokenizer = self.get_tokenizer()
        hf_processor.extra_special_tokens  # noqa: B018

    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs):
        vit_config = self.get_hf_config().vit_config
        return self.ctx.get_hf_processor(
            Ovis2_5Processor,
            patch_size=vit_config.patch_size,
            hidden_stride=vit_config.hidden_stride,
            temporal_patch_size=vit_config.temporal_patch_size,
        )

    def get_default_tok_params(self) -> TokenizeParams:
        return super().get_default_tok_params().with_kwargs(add_special_tokens=False)

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
    ) -> int:
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
        mm_options: MultiModalDummyOptions,
    ) -> MultiModalDataDict:
        target_width, target_height = self.info.get_image_size_with_most_features()
        target_num_frames = self.info.get_num_frames_with_most_features(
            seq_len, mm_counts
        )

        mm_data = {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=mm_counts.get("image", 0),
                overrides=mm_options.get("image"),
            ),
            "video": self._get_dummy_videos(
                width=target_width,
                height=target_height,
                num_frames=target_num_frames,
                num_videos=mm_counts.get("video", 0),
                overrides=mm_options.get("video"),
            ),
        }
        return mm_data


class Ovis2_5MultiModalProcessor(BaseMultiModalProcessor[Ovis2_5ProcessingInfo]):
    def visual_indicators_to_visual_tokens(
        self,
        visual_indicators: list[int],
    ) -> list[int]:
        """Filter image indicators placeholders and convert them to corresponding
        tokens in visual tokenizer.
        """
        hf_config = self.info.get_hf_config()
        vte_vocab_size = hf_config.visual_vocab_size
        return [
            vte_vocab_size - len(INDICATOR_IDS) + (x - INDICATOR_IDS[0])
            for x in visual_indicators
            if x >= INDICATOR_IDS[0]
        ]

    def _get_hf_mm_text(self, mm_counts: Mapping[str, int]) -> str:
        return self.dummy_inputs.get_dummy_text(mm_counts)

    def _postprocess_hf_mm_data(
        self,
        mm_data: Mapping[str, object],
        hf_processor_mm_kwargs: Mapping[str, object],
        processed_data: BatchFeature,
    ) -> BatchFeature:
        if not mm_data:
            return processed_data

        hf_processor = self.info.get_hf_processor()

        if "videos" in mm_data:
            visual_indicators = [
                hf_processor.construct_visual_indicators((1, 1, 1), True)
                for grid in processed_data["video_grids"]
            ]
            indicator_tokens = [
                self.visual_indicators_to_visual_tokens(indicator)
                for indicator in visual_indicators
            ]
            processed_data["video_indicator_tokens"] = torch.tensor(indicator_tokens)
        if "images" in mm_data:
            visual_indicators = [
                hf_processor.construct_visual_indicators((1, 1, 1), False)
                for grid in processed_data["grids"]
            ]
            indicator_tokens = [
                self.visual_indicators_to_visual_tokens(indicator)
                for indicator in visual_indicators
            ]

            processed_data["indicator_tokens"] = torch.tensor(indicator_tokens)

        return processed_data

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            grids=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
            indicator_tokens=MultiModalFieldConfig.batched("image"),
            video_pixel_values=MultiModalFieldConfig.batched("video"),
            video_indicator_tokens=MultiModalFieldConfig.batched("video"),
            video_grids=MultiModalFieldConfig.batched("video", keep_on_cpu=True),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> list[PromptReplacement]:
        hf_processor = self.info.get_hf_processor()
        placeholder = {
            "image": hf_processor.get_token_value("image_token"),
            "video": hf_processor.get_token_value("video_token"),
        }

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
                target=[placeholder[modality]],
                replacement=partial(get_replacement_ovis, modality=modality),
            )
            for modality in ("image", "video")
        ]


@MULTIMODAL_REGISTRY.register_processor(
    Ovis2_5MultiModalProcessor,
    info=Ovis2_5ProcessingInfo,
    dummy_inputs=Ovis2_5DummyInputsBuilder,
)
class Ovis2_5(nn.Module, SupportsMultiModal, SupportsPP, SupportsEncoderCudaGraph):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return IMAGE_TOKEN
        if modality.startswith("video"):
            return VIDEO_TOKEN

        raise ValueError("Only image or video modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config: PreTrainedConfig = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = vllm_config.model_config.multimodal_config

        with self._mark_language_model(vllm_config):
            self.llm = init_vllm_registered_model(
                vllm_config=vllm_config.with_hf_config(config.text_config),
                prefix=maybe_prefix(prefix, "llm"),
            )

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual_tokenizer = VisualTokenizer(
                config=config.vit_config,
                visual_vocab_size=config.visual_vocab_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual_tokenizer"),
            )
            self.vte = VisualEmbedding(config.visual_vocab_size, config.hidden_size)

        self.image_pad_token_id: int = IMAGE_PAD_TOKEN_ID

        language_model = self.get_language_model()
        assert supports_pp(language_model)
        self.make_empty_intermediate_tensors = (
            language_model.make_empty_intermediate_tensors
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

    def _parse_and_validate_multimodal_inputs(
        self, **kwargs: object
    ) -> Ovis2_5MultiModalInputs:
        modalities: Ovis2_5MultiModalInputs = {}

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
                assert image_input is not None
                image_embeddings = self._process_visual_input(image_input)
                multimodal_embeddings += tuple(image_embeddings)
            if modality == "videos":
                video_input = modalities["videos"]
                assert video_input is not None
                video_embeddings = self._process_visual_input(video_input)
                multimodal_embeddings += tuple(video_embeddings)

        return multimodal_embeddings

    # -- SupportsEncoderCudaGraph protocol methods --
    #
    # Only the ViT (`VisualTokenizer.encode`) is captured. The visual head,
    # softmax over the visual vocabulary and the `vte` lookup are a handful
    # of large GEMMs that gain nothing from CUDA graphs, so they run in
    # `postprocess_encoder_output` together with the indicator embeddings.

    def get_encoder_cudagraph_config(self):
        from vllm.v1.worker.encoder_cudagraph_defs import (
            EncoderCudaGraphConfig,
        )

        return EncoderCudaGraphConfig(
            modalities=["image", "video"],
            buffer_keys=[
                "pixel_values",
                "pos_embeds",
                "rotary_pos_emb_cos",
                "rotary_pos_emb_sin",
                "window_index",
                "reverse_indices",
                "cu_seqlens",
                "cu_window_seqlens",
                "max_seqlen_full",
                "max_seqlen_window",
            ],
            padding_logics={
                "cu_seqlens": _pad_cumulative_seqlens_buffer,
                "cu_window_seqlens": _pad_cumulative_seqlens_buffer,
            },
            out_hidden_size=self.config.hidden_size,
            max_frames_per_video=self.get_max_frames_per_video(),
        )

    def get_input_modality(
        self,
        mm_kwargs: dict[str, Any],
    ) -> str:
        if "pixel_values" in mm_kwargs:
            return "image"
        if "video_pixel_values" in mm_kwargs:
            return "video"
        raise AssertionError("This line should be unreachable.")

    def get_max_frames_per_video(self) -> int:
        info = MULTIMODAL_REGISTRY.get_processing_info(self.model_config)
        assert isinstance(info, Ovis2_5ProcessingInfo)
        return info.get_num_frames_with_most_features(
            seq_len=self.model_config.max_model_len,
            mm_counts={"video": self.multimodal_config.get_limit_per_prompt("video")},
        )

    def get_encoder_cudagraph_budget_range(
        self,
        vllm_config: VllmConfig,
    ) -> tuple[int, int]:
        # Min: smallest image allowed by the processor (MIN_PIXELS) plus the
        # start/end indicator tokens.
        vit_config = self.config.vit_config
        pixels_per_token = (vit_config.patch_size * vit_config.hidden_stride) ** 2
        min_budget = MIN_PIXELS // pixels_per_token + 2
        max_budget = min(
            vllm_config.scheduler_config.max_num_batched_tokens,
            self.model_config.max_model_len,
        )
        return (min_budget, max_budget)

    def _get_encoder_cudagraph_inputs(
        self,
        mm_kwargs: dict[str, Any],
    ) -> tuple[list[torch.Tensor], list[list[int]], list[torch.Tensor]]:
        """Return per-item (pixel values, [t, h, w] grid, indicator tokens)."""
        pv_key, grid_key, ind_key = _MM_INPUT_KEYS[self.get_input_modality(mm_kwargs)]
        pixel_values = list(mm_kwargs[pv_key])
        grids = [g.reshape(3).tolist() for g in mm_kwargs[grid_key]]
        indicator_tokens = list(mm_kwargs[ind_key])
        return pixel_values, grids, indicator_tokens

    def get_encoder_cudagraph_item_specs(
        self,
        mm_kwargs: dict[str, Any],
    ):
        from vllm.v1.worker.encoder_cudagraph_defs import EncoderItemSpec

        merge_unit = self.config.vit_config.hidden_stride**2
        _, grids, indicator_tokens = self._get_encoder_cudagraph_inputs(mm_kwargs)
        return [
            EncoderItemSpec(
                input_size=t * h * w,
                # Visual tokens plus the indicator tokens wrapped around them.
                output_tokens=t * h * w // merge_unit + indicators.shape[0],
            )
            for (t, h, w), indicators in zip(grids, indicator_tokens)
        ]

    def select_encoder_cudagraph_items(
        self,
        mm_kwargs: dict[str, Any],
        indices: list[int],
    ) -> dict[str, Any]:
        keys = _MM_INPUT_KEYS[self.get_input_modality(mm_kwargs)]
        return {key: [mm_kwargs[key][i] for i in indices] for key in keys}

    def postprocess_encoder_output(
        self,
        outputs: dict[str, torch.Tensor],
        indices: list[int],
        per_item_out_tokens: list[int],
        dest: dict[int, torch.Tensor] | list[torch.Tensor | None],
        clone: bool = False,
        batch_mm_kwargs: dict[str, Any] | None = None,
    ) -> None:
        assert batch_mm_kwargs is not None
        _, grids, indicator_tokens = self._get_encoder_cudagraph_inputs(batch_mm_kwargs)
        merge_unit = self.config.vit_config.hidden_stride**2
        num_visual_tokens = [t * h * w // merge_unit for t, h, w in grids]

        # The captured output buffer is sized to the token budget; only the
        # leading rows belong to real items.
        features = outputs["default"][: sum(num_visual_tokens)]
        visual_tokens = self.visual_tokenizer.features_to_tokens(features)
        visual_embeds = self.vte(visual_tokens).split(num_visual_tokens)

        # `torch.cat` always allocates, so the result never aliases the
        # graph output buffer and `clone` needs no extra handling.
        for idx, visual, indicators in zip(indices, visual_embeds, indicator_tokens):
            indicator_embeds = self.vte(indicators)
            dest[idx] = torch.cat(
                [indicator_embeds[:1], visual, indicator_embeds[1:]], dim=0
            )

    def prepare_encoder_cudagraph_capture_inputs(
        self,
        token_budget: int,
        max_batch_size: int,
        max_frames_per_batch: int,
        device: torch.device,
        dtype: torch.dtype,
        path: str = "default",
        axis_keys: tuple[Hashable, ...] | None = None,
    ):
        from vllm.v1.worker.encoder_cudagraph_defs import (
            EncoderCudaGraphCaptureInputs,
        )

        vit_config = self.config.vit_config
        stride = vit_config.hidden_stride
        # Ceil so that the total captured capacity is never below token_budget.
        per_item_tokens = (token_budget + max_batch_size - 1) // max_batch_size

        frames_per_item = max_frames_per_batch // max_batch_size
        if frames_per_item > 1:
            # Video-format grid so cu_seqlens is sized for one attention
            # sequence per frame.
            tokens_per_frame = (
                per_item_tokens + frames_per_item - 1
            ) // frames_per_item
            grid_config = [
                [frames_per_item, stride, tokens_per_frame * stride]
            ] * max_batch_size
        else:
            grid_config = [[1, stride, per_item_tokens * stride]] * max_batch_size

        total_patches = sum(t * h * w for t, h, w in grid_config)
        patch_dim = (
            vit_config.num_channels
            * vit_config.temporal_patch_size
            * vit_config.patch_size**2
        )
        # Contents are overwritten before every replay.
        dummy_pixel_values = torch.zeros(
            total_patches, patch_dim, device=device, dtype=dtype
        )

        # max_seqlen is baked into the graph at capture time, so capture
        # with upper bounds covering any replay: one item spanning the whole
        # budget for full attention, and one full window for window attention.
        encoder = self.visual_tokenizer.vit.vision_model.encoder
        window_merged = encoder.window_size // stride // encoder.patch_size
        max_num_seqs = max(max_batch_size, max_frames_per_batch)
        metadata = self.visual_tokenizer.vit.prepare_encoder_metadata(
            grid_config,
            max_num_seqs=max_num_seqs,
            max_num_window_seqs=encoder.get_encoder_cudagraph_max_window_seqs(
                token_budget, max_num_seqs
            ),
            max_seqlen_override=token_budget * stride**2,
            max_seqlen_window_override=(window_merged * stride) ** 2,
            device=device,
        )
        return EncoderCudaGraphCaptureInputs(
            values=metadata | {"pixel_values": dummy_pixel_values}
        )

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, Any],
        max_batch_size: int,
        max_frames_per_batch: int,
        path: str = "default",
    ):
        from vllm.v1.worker.encoder_cudagraph_defs import (
            EncoderCudaGraphReplayBuffers,
        )

        pixel_values, grids, _ = self._get_encoder_cudagraph_inputs(mm_kwargs)
        # Sized to the actual batch; the manager pads cu*_seqlens into the
        # captured buffers via `padding_logics`.
        metadata = self.visual_tokenizer.vit.prepare_encoder_metadata(grids)
        return EncoderCudaGraphReplayBuffers(
            values=metadata | {"pixel_values": torch.cat(pixel_values)}
        )

    def encoder_cudagraph_forward(
        self,
        values: dict[str, torch.Tensor],
        path: str = "default",
    ) -> torch.Tensor:
        pixel_values = values.pop("pixel_values")
        return self.visual_tokenizer.encode(pixel_values, None, encoder_metadata=values)

    def encoder_eager_forward(
        self,
        mm_kwargs: dict[str, Any],
        path: str = "default",
    ) -> torch.Tensor:
        pixel_values, grids, _ = self._get_encoder_cudagraph_inputs(mm_kwargs)
        return self.visual_tokenizer.encode(
            torch.cat(pixel_values).to(self.visual_tokenizer.dtype), grids
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
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
        return self.llm.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
