# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Bailing MoE V3 vision-language model."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from typing import Any, cast, get_type_hints

import numpy as np
import regex as re
import torch
import torch.nn as nn
from transformers import AutoProcessor, BatchFeature

from vllm.config import VllmConfig
from vllm.config.multimodal import MultiModalDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems, VideoProcessorItems
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.processing.processor import (
    MultiModalPromptUpdates,
    PlaceholderFeaturesInfo,
)
from vllm.sequence import IntermediateTensors

from .bailing_moe_v3 import BailingMoeV3ForCausalLM
from .interfaces import (
    HasInnerState,
    IsHybrid,
    MultiModalEmbeddings,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
    _require_is_multimodal,
)
from .qwen2_5_vl import (
    Qwen2_5_VLDummyInputsBuilder,
    Qwen2_5_VLImageEmbeddingInputs,
    Qwen2_5_VLImageInputs,
    Qwen2_5_VLImagePixelInputs,
    Qwen2_5_VLVideoEmbeddingInputs,
    Qwen2_5_VLVideoInputs,
    Qwen2_5_VLVideoPixelInputs,
)
from .qwen2_vl import (
    Qwen2VLMultiModalDataParser,
    Qwen2VLProcessingInfo,
    _create_qwen2vl_field_factory,
)
from .qwen3_vl import Qwen3_VisionTransformer
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    _merge_multimodal_embeddings,
    maybe_prefix,
)
from .vision import run_dp_sharded_mrope_vision_model

IMAGE_TOKEN = "<|image_pad|>"
IMAGE_PLACEHOLDER = f"<|vision_start|>{IMAGE_TOKEN}<|vision_end|>"
VIDEO_TOKEN = "<|video_pad|>"
VIDEO_PLACEHOLDER = f"<|video_start|>{VIDEO_TOKEN}<|video_end|>"


def _get_bailing_mrope_positions(
    input_tokens: list[int],
    mm_features: list[MultiModalFeatureSpec],
    merge_size: int,
) -> tuple[torch.Tensor, int]:
    """Bailing's centered spatial positions and two-step temporal positions."""
    position_blocks: list[np.ndarray] = []
    consumed = 0

    segments = []
    for feature in mm_features:
        if feature.modality not in ("image", "video"):
            raise ValueError(f"Unsupported modality: {feature.modality}")
        assert feature.data is not None
        grid = feature.data[f"{feature.modality}_grid_thw"].data
        assert isinstance(grid, torch.Tensor)
        t, h, w = grid.tolist()
        mask = feature.mm_position.is_embed
        if feature.modality == "video" and mask is not None:
            indices = torch.where(mask)[0].tolist()
            frame_size = (h // merge_size) * (w // merge_size)
            if len(indices) != t * frame_size:
                raise ValueError("Video placeholder mask does not match the patch grid")
            for frame in range(t):
                frame_indices = indices[frame * frame_size : (frame + 1) * frame_size]
                first = frame_indices[0]
                if frame_indices != list(range(first, first + frame_size)):
                    raise ValueError("Video frame patch tokens must be contiguous")
                segments.append((feature.mm_position.offset + first, 1, h, w))
        else:
            segments.append((feature.mm_position.offset, t, h, w))

    for offset, t, h, w in sorted(segments):
        text_len = offset - consumed
        start = position_blocks[-1][0].max() + 1 if position_blocks else 0
        position_blocks.append(
            np.broadcast_to(np.arange(text_len), (3, text_len)) + start
        )

        h, w = h // merge_size, w // merge_size
        grid = np.indices((t, h, w)).reshape(3, -1)
        grid[0] = grid[0] * 2 + text_len + start
        grid[1] += grid[0] - (h - 1) // 2
        grid[2] += grid[0] - (w - 1) // 2
        position_blocks.append(grid)
        consumed = offset + t * h * w

    if consumed < len(input_tokens):
        start = position_blocks[-1][0].max() + 1 if position_blocks else 0
        text_len = len(input_tokens) - consumed
        position_blocks.append(
            np.broadcast_to(np.arange(text_len), (3, text_len)) + start
        )

    if not position_blocks:
        positions = np.empty((3, 0), dtype=np.int64)
        return torch.from_numpy(positions), 0

    positions = np.concatenate(position_blocks, axis=1).reshape(3, -1)
    position_delta = int(positions[0].max() + 1 - len(input_tokens))
    return torch.from_numpy(positions), position_delta


class BailingMoeV3VLProcessingInfo(Qwen2VLProcessingInfo):
    """Processing metadata for the remote Bailing multimodal processor."""

    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs: object) -> AutoProcessor:
        # The checkpoint supplies its processor through trust_remote_code.
        return self.ctx.get_hf_processor(**kwargs)

    def get_data_parser(self):
        return Qwen2VLMultiModalDataParser(
            self.get_hf_config().vision_config.spatial_merge_size,
            video_needs_metadata=True,
            expected_hidden_size=self._get_expected_hidden_size(),
            allow_missing_mm_embeddings=self.allow_missing_mm_embeddings,
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "video": None}

    def get_supported_mm_processor_kwargs(self) -> dict[str, set[str]]:
        processor = self.get_hf_processor()
        kwargs = get_type_hints(processor.valid_processor_kwargs)
        vision_kwargs = set(processor.image_processor.valid_kwargs.__annotations__)
        return {
            "text_kwargs": set(kwargs["text_kwargs"].__annotations__),
            "images_kwargs": vision_kwargs,
            "videos_kwargs": vision_kwargs,
        }

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {
            "image": self.get_max_image_tokens(),
            "video": self.get_max_video_tokens(seq_len, mm_counts),
        }


class BailingMoeV3VLDummyInputsBuilder(Qwen2_5_VLDummyInputsBuilder):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        return IMAGE_PLACEHOLDER * num_images + VIDEO_PLACEHOLDER * num_videos

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: MultiModalDummyOptions,
    ) -> MultiModalDataDict:
        target_width, target_height = self.info.get_image_size_with_most_features()
        return {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=mm_counts.get("image", 0),
                overrides=mm_options.get("image"),
            ),
            "video": self._get_dummy_videos(
                width=target_width,
                height=target_height,
                num_frames=self.info.get_num_frames_with_most_features(
                    seq_len, mm_counts
                ),
                num_videos=mm_counts.get("video", 0),
                overrides=mm_options.get("video"),
            ),
        }


class BailingMoeV3VLMultiModalProcessor(BaseMultiModalProcessor[Qwen2VLProcessingInfo]):
    def _get_hf_mm_text(self, mm_counts: Mapping[str, int]) -> str:
        return self.dummy_inputs.get_dummy_text(mm_counts)

    def _apply_hf_processor_main(
        self,
        mm_items: MultiModalDataItems,
        hf_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        hf_data, hf_kwargs, passthrough_data = self._get_hf_mm_inputs(
            mm_items, hf_kwargs
        )
        if not hf_data:
            return self._finalize_hf_mm_data(hf_data, hf_kwargs, passthrough_data)

        videos = hf_data.pop("videos", [])
        result = self._call_hf_processor(hf_data, hf_kwargs)
        if not videos:
            return self._finalize_hf_mm_data(
                hf_data, hf_kwargs, passthrough_data, result
            )

        processor = self.info.get_hf_processor(**hf_kwargs)
        image_processor = processor.image_processor
        if not getattr(image_processor, "use_interleaved_frame_timestamp", False):
            raise ValueError(
                "Bailing video inputs require interleaved frame timestamps"
            )
        metadata = mm_items.get_items("video", VideoProcessorItems).metadata
        video_kwargs = dict(
            cast(Mapping[str, object], hf_kwargs.get("videos_kwargs", {}))
        )
        pixels, grids, prompts = [], [], []
        assert isinstance(videos, list)
        for index, video in enumerate(videos):
            meta = metadata[index] if isinstance(metadata, list) else metadata
            meta = meta or {}
            if isinstance(video, tuple):
                video, meta = video
            video = torch.as_tensor(np.asarray(video))
            if not len(video):
                raise ValueError("A video must contain at least one frame")
            if "resmp_ts" in meta:
                timestamps = list(meta["resmp_ts"])
            else:
                fps = float(meta.get("fps", 2.0))
                if fps <= 0:
                    raise ValueError("Video FPS must be positive")
                indices = meta.get("frames_indices", list(range(len(video))))
                timestamps = [float(i) / fps for i in indices]
            if len(timestamps) != len(video):
                raise ValueError("Video timestamps must match the sampled frames")
            padding = -len(video) % image_processor.temporal_patch_size
            if padding:
                video = torch.cat((video, video[-1:].repeat(padding, 1, 1, 1)))
                timestamps += [timestamps[-1]] * padding
            data = image_processor(
                images=None,
                videos=[video],
                videos_timestamps_seconds=[timestamps],
                return_tensors="pt",
                **video_kwargs,
            )
            grids.append(data["video_grid_thw"][0].clone())
            pixels.append(
                data["pixel_values_videos"].to(self.info.ctx.model_config.dtype)
            )
            # Use the checkpoint's expansion to retain its exact timestamps/wrappers.
            text = processor._expand_video_tokens([VIDEO_PLACEHOLDER], dict(data))[0]
            prompts.append(
                torch.tensor(
                    processor.tokenizer.encode(text, add_special_tokens=False),
                    dtype=torch.long,
                )
            )
        result.update(
            pixel_values_videos=torch.cat(pixels),
            video_grid_thw=torch.stack(grids),
            video_prompt_ids=prompts,
        )
        return self._finalize_hf_mm_data(hf_data, hf_kwargs, passthrough_data, result)

    def _apply_prompt_updates(
        self,
        token_ids: list[int],
        mm_prompt_updates: MultiModalPromptUpdates,
    ) -> tuple[list[int], Mapping[str, list[PlaceholderFeaturesInfo]]]:
        if any(mm_prompt_updates.values()):
            tokenizer = self.info.get_tokenizer()
            text = tokenizer.decode(token_ids)
            # HF wraps each item and adds a newline before tokenization.
            for modality, token, placeholder in (
                ("image", IMAGE_TOKEN, IMAGE_PLACEHOLDER),
                ("video", VIDEO_TOKEN, VIDEO_PLACEHOLDER),
            ):
                if updates := mm_prompt_updates.get(modality):
                    wrapped = tokenizer.encode(placeholder, add_special_tokens=False)
                    suffix = "" if updates[0][0].target == wrapped else "\n"
                    text = re.sub(
                        f"{re.escape(placeholder)}|{re.escape(token)}",
                        lambda _, placeholder=placeholder, suffix=suffix: (
                            placeholder + suffix
                        ),
                        text,
                        count=len(updates),
                    )
            token_ids = tokenizer.encode(text, add_special_tokens=False)

        return super()._apply_prompt_updates(token_ids, mm_prompt_updates)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        merge_size = self.info.get_hf_config().vision_config.spatial_merge_size
        fields = _create_qwen2vl_field_factory(merge_size)(hf_inputs)
        return dict(
            fields,
            video_prompt_ids=MultiModalFieldConfig.batched("video", keep_on_cpu=True),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()
        merge_unit = self.info.get_hf_config().vision_config.spatial_merge_size ** 2

        def get_replacement(item_idx: int, modality: str, token_id: int):
            grid_thw = out_mm_kwargs[modality][item_idx][f"{modality}_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)
            item = out_mm_kwargs[modality][item_idx]
            if modality == "video" and "video_prompt_ids" in item:
                ids = item["video_prompt_ids"].data
                if isinstance(ids, torch.Tensor):
                    ids = ids.tolist()
                return PromptUpdateDetails.select_token_id(
                    cast(list[int], ids), token_id
                )
            return [token_id] * (int(grid_thw.prod()) // merge_unit)

        return [
            PromptReplacement(
                modality=modality,
                target=(
                    tokenizer.encode(VIDEO_PLACEHOLDER, add_special_tokens=False)
                    if modality == "video"
                    and "video_prompt_ids" in out_mm_kwargs[modality][0]
                    else [vocab[token]]
                ),
                replacement=partial(
                    get_replacement, modality=modality, token_id=vocab[token]
                ),
            )
            for modality, token in (("image", IMAGE_TOKEN), ("video", VIDEO_TOKEN))
            if modality in out_mm_kwargs
        ]


class BailingMoeV3VisionPatchMerger(nn.Module):
    """Normalize patches and merge each 2x2 group without an MLP."""

    def __init__(
        self,
        context_dim: int,
        spatial_merge_size: int,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * spatial_merge_size**2
        self.norm = nn.LayerNorm(context_dim, eps=norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        return hidden_states.reshape(-1, self.hidden_size)


class BailingMoeV3VisionTransformer(Qwen3_VisionTransformer):
    """Qwen3 ViT with Bailing's optional parameter-free patch merger."""

    def __init__(
        self,
        vision_config: Any,
        norm_eps: float = 1e-6,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        # Honor whole-tower exclusions before the parent creates linear layers.
        if (
            isinstance(quant_config, Fp8Config)
            and prefix in quant_config.ignored_layers
        ):
            quant_config = None

        super().__init__(
            vision_config,
            norm_eps=norm_eps,
            quant_config=quant_config,
            prefix=prefix,
        )
        if vision_config.disable_merger_proj:
            self.merger = BailingMoeV3VisionPatchMerger(
                context_dim=vision_config.hidden_size,
                spatial_merge_size=vision_config.spatial_merge_size,
                norm_eps=norm_eps,
            )

        # HF computes vision frequencies and the rotary products in FP32,
        # then casts the rotated Q/K back to the model dtype.
        head_dim = vision_config.hidden_size // vision_config.num_heads
        self.rotary_pos_emb = get_rope(
            head_size=head_dim,
            max_position=8192,
            is_neox_style=True,
            rope_parameters={"partial_rotary_factor": 0.5},
            dtype=torch.float32,
        )
        for block in self.blocks:
            block.attn.apply_rotary_emb.enable_fp32_compute = True


class BailingMoeV3VLProjector(nn.Module):
    """Map merged vision features to text width, optionally sharded over TP."""

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        disable_tp: bool = False,
    ) -> None:
        super().__init__()
        if (
            isinstance(quant_config, Fp8Config)
            and prefix in quant_config.ignored_layers
        ):
            quant_config = None

        self.linear_fc1 = ColumnParallelLinear(
            vision_hidden_size,
            text_hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "linear_fc1"),
            disable_tp=disable_tp,
        )
        self.act_fn = nn.GELU()
        self.linear_fc2 = RowParallelLinear(
            text_hidden_size,
            text_hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "linear_fc2"),
            disable_tp=disable_tp,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.linear_fc1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states, _ = self.linear_fc2(hidden_states)
        return hidden_states


class _BailingVisionEncoder(nn.Module):
    """Expose the tower and projector together to the image DP runner."""

    def __init__(
        self,
        visual: BailingMoeV3VisionTransformer,
        projector: BailingMoeV3VLProjector,
        vision_out_dim: int,
    ) -> None:
        super().__init__()
        self.visual = visual
        self.projector = projector
        self.vision_out_dim = vision_out_dim
        self.spatial_merge_size = visual.spatial_merge_size
        self.out_hidden_size = projector.linear_fc2.output_size

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: list[list[int]]
    ) -> torch.Tensor:
        features = self.visual(pixel_values, grid_thw=grid_thw)
        return self.projector(features[..., : self.vision_out_dim])


@MULTIMODAL_REGISTRY.register_processor(
    BailingMoeV3VLMultiModalProcessor,
    info=BailingMoeV3VLProcessingInfo,
    dummy_inputs=BailingMoeV3VLDummyInputsBuilder,
)
class BailingMoeV3VLForConditionalGeneration(
    nn.Module,
    HasInnerState,
    IsHybrid,
    SupportsMultiModal,
    SupportsPP,
    SupportsMRoPE,
):
    """Native vLLM wrapper for `BailingMoeV3VLConfig` checkpoints."""

    supports_encoder_tp_data = True

    packed_modules_mapping = {
        **BailingMoeV3ForCausalLM.packed_modules_mapping,
        # The vision checkpoint already stores Q/K/V in one tensor.
        "qkv": ["qkv"],
    }

    hf_to_vllm_mapper = BailingMoeV3ForCausalLM.hf_to_vllm_mapper | WeightsMapper(
        orig_to_new_prefix={
            # Match module names as well as parameter names for quantization.
            "model.visual": "visual",
            "model.linear_proj": "linear_proj",
            "linear_proj.0": "linear_proj.linear_fc1",
            "linear_proj.2": "linear_proj.linear_fc2",
            "lm_head": "language_model.lm_head",
            "model.": "language_model.model.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return IMAGE_PLACEHOLDER
        if modality.startswith("video"):
            return VIDEO_PLACEHOLDER
        raise ValueError(f"Unsupported modality: {modality}")

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        text_config = vllm_config.model_config.hf_config.text_config
        return BailingMoeV3ForCausalLM.get_mamba_state_shape_from_config(
            vllm_config.with_hf_config(text_config)
        )

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]:
        text_config = vllm_config.model_config.hf_config.text_config
        return BailingMoeV3ForCausalLM.get_mamba_state_dtype_from_config(
            vllm_config.with_hf_config(text_config)
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple:
        return BailingMoeV3ForCausalLM.get_mamba_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        vision_config = config.vision_config
        text_config = config.text_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.model_config = vllm_config.model_config
        multimodal_config = vllm_config.model_config.multimodal_config
        assert multimodal_config is not None
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"

        self.vision_out_dim = vision_config.out_hidden_size
        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = BailingMoeV3VisionTransformer(
                vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )
            self.linear_proj = BailingMoeV3VLProjector(
                vision_hidden_size=self.vision_out_dim,
                text_hidden_size=text_config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "linear_proj"),
                disable_tp=self.use_data_parallel,
            )

        with self._mark_language_model(vllm_config):
            self.language_model = BailingMoeV3ForCausalLM(
                vllm_config=vllm_config.with_hf_config(text_config),
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> Qwen2_5_VLImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None
        if pixel_values is not None:
            return Qwen2_5_VLImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
        return Qwen2_5_VLImageEmbeddingInputs(
            type="image_embeds",
            image_embeds=image_embeds,
            image_grid_thw=image_grid_thw,
        )

    def _parse_and_validate_video_input(
        self, **kwargs: object
    ) -> Qwen2_5_VLVideoInputs | None:
        pixels = kwargs.get("pixel_values_videos")
        embeds = kwargs.get("video_embeds")
        grid = kwargs.get("video_grid_thw")
        if pixels is not None:
            return Qwen2_5_VLVideoPixelInputs(
                type="pixel_values_videos",
                pixel_values_videos=pixels,
                video_grid_thw=grid,
                second_per_grid_ts=None,
            )
        if embeds is not None:
            return Qwen2_5_VLVideoEmbeddingInputs(
                type="video_embeds", video_embeds=embeds, video_grid_thw=grid
            )
        return None

    def _process_vision_input(
        self,
        vision_input: Qwen2_5_VLImageInputs | Qwen2_5_VLVideoInputs,
        modality: str,
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = vision_input[f"{modality}_grid_thw"]
        assert grid_thw.ndim == 2
        embeds_key = f"{modality}_embeds"
        if vision_input["type"] == embeds_key:
            embeddings = vision_input[embeds_key]
        else:
            pixels = vision_input[vision_input["type"]].type(self.visual.dtype)
            if self.use_data_parallel:
                # Keep the adapter transient to preserve checkpoint module names.
                encoder = _BailingVisionEncoder(
                    self.visual, self.linear_proj, self.vision_out_dim
                )
                return run_dp_sharded_mrope_vision_model(
                    encoder, pixels, grid_thw.tolist(), rope_type="rope_3d"
                )
            features = self.visual(pixels, grid_thw=grid_thw)
            embeddings = self.linear_proj(features[..., : self.vision_out_dim])
        sizes = (grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        return embeddings.to(dtype=self.visual.dtype).split(sizes)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        embeddings: tuple[torch.Tensor, ...] = ()
        seen: set[str] = set()
        parse: Callable[..., Qwen2_5_VLImageInputs | Qwen2_5_VLVideoInputs | None]
        # Match the modality order supplied by the multimodal runner.
        for key in kwargs:
            if key in ("pixel_values", "image_embeds"):
                modality = "image"
                parse = self._parse_and_validate_image_input
            elif key in ("pixel_values_videos", "video_embeds"):
                modality = "video"
                parse = self._parse_and_validate_video_input
            else:
                continue
            if modality in seen:
                continue
            seen.add(modality)
            vision_input = parse(**kwargs)
            if vision_input is not None:
                embeddings += self._process_vision_input(vision_input, modality)
        return embeddings

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
        )
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=_require_is_multimodal(is_multimodal),
        )

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        return _get_bailing_mrope_positions(
            input_tokens, mm_features, self.config.vision_config.spatial_merge_size
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
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="linear_proj",
            tower_model="visual.",
        )

    def get_mm_lora_token_counts(
        self,
        *,
        modality: str,
        mm_kwargs: MultiModalKwargsItem | None,
        num_mm_embeds: int,
    ) -> tuple[int, int | None]:
        del modality, mm_kwargs
        return num_mm_embeds * self.visual.spatial_merge_size**2, num_mm_embeds
