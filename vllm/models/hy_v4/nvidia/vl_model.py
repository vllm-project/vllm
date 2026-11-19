# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The HunYuan team.

from collections.abc import Iterable
from typing import Annotated

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.fusion.mm_input_norm import build_mm_input_norm
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsEncoderCudaGraph,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.transformers_utils.processors.hy_v4 import (
    vision_token_count,
    vision_token_grid,
)
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .mm_preprocess import (
    HYV4VLDummyInputsBuilder,
    HYV4VLMultiModalProcessor,
    HYV4VLProcessingInfo,
)
from .model import HYV4ForCausalLM
from .vision import HYV4VisionTransformer
from .vl_cudagraph import HYV4VLEncoderCudaGraphMixin


class HYV4VLImagePixelInputs(TensorSchema):
    """Flattened image patches and per-image [T, H, W] grids."""

    pixel_values: Annotated[torch.Tensor, TensorShape("np", "cps")]
    image_grid_thw: Annotated[torch.Tensor, TensorShape("ni", 3)]


class HYV4VLVideoPixelInputs(TensorSchema):
    """Flattened video patches and per-video [T, H, W] grids."""

    pixel_values_videos: Annotated[torch.Tensor, TensorShape("np", "cps")]
    video_grid_thw: Annotated[torch.Tensor, TensorShape("nv", 3)]


@MULTIMODAL_REGISTRY.register_processor(
    HYV4VLMultiModalProcessor,
    info=HYV4VLProcessingInfo,
    dummy_inputs=HYV4VLDummyInputsBuilder,
)
class HYV4VLForConditionalGeneration(
    nn.Module,
    HYV4VLEncoderCudaGraphMixin,
    SupportsMultiModal,
    SupportsEncoderCudaGraph,
    SupportsMRoPE,
    SupportsPP,
):
    """HYV4 VL composition using the platform-selected HYV4 text model."""

    supports_mm_device_do_normalize = True
    language_model_architecture = "HYV4ForCausalLM"

    packed_modules_mapping = {
        **HYV4ForCausalLM.packed_modules_mapping,
        "qkv": ["q_proj", "k_proj", "v_proj"],
    }
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "vit.vit.": "visual.",
            "vit.": "visual.",
            "model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
        },
        orig_to_new_substr={
            "self_attn.query_layernorm": "self_attn.q_norm",
            "self_attn.key_layernorm": "self_attn.k_norm",
            "mlp.gate.wg": "mlp.gate",
        },
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str:
        if modality in ("image", "video"):
            return f"<{modality}>"
        raise ValueError(f"Unsupported HYV4 VL modality: {modality}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.model_config = vllm_config.model_config
        self.config = vllm_config.model_config.hf_config
        self.multimodal_config = vllm_config.model_config.get_multimodal_config()
        enabled = any(
            self.multimodal_config.get_limit_per_prompt(m) for m in ("image", "video")
        )
        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = None
            if enabled:
                quant = vllm_config.quant_config
                self.visual = HYV4VisionTransformer(
                    self.config.vision_config,
                    quant_config=quant,
                    prefix=maybe_prefix(prefix, "visual"),
                    input_norm=build_mm_input_norm(vllm_config.model_config),
                )
        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=self.config.text_config,
                architectures=[self.language_model_architecture],
                prefix=maybe_prefix(prefix, "language_model"),
            )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> HYV4VLImagePixelInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        if pixel_values is None:
            return None
        if isinstance(pixel_values, list):
            pixel_values = torch.cat(
                [item.reshape(-1, item.shape[-1]) for item in pixel_values]
            )
        elif isinstance(pixel_values, torch.Tensor) and pixel_values.ndim > 0:
            pixel_values = pixel_values.reshape(-1, pixel_values.shape[-1])
        grid = kwargs.pop("image_grid_thw", None)
        if isinstance(grid, list):
            grid = torch.stack(grid)
        if isinstance(grid, torch.Tensor) and grid.ndim > 0:
            grid = grid.reshape(-1, grid.shape[-1])
        config = self.config.vision_config
        return HYV4VLImagePixelInputs(
            pixel_values=pixel_values,
            image_grid_thw=grid,
            resolve_bindings={"cps": config.num_channels * config.patch_size**2},
        )

    def _parse_and_validate_video_input(
        self, **kwargs: object
    ) -> HYV4VLVideoPixelInputs | None:
        pixel_values = kwargs.pop("pixel_values_videos", None)
        if pixel_values is None:
            return None
        if isinstance(pixel_values, list):
            pixel_values = torch.cat(
                [item.reshape(-1, item.shape[-1]) for item in pixel_values]
            )
        elif isinstance(pixel_values, torch.Tensor) and pixel_values.ndim > 0:
            pixel_values = pixel_values.reshape(-1, pixel_values.shape[-1])
        grid = kwargs.pop("video_grid_thw", None)
        if isinstance(grid, list):
            grid = torch.stack(grid)
        if isinstance(grid, torch.Tensor) and grid.ndim > 0:
            grid = grid.reshape(-1, grid.shape[-1])
        config = self.config.vision_config
        return HYV4VLVideoPixelInputs(
            pixel_values_videos=pixel_values,
            video_grid_thw=grid,
            resolve_bindings={"cps": config.num_channels * config.patch_size**2},
        )

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        if (
            kwargs.get("image_embeds") is not None
            or kwargs.get("video_embeds") is not None
        ):
            raise ValueError("HYV4 VL does not support precomputed visual embeddings")
        outputs = []
        parsers = {
            "pixel_values": ("image", self._parse_and_validate_image_input),
            "pixel_values_videos": ("video", self._parse_and_validate_video_input),
        }
        for key, value in kwargs.items():
            if key not in parsers or value is None:
                continue
            modality, parse_input = parsers[key]
            if self.visual is None:
                raise ValueError(
                    "HYV4 visual inputs are disabled by limit_mm_per_prompt"
                )
            mm_input = parse_input(**kwargs)
            assert mm_input is not None
            grids = mm_input[f"{modality}_grid_thw"].tolist()
            counts = [
                vision_token_count(g, self.config.vision_config, modality)
                for g in grids
            ]
            embeds = self.visual(mm_input[key], grids)
            for embedding, count in zip(embeds, counts, strict=True):
                if embedding.shape != (count, self.config.text_config.hidden_size):
                    raise ValueError(
                        f"{modality} visual embeddings do not match token grid"
                    )
            outputs.extend(embeds)
        return tuple(outputs)

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def compute_logits(self, hidden_states):
        return self.language_model.compute_logits(hidden_states)

    def get_expert_mapping(self):
        return self.language_model.get_expert_mapping()

    def get_mm_mapping(self):
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="visual.perceive",
            tower_model=["visual.embeddings", "visual.layers"],
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_prefixes=["visual."] if self.visual is None else None,
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        tokens = torch.tensor(input_tokens, dtype=torch.long)
        positions = torch.arange(len(input_tokens)).repeat(3, 1)
        if not input_tokens or not mm_features:
            return positions, 0
        data = MultiModalFeatureSpec.gather_kwargs(
            mm_features, {"image_grid_thw", "video_grid_thw"}
        )
        image_grids = [grid.tolist() for grid in data.get("image_grid_thw", [])]
        video_grids = [grid.tolist() for grid in data.get("video_grid_thw", [])]
        config = self.config
        vision = config.vision_config
        extra = int(bool(vision.cat_extra_token))
        starts = torch.where(tokens == config.image_start_token_id)[0].tolist()
        ends = torch.where(tokens == config.image_end_token_id)[0].tolist()
        if len(starts) != len(ends):
            raise ValueError("HYV4 image/frame start and end tokens must be paired")
        start_pos = start_index = mm_index = scan = 0
        image_index = video_index = -1
        inside_video = False
        while mm_index < len(starts):
            marker = starts[mm_index]
            pos = marker + 1 + extra
            while scan < marker:
                if input_tokens[scan] == config.video_start_token_id:
                    inside_video = True
                elif input_tokens[scan] == config.video_end_token_id:
                    inside_video = False
                scan += 1
            is_video = inside_video or (
                image_index + 1 >= len(image_grids)
                and video_index + 1 < len(video_grids)
            )
            if start_pos < pos:
                positions[:, start_pos:pos] = (
                    torch.arange(pos - start_pos) + start_index
                )
                start_index += pos - start_pos
            if is_video:
                video_index += 1
                if video_index >= len(video_grids):
                    raise ValueError("Video placeholders do not match video_grid_thw")
                frames, rows, cols = vision_token_grid(
                    video_grids[video_index], vision, "video"
                )
            else:
                image_index += 1
                if image_index >= len(image_grids):
                    raise ValueError("Image placeholders do not match image_grid_thw")
                frames, rows, cols = vision_token_grid(
                    image_grids[image_index], vision, "image"
                )
            if mm_index + frames > len(starts) or mm_index + frames > len(ends):
                raise ValueError("Missing HYV4 image/frame boundary tokens")
            count = rows * cols
            for frame in range(frames):
                begin = starts[mm_index + frame] + 1 + extra
                end = begin + count
                pad_id = config.video_token_id if is_video else config.image_token_id
                if end + extra != ends[mm_index + frame] or not torch.all(
                    tokens[begin - extra : end + extra] == pad_id
                ):
                    raise ValueError(
                        "HYV4 visual placeholders do not match "
                        "frame boundaries and grid"
                    )
                positions[0, begin:end] = start_index
                positions[1, begin:end] = (
                    torch.arange(rows).repeat_interleave(cols) + start_index
                )
                positions[2, begin:end] = torch.arange(cols).repeat(rows) + start_index
                start_index += max(rows, cols)
                if is_video and frame + 1 < frames:
                    bridge_start = ends[mm_index + frame]
                    bridge_end = starts[mm_index + frame + 1] + 1
                    positions[:, bridge_start:bridge_end] = (
                        torch.arange(bridge_end - bridge_start) + start_index
                    )
                    start_index += bridge_end - bridge_start
            start_pos = ends[mm_index + frames - 1] if is_video else pos + count
            mm_index += frames
        if start_pos < len(input_tokens):
            positions[:, start_pos:] = (
                torch.arange(len(input_tokens) - start_pos) + start_index
            )
        if image_index + 1 != len(image_grids) or video_index + 1 != len(video_grids):
            raise ValueError("Unused HYV4 media grids after position construction")
        return positions, int(positions.max()) + 1 - len(input_tokens)
