# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# copied from : https://github.com/huggingface/transformers
import ast
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from itertools import accumulate
from typing import Annotated, Literal

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import LayerNorm, LayerNorm2d
from timm.models.regnet import RegStage
from transformers import BatchFeature, CLIPVisionConfig, SiglipVisionConfig

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions, MultiModalConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import BaseMultiModalProcessorCache
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
    InputProcessingContext,
    PromptReplacement,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .clip import CLIPVisionModel
from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .siglip import SiglipVisionModel
from .utils import (
    AutoWeightsLoader,
    flatten_bn,
    init_vllm_registered_model,
    maybe_prefix,
)
from .vision import get_vision_encoder_info

EOT = "<|endofturn|>"
IMAGE_TOKEN: str = "<|dummy3|>"
VIDEO_TOKEN: str = "<|_unuse_missing_100270|>"


# Based on combine_frames_into_images in
# https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B/blob/main/processing_hyperclovax.py
def get_num_combined_frames(
    num_frames: int,
    max_grid_shape: tuple[int, int] = (3, 3),
) -> int:
    max_num_grids = max_grid_shape[0] * max_grid_shape[1]

    # Calculate the number of canvases needed.
    num_canvases = num_frames // max_num_grids
    leftover_frames = num_frames % max_num_grids

    return num_canvases + (leftover_frames > 0)


class HCXVisionImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - n: Number of images
        - g: Number of grids
        - c: Number of channels (3)
        - h: Height
        - w: Width
    """

    type: Literal["pixel_values"] = "pixel_values"
    pixel_values_images: Annotated[
        list[torch.Tensor], TensorShape("n", "g", 3, "h", "w", dynamic_dims={"g"})
    ]
    image_sizes_images: Annotated[torch.Tensor, TensorShape("n", 2)]


HCXVisionImageInputs = HCXVisionImagePixelInputs


class HCXVisionVideoPixelInputs(TensorSchema):
    """
    Dimensions:
        - n: Number of videos
        - f: Number of frames
        - g: Number of grids
        - c: Number of channels (3)
        - h: Height
        - w: Width
    """

    type: Literal["pixel_values_videos"] = "pixel_values_videos"
    pixel_values_videos: Annotated[
        list[list[torch.Tensor]],
        TensorShape("n", "f", "g", 3, "h", "w", dynamic_dims={"f", "g"}),
    ]


HCXVisionVideoInputs = HCXVisionVideoPixelInputs


class HCXVisionProcessingInfo(BaseProcessingInfo):
    def get_vision_encoder_info(self):
        return get_vision_encoder_info(self.get_hf_config())

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "video": None}

    def get_num_image_tokens(
        self,
        *,
        vision_query_length: int | list[int],
    ) -> int:
        if isinstance(vision_query_length, int):
            return vision_query_length
        else:
            return sum(vision_query_length)

    def get_num_video_tokens(
        self,
        *,
        vision_query_length: int | list[int],
    ) -> int:
        if isinstance(vision_query_length, int):
            return vision_query_length
        else:
            return sum(vision_query_length)

    def get_image_size_with_most_features(self) -> ImageSize:
        vision_encoder_info = self.get_vision_encoder_info()
        width = height = vision_encoder_info.get_image_size()
        return ImageSize(width=width, height=height)

    def get_max_image_tokens(self) -> int:
        target_width, target_height = self.get_image_size_with_most_features()

        return self.get_num_image_tokens(
            image_width=target_width,
            image_height=target_height,
        )


class HCXVisionDummyInputsBuilder(BaseDummyInputsBuilder[HCXVisionProcessingInfo]):
    def get_dummy_text(
        self,
        mm_counts: Mapping[str, int],
    ) -> str:
        dummy_text = IMAGE_TOKEN * mm_counts.get(
            "image", 0
        ) + VIDEO_TOKEN * mm_counts.get("video", 0)
        return dummy_text

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        target_width, target_height = self.info.get_image_size_with_most_features()
        target_num_frames = 32

        image_overrides = mm_options.get("image") if mm_options else None
        video_overrides = mm_options.get("video") if mm_options else None

        return {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
                overrides=image_overrides,
            ),
            "video": self._get_dummy_videos(
                width=target_width - 1,
                height=target_height - 1,
                num_frames=target_num_frames,
                num_videos=num_videos,
                overrides=video_overrides,
            ),
        }


class HCXVisionMultiModalProcessor(BaseMultiModalProcessor[HCXVisionProcessingInfo]):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        for video_idx, video_arr in enumerate(mm_data.get("videos", [])):
            if video_arr.dtype != np.uint8:
                mm_data["videos"][video_idx] = video_arr.astype(np.uint8)

        processed_outputs = self.info.ctx.call_hf_processor(
            hf_processor=self.info.get_hf_processor(**mm_kwargs),
            data=dict(
                text=prompt,
                images=None,
                videos=None,
            ),
        )  # text-only

        if len(mm_data) > 0:
            images = mm_data.get("images")
            videos = mm_data.get("videos")

            # batchify input as a single item
            _processed_outputs = self.info.ctx.call_hf_processor(
                hf_processor=self.info.get_hf_processor(**mm_kwargs),
                data=dict(
                    text=None,
                    images=None if images is None else [images],
                    videos=None if videos is None else [videos],
                ),
            )  # mm-only

            for k, v in _processed_outputs.items():
                if isinstance(v, list) and len(v) > 0:
                    assert len(v) == 1
                    _processed_outputs[k] = v[0]

            if images:
                _processed_outputs["image_sizes_images"] = torch.tensor(
                    _processed_outputs["image_sizes_images"]
                )
                _processed_outputs["vision_query_lengths_images"] = torch.tensor(
                    _processed_outputs["vision_query_lengths_images"]
                )

            if videos:
                _idx_per_video = [
                    0,
                    *accumulate(
                        get_num_combined_frames(len(video)) for video in videos
                    ),
                ]
                _processed_outputs["pixel_values_videos"] = [
                    _processed_outputs["pixel_values_videos"][
                        _idx_per_video[i] : _idx_per_video[i + 1]
                    ]
                    for i in range(len(videos))
                ]
                _processed_outputs["vision_query_lengths_videos"] = [
                    torch.tensor(
                        _processed_outputs["vision_query_lengths_videos"][
                            _idx_per_video[i] : _idx_per_video[i + 1]
                        ]
                    )
                    for i in range(len(videos))
                ]

            processed_outputs.update(_processed_outputs)

        return processed_outputs

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()
        placeholder = {
            "image": hf_config.image_token_id,
            "video": hf_config.video_token_id,
        }

        def get_replacement_hyperclovax(
            item_idx: int,
            modality: str,
            out_mm_kwargs: MultiModalKwargsItems,
        ):
            out_item = out_mm_kwargs[modality][item_idx]

            if modality == "image":
                lens = out_item["vision_query_lengths_images"].data.tolist()
                num_tokens = self.info.get_num_image_tokens(vision_query_length=lens)
            elif modality == "video":
                lens = out_item["vision_query_lengths_videos"].data.tolist()
                num_tokens = self.info.get_num_video_tokens(vision_query_length=lens)
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
                    get_replacement_hyperclovax,
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
        return dict(
            pixel_values_images=MultiModalFieldConfig.batched("image"),
            image_sizes_images=MultiModalFieldConfig.batched("image"),
            vision_query_lengths_images=MultiModalFieldConfig.batched("image"),
            pixel_values_videos=MultiModalFieldConfig.batched("video"),
            vision_query_lengths_videos=MultiModalFieldConfig.batched("video"),
        )


def _build_hcxvision_hf_info(
    ctx: InputProcessingContext,
) -> HCXVisionProcessingInfo:
    return HCXVisionProcessingInfo(ctx)


def _build_hcxvision_hf_processor(
    info: HCXVisionProcessingInfo,
    dummy_inputs: BaseDummyInputsBuilder[HCXVisionProcessingInfo],
    *,
    cache: BaseMultiModalProcessorCache | None = None,
) -> BaseMultiModalProcessor:
    if isinstance(info, HCXVisionProcessingInfo):
        return HCXVisionMultiModalProcessor(
            info,
            dummy_inputs,  # type: ignore
            cache=cache,
        )

    raise NotImplementedError(type(info))


def init_vision_tower_for_hcxvision(
    vision_config,
    quant_config: QuantizationConfig | None,
    multimodal_config: MultiModalConfig | None,
    *,
    use_nth_layer: int | None = None,
    require_post_norm: bool | None = None,
    prefix: str = "",
) -> CLIPVisionModel | SiglipVisionModel:
    num_hidden_layers = vision_config.num_hidden_layers
    if not isinstance(use_nth_layer, int):
        pass
    elif use_nth_layer >= 0:
        num_hidden_layers = use_nth_layer + 1
    else:
        num_hidden_layers = num_hidden_layers + use_nth_layer + 1

    if isinstance(vision_config, CLIPVisionConfig):
        return CLIPVisionModel(
            vision_config,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            num_hidden_layers_override=num_hidden_layers,
            require_post_norm=require_post_norm,
            prefix=prefix,
        )
    elif isinstance(vision_config, SiglipVisionConfig):
        return SiglipVisionModel(
            vision_config,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            num_hidden_layers_override=num_hidden_layers,
            require_post_norm=require_post_norm,
            prefix=prefix,
        )

    msg = f"Unsupported vision config: {type(vision_config)}"
    raise NotImplementedError(msg)


class HCXVisionMlp(nn.Module):
    def __init__(
        self,
        mm_projector_type,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mm_projector_type = mm_projector_type
        if self.mm_projector_type == "mlp":
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden_features, out_features)
        elif self.mm_projector_type == "inverted_mlp":
            self.fc1 = nn.Linear(in_features, 2 * hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Linear(2 * hidden_features, out_features)
        else:
            raise NotImplementedError(
                "{} is not implemented".format(self.mm_projector_type)
            )

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class HCXVisionCAbstractor(nn.Module):
    """
    This module is based on C-Abstractor, whose license is under apache-2.0.
    You can check the original code at
    https://github.com/khanrc/honeybee/blob/main/honeybee/projectors/projectors.py
    and we made necessary modifications.
    """

    def __init__(
        self,
        num_queries: int,
        num_input_tokens: int,
        encoder_hidden_size: int,
        hidden_size: int,
        output_hidden_size: int,
        pos_emb: bool = True,
        prenorm: bool = False,
    ):
        super().__init__()
        self.num_input_tokens = num_input_tokens
        self.output_hidden_size = output_hidden_size

        # Positional embedding
        if pos_emb:
            self.pos_emb = torch.nn.Parameter(
                torch.zeros(1, num_input_tokens, encoder_hidden_size)
            )
            self.pos_emb.data.normal_(mean=0.0, std=0.02)
        else:
            self.pos_emb = None

        # (Optional) Pre-normalization layer
        if prenorm:
            self.prenorm = LayerNorm(encoder_hidden_size)
        else:
            self.prenorm = None

        self.build_net(
            num_queries, encoder_hidden_size, hidden_size, output_hidden_size
        )
        self.dtype = next(self.parameters()).dtype

    def forward(
        self,
        x: torch.Tensor,
        num_queries_vis_abstractors: list[list[int]] | None = None,
        num_grids: list[int] | None = None,
    ) -> torch.Tensor:
        if self.prenorm is not None:
            x = self.prenorm(x)

        if self.pos_emb is not None:
            x = x + self.pos_emb

        x = self._forward(
            x,
            num_queries_vis_abstractors=num_queries_vis_abstractors,
            num_grids=num_grids,
        )  # (B, L, output_hidden_size)

        return x

    def _forward(
        self,
        x: torch.Tensor,
        num_queries_vis_abstractors: list[list[int]] | None = None,
        num_grids: list[int] | None = None,
    ) -> torch.Tensor:
        # x: [B, L, dim]
        B, L, dim = x.shape
        hw = int(L**0.5)
        x = rearrange(x, "b (h w) d -> b d h w", h=hw, w=hw)

        if num_queries_vis_abstractors is not None:
            assert num_grids is not None
            return self._forward_adaptive_num_query(
                x, num_queries_vis_abstractors, num_grids
            )

        x = self.net(x)
        x = rearrange(x, "b d h w -> b (h w) d")
        x = self.readout(x)
        return x

    def _forward_adaptive_num_query(
        self,
        x: torch.Tensor,
        num_queries_vis_abstractors: list[list[int]] | None = None,
        num_grids: list[int] | None = None,
    ) -> list[torch.Tensor]:
        # self.net is consisted by 3 layers (s1, sampler, s2)
        assert len(self.net) == 3

        x = self.net[0](x)  # s1
        new_x = []
        for i, num_queries in enumerate(num_queries_vis_abstractors):
            hw = int(num_queries**0.5)
            sampler = nn.AdaptiveAvgPool2d((hw, hw))
            out = sampler(x[num_grids[i] : num_grids[i + 1], :])
            out = self.net[2](out)  # s2

            out = rearrange(out, "b d h w -> b (h w) d")
            out = self.readout(out)

            new_x.append(out)
        return new_x

    def build_net(
        self,
        n_queries: int,
        encoder_hidden_size: int,
        hidden_size: int,
        output_hidden_size: int,
        depth: int = 3,
        mlp_depth: int = 2,
    ):
        assert (n_queries**0.5).is_integer(), (
            f"n_queries must be square number. n_queries: {n_queries}"
        )
        hw = int(n_queries**0.5)

        # RegBlock = ResBlock + SE
        RegBlock = partial(
            RegStage,
            stride=1,
            dilation=1,
            act_layer=nn.SiLU,
            norm_layer=LayerNorm2d,
        )

        s1 = RegBlock(
            depth,
            encoder_hidden_size,
            hidden_size,
        )
        sampler = nn.AdaptiveAvgPool2d((hw, hw))
        s2 = RegBlock(
            depth,
            hidden_size,
            hidden_size,
        )

        self.net = nn.Sequential(s1, sampler, s2)
        self.readout = self.build_mlp(mlp_depth, hidden_size, output_hidden_size)

    def build_mlp(
        self,
        depth: int,
        hidden_size: int,
        output_hidden_size: int,
    ):
        layers = [nn.Linear(hidden_size, output_hidden_size)]
        for _ in range(1, depth):
            layers.append(nn.SiLU())
            layers.append(nn.Linear(output_hidden_size, output_hidden_size))
        return nn.Sequential(*layers)


@MULTIMODAL_REGISTRY.register_processor(
    _build_hcxvision_hf_processor,
    info=_build_hcxvision_hf_info,
    dummy_inputs=HCXVisionDummyInputsBuilder,
)
class HCXVisionForCausalLM(nn.Module, SupportsMultiModal, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        # init configs
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        # text_config
        text_config = config.text_config
        if text_config.model_type in ["gpt2", "hyperclovax", "llama"]:
            text_config._attn_implementation = "sdpa"
        if text_config.model_type != "hyperclovax":
            text_config.logits_scaling = 1.0
        # vision_config
        vision_config = config.vision_config
        vision_config.auto_map = {}
        vision_config.anyres = config.anyres
        vision_config.max_num_grids = config.max_num_grids
        self.dtype = vllm_config.model_config.dtype

        ## possible_resolution should be matched with preprocessor_config.json
        config.possible_resolutions = self._init_possible_resolutions(
            config, vision_config
        )

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.vision_model = init_vision_tower_for_hcxvision(
                vision_config,
                quant_config=quant_config,
                multimodal_config=multimodal_config,
                use_nth_layer=getattr(config, "use_nth_layer", -1),
                require_post_norm=False,
                prefix=maybe_prefix(prefix, "vision_model"),
            )
            self.mm_projector = self._init_mm_projector(
                config, text_config, vision_config
            )

            if config.anyres:
                self.image_newline = nn.Parameter(
                    torch.empty(text_config.hidden_size, dtype=self.dtype)
                )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=text_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.config = config
        self.vision_config = vision_config
        self.text_config = text_config

        # use_sum_loss = bool(kwargs.pop("use_sum_loss", False))
        # self.reduction = self._init_reduction_type(use_sum_loss)

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return IMAGE_TOKEN
        if modality.startswith("video"):
            return VIDEO_TOKEN

        raise ValueError("Only image or video modality is supported")

    def _parse_and_validate_image_input(
        self,
        **kwargs: object,
    ) -> HCXVisionImageInputs | None:
        pixel_values_images = kwargs.pop("pixel_values_images", None)

        if pixel_values_images is None:
            return None

        image_sizes_images = kwargs.pop("image_sizes_images")

        return HCXVisionImagePixelInputs(
            pixel_values_images=pixel_values_images,
            image_sizes_images=image_sizes_images,
        )

    def _parse_and_validate_video_input(
        self,
        **kwargs: object,
    ) -> HCXVisionVideoInputs | None:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)

        if pixel_values_videos is None:
            return None

        return HCXVisionVideoPixelInputs(
            pixel_values_videos=pixel_values_videos,
        )

    def _process_image_input(
        self,
        image_input: HCXVisionImageInputs,
    ) -> tuple[torch.Tensor, ...]:
        return self.forward_images(
            pixel_values_images=image_input["pixel_values_images"],
            image_sizes_images=image_input["image_sizes_images"],
        )

    def _process_video_input(
        self,
        video_input: HCXVisionVideoInputs,
    ) -> tuple[torch.Tensor, ...]:
        return self.forward_videos(
            pixel_values_videos=video_input["pixel_values_videos"],
        )

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if input_key == "pixel_values_images" and "images" not in modalities:
                modalities["images"] = self._parse_and_validate_image_input(**kwargs)
            if input_key == "pixel_values_videos" and "videos" not in modalities:
                modalities["videos"] = self._parse_and_validate_video_input(**kwargs)

        return modalities

    def embed_multimodal(
        self,
        **kwargs: object,
    ) -> MultiModalEmbeddings:
        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not modalities:
            return []

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in modalities:
            if modality == "images":
                image_input = modalities["images"]
                image_embeddings = self._process_image_input(image_input)
                multimodal_embeddings += tuple(image_embeddings)
            if modality == "videos":
                video_input = modalities["videos"]
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

    def forward_images(
        self,
        pixel_values_images: list[torch.Tensor],
        image_sizes_images: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        pixel_values_image_flat = flatten_bn(pixel_values_images, concat=True)

        visual_token_idx = 0 if "siglip" in self.vision_config.model_type else 1
        image_forward_outs = self.vision_model(pixel_values_image_flat)[
            :, visual_token_idx:
        ]

        image_forward_outs = image_forward_outs.to(dtype=self.mm_projector.dtype)
        image_forward_outs = self.mm_projector(image_forward_outs)  # b (h w) d

        split_sizes = [len(item) for item in pixel_values_images]
        image_forward_outs = torch.split(image_forward_outs, split_sizes, dim=0)

        # newline for anyres postprocessing
        image_features = anyres_postprocessing(
            image_forward_outs=image_forward_outs,
            image_sizes=image_sizes_images.tolist(),
            num_queries_vis_abstractor=self.config.num_queries_vis_abstractor_image,
            unpad=self.config.unpad,
            patch_size=self.vision_config.patch_size,
            grid_size=self.vision_config.image_size,
            image_newline=self.image_newline,
            possible_resolutions=self.config.possible_resolutions,
        )

        return tuple(image_features)

    def forward_videos(
        self,
        pixel_values_videos: list[list[torch.Tensor]],
    ) -> tuple[torch.Tensor, ...]:
        pixel_values_videos_flat = flatten_bn(
            [frame for frames in pixel_values_videos for frame in frames],
            concat=True,
        )

        visual_token_idx = 0 if "siglip" in self.vision_config.model_type else 1
        video_forward_outs = self.vision_model(pixel_values_videos_flat)[
            :, visual_token_idx:
        ]

        video_forward_outs = video_forward_outs.to(dtype=self.mm_projector.dtype)

        # Run MM-Projector
        # len(num_grids) == len(num_queries_vis_abstractors) + 1
        grid_idx = 0
        # e.g. [0, 9, 18, 19, 27, 28, 36, 37, 45, 46, 54, 55, 56]
        num_grids = [grid_idx]
        # e.g. [81, 81, 81, 9, 81, 9, 81, 9, 81, 9, 81, 9]
        num_queries_vis_abstractors = []
        len_total_frames = video_forward_outs.shape[0]

        if self.config.first_last_frames_slow:
            # slowfast (first_last_frames_slow)
            assert len_total_frames != 0
            if len_total_frames <= 2:
                num_queries_vis_abstractors.append(
                    self.config.num_queries_vis_abstractor_video_slow
                )
                grid_idx += len_total_frames
                num_grids.append(grid_idx)
            else:
                num_queries_vis_abstractors.append(
                    self.config.num_queries_vis_abstractor_video_slow
                )
                grid_idx += 1
                num_grids.append(grid_idx)

                num_queries_vis_abstractors.append(
                    self.config.num_queries_vis_abstractor_video_fast
                )
                grid_idx += len_total_frames - 2
                num_grids.append(grid_idx)

                num_queries_vis_abstractors.append(
                    self.config.num_queries_vis_abstractor_video_slow
                )
                grid_idx += 1
                num_grids.append(grid_idx)
        else:
            # slowfast
            for pixel_values_frames in pixel_values_videos:
                for pixel_values_frame in pixel_values_frames:
                    if len(pixel_values_frame) > 0:
                        num_queries_vis_abstractors.append(
                            self.config.num_queries_vis_abstractor_video_slow
                        )
                        grid_idx += 1
                        num_grids.append(grid_idx)
                        num_queries_vis_abstractors.append(
                            self.config.num_queries_vis_abstractor_video_fast
                        )
                        grid_idx = grid_idx + len(pixel_values_frame) - 1
                        num_grids.append(grid_idx)

        video_forward_outs = self.mm_projector(
            video_forward_outs, num_queries_vis_abstractors, num_grids
        )

        video_features = []  # what we want to return
        target_features = []
        target_group_size = 0
        group_counter = 0
        video_groups = [
            len(frame) for frames in pixel_values_videos for frame in frames
        ]  # for concat video features after projector

        for forward_out in video_forward_outs:
            target_group_size += len(forward_out)
            target_features.append(forward_out.flatten(0, 1))

            video_group_size = video_groups[group_counter]
            if video_group_size == target_group_size:
                video_features.append(torch.cat(target_features, dim=0))
                target_features = []
                group_counter += 1
                target_group_size = 0

            elif video_group_size < target_group_size:
                raise RuntimeError(f"{video_group_size=} < {target_group_size=}")

        assert len(target_features) == 0, (
            f"target_features is not empty!! {target_features}"
        )
        assert len(video_groups) == len(video_features)

        feats_per_video = [len(video) for video in pixel_values_videos]
        idxs_per_video = [0, *accumulate(feats_per_video)]
        return tuple(
            torch.cat(video_features[idxs_per_video[i] : idxs_per_video[i + 1]])
            for i in range(len(feats_per_video))
        )

    def _prepare_multimodal_kwargs(self, **kwargs: object):
        output = defaultdict(list)
        for k, v in kwargs.items():
            if len(v) < 1 or len(v[0]) < 1:
                continue  # if empty batch of empty sample

            new_k, is_video = k, False
            if not k.endswith("_images") and not k.endswith("_videos"):
                pass
            else:
                new_k, is_video = k.split("_")[:-1], k.split("_")[-1]
                new_k = "_".join(new_k)
                is_video = is_video == "videos"

            for _sample_idx, _v in enumerate(v):  # batch -> sample
                if new_k not in ["pixel_values"]:
                    if len(output[new_k]) < _sample_idx + 1:
                        output[new_k].append(list())
                    _v = _v.detach().cpu().numpy().tolist()
                    output[new_k][_sample_idx] += _v
                elif isinstance(_v, torch.Tensor):
                    if len(output[new_k]) < _sample_idx + 1:
                        output[new_k].append(list())
                        output["is_videos"].append(list())
                    _v = list(torch.unbind(_v, dim=0))
                    output[new_k][_sample_idx] += _v
                    output["is_videos"][_sample_idx] += [
                        is_video,
                    ] * len(_v)
        return dict(output)

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
        return loader.load_weights(weights)

    def _init_possible_resolutions(
        self,
        config,
        vision_config,
    ):
        if not getattr(config, "possible_resolutions", []):
            possible_resolutions = []
            if config.anyres:
                assert config.max_num_grids > 0
                for i in range(1, config.max_num_grids + 1):
                    for j in range(1, config.max_num_grids + 1):
                        if i == 1 and j == 1 and not config.use_1x1_grid:
                            continue
                        if i * j <= config.max_num_grids:
                            possible_resolutions.append([i, j])

                possible_resolutions = [
                    [ys * vision_config.image_size, xs * vision_config.image_size]
                    for ys, xs in possible_resolutions
                ]
            return possible_resolutions
        else:
            return config.possible_resolutions

    def _init_mm_projector(
        self,
        config,
        text_config,
        vision_config,
    ):
        input_hidden_size = vision_config.hidden_size
        if config.mm_projector_type == "linear":
            mm_projector = nn.Linear(input_hidden_size, text_config.hidden_size)
            mm_projector.dtype = next(mm_projector.parameters()).dtype
        elif config.mm_projector_type == "cabstractor":
            mm_projector = HCXVisionCAbstractor(
                num_queries=config.num_queries_vis_abstractor_image,
                num_input_tokens=(vision_config.image_size // vision_config.patch_size)
                ** 2,
                encoder_hidden_size=input_hidden_size,
                hidden_size=input_hidden_size,
                output_hidden_size=text_config.hidden_size,
                pos_emb=config.proj_pos_emb,
                prenorm=config.proj_prenorm,
            )
        else:
            mm_projector = HCXVisionMlp(
                config.mm_projector_type,
                input_hidden_size,
                hidden_features=input_hidden_size,
                out_features=self.text_config.hidden_size,
            )
        return mm_projector


def unpad_image(tensor: torch.Tensor, original_size: tuple[int, int]) -> torch.Tensor:
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


def select_best_resolution(original_size: tuple, possible_resolutions: list) -> tuple:
    original_height, original_width = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for height, width in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = (
            int(original_width * scale),
            int(original_height * scale),
        )
        effective_resolution = min(
            downscaled_width * downscaled_height, original_width * original_height
        )
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution
            and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (height, width)

    return best_fit


def get_anyres_image_grid_shape(
    image_size: tuple[int, int],
    grid_pinpoints: str | list[tuple[int, int]],
    patch_size: int,
) -> tuple[int, int]:
    possible_resolutions = (
        grid_pinpoints
        if isinstance(grid_pinpoints, list)
        else ast.literal_eval(grid_pinpoints)
    )

    original_width, original_height = image_size
    height, width = select_best_resolution(
        (original_height, original_width), possible_resolutions
    )
    return width // patch_size, height // patch_size


def reshape_and_unpad_image_features(
    image_feature: torch.Tensor,
    height: int,
    width: int,
    image_size: tuple[int, int],
    possible_resolutions: list[tuple[int, int]],
    grid_size: int,
    unpad: bool,
    image_newline: torch.Tensor,
) -> torch.Tensor:
    base_image_feature = image_feature[0]
    image_feature = image_feature[1:]

    assert height * width == base_image_feature.shape[0], (
        f"{height=} * {width=} != {base_image_feature.shape[0]=}"
    )

    num_patch_width, num_patch_height = get_anyres_image_grid_shape(
        image_size, possible_resolutions, grid_size
    )
    image_feature = image_feature.view(
        num_patch_height, num_patch_width, height, width, -1
    )

    if unpad:
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        image_feature = unpad_image(image_feature, image_size)
        image_feature = torch.cat(
            (
                image_feature,
                image_newline[:, None, None]
                .expand(*image_feature.shape[:-1], 1)
                .to(image_feature.device),
            ),
            dim=-1,
        )
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
    else:
        image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
        image_feature = image_feature.flatten(0, 3)
    image_feature = torch.cat((base_image_feature, image_feature), dim=0)

    return image_feature


def anyres_postprocessing(
    image_forward_outs: list[torch.Tensor],
    image_sizes: list[list[int]],
    possible_resolutions: list[tuple[int, int]],
    patch_size: int,
    grid_size: int,
    image_newline: torch.Tensor,
    num_queries_vis_abstractor: int = -1,
    unpad: bool = False,
) -> list[torch.Tensor]:
    height = width = grid_size // patch_size

    if num_queries_vis_abstractor > 0:
        assert (num_queries_vis_abstractor**0.5).is_integer(), (
            "n_queries must be square number"
        )
        height = width = int(num_queries_vis_abstractor**0.5)

    # post-processing (unpad, add newline)
    new_image_features = []
    for image_idx, image_feature in enumerate(image_forward_outs):
        if image_feature.shape[0] > 1:
            image_feature = reshape_and_unpad_image_features(
                image_feature=image_feature,
                height=height,
                width=width,
                image_size=image_sizes[image_idx],
                possible_resolutions=possible_resolutions,
                grid_size=grid_size,  # Pass grid info if needed by helper
                unpad=unpad,
                image_newline=image_newline,
            )
        else:
            image_feature = image_feature[0]
            image_feature = torch.cat(
                (image_feature, image_newline[None].to(image_feature.device)), dim=0
            )
        new_image_features.append(image_feature)

    return new_image_features
