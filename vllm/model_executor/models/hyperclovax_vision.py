# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# copied from : https://github.com/huggingface/transformers
import ast
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from itertools import chain
from typing import Any, Literal, Optional, TypedDict, Union

import numpy as np
import PIL
from einops import rearrange
from PIL import Image

if sys.version_info >= (3, 11):
    import typing
    Unpack = typing.Unpack
else:
    import typing_extensions
    Unpack = typing_extensions.Unpack

import torch
import torch.nn as nn
from timm.layers import LayerNorm, LayerNorm2d
from timm.models.regnet import RegStage
from transformers import BatchFeature, CLIPVisionConfig, SiglipVisionConfig
from transformers.modeling_utils import no_init_weights

from vllm.config import VllmConfig
from vllm.inputs import InputProcessingContext
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs)
from vllm.multimodal.parse import ImageSize, MultiModalDataItems
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, ProcessingCache,
                                        PromptReplacement, PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors

from .clip import CLIPVisionModel
from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .siglip import SiglipVisionModel
from .utils import AutoWeightsLoader, init_vllm_registered_model, maybe_prefix
from .vision import get_vision_encoder_info

EOT = "<|endofturn|>"
IMAGE_TOKEN: str = "<|dummy3|>"
VIDEO_TOKEN: str = "<|_unuse_missing_100270|>"


class HCXVisionMultimodalPixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values_images: list[torch.Tensor]
    """
    Shape: `[(num_grids, num_channels, height, width), ...]` if anyres
    
    Note that `height` or `width` may be different per batch and image,
    in which case the data is passed as a list instead of a batched tensor.
    """
    image_sizes_images: list[tuple[Union[int, float]]]
    """
    Shape: `[(height, width), ...]`
    """
    vision_query_lengths_images: list[Union[int, float]]
    pixel_values_videos: list[tuple[Union[int, float]]]
    """
    Shape: `[(num_grids, num_channels, height, width), ...]` if anyres
    """
    vision_query_lengths_videos: list[Union[int, float]]


HCXVisionMultimodalInputs = Union[HCXVisionMultimodalPixelInputs]


class HCXVisionProcessingInfo(BaseProcessingInfo):

    def get_vision_encoder_info(self):
        return get_vision_encoder_info(self.get_hf_config())

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None, "video": None}

    def get_num_image_tokens(
        self,
        *,
        vision_query_length: Union[int, list[int]],
    ) -> int:
        if isinstance(vision_query_length, int):
            return vision_query_length
        else:
            return sum(vision_query_length)

    def get_num_video_tokens(
        self,
        *,
        vision_query_length: Union[int, list[int]],
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


class HCXVisionDummyInputsBuilder(
        BaseDummyInputsBuilder[HCXVisionProcessingInfo]):

    def get_dummy_text(
        self,
        mm_counts: Mapping[str, int],
    ) -> str:
        dummy_text = IMAGE_TOKEN * mm_counts.get(
            "image", 0) + VIDEO_TOKEN * mm_counts.get("video", 0)
        return dummy_text

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        target_width, target_height = \
            self.info.get_image_size_with_most_features()
        target_num_frames = 32
        return {
            "image":
            self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
            ),
            "video":
            self._get_dummy_videos(
                width=target_width - 1,
                height=target_height - 1,
                num_frames=target_num_frames,
                num_videos=num_videos,
            )
        }


class HCXVisionMultiModalProcessor(
        BaseMultiModalProcessor[HCXVisionProcessingInfo]):

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:

        def replace_multimodal_token(
            token_ids: torch.Tensor,
            target_token: int,
            repeats: list,
        ):
            output = list()
            _repeats_idx = 0
            for token_id in token_ids:
                if token_id == target_token:
                    output += [
                        token_id.item(),
                    ] * repeats[_repeats_idx]
                    _repeats_idx += 1
                else:
                    output += [
                        token_id.item(),
                    ]
            return torch.tensor(output, device=token_ids.device)

        for video_idx, video_arr in enumerate(mm_data.get("videos", list())):
            if video_arr.dtype == np.uint8:
                continue
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
            # batchify input as a single item
            images = mm_data.get("images", None)
            num_images = 0
            if images is not None:
                num_images = len(images)
                images = [
                    images,
                ]  # batchify

            videos = mm_data.get("videos",
                                 None)  # list of video in single conversation
            num_videos = 0
            if videos is not None:
                num_videos = len(videos)
                videos = [
                    videos,
                ]  # batchify

            _processed_outputs = self.info.ctx.call_hf_processor(
                hf_processor=self.info.get_hf_processor(**mm_kwargs),
                data=dict(
                    text=None,
                    images=images,
                    videos=videos,
                ),
            )  # mm-only

            for k, v in _processed_outputs.items():
                if len(v) < 1:
                    continue
                elif k.endswith("_images"):
                    # list of list of 4D tensor -> list of 4D tensor
                    _processed_outputs[k] = v[0]
                elif k.endswith("_videos"):
                    # list of list of 4D tensor -> list of 4D tensor
                    v = v[0]
                    if k == "pixel_values_videos":
                        v = torch.cat(v, dim=0)
                        _c, _w, _h = v.shape[-3:]
                        v = v.reshape(num_videos, -1, _c, _w, _h)
                        v = list(torch.unbind(v, dim=0))
                    _processed_outputs[k] = v

            if num_images > 0:
                tokenizer = self.info.get_tokenizer()
                processed_outputs["input_ids"] = torch.stack([
                    replace_multimodal_token(
                        token_ids=_input_ids,
                        target_token=tokenizer.convert_tokens_to_ids(
                            IMAGE_TOKEN),
                        repeats=_processed_outputs[
                            "vision_query_lengths_images"],
                    ) for _input_ids in processed_outputs["input_ids"]
                ],
                                                             dim=0)

            if num_videos > 0:
                tokenizer = self.info.get_tokenizer()
                processed_outputs["input_ids"] = torch.stack([
                    replace_multimodal_token(
                        token_ids=_input_ids,
                        target_token=tokenizer.convert_tokens_to_ids(
                            VIDEO_TOKEN),
                        repeats=_processed_outputs[
                            "vision_query_lengths_videos"],
                    ) for _input_ids in processed_outputs["input_ids"]
                ],
                                                             dim=0)

                _ratios = [
                    len(_pixel_values) for _pixel_values in
                    _processed_outputs["pixel_values_videos"]
                ]
                _num_per_videos = [
                    int(_e / sum(_ratios) *
                        len(_processed_outputs["vision_query_lengths_videos"]))
                    for _e in _ratios
                ]
                _processed_outputs["vision_query_lengths_videos"] = [
                    _processed_outputs["vision_query_lengths_videos"]
                    [sum(_num_per_videos[:_i]):sum(_num_per_videos[:_i + 1])]
                    for _i in range(0, num_videos)
                ]

            processed_outputs.update(_processed_outputs)

        return processed_outputs

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()
        placeholder = {
            "image": hf_config.image_token_id,
            "video": hf_config.video_token_id,
        }

        def get_replacement_hyperclovax(
            item_idx: int,
            modality: str,
            out_mm_kwargs: MultiModalKwargs,
        ):
            num_tokens = None
            if modality == "image":
                num_tokens = self.info.get_num_image_tokens(
                    vision_query_length=out_mm_kwargs[
                        "vision_query_lengths_images"][item_idx], )
            if modality == "video":
                num_tokens = self.info.get_num_video_tokens(
                    vision_query_length=out_mm_kwargs[
                        "vision_query_lengths_videos"][item_idx], )
            assert isinstance(num_tokens, int)
            return [
                placeholder[modality],
            ] * num_tokens

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
            ) for modality in ("image", "video")
        ]

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            # image
            pixel_values_images=MultiModalFieldConfig.batched("image"),
            image_sizes_images=MultiModalFieldConfig.batched("image"),
            vision_query_lengths_images=MultiModalFieldConfig.batched("image"),
            num_queries_vis_abstractors_images=MultiModalFieldConfig.batched(
                "image"),
            num_queries_vis_abstractors_slow_images=MultiModalFieldConfig.
            batched("image"),
            first_last_frames_slows_images=MultiModalFieldConfig.batched(
                "image"),
            # video
            pixel_values_videos=MultiModalFieldConfig.batched("video"),
            image_sizes_videos=MultiModalFieldConfig.batched("video"),
            vision_query_lengths_videos=MultiModalFieldConfig.batched("video"),
            num_queries_vis_abstractors_videos=MultiModalFieldConfig.batched(
                "video"),
            num_queries_vis_abstractors_slow_videos=MultiModalFieldConfig.
            batched("video"),
            first_last_frames_slows_videos=MultiModalFieldConfig.batched(
                "video"),
        )


def _build_hcxvision_hf_info(
    ctx: InputProcessingContext, ) -> HCXVisionProcessingInfo:
    return HCXVisionProcessingInfo(ctx)


def _build_hcxvision_hf_processor(
    info: HCXVisionProcessingInfo,
    dummy_inputs: BaseDummyInputsBuilder[HCXVisionProcessingInfo],
    *,
    cache: Optional[ProcessingCache] = None,
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
    quant_config: Optional[QuantizationConfig],
    *,
    use_nth_layer: Optional[int] = None,
    require_post_norm: Optional[bool] = None,
    prefix: str = "",
) -> Union[CLIPVisionModel, SiglipVisionModel]:
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
            num_hidden_layers_override=num_hidden_layers,
            require_post_norm=require_post_norm,
            prefix=prefix,
        )
    elif isinstance(vision_config, SiglipVisionConfig):
        return SiglipVisionModel(
            vision_config,
            quant_config=quant_config,
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
            raise NotImplementedError("{} is not implemented".format(
                self.mm_projector_type))

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
                torch.zeros(1, num_input_tokens, encoder_hidden_size))
            self.pos_emb.data.normal_(mean=0.0, std=0.02)
        else:
            self.pos_emb = None

        # (Optional) Pre-normalization layer
        if prenorm:
            self.prenorm = LayerNorm(encoder_hidden_size)
        else:
            self.prenorm = None

        self.build_net(num_queries, encoder_hidden_size, hidden_size,
                       output_hidden_size)
        self.dtype = next(self.parameters()).dtype

    def forward(
        self,
        x: torch.Tensor,
        num_queries_vis_abstractors: Optional[list[list[int]]] = None,
        num_grids: Optional[list[int]] = None,
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
        num_queries_vis_abstractors: Optional[list[list[int]]] = None,
        num_grids: Optional[list[int]] = None,
    ) -> torch.Tensor:
        # x: [B, L, dim]
        B, L, dim = x.shape
        hw = int(L**0.5)
        x = rearrange(x, "b (h w) d -> b d h w", h=hw, w=hw)

        if num_queries_vis_abstractors is not None:
            assert num_grids is not None
            return self._forward_adaptive_num_query(
                x, num_queries_vis_abstractors, num_grids)

        x = self.net(x)
        x = rearrange(x, "b d h w -> b (h w) d")
        x = self.readout(x)
        return x

    def _forward_adaptive_num_query(
        self,
        x: torch.Tensor,
        num_queries_vis_abstractors: Optional[list[list[int]]] = None,
        num_grids: Optional[list[int]] = None,
    ) -> list[torch.Tensor]:
        # self.net is consisted by 3 layers (s1, sampler, s2)
        assert len(self.net) == 3

        x = self.net[0](x)  # s1
        new_x = []
        for i, num_queries in enumerate(num_queries_vis_abstractors):
            hw = int(num_queries**0.5)
            sampler = nn.AdaptiveAvgPool2d((hw, hw))
            out = sampler(x[num_grids[i]:num_grids[i + 1], :])
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
        assert (n_queries**0.5).is_integer(
        ), f"n_queries must be square number. n_queries: {n_queries}"
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
        self.readout = self.build_mlp(mlp_depth, hidden_size,
                                      output_hidden_size)

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
    dummy_inputs=HCXVisionDummyInputsBuilder)
class HCXVisionForCausalLM(nn.Module, SupportsMultiModal, SupportsPP):

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"]
    }

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        **kwargs: Optional[Any],
    ) -> None:
        super().__init__()

        # init configs
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
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
            config, vision_config)

        # init models & parameters
        with no_init_weights():  # weight will be loaded in from_pretrained
            self.vision_model = init_vision_tower_for_hcxvision(
                vision_config,
                quant_config,
                use_nth_layer=getattr(config, "use_nth_layer", -1),
                require_post_norm=False,
                prefix=maybe_prefix(prefix, "vision_model"),
            )
        self.mm_projector = self._init_mm_projector(config, text_config,
                                                    vision_config)

        self.lm_head_vocab_size = getattr(text_config, "padded_vocab_size",
                                          text_config.vocab_size)
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        if config.anyres:
            self.image_newline = nn.Parameter(
                torch.empty(text_config.hidden_size, dtype=self.dtype))

        self.config = config
        self.vision_config = vision_config
        self.text_config = text_config

        # use_sum_loss = bool(kwargs.pop("use_sum_loss", False))
        # self.reduction = self._init_reduction_type(use_sum_loss)

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("image"):
            return IMAGE_TOKEN
        if modality.startswith("video"):
            return VIDEO_TOKEN

        raise ValueError("Only image or video modality is supported")

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(
        self,
        **kwargs: Unpack[HCXVisionMultimodalInputs],
    ) -> Optional[MultiModalEmbeddings]:

        multimodal_embeddings = list()
        if kwargs.get("pixel_values_images") is not None:
            for _pixel_values_images, _image_sizes_images in zip(
                    kwargs["pixel_values_images"],
                    kwargs["image_sizes_images"]):
                _pixel_values_images = _pixel_values_images.unsqueeze(dim=0)
                _image_sizes_images = _image_sizes_images.unsqueeze(dim=0)
                _len_pixel_values_images = [
                    len(pixel_value) for pixel_value in _pixel_values_images
                ]
                if isinstance(_image_sizes_images, torch.Tensor):
                    _image_sizes_images = _image_sizes_images.detach().cpu(
                    ).tolist()
                _multimodal_embeddings_images = self.forward_images(
                    pixel_values_images=_pixel_values_images,
                    image_sizes_images=_image_sizes_images,
                    len_pixel_values_images=_len_pixel_values_images,
                )
                _multimodal_embeddings_images = torch.cat(
                    _multimodal_embeddings_images, dim=0)
                multimodal_embeddings.append(_multimodal_embeddings_images)

        if kwargs.get("pixel_values_videos") is not None:
            for _pixel_values_videos, _vision_query_lengths_videos in zip(
                    kwargs["pixel_values_videos"],
                    kwargs["vision_query_lengths_videos"]):
                _len_pixel_values_videos = [
                    len(_vision_query_lengths)
                    for _vision_query_lengths in _vision_query_lengths_videos
                ]
                _c, _w, _h = _pixel_values_videos.shape[-3:]
                _pixel_values_videos = _pixel_values_videos.reshape(
                    sum(_len_pixel_values_videos), -1, _c, _w,
                    _h).unsqueeze(dim=0)
                _multimodal_embeddings_videos = self.forward_videos(
                    pixel_values_videos=_pixel_values_videos,
                    len_pixel_values_videos=_len_pixel_values_videos,
                )
                _multimodal_embeddings_videos = torch.cat(
                    _multimodal_embeddings_videos, dim=0)
                multimodal_embeddings.append(_multimodal_embeddings_videos)
        return multimodal_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
        **kwargs,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if (kwargs.get("pixel_values_images") is not None
                or kwargs.get("pixel_values_videos")
                is not None):  # v0 compatibility
            multimodal_embeddings = self.get_multimodal_embeddings(**kwargs)
        if multimodal_embeddings is not None:
            multimodal_embeddings = torch.cat(multimodal_embeddings, dim=0)
            _mask_image = input_ids == self.config.image_token_id
            _mask_video = input_ids == self.config.video_token_id
            assert _mask_image.sum() + _mask_video.sum() == len(
                multimodal_embeddings)

            if multimodal_embeddings.dtype != inputs_embeds.dtype:
                multimodal_embeddings = multimodal_embeddings.to(
                    dtype=inputs_embeds.dtype)
            if multimodal_embeddings.device != inputs_embeds.device:
                multimodal_embeddings = multimodal_embeddings.to(
                    device=inputs_embeds.device)

            if _mask_image.sum() > 0:
                inputs_embeds[
                    _mask_image] = multimodal_embeddings[:sum(_mask_image)]
            if _mask_video.sum() > 0:
                inputs_embeds[_mask_video] = multimodal_embeddings[
                    -sum(_mask_video):]
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids=input_ids,
                                                      **kwargs)
            input_ids = None
        hidden_states = self.language_model.model(input_ids,
                                                  positions,
                                                  intermediate_tensors,
                                                  inputs_embeds=inputs_embeds)
        return hidden_states

    def forward_images(
        self,
        pixel_values_images: list[list[torch.FloatTensor]],
        image_sizes_images: list[list[tuple[int, int]]],
        len_pixel_values_images: list[int],
    ) -> list[list[torch.Tensor]]:
        if sum(len_pixel_values_images) == 0:
            return None

        concat_pixel_values_images = torch.cat(list(
            chain(*pixel_values_images)),
                                               dim=0)

        visual_token_idx = 0 if "siglip" in self.vision_config.model_type else 1
        image_forward_outs = self.vision_model(
            concat_pixel_values_images)[:, visual_token_idx:]

        image_forward_outs = image_forward_outs.to(
            dtype=self.mm_projector.dtype)
        image_forward_outs = self.mm_projector(image_forward_outs)  # b (h w) d

        split_sizes = [
            pixel_value.shape[0] for pixel_value in chain(*pixel_values_images)
        ]
        image_forward_outs = torch.split(image_forward_outs,
                                         split_sizes,
                                         dim=0)

        # newline for anyres postprocessing
        image_features = anyres_postprocessing(
            image_forward_outs=image_forward_outs,
            image_sizes=[
                image_size for image_sizes in image_sizes_images
                for image_size in image_sizes
            ],
            num_queries_vis_abstractor=self.config.
            num_queries_vis_abstractor_image,
            unpad=self.config.unpad,
            patch_size=self.vision_config.patch_size,
            grid_size=self.vision_config.image_size,
            image_newline=self.image_newline,
            possible_resolutions=self.config.possible_resolutions,
        )
        return image_features

    def forward_videos(
        self,
        pixel_values_videos: list[list[torch.FloatTensor]],
        len_pixel_values_videos: list[int],
    ) -> list[torch.Tensor]:

        len_video_grids = sum(len_pixel_values_videos)
        if len_video_grids == 0:
            return None

        # Run Vision Model
        concat_pixel_values_videos = torch.cat(list(
            chain(*pixel_values_videos)),
                                               dim=0)

        visual_token_idx = 0 if "siglip" in self.vision_config.model_type else 1
        video_forward_outs = self.vision_model(
            concat_pixel_values_videos)[:, visual_token_idx:]

        video_forward_outs = video_forward_outs.to(
            dtype=self.mm_projector.dtype)

        # Run MM-Projector
        # len(num_grids) == len(num_queries_vis_abstractors) + 1
        grid_idx = 0
        num_grids = [
            grid_idx
        ]  # e.g. [0, 9, 18, 19, 27, 28, 36, 37, 45, 46, 54, 55, 56]
        num_queries_vis_abstractors = [
        ]  # e.g. [81, 81, 81, 9, 81, 9, 81, 9, 81, 9, 81, 9]
        len_total_frames = video_forward_outs.shape[0]

        if self.config.first_last_frames_slow:
            # slowfast (first_last_frames_slow)
            assert len_total_frames != 0
            if len_total_frames <= 2:
                num_queries_vis_abstractors.append(
                    self.config.num_queries_vis_abstractor_video_slow)
                grid_idx += len_total_frames
                num_grids.append(grid_idx)
            else:
                num_queries_vis_abstractors.append(
                    self.config.num_queries_vis_abstractor_video_slow)
                grid_idx += 1
                num_grids.append(grid_idx)

                num_queries_vis_abstractors.append(
                    self.config.num_queries_vis_abstractor_video_fast)
                grid_idx += len_total_frames - 2
                num_grids.append(grid_idx)

                num_queries_vis_abstractors.append(
                    self.config.num_queries_vis_abstractor_video_slow)
                grid_idx += 1
                num_grids.append(grid_idx)
        else:
            # slowfast
            for pixel_values_frames in pixel_values_videos:
                for pixel_values_frame in pixel_values_frames:
                    if len(pixel_values_frame) > 0:
                        num_queries_vis_abstractors.append(
                            self.config.num_queries_vis_abstractor_video_slow)
                        grid_idx += 1
                        num_grids.append(grid_idx)
                        num_queries_vis_abstractors.append(
                            self.config.num_queries_vis_abstractor_video_fast)
                        grid_idx = grid_idx + len(pixel_values_frame) - 1
                        num_grids.append(grid_idx)

        video_forward_outs = self.mm_projector(video_forward_outs,
                                               num_queries_vis_abstractors,
                                               num_grids)

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
                raise RuntimeError(f"video_group_size < target_group_size!! \
                        [{video_group_size} < {target_group_size}]")

        assert len(target_features
                   ) == 0, f"target_features is not empty!! {target_features}"
        assert len(video_groups) == len(video_features)

        return video_features

    def _prepare_multimodal_kwargs(self, **kwargs: object):
        output = defaultdict(list)
        for k, v in kwargs.items():
            if len(v) < 1 or len(v[0]) < 1:
                continue  # if empty batch of empty sample

            new_k, is_video = k, False
            if (not k.endswith("_images") and not k.endswith("_videos")):
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
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

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

                possible_resolutions = [[
                    ys * vision_config.image_size,
                    xs * vision_config.image_size
                ] for ys, xs in possible_resolutions]
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
            mm_projector = nn.Linear(input_hidden_size,
                                     text_config.hidden_size)
            mm_projector.dtype = next(mm_projector.parameters()).dtype
        elif config.mm_projector_type == "cabstractor":
            mm_projector = HCXVisionCAbstractor(
                num_queries=config.num_queries_vis_abstractor_image,
                num_input_tokens=(vision_config.image_size //
                                  vision_config.patch_size)**2,
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


def unpad_image(tensor: torch.Tensor,
                original_size: tuple[int, int]) -> torch.Tensor:
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


def select_best_resolution(original_size: tuple,
                           possible_resolutions: list) -> tuple:
    original_height, original_width = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for height, width in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(
            original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height,
                                   original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
                effective_resolution == max_effective_resolution
                and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (height, width)

    return best_fit


def get_anyres_image_grid_shape(
    image_size: tuple[int, int],
    grid_pinpoints: Union[str, list[tuple[int, int]]],
    patch_size: int,
) -> tuple[int, int]:
    possible_resolutions = grid_pinpoints if isinstance(
        grid_pinpoints, list) else ast.literal_eval(grid_pinpoints)

    original_width, original_height = image_size
    height, width = select_best_resolution((original_height, original_width),
                                           possible_resolutions)
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

    assert (height * width == base_image_feature.shape[0]
            ), f"height: {height}, width: {width}, \
        base_image_feature.shape[0]: {base_image_feature.shape[0]}"

    num_patch_width, num_patch_height = get_anyres_image_grid_shape(
        image_size, possible_resolutions, grid_size)
    image_feature = image_feature.view(num_patch_height, num_patch_width,
                                       height, width, -1)

    if unpad:
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        image_feature = unpad_image(image_feature, image_size)
        image_feature = torch.cat(
            (
                image_feature,
                image_newline[:, None, None].expand(
                    *image_feature.shape[:-1], 1).to(image_feature.device),
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
    image_forward_outs: list[torch.FloatTensor],
    image_sizes: list[list[int]],
    possible_resolutions: list[tuple[int, int]],
    patch_size: int,
    grid_size: int,
    image_newline: torch.FloatTensor,
    num_queries_vis_abstractor: int = -1,
    unpad: bool = False,
) -> list[torch.FloatTensor]:
    height = width = grid_size // patch_size

    if num_queries_vis_abstractor > 0:
        assert (num_queries_vis_abstractor**0.5
                ).is_integer(), "n_queries must be square number"
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
                (image_feature, image_newline[None].to(image_feature.device)),
                dim=0)
        new_image_features.append(image_feature)
    image_features = new_image_features
    return image_features


def resize_image(
    image: Union[np.ndarray, PIL.Image.Image],
    max_side: int = 378,
) -> np.ndarray:
    image_arr = image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    width, height = image.size
    cur_max_size = max(width, height)
    if cur_max_size <= max_side:
        return image_arr

    scale = max_side / cur_max_size
    width = int(width * scale)
    height = int(height * scale)
    image = image.resize((width, height), Image.LANCZOS)
    image_arr = np.array(image)
    return image_arr
