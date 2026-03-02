# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Annotated, Any, Literal, TypeAlias

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from transformers import PretrainedConfig
from transformers.activations import GELUActivation
from transformers.feature_extraction_utils import BatchFeature

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    ImageItem,
    ModalityData,
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    VideoItem,
)
from vllm.multimodal.parse import (
    DictEmbeddingItems,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import SupportsLoRA, SupportsMRoPE, SupportsMultiModal, SupportsPP
from .keye import (
    BaseKeyeModule,
    BaseMultiModalProcessor,
    KeyeBaseDummyInputsBuilder,
    KeyeProcessingInfo,
)

logger = init_logger(__name__)


def split_thw(grid_thw: torch.Tensor) -> torch.Tensor:
    """
    Split grid_thw in t dimension.

    Args:
        grid_thw: [N, 3] tensor of [t, h, w]

    Returns:
        [Σt, 3] tensor where each row is [1, h, w]

    Example:
    >>> grid_thw = torch.tensor([[2, 3, 4], [1, 5, 6]])
    >>> split_thw(grid_thw)
    tensor([[1, 3, 4],
           [1, 3, 4],
           [1, 5, 6]])
    """
    t = grid_thw[:, 0]
    h_w = grid_thw[:, 1:]
    ones = torch.ones_like(h_w[:, :1])
    return torch.cat([ones, h_w], dim=1).repeat_interleave(t, dim=0)


def get_num_patches(
    grid_thw: torch.Tensor, num_frames: list[int] | torch.Tensor
) -> list[int]:
    """
    Return num_patches per video.

    Args:
        grid_thw: Tensor with shape [N, 3] containing temporal, height, width
            dimensions
        num_frames: List or tensor indicating the number of frames per video

    Returns:
        List of ints representing the number of patches for each video

    Examples:
        >>> # Suppose there are 2 videos with a total of 3 grids
        >>> grid_thw = torch.tensor(
        ...     [
        ...         [2, 2, 2],  # grid 0: 2*2*2=8 patches
        ...         [2, 2, 2],  # grid 1: 2*2*2=8 patches
        ...         [1, 1, 1],
        ...     ]
        ... )  # grid 2: 1*1*1=1 patches
        >>> num_frames = [2, 1]  # The first video contains 2 grids,
                                   the second contains 1 grid.
        >>> get_num_patches(grid_thw, num_frames)
        tensor([16, 1])  # Total patches for first video: 8+8=16,
                           second video: 1.
    """

    assert len(grid_thw.shape) == 2
    if isinstance(num_frames, torch.Tensor):
        num_frames = num_frames.clone().tolist()

    num_grids_per_frame = grid_thw.prod(dim=1)
    start_idx_per_video = [0, *itertools.accumulate(num_frames)]
    num_patches = [
        num_grids_per_frame[start_idx_per_video[i] : start_idx_per_video[i + 1]].sum()
        for i in range(len(num_frames))
    ]
    return (
        torch.stack(num_patches)
        if num_patches
        else torch.zeros(0, dtype=grid_thw.dtype, device=grid_thw.device)
    )


class KeyeVL1_5ImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - bnp: Batch size * Number of patches
        - c: Number of channels
        - ps: Patch size
        - ni: Number of images
        - g: Grid dimensions (3 for t, h, w)
    """

    type: Literal["pixel_values"]

    pixel_values: Annotated[
        torch.Tensor, TensorShape("bnp", 3, "ps", "ps", dynamic_dims={"bnp"})
    ]

    image_grid_thw: Annotated[torch.Tensor, TensorShape("ni", 3)]


class KeyeVL1_5ImageEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - nf: Number of image features
        - hs: Hidden size (must match the hidden size of language model
          backbone)
        - ni: Number of images
        - g: Grid dimensions (3 for t, h, w)
    """

    type: Literal["image_embeds"]
    image_embeds: Annotated[torch.Tensor, TensorShape("nf", "hs")]
    image_grid_thw: Annotated[torch.Tensor, TensorShape("ni", 3)]


KeyeVL1_5ImageInputs: TypeAlias = (
    KeyeVL1_5ImagePixelInputs | KeyeVL1_5ImageEmbeddingInputs
)


class KeyeVL1_5VideoPixelInputs(TensorSchema):
    """
    Dimensions:
        - bnp: Batch size * Number of patches
        - c: Number of channels
        - ps: Patch size
        - ni: Number of images
        - g: Grid dimensions (3 for t, h, w)
    """

    type: Literal["pixel_values_videos"]
    pixel_values_videos: Annotated[
        torch.Tensor, TensorShape("bnp", 3, "ps", "ps", dynamic_dims={"bnp"})
    ]
    video_grid_thw: Annotated[torch.Tensor, TensorShape("nv", 3)]

    num_frames: torch.Tensor


class KeyeVL1_5VideoEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - nf: Number of video features
        - hs: Hidden size (must match the hidden size of language model
          backbone)
        - nv: Number of videos
        - g: Grid dimensions (3 for t, h, w)
    """

    type: Literal["video_embeds"]
    video_embeds: Annotated[torch.Tensor, TensorShape("nf", "hs")]
    video_grid_thw: Annotated[torch.Tensor, TensorShape("nv", 3)]
    num_frames: torch.Tensor


KeyeVL1_5VideoInputs: TypeAlias = (
    KeyeVL1_5VideoPixelInputs | KeyeVL1_5VideoEmbeddingInputs
)


class KeyeVL1_5Projector(nn.Module):
    def __init__(
        self,
        text_config: PretrainedConfig,
        vision_config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.text_config = text_config
        self.vision_config = vision_config
        self.merge_kernel_size = (2, 2)

        self.hidden_size = (
            self.vision_config.hidden_size
            * self.merge_kernel_size[0]
            * self.merge_kernel_size[1]
        )

        self.pre_norm = torch.nn.LayerNorm(self.hidden_size, eps=1e-05)
        self.act = GELUActivation()

        self.linear_1 = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_1",
        )
        self.linear_2 = RowParallelLinear(
            self.hidden_size,
            self.text_config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_2",
        )

    def forward(
        self,
        image_features: torch.Tensor | tuple[torch.Tensor] | list[torch.Tensor],
        image_grid_thw: list[tuple[int, int, int]],
    ) -> torch.Tensor | list[torch.Tensor]:
        m1, m2 = self.merge_kernel_size
        if isinstance(image_features, (list, tuple)):
            processed_features = list()
            for image_feature, image_grid in zip(image_features, image_grid_thw):
                t, h, w = image_grid
                image_feature = rearrange(
                    image_feature,
                    "(t h p1 w p2) d -> (t h w) (p1 p2 d)",
                    t=t,
                    h=h // m1,
                    p1=m1,
                    w=w // m2,
                    p2=m2,
                )
                image_feature = self.pre_norm(image_feature)
                hidden_states, _ = self.linear_1(image_feature)
                hidden_states = self.act(hidden_states)
                hidden_states, _ = self.linear_2(hidden_states)
                processed_features.append(hidden_states)

            return processed_features

        dims = image_features.shape[:-1]
        dim = image_features.shape[-1]
        image_features = image_features.view(np.prod(dims), dim)
        hidden_states = self.pre_norm(image_features.view(-1, self.hidden_size))
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)

        return hidden_states.view(*dims, -1)


def _keye_field_config(
    hf_inputs: Mapping[str, torch.Tensor],
):
    image_grid_thw = hf_inputs.get(
        "image_grid_thw", torch.empty((0, 3), dtype=torch.int64)
    )
    image_grid_sizes = image_grid_thw.prod(-1)

    video_grid_thw = hf_inputs.get(
        "video_grid_thw", torch.empty((0, 3), dtype=torch.int64)
    )
    video_grid_thw = split_thw(video_grid_thw)
    num_frames = hf_inputs.get("num_frames", video_grid_thw[:, 0]).clone().tolist()

    video_num_patches = get_num_patches(video_grid_thw, num_frames)

    video_num_grids = []
    if len(num_frames) > 0:
        i = 0
        j = 1
        cur_frames = num_frames[i]
        for t, _, _ in video_grid_thw.tolist():
            cur_frames -= t
            if cur_frames == 0:
                video_num_grids.append(j)
                i += 1
                if i < len(num_frames):
                    cur_frames = num_frames[i]
                j = 1
            else:
                j += 1
    video_num_grids = torch.tensor(video_num_grids)
    return dict(
        pixel_values=MultiModalFieldConfig.flat_from_sizes("image", image_grid_sizes),
        image_embeds=MultiModalFieldConfig.flat_from_sizes("image", image_grid_sizes),
        image_grid_thw=MultiModalFieldConfig.batched("image"),
        pixel_values_videos=MultiModalFieldConfig.flat_from_sizes(
            "video", video_num_patches
        ),
        video_embeds=MultiModalFieldConfig.flat_from_sizes("video", video_num_patches),
        video_grid_thw=MultiModalFieldConfig.flat_from_sizes("video", video_num_grids),
        num_frames=MultiModalFieldConfig.batched("video"),
    )


class KeyeVL1_5MultiModalDataParser(MultiModalDataParser):
    def _parse_image_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[ImageItem],
    ) -> ModalityDataItems[Any, Any] | None:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="image",
                required_fields={
                    "image_embeds",
                    "image_grid_thw",
                },
                fields_factory=_keye_field_config,
            )

        return super()._parse_image_data(data)

    def _parse_video_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[VideoItem],
    ) -> ModalityDataItems[Any, Any] | None:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="video",
                required_fields={
                    "video_embeds",
                    "video_grid_thw",
                },
                fields_factory=_keye_field_config,
            )

        return super()._parse_video_data(data)


class KeyeVL1_5ProcessingInfo(KeyeProcessingInfo):
    def get_data_parser(self):
        return KeyeVL1_5MultiModalDataParser(
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_max_frame_per_video(self) -> int:
        return 2048

    def get_supported_mm_limits(
        self,
    ) -> Mapping[str, int | None]:
        return {"image": None, "video": 1}


class KeyeVL1_5MultiModalProcessor(BaseMultiModalProcessor[KeyeVL1_5ProcessingInfo]):
    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()
        image_token_id = vocab[hf_processor.image_token]
        video_token_id = vocab[hf_processor.video_token]
        placeholder = {"image": image_token_id, "video": video_token_id}
        merge_length = image_processor.merge_size**2

        out_mm_kwargs_data = out_mm_kwargs.get_data()
        frame_types: list[torch.Tensor] = hf_processor_mm_kwargs.get(
            "frame_types", None
        )
        timestamps: list[torch.Tensor] = hf_processor_mm_kwargs.get("timestamps", None)
        num_videos = mm_items.get_count("video", strict=False)

        if frame_types is None:
            frame_types = [None] * num_videos
        assert len(frame_types) == num_videos, (
            f"Number of frame_types={len(frame_types)} "
            f"doesn't equal to number of videos={num_videos}"
        )
        if timestamps is None:
            timestamps = [None] * num_videos
        assert len(timestamps) == num_videos, (
            f"Number of timestamps={len(timestamps)} "
            f"doesn't equal to number of videos={num_videos}"
        )

        video_grid_thw = out_mm_kwargs_data.get(
            "video_grid_thw", torch.empty((0, 3), dtype=torch.int64)
        )
        num_frames = out_mm_kwargs_data.get(
            "num_frames", torch.tensor([], dtype=torch.int64)
        )

        assert len(num_frames) == num_videos, (
            f"Size of num_frames={len(num_frames)} "
            f"doesn't equal to number of videos={num_videos}"
        )

        video_grid_hws = split_thw(video_grid_thw)
        assert int(num_frames.sum().tolist()) == video_grid_hws.shape[0], (
            f"The first dimension of `video_grid_hws`={video_grid_hws.shape[0]}"
            f"doesn't equal to num of frames."
        )

        cu_seqlens = torch.cumsum(torch.tensor([0] + num_frames.tolist()), dim=-1)

        def get_replacement_keye(item_idx: int, modality: str):
            """
            Args:
                item_idx(int): The item index of modality to replace
                modality(str): The modality
            """
            if modality == "image":
                out_item = out_mm_kwargs[modality][item_idx]
                grid_thw = out_item[f"{modality}_grid_thw"].data
                assert isinstance(grid_thw, torch.Tensor)

                num_tokens = int(grid_thw.prod()) // merge_length
                return [image_token_id] * num_tokens
            elif modality == "video":
                placeholders = []
                video_timestamps = timestamps[item_idx]
                video_frame_types = frame_types[item_idx]
                grid_thw = video_grid_hws[
                    cu_seqlens[item_idx] : cu_seqlens[item_idx + 1]
                ]

                nframes = grid_thw.shape[0]

                if video_timestamps is None:
                    video_timestamps = [""] * nframes
                else:
                    video_timestamps = [format(ts, ".1f") for ts in video_timestamps]

                if video_frame_types is None:
                    video_frame_types = [0] * nframes
                for i, sub_thw in enumerate(grid_thw):
                    s = f"{hf_processor.frame_token}{video_timestamps[i]}"
                    if video_frame_types[i] == 1:
                        s += hf_processor.fast_start
                    placeholders.extend(tokenizer.encode(s))
                    num_frame_tokens = int(sub_thw.prod()) // merge_length
                    placeholders.extend([video_token_id] * num_frame_tokens)
                    if video_frame_types[i] == 1:
                        placeholders.append(vocab[hf_processor.fast_end])

                return PromptUpdateDetails.select_token_id(
                    placeholders, embed_token_id=video_token_id
                )
            else:
                raise ValueError(f"Unsupported modality {modality}")

        return [
            PromptReplacement(
                modality=modality,
                target=[placeholder[modality]],
                replacement=partial(get_replacement_keye, modality=modality),
            )
            for modality in ("image", "video")
        ]

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _keye_field_config(hf_inputs)


class KeyeVL1_5DummyInputsBuilder(
    KeyeBaseDummyInputsBuilder[KeyeVL1_5ProcessingInfo]
): ...


@MULTIMODAL_REGISTRY.register_processor(
    KeyeVL1_5MultiModalProcessor,
    info=KeyeVL1_5ProcessingInfo,
    dummy_inputs=KeyeVL1_5DummyInputsBuilder,
)
class KeyeVL1_5ForConditionalGeneration(
    BaseKeyeModule, SupportsMultiModal, SupportsLoRA, SupportsPP, SupportsMRoPE
):
    def _build_projector(
        self,
        text_config: PretrainedConfig,
        vision_config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> nn.Module:
        return KeyeVL1_5Projector(text_config, vision_config, quant_config, prefix)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config: PretrainedConfig = vllm_config.model_config.hf_config
        self.merge_size = config.vision_config.spatial_merge_size
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> KeyeVL1_5ImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            return KeyeVL1_5ImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        if image_embeds is not None:
            return KeyeVL1_5ImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw,
            )

    def _parse_and_validate_video_input(
        self, **kwargs: object
    ) -> KeyeVL1_5VideoInputs | None:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        num_frames = kwargs.pop("num_frames", None)

        if pixel_values_videos is None and video_embeds is None:
            return None

        if pixel_values_videos is not None:
            return KeyeVL1_5VideoPixelInputs(
                type="pixel_values_videos",
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                num_frames=num_frames,
            )

        if video_embeds is not None:
            return KeyeVL1_5VideoEmbeddingInputs(
                type="video_embeds",
                video_embeds=video_embeds,
                video_grid_thw=video_grid_thw,
                num_frames=num_frames,
            )

    def _process_video_input(
        self, video_input: KeyeVL1_5VideoInputs
    ) -> tuple[torch.Tensor, ...]:
        video_type = video_input["type"]
        video_grid_thw = split_thw(video_input["video_grid_thw"])
        pixel_values_videos = video_input.get("pixel_values_videos", None)

        video_embeds = self._process_video_embeds(
            video_type, video_grid_thw, pixel_values_videos
        )
        video_embeds = torch.concat(video_embeds, dim=0)

        num_frames = video_input["num_frames"].clone().tolist()

        num_patches = get_num_patches(video_grid_thw, num_frames).tolist()

        patch_cu_seqlens = torch.cumsum(
            torch.tensor([0] + num_patches).detach().clone(), dim=-1
        )
        patch_cu_seqlens = torch.div(
            patch_cu_seqlens, self.merge_size**2, rounding_mode="floor"
        )

        new_video_embeds = []
        for idx in range(patch_cu_seqlens.shape[0] - 1):
            start = patch_cu_seqlens[idx]
            end = patch_cu_seqlens[idx + 1]
            new_video_embeds.append(video_embeds[start:end])
        return tuple(new_video_embeds)

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        kwargs = MultiModalFeatureSpec.gather_kwargs(
            mm_features,
            {"image_grid_thw", "video_grid_thw"},
        )
        image_grid_thw = [item.tolist() for item in kwargs.get("image_grid_thw", [])]
        video_grid_thw = [item.tolist() for item in kwargs.get("video_grid_thw", [])]

        if isinstance(video_grid_thw, list) and len(video_grid_thw) > 0:
            video_grid_thw = video_grid_thw[0]

        def split_thw(grid_thw: torch.Tensor | list[int]) -> list[list[int]]:
            """
            Split grid_thw along the t dimension.

            Args:
                grid_thw: shape [N, 3] tensor or nested list of [t, h, w].

            Returns:
                List of [1, h, w] rows, repeated t times for each original row.
            """

            if isinstance(grid_thw, list):
                grid_thw = torch.tensor(grid_thw, dtype=torch.long)

            if grid_thw.numel() == 0:
                return []

            t, hw = grid_thw[:, 0], grid_thw[:, 1:]
            ones = torch.ones_like(hw[:, :1])  # [N,1]
            out = torch.cat([ones, hw], dim=1).repeat_interleave(t, dim=0)
            return out.tolist()

        video_grid_thw = split_thw(video_grid_thw)

        hf_config = self.config
        image_token_id = hf_config.image_token_id
        video_token_id = hf_config.video_token_id
        spatial_merge_size = hf_config.vision_config.spatial_merge_size

        image_nums = len(image_grid_thw)
        frame_nums = len(video_grid_thw)
        llm_pos_ids_list: list = []

        st = 0
        remain_images, remain_frames = image_nums, frame_nums

        image_index, video_index = 0, 0
        for _ in range(image_nums + frame_nums):
            if remain_images > 0:
                try:
                    ed_image = input_tokens.index(image_token_id, st)
                except ValueError:
                    ed_image = len(input_tokens) + 1
            else:
                ed_image = len(input_tokens) + 1
            if remain_frames > 0:
                try:
                    ed_video = input_tokens.index(video_token_id, st)
                except ValueError:
                    ed_video = len(input_tokens) + 1
            else:
                ed_video = len(input_tokens) + 1

            if ed_image < ed_video:
                t, h, w = image_grid_thw[image_index]
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = video_grid_thw[video_index]
                video_index += 1
                remain_frames -= 1
                ed = ed_video

            llm_grid_t, llm_grid_h, llm_grid_w = (
                t,
                h // spatial_merge_size,
                w // spatial_merge_size,
            )
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )

            t_index = (
                (
                    torch.arange(llm_grid_t)
                    .view(-1, 1)
                    .expand(-1, llm_grid_h * llm_grid_w)
                )
                .long()
                .flatten()
            )

            h_index = (
                torch.arange(llm_grid_h)
                .view(1, -1, 1)
                .expand(llm_grid_t, -1, llm_grid_w)
                .flatten()
            )
            w_index = (
                torch.arange(llm_grid_w)
                .view(1, 1, -1)
                .expand(llm_grid_t, llm_grid_h, -1)
                .flatten()
            )
            llm_pos_ids_list.append(
                torch.stack([t_index, h_index, w_index]) + text_len + st_idx
            )
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        mrope_position_delta = (llm_positions.max() + 1 - len(input_tokens)).item()

        return llm_positions, mrope_position_delta
