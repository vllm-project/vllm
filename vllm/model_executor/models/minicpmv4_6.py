# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only MiniCPM-V 4.6 model (MiniCPMV4_6ForConditionalGeneration)."""

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch
from PIL import Image as PILImage
from torch import nn
from transformers import MiniCPMV4_6Config

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    NestedTensors,
)
from vllm.multimodal.parse import ImageProcessorItems, ImageSize, VideoProcessorItems
from vllm.multimodal.processing.processor import (
    PromptReplacement,
    PromptUpdateDetails,
    ResolvedPromptUpdate,
    cached_encode,
)
from vllm.sequence import IntermediateTensors

from .idefics2_vision_model import Idefics2VisionTransformer
from .interfaces import (
    HasInnerState,
    IsHybrid,
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
    _require_is_multimodal,
)
from .minicpmv import (
    MiniCPMVDummyInputsBuilder,
    MiniCPMVImageEmbeddingInputs,
    MiniCPMVImageEmbeddingItems,
    MiniCPMVImagePixelInputs,
    MiniCPMVMultiModalProcessor,
    MiniCPMVProcessingInfo,
    MiniCPMVVideoEmbeddingItems,
    _image_kwargs_from_video,
)
from .module_mapping import MultiModelKeys
from .qwen3_5 import Qwen3_5ForCausalLM
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    _merge_multimodal_embeddings,
    flatten_bn,
    maybe_prefix,
)
from .vision import is_vit_use_data_parallel


def _minicpmv4_6_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    fields = dict(
        pixel_values=MultiModalFieldConfig.batched("image"),
        tgt_sizes=MultiModalFieldConfig.batched("image"),
        image_embeds=MultiModalFieldConfig.batched("image"),
        video_pixel_values=MultiModalFieldConfig.batched("video"),
        video_image_sizes=MultiModalFieldConfig.batched("video", keep_on_cpu=True),
        video_tgt_sizes=MultiModalFieldConfig.batched("video"),
        video_embeds=MultiModalFieldConfig.batched("video"),
    )
    if "use_vit_merger" in hf_inputs:
        fields["use_vit_merger"] = MultiModalFieldConfig.batched(
            "image", keep_on_cpu=True
        )
    if "video_use_vit_merger" in hf_inputs:
        fields["video_use_vit_merger"] = MultiModalFieldConfig.batched(
            "video", keep_on_cpu=True
        )
    return fields


def _expand_flags(value: object, num_items: int) -> list[bool | None]:
    """Broadcast a batched per-item flag to exactly one entry per item."""
    if value is None:
        return [None] * num_items

    raw: list[object]
    if isinstance(value, torch.Tensor):
        raw = value.reshape(-1).tolist()
    elif isinstance(value, list | tuple):
        raw = []
        for item in value:
            if isinstance(item, torch.Tensor):
                raw.extend(item.reshape(-1).tolist())
            else:
                raw.append(item)
    else:
        raw = [value]

    if len(raw) == num_items:
        return [bool(v) for v in raw]
    if len(raw) == 1:
        return [bool(raw[0])] * num_items
    # Unexpected shape: keep one decision for the whole batch instead of
    # failing the request.
    return [any(bool(v) for v in raw)] * num_items


def _vision_downsample_mode(flag: bool | None) -> str | None:
    if flag is None:
        return None
    return "16x" if flag else "4x"


def _select_vision_items(
    image_input: Mapping[str, object],
    item_indices: list[int],
    num_slices: list[int],
):
    """Slice a vision input down to the given items, keeping their order."""
    offsets = [0]
    for n in num_slices:
        offsets.append(offsets[-1] + n)

    pixel_values: list[torch.Tensor] = []
    tgt_sizes: list[torch.Tensor] = []
    sizes: list[int] = []
    for i in item_indices:
        start, end = offsets[i], offsets[i + 1]
        pixel_values.extend(image_input["pixel_values"][start:end])
        tgt_sizes.append(image_input["tgt_sizes"][start:end])
        sizes.append(end - start)

    return MiniCPMVImagePixelInputs(
        type="pixel_values",
        pixel_values=pixel_values,
        tgt_sizes=torch.cat(tgt_sizes, dim=0),
        num_slices=torch.tensor(sizes),
    )


# HF-style modality-scoped keys inside `mm_processor_kwargs`. They scope a value
# to a single modality and must not be forwarded to the processors, which take
# flat arguments only.
_MODALITY_SCOPED_MM_KWARGS = ("images_kwargs", "videos_kwargs", "audio_kwargs")


def _flat_processor_kwargs(
    mm_kwargs: Mapping[str, object],
    downsample_mode: str,
) -> dict[str, object]:
    """Drop the modality-scoped nesting and pin the resolved downsample mode.

    The placeholder count and the vision tower must read the same mode, so the
    value resolved for this modality is passed down explicitly rather than
    leaving the processor to fall back to its own default.
    """
    kwargs = {
        key: value
        for key, value in mm_kwargs.items()
        if key not in _MODALITY_SCOPED_MM_KWARGS
    }
    kwargs["downsample_mode"] = downsample_mode
    return kwargs


class MiniCPMV4_6MultiModalProcessor(MiniCPMVMultiModalProcessor):
    def _resolve_downsample_mode(
        self,
        mm_kwargs: Mapping[str, object],
        modality: str = "image",
    ) -> str:
        # A flat `downsample_mode` applies to both modalities; the nested
        # `images_kwargs` / `videos_kwargs` forms scope it to one, matching
        # MiniCPMV4_6Processor.
        info = self.info
        assert isinstance(info, MiniCPMV4_6ProcessingInfo)
        merged = info.ctx.get_merged_mm_kwargs(mm_kwargs, modality=modality)
        downsample_mode = merged.get("downsample_mode")
        if downsample_mode is not None:
            return str(downsample_mode)
        return info._get_downsample_mode()

    def _resolve_max_slice_nums(
        self,
        mm_kwargs: Mapping[str, object],
        modality: str = "image",
    ) -> int | None:
        # Per-request max_slice_nums. None keeps the processor default.
        # Resolved like `downsample_mode` so the nested per-modality form
        # (`images_kwargs` / `videos_kwargs`) is honoured too. This must match
        # what `process_images` passes to the image processor: the placeholder
        # count and the visual token count are derived from it separately, so
        # they diverge (and the engine aborts) if the two sides read different
        # values.
        info = self.info
        assert isinstance(info, MiniCPMV4_6ProcessingInfo)
        merged = info.ctx.get_merged_mm_kwargs(mm_kwargs, modality=modality)
        max_slice = merged.get("max_slice_nums")
        if max_slice is None:
            return None
        return int(max_slice)

    def get_image_prompt_texts(
        self,
        image_size,
        image_idx: int = 0,
        downsample_mode: str | None = None,
        max_slice_nums: int | None = None,
    ) -> str:
        info = self.info
        assert isinstance(info, MiniCPMV4_6ProcessingInfo)
        return info.get_slice_image_placeholder(
            image_size,
            image_idx=image_idx,
            max_slice_nums=max_slice_nums,
            downsample_mode=downsample_mode,
        )

    def _video_local_id_prefix(self, video_idx: int) -> str:
        """`<image_id>{idx}</image_id>` prefix emitted in front of a video placeholder.

        MiniCPM-V 4.6 numbers each video with a local id, matching
        MiniCPMV4_6Processor. 4.7 treats a video as one temporal sequence and is
        trained without these ids, so it overrides this to return an empty string.
        """
        tokenizer = self.info.get_tokenizer()
        id_start = getattr(tokenizer, "image_id_start_token", "<image_id>")
        id_end = getattr(tokenizer, "image_id_end_token", "</image_id>")
        return f"{id_start}{video_idx}{id_end}"

    def get_video_prompt_texts(
        self,
        image_size,
        num_frames: int,
        downsample_mode: str | None = None,
        video_idx: int = 0,
    ) -> str:
        # Match transformers v5.7+ MiniCPMV4_6Processor video formatting:
        #   <image_id>{video_idx}</image_id>(<image>VIDEO*src</image>
        #     <slice>VIDEO*patch</slice>...)*num_frames
        # Crucially the visual token inside each frame is ``<|video_pad|>``
        # (tokenizer.video_token), NOT ``<|image_pad|>`` — they share the same
        # embedding-injection role but the language model is conditioned on
        # which one is used. Using image_token for video silently produces
        # garbage descriptions.
        info = self.info
        assert isinstance(info, MiniCPMV4_6ProcessingInfo)
        grids, source_tokens, patch_tokens = info._compute_visual_tokens(
            image_size,
            max_slice_nums=info.get_video_max_slice_num(),
            downsample_mode=downsample_mode,
        )
        tokenizer = info.get_tokenizer()
        video_token = getattr(tokenizer, "video_token", "<|video_pad|>")
        image_start = getattr(tokenizer, "image_start_token", "<image>")
        image_end = getattr(tokenizer, "image_end_token", "</image>")
        slice_start = getattr(tokenizer, "slice_start_token", "<slice>")
        slice_end = getattr(tokenizer, "slice_end_token", "</slice>")

        per_frame = image_start + video_token * source_tokens + image_end
        if grids[0] > 0 and grids[1] > 0 and patch_tokens > 0:
            slice_ph = slice_start + video_token * patch_tokens + slice_end
            rows = [slice_ph * grids[1] for _ in range(grids[0])]
            per_frame += "\n".join(rows)

        body = per_frame * num_frames
        return self._video_local_id_prefix(video_idx) + body

    def process_images(
        self,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, NestedTensors]:
        if (images := mm_data.get("images")) is None:
            return {}

        mm_items = self.info.parse_mm_data({"image": images}, validate=False)
        parsed_images = mm_items.get_items(
            "image", (MiniCPMVImageEmbeddingItems, ImageProcessorItems)
        )

        if isinstance(parsed_images, MiniCPMVImageEmbeddingItems):
            return {}

        # transformers v5.7+ MiniCPMV4_6ImageProcessor returns
        # `pixel_values` (1, C, P, sum_W) where all slices are fused along W
        # (NaViT-style), and `target_sizes` (n_slices, 2). vLLM expects each
        # image entry to be a 4D tensor (n_slices, C, P, L_max_padded).
        n_images = len(parsed_images)
        image_processor = self.info.get_image_processor()
        patch_size = image_processor.patch_size
        ds_mode = self._resolve_downsample_mode(mm_kwargs, modality="image")
        image_proc_kwargs = _flat_processor_kwargs(mm_kwargs, ds_mode)
        # `_flat_processor_kwargs` drops the nested form, so the value resolved
        # from it has to be pinned back for the processor and the prompt text
        # to agree on the slice count.
        image_max_slice = self._resolve_max_slice_nums(mm_kwargs, modality="image")
        if image_max_slice is not None:
            image_proc_kwargs["max_slice_nums"] = image_max_slice
        per_image_pixel_values: list[torch.Tensor] = []
        per_image_tgt_sizes: list[torch.Tensor] = []
        for image in parsed_images:
            ip_out = image_processor([image], **image_proc_kwargs)
            pv = ip_out["pixel_values"]  # (1, C, P, sum_W)
            ts = ip_out["target_sizes"]  # (n_slices, 2)
            if pv.ndim == 4 and pv.shape[0] == 1:
                pv = pv.squeeze(0)  # (C, P, sum_W)
            ts_long = ts.to(torch.long)
            split_widths = (ts_long[:, 0] * ts_long[:, 1] * patch_size).tolist()
            slices = torch.split(pv, split_widths, dim=-1)
            n_slices = len(slices)
            l_max = max(s.shape[-1] for s in slices)
            out = torch.zeros(
                n_slices,
                pv.shape[0],
                pv.shape[1],
                l_max,
                dtype=pv.dtype,
                device=pv.device,
            )
            for i, s in enumerate(slices):
                out[i, :, :, : s.shape[-1]] = s
            per_image_pixel_values.append(out)
            per_image_tgt_sizes.append(ts_long)

        image_inputs: dict = {
            "pixel_values": per_image_pixel_values,
            "tgt_sizes": per_image_tgt_sizes,
        }

        insert_layer_id = getattr(
            self.info.get_hf_config(),
            "insert_layer_id",
            -1,
        )
        merger_flag = ds_mode != "4x" and insert_layer_id >= 0
        image_inputs["use_vit_merger"] = [
            torch.tensor([merger_flag], dtype=torch.bool) for _ in range(n_images)
        ]
        return image_inputs

    def process_videos(
        self,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, NestedTensors]:
        if (videos := mm_data.get("videos")) is None:
            return {}

        mm_items = self.info.parse_mm_data({"video": videos}, validate=False)
        parsed_videos = mm_items.get_items(
            "video", (MiniCPMVVideoEmbeddingItems, VideoProcessorItems)
        )

        if isinstance(parsed_videos, MiniCPMVVideoEmbeddingItems):
            return {}

        # Treat each video as a sequence of frames. The transformers v5.7+
        # `MiniCPMV4_6ImageProcessor` returns NaViT-style fused `pixel_values`;
        # we run it per-frame, split the slices, then re-pack each video into
        # a single 4D tensor (sum_slices, C, P, L_max_video).
        image_processor = self.info.get_image_processor()
        patch_size = image_processor.patch_size
        video_max_slice = self.info.get_video_max_slice_num()
        video_ds_mode = self._resolve_downsample_mode(mm_kwargs, modality="video")
        video_mm_kwargs = {
            **_flat_processor_kwargs(mm_kwargs, video_ds_mode),
            "max_slice_nums": video_max_slice,
        }

        per_video_pixel_values: list[torch.Tensor] = []
        per_video_tgt_sizes: list[torch.Tensor] = []
        per_video_image_sizes: list[torch.Tensor] = []

        for video in parsed_videos:
            # video is iterable of frames (PIL Image or numpy array).
            all_slices: list[torch.Tensor] = []
            ts_list: list[torch.Tensor] = []
            frame_sizes: list[torch.Tensor] = []
            for frame in video:
                # Record per-frame (W, H) for video_image_sizes so that
                # get_video_prompt_texts can consume a consistent frame size.
                if isinstance(frame, PILImage.Image):
                    w, h = frame.size
                elif isinstance(frame, np.ndarray):
                    if frame.ndim == 3 and frame.shape[-1] in (1, 3, 4):
                        # HWC (e.g. from np.array(PIL.Image))
                        h, w = frame.shape[0], frame.shape[1]
                    else:
                        # CHW
                        _, h, w = frame.shape
                elif isinstance(frame, torch.Tensor):
                    if frame.ndim == 3 and frame.shape[-1] in (1, 3, 4):
                        h, w = frame.shape[0], frame.shape[1]
                    else:
                        _, h, w = frame.shape
                else:
                    raise TypeError(f"Unsupported frame type: {type(frame)}")
                frame_sizes.append(torch.tensor([w, h], dtype=torch.long, device="cpu"))

                ip_out = image_processor([frame], **video_mm_kwargs)
                pv = ip_out["pixel_values"]  # (1, C, P, sum_W)
                ts = ip_out["target_sizes"]  # (n_slices, 2)
                if pv.ndim == 4 and pv.shape[0] == 1:
                    pv = pv.squeeze(0)  # (C, P, sum_W)
                ts_long = ts.to(torch.long)
                split_widths = (ts_long[:, 0] * ts_long[:, 1] * patch_size).tolist()
                slices = torch.split(pv, split_widths, dim=-1)
                all_slices.extend(slices)
                ts_list.append(ts_long)

            if not all_slices:
                continue

            l_max = max(s.shape[-1] for s in all_slices)
            n_total = len(all_slices)
            C, P = all_slices[0].shape[0], all_slices[0].shape[1]
            out = torch.zeros(
                n_total,
                C,
                P,
                l_max,
                dtype=all_slices[0].dtype,
                device=all_slices[0].device,
            )
            for i, s in enumerate(all_slices):
                out[i, :, :, : s.shape[-1]] = s

            per_video_pixel_values.append(out)
            per_video_tgt_sizes.append(torch.cat(ts_list, dim=0))
            per_video_image_sizes.append(torch.stack(frame_sizes))

        if not per_video_pixel_values:
            return {}

        insert_layer_id = getattr(self.info.get_hf_config(), "insert_layer_id", -1)
        merger_flag = video_ds_mode != "4x" and insert_layer_id >= 0

        return {
            "video_pixel_values": per_video_pixel_values,
            "video_tgt_sizes": per_video_tgt_sizes,
            "video_image_sizes": per_video_image_sizes,
            "video_use_vit_merger": [
                torch.tensor([merger_flag], dtype=torch.bool)
                for _ in per_video_pixel_values
            ],
        }

    def _get_prompt_updates(
        self,
        mm_items,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs,
    ):
        image_ds_mode = self._resolve_downsample_mode(
            hf_processor_mm_kwargs, modality="image"
        )
        video_ds_mode = self._resolve_downsample_mode(
            hf_processor_mm_kwargs, modality="video"
        )
        image_max_slice = self._resolve_max_slice_nums(hf_processor_mm_kwargs)

        placeholders = [
            ("image", self.info.image_pattern),
            ("video", self.info.video_pattern),
        ]
        tokenizer = self.info.get_tokenizer()
        additional_placeholders = []
        for modality, pattern in placeholders:
            sub_pattern = tokenizer.decode(
                cached_encode(tokenizer, pattern, add_special_tokens=False)
            )
            if sub_pattern != pattern:
                additional_placeholders.append((modality, sub_pattern))
        placeholders += additional_placeholders

        # The 4.6 chat_template emits `<|image_pad|>` / `<|video_pad|>` rather
        # than `<unk>`, so use those tokens as the embedding selector.
        image_embed_text = getattr(tokenizer, "image_token", "<|image_pad|>")
        video_embed_text = getattr(tokenizer, "video_token", "<|video_pad|>")
        vocab = tokenizer.get_vocab()
        image_embed_ids = [vocab[image_embed_text]]
        video_embed_ids = [vocab[video_embed_text]]

        def get_image_replacement(item_idx: int):
            images = mm_items.get_items(
                "image",
                (MiniCPMVImageEmbeddingItems, ImageProcessorItems),
            )
            image_size = images.get_image_size(item_idx)
            return PromptUpdateDetails.select_token_ids(
                cached_encode(
                    tokenizer,
                    self.get_image_prompt_texts(
                        image_size,
                        item_idx,
                        downsample_mode=image_ds_mode,
                        max_slice_nums=image_max_slice,
                    ),
                    add_special_tokens=False,
                ),
                image_embed_ids,
            )

        def get_video_replacement(item_idx: int):
            # Prefer video_image_sizes from processed data so that the
            # placeholder count is driven by the same frame sizes that the
            # vision tower will actually consume.
            video_mm_kwargs = out_mm_kwargs.get("video")
            if video_mm_kwargs is not None and item_idx < len(video_mm_kwargs):
                video_item = video_mm_kwargs[item_idx]
                image_sizes_elem = video_item.get("video_image_sizes")
                if image_sizes_elem is not None and image_sizes_elem.data is not None:
                    # image_sizes_elem.data: (num_frames, 2) – each row is [W, H]
                    image_sizes = image_sizes_elem.data
                    num_frames = image_sizes.shape[0]
                    frame_size = ImageSize(
                        width=int(image_sizes[0, 0].item()),
                        height=int(image_sizes[0, 1].item()),
                    )
                    return PromptUpdateDetails.select_token_ids(
                        cached_encode(
                            tokenizer,
                            self.get_video_prompt_texts(
                                frame_size,
                                num_frames,
                                downsample_mode=video_ds_mode,
                                video_idx=item_idx,
                            ),
                            add_special_tokens=False,
                        ),
                        video_embed_ids,
                    )

            videos = mm_items.get_items(
                "video",
                (MiniCPMVVideoEmbeddingItems, VideoProcessorItems),
            )
            frame_size = videos.get_frame_size(item_idx)
            num_frames = videos.get_num_frames(item_idx)
            return PromptUpdateDetails.select_token_ids(
                cached_encode(
                    tokenizer,
                    self.get_video_prompt_texts(
                        frame_size,
                        num_frames,
                        downsample_mode=video_ds_mode,
                        video_idx=item_idx,
                    ),
                    add_special_tokens=False,
                ),
                video_embed_ids,
            )

        get_replacement = {
            "image": get_image_replacement,
            "video": get_video_replacement,
        }

        return [
            PromptReplacement(
                modality=modality,
                target=cached_encode(tokenizer, pattern, add_special_tokens=False),
                replacement=get_replacement[modality],
            )
            for modality, pattern in placeholders
        ]

    def _recompute_cached_prompt_update(
        self, cached_update: ResolvedPromptUpdate, new_item_idx: int
    ) -> ResolvedPromptUpdate:
        new_update = super()._recompute_cached_prompt_update(
            cached_update, new_item_idx
        )
        # MiniCPM-V 4.6 prefixes video placeholders with `<image_id>{idx}</image_id>`
        # (the base class only rewrites the image modality). 4.7 emits no such
        # prefix, so there is nothing to renumber.
        if cached_update.modality == "video":
            old_prefix = self._video_local_id_prefix(cached_update.item_idx)
            new_prefix = self._video_local_id_prefix(new_item_idx)
            if old_prefix and old_prefix != new_prefix:
                tokenizer = self.info.get_tokenizer()
                video_token = getattr(tokenizer, "video_token", "<|video_pad|>")

                text = tokenizer.decode(cached_update.content.full)

                new_update = new_update.with_content(
                    PromptUpdateDetails.select_token_ids(
                        cached_encode(
                            tokenizer,
                            text.replace(old_prefix, new_prefix, 1),
                            add_special_tokens=False,
                        ),
                        cached_encode(tokenizer, video_token, add_special_tokens=False),
                    )
                )
        return new_update

    def _get_mm_fields_config(
        self,
        hf_inputs,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _minicpmv4_6_field_config(hf_inputs)


class MiniCPMV4_6ProcessingInfo(MiniCPMVProcessingInfo):
    # transformers v5.7+ chat_template emits these as image/video placeholders.
    image_pattern = "<|image_pad|>"
    video_pattern = "<|video_pad|>"

    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs: object):
        # MiniCPM-V 4.6 keeps the native transformers MiniCPMV4_6Processor:
        # this model has its own image/video handling and prompt-update logic
        # below, so it does not need (and is incompatible with) the vendored
        # MiniCPMVProcessor used by 2.x/4.0/4.5, whose __init__ assumes a
        # legacy `image_processor.version` attribute that 4.6 no longer has.
        hf_processor = self.ctx.get_hf_processor(**kwargs)

        # NumPy arrays are considered as Iterable but not Sequence in
        # https://github.com/huggingface/transformers/blob/main/src/transformers/image_transforms.py#L428
        image_processor = getattr(hf_processor, "image_processor", None)
        if image_processor is not None:
            # transformers v5+ renamed `mean`/`std` -> `image_mean`/`image_std`
            for attr in ("mean", "std", "image_mean", "image_std"):
                val = getattr(image_processor, attr, None)
                if isinstance(val, np.ndarray):
                    setattr(image_processor, attr, val.tolist())

        return hf_processor

    def _get_expected_hidden_size(self) -> int:
        config = self.get_hf_config()
        if hasattr(config, "text_config") and config.text_config is not None:
            return config.text_config.hidden_size
        return config.hidden_size

    def get_model_version(self):
        return (4, 6)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "video": None}

    def get_image_max_slice_num(self) -> int:
        config = self.get_hf_config()
        if hasattr(config, "slice_config") and config.slice_config is not None:
            return getattr(config.slice_config, "max_slice_nums", 9)
        return getattr(config, "max_slice_nums", 9)

    def get_video_max_slice_num(self) -> int:
        # Override the base class default of 1: transformers v5.7+
        # `MiniCPMV4_6VideoProcessor` keeps the same max_slice_nums (default 9)
        # as the image processor so that high-res frames get sliced.
        try:
            hf_processor = self.get_hf_processor()
            video_processor = getattr(hf_processor, "video_processor", None)
            if video_processor is not None:
                return int(getattr(video_processor, "max_slice_nums", 9))
        except Exception:
            pass
        return self.get_image_max_slice_num()

    def _get_downsample_mode(
        self,
        downsample_mode: str | None = None,
    ) -> str:
        if downsample_mode is not None:
            return downsample_mode
        image_processor = self.get_image_processor()
        return getattr(image_processor, "downsample_mode", "16x")

    def _compute_visual_tokens(
        self,
        image_size: ImageSize,
        max_slice_nums: int | None = None,
        downsample_mode: str | None = None,
    ) -> tuple[list[int], int, int]:
        """Compute grid, source_image_visual_tokens and patch_visual_tokens.

        Args:
            image_size: Size of the source image.
            max_slice_nums: Maximum number of slices, or None for the default.
            downsample_mode: ``"16x"`` (default, full merge) or ``"4x"``
                (skip vit_merger, 4x more visual tokens).

        Returns:
            (grids, source_image_visual_tokens, patch_visual_tokens)
            grids is [0, 0] when no slicing occurs.

        """
        image_processor = self.get_image_processor()
        if max_slice_nums is None:
            max_slice_nums = image_processor.max_slice_nums

        patch_size = image_processor.patch_size
        scale_res = image_processor.scale_resolution
        downsample_mode = self._get_downsample_mode(downsample_mode)
        token_divisor = 4 if downsample_mode == "4x" else 16

        # vLLM ImageSize is (width, height); transformers expects (height, width)
        hf_image_size = (image_size.height, image_size.width)

        # transformers v5.7+ requires `scale_resolution` arg
        try:
            grids = image_processor.get_sliced_grid(
                hf_image_size,
                max_slice_nums,
                scale_res,
            )
        except TypeError:
            grids = image_processor.get_sliced_grid(
                hf_image_size,
                max_slice_nums,
            )

        if grids is None:
            best_size = image_processor.find_best_resize(
                hf_image_size,
                scale_res,
                patch_size,
                allow_upscale=True,
            )
            source_tokens = (
                best_size[0] * best_size[1] // (patch_size * patch_size * token_divisor)
            )
            return [0, 0], source_tokens, 0

        best_resize = image_processor.find_best_resize(
            hf_image_size,
            scale_res,
            patch_size,
        )
        source_tokens = (
            best_resize[0] * best_resize[1] // (patch_size * patch_size * token_divisor)
        )
        refine_size = image_processor.get_refine_size(
            hf_image_size,
            grids,
            scale_res,
            patch_size,
            allow_upscale=True,
        )
        patch_w = refine_size[0] // grids[0]
        patch_h = refine_size[1] // grids[1]
        patch_tokens = patch_w * patch_h // (patch_size * patch_size * token_divisor)
        return grids, source_tokens, patch_tokens

    def get_slice_image_placeholder(
        self,
        image_size,
        image_idx: int = 0,
        max_slice_nums: int | None = None,
        use_image_id: bool = True,
        downsample_mode: str | None = None,
    ) -> str:
        grids, source_tokens, patch_tokens = self._compute_visual_tokens(
            image_size,
            max_slice_nums,
            downsample_mode=downsample_mode,
        )
        image_processor = self.get_image_processor()
        # transformers v5.7+ removed `get_slice_image_placeholder` from the
        # image_processor and moved the logic into MiniCPMV4_6Processor.
        # Replicate it here using tokenizer special tokens.
        if hasattr(image_processor, "get_slice_image_placeholder"):
            return image_processor.get_slice_image_placeholder(
                grids,
                image_idx=image_idx,
                max_slice_nums=max_slice_nums,
                use_image_id=use_image_id,
                source_image_visual_tokens=source_tokens,
                patch_visual_tokens=patch_tokens,
            )
        tokenizer = self.get_tokenizer()
        image_token = getattr(tokenizer, "image_token", "<|image_pad|>")
        image_start = getattr(tokenizer, "image_start_token", "<image>")
        image_end = getattr(tokenizer, "image_end_token", "</image>")
        slice_start = getattr(tokenizer, "slice_start_token", "<slice>")
        slice_end = getattr(tokenizer, "slice_end_token", "</slice>")
        id_start = getattr(tokenizer, "image_id_start_token", "<image_id>")
        id_end = getattr(tokenizer, "image_id_end_token", "</image_id>")

        placeholder = image_start + image_token * source_tokens + image_end
        if use_image_id:
            placeholder = f"{id_start}{image_idx}{id_end}" + placeholder

        num_rows, num_cols = grids[0], grids[1]
        if num_cols > 0 and num_rows > 0 and patch_tokens > 0:
            slice_ph = slice_start + image_token * patch_tokens + slice_end
            slices = [slice_ph * num_cols for _ in range(num_rows)]
            placeholder += "\n".join(slices)
        return placeholder

    def get_num_image_tokens(
        self,
        image_size,
        max_slice_nums: int | None = None,
        downsample_mode: str | None = None,
    ) -> int:
        grids, source_tokens, patch_tokens = self._compute_visual_tokens(
            image_size,
            max_slice_nums,
            downsample_mode=downsample_mode,
        )
        return source_tokens + grids[0] * grids[1] * patch_tokens


def _stack_vit_merger_qkv(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    """Fold separate vit_merger q/k/v projections into the fused qkv_proj module.

    HF checkpoints store the window-attention projections separately, while
    vLLM keeps them fused. The shard_id mechanism on the submodule's
    WeightsMapper cannot do this: AutoWeightsLoader still recurses over every
    incoming key after calling a submodule's load_weights, so a "k_proj" key
    with no matching child raises before the mapping is honoured.
    """
    pairs = (
        ("vit_merger.self_attn.q_proj.", "q"),
        ("vit_merger.self_attn.k_proj.", "k"),
        ("vit_merger.self_attn.v_proj.", "v"),
    )
    for name, tensor in weights:
        for src, shard in pairs:
            if src in name:
                name = name.replace(src, "vit_merger.self_attn.qkv_proj.", 1)
                tensor.shard_id = shard
                break
        yield name, tensor


class MiniCPMV4_6ViTWindowAttentionSelfAttn(nn.Module):
    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        use_data_parallel = is_vit_use_data_parallel()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        tp_size = 1 if use_data_parallel else get_tensor_model_parallel_world_size()
        assert self.num_heads % tp_size == 0
        self.num_heads_per_partition = self.num_heads // tp_size

        self.qkv_proj = QKVParallelLinear(
            self.embed_dim,
            self.head_dim,
            self.num_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            disable_tp=use_data_parallel,
        )
        self.out_proj = RowParallelLinear(
            self.embed_dim,
            self.embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
            disable_tp=use_data_parallel,
        )
        self.attn = MMEncoderAttention(
            self.num_heads_per_partition,
            self.head_dim,
            self.scale,
            prefix=f"{prefix}.attn",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        attn_out = self.attn(q, k, v)
        out, _ = self.out_proj(attn_out)
        return out


class MiniCPMV4_6ViTWindowAttentionMerger(nn.Module):
    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.window_kernel_size = (2, 2)
        self.embed_dim = config.hidden_size

        self.self_attn = MiniCPMV4_6ViTWindowAttentionSelfAttn(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.layer_norm1 = nn.LayerNorm(
            self.embed_dim,
            eps=config.layer_norm_eps,
        )

        hidden_4x = self.embed_dim * 4
        inter_4x = config.intermediate_size * 4

        self.pre_norm = nn.LayerNorm(hidden_4x, eps=config.layer_norm_eps)
        self.linear_1 = nn.Linear(hidden_4x, inter_4x, bias=True)
        self.act = get_act_fn("gelu_pytorch_tanh")
        self.linear_2 = nn.Linear(inter_4x, self.embed_dim, bias=True)

    def _apply_window_attention(
        self,
        valid_states: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        D = valid_states.shape[-1]
        wh, ww = self.window_kernel_size
        nh, nw = H // wh, W // ww
        num_windows = nh * nw

        x = valid_states.view(H, W, D)
        x = x.view(nh, wh, nw, ww, D).permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(num_windows, wh * ww, D)

        x = self.self_attn(x)

        x = x.view(nh, nw, wh, ww, D).permute(0, 2, 1, 3, 4).contiguous()
        return x.view(H * W, D)

    def _apply_mlp_downsample(
        self,
        valid_states: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        D = valid_states.shape[-1]
        wh, ww = self.window_kernel_size
        nh, nw = H // wh, W // ww

        x = valid_states.view(H, W, D)
        x = x.view(nh, wh, nw, ww, D).permute(0, 2, 1, 3, 4).contiguous()

        residual = x.reshape(nh * nw, wh * ww, D).mean(dim=1)
        x = x.reshape(nh * nw, wh * ww * D)

        x = self.pre_norm(x)
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x + residual

    def forward(
        self,
        hidden_states: torch.Tensor,
        tgt_sizes: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        B, _L, D = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        all_merged = []
        new_tgt_sizes = torch.zeros_like(tgt_sizes)

        for b in range(B):
            H, W = tgt_sizes[b].tolist()
            hs = hidden_states[b, : H * W, :]

            residual = hs
            hs = self.layer_norm1(hs)
            hs = residual + self._apply_window_attention(hs, H, W)

            wh, ww = self.window_kernel_size
            new_H, new_W = H // wh, W // ww
            all_merged.append(self._apply_mlp_downsample(hs, H, W))
            new_tgt_sizes[b] = torch.tensor(
                [new_H, new_W],
                device=device,
                dtype=tgt_sizes.dtype,
            )

        new_num_patches = new_tgt_sizes[:, 0] * new_tgt_sizes[:, 1]
        new_max_patches = int(new_num_patches.max().item())
        new_hidden = torch.zeros(
            B,
            new_max_patches,
            D,
            device=device,
            dtype=dtype,
        )
        for b, merged in enumerate(all_merged):
            new_hidden[b, : merged.shape[0], :] = merged

        # Build new attention mask after spatial downsampling
        new_attention_mask: torch.Tensor | None = None
        if attention_mask is not None:
            mask = torch.zeros(
                B,
                new_max_patches,
                dtype=torch.bool,
                device=device,
            )
            for b in range(B):
                mask[b, : int(new_num_patches[b].item())] = True
            min_val = torch.finfo(dtype).min
            new_attention_mask = (~mask).to(dtype=dtype) * min_val
            new_attention_mask = new_attention_mask[:, None, None, :]

        return new_hidden, new_tgt_sizes, new_attention_mask


class MiniCPMV4_6DownsampleMLP(nn.Module):
    """Match HF (transformers v5.7+) parameter naming: pre_norm/linear_1/
    act/linear_2 (instead of pre_norm + Sequential(mlp.0/mlp.2))."""

    def __init__(
        self,
        hidden_size: int,
        llm_embed_dim: int,
        merge_kernel_size: tuple[int, int] = (2, 2),
    ):
        super().__init__()
        self.merge_kernel_size = merge_kernel_size
        self.hidden_size = hidden_size * merge_kernel_size[0] * merge_kernel_size[1]
        self.pre_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.act = get_act_fn("gelu")
        self.linear_2 = nn.Linear(self.hidden_size, llm_embed_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_norm(x)
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x


class MiniCPMV4_6Merger(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        llm_embed_dim: int,
        merge_kernel_size: tuple[int, int] = (2, 2),
        times: int = 1,
    ):
        super().__init__()
        self.merge_kernel_size = merge_kernel_size
        self.times = times
        self.mlp = nn.ModuleList(
            [
                MiniCPMV4_6DownsampleMLP(
                    hidden_size,
                    llm_embed_dim if i == times - 1 else hidden_size,
                    merge_kernel_size,
                )
                for i in range(times)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        tgt_sizes: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Args:
        hidden_states: (B, max_patches, D) padded batch.
        tgt_sizes: (B, 2) actual (H, W) per sample.

        """
        m1, m2 = self.merge_kernel_size
        results = []

        for b in range(len(tgt_sizes)):
            h, w = tgt_sizes[b].tolist()
            n_patches = h * w
            hs = hidden_states[b, :n_patches, :]

            hs = hs.reshape(h // m1, m1, w // m2, m2, -1)
            hs = hs.permute(0, 2, 1, 3, 4).reshape(
                (h // m1) * (w // m2),
                m1 * m2 * hs.shape[-1],
            )
            hs = self.mlp[0](hs)

            if self.times > 1:
                cur_h, cur_w = h // m1, w // m2
                for t in range(1, self.times):
                    cur_h, cur_w = cur_h // m1, cur_w // m2
                    hs = hs.reshape(cur_h, m1, cur_w, m2, -1)
                    hs = hs.permute(0, 2, 1, 3, 4).reshape(
                        cur_h * cur_w,
                        m1 * m2 * hs.shape[-1],
                    )
                    hs = self.mlp[t](hs)

            results.append(hs)

        return results


@MULTIMODAL_REGISTRY.register_processor(
    MiniCPMV4_6MultiModalProcessor,
    info=MiniCPMV4_6ProcessingInfo,
    dummy_inputs=MiniCPMVDummyInputsBuilder,
)
class MiniCPMV4_6ForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsLoRA,
    SupportsPP,
    HasInnerState,
    IsHybrid,
    SupportsMRoPE,
):
    supports_encoder_tp_data = True

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
        },
        orig_to_new_prefix={
            # transformers v5.7+ uses `vision_tower` and nests `vit_merger`
            # inside it. Order matters: more specific prefix must come first.
            "model.vision_tower.vit_merger.": "vit_merger.",
            "model.vision_tower.": "vpm.",
            "model.vpm.": "vpm.",
            "model.vit_merger.": "vit_merger.",
            "model.merger.": "merger.",
            "model.language_model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
            "mtp.": None,
        },
    )

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        # transformers v5.7+ chat_template uses these tokens.
        if modality.startswith("image"):
            return "<|image_pad|>"
        if modality.startswith("video"):
            return "<|video_pad|>"
        raise ValueError("Only image or video modality is supported")

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list["MultiModalFeatureSpec"],
    ) -> tuple[torch.Tensor, int]:
        """MiniCPM-V uses embedding injection for vision, not spatial M-RoPE.

        All tokens (text and vision placeholders) get identical sequential
        positions duplicated across the 3 M-RoPE channels expected by the
        Qwen3.5 backbone.
        """
        seq_len = len(input_tokens)
        positions = torch.arange(seq_len).unsqueeze(0).expand(3, -1)
        return positions, 0

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: MiniCPMV4_6Config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        assert multimodal_config is not None

        self.config = config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"

        # --- Vision tower ---
        with self._mark_tower_model(vllm_config, {"image"}):
            self.vpm = Idefics2VisionTransformer(
                config.vision_config,
                quant_config=quant_config,
                apply_encoder_attention_mask=True,
                prefix=maybe_prefix(prefix, "vpm"),
            )
            if config.drop_vision_last_layer:
                self.vpm.encoder.layers = self.vpm.encoder.layers[:-1]

            self.vit_merger = MiniCPMV4_6ViTWindowAttentionMerger(
                config.vision_config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "vit_merger"),
            )
            self.merger = MiniCPMV4_6Merger(
                hidden_size=config.vision_config.hidden_size,
                llm_embed_dim=config.text_config.hidden_size,
            )

        # --- Language model ---
        # Temporarily swap top-level model_type so that Qwen3_5ForCausalLM
        # picks up the expected text config when introspecting the hf config.
        with self._mark_language_model(vllm_config):
            saved_model_type = config.model_type
            config.model_type = "qwen3_5_text"
            try:
                self.language_model = Qwen3_5ForCausalLM(
                    vllm_config=vllm_config,
                    prefix=maybe_prefix(prefix, "language_model"),
                )
            finally:
                config.model_type = saved_model_type

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    # ----- Multimodal parsing -----

    def _parse_and_validate_vision_input(
        self,
        **kwargs: object,
    ) -> MiniCPMVImagePixelInputs | MiniCPMVImageEmbeddingInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if image_embeds is not None:
            return MiniCPMVImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds,
            )

        assert isinstance(pixel_values, torch.Tensor | list)
        tgt_sizes = kwargs.pop("tgt_sizes")
        num_slices_flat = torch.tensor([len(ps) for ps in pixel_values])
        pixel_values_flat = flatten_bn(pixel_values)
        tgt_sizes_flat = flatten_bn(tgt_sizes, concat=True)

        return MiniCPMVImagePixelInputs(
            type="pixel_values",
            pixel_values=pixel_values_flat,
            tgt_sizes=tgt_sizes_flat,
            num_slices=num_slices_flat,
        )

    # ----- Vision forward -----

    def get_vision_hidden_states(
        self,
        data: MiniCPMVImagePixelInputs,
        downsample_mode: str | None = None,
    ) -> list[torch.Tensor]:
        pixel_values = data["pixel_values"]
        tgt_sizes = data["tgt_sizes"]

        B = len(pixel_values)
        P = pixel_values[0].shape[-2]
        L = max(item.shape[-1] for item in pixel_values)
        device = pixel_values[0].device
        target_dtype = self.vpm.embeddings.patch_embedding.weight.dtype

        all_pixel_values = torch.zeros(
            B,
            3,
            P,
            L,
            dtype=target_dtype,
            device=device,
        )
        for i, pv in enumerate(pixel_values):
            all_pixel_values[i, ..., : pv.shape[-1]] = pv.to(target_dtype)

        num_patches = tgt_sizes.prod(-1)
        max_patches = int(num_patches.max().item())
        patch_attn_mask = torch.zeros(
            B,
            max_patches,
            dtype=torch.bool,
            device=device,
        )
        for i in range(B):
            patch_attn_mask[i, : num_patches[i]] = True

        hidden_states = self.vpm.embeddings(
            all_pixel_values,
            patch_attention_mask=patch_attn_mask.unsqueeze(1),
            tgt_sizes=tgt_sizes,
        )

        if torch.any(~patch_attn_mask):
            mask_dtype = hidden_states.dtype
            min_val = torch.finfo(mask_dtype).min
            attention_mask = (~patch_attn_mask).to(dtype=mask_dtype) * min_val
            attention_mask = attention_mask[:, None, None, :]
        else:
            attention_mask = None

        # Encoder layers with mid-encoder merger injection
        insert_layer_id = getattr(self.config, "insert_layer_id", -1)
        if downsample_mode is None:
            downsample_mode = getattr(self.config, "downsample_mode", "16x")
        use_vit_merger = downsample_mode != "4x" and insert_layer_id >= 0

        for layer in self.vpm.encoder.layers[: insert_layer_id + 1]:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        if use_vit_merger:
            hidden_states, tgt_sizes, attention_mask = self.vit_merger(
                hidden_states,
                tgt_sizes,
                attention_mask,
            )

        for layer in self.vpm.encoder.layers[insert_layer_id + 1 :]:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        # 4. Post layernorm
        hidden_states = self.vpm.post_layernorm(hidden_states)

        # 5. MLP merger → list of per-slice tensors
        return self.merger(hidden_states, tgt_sizes)

    def _process_vision_input(self, image_input, use_vit_merger=None):
        if image_input["type"] == "image_embeds":
            return image_input["image_embeds"]

        num_slices = image_input["num_slices"].tolist()
        flags = _expand_flags(use_vit_merger, len(num_slices))

        # The merger flag is per request, but the model runner may batch items
        # from several concurrent requests. Encoding such a mixed batch would
        # apply one request's downsample mode to another request's items, so
        # group by flag and encode each group separately.
        encoded: dict[int, torch.Tensor] = {}
        for flag in dict.fromkeys(flags):
            item_indices = [i for i, f in enumerate(flags) if f == flag]
            group_input = _select_vision_items(image_input, item_indices, num_slices)
            group_features = self.get_vision_hidden_states(
                group_input,
                downsample_mode=_vision_downsample_mode(flag),
            )

            idx = 0
            for i in item_indices:
                n = num_slices[i]
                encoded[i] = torch.cat(group_features[idx : idx + n], dim=0)
                idx += n

        return [encoded[i] for i in range(len(num_slices))]

    # ----- Multimodal embedding interface -----

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        use_vit_merger = kwargs.pop("use_vit_merger", None)
        video_use_vit_merger = kwargs.pop("video_use_vit_merger", None)

        image_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ("pixel_values", "image_embeds", "tgt_sizes")
        }
        video_kwargs = _image_kwargs_from_video(kwargs)

        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        if (
            image_kwargs.get("pixel_values") is not None
            or image_kwargs.get("image_embeds") is not None
        ):
            image_input = self._parse_and_validate_vision_input(**image_kwargs)
            if image_input is not None:
                multimodal_embeddings += tuple(
                    self._process_vision_input(
                        image_input,
                        use_vit_merger=use_vit_merger,
                    )
                )

        if (
            video_kwargs.get("pixel_values") is not None
            or video_kwargs.get("image_embeds") is not None
        ):
            video_input = self._parse_and_validate_vision_input(**video_kwargs)
            if video_input is not None:
                multimodal_embeddings += tuple(
                    self._process_vision_input(
                        video_input,
                        use_vit_merger=video_use_vit_merger,
                    )
                )

        if not multimodal_embeddings:
            return []
        return multimodal_embeddings

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

        is_multimodal = _require_is_multimodal(is_multimodal)
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    # ----- Forward / Logits -----

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            inputs_embeds = None

        return self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    # ----- Weight loading -----

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(
            _stack_vit_merger_qkv(weights),
            mapper=self.hf_to_vllm_mapper,
        )

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector=["vit_merger", "merger"],
            tower_model="vpm",
        )

    # ----- Mamba / Hybrid state helpers (same as Qwen3.5 VLM) -----

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()
