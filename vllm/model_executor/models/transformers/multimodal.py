# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Transformers modeling backend mixin for multi-modal models."""

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from functools import cached_property
from typing import TYPE_CHECKING, Any

import torch
import transformers

from vllm.compilation.decorators import should_torch_compile_mm_encoder
from vllm.config.utils import getattr_iter
from vllm.inputs import MultiModalDataBuiltins, MultiModalDataDict
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMRoPE,
    SupportsMultiModal,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargsItems
from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalFieldConfig
from vllm.multimodal.parse import ImageSize, MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
    cached_encode,
)
from vllm.multimodal.processing.processor import (
    HFMultiModalInputs,
    MultiModalProcessingResult,
    PlaceholderFeaturesInfo,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import get_submodel_config_name
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.torch_utils import async_tensor_h2d

from .base import Base

if TYPE_CHECKING:
    from transformers import BatchFeature, PreTrainedModel

    from vllm.config import VllmConfig
    from vllm.config.multimodal import MultiModalDummyOptions

logger = init_logger(__name__)

_MODALITY_TO_TOKEN_TYPE_ID = {"image": 1, "video": 2, "audio": 3}
_MODALITY_SIZE_KEYS = {
    "audio": "num_audio_tokens",
    "image": "num_image_patches",
    "video": "num_video_patches",
}
# NOTE: Profiling cap as in llava_onevision._MAX_FRAMES_PER_VIDEO
# past which a pixel budget only shrinks the frames, so Qwen3-VL's most is
# 12168 tokens at 16 frames and 9600 at its `max_frames` of 768
_MAX_FRAMES_PER_VIDEO = 16
# Arbitrary large size, bounded only by the processor's own resizing
_MAX_DUMMY_SIDE = 10_000
# Tiny inputs trip processors' channel-axis inference
_MIN_DUMMY_SIDE = 224


def _get_embed_token_id(
    replacement_ids: list[int], preferred: int | None = None
) -> int:
    """The token holding the embeddings, which an expansion repeats."""
    if preferred is not None and replacement_ids.count(preferred) > 1:
        return preferred
    return int(torch.tensor(replacement_ids).mode().values)


def _validate_one_audio_per_video(num_audios: int, num_videos: int) -> None:
    if num_videos > num_audios:
        raise ValueError(
            "use_audio_in_video needs one audio per video, got "
            f"num_audios={num_audios} and num_videos={num_videos}"
        )


def _count_embed_tokens(
    seqs: list[list[int]], preferred: int | None = None
) -> torch.Tensor:
    """Number of embedding tokens in each item's replacement."""
    counts = []
    for seq in seqs:
        if preferred is not None and (count := seq.count(preferred)):
            counts.append(count)
            continue
        counts.append(seq.count(_get_embed_token_id(seq)))
    return torch.tensor(counts)


class MultiModalProcessingInfo(BaseProcessingInfo):
    def _get_audio_processor(self) -> Any:
        # TODO: drop feature_extractor branch once huggingface/transformers#44394 lands.
        return getattr_iter(
            self.get_hf_processor(), ("audio_processor", "feature_extractor")
        )

    @cached_property
    def _is_audio_model(self) -> bool:
        return self._get_audio_processor() is not None

    @cached_property
    def _is_image_model(self) -> bool:
        return hasattr(self.get_hf_processor(), "image_processor")

    @cached_property
    def _is_video_model(self) -> bool:
        if not hasattr(self.get_hf_processor(), "video_processor"):
            return False
        try:
            Base.check_version("5.18.0", "video inputs")
        except ImportError as e:
            logger.info_once("%s, so video inputs are disabled.", e)
            return False
        num_frames = self._get_min_video_frames()
        side = _MIN_DUMMY_SIDE
        # TODO: Drop the except branch once every video processor can count, see
        # https://github.com/huggingface/transformers/issues/43329
        try:
            mm_tokens = self._get_num_mm_tokens(video_sizes=([num_frames, side, side],))
        except AttributeError:
            logger.info_once(
                "%s cannot count video tokens yet, so the Transformers modeling "
                "backend serves this model without video inputs. Please report "
                "this to transformers so it can be fixed, see "
                "https://github.com/huggingface/transformers/issues/43329",
                type(self.get_hf_processor()).__name__,
            )
            return False
        return mm_tokens["num_video_tokens"] is not None

    @cached_property
    def _video_needs_metadata(self) -> bool:
        video_processor = getattr(self.get_hf_processor(), "video_processor", None)
        return getattr(video_processor, "do_sample_frames", False)

    def _get_min_video_frames(self) -> int:
        video_processor = self.get_hf_processor().video_processor
        # A processor without `temporal_patch_size` groups no frames, so one is enough
        return getattr(video_processor, "temporal_patch_size", 1)

    def _get_supported_modalities(self) -> list[str]:
        modalities = []
        if self._is_audio_model:
            modalities.append("audio")
        if self._is_image_model:
            modalities.append("image")
        if self._is_video_model:
            modalities.append("video")
        if not modalities:
            raise ValueError(
                f"{type(self.get_hf_processor()).__name__} exposes no image, video "
                "or audio processor, so the Transformers modeling backend cannot "
                "serve this model as multi-modal."
            )
        return modalities

    @cached_property
    def _fuses_audio_into_video(self) -> bool:
        return bool(self.ctx.get_merged_mm_kwargs({}).get("use_audio_in_video"))

    def _get_audio_sampling_rate(self) -> float:
        sub = self._get_audio_processor()
        if sub is not None and hasattr(sub, "sampling_rate"):
            return sub.sampling_rate
        return 16000.0

    def get_data_parser(self) -> MultiModalDataParser:
        target_sr = self._get_audio_sampling_rate() if self._is_audio_model else None
        return MultiModalDataParser(
            target_sr=target_sr,
            video_needs_metadata=self._video_needs_metadata,
            expected_hidden_size=self._get_expected_hidden_size(),
            allow_missing_mm_embeddings=self.allow_missing_mm_embeddings,
        )

    def get_supported_mm_limits(self):
        return dict.fromkeys(self._get_supported_modalities())

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        modalities = self._get_supported_modalities()
        max_tokens = {}
        if "audio" in modalities:
            max_tokens["audio"] = self.get_max_audio_tokens()
        if "image" in modalities:
            max_tokens["image"] = self.get_max_image_tokens()
        if "video" in modalities:
            max_tokens["video"] = self.get_max_video_tokens(seq_len, mm_counts)
        return max_tokens

    def get_max_audio_tokens(self) -> int:
        config = self.get_hf_config()
        audio_config_names = ("audio_config", "encoder_config")
        names = ("max_source_positions", "max_position_embeddings", "max_pos_emb")
        submodel = get_submodel_config_name(config)
        if submodel is not None:
            config = getattr(config, submodel)
        audio_config = getattr_iter(config, audio_config_names, default=config)
        val = getattr_iter(audio_config, names)
        if val is None:
            val = getattr(self.get_hf_processor(), "audio_seq_length", None)
        if val is not None:
            return int(val)
        raise ValueError(
            f"Unable to get max input length from {type(audio_config).__name__}. "
            f"The following attribute names were checked: {names}, and "
            "`audio_seq_length` on the processor."
        )

    def _get_num_mm_tokens(self, **sizes: Sequence[Sequence[int]]) -> Any:
        processor = self.get_hf_processor()
        multimodal_config = self.ctx.model_config.get_multimodal_config()
        mm_processor_kwargs = multimodal_config.mm_processor_kwargs or {}
        return processor._get_num_multimodal_tokens(**sizes, **mm_processor_kwargs)

    def get_max_image_tokens(self) -> int:
        size = self.get_image_size_with_most_features()
        return self._get_num_image_tokens(size)

    def get_max_video_tokens(self, seq_len: int, mm_counts: Mapping[str, int]) -> int:
        num_frames = self.get_num_frames_with_most_features(seq_len, mm_counts)
        return self._get_max_video_tokens_for_frames(num_frames)

    def _get_max_video_tokens_for_frames(self, num_frames: int) -> int:
        size = self.get_video_size_with_most_features(num_frames)
        return self._get_num_video_tokens(num_frames, size)

    def _get_num_image_tokens(self, size: ImageSize) -> int:
        mm_tokens = self._get_num_mm_tokens(image_sizes=([size.height, size.width],))
        return mm_tokens["num_image_tokens"][0]

    def _get_num_video_tokens(self, num_frames: int, size: ImageSize) -> int:
        video_sizes = ([num_frames, size.height, size.width],)
        mm_tokens = self._get_num_mm_tokens(video_sizes=video_sizes)
        return mm_tokens["num_video_tokens"][0]

    def _get_size_candidates(
        self, sub_processor: Any, divisors: tuple[int, ...]
    ) -> list[ImageSize]:
        """Candidate sizes read off a sub-processor's `size`.

        The keys are one of `VALID_SIZE_DICT_KEYS`, so the bound is either exact
        or an area budget, which `divisors` splits over its items. `shortest_edge`
        bounds only the small side, so it yields no candidate.

        `longest_edge` is read as the area budget Qwen and GLM use it for. A
        processor using it as an edge length (SmolVLM) yields a tiny candidate,
        which is only chosen if it ties on tokens, so it is still correct.
        """
        size = getattr(sub_processor, "size", None) or {}
        height = size.get("height", size.get("max_height"))
        width = size.get("width", size.get("max_width"))
        if height and width:
            return [ImageSize(width=width, height=height)]
        if max_pixels := size.get("max_pixels", size.get("longest_edge")):
            sides = {math.isqrt(max_pixels // divisor) for divisor in divisors}
            return [
                ImageSize(width=side, height=side) for side in sorted(sides) if side
            ]
        return []

    def _get_size_with_most_tokens(
        self, candidates: list[ImageSize], count: Callable[[ImageSize], int]
    ) -> ImageSize:
        """Smallest size found that yields the most tokens.

        `size` bounds the resized output, not the token count, so the candidates
        are compared by token count against an arbitrary large fallback, which
        tiling processors need. The fallback goes last so a candidate wins a tie.
        Resize rounding can give a large input fewer tokens (Qwen2.5-VL), so the
        candidates matter for the count, not only for memory.

        The winner is then halved while the count holds, so processors with no
        usable bound aren't profiled on huge dummy inputs.
        """
        sizes = [*candidates, ImageSize(width=_MAX_DUMMY_SIDE, height=_MAX_DUMMY_SIDE)]
        num_tokens = [count(size) for size in sizes]
        most_tokens = max(num_tokens)
        size = sizes[num_tokens.index(most_tokens)]

        while min(size.width, size.height) // 2 >= _MIN_DUMMY_SIDE:
            smaller = ImageSize(width=size.width // 2, height=size.height // 2)
            if count(smaller) < most_tokens:
                break
            size = smaller
        return size

    def get_image_size_with_most_features(self) -> ImageSize:
        return self._get_size_with_most_tokens(
            self._get_size_candidates(self.get_hf_processor().image_processor, (1,)),
            self._get_num_image_tokens,
        )

    def get_video_size_with_most_features(self, num_frames: int) -> ImageSize:
        """Per-frame size that yields the most video tokens.

        The video processor's `size` budget may cover one frame, every frame or
        every temporal patch, so all three splits are tried.
        """
        video_processor = self.get_hf_processor().video_processor
        grid_t = max(num_frames // self._get_min_video_frames(), 1)
        return self._get_size_with_most_tokens(
            self._get_size_candidates(video_processor, (num_frames, grid_t, 1)),
            lambda size: self._get_num_video_tokens(num_frames, size),
        )

    def get_num_frames_with_most_features(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> int:
        max_videos = max(mm_counts.get("video", 0), 1)
        step = self._get_min_video_frames()
        num_frames = step
        while (
            num_frames + step <= _MAX_FRAMES_PER_VIDEO
            and self._get_max_video_tokens_for_frames(num_frames + step) * max_videos
            <= seq_len
        ):
            num_frames += step
        return num_frames


class MultiModalDummyInputsBuilder(BaseDummyInputsBuilder[MultiModalProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        text = ""
        if self.info._is_audio_model and (num_audios := mm_counts.get("audio", 0)):
            processor = self.info.get_hf_processor()
            audio_token = getattr(processor, "audio_token", "")
            # Separated so that adjacent placeholders stay distinguishable
            text += " ".join([audio_token] * num_audios)
        if self.info._is_image_model and (num_images := mm_counts.get("image", 0)):
            processor = self.info.get_hf_processor()
            image_token = getattr(processor, "image_token", "")
            # Some processors (e.g. HunYuanVL) reject a bare image token and
            # require each one to be wrapped in its start/end markers.
            start_token = getattr(processor, "image_start_token", "")
            end_token = getattr(processor, "image_end_token", "")
            text += f"{start_token}{image_token}{end_token}" * num_images
        if self.info._is_video_model and (num_videos := mm_counts.get("video", 0)):
            processor = self.info.get_hf_processor()
            video_token = getattr(processor, "video_token", "")
            text += video_token * num_videos
        return text

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: "MultiModalDummyOptions",
    ) -> MultiModalDataDict:
        data = MultiModalDataBuiltins()
        if self.info._is_audio_model and (num_audios := mm_counts.get("audio", 0)):
            sampling_rate = self.info._get_audio_sampling_rate()
            sub = self.info._get_audio_processor()
            chunk_length = getattr(sub, "chunk_length", None) if sub else None
            if chunk_length is None:
                chunk_length = 30
            data["audio"] = self._get_dummy_audios(
                length=int(chunk_length * sampling_rate),
                num_audios=num_audios,
                overrides=mm_options.get("audio"),
            )
        if self.info._is_image_model and (num_images := mm_counts.get("image", 0)):
            width, height = self.info.get_image_size_with_most_features()
            data["image"] = self._get_dummy_images(
                width=width,
                height=height,
                num_images=num_images,
                overrides=mm_options.get("image"),
            )
        if self.info._is_video_model and (num_videos := mm_counts.get("video", 0)):
            num_frames = self.info.get_num_frames_with_most_features(seq_len, mm_counts)
            width, height = self.info.get_video_size_with_most_features(num_frames)
            videos = self._get_dummy_videos(
                width=width,
                height=height,
                num_frames=num_frames,
                num_videos=num_videos,
                overrides=mm_options.get("video"),
            )
            # Only processors that sample frames need metadata, which has the
            # dummy frames consumed verbatim
            if self.info._video_needs_metadata:
                video_processor = self.info.get_hf_processor().video_processor
                fps = getattr(video_processor, "fps", None)
                videos = [
                    (
                        video,
                        {
                            "fps": fps,
                            "duration": len(video) / fps if fps else None,
                            "total_num_frames": len(video),
                            "frames_indices": list(range(len(video))),
                            "video_backend": "opencv",
                            "do_sample_frames": False,
                        },
                    )
                    for video in videos
                ]
            data["video"] = videos
        return data


class MultiModalProcessor(BaseMultiModalProcessor[MultiModalProcessingInfo]):
    """Locates placeholders from the `text_replacement_offsets` the HF processor
    reports, expressing each one as a `PromptUpdate`.

    Stating the expansion as an update is what lets it be rebuilt from an
    unexpanded prompt, so this processor takes the base class's processing path
    and with it the multi-modal processor cache.
    """

    def _get_hf_mm_inputs(
        self,
        mm_items: MultiModalDataItems,
        hf_kwargs: Mapping[str, object],
    ) -> HFMultiModalInputs:
        """Pass the video metadata along to the HF processor."""
        hf_inputs = super()._get_hf_mm_inputs(mm_items, hf_kwargs)
        hf_data = hf_inputs.hf_data

        if "videos" not in hf_data or not self.info._video_needs_metadata:
            return hf_inputs

        videos = hf_data["videos"]
        assert isinstance(videos, Sequence)
        videos, metadata = zip(*videos)
        hf_data["videos"] = list(videos)
        hf_data["video_metadata"] = [
            {k: v for k, v in item.items() if k != "do_sample_frames"}
            for item in metadata
        ]
        do_sample_frames = {item.get("do_sample_frames", False) for item in metadata}
        if len(do_sample_frames) > 1:
            raise ValueError(
                "The videos in a request must agree on `do_sample_frames`, "
                "since the HF processor takes one value for all of them."
            )

        return hf_inputs._replace(
            hf_kwargs={
                "do_sample_frames": do_sample_frames.pop(),
                **hf_inputs.hf_kwargs,
            }
        )

    def _get_modality_field_names(self, modality: str) -> set[str]:
        """Names of the fields the sub-processor for `modality` produces."""
        # TODO: use else branch only once huggingface/transformers#44394 lands.
        if modality == "audio":
            sub_processor = self.info._get_audio_processor()
        else:
            processor = self.info.get_hf_processor()
            sub_processor = getattr(processor, f"{modality}_processor", None)

        # Pre-computed embeddings bypass the sub-processor entirely
        names = {f"{modality}_embeds"}
        for name in getattr(sub_processor, "model_input_names", None) or ():
            # Companion masks are emitted but not always declared
            names.update((name, f"{name}_mask"))
            if name == "input_features":
                names.add("feature_attention_mask")
        return names

    def _partition_keys_by_modality(
        self,
        keys: list[str],
        modalities: list[str],
    ) -> dict[str, list[str]]:
        """Attribute each HF processor output key to the modality that produced it."""
        if len(modalities) == 1:
            return {modalities[0]: keys}

        claimed = {m: self._get_modality_field_names(m) for m in modalities}

        owned: dict[str, list[str]] = {modality: [] for modality in modalities}
        unclaimed = []
        for key in keys:
            for modality in modalities:
                if key in claimed[modality]:
                    owned[modality].append(key)
                    break
            else:
                unclaimed.append(key)

        if unclaimed:
            logger.warning_once(
                "Unable to attribute %s to any of the modalities %s, so they "
                "will not be passed to the model. Add them to the relevant "
                "sub-processor's `model_input_names` to fix this.",
                tuple(unclaimed),
                tuple(modalities),
            )

        return owned

    def _get_slice_dim(self, data: Any, total_rows: int) -> int:
        """Which dimension of a field holds the rows belonging to each item.

        Some processors (e.g., Idefics3) return image fields with a leading batch
        dimension, putting the rows one dimension further in.
        """
        if not isinstance(data, torch.Tensor) or data.ndim < 2:
            return 0
        if data.shape[0] != total_rows and data.shape[1] == total_rows:
            return 1
        return 0

    def _get_mm_fields_config(
        self,
        hf_inputs: "BatchFeature",
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        # HF Processors always return a mask but vLLM doesn't need it
        hf_inputs.pop("attention_mask", None)

        # Absent if the modality had no items
        sizes = {
            modality: hf_inputs.get(key)
            for modality, key in _MODALITY_SIZE_KEYS.items()
        }
        modalities = [m for m, size in sizes.items() if size is not None]

        # Keys we wrote ourselves, rather than ones a sub-processor produced
        own_keys = {*_MODALITY_SIZE_KEYS.values(), "num_video_tokens"} | {
            f"{modality}_replacement_{suffix}"
            for modality in modalities
            for suffix in ("ids", "sizes")
        }
        # Registered by name below, so no sub-processor has to claim them
        own_keys |= {
            "image_grid_thw",
            "video_grid_thw",
            "second_per_grid_ts",
            "audio_feature_lengths",
            "use_audio_in_video",
        }
        keys = [key for key in hf_inputs if key not in own_keys]
        owned = self._partition_keys_by_modality(keys, modalities)

        mm_fields = {
            key: self._get_field_config(modality, key, hf_inputs, sizes[modality])
            for modality in modalities
            for key in owned[modality]
        }

        if "audio" in modalities and "audio_feature_lengths" in hf_inputs:
            mm_fields["audio_feature_lengths"] = MultiModalFieldConfig.batched(
                "audio", keep_on_cpu=True
            )

        for modality in modalities:
            # One row per item, and only ever read on the CPU
            mm_fields[_MODALITY_SIZE_KEYS[modality]] = MultiModalFieldConfig.batched(
                modality, keep_on_cpu=True
            )
            replacement_sizes = hf_inputs.get(f"{modality}_replacement_sizes")
            if replacement_sizes is not None:
                mm_fields[f"{modality}_replacement_ids"] = (
                    MultiModalFieldConfig.flat_from_sizes(
                        modality, replacement_sizes, keep_on_cpu=True
                    )
                )

        if "image" in modalities:
            # Always one row per item, whatever they describe
            mm_fields["image_grid_thw"] = MultiModalFieldConfig.batched(
                "image", keep_on_cpu=True
            )
        if "video" in modalities:
            mm_fields["video_grid_thw"] = MultiModalFieldConfig.batched(
                "video", keep_on_cpu=True
            )
            if "use_audio_in_video" in hf_inputs:
                mm_fields["use_audio_in_video"] = MultiModalFieldConfig.shared(
                    "video", len(sizes["video"]), keep_on_cpu=True
                )
            mm_fields["second_per_grid_ts"] = MultiModalFieldConfig.batched(
                "video", keep_on_cpu=True
            )

        return mm_fields

    def _get_field_config(
        self,
        modality: str,
        key: str,
        hf_inputs: "BatchFeature",
        sizes: torch.Tensor,
    ) -> MultiModalFieldConfig:
        data = hf_inputs[key]
        # Un-padded fields are already one entry per item, so index rather than slice
        if modality == "audio" or isinstance(data, list):
            return MultiModalFieldConfig.batched(modality)

        total = int(sizes.sum())
        dim = self._get_slice_dim(data, total)
        if modality != "video":
            return MultiModalFieldConfig.flat_from_sizes(modality, sizes, dim=dim)

        rows = data.shape[dim]
        if rows == len(sizes):
            return MultiModalFieldConfig.batched("video")
        if rows == total:
            return MultiModalFieldConfig.flat_from_sizes("video", sizes, dim=dim)
        # Gemma 4 concatenates every video's frames along one axis, so a per-frame
        # field has one row per frame rather than per patch
        num_frames = hf_inputs.get("num_frames_per_video")
        if num_frames is not None and rows == int(num_frames.sum()):
            return MultiModalFieldConfig.flat_from_sizes("video", num_frames, dim=dim)
        # VideoLLaMA3's compression mask has one row per token the processor counts
        num_video_tokens = hf_inputs["num_video_tokens"]
        if rows == int(num_video_tokens.sum()):
            return MultiModalFieldConfig.flat_from_sizes(
                "video", num_video_tokens, dim=dim
            )
        # NOTE: Any other layout would need a per-model guess, which we'd like to avoid
        raise ValueError(
            f"{type(self.info.get_hf_processor()).__name__} returned {rows} row(s) of "
            f"`{key}` for {len(sizes)} video(s) with {total} patch(es), so the rows "
            "cannot be attributed to a video."
        )

    def _get_hf_mm_text(self, mm_counts: Mapping[str, int]) -> str:
        if self.info._fuses_audio_into_video:
            mm_counts = {
                **mm_counts,
                "audio": max(0, mm_counts.get("audio", 0) - mm_counts.get("video", 0)),
            }
        return self.dummy_inputs.get_dummy_text(mm_counts)

    def _unpad_audios(
        self,
        hf_inputs: "BatchFeature",
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> None:
        """Replace the audio fields with each audio processed on its own.

        Processors pad every audio up to the longest in the call, which would leave
        an item's data dependent on what it was processed with. Unlike images,
        nothing in the output states how long each one really is, and processors
        pad a lone audio too, so the only way to know what an audio produces by
        itself is to process it by itself.
        """
        audios = mm_data.get("audio")
        if not isinstance(audios, Iterable) or not audios:
            return
        if len({len(audio) for audio in audios}) == 1:
            return

        alone = [
            self.info.ctx.call_hf_processor(
                self.info.get_hf_processor(**mm_kwargs),
                dict(
                    text=self.dummy_inputs.get_dummy_text({"audio": 1}), audio=[audio]
                ),
                mm_kwargs,
            )
            for audio in audios
        ]

        for key in self._get_modality_field_names("audio"):
            if isinstance(hf_inputs.get(key), torch.Tensor):
                hf_inputs[key] = [output[key][0] for output in alone]

    def _unpad_images(self, hf_inputs: "BatchFeature") -> None:
        """Trim each image back to its own size when the processor padded them all
        to the largest in the batch.

        An image's data has to depend on nothing but that image, or the multi-modal
        processor cache would store it under that image's hash and later reuse it
        beside a different neighbour. Padding is re-applied when the encoder runs.
        """
        pixel_values = hf_inputs.get("pixel_values")
        image_sizes = hf_inputs.get("image_sizes")
        if not isinstance(pixel_values, torch.Tensor):
            return
        if not isinstance(image_sizes, torch.Tensor):
            return
        if pixel_values.ndim != 4 or len(pixel_values) != len(image_sizes):
            return

        # The sizes describe the trailing dimensions only if the largest of them is
        # what the batch was padded up to. Otherwise they mean something else, as
        # in llava-onevision, where they are the sizes before any processing.
        maxima = image_sizes.max(dim=0).values
        if maxima.tolist() != list(pixel_values.shape[-2:]):
            return

        hf_inputs["pixel_values"] = [
            image[..., :height, :width]
            for image, (height, width) in zip(pixel_values, image_sizes.tolist())
        ]

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        """Replace each modality's placeholder token with the token ids that item's
        replacement text encodes to, marking which of them hold embeddings."""
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()

        def get_target_token_ids(token: str | int | Sequence[int]) -> list[int]:
            if isinstance(token, str):
                return cached_encode(tokenizer, token, add_special_tokens=False)
            if isinstance(token, int):
                return [token]
            return list(token)

        updates = []
        for modality, items in out_mm_kwargs.items():
            token = getattr(hf_processor, f"{modality}_token")
            target = get_target_token_ids(token)
            preferred = target[0] if len(target) == 1 else None
            # Popped so they are neither cached nor sent to the model; the updates
            # they produce are cached alongside the item instead
            replacements = []
            for item in items:
                ids = item.pop(f"{modality}_replacement_ids").data
                assert isinstance(ids, torch.Tensor)
                replacement_ids = ids.tolist()
                replacements.append(
                    PromptUpdateDetails.select_token_id(
                        replacement_ids,
                        _get_embed_token_id(replacement_ids, preferred),
                    )
                )
            updates.append(
                PromptReplacement(
                    modality=modality,
                    target=target,
                    replacement=replacements.__getitem__,
                )
            )
        return updates

    def _derive_audio_from_video_placeholders(
        self,
        placeholders: Mapping[str, list[PlaceholderFeaturesInfo]],
    ) -> Mapping[str, list[PlaceholderFeaturesInfo]]:
        """The placeholders for audios folded into videos, which share their span."""
        if "video" not in placeholders:
            return placeholders

        hf_processor = self.info.get_hf_processor()
        tokenizer = self.info.get_tokenizer()
        audio_token_id = cached_encode(
            tokenizer, hf_processor.audio_token, add_special_tokens=False
        )[0]

        audio_placeholders = list(placeholders.get("audio", []))
        for video in placeholders["video"]:
            audio_placeholders.append(
                PlaceholderFeaturesInfo(
                    modality="audio",
                    item_idx=len(audio_placeholders),
                    start_idx=video.start_idx,
                    tokens=video.tokens,
                    is_embed=torch.tensor(video.tokens).eq(audio_token_id),
                )
            )
        return {**placeholders, "audio": audio_placeholders}

    def _maybe_apply_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        mm_res: MultiModalProcessingResult,
    ) -> tuple[list[int], Mapping[str, list[PlaceholderFeaturesInfo]]]:
        if not self.info._fuses_audio_into_video:
            return super()._maybe_apply_prompt_updates(mm_items, mm_res)

        mm_item_counts = mm_items.get_all_counts()
        self._validate_mm_kwargs(mm_res.kwargs, mm_item_counts)
        self._validate_mm_updates(mm_res.prompt_updates, mm_item_counts)

        num_audios = mm_item_counts.get("audio", 0)
        num_videos = mm_item_counts.get("video", 0)
        _validate_one_audio_per_video(num_audios, num_videos)
        num_standalone = num_audios - num_videos
        updates = dict(mm_res.prompt_updates)
        if "audio" in updates:
            updates["audio"] = updates["audio"][:num_standalone]
        prompt_ids, placeholders = self._apply_prompt_updates(
            mm_res.prompt_ids, updates
        )
        placeholders = self._derive_audio_from_video_placeholders(placeholders)
        self._validate_mm_placeholders(placeholders, mm_item_counts)
        return prompt_ids, placeholders

    def _get_num_image_patches(
        self,
        hf_inputs: "BatchFeature",
        mm_data: Mapping[str, object],
        num_images: int,
    ) -> torch.Tensor:
        """How many rows of the image fields belong to each image.

        Taken from whichever per-image count the processor reports,
        and checked against the data it has to slice.
        """
        if (grid := hf_inputs.get("image_grid_thw")) is not None:
            num_patches = grid.prod(-1)
        elif (counts := self._get_num_patches_per_image(mm_data)) is not None:
            num_patches = torch.tensor(counts)
        else:
            num_patches = torch.ones(num_images, dtype=torch.long)

        image_data = hf_inputs.get("pixel_values", hf_inputs.get("image_patches"))
        if isinstance(image_data, torch.Tensor):
            total = int(num_patches.sum())
            rows = image_data.shape[self._get_slice_dim(image_data, total)]
            if rows != total:
                raise ValueError(
                    f"{type(self.info.get_hf_processor()).__name__} returned "
                    f"{rows} row(s) of image data for {num_images} image(s), which "
                    f"cannot be split into the {num_patches.tolist()} row(s) per "
                    "image derived from its outputs, so the rows cannot be "
                    "attributed to an image. Gemma3 does this when "
                    "`do_pan_and_scan` crops an image."
                )
        return num_patches

    def _get_num_patches_per_image(
        self, mm_data: Mapping[str, object]
    ) -> list[int] | None:
        """Ask the HF processor how many rows of image data each image produces."""
        images = mm_data.get("images")
        if not isinstance(images, Iterable) or not images:
            return None
        try:
            sizes = [(image.height, image.width) for image in images]
            mm_tokens = self.info.get_hf_processor()._get_num_multimodal_tokens(
                image_sizes=sizes, **self.info.ctx.get_merged_mm_kwargs({})
            )
            return list(mm_tokens["num_image_patches"])
        except (AttributeError, KeyError, TypeError):
            return None

    def _apply_hf_processor_main(
        self,
        mm_items: MultiModalDataItems,
        hf_kwargs: Mapping[str, object],
    ) -> "BatchFeature":
        hf_data, hf_kwargs, passthrough_data = self._get_hf_mm_inputs(
            mm_items, hf_kwargs
        )

        if not hf_data:
            return self._finalize_hf_mm_data(hf_data, hf_kwargs, passthrough_data)

        prompt_text = hf_data.pop("text")
        assert isinstance(prompt_text, str)

        use_audio_in_video = self.info._fuses_audio_into_video
        override = hf_kwargs.get("use_audio_in_video")
        if override is not None and bool(override) != use_audio_in_video:
            raise ValueError(
                "use_audio_in_video is a server-level setting and cannot be "
                "overridden per request; pass it in --mm-processor-kwargs instead."
            )
        item_counts = mm_items.get_all_counts()
        num_audios = item_counts.get("audio", 0)
        if use_audio_in_video:
            _validate_one_audio_per_video(num_audios, item_counts.get("video", 0))

        # Ask for the replacement each placeholder expands to, and record it as
        # per-item fields: its token ids, and the tokens or patches behind them
        if has_mm_data := any(hf_data.values()):
            hf_data = {**hf_data, "return_text_replacement_offsets": True}

        try:
            hf_inputs = self.info.ctx.call_hf_processor(
                self.info.get_hf_processor(**hf_kwargs),
                dict(text=prompt_text, **hf_data),
                hf_kwargs,
            )
        except ValueError:
            if any(hf_data.values()):
                raise
            # Some processors reject a prompt holding placeholders with
            # no data to go with them, so tokenize it without them
            tokenizer = self.info.get_tokenizer()
            hf_inputs = transformers.BatchFeature(
                dict(input_ids=[tokenizer.encode(prompt_text)]), tensor_type="pt"
            )

        if self.info.ctx.model_config.uses_mrope:
            feature_attention_mask = getattr_iter(
                hf_inputs, ("feature_attention_mask", "input_features_mask"), None
            )
            if feature_attention_mask is not None:
                hf_inputs["audio_feature_lengths"] = feature_attention_mask.sum(-1)

        self._unpad_images(hf_inputs)
        self._unpad_audios(hf_inputs, hf_data, hf_kwargs)

        video_second_per_grid = hf_inputs.pop("video_second_per_grid", None)
        if video_second_per_grid is not None:
            hf_inputs["second_per_grid_ts"] = video_second_per_grid

        if use_audio_in_video:
            hf_inputs["use_audio_in_video"] = torch.tensor(True)

        # Drop the inputs the model would reject
        hf_inputs.pop("mm_token_type_ids", None)
        hf_inputs.pop("token_type_ids", None)

        offsets = hf_inputs.pop("text_replacement_offsets", None)
        # Some processors return an empty batch as a tensor rather than a list
        if offsets is None or len(offsets) == 0 or len(offsets[0]) == 0:
            if has_mm_data:
                raise ValueError(
                    f"{type(self.info.get_hf_processor()).__name__} returned no "
                    "text replacement offsets, so the Transformers modeling backend "
                    "cannot locate the placeholder of each item. Its `__call__` has "
                    "to reach `ProcessorMixin.get_text_with_replacements` with one "
                    "replacement per item, which usually means implementing "
                    "`replace_<modality>_token`. Please report this to transformers "
                    "so it can be fixed."
                )
            hf_inputs.pop("input_ids", None)
            return self._finalize_hf_mm_data(
                hf_data, hf_kwargs, passthrough_data, hf_inputs
            )

        tokenizer = self.info.get_tokenizer()
        replacements = defaultdict[str, list[list[int]]](list)
        for entry in offsets[0]:
            replacements[entry["type"]].append(
                cached_encode(tokenizer, entry["replacement"], add_special_tokens=False)
            )

        audio_token_id = None
        if "use_audio_in_video" in hf_inputs and replacements.get("video"):
            audio_token_id = cached_encode(
                tokenizer,
                self.info.get_hf_processor().audio_token,
                add_special_tokens=False,
            )[0]
            fused = [seq for seq in replacements["video"] if audio_token_id in seq]
            if len(replacements["audio"]) + len(fused) != num_audios:
                raise ValueError(
                    f"use_audio_in_video fused {len(fused)} of "
                    f"{len(replacements['video'])} videos with an audio, which "
                    f"accounts for {len(replacements['audio']) + len(fused)} of "
                    f"{num_audios} audios"
                )
            # Standalone audios first, so item indices match the derived placeholders
            replacements["audio"].extend(fused)
            if not replacements["audio"]:
                del replacements["audio"]

        for modality, seqs in replacements.items():
            hf_inputs[f"{modality}_replacement_ids"] = torch.tensor(
                [token_id for seq in seqs for token_id in seq]
            )
            hf_inputs[f"{modality}_replacement_sizes"] = torch.tensor(
                [len(seq) for seq in seqs]
            )
            if modality == "image":
                hf_inputs["num_image_patches"] = self._get_num_image_patches(
                    hf_inputs, hf_data, len(seqs)
                )
            elif modality == "audio":
                hf_inputs["num_audio_tokens"] = _count_embed_tokens(
                    seqs, audio_token_id
                )
            elif modality == "video":
                grid = hf_inputs.get("video_grid_thw")
                hf_inputs["num_video_patches"] = (
                    grid.prod(-1)
                    if grid is not None
                    else torch.ones(len(seqs), dtype=torch.long)
                )
                hf_inputs["num_video_tokens"] = _count_embed_tokens(seqs)

        hf_inputs.pop("input_ids")

        return self._finalize_hf_mm_data(
            hf_data, hf_kwargs, passthrough_data, hf_inputs
        )


class MultiModalMixin(SupportsMultiModal, SupportsMRoPE, Base):
    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = ""):
        # Skip SupportsMRoPE.__init__ and call the next class in MRO
        super(SupportsMRoPE, self).__init__(vllm_config=vllm_config, prefix=prefix)

    def _find_encoder_classes(
        self, model: "PreTrainedModel"
    ) -> dict[str, type["PreTrainedModel"]]:
        """Modalities whose encoder cannot be told apart from the model itself are
        omitted, as are those `get_encoder` rejects."""
        encoder_classes: dict[str, type[PreTrainedModel]] = {}
        for modality in _MODALITY_TO_TOKEN_TYPE_ID:
            try:
                encoder_cls = type(model.get_encoder(modality=modality))
            except (TypeError, ValueError):
                continue
            if encoder_cls is not type(model):
                encoder_classes[modality] = encoder_cls
        return encoder_classes

    @contextmanager
    def _mark_model_components(self, vllm_config: "VllmConfig"):
        model_config = vllm_config.model_config
        encoder_classes = self._pre_trained_model_classes.encoders
        if not encoder_classes:
            logger.debug("No encoders identified, so no components will be marked")
            yield
            return

        if model_config.skip_tokenizer_init:
            # Determining the supported modalities needs the HF processor, which in
            # turn needs a tokenizer
            mm_config = model_config.get_multimodal_config()
            if mm_config.mm_encoder_only or any(
                mm_config.get_limit_per_prompt(modality) == 0
                for modality in encoder_classes
            ):
                logger.warning_once(
                    "Unable to determine the supported modalities without a "
                    "tokenizer, so no model components will be skipped."
                )
            yield
            return

        # Modalities we don't serve report a limit of 999, which would stop their
        # encoder ever being skipped
        supported_modalities = MULTIMODAL_REGISTRY.get_processing_info(
            model_config
        ).supported_mm_limits

        # One encoder often serves several modalities, and may only be skipped when
        # all of them are disabled, so mark it once for the whole set
        modalities_by_encoder = defaultdict(set)
        for modality, encoder_cls in encoder_classes.items():
            if modality in supported_modalities:
                modalities_by_encoder[encoder_cls].add(modality)

        with ExitStack() as stack:
            stack.enter_context(
                self._mark_language_model(
                    vllm_config, targets=self._pre_trained_model_classes.decoder
                )
            )
            for encoder_cls, modalities in modalities_by_encoder.items():
                stack.enter_context(
                    self._mark_tower_model(vllm_config, modalities, targets=encoder_cls)
                )
            yield

    def _decorate_for_torch_compile(self):
        """Decorate the model's decoder and encoder classes to indicate to vLLM
        that they support torch compile if `can_enable_torch_compile` and
        `should_torch_compile_mm_encoder` are True respectively.
        """
        super()._decorate_for_torch_compile()
        # Decorate the encoder model classes to support torch compile if needed
        if self.compilation_config.compile_mm_encoder:
            encoder_classes = self._pre_trained_model_classes.encoders
            if not encoder_classes:
                raise ValueError(
                    "Unable to infer any encoder classes from the model. "
                    "You must either: update the model so that "
                    "https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.get_encoder"
                    " can detect the encoders correctly, or remove "
                    "'compile_mm_encoder'."
                )
            logger.warning_once(
                "Multimodal encoder compilation with the Transformers modeling backend "
                "is an experimental feature. It relies on:\n"
                "- The encoder being torch compilable.\n"
                "- All encoder tensor inputs must be type hinted as either "
                "`torch.Tensor` or `torch.FloatTensor`.\n"
                "- The 0-th dimension of all tensor inputs to the encoder being the "
                "dynamic dimension (e.g. sequence length, number of patches).\n"
                "Please report any issues you encounter to help us improve it."
            )
            # One encoder can serve several modalities, and must only be decorated once
            for encoder_cls in dict.fromkeys(encoder_classes.values()):
                self._decorate_cls_for_torch_compile(
                    cls=encoder_cls,
                    # TODO: properly infer dynamic_arg_dims based on the encoder's
                    # forward method signature. We assume dim 0 for all tensor inputs.
                    dynamic_arg_dims=None,
                    enable_if=should_torch_compile_mm_encoder,
                    is_encoder=True,
                )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        # Positions shape handling for MRoPE models
        if self.model_config.uses_mrope:
            # [3, seq_len] -> [3, 1, seq_len]
            positions = positions[:, None].contiguous()
        model_output = super().forward(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return model_output

    def get_language_model(self) -> torch.nn.Module:
        """Transformers modeling backend multimodal classes do not contain a separate
        vLLM language model class. Therefore, in order to return a language model vLLM
        class, we use a wrapper to give `self` the same interface as a text model."""
        # Exclude self and object
        bases = self.__class__.mro()[1:-1]
        # Keep only classes defined in `vllm.model_executor.models.transformers`
        bases = [b for b in bases if ".transformers." in b.__module__]
        # Exclude MultiModalMixin itself
        bases = [b for b in bases if b is not MultiModalMixin]

        class LanguageModel(*bases):  # type: ignore[misc]
            def __init__(self, multimodal_model):
                # Don't call super().__init__() to avoid re-initialization
                self.__dict__.update(multimodal_model.__dict__)

            model = getattr_iter(self.model, ("language_model", "text_model"), None)

        return LanguageModel(self)

    def get_mm_mapping(self) -> MultiModelKeys:
        """Get the module prefix in multimodal models"""
        for name in ("language_model", "text_model"):
            if getattr(self.model, name, None) is not None:
                return MultiModelKeys.from_string_field(language_model=f"model.{name}")
        raise ValueError(
            "Could not locate the language model submodule for LoRA support"
        )

    def _split_embeddings(
        self, embeddings: torch.Tensor, split_sizes: list[int]
    ) -> list[torch.Tensor]:
        total_expected = sum(split_sizes)

        # Flatten to 2D: [total_tokens, hidden_dim]
        if embeddings.ndim > 2:
            embeddings = embeddings.reshape(-1, embeddings.shape[-1])

        total_tokens = embeddings.shape[0]
        if total_tokens == total_expected:
            # Direct match: split_sizes are actual token counts
            token_split_sizes = split_sizes
        elif total_expected > 0 and total_tokens % total_expected == 0:
            # Uniform expansion: each item expands to N tokens
            tokens_per_item = total_tokens // total_expected
            token_split_sizes = [s * tokens_per_item for s in split_sizes]
        elif total_expected > 0:
            # TODO: make this an error once we know profiling never relies on it
            if total_tokens == 0:
                raise ValueError(
                    "Encoder returned empty embeddings. "
                    f"Expected {total_expected} tokens from "
                    f"split_sizes={split_sizes}"
                )
            # Keep the counts out of the message: `warning_once` keys its cache on
            # the args, so varying them would log on every new pair
            logger.warning_once(
                "Encoder returned a different number of tokens than expected; "
                "padding or truncating to fit. The embeddings are not trustworthy "
                "outside of memory profiling."
            )
            logger.debug(
                "Encoder returned %s tokens but %s were expected",
                total_tokens,
                total_expected,
            )
            if total_tokens < total_expected:
                repeat_factor = (total_expected + total_tokens - 1) // total_tokens
                embeddings = embeddings.repeat(repeat_factor, 1)
            embeddings = embeddings[:total_expected]
            token_split_sizes = split_sizes
        else:
            return []

        return list(torch.split(embeddings, token_split_sizes, dim=0))

    def _process_audio_input(self, **kwargs) -> list[torch.Tensor] | None:
        input_features: torch.Tensor | None = kwargs.pop("input_features", None)
        if input_features is None:
            input_features = kwargs.pop("input_values", None)
        if input_features is None:
            return None

        num_audio_tokens = kwargs.pop("num_audio_tokens")
        kwargs.pop("token_type_ids", None)
        kwargs.pop("mm_token_type_ids", None)

        # Per-audio token counts are needed as Python ints to split.
        with gpu_sync_allowed():
            split_sizes = num_audio_tokens.flatten().tolist()
        if isinstance(input_features, torch.Tensor):
            # HuggingFace's `get_audio_features` implementations branch on
            # per-sample feature lengths internally.
            with gpu_sync_allowed():
                audio_output = self.model.get_audio_features(
                    input_features, return_dict=True, **kwargs
                )
            return self._split_embeddings(audio_output.pooler_output, split_sizes)

        # Audios the processor left un-padded arrive as a list once their
        # lengths differ. Encode them one at a time so that none of them is
        # padded to match another.
        embeddings: list[torch.Tensor] = []
        for index, features in enumerate(input_features):
            audio_output = self.model.get_audio_features(
                features.unsqueeze(0),
                return_dict=True,
                **self._select_item_kwargs(kwargs, index, len(input_features)),
            )
            embeddings.extend(
                self._split_embeddings(audio_output.pooler_output, [split_sizes[index]])
            )
        return embeddings

    def _process_image_input(self, **kwargs) -> list[torch.Tensor] | None:
        pixel_values: torch.Tensor | None = kwargs.pop("pixel_values", None)
        image_embeds: torch.Tensor | None = kwargs.pop("image_embeds", None)
        # Model might use `image_patches` instead of `pixel_values`
        if pixel_values is None:
            pixel_values = kwargs.pop("image_patches", None)

        if image_embeds is not None:
            return [image_embeds]

        if pixel_values is None:
            return None

        return self._process_vision_input(
            "image", pixel_values, kwargs.pop("num_image_patches"), **kwargs
        )

    def _process_video_input(self, **kwargs) -> list[torch.Tensor] | None:
        pixel_values_videos: torch.Tensor | None = kwargs.pop(
            "pixel_values_videos", None
        )
        if pixel_values_videos is None:
            return None

        kwargs.pop("second_per_grid_ts", None)
        return self._process_vision_input(
            "video", pixel_values_videos, kwargs.pop("num_video_patches"), **kwargs
        )

    def _process_vision_input(
        self,
        modality: str,
        pixel_values: torch.Tensor,
        num_patches: torch.Tensor,
        **kwargs,
    ) -> list[torch.Tensor]:
        split_sizes = num_patches.flatten().tolist()
        if isinstance(pixel_values, torch.Tensor):
            vision_embeddings = self._get_features(modality, pixel_values, **kwargs)
            if isinstance(vision_embeddings, torch.Tensor):
                return self._split_embeddings(vision_embeddings, split_sizes)
            return list(vision_embeddings)

        # Items the processor left un-padded arrive as a list once their
        # shapes differ. Encode them one at a time so that none of them is
        # padded to match another.
        embeddings: list[torch.Tensor] = []
        for index, item in enumerate(pixel_values):
            features = self._get_features(
                modality,
                item.unsqueeze(0),
                **self._select_item_kwargs(kwargs, index, len(pixel_values)),
            )
            # Encoders which return one entry per item return a single entry
            if not isinstance(features, torch.Tensor):
                features = torch.cat(list(features))
            embeddings.extend(self._split_embeddings(features, [split_sizes[index]]))
        return embeddings

    def _select_item_kwargs(
        self, kwargs: dict[str, Any], index: int, num_items: int
    ) -> dict[str, Any]:
        """Narrow the entries of `kwargs` that hold one row per item down to the item
        at `index`. Length is all there is to match on, so an unrelated entry of the
        same length is narrowed too."""
        return {
            key: value[index : index + 1]
            if isinstance(value, (torch.Tensor, list)) and len(value) == num_items
            else value
            for key, value in kwargs.items()
        }

    def _get_features(self, modality: str, pixel_values: torch.Tensor, **kwargs) -> Any:
        # grid_thw fields are registered keep_on_cpu; restore the on-device
        # placement that HF get_*_features implementations expect.
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and value.is_cpu:
                kwargs[key] = async_tensor_h2d(value, pixel_values.device)

        # The underlying HuggingFace `get_*_features` implementations
        # contain model-internal syncs (e.g. Idefics3 filters all-zero
        # padding images via boolean-mask indexing, LlavaOnevision
        # branches on per-sample batch counts).
        with gpu_sync_allowed():
            get_modality_features = getattr(self.model, f"get_{modality}_features")
            features = get_modality_features(pixel_values, return_dict=True, **kwargs)
        return features.pooler_output

    def embed_multimodal(self, **kwargs) -> MultiModalEmbeddings:
        # Each helper detects its own inputs. We are called once per modality, so the
        # leftovers a helper forwards to the HF model can't belong to the other one.
        embeddings: list[torch.Tensor] = []
        for process_input in (
            self._process_audio_input,
            self._process_image_input,
            self._process_video_input,
        ):
            embeddings.extend(process_input(**kwargs) or [])
        return embeddings

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        kwargs = MultiModalFeatureSpec.gather_kwargs(
            mm_features,
            {
                "image_grid_thw",
                "video_grid_thw",
                "second_per_grid_ts",
                "audio_feature_lengths",
                "use_audio_in_video",
            },
        )

        image_grid_thw = kwargs.get("image_grid_thw", [])
        video_grid_thw = kwargs.get("video_grid_thw", [])
        second_per_grid_ts = kwargs.get("second_per_grid_ts", [])
        audio_feature_lengths = kwargs.get("audio_feature_lengths", [])

        image_grid_thw = torch.stack(image_grid_thw) if image_grid_thw else None
        video_grid_thw = torch.stack(video_grid_thw) if video_grid_thw else None
        second_per_grid_ts = (
            torch.stack(second_per_grid_ts) if second_per_grid_ts else None
        )
        audio_feature_lengths = (
            torch.stack(audio_feature_lengths) if audio_feature_lengths else None
        )
        use_audio_in_video = any(kwargs.get("use_audio_in_video", []))

        # `get_rope_index` doesn't always accept arbitrary `kwargs`
        if not hasattr(self, "_get_rope_index_kwarg_names"):
            import inspect

            params = inspect.signature(self.model.get_rope_index).parameters
            self._get_rope_index_kwarg_names = (
                None
                if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
                else frozenset(params)
            )

        kwarg_names = self._get_rope_index_kwarg_names

        def accepts_kwarg(name: str) -> bool:
            return kwarg_names is None or name in kwarg_names

        seconds_name = (
            "second_per_grid_ts"
            if accepts_kwarg("second_per_grid_ts")
            else "second_per_grids"
        )
        audio_name = (
            "audio_feature_lengths"
            if accepts_kwarg("audio_feature_lengths")
            else "audio_seqlens"
        )

        # Drop a grid the model can't accept only when there is nothing to pass.
        kwargs = {
            name: value
            for name, value in (
                ("image_grid_thw", image_grid_thw),
                ("video_grid_thw", video_grid_thw),
                (seconds_name, second_per_grid_ts),
                (audio_name, audio_feature_lengths),
            )
            if value is not None or accepts_kwarg(name)
        }
        if accepts_kwarg("use_audio_in_video"):
            kwargs["use_audio_in_video"] = use_audio_in_video
        if accepts_kwarg("mm_token_type_ids"):
            mm_token_type_ids = torch.zeros(len(input_tokens), dtype=torch.int)
            for feature in mm_features:
                position = feature.mm_position
                offset, length = position.offset, position.length
                is_embed = position.is_embed
                if is_embed is None:
                    is_embed = slice(None)
                mm_token_type_id = _MODALITY_TO_TOKEN_TYPE_ID[feature.modality]
                mm_token_type_ids[offset : offset + length][is_embed] = mm_token_type_id
            kwargs["mm_token_type_ids"] = mm_token_type_ids.unsqueeze(0)

        mrope_positions, mrope_position_delta = self.model.get_rope_index(
            input_ids=torch.tensor(input_tokens).unsqueeze(0),
            **kwargs,
        )

        mrope_positions = mrope_positions[:, 0]
        mrope_position_delta = mrope_position_delta[0].item()

        return mrope_positions, mrope_position_delta
