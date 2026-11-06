# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only LFM3-Omni model: images, videos and speech in, text out.

Prompts follow the Transformers processor: every image, video and clip is wrapped
in one span pair, a video is a series of timestamped frame groups, and with
`use_audio_in_video` each video takes the first clip after it as its soundtrack,
split across its frame groups. The speech output head is not served and its
weights are skipped on load.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import torch
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.config.multimodal import MultiModalDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems, VideoProcessorItems
from vllm.multimodal.processing import (
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.processing.processor import (
    MultiModalProcessingResult,
    PlaceholderFeaturesInfo,
    ResolvedPromptUpdate,
)
from vllm.transformers_utils.configs.lfm3_omni import Lfm3OmniConfig
from vllm.transformers_utils.processors.lfm3_omni import Lfm3OmniProcessor
from vllm.utils.gpu_sync_debug import gpu_sync_allowed

from .interfaces import MultiModalEmbeddings
from .lfm3_audio import (
    Lfm3AudioEncoder,
    Lfm3AudioForConditionalGeneration,
    Lfm3AudioProcessingInfo,
    Lfm3AudioProjector,
    split_audio_features,
    suppress_speech_mode,
)
from .lfm3_vl import (
    Lfm3VlDummyInputsBuilder,
    Lfm3VlForConditionalGeneration,
    Lfm3VlProcessingInfo,
)
from .qwen2_5_omni_thinker import merge_interleaved_embeddings
from .qwen3_vl import Qwen3VLMultiModalDataParser, Qwen3VLMultiModalProcessor
from .utils import maybe_prefix


class Lfm3OmniProcessingInfo(Lfm3AudioProcessingInfo, Lfm3VlProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Lfm3OmniConfig)

    def get_hf_processor(self, **kwargs: object) -> Lfm3OmniProcessor:
        return self.ctx.get_hf_processor(Lfm3OmniProcessor, **kwargs)

    def get_data_parser(self):
        return Qwen3VLMultiModalDataParser(
            self.get_hf_config().vision_config.spatial_merge_size,
            video_needs_metadata=True,
            target_sr=self.get_feature_extractor().sampling_rate,
            target_channels=1,
            expected_hidden_size=self._get_expected_hidden_size(),
            allow_missing_mm_embeddings=self.allow_missing_mm_embeddings,
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "video": None, "audio": None}

    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int] | None = None
    ) -> Mapping[str, int]:
        return {
            **Lfm3VlProcessingInfo.get_mm_max_tokens_per_item(
                self, seq_len, mm_counts or {}
            ),
            **super().get_mm_max_tokens_per_item(seq_len, mm_counts),
        }


class Lfm3OmniDummyInputsBuilder(Lfm3VlDummyInputsBuilder):
    info: Lfm3OmniProcessingInfo

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        audio = "<|vision_start|><|audio|><|vision_end|>"
        return super().get_dummy_text(mm_counts) + audio * mm_counts.get("audio", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: MultiModalDummyOptions,
    ) -> MultiModalDataDict:
        return {
            **super().get_dummy_mm_data(seq_len, mm_counts, mm_options),
            "audio": self._get_dummy_audios(
                length=self.info.get_max_audio_samples(),
                num_audios=mm_counts.get("audio", 0),
                overrides=mm_options.get("audio"),
            ),
        }


@dataclass
class _VideoGroups(PromptUpdateDetails):
    """A video placeholder that takes a soundtrack, kept as frame groups until
    the prompt shows which clip that is."""

    groups: list[list[int]] = field(default_factory=list)
    group_starts: list[float] = field(default_factory=list)


class Lfm3OmniMultiModalProcessor(Qwen3VLMultiModalProcessor):
    info: Lfm3OmniProcessingInfo

    def _get_mm_fields_config(
        self, hf_inputs: BatchFeature, hf_processor_mm_kwargs: Mapping[str, object]
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {
            **super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs),
            "input_features": MultiModalFieldConfig.batched("audio"),
            "audio_embeds": MultiModalFieldConfig.batched("audio"),
        }

    def _postprocess_hf_mm_data(
        self,
        mm_inputs: Mapping[str, object],
        hf_processor_mm_kwargs: Mapping[str, object],
        processed_data: BatchFeature,
    ) -> BatchFeature:
        return split_audio_features(processed_data)

    def _span_token_ids(self) -> tuple[int, int]:
        processor = self.info.get_hf_processor()
        return processor.vision_start_token_id, processor.vision_end_token_id

    def _postprocess_prompt(self, prompt: list[int]) -> list[int]:
        # Each placeholder gets exactly one span pair, so drop the pair a chat
        # template already put around it.
        processor = self.info.get_hf_processor()
        start, end = self._span_token_ids()
        placeholders = {
            processor.image_token_id,
            processor.video_token_id,
            processor.audio_token_id,
        }
        stripped: list[int] = []
        index = 0
        while index < len(prompt):
            if (
                prompt[index] == start
                and index + 2 < len(prompt)
                and prompt[index + 1] in placeholders
                and prompt[index + 2] == end
            ):
                stripped.append(prompt[index + 1])
                index += 3
            else:
                stripped.append(prompt[index])
                index += 1
        return stripped

    def _wrap(self, tokens: list[int]) -> list[int]:
        start, end = self._span_token_ids()
        return [start, *tokens, end]

    def _with_soundtrack(
        self, video: _VideoGroups, num_tokens: int
    ) -> PromptUpdateDetails:
        """Splits the soundtrack's encoder tokens at the first frame of each
        group, after the group."""
        processor = self.info.get_hf_processor()
        extractor = processor.feature_extractor
        seconds_per_token = (
            extractor.hop_length * extractor.encoder_stride / extractor.sampling_rate
        )
        bounds = [0] + [
            min(num_tokens, round(start / seconds_per_token))
            for start in video.group_starts[1:]
        ]
        bounds = [max(bounds[: i + 1]) for i in range(len(bounds))] + [num_tokens]
        tokens: list[int] = []
        for group, first, last in zip(video.groups, bounds, bounds[1:]):
            tokens += group + [processor.audio_token_id] * (last - first)
        return PromptUpdateDetails.select_token_id(
            self._wrap(tokens), processor.video_token_id
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        extractor = processor.feature_extractor
        start, end = self._span_token_ids()
        temporal_patch_size = processor.video_processor.temporal_patch_size
        merge_length = processor.image_processor.merge_size**2
        use_audio_in_video = bool(hf_processor_mm_kwargs.get("use_audio_in_video"))

        def wrapped(token_id: int, count: int) -> PromptUpdateDetails:
            return PromptUpdateDetails.select_token_id(
                self._wrap([token_id] * count), token_id
            )

        def image_replacement(item_idx: int) -> PromptUpdateDetails:
            grid_thw = out_mm_kwargs["image"][item_idx]["image_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)
            return wrapped(
                processor.image_token_id, int(grid_thw.prod()) // merge_length
            )

        def audio_replacement(item_idx: int) -> PromptUpdateDetails:
            item = out_mm_kwargs["audio"][item_idx]
            if (embeds := item.get("audio_embeds")) is not None:
                count = len(embeds.data)
            else:
                features = item["input_features"].data
                assert isinstance(features, torch.Tensor)
                count = extractor.num_encoder_tokens(features.shape[-1])
            return wrapped(processor.audio_token_id, count)

        def video_replacement(item_idx: int) -> PromptUpdateDetails:
            item = out_mm_kwargs["video"][item_idx]
            grid_thw = item["video_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)
            timestamps = item["timestamps"].data
            assert isinstance(timestamps, (list, torch.Tensor))
            tokens_per_group = int(grid_thw[1:].prod()) // merge_length
            groups = [
                self.get_video_repl(
                    tokens_per_frame=[tokens_per_group],
                    timestamps=[timestamp],
                    tokenizer=tokenizer,
                    vision_start_token_id=start,
                    vision_end_token_id=end,
                    video_token_id=processor.video_token_id,
                ).full
                for timestamp in timestamps
            ]
            details = PromptUpdateDetails.select_token_id(
                self._wrap([token for group in groups for token in group]),
                processor.video_token_id,
            )
            if not use_audio_in_video:
                return details
            # The soundtrack splits at the first frame of each group, which the
            # averaged timestamps lose, so take the frame indices the loader
            # sampled; frames resampled by the HF processor change the count.
            videos = mm_items.get("video")
            video = (
                videos.get(item_idx)
                if isinstance(videos, VideoProcessorItems)
                else None
            )
            metadata = video[1] if isinstance(video, tuple) else {}
            indices = metadata.get("frames_indices", [])
            if -(-len(indices) // temporal_patch_size) != len(groups):
                raise ValueError(
                    "`use_audio_in_video` needs videos sampled by the video loader"
                )
            return _VideoGroups(
                full=details.full,
                is_embed=details.is_embed,
                groups=groups,
                group_starts=[
                    index / metadata["fps"] for index in indices[::temporal_patch_size]
                ],
            )

        return [
            PromptReplacement("image", [processor.image_token_id], image_replacement),
            PromptReplacement("video", [processor.video_token_id], video_replacement),
            PromptReplacement("audio", [processor.audio_token_id], audio_replacement),
        ]

    def _soundtracks(self, prompt: list[int]) -> dict[int, int]:
        """The audio item each video takes as its soundtrack: the first clip
        after the video that no earlier video took."""
        processor = self.info.get_hf_processor()
        clips = [i for i, t in enumerate(prompt) if t == processor.audio_token_id]
        taken: list[int] = []
        for video in (i for i, t in enumerate(prompt) if t == processor.video_token_id):
            clip_idx = next(
                (i for i, clip in enumerate(clips) if clip > video and i not in taken),
                None,
            )
            if clip_idx is None:
                raise ValueError(
                    "`use_audio_in_video` needs an audio clip after each video"
                )
            taken.append(clip_idx)
        return dict(enumerate(taken))

    def _maybe_apply_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        mm_res: MultiModalProcessingResult,
    ) -> tuple[list[int], Mapping[str, list[PlaceholderFeaturesInfo]]]:
        if not any(
            isinstance(update.content, _VideoGroups)
            for item in mm_res.prompt_updates.get("video", [])
            for update in item
        ):
            return super()._maybe_apply_prompt_updates(mm_items, mm_res)

        mm_item_counts = mm_items.get_all_counts()
        self._validate_mm_kwargs(mm_res.kwargs, mm_item_counts)
        self._validate_mm_updates(mm_res.prompt_updates, mm_item_counts)

        # A soundtrack's own placeholder expands to nothing; its tokens follow
        # its video's frame groups instead.
        soundtracks = self._soundtracks(mm_res.prompt_ids)
        updates: dict[str, list[list[ResolvedPromptUpdate]]] = {
            modality: [list(item) for item in items]
            for modality, items in mm_res.prompt_updates.items()
        }
        audio_token_id = self.info.get_hf_processor().audio_token_id
        for video_idx, audio_idx in soundtracks.items():
            (video,) = updates["video"][video_idx]
            (audio,) = updates["audio"][audio_idx]
            assert isinstance(video.content, _VideoGroups)
            content = self._with_soundtrack(
                video.content, audio.content.full.count(audio_token_id)
            )
            updates["video"][video_idx] = [replace(video, content=content)]
            updates["audio"][audio_idx] = [
                replace(audio, content=PromptUpdateDetails.from_seq([]))
            ]

        prompt_ids, placeholders = self._apply_prompt_updates(
            mm_res.prompt_ids, updates
        )
        audio_placeholders = list(placeholders.get("audio", []))
        for video_placeholder in placeholders["video"]:
            audio_idx = soundtracks[video_placeholder.item_idx]
            audio_placeholders.append(
                replace(
                    video_placeholder,
                    modality="audio",
                    item_idx=audio_idx,
                    is_embed=torch.tensor(video_placeholder.tokens) == audio_token_id,
                )
            )
        placeholders = {
            **placeholders,
            "audio": sorted(audio_placeholders, key=lambda info: info.item_idx),
        }
        self._validate_mm_placeholders(placeholders, mm_item_counts)
        return prompt_ids, placeholders


@MULTIMODAL_REGISTRY.register_processor(
    Lfm3OmniMultiModalProcessor,
    info=Lfm3OmniProcessingInfo,
    dummy_inputs=Lfm3OmniDummyInputsBuilder,
)
class Lfm3OmniForConditionalGeneration(Lfm3VlForConditionalGeneration):
    hf_to_vllm_mapper = (
        Lfm3VlForConditionalGeneration.hf_to_vllm_mapper
        | Lfm3AudioForConditionalGeneration.hf_to_vllm_mapper
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return "<|vision_start|><|audio|><|vision_end|>"
        return super().get_placeholder_str(modality, i)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config: Lfm3OmniConfig = vllm_config.model_config.hf_config
        with self._mark_tower_model(vllm_config, "audio"):
            self.audio_encoder = Lfm3AudioEncoder(config.audio_encoder_config)
            self.audio_projector = Lfm3AudioProjector(
                config, prefix=maybe_prefix(prefix, "audio_projector")
            )

    _parse_and_validate_audio_input = (
        Lfm3AudioForConditionalGeneration._parse_and_validate_audio_input
    )

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return super().embed_multimodal(**kwargs)
        if audio_input["type"] == "audio_embeds":
            return list(audio_input["audio_embeds"])
        dtype = self.audio_encoder.downsample.conv_in.weight.dtype
        return [
            self.audio_projector(self.audio_encoder(features[None].to(dtype))[0])
            for features in audio_input["input_features"]
        ]

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return suppress_speech_mode(self.config, super().compute_logits(hidden_states))

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if multimodal_embeddings and is_multimodal is not None:
            with gpu_sync_allowed():
                input_ids_cpu = input_ids.cpu()
                is_video = is_multimodal & (input_ids_cpu == self.config.video_token_id)
                is_audio = is_multimodal & (input_ids_cpu == self.config.audio_token_id)
                video_and_audio = bool(is_video.any() and is_audio.any())
            if video_and_audio:
                # A soundtrack's tokens sit inside its video's span, so the
                # embeddings are not in position order: scatter each modality
                # over its own tokens.
                inputs_embeds = self._embed_text_input_ids(
                    input_ids,
                    self.get_language_model().embed_input_ids,
                    is_multimodal=is_multimodal,
                )
                return merge_interleaved_embeddings(
                    inputs_embeds,
                    multimodal_embeddings,
                    is_video,
                    is_audio,
                    is_multimodal,
                )
        return super().embed_input_ids(
            input_ids, multimodal_embeddings, is_multimodal=is_multimodal
        )
