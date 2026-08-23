# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM composition layer for Dots3Note image and audio encoders."""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    IntermediateTensors,
    WeightsMapper,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

from ..common.processor import (
    AUDIO_END,
    AUDIO_PAD,
    AUDIO_START,
    IMAGE_END,
    IMAGE_PAD,
    IMAGE_START,
    VIDEO_PLACEHOLDER,
    Dots3NoteDummyInputsBuilder,
    Dots3NoteMultiModalProcessor,
    Dots3NoteProcessingInfo,
    load_note_config_section,
)
from .audio import Dots3NoteAudioConfig, Dots3NoteAudioModel
from .model import Dots3NoteLanguageModelForCausalLM
from .vision import DotsMoEVitConfig, DotsMoEVitModel


@MULTIMODAL_REGISTRY.register_processor(
    Dots3NoteMultiModalProcessor,
    info=Dots3NoteProcessingInfo,
    dummy_inputs=Dots3NoteDummyInputsBuilder,
)
class Dots3NoteForCausalLM(nn.Module, SupportsMultiModal, SupportsPP):
    """Dots3Note model with optional image and audio towers."""

    supports_encoder_tp_data = True

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
            "mtp.": "language_model.mtp.",
            "vision_encoder.": "visual.",
            "audio_encoder.": "audio_tower.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return f"{IMAGE_START}{IMAGE_PAD}{IMAGE_END}"
        if modality.startswith("audio"):
            return f"{AUDIO_START}{AUDIO_PAD}{AUDIO_END}"
        if modality.startswith("video"):
            return VIDEO_PLACEHOLDER
        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        model_config = vllm_config.model_config
        self.config = model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.multimodal_config = model_config.multimodal_config
        assert self.multimodal_config is not None

        added_tokens = get_hf_file_to_dict(
            "added_tokens.json",
            model_config.model,
            model_config.revision,
        )
        if added_tokens is None or IMAGE_PAD not in added_tokens:
            raise ValueError("NOTE tokenizer is missing the image padding token")
        self.config.image_token_index = int(added_tokens[IMAGE_PAD])

        vision_config_dict = load_note_config_section(
            model_config.model,
            model_config.revision,
            "vision_config",
        )
        audio_config_dict = load_note_config_section(
            model_config.model,
            model_config.revision,
            "audio_config",
        )
        video_enabled = (
            vision_config_dict is not None
            and self.multimodal_config.get_limit_per_prompt("video") > 0
        )
        image_enabled = vision_config_dict is not None and (
            self.multimodal_config.get_limit_per_prompt("image") > 0 or video_enabled
        )
        audio_enabled = audio_config_dict is not None and (
            self.multimodal_config.get_limit_per_prompt("audio") > 0 or video_enabled
        )

        with self._mark_tower_model(vllm_config, {"image", "audio", "video"}):
            self.visual: DotsMoEVitModel | None
            if image_enabled:
                assert vision_config_dict is not None
                self.visual = DotsMoEVitModel(DotsMoEVitConfig(**vision_config_dict))
            else:
                self.visual = None
            self.audio_tower: Dots3NoteAudioModel | None
            if audio_enabled:
                assert audio_config_dict is not None
                self.audio_tower = Dots3NoteAudioModel(
                    Dots3NoteAudioConfig(**audio_config_dict)
                )
            else:
                self.audio_tower = None
            # The native encoder service casts each complete tower before
            # loading its checkpoint.  This also converts explicitly-created
            # floating buffers (for example RoPE tables and router state),
            # which merely constructing under vLLM's default dtype does not.
            if self.visual is not None:
                self.visual.to(dtype=model_config.dtype)
            if self.audio_tower is not None:
                self.audio_tower.to(dtype=model_config.dtype)
        with self._mark_language_model(vllm_config):
            self.language_model = Dots3NoteLanguageModelForCausalLM(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        orig_to_new_prefix = dict[str, None]()
        if self.visual is None:
            orig_to_new_prefix["visual."] = None
        if self.audio_tower is None:
            orig_to_new_prefix["audio_tower."] = None
        if orig_to_new_prefix:
            self.hf_to_vllm_mapper = self.hf_to_vllm_mapper | WeightsMapper(
                orig_to_new_prefix=orig_to_new_prefix
            )

    def _process_image_input(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if self.visual is None:
            return ()
        image_embeds = self.visual(pixel_values, image_grid_thw)
        merge_size = self.visual.spatial_merge_size
        sizes = (image_grid_thw.prod(-1) // merge_size**2).tolist()
        return image_embeds.split(sizes)

    def _process_audio_input(
        self,
        audio_values: torch.Tensor,
        audio_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if self.audio_tower is None:
            return ()
        if audio_values.dtype != torch.int32:
            raise TypeError(
                "NOTE audio values must carry float32 waveform bits as int32, "
                f"got {audio_values.dtype}"
            )
        waveforms = audio_values.contiguous().view(torch.float32)
        audio_embeds, item_lengths = self.audio_tower(waveforms, audio_lengths)
        return audio_embeds.split(item_lengths)

    def _process_video_input(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        audio_values: torch.Tensor,
        audio_lengths: torch.Tensor,
        modalities: torch.Tensor,
        frame_counts: torch.Tensor,
        audio_counts: torch.Tensor,
        emission_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        image_embeds = self._process_image_input(pixel_values, image_grid_thw)
        audio_embeds = (
            self._process_audio_input(audio_values, audio_lengths)
            if audio_lengths.numel()
            else ()
        )
        orders = modalities.tolist()
        frame_counts_list = frame_counts.tolist()
        audio_counts_list = audio_counts.tolist()
        emission_counts_list = emission_counts.tolist()
        outputs: list[torch.Tensor] = []
        image_idx = audio_idx = order_idx = 0
        for num_frames, num_audios, num_emissions in zip(
            frame_counts_list,
            audio_counts_list,
            emission_counts_list,
        ):
            video_order = orders[order_idx : order_idx + num_emissions]
            video_parts: list[torch.Tensor] = []
            video_image_start = image_idx
            video_audio_start = audio_idx
            for modality in video_order:
                if modality == 0:
                    video_parts.append(image_embeds[image_idx])
                    image_idx += 1
                elif modality == 1:
                    if audio_idx >= len(audio_embeds):
                        raise ValueError("NOTE video audio tower output is missing")
                    video_parts.append(audio_embeds[audio_idx])
                    audio_idx += 1
                else:
                    raise ValueError(f"Unknown NOTE video modality id: {modality}")
            if image_idx - video_image_start != num_frames:
                raise ValueError("NOTE video frame order/count mismatch")
            if audio_idx - video_audio_start != num_audios:
                raise ValueError("NOTE video audio order/count mismatch")
            outputs.append(torch.cat(video_parts))
            order_idx += num_emissions
        if image_idx != len(image_embeds) or audio_idx != len(audio_embeds):
            raise ValueError("NOTE video encoder outputs were not fully consumed")
        return tuple(outputs)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        multimodal_embeddings: list[torch.Tensor] = []
        handled: set[str] = set()
        for input_key in kwargs:
            if input_key == "pixel_values" and "image" not in handled:
                pixel_values = kwargs.get("pixel_values")
                image_grid_thw = kwargs.get("image_grid_thw")
                if isinstance(pixel_values, torch.Tensor) and isinstance(
                    image_grid_thw, torch.Tensor
                ):
                    multimodal_embeddings.extend(
                        self._process_image_input(pixel_values, image_grid_thw)
                    )
                handled.add("image")
            elif input_key == "audio_values" and "audio" not in handled:
                audio_values = kwargs.get("audio_values")
                audio_lengths = kwargs.get("audio_lengths")
                if isinstance(audio_values, torch.Tensor) and isinstance(
                    audio_lengths, torch.Tensor
                ):
                    multimodal_embeddings.extend(
                        self._process_audio_input(audio_values, audio_lengths)
                    )
                handled.add("audio")
            elif input_key == "video_pixel_values" and "video" not in handled:
                video_inputs = (
                    kwargs.get("video_pixel_values"),
                    kwargs.get("video_image_grid_thw"),
                    kwargs.get("video_audio_values"),
                    kwargs.get("video_audio_lengths"),
                    kwargs.get("video_modalities"),
                    kwargs.get("video_frame_counts"),
                    kwargs.get("video_audio_counts"),
                    kwargs.get("video_emission_counts"),
                )
                if all(isinstance(value, torch.Tensor) for value in video_inputs):
                    multimodal_embeddings.extend(
                        self._process_video_input(*video_inputs)  # type: ignore[arg-type]
                    )
                handled.add("video")
        return tuple(multimodal_embeddings)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        return self.language_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(
            weights,
            mapper=self.hf_to_vllm_mapper,
        )

    def process_weights_after_loading(self) -> None:
        if self.visual is not None:
            self.visual.process_weights_after_loading()

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            tower_model=["visual", "audio_tower"],
        )
