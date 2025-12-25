# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, Literal, TypeAlias

import torch
import torch.nn as nn
from transformers import BatchFeature
from transformers.models.glmasr import GlmAsrConfig, GlmAsrEncoder, GlmAsrProcessor
from transformers.models.whisper import WhisperFeatureExtractor

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    DictEmbeddingItems,
    ModalityData,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from .utils import AutoWeightsLoader, init_vllm_registered_model, maybe_prefix

DEFAULT_MAX_AUDIO_LEN_S = 655
DEFAULT_MERGE_FACTOR = 4


class GlmAsrFeatureInputs(TensorSchema):
    """
    Dimensions:
        - num_chunks: Number of audio chunks (flattened)
        - nmb: Number of mel bins
        - num_audios: Number of original audio files
    """

    type: Literal["audio_features"]
    input_features: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("num_chunks", "nmb", 3000),
    ]
    feature_attention_mask: Annotated[
        torch.Tensor,
        TensorShape("num_chunks", 3000),
    ]
    chunk_counts: Annotated[
        torch.Tensor,
        TensorShape("num_audios"),
    ]


class GlmAsrEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size
        - naf: Number of audio features
        - hs: Hidden size (must match the hidden size of language model
          backbone)
    """

    type: Literal["audio_embeds"] = "audio_embeds"
    audio_embeds: Annotated[
        list[torch.Tensor],
        TensorShape("bn", "naf", "hs", dynamic_dims={"naf"}),
    ]


GlmAsrInputs: TypeAlias = GlmAsrFeatureInputs | GlmAsrEmbeddingInputs


class GlmAsrMultiModalProjector(nn.Module):
    def __init__(self, config: GlmAsrConfig):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.audio_config.intermediate_size,
            config.text_config.hidden_size * 2,
        )
        self.act = get_act_fn(config.projector_hidden_act)
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size * 2,
            config.text_config.hidden_size,
        )

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(audio_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class GlmAsrProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> GlmAsrConfig:
        return self.ctx.get_hf_config(GlmAsrConfig)

    def get_hf_processor(self, **kwargs: object) -> GlmAsrProcessor:
        return self.ctx.get_hf_processor(GlmAsrProcessor, **kwargs)

    def get_feature_extractor(self, **kwargs: object) -> WhisperFeatureExtractor:
        hf_processor = self.get_hf_processor(**kwargs)
        feature_extractor = hf_processor.feature_extractor
        assert isinstance(feature_extractor, WhisperFeatureExtractor)
        return feature_extractor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}


class GlmAsrDummyInputsBuilder(BaseDummyInputsBuilder[GlmAsrProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        hf_processor = self.info.get_hf_processor()
        return hf_processor.audio_token * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        feature_extractor = self.info.get_feature_extractor()
        sampling_rate = feature_extractor.sampling_rate
        num_audios = mm_counts.get("audio", 0)
        audio_overrides = mm_options.get("audio") if mm_options else None

        max_audio_len = getattr(
            self.info.get_hf_processor(), "max_audio_len", DEFAULT_MAX_AUDIO_LEN_S
        )
        audio_len = int(max_audio_len * sampling_rate)

        return {
            "audio": self._get_dummy_audios(
                length=audio_len, num_audios=num_audios, overrides=audio_overrides
            )
        }


def _glmasr_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    chunk_counts = hf_inputs.get("chunk_counts")
    if chunk_counts is not None:
        return dict(
            audio_embeds=MultiModalFieldConfig.batched("audio"),
            input_features=MultiModalFieldConfig.flat_from_sizes(
                "audio", chunk_counts, dim=0
            ),
            feature_attention_mask=MultiModalFieldConfig.flat_from_sizes(
                "audio", chunk_counts, dim=0
            ),
            chunk_counts=MultiModalFieldConfig.batched("audio"),
        )
    return dict(
        audio_embeds=MultiModalFieldConfig.batched("audio"),
        input_features=MultiModalFieldConfig.batched("audio"),
        feature_attention_mask=MultiModalFieldConfig.batched("audio"),
        chunk_counts=MultiModalFieldConfig.batched("audio"),
    )


class GlmAsrMultiModalDataParser(MultiModalDataParser):
    def _parse_audio_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[Any],
    ) -> ModalityDataItems[Any, Any] | None:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="audio",
                required_fields={"audio_embeds"},
                fields_factory=_glmasr_field_config,
            )
        return super()._parse_audio_data(data)


class GlmAsrMultiModalProcessor(BaseMultiModalProcessor[GlmAsrProcessingInfo]):
    def _get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.info.get_feature_extractor()
        return GlmAsrMultiModalDataParser(target_sr=feature_extractor.sampling_rate)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: dict[str, object],
        mm_kwargs: Mapping[str, Any],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        audios = mm_data.pop("audios", [])
        if audios:
            mm_data["audio"] = audios

        if not mm_data.get("audio", []):
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        processor = self.info.get_hf_processor(**mm_kwargs)
        feature_extractor = processor.feature_extractor
        mm_kwargs = dict(**mm_kwargs, sampling_rate=feature_extractor.sampling_rate)

        audio_list = mm_data.get("audio")
        if not isinstance(audio_list, list):
            audio_list = [audio_list]

        chunk_counts = []
        sampling_rate = feature_extractor.sampling_rate
        chunk_length = feature_extractor.chunk_length
        max_audio_len = getattr(processor, "max_audio_len", DEFAULT_MAX_AUDIO_LEN_S)
        window_size = int(sampling_rate * chunk_length)
        max_windows = int(max_audio_len // chunk_length)

        for audio in audio_list:
            n_samples = len(audio) if isinstance(audio, list) else audio.shape[0]
            n_win = max(1, (n_samples + window_size - 1) // window_size)
            if n_win > max_windows:
                n_win = max_windows
            chunk_counts.append(n_win)

        outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

        if "input_features_mask" in outputs:
            outputs["feature_attention_mask"] = outputs.pop("input_features_mask")

        outputs["chunk_counts"] = torch.tensor(chunk_counts, dtype=torch.long)
        return outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _glmasr_field_config(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()
        config = self.info.get_hf_config()

        audio_token = getattr(processor, "audio_token", "<|pad|>")
        audio_token_id = vocab.get(audio_token)
        if audio_token_id is None:
            audio_token_id = processor.audio_token_id

        # Get merge_factor from config, fallback to default
        merge_factor = getattr(config, "merge_factor", DEFAULT_MERGE_FACTOR)

        out_mm_data = out_mm_kwargs.get_data()
        feature_attention_mask = out_mm_data.get("feature_attention_mask")
        chunk_counts = out_mm_data.get("chunk_counts")

        def get_replacement_glmasr(item_idx: int):
            if feature_attention_mask is not None:
                if chunk_counts is not None:
                    counts = (
                        chunk_counts.tolist()
                        if isinstance(chunk_counts, torch.Tensor)
                        else chunk_counts
                    )
                    start_idx = sum(counts[:item_idx])
                    count = counts[item_idx]
                    end_idx = start_idx + count

                    if isinstance(feature_attention_mask, list):
                        mask_list = feature_attention_mask[start_idx:end_idx]
                        if len(mask_list) > 0 and isinstance(
                            mask_list[0], torch.Tensor
                        ):
                            mask = torch.stack(mask_list)
                        else:
                            mask = torch.tensor(mask_list)
                    else:
                        mask = feature_attention_mask[start_idx:end_idx]
                else:
                    if isinstance(feature_attention_mask, list):
                        mask = feature_attention_mask[item_idx]
                    else:
                        mask = feature_attention_mask[item_idx].unsqueeze(0)

                audio_lengths = mask.sum(-1)
                for padding, kernel_size, stride in [(1, 3, 1), (1, 3, 2)]:
                    audio_lengths = (
                        audio_lengths + 2 * padding - (kernel_size - 1) - 1
                    ) // stride + 1
                audio_output_lengths = (
                    audio_lengths - merge_factor
                ) // merge_factor + 1
                num_features = audio_output_lengths.sum().item()
            else:
                audio_embeds = out_mm_data["audio_embeds"][item_idx]
                num_features = audio_embeds.shape[0]

            if num_features == 0:
                raise ValueError("Audio is too short")

            audio_tokens = [audio_token_id] * int(num_features)
            return PromptUpdateDetails.select_token_id(
                audio_tokens,
                embed_token_id=audio_token_id,
            )

        return [
            PromptReplacement(
                modality="audio",
                target=audio_token,
                replacement=get_replacement_glmasr,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    GlmAsrMultiModalProcessor,
    info=GlmAsrProcessingInfo,
    dummy_inputs=GlmAsrDummyInputsBuilder,
)
class GlmAsrForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config

        self.audio_tower = GlmAsrEncoder(config.audio_config)
        self.multi_modal_projector = GlmAsrMultiModalProjector(config)
        self.quant_config = quant_config

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["LlamaForCausalLM"],
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return "<|begin_of_audio|><|pad|><|end_of_audio|>"

        raise ValueError("Only audio modality is supported")

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model.",
            connector="multi_modal_projector.",
            tower_model="audio_tower.",
        )

    def _parse_and_validate_audio_input(self, **kwargs: object) -> GlmAsrInputs | None:
        input_features = kwargs.pop("input_features", None)
        audio_embeds = kwargs.pop("audio_embeds", None)
        feature_attention_mask = kwargs.pop("feature_attention_mask", None)
        chunk_counts = kwargs.pop("chunk_counts", None)

        if input_features is None and audio_embeds is None:
            return None

        if audio_embeds is not None:
            return GlmAsrEmbeddingInputs(type="audio_embeds", audio_embeds=audio_embeds)

        if input_features is not None:
            return GlmAsrFeatureInputs(
                type="audio_features",
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                chunk_counts=chunk_counts,
            )

        raise AssertionError("This line should be unreachable.")

    def _process_audio_input(
        self, audio_input: GlmAsrInputs
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if audio_input["type"] == "audio_embeds":
            audio_embeds = audio_input["audio_embeds"]
            return tuple(audio_embeds)

        input_features = audio_input["input_features"]
        feature_attention_mask = audio_input["feature_attention_mask"]
        chunk_counts = audio_input.get("chunk_counts")

        if isinstance(input_features, list):
            input_features = torch.cat(input_features, dim=0)
            feature_attention_mask = torch.cat(feature_attention_mask, dim=0)

        if chunk_counts is None:
            chunk_counts = [1] * input_features.shape[0]
        elif isinstance(chunk_counts, torch.Tensor):
            chunk_counts = chunk_counts.tolist()
        elif (
            isinstance(chunk_counts, list)
            and chunk_counts
            and isinstance(chunk_counts[0], torch.Tensor)
        ):
            chunk_counts = [c.item() for c in chunk_counts]

        audio_outputs = self.audio_tower(input_features)
        audio_hidden_states = audio_outputs.last_hidden_state
        audio_hidden_states = audio_hidden_states.reshape(
            input_features.shape[0],
            -1,
            self.config.audio_config.intermediate_size,
        )
        audio_features = self.multi_modal_projector(audio_hidden_states)

        # Get merge_factor from config, fallback to default
        merge_factor = getattr(self.config, "merge_factor", DEFAULT_MERGE_FACTOR)

        audio_lengths = feature_attention_mask.sum(-1)
        for padding, kernel_size, stride in [(1, 3, 1), (1, 3, 2)]:
            audio_lengths = (
                audio_lengths + 2 * padding - (kernel_size - 1) - 1
            ) // stride + 1
        audio_output_lengths = (audio_lengths - merge_factor) // merge_factor + 1

        num_chunks, max_audio_tokens, embed_dim = audio_features.shape
        audio_output_lengths = audio_output_lengths.unsqueeze(1)
        audio_features_mask = (
            torch.arange(max_audio_tokens)
            .expand(num_chunks, max_audio_tokens)
            .to(audio_output_lengths.device)
            < audio_output_lengths
        )
        masked_audio_features = audio_features[audio_features_mask].view(-1, embed_dim)

        chunk_embeddings = torch.split(
            masked_audio_features, audio_output_lengths.flatten().tolist()
        )

        grouped_embeddings = []
        current_idx = 0
        for count in chunk_counts:
            audio_chunks = chunk_embeddings[current_idx : current_idx + count]
            grouped_embeddings.append(torch.cat(audio_chunks, dim=0))
            current_idx += count
        return tuple(grouped_embeddings)

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return []
        masked_audio_features = self._process_audio_input(audio_input)
        return masked_audio_features

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
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_prefixes = ["audio_tower.embed_positions"]
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights)
