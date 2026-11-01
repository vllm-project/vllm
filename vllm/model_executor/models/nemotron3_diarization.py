# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wrapper around the Transformers implementation of NVIDIA Nemotron-3-Diarization.

The model is a streaming Sortformer speaker diarizer: for every 10 ms spectrogram
frame of a recording it returns the logits of up to 8 speakers, numbered in order
of first arrival. vLLM serves it offline, one whole recording per request, through
the `plugin` pooling task. The Transformers forward chunks the recording and keeps
the speaker cache internally, so no state is carried between requests.
"""

from collections.abc import Iterable, Mapping, Sequence

import torch
import torch.nn as nn
from transformers import BatchFeature, Nemotron3DiarizationConfig
from transformers import (
    Nemotron3DiarizationForAudioFrameClassification as HFNemotron3Diarization,
)

from vllm.config import VllmConfig
from vllm.config.multimodal import MultiModalDummyOptions
from vllm.inputs import MultiModalDataDict, MultiModalInput, mm_input
from vllm.model_executor.layers.pooler import IdentityPooler
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    PlaceholderRange,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    ProcessorInputs,
    PromptUpdate,
    TimingContext,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata

from .interfaces import IsAttentionFree, MultiModalEmbeddings, SupportsMultiModal
from .interfaces_base import attn_type

# Length of the dummy recording used for profiling. The model has no KV cache to
# size, so a short clip that exercises the forward is enough.
_DUMMY_AUDIO_SECONDS = 5


class Nemotron3DiarizationProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> Nemotron3DiarizationConfig:
        return self.ctx.get_hf_config(Nemotron3DiarizationConfig)

    def get_feature_extractor(self, **kwargs: object):
        return self.get_hf_processor(**kwargs).feature_extractor

    def get_data_parser(self):
        return MultiModalDataParser(
            target_sr=self.get_feature_extractor().sampling_rate,
            target_channels=1,
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}


class Nemotron3DiarizationDummyInputsBuilder(
    BaseDummyInputsBuilder[Nemotron3DiarizationProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: MultiModalDummyOptions,
    ) -> MultiModalDataDict:
        sampling_rate = self.info.get_feature_extractor().sampling_rate
        return {
            "audio": self._get_dummy_audios(
                length=_DUMMY_AUDIO_SECONDS * sampling_rate,
                num_audios=mm_counts.get("audio", 0),
                overrides=mm_options.get("audio"),
            )
        }


class Nemotron3DiarizationMultiModalProcessor(
    BaseMultiModalProcessor[Nemotron3DiarizationProcessingInfo]
):
    # The model consumes a spectrogram, not tokens.
    requires_tokenizer = False

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            input_features=MultiModalFieldConfig.batched("audio"),
            attention_mask=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        return []

    def apply(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ) -> MultiModalInput:
        mm_items = inputs.mm_data_items
        hf_processor_mm_kwargs = inputs.hf_processor_mm_kwargs

        with timing_ctx.record("apply_hf_processor"):
            audios = mm_items.get_items("audio", AudioProcessorItems)
            feature_extractor = self.info.get_feature_extractor(
                **hf_processor_mm_kwargs
            )
            # Same features as `Nemotron3DiarizationProcessor` in offline mode
            mm_processed_data = feature_extractor(
                [audios.get(i) for i in range(len(audios))],
                sampling_rate=feature_extractor.sampling_rate,
                return_attention_mask=True,
                return_tensors="pt",
            )

        mm_kwargs = MultiModalKwargsItems.from_hf_inputs(
            mm_processed_data,
            self._get_mm_fields_config(mm_processed_data, hf_processor_mm_kwargs),
        )

        with timing_ctx.record("get_mm_hashes"):
            mm_hashes = inputs.get_mm_hashes(
                self.info.model_id,
                self.info.ctx.get_mm_config().mm_hasher_algorithm,
            )

        return mm_input(
            prompt_token_ids=[0],
            mm_kwargs=mm_kwargs,
            mm_hashes=mm_hashes,
            mm_placeholders={"audio": [PlaceholderRange(offset=0, length=0)]},
        )


class Nemotron3DiarizationPooler(IdentityPooler):
    """Trims the padded logits of a step back to the frames of each recording."""

    def __init__(self) -> None:
        super().__init__()
        self.num_frames: list[int] | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        num_frames, self.num_frames = self.num_frames, None
        if num_frames is None:
            return hidden_states
        return [logits[:n] for logits, n in zip(hidden_states, num_frames)]


@attn_type("attention_free")
@MULTIMODAL_REGISTRY.register_processor(
    Nemotron3DiarizationMultiModalProcessor,
    info=Nemotron3DiarizationProcessingInfo,
    dummy_inputs=Nemotron3DiarizationDummyInputsBuilder,
)
class Nemotron3DiarizationForAudioFrameClassification(
    nn.Module, IsAttentionFree, SupportsMultiModal
):
    supports_multimodal_raw_input_only = True
    is_pooling_model = True

    # The Transformers model is wrapped as is, under `self.model`
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"": "model."})

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return None

        raise ValueError("Only audio modality is supported")

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: Nemotron3DiarizationConfig = vllm_config.model_config.hf_config
        self.model = HFNemotron3Diarization(config)
        self.num_speakers = config.head_config.num_speakers

        # Transformers computes the RoPE frequencies on the CPU. Computing them on
        # the GPU, where the model is built, shifts the logits by up to ~2e-3.
        rotary_emb = self.model.model.audio_tower.rotary_emb
        if rotary_emb.rope_type == "default":
            with torch.device("cpu"):
                inv_freq, _ = rotary_emb.compute_default_rope_parameters(
                    rotary_emb.config
                )
            rotary_emb.inv_freq.copy_(inv_freq)
            rotary_emb.original_inv_freq.copy_(inv_freq)

        self.pooler = Nemotron3DiarizationPooler()

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # The prompt is a single placeholder token that the model does not use.
        return torch.empty((input_ids.shape[0], 0))

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        input_features: torch.Tensor | list[torch.Tensor] | None = None,
        attention_mask: torch.Tensor | list[torch.Tensor] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if input_features is None:
            # No recording in this step: the output still needs a batch dimension
            self.pooler.num_frames = None
            return positions.new_zeros(
                (positions.shape[0], 0, self.num_speakers), dtype=self.model.dtype
            )
        assert attention_mask is not None

        logits = list[torch.Tensor]()
        # One recording per forward: the offline forward chunks it and updates the
        # speaker cache, which syncs with the host.
        with gpu_sync_allowed():
            for features, mask in zip(input_features, attention_mask):
                output = self.model(
                    input_features=features[None].to(self.model.dtype),
                    attention_mask=mask[None],
                )
                logits.append(output.logits[0])

        # The runner pools a single tensor, so recordings of different lengths
        # are padded here and trimmed back by the pooler.
        self.pooler.num_frames = [x.shape[0] for x in logits]
        return nn.utils.rnn.pad_sequence(logits, batch_first=True)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
