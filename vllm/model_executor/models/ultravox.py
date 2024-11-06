# Adapted from https://github.com/fixie-ai/ultravox/blob/ecd58c4041030bae2ad15aa6bcf04ab43199ea02/ultravox/model/ultravox_model.py
"""PyTorch Ultravox model."""

import math
from functools import cached_property, lru_cache
from typing import (Iterable, List, Literal, Mapping, Optional, Tuple,
                    TypedDict, Union, cast)

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from transformers.models.whisper import WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, MultiModalConfig
from vllm.inputs import (INPUT_REGISTRY, DecoderOnlyInputs, DummyData,
                         InputContext, token_inputs)
from vllm.model_executor.layers.activation import SiluAndMul, get_act_fn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.model_loader.loader import DefaultModelLoader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import (MULTIMODAL_REGISTRY, MultiModalInputs,
                             NestedTensors)
from vllm.multimodal.utils import (cached_get_tokenizer,
                                   consecutive_placeholder_ranges,
                                   repeat_and_pad_placeholder_tokens)
from vllm.sequence import IntermediateTensors, SequenceData
from vllm.transformers_utils.configs.ultravox import UltravoxConfig
from vllm.utils import is_list_of

from .interfaces import SupportsMultiModal, SupportsPP
from .utils import (AutoWeightsLoader, WeightsMapper, flatten_bn,
                    init_vllm_registered_model,
                    merge_multimodal_embeddings_from_map)

_AUDIO_PLACEHOLDER_TOKEN = 128002
_AUDIO_TOKENS_PER_SECOND = 6.25


class UltravoxAudioFeatureInputs(TypedDict):
    type: Literal["audio_features"]
    data: NestedTensors
    """Shape: `(batch_size, num_audios, 80, M)`"""


class UltravoxAudioEmbeddingInputs(TypedDict):
    type: Literal["audio_embeds"]
    data: NestedTensors
    """Shape: `(batch_size, num_audios, audio_feature_size, hidden_size)`"""


UltravoxAudioInputs = Union[UltravoxAudioFeatureInputs,
                            UltravoxAudioEmbeddingInputs]


@lru_cache
def cached_feature_extractor(model_id: str) -> WhisperFeatureExtractor:
    return WhisperFeatureExtractor.from_pretrained(model_id)


def whisper_feature_extractor(ctx: InputContext) -> WhisperFeatureExtractor:
    return cached_feature_extractor(
        ctx.get_hf_config(UltravoxConfig).audio_model_id)


def get_ultravox_max_audio_tokens(ctx: InputContext):
    feature_extractor = whisper_feature_extractor(ctx)
    return math.ceil(feature_extractor.chunk_length * _AUDIO_TOKENS_PER_SECOND)


def dummy_seq_data_for_ultravox(
    ctx: InputContext,
    seq_len: int,
    audio_count: int,
):
    audio_length = min(get_ultravox_max_audio_tokens(ctx),
                       seq_len // audio_count)

    return SequenceData.from_prompt_token_counts(
        (_AUDIO_PLACEHOLDER_TOKEN, audio_length * audio_count),
        (0, seq_len - audio_length * audio_count)), {
            "audio":
            consecutive_placeholder_ranges(num_items=audio_count,
                                           item_size=audio_length)
        }


def dummy_audio_for_ultravox(
    ctx: InputContext,
    audio_count: int,
):
    feature_extractor = whisper_feature_extractor(ctx)
    audio_and_sr = (np.array([0.0] * feature_extractor.chunk_length), 1)
    return {"audio": [audio_and_sr] * audio_count}


def dummy_data_for_ultravox(
    ctx: InputContext,
    seq_len: int,
    mm_counts: Mapping[str, int],
):
    audio_count = mm_counts["audio"]
    seq_data, ranges = dummy_seq_data_for_ultravox(ctx, seq_len, audio_count)
    mm_dict = dummy_audio_for_ultravox(ctx, audio_count)

    return DummyData(seq_data, mm_dict, ranges)


def input_mapper_for_ultravox(ctx: InputContext, data: object):
    if not isinstance(data, list):
        data = [data]

    if len(data) == 0:
        return MultiModalInputs()

    # If the audio inputs are embeddings, no need for preprocessing
    if is_list_of(data, torch.Tensor, check="all"):
        return MultiModalInputs({"audio_embeds": data})

    audio_features = []
    for audio_input in data:
        if not isinstance(audio_input, tuple):
            raise NotImplementedError(
                f"Unsupported data type: {type(audio_input)}")

        (audio, sr) = cast(Tuple[np.ndarray, Union[float, int]], audio_input)
        feature_extractor = whisper_feature_extractor(ctx)

        if sr != feature_extractor.sampling_rate:
            try:
                import librosa
            except ImportError:
                raise ImportError(
                    "Please install vllm[audio] for audio support.") from None
            audio = librosa.resample(audio,
                                     orig_sr=sr,
                                     target_sr=feature_extractor.sampling_rate)
            sr = feature_extractor.sampling_rate

        minimum_audio_length = feature_extractor.n_fft // 2 + 1
        if len(audio) < minimum_audio_length:
            # Not enough audio; pad it.
            audio = np.pad(audio, (0, minimum_audio_length - len(audio)))

        single_audio_features = feature_extractor(
            audio, sampling_rate=sr, padding="longest",
            return_tensors="pt")["input_features"]

        # Remove the batch dimension because we're wrapping it in a list.
        audio_features.append(single_audio_features.squeeze(0))

    return MultiModalInputs({"audio_features": audio_features})


def input_processor_for_ultravox(ctx: InputContext, inputs: DecoderOnlyInputs):
    multi_modal_data = inputs.get("multi_modal_data")
    if multi_modal_data is None or "audio" not in multi_modal_data:
        return inputs

    if "multi_modal_placeholders" in inputs and "audio" in inputs[
            "multi_modal_placeholders"]:
        # The inputs already have placeholders.
        return inputs

    feature_extractor = whisper_feature_extractor(ctx)
    audios = multi_modal_data["audio"]
    if not isinstance(audios, list):
        audios = [audios]

    audio_token_counts = []
    for audio in audios:
        if isinstance(audio, torch.Tensor):
            audio_num_tokens = audio.shape[1]
            audio_token_counts.append(audio_num_tokens)
        else:
            audio_data, sample_rate = audio
            audio_length = audio_data.shape[0]
            if sample_rate != feature_extractor.sampling_rate:
                # Account for resampling.
                adjustment = feature_extractor.sampling_rate / sample_rate
                audio_length = math.ceil(adjustment * audio_length)

            feature_extractor_output_length = math.ceil(
                (audio_length - (feature_extractor.hop_length - 1)) /
                feature_extractor.hop_length)

            uv_config = ctx.get_hf_config(UltravoxConfig)
            audio_num_tokens = min(
                max(
                    1,
                    math.ceil(feature_extractor_output_length /
                              (uv_config.stack_factor * 2))),
                get_ultravox_max_audio_tokens(ctx))
            audio_token_counts.append(audio_num_tokens)

    tokenizer = cached_get_tokenizer(ctx.model_config.tokenizer)

    new_prompt, new_token_ids, ranges = repeat_and_pad_placeholder_tokens(
        tokenizer,
        inputs.get("prompt"),
        inputs["prompt_token_ids"],
        placeholder_token_id=_AUDIO_PLACEHOLDER_TOKEN,
        repeat_count=audio_token_counts,
    )

    # NOTE: Create a defensive copy of the original inputs
    return token_inputs(prompt_token_ids=new_token_ids,
                        prompt=new_prompt,
                        multi_modal_data=multi_modal_data,
                        multi_modal_placeholders={"audio": ranges})


class StackAudioFrames(nn.Module):
    """
    Stack the audio embedding frames to reduce the sequence length by a factor
    of `stack_factor`.
    """

    def __init__(self, stack_factor: int = 8):
        super().__init__()
        self.stack_factor = stack_factor

    def forward(self, audio_embeds: torch.Tensor) -> torch.Tensor:
        B, T, C = audio_embeds.shape
        T_pad = (T + self.stack_factor -
                 1) // self.stack_factor * self.stack_factor
        audio_embeds = F.pad(audio_embeds, (0, 0, 0, T_pad - T))
        B, T, C = audio_embeds.shape
        audio_embeds = audio_embeds.view(B, T // self.stack_factor,
                                         C * self.stack_factor)
        return audio_embeds


class FlippedSiluAndMul(SiluAndMul):
    """Ultravox is trained with SwiGLU with flipped halves."""

    def forward(self, x: torch.Tensor):
        a, b = x.chunk(2, dim=-1)
        flipped = torch.cat((b, a), dim=-1)
        return super().forward(flipped)


class UltravoxProjector(nn.Module):

    def __init__(self, config: UltravoxConfig):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self._pad_and_stack = StackAudioFrames(config.stack_factor)
        dim = config.audio_config.hidden_size * config.stack_factor
        self.ln_pre = RMSNorm(dim)
        self.linear_1 = nn.Linear(dim, self.hidden_dim, bias=False)
        dim = self.hidden_dim

        if config.projector_act == "swiglu":
            self.act = FlippedSiluAndMul()
            dim = dim // 2
        else:
            self.act = get_act_fn(config.projector_act)

        self.linear_2 = nn.Linear(dim,
                                  config.text_config.hidden_size,
                                  bias=False)
        self.ln_post = RMSNorm(config.text_config.hidden_size)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        audio_features = self._pad_and_stack(audio_features)
        audio_features = self.ln_pre(audio_features)
        hidden_states = self.linear_1(audio_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.ln_post(hidden_states)
        return hidden_states


class ModifiedWhisperEncoder(WhisperEncoder):
    """
    Encoder portion of OpenAI's Whisper model.

    This implementation is a slightly modified version of HF Transformers'
    Whisper Encoder, with only a few fixes:
    1. base_model_prefix updated to allow for doing `.from_pretrained`
       directly on the encoder
    2. allow less than 30 second of audio padding to be passed in:
        - relaxed ValueError check for `input_features` length to be less
           than or equal to `expected_seq_length` instead of strictly equal
        - embed_pos is now sliced to match the length of `inputs_embeds`

    Original: https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py
    See commentary: https://github.com/huggingface/transformers/issues/25744
    """

    base_model_prefix = "model.encoder"

    def forward(
        self,
        input_features,
    ):
        expected_seq_length = (self.config.max_source_positions *
                               self.conv1.stride[0] * self.conv2.stride[0])
        if input_features.shape[-1] > expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length "
                f"{expected_seq_length} or less, but found "
                f"{input_features.shape[-1]}. Make sure to pad the input mel "
                f"features to {expected_seq_length}.")

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight[:inputs_embeds.size(-2)]

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states,
                                              p=self.dropout,
                                              training=self.training)

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                None,
                layer_head_mask=None,
            )

            hidden_states = layer_outputs[0]

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


@MULTIMODAL_REGISTRY.register_input_mapper("audio", input_mapper_for_ultravox)
@MULTIMODAL_REGISTRY.register_max_multimodal_tokens(
    "audio", get_ultravox_max_audio_tokens)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_ultravox)
@INPUT_REGISTRY.register_input_processor(input_processor_for_ultravox)
class UltravoxModel(nn.Module, SupportsMultiModal, SupportsPP):

    def __init__(self,
                 config: UltravoxConfig,
                 multimodal_config: MultiModalConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional["QuantizationConfig"] = None):
        super().__init__()
        self.config = config
        self.multi_modal_config = multimodal_config
        assert self.multi_modal_config

        self.secondary_weights = []
        self.audio_tower = ModifiedWhisperEncoder(config.audio_config)
        if config.audio_model_id is not None:
            self.secondary_weights.append(
                DefaultModelLoader.Source(
                    model_or_path=config.audio_model_id,
                    revision=None,
                    prefix="audio_tower.",
                ))
        self.multi_modal_projector = UltravoxProjector(config)
        self.language_model = init_vllm_registered_model(
            config.text_config,
            cache_config,
            quant_config,
            prefix="language_model")
        if config.text_model_id is not None:
            self.secondary_weights.append(
                DefaultModelLoader.Source(model_or_path=config.text_model_id,
                                          revision=None,
                                          prefix="language_model."))

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    @cached_property
    def sampler(self):
        if hasattr(self.language_model, "sampler"):
            return self.language_model.sampler

        return Sampler()

    def _audio_features_to_embeddings(
            self, input_features: torch.Tensor) -> torch.Tensor:
        audio_input = input_features.to(self.audio_tower.dtype)
        audio_features = self.audio_tower(audio_input)
        audio_features = audio_features.to(self.audio_tower.dtype)
        audio_embeddings = self.multi_modal_projector(audio_features)
        return audio_embeddings

    def _parse_and_validate_audio_input(
            self, **kwargs: object) -> Optional[UltravoxAudioInputs]:
        audio_features = kwargs.pop("audio_features", None)
        audio_embeds = kwargs.pop("audio_embeds", None)

        if audio_features is None and audio_embeds is None:
            return None

        if audio_features is not None:
            if not isinstance(audio_features, (torch.Tensor, list)):
                raise ValueError("Incorrect type of audio features. "
                                 f"Got type: {type(audio_features)}")

            return UltravoxAudioFeatureInputs(type="audio_features",
                                              data=audio_features)

        if audio_embeds is not None:
            if not isinstance(audio_embeds, (torch.Tensor, list)):
                raise ValueError("Incorrect type of audio embeds. "
                                 f"Got type: {type(audio_embeds)}")

            return UltravoxAudioEmbeddingInputs(type="audio_embeds",
                                                data=audio_embeds)

        raise AssertionError("This line should be unreachable.")

    def _process_audio_input(
            self, audio_input: UltravoxAudioInputs) -> NestedTensors:
        if audio_input["type"] == "audio_embeds":
            return audio_input["data"]

        audio_features = audio_input["data"]
        if isinstance(audio_features, torch.Tensor):
            # Combine the B and N dimensions for the encoder/projector
            flattened = flatten_bn(audio_features)
            flattened_embeddings = self._audio_features_to_embeddings(
                flattened)

            # Restore the original dimensions
            embeddings = flattened_embeddings.unflatten(
                0, audio_features.shape[:2])
            return embeddings

        result = []
        # TODO: Batch heterogeneous tensors through the encoder/projector
        for audio_features_item in audio_features:
            if isinstance(audio_features_item, torch.Tensor):
                result.append(
                    self._audio_features_to_embeddings(audio_features_item))
            else:
                embeddings = [
                    # Add a batch dimension to embed it, then remove it.
                    self._audio_features_to_embeddings(tensor.unsqueeze(0)
                                                       ).squeeze(0)
                    for tensor in audio_features_item
                ]
                result.append(embeddings)

        return result

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                kv_caches: List[torch.Tensor],
                attn_metadata: AttentionMetadata,
                intermediate_tensors: Optional[torch.Tensor],
                **kwargs) -> Union[torch.Tensor, IntermediateTensors]:
        """Run forward pass for Ultravox

        One key thing to understand is the `input_ids` already accounts for the
        positions of the to-be-inserted audio embeddings. The to-be-inserted
        audio has a size that is essentially 6.25 tokens per second of audio.

        This way, the `positions` and `attn_metadata` are consistent
        with the `input_ids`.

        Args:
            audio_features: A batch of audio inputs [B, N, 80, M].
        """
        if intermediate_tensors is not None:
            input_ids = None
            inputs_embeds = None
        else:
            audio_input = self._parse_and_validate_audio_input(**kwargs)
            if audio_input is not None:
                audio_embeddings = self._process_audio_input(audio_input)
                inputs_embeds = self.language_model.model.get_input_embeddings(
                    input_ids)

                merge_multimodal_embeddings_from_map(
                    inputs_embeds, audio_embeddings,
                    attn_metadata.multi_modal_placeholder_index_maps["audio"])
                input_ids = None
            else:
                inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.language_model.sample(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        hf_to_vllm_mapper = WeightsMapper(
            orig_to_new_prefix={"audio_tower.model.encoder.": "audio_tower."})

        loader = AutoWeightsLoader(self,
                                   ignore_unexpected_prefixes=["audio_tower."])
        loader.load_weights(weights, mapper=hf_to_vllm_mapper)
