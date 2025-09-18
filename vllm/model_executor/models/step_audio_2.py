# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property
from typing import Optional, TypedDict, Union

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BatchFeature, PretrainedConfig, TensorType

from vllm.config import VllmConfig
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs, NestedTensors)
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate, PromptUpdateDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs import StepAudio2EncoderConfig
from vllm.transformers_utils.tokenizer import AnyTokenizer

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .utils import (flatten_bn, init_vllm_registered_model, maybe_prefix,
                    merge_multimodal_embeddings)

AUDIO_PATCH_TOKEN_ID = 151690


class Step1fAudioInputs(TypedDict):
    audio_mels: torch.Tensor
    """Shape: `(num_audios * num_frames, num_mel_bins)`"""

    audio_lens: list[int]
    """Shape: `(num_audios,)`"""


class Step1fProcessor:

    _mel_filters_cache = {}

    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: AnyTokenizer,
    ) -> None:
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer

        self.audio_token = "<audio_patch>"
        self.n_mels = 128
        self.max_chunk_size = 29  # from audio encoder position embedding length equals 1500, means 29.98s audio # noqa: E501
        self.sampling_rate = 16000
        self._mel_filters = torch.from_numpy(
            librosa.filters.mel(sr=self.sampling_rate,
                                n_fft=400,
                                n_mels=self.n_mels))

    @property
    def audio_token_id(self) -> int:
        return self.tokenizer.get_vocab()[self.audio_token]

    def _log_mel_spectrogram(
        self,
        audio: np.ndarray,
        padding: int = 0,
    ):
        audio = F.pad(torch.from_numpy(audio.astype(np.float32)), (0, padding))
        window = torch.hann_window(400).to(audio.device)
        stft = torch.stft(audio, 400, 160, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs()**2
        filters = self._mel_filters
        mel_spec = filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec.t()

    def preprocess_audio(self, audio_tensor: np.ndarray) -> torch.Tensor:
        return self._log_mel_spectrogram(audio_tensor, padding=479)

    def get_num_audio_tokens(self, max_feature_len: int) -> int:
        encoder_output_dim = (
            max_feature_len +
            1) // 2 // 2  # from hych: align with log-to-mel padding 479
        padding = 1
        kernel_size = 3
        stride = 2
        adapter_output_dim = (encoder_output_dim + 2 * padding -
                              kernel_size) // stride + 1
        return adapter_output_dim

    def _get_audio_repl(
        self,
        audio_feat_len: int,
    ) -> tuple[str, list[int]]:
        num_audio_tokens = self.get_num_audio_tokens(audio_feat_len)
        text = "<audio_start>" + "<audio_patch>" * num_audio_tokens + "<audio_end>"  # noqa: E501
        token_ids = [self.tokenizer.convert_tokens_to_ids("<audio_start>")
                     ] + [self.audio_token_id] * num_audio_tokens + [
                         self.tokenizer.convert_tokens_to_ids("<audio_end>")
                     ]
        return text, token_ids

    def replace_placeholder(self, text: str, placeholder: str,
                            repls: list[str]) -> str:
        parts = text.split(placeholder)

        if len(parts) - 1 != len(repls):
            raise ValueError(
                "The number of placeholders does not match the number of replacements."  # noqa: E501
            )

        result = [parts[0]]
        for i, repl in enumerate(repls):
            result.append(repl)
            result.append(parts[i + 1])

        return "".join(result)

    def __call__(
        self,
        text: Optional[Union[str, list[str]]] = None,
        audios: Union[np.ndarray, list[np.ndarray]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        if text is None:
            text = []
        if not isinstance(text, list):
            text = [text]
        if audios is None:
            audios = []
        if not isinstance(audios, list):
            audios = [audios]

        if len(audios) == 0:
            audio_inputs = {}
            text_inputs = self.tokenizer(text)
        else:
            audio_mels_lst = []
            audio_repl_str_lst = []
            audio_repl_ids_lst = []
            for audio in audios:
                audio_mels = self.preprocess_audio(audio)
                audio_mels_lst.append(audio_mels)
                audio_repl_str, audio_repl_ids = self._get_audio_repl(
                    audio_mels.shape[0])
                audio_repl_str_lst.append(audio_repl_str)
                audio_repl_ids_lst.extend(audio_repl_ids)
            audio_inputs = {
                "audio_mels":
                torch.concat(audio_mels_lst),
                "audio_lens":
                [audio_mels.shape[0] for audio_mels in audio_mels_lst]
            }

            text = [
                self.replace_placeholder(t, self.audio_token,
                                         audio_repl_str_lst) for t in text
            ]
            text_inputs = self.tokenizer(text)

        return BatchFeature(
            {
                **text_inputs,
                **audio_inputs,
            },
            tensor_type=return_tensors,
        )


class Step1fAudioProcessingInfo(BaseProcessingInfo):

    def get_hf_processor(self) -> Step1fProcessor:
        return Step1fProcessor(
            self.get_hf_config(),
            self.get_tokenizer(),
        )

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"audio": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        hf_processor = self.get_hf_processor()
        max_audio_length = int(hf_processor.sampling_rate *
                               hf_processor.max_chunk_size)
        dummy_audio_tensor = np.zeros(max_audio_length, dtype=np.float32)
        dummy_audio_mels = hf_processor.preprocess_audio(dummy_audio_tensor)
        num_audio_tokens = len(
            hf_processor._get_audio_repl(dummy_audio_mels.shape[0])[1])
        return {"audio": num_audio_tokens}

    def get_num_mm_tokens(self, mm_data: MultiModalDataDict) -> int:
        if len(mm_data) != 1 or "audio" not in mm_data:
            raise ValueError(
                "mm_data could only contain one key 'audio' for step1f")

        audio_data = mm_data["audio"]
        if not isinstance(audio_data, (list, tuple)):
            audio_data = [audio_data]

        hf_processor = self.get_hf_processor()
        total_tokens = 0
        for audio in audio_data:
            # Handle audio data format: (audio_array, sample_rate)
            if isinstance(audio, tuple) and len(audio) == 2:
                audio_array, sample_rate = audio
                # Calculate resampled length without actual resampling
                if hasattr(audio_array, '__len__') and sample_rate > 0:
                    original_length = len(audio_array)
                    # Resample to 16000Hz
                    resampled_length = int(original_length * 16000 /
                                           sample_rate)
                    audio_array = np.zeros(resampled_length, dtype=np.float32)
                else:
                    # Fallback to original length if we can't calculate
                    audio_array = np.zeros(len(audio_array), dtype=np.float32)
            else:
                # Assume it's already a numpy array at 16000Hz
                audio_array = audio

            audio_mels = hf_processor.preprocess_audio(audio_array)
            audio_repl_ids = hf_processor._get_audio_repl(
                audio_mels.shape[0])[1]
            total_tokens += len(audio_repl_ids)

        return total_tokens

    def check_valid_mm(self,
                       mm_data: MultiModalDataDict,
                       token_ids=None) -> bool:
        if len(mm_data) != 1 or "audio" not in mm_data:
            raise ValueError(
                "mm_data could only contain one key 'audio' for step1f")

        audio_data = mm_data["audio"]

        for audio in audio_data:
            if isinstance(audio, tuple) and len(audio) == 2:
                audio_array, sample_rate = audio
                if hasattr(audio_array, '__len__') and sample_rate > 0:
                    duration = len(audio_array) / sample_rate
                    if duration > 29.98:
                        raise ValueError(
                            f"Audio duration {duration:.2f} seconds exceeds the maximum allowed duration of 29.98 seconds"  # noqa: E501
                        )
            else:
                raise ValueError(
                    "Audio data format error, should be a tuple (array, sample_rate)"  # noqa: E501
                )

        # Validate token count consistency if token_ids are provided
        if token_ids is not None:
            tokenizer = self.get_tokenizer()
            audio_token_id = tokenizer.get_vocab()["<audio_patch>"]

            # Count audio placeholder tokens in token_ids
            placeholder_token_count = sum(1 for token_id in token_ids
                                          if token_id == audio_token_id)

            # Expected placeholder count should match the number of audio items in mm_data # noqa: E501
            expected_placeholder_count = len(audio_data)

            if placeholder_token_count != expected_placeholder_count:
                raise ValueError(
                    f"Mismatch between multimodal placeholder tokens in prompt ({placeholder_token_count}) and mm_data count ({expected_placeholder_count})"  # noqa: E501
                )

        return True


class Step1fAudioDummyInputsBuilder(
        BaseDummyInputsBuilder[Step1fAudioProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        return "<audio_patch>" * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        hf_processor = self.info.get_hf_processor()
        audio_len = int(hf_processor.sampling_rate *
                        hf_processor.max_chunk_size)
        num_audios = mm_counts.get("audio", 0)
        return {
            "audio":
            self._get_dummy_audios(length=audio_len, num_audios=num_audios)
        }


class Step1fAudioMultiModalProcessor(
        BaseMultiModalProcessor[Step1fAudioProcessingInfo]):

    def _get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(target_sr=16000)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        audio_lens = hf_inputs.get("audio_lens", torch.empty(0))

        return dict(
            audio_mels=MultiModalFieldConfig.flat_from_sizes(
                "audio", audio_lens),
            audio_lens=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        audio_token_id = processor.audio_token_id

        out_mm_data = out_mm_kwargs.get_data()

        def get_replacement_step_audio(item_idx: int):
            batched_audio_lens = out_mm_data.get("audio_lens").tolist()
            num_feature_len = batched_audio_lens[item_idx]
            audio_repl_ids = processor._get_audio_repl(num_feature_len)[1]

            return PromptUpdateDetails.select_token_id(
                seq=audio_repl_ids,
                embed_token_id=audio_token_id,
            )

        return [
            PromptReplacement(
                modality="audio",
                target=[audio_token_id],
                replacement=get_replacement_step_audio,
            )
        ]


def make_non_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return ~mask


def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    assert mask.dtype == torch.bool
    assert dtype in [torch.float32, torch.bfloat16, torch.float16]
    mask = mask.to(dtype)
    # attention mask bias
    # NOTE(Mddct): torch.finfo jit issues
    #     chunk_masks = (1.0 - chunk_masks) * torch.finfo(dtype).min
    mask = (1.0 - mask) * -1.0e+10
    return mask


class MultiHeadAttention(nn.Module):

    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(self,
                      q: torch.Tensor,
                      k: torch.Tensor,
                      v: torch.Tensor,
                      mask: Optional[torch.Tensor] = None):
        _, T, D = q.shape
        scale = (D // self.n_head)**-0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k  # (B, n_head, T, T)
        if mask is not None:
            qk = qk + mask
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class ResidualAttentionBlock(nn.Module):

    def __init__(self, n_state: int, n_head: int):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(nn.Linear(n_state, n_mlp), nn.GELU(),
                                 nn.Linear(n_mlp, n_state))
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):

    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int,
                 n_layer: int):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state,
                               n_state,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.positional_embedding = nn.Embedding(n_ctx, n_state)

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)])
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        self.after_norm = nn.LayerNorm(n_state)

    def forward(self, x: torch.Tensor,
                x_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x : torch.Tensor, shape = (batch_size, n_mels, T)
            the mel spectrogram of the audio
        x_len: torch.Tensor, shape = (batch_size,)
            length of each audio in x
        """
        T = x.size(-1)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)  # (B, T // 2, n_state)
        mask = make_non_pad_mask(x_len, T).unsqueeze(1)  # (B, 1, T)
        mask = mask_to_bias(mask[:, :, (T + 1) % 2::2],
                            x.dtype)  # (B, 1, T // 2)

        x = (x + self.positional_embedding.weight[:x.shape[1], :]).to(
            x.dtype).contiguous()

        for block in self.blocks:
            x = block(x, mask.unsqueeze(1))

        x = x.permute(0, 2, 1)
        x = self.avg_pooler(x)
        x = x.permute(0, 2, 1)
        x_len = (x_len + 1) // 2 // 2
        x = self.after_norm(x)
        return x, x_len


class Adaptor(nn.Module):

    def __init__(self,
                 n_state: int = 1280,
                 n_hidden: int = 3072,
                 kernel_size: int = 7,
                 stride: int = 4,
                 adapter_state: int = 2048):
        super().__init__()
        self.stride = stride
        if self.stride != -1:
            self.conv = nn.Conv1d(n_state,
                                  n_state,
                                  kernel_size,
                                  stride,
                                  padding=1)
        self.linear1 = nn.Linear(n_state, adapter_state)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(adapter_state, n_hidden)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """
        x : torch.Tensor, shape = (batch_size, T, n_features)
        """

        if self.stride != -1:
            x = x.permute(0, 2, 1)  # (B, n_state, T)
            x = F.gelu(self.conv(x))
            x = x.permute(0, 2, 1)  # (B, T//stride, n_state)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class StepASREncoder(nn.Module):

    def __init__(self, config: StepAudio2EncoderConfig):
        super().__init__()
        config = config.get_audio_encoder_config()
        self.config = config
        self.encoder = AudioEncoder(config.n_mels, config.n_audio_ctx,
                                    config.n_audio_state, config.n_audio_head,
                                    config.n_audio_layer)
        self.adaptor = Adaptor(config.n_audio_state, config.llm_dim,
                               config.kernel_size, config.adapter_stride,
                               config.adapter_state)

    def forward(self, x: torch.Tensor,
                x_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, x_len = self.encoder(x, x_len)
        x = self.adaptor(x)
        x_len = (
            x_len - 1
        ) // 2 + 1  # FIXME(ys): hard code for audio token num, padding 1, kernel size 3, stride 2 # noqa: E501
        return x, x_len


@MULTIMODAL_REGISTRY.register_processor(
    Step1fAudioMultiModalProcessor,
    info=Step1fAudioProcessingInfo,
    dummy_inputs=Step1fAudioDummyInputsBuilder)
class StepAudio2ForCausalLM(nn.Module, SupportsMultiModal, SupportsPP):

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("audio"):
            return "<audio_patch>"

        raise ValueError("Only audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        self.encoder = AudioEncoder(config.audio_encoder_config.n_mels,
                                    config.audio_encoder_config.n_audio_ctx,
                                    config.audio_encoder_config.n_audio_state,
                                    config.audio_encoder_config.n_audio_head,
                                    config.audio_encoder_config.n_audio_layer)
        self.adapter = Adaptor(config.audio_encoder_config.n_audio_state,
                               config.audio_encoder_config.llm_dim,
                               config.audio_encoder_config.kernel_size,
                               config.audio_encoder_config.adapter_stride)

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"))

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    @cached_property
    def sampler(self):
        if hasattr(self.language_model, "sampler"):
            return self.language_model.sampler

        return get_sampler()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def _parse_and_validate_audio_input(
            self, **kwargs: object) -> Optional[Step1fAudioInputs]:
        audio_mels = kwargs.pop("audio_mels", None)
        audio_lens = kwargs.pop("audio_lens", None)

        if audio_mels is None:
            return None

        audio_mels = flatten_bn(audio_mels, concat=True)
        audio_lens = flatten_bn(audio_lens, concat=True).tolist()

        audio_mels_lst = []
        cur_idx = 0
        for audio_len in audio_lens:
            audio_mels_lst.append(audio_mels[cur_idx:cur_idx + audio_len])
            cur_idx += audio_len

        max_len = max(x.size(0) for x in audio_mels_lst)
        audio_mels = torch.stack(
            [F.pad(x, (0, 0, 0, max_len - x.size(0))) for x in audio_mels_lst],
            dim=0)

        return Step1fAudioInputs(
            audio_mels=audio_mels.to(self.dtype).to(self.device),
            audio_lens=audio_lens,
        )

    def _process_audio_input(
            self, audio_input: Step1fAudioInputs) -> tuple[torch.Tensor, ...]:
        audio_mels = audio_input["audio_mels"]
        audio_lens = torch.tensor(audio_input["audio_lens"],
                                  device=self.device)

        audio_mels = audio_mels.permute(0, 2,
                                        1)  # (B, T, n_mels) -> (B, n_mels, T)

        audio_features, audio_lens = self.encoder(audio_mels, audio_lens)
        audio_features = self.adapter(audio_features)
        audio_feature_lens = (
            audio_lens - 1
        ) // 2 + 1  # FIXME(ys): hard code for audio token num, padding 1, kernel size 3, stride 2 # noqa: E501

        audio_feature_list = [
            audio_features[i, :audio_feature_lens[i]]
            for i in range(audio_features.size(0))
        ]

        return audio_feature_list

    def get_multimodal_embeddings(self, **kwargs) -> Optional[NestedTensors]:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return None
        else:
            audio_embeddings = self._process_audio_input(audio_input)
            return audio_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.model.get_input_embeddings(
            input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                AUDIO_PATCH_TOKEN_ID)
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
        elif inputs_embeds is None:
            audio_embeddings = self.get_multimodal_embeddings(**kwargs)
            # always pass the input via `inputs_embeds`
            # to make sure the computation graph is consistent
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      audio_embeddings)
            input_ids = None

        hidden_states = self.language_model(input_ids,
                                            positions,
                                            intermediate_tensors,
                                            inputs_embeds=inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.language_model.sample(logits, sampling_metadata)

    def maybe_remap_params(self, name):
        if name.startswith("model."):
            name = name.replace("model.", "language_model.model.")
        if name.startswith("lm_head"):
            name = name.replace("lm_head", "language_model.lm_head")
        return name

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params = set()
        for name, loaded_weight in weights:

            name = self.maybe_remap_params(name)

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        params_need_to_load = []
        for name in params_dict:
            params_need_to_load.append(name)
        params_need_to_load = set(params_need_to_load)

        if params_need_to_load != loaded_params:
            param_name_example = list(params_need_to_load - loaded_params)[0]
            raise RuntimeError(
                f"Some parameters like {param_name_example} are not in the checkpoint and will falsely use random initialization"  # noqa: E501
            )
