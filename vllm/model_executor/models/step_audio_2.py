 # SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Step-Audio2 implementation for vLLM

from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property
from typing import Annotated, Literal, Optional, TypedDict, Union

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
                                    MultiModalKwargs, MultiModalKwargsItems, NestedTensors)
from vllm.multimodal.parse import (DictEmbeddingItems, MultiModalDataItems,
                                   MultiModalDataParser)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate, PromptUpdateDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .utils import (flatten_bn, init_vllm_registered_model, maybe_prefix,
                    merge_multimodal_embeddings)

# Audio placeholder strings
_AUDIO_PLACEHOLDER = "<audio_patch>"
_AUDIO_START_PLACEHOLDER = "<audio_start>"
_AUDIO_END_PLACEHOLDER = "<audio_end>"


class StepAudio2AudioFeatureInputs(TensorSchema):
    """
    Audio feature inputs as mel spectrograms.

    Dimensions:
    - b: batch size
    - t: time frames (variable)
    - nmb: number of mel bins (128)
    """
    type: Literal["audio_features"]
    audio_mels: Annotated[Union[torch.Tensor, list[torch.Tensor]],
                          TensorShape("b", "t", "nmb", dynamic_dims={"t"})]
    audio_lens: Annotated[Union[torch.Tensor, list[int]],
                         TensorShape("b")]


class StepAudio2AudioEmbeddingInputs(TensorSchema):
    """
    Pre-computed audio embeddings.

    Dimensions:
    - b: batch size
    - naf: number of audio features
    - hs: hidden size
    """
    type: Literal["audio_embeds"]
    audio_embeds: Annotated[Union[torch.Tensor, list[torch.Tensor]],
                           TensorShape("b", "naf", "hs")]


StepAudio2AudioInputs = Union[StepAudio2AudioFeatureInputs,
                             StepAudio2AudioEmbeddingInputs]


def make_non_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of non-padded part."""
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return ~mask


def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert bool-tensor to float-tensor for flash attention."""
    assert mask.dtype == torch.bool
    assert dtype in [torch.float32, torch.bfloat16, torch.float16]
    mask = mask.to(dtype)
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

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      mask: Optional[torch.Tensor] = None):
        _, T, D = q.shape
        scale = (D // self.n_head) ** -0.25
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
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_mlp),
            nn.GELU(),
            nn.Linear(n_mlp, n_state)
        )
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = x + self.attn(self.attn_ln(x), mask=mask)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    """Step-Audio2 audio encoder based on Whisper architecture."""

    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.positional_embedding = nn.Embedding(n_ctx, n_state)

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        self.after_norm = nn.LayerNorm(n_state)

    def forward(self, x: torch.Tensor, x_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        mask = mask_to_bias(mask[:, :, (T + 1) % 2::2], x.dtype)  # (B, 1, T // 2)

        x = (x + self.positional_embedding.weight[:x.shape[1], :]).to(x.dtype).contiguous()

        for block in self.blocks:
            x = block(x, mask.unsqueeze(1))

        x = x.permute(0, 2, 1)
        x = self.avg_pooler(x)
        x = x.permute(0, 2, 1)
        x_len = (x_len + 1) // 2 // 2
        x = self.after_norm(x)
        return x, x_len


class Adaptor(nn.Module):
    """Adapter module to project audio features to LLM dimensions."""

    def __init__(self, n_state: int = 1280, n_hidden: int = 3584,
                 kernel_size: int = 3, stride: int = 2, adapter_state: int = 2048):
        super().__init__()
        self.stride = stride
        if self.stride != -1:
            self.conv = nn.Conv1d(n_state, n_state, kernel_size, stride, padding=1)
        self.linear1 = nn.Linear(n_state, adapter_state)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(adapter_state, n_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class StepAudio2MultiModalDataParser(MultiModalDataParser):
    """Custom parser to support dict-based audio embeddings."""

    def _parse_audio_data(self, data: object) -> MultiModalDataItems:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="audio",
                required_fields={"audio_embeds"},
                fields_factory=lambda hf_inputs: {
                    "audio_embeds": MultiModalFieldConfig.batched("audio")
                }
            )
        return super()._parse_audio_data(data)


class StepAudio2ProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"audio": None}

    def get_hf_processor(self, **kwargs):
        """Get the HuggingFace processor for Step-Audio2."""
        from vllm.transformers_utils.processors.step_audio2 import StepAudio2Processor

        tokenizer = self.get_tokenizer(**kwargs)
        config = self.get_hf_config()

        processor = StepAudio2Processor(
            tokenizer=tokenizer,
            config=config,
            **kwargs
        )

        # Set standardized audio token replacement patterns
        processor.audio_token_replacement = _AUDIO_PLACEHOLDER
        processor.audio_start_replacement = _AUDIO_START_PLACEHOLDER
        processor.audio_end_replacement = _AUDIO_END_PLACEHOLDER

        # Get token IDs from config or processor
        if hasattr(config, 'audio_token_index'):
            processor.audio_replacement_token_id = config.audio_token_index
        else:
            # Use processor's own token ID
            processor.audio_replacement_token_id = processor.audio_token_id

        return processor

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        """Calculate maximum tokens per audio item for profiling."""
        processor = self.get_hf_processor()
        # Maximum audio length: 29 seconds at 16kHz
        max_audio_length = int(16000 * 29)
        dummy_audio = np.zeros(max_audio_length, dtype=np.float32)
        dummy_mels = processor.preprocess_audio(dummy_audio)
        # Get the full replacement token sequence including start/end tokens
        _, token_ids = processor._get_audio_repl(dummy_mels.shape[0])
        return {"audio": len(token_ids)}

    def get_num_mm_tokens(self, mm_data: MultiModalDataDict) -> int:
        """Calculate actual token count for given audio data."""
        if len(mm_data) != 1 or "audio" not in mm_data:
            raise ValueError(
                "mm_data must contain exactly one key 'audio' for Step-Audio2")

        audio_data = mm_data["audio"]
        if not isinstance(audio_data, (list, tuple)):
            audio_data = [audio_data]

        processor = self.get_hf_processor()
        total_tokens = 0

        for audio in audio_data:
            # Handle different audio data formats
            if isinstance(audio, tuple) and len(audio) == 2:
                audio_array, sample_rate = audio
                # Resample to 16kHz if needed
                if sample_rate != 16000 and hasattr(audio_array, '__len__'):
                    # Calculate resampled length
                    resampled_length = int(len(audio_array) * 16000 / sample_rate)
                    audio_array = np.zeros(resampled_length, dtype=np.float32)
            else:
                # Assume it's already a numpy array at 16kHz
                audio_array = audio

            # Process audio to mel-spectrogram
            audio_mels = processor.preprocess_audio(audio_array)
            # Get the full replacement token sequence
            _, token_ids = processor._get_audio_repl(audio_mels.shape[0])
            total_tokens += len(token_ids)

        return total_tokens

    def check_valid_mm(
        self,
        mm_data: MultiModalDataDict,
        token_ids=None
    ) -> bool:
        """Validate audio data and check duration limits."""
        if len(mm_data) != 1 or "audio" not in mm_data:
            raise ValueError(
                "mm_data must contain exactly one key 'audio' for Step-Audio2")

        audio_data = mm_data["audio"]
        if not isinstance(audio_data, (list, tuple)):
            audio_data = [audio_data]

        # Check each audio item for duration limits
        for audio in audio_data:
            if isinstance(audio, tuple) and len(audio) == 2:
                audio_array, sample_rate = audio
                if hasattr(audio_array, '__len__') and sample_rate > 0:
                    duration = len(audio_array) / sample_rate
                    if duration > 29.98:
                        raise ValueError(
                            f"Audio duration {duration:.2f}s exceeds maximum "
                            f"allowed duration of 29.98s")
            else:
                # For raw arrays, assume 16kHz sample rate
                if hasattr(audio, '__len__'):
                    duration = len(audio) / 16000
                    if duration > 29.98:
                        raise ValueError(
                            f"Audio duration {duration:.2f}s exceeds maximum "
                            f"allowed duration of 29.98s")

        # Validate token count consistency if token_ids are provided
        if token_ids is not None:
            tokenizer = self.get_tokenizer()
            processor = self.get_hf_processor()
            audio_placeholder = getattr(processor, 'audio_token_replacement', _AUDIO_PLACEHOLDER)
            audio_token_id = tokenizer.get_vocab().get(audio_placeholder)

            # Count audio placeholder tokens in token_ids
            placeholder_token_count = sum(
                1 for token_id in token_ids if token_id == audio_token_id
            )

            # Expected placeholder count should match audio items
            expected_placeholder_count = len(audio_data)

            if placeholder_token_count != expected_placeholder_count:
                raise ValueError(
                    f"Mismatch between multimodal placeholder tokens in prompt "
                    f"({placeholder_token_count}) and mm_data count "
                    f"({expected_placeholder_count})")

        return True


class StepAudio2DummyInputsBuilder(BaseDummyInputsBuilder[StepAudio2ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)

        # Get placeholder from processor for consistency
        processor = self.info.get_hf_processor()
        audio_token = getattr(processor, 'audio_token_replacement', _AUDIO_PLACEHOLDER)

        return audio_token * num_audios

    def get_dummy_mm_data(self, seq_len: int, mm_counts: Mapping[str, int]) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        # Create dummy audio data (25 seconds at 16kHz)
        audio_len = 25 * 16000
        return {"audio": self._get_dummy_audios(length=audio_len, num_audios=num_audios)}


class StepAudio2MultiModalProcessor(BaseMultiModalProcessor[StepAudio2ProcessingInfo]):
    def _get_data_parser(self) -> MultiModalDataParser:
        """Return custom parser that supports dict-based embeddings."""
        return StepAudio2MultiModalDataParser(target_sr=16000)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """Process inputs with proper fallback for text-only."""
        # Handle text-only input
        if not mm_data.get("audio", []):
            prompt_ids = self.info.get_tokenizer().encode(
                prompt, add_special_tokens=False
            )
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        # Ensure sampling rate is set for audio processing
        mm_kwargs = dict(**mm_kwargs, sampling_rate=16000)

        # Call parent processor
        return super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """Get multimodal field configuration for audio inputs."""
        audio_lens = hf_inputs.get("audio_lens", torch.empty(0))

        return {
            "audio_mels": MultiModalFieldConfig.flat_from_sizes("audio", audio_lens),
            "audio_lens": MultiModalFieldConfig.batched("audio"),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> list[PromptUpdate]:
        """Get prompt updates for audio inputs."""
        # Check if we have any audio data to process
        out_mm_data = out_mm_kwargs.get_data() if out_mm_kwargs else {}

        # If no audio data, return empty list
        if not out_mm_data or ("audio_mels" not in out_mm_data and "audio_lens" not in out_mm_data):
            return []

        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        # Use string placeholder for standardization
        audio_placeholder = getattr(processor, 'audio_token_replacement', _AUDIO_PLACEHOLDER)

        def get_replacement_audio(item_idx: int):
            # Get audio lengths from processed data
            if "audio_lens" in out_mm_data:
                audio_lens = out_mm_data["audio_lens"]
                if isinstance(audio_lens, torch.Tensor):
                    audio_lens = audio_lens.tolist()
                num_feature_len = audio_lens[item_idx] if item_idx < len(audio_lens) else 0
            else:
                # Default to standard length if no lens provided
                num_feature_len = 1500  # Default for 30s audio

            # Get replacement tokens from processor
            _, audio_repl_ids = processor._get_audio_repl(num_feature_len)

            # Get the audio token ID from processor
            audio_token_id = getattr(processor, 'audio_replacement_token_id', processor.audio_token_id)

            return PromptUpdateDetails.select_token_id(
                seq=audio_repl_ids,
                embed_token_id=audio_token_id,
            )

        return [
            PromptReplacement(
                modality="audio",
                target=audio_placeholder,
                replacement=get_replacement_audio,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    StepAudio2MultiModalProcessor,
    info=StepAudio2ProcessingInfo,
    dummy_inputs=StepAudio2DummyInputsBuilder
)
class StepAudio2ForCausalLM(nn.Module, SupportsMultiModal, SupportsPP):
    """Step-Audio2 model for causal language modeling with audio understanding."""

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality == "audio" or modality.startswith("audio"):
            return _AUDIO_PLACEHOLDER
        return None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        # Audio encoder configuration
        audio_config = config.audio_encoder_config
        self.encoder = AudioEncoder(
            n_mels=audio_config.n_mels,
            n_ctx=audio_config.n_audio_ctx,
            n_state=audio_config.n_audio_state,
            n_head=audio_config.n_audio_head,
            n_layer=audio_config.n_audio_layer
        )

        # Adapter configuration
        self.adapter = Adaptor(
            n_state=audio_config.n_audio_state,
            n_hidden=audio_config.llm_dim,
            kernel_size=audio_config.kernel_size,
            stride=audio_config.adapter_stride
        )

        # Language model backbone (Qwen2)
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model")
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

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

    def _parse_and_validate_audio_input(self, **kwargs: object) -> Optional[StepAudio2AudioInputs]:
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
            dim=0
        )

        return StepAudio2AudioFeatureInputs(
            type="audio_features",
            audio_mels=audio_mels.to(self.dtype).to(self.device),
            audio_lens=audio_lens,
        )

    def _process_audio_input(self, audio_input: StepAudio2AudioInputs) -> tuple[torch.Tensor, ...]:
        audio_mels = audio_input["audio_mels"]
        audio_lens = torch.tensor(audio_input["audio_lens"], device=self.device)

        # Permute to (B, n_mels, T)
        audio_mels = audio_mels.permute(0, 2, 1)

        # Pass through encoder and adapter
        audio_features, audio_lens = self.encoder(audio_mels, audio_lens)
        audio_features = self.adapter(audio_features)
        audio_feature_lens = (audio_lens - 1) // 2 + 1

        # Split features by length
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
        inputs_embeds = self.language_model.model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            # Get audio token ID from config or tokenizer
            config = self.config
            audio_token_id = getattr(config, 'audio_token_index', None)
            if audio_token_id is None:
                # Fallback to tokenizer to get ID
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    config._name_or_path, trust_remote_code=True
                )
                audio_token_id = tokenizer.convert_tokens_to_ids(_AUDIO_PLACEHOLDER)

            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings, audio_token_id
            )
        return inputs_embeds

    def get_language_model(self):
        """Get the language model component for consistency with other models."""
        return self.language_model

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
            # Always pass the input via `inputs_embeds` for consistency
            inputs_embeds = self.get_input_embeddings(input_ids, audio_embeddings)
            input_ids = None

        hidden_states = self.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states, sampling_metadata)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.language_model.sample(logits, sampling_metadata)

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
            # Remap parameter names if necessary
            if name.startswith("model."):
                name = name.replace("model.", "language_model.model.")
            if name.startswith("lm_head"):
                name = name.replace("lm_head", "language_model.lm_head")

            # Handle stacked parameters
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
                # Load regular parameters
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)

        return loaded_params