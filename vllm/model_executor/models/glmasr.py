# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from typing import Annotated, Any, Literal, TypeAlias, cast

import numpy as np
import torch
import torch.nn as nn
from transformers import BatchFeature
from transformers.models.glmasr import GlmAsrConfig, GlmAsrProcessor
from transformers.models.whisper import WhisperFeatureExtractor

from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs.data import PromptType
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
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
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.transformers_utils.processor import cached_processor_from_config
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .audioflamingo3 import (
    AudioFlamingo3MultiModalDataParser,
    AudioFlamingo3ProcessingInfo,
)
from .audioflamingo3 import (
    _audioflamingo3_field_config as _glmasr_field_config,
)
from .glmasr_utils import (
    DEFAULT_CONV_PARAMS,
    DEFAULT_MAX_AUDIO_LEN_S,
    DEFAULT_MERGE_FACTOR,
    GlmAsrEncoder,
    _flatten_audio_features_by_length,
    _get_audio_output_lengths_for_tower,
    _get_num_features_for_item,
    _group_audio_embeddings,
    _normalize_chunk_counts,
)
from .interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
    SupportsTranscription,
)
from .utils import AutoWeightsLoader, init_vllm_registered_model, maybe_prefix
from .whisper import ISO639_1_SUPPORTED_LANGS

logger = logging.getLogger(__name__)


# =============================================================================
# GPU-accelerated Whisper Feature Extractor
# =============================================================================


@lru_cache(maxsize=1)
def _get_mel_filters(
    n_fft: int = 400,
    n_mels: int = 80,
    sampling_rate: int = 16000,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Compute mel filterbank matrix (cached).
    Matches WhisperFeatureExtractor's mel_filter_bank with slaney norm/scale.
    """
    if device is None:
        device = torch.device("cpu")
    # Frequency bins
    n_freqs = n_fft // 2 + 1
    all_freqs = torch.linspace(0, sampling_rate // 2, n_freqs, device=device)

    # Mel scale conversion (slaney)
    min_mel = 0.0
    max_mel = 2595.0 * np.log10(1.0 + (sampling_rate / 2) / 700.0)
    mels = torch.linspace(min_mel, max_mel, n_mels + 2, device=device)
    mel_freqs = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    # Create filterbank
    mel_filters = torch.zeros(n_mels, n_freqs, device=device)
    for i in range(n_mels):
        lower = mel_freqs[i]
        center = mel_freqs[i + 1]
        upper = mel_freqs[i + 2]

        # Lower slope
        lower_slope = (all_freqs - lower) / (center - lower + 1e-10)
        # Upper slope
        upper_slope = (upper - all_freqs) / (upper - center + 1e-10)

        mel_filters[i] = torch.maximum(
            torch.zeros_like(all_freqs),
            torch.minimum(lower_slope, upper_slope),
        )

    # Slaney normalization
    enorm = 2.0 / (mel_freqs[2 : n_mels + 2] - mel_freqs[:n_mels])
    mel_filters *= enorm.unsqueeze(1)

    return mel_filters


class GPUWhisperFeatureExtractor:
    """
    GPU-accelerated Whisper feature extractor using PyTorch.
    Computes log-mel spectrogram matching WhisperFeatureExtractor output.

    Key parameters (Whisper defaults):
        - n_fft: 400 (25ms window at 16kHz)
        - hop_length: 160 (10ms hop at 16kHz)
        - n_mels: 80
        - chunk_length: 30 seconds
        - sampling_rate: 16000
    """

    def __init__(
        self,
        feature_size: int = 80,
        sampling_rate: int = 16000,
        hop_length: int = 160,
        chunk_length: int = 30,
        n_fft: int = 400,
        padding_value: float = 0.0,
        device: str | torch.device = "cuda",
    ):
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_fft = n_fft
        self.padding_value = padding_value
        self.device = torch.device(device) if isinstance(device, str) else device

        # Derived parameters
        self.n_samples = chunk_length * sampling_rate  # 480000 for 30s
        self.nb_max_frames = self.n_samples // hop_length  # 3000 frames

        # Pre-compute window and mel filters on device
        self._window: torch.Tensor | None = None
        self._mel_filters: torch.Tensor | None = None

    def _ensure_buffers(self, device: torch.device) -> None:
        """Lazily initialize buffers on the target device."""
        if self._window is None or self._window.device != device:
            self._window = torch.hann_window(self.n_fft, device=device)

        if self._mel_filters is None or self._mel_filters.device != device:
            self._mel_filters = _get_mel_filters(
                n_fft=self.n_fft,
                n_mels=self.feature_size,
                sampling_rate=self.sampling_rate,
                device=device,
            )

    def __call__(
        self,
        raw_speech: list[np.ndarray] | np.ndarray | torch.Tensor,
        sampling_rate: int | None = None,
        padding: str = "max_length",
        max_length: int | None = None,
        return_attention_mask: bool = True,
        return_tensors: str = "pt",
        device: str | torch.device | None = None,
    ) -> BatchFeature:
        """
        Extract log-mel spectrogram features from audio.

        Args:
            raw_speech: Audio waveform(s), can be list of arrays or batched
            sampling_rate: Expected sample rate (must match self.sampling_rate)
            padding: Padding strategy ('max_length' or 'longest')
            max_length: Max samples (default: self.n_samples = 30s * 16kHz)
            return_attention_mask: Whether to return attention mask
            return_tensors: Output format ('pt' for PyTorch)
            device: Device for computation (default: self.device)

        Returns:
            BatchFeature with 'input_features' and optionally 'attention_mask'
        """
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"Expected sampling_rate={self.sampling_rate}, got {sampling_rate}"
            )

        device = torch.device(device) if device else self.device
        max_length = max_length or self.n_samples

        # Convert inputs to list of 1D tensors
        if isinstance(raw_speech, np.ndarray):
            raw_speech = [raw_speech] if raw_speech.ndim == 1 else list(raw_speech)
        elif isinstance(raw_speech, torch.Tensor):
            raw_speech = (
                [raw_speech.numpy()]
                if raw_speech.ndim == 1
                else [s.numpy() for s in raw_speech]
            )

        batch_size = len(raw_speech)

        # Get actual lengths before padding
        lengths = [len(s) for s in raw_speech]

        # Pad/truncate to max_length
        if padding == "max_length":
            target_length = max_length
        else:  # 'longest'
            target_length = min(max(lengths), max_length)

        # Create padded batch tensor
        padded_waveforms = torch.zeros(
            batch_size, target_length, dtype=torch.float32, device=device
        )
        attention_mask = torch.zeros(
            batch_size, target_length, dtype=torch.int32, device=device
        )

        for i, waveform in enumerate(raw_speech):
            if isinstance(waveform, np.ndarray):
                waveform = torch.from_numpy(waveform)
            waveform = waveform.to(device=device, dtype=torch.float32)

            # Truncate if needed
            actual_len = min(len(waveform), target_length)
            padded_waveforms[i, :actual_len] = waveform[:actual_len]
            attention_mask[i, :actual_len] = 1

        # Extract features on GPU
        input_features = self._extract_fbank_features(padded_waveforms)

        # Rescale attention mask from samples to frames
        # STFT produces L//hop_length + 1 frames, but we drop the last one
        frame_attention_mask = attention_mask[:, :: self.hop_length]
        # Trim to match actual frame count (we drop last frame in _extract)
        if attention_mask.shape[1] % self.hop_length != 0:
            frame_attention_mask = frame_attention_mask[:, :-1]

        result: dict[str, Any] = {"input_features": input_features}
        if return_attention_mask:
            result["attention_mask"] = frame_attention_mask

        return BatchFeature(data=result, tensor_type=return_tensors)

    def _extract_fbank_features(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Compute log-mel spectrogram for batched waveforms.

        Args:
            waveforms: [batch, samples] float32 tensor on target device

        Returns:
            [batch, n_mels, frames] float32 tensor (log-mel spectrogram)
        """
        device = waveforms.device
        self._ensure_buffers(device)

        # STFT: [batch, samples] -> [batch, n_fft//2+1, frames] complex
        stft = torch.stft(
            waveforms,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self._window,
            return_complex=True,
        )

        # Power spectrogram, drop last frame (matching HF implementation)
        magnitudes = stft[..., :-1].abs() ** 2  # [batch, n_freqs, frames]

        # Apply mel filterbank: [n_mels, n_freqs] @ [batch, n_freqs, frames]
        # -> [batch, n_mels, frames]
        mel_spec = torch.matmul(self._mel_filters, magnitudes)

        # Log scale with floor
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()

        # Per-sample normalization (max - 8.0 floor, then scale)
        max_val = log_spec.amax(dim=(1, 2), keepdim=True)
        log_spec = torch.maximum(log_spec, max_val - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec


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
        TensorShape("num_chunks", "nmb", "chunk_length", dynamic_dims={"chunk_length"}),
    ]
    feature_attention_mask: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("num_chunks", "chunk_length", dynamic_dims={"chunk_length"}),
    ]
    chunk_counts: Annotated[
        torch.Tensor | list[torch.Tensor],
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
    def __init__(
        self,
        config: GlmAsrConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.linear_1 = ColumnParallelLinear(
            input_size=config.audio_config.intermediate_size,
            output_size=config.text_config.hidden_size * 2,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_1",
        )
        self.act = get_act_fn(config.projector_hidden_act)
        self.linear_2 = RowParallelLinear(
            input_size=config.text_config.hidden_size * 2,
            output_size=config.text_config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_2",
        )

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.linear_1(audio_features)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states


class GlmAsrProcessingInfo(AudioFlamingo3ProcessingInfo):
    def get_hf_config(self) -> GlmAsrConfig:
        return self.ctx.get_hf_config(GlmAsrConfig)

    def get_hf_processor(self, **kwargs: object) -> GlmAsrProcessor:
        return self.ctx.get_hf_processor(GlmAsrProcessor, **kwargs)

    def get_feature_extractor(self, **kwargs: object) -> WhisperFeatureExtractor:
        # Reuse parent implementation, but add type annotation and assertion
        feature_extractor = super().get_feature_extractor(**kwargs)
        assert isinstance(feature_extractor, WhisperFeatureExtractor)
        return feature_extractor


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


class GlmAsrMultiModalDataParser(AudioFlamingo3MultiModalDataParser):
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


class GlmAsrMultiModalProcessor(BaseMultiModalProcessor["GlmAsrProcessingInfo"]):
    """
    GLM-ASR processor that inherits directly from BaseMultiModalProcessor
    for better performance and cleaner implementation.
    Uses GPU-accelerated feature extraction for improved throughput.
    """

    # Shared GPU feature extractor instance (lazy initialized)
    _gpu_feature_extractor: GPUWhisperFeatureExtractor | None = None

    @classmethod
    def _get_gpu_feature_extractor(
        cls,
        hf_feature_extractor: WhisperFeatureExtractor,
        device: str = "cuda",
    ) -> GPUWhisperFeatureExtractor:
        """Get or create GPU feature extractor matching HF config."""
        if cls._gpu_feature_extractor is None:
            cls._gpu_feature_extractor = GPUWhisperFeatureExtractor(
                feature_size=hf_feature_extractor.feature_size,
                sampling_rate=hf_feature_extractor.sampling_rate,
                hop_length=hf_feature_extractor.hop_length,
                chunk_length=hf_feature_extractor.chunk_length,
                n_fft=hf_feature_extractor.n_fft,
                padding_value=hf_feature_extractor.padding_value,
                device=device,
            )
        return cls._gpu_feature_extractor

    def _get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.info.get_feature_extractor()
        return GlmAsrMultiModalDataParser(target_sr=feature_extractor.sampling_rate)

    def _calculate_chunk_counts(
        self,
        audio_list: list[Any],
        feature_extractor: WhisperFeatureExtractor,
        processor: GlmAsrProcessor,
    ) -> list[int]:
        """Calculate chunk counts for each audio."""
        sampling_rate = feature_extractor.sampling_rate
        chunk_length = feature_extractor.chunk_length
        max_audio_len = getattr(processor, "max_audio_len", DEFAULT_MAX_AUDIO_LEN_S)
        window_size = int(sampling_rate * chunk_length)
        max_windows = int(max_audio_len // chunk_length)

        chunk_counts = []
        for audio in audio_list:
            n_samples = len(audio) if isinstance(audio, list) else audio.shape[0]
            n_chunks = max(1, (n_samples + window_size - 1) // window_size)
            chunk_counts.append(min(n_chunks, max_windows))
        return chunk_counts

    # @torch.compile(fullgraph=True)
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: dict[str, object],
        mm_kwargs: Mapping[str, Any],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """
        Call processor with GPU-accelerated feature extraction.
        """
        # Normalize input: handle deprecated key and list conversion.
        if "audios" in mm_data:
            mm_data["audio"] = mm_data.pop("audios")

        audio = mm_data.get("audio", [])
        audio_list = [audio] if audio and not isinstance(audio, list) else audio

        # Early return for text-only.
        if not audio_list:
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        # Get processor for tokenizer and config
        processor = self.info.get_hf_processor(**mm_kwargs)
        hf_feature_extractor = processor.feature_extractor
        tokenizer = processor.tokenizer

        # ===== Audio chunking (CPU, fast) =====
        sampling_rate = hf_feature_extractor.sampling_rate
        chunk_length = hf_feature_extractor.chunk_length
        max_audio_len = getattr(processor, "max_audio_len", DEFAULT_MAX_AUDIO_LEN_S)
        window_size = int(sampling_rate * chunk_length)
        max_windows = int(max_audio_len // chunk_length)

        per_sample_windows: list[int] = []
        flat_chunks: list[np.ndarray] = []

        for audio_el in audio_list:
            # Convert to numpy if needed
            if isinstance(audio_el, torch.Tensor):
                audio_el = audio_el.numpy()
            elif isinstance(audio_el, list):
                audio_el = np.array(audio_el, dtype=np.float32)

            n_samples = int(audio_el.shape[0])
            n_win = max(1, (n_samples + window_size - 1) // window_size)
            if n_win > max_windows:
                n_win = max_windows

            per_sample_windows.append(n_win)
            time_cap = min(n_samples, n_win * window_size)

            for i in range(n_win):
                start = i * window_size
                end = min((i + 1) * window_size, time_cap)
                flat_chunks.append(audio_el[start:end])

        # ===== GPU Feature Extraction =====
        # Check if CUDA is available, fallback to CPU if not
        use_gpu = torch.cuda.is_available()
        device = "cuda" if use_gpu else "cpu"

        if use_gpu:
            # Use GPU-accelerated feature extractor
            gpu_extractor = self._get_gpu_feature_extractor(
                hf_feature_extractor, device=device
            )
            audio_inputs = gpu_extractor(
                flat_chunks,
                sampling_rate=sampling_rate,
                return_attention_mask=True,
                return_tensors="pt",
            )
        else:
            # Fallback to HF CPU implementation
            audio_inputs = hf_feature_extractor(
                flat_chunks,
                sampling_rate=sampling_rate,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            )

        # ===== Process attention mask =====
        padding_mask = audio_inputs.pop("attention_mask")
        input_features_mask = padding_mask

        # ===== Compute audio token lengths =====
        chunk_lengths = padding_mask.sum(-1)  # [num_chunks]
        audio_lengths = torch.stack(
            [
                chunk_lengths[
                    sum(per_sample_windows[:i]) : sum(per_sample_windows[: i + 1])
                ].sum()
                for i in range(len(per_sample_windows))
            ]
        )

        # Apply convolution formula to get token counts
        merge_factor = 4
        for padding, kernel_size, stride in [(1, 3, 1), (1, 3, 2)]:
            audio_lengths = (
                audio_lengths + 2 * padding - (kernel_size - 1) - 1
            ) // stride + 1
        audio_tokens_lengths = (audio_lengths - merge_factor) // merge_factor + 1

        # ===== Expand audio tokens in text =====
        import regex as re

        audio_token = getattr(processor, "audio_token", "<|pad|>")
        text_list = [prompt]

        for i, audio_length in enumerate(audio_tokens_lengths):
            if i < len(text_list):
                expanded = re.sub(
                    re.escape(audio_token),
                    audio_token * int(audio_length),
                    text_list[i],
                )
                text_list[i] = expanded

        # ===== Tokenize text =====
        text_inputs = tokenizer(
            text_list,
            return_tensors="pt",
            padding=True,
            **tok_kwargs,
        )

        # ===== Combine outputs =====
        # Move input_features to CPU for compatibility
        input_features = audio_inputs["input_features"]
        if input_features.device.type != "cpu":
            input_features = input_features.cpu()
        if input_features_mask.device.type != "cpu":
            input_features_mask = input_features_mask.cpu()

        outputs = BatchFeature(
            data={
                **text_inputs,
                "input_features": input_features,
                "feature_attention_mask": input_features_mask,
            },
            tensor_type="pt",
        )

        outputs["chunk_counts"] = torch.tensor(per_sample_windows, dtype=torch.long)

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

        merge_factor = getattr(config, "merge_factor", DEFAULT_MERGE_FACTOR)
        out_mm_data = out_mm_kwargs.get_data()
        feature_attention_mask = out_mm_data.get("feature_attention_mask")
        chunk_counts = out_mm_data.get("chunk_counts")

        def get_replacement_glmasr(item_idx: int):
            conv_params = getattr(config, "conv_params", DEFAULT_CONV_PARAMS)
            audio_embeds = out_mm_data.get("audio_embeds")
            num_features = _get_num_features_for_item(
                feature_attention_mask,
                chunk_counts,
                item_idx,
                audio_embeds,
                merge_factor,
                conv_params,
            )

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
    nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA, SupportsTranscription
):
    supported_languages = ISO639_1_SUPPORTED_LANGS

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

        # Use optimized vLLM native encoder
        self.audio_tower = GlmAsrEncoder(
            config.audio_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "audio_tower"),
        )
        self.multi_modal_projector = GlmAsrMultiModalProjector(
            config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "multi_modal_projector"),
        )
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
        audio_embeds = kwargs.pop("audio_embeds", None)
        if audio_embeds is not None:
            return GlmAsrEmbeddingInputs(type="audio_embeds", audio_embeds=audio_embeds)

        input_features = kwargs.pop("input_features", None)
        if input_features is None:
            return None

        return GlmAsrFeatureInputs(
            type="audio_features",
            input_features=input_features,
            feature_attention_mask=kwargs.pop("feature_attention_mask", None),
            chunk_counts=kwargs.pop("chunk_counts", None),
        )

    def _process_audio_input(
        self, audio_input: GlmAsrInputs
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if audio_input["type"] == "audio_embeds":
            return tuple(audio_input["audio_embeds"])

        input_features = audio_input["input_features"]
        feature_attention_mask = audio_input["feature_attention_mask"]

        if isinstance(input_features, list):
            input_features = torch.cat(input_features, dim=0)
            feature_attention_mask = torch.cat(feature_attention_mask, dim=0)

        num_chunks = input_features.shape[0]
        chunk_counts = _normalize_chunk_counts(
            audio_input.get("chunk_counts"), num_chunks=num_chunks
        )

        # Convert input_features to model dtype (e.g., bfloat16) to match model weights
        input_features = input_features.to(dtype=self.audio_tower.conv1.weight.dtype)

        # audio_tower returns [batch_size, seq_len, hidden_size] where hidden_size=1280
        audio_hidden_states = self.audio_tower(input_features).last_hidden_state

        # GLM-ASR merges consecutive frames: 4 frames with hidden_size=1280
        # -> 1 frame with intermediate_size=5120
        hidden_size = self.config.audio_config.hidden_size
        intermediate_size = self.config.audio_config.intermediate_size
        merge_ratio = intermediate_size // hidden_size

        # Truncate sequence length to be divisible by merge_ratio
        seq_len = audio_hidden_states.shape[1]
        seq_len_truncated = (seq_len // merge_ratio) * merge_ratio
        if seq_len_truncated < seq_len:
            audio_hidden_states = audio_hidden_states[:, :seq_len_truncated, :]

        # Reshape to merge consecutive frames
        audio_hidden_states = audio_hidden_states.reshape(
            num_chunks,
            -1,
            intermediate_size,
        )

        audio_features = self.multi_modal_projector(audio_hidden_states)

        merge_factor = getattr(self.config, "merge_factor", DEFAULT_MERGE_FACTOR)
        conv_params = getattr(self.config, "conv_params", DEFAULT_CONV_PARAMS)

        audio_output_lengths = _get_audio_output_lengths_for_tower(
            self.audio_tower,
            feature_attention_mask.sum(-1),
            merge_factor,
            conv_params,
        )

        masked_audio_features = _flatten_audio_features_by_length(
            audio_features, audio_output_lengths
        )

        chunk_embeddings = torch.split(
            masked_audio_features, audio_output_lengths.flatten().tolist()
        )
        result = _group_audio_embeddings(chunk_embeddings, chunk_counts)

        return result

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

    @classmethod
    def _get_audio_token(cls, model_config: ModelConfig) -> str:
        """Get the audio token from processor.

        Similar to get_placeholder_str but returns single token.
        """
        processor = cached_processor_from_config(model_config)
        return getattr(processor, "audio_token", "<|pad|>")

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig:
        processor = cached_processor_from_config(model_config)
        feature_extractor = processor.feature_extractor
        max_audio_clip_s = getattr(processor, "max_audio_len", DEFAULT_MAX_AUDIO_LEN_S)
        return SpeechToTextConfig(
            max_audio_clip_s=max_audio_clip_s,
            sample_rate=feature_extractor.sampling_rate,
        )

    @classmethod
    def get_generation_prompt(
        cls,
        audio: np.ndarray,
        model_config: ModelConfig,
        stt_config: SpeechToTextConfig,
        language: str | None,
        task_type: Literal["transcribe", "translate"],
        request_prompt: str,
        to_language: str | None,
    ) -> PromptType:
        """Get the generation prompt to be used for transcription requests."""
        tokenizer = cached_tokenizer_from_config(model_config)
        audio_token = cls._get_audio_token(model_config)

        if task_type == "translate":
            full_lang_name_to = cls.supported_languages.get(to_language, to_language)
            user_content = f"{audio_token}translate the speech to {full_lang_name_to}"
        elif task_type == "transcribe":
            user_content = (
                f"{audio_token}can you transcribe the speech into a written format?"
            )
        else:
            raise ValueError(f"Unsupported task type {task_type}")

        messages = [{"role": "user", "content": user_content}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        prompt_token_ids = tokenizer.encode(prompt)
        prompt_dict = {
            "prompt_token_ids": prompt_token_ids,
            "multi_modal_data": {"audio": audio},
        }
        return cast(PromptType, prompt_dict)
