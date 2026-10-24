# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from https://github.com/fixie-ai/ultravox/blob/ecd58c4041030bae2ad15aa6bcf04ab43199ea02/ultravox/model/ultravox_model.py
"""PyTorch Ultravox model."""

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, Literal, NamedTuple, TypeAlias

import torch
from torch import nn
from torch.nn import functional as F
from transformers import BatchFeature, ProcessorMixin
from transformers.models.whisper import WhisperFeatureExtractor

from vllm.config import VllmConfig
from vllm.config.multimodal import MultiModalDummyOptions
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.layers.activation import MulAndSilu, get_act_fn
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    FLASHINFER_BATCH_BUCKETS,
    bucket_flashinfer_max_seqlen,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader import DefaultModelLoader
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.vision import get_fp8_padded_hidden_size
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    NestedTensors,
)
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    cached_encode,
)
from vllm.multimodal.processing.processor import HFMultiModalInputs
from vllm.renderers import TokenizeParams
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.ultravox import UltravoxConfig
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.math_utils import round_up
from vllm.utils.tensor_schema import TensorSchema, TensorShape
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from .interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    flatten_bn,
    init_vllm_registered_model,
    maybe_prefix,
)
from .whisper import (
    WhisperEncoder,
    WhisperEncoderLayer,
    _create_fake_bias_for_k_proj,
)

_AUDIO_PLACEHOLDER_OVERRIDE = "<|audio|>"
_MAX_ENCODER_BATCH_SIZE = 16


class UltravoxAudioFeatureInputs(TensorSchema):
    """Dimensions:
    - b: batch size
    - n: number of chunks
    - t: Time frames (M)
    - nmb: Number of mel bins
    """

    type: Literal["audio_features"]
    data: Annotated[
        torch.Tensor | list[torch.Tensor] | list[list[torch.Tensor]],
        TensorShape("bn", "nmb", "t"),
    ]
    lens: Annotated[torch.Tensor, TensorShape("bn")]
    """
    Length of the audio frames per chunk. Used for attention mask in WhisperEncoder.
    """
    token_len: Annotated[torch.Tensor, TensorShape("bn")]
    """Length of the audio tokens per chunk. Used for flattening the audio features."""
    num_chunks: Annotated[torch.Tensor, TensorShape("n")]
    """Number of chunks per audio. Used for flattening the audio features."""


class UltravoxAudioEmbeddingInputs(TensorSchema):
    """Dimensions:
    - b: batch size
    - na: number of audios
    - afs: audio feature size
    - hs: hidden size
    """

    type: Literal["audio_embeds"]
    data: Annotated[
        torch.Tensor | list[torch.Tensor], TensorShape("b", "na", "afs", "hs")
    ]


UltravoxAudioInputs: TypeAlias = (
    UltravoxAudioFeatureInputs | UltravoxAudioEmbeddingInputs
)


class UltravoxMultiModalDataParser(MultiModalDataParser):
    embedding_fields = {
        "audio": {"audio_embeds": "values", "audio_num_tokens": "metadata"},
    }


class UltravoxProcessingInfo(BaseProcessingInfo):
    def get_hf_processor(self, **kwargs: object) -> ProcessorMixin:
        config = self.ctx.model_config.hf_config
        hf_processor = self.ctx.get_hf_processor(**kwargs)

        # NOTE: Ultravox processing definition uses '<|eot_id|>' as the
        # placeholder that will cause confusion with the actual end of turn
        # token, thus we override placeholder with a reserved token.
        hf_processor.audio_token_replacement = _AUDIO_PLACEHOLDER_OVERRIDE
        hf_processor.audio_replacement_token_id = config.audio_token_index

        return hf_processor

    def get_feature_extractor(self, **kwargs: object) -> WhisperFeatureExtractor:
        hf_processor = self.get_hf_processor(**kwargs)

        # Changed in https://huggingface.co/fixie-ai/ultravox-v0_5-llama-3_2-1b/commit/9a3c571b8fdaf1e66dd3ea61bbcb6db5c70a438e
        audio_processor = hf_processor.audio_processor  # type: ignore
        if isinstance(audio_processor, WhisperFeatureExtractor):
            return audio_processor

        feature_extractor = audio_processor.feature_extractor  # type: ignore
        assert isinstance(feature_extractor, WhisperFeatureExtractor)
        return feature_extractor

    def get_default_tok_params(self) -> TokenizeParams:
        return super().get_default_tok_params().with_kwargs(add_special_tokens=False)

    def get_data_parser(self):
        feature_extractor = self.get_feature_extractor()

        return UltravoxMultiModalDataParser(
            target_sr=feature_extractor.sampling_rate,
            target_channels=self.get_target_channels(),
            expected_hidden_size=self._get_expected_hidden_size(),
            allow_missing_mm_embeddings=self.allow_missing_mm_embeddings,
        )

    def get_target_channels(self) -> int:
        """Return target audio channels for Ultravox models (mono)."""
        return 1

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}


class UltravoxDummyInputsBuilder(BaseDummyInputsBuilder[UltravoxProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)

        return "<|audio|>" * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: MultiModalDummyOptions,
    ) -> MultiModalDataDict:
        feature_extractor = self.info.get_feature_extractor()

        sampling_rate = feature_extractor.sampling_rate
        audio_len = (
            feature_extractor.chunk_length * sampling_rate * _MAX_ENCODER_BATCH_SIZE
        )

        return {
            "audio": self._get_dummy_audios(
                length=audio_len,
                num_audios=mm_counts.get("audio", 0),
                overrides=mm_options.get("audio"),
            )
        }


class UltravoxMultiModalProcessor(BaseMultiModalProcessor[UltravoxProcessingInfo]):
    def _get_hf_mm_text(self, mm_counts: Mapping[str, int]) -> str:
        return self.dummy_inputs.get_dummy_text(mm_counts)

    def _get_hf_mm_inputs(
        self,
        mm_items: MultiModalDataItems,
        hf_kwargs: Mapping[str, object],
    ) -> HFMultiModalInputs:
        hf_inputs = super()._get_hf_mm_inputs(mm_items, hf_kwargs)

        # The Ultravox processor accepts "audios" instead of "audio"
        hf_data = hf_inputs.hf_data
        if "audio" in hf_data:
            hf_data["audios"] = hf_data.pop("audio")

        feature_extractor = self.info.get_feature_extractor(**hf_kwargs)
        normalized_kwargs = dict(hf_inputs.hf_kwargs)
        # The remote processor already hardcodes truncation=False and forwards
        # its kwargs to the audio processor, where another value would collide.
        normalized_kwargs.pop("truncation", None)
        return hf_inputs._replace(
            hf_kwargs=dict(
                normalized_kwargs,
                sampling_rate=feature_extractor.sampling_rate,
                include_audio_num_chunks=True,
            )
        )

    def _postprocess_hf_mm_data(
        self,
        mm_data: Mapping[str, object],
        hf_processor_mm_kwargs: Mapping[str, object],
        processed_data: BatchFeature,
    ) -> BatchFeature:
        if "audio_values" in processed_data:
            processed_data["audio_features"] = processed_data.pop("audio_values")

        return processed_data

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        num_chunks = hf_inputs.get("audio_num_chunks", torch.zeros(0))
        return dict(
            # to handle longer than 30s audio, each audio might be split
            # into multiple chunks as such, their batch dimension can be
            # higher than the number of audio samples
            audio_features=MultiModalFieldConfig.flat_from_sizes("audio", num_chunks),
            audio_token_len=MultiModalFieldConfig.flat_from_sizes("audio", num_chunks),
            # Only ever used to derive the encoder attention metadata on the
            # host, so keep it there.
            audio_lens=MultiModalFieldConfig.flat_from_sizes(
                "audio", num_chunks, keep_on_cpu=True
            ),
            # num_chunks can convert audio_chunked to audio batch dimension
            audio_num_chunks=MultiModalFieldConfig.batched("audio", keep_on_cpu=True),
            audio_embeds=MultiModalFieldConfig.batched("audio"),
            audio_num_tokens=MultiModalFieldConfig.batched("audio", keep_on_cpu=True),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()

        replacement_id = hf_processor.audio_replacement_token_id  # type: ignore

        # Each audio can be split into multiple chunks.
        # chunks_start_idx[i] indicates the start index of the chunks
        # belonging to the i-th audio.
        out_mm_data = out_mm_kwargs.get_data()
        num_chunks = out_mm_data.get("audio_num_chunks", torch.zeros(0))
        chunks_start_idx: torch.Tensor = torch.cumsum(
            num_chunks, dim=0, dtype=torch.int32
        )
        chunks_start_idx = torch.cat(
            [torch.tensor([0], dtype=torch.int32), chunks_start_idx]
        )

        def get_replacement_ultravox(item_idx: int):
            if "audio_num_tokens" in out_mm_data:
                audio_num_tokens = out_mm_data["audio_num_tokens"]
                assert isinstance(audio_num_tokens, torch.Tensor)
                return [replacement_id] * int(audio_num_tokens[item_idx])
            start = chunks_start_idx[item_idx]
            end = chunks_start_idx[item_idx + 1]
            audio_token_lens = out_mm_data["audio_token_len"]
            assert isinstance(audio_token_lens, torch.Tensor)
            audio_token_len = audio_token_lens[start:end].sum()
            return [replacement_id] * int(audio_token_len)  # type: ignore

        return [
            PromptReplacement(
                modality="audio",
                # `<|audio|>` is paired with `replacement_id` in
                # `UltravoxProcessingInfo.get_hf_processor`, but it is not
                # guaranteed to be a single vocab entry, so match the encoded
                # placeholder text instead of `replacement_id`
                target=cached_encode(
                    tokenizer,
                    hf_processor.audio_token_replacement,  # type: ignore
                    add_special_tokens=False,
                ),
                replacement=get_replacement_ultravox,
            )
        ]


class StackAudioFrames(nn.Module):
    """Stack the audio embedding frames to reduce the sequence length by a factor
    of `stack_factor`.
    """

    def __init__(self, stack_factor: int = 8):
        super().__init__()
        self.stack_factor = stack_factor

    def forward(self, audio_embeds: torch.Tensor) -> torch.Tensor:
        B, T, C = audio_embeds.shape
        T_pad = (T + self.stack_factor - 1) // self.stack_factor * self.stack_factor
        audio_embeds = F.pad(audio_embeds, (0, 0, 0, T_pad - T))
        B, T, C = audio_embeds.shape
        audio_embeds = audio_embeds.view(
            B, T // self.stack_factor, C * self.stack_factor
        )
        return audio_embeds


class _ChunkPackPlan(NamedTuple):
    """Host-side gather plan for one rectangular mel batch.

    `pack_index` selects conv frames `t < feat_len[b]` from row-major
    `[B, T_conv]`. Zero-length rows are omitted. `cu_seqlens` has one
    segment per kept row.
    """

    pack_index: torch.Tensor
    feat_lens: list[int]
    positive_feat_lens: list[int]
    cu_seqlens: torch.Tensor
    max_seqlen: int
    t_conv: int


def _feat_len(mel_len: int) -> int:
    """Conv2 frames for a mel length. Matches `(L - 1) // 2 + 1`."""
    return (int(mel_len) - 1) // 2 + 1


def _connector_tokens(feat_len: int, stack_factor: int) -> int:
    if feat_len <= 0:
        return 0
    return (feat_len + stack_factor - 1) // stack_factor


def _upper_bound_mm_lora_counts(
    num_mm_embeds: int,
    max_source_positions: int,
    stack_factor: int,
) -> tuple[int, int]:
    """Init-time punica budget. `num_mm_embeds` is the encoder token budget.

    Both tower terms are required. Full 1500-frame chunks can exceed
    `num_mm_embeds * stack_factor` (`8192` -> `66000`), and short chunks can
    exceed `num_chunks * max_source_positions` (`3008` -> `24064`).
    """
    if num_mm_embeds <= 0 or max_source_positions <= 0 or stack_factor <= 0:
        return 0, 0
    max_tokens_per_chunk = math.ceil(max_source_positions / stack_factor)
    num_chunks = math.ceil(num_mm_embeds / max_tokens_per_chunk)
    tower = max(num_chunks * max_source_positions, num_mm_embeds * stack_factor)
    connector = num_chunks * max_tokens_per_chunk
    return tower, connector


def _audio_lens_tensor(mm_kwargs: Mapping[str, Any]) -> torch.Tensor | None:
    elem = mm_kwargs.get("audio_lens")
    if elem is None:
        return None
    # Field elements store the tensor on `.data`. A bare tensor's `.data`
    # is a storage alias, so only unwrap real field objects.
    data = elem.data if hasattr(elem, "field") else elem
    if isinstance(elem, torch.Tensor):
        data = elem
    if not isinstance(data, torch.Tensor) or data.numel() == 0:
        return None
    return data


def _counts_from_audio_lens(
    audio_lens: torch.Tensor, stack_factor: int
) -> tuple[int, int]:
    """Per-item tower and connector rows from host mel lengths.

    Tower is `sum((L - 1) // 2 + 1)`. Connector is `sum(ceil(feat / stack))`.
    A device tensor is not copied back; callers keep `audio_lens` on the host.
    """
    if audio_lens.device.type != "cpu":
        raise RuntimeError(
            "Ultravox LoRA token counts need host audio_lens. "
            "Refusing to copy a device length tensor back to the host."
        )
    tower = 0
    connector = 0
    for mel_len in audio_lens.reshape(-1).to(dtype=torch.int64).tolist():
        feat = _feat_len(mel_len)
        if feat <= 0:
            continue
        tower += feat
        connector += _connector_tokens(feat, stack_factor)
    return tower, connector


def _build_cpu_pack_plan(audio_lens: torch.Tensor, mel_t: int) -> _ChunkPackPlan:
    """Gather index for valid conv frames. `audio_lens` must already be on CPU."""
    if audio_lens.device.type != "cpu":
        raise RuntimeError(
            "The Ultravox pack index is built from host audio_lens "
            "before the tower runs."
        )
    mel_lengths = audio_lens.reshape(-1).to(dtype=torch.int64).tolist()
    t_conv = (mel_t - 1) // 2 + 1
    feat_lens: list[int] = []
    positive: list[int] = []
    parts: list[torch.Tensor] = []
    for row, mel_len in enumerate(mel_lengths):
        feat = _feat_len(mel_len)
        if feat < 0:
            feat = 0
        feat_lens.append(feat)
        if feat <= 0:
            continue
        if feat > t_conv:
            raise ValueError(
                f"Mel length {mel_len} expands to {feat} frames, but the "
                f"padded mel width {mel_t} only has {t_conv} conv frames."
            )
        base = row * t_conv
        parts.append(torch.arange(base, base + feat, dtype=torch.int64))
        positive.append(feat)
    pack_index = torch.cat(parts) if parts else torch.empty(0, dtype=torch.int64)
    cu_seqlens = torch.zeros(len(positive) + 1, dtype=torch.int32)
    running = 0
    for i, feat in enumerate(positive, start=1):
        running += feat
        cu_seqlens[i] = running
    max_seqlen = max(positive) if positive else 0
    return _ChunkPackPlan(
        pack_index=pack_index,
        feat_lens=feat_lens,
        positive_feat_lens=positive,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        t_conv=t_conv,
    )


def _zero_padded_conv1_tail(
    conv1_out: torch.Tensor, audio_lens: torch.Tensor
) -> torch.Tensor:
    """Zero conv1 time `t >= audio_lens` before conv2.

    That axis is still the mel length. Odd `L` makes conv2 read one past `L`;
    a later right-pad is nonzero there unless this tap is cleared. Zeroing
    `t > L`, or zeroing conv2-time, leaves the odd-length frame wrong.
    Host lengths are copied up with `async_tensor_h2d`. A device length tensor
    stays on device.
    """
    time = conv1_out.shape[-1]
    positions = torch.arange(time, device=conv1_out.device)
    flat_lens = audio_lens.reshape(-1)
    if flat_lens.device.type == "cpu":
        lens = async_tensor_h2d(
            flat_lens.to(dtype=torch.int64),
            device=conv1_out.device,
            dtype=torch.int64,
        )
    elif flat_lens.dtype == torch.int64:
        lens = flat_lens
    else:
        lens = flat_lens.to(dtype=torch.int64)
    mask = positions.unsqueeze(0) >= lens.unsqueeze(1)
    return conv1_out.masked_fill(mask.unsqueeze(1), 0)


def _flashinfer_batch_bucket(batch_size: int) -> int:
    for bucket in FLASHINFER_BATCH_BUCKETS:
        if bucket >= batch_size:
            return bucket
    return round_up(batch_size, FLASHINFER_BATCH_BUCKETS[0])


def _prepare_flashinfer_cu_seqlens(
    cu_seqlens_cpu: torch.Tensor,
    *,
    hidden_size: int,
    tp_size: int,
    fp8_padded_hidden_size: int | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale and batch-bucket pad FlashInfer offsets with tensor ops.

    `sequence_lengths` is padded with 0 out to the `add_padding_to_seqlens`
    bucket. Each cumulative vector (length `batch + 1`) is padded to
    `bucket + 1` by repeating its last value. The BF16 path concatenates the
    Q/K section with a `* 3` V section. FP8 uses
    `fp8_padded_hidden_size // tp_size` for both sections and does not
    multiply by 3. Bucketing does not change which frames attend.
    """
    batch_size = int(cu_seqlens_cpu.shape[0] - 1)
    cu = async_tensor_h2d(cu_seqlens_cpu, device=device, dtype=torch.int32)
    seq_lens = cu[1:] - cu[:-1]
    bucket = _flashinfer_batch_bucket(batch_size)
    extra = bucket - batch_size
    if extra > 0:
        seq_lens = torch.cat(
            [
                seq_lens,
                torch.zeros(extra, dtype=seq_lens.dtype, device=seq_lens.device),
            ]
        )

    def _pad_cumulative(cum: torch.Tensor) -> torch.Tensor:
        if extra <= 0:
            return cum
        return torch.cat([cum, cum[-1:].expand(extra)])

    if fp8_padded_hidden_size is not None:
        scale = fp8_padded_hidden_size // tp_size
        scaled = _pad_cumulative(cu * scale)
        return torch.cat([scaled, scaled]), seq_lens

    scale = hidden_size // tp_size
    scaled = cu * scale
    return torch.cat([_pad_cumulative(scaled), _pad_cumulative(scaled * 3)]), seq_lens


_DEVICE_CU_BACKENDS = {
    AttentionBackendEnum.FLASH_ATTN,
    AttentionBackendEnum.ROCM_AITER_FA,
    AttentionBackendEnum.TRITON_ATTN,
    AttentionBackendEnum.FLASHINFER,
}


def _build_packed_attn_metadata(
    attn: MMEncoderAttention,
    cu_seqlens_cpu: torch.Tensor,
    max_seqlen: int,
    hidden_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor | None]:
    """Attention metadata for one packed segment per positive chunk.

    `max_seqlen` is wrapped as a 0-dim CPU tensor so the flash wrapper's
    `.item()` does not touch CUDA. FlashInfer buckets that integer on the
    host first. SDPA keeps the unscaled `cu_seqlens` on CPU; its wrapper
    `.tolist()` is not done here.
    """
    if cu_seqlens_cpu.device.type != "cpu":
        raise RuntimeError("packed cu_seqlens must be built on the host")
    cu_seqlens_cpu = cu_seqlens_cpu.to(dtype=torch.int32)
    backend = attn.attn_backend
    if backend == AttentionBackendEnum.FLASHINFER:
        max_seqlen = bucket_flashinfer_max_seqlen(max_seqlen)
    max_seqlen_t = torch.tensor(max_seqlen, dtype=torch.int32)

    if backend not in _DEVICE_CU_BACKENDS:
        return {
            "cu_seqlens": cu_seqlens_cpu,
            "max_seqlen": max_seqlen_t,
            "sequence_lengths": None,
        }

    if backend != AttentionBackendEnum.FLASHINFER:
        return {
            "cu_seqlens": async_tensor_h2d(
                cu_seqlens_cpu, device=device, dtype=torch.int32
            ),
            "max_seqlen": max_seqlen_t,
            "sequence_lengths": None,
        }

    tp_size = get_tensor_model_parallel_world_size()
    fp8_padded_hidden_size = None
    if getattr(attn, "fp8_enabled", False):
        fp8_padded_hidden_size = get_fp8_padded_hidden_size(
            attn.num_heads * tp_size,
            attn.head_size,
        )
    cu_seqlens, sequence_lengths = _prepare_flashinfer_cu_seqlens(
        cu_seqlens_cpu,
        hidden_size=hidden_size,
        tp_size=tp_size,
        fp8_padded_hidden_size=fp8_padded_hidden_size,
        device=device,
    )
    return {
        "cu_seqlens": cu_seqlens,
        "max_seqlen": max_seqlen_t,
        "sequence_lengths": sequence_lengths,
    }


def _positive_feat_lens(feat_lens: Sequence[int] | torch.Tensor) -> list[int]:
    if isinstance(feat_lens, torch.Tensor):
        if feat_lens.device.type != "cpu":
            raise RuntimeError("feat_lens must stay on the host")
        values = feat_lens.reshape(-1).to(dtype=torch.int64).tolist()
    else:
        values = [int(v) for v in feat_lens]
    return [v for v in values if v > 0]


def _stack_packed_frames(
    stack: StackAudioFrames,
    packed_frames: torch.Tensor,
    feat_lens: Sequence[int] | torch.Tensor,
) -> tuple[torch.Tensor, list[int]]:
    """Stack each chunk on its own, then concatenate connector tokens.

    The stack is not applied to the concatenated pack, so a short chunk cannot
    absorb the next chunk's frames. The last token is zero-filled when
    `feat_len % stack_factor != 0` (at most `stack_factor - 1` frames).
    """
    lengths = _positive_feat_lens(feat_lens)
    stacked_dim = packed_frames.shape[-1] * stack.stack_factor
    if not lengths:
        return packed_frames.new_empty((0, stacked_dim)), []
    tokens: list[torch.Tensor] = []
    conn_lens: list[int] = []
    for chunk in packed_frames.split(lengths, dim=0):
        stacked = stack(chunk.unsqueeze(0)).squeeze(0)
        tokens.append(stacked)
        conn_lens.append(int(stacked.shape[0]))
    return torch.cat(tokens, dim=0), conn_lens


def _chunk_position_ids(conn_lens: Sequence[int]) -> torch.Tensor:
    total = sum(conn_lens)
    pos = torch.empty(total, dtype=torch.long)
    offset = 0
    for length in conn_lens:
        pos[offset : offset + length] = torch.arange(length, dtype=torch.long)
        offset += length
    return pos


def _cu_from_lengths(lengths: Sequence[int]) -> tuple[torch.Tensor, int]:
    cu = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    running = 0
    for i, length in enumerate(lengths, start=1):
        running += int(length)
        cu[i] = running
    max_seqlen = max(lengths) if lengths else 0
    return cu, int(max_seqlen)


# Maps HF whisper encoder layer names to vLLM's fused qkv_proj and nested
# mlp; `_create_fake_bias_for_k_proj` supplies the fused bias' k-slice
# (HF whisper k_proj has no bias).
_WHISPER_ENCODER_MAPPER = WeightsMapper(
    orig_to_new_substr={".fc1.": ".mlp.fc1.", ".fc2.": ".mlp.fc2."},
    orig_to_new_stacked={
        ".self_attn.q_proj": (".self_attn.qkv_proj", "q"),
        ".self_attn.k_proj": (".self_attn.qkv_proj", "k"),
        ".self_attn.v_proj": (".self_attn.qkv_proj", "v"),
    },
)


def _load_whisper_layer_weights(
    module: nn.Module, weights: Iterable[tuple[str, torch.Tensor]]
) -> set[str]:
    weights = _create_fake_bias_for_k_proj(weights, ".self_attn.k_proj.weight")
    loader = AutoWeightsLoader(module)
    return loader.load_weights(weights, mapper=_WHISPER_ENCODER_MAPPER)


class UltravoxFeedForwardProjector(nn.Module):
    def __init__(
        self,
        config: UltravoxConfig,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self._pad_and_stack = StackAudioFrames(config.stack_factor)
        dim_in = config.audio_config.hidden_size * config.stack_factor
        self.ln_pre = RMSNorm(dim_in)
        # ReplicatedLinear (a vLLM-native linear) so the connector can be
        # wrapped for LoRA; a plain nn.Linear is left unwrapped by from_layer.
        self.linear_1 = ReplicatedLinear(
            dim_in,
            self.hidden_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_1",
        )
        dim_mid = self.hidden_dim

        if config.projector_act == "swiglu":
            self.act = MulAndSilu()
            dim_mid = dim_mid // 2
        else:
            self.act = get_act_fn(config.projector_act)

        dim_out = config.text_config.hidden_size
        self.linear_2 = ReplicatedLinear(
            dim_mid,
            dim_out,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_2",
        )

        # Ultravox v0.4.1 and below use layer_norm after the second linear layer
        # while v0.5.0 and above uses layer_norm after the first linear layer.
        if config.projector_ln_mid:
            self.ln_mid: nn.Module = RMSNorm(dim_mid)
            self.ln_post = nn.Identity()
        else:
            self.ln_mid = nn.Identity()
            self.ln_post = RMSNorm(dim_out)

    def _project_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.ln_pre(hidden)
        hidden, _ = self.linear_1(hidden)
        hidden = self.act(hidden)
        hidden = self.ln_mid(hidden)
        hidden, _ = self.linear_2(hidden)
        hidden = self.ln_post(hidden)
        return hidden

    def forward(
        self,
        audio_features: torch.Tensor,
        audio_token_len: torch.Tensor,
        feat_lens: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        # `audio_token_len` is not read back. Packed rows come from host
        # `feat_lens`. A device length falls through to the rectangle path.
        del audio_token_len
        if feat_lens is not None and audio_features.dim() == 2:
            tokens, _ = _stack_packed_frames(
                self._pad_and_stack, audio_features, feat_lens
            )
            return self._project_hidden(tokens)
        return self._project_hidden(self._pad_and_stack(audio_features))


class UltravoxTransformerProjector(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        prefix: str = "",
    ):
        super().__init__()
        config: UltravoxConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        audio_config = config.audio_config

        self._pad_and_stack = StackAudioFrames(config.stack_factor)
        dim_in = audio_config.hidden_size * config.stack_factor

        self.ln_pre = RMSNorm(dim_in)
        self.linear_in = ReplicatedLinear(
            dim_in,
            audio_config.d_model,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_in",
        )

        self.embed_positions = nn.Embedding(
            audio_config.max_source_positions,
            audio_config.d_model,
        )

        audio_vllm_config = vllm_config.with_hf_config(
            audio_config, architectures=["UltravoxModel"]
        )
        self.layers = nn.ModuleList(
            [
                WhisperEncoderLayer(
                    vllm_config=audio_vllm_config,
                    prefix=f"{prefix}.layers.{idx}",
                )
                for idx in range(config.num_projector_layers)
            ]
        )
        self._d_model = audio_config.d_model

        self.ln_post = RMSNorm(audio_config.d_model)
        self.linear_out = ReplicatedLinear(
            audio_config.d_model,
            config.text_config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_out",
        )

    def _run_layers(
        self,
        hidden_states: torch.Tensor,
        conn_lens: Sequence[int],
        *,
        per_chunk_positions: bool,
    ) -> torch.Tensor:
        if per_chunk_positions:
            pos_ids = async_tensor_h2d(
                _chunk_position_ids(conn_lens),
                device=hidden_states.device,
                dtype=torch.long,
            )
            hidden_states = hidden_states + self.embed_positions(pos_ids)
            cu_seqlens, max_seqlen = _cu_from_lengths(conn_lens)
        else:
            # Device lengths are not read. `max_seqlen` is this tensor's time
            # axis, not the flattened row count.
            seq_len = hidden_states.shape[-2]
            positions = self.embed_positions(
                torch.arange(seq_len, device=hidden_states.device)
            )
            hidden_states = hidden_states + positions
            batch_size = hidden_states.shape[0]
            cu_seqlens = torch.arange(
                0,
                (batch_size + 1) * seq_len,
                step=max(seq_len, 1),
                dtype=torch.int32,
            )
            if seq_len == 0:
                cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32)
            max_seqlen = seq_len
            hidden_states = hidden_states.reshape(
                batch_size * seq_len, hidden_states.shape[-1]
            )

        if hidden_states.shape[0] == 0 or len(self.layers) == 0:
            return hidden_states

        attn_metadata = _build_packed_attn_metadata(
            self.layers[0].self_attn.attn,
            cu_seqlens,
            max_seqlen,
            hidden_states.shape[-1],
            hidden_states.device,
        )
        for layer in self.layers:
            hidden_states = layer(hidden_states, **attn_metadata)
        return hidden_states

    def forward(
        self,
        audio_features: torch.Tensor,
        audio_token_len: torch.Tensor,
        feat_lens: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        del audio_token_len
        if feat_lens is not None and audio_features.dim() == 2:
            tokens, conn_lens = _stack_packed_frames(
                self._pad_and_stack, audio_features, feat_lens
            )
            hidden_states = self.ln_pre(tokens)
            hidden_states, _ = self.linear_in(hidden_states)
            hidden_states = self._run_layers(
                hidden_states, conn_lens, per_chunk_positions=True
            )
        else:
            audio_features = self._pad_and_stack(audio_features)
            hidden_states = self.ln_pre(audio_features)
            hidden_states, _ = self.linear_in(hidden_states)
            batch_size, seq_len, _ = hidden_states.shape
            hidden_states = self._run_layers(
                hidden_states,
                [seq_len] * batch_size,
                per_chunk_positions=False,
            )
            hidden_states = hidden_states.reshape(batch_size, seq_len, -1)

        hidden_states = self.ln_post(hidden_states)
        hidden_states, _ = self.linear_out(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return _load_whisper_layer_weights(self, weights)


class UltravoxWhisperEncoder(WhisperEncoder):
    """Ultravox's `ModifiedWhisperEncoder` on top of vLLM's `WhisperEncoder`.

    Like the original (a modified HF whisper encoder,
    see https://github.com/huggingface/transformers/issues/25744), it accepts
    mel inputs shorter than 30s. Valid conv frames are packed from a host
    index built before this tower runs, and attention is one segment per
    positive chunk. The linears are vLLM-native so the tower can be wrapped
    for LoRA. Packed length varies with `audio_lens`, so this tower does not
    use the Whisper encoder CUDA-graph protocol.
    """

    @property
    def dtype(self) -> torch.dtype:
        return self.conv1.weight.dtype

    @property
    def max_context_length(self) -> int:
        return self.max_source_positions * self.total_stride

    def _get_feat_extract_output_lengths(
        self, input_lengths: torch.Tensor
    ) -> torch.Tensor:
        return (input_lengths - 1) // 2 + 1

    # Ultravox requires chunk lengths in addition to Whisper's input features.
    def forward(  # type: ignore[override]
        self,
        input_features: torch.Tensor,
        audio_lens: torch.Tensor,
        pack: _ChunkPackPlan | None = None,
    ) -> torch.Tensor:
        expected_seq_length = self.max_context_length
        mel_t = input_features.shape[-1]
        if mel_t > expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length "
                f"{expected_seq_length} or less, but found "
                f"{mel_t}. Make sure to pad the input mel "
                f"features to {expected_seq_length}."
            )

        # Host lengths are already on CPU (`keep_on_cpu`). Build the gather
        # index before any conv so this forward does not read them back.
        if pack is None and audio_lens.device.type == "cpu":
            pack = _build_cpu_pack_plan(audio_lens, mel_t)

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = _zero_padded_conv1_tail(inputs_embeds, audio_lens)
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        # Python int. A device-length fallback uses this, not `flat.shape[0]`.
        t_conv = inputs_embeds.shape[-2]
        embed_pos = self.embed_positions.weight[:t_conv]

        hidden_states = (inputs_embeds + embed_pos).to(inputs_embeds.dtype)

        batch_size, seq_len, d_model = hidden_states.shape
        flat = hidden_states.reshape(batch_size * seq_len, d_model)
        if pack is None:
            return self._forward_full_rows(flat, batch_size, t_conv, d_model)

        if t_conv != pack.t_conv:
            raise RuntimeError(
                f"conv2 time {t_conv} does not match the host pack plan "
                f"({pack.t_conv})."
            )
        if pack.pack_index.numel() == 0:
            return flat.new_empty((0, d_model))

        index = async_tensor_h2d(pack.pack_index, device=flat.device, dtype=torch.long)
        packed = flat.index_select(0, index)
        packed = self._run_packed_layers(
            packed, pack.cu_seqlens, pack.max_seqlen, d_model
        )
        return self.layer_norm(packed)

    def _run_packed_layers(
        self,
        hidden: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        hidden_size: int,
    ) -> torch.Tensor:
        if hidden.shape[0] == 0 or len(self.layers) == 0:
            return hidden
        attn_metadata = _build_packed_attn_metadata(
            self.layers[0].self_attn.attn,
            cu_seqlens,
            max_seqlen,
            hidden_size,
            hidden.device,
        )
        for encoder_layer in self.layers:
            hidden = encoder_layer(hidden, **attn_metadata)
        return hidden

    def _forward_full_rows(
        self,
        flat: torch.Tensor,
        batch_size: int,
        t_conv: int,
        d_model: int,
    ) -> torch.Tensor:
        """Device `audio_lens` stayed on device, so the rows are not packed.

        `max_seqlen` is the conv2 time axis (`[B, T, D]`), not `flat.shape[0]`.
        """
        if t_conv == 0:
            cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32)
        else:
            cu_seqlens = torch.arange(
                0, (batch_size + 1) * t_conv, step=t_conv, dtype=torch.int32
            )
        flat = self._run_packed_layers(flat, cu_seqlens, t_conv, d_model)
        flat = self.layer_norm(flat)
        return flat.reshape(batch_size, t_conv, d_model)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return _load_whisper_layer_weights(self, weights)


@MULTIMODAL_REGISTRY.register_processor(
    UltravoxMultiModalProcessor,
    info=UltravoxProcessingInfo,
    dummy_inputs=UltravoxDummyInputsBuilder,
)
class UltravoxModel(nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "audio_tower.model.encoder.": "audio_tower.",
            # A whisper checkpoint loaded via `audio_model_id` also carries
            # decoder and LM-head weights the tower does not use.
            "audio_tower.model.": None,
            "audio_tower.proj_out.": None,
        }
    )

    supports_tower_connector_lora = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return "<|audio|>"

        raise ValueError("Only audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: UltravoxConfig = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.multi_modal_config = multimodal_config
        assert self.multi_modal_config

        # Tower/connector LoRA installs one punica map for the whole scheduled
        # encoder batch. Splitting that batch would apply the wrong ids, so
        # the tower and projector each run once. Lengths stay the real mel
        # lengths; chunks are not padded to 3000.
        self.tower_connector_lora = bool(
            lora_config is not None and lora_config.enable_tower_connector_lora
        )

        self.configure_mm_token_handling(
            self.config.vocab_size,
            [self.config.audio_token_index],
        )

        # The towers live in their own Hub repos, so the revision of the
        # Ultravox repo does not apply to them.
        self.secondary_weights = []
        if config.audio_model_id is not None:
            # this prefix is not for initialization, but for loading weights
            # note the trailing dot
            self.secondary_weights.append(
                DefaultModelLoader.Source(
                    model_or_path=config.audio_model_id,
                    revision=None,
                    prefix="audio_tower.",
                )
            )
        if config.text_model_id is not None:
            # this prefix is not for initialization, but for loading weights
            # note the trailing dot
            self.secondary_weights.append(
                DefaultModelLoader.Source(
                    model_or_path=config.text_model_id,
                    revision=None,
                    prefix="language_model.",
                )
            )

        with self._mark_tower_model(vllm_config, "audio"):
            self.audio_tower = UltravoxWhisperEncoder(
                vllm_config=vllm_config.with_hf_config(
                    config.audio_config, architectures=["UltravoxModel"]
                ),
                prefix=maybe_prefix(prefix, "audio_tower"),
                enable_pp=False,
            )
            if config.num_projector_layers > 0:
                self.multi_modal_projector = UltravoxTransformerProjector(
                    vllm_config,
                    prefix=maybe_prefix(prefix, "multi_modal_projector"),
                )
            else:
                self.multi_modal_projector = UltravoxFeedForwardProjector(
                    config,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "multi_modal_projector"),
                )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.wrapped_model_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def get_mm_mapping(self) -> MultiModelKeys:
        """Get the module prefix in multimodal models."""
        return MultiModelKeys.from_string_field(
            language_model="language_model.",
            connector="multi_modal_projector.",
            tower_model="audio_tower.",
        )

    def _get_max_tokens_per_chunk(self) -> int:
        # Audio is chunked at the tower's full context (30s -> 1500 encoder
        # frames); the projector stacks `stack_factor` frames per LM token, so
        # a full chunk yields ceil(max_source_positions / stack_factor) tokens
        # and only an audio's last chunk can yield fewer.
        return math.ceil(
            self.config.audio_config.max_source_positions / self.config.stack_factor
        )

    def get_mm_lora_token_counts(
        self,
        *,
        modality: str,
        mm_kwargs: MultiModalKwargsItem | None,
        num_mm_embeds: int,
    ) -> tuple[int, int | None]:
        del modality
        if mm_kwargs is None:
            # Init sizing. `num_mm_embeds` is the encoder budget. Do not read
            # `audio_lens` and do not return a packed sum or `(0, 0)`.
            return _upper_bound_mm_lora_counts(
                num_mm_embeds,
                self.config.audio_config.max_source_positions,
                self.config.stack_factor,
            )
        audio_lens = _audio_lens_tensor(mm_kwargs)
        if audio_lens is None:
            # `(0, None)` would make `set_active_mm_loras` return before the
            # connector map is installed for a later audio item.
            return 0, 0
        return _counts_from_audio_lens(audio_lens, self.config.stack_factor)

    def _audio_features_to_embeddings(
        self,
        input_features: torch.Tensor,
        audio_lens: torch.Tensor,
        audio_token_len: torch.Tensor,
    ) -> torch.Tensor:
        audio_features = input_features.to(self.audio_tower.dtype)
        batch_size = audio_features.size(0)
        audio_embeddings = []

        # With tower/connector LoRA, punica maps the first `x.size(0)` entries
        # of the token-LoRA mapping onto each linear. One pass keeps those
        # rows in scheduler order. Otherwise cap the batch for memory.
        encoder_batch_size = (
            max(batch_size, 1) if self.tower_connector_lora else _MAX_ENCODER_BATCH_SIZE
        )
        for start in range(0, batch_size, encoder_batch_size):
            end = min(start + encoder_batch_size, batch_size)
            batch_features = audio_features[start:end]
            batch_lens = audio_lens[start:end]
            pack = None
            if batch_lens.device.type == "cpu":
                pack = _build_cpu_pack_plan(batch_lens, batch_features.shape[-1])
            batch_hidden = self.audio_tower(batch_features, batch_lens, pack)
            batch_hidden = batch_hidden.to(self.audio_tower.dtype)
            if pack is None:
                batch_embeddings = self.multi_modal_projector(
                    batch_hidden, audio_token_len[start:end]
                )
            else:
                batch_embeddings = self.multi_modal_projector(
                    batch_hidden,
                    audio_token_len[start:end],
                    feat_lens=pack.positive_feat_lens,
                )
            audio_embeddings.append(batch_embeddings)

        if not audio_embeddings:
            return audio_features.new_empty((0, 0))
        return torch.cat(audio_embeddings, dim=0)

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> UltravoxAudioInputs | None:
        audio_features = kwargs.pop("audio_features", None)
        audio_embeds = kwargs.pop("audio_embeds", None)
        audio_lens = kwargs.pop("audio_lens", None)
        audio_token_len = kwargs.pop("audio_token_len", None)
        audio_num_chunks = kwargs.pop("audio_num_chunks", None)

        if audio_features is None and audio_embeds is None:
            return None

        if audio_features is not None:
            return UltravoxAudioFeatureInputs(
                type="audio_features",
                data=audio_features,
                lens=audio_lens,
                token_len=audio_token_len,
                num_chunks=audio_num_chunks,
            )

        if audio_embeds is not None:
            return UltravoxAudioEmbeddingInputs(type="audio_embeds", data=audio_embeds)

        raise AssertionError("This line should be unreachable.")

    def _process_audio_input(
        self,
        audio_input: UltravoxAudioInputs,
    ) -> NestedTensors | tuple[torch.Tensor, ...]:
        if audio_input["type"] == "audio_embeds":
            return audio_input["data"]

        # Pad and concatenate audio features to the batch max only.
        # [[B1, 80, M1], [B2, 80, M2]] -> [B1+B2, 80, max(M1, M2)]
        # Over-long mels are not trimmed; the tower raises on them.
        audio_features = pad_and_concat_to_dim3(audio_input["data"])

        audio_lens = audio_input["lens"]
        audio_token_len = audio_input["token_len"]

        embeddings = self._audio_features_to_embeddings(
            audio_features, audio_lens, audio_token_len
        )

        # Packed connector rows are already the valid tokens, including a
        # zero-filled tail when `feat_len % stack_factor != 0`. A device-length
        # fallback is still a rectangle and is trimmed here.
        with gpu_sync_allowed():
            embed_lens = [
                chunk_lens.sum().item()
                for chunk_lens in audio_token_len.split(
                    audio_input["num_chunks"].tolist()
                )
            ]
            if embeddings.dim() == 2:
                return embeddings.split(embed_lens)

            max_len = embeddings.shape[1]
            indices = torch.arange(max_len, device=embeddings.device).expand(
                embeddings.shape[0], -1
            )
            mask = indices < audio_token_len[:, None]
            flattened_embeddings = embeddings[mask]
            return flattened_embeddings.split(embed_lens)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return []
        audio_embeddings = self._process_audio_input(audio_input)
        return audio_embeddings

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # This is to satisfy the type checker for each overload
        if multimodal_embeddings is None or is_multimodal is None:
            return super().embed_input_ids(input_ids)

        return super().embed_input_ids(
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Run forward pass for Ultravox.

        One key thing to understand is the `input_ids` already accounts for the
        positions of the to-be-inserted audio embeddings. The to-be-inserted
        audio has a size that is essentially 6.25 tokens per second of audio.

        This way, the `positions` and `attn_metadata` are consistent
        with the `input_ids`.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Position indices for the input tokens.
            intermediate_tensors: Intermediate tensors from prior forward pass.
            inputs_embeds: Optional tensor of input embeddings.
            **kwargs: Multimodal inputs for this batch, forwarded to the
                multimodal embedding path.

        """
        if intermediate_tensors is not None:
            inputs_embeds = None

        language_model = self.language_model
        if hasattr(language_model, "language_model"):
            language_model = language_model.language_model

        hidden_states = language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


def pad_and_concat_to_dim3(
    features: torch.Tensor | list[torch.Tensor] | list[list[torch.Tensor]],
) -> torch.Tensor:
    """Pad and concatenate a list of tensors.

    output:
        Tensor of shape [B, C, M] where M is the maximum length of the input
        tensors, B is the sum of the batch sizes of the input tensors.
        C must be the same for all input tensors.
    """
    if isinstance(features, torch.Tensor):
        if features.ndim > 3:
            # Flatten [B, N, 80, M] -> [B * N, 80, M]
            features = flatten_bn(features)

        return features

    tensors = [pad_and_concat_to_dim3(f) for f in features]

    max_len = max(f.shape[-1] for f in tensors)
    # Ensure all features have dim=3
    tensors = [f.view(-1, *f.shape[-2:]) for f in tensors]
    # Pad and concatenate:
    # [[B1, 80, M1], [B2, 80, M2]] -> [B1+B2, 80, max(M1, M2)]
    tensors = [F.pad(f, (0, max_len - f.shape[-1])) for f in tensors]
    return torch.cat(tensors)
