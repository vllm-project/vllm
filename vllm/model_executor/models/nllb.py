# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NLLB / M2M100 (``M2M100ForConditionalGeneration``) encoder-decoder model.

Encoder/decoder implementation and weight loading for the multilingual NLLB
family (``facebook/nllb-200-distilled-600M`` and the M2M100 checkpoints it
shares an architecture with). The source sentence is fed to the encoder as a
single "text" modality via an ``EncDecMultiModalProcessor`` (the same pattern
Whisper uses for audio), while the decoder runs as a normal V1 encoder-decoder.
The attention/QKV fusion, the source-text-as-modality processor, and the
tied-embedding weight loader share MarianMT's structure.

Differences from Marian (all reflected below):
  * **Pre-norm** layers: each ``*_layer_norm`` is applied *before* its sub-block,
    with the residual added after (Marian is post-norm).
  * A **final ``layer_norm``** after all layers in both the encoder and decoder
    (Marian has none).
  * Positional embeddings are **computed sinusoids** (tensor2tensor
    ``cat[sin, cos]`` layout) held in a **non-persistent buffer** -- they are NOT
    in the checkpoint. Position ids carry the M2M100 offset
    (``index + padding_idx + 1``); for NLLB ``padding_idx=1`` so the first real
    position is 2.
  * **No ``final_logits_bias``** (M2M100 has none); the LM head is the raw tied
    ``model.shared`` weight.
  * ``scale_embedding=True`` (``sqrt(d_model)``) and ``relu`` activation, both
    read from the config exactly as Marian reads its own.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
from torch import nn
from transformers import M2M100Config
from transformers.utils import logging

from vllm.config import CacheConfig, VllmConfig
from vllm.config.lora import LoRAConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.cross_attention import CrossAttention
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ModalityData,
    ModalityDataItems,
    ModalityDataParser,
    MultiModalDataItems,
    MultiModalDataParser,
    ProcessorBatchItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseProcessingInfo,
    EncDecMultiModalProcessor,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.collection_utils import is_list_of
from vllm.v1.attention.backend import AttentionType

from .interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsQuant,
)
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    cast_overflow_tensors,
    maybe_prefix,
)

logger = logging.get_logger(__name__)

# NLLB language tags are ISO-639-3 code + ``_`` + a 4-letter script, e.g.
# ``"deu_Latn"``. Used to pick the language tokens out of the tokenizer's added
# tokens (and to exclude control tokens such as ``<mask>``).
_LANG_CODE_RE = re.compile(r"^[a-z]{3}_[A-Z][a-z]{3}$")

# Friendly aliases -> NLLB-200 language tag. Lets callers pass a common
# ISO-639-1 code ("de") or an English language name ("German") instead of the
# full NLLB tag ("deu_Latn"). Only consulted for NLLB-200-style checkpoints (see
# ``_normalize_lang_code``); a candidate is accepted only if the checkpoint
# actually declares it, so this never invents an unsupported language.
_ISO_TO_NLLB: dict[str, str] = {
    "en": "eng_Latn",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "ru": "rus_Cyrl",
    "uk": "ukr_Cyrl",
    "cs": "ces_Latn",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "ro": "ron_Latn",
    "bg": "bul_Cyrl",
    "el": "ell_Grek",
    "hu": "hun_Latn",
    "fi": "fin_Latn",
    "sv": "swe_Latn",
    "da": "dan_Latn",
    "no": "nob_Latn",
    "nb": "nob_Latn",
    "nn": "nno_Latn",
    "is": "isl_Latn",
    "tr": "tur_Latn",
    "ar": "arb_Arab",
    "he": "heb_Hebr",
    "fa": "pes_Arab",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "ur": "urd_Arab",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "th": "tha_Thai",
    "vi": "vie_Latn",
    "id": "ind_Latn",
    "ms": "zsm_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "zh": "zho_Hans",
    "ca": "cat_Latn",
    "eu": "eus_Latn",
    "gl": "glg_Latn",
    "et": "est_Latn",
    "lv": "lvs_Latn",
    "lt": "lit_Latn",
    "hr": "hrv_Latn",
    "sr": "srp_Cyrl",
    "af": "afr_Latn",
    "sq": "als_Latn",
    "hy": "hye_Armn",
    "ka": "kat_Geor",
    "az": "azj_Latn",
    "kk": "kaz_Cyrl",
    "sw": "swh_Latn",
}
_NAME_TO_NLLB: dict[str, str] = {
    "english": "eng_Latn",
    "german": "deu_Latn",
    "french": "fra_Latn",
    "spanish": "spa_Latn",
    "italian": "ita_Latn",
    "portuguese": "por_Latn",
    "dutch": "nld_Latn",
    "polish": "pol_Latn",
    "russian": "rus_Cyrl",
    "ukrainian": "ukr_Cyrl",
    "czech": "ces_Latn",
    "romanian": "ron_Latn",
    "bulgarian": "bul_Cyrl",
    "greek": "ell_Grek",
    "hungarian": "hun_Latn",
    "finnish": "fin_Latn",
    "swedish": "swe_Latn",
    "danish": "dan_Latn",
    "norwegian": "nob_Latn",
    "turkish": "tur_Latn",
    "arabic": "arb_Arab",
    "hebrew": "heb_Hebr",
    "persian": "pes_Arab",
    "hindi": "hin_Deva",
    "bengali": "ben_Beng",
    "urdu": "urd_Arab",
    "thai": "tha_Thai",
    "vietnamese": "vie_Latn",
    "indonesian": "ind_Latn",
    "japanese": "jpn_Jpan",
    "korean": "kor_Hang",
    "chinese": "zho_Hans",
    "croatian": "hrv_Latn",
    "serbian": "srp_Cyrl",
    "swahili": "swh_Latn",
}

# M2M100. Unlike NLLB, M2M100 names its languages with bare ISO-639-1
# codes ("de", "fr", "zh") exposed through the tokenizer's ``lang_code_to_id``,
# each mapped to a ``__de__``-style token id. To normalize the SAME friendly
# inputs we accept for NLLB (English names, and NLLB-style tags a caller might
# reuse) onto M2M100's native ISO codes, we derive two reverse maps from the
# NLLB tables above -- keeping a single source of truth for the language list.
# ``setdefault`` keeps the primary ISO code when several map to one NLLB tag
# (e.g. "no"/"nb"/"nn" -> "nob_Latn" resolves back to "no").
_NLLB_TO_ISO: dict[str, str] = {}
for _iso, _tag in _ISO_TO_NLLB.items():
    _NLLB_TO_ISO.setdefault(_tag, _iso)
_NAME_TO_ISO: dict[str, str] = {
    _name: _NLLB_TO_ISO[_tag]
    for _name, _tag in _NAME_TO_NLLB.items()
    if _tag in _NLLB_TO_ISO
}

# Sentinel distinguishing "not yet computed" from a cached ``None`` result
# (``None`` legitimately means "this checkpoint is NLLB, not M2M100").
_UNSET = object()


class M2M100SinusoidalPositionalEmbedding(nn.Module):
    """Static sinusoidal positions, computed (not stored in the checkpoint).

    Mirrors HF ``M2M100SinusoidalPositionalEmbedding``: the tensor2tensor
    ``cat[sin, cos]`` layout, an ``offset`` of ``padding_idx + 1`` baked into the
    position ids (so the first real token maps to row ``padding_idx + 1``), and a
    zeroed ``padding_idx`` row. The table lives in a **non-persistent** buffer so
    it is neither reported missing (it is not a parameter) nor unexpected (it is
    not a checkpoint key).
    """

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        # HF reserves ``offset`` (=2) extra rows so index_select never overflows
        # for positions up to ``padding_idx + num_positions``.
        self.offset = 2
        weights = self._get_embedding(
            num_positions + self.offset, embedding_dim, padding_idx
        )
        self.register_buffer("weights", weights, persistent=False)

    @staticmethod
    def _get_embedding(
        num_embeddings: int, embedding_dim: int, padding_idx: int | None
    ) -> torch.Tensor:
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.int64).float() * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.int64).float().unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb.to(torch.get_default_dtype())

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        # vLLM passes 0-based absolute positions per sequence; apply the M2M100
        # offset (position id = index + padding_idx + 1) before the lookup.
        idx = positions + self.padding_idx + 1
        return self.weights.index_select(0, idx.view(-1))


class M2M100ScaledWordEmbedding(VocabParallelEmbedding):
    """Word embedding scaled by ``sqrt(d_model)`` (``scale_embedding=True``)."""

    def __init__(
        self, num_embeddings: int, embedding_dim: int, embed_scale: float = 1.0
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale


class M2M100EncoderAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        config: M2M100Config | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.d_model = config.d_model
        self.embed_dim = embed_dim
        self.total_num_heads = num_heads
        self.total_num_kv_heads = self.total_num_heads
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            self.d_model,
            self.d_model // self.total_num_heads,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
        )

        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=bias,
            quant_config=quant_config,
        )

        tp_world_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_world_size == 0
        self.num_heads = self.total_num_heads // tp_world_size

        if self.total_num_kv_heads >= tp_world_size:
            assert self.total_num_kv_heads % tp_world_size == 0
        else:
            assert tp_world_size % self.total_num_kv_heads == 0
        self.num_kv_heads = self.num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.attn = MMEncoderAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        is_2d = q.dim() == 2
        if is_2d:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)

        attn_output = self.attn(q, k, v)

        output, _ = self.out_proj(attn_output)
        if is_2d:
            output = output.squeeze(0)
        return output


class M2M100DecoderSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        config: M2M100Config | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.d_model = config.d_model
        self.embed_dim = embed_dim
        self.total_num_heads = num_heads
        self.total_num_kv_heads = self.total_num_heads
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            self.d_model,
            self.d_model // self.total_num_heads,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
        )

        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=bias,
            quant_config=quant_config,
        )

        tp_world_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_world_size == 0
        self.num_heads = self.total_num_heads // tp_world_size

        if self.total_num_kv_heads >= tp_world_size:
            assert self.total_num_kv_heads % tp_world_size == 0
        else:
            assert tp_world_size % self.total_num_kv_heads == 0
        self.num_kv_heads = self.num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=AttentionType.DECODER,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        attn_output = self.attn(q, k, v)

        output, _ = self.out_proj(attn_output)
        return output


class M2M100CrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        config: M2M100Config | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.prefix = prefix
        self.d_model = config.d_model
        self.embed_dim = embed_dim
        self.total_num_heads = num_heads
        self.total_num_kv_heads = self.total_num_heads
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.kv_size = self.total_num_kv_heads * self.head_dim

        # Q projects decoder hidden states.
        self.q_proj = ColumnParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )

        # KV projects encoder hidden states; total_num_heads=0 avoids an
        # unused Q shard inside the fused QKV layer.
        self.kv_proj = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.head_dim,
            total_num_heads=0,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_proj",
        )

        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=bias,
            quant_config=quant_config,
        )

        tp_world_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_world_size == 0
        self.num_heads = self.total_num_heads // tp_world_size

        if self.total_num_kv_heads >= tp_world_size:
            assert self.total_num_kv_heads % tp_world_size == 0
        else:
            assert tp_world_size % self.total_num_kv_heads == 0
        self.num_kv_heads = self.num_heads  # No GQA in M2M100
        self.attn = CrossAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=AttentionType.ENCODER_DECODER,
        )

    def forward(
        self,
        decoder_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q, _ = self.q_proj(decoder_hidden_states)

        # Encoder K/V are computed once during prefill; afterwards they live in
        # the cross-attention KV cache.
        if encoder_hidden_states is not None:
            kv, _ = self.kv_proj(encoder_hidden_states)
            k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
        else:
            k = v = None

        attn_output = self.attn(q, k, v)
        output, _ = self.out_proj(attn_output)
        return output


class M2M100EncoderLayer(nn.Module):
    """Pre-norm encoder layer (LN before each sub-block, residual after)."""

    def __init__(
        self,
        config: M2M100Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = M2M100EncoderAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.activation_fn = get_act_fn(config.activation_function)

        ffn_has_bias = True
        self.fc1 = ColumnParallelLinear(
            self.embed_dim,
            config.encoder_ffn_dim,
            bias=ffn_has_bias,
            quant_config=quant_config,
        )
        self.fc2 = RowParallelLinear(
            config.encoder_ffn_dim,
            self.embed_dim,
            bias=ffn_has_bias,
            quant_config=quant_config,
        )

        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Pre-norm self attention.
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        # Pre-norm feed-forward.
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        fc1_out, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(fc1_out)
        hidden_states, _ = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            hidden_states = cast_overflow_tensors(hidden_states)

        return hidden_states


class M2M100DecoderLayer(nn.Module):
    """Pre-norm decoder layer (self-attn, cross-attn, FFN; LN before each)."""

    def __init__(
        self,
        config: M2M100Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = M2M100DecoderSelfAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.activation_fn = get_act_fn(config.activation_function)

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # Named "encoder_attn" to match the pretrained weights (this is the
        # decoder's cross-attention over encoder outputs).
        self.encoder_attn = M2M100CrossAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.encoder_attn",
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        ffn_has_bias = True
        self.fc1 = ColumnParallelLinear(
            self.embed_dim,
            config.decoder_ffn_dim,
            bias=ffn_has_bias,
            quant_config=quant_config,
        )
        self.fc2 = RowParallelLinear(
            config.decoder_ffn_dim,
            self.embed_dim,
            bias=ffn_has_bias,
            quant_config=quant_config,
        )

        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        decoder_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-norm self attention.
        residual = decoder_hidden_states
        hidden_states = self.self_attn_layer_norm(decoder_hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        # Pre-norm cross attention.
        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states = self.encoder_attn(
            decoder_hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )
        hidden_states = residual + hidden_states

        # Pre-norm feed-forward.
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        fc1_out, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(fc1_out)
        hidden_states, _ = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class M2M100Encoder(nn.Module):
    """Transformer encoder of ``config.encoder_layers`` [`M2M100EncoderLayer`]."""

    def __init__(
        self,
        config: M2M100Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        lora_config: LoRAConfig | None = None,
        embed_tokens: nn.Embedding | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.cache_config = cache_config
        self.quant_config = quant_config
        self.lora_config = lora_config
        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.embed_tokens = M2M100ScaledWordEmbedding(
            config.vocab_size, embed_dim, embed_scale=embed_scale
        )
        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        # Computed sinusoids with the M2M100 padding offset; not in the checkpoint.
        self.embed_positions = M2M100SinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
            self.padding_idx,
        )
        self.layers = nn.ModuleList(
            [
                M2M100EncoderLayer(
                    config,
                    cache_config,
                    quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(config.encoder_layers)
            ]
        )
        # M2M100 has no layernorm_embedding, but does have a final layer_norm.
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        embed_pos = self.embed_positions(positions)
        embed_pos = embed_pos.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        hidden_states = inputs_embeds + embed_pos

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states=hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class M2M100Decoder(nn.Module):
    """Transformer decoder of ``config.decoder_layers`` [`M2M100DecoderLayer`]."""

    def __init__(
        self,
        config: M2M100Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        lora_config: LoRAConfig | None = None,
        embed_tokens: nn.Embedding | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.cache_config = cache_config
        self.quant_config = quant_config
        self.lora_config = lora_config
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = M2M100ScaledWordEmbedding(
            config.vocab_size, config.d_model, embed_scale=embed_scale
        )
        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = M2M100SinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            self.padding_idx,
        )
        self.layers = nn.ModuleList(
            [
                M2M100DecoderLayer(
                    config,
                    cache_config,
                    quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(config.decoder_layers)
            ]
        )
        # M2M100 has no layernorm_embedding, but does have a final layer_norm.
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        decoder_positions: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            assert decoder_input_ids is not None
            inputs_embeds = self.embed_input_ids(decoder_input_ids)

        embed_pos = self.embed_positions(decoder_positions)
        embed_pos = embed_pos.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        hidden_states = inputs_embeds + embed_pos

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                decoder_hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.embed_tokens(input_ids)


class M2M100Model(nn.Module, SupportsQuant):
    _tied_weights_keys = [
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
    ]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config

        lora_vocab = (
            (lora_config.lora_extra_vocab_size * (lora_config.max_loras or 1))
            if lora_config
            else 0
        )
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        self.encoder = M2M100Encoder(
            config, cache_config, quant_config=quant_config, prefix=f"{prefix}.encoder"
        )
        self.decoder = M2M100Decoder(
            config, cache_config, quant_config=quant_config, prefix=f"{prefix}.decoder"
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        encoder_outputs: torch.Tensor | None,
    ) -> torch.Tensor:
        decoder_outputs = self.decoder(
            decoder_input_ids=input_ids,
            decoder_positions=positions,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_outputs,
        )
        return decoder_outputs

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        # Cross-attention fuses only K/V, keeping Q separate.
        cross_attn_stacked_params_mapping = [
            ("kv_proj", "k_proj", "k"),
            ("kv_proj", "v_proj", "v"),
        ]

        other_weights = []
        loaded_stacked_params = []
        model_params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in cross_attn_stacked_params_mapping:
                if weight_name not in name or "encoder_attn" not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name not in model_params_dict:
                    continue
                param = model_params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_stacked_params.append(name)
                break
            else:
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in name or "encoder_attn" in name:
                        # Skip cross-attn q_proj here; it loads normally.
                        continue
                    name = name.replace(weight_name, param_name)
                    if name not in model_params_dict:
                        continue
                    param = model_params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    loaded_stacked_params.append(name)
                    break
                else:
                    if name in model_params_dict:
                        other_weights.append((name, loaded_weight))

        loader = AutoWeightsLoader(self)
        loaded_params = loader.load_weights(other_weights)
        loaded_params.update(loaded_stacked_params)
        return loaded_params


class M2M100ProcessingInfo(BaseProcessingInfo):
    """Processing information for NLLB / M2M100 encoder-decoder models."""

    def get_hf_config(self) -> M2M100Config:
        return self.ctx.get_hf_config(M2M100Config)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # The encoder input (the source sentence) is a single "text" modality;
        # the decoder side is plain text.
        return {"text": 1}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        # The encoder handles up to max_position_embeddings source tokens;
        # return it directly to avoid complex profiling.
        config = self.get_hf_config()
        return {"text": config.max_position_embeddings}

    def get_data_parser(self) -> MultiModalDataParser:
        return TextDataParser()

    def _get_m2m100_lang_map(self) -> dict[str, int] | None:
        """The tokenizer's language map iff this is an M2M100 checkpoint.

        M2M100 (as opposed to NLLB) exposes its language tokens through the
        tokenizer's ``lang_code_to_id`` -- bare ISO-639-1 codes (``"de"``,
        ``"fr"``) each mapped to the corresponding ``__de__``-style token id.
        NLLB has no such attribute; its ``deu_Latn`` tags live only in
        ``added_tokens_decoder``. So the presence of a populated
        ``lang_code_to_id`` both identifies the family and provides the
        authoritative code -> forced-BOS-token-id resolution. Returns the map
        for M2M100, or ``None`` for NLLB. Result (including ``None``) is cached.
        """
        cached = getattr(self, "_m2m100_lang_map_cache", _UNSET)
        if cached is not _UNSET:
            return cached
        tokenizer = self.get_tokenizer()
        lang_map = getattr(tokenizer, "lang_code_to_id", None)
        result = dict(lang_map) if lang_map else None
        self._m2m100_lang_map_cache = result
        return result

    def get_lang_codes(self) -> frozenset[str]:
        """The set of language codes this NLLB/M2M100 checkpoint recognizes.

        For **M2M100** these are the bare ISO-639-1 codes from the tokenizer's
        ``lang_code_to_id`` (see :meth:`_get_m2m100_lang_map`). For **NLLB**,
        which adds one special token per supported language (e.g. ``"deu_Latn"``,
        ``"fra_Latn"``), they are read from the tokenizer's
        ``added_tokens_decoder``. Using this authoritative set (rather than "any
        token that isn't unk") rejects strings that happen to tokenize to an
        ordinary sub-word token -- e.g. a bare ISO code like ``"de"`` (a real
        sub-word, not an NLLB language tag) -- which would otherwise silently
        mis-condition the decoder.
        """
        cached = getattr(self, "_lang_codes_cache", None)
        if cached is not None:
            return cached
        lang_map = self._get_m2m100_lang_map()
        if lang_map is not None:
            frozen = frozenset(lang_map.keys())
            self._lang_codes_cache = frozen
            return frozen
        tokenizer = self.get_tokenizer()
        codes: set[str] = set()
        added = getattr(tokenizer, "added_tokens_decoder", None) or {}
        for token in added.values():
            content = getattr(token, "content", token)
            # NLLB language tags are ``lll_Ssss`` (ISO-639-3 ``_`` script);
            # this pattern excludes control tokens like ``<mask>``/``</s>``.
            if _LANG_CODE_RE.match(str(content)):
                codes.add(str(content))
        frozen = frozenset(codes)
        self._lang_codes_cache = frozen
        return frozen

    def _normalize_lang_code(self, target_language: str) -> str:
        """Map a friendly language identifier to this checkpoint's native code.

        For **NLLB** the native code is the full tag (``"deu_Latn"``); a common
        ISO-639-1 code (``"de"``) or an English name (``"German"``) is mapped to
        it. For **M2M100** the native code is the bare ISO-639-1 code (``"de"``);
        an English name (``"German"``) or an NLLB-style tag (``"deu_Latn"``, in
        case a caller reuses one) is mapped to it. A mapped candidate is accepted
        only if the checkpoint actually declares it, so this never invents an
        unsupported language -- an unmapped/unknown code is returned unchanged
        and rejected downstream.

        A non-string target raises a clean ``ValueError`` rather than an
        ``AttributeError`` from ``.strip()`` (the runtime path already guards
        with ``isinstance(prompt, str)``, but this method is public).
        """
        if not isinstance(target_language, str):
            raise ValueError(
                "Target language must be a string, got "
                f"{type(target_language).__name__}."
            )
        code = target_language.strip()
        valid_codes = self.get_lang_codes()
        if not valid_codes:
            # Could not recover the code set: leave as-is (rejected downstream).
            return code
        if code in valid_codes:
            return code
        if self._get_m2m100_lang_map() is not None:
            # M2M100: native codes are bare (lower-case) ISO-639-1. Accept a
            # case-variant ISO code ("DE"), an English name, or an NLLB-style tag
            # a caller might reuse -- matching the case-insensitivity NLLB gets
            # for free from its ISO/name lookups below.
            candidate = (
                (code.lower() if code.lower() in valid_codes else None)
                or _NAME_TO_ISO.get(code.lower())
                or _NLLB_TO_ISO.get(code)
            )
        else:
            # NLLB: native codes are ``lll_Ssss`` tags.
            candidate = _ISO_TO_NLLB.get(code.lower()) or _NAME_TO_NLLB.get(
                code.lower()
            )
        if candidate is not None and candidate in valid_codes:
            return candidate
        return code

    def get_lang_token_id(self, target_language: str) -> int:
        """Resolve a target-language code to its forced-BOS token id.

        Both families condition the output language by placing this token
        immediately after ``decoder_start_token_id`` (the ``forced_bos_token_id``
        mechanism): the decoder sequence begins ``[eos, <tgt-lang>]``. The
        requested target is first normalized (see :meth:`_normalize_lang_code`).

        **M2M100** resolves the ISO code through the tokenizer's own
        ``lang_code_to_id`` -- ``convert_tokens_to_ids("de")`` would return an
        ordinary sub-word id (a real token, not unk), so it must NOT be used;
        the ``__de__`` language-token id comes from the language map instead.
        **NLLB** looks the ``deu_Latn`` tag up with ``convert_tokens_to_ids``.
        Only codes the checkpoint actually declares (see :meth:`get_lang_codes`)
        are accepted; anything else is rejected so an unsupported target surfaces
        as a clean error rather than a mis-conditioned (garbage) translation.
        """
        resolved = self._normalize_lang_code(target_language)
        lang_map = self._get_m2m100_lang_map()
        if lang_map is not None:
            if resolved not in lang_map:
                raise ValueError(
                    f"Unknown M2M100 target language code: {target_language!r}. "
                    "Expected an ISO-639-1 code such as 'de' or 'fr', or an "
                    "English language name (e.g. 'German')."
                )
            return lang_map[resolved]
        valid_codes = self.get_lang_codes()
        # Fall back to the unk check if we could not recover the code set
        # (defensive: never regress to accepting an unk-mapped code).
        tokenizer = self.get_tokenizer()
        token_id = tokenizer.convert_tokens_to_ids(resolved)
        is_valid = (
            resolved in valid_codes
            if valid_codes
            else token_id is not None and token_id != tokenizer.unk_token_id
        )
        if not is_valid:
            raise ValueError(
                f"Unknown NLLB target language code: {target_language!r}. "
                "Expected an NLLB code such as 'deu_Latn' or 'fra_Latn', a "
                "recognized ISO-639-1 code (e.g. 'de'), or an English language "
                "name (e.g. 'German')."
            )
        return token_id

    def get_default_tok_params(self):
        # This governs the DECODER-prompt tokenization path: keep
        # add_special_tokens=False so an empty decoder prompt stays empty and
        # the runtime prepends only decoder_start_token_id (=eos=2). The ENCODER
        # source is tokenized separately in _apply_hf_processor_main /
        # _get_prompt_updates with add_special_tokens=True (the NLLB tokenizer
        # then prepends the src-lang code and appends </s>).
        return super().get_default_tok_params().with_kwargs(add_special_tokens=False)


class M2M100DummyInputsBuilder(BaseDummyInputsBuilder[M2M100ProcessingInfo]):
    """Builds dummy inputs for profiling NLLB / M2M100 models."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        # The decoder prompt is separate from the encoder; minimal dummy text.
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_texts = mm_counts.get("text", 0)
        if num_texts == 0:
            return {}
        # Simple repeated words of the profiled length for the encoder side.
        # Leave room for the two special tokens the tokenizer adds
        # (add_special_tokens=True: a leading src-lang code + trailing </s>), so
        # the total token count -- and hence the max encoder position -- stays
        # within max_position_embeddings.
        num_words = max(seq_len - 2, 1)
        dummy_text = " ".join(["word"] * num_words)
        return {"text": dummy_text}


class TextProcessorItems(ProcessorBatchItems[str]):
    """Data items for the text modality (encoder input is source text)."""

    def __init__(self, data) -> None:
        if data is None:
            data = [""]
        elif isinstance(data, str):
            data = [data]
        super().__init__(data, "text")


class TextDataParser(MultiModalDataParser):
    def __init__(self):
        super().__init__()

    def _parse_text_data(
        self,
        data: ModalityData[str],
    ) -> ModalityDataItems[Any, Any] | None:
        if data is None or not len(data):
            return TextProcessorItems(None)

        if isinstance(data, str) or is_list_of(data, str):
            return TextProcessorItems(data)
        raise TypeError(
            f"Text data must be a string or list of strings, got {type(data)}"
        )

    def _get_subparsers(self) -> Mapping[str, ModalityDataParser]:
        return {
            "text": self._parse_text_data,
        }


class M2M100MultiModalProcessor(EncDecMultiModalProcessor[M2M100ProcessingInfo]):
    """Multimodal processor for NLLB / M2M100 (source text as a "text" modality)."""

    def create_encoder_prompt(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
    ) -> str | list[int]:
        # Single placeholder token; expanded to the real source length by
        # _get_prompt_updates. The actual token ids come from the mm kwargs.
        return [0]

    def create_decoder_prompt(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
    ) -> str | list[int]:
        """Build the NLLB decoder start from the requested target language.

        NLLB's decoder sequence begins ``[decoder_start=eos, forced_bos=<tgt-lang>]``.
        The runtime prepends ``decoder_start_token_id`` (=eos=2 for NLLB), so we
        return ``[<tgt-lang token id>]`` as the forced BOS. The target language
        is passed as the decoder prompt string (an NLLB code such as
        ``"deu_Latn"``).

        A list of token ids is passed through unchanged: that is an explicit
        decoder sequence (e.g. teacher forcing) which already carries its lang
        tag. An empty string yields no forced BOS (e.g. profiling) -- the runtime
        still prepends the decoder start.
        """
        if isinstance(prompt, str):
            target_language = prompt.strip()
            if not target_language:
                return []
            return [self.info.get_lang_token_id(target_language)]
        return prompt

    def _apply_hf_processor_main(
        self,
        mm_items: MultiModalDataItems,
        hf_kwargs: Mapping[str, object],
    ):
        """Tokenize the source text directly (these models have only a
        tokenizer, no HF Processor). The decoder prompt is built separately by
        ``create_decoder_prompt``, so this emits only ``encoder_input_ids``.

        Uses ``add_special_tokens=True`` so the NLLB tokenizer prepends the
        source-language code and appends </s>, matching HuggingFace and
        ``_get_prompt_updates``.
        """
        from transformers.feature_extraction_utils import BatchFeature

        num_text_items = mm_items.get_count("text", strict=False)
        if num_text_items == 0:
            return BatchFeature({})

        tokenizer = self.info.get_tokenizer()
        text_items = mm_items.get_items("text", TextProcessorItems)
        encoder_text = text_items.get(0)

        encoder_tokenized = tokenizer(
            encoder_text,
            return_tensors="pt",
            add_special_tokens=True,
        )
        return BatchFeature({"encoder_input_ids": encoder_tokenized["input_ids"]})

    def _get_mm_fields_config(
        self,
        hf_inputs,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(encoder_input_ids=MultiModalFieldConfig.batched("text"))

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        """Expand the single ``[0]`` encoder placeholder to N placeholders,
        where N is the tokenized source length (``add_special_tokens=True``,
        matching ``_apply_hf_processor_main`` -- includes the src-lang code and
        the trailing </s>)."""
        from vllm.multimodal.processing import PromptReplacement

        num_text_items = mm_items.get_count("text", strict=False)
        if num_text_items == 0:
            return []

        text_items = mm_items.get_items("text", TextProcessorItems)
        tokenizer = self.info.get_tokenizer()
        text = text_items.get(0)
        num_tokens = len(tokenizer.encode(text, add_special_tokens=True))

        return [
            PromptReplacement(
                modality="text",
                target=[0],
                replacement=[0] * num_tokens,
            )
        ]

    def build_data_parser(self) -> MultiModalDataParser:
        return TextDataParser()


@MULTIMODAL_REGISTRY.register_processor(
    M2M100MultiModalProcessor,
    info=M2M100ProcessingInfo,
    dummy_inputs=M2M100DummyInputsBuilder,
)
class M2M100ForConditionalGeneration(nn.Module, SupportsQuant, SupportsMultiModal):
    """NLLB / M2M100 for conditional generation (encoder-decoder + tied LM head).

    The source sentence is fed as a "text" modality through
    ``M2M100MultiModalProcessor`` (registered above). ``load_weights`` consumes a
    real ``facebook/nllb-200-*`` or M2M100 checkpoint with zero missing or
    unexpected keys. The decoder's first token is the target-language
    ``forced_bos_token_id`` resolved by ``create_decoder_prompt``.
    """

    # HF M2M100 checkpoint keys already carry the ``model.`` prefix and use
    # standard ``.weight``/``.bias``/``*_layer_norm`` names, so the mapper is an
    # identity map.
    hf_to_vllm_mapper = WeightsMapper()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        lora_config = vllm_config.lora_config
        # NLLB / M2M100 tie the LM head to the shared embeddings.
        assert config.tie_word_embeddings
        self.config = config
        self.model = M2M100Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size

        # M2M100's lm_head is the raw tied ``model.shared`` weight; unlike Marian
        # there is NO final_logits_bias.
        self.lm_head = ParallelLMHead(config.vocab_size, config.d_model)
        self.logits_processor = LogitsProcessor(
            self.unpadded_vocab_size, config.vocab_size
        )

    def get_language_model(self) -> nn.Module:
        return self.model.decoder

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.decoder.embed_tokens(input_ids)

    def embed_multimodal(self, **kwargs) -> MultiModalEmbeddings:
        encoder_input_ids_list = self._parse_and_validate_encoder_input(**kwargs)

        if not encoder_input_ids_list:
            raise ValueError(
                "encoder_input_ids_list is empty - this should not happen. "
                "Check that multimodal data is being passed correctly."
            )

        # Process each unique encoder input once; duplicates alias the same
        # output tensor (safe for read-only consumers).
        encoder_outputs: list[torch.Tensor] = []
        cached_outputs: dict[tuple[int, ...], torch.Tensor] = {}
        for encoder_input_ids in encoder_input_ids_list:
            cache_key = tuple(encoder_input_ids.reshape(-1).tolist())
            encoder_output = cached_outputs.get(cache_key)
            if encoder_output is not None:
                encoder_outputs.append(encoder_output)
                continue

            encoder_positions = torch.arange(
                encoder_input_ids.size(-1),
                dtype=torch.long,
                device=encoder_input_ids.device,
            )
            encoder_output = self.model.encoder(
                input_ids=encoder_input_ids.squeeze(0),
                positions=encoder_positions,
            )
            cached_outputs[cache_key] = encoder_output
            encoder_outputs.append(encoder_output)
        return encoder_outputs

    def _parse_and_validate_encoder_input(self, **kwargs: object) -> list[torch.Tensor]:
        encoder_input_ids = kwargs.get("encoder_input_ids", kwargs.get("input_ids"))

        if encoder_input_ids is None:
            return []

        if not isinstance(encoder_input_ids, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of encoder input_ids. "
                f"Got type: {type(encoder_input_ids)}"
            )

        if isinstance(encoder_input_ids, list):
            result = []
            for item in encoder_input_ids:
                if isinstance(item, torch.Tensor) and item.dim() == 0:
                    item = item.unsqueeze(0)
                result.append(item)
            return result
        return encoder_input_ids.unsqueeze(1).unbind(dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_outputs: list[torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # EncoderDecoderModelState passes an empty list on decode steps.
        enc_states = torch.cat(encoder_outputs, dim=0) if encoder_outputs else None

        return self.model(
            input_ids, positions, inputs_embeds, encoder_outputs=enc_states
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        # No final_logits_bias for M2M100.
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights_tuple_list = list(weights)

        shared_embedding_weight = None
        for name, loaded_weight in weights_tuple_list:
            if (
                "shared.weight" in name
                or "encoder.embed_tokens.weight" in name
                or "decoder.embed_tokens.weight" in name
                or "lm_head.weight" in name
            ):
                if shared_embedding_weight is not None:
                    # NLLB / M2M100 tie model.shared, both embed_tokens and
                    # lm_head; they alias one tensor, so only the first copy is
                    # used.
                    continue
                shared_embedding_weight = loaded_weight

        # ``model.shared.weight`` has no matching parameter in this module, and
        # the sinusoidal position tables are non-persistent buffers absent from
        # the checkpoint -> ignore any such keys as "unexpected".
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_prefixes=[
                "model.shared.",
                "model.encoder.embed_positions",
                "model.decoder.embed_positions",
            ],
        )
        loaded_params = loader.load_weights(
            weights_tuple_list, mapper=self.hf_to_vllm_mapper
        )

        if shared_embedding_weight is not None:
            weight_loader = getattr(
                self.lm_head.weight, "weight_loader", default_weight_loader
            )
            weight_loader(self.lm_head.weight, shared_embedding_weight)

            self.model.encoder.embed_tokens.weight = self.lm_head.weight
            self.model.decoder.embed_tokens.weight = self.lm_head.weight
            loaded_params.update(
                {
                    "model.encoder.embed_tokens.weight",
                    "lm_head.weight",
                    "model.decoder.embed_tokens.weight",
                }
            )

        return loaded_params
