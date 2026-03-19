# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.inputs import MultiModalDataDict, PromptType, TextPrompt
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import (
    Attention,
    CrossAttention,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseProcessingInfo,
    EncDecMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
)
from vllm.renderers import TokenizeParams
from vllm.transformers_utils.processors.cohere_asr import (
    INF_VAL,
    CohereASRFeatureExtractor,
    CohereASRProcessor,
)
from vllm.v1.attention.backend import (
    AttentionType,
)

from .interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsTranscription,
)
from .utils import AutoWeightsLoader, WeightsMapper, make_layers

logger = init_logger(__name__)

# From https://platform.openai.com/docs/guides/speech-to-text/supported-languages

ISO639_1_SUPPORTED_LANGS = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "el": "Greek",
    "ar": "Arabic",
    "ko": "Korean",
    "ja": "Japanese",
    "vi": "Vietnamese",
    "zh": "Chinese",
}


class CohereASRAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        attn_type: AttentionType = AttentionType.DECODER,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        if self.total_num_heads >= tp_size:
            # Number of heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_heads % tp_size == 0
        else:
            # Number of heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_heads == 0
        self.num_kv_heads = max(1, self.total_num_heads // tp_size)
        self.head_dim = self.embed_dim // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.attn_type = attn_type

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: "
                f"{self.embed_dim} and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self._init_qkv(embed_dim, bias, quant_config, prefix=prefix)

        self.out_projection = RowParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.out_projection",
        )
        if attn_type == AttentionType.ENCODER:
            raise NotImplementedError(
                "CohereASRAttention does not support Encoder Self-Attention yet."
            )

        elif self.attn_type == AttentionType.ENCODER_DECODER:
            self.attn = CrossAttention(
                self.num_heads,
                self.head_dim,
                self.scaling,
                num_kv_heads=self.num_kv_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.attn",
                attn_type=self.attn_type,
            )
        else:  # AttentionType.DECODER (regular decoder self-attention)
            self.attn = Attention(
                self.num_heads,
                self.head_dim,
                self.scaling,
                num_kv_heads=self.num_kv_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.attn",
                attn_type=self.attn_type,
            )

    def _init_qkv(
        self,
        embed_dim: int,
        bias: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        self.qkv_proj = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        attn_output = self.attn(q, k, v)

        output, _ = self.out_projection(attn_output)

        return output


class CohereASRCrossAttention(CohereASRAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            attn_type=AttentionType.ENCODER_DECODER,
        )

    def _init_qkv(
        self,
        embed_dim: int,
        bias: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        self.q_proj = ColumnParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.kv_proj = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.head_dim,
            total_num_heads=0,
            total_num_kv_heads=self.total_num_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_proj",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
    ) -> torch.Tensor:
        q, _ = self.q_proj(hidden_states)

        # Encoder hidden states are only computed once during prefill phase.
        # Afterwards, the keys and values should be available in the kv-cache.
        if encoder_hidden_states is not None:
            kv, _ = self.kv_proj(encoder_hidden_states)
            k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
        else:
            k = v = None

        attn_output = self.attn(q, k, v)

        output, _ = self.out_projection(attn_output)

        return output


# ----- Decoder START -----
class CohereASRMLP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        act_fn: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.activation_fn = get_act_fn(act_fn)
        self.dense_in = ColumnParallelLinear(
            input_size=embed_dim,
            output_size=ffn_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.dense_out = RowParallelLinear(
            input_size=ffn_dim,
            output_size=embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.dense_in(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.dense_out(hidden_states)
        return hidden_states


class FixedPositionalEncoding(nn.Module):
    """
    Fixed positional encoding (embedding layer) from sine and cosine functions
    of different frequencies according to https://arxiv.org/abs/1706.03762

    Args:
        hidden_size: size of the embeddings in the model, also known as d_model
        max_sequence_length: maximum allowed length of the input sequence
    """

    def __init__(self, hidden_size: int, max_sequence_length: int = 512) -> None:
        super().__init__()

        self._hidden_size = hidden_size
        self._max_sequence_length = max_sequence_length
        self._build_pos_enc(
            hidden_size=self._hidden_size, max_sequence_length=self._max_sequence_length
        )

    def _build_pos_enc(self, hidden_size: int, max_sequence_length: int) -> None:
        """Builds/replaces pre-computed positional encoding."""
        pos_enc = torch.zeros(max_sequence_length, hidden_size)
        position = torch.arange(0.0, max_sequence_length).unsqueeze(1)
        coef = -math.log(10000.0) / hidden_size
        div_term = torch.exp(coef * torch.arange(0.0, hidden_size, 2))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc.div_(math.sqrt(hidden_size))
        self.register_buffer("pos_enc", pos_enc)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        embeddings = torch.embedding(self.pos_enc, position_ids)
        return embeddings


class CohereASRDecoderLayer(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config.transf_decoder["config_dict"]
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.hidden_dim = config.get("hidden_size")
        self.ffn_dim = config.get("inner_size")
        self.act_fn = config.get("hidden_act")
        self.num_heads = config.get("num_attention_heads")

        # self_attn
        self.layer_norm_1 = nn.LayerNorm(self.hidden_dim)
        self.first_sub_layer = CohereASRAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            attn_type=AttentionType.DECODER,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.first_sub_layer",
        )

        # cross attn to attend to encoder
        self.layer_norm_2 = nn.LayerNorm(self.hidden_dim)
        self.second_sub_layer = CohereASRCrossAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.second_sub_layer",
        )

        self.layer_norm_3 = nn.LayerNorm(self.hidden_dim)
        self.third_sub_layer = CohereASRMLP(
            embed_dim=self.hidden_dim,
            ffn_dim=self.ffn_dim,
            act_fn=self.act_fn,
            quant_config=quant_config,
            prefix=f"{prefix}.third_sub_layer",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm_1(hidden_states)
        hidden_states = self.first_sub_layer(hidden_states=hidden_states)

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm_2(hidden_states)
        hidden_states = self.second_sub_layer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm_3(hidden_states)
        hidden_states = self.third_sub_layer(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class TransformerEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_target_positions: int,
        padding_idx: int,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx)
        self.position_embedding = FixedPositionalEncoding(
            hidden_size=hidden_size,
            max_sequence_length=max_target_positions,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        inputs_embeds = self.token_embedding(input_ids)
        positions = self.position_embedding(positions)
        embeddings = inputs_embeds + positions
        embeddings = self.layer_norm(embeddings)
        return embeddings


@support_torch_compile(dynamic_arg_dims={"input_ids": 0, "positions": -1})
class CohereASRDecoder(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.padding_idx = 2
        config_dict = config.transf_decoder["config_dict"]
        self.max_target_positions = config_dict.get("max_sequence_length")
        self.hidden_size = config_dict.get("hidden_size")
        self.num_decoder_layers = config_dict.get("num_layers")
        self.vocab_size = config.head["num_classes"]

        self.embedding = TransformerEmbedding(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            max_target_positions=self.max_target_positions,
            padding_idx=self.padding_idx,
        )

        self.start_layer, self.end_layer, self.layers = make_layers(
            self.num_decoder_layers,
            lambda prefix: CohereASRDecoderLayer(
                vllm_config=vllm_config, prefix=f"{prefix}.layers"
            ),
            prefix=f"{prefix}.layers",
        )
        self.final_layer_norm = nn.LayerNorm(self.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden_states = self.get_input_embeddings(input_ids, positions)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )

        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states

    def get_input_embeddings(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        return self.embedding(input_ids, positions)


# ----- Decoder END -----


# ----- Encoder START -----
class MaskedConvSequential(nn.Sequential):
    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)  # (batch, 1, time, features)
        current_lengths = lengths.clone().float()
        mask = self._create_mask(x, current_lengths.long())

        # Process through each layer with mask propagation
        for i, layer in enumerate(self):
            # Apply current mask before layer
            x = self.apply_channel_mask(x, mask)

            # Apply layer
            x = layer(x)

            # Update lengths for stride operations with proper padding
            if hasattr(layer, "stride") and layer.stride != (1, 1):
                if hasattr(layer, "_left_padding"):
                    padding = (
                        layer._left_padding,
                        layer._right_padding,
                    )  # CausalConv2D
                else:
                    padding = layer.padding
                current_lengths = self.calculate_conv_output_size(
                    current_lengths, layer.kernel_size[0], layer.stride[0], padding
                )
                mask = self._create_mask(x, current_lengths.long())

        # Final masking
        x = self.apply_channel_mask(x, mask)
        return x, current_lengths.long()

    def _create_mask(self, tensor: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Create broadcastable mask from per-sample lengths.

        Returns a (B, 1, T, 1) mask that broadcasts over channels and
        features without materializing a full (B, C, T, F) tensor.
        """
        batch_size, channels, time, features = tensor.shape
        time_mask = torch.arange(time, device=tensor.device).expand(
            batch_size, time
        ) < lengths.unsqueeze(1)
        return time_mask.to(tensor.dtype).unsqueeze(1).unsqueeze(-1)

    def apply_channel_mask(
        self, tensor: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply mask in-place via broadcasting.

        tensor: (B, C, T, F),  mask: (B, 1, T, 1)
        """
        tensor.mul_(mask)
        return tensor

    def calculate_conv_output_size(
        self,
        input_size: torch.Tensor,
        kernel_size: int,
        stride: int,
        padding: tuple[int, int],
    ):
        """Calculate exact output size after convolution."""
        return (input_size + padding[0] + padding[1] - kernel_size) // stride + 1


class ConvSubsampling(nn.Module):
    def __init__(
        self,
        subsampling: str,
        subsampling_factor: int,
        feat_in: int,
        feat_out: int,
        conv_channels: int,
        subsampling_conv_chunking_factor: int = 1,
        activation: nn.Module | None = None,
        is_causal: bool = False,
    ) -> None:
        super().__init__()
        if activation is None:
            activation = nn.ReLU()

        if subsampling_factor % 2 != 0:
            raise ValueError("Sampling factor should be a multiply of 2!")
        self._sampling_num = int(math.log(subsampling_factor, 2))

        if (
            subsampling_conv_chunking_factor != -1
            and subsampling_conv_chunking_factor != 1
            and subsampling_conv_chunking_factor % 2 != 0
        ):
            raise ValueError(
                "subsampling_conv_chunking_factor should be -1, 1, or a power of 2"
            )

        in_channels = 1
        layers = []

        assert subsampling == "dw_striding"
        self._stride = 2
        self._kernel_size = 3
        self._ceil_mode = False

        assert not is_causal

        self._left_padding = (self._kernel_size - 1) // 2
        self._right_padding = (self._kernel_size - 1) // 2

        # Layer 1
        # [1, T, num_melspec] -> [conv_channels, T//2, num_melspec//2]
        layers.append(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=conv_channels,
                kernel_size=self._kernel_size,
                stride=self._stride,
                padding=self._left_padding,
            )
        )
        in_channels = conv_channels
        layers.append(activation)

        for i in range(self._sampling_num - 1):
            # [conv_channels, T//2^i, num_melspec//2^i] ->
            # [conv_channels, T//2^(i+1), num_melspec//2^(i+1)]
            # depthwise conv
            layers.append(
                torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=self._kernel_size,
                    stride=self._stride,
                    padding=self._left_padding,
                    groups=in_channels,
                )
            )

            # [conv_channels, T//2^(i+1), num_melspec//2^(i+1)]
            # -> [conv_channels, T//2^(i+1), num_melspec//2^(i+1)]
            # pointwise conv
            layers.append(
                torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=conv_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                )
            )
            layers.append(activation)
            in_channels = conv_channels

        in_length = torch.tensor(feat_in, dtype=torch.float)
        out_length = self.calc_length(
            lengths=in_length,
            all_paddings=self._left_padding + self._right_padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )

        # reshape:
        # [conv_channels, T//sub_factor, num_melspec//sub_factor]
        # -> [T//sub_factor, conv_channels * (num_melspec//sub_factor)]
        # mlp:
        # [T//sub_factor, conv_channels * (num_melspec//sub_factor)]
        # -> [T//sub_factor, feat_out]
        self.out = torch.nn.Linear(conv_channels * int(out_length), feat_out)
        self.conv2d_subsampling = True
        self.conv = MaskedConvSequential(*layers)

    def calc_length(
        self,
        lengths: torch.Tensor,
        all_paddings: int,
        kernel_size: int,
        stride: int,
        ceil_mode: bool,
        repeat_num: int = 1,
    ) -> torch.Tensor:
        """Calculates the output length of a Tensor passed
        through a convolution or max pooling layer"""
        add_pad: float = all_paddings - kernel_size
        one: float = 1.0
        for i in range(repeat_num):
            lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
            lengths = torch.ceil(lengths) if ceil_mode else torch.floor(lengths)
        return lengths.to(dtype=torch.int)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, lengths = self.conv(x, lengths)

        if self.conv2d_subsampling:
            b, c, t, f = x.size()
            x = self.out(x.transpose(1, 2).reshape(b, t, -1))
        # Transpose to Channel Last mode
        else:
            x = x.transpose(1, 2)

        return x, lengths


class PositionalEncoding(torch.nn.Module):
    """Fixed sinusoidal positional encoding.
    Args:
        d_model (int): embedding dim
        max_len (int): maximum input length
        xscale (bool): whether to scale the input by sqrt(d_model)
    """

    def __init__(
        self, d_model: int, max_len: int = 5000, xscale: float | None = None
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.xscale = xscale
        self.max_len = max_len

    def create_pe(self, positions: torch.Tensor, dtype: torch.dtype) -> None:
        pos_length = positions.size(0)
        pe = torch.zeros(pos_length, self.d_model, device=positions.device)
        div_term = torch.exp(
            torch.arange(
                0, self.d_model, 2, dtype=torch.float32, device=positions.device
            )
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        pe = pe.unsqueeze(0).to(dtype)
        if hasattr(self, "pe"):
            self.pe = pe
        else:
            self.register_buffer("pe", pe, persistent=False)

    def forward(
        self, x: torch.Tensor, cache_len: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Adds positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, feature_size)
            cache_len (int): the size of the cache which is used to shift positions
        Returns:
            x+pos_emb (torch.Tensor): Its shape is (batch, time, feature_size)
            pos_emb (torch.Tensor): Its shape is (1, time, feature_size)
        """
        input_len = x.size(1) + cache_len
        if self.xscale:
            x = x * self.xscale
        pos_emb = self.pe[:, :input_len]
        x = x + pos_emb
        return x, pos_emb


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding for TransformerXL's layers
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): embedding dim
        max_len (int): maximum input length
        xscale (bool): whether to scale the input by sqrt(d_model)
    """

    def extend_pe(self, length: int, device: torch.device, dtype: torch.dtype) -> None:
        """Reset and extend the positional encodings if needed."""
        needed_size = 2 * length - 1
        if hasattr(self, "pe") and self.pe.size(1) >= needed_size:
            return
        positions = torch.arange(
            length - 1, -length, -1, dtype=torch.float32, device=device
        ).unsqueeze(1)
        self.create_pe(positions=positions, dtype=dtype)

    def forward(
        self, x: torch.Tensor, cache_len: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, feature_size)
            cache_len (int): the size of the cache which is used to shift positions
        Returns:
            x (torch.Tensor): Its shape is (batch, time, feature_size)
            pos_emb (torch.Tensor): Its shape is (1, time, feature_size)
        """

        if self.xscale:
            x = x * self.xscale

        input_len = x.size(1) + cache_len
        center_pos = self.pe.size(1) // 2 + 1
        start_pos = center_pos - input_len
        end_pos = center_pos + input_len - 1
        pos_emb = self.pe[:, start_pos:end_pos]

        return x, pos_emb


class Swish(nn.SiLU):
    """
    Swish activation function introduced in 'https://arxiv.org/abs/1710.05941'
    Mathematically identical to SiLU. See note in nn.SiLU for references.
    """


class ConformerFeedForward(nn.Module):
    """
    feed-forward module of Conformer model.
    use_bias (bool): Apply bias to all Linear and Conv1d
        layers to improve activation flow and stabilize
        training of huge models.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: nn.Module | None = None,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        if activation is None:
            activation = Swish()
        self.linear1 = nn.Linear(d_model, d_ff, bias=use_bias)
        self.activation = activation
        self.linear2 = nn.Linear(d_ff, d_model, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class CausalConv1D(nn.Conv1d):
    """
    A causal version of nn.Conv1d where each step would
    have limited access to locations on its right or left.
    All arguments are the same as nn.Conv1d except padding.

    If padding is set None, then paddings are set
    automatically to make it a causal convolution where
    each location would not see any steps on its right.

    If padding is set as a list (size of 2), then
    padding[0] would be used as left padding and
    padding[1] as right padding. It would make it possible
    to control the number of steps to be accessible on the
    right and left. This mode is not supported when
    stride > 1. padding[0]+padding[1] should be equal to
    (kernel_size - 1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: str | int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        if padding is None:
            self._left_padding = kernel_size - 1
            self._right_padding = stride - 1
        else:
            if stride != 1 and padding != kernel_size - 1:
                raise ValueError("No striding allowed for non-symmetric convolutions!")
            if isinstance(padding, int):
                self._left_padding = padding
                self._right_padding = padding
            elif (
                isinstance(padding, list)
                and len(padding) == 2
                and padding[0] + padding[1] == kernel_size - 1
            ):
                self._left_padding = padding[0]
                self._right_padding = padding[1]
            else:
                raise ValueError(f"Invalid padding param: {padding}!")

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, pad=(self._left_padding, self._right_padding))
        return super().forward(x)


class ConformerConvolution(nn.Module):
    """The convolution module for the Conformer model.
    Args:
        d_model (int): hidden dimension
        kernel_size (int): kernel size for depthwise convolution
        pointwise_activation (str): name of the activation
            function to be used for the pointwise conv.
            Note that Conformer uses a special key `glu_`
            which is treated as the original default from
            the paper.
        use_bias (bool): Use bias in all Linear and Conv1d
            layers to improve activation flow and stabilize
            training of huge models. Defaults to True
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        norm_type: str = "batch_norm",
        conv_context_size: int | None = None,
        pointwise_activation: str = "glu_",
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        assert (kernel_size - 1) % 2 == 0

        if conv_context_size is None:
            conv_context_size = (kernel_size - 1) // 2

        assert pointwise_activation == "glu_"
        dw_conv_input_dim = d_model

        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=use_bias,
        )

        self.depthwise_conv = CausalConv1D(
            in_channels=dw_conv_input_dim,
            out_channels=dw_conv_input_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=conv_context_size,
            groups=dw_conv_input_dim,
            bias=use_bias,
        )

        assert norm_type == "batch_norm"
        self.batch_norm = nn.BatchNorm1d(dw_conv_input_dim)

        self.activation = Swish()
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=dw_conv_input_dim,
            out_channels=d_model,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=use_bias,
        )

    def forward(
        self, x: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)

        x = nn.functional.glu(x, dim=1)

        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)

        x = self.depthwise_conv(x)

        x = self.batch_norm(x)

        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)
        return x


class CohereASRMultiHeadAttention(nn.Module):
    """Multi-Head Attention layer of Transformer.
    Args:
        n_head (int): number of heads
        n_feat (int): size of the features
        use_bias (bool): whether to remove bias in linear and conv layers
    """

    def __init__(
        self,
        n_head: int,
        n_feat: int,
        use_bias: bool = True,
    ) -> None:
        """Construct an MultiHeadedAttention object."""
        super().__init__()

        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.s_d_k = math.sqrt(self.d_k)
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.linear_v = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.linear_out = nn.Linear(n_feat, n_feat, bias=use_bias)

    def forward_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transforms query, key and value.
        Args:
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value (torch.Tensor): (batch, time2, size)
        returns:
            q (torch.Tensor): (batch, head, time1, size)
            k (torch.Tensor): (batch, head, time2, size)
            v (torch.Tensor): (batch, head, time2, size)
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        return q, k, v

    def forward_attention(
        self,
        value: torch.Tensor,
        scores: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute attention context vector.
        Args:
            value (torch.Tensor): (batch, time2, size)
            scores(torch.Tensor): (batch, time1, time2)
            mask(torch.Tensor): (batch, time1, time2)
        returns:
            value (torch.Tensor): transformed `value`
                (batch, time2, d_model) weighted by the
                attention scores
        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch, 1, time1, time2)
            scores = scores.masked_fill(mask, -INF_VAL)
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        x = torch.matmul(attn, value)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).reshape(
            n_batch, -1, self.h * self.d_k
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None,
        pos_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute 'Scaled Dot Product Attention'.
        Args:
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value(torch.Tensor): (batch, time2, size)
            mask (torch.Tensor): (batch, time1, time2)

        returns:
            output (torch.Tensor): transformed `value`
                (batch, time1, d_model) weighted by the
                query dot key attention
        """
        q, k, v = self.forward_qkv(query, key, value)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.s_d_k
        return self.forward_attention(v, scores, mask)


class RelPositionMultiHeadAttention(CohereASRMultiHeadAttention):
    """Multi-Head Attention layer of Transformer-XL with
    support of relative positional encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): number of heads
        n_feat (int): size of the features
        use_bias (bool): whether to apply bias in linear
            and conv layers of MultiHeadAttention
    """

    def __init__(
        self,
        n_head: int,
        n_feat: int,
        pos_bias_u: nn.Parameter | torch.Tensor | None,
        pos_bias_v: nn.Parameter | torch.Tensor | None,
        use_bias: bool = True,
    ) -> None:
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(
            n_head=n_head,
            n_feat=n_feat,
            use_bias=use_bias,
        )
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable biases are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        if pos_bias_u is None or pos_bias_v is None:
            self.pos_bias_u = nn.Parameter(
                torch.zeros(self.h, self.d_k), requires_grad=False
            )
            self.pos_bias_v = nn.Parameter(
                torch.zeros(self.h, self.d_k), requires_grad=False
            )
        else:
            self.pos_bias_u = pos_bias_u
            self.pos_bias_v = pos_bias_v

    def rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Compute relative positional encoding.
        Args:
            x (torch.Tensor): (batch, nheads, time, 2*time-1)
        """
        b, h, qlen, pos_len = x.size()  # (b, h, t1, t2)
        # need to add a column of zeros on the left side of
        # last dimension to perform the relative shifting
        x = torch.nn.functional.pad(x, pad=(1, 0))  # (b, h, t1, t2+1)
        x = x.view(b, h, -1, qlen)  # (b, h, t2+1, t1)
        # need to drop the first row
        x = x[:, :, 1:].view(b, h, qlen, pos_len)  # (b, h, t1, t2)
        return x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None,
        pos_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value(torch.Tensor): (batch, time2, size)
            mask (torch.Tensor): (batch, time1, time2)
            pos_emb (torch.Tensor) : (batch, time1, size)

        Returns:
            output (torch.Tensor): transformed `value`
                (batch, time1, d_model) weighted by the
                query dot key attention
        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        # drops extra elements in the matrix_bd to match the matrix_ac's size
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        matrix_bd = matrix_bd[:, :, :, : matrix_ac.size(-1)]
        scores = (matrix_ac + matrix_bd) / self.s_d_k  # (batch, head, time1, time2)
        return self.forward_attention(v, scores, mask)


class ConformerLayer(torch.nn.Module):
    """A single block of the Conformer encoder.

    Args:
        d_model (int): input dimension of
            MultiheadAttentionMechanism and
            PositionwiseFeedForward
        d_ff (int): hidden dimension of
            PositionwiseFeedForward
        self_attention_model (str): type of the attention
            layer and positional encoding
        n_heads (int): number of heads for multi-head
            attention
        conv_kernel_size (int): kernel size for depthwise
            convolution in convolution module
        use_bias (bool): Apply bias to all Linear and
            Conv1d layers from each ConformerLayer to
            improve activation flow and stabilize training
            of huge models. Defaults to True.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        self_attention_model: str = "rel_pos",
        n_heads: int = 4,
        conv_kernel_size: int = 31,
        conv_norm_type: str = "batch_norm",
        conv_context_size: int | None = None,
        pos_bias_u: nn.Parameter | torch.Tensor | None = None,
        pos_bias_v: nn.Parameter | torch.Tensor | None = None,
        att_context_size: list[int] | None = None,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        if att_context_size is None:
            att_context_size = [-1, -1]

        self.self_attention_model = self_attention_model
        self.fc_factor = 0.5

        # first feed forward module
        self.norm_feed_forward1 = nn.LayerNorm(d_model)
        self.feed_forward1 = ConformerFeedForward(
            d_model=d_model, d_ff=d_ff, use_bias=use_bias
        )

        # convolution module
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = ConformerConvolution(
            d_model=d_model,
            kernel_size=conv_kernel_size,
            norm_type=conv_norm_type,
            conv_context_size=conv_context_size,
            use_bias=use_bias,
        )

        # multi-headed self-attention module
        self.norm_self_att = nn.LayerNorm(d_model)

        assert self_attention_model == "rel_pos"

        self.self_attn = RelPositionMultiHeadAttention(
            n_head=n_heads,
            n_feat=d_model,
            pos_bias_u=pos_bias_u,
            pos_bias_v=pos_bias_v,
            use_bias=use_bias,
        )

        # second feed forward module
        self.norm_feed_forward2 = nn.LayerNorm(d_model)
        self.feed_forward2 = ConformerFeedForward(
            d_model=d_model, d_ff=d_ff, use_bias=use_bias
        )

        self.norm_out = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        att_mask: torch.Tensor | None = None,
        pos_emb: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input signals (B, T, d_model)
            att_mask (torch.Tensor): attention masks(B, T, T)
            pos_emb (torch.Tensor): (L, 1, d_model)
            pad_mask (torch.tensor): padding mask
        Returns:
            x (torch.Tensor): (B, T, d_model)
        """
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual = residual + x * self.fc_factor

        x = self.norm_self_att(residual)
        if self.self_attention_model == "rel_pos":
            x = self.self_attn(
                query=x,
                key=x,
                value=x,
                mask=att_mask,
                pos_emb=pos_emb,
            )
        elif self.self_attention_model == "rel_pos_local_attn":
            x = self.self_attn(
                query=x,
                key=x,
                value=x,
                pad_mask=pad_mask,
                pos_emb=pos_emb,
            )
        elif self.self_attention_model == "abs_pos":
            x = self.self_attn(query=x, key=x, value=x, mask=att_mask)
        else:
            x = None

        residual = residual + x

        x = self.norm_conv(residual)
        x = self.conv(x, pad_mask=pad_mask)
        residual = residual + x

        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        residual = residual + x * self.fc_factor

        x = self.norm_out(residual)

        return x


class ConformerEncoder(nn.Module):
    """
    The encoder for ASR model of Conformer.
    Based on this paper:
    'Conformer: Convolution-augmented Transformer for
    Speech Recognition' by Anmol Gulati et al.
    https://arxiv.org/abs/2005.08100
    """

    def __init__(self, *, vllm_config: VllmConfig):
        super().__init__()

        self.hf_config = vllm_config.model_config.hf_config

        feat_in = self.hf_config.encoder["feat_in"]
        n_layers = self.hf_config.encoder["n_layers"]
        d_model = self.hf_config.encoder["d_model"]
        feat_out = self.hf_config.encoder["feat_out"]
        causal_downsampling = self.hf_config.encoder["causal_downsampling"]
        subsampling = self.hf_config.encoder["subsampling"]
        subsampling_factor = self.hf_config.encoder["subsampling_factor"]
        subsampling_conv_chunking_factor = self.hf_config.encoder.get(
            "subsampling_conv_chunking_factor", 1
        )
        subsampling_conv_channels = self.hf_config.encoder["subsampling_conv_channels"]
        ff_expansion_factor = self.hf_config.encoder["ff_expansion_factor"]
        self_attention_model = self.hf_config.encoder["self_attention_model"]
        n_heads = self.hf_config.encoder["n_heads"]
        att_context_size = self.hf_config.encoder["att_context_size"]
        att_context_probs = self.hf_config.encoder.get("att_context_probs", None)
        att_context_style = self.hf_config.encoder.get("att_context_style", "regular")
        xscaling = self.hf_config.encoder["xscaling"]
        untie_biases = self.hf_config.encoder["untie_biases"]
        pos_emb_max_len = self.hf_config.encoder["pos_emb_max_len"]
        conv_kernel_size = self.hf_config.encoder["conv_kernel_size"]
        conv_norm_type = self.hf_config.encoder["conv_norm_type"]
        conv_context_size = self.hf_config.encoder["conv_context_size"]
        use_bias = self.hf_config.encoder.get("use_bias", True)

        d_ff = d_model * ff_expansion_factor
        self.d_model = d_model
        self._feat_in = feat_in
        self.att_context_style = att_context_style
        self.subsampling_factor = subsampling_factor

        self.self_attention_model = self_attention_model

        # Setting up the att_context_size
        (
            _,
            self.att_context_size,
            _,
            self.conv_context_size,
        ) = self._calc_context_sizes(
            att_context_style=att_context_style,
            att_context_size=att_context_size,
            att_context_probs=att_context_probs,
            conv_context_size=conv_context_size,
            conv_kernel_size=conv_kernel_size,
        )

        if xscaling:
            self.xscale = math.sqrt(d_model)
        else:
            self.xscale = None

        # Subsampling
        if subsampling_conv_channels == -1:
            subsampling_conv_channels = d_model
        assert subsampling and subsampling_factor > 1 and subsampling == "dw_striding"

        self.pre_encode = ConvSubsampling(
            subsampling=subsampling,
            subsampling_factor=subsampling_factor,
            feat_in=feat_in,
            feat_out=d_model,
            conv_channels=subsampling_conv_channels,
            subsampling_conv_chunking_factor=subsampling_conv_chunking_factor,
            activation=nn.ReLU(True),
            is_causal=causal_downsampling,
        )

        self._feat_out = d_model

        # Biases for relative positional encoding
        if not untie_biases and self_attention_model == "rel_pos":
            d_head = d_model // n_heads
            # Register as buffers instead of parameters since they're not trainable
            # and need to respect dtype during weight loading
            self.register_buffer(
                "pos_bias_u", torch.zeros(n_heads, d_head), persistent=True
            )
            self.register_buffer(
                "pos_bias_v", torch.zeros(n_heads, d_head), persistent=True
            )
            pos_bias_u = self.pos_bias_u
            pos_bias_v = self.pos_bias_v
        else:
            pos_bias_u = None
            pos_bias_v = None

        # Positional encodings
        self.pos_emb_max_len = pos_emb_max_len
        assert self_attention_model == "rel_pos"
        self.pos_enc = RelPositionalEncoding(
            d_model=d_model,
            max_len=pos_emb_max_len,
            xscale=self.xscale,
        )

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = ConformerLayer(
                d_model=d_model,
                d_ff=d_ff,
                self_attention_model=self_attention_model,
                n_heads=n_heads,
                conv_kernel_size=conv_kernel_size,
                conv_norm_type=conv_norm_type,
                conv_context_size=self.conv_context_size,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
                att_context_size=self.att_context_size,
                use_bias=use_bias,
            )
            self.layers.append(layer)

        if feat_out > 0 and feat_out != self._feat_out:
            self.out_proj = nn.Linear(self._feat_out, feat_out)
            self._feat_out = feat_out
        else:
            self.out_proj = None
            self._feat_out = d_model
        self.set_max_audio_length(self.pos_emb_max_len)

    def get_num_encoder_cross_attn_tokens(self, num_encoder_input_tokens: int) -> int:
        num_encoder_cross_attn_tokens = math.ceil(
            num_encoder_input_tokens / self.subsampling_factor
        )
        return num_encoder_cross_attn_tokens

    def set_max_audio_length(self, max_audio_length: int) -> None:
        """
        Sets maximum input length.
        Pre-calculates internal seq_range mask.

        Args:
            max_audio_length (int): New maximum sequence length.
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        self.pos_enc.extend_pe(max_audio_length, device, dtype)

    def forward(
        self,
        audio_signal: torch.Tensor,
        length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if audio_signal.shape[-2] != self._feat_in:
            raise ValueError(
                f"audio_signal should have shape "
                f"(batch, {self._feat_in}, n_frame) but "
                f"got last dimension "
                f"{audio_signal.shape[-2]}."
            )

        return self.forward_internal(
            audio_signal,
            length,
        )

    def forward_internal(
        self,
        audio_signal: torch.Tensor,
        length: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if length is None:
            length = audio_signal.new_full(
                (audio_signal.size(0),),
                audio_signal.size(-1),
                dtype=torch.int64,
                device=audio_signal.device,
            )

        cur_att_context_size = self.att_context_size
        audio_signal = torch.transpose(audio_signal, 1, 2)

        audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
        length = length.to(torch.int64)

        max_audio_length = audio_signal.size(1)

        padding_length = length

        audio_signal, pos_emb = self.pos_enc(x=audio_signal, cache_len=0)

        pad_mask, att_mask = self._create_masks(
            att_context_size=cur_att_context_size,
            padding_length=padding_length,
            max_audio_length=max_audio_length,
            offset=None,
            device=audio_signal.device,
        )

        for lth, layer in enumerate(self.layers):
            audio_signal = layer(
                x=audio_signal,
                att_mask=att_mask,
                pos_emb=pos_emb,
                pad_mask=pad_mask,
            )

        if self.out_proj is not None:
            audio_signal = self.out_proj(audio_signal)

        audio_signal = torch.transpose(audio_signal, 1, 2)
        length = length.to(dtype=torch.int64)

        return audio_signal, length

    def _create_masks(
        self,
        att_context_size: list[int],
        padding_length: torch.Tensor,
        max_audio_length: int,
        offset: torch.Tensor | None,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.self_attention_model != "rel_pos_local_attn":
            att_mask = torch.ones(
                1, max_audio_length, max_audio_length, dtype=torch.bool, device=device
            )

            if self.att_context_style == "regular":
                if att_context_size[0] >= 0:
                    att_mask = att_mask.triu(diagonal=-att_context_size[0])
                if att_context_size[1] >= 0:
                    att_mask = att_mask.tril(diagonal=att_context_size[1])
            elif self.att_context_style == "chunked_limited":
                # When right context is unlimited, just the
                # left side of masking needs to get updated
                if att_context_size[1] == -1:
                    if att_context_size[0] >= 0:
                        att_mask = att_mask.triu(diagonal=-att_context_size[0])
                else:
                    chunk_size = att_context_size[1] + 1
                    # left_chunks_num specifies the number
                    # of chunks to be visible by each chunk
                    # on the left side
                    if att_context_size[0] >= 0:
                        left_chunks_num = att_context_size[0] // chunk_size
                    else:
                        left_chunks_num = 10000

                    chunk_idx = torch.arange(
                        0, max_audio_length, dtype=torch.int, device=att_mask.device
                    )
                    chunk_idx = torch.div(chunk_idx, chunk_size, rounding_mode="trunc")
                    diff_chunks = chunk_idx.unsqueeze(1) - chunk_idx.unsqueeze(0)
                    chunked_limited_mask = torch.logical_and(
                        torch.le(diff_chunks, left_chunks_num), torch.ge(diff_chunks, 0)
                    )
                    att_mask = torch.logical_and(
                        att_mask, chunked_limited_mask.unsqueeze(0)
                    )
        else:
            att_mask = None

        # pad_mask is the masking to be used to ignore paddings
        pad_mask = torch.arange(0, max_audio_length, device=device).expand(
            padding_length.size(0), -1
        ) < padding_length.unsqueeze(-1)

        if offset is not None:
            pad_mask_off = torch.arange(0, max_audio_length, device=device).expand(
                padding_length.size(0), -1
            ) >= offset.unsqueeze(-1)
            pad_mask = pad_mask_off.logical_and(pad_mask)

        if att_mask is not None:
            # pad_mask_for_att_mask is the mask which helps to ignore paddings
            pad_mask_for_att_mask = pad_mask.unsqueeze(1).repeat(
                [1, max_audio_length, 1]
            )
            pad_mask_for_att_mask = torch.logical_and(
                pad_mask_for_att_mask, pad_mask_for_att_mask.transpose(1, 2)
            )
            # att_mask is the masking to be used by MHA
            # layers to ignore tokens not supposed to be
            # visible
            att_mask = att_mask[:, :max_audio_length, :max_audio_length]
            # paddings should also get ignored, so
            # pad_mask_for_att_mask is used to ignore their
            # corresponding scores
            att_mask = torch.logical_and(
                pad_mask_for_att_mask, att_mask.to(pad_mask_for_att_mask.device)
            )
            att_mask = ~att_mask

        pad_mask = ~pad_mask
        return pad_mask, att_mask

    def _calc_context_sizes(
        self,
        att_context_size: list[int] | list[list[int]] | None,
        att_context_probs: list[float] | None,
        att_context_style: str,
        conv_context_size: list[int] | str | None,
        conv_kernel_size: int,
    ) -> tuple[list[list[int]], list[int], list[float], list[int]]:
        # convert att_context_size to a standard list of lists
        if att_context_size:
            att_context_size_all = list(att_context_size)
            if isinstance(att_context_size_all[0], int):
                att_context_size_all = [att_context_size_all]
            for i, att_cs in enumerate(att_context_size_all):
                if att_context_style == "chunked_limited":
                    if att_cs[0] > 0 and att_cs[0] % (att_cs[1] + 1) > 0:
                        raise ValueError(
                            f"att_context_size[{i}][0] % "
                            f"(att_context_size[{i}][1]"
                            f" + 1) should be zero!"
                        )
                    if att_cs[1] < 0 and len(att_context_size_all) <= 1:
                        raise ValueError(
                            f"Right context "
                            f"(att_context_size[{i}][1])"
                            f" can not be unlimited for"
                            f" chunked_limited style!"
                        )
        else:
            att_context_size_all = [[-1, -1]]

        if att_context_probs:
            if len(att_context_probs) != len(att_context_size_all):
                raise ValueError(
                    "The size of the att_context_probs "
                    "should be the same as att_context_size."
                )
            att_context_probs = list(att_context_probs)
            if sum(att_context_probs) != 1:
                raise ValueError(
                    "The sum of numbers in "
                    "att_context_probs should be equal "
                    "to one to be a distribution."
                )
        else:
            att_context_probs = [1.0 / len(att_context_size_all)] * len(
                att_context_size_all
            )

        if conv_context_size is not None:
            if not isinstance(conv_context_size, list) and not isinstance(
                conv_context_size, str
            ):
                raise ValueError(
                    "Invalid conv_context_size! It should "
                    "be the string 'causal' or a list of "
                    "two integers."
                )
            if conv_context_size == "causal":
                conv_context_size = [conv_kernel_size - 1, 0]
            else:
                total = conv_context_size[0] + conv_context_size[1] + 1
                if total != conv_kernel_size:
                    raise ValueError(
                        f"Invalid conv_context_size: {self.conv_context_size}!"
                    )
        else:
            conv_context_size = [
                (conv_kernel_size - 1) // 2,
                (conv_kernel_size - 1) // 2,
            ]
        return (
            att_context_size_all,
            att_context_size_all[0],
            att_context_probs,
            conv_context_size,
        )


# ----- Encoder END -----


# This subclass is specific to vLLM in order for
# `_mark_composite_model` to target this module
class CohereASRProjector(nn.Linear):
    pass


class CohereASRModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.encoder = ConformerEncoder(vllm_config=vllm_config)

        self.decoder = CohereASRDecoder(
            vllm_config=vllm_config, prefix=f"{prefix}.decoder"
        )

        if self.encoder.d_model != self.decoder.hidden_size:
            self.encoder_decoder_proj = CohereASRProjector(
                self.encoder.d_model, self.decoder.hidden_size
            )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        encoder_outputs: list[torch.Tensor],
    ) -> torch.Tensor:
        enc_states = torch.cat(encoder_outputs, dim=0) if len(encoder_outputs) else None
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            positions=positions,
            encoder_hidden_states=enc_states,
        )

        return decoder_outputs

    def get_encoder_outputs(
        self,
        input_features: torch.Tensor | list[torch.Tensor] | None,
        seq_lens: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if input_features is None:
            return None

        if isinstance(input_features, torch.Tensor):
            encoder_input_length = seq_lens
            out, encoder_output_length = self.encoder(
                input_features, length=encoder_input_length
            )  # B x D x T
            out = out.permute(0, 2, 1)

            if hasattr(self, "encoder_decoder_proj"):
                out = self.encoder_decoder_proj(out)

            # Convert padded tensor to packed
            outs = []
            for i, feat in enumerate(out):
                feat_len = encoder_output_length[i]
                outs.append(feat[:feat_len, :])

            return outs
        else:
            raise NotImplementedError("List input_features not supported")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".first_sub_layer.qkv_proj", ".first_sub_layer.query_net", "q"),
            (".first_sub_layer.qkv_proj", ".first_sub_layer.key_net", "k"),
            (".first_sub_layer.qkv_proj", ".first_sub_layer.value_net", "v"),
            (".second_sub_layer.kv_proj", ".second_sub_layer.key_net", "k"),
            (".second_sub_layer.kv_proj", ".second_sub_layer.value_net", "v"),
        ]
        params_dict = dict(self.named_parameters())
        buffers_dict = dict(self.named_buffers())
        params_dict.update(buffers_dict)

        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                # if name.endswith(".bias") and name not in params_dict:
                #     continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)

                # Convert buffer dtype to match loaded weight for pos_bias tensors
                if "pos_bias" in name and param.dtype != loaded_weight.dtype:
                    logger.info(
                        "Converting buffer %s dtype from %s to %s for loading.",
                        name,
                        param.dtype,
                        loaded_weight.dtype,
                    )
                    param.data = param.data.to(loaded_weight.dtype)

                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class CohereASRProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> PretrainedConfig:
        return self.ctx.get_hf_config()

    def get_default_tok_params(self) -> TokenizeParams:
        # Special tokens should be provided by the user based on the
        # task and language of their request. Also needed to avoid
        # appending an EOS token to the prompt which disrupts generation.
        return super().get_default_tok_params().with_kwargs(add_special_tokens=False)

    def get_hf_processor(self, **kwargs: object) -> CohereASRProcessor:
        if not hasattr(self, "_cached_hf_processor"):
            hf_config = self.get_hf_config()
            preproc = hf_config.preprocessor

            sample_rate = preproc.get("sample_rate", 16000)
            window_size = preproc.get("window_size", 0.02)
            window_stride = preproc.get("window_stride", 0.01)

            feature_extractor = CohereASRFeatureExtractor(
                feature_size=preproc.get("features", 64),
                sampling_rate=sample_rate,
                padding_value=preproc.get("pad_value", 0.0),
                max_duration=hf_config.max_audio_clip_s,
                n_window_size=int(window_size * sample_rate),
                n_window_stride=int(window_stride * sample_rate),
                window=preproc.get("window", "hann"),
                normalize=preproc.get("normalize", "per_feature"),
                n_fft=preproc.get("n_fft", None),
                preemph=preproc.get("preemph", 0.97),
                lowfreq=preproc.get("lowfreq", 0),
                highfreq=preproc.get("highfreq", None),
                log=preproc.get("log", True),
                log_zero_guard_type=preproc.get("log_zero_guard_type", "add"),
                log_zero_guard_value=preproc.get("log_zero_guard_value", 2**-24),
                dither=preproc.get("dither", 1e-05),
                pad_to=preproc.get("pad_to", 16),
                frame_splicing=preproc.get("frame_splicing", 1),
                exact_pad=preproc.get("exact_pad", False),
                mag_power=preproc.get("mag_power", 2.0),
                mel_norm=preproc.get("mel_norm", "slaney"),
                stft_exact_pad=preproc.get("stft_exact_pad", False),
                stft_conv=preproc.get("stft_conv", False),
                device="cpu",
            )

            tokenizer = self.ctx.tokenizer
            self._cached_hf_processor = CohereASRProcessor(
                feature_extractor=feature_extractor,
                tokenizer=tokenizer,
            )
        return self._cached_hf_processor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.get_feature_extractor()
        return MultiModalDataParser(target_sr=feature_extractor.sampling_rate)

    def get_feature_extractor(self, **kwargs: object) -> CohereASRFeatureExtractor:
        hf_processor = self.get_hf_processor(**kwargs)
        feature_extractor = hf_processor.feature_extractor
        assert isinstance(feature_extractor, CohereASRFeatureExtractor)
        return feature_extractor

    def get_num_audio_tokens(self, num_samples: int) -> int:
        num_tokens = self.get_feature_extractor().get_seq_len(num_samples)
        config = self.get_hf_config()
        subsampling_factor = config.encoder["subsampling_factor"]
        num_tokens = math.ceil(num_tokens / subsampling_factor)
        return num_tokens


class CohereASRDummyInputsBuilder(BaseDummyInputsBuilder[CohereASRProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)

        return "<|startoftranscript|>" * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options=None,
        mm_processor_kwargs=None,
    ) -> MultiModalDataDict:
        feature_extractor = self.info.get_feature_extractor()

        sampling_rate = feature_extractor.sampling_rate
        audio_len = feature_extractor.max_duration * sampling_rate
        num_audios = mm_counts.get("audio", 0)

        return {
            "audio": self._get_dummy_audios(length=audio_len, num_audios=num_audios)
        }


class CohereASRMultiModalProcessor(EncDecMultiModalProcessor[CohereASRProcessingInfo]):
    skip_decoder_start_token: bool = True

    @property
    def pad_dummy_encoder_prompt(self) -> bool:
        return True

    def create_encoder_prompt(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
    ) -> str | list[int]:
        return [0]

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ):
        if mm_data:
            feature_extractor = self.info.get_feature_extractor(**mm_kwargs)
            mm_data = dict(audio=mm_data.pop("audios"))
            mm_kwargs = dict(
                **mm_kwargs,
                sampling_rate=feature_extractor.sampling_rate,
            )
        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )
        if "labels" in processed_outputs:
            processed_outputs["input_ids"] = processed_outputs.pop("labels")
        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            input_features=MultiModalFieldConfig.batched("audio"),
            length=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        def get_audio_replacement_cohere_asr(item_idx: int):
            audios = mm_items.get_items("audio", AudioProcessorItems)
            audio_len = audios.get_audio_length(item_idx)
            num_tokens = self.info.get_num_audio_tokens(num_samples=audio_len)
            return [0] * num_tokens

        return [
            PromptReplacement(
                modality="audio",
                target=[0],
                replacement=get_audio_replacement_cohere_asr,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    CohereASRMultiModalProcessor,
    info=CohereASRProcessingInfo,
    dummy_inputs=CohereASRDummyInputsBuilder,
)
class CohereASRForConditionalGeneration(
    nn.Module, SupportsTranscription, SupportsMultiModal
):
    packed_modules_mapping = {
        "self_attn.qkv_proj": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
        ],
        "encoder_attn.kv_proj": ["encoder_attn.k_proj", "encoder_attn.v_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={".fc1.": ".mlp.fc1.", ".fc2.": ".mlp.fc2."}
    )

    supports_transcription_only = True
    supported_languages = ISO639_1_SUPPORTED_LANGS
    skip_warmup_audio_preprocessing = True

    @classmethod
    def validate_language(cls, language: str | None) -> str | None:
        if language is None:
            logger.warning(
                "Defaulting to language='en'. If you wish to transcribe "
                "audio in a different language, pass the `language` field "
                "in the TranscriptionRequest."
            )
            language = "en"
        return super().validate_language(language)

    @classmethod
    def get_generation_prompt(
        cls,
        audio: np.ndarray,
        model_config: ModelConfig,  # not needed here
        stt_config: SpeechToTextConfig,
        language: str | None,
        task_type: Literal["transcribe", "translate"],
        request_prompt: str,
        to_language: str | None,
    ) -> PromptType:
        if language is None:
            raise ValueError(
                "Language must be specified when creating the CohereASR prompt"
            )

        # NOTE: this function is used only by online inference and not offline inference
        # CohereASR doesnt have encoder prompt
        language_tag = f"<|{language}|><|{language}|>"
        pnc = True  # TODO(ekagra): make this configurable later
        pnc_tag = "<|pnc|>" if pnc else "<|nopnc|>"
        default_prompt = (
            f"<|startofcontext|><|startoftranscript|>"
            f"<|emo:undefined|>{language_tag}{pnc_tag}"
            f"<|noitn|><|notimestamp|><|nodiarize|>"
        )
        prompt_text = request_prompt if request_prompt else default_prompt

        return TextPrompt(
            prompt=prompt_text,
            multi_modal_data={"audio": (audio, stt_config.sample_rate)},
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        # Required as part of SupportsMultiModal interface.
        if modality.startswith("audio"):
            return None

        raise ValueError("Only audio modality is supported")

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig:
        sampling_rate = model_config.hf_config.sample_rate
        assert sampling_rate == 16000
        max_audio_clip_s = model_config.hf_config.max_audio_clip_s
        overlap_chunk_second = model_config.hf_config.overlap_chunk_second

        return SpeechToTextConfig(
            max_audio_clip_s=max_audio_clip_s,
            overlap_chunk_second=overlap_chunk_second,
            sample_rate=sampling_rate,
        )

    @classmethod
    def get_num_audio_tokens(
        cls,
        audio_duration_s: float,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
    ) -> int | None:
        hop_length = model_config.hf_config.preprocessor.get("window_stride")
        assert hop_length is not None
        return math.ceil(audio_duration_s * stt_config.sample_rate / hop_length)

    def get_num_encoder_cross_attn_tokens(self, num_encoder_input_tokens: int) -> int:
        return self.model.encoder.get_num_encoder_cross_attn_tokens(
            num_encoder_input_tokens
        )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.dtype = vllm_config.model_config.dtype

        with self._mark_composite_model(
            vllm_config,
            language_targets=CohereASRDecoder,
            tower_targets={"audio": (ConformerEncoder, CohereASRProjector)},
        ):
            self.model = CohereASRModel(vllm_config=vllm_config, prefix=prefix)

        head_config = config.head

        self.proj_out = ParallelLMHead(
            head_config["num_classes"],
            head_config["hidden_size"],
            quant_config=quant_config,
            bias=True,
        )  # NOTE: bias is True

        logit_scale = getattr(head_config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            head_config["num_classes"], scale=logit_scale
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_outputs: list[torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if encoder_outputs is None:
            encoder_outputs = []
        decoder_outputs = self.model(
            input_ids=input_ids,
            positions=positions,
            encoder_outputs=encoder_outputs,
        )

        return decoder_outputs

    def get_language_model(self) -> torch.nn.Module:
        # Required as part of SupportsMultiModal interface.
        return self.model.decoder

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        # Required as part of SupportsMultiModal interface.
        audio_input, seq_lens = self._parse_and_validate_audio_input(**kwargs)

        if hasattr(audio_input, "input_features"):
            out = self.model.get_encoder_outputs(audio_input["input_features"])
        else:
            out = self.model.get_encoder_outputs(audio_input, seq_lens)

        return out

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_features = kwargs.pop("input_features", None)
        length = kwargs.pop("length", None)

        if input_features is None:
            raise ValueError("Audio features are required for CohereASR model.")

        if not isinstance(input_features, (torch.Tensor, list)):
            raise ValueError(
                f"Incorrect type of audio features. Got type: {type(input_features)}"
            )

        if isinstance(input_features, torch.Tensor):
            seq_lens = length.reshape(-1)
        else:
            input_features = [
                feat.to(self.dtype).squeeze(0).transpose(1, 0)
                for feat in input_features
            ]
            seq_lens = length.reshape(-1)
            input_features = torch.nn.utils.rnn.pad_sequence(
                input_features, batch_first=True, padding_value=0.0
            )
            input_features = input_features.transpose(1, 2)

        return input_features, seq_lens

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.logits_processor(self.proj_out, hidden_states, self.proj_out.bias)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def transform(inputs):
            name, loaded_weight = inputs

            if name.startswith("transf_decoder._decoder"):
                name = name.replace("transf_decoder._decoder", "decoder")
            if name.startswith("transf_decoder._embedding"):
                name = name.replace("transf_decoder._embedding", "decoder.embedding")
            if "second_sub_layer.query_net" in name:
                name = name.replace(
                    "second_sub_layer.query_net", "second_sub_layer.q_proj"
                )

            if name in ["log_softmax.mlp.layer0.weight", "log_softmax.mlp.layer0.bias"]:
                name = name.replace("log_softmax.mlp.layer0", "proj_out")
            else:
                name = "model." + name

            return name, loaded_weight

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=[
                "model.preprocessor.featurizer.fb",
                "model.preprocessor.featurizer.window",
            ],
            skip_substrs=["model.conv.batch_norm.num_batches_tracked"],
        )

        return loader.load_weights(
            map(transform, weights), mapper=self.hf_to_vllm_mapper
        )
