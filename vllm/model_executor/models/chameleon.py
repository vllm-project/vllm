# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property
from typing import Any, Literal, Optional, TypedDict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (BatchFeature, ChameleonConfig, ChameleonProcessor,
                          ChameleonVQVAEConfig)

from vllm.attention import Attention
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, row_parallel_weight_loader)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.utils import set_weight_attrs
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate, PromptUpdateDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors

from .interfaces import (MultiModalEmbeddings, SupportsMultiModal, SupportsPP,
                         SupportsQuant)
from .utils import (flatten_bn, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix, merge_multimodal_embeddings)

logger = init_logger(__name__)


class ChameleonImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    """Shape: `(batch_size * num_images, num_channels, height, width)`"""


class ChameleonProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(ChameleonConfig)

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(ChameleonProcessor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": 1}

    def get_num_image_tokens(self) -> int:
        processor = self.get_hf_processor()
        return processor.image_seq_length


class ChameleonDummyInputsBuilder(
        BaseDummyInputsBuilder[ChameleonProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        processor = self.info.get_hf_processor()
        image_token = processor.image_token

        return image_token * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        config = self.info.get_hf_config()

        width = height = config.vq_config.resolution
        num_images = mm_counts.get("image", 0)

        return {
            "image":
            self._get_dummy_images(width=width,
                                   height=height,
                                   num_images=num_images)
        }


class ChameleonMultiModalProcessor(
        BaseMultiModalProcessor[ChameleonProcessingInfo]):

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if not mm_data:
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        return super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
        )

    def _apply_hf_processor_tokens_only(
        self,
        prompt_tokens: list[int],
    ) -> list[int]:
        # HF processor adds sep token for chat mode
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        sep_token_id = vocab[tokenizer.sep_token]  # type: ignore

        return prompt_tokens + [sep_token_id]

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(pixel_values=MultiModalFieldConfig.batched("image"))

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        image_start_id = vocab[processor.image_start_token]
        image_token_id = vocab[processor.image_token]
        image_end_id = vocab[processor.image_end_token]

        num_image_tokens = self.info.get_num_image_tokens()
        image_tokens = [image_token_id] * num_image_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=PromptUpdateDetails.select_token_id(
                    [image_start_id] + image_tokens + [image_end_id],
                    embed_token_id=image_token_id,
                ),
            )
        ]


class ChameleonLayerNorm(nn.LayerNorm):

    def __init__(self, hidden_size, *args, **kwargs):
        super().__init__(hidden_size, *args, **kwargs)
        self.normalized_shape = (hidden_size[-1], )

        set_weight_attrs(self.weight,
                         {"weight_loader": row_parallel_weight_loader})
        set_weight_attrs(self.bias,
                         {"weight_loader": row_parallel_weight_loader})

    def forward(self, hidden_states):
        hidden_states = F.layer_norm(hidden_states,
                                     self.normalized_shape,
                                     None,
                                     None,
                                     eps=1e-5)
        hidden_states = hidden_states * self.weight + self.bias
        return hidden_states


# Copied from vllm.model_executor.models.llama.LlamaMLP -> ChameleonMLP
class ChameleonMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config)
        self.down_proj = RowParallelLinear(input_size=intermediate_size,
                                           output_size=hidden_size,
                                           bias=bias,
                                           quant_config=quant_config)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


# Modified from vllm.model_executor.models.llama.LlamaAttention -> ChameleonAttention #noqa
class ChameleonAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict[str, Any]] = None,
        max_position_embeddings: int = 4096,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
        )
        self.q_norm = ChameleonLayerNorm((self.num_heads, self.head_dim))
        self.k_norm = ChameleonLayerNorm((self.num_kv_heads, self.head_dim))
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn")

    def _apply_qk_norm(self, q: torch.Tensor,
                       k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # reshape for layernorm
        q = q.reshape(-1, self.num_heads, self.head_dim)
        k = k.reshape(-1, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.view(*q.shape[:-2], -1)
        k = k.view(*k.shape[:-2], -1)
        return q, k

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self._apply_qk_norm(q, k)

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class ChameleonDecoderLayer(nn.Module):

    def __init__(
        self,
        config: ChameleonConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          4096)

        self.self_attn = ChameleonAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=False,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = ChameleonMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class ChameleonSwinDecoderLayer(nn.Module):

    def __init__(
        self,
        config: ChameleonConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          4096)

        self.self_attn = ChameleonAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=False,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = ChameleonMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:

        residual = hidden_states
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        # Fully Connected
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, residual


# Copied from transformers.models.chameleon.modeling_chameleon.ChameleonVQVAEVectorQuantizer #noqa
class ChameleonVQVAEVectorQuantizer(nn.Module):

    def __init__(self, config: ChameleonVQVAEConfig):
        super().__init__()
        self.num_embeddings = config.num_embeddings
        self.embedding_dim = config.embed_dim
        self.beta = getattr(config, "beta", 0.25)

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.re_embed = self.num_embeddings

    def forward(self, hidden_state: torch.Tensor):
        hidden_state = hidden_state.permute(0, 2, 3, 1).contiguous()
        hidden_state_flattened = hidden_state.view(-1, self.embedding_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        distances = (
            torch.sum(hidden_state_flattened**2, dim=1, keepdim=True) +
            torch.sum(self.embedding.weight**2, dim=1) -
            2 * torch.einsum("bd,dn->bn", hidden_state_flattened,
                             self.embedding.weight.transpose(0, 1)))

        min_encoding_indices = torch.argmin(distances, dim=1)
        hidden_state_quant = self.embedding(min_encoding_indices).view(
            hidden_state.shape)

        # compute loss for embedding
        loss = torch.mean((hidden_state_quant.detach() - hidden_state)**
                          2) + self.beta * torch.mean(
                              (hidden_state_quant - hidden_state.detach())**2)

        # preserve gradients
        hidden_state_quant = hidden_state + (hidden_state_quant -
                                             hidden_state).detach()

        # reshape back to match original input shape
        hidden_state_quant = hidden_state_quant.permute(0, 3, 1,
                                                        2).contiguous()

        return hidden_state_quant, loss, min_encoding_indices


# Copied from transformers.models.chameleon.modeling_chameleon.ChameleonVQVAEEncoderConvDownsample #noqa
class ChameleonVQVAEEncoderConvDownsample(nn.Module):

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              in_channels,
                              kernel_size=3,
                              stride=2,
                              padding=0)

    def forward(self, hidden_states: torch.Tensor):
        # no asymmetric padding in torch conv, must do it ourselves
        hidden_states = F.pad(hidden_states,
                              pad=(0, 1, 0, 1),
                              mode="constant",
                              value=0)
        hidden_states = self.conv(hidden_states)
        return hidden_states


# Copied from transformers.models.chameleon.modeling_chameleon.ChameleonVQVAEEncoderResnetBlock #noqa
class ChameleonVQVAEEncoderResnetBlock(nn.Module):

    def __init__(
        self,
        config: ChameleonVQVAEConfig,
        in_channels: int,
        out_channels=None,
        conv_shortcut=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None \
            else out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = torch.nn.GroupNorm(num_groups=32,
                                        num_channels=in_channels,
                                        eps=1e-6,
                                        affine=True)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.norm2 = torch.nn.GroupNorm(num_groups=32,
                                        num_channels=out_channels,
                                        eps=1e-6,
                                        affine=True)
        self.dropout = torch.nn.Dropout(config.dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, hidden_states: torch.Tensor):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states *= torch.sigmoid(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states *= torch.sigmoid(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                residual = self.conv_shortcut(residual)
            else:
                residual = self.nin_shortcut(residual)

        return residual + hidden_states


# Copied from transformers.models.chameleon.modeling_chameleon.ChameleonVQVAEEncoderAttnBlock #noqa
class ChameleonVQVAEEncoderAttnBlock(nn.Module):

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=32,
                                       num_channels=in_channels,
                                       eps=1e-6,
                                       affine=True)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, hidden_states: torch.Tensor):
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        query_states = self.q(hidden_states)
        key_states = self.k(hidden_states)
        value_states = self.v(hidden_states)

        # compute attention
        batch_size, channels, height, width = query_states.shape
        query_states = query_states.reshape(batch_size, channels,
                                            height * width).permute(0, 2, 1)
        key_states = key_states.reshape(batch_size, channels, height * width)
        attn_weights = torch.bmm(query_states, key_states)
        attn_weights = attn_weights * (int(channels)**(-0.5))
        attn_weights = F.softmax(attn_weights, dim=2)

        # attend to values
        value_states = value_states.reshape(batch_size, channels,
                                            height * width)
        attn_weights = attn_weights.permute(0, 2, 1)
        attn_output = torch.bmm(value_states,
                                attn_weights).reshape(batch_size, channels,
                                                      height, width)

        attn_output = self.proj_out(attn_output)
        return residual + attn_output


# Copied from transformers.models.chameleon.modeling_chameleon.ChameleonVQVAEEncoder #noqa
class ChameleonVQVAEEncoder(nn.Module):

    def __init__(self, config: ChameleonVQVAEConfig):
        super().__init__()

        self.num_resolutions = len(config.channel_multiplier)
        self.num_res_blocks = config.num_res_blocks
        base_channels = config.base_channels
        resolution = config.resolution
        in_channels = config.in_channels
        double_latent = config.double_latent
        latent_channels = config.latent_channels
        channel_multiplier = config.channel_multiplier

        self.conv_in = torch.nn.Conv2d(in_channels,
                                       base_channels,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_channel_multiplier = (1, ) + tuple(channel_multiplier)
        self.in_channel_multiplier = in_channel_multiplier
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = base_channels * in_channel_multiplier[i_level]
            block_out = base_channels * channel_multiplier[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ChameleonVQVAEEncoderResnetBlock(
                        config=config,
                        in_channels=block_in,
                        out_channels=block_out,
                    ))
                block_in = block_out
                if (config.attn_resolutions is not None
                        and curr_res in config.attn_resolutions
                        and config.attn_type == "vanilla"):
                    attn.append(ChameleonVQVAEEncoderAttnBlock(block_in))

            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = ChameleonVQVAEEncoderConvDownsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ChameleonVQVAEEncoderResnetBlock(
            config=config,
            in_channels=block_in,
            out_channels=block_in,
        )
        self.mid.attn_1 = ChameleonVQVAEEncoderAttnBlock(
            block_in) if config.attn_type == "vanilla" else nn.Identity()
        self.mid.block_2 = ChameleonVQVAEEncoderResnetBlock(
            config=config,
            in_channels=block_in,
            out_channels=block_in,
        )

        self.norm_out = torch.nn.GroupNorm(num_groups=32,
                                           num_channels=block_in,
                                           eps=1e-6,
                                           affine=True)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            2 * latent_channels if double_latent else latent_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, pixel_values: torch.Tensor):
        pixel_values = pixel_values.to(self.conv_in.weight.dtype)

        # downsampling
        hidden_states = [self.conv_in(pixel_values)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                hidden_state = self.down[i_level].block[i_block](
                    hidden_states[-1])
                if len(self.down[i_level].attn) > 0:
                    hidden_state = self.down[i_level].attn[i_block](
                        hidden_state)
                hidden_states.append(hidden_state)
            if i_level != self.num_resolutions - 1:
                hidden_states.append(self.down[i_level].downsample(
                    hidden_states[-1]))

        # middle
        last_hidden_state = hidden_states[-1]
        last_hidden_state = self.mid.block_1(last_hidden_state)
        last_hidden_state = self.mid.attn_1(last_hidden_state)
        last_hidden_state = self.mid.block_2(last_hidden_state)

        # end
        last_hidden_state = self.norm_out(last_hidden_state)
        last_hidden_state *= torch.sigmoid(last_hidden_state)
        last_hidden_state = self.conv_out(last_hidden_state)
        return last_hidden_state


# Adapted from transformers.models.chameleon.modeling_chameleon.ChameleonVQVAE #noqa
class ChameleonVQVAE(nn.Module):

    def __init__(self, config: ChameleonVQVAEConfig):
        super().__init__()
        self.encoder = ChameleonVQVAEEncoder(config)
        self.quantize = ChameleonVQVAEVectorQuantizer(config)
        self.quant_conv = torch.nn.Conv2d(config.latent_channels,
                                          config.embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(config.embed_dim,
                                               config.latent_channels, 1)
        self.eval()  # Chameleon's VQ model is frozen

    def encode(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = self.encoder(pixel_values)
        hidden_states = self.quant_conv(hidden_states)
        quant, emb_loss, indices = self.quantize(hidden_states)
        return quant, emb_loss, indices


# Copied from transformers.models.chameleon.modeling_chameleon.ChameleonImageVocabularyMapping #noqa
class ChameleonImageVocabularyMapping:
    """
    A class for mapping discrete image tokens from VQGAN to BPE tokens.
    """

    def __init__(self, vocab_map: dict[str, int]):
        self.vocab_map = vocab_map
        self.image_token_id = vocab_map.get("<image>")

    @cached_property
    def val2name(self):
        return {v: k for k, v in self.vocab_map.items()}

    @cached_property
    def image_tokens(self):
        return sorted([
            val for name, val in self.vocab_map.items()
            if name.startswith("IMGIMG")
        ])

    @cached_property
    def bpe2img(self):
        img_tkn_chr_mapping = {chr(ord("A") + i): str(i) for i in range(10)}

        def remap(old_name: str) -> str:
            return "".join(
                img_tkn_chr_mapping.get(c, c)
                for c in old_name[len("IMGIMG"):-1])

        return {
            tok: int(remap(self.val2name[tok]))
            for tok in self.image_tokens
        }

    @cached_property
    def img2bpe(self):
        return {v: k for k, v in self.bpe2img.items()}

    @cached_property
    def bpe2img_search_tensors(self):
        return torch.tensor(sorted(self.bpe2img.keys())), torch.tensor(
            sorted(self.bpe2img.values()))

    @cached_property
    def img2bpe_mapping_tensor(self):
        mapping = torch.zeros(max(self.img2bpe.keys()) + 1, dtype=torch.int)
        for k, v in self.img2bpe.items():
            mapping[k] = v
        return mapping

    def convert_img2bpe(self, img_batch: torch.Tensor) -> torch.Tensor:
        device = img_batch.device
        img_tokens = self.img2bpe_mapping_tensor[img_batch.to("cpu")]
        return img_tokens.to(device)


class ChameleonModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )
        self.vocabulary_mapping = ChameleonImageVocabularyMapping(
            config.vocabulary_map)
        decoder_layer = ChameleonDecoderLayer if not self.config.swin_norm \
            else ChameleonSwinDecoderLayer

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: decoder_layer(config=config,
                                         cache_config=cache_config,
                                         quant_config=quant_config,
                                         prefix=prefix),
            prefix=f"{prefix}.layers",
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.vqmodel = ChameleonVQVAE(config.vq_config)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def get_image_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Tokenizes images into discrete tokens with VQGAN module. Converts
        obtained image tokens into BPE tokens and wraps with "boi" and "eoi"
        special tokens.
        """
        batch_size = pixel_values.shape[0]
        _, _, image_toks = self.vqmodel.encode(pixel_values)
        bpe_toks = self.vocabulary_mapping.convert_img2bpe(image_toks)
        bpe_toks = bpe_toks.view(batch_size, -1)
        return bpe_toks

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for layer in self.layers[self.start_layer:self.end_layer]:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


@MULTIMODAL_REGISTRY.register_processor(
    ChameleonMultiModalProcessor,
    info=ChameleonProcessingInfo,
    dummy_inputs=ChameleonDummyInputsBuilder)
class ChameleonForConditionalGeneration(nn.Module, SupportsMultiModal,
                                        SupportsPP, SupportsQuant):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"]
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config
        self.model = ChameleonModel(vllm_config=vllm_config,
                                    prefix=maybe_prefix(prefix, "model"))
        self.unpadded_vocab_size = config.vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size, logit_scale)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def _validate_pixel_values(self, data: torch.Tensor) -> torch.Tensor:
        vq_config: ChameleonVQVAEConfig = self.config.vq_config
        expected_dims = (3, vq_config.resolution, vq_config.resolution)
        actual_dims = tuple(data.shape[1:])

        if actual_dims != expected_dims:
            expected_expr = ("batch_size", *map(str, expected_dims))
            raise ValueError(
                f"The expected shape of pixel values is {expected_expr}. "
                f"You supplied {tuple(data.shape)}.")

        return data

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[ChameleonImagePixelInputs]:
        pixel_values = kwargs.pop("pixel_values", None)

        if pixel_values is None:
            return None

        if not isinstance(pixel_values, (torch.Tensor, list)):
            raise ValueError("Incorrect type of pixel values. "
                             f"Got type: {type(pixel_values)}")

        pixel_values = flatten_bn(pixel_values, concat=True)

        return ChameleonImagePixelInputs(
            type="pixel_values",
            data=self._validate_pixel_values(pixel_values),
        )

    def get_language_model(self) -> torch.nn.Module:
        return self.model

    def get_multimodal_embeddings(self,
                                  **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []
        assert self.model.vqmodel is not None
        image_tokens = self.model.get_image_tokens(image_input["data"].to(
            self.config.torch_dtype))
        vision_embeddings = self.model.get_input_embeddings(image_tokens)
        return vision_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:

        inputs_embeds = self.model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None \
            and len(multimodal_embeddings) != 0:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                self.model.vocabulary_mapping.image_token_id)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:

        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None

        hidden_states = self.model(input_ids,
                                   positions,
                                   intermediate_tensors,
                                   inputs_embeds=inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)

        # Disallow image tokens which does not include special
        # begin-image and end-image tokens
        if logits is not None:
            image_tokens = self.model.vocabulary_mapping.image_tokens
            logits[:, image_tokens] = torch.finfo(logits.dtype).min

        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue

            # With tie_word_embeddings, we can skip lm_head.weight
            # The weight might appear unnecessarily in the files if the model is
            # processed with quantization, LoRA, fine-tuning, etc.
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue

            use_default_weight_loading = False
            if "vqmodel" in name:
                if self.model.vqmodel is not None:
                    # We only do sharding for language model and
                    # not vqvae for now.
                    use_default_weight_loading = True
            else:
                for (param_name, weight_name,
                     shard_id) in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    # Remapping the name of FP8 kv-scale.
                    if name.endswith("kv_scale"):
                        remapped_kv_scale_name = name.replace(
                            ".kv_scale", ".attn.kv_scale")
                        if remapped_kv_scale_name not in params_dict:
                            logger.warning_once(
                                "Found kv scale in the checkpoint (e.g. %s), but not found the expected name in the model (e.g. %s). kv-scale is not loaded.",  # noqa: E501
                                name,
                                remapped_kv_scale_name,
                            )
                            continue
                        else:
                            name = remapped_kv_scale_name
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            if use_default_weight_loading and name in params_dict:
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
