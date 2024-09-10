from array import array
from functools import cached_property
from typing import (Any, Dict, Iterable, List, Literal, Mapping, Optional,
                    Tuple, TypedDict)

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from transformers import ChameleonConfig, ChameleonVQVAEConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, MultiModalConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.inputs import INPUT_REGISTRY, InputContext, LLMInputs
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, row_parallel_weight_loader)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.utils import set_weight_attrs
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.utils import (cached_get_tokenizer,
                                   repeat_and_pad_placeholder_tokens)
from vllm.sequence import (VLLM_TOKEN_ID_ARRAY_TYPE, IntermediateTensors,
                           SequenceData)
from vllm.utils import print_warning_once

from .interfaces import SupportsMultiModal

logger = init_logger(__name__)

# These configs are not part of the model config but the preprocessor
# and processor files, so we hardcode them in the model file for now.
CHAMELEON_CROP_SIZE_HEIGHT = CHAMELEON_CROP_SIZE_WIDTH = 512
CHAMELEON_IMAGE_SEQ_LENGTH = 1024
CHAMELEON_IMAGE_TOKEN_ID = 8711
CHAMELEON_IMAGE_START_TOKEN_ID = 8197
CHAMELEON_IMAGE_END_TOKEN_ID = 8196
CHAMELEON_SEP_TOKEN_ID = 8710


class ChameleonImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    """Shape: `(batch_size * num_images, num_channels, height, width)`"""


def get_max_chameleon_image_tokens(ctx: InputContext):
    return CHAMELEON_IMAGE_SEQ_LENGTH


def dummy_seq_data_for_chameleon(
    seq_len: int,
    num_images: int,
    *,
    image_token_id: int,
    image_feature_size_override: Optional[int] = None,
):
    if image_feature_size_override is None:
        image_feature_size = CHAMELEON_IMAGE_SEQ_LENGTH
    else:
        image_feature_size = image_feature_size_override

    token_ids = array(VLLM_TOKEN_ID_ARRAY_TYPE,
                      [image_token_id]) * image_feature_size * num_images
    token_ids += array(VLLM_TOKEN_ID_ARRAY_TYPE,
                       [0]) * (seq_len - image_feature_size * num_images)
    return SequenceData(token_ids)


def dummy_image_for_chameleon(
    num_images: int,
    *,
    image_width_override: Optional[int] = None,
    image_height_override: Optional[int] = None,
):
    width = CHAMELEON_CROP_SIZE_WIDTH
    height = CHAMELEON_CROP_SIZE_HEIGHT
    if image_width_override is not None:
        width = image_width_override
    if image_height_override is not None:
        height = image_height_override

    image = Image.new("RGB", (width, height), color=0)
    return {"image": image if num_images == 1 else [image] * num_images}


def dummy_data_for_chameleon(ctx: InputContext, seq_len: int,
                             mm_counts: Mapping[str, int]):
    num_images = mm_counts["image"]

    seq_data = dummy_seq_data_for_chameleon(
        seq_len,
        num_images,
        image_token_id=CHAMELEON_IMAGE_TOKEN_ID,
    )

    mm_data = dummy_image_for_chameleon(num_images)
    return seq_data, mm_data


def input_processor_for_chameleon(ctx: InputContext, llm_inputs: LLMInputs):

    """
    Processing input prompt to insert required tokens for image placeholder.

    See https://github.com/huggingface/transformers/blob/0fdea8607d7e01eb0e38a1ebeb7feee30a22f0cf/src/transformers/models/chameleon/processing_chameleon.py#L58
    """ # noqa

    multi_modal_data = llm_inputs.get("multi_modal_data")
    if multi_modal_data is None or "image" not in multi_modal_data:
        return llm_inputs

    model_config = ctx.model_config
    tokenizer = cached_get_tokenizer(model_config.tokenizer)
    new_prompt, new_token_ids = repeat_and_pad_placeholder_tokens(
        tokenizer,
        llm_inputs.get("prompt"),
        llm_inputs["prompt_token_ids"],
        placeholder_token_id=CHAMELEON_IMAGE_TOKEN_ID,
        repeat_count=CHAMELEON_IMAGE_SEQ_LENGTH,
        pad_token_left=CHAMELEON_IMAGE_START_TOKEN_ID,
        pad_token_right=CHAMELEON_IMAGE_END_TOKEN_ID,
    )

    # Appending sep token for chat mode to follow default processor
    # behavior
    if new_prompt is not None:
        new_prompt += tokenizer.sep_token
    new_token_ids += [CHAMELEON_SEP_TOKEN_ID]

    # NOTE: Create a defensive copy of the original inputs
    return LLMInputs(prompt_token_ids=new_token_ids,
                     prompt=new_prompt,
                     multi_modal_data=multi_modal_data)


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
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 4096,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        cache_config: Optional[CacheConfig] = None,
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
                              quant_config=quant_config)

    def _apply_qk_norm(self, q: torch.Tensor,
                       k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self._apply_qk_norm(q, k)

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class ChameleonDecoderLayer(nn.Module):

    def __init__(
        self,
        config: ChameleonConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
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
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
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
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        residual = hidden_states
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
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
                    hidden_states[-1], )
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = self.encoder(pixel_values)
        hidden_states = self.quant_conv(hidden_states)
        quant, emb_loss, indices = self.quantize(hidden_states)
        return quant, emb_loss, indices


# Copied from transformers.models.chameleon.modeling_chameleon.ChameleonImageVocabularyMapping #noqa
class ChameleonImageVocabularyMapping:
    """
    A class for mapping discrete image tokens from VQGAN to BPE tokens.
    """

    def __init__(self, vocab_map: Dict[str, int]):
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

    def __init__(
        self,
        config: ChameleonConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )
        self.vocabulary_mapping = ChameleonImageVocabularyMapping(
            config.vocabulary_map)
        decoder_layer = ChameleonDecoderLayer if not self.config.swin_norm \
            else ChameleonSwinDecoderLayer
        self.layers = nn.ModuleList([
            decoder_layer(config=config,
                          cache_config=cache_config,
                          quant_config=quant_config)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.vqmodel = ChameleonVQVAE(config.vq_config)

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
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids)
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                attn_metadata,
                residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


@MULTIMODAL_REGISTRY.register_image_input_mapper()
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_chameleon_image_tokens)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_chameleon)
@INPUT_REGISTRY.register_input_processor(input_processor_for_chameleon)
class ChameleonForConditionalGeneration(nn.Module, SupportsMultiModal):

    def __init__(
        self,
        config: ChameleonConfig,
        multimodal_config: MultiModalConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.multimodal_config = multimodal_config
        self.model = ChameleonModel(config, cache_config, quant_config)
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
        self.sampler = Sampler()

    def _validate_pixel_values(self, data: torch.Tensor) -> torch.Tensor:

        expected_dims = (3, CHAMELEON_CROP_SIZE_HEIGHT,
                         CHAMELEON_CROP_SIZE_WIDTH)
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

        if not isinstance(pixel_values, torch.Tensor):
            raise ValueError("Incorrect type of pixel values. "
                             f"Got type: {type(pixel_values)}")

        # Remove the N dimension until multiple images are supported.
        pixel_values = pixel_values.squeeze(1)

        return ChameleonImagePixelInputs(
            type="pixel_values",
            data=self._validate_pixel_values(pixel_values),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs,
    ) -> torch.Tensor:

        image_input = self._parse_and_validate_image_input(**kwargs)

        if image_input is not None:
            assert self.model.vqmodel is not None
            image_tokens = self.model.get_image_tokens(image_input["data"].to(
                self.config.torch_dtype))
            image_token_id = self.model.vocabulary_mapping.image_token_id
            special_image_mask = input_ids == image_token_id
            image_tokens = image_tokens.to(input_ids.device, input_ids.dtype)
            input_ids = input_ids.masked_scatter(special_image_mask,
                                                 image_tokens)

        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata)
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

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
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
                            print_warning_once(
                                "Found kv scale in the checkpoint (e.g. "
                                f"{name}), but not found the expected name in "
                                f"the model (e.g. {remapped_kv_scale_name}). "
                                "kv-scale is not loaded.")
                            continue
                        else:
                            name = remapped_kv_scale_name
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            if use_default_weight_loading and name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
