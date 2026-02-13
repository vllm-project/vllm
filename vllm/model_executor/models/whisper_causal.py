# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import functools
import math
from dataclasses import replace
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.mistral import MistralMLP
from vllm.model_executor.models.whisper import WhisperPosEmbedType
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    AttentionType,
    CommonAttentionMetadata,
    subclass_attention_backend_with_overrides,
)
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.kv_cache_interface import AttentionSpec

from .utils import make_layers

CausalRMSNorm = partial(RMSNorm, eps=1e-5)


def _pad1d(
    x: torch.Tensor,
    paddings: tuple[int, int],
    mode: str = "constant",
    value: float = 0.0,
) -> torch.Tensor:
    """Tiny wrapper around F.pad, just to allow for
    reflect padding on small input.
    If this is the case, we insert extra 0 padding
    to the right before the reflection happen.
    """
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


class WhisperCausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self._stride = self.stride[0]
        self._effective_kernel_size = (kernel_size - 1) * self.dilation[0] + 1
        self._padding_total = self._effective_kernel_size - self._stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_frames = (
            x.shape[-1] - self._effective_kernel_size + self._padding_total
        ) / self._stride + 1
        target_length = (math.ceil(n_frames) - 1) * self._stride + (
            self._effective_kernel_size - self._padding_total
        )
        extra_padding = target_length - x.shape[-1]
        x = _pad1d(x, (self._padding_total, extra_padding), mode="constant")
        return super().forward(x)


@functools.lru_cache
def create_whisper_attention_backend_with_block_pooling(
    underlying_attn_backend: AttentionBackend, block_pool_size: int
) -> type[AttentionBackend]:
    prefix = "WhisperCausalAttentionWithBlockPooling_"
    underlying_builder = underlying_attn_backend.get_builder_cls()
    underlying_impl = underlying_attn_backend.get_impl_cls()

    class WhisperCausalAttentionWithBlockPoolingBuilder(underlying_builder):  # type: ignore
        def __init__(
            self,
            kv_cache_spec: AttentionSpec,
            layer_names: list[str],
            vllm_config: VllmConfig,
            device: torch.device,
        ):
            assert kv_cache_spec.num_kv_heads % block_pool_size == 0
            kv_cache_spec = replace(
                kv_cache_spec,
                block_size=kv_cache_spec.block_size * block_pool_size,
                num_kv_heads=kv_cache_spec.num_kv_heads // block_pool_size,
            )
            super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
            fast_build: bool = False,
        ) -> AttentionMetadata:
            new_common_attn_metadata = copy.deepcopy(common_attn_metadata)
            new_common_attn_metadata.query_start_loc *= block_pool_size
            new_common_attn_metadata.query_start_loc_cpu *= block_pool_size
            new_common_attn_metadata.seq_lens *= block_pool_size
            new_common_attn_metadata._seq_lens_cpu *= block_pool_size
            new_common_attn_metadata._num_computed_tokens_cpu *= block_pool_size
            new_common_attn_metadata.num_actual_tokens *= block_pool_size
            new_common_attn_metadata.max_query_len *= block_pool_size
            new_common_attn_metadata.max_seq_len *= block_pool_size
            original_slot_mapping = common_attn_metadata.slot_mapping
            common_prefix_len *= block_pool_size
            new_common_attn_metadata.slot_mapping = (
                (
                    original_slot_mapping.unsqueeze(1) * block_pool_size
                    + torch.arange(block_pool_size, device=original_slot_mapping.device)
                )
                .flatten()
                .clamp(min=-1)
            )
            return super().build(
                common_prefix_len, new_common_attn_metadata, fast_build
            )

    # NOTE: We need a custom impl so we can use the transformed slot_mapping
    # computed by `WhisperCausalAttentionWithBlockPoolingBuilder` instead of
    # the one from `forward_context.slot_mapping` (gpu_model_runner).
    # This follows the same pattern as CrossAttentionImpl.
    class WhisperCausalAttentionWithBlockPoolingImpl(underlying_impl):  # type: ignore[valid-type,misc]
        def forward(
            self,
            layer: torch.nn.Module,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata,
            output: torch.Tensor | None = None,
            output_scale: torch.Tensor | None = None,
            output_block_scale: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if (
                not underlying_attn_backend.forward_includes_kv_cache_update
                and attn_metadata is not None
                and layer.kv_sharing_target_layer_name is None
                and key is not None
                and value is not None
            ):
                self.do_kv_cache_update(
                    layer, key, value, kv_cache, attn_metadata.slot_mapping
                )

            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

    if not issubclass(underlying_attn_backend, FlashAttentionBackend):
        raise NotImplementedError(
            f"{underlying_attn_backend} is not yet supported."
            "Contributions to support more backends are much "
            "appreciated."
        )

    attn_backend = subclass_attention_backend_with_overrides(
        name_prefix=prefix,
        attention_backend_cls=underlying_attn_backend,
        overrides={
            "get_builder_cls": lambda: WhisperCausalAttentionWithBlockPoolingBuilder,
            "get_impl_cls": lambda: WhisperCausalAttentionWithBlockPoolingImpl,
            "get_kv_cache_shape": lambda num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            cache_dtype_str: (
                2,
                num_blocks,
                # we stretch each block by `block_pool_size`
                block_size * block_pool_size,
                num_kv_heads // block_pool_size,
                head_size,
            ),  # TODO: generalize to other backends
            "forward_includes_kv_cache_update": True,
        },
    )

    return attn_backend


class WhisperCausalAttentionWithBlockPooling(Attention):
    """Attention layer with block pooling."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        logits_soft_cap: float | None = None,
        per_layer_sliding_window: int | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        block_pool_size: int = 1,
        attn_backend: type[AttentionBackend] | None = None,
        **extra_impl_args,
    ) -> None:
        self.block_pool_size = block_pool_size
        dtype = torch.get_default_dtype()

        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            block_size = cache_config.block_size
        else:
            kv_cache_dtype = "auto"
            block_size = 16

        underlying_attn_backend = get_attn_backend(
            head_size,
            dtype,
            kv_cache_dtype,
            block_size,
            attn_type=attn_type,
        )
        attn_backend = create_whisper_attention_backend_with_block_pooling(
            underlying_attn_backend, block_pool_size
        )

        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            cache_config=cache_config,
            quant_config=quant_config,
            logits_soft_cap=logits_soft_cap,
            per_layer_sliding_window=per_layer_sliding_window,
            prefix=prefix,
            attn_type=attn_type,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            attn_backend=attn_backend,
            **extra_impl_args,
        )

    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        kv_cache_spec = super().get_kv_cache_spec(vllm_config)
        assert isinstance(kv_cache_spec, AttentionSpec)
        kv_cache_spec = replace(
            kv_cache_spec,
            num_kv_heads=self.block_pool_size * kv_cache_spec.num_kv_heads,
        )
        return kv_cache_spec


class WhisperCausalAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        bias: bool = True,
        attn_type: AttentionType = AttentionType.DECODER,
        per_layer_sliding_window: int | None = None,
        block_pool_size: int = 1,
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
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.attn_type = attn_type

        self.scaling = self.head_dim**-0.5

        self._init_qkv(embed_dim, bias, quant_config, prefix=prefix)
        self.out_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=embed_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
        assert block_pool_size > 1, (
            f"Causal attention only supports block_pool_size>1, not {block_pool_size}."
        )
        self.attn = WhisperCausalAttentionWithBlockPooling(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=AttentionType.DECODER,
            per_layer_sliding_window=per_layer_sliding_window,
            block_pool_size=block_pool_size,
        )

        assert per_layer_sliding_window is not None, (
            "rope can only used in combination with a sliding window"
        )
        self._init_rotary_emb(max_position_embeddings)

    def _init_rotary_emb(self, max_position_embeddings: int) -> None:
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            is_neox_style=False,
            rope_parameters={"rope_theta": 1e6},
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
        positions: torch.Tensor | None = None,
    ):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        assert positions is not None
        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)

        output, _ = self.out_proj(attn_output)

        return output


class WhisperCausalEncoderLayer(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        sliding_window = getattr(config, "sliding_window", None)
        block_pool_size = config.block_pool_size
        assert block_pool_size > 1

        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.embed_dim = config.d_model
        self.head_dim = self.embed_dim // config.encoder_attention_heads
        self.self_attn = WhisperCausalAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            head_dim=config.encoder_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            block_pool_size=block_pool_size,
            per_layer_sliding_window=sliding_window,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.self_attn_layer_norm = CausalRMSNorm(self.embed_dim)

        self.mlp = MistralMLP(
            hidden_size=config.d_model,
            intermediate_size=config.encoder_ffn_dim,
            hidden_act="silu",
            quant_config=quant_config,
            bias=True,
            gate_up_proj_bias=False,
            prefix=f"{prefix}.mlp",
        )
        self.final_layer_norm = CausalRMSNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor | None = None,
    ):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states, positions=positions)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class WhisperCausalEncoder(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        embed_dim = config.d_model

        assert WhisperPosEmbedType(config.pos_embed) == WhisperPosEmbedType.ROPE
        assert config.is_causal

        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.conv1 = WhisperCausalConv1d(self.num_mel_bins, embed_dim, kernel_size=3)
        self.conv2 = WhisperCausalConv1d(embed_dim, embed_dim, stride=2, kernel_size=3)

        self.total_stride = self.conv1.stride[0] * self.conv2.stride[0]
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.encoder_layers,
            lambda prefix: WhisperCausalEncoderLayer(
                vllm_config=vllm_config, prefix=f"{prefix}.layers"
            ),
            prefix=f"{prefix}.layers",
        )
        self.layer_norm = CausalRMSNorm(config.d_model)

    def forward_conv(
        self, input_features: torch.Tensor | list[torch.Tensor]
    ) -> torch.Tensor:
        hidden_states = []
        for features in input_features:
            embeds = nn.functional.gelu(self.conv1(features))
            embeds = nn.functional.gelu(self.conv2(embeds))

            embeds = embeds.transpose(-1, -2).to(embeds.dtype)

            hidden_states.append(embeds)

        hidden_states = torch.cat(hidden_states)

        return hidden_states

    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, positions)

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states
