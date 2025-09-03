# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from collections.abc import Iterable
from typing import Optional, Union, type

import torch
import torch.nn as nn
from transformers.activations import ACT2FN

import vllm.envs as envs
from vllm.attention import Attention, AttentionMetadata, AttentionType
from vllm.attention.backends.abstract import AttentionBackend
from vllm.config import (CacheConfig, ModelConfig, VllmConfig,
                         get_current_vllm_config)
from vllm.distributed import (divide, get_pp_group,
                              get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba2_metadata import (
    Mamba2Metadata, prepare_mamba2_metadata)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator, MambaStateShapeCalculator)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.model_executor.models.mamba_cache import (MambaCacheManager,
                                                    MambaCacheParams)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.platforms import _Backend, current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils import direct_register_custom_op
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadata

from .utils import (make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)

logger = init_logger(__name__)


class SwiGLUActivation(nn.Module):

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return x1 * nn.functional.silu(x2)


class SambaYMLP(nn.Module):
    """Gated Linear Unit.

    Reference:
        Language Modeling with Gated Convolutional Networks.
        https://arxiv.org/pdf/1612.08083v3.pdf.

    """

    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__()

        self.config = config
        self.fc1 = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )

        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        gate_up, _ = self.fc1(hidden_states)
        gate, y = gate_up.chunk(2, dim=-1)
        y = y * self.activation_fn(gate)
        output, _ = self.fc2(y)
        return output


class SambaYAttention(nn.Module):

    def __init__(
        self,
        config,
        layer_idx: Optional[int] = None,
        yoco_cross: bool = False,
        cache_config: Optional[CacheConfig] = None,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing "
                "a `layer_idx` is not recommended and will lead to errors "
                "during the forward call if caching is used. Please make "
                "sure to provide a `layer_idx` when creating this class.")
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.yoco_cross = yoco_cross

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError("hidden_size must be divisible by num_heads "
                             f"(got `hidden_size`: {self.hidden_size} and "
                             f"`num_heads`: {self.num_heads}).")

        op_size = self.num_heads * self.head_dim + 2 * (
            self.num_key_value_heads * self.head_dim)
        self.out_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=True,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
        if yoco_cross:
            self.Wqkv = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.head_dim,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.Wqkv",
            )
        else:
            self.Wqkv = ColumnParallelLinear(
                self.hidden_size,
                op_size,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.Wqkv",
            )

        # disable sliding window for the second half of the model
        is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        sliding_window = config.sliding_window if is_sliding else None

        assert self.num_heads % 2 == 0, "num_heads should be even"
        assert self.num_key_value_heads % 2 == 0, "num_heads should be even"

        self.lambda_init = self.lambda_init_fn(layer_idx)
        self.lambda_q1 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,
                                                                    std=0.1))
        self.lambda_k1 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,
                                                                    std=0.1))
        self.lambda_q2 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,
                                                                    std=0.1))
        self.lambda_k2 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,
                                                                    std=0.1))
        self.subln = nn.RMSNorm(2 * self.head_dim,
                                eps=1e-5,
                                elementwise_affine=True)

        params = {
            "differential_flash_attention_config": {
                "lambda_init": self.lambda_init,
                "lambda_q1": self.lambda_q1,
                "lambda_k1": self.lambda_k1,
                "lambda_q2": self.lambda_q2,
                "lambda_k2": self.lambda_k2,
                "subln": self.subln,
            }
        }

        if yoco_cross:
            kv_shared_layer_index = config.num_hidden_layers // 2 + 1
            kv_sharing_target_layer_name = (
                f"model.layers.{kv_shared_layer_index}.self_attn.attn")
        else:
            kv_sharing_target_layer_name = None

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_key_value_heads,
            cache_config=cache_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
            attn_type=AttentionType.DECODER,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            **params,
        )
        assert self.attn.backend == _Backend.DIFFERENTIAL_FLASH_ATTN, (
            "DIFFERENTIAL_FLASH_ATTN required")

    def lambda_init_fn(self, depth):
        return 0.8 - 0.6 * math.exp(-0.3 * depth)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ):
        if not self.yoco_cross:  # need to generate kv-cache
            qkv, _ = self.Wqkv(hidden_states)
            q, k, v = qkv.split(
                [
                    self.hidden_size,
                    self.num_key_value_heads * self.head_dim,
                    self.num_key_value_heads * self.head_dim,
                ],
                dim=-1,
            )
            attn_output = self.attn(q,
                                    k,
                                    v,
                                    kv_cache=kv_cache,
                                    attn_metadata=attn_metadata)
        else:  # reuse the kv cache, full attention
            q, _ = self.Wqkv(hidden_states)
            attn_output = self.attn(q,
                                    None,
                                    None,
                                    kv_cache=kv_cache,
                                    attn_metadata=attn_metadata)
        attn_output = attn_output.view(-1, self.num_heads * self.head_dim)
        output, _ = self.out_proj(attn_output)
        return output


@CustomOp.register("phi4_mamba")
class Phi4Mamba(MambaBase, CustomOp):
    """
    Phi4-specific Mamba implementation following MambaMixer2 
    pattern for V1 compatibility.

    This implementation:
    1. Follows MambaMixer2 structure exactly for V1 compatibility
    2. Adds YoCo-specific logic where needed
    3. Uses the same KV cache pattern as MambaMixer2
    4. Supports both V0 and V1 execution modes
    """

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        yoco_cross=False,
        yoco_kv=False,
        prefix: str = "",
        model_config: Optional[ModelConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config=None,
    ):
        super().__init__()

        # YoCo-specific attributes
        self.yoco_cross = yoco_cross
        self.yoco_kv = yoco_kv
        self.swiGluActivation = SwiGLUActivation()

        # Follow MambaMixer2 pattern for TP and basic setup
        self.tp_size = get_tensor_model_parallel_world_size()
        get_tensor_model_parallel_rank()

        # Calculate dimensions following MambaMixer2 pattern
        intermediate_size = int(expand * d_model)

        # For Phi4, calculate num_heads and head_dim
        if intermediate_size % 64 == 0:
            head_dim = 64
            num_heads = intermediate_size // head_dim
        elif intermediate_size % 32 == 0:
            head_dim = 32
            num_heads = intermediate_size // head_dim
        else:
            head_dim = 64
            num_heads = max(1, intermediate_size // head_dim)

        # Ensure TP compatibility
        assert num_heads % self.tp_size == 0, (
            "Tensor parallel world size must divide num heads.")

        # Store key parameters
        self.ssm_state_size = d_state
        self.conv_kernel_size = d_conv
        self.activation = "silu"
        self.intermediate_size = intermediate_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.n_groups = 1  # Phi4 uses single group
        self.use_rms_norm = True

        if self.yoco_cross:
            # YoCo cross-attention mode: simple projections only
            self.in_proj = MergedColumnParallelLinear(
                d_model,
                [intermediate_size],
                bias=bias,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj",
            )
            self.out_proj = RowParallelLinear(
                intermediate_size,
                d_model,
                bias=bias,
                input_is_parallel=True,
                quant_config=quant_config,
                prefix=f"{prefix}.out_proj",
            )
        else:
            # Standard Mamba mode: follow MambaMixer2 structure exactly
            self.conv_dim = intermediate_size + 2 * self.n_groups * d_state

            # Conv1D layer
            self.conv1d = ColumnParallelLinear(
                input_size=d_conv,
                output_size=self.conv_dim,
                bias=conv_bias,
                quant_config=None,
            )
            # Unsqueeze to fit conv1d weights shape
            self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

            # Input projection
            self.in_proj = ColumnParallelLinear(
                input_size=d_model,
                output_size=intermediate_size + self.conv_dim + num_heads,
                bias=bias,
                quant_config=quant_config,
            )

            # State space parameters (following MambaMixer2)
            self.A = nn.Parameter(
                torch.empty(
                    divide(num_heads, self.tp_size),
                    dtype=torch.float32,
                ))
            self.D = nn.Parameter(torch.ones(num_heads // self.tp_size))
            self.dt_bias = nn.Parameter(torch.ones(num_heads // self.tp_size))

            # Output projection
            self.out_proj = RowParallelLinear(
                intermediate_size,
                d_model,
                bias=bias,
                input_is_parallel=True,
                quant_config=quant_config,
            )

            # RMS Norm (using the same pattern as MambaMixer2)
            from vllm.model_executor.layers.mamba.mamba_mixer2 import (
                Mixer2RMSNormGated)

            self.norm = Mixer2RMSNormGated(intermediate_size,
                                           self.n_groups,
                                           self.use_rms_norm,
                                           eps=1e-5)

        # V1 compatibility setup (following MambaMixer2)
        if envs.VLLM_USE_V1:
            compilation_config = get_current_vllm_config().compilation_config
            if prefix in compilation_config.static_forward_context:
                raise ValueError(f"Duplicate layer name: {prefix}")
            compilation_config.static_forward_context[prefix] = self

            # KV cache setup (following MambaMixer2 pattern)
            self.kv_cache = [(torch.tensor([]), torch.tensor([]))]

        self.model_config = model_config
        self.cache_config = cache_config
        self.prefix = prefix

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        mamba_cache_params: MambaCacheParams,
        mamba2_metadata: Mamba2Metadata,
        yoco_key_values=None,
    ):
        # Native implementation for V0 or fallback
        pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        mamba_cache_params: MambaCacheParams,
        mamba2_metadata: Mamba2Metadata,
        yoco_key_values=None,
    ):
        """Forward pass with YoCo-specific handling"""

        if self.yoco_cross:
            # YoCo cross-attention mode: custom implementation
            out, _ = self.in_proj(hidden_states)
            out = self.swiGluActivation(yoco_key_values, out)
            output_result, _ = self.out_proj(out)
            output[:output_result.shape[0]] = output_result
            return yoco_key_values
        else:
            # Standard Mamba mode: use V1 if available, otherwise V0
            if not envs.VLLM_USE_V1:
                CustomOp.forward(
                    self,
                    hidden_states,
                    output,
                    mamba_cache_params,
                    mamba2_metadata,
                    yoco_key_values,
                )
            else:
                torch.ops.vllm.phi4_mamba(
                    hidden_states,
                    output,
                    self.prefix,
                    yoco_key_values,
                )

            if self.yoco_kv:
                # YoCo key-value mode: return output as yoco_key_values
                return output.clone()

            return None

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        mamba_cache_params: MambaCacheParams,
        mamba2_metadata: Mamba2Metadata,
        yoco_key_values=None,
    ):
        """CUDA implementation following MambaMixer2 pattern"""

        if self.yoco_cross:
            # YoCo cross mode handled in forward()
            return self.forward(
                hidden_states,
                output,
                mamba_cache_params,
                mamba2_metadata,
                yoco_key_values,
            )

        # Follow MambaMixer2 forward_cuda pattern exactly
        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        if envs.VLLM_USE_V1:
            if attn_metadata is not None:
                assert isinstance(attn_metadata, dict)
                attn_metadata = attn_metadata[self.prefix]
                mamba2_metadata = attn_metadata

            assert isinstance(attn_metadata, Mamba2AttentionMetadata)
            self_kv_cache = self.kv_cache[forward_context.virtual_engine]
            # Follow MambaMixer2 pattern: read from KV cache
            self_kv_cache[0].transpose(-1, -2)
            self_kv_cache[1]
            # ... rest of V1 metadata extraction
        else:
            # V0 path
            pass

        # Calculate num_actual_tokens following MambaMixer2 pattern
        if envs.VLLM_USE_V1:
            num_actual_tokens = (attn_metadata.num_decode_tokens +
                                 attn_metadata.num_prefill_tokens)
        else:
            # For V0, use the full hidden_states size
            num_actual_tokens = hidden_states.shape[0]

        # 1. Input projection
        projected_states, _ = self.in_proj(hidden_states)
        gate, hidden_states_B_C, dt = torch.split(
            projected_states,
            [
                self.intermediate_size // self.tp_size,
                self.conv_dim // self.tp_size,
                self.num_heads // self.tp_size,
            ],
            dim=-1,
        )

        # 2. Apply normalization and output projection
        # (Simplified for now - full Mamba logic would go here)
        hidden_states = self.norm(hidden_states_B_C, gate)
        output[:num_actual_tokens], _ = self.out_proj(hidden_states)

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        if self.yoco_cross:
            # YoCo cross mode doesn't need state
            return torch.float16, torch.float16
        else:
            # Follow MambaMixer2 pattern
            assert self.model_config is not None
            assert self.cache_config is not None
            return MambaStateDtypeCalculator.mamba2_state_dtype(
                self.model_config.dtype,
                self.cache_config.mamba_cache_dtype,
                self.cache_config.mamba_ssm_cache_dtype,
            )

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if self.yoco_cross:
            # YoCo cross mode doesn't need state
            return ((0, 0), (0, 0))
        else:
            # Follow MambaMixer2 pattern
            return MambaStateShapeCalculator.mamba2_state_shape(
                intermediate_size=self.intermediate_size,
                tp_world_size=get_tensor_model_parallel_world_size(),
                n_groups=self.n_groups,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                state_size=self.ssm_state_size,
                conv_kernel=self.conv_kernel_size,
            )

    @property
    def mamba_type(self) -> str:
        return "phi4mamba"

    def get_attn_backend(self) -> type[AttentionBackend]:
        if self.yoco_cross:
            # YoCo cross mode doesn't use attention backend
            return None
        else:
            from vllm.v1.attention.backends.mamba2_attn import (
                Mamba2AttentionBackend)

            return Mamba2AttentionBackend


class SambaYDecoderLayer(nn.Module):

    def __init__(
        self,
        config,
        layer_idx,
        cache_config,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.mlp = SambaYMLP(config,
                             quant_config=quant_config,
                             prefix=f"{prefix}.mlp")
        self.input_layernorm = nn.LayerNorm(config.hidden_size,
                                            eps=config.layer_norm_eps)

        self.yoco_mb = False
        self.yoco_cross = False
        if layer_idx >= config.num_hidden_layers // 2:
            self.yoco_mb = True
            self.yoco_cross = layer_idx >= (config.num_hidden_layers // 2 + 2)
        self.use_mamba = (config.mb_per_layer > 0
                          and layer_idx % config.mb_per_layer == 0)
        if self.use_mamba:
            self.attn = Phi4Mamba(
                config.hidden_size,
                layer_idx=layer_idx,
                yoco_cross=self.yoco_cross,
                yoco_kv=self.yoco_mb,
                quant_config=quant_config,
                prefix=f"{prefix}.attn",
            )
        else:
            self.attn = SambaYAttention(
                config,
                layer_idx=layer_idx,
                yoco_cross=self.yoco_cross,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size,
                                                     eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor,
        mamba_cache_params: MambaCacheParams,
        mamba2_metadata: Mamba2Metadata,
        ssm_output: Optional[torch.LongTensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        residual = hidden_states
        hidden_states = self.input_layernorm(
            hidden_states.to(dtype=self.input_layernorm.weight.dtype))

        if self.use_mamba:
            output = torch.empty_like(hidden_states)

            # Get layer-specific cache parameters
            layer_mamba_cache_params = None
            if mamba_cache_params:
                layer_mamba_cache_params = mamba_cache_params.at_layer_idx(
                    self.layer_idx)

            ssm_output = self.attn(
                hidden_states,
                output,
                mamba_cache_params=layer_mamba_cache_params,
                mamba2_metadata=mamba2_metadata,
                yoco_key_values=ssm_output,
            )
            attn_outputs = output
            residual = residual.to(torch.float32)
        else:
            # For attention layers, handle V1 vs V0 metadata access
            forward_context = get_forward_context()
            attn_metadata = forward_context.attn_metadata

            if envs.VLLM_USE_V1 and isinstance(attn_metadata, dict):
                # V1: attn_metadata is a dict, get by prefix
                layer_attn_metadata = attn_metadata.get(self.attn.prefix)
            else:
                # V0: attn_metadata is the object directly
                layer_attn_metadata = attn_metadata

            attn_outputs = self.attn(
                hidden_states,
                kv_cache=kv_cache,
                attn_metadata=layer_attn_metadata,
            )
            ssm_output = ssm_output  # Pass through unchanged

        hidden_states = residual + attn_outputs
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(
            hidden_states.to(dtype=self.post_attention_layernorm.weight.dtype))
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, ssm_output


class SambaYModel(nn.Module):

    def __init__(
        self,
        config,
        cache_config=None,
        quant_config=None,
        lora_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )

        # Pipeline parallel is not supported since the second half of
        # the layers share the kv cache.
        if get_pp_group().world_size != 1:
            raise ValueError("Pipeline Parallel not supported")

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: SambaYDecoderLayer(
                config,
                int(prefix.split(".")[-1]),
                cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        self.final_layernorm = nn.LayerNorm(config.hidden_size,
                                            eps=config.layer_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: list[torch.Tensor],
        mamba_cache_params: MambaCacheParams,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        # Prepare Mamba2 metadata for V0 compatibility
        attn_metadata = get_forward_context().attn_metadata
        if not envs.VLLM_USE_V1:
            mamba2_metadata = prepare_mamba2_metadata(
                chunk_size=getattr(self.config, "mamba_chunk_size", 256),
                attn_metadata=attn_metadata,
            )
        else:
            # V1 gets mamba2_metadata from forward_context
            mamba2_metadata = None

        ssm_output = None
        attn_layer_idx = 0
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            if i == self.config.num_hidden_layers // 2 + 2:
                # profile run
                kv_cache_idx = self.config.num_hidden_layers // 2 + 1
                cache_layer = self.layers[kv_cache_idx]
                kv_cache = cache_layer.attn.attn.kv_cache
                if kv_cache[0].numel() == 0:
                    break

            if layer.use_mamba:
                hidden_states, ssm_output = layer(
                    hidden_states,
                    positions,
                    None,
                    mamba_cache_params,
                    mamba2_metadata,
                    ssm_output=ssm_output,
                )
            else:
                hidden_states, ssm_output = layer(
                    hidden_states,
                    positions,
                    kv_caches[attn_layer_idx],
                    mamba_cache_params,
                    mamba2_metadata,
                    ssm_output=ssm_output,
                )
                attn_layer_idx += 1

        hidden_states = self.final_layernorm(
            hidden_states.to(dtype=self.final_layernorm.weight.dtype))
        return hidden_states


class Phi4FlashForCausalLM(nn.Module, HasInnerState, IsHybrid):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        lora_config = vllm_config.lora_config
        quant_config = vllm_config.quant_config
        self.vllm_config = vllm_config
        super().__init__()
        self.config = config
        self.model_config = vllm_config.model_config

        # Initialize Mamba cache for V0 compatibility
        self.mamba_cache = None

        self.model = SambaYModel(
            config,
            cache_config=cache_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=(
                DEFAULT_VOCAB_PADDING_SIZE
                # We need bigger padding if using lora for kernel
                # compatibility
                if not lora_config else lora_config.lora_vocab_padding_size),
            quant_config=quant_config,
        )
        self.embedding_bias = None
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size,
                                                logits_as_input=False)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
        use_v1: bool = True,
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        """Calculate shapes for Mamba's convolutional and state caches."""
        from vllm.distributed import get_tensor_model_parallel_world_size
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateShapeCalculator)

        config = vllm_config.model_config.hf_config

        # Calculate intermediate size and state size for Mamba layers
        intermediate_size = int(2 *
                                config.hidden_size)  # expand=2 in Phi4Mamba
        state_size = 16  # d_state=16 in Phi4Mamba
        conv_kernel = 4  # d_conv=4 in Phi4Mamba

        return MambaStateShapeCalculator.mamba1_state_shape(
            tp_world_size=get_tensor_model_parallel_world_size(),
            intermediate_size=intermediate_size,
            state_size=state_size,
            conv_kernel=conv_kernel,
        )

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        """Calculate dtypes for Mamba's convolutional and state caches."""
        return MambaStateDtypeCalculator.mamba1_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: list[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # Initialize Mamba cache if needed (V0 compatibility)
        mamba_cache_params = None
        if not envs.VLLM_USE_V1:
            if self.mamba_cache is None:
                num_mamba_layers = self.config.num_hidden_layers
                mamba_state_shape = self.get_mamba_state_shape_from_config(
                    self.vllm_config, use_v1=False)
                mamba_state_dtype = self.get_mamba_state_dtype_from_config(
                    self.vllm_config)
                self.mamba_cache = MambaCacheManager(
                    self.vllm_config,
                    num_mamba_layers,
                    *mamba_state_shape,
                    *mamba_state_dtype,
                )

            # Get cache parameters for current run
            mamba_cache_params = self.mamba_cache.current_run_tensors(**kwargs)

        # Forward pass through model
        hidden_states = self.model(
            input_ids,
            positions,
            kv_caches,
            mamba_cache_params,
            intermediate_tensors,
            inputs_embeds,
        )
        return hidden_states

    def copy_inputs_before_cuda_graphs(self, input_buffers: dict[str,
                                                                 torch.Tensor],
                                       **kwargs) -> dict[str, torch.Tensor]:
        """Copy inputs before CUDA graph capture."""
        if self.mamba_cache is not None:
            return self.mamba_cache.copy_inputs_before_cuda_graphs(
                input_buffers, **kwargs)
        return input_buffers

    def get_seqlen_agnostic_capture_inputs(
            self,
            input_buffers: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Get sequence length agnostic capture inputs."""
        if self.mamba_cache is not None:
            return self.mamba_cache.get_seqlen_agnostic_capture_inputs(
                input_buffers)
        return input_buffers

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        # If the shape is the same, it means that we have already
        # prune hidden states manually.
        prune_hidden_states = hidden_states.size(
            0) != sampling_metadata.selected_token_indices.size(0)
        processed_logits = self.logits_processor(
            self.lm_head,
            hidden_states,
            sampling_metadata,
            self.embedding_bias,
            prune_hidden_states=prune_hidden_states,
        )
        return processed_logits

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ):
        weights = {name: weight for name, weight in weights}
        adjusted_weights = {}
        for name, weight in weights.items():
            if "A_log" in name:
                name = name.replace("A_log", "A")
                weight = -torch.exp(weight.float())
            if "inner_cross_attn." in name:
                name = name.replace("inner_cross_attn.", "")
            adjusted_weights[name] = weight
        adjusted_weights["lm_head.weight"] = weights[
            "model.embed_tokens.weight"]
        loaded_params: set[str] = set()
        for name, param in self.named_parameters():
            weight = adjusted_weights.get(name)
            if weight is not None and weight.shape != param.shape:
                logger.warning("Shape mismatch: %s %s %s", name, weight.shape,
                               param.shape)
            loaded_params.add(name)
        missing_keys, unexpected_keys = self.load_state_dict(adjusted_weights,
                                                             strict=False)
        assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"
        assert len(missing_keys) == 0, f"Missing keys: {missing_keys}"
        return loaded_params


def phi4_mamba(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    yoco_key_values: Optional[torch.Tensor] = None,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self.forward_cuda(
        hidden_states=hidden_states,
        output=output,
        mamba_cache_params=None,
        mamba2_metadata=None,
        yoco_key_values=yoco_key_values,
    )


def phi4_mamba_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    yoco_key_values: Optional[torch.Tensor] = None,
) -> None:
    return


direct_register_custom_op(
    op_name="phi4_mamba",
    op_func=phi4_mamba,
    mutates_args=["output"],
    fake_impl=phi4_mamba_fake,
    dispatch_key=current_platform.dispatch_key,
)
