# SPDX-License-Identifier: Apache-2.0
"""Inference-only PLaMo2 model."""
import math
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.attention.layer import Attention
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn, causal_conv1d_update)
from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    selective_scan_fn, selective_state_update)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    composed_weight_loader, default_weight_loader, sharded_weight_loader)
from vllm.model_executor.models.interfaces import (HasInnerState, IsHybrid,
                                                   SupportsV0Only)
from vllm.model_executor.models.mamba_cache import (MambaCacheManager,
                                                    MambaCacheParams)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import IntermediateTensors
from vllm.utils import LayerBlockType


# Only used for type hinting.
class Plamo2Config(PretrainedConfig):  # type: ignore
    model_type: str = "plamo2"

    hidden_size: int
    num_hidden_layers: int
    rms_norm_eps: float
    # Attention
    num_attention_heads: int
    hidden_size_per_head: int
    num_key_value_heads: int
    # Mamba
    mamba_d_state: int
    mamba_d_conv: int
    mamba_num_heads: int
    mamba_step: int
    # MLP
    intermediate_size: int
    # Tokenizer
    vocab_size: int


class Plamo2PreTrainedModel(PreTrainedModel):  # type: ignore

    def _init_weights(self, module: torch.nn.Module) -> None:
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


def get_initial_dt_bias(num_heads: int) -> torch.Tensor:
    dt_min = 0.001
    dt_max = 0.1
    dt = torch.exp(
        torch.rand(num_heads) * (math.log(dt_max) - math.log(dt_min)) +
        math.log(dt_min))
    dt = torch.clamp(dt, 1e-4)
    inv_dt = dt + torch.log(-torch.expm1(-dt))
    return inv_dt


def is_mamba(config: Plamo2Config, i: int) -> bool:
    assert config.mamba_step > 1

    if config.num_hidden_layers <= (config.mamba_step // 2):
        # use attention in last layer
        return i != config.num_hidden_layers - 1
    return (i % config.mamba_step) != (config.mamba_step // 2)


# TODO(Shinichi): Replace this with RMSNorm.
def _rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor,
              eps: float) -> torch.Tensor:
    input_shape = hidden_states.shape
    hidden_states = hidden_states.reshape(input_shape[:-1] + weight.shape)
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    hidden_states = hidden_states.to(input_dtype)
    hidden_states = weight * hidden_states
    return hidden_states.reshape(input_shape)


def _swiglu(h: torch.Tensor) -> torch.Tensor:
    h0, h1 = h.chunk(2, dim=-1)
    return torch.nn.functional.silu(h0) * h1


# Adapted from transformers.models.mamba.modeling_mamba.MambaMixer
class Plamo2MambaMixer(nn.Module):
    # TODO(Shinichi): Rebase on Mamba2 implementation.

    def __init__(self,
                 config: Plamo2Config,
                 cache_config: CacheConfig,
                 quant_config: QuantizationConfig,
                 max_model_len: int,
                 prefix: str = "",
                 **kwargs) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.intermediate_size = (config.mamba_num_heads *
                                  config.hidden_size_per_head)
        self.hidden_size_per_head = config.hidden_size_per_head
        self.num_heads = config.mamba_num_heads
        self.time_step_rank = max(64, self.hidden_size // 16)
        self.use_conv_bias = False
        self.use_bias = False
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.intermediate_size,
            bias=self.use_conv_bias,
        )
        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `set_weight_attrs`
        # doesn't allow to override it
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        self.in_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.intermediate_size] * 2,
            bias=self.use_bias,
            prefix=f"{prefix}.in_proj",
        )
        # selective projection used to make dt, B and C input dependent
        self.bcdt_proj = RowParallelLinear(
            self.intermediate_size,
            self.time_step_rank + self.ssm_state_size * 2,
            bias=False,
            prefix=f"{prefix}.bcdt_proj",
        )
        # time step projection (discretization) -
        # In the forward we need to apply dt_proj without the bias,
        # as the bias is added in the selective scan kernel.
        self.dt_proj = ColumnParallelLinear(
            self.time_step_rank,
            self.num_heads,
            bias=False,
            prefix=f"{prefix}.dt_proj",
        )
        self.dt_bias = torch.nn.Parameter(get_initial_dt_bias(self.num_heads))

        tp_size = get_tensor_model_parallel_world_size()
        self.A = nn.Parameter(
            torch.empty(
                self.intermediate_size // tp_size,
                self.ssm_state_size,
                dtype=torch.float32,
            ))
        self.D = nn.Parameter(torch.ones(self.intermediate_size // tp_size))

        set_weight_attrs(self.D, {"weight_loader": sharded_weight_loader(0)})
        a_weight_loader = composed_weight_loader(
            sharded_weight_loader(0), lambda x: -torch.exp(x.float()))
        set_weight_attrs(self.A, {"weight_loader": a_weight_loader})

        self.out_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=self.use_bias,
            input_is_parallel=True,
            prefix=f"{prefix}.out_proj",
        )
        # The activation function is fixed to SiLU.
        self.activation = "silu"

        self.dt_norm = RMSNorm(self.time_step_rank, eps=config.rms_norm_eps)
        self.B_norm = RMSNorm(self.ssm_state_size, eps=config.rms_norm_eps)
        self.C_norm = RMSNorm(self.ssm_state_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mamba_cache_params: MambaCacheParams,
        **kwargs,
    ) -> torch.Tensor:

        attn_metadata: AttentionMetadata = get_forward_context().attn_metadata

        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(hidden_states)[0]
        # Reshaping the projected states as in modeling_plamo.py.
        length = len(hidden_states)
        projected_states = projected_states.reshape(length, self.num_heads, -1)
        gate, hidden_states = torch.split(
            projected_states,
            [self.hidden_size_per_head, self.hidden_size_per_head],
            dim=-1)
        hidden_states = hidden_states.reshape(length, -1).transpose(0, 1)
        gate = gate.reshape(length, -1).transpose(0, 1)

        # 2. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0),
                                               self.conv1d.weight.size(2))

        if attn_metadata.query_start_loc is not None \
            and attn_metadata.context_lens_tensor is not None:
            # |---------- N-1 iteration --------|
            # |---------------- N iteration ---------------------|
            # |- tokenA -|......................|-- newTokens ---|
            # |---------- context_len ----------|
            # |-------------------- seq_len ---------------------|
            #                                   |-- query_len ---|
            hidden_states = causal_conv1d_fn(
                hidden_states,
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=mamba_cache_params.conv_state,
                has_initial_state=attn_metadata.context_lens_tensor > 0,
                cache_indices=mamba_cache_params.state_indices_tensor,
                query_start_loc=attn_metadata.query_start_loc)
        else:
            hidden_states = causal_conv1d_update(
                hidden_states.transpose(0, 1),
                mamba_cache_params.conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=mamba_cache_params.state_indices_tensor)
            hidden_states = hidden_states.transpose(0, 1)

        # 3. State Space Model sequence transformation
        # 3.a. input varying initialization of time_step, B and C
        ssm_parameters = self.bcdt_proj(hidden_states.transpose(-2, -1))[0]

        # Splitting the ssm_parameters as in modeling_plamo.py.
        B, C, time_step = torch.split(
            ssm_parameters,
            [self.ssm_state_size, self.ssm_state_size, self.time_step_rank],
            dim=-1,
        )
        time_step = self.dt_norm(time_step.contiguous())
        B = self.B_norm(B.contiguous())
        C = self.C_norm(C.contiguous())

        discrete_time_step = self.dt_proj(time_step)[0].transpose(-2, -1)
        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        time_proj_bias = (self.dt_bias.float() if hasattr(
            self.dt_proj, "bias") else None)

        # Broadcasting as in modeling_plamo.py.
        discrete_time_step = discrete_time_step.transpose(
            0, 1)[..., None].expand(-1, -1, self.hidden_size_per_head)
        discrete_time_step = discrete_time_step.reshape(
            -1, self.intermediate_size).transpose(0, 1)
        time_proj_bias = time_proj_bias[...,
                                        None].expand(-1,
                                                     self.hidden_size_per_head)
        time_proj_bias = time_proj_bias.reshape(self.intermediate_size)

        if attn_metadata.query_start_loc is not None \
            and attn_metadata.context_lens_tensor is not None:
            scan_outputs = selective_scan_fn(
                hidden_states,
                mamba_cache_params.ssm_state,
                discrete_time_step,
                self.A,
                B.transpose(-2, -1),
                C.transpose(-2, -1),
                self.D.float(),
                gate,
                time_proj_bias,
                delta_softplus=True,
                cache_indices=mamba_cache_params.state_indices_tensor,
                has_initial_state=attn_metadata.context_lens_tensor > 0,
                query_start_loc=attn_metadata.query_start_loc)
        else:
            scan_outputs = selective_state_update(
                mamba_cache_params.ssm_state,
                hidden_states.transpose(0, 1),
                discrete_time_step.transpose(0, 1),
                self.A,
                B,
                C,
                self.D,
                gate.transpose(0, 1),
                time_proj_bias,
                dt_softplus=True,
                state_batch_indices=mamba_cache_params.state_indices_tensor)
            scan_outputs = scan_outputs.transpose(0, 1)

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_outputs.transpose(-2,
                                                                     -1))[0]
        return contextualized_states


class DenseMLP(nn.Module):

    def __init__(
        self,
        config: Plamo2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(
            self.hidden_size, [self.intermediate_size] * 2,
            bias=False,
            prefix=f"{prefix}.gate_up_proj",
            quant_config=quant_config)
        self.down_proj = RowParallelLinear(self.intermediate_size,
                                           self.hidden_size,
                                           bias=False,
                                           prefix=f"{prefix}.down_proj",
                                           quant_config=quant_config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        h = self.gate_up_proj(hidden_states)[0]
        h = _swiglu(h)
        output, _ = self.down_proj(h)
        return output  # type: ignore


class Plamo2AttentionMixer(nn.Module):

    def __init__(self,
                 config: Plamo2Config,
                 cache_config: CacheConfig,
                 quant_config: QuantizationConfig,
                 max_model_len: int | None = None,
                 prefix: str = "",
                 **kwargs) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.hidden_size_per_head
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim,
                                        config.hidden_size,
                                        bias=False,
                                        quant_config=quant_config)

        self.rope_theta = config.rope_theta if hasattr(config,
                                                       "rope_theta") else 10000
        self.rope_scaling = config.rope_scaling if hasattr(
            config, "rope_scaling") else None

        assert max_model_len is not None, "max_model_len must be provided"
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_model_len,
            base=self.rope_theta,
            rope_scaling=self.rope_scaling,
        )
        self.q_weight = torch.nn.Parameter(
            torch.ones((self.num_heads, config.hidden_size_per_head)))
        self.k_weight = torch.nn.Parameter(
            torch.ones((self.num_kv_heads, config.hidden_size_per_head)))

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = _rms_norm(q, self.q_weight, 1e-6)
        k = _rms_norm(k, self.k_weight, 1e-6)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Plamo2DecoderLayer(nn.Module):

    def __init__(self,
                 vllm_config: VllmConfig,
                 layer_idx: int,
                 max_model_len: int | None = None,
                 prefix: str = "",
                 **kwargs) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        max_model_len = vllm_config.scheduler_config.max_model_len

        self.is_mamba = is_mamba(config, layer_idx)
        if self.is_mamba:
            self.mixer = Plamo2MambaMixer(config=config,
                                          cache_config=cache_config,
                                          quant_config=quant_config,
                                          max_model_len=max_model_len,
                                          prefix=f"{prefix}.mixer")
        else:
            self.mixer = Plamo2AttentionMixer(config=config,
                                              cache_config=cache_config,
                                              quant_config=quant_config,
                                              max_model_len=max_model_len,
                                              prefix=f"{prefix}.mixer")

        self.mlp = DenseMLP(config=config,
                            quant_config=quant_config,
                            prefix=f"{prefix}.mlp")
        self.pre_mixer_norm = RMSNorm(config.hidden_size,
                                      eps=config.rms_norm_eps)
        self.post_mixer_norm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.pre_mlp_norm = RMSNorm(config.hidden_size,
                                    eps=config.rms_norm_eps)
        self.post_mlp_norm = RMSNorm(config.hidden_size,
                                     eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        mamba_cache_params: MambaCacheParams,
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.pre_mixer_norm(hidden_states)
        else:
            hidden_states, residual = self.pre_mixer_norm(
                hidden_states, residual)

        hidden_states = self.mixer(positions=positions,
                                   hidden_states=hidden_states,
                                   residual=residual,
                                   mamba_cache_params=mamba_cache_params)
        hidden_states = self.post_mixer_norm(hidden_states)
        # Fully Connected
        hidden_states, residual = self.pre_mlp_norm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_norm(hidden_states)
        return hidden_states, residual


class Plamo2Decoder(torch.nn.Module):

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        num_hidden_layers = vllm_config.model_config.hf_config.num_hidden_layers

        self.layers = nn.ModuleList([
            Plamo2DecoderLayer(vllm_config=vllm_config,
                               layer_idx=i,
                               prefix=f"{prefix}.layers.{i}")
            for i in range(num_hidden_layers)
        ])

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        mamba_cache_params: MambaCacheParams,
    ) -> torch.Tensor:
        mamba_cache_index = 0
        for layer in self.layers:
            layer_mamba_cache_params = None
            if layer.is_mamba:
                layer_mamba_cache_params = mamba_cache_params.at_layer_idx(
                    mamba_cache_index)
                mamba_cache_index += 1

            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                mamba_cache_params=layer_mamba_cache_params)
        return hidden_states, residual


class Plamo2Model(Plamo2PreTrainedModel):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config.model_config.hf_config)

        config = vllm_config.model_config.hf_config

        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.org_vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            prefix=f"{prefix}.embed_tokens",
        )
        self.layers = Plamo2Decoder(vllm_config, prefix=f"{prefix}.layers")
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mamba_cache_params: MambaCacheParams,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # TODO(Shinichi): Implement pipeline parallelism.
        hidden_states = self.embed_tokens(input_ids)
        residual = None

        hidden_states, residual = self.layers(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
            mamba_cache_params=mamba_cache_params)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Plamo2ForCausalLM(Plamo2PreTrainedModel, HasInnerState, IsHybrid,
                        SupportsV0Only):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config = vllm_config.model_config.hf_config
        scheduler_config = vllm_config.scheduler_config
        assert not vllm_config.cache_config.enable_prefix_caching, \
            "PLaMo2 currently does not support prefix caching"

        super().__init__(config)
        self.config = config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.scheduler_config = scheduler_config

        # ModelConfig.get_head_size assumes head_dim is set or calculated as
        # hidden_size // num_attention_heads. However, this is not always
        # the case for PLaMo2, as indicated by the FIXME comment.
        self.config.head_dim = self.config.hidden_size_per_head

        self.model = Plamo2Model(vllm_config=vllm_config,
                                 prefix=maybe_prefix(prefix, "model"))
        self.vocab_size = self.config.vocab_size
        self.unpadded_vocab_size = self.config.vocab_size
        num_embeddings = ((self.vocab_size + 15) // 16) * 16
        self.lm_head = ParallelLMHead(
            num_embeddings,
            self.config.hidden_size,
            org_num_embeddings=self.config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE,
            prefix=f"{prefix}.lm_head",
        )
        if self.config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

        # Used to track and store by the Mamba cache between steps.
        self.mamba_cache: Optional[MambaCacheManager] = None

        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                self.config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                **kwargs):
        if self.mamba_cache is None:
            num_mamba_layers = self.model_config.get_num_layers_by_block_type(
                self.vllm_config.parallel_config, LayerBlockType.mamba)

            self.mamba_cache = MambaCacheManager(
                self.vllm_config, self.lm_head.weight.dtype, num_mamba_layers,
                *self._get_mamba_cache_shape())

        mamba_cache_params = self.mamba_cache.current_run_tensors(**kwargs)

        hidden_states = self.model(input_ids, positions, mamba_cache_params,
                                   intermediate_tensors, inputs_embeds)
        return hidden_states

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.mamba_cache.copy_inputs_before_cuda_graphs(
            input_buffers, **kwargs)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self.mamba_cache.get_seqlen_agnostic_capture_inputs(batch_size)

    def _get_mamba_cache_shape(
            self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        world_size = get_tensor_model_parallel_world_size()
        hidden_size = (self.config.mamba_num_heads *
                       self.config.hidden_size_per_head)
        conv_state_shape = (
            hidden_size // world_size,
            self.config.mamba_d_conv - 1,
        )
        temporal_state_shape = (
            hidden_size // world_size,
            self.config.mamba_d_state,
        )
        return conv_state_shape, temporal_state_shape

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:

            # Both tie_word_embeddings=True and lm_head.weight in the safetensor
            # at the same time causes dict key access error.
            if name == "lm_head.weight" and self.config.tie_word_embeddings:
                assert "lm_head.weight" not in params_dict
                continue

            # Update the weight names to be compatible with the vllm version
            # of the model.
            # Do not change the order of the replacements.
            replacements = {
                # Rename incompatible weight names.
                ".A_log": ".A",
                ".B_norm_weight": ".B_norm.weight",
                ".C_norm_weight": ".C_norm.weight",
                ".dt_norm_weight": ".dt_norm.weight",
            }
            # Apply replacements based on the defined mappings
            for old, new in replacements.items():
                if old in name:
                    name = name.replace(old, new)

            # Broadcast the loaded weight to match the model's parameter shape.
            if ".A" in name:
                loaded_weight = loaded_weight[:, None, None].expand(
                    -1, self.config.hidden_size_per_head,
                    self.config.mamba_d_state)
                loaded_weight = loaded_weight.reshape(
                    -1, self.config.mamba_d_state)
            elif ".D" in name:
                loaded_weight = loaded_weight[:, None].expand(
                    -1, self.config.hidden_size_per_head)
                loaded_weight = loaded_weight.reshape(-1)
            # Offset parameter with vllm's RMSNorm haven't been supported yet.
            if ".pre_mixer_norm" in name:
                loaded_weight += 1.0
            elif ".post_mixer_norm" in name:
                loaded_weight += 1.0 / 5
            elif ".pre_mlp_norm" in name:
                loaded_weight += 1.0
            elif ".post_mlp_norm" in name:
                loaded_weight += 1.0 / (5**1.5)
            elif "model.norm.weight" in name:
                loaded_weight += 1.0

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
