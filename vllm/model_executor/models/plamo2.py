# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only PLaMo2 model."""
from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba2_metadata import (
    Mamba2Metadata, prepare_mamba2_metadata)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn, causal_conv1d_update)
from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    selective_state_update)
from vllm.model_executor.layers.mamba.ops.ssd_combined import (
    mamba_chunk_scan_combined)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    composed_weight_loader, default_weight_loader, sharded_weight_loader)
from vllm.model_executor.models.interfaces import (HasInnerState, IsHybrid,
                                                   SupportsPP, SupportsV0Only)
from vllm.model_executor.models.mamba_cache import (MambaCacheManager,
                                                    MambaCacheParams)
from vllm.model_executor.models.utils import (
    is_pp_missing_parameter, make_empty_intermediate_tensors_factory,
    make_layers, maybe_prefix)
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


def is_mamba(config: Plamo2Config, i: int) -> bool:
    assert config.mamba_step > 1

    if config.num_hidden_layers <= (config.mamba_step // 2):
        # use attention in last layer
        return i != config.num_hidden_layers - 1
    return (i % config.mamba_step) != (config.mamba_step // 2)


# Adapted from:
# vllm.model_executor.layers.mamba.mamba_mixer2.MambaMixer2
# transformers.models.mamba.modeling_mamba.MambaMixer
class Plamo2MambaMixer(nn.Module):

    def __init__(self,
                 vllm_config: VllmConfig,
                 *,
                 prefix: str = "",
                 **kwargs) -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.hidden_size = self.config.hidden_size
        self.ssm_state_size = self.config.mamba_d_state
        self.conv_kernel_size = self.config.mamba_d_conv
        self.intermediate_size = (self.config.mamba_num_heads *
                                  self.config.hidden_size_per_head)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.intermediate_size_per_tp_worker = \
            self.intermediate_size // self.tp_size
        self.head_dim = self.config.hidden_size_per_head
        self.num_heads = self.config.mamba_num_heads
        self.time_step_rank = max(64, self.hidden_size // 16)
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.intermediate_size,
            bias=False,
            prefix=f"{prefix}.conv1d",
            return_bias=False,
        )
        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `set_weight_attrs`
        # doesn't allow to override it
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        self.in_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.intermediate_size] * 2,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.in_proj",
            return_bias=False,
        )
        # selective projection used to make dt, B and C input dependent
        self.bcdt_proj = RowParallelLinear(
            self.intermediate_size,
            self.time_step_rank + self.ssm_state_size * 2,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.bcdt_proj",
            return_bias=False,
        )
        # time step projection (discretization) -
        # In the forward we need to apply dt_proj without the bias,
        # as the bias is added in the selective scan kernel.
        self.dt_proj = ColumnParallelLinear(
            self.time_step_rank,
            self.num_heads,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.dt_proj",
            return_bias=False,
        )

        self.A = nn.Parameter(
            torch.empty(
                divide(self.num_heads, self.tp_size),
                dtype=torch.float32,
            ))
        self.D = nn.Parameter(torch.ones(divide(self.num_heads, self.tp_size)))
        self.dt_bias = nn.Parameter(
            torch.ones(divide(self.num_heads, self.tp_size)))

        set_weight_attrs(self.D, {"weight_loader": sharded_weight_loader(0)})
        a_weight_loader = composed_weight_loader(
            sharded_weight_loader(0), lambda x: -torch.exp(x.float()))
        set_weight_attrs(self.A, {"weight_loader": a_weight_loader})
        set_weight_attrs(self.dt_bias,
                         {"weight_loader": sharded_weight_loader(0)})

        self.out_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=self.quant_config,
            prefix=f"{prefix}.out_proj",
            return_bias=False,
        )
        # The activation function is fixed to SiLU.
        self.activation = "silu"

        self.dt_norm = RMSNorm(self.time_step_rank,
                               eps=self.config.rms_norm_eps)
        self.B_norm = RMSNorm(self.ssm_state_size,
                              eps=self.config.rms_norm_eps)
        self.C_norm = RMSNorm(self.ssm_state_size,
                              eps=self.config.rms_norm_eps)

    def _project_ssm_parameters(self, hidden_states):
        ssm_parameters = self.bcdt_proj(hidden_states)
        B, C, time_step = torch.split(
            ssm_parameters,
            [self.ssm_state_size, self.ssm_state_size, self.time_step_rank],
            dim=-1,
        )

        # vllm._custom_ops.rms_norm requires contiguous input tensors.
        time_step = self.dt_norm(time_step.contiguous())
        B = self.B_norm(B.contiguous())
        C = self.C_norm(C.contiguous())
        dt = self.dt_proj(time_step)
        return B, C, dt

    def forward(
        self,
        hidden_states: torch.Tensor,
        mamba_cache_params: MambaCacheParams,
        mamba2_metadata: Mamba2Metadata,
        **kwargs,
    ) -> torch.Tensor:

        # mamba2_metadata contains metadata necessary for the mamba2 triton
        # kernels to operate in continuous batching and in chunked prefill
        # modes; they are computed at top-level model forward since they
        # stay the same and reused for all mamba layers in the same iteration
        attn_metadata: AttentionMetadata = get_forward_context().attn_metadata

        num_prefills = attn_metadata.num_prefills  # request count
        num_decodes = attn_metadata.num_decode_tokens  # token count (=request)
        num_prefill_tokens = attn_metadata.num_prefill_tokens  # token count
        has_prefill = num_prefills > 0
        has_decode = num_decodes > 0

        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(hidden_states)
        gate, hidden_states = projected_states.chunk(2, dim=-1)

        # 2. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0),
                                               self.conv1d.weight.size(2))

        # Separate prefill and decode by splitting varlen input
        # Split along token dimension
        hidden_states_p, hidden_states_d = torch.split(
            hidden_states,
            [num_prefill_tokens, num_decodes],
            dim=0,
        )
        gate_p, gate_d = torch.split(gate, [num_prefill_tokens, num_decodes],
                                     dim=0)
        # Split along batch dimension
        state_indices_tensor_p, state_indices_tensor_d = torch.split(
            mamba_cache_params.state_indices_tensor,
            [num_prefills, num_decodes],
            dim=0,
        )
        query_start_loc_p = (attn_metadata.query_start_loc[:num_prefills + 1]
                             if has_prefill else None)

        # Preallocate output tensor to avoid memcpy cost for merging prefill
        # and decode outputs
        preallocated_ssm_out = torch.empty(
            [
                num_prefill_tokens + num_decodes,
                (self.num_heads // self.tp_size) * self.head_dim
            ],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        preallocated_ssm_out_p, preallocated_ssm_out_d = torch.split(
            preallocated_ssm_out,
            [num_prefill_tokens, num_decodes],
            dim=0,
        )

        # Process prefill requests
        if has_prefill:
            # 2. Convolution sequence transformation
            # - "cache_indices" updates the conv_state cache in positions
            # pointed to by "mamba_cache_params.state_indices_tensor"
            hidden_states_p = causal_conv1d_fn(
                hidden_states_p.transpose(0, 1),
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=mamba_cache_params.conv_state,
                has_initial_state=mamba2_metadata.has_initial_states,
                cache_indices=state_indices_tensor_p,
                query_start_loc=query_start_loc_p)
            hidden_states_p = hidden_states_p.transpose(0, 1)
            hidden_states_p = hidden_states_p[:num_prefill_tokens]
            # In some instances, the following `bcdt_proj` op
            # requires contiguous inputs
            # (e.g. if the Marlin kernel is used).
            hidden_states_p = hidden_states_p.contiguous()

            B, C, dt = self._project_ssm_parameters(hidden_states_p)

            # 3. State Space Model sequence transformation
            initial_states = None
            if (mamba2_metadata.has_initial_states is not None
                    and mamba2_metadata.prep_initial_states):
                # making a copy of the states
                initial_states = torch.where(
                    mamba2_metadata.has_initial_states[:, None, None, None],
                    mamba_cache_params.ssm_state[state_indices_tensor_p], 0)
            varlen_state = mamba_chunk_scan_combined(
                hidden_states_p.view(1, num_prefill_tokens,
                                     self.num_heads // self.tp_size,
                                     self.head_dim),
                dt.unsqueeze(0),
                self.A,
                B.view(1, num_prefill_tokens, 1, -1),
                C.view(1, num_prefill_tokens, 1, -1),
                chunk_size=mamba2_metadata.chunk_size,
                D=self.D,
                z=gate_p.view(1, num_prefill_tokens,
                              self.num_heads // self.tp_size, self.head_dim),
                dt_bias=self.dt_bias,
                seq_idx=mamba2_metadata.seq_idx,
                chunk_indices=mamba2_metadata.chunk_indices,
                chunk_offsets=mamba2_metadata.chunk_offsets,
                cu_seqlens=attn_metadata.query_start_loc[:num_prefills + 1],
                initial_states=initial_states,
                return_varlen_states=True,
                return_final_states=False,
                dt_softplus=True,
                dt_limit=(0.0, float("inf")),
                out=preallocated_ssm_out_p.view(1, num_prefill_tokens, -1,
                                                self.head_dim),
            )

            # update ssm states
            # - varlen state is a (batch, nheads, headdim, dstate) tensor
            mamba_cache_params.ssm_state[state_indices_tensor_p] = varlen_state

        # Process decode requests
        if has_decode:
            # 2. Convolution sequence transformation
            hidden_states_d = causal_conv1d_update(
                hidden_states_d,
                mamba_cache_params.conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=state_indices_tensor_d)

            B, C, dt = self._project_ssm_parameters(hidden_states_d)

            # 3. State Space Model sequence transformation
            A = self.A[:, None, ...][:, :,
                                     None].expand(-1, self.head_dim,
                                                  self.config.mamba_d_state)
            dt = dt[:, :, None].expand(-1, -1, self.head_dim)
            dt_bias = self.dt_bias[:, None, ...].expand(-1, self.head_dim)
            D = self.D[:, None, ...].expand(-1, self.head_dim)
            B = B.unsqueeze(1)
            C = C.unsqueeze(1)
            hidden_states_d = hidden_states_d.view(
                -1, self.num_heads // self.tp_size, self.head_dim)

            # - the hidden is reshaped into (bs, num_heads, head_dim)
            # - mamba_cache_params.ssm_state's slots will be selected
            #   using state_indices_tensor_d
            selective_state_update(
                mamba_cache_params.ssm_state,
                hidden_states_d,
                dt,
                A,
                B,
                C,
                D,
                z=gate_d.reshape(num_decodes, -1, self.head_dim),
                dt_bias=dt_bias,
                dt_softplus=True,
                state_batch_indices=state_indices_tensor_d,
                out=preallocated_ssm_out_d.view(num_decodes, -1,
                                                self.head_dim),
            )
            assert self.num_heads % self.tp_size == 0

        # 4. Final linear projection
        out = self.out_proj(preallocated_ssm_out)
        return out


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
            self.hidden_size,
            [self.intermediate_size] * 2,
            bias=False,
            prefix=f"{prefix}.gate_up_proj",
            quant_config=quant_config,
            return_bias=False,
        )
        self.act = SiluAndMul()
        self.down_proj = RowParallelLinear(self.intermediate_size,
                                           self.hidden_size,
                                           bias=False,
                                           prefix=f"{prefix}.down_proj",
                                           quant_config=quant_config,
                                           return_bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        h = self.gate_up_proj(hidden_states)
        h = self.act(h)
        return self.down_proj(h)


@support_torch_compile
class Plamo2AttentionMixer(nn.Module):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 **kwargs) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
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
        max_position = config.max_position_embeddings
        if hasattr(vllm_config.model_config, "max_model_len") and isinstance(
                vllm_config.model_config.max_model_len, int):
            max_position = min(max_position,
                               vllm_config.model_config.max_model_len)

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=self.rope_theta,
            rope_scaling=self.rope_scaling,
        )
        self.q_norm = RMSNorm(config.hidden_size_per_head,
                              eps=config.rms_norm_eps)
        self.q_norm.weight = torch.nn.Parameter(
            torch.ones((self.num_heads, config.hidden_size_per_head)))
        set_weight_attrs(self.q_norm.weight,
                         {"weight_loader": sharded_weight_loader(0)})
        self.k_norm = RMSNorm(config.hidden_size_per_head,
                              eps=config.rms_norm_eps)
        self.k_norm.weight = torch.nn.Parameter(
            torch.ones((self.num_kv_heads, config.hidden_size_per_head)))
        # Tensor-parallelism shards the K norm weights to the tp ranks
        # in a head-wise manner. This approach does not work if there is only
        # a single KV head, as is the case for PLaMo 2-1B.
        if self.total_num_kv_heads != 1:
            set_weight_attrs(self.k_norm.weight,
                             {"weight_loader": sharded_weight_loader(0)})

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
        **kwargs,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q_shape = q.shape
        q = q.reshape(q_shape[:-1] + self.q_norm.weight.shape)
        q = self.q_norm.forward_native(q).reshape(q_shape)
        k_shape = k.shape
        k = k.reshape(k_shape[:-1] + self.k_norm.weight.shape)
        k = self.k_norm.forward_native(k).reshape(k_shape)

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Plamo2DecoderLayer(nn.Module):

    def __init__(self,
                 vllm_config: VllmConfig,
                 layer_idx: int,
                 prefix: str = "",
                 **kwargs) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.is_mamba = is_mamba(config, layer_idx)
        if self.is_mamba:
            self.mixer = Plamo2MambaMixer(vllm_config=vllm_config,
                                          prefix=f"{prefix}.mixer")
        else:
            self.mixer = Plamo2AttentionMixer(vllm_config=vllm_config,
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
        mamba2_metadata: Mamba2Metadata,
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.pre_mixer_norm(hidden_states)
        else:
            hidden_states, residual = self.pre_mixer_norm(
                hidden_states, residual)

        hidden_states = self.mixer(
            positions=positions,
            hidden_states=hidden_states,
            mamba_cache_params=mamba_cache_params,
            mamba2_metadata=mamba2_metadata,
        )
        hidden_states = self.post_mixer_norm(hidden_states)
        # Fully Connected
        hidden_states, residual = self.pre_mlp_norm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_norm(hidden_states)
        return hidden_states, residual


class Plamo2Decoder(torch.nn.Module):

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        extra_kwargs = {"is_lora_enabled": bool(vllm_config.lora_config)}

        def get_layer(prefix: str):
            layer_idx = int(prefix.rsplit(".", 1)[1])
            return Plamo2DecoderLayer(vllm_config=vllm_config,
                                      layer_idx=layer_idx,
                                      prefix=prefix,
                                      **extra_kwargs)

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        mamba_cache_params: MambaCacheParams,
        mamba2_metadata: Mamba2Metadata,
    ) -> torch.Tensor:
        mamba_cache_index = 0
        for layer in self.layers[self.start_layer:self.end_layer]:
            layer_mamba_cache_params = None
            if layer.is_mamba:
                layer_mamba_cache_params = mamba_cache_params.at_layer_idx(
                    mamba_cache_index)
                mamba_cache_index += 1

            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                mamba_cache_params=layer_mamba_cache_params,
                mamba2_metadata=mamba2_metadata,
            )
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
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))
        self.layers = Plamo2Decoder(vllm_config, prefix=f"{prefix}.layers")
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mamba_cache_params: MambaCacheParams,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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

        attn_metadata: AttentionMetadata = get_forward_context().attn_metadata
        mamba2_metadata = prepare_mamba2_metadata(
            chunk_size=self.config.mamba_chunk_size,
            attn_metadata=attn_metadata,
        )

        hidden_states, residual = self.layers(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
            mamba_cache_params=mamba_cache_params,
            mamba2_metadata=mamba2_metadata,
        )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Plamo2ForCausalLM(Plamo2PreTrainedModel, HasInnerState, SupportsPP,
                        IsHybrid, SupportsV0Only):
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
        self.sampler = get_sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

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
            self) -> tuple[tuple[int, int], tuple[int, int, int]]:
        world_size = get_tensor_model_parallel_world_size()
        hidden_size = (self.config.mamba_num_heads *
                       self.config.hidden_size_per_head)
        conv_state_shape = (
            hidden_size // world_size,
            self.config.mamba_d_conv - 1,
        )
        temporal_state_shape = (
            divide(self.config.mamba_num_heads, world_size),
            self.config.hidden_size_per_head,
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

    def sample(
        self,
        logits: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
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
                ".q_weight": ".q_norm.weight",
                ".k_weight": ".k_norm.weight",
            }
            # Apply replacements based on the defined mappings
            for old, new in replacements.items():
                if old in name:
                    name = name.replace(old, new)

            # Reshape the in_proj weights to match the shape expected
            # by MergedColumnParallelLinear.
            # This works both for unquantized weights and
            # for quantized weights.
            # In the quantized case, the weights are already transposed.
            # Also, in addition to the quantized weights,
            # the zero points and scales have to be reshaped as well.
            # Packing should not be affected by this.
            if ".mixer.in_proj.weight" in name \
                or "mixer.in_proj.qweight" in name \
                or "mixer.in_proj.scales" in name \
                or "mixer.in_proj.qzeros" in name:
                if "mixer.in_proj.weight" in name:
                    loaded_weight = loaded_weight.transpose(0, 1)
                # for weight:
                # loaded_weight.shape[0] == self.config.hidden_size
                # for qweight:
                # loaded_weight.shape[0] == self.config.hidden_size // param.pack_factor  # noqa
                # for scales and qzeros:
                # loaded_weight.shape[0] == self.config.hidden_size // self.vllm_config.quant_config.group_size  # noqa
                loaded_weight = loaded_weight.reshape(
                    loaded_weight.shape[0], self.config.mamba_num_heads, -1)
                gate_weight, hidden_states_weight = loaded_weight.chunk(2,
                                                                        dim=-1)
                gate_weight = gate_weight.reshape(loaded_weight.shape[0], -1)
                hidden_states_weight = hidden_states_weight.reshape(
                    loaded_weight.shape[0], -1)
                loaded_weight = torch.cat([gate_weight, hidden_states_weight],
                                          dim=-1)
                if "mixer.in_proj.weight" in name:
                    loaded_weight = loaded_weight.transpose(0, 1)

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

            # Skip layers on other devices.
            if is_pp_missing_parameter(name, self):
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
