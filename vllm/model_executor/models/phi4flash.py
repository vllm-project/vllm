# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from collections.abc import Iterable
from typing import Optional, Union

import torch
import torch.nn as nn
from transformers.activations import ACT2FN

import vllm.envs as envs
from vllm.attention import Attention, AttentionMetadata, AttentionType
from vllm.attention.selector import _Backend
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn, causal_conv1d_update)
from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    selective_scan_fn, selective_state_update)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.models.interfaces import (HasInnerState, IsHybrid,
                                                   SupportsV0Only)
from vllm.model_executor.models.mamba_cache import (MambaCacheManager,
                                                    MambaCacheParams)
from vllm.sequence import IntermediateTensors

from .utils import make_layers, maybe_prefix

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

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.fc1 = nn.Linear(config.hidden_size,
                             2 * config.intermediate_size,
                             bias=False)
        self.fc2 = nn.Linear(config.intermediate_size,
                             config.hidden_size,
                             bias=False)

        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        y = self.fc1(hidden_states)
        gate, y = y.chunk(2, dim=-1)
        y = y * self.activation_fn(gate)
        return self.fc2(y)


def get_virtual_engine():
    forward_context: ForwardContext = get_forward_context()
    return forward_context.virtual_engine


class SambaYAttention(nn.Module):

    def __init__(self,
                 config,
                 layer_idx: Optional[int] = None,
                 yoco_cross: bool = False,
                 cache_config: Optional[CacheConfig] = None,
                 prefix: str = ""):
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
        self.out_proj = nn.Linear(self.num_heads * self.head_dim,
                                  self.hidden_size,
                                  bias=True)
        if yoco_cross:
            self.Wqkv = nn.Linear(self.hidden_size,
                                  self.num_heads * self.head_dim,
                                  bias=True)
        else:
            self.Wqkv = nn.Linear(self.hidden_size, op_size, bias=True)

        # disable sliding window for the second half of the model
        is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        sliding_window = config.sliding_window if is_sliding else None

        assert self.num_heads % 2 == 0, 'num_heads should be even'
        assert self.num_key_value_heads % 2 == 0, 'num_heads should be even'

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
            'differential_flash_attention_config': {
                'lambda_init': self.lambda_init,
                'lambda_q1': self.lambda_q1,
                'lambda_k1': self.lambda_k1,
                'lambda_q2': self.lambda_q2,
                'lambda_k2': self.lambda_k2,
                "subln": self.subln,
            }
        }

        if yoco_cross:
            kv_shared_layer_index = config.num_hidden_layers // 2 + 1
            kv_sharing_target_layer_name = \
                f"model.layers.{kv_shared_layer_index}.self_attn.attn"
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
            **params)
        assert self.attn.backend == _Backend.DIFFERENTIAL_FLASH_ATTN,\
              "DIFFERENTIAL_FLASH_ATTN required"

    def lambda_init_fn(self, depth):
        return 0.8 - 0.6 * math.exp(-0.3 * depth)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):

        if not self.yoco_cross:  # need to generate kv-cache
            qkv = self.Wqkv(hidden_states)
            q, k, v = qkv.split([
                self.hidden_size, self.num_key_value_heads * self.head_dim,
                self.num_key_value_heads * self.head_dim
            ],
                                dim=-1)
            attn_output = self.attn(q, k, v)
        else:  # reuse the kv cache, full attention
            q = self.Wqkv(hidden_states)
            attn_output = self.attn(q, None, None)
        attn_output = attn_output.view(-1, self.num_heads * self.head_dim)
        return self.out_proj(attn_output)


class Phi4Mamba(nn.Module):

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",  # difference
        dt_scale=1.0,  # difference
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        yoco_cross=False,
        yoco_kv=False,
    ):
        factory_kwargs = {"params_dtype": dtype}  # difference
        super().__init__()
        self.yoco_cross = yoco_cross
        self.yoco_kv = yoco_kv
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model /
                                 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.swiGluActivation = SwiGLUActivation()
        if self.yoco_cross:
            self.in_proj = MergedColumnParallelLinear(self.d_model,
                                                      [self.d_inner],
                                                      bias=bias,
                                                      **factory_kwargs)
            self.out_proj = RowParallelLinear(self.d_inner,
                                              self.d_model,
                                              bias=bias,
                                              **factory_kwargs)
            return
        self.conv1d = ColumnParallelLinear(
            input_size=d_conv,
            output_size=self.d_inner,
            bias=conv_bias,
            params_dtype=dtype,
        )
        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `set_weight_attrs`
        # doesn't allow to override it
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        self.in_proj = MergedColumnParallelLinear(
            self.d_model,
            [self.d_inner] * 2,
            bias=bias,
            params_dtype=dtype,
        )

        # selective projection used to make dt, B and C input dependent
        self.x_proj = RowParallelLinear(
            self.d_inner,
            self.dt_rank + self.d_state * 2,
            bias=False,
            params_dtype=dtype,
        )

        # time step projection (discretization) -
        # In the forward we need to apply dt_proj without the bias,
        # as the bias is added in the selective scan kernel.
        self.dt_proj = ColumnParallelLinear(
            self.dt_rank,
            self.d_inner,
            bias=True,
            skip_bias_add=True,
            params_dtype=dtype,
        )

        # # D "skip" parameter
        # self.D = nn.Parameter(torch.ones(self.d_inner))  # Keep in fp32
        self.A = nn.Parameter(
            torch.empty(
                self.d_inner,
                self.d_state,
                dtype=torch.float32,
            ))
        self.D = nn.Parameter(torch.ones(self.d_inner, dtype=torch.float32))

        self.out_proj = RowParallelLinear(
            self.d_inner,
            self.d_model,
            bias=bias,
            input_is_parallel=True,
            params_dtype=dtype,
        )
        self.activation = "silu"

    def forward(self,
                hidden_states: torch.Tensor,
                attn_metadata: AttentionMetadata,
                mamba_cache_params: MambaCacheParams,
                yoco_key_values=None) -> torch.Tensor:

        if self.yoco_cross:
            out = self.in_proj(hidden_states)[0]
            out = self.swiGluActivation(yoco_key_values, out)
            out = self.out_proj(out)
            return out[0], yoco_key_values

        # 1. Gated MLP's linear projection
        # projected_states = self.in_proj(hidden_states)[0].transpose(-2, -1)
        projected_states = self.in_proj(
            hidden_states.to(self.in_proj.weight.dtype))[0].transpose(-2, -1)
        hidden_states, gate = projected_states.chunk(2, dim=-2)

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
        ssm_parameters = self.x_proj(hidden_states.transpose(-2, -1))[0]

        time_step, B, C = torch.split(
            ssm_parameters,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )

        # Note that Jamba normalizes B, C, and time_step here but Mamba doesn't.

        discrete_time_step = self.dt_proj(time_step)[0].transpose(-2, -1)
        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        time_proj_bias = (self.dt_proj.bias.float() if hasattr(
            self.dt_proj, "bias") else None)

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
                # z,
                None if self.yoco_kv else gate,
                time_proj_bias,
                delta_softplus=True,
                cache_indices=mamba_cache_params.state_indices_tensor,
                has_initial_state=attn_metadata.context_lens_tensor > 0,
                query_start_loc=attn_metadata.query_start_loc)
        else:
            scan_outputs = torch.empty_like(hidden_states.transpose(0, 1))
            selective_state_update(
                mamba_cache_params.ssm_state,
                hidden_states.transpose(0, 1),
                discrete_time_step.transpose(0, 1),
                self.A,
                B,
                C,
                self.D,
                # z
                # gate.transpose(0, 1),
                None if self.yoco_kv else gate.transpose(0, 1),
                time_proj_bias,
                dt_softplus=True,
                state_batch_indices=mamba_cache_params.state_indices_tensor,
                out=scan_outputs)
            scan_outputs = scan_outputs.transpose(0, 1)

        # 4. Final linear projection
        if self.yoco_kv:
            # gate = gate.transpose(-1,-2).contiguous()
            yoco_key_values = scan_outputs.transpose(-2, -1)
            scan_outputs = self.swiGluActivation(scan_outputs, gate)

        contextualized_states = self.out_proj(scan_outputs.transpose(-2,
                                                                     -1))[0]

        return contextualized_states, yoco_key_values


class SambaYDecoderLayer(nn.Module):

    def __init__(
        self,
        config,
        layer_idx,
        cache_config,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.mlp = SambaYMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size,
                                            eps=config.layer_norm_eps)

        self.yoco_mb = False
        self.yoco_cross = False
        if layer_idx >= config.num_hidden_layers // 2:
            self.yoco_mb = True
            self.yoco_cross = (layer_idx
                               >= (config.num_hidden_layers // 2 + 2))
        self.use_mamba = config.mb_per_layer > 0 and \
            layer_idx % config.mb_per_layer == 0
        if self.use_mamba:
            factory_kwargs = {"dtype": None}
            self.attn = Phi4Mamba(config.hidden_size,
                                  layer_idx=layer_idx,
                                  yoco_cross=self.yoco_cross,
                                  yoco_kv=self.yoco_mb,
                                  **factory_kwargs)
        else:
            self.attn = SambaYAttention(config,
                                        layer_idx=layer_idx,
                                        yoco_cross=self.yoco_cross,
                                        cache_config=cache_config,
                                        prefix=f"{prefix}.self_attn")
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size,
                                                     eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: AttentionMetadata,
        mamba_cache_params: MambaCacheParams,
        ssm_output: Optional[torch.LongTensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if self.use_mamba:
            assert mamba_cache_params is not None
        else:
            assert mamba_cache_params is None

        residual = hidden_states
        hidden_states = self.input_layernorm(
            hidden_states.to(dtype=self.input_layernorm.weight.dtype))

        if self.use_mamba:
            attn_outputs, ssm_output = self.attn(hidden_states,
                                                 attn_metadata,
                                                 mamba_cache_params,
                                                 yoco_key_values=ssm_output)
            residual = residual.to(torch.float32)
        else:
            attn_outputs = self.attn(hidden_states, )
        hidden_states = residual + attn_outputs
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(
            hidden_states.to(dtype=self.post_attention_layernorm.weight.dtype))
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, ssm_output


class SambaYModel(nn.Module):

    def __init__(self,
                 config,
                 cache_config=None,
                 quant_config=None,
                 lora_config=None,
                 prefix: str = "") -> None:
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
            lambda prefix: SambaYDecoderLayer(config,
                                              int(prefix.split('.')[-1]),
                                              cache_config,
                                              prefix=prefix),
            prefix=f"{prefix}.layers")
        self.final_layernorm = nn.LayerNorm(config.hidden_size,
                                            eps=config.layer_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        attn_metadata: AttentionMetadata,
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

        mamba_state_idx = 0
        ssm_output = None
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            if i == self.config.num_hidden_layers // 2 + 2:
                # profile run
                kv_cache_idx = self.config.num_hidden_layers // 2 + 1
                cache_layer = self.layers[kv_cache_idx]
                kv_cache = cache_layer.attn.attn.kv_cache
                if kv_cache[0].numel() == 0:
                    break

                # Starting from this layer, we do not need to calculate
                # the kv cache since we reuse the kv cache from last layer.
                # If in prefill phase, we can <s>prune></s> truncate
                # the hidden state to save computation cost.
                if attn_metadata.prefill_metadata and not envs.VLLM_USE_V1:
                    selected_token_indices = torch.cumsum(
                        attn_metadata.seq_lens_tensor, dim=0) - 1
                    hidden_states = hidden_states.index_select(
                        0, selected_token_indices)
                    ssm_output = ssm_output.index_select(
                        0, selected_token_indices)

            if layer.use_mamba:
                if i < self.config.num_hidden_layers // 2 or \
                    not layer.yoco_cross:
                    mamba_cache = mamba_cache_params.at_layer_idx(
                        mamba_state_idx)
                    mamba_state_idx += 1
                else:
                    mamba_cache = mamba_cache_params.at_layer_idx(
                        mamba_state_idx - 1)

                hidden_states, ssm_output = layer(hidden_states,
                                                  positions,
                                                  attn_metadata,
                                                  mamba_cache,
                                                  ssm_output=ssm_output)
            else:
                hidden_states, ssm_output = layer(
                    hidden_states,
                    positions,
                    attn_metadata,
                    None,  # mamba_cache_params
                    ssm_output=ssm_output)

        hidden_states = self.final_layernorm(
            hidden_states.to(dtype=self.final_layernorm.weight.dtype))
        return hidden_states


class Phi4FlashForCausalLM(nn.Module, HasInnerState, IsHybrid, SupportsV0Only):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        lora_config = vllm_config.lora_config
        quant_config = vllm_config.quant_config
        scheduler_config = vllm_config.scheduler_config
        self.compilation_config = vllm_config.compilation_config
        self.vllm_config = vllm_config
        # Prefix caching and chunked prefill is not supported for this model.
        assert not cache_config.enable_prefix_caching, \
            "Phi4flash currently does not support prefix caching"
        assert not scheduler_config.chunked_prefill_enabled, \
            "Phi4Flash currently does not support prefix caching"
        super().__init__()
        self.config = config
        self.model_config = vllm_config.model_config
        self.scheduler_config = scheduler_config
        self.model = SambaYModel(config,
                                 cache_config=cache_config,
                                 prefix=maybe_prefix(prefix, "model"))
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
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.embedding_bias = None
        # Used to track and store by the Mamba cache between steps.
        self.mamba_cache: Optional[MambaCacheManager] = None
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size,
                                                logits_as_input=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if self.mamba_cache is None:
            num_mamba_layers = self.config.num_hidden_layers \
                // 2 // self.config.mb_per_layer + 1
            self.mamba_cache = MambaCacheManager(
                self.vllm_config,
                num_mamba_layers,
                *self._get_mamba_cache_shape(),
                self.lm_head.weight.dtype,
                self.lm_head.weight.dtype,
            )
        mamba_cache_params = self.mamba_cache.current_run_tensors(**kwargs)

        attn_metadata = get_forward_context().attn_metadata
        # input_ids and hidden_states isn't a one-to-one mapping in prefill
        # stage due to YOCO optimization.
        hidden_states = self.model(input_ids, positions, attn_metadata,
                                   mamba_cache_params, intermediate_tensors,
                                   inputs_embeds)
        return hidden_states

    def _get_mamba_cache_shape(
            self
    ) -> tuple[Optional[tuple[int, int]], Optional[tuple[int, int]]]:
        world_size = get_tensor_model_parallel_world_size()
        hidden_size = self.config.hidden_size
        mamba_expand = self.config.mamba_expand  # 2
        mamba_d_conv = self.config.mamba_d_conv  # 4
        mamba_d_state = self.config.mamba_d_state  # 16
        conv_state_shape = (
            mamba_expand * hidden_size // world_size,
            mamba_d_conv - 1,
        )
        temporal_state_shape = (
            mamba_expand * hidden_size // world_size,
            mamba_d_state,
        )
        return conv_state_shape, temporal_state_shape

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.mamba_cache.copy_inputs_before_cuda_graphs(
            input_buffers, **kwargs)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self.mamba_cache.get_seqlen_agnostic_capture_inputs(batch_size)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        processed_logits = self.logits_processor(
            self.lm_head,
            hidden_states,
            self.embedding_bias,
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
