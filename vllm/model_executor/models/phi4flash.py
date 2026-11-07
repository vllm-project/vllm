# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Phi-4 Flash model."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers.activations import ACT2FN

from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_mixer import (
    split_batch_to_prefill_and_decode,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_scan_fn
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import selective_state_update
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.model_executor.models.utils import (
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.differential_flash_attn import (
    DifferentialFlashAttentionBackend,
)
from vllm.v1.attention.backends.mamba1_attn import Mamba1AttentionMetadata
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum


class SwiGLUActivation(nn.Module):
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return x1 * nn.functional.silu(x2)


class SambaYMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.fc1 = nn.Linear(
            config.hidden_size, 2 * config.intermediate_size, bias=False
        )
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        y = self.fc1(hidden_states)
        gate, y = y.chunk(2, dim=-1)
        y = y * self.activation_fn(gate)
        return self.fc2(y)


class SambaYAttention(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.yoco_cross = config.layers_block_type[layer_idx] == "shared_attention"

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                "hidden_size must be divisible by num_heads "
                f"(got hidden_size={self.hidden_size}, "
                f"num_heads={self.num_heads})."
            )
        if self.num_heads % 2 != 0 or self.num_key_value_heads % 2 != 0:
            raise ValueError(
                "Phi-4 Flash differential attention requires an even number of heads."
            )

        op_size = self.num_heads * self.head_dim + 2 * (
            self.num_key_value_heads * self.head_dim
        )
        self.out_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=True
        )
        if self.yoco_cross:
            self.Wqkv = nn.Linear(
                self.hidden_size, self.num_heads * self.head_dim, bias=True
            )
            sliding_window = None
            kv_shared_layer_index = config.num_hidden_layers // 2 + 1
            kv_sharing_target_layer_name = (
                f"model.layers.{kv_shared_layer_index}.attn.attn"
            )
        else:
            self.Wqkv = nn.Linear(self.hidden_size, op_size, bias=True)
            sliding_window = config.sliding_window[layer_idx]
            kv_sharing_target_layer_name = None

        lambda_init = self._lambda_init(layer_idx)
        self.lambda_q1 = nn.Parameter(
            torch.empty(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k1 = nn.Parameter(
            torch.empty(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_q2 = nn.Parameter(
            torch.empty(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k2 = nn.Parameter(
            torch.empty(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.subln = nn.RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_key_value_heads,
            cache_config=cache_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            attn_backend=DifferentialFlashAttentionBackend,
            differential_flash_attention_config={
                "lambda_init": lambda_init,
                "lambda_q1": self.lambda_q1,
                "lambda_k1": self.lambda_k1,
                "lambda_q2": self.lambda_q2,
                "lambda_k2": self.lambda_k2,
                "subln": self.subln,
            },
        )

    @staticmethod
    def _lambda_init(depth: int) -> float:
        return 0.8 - 0.6 * math.exp(-0.3 * depth)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.yoco_cross:
            q = self.Wqkv(hidden_states)
            attn_output = self.attn(q, None, None)
        else:
            qkv = self.Wqkv(hidden_states)
            q, k, v = qkv.split(
                [
                    self.hidden_size,
                    self.num_key_value_heads * self.head_dim,
                    self.num_key_value_heads * self.head_dim,
                ],
                dim=-1,
            )
            attn_output = self.attn(q, k, v)
        return self.out_proj(attn_output.view(-1, self.num_heads * self.head_dim))


class Phi4FlashYocoCrossMamba(nn.Module):
    def __init__(
        self, hidden_size: int, expand: int = 2, bias: bool = False, prefix: str = ""
    ) -> None:
        super().__init__()
        self.d_inner = hidden_size * expand
        self.swi_glu_activation = SwiGLUActivation()
        self.in_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.d_inner],
            bias=bias,
            prefix=f"{prefix}.in_proj",
        )
        self.out_proj = RowParallelLinear(
            self.d_inner,
            hidden_size,
            bias=bias,
            prefix=f"{prefix}.out_proj",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        yoco_key_values: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if yoco_key_values is None:
            raise ValueError(
                "Phi-4 Flash YOCO Mamba layer requires cached SSM outputs."
            )
        out = self.in_proj(hidden_states)[0]
        out = self.swi_glu_activation(yoco_key_values, out)
        out = self.out_proj(out)[0]
        return out, yoco_key_values


class Phi4FlashMamba(MambaBase):
    def __init__(
        self,
        hidden_size: int,
        *,
        ssm_state_size: int,
        conv_kernel_size: int,
        expand: int,
        time_step_rank: int,
        use_conv_bias: bool,
        use_bias: bool,
        yoco_kv: bool,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.ssm_state_size = ssm_state_size
        self.conv_kernel_size = conv_kernel_size
        self.expand = expand
        self.intermediate_size = self.expand * self.hidden_size
        self.time_step_rank = time_step_rank
        self.yoco_kv = yoco_kv
        self.activation = "silu"
        self.prefix = prefix
        self.model_config = model_config
        self.cache_config = cache_config
        self.swi_glu_activation = SwiGLUActivation()

        self.conv1d = ColumnParallelLinear(
            input_size=conv_kernel_size,
            output_size=self.intermediate_size,
            bias=use_conv_bias,
            prefix=f"{prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        self.in_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.intermediate_size] * 2,
            bias=use_bias,
            prefix=f"{prefix}.in_proj",
        )
        self.x_proj = RowParallelLinear(
            self.intermediate_size,
            self.time_step_rank + self.ssm_state_size * 2,
            bias=False,
            prefix=f"{prefix}.x_proj",
        )
        self.dt_proj = ColumnParallelLinear(
            self.time_step_rank,
            self.intermediate_size,
            bias=True,
            skip_bias_add=True,
            prefix=f"{prefix}.dt_proj",
        )

        def weight_loader(
            param: torch.nn.Parameter, loaded_weight: torch.Tensor
        ) -> None:
            tp_rank = get_tensor_model_parallel_rank()
            tp_size = get_tensor_model_parallel_world_size()
            shard = loaded_weight.data.split(loaded_weight.shape[0] // tp_size, dim=0)[
                tp_rank
            ]
            param.data.copy_(shard)

        def a_weight_loader(
            param: torch.nn.Parameter, loaded_weight: torch.Tensor
        ) -> None:
            weight_loader(param, -torch.exp(loaded_weight.float()))

        tp_size = get_tensor_model_parallel_world_size()
        self.A = nn.Parameter(
            torch.empty(
                self.intermediate_size // tp_size,
                self.ssm_state_size,
                dtype=torch.float32,
            )
        )
        self.D = nn.Parameter(
            torch.ones(self.intermediate_size // tp_size, dtype=torch.float32)
        )
        set_weight_attrs(self.A, {"weight_loader": a_weight_loader})
        set_weight_attrs(self.D, {"weight_loader": weight_loader})

        self.out_proj = RowParallelLinear(
            self.intermediate_size,
            hidden_size,
            bias=use_bias,
            input_is_parallel=True,
            prefix=f"{prefix}.out_proj",
        )
        self.kv_cache = (torch.tensor([]), torch.tensor([]))

    def _ssm_transform(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ssm_params = self.x_proj(x)[0]
        time_step, b, c = torch.split(
            ssm_params,
            [self.time_step_rank, self.ssm_state_size, self.ssm_state_size],
            dim=-1,
        )
        discrete_time_step = self.dt_proj(time_step)[0].transpose(-2, -1)
        return discrete_time_step, b, c

    def _time_proj_bias(self) -> torch.Tensor | None:
        if hasattr(self.dt_proj, "bias") and self.dt_proj.bias is not None:
            return self.dt_proj.bias.float()
        return None

    def forward(
        self,
        hidden_states: torch.Tensor,
        yoco_key_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        forward_context: ForwardContext = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        attn_metadata = None
        if attn_metadata_raw is not None:
            assert isinstance(attn_metadata_raw, dict)
            attn_metadata = attn_metadata_raw[self.prefix]
            assert isinstance(attn_metadata, Mamba1AttentionMetadata)

        projected_states = self.in_proj(hidden_states)[0].transpose(-2, -1)
        hidden_states_bc, gate = projected_states.chunk(2, dim=-2)
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        if attn_metadata is None:
            out = self.out_proj(hidden_states_bc.transpose(-2, -1))[0]
            return out, yoco_key_values

        assert self.cache_config is not None
        conv_state = (
            self.kv_cache[0]
            if is_conv_state_dim_first()
            else self.kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self.kv_cache[1]
        mamba_block_size = self.cache_config.mamba_block_size
        is_mamba_cache_all = self.cache_config.mamba_cache_mode == "all"

        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_prefills = attn_metadata.num_prefills
        num_decodes = attn_metadata.num_decode_tokens
        num_actual_tokens = num_prefill_tokens + num_decode_tokens

        split = split_batch_to_prefill_and_decode(
            hidden_states_bc,
            gate,
            num_prefill_tokens,
            num_decode_tokens,
        )
        hidden_states_bc_p = split.hidden_states_BC_p
        hidden_states_bc_d = split.hidden_states_BC_d
        gate_p = split.gate_p
        gate_d = split.gate_d

        if is_mamba_cache_all:
            block_idx_last_computed_token_d, block_idx_last_computed_token_p = (
                torch.split(
                    attn_metadata.block_idx_last_computed_token,
                    [num_decodes, num_prefills],
                    dim=0,
                )
            )
            block_idx_last_scheduled_token_d, block_idx_last_scheduled_token_p = (
                torch.split(
                    attn_metadata.block_idx_last_scheduled_token,
                    [num_decodes, num_prefills],
                    dim=0,
                )
            )
            block_idx_first_scheduled_token_p = (
                attn_metadata.block_idx_first_scheduled_token_p
            )
            num_computed_tokens_p = attn_metadata.num_computed_tokens_p
        else:
            block_idx_last_computed_token_d = None
            block_idx_last_computed_token_p = None
            block_idx_last_scheduled_token_d = None
            block_idx_last_scheduled_token_p = None
            block_idx_first_scheduled_token_p = None
            num_computed_tokens_p = None

        outputs = []
        time_proj_bias = self._time_proj_bias()

        if num_prefill_tokens > 0:
            conv_out_p = causal_conv1d_fn(
                hidden_states_bc_p,
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=attn_metadata.has_initial_states_p,
                cache_indices=attn_metadata.state_indices_tensor_p,
                query_start_loc=attn_metadata.query_start_loc_p,
                block_idx_first_scheduled_token=block_idx_first_scheduled_token_p,
                block_idx_last_scheduled_token=block_idx_last_scheduled_token_p,
                initial_state_idx=block_idx_last_computed_token_p,
                num_computed_tokens=num_computed_tokens_p,
                block_size_to_align=mamba_block_size,
                metadata=attn_metadata,
            )
            discrete_time_step_p, b_p, c_p = self._ssm_transform(
                conv_out_p.transpose(-2, -1)
            )
            scan_out_p = selective_scan_fn(
                conv_out_p,
                ssm_state,
                discrete_time_step_p,
                self.A,
                b_p.transpose(-2, -1),
                c_p.transpose(-2, -1),
                self.D.float(),
                None if self.yoco_kv else gate_p,
                time_proj_bias,
                delta_softplus=True,
                cache_indices=attn_metadata.state_indices_tensor_p,
                has_initial_state=attn_metadata.has_initial_states_p,
                query_start_loc=attn_metadata.query_start_loc_p,
                block_size=mamba_block_size,
                block_idx_first_scheduled_token=block_idx_first_scheduled_token_p,
                block_idx_last_scheduled_token=block_idx_last_scheduled_token_p,
                initial_state_idx=block_idx_last_computed_token_p,
                cu_chunk_seqlen=attn_metadata.cu_chunk_seqlen_p,
                last_chunk_indices=attn_metadata.last_chunk_indices_p,
            )
            outputs.append(scan_out_p)

        if num_decode_tokens > 0:
            assert attn_metadata.state_indices_tensor_d is not None
            if is_mamba_cache_all:
                state_indices_tensor_d_input = (
                    attn_metadata.state_indices_tensor_d.gather(
                        1, block_idx_last_computed_token_d.unsqueeze(1)
                    ).squeeze(1)
                )
                state_indices_tensor_d_output = (
                    attn_metadata.state_indices_tensor_d.gather(
                        1, block_idx_last_scheduled_token_d.unsqueeze(1)
                    ).squeeze(1)
                )
            else:
                state_indices_tensor_d_input = attn_metadata.state_indices_tensor_d
                state_indices_tensor_d_output = attn_metadata.state_indices_tensor_d

            conv_out_d = causal_conv1d_update(
                hidden_states_bc_d.transpose(0, 1),
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=attn_metadata.state_indices_tensor_d,
                block_idx_last_scheduled_token=block_idx_last_scheduled_token_d,
                initial_state_idx=block_idx_last_computed_token_d,
            ).transpose(0, 1)
            discrete_time_step_d, b_d, c_d = self._ssm_transform(
                conv_out_d.transpose(-2, -1)
            )
            scan_out_d = torch.empty_like(hidden_states_bc_d.transpose(0, 1))
            selective_state_update(
                ssm_state,
                conv_out_d.transpose(0, 1),
                discrete_time_step_d.transpose(0, 1),
                self.A,
                b_d,
                c_d,
                self.D,
                time_proj_bias,
                z=None if self.yoco_kv else gate_d.transpose(0, 1),
                dt_softplus=True,
                state_batch_indices=state_indices_tensor_d_input,
                dst_state_batch_indices=state_indices_tensor_d_output,
                out=scan_out_d,
            )
            outputs.insert(0, scan_out_d.transpose(0, 1))

        scan_outputs = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
        if self.yoco_kv:
            yoco_key_values = scan_outputs.transpose(-2, -1)
            scan_outputs = self.swi_glu_activation(
                scan_outputs, gate[..., :num_actual_tokens]
            )
        out = self.out_proj(scan_outputs.transpose(-2, -1))[0]
        return out, yoco_key_values

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        assert self.model_config is not None
        assert self.cache_config is not None
        return MambaStateDtypeCalculator.mamba1_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return MambaStateShapeCalculator.mamba1_state_shape(
            tp_world_size=get_tensor_model_parallel_world_size(),
            intermediate_size=self.intermediate_size,
            state_size=self.ssm_state_size,
            conv_kernel=self.conv_kernel_size,
        )

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.MAMBA1


class SambaYDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.use_mamba = (
            config.mb_per_layer > 0 and layer_idx % config.mb_per_layer == 0
        )
        self.yoco_cross = layer_idx >= (config.num_hidden_layers // 2 + 2)
        self.mlp = SambaYMLP(config)
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

        if self.use_mamba:
            if self.yoco_cross:
                self.attn = Phi4FlashYocoCrossMamba(
                    config.hidden_size,
                    expand=config.mamba_expand,
                    bias=config.mamba_proj_bias,
                    prefix=f"{prefix}.attn",
                )
            else:
                self.attn = Phi4FlashMamba(
                    config.hidden_size,
                    ssm_state_size=config.mamba_d_state,
                    conv_kernel_size=config.mamba_d_conv,
                    expand=config.mamba_expand,
                    time_step_rank=config.mamba_dt_rank,
                    use_conv_bias=config.mamba_conv_bias,
                    use_bias=config.mamba_proj_bias,
                    yoco_kv=layer_idx >= config.num_hidden_layers // 2,
                    model_config=model_config,
                    cache_config=cache_config,
                    prefix=f"{prefix}.attn",
                )
        else:
            self.attn = SambaYAttention(
                config,
                layer_idx=layer_idx,
                cache_config=cache_config,
                prefix=f"{prefix}.attn",
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        ssm_output: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del positions
        residual = hidden_states
        hidden_states = self.input_layernorm(
            hidden_states.to(dtype=self.input_layernorm.weight.dtype)
        )

        if self.use_mamba:
            attn_outputs, ssm_output = self.attn(hidden_states, ssm_output)
            residual = residual.to(torch.float32)
        else:
            attn_outputs = self.attn(hidden_states)
        hidden_states = residual + attn_outputs
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(
            hidden_states.to(dtype=self.post_attention_layernorm.weight.dtype)
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, ssm_output


class SambaYModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        if get_pp_group().world_size != 1:
            raise ValueError("Pipeline parallel is not supported for Phi-4 Flash.")

        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda layer_prefix: SambaYDecoderLayer(
                config,
                int(layer_prefix.rsplit(".", 1)[1]),
                model_config=model_config,
                cache_config=cache_config,
                prefix=layer_prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        self.final_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            raise ValueError(
                "Phi-4 Flash does not support pipeline parallel intermediate tensors."
            )
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)

        ssm_output = None
        shared_cache_layer_idx = self.config.num_hidden_layers // 2 + 1
        shared_cache_start_idx = self.config.num_hidden_layers // 2 + 2
        for layer_idx in range(self.start_layer, self.end_layer):
            if layer_idx == shared_cache_start_idx:
                shared_cache = self.layers[shared_cache_layer_idx].attn.attn.kv_cache
                if shared_cache.numel() == 0:
                    break
            hidden_states, ssm_output = self.layers[layer_idx](
                hidden_states,
                positions,
                ssm_output=ssm_output,
            )
        return self.final_layernorm(
            hidden_states.to(dtype=self.final_layernorm.weight.dtype)
        )


class Phi4FlashForCausalLM(nn.Module, HasInnerState, IsHybrid):
    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.mamba1_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config
        return MambaStateShapeCalculator.mamba1_state_shape(
            tp_world_size=parallel_config.tensor_parallel_size,
            intermediate_size=hf_config.mamba_expand * hf_config.hidden_size,
            state_size=hf_config.mamba_d_state,
            conv_kernel=hf_config.mamba_d_conv,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.mamba1_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        scheduler_config = vllm_config.scheduler_config
        lora_config = vllm_config.lora_config
        quant_config = vllm_config.quant_config

        if cache_config.enable_prefix_caching:
            raise ValueError("Phi-4 Flash does not support prefix caching.")
        if scheduler_config.chunked_prefill_enabled:
            raise ValueError("Phi-4 Flash does not support chunked prefill.")

        self.config = config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.scheduler_config = scheduler_config
        self.model = SambaYModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        unpadded_vocab_size = config.vocab_size
        if lora_config:
            unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=(
                DEFAULT_VOCAB_PADDING_SIZE
                if not lora_config
                else lora_config.lora_vocab_padding_size
            ),
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            unpadded_vocab_size,
            config.vocab_size,
            logits_as_input=False,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        embed_weight: torch.Tensor | None = None
        for name, loaded_weight in weights:
            if name == "model.embed_tokens.weight":
                embed_weight = loaded_weight
            if "A_log" in name:
                name = name.replace("A_log", "A")
            if "inner_cross_attn." in name:
                name = name.replace("inner_cross_attn.", "")
            if name.endswith(".bias") and name not in params_dict:
                continue
            if is_pp_missing_parameter(name, self):
                continue
            param = params_dict.get(name)
            if param is None:
                continue
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        if embed_weight is not None and "lm_head.weight" in params_dict:
            weight_loader = getattr(
                params_dict["lm_head.weight"], "weight_loader", default_weight_loader
            )
            weight_loader(params_dict["lm_head.weight"], embed_weight)
            loaded_params.add("lm_head.weight")
        return loaded_params
