# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch

from vllm import envs
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.config import get_current_vllm_config
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.mamba.mamba2_metadata import (Mamba2Metadata,
                                                              update_metadata)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn, causal_conv1d_update)
from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op
from vllm.v1.attention.backends.mamba_attn import Mamba2AttentionMetadata


@dataclass
class ConvCacheParams:
    conv_state: torch.Tensor = torch.Tensor()
    state_indices_tensor: torch.Tensor = torch.Tensor()

    def at_layer_idx(self, layer_idx):
        return ConvCacheParams(self.conv_state[layer_idx],
                               self.state_indices_tensor)


@CustomOp.register("short_conv")
class ShortConv(CustomOp):

    def __init__(self, config, dim: int, layer_idx: int, prefix: str = ""):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.conv_dim = dim
        self.L_cache = config.conv_L_cache
        self.bias = config.conv_bias

        self.conv = ColumnParallelLinear(
            input_size=self.L_cache,
            output_size=dim,
            bias=self.bias,
            prefix=f"{prefix}.conv1d",
        )
        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `set_weight_attrs`
        # doesn't allow to override it
        self.conv.weight.data = self.conv.weight.data.unsqueeze(1)

        self.in_proj = MergedColumnParallelLinear(
            input_size=dim,
            output_sizes=[dim] * 3,
            bias=self.bias,
            prefix=f"{prefix}.in_proj",
        )
        self.out_proj = RowParallelLinear(
            input_size=dim,
            output_size=dim,
            bias=self.bias,
            prefix=f"{prefix}.out_proj",
        )

        if envs.VLLM_USE_V1:
            compilation_config = get_current_vllm_config().compilation_config
            if prefix in compilation_config.static_forward_context:
                raise ValueError(f"Duplicate layer name: {prefix}")
            compilation_config.static_forward_context[prefix] = self
            # The outer list is for v0 PP virtual engine. Though this code path
            # only runs for v1, we have to do this to unify with the interface
            # of Attention + v0 PP.
            # The inner tuple is (conv_state,)
            self.kv_cache = [(torch.tensor([]))]

        # For compatibility with StaticCacheSpec utils
        self.chunk_size = 1
        self.prefix = prefix

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        conv_cache_params: ConvCacheParams,
        conv_metadata: Mamba2Metadata,
    ):
        return

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        conv_cache_params: ConvCacheParams,
        conv_metadata: Mamba2Metadata,
    ):
        if not envs.VLLM_USE_V1:
            CustomOp.forward(self, hidden_states, output, conv_cache_params,
                             conv_metadata)
        else:
            torch.ops.vllm.short_conv(
                hidden_states,
                output,
                self.prefix,
            )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        conv_cache_params: ConvCacheParams,
        conv_metadata: Mamba2Metadata,
    ):
        forward_context = get_forward_context()
        # Mamba2Metadata contains metadata necessary for the mamba2 triton
        # kernels to operate in continuous batching and in chunked prefill
        # modes; they are computed at top-level model forward since they
        # stay the same and reused for all mamba layers in the same iteration
        attn_metadata: AttentionMetadata = forward_context.attn_metadata
        if envs.VLLM_USE_V1:
            if attn_metadata is not None:
                assert isinstance(attn_metadata, dict)
                attn_metadata = attn_metadata[self.prefix]
                conv_metadata = attn_metadata
                assert isinstance(attn_metadata, Mamba2AttentionMetadata)
                self_kv_cache = self.kv_cache[forward_context.virtual_engine]
                conv_state = self_kv_cache[0].transpose(-1, -2)
                state_indices_tensor = attn_metadata.state_indices_tensor
                has_initial_states_p = attn_metadata.has_initial_states
        else:
            conv_state = conv_cache_params.conv_state
            state_indices_tensor = conv_cache_params.state_indices_tensor
            has_initial_states_p = conv_metadata.has_initial_states

        BCx, _ = self.in_proj(hidden_states)

        B, C, x = BCx.chunk(3, dim=-1)

        conv_weights = self.conv.weight.view(self.conv.weight.size(0),
                                             self.conv.weight.size(2))

        if envs.VLLM_USE_V1 and attn_metadata is None:
            # V1 profile run
            Bx = (B * x).contiguous()
            hidden_states = C * Bx
            contextualized_states, _ = self.out_proj(hidden_states)
            return contextualized_states

        num_prefills = attn_metadata.num_prefills  # request count
        num_decodes = attn_metadata.num_decode_tokens  # token count (=request)
        num_prefill_tokens = attn_metadata.num_prefill_tokens  # token count
        has_prefill = num_prefills > 0
        has_decode = num_decodes > 0
        num_actual_tokens = num_decodes + num_prefill_tokens

        # NOTE: V0 put prefill before decode, v1 puts decode before prefill
        # Separate prefill and decode by splitting varlen input
        # Split along token dimension
        if envs.VLLM_USE_V1:
            B_d, B_p = torch.split(
                B[:num_actual_tokens],
                [num_decodes, num_prefill_tokens],
                dim=0,
            )
            C_d, C_p = torch.split(
                C[:num_actual_tokens],
                [num_decodes, num_prefill_tokens],
                dim=0,
            )
            x_d, x_p = torch.split(
                x[:num_actual_tokens],
                [num_decodes, num_prefill_tokens],
                dim=0,
            )
            # Split along batch dimension
            state_indices_tensor_d, state_indices_tensor_p = torch.split(
                state_indices_tensor[:num_actual_tokens],
                [num_decodes, num_prefills],
                dim=0,
            )
            query_start_loc_p = (
                attn_metadata.query_start_loc[-num_prefills - 1:] -
                num_decodes if has_prefill else None)
        else:
            B_p, B_d = torch.split(
                B,
                [num_prefill_tokens, num_decodes],
                dim=0,
            )
            C_p, C_d = torch.split(
                C,
                [num_prefill_tokens, num_decodes],
                dim=0,
            )
            x_p, x_d = torch.split(
                x,
                [num_prefill_tokens, num_decodes],
                dim=0,
            )
            # Split along batch dimension
            state_indices_tensor_p, state_indices_tensor_d = torch.split(
                state_indices_tensor,
                [num_prefills, num_decodes],
                dim=0,
            )
            query_start_loc_p = (attn_metadata.query_start_loc[:num_prefills +
                                                               1]
                                 if has_prefill else None)

        conv_output_list = []

        if has_prefill:
            Bx_p = (B_p * x_p).transpose(0, 1)
            if conv_metadata.cu_seqlen is None:
                conv_metadata = update_metadata(Bx_p, query_start_loc_p,
                                                conv_metadata)
            Bx = causal_conv1d_fn(Bx_p,
                                  conv_weights,
                                  self.conv.bias,
                                  activation=None,
                                  conv_states=conv_state,
                                  has_initial_state=has_initial_states_p,
                                  cache_indices=state_indices_tensor_p,
                                  metadata=conv_metadata,
                                  query_start_loc=query_start_loc_p).transpose(
                                      0, 1)[:num_prefill_tokens]

            y = C_p * Bx
            conv_output_list.append(y)

        if has_decode:
            Bx_d = (B_d * x_d).contiguous()
            Bx = causal_conv1d_update(
                Bx_d,
                conv_state,
                conv_weights,
                self.conv.bias,
                activation=None,
                conv_state_indices=state_indices_tensor_d)
            y = C_d * Bx
            if envs.VLLM_USE_V1:
                conv_output_list.insert(0, y)
            else:
                conv_output_list.append(y)

        # Merge prefill and decode outputs before passing to gated MLP
        hidden_states = torch.vstack(conv_output_list)

        # Final linear projection
        output[:num_actual_tokens], _ = self.out_proj(hidden_states)

    def get_state_shape(self) -> tuple[tuple[int, ...]]:
        world_size = get_tensor_model_parallel_world_size()
        # contiguous along 'dim' axis
        conv_state_shape = (
            self.L_cache - 1,
            divide(self.conv_dim, world_size),
        )
        return (conv_state_shape, )


def short_conv(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self.forward_cuda(hidden_states=hidden_states,
                      output=output,
                      conv_cache_params=None,
                      conv_metadata=None)


def short_conv_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="short_conv",
    op_func=short_conv,
    mutates_args=["output"],
    fake_impl=short_conv_fake,
    dispatch_key=current_platform.dispatch_key,
)
