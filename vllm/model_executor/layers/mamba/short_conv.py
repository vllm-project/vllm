# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

import torch

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.config import CacheConfig, ModelConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.utils import direct_register_custom_op
from vllm.v1.attention.backends.short_conv_attn import ShortConvAttentionMetadata


@CustomOp.register("short_conv")
class ShortConv(MambaBase, CustomOp):
    def __init__(
        self,
        config,
        dim: int,
        layer_idx: int,
        model_config: Optional[ModelConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
    ):
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

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.kv_cache = (torch.tensor([]),)

        self.model_config = model_config
        self.cache_config = cache_config
        self.prefix = prefix

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        return

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        torch.ops.vllm.short_conv(
            hidden_states,
            output,
            self.prefix,
        )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        forward_context = get_forward_context()
        # ShortConvAttentionMetadata contains metadata necessary for the
        # short_conv triton kernels to operate in continuous batching and in
        # chunked prefill modes; they are computed at top-level model forward
        # since they stay the same and reused for all mamba layers in the same
        # iteration.
        attn_metadata: AttentionMetadata = forward_context.attn_metadata
        
        assert self.cache_config is not None
        mamba_block_size = self.cache_config.mamba_block_size
        prefix_caching_enabled = self.cache_config.enable_prefix_caching             
        if attn_metadata is not None:
            assert isinstance(attn_metadata, dict)
            attn_metadata = attn_metadata[self.prefix]
            assert isinstance(attn_metadata, ShortConvAttentionMetadata)
            self_kv_cache = self.kv_cache[forward_context.virtual_engine]
            conv_state = self_kv_cache[0].transpose(-1, -2)
            state_indices_tensor = attn_metadata.state_indices_tensor
            has_initial_states_p = attn_metadata.has_initial_states_p
            
            #WILL - check them
            seq_idx_p = attn_metadata.seq_idx_p
            query_start_loc_p = attn_metadata.query_start_loc_p

        BCx, _ = self.in_proj(hidden_states)

        B, C, x = BCx.chunk(3, dim=-1)

        conv_weights = self.conv.weight.view(
            self.conv.weight.size(0), self.conv.weight.size(2)
        )

        if attn_metadata is None:
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

        # NOTE: V1 puts decode before prefill
        # Separate prefill and decode by splitting varlen input
        # Split along token dimension
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
            state_indices_tensor,
            [num_decodes, num_prefills],
            dim=0,
        )
        
        

        if prefix_caching_enabled:         
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
            # Prefill-only variables:
            block_idx_first_scheduled_token_p = (
                attn_metadata.block_idx_first_scheduled_token_p
            )
            
            #print("\n이값은 메타데이터의 block_idx_first_scheduled_token_p", block_idx_first_scheduled_token_p)
            
            num_computed_tokens_p = attn_metadata.num_computed_tokens_p
            
        else:
            block_idx_last_computed_token_d = None
            block_idx_last_computed_token_p = None
            block_idx_last_scheduled_token_d = None
            block_idx_last_scheduled_token_p = None
            block_idx_first_scheduled_token_p = None
            num_computed_tokens_p = None

        conv_output_list = []

        if has_prefill:
            Bx_p = (B_p * x_p).transpose(0, 1)
            
            #WILL For Triton
            state_indices_tensor_p = state_indices_tensor_p.view(-1)
            Bx = causal_conv1d_fn(
                Bx_p,
                conv_weights,
                self.conv.bias,
                activation=None,
                conv_states=conv_state,
                has_initial_state=has_initial_states_p,
                cache_indices=state_indices_tensor_p,
                block_idx_first_scheduled_token=block_idx_first_scheduled_token_p,
                block_idx_last_scheduled_token=(block_idx_last_scheduled_token_p if block_idx_last_scheduled_token_p is not None else None),
                initial_state_idx=block_idx_last_computed_token_p,
                num_computed_tokens=num_computed_tokens_p,
                block_size_to_align=mamba_block_size,
                metadata=attn_metadata,
                query_start_loc=query_start_loc_p,
            ).transpose(0, 1)[:num_prefill_tokens]

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
                conv_state_indices=state_indices_tensor_d,
                block_idx_last_scheduled_token=block_idx_last_scheduled_token_d,
                initial_state_idx=block_idx_last_computed_token_d,      
            )
                        
            y = C_d * Bx
            conv_output_list.insert(0, y)

        # Merge prefill and decode outputs before passing to gated MLP
        hidden_states = torch.vstack(conv_output_list)

        # Final linear projection
        output[:num_actual_tokens], _ = self.out_proj(hidden_states)

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        assert self.model_config is not None
        assert self.cache_config is not None
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...]]:
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=get_tensor_model_parallel_world_size(),
            intermediate_size=self.conv_dim,
            conv_kernel=self.L_cache,
        )

    @property
    def mamba_type(self) -> str:
        return "short_conv"

    def get_attn_backend(self) -> type["AttentionBackend"]:
        from vllm.v1.attention.backends.short_conv_attn import ShortConvAttentionBackend

        return ShortConvAttentionBackend


def short_conv(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self.forward_cuda(hidden_states=hidden_states, output=output)


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
)
