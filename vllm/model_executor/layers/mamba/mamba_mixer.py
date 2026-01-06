# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import NamedTuple

import torch
from torch import nn
from torch.nn.parameter import Parameter

from vllm.config import CacheConfig, ModelConfig, get_current_vllm_config
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.layernorm import RMSNorm
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
from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    selective_scan_fn,
    selective_state_update,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.mamba1_attn import Mamba1AttentionMetadata


# Adapted from transformers.models.mamba.modeling_mamba.MambaMixer
@CustomOp.register("mamba_mixer")
class MambaMixer(MambaBase, CustomOp):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute
    the `contextualized_states`. A, D are input independent
    (see Mamba paper [1] Section 3.5.2 "Interpretation of A"
    for why A isn't selective) ∆, B, C are input-dependent
    (this is a key difference between Mamba and the linear time
    invariant S4, and is why Mamba is called
    **selective** state spaces)
    """

    def __init__(
        self,
        hidden_size: int,
        ssm_state_size: int,
        conv_kernel_size: int,
        intermediate_size: int,
        time_step_rank: int,
        use_conv_bias: bool,
        use_bias: bool,
        use_rms_norm: bool,
        rms_norm_has_weight: bool = True,
        rms_norm_eps: float = 1e-5,
        activation="silu",
        is_lora_enabled: bool = False,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.time_step_rank = time_step_rank
        self.ssm_state_size = ssm_state_size
        self.use_rms_norm = use_rms_norm
        self.activation = activation
        self.is_lora_enabled = is_lora_enabled
        self.conv_kernel_size = conv_kernel_size
        self.intermediate_size = intermediate_size

        self.conv1d = ColumnParallelLinear(
            input_size=conv_kernel_size,
            output_size=intermediate_size,
            bias=use_conv_bias,
            prefix=f"{prefix}.conv1d",
        )
        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `set_weight_attrs`
        # doesn't allow to override it
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        self.in_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=use_bias,
            prefix=f"{prefix}.in_proj",
        )

        # selective projection used to make dt, B and C input dependent
        self.x_proj = RowParallelLinear(
            intermediate_size,
            time_step_rank + ssm_state_size * 2,
            bias=False,
            prefix=f"{prefix}.x_proj",
        )
        # time step projection (discretization) -
        # In the forward we need to apply dt_proj without the bias,
        # as the bias is added in the selective scan kernel.
        self.dt_proj = ColumnParallelLinear(
            time_step_rank,
            intermediate_size,
            bias=True,
            skip_bias_add=True,
            prefix=f"{prefix}.dt_proj",
        )

        def weight_loader(param: Parameter, loaded_weight: torch.Tensor):
            tp_rank = get_tensor_model_parallel_rank()
            tp_size = get_tensor_model_parallel_world_size()
            param.data.copy_(
                loaded_weight.data.split(loaded_weight.shape[0] // tp_size, dim=0)[
                    tp_rank
                ]
            )

        def A_weight_loader(param: Parameter, loaded_weight: torch.Tensor):
            weight_loader(param, -torch.exp(loaded_weight.float()))

        tp_size = get_tensor_model_parallel_world_size()
        self.A = nn.Parameter(
            torch.empty(
                intermediate_size // tp_size,
                ssm_state_size,
                dtype=torch.float32,
            )
        )
        self.D = nn.Parameter(torch.ones(intermediate_size // tp_size))

        set_weight_attrs(self.D, {"weight_loader": weight_loader})
        set_weight_attrs(self.A, {"weight_loader": A_weight_loader})

        self.out_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=use_bias,
            input_is_parallel=True,
            prefix=f"{prefix}.out_proj",
        )

        self.dt_layernorm = (
            RMSNorm(
                time_step_rank,
                eps=rms_norm_eps,
                has_weight=rms_norm_has_weight,
            )
            if use_rms_norm
            else None
        )

        self.b_layernorm = (
            RMSNorm(
                ssm_state_size,
                eps=rms_norm_eps,
                has_weight=rms_norm_has_weight,
            )
            if use_rms_norm
            else None
        )

        self.c_layernorm = (
            RMSNorm(
                ssm_state_size,
                eps=rms_norm_eps,
                has_weight=rms_norm_has_weight,
            )
            if use_rms_norm
            else None
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        # The inner tuple is (conv_state, ssm_state)
        self.kv_cache = (torch.tensor([]), torch.tensor([]))

        self.model_config = model_config
        self.cache_config = cache_config
        self.prefix = prefix

    def _ssm_transform(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.is_lora_enabled:
            #  Lora kernel requires contiguous tensor.
            ssm_params = self.x_proj(x.contiguous())[0]
        else:
            ssm_params = self.x_proj(x)[0]
        time_step, B, C = torch.split(
            ssm_params,
            [self.time_step_rank, self.ssm_state_size, self.ssm_state_size],
            dim=-1,
        )
        if self.use_rms_norm:
            assert self.dt_layernorm is not None
            assert self.b_layernorm is not None
            assert self.c_layernorm is not None
            time_step = self.dt_layernorm(time_step.contiguous())
            B = self.b_layernorm(B.contiguous())
            C = self.c_layernorm(C.contiguous())
        discrete_time_step = self.dt_proj(time_step)[0].transpose(-2, -1)
        return discrete_time_step, B, C

    def forward(self, hidden_states: torch.Tensor, output: torch.Tensor):
        torch.ops.vllm.mamba_mixer(
            hidden_states,
            output,
            self.prefix,
        )

    def forward_native(self, hidden_states: torch.Tensor, output: torch.Tensor):
        pass

    def forward_cuda(self, hidden_states: torch.Tensor, output: torch.Tensor):
        """
        Run the Mamba-1 SSM pipeline.

        Steps
        -----
        1. Apply the gated-MLP linear projection to the raw input.
        2. Pass the projected sequence through the convolutional mixing layer.
        3. Feed the result into the State-Space Model (SSM) blocks.
        4. Perform the recurrence y ← SSM(A, B, C, Δ)(x)
           to produce contextual representations.
        5. Project the contextualised sequence back
           to the output embedding dimension.

        Batch handling
        --------------
        Prefill and decode tokens are processed by dedicated CUDA
        kernels for both the convolutional (conv1d) and SSM stages.
        In the case of a mixed batch (containing both prefill and
        decode tokens), both sets of kernels are executed independently
        and their outputs are concatenated before the final output projection.
        """

        forward_context: ForwardContext = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        assert self.cache_config is not None
        mamba_block_size = self.cache_config.mamba_block_size
        prefix_caching_enabled = self.cache_config.enable_prefix_caching

        if attn_metadata is not None:
            assert isinstance(attn_metadata, dict)
            attn_metadata = attn_metadata[self.prefix]
            assert isinstance(attn_metadata, Mamba1AttentionMetadata)
            query_start_loc_p = attn_metadata.query_start_loc_p
            state_indices_tensor = attn_metadata.state_indices_tensor
            self_kv_cache = self.kv_cache[forward_context.virtual_engine]
            conv_state = self_kv_cache[0].transpose(-1, -2)
            ssm_state = self_kv_cache[1]
            has_initial_states_p = attn_metadata.has_initial_states_p

        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(hidden_states)[0].transpose(-2, -1)
        hidden_states_BC, gate = projected_states.chunk(2, dim=-2)

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        if attn_metadata is None:
            # V1 profile run
            hidden_states_BC = hidden_states_BC.contiguous()
            return self.out_proj(hidden_states_BC.transpose(-2, -1))[0]

        num_prefill_tokens = attn_metadata.num_prefill_tokens  # token count
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_prefills = attn_metadata.num_prefills  # request count
        num_decodes = attn_metadata.num_decode_tokens  # token count (=request)
        has_prefill = num_prefill_tokens > 0
        has_decode = num_decode_tokens > 0
        num_actual_tokens = num_prefill_tokens + num_decode_tokens

        prefill_decode_split = split_batch_to_prefill_and_decode(
            hidden_states_BC,
            gate,
            state_indices_tensor,
            num_prefill_tokens,
            num_prefills,
            num_decode_tokens,
        )
        hidden_states_BC_p = prefill_decode_split.hidden_states_BC_p
        hidden_states_BC_d = prefill_decode_split.hidden_states_BC_d
        gate_p = prefill_decode_split.gate_p
        gate_d = prefill_decode_split.gate_d
        state_indices_tensor_p = prefill_decode_split.state_indices_tensor_p
        state_indices_tensor_d = prefill_decode_split.state_indices_tensor_d

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

        ssm_outputs = []

        if has_prefill:
            # 2. Convolution sequence transformation
            conv_out_p = causal_conv1d_fn(
                hidden_states_BC_p,
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_states_p,
                cache_indices=state_indices_tensor_p,
                query_start_loc=query_start_loc_p,
                block_idx_first_scheduled_token=block_idx_first_scheduled_token_p,
                block_idx_last_scheduled_token=block_idx_last_scheduled_token_p,
                initial_state_idx=block_idx_last_computed_token_p,
                num_computed_tokens=num_computed_tokens_p,
                block_size_to_align=mamba_block_size,
            )
            # 3. State Space Model sequence transformations.
            discrete_time_step_p, B_p, C_p = self._ssm_transform(
                conv_out_p.transpose(-2, -1)
            )
            time_proj_bias = self._time_proj_bias()

            # 4. Perform the recurrence y ← SSM(A, B, C, Δ)(x)
            scan_out_p = selective_scan_fn(
                conv_out_p,
                ssm_state,
                discrete_time_step_p,
                self.A,
                B_p.transpose(-2, -1),
                C_p.transpose(-2, -1),
                self.D.float(),
                gate_p,
                time_proj_bias,
                delta_softplus=True,
                cache_indices=state_indices_tensor_p,
                has_initial_state=has_initial_states_p,
                query_start_loc=query_start_loc_p,
                block_size=mamba_block_size,
                block_idx_first_scheduled_token=block_idx_first_scheduled_token_p,
                block_idx_last_scheduled_token=block_idx_last_scheduled_token_p,
                initial_state_idx=block_idx_last_computed_token_p,
            )
            ssm_outputs.append(scan_out_p)

        if has_decode:
            if prefix_caching_enabled:
                state_indices_tensor_d_input = state_indices_tensor_d.gather(
                    1, block_idx_last_computed_token_d.unsqueeze(1)
                ).squeeze(1)
                state_indices_tensor_d_output = state_indices_tensor_d.gather(
                    1, block_idx_last_scheduled_token_d.unsqueeze(1)
                ).squeeze(1)
            else:
                state_indices_tensor_d_input = state_indices_tensor_d
                state_indices_tensor_d_output = state_indices_tensor_d
            # 2. Convolution sequence transformation
            conv_out_d = causal_conv1d_update(
                hidden_states_BC_d.transpose(0, 1),
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=state_indices_tensor_d,
                block_idx_last_scheduled_token=block_idx_last_scheduled_token_d,
                initial_state_idx=block_idx_last_computed_token_d,
            ).transpose(0, 1)

            # 3. State Space Model sequence transformation.
            discrete_time_step_d, B_d, C_d = self._ssm_transform(
                conv_out_d.transpose(-2, -1)
            )
            time_proj_bias = self._time_proj_bias()

            # 4. Perform the recurrence y ← SSM(A, B, C, Δ)(x)
            scan_outputs_d = torch.empty_like(hidden_states_BC_d.transpose(0, 1))
            selective_state_update(
                ssm_state,
                conv_out_d.transpose(0, 1),
                discrete_time_step_d.transpose(0, 1),
                self.A,
                B_d,
                C_d,
                self.D,
                gate_d.transpose(0, 1),
                time_proj_bias,
                dt_softplus=True,
                state_batch_indices=state_indices_tensor_d_input,
                dst_state_batch_indices=state_indices_tensor_d_output,
                out=scan_outputs_d,
            )
            scan_outputs_d = scan_outputs_d.transpose(0, 1)

            ssm_outputs.insert(0, scan_outputs_d)

        scan_outputs_combined = (
            ssm_outputs[0] if len(ssm_outputs) == 1 else torch.cat(ssm_outputs, dim=-1)
        )

        # 5. Final output projection
        if self.is_lora_enabled:  # Lora kernel requires contiguous tensor.
            scan_outputs_combined = scan_outputs_combined.transpose(-2, -1).contiguous()
            out = self.out_proj(scan_outputs_combined)[0]
        else:
            out = self.out_proj(scan_outputs_combined.transpose(-2, -1))[0]

        output[:num_actual_tokens] = out

    def get_state_dtype(self) -> tuple[torch.dtype]:
        assert self.model_config is not None
        assert self.cache_config is not None
        return MambaStateDtypeCalculator.mamba1_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.mamba1_state_shape(
            tp_world_size=get_tensor_model_parallel_world_size(),
            intermediate_size=self.intermediate_size,
            state_size=self.ssm_state_size,
            conv_kernel=self.conv_kernel_size,
        )

    @property
    def mamba_type(self) -> str:
        return "mamba1"

    def _time_proj_bias(self) -> torch.Tensor | None:
        if hasattr(self.dt_proj, "bias") and self.dt_proj.bias is not None:
            return self.dt_proj.bias.float()
        return None


class PrefillDecodeSplit(NamedTuple):
    hidden_states_BC_p: torch.Tensor
    hidden_states_BC_d: torch.Tensor
    gate_p: torch.Tensor
    gate_d: torch.Tensor
    state_indices_tensor_p: torch.Tensor
    state_indices_tensor_d: torch.Tensor


def split_batch_to_prefill_and_decode(
    hidden_states_BC: torch.Tensor,
    gate: torch.Tensor,
    state_indices_tensor: torch.Tensor,
    num_prefill_tokens: int,
    num_prefills: int,
    num_decode_tokens: int,
) -> PrefillDecodeSplit:
    num_actual_tokens = num_prefill_tokens + num_decode_tokens

    # In v1, decode tokens come first, then prefill tokens.
    hidden_states_BC_d, hidden_states_BC_p = torch.split(
        hidden_states_BC[..., :num_actual_tokens],
        [num_decode_tokens, num_prefill_tokens],
        dim=-1,
    )
    gate_d, gate_p = torch.split(
        gate[..., :num_actual_tokens], [num_decode_tokens, num_prefill_tokens], dim=-1
    )

    # num_decode_tokens accounts for CUDA graph padding when applicable
    state_indices_tensor_d, state_indices_tensor_p = torch.split(
        state_indices_tensor[: num_decode_tokens + num_prefills],
        [num_decode_tokens, num_prefills],
        dim=0,
    )

    return PrefillDecodeSplit(
        hidden_states_BC_p=hidden_states_BC_p,
        hidden_states_BC_d=hidden_states_BC_d,
        gate_p=gate_p,
        gate_d=gate_d,
        state_indices_tensor_p=state_indices_tensor_p,
        state_indices_tensor_d=state_indices_tensor_d,
    )


def mamba_mixer(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self.forward_cuda(hidden_states=hidden_states, output=output)


def mamba_mixer_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="mamba_mixer",
    op_func=mamba_mixer,
    mutates_args=["output"],
    fake_impl=mamba_mixer_fake,
)
