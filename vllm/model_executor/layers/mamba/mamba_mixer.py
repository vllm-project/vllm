import torch
from torch import nn
from torch.nn.parameter import Parameter

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn, causal_conv1d_update)
from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    selective_scan_fn, selective_state_update)
from vllm.model_executor.models.mamba_cache import MambaCacheParams
from vllm.model_executor.utils import set_weight_attrs


# Adapted from transformers.models.mamba.modeling_mamba.MambaMixer
@CustomOp.register("mamba_mixer")
class MambaMixer(CustomOp):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute
    the `contextualized_states`. A, D are input independent
    (see Mamba paper [1] Section 3.5.2 "Interpretation of A"
    for why A isn't selective) ∆, B, C are input-dependent
    (this is a key difference between Mamba and the linear time
    invariant S4, and is why Mamba is called
    **selective** state spaces)
    """

    def __init__(self,
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
                 activation="silu"):
        super().__init__()
        self.time_step_rank = time_step_rank
        self.ssm_state_size = ssm_state_size
        self.use_rms_norm = use_rms_norm
        self.activation = activation

        self.conv1d = ColumnParallelLinear(
            input_size=conv_kernel_size,
            output_size=intermediate_size,
            bias=use_conv_bias,
        )
        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `set_weight_attrs`
        # doesn't allow to override it
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        self.in_proj = MergedColumnParallelLinear(hidden_size,
                                                  [intermediate_size] * 2,
                                                  bias=use_bias)
        # selective projection used to make dt, B and C input dependent
        self.x_proj = RowParallelLinear(
            intermediate_size,
            time_step_rank + ssm_state_size * 2,
            bias=False,
        )
        # time step projection (discretization) -
        # In the forward we need to apply dt_proj without the bias,
        # as the bias is added in the selective scan kernel.
        self.dt_proj = ColumnParallelLinear(time_step_rank,
                                            intermediate_size,
                                            bias=True,
                                            skip_bias_add=True)

        def weight_loader(param: Parameter, loaded_weight: torch.Tensor):
            tp_rank = get_tensor_model_parallel_rank()
            tp_size = get_tensor_model_parallel_world_size()
            param.data.copy_(
                loaded_weight.data.split(loaded_weight.shape[0] // tp_size,
                                         dim=0)[tp_rank])

        def A_weight_loader(param: Parameter, loaded_weight: torch.Tensor):
            weight_loader(param, -torch.exp(loaded_weight.float()))

        tp_size = get_tensor_model_parallel_world_size()
        self.A = nn.Parameter(
            torch.empty(
                intermediate_size // tp_size,
                ssm_state_size,
                dtype=torch.float32,
            ))
        self.D = nn.Parameter(torch.ones(intermediate_size // tp_size))

        set_weight_attrs(self.D, {"weight_loader": weight_loader})
        set_weight_attrs(self.A, {"weight_loader": A_weight_loader})

        self.out_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=use_bias,
            input_is_parallel=True,
        )

        self.dt_layernorm = RMSNorm(
            time_step_rank,
            eps=rms_norm_eps,
            has_weight=rms_norm_has_weight,
        ) if use_rms_norm else None

        self.b_layernorm = RMSNorm(
            ssm_state_size,
            eps=rms_norm_eps,
            has_weight=rms_norm_has_weight,
        ) if use_rms_norm else None

        self.c_layernorm = RMSNorm(
            ssm_state_size,
            eps=rms_norm_eps,
            has_weight=rms_norm_has_weight,
        ) if use_rms_norm else None

    def forward_native(self, hidden_states: torch.Tensor,
                       attn_metadata: AttentionMetadata,
                       conv_state: torch.Tensor, ssm_state: torch.Tensor):
        pass

    def forward_cuda(self, hidden_states: torch.Tensor,
                     attn_metadata: AttentionMetadata,
                     mamba_cache_params: MambaCacheParams):

        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(hidden_states)[0].transpose(-2, -1)
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
