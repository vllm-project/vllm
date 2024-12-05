import torch
from torch import nn
from torch.nn.parameter import Parameter

# Added by the IBM Team, 2024

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)

from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn, causal_conv1d_update)
from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    selective_state_update)
from vllm.model_executor.layers.mamba.ops.ssd_combined import (
    mamba_chunk_scan_combined)
from vllm.model_executor.models.mamba_cache import MambaCacheParams
from vllm.model_executor.utils import set_weight_attrs


from typing import Tuple, Union, Optional
from vllm.model_executor.custom_op import CustomOp

# Adapted from transformers.models.mamba2.modeling_mamba2.MambaRMSNormGated
@CustomOp.register("mixer2_gated_rms_norm")
class Mixer2RMSNormGated(CustomOp):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward_native(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
    ):
        pass

    def forward_cuda(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        from vllm import _custom_ops as ops

        # cast gate to float32 before silu
        out = torch.empty_like(x)
        y = x * nn.functional.silu(gate.to(torch.float32))
        ops.rms_norm(
            out,
            y.to(x.dtype),
            self.weight.data,
            self.variance_epsilon,
        )
        return out

# Adapted from transformers.models.mamba.modeling_mamba.MambaMixer
@CustomOp.register("mamba_mixer2") 
class MambaMixer2(CustomOp):
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
                 n_groups: int = 1,
                 num_heads: int = 128,
                 head_dim: int = 64,
                 rms_norm_eps: float = 1e-5,
                 activation="silu",
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.time_step_rank = time_step_rank
        self.ssm_state_size = ssm_state_size
        self.use_rms_norm = use_rms_norm
        self.activation = activation

        self.chunk_size = 256
        self.intermediate_size = intermediate_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.n_groups = n_groups
        self.conv_dim = intermediate_size + 2 * n_groups * ssm_state_size
        self.conv1d = ColumnParallelLinear(
            input_size=conv_kernel_size,
            output_size=self.conv_dim,
            bias=use_conv_bias,
            quant_config=None,
        )
        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `set_weight_attrs`
        # doesn't allow to override it
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        self.in_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=intermediate_size + self.conv_dim + self.num_heads,
            bias=use_bias,
            quant_config=quant_config)

        # unlike mamba_mixer.py (v1), we do not TP the A matrix as it is 
        # already quite small. 
        # - same for dt_bias and D

        def A_weight_loader(param: Parameter, loaded_weight: torch.Tensor):
            param.data.copy_(-torch.exp(loaded_weight.float()))

        self.A = nn.Parameter(
            torch.empty(
                num_heads,
                dtype=torch.float32,
            ))
        set_weight_attrs(self.A, {"weight_loader": A_weight_loader})

        self.dt_bias = nn.Parameter(torch.ones(num_heads))
        self.D = nn.Parameter(torch.ones(num_heads))

        self.out_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=use_bias,
            input_is_parallel=True,
            quant_config=quant_config)

        self.norm = Mixer2RMSNormGated(
            intermediate_size, eps=rms_norm_eps
        )

    def forward_native(self, hidden_states: torch.Tensor,
                       attn_metadata: AttentionMetadata,
                       conv_state: torch.Tensor, ssm_state: torch.Tensor):
        pass

    def forward_cuda(self, hidden_states: torch.Tensor,
                     attn_metadata: AttentionMetadata,
                     mamba_cache_params: MambaCacheParams):


        seq_len, _ = hidden_states.shape
        groups_time_state_size = self.n_groups * self.ssm_state_size

        # - doing it differently from mixer v1; little confused with its logic
        # - we need to do is to detect if there is any prefill; if there are 
        #   no prefils, then each example will be coming in one sample at a time
        # - on the other hand v1 checks for "query_start_loc" and "context_lens_tensor"
        #   however we have noticed that, even when the samples are coming in
        #   one at a time, they are still non-NO.e
        #   * "query_start_loc" = [0, 1, ..]
        #   * "context_lens_tensor" = [8, ...]
        has_prefill = attn_metadata.num_prefills > 0 

        # 1. Gated MLP's linear projection
        projected_states, _ = self.in_proj(hidden_states)
        gate, hidden_states_B_C, dt = torch.split(
            projected_states,
            [self.intermediate_size, self.conv_dim, self.num_heads],
            dim=-1,
        )

        # 2. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0),
                                               self.conv1d.weight.size(2))

        if has_prefill:
            # |---------- N-1 iteration --------|
            # |---------------- N iteration ---------------------|
            # |- tokenA -|......................|-- newTokens ---|
            # |---------- context_len ----------|
            # |-------------------- seq_len ---------------------|
            #                                   |-- query_len ---|

            # - "cache_indices" upates the conv_state cache in positions
            #   pointed to by "mamba_cache_params.state_indices_tensor"
            hidden_states_B_C = causal_conv1d_fn(
                hidden_states_B_C.transpose(0, 1),
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=mamba_cache_params.conv_state,
                has_initial_state=attn_metadata.context_lens_tensor > 0,
                cache_indices=mamba_cache_params.state_indices_tensor,
                query_start_loc=attn_metadata.query_start_loc
            ).transpose(0, 1)[:seq_len]
        else:
            hidden_states_B_C = causal_conv1d_update(
                hidden_states_B_C,
                mamba_cache_params.conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=mamba_cache_params.state_indices_tensor
            )

        # - get hidden_states, B and C after depthwise convolution.
        hidden_states, B, C = torch.split(
            hidden_states_B_C,
            [self.intermediate_size, groups_time_state_size, groups_time_state_size],
            dim=-1,
        )

        # 3. State Space Model sequence transformation
        if has_prefill:
            
            # FIXME: we are having problems using mamba_chunk_scan_combined
            # with chunked prefill. This is because there is no
            # initial_states requires initial_states.shape[0] to match
            # the batch size, but cu_seqlens requires batch_size = 1.
            # Therefore as of now, initial_states and cu_seqlens are 
            # mutually exclusive.

            initial_states = None
            # if any(attn_metadata.context_lens_tensor > 0):
            #     initial_states = mamba_cache_params.ssm_state[
            #         mamba_cache_params.state_indices_tensor
            #     ]

            scan_output, varlen_state = mamba_chunk_scan_combined(
                hidden_states.view(1, seq_len, -1, self.head_dim),
                dt.unsqueeze(0),
                self.A,
                B.view(1, seq_len, self.n_groups, -1),
                C.view(1, seq_len, self.n_groups, -1),
                chunk_size=self.chunk_size,
                D=self.D,
                z=None,
                dt_bias=self.dt_bias,
                seq_idx=attn_metadata.seq_idx.unsqueeze(0),
                cu_seqlens=attn_metadata.query_start_loc,
                initial_states=initial_states,
                return_varlen_states=True,
                return_final_states=False,
                dt_softplus=True,
                dt_limit=(0.0, float("inf")),
            )

            # update ssm states
            # - varlen state is a (batch, nheads, headdim, dstate) tensor
            for i, idx in enumerate(mamba_cache_params.state_indices_tensor):
                mamba_cache_params.ssm_state[idx].copy_(varlen_state[i])

            # - reshape
            hidden_states = scan_output.view(seq_len, -1)
        else:

            # NOTE: can be optimized? 
            A = self.A[:, None, ...][:, :, None].expand(-1, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            dt = dt[:, :, None].expand(-1, -1, self.head_dim)
            dt_bias = self.dt_bias[:, None, ...].expand(-1, self.head_dim)
            D = self.D[:, None, ...].expand(-1, self.head_dim)
            B = B.view(-1, self.n_groups, B.shape[1] // self.n_groups)
            C = C.view(-1, self.n_groups, C.shape[1] // self.n_groups)
            hidden_states_reshaped = hidden_states.view(-1, self.num_heads, self.head_dim)

            # - the hidden is reshaped into number of current batches
            # - in this case there is no more prefil, so the batches gen
            #   1 token at a time
            # - thus hidden will be (bs, num_heads, head_dim)
            # - mamba_cache_params.ssm_state's slots will be selected
            #   using "mamba_cache_params.state_indices_tensor", just as
            #   above in the prefill case

            hidden_states = selective_state_update(
                mamba_cache_params.ssm_state,
                hidden_states_reshaped,
                dt,
                A, 
                B,
                C,
                D, 
                z=None,
                dt_bias=dt_bias,
                dt_softplus=True,
                state_batch_indices=mamba_cache_params.state_indices_tensor,
            )
            hidden_states = hidden_states.view(-1, self.num_heads * self.head_dim)

        # # 4. gated MLP
        hidden_states = self.norm(hidden_states, gate)

        # # 5. Final linear projection
        out, _ = self.out_proj(hidden_states)
        return out 