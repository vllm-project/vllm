# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.distributed import (divide, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce)
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.mamba.mamba2_metadata import Mamba2Metadata
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn, causal_conv1d_update)
from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    selective_state_update)
from vllm.model_executor.layers.mamba.ops.ssd_combined import (
    mamba_chunk_scan_combined)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import (
    LoaderFunction, composed_weight_loader, sharded_weight_loader)
from vllm.model_executor.models.mamba_cache import MambaCacheParams
from vllm.model_executor.utils import set_weight_attrs

# Added by the IBM Team, 2024


# Adapted from transformers.models.mamba2.modeling_mamba2.MambaRMSNormGated
@CustomOp.register("mixer2_gated_rms_norm")
class Mixer2RMSNormGated(CustomOp):

    def __init__(self, full_hidden_size, full_n_groups, eps=1e-6):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.full_hidden_size = full_hidden_size
        self.group_size = full_hidden_size // full_n_groups
        self.per_rank_hidden_size = full_hidden_size // self.tp_size
        self.n_groups = full_hidden_size // self.group_size

        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(self.per_rank_hidden_size))
        set_weight_attrs(self.weight,
                         {"weight_loader": sharded_weight_loader(0)})
        assert self.full_hidden_size % self.tp_size== 0,\
            "Tensor parallel world size must divide hidden size."

    def forward_native(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
    ):
        # Three tensor-parallel cases:
        #   1. n_groups is 1
        #      In this case we parallelize along the reduction dim.
        #      Each rank computes a local sum of squares followed by AllReduce
        #   2. tp_size divides n_groups
        #      Each rank only reduces within its local group(s).
        #      No collective ops necessary.
        #   3. The general case can be pretty complicated so we AllGather
        #      the input and then redundantly compute the RMSNorm.
        input_dtype = x.dtype
        x = x * nn.functional.silu(gate.to(torch.float32))

        if self.n_groups == 1:
            if self.tp_size > 1:
                # Compute local sum and then reduce to obtain global sum
                local_sums = x.pow(2).sum(dim=-1, keepdim=True)
                global_sums = tensor_model_parallel_all_reduce(local_sums)
                # Calculate the variance
                count = self.tp_size * x.shape[-1]
                variance = (global_sums / count)

            else:
                variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.variance_epsilon)
        else:
            redundant_tp: bool = self.n_groups % self.tp_size != 0
            if redundant_tp:
                # To handle the general case, redundantly apply the variance
                x = tensor_model_parallel_all_gather(x, -1)

            *prefix_dims, hidden_dim = x.shape
            group_count = hidden_dim // self.group_size
            x_grouped = x.view(*prefix_dims, group_count, self.group_size)
            variance = x_grouped.pow(2).mean(-1, keepdim=True)
            x_grouped = x_grouped * torch.rsqrt(variance +
                                                self.variance_epsilon)
            x = x_grouped.view(*prefix_dims, hidden_dim)

            if redundant_tp:
                start = self.per_rank_hidden_size * self.tp_rank
                end = start + self.per_rank_hidden_size
                x = x[..., start:end]

        return self.weight * x.to(input_dtype)

    def forward_cuda(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        if self.tp_size > 1 or self.n_groups != 1:
            return self.forward_native(x, gate)

        from vllm import _custom_ops as ops

        # cast x and gate to float32 before silu
        out = torch.empty_like(x)
        y = x * nn.functional.silu(gate.to(torch.float32))
        ops.rms_norm(
            out,
            y.to(x.dtype),
            self.weight.data,
            self.variance_epsilon,
        )
        return out


def extra_groups_for_head_shards(ngroups: int, tp_size: int):
    """Compute the increase in group numbers to account for 
    replication in order to accompany the head shards."""

    # in the case ngoups % tp_size == 0, this will be zero
    if ngroups % tp_size == 0:
        return 0

    # for n_groups == 1, this is exactly tp_size - n_groups
    return tp_size - ngroups


def mamba_v2_sharded_weight_loader(
    shard_spec: List[Tuple[int, int, float]],
    tp_size: int,
    tp_rank: int,
) -> LoaderFunction:
    """Create a weight loader for mamba v2. This ensures that the projections 
    are correctly sharded so that they can be split into x, B, C. It also 
    ensures the the all the groups corresponding to a head shard is placed 
    together with it.
    """

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:

        # - track boundary of (sharded) param, and loaded_weight, respectively
        boundary, loaded_boundary = 0, 0

        # - iterate over the shard specs
        for full_dim, extra, duplicate_groups in shard_spec:
            # - full dim is the model dim (before TP).
            # - extra > 0, means there is expected overall increase
            #   of dimensions. This is so because of replication.
            # - ratio is used map the tp_rank to the actual shard
            #   rank. This is useful when there is replication of
            #   groups to accompany head shards.

            # - size of the loaded shard
            shard_size = full_dim // tp_size

            # - compute the rank into the loaded shard.
            # - if there is replication, different TP shards will
            #   take from the same rank.
            # NOTE: currently we only support duplication
            # in the case where num_groups == 1
            rank = 0 if duplicate_groups else tp_rank

            # - leftmost boundary index into loaded weight.
            loaded_skip = rank * shard_size
            loaded_start_idx = loaded_boundary + loaded_skip

            # - take these many dims from the loaded weight.
            take = min(shard_size, full_dim - extra - loaded_skip)

            # - always shard on dim 0
            # - the ignore is for a mundane mypy error as it does not
            #   seem to handle slices well.
            # https://github.com/python/mypy/issues/2410
            param.data[
                boundary:(boundary + take),  # type: ignore[misc]
                ...] = loaded_weight[loaded_start_idx:(  # type: ignore[misc]
                    loaded_start_idx + take)]  # type: ignore[misc]

            # move indexing boundaries
            boundary += shard_size
            loaded_boundary += (full_dim - extra)

    return loader


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
                 use_conv_bias: bool,
                 use_bias: bool,
                 n_groups: int = 1,
                 num_heads: int = 128,
                 head_dim: int = 64,
                 rms_norm_eps: float = 1e-5,
                 activation="silu",
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()

        # For TP, the sharding plan is as follows:
        # - for the conv modules, since
        #   conv_dim = intermediate_size * 2 * n_groups * ssm_state_size,
        #   we shard intermediate_size and n_groups
        # - since intermediate_size = n_heads * head_dim, sharding on
        #   intermediate_size is achieved by sharding on n_heads.
        # - IF, world_size divides groups, then sharding
        #   (n_groups / world_size, n_heads / world_size)
        #   also maintains the invariant n_heads % n_groups == 0
        # - HOWEVER IF, world_size DOES NOT divide groups, then we need
        #   to allocate extra space in the shard, such that groups
        #   may be replicated to follow the head shard.
        # - NOTE: currently for the world size DOES NOT divide groups
        #   case, we only support the case when n_groups == 1
        self.tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        assert num_heads % self.tp_size == 0, \
            "Tensor parallel world size must divide num heads."

        assert (n_groups % self.tp_size) == 0 or n_groups == 1, \
            (
                "If tensor parallel world size does not divide num_heads, "
                "then num_groups must equal 1."
            )

        assert self.tp_size == 1 or quant_config is None, \
            "Tensor parallel currently not supported for quantized models."

        self.ssm_state_size = ssm_state_size
        self.activation = activation

        self.intermediate_size = intermediate_size
        self.head_dim = head_dim
        self.num_heads = num_heads

        self.n_groups = n_groups
        if n_groups % self.tp_size != 0:
            # - for TP we shard conv_dim by sharding on n_groups,
            # - but if n_groups cannot divide tp_size, we need to
            #   extend some extra groups
            self.n_groups = n_groups + extra_groups_for_head_shards(
                n_groups, self.tp_size)

        self.conv_dim = (intermediate_size +
                         2 * self.n_groups * ssm_state_size)
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

        self.in_proj = ColumnParallelLinear(input_size=hidden_size,
                                            output_size=intermediate_size +
                                            self.conv_dim + self.num_heads,
                                            bias=use_bias,
                                            quant_config=quant_config)

        # - because in_proj is a concatenation of 3 weights, we
        #   need to interleave them before sharding
        # - use the custom weight loader mamba_v2_sharded_weight_loader
        #   for conv1d.bias, covn1d.weight and in_proj.weight
        # - need to set these settings, to assign the groups to the head shards
        group_shard_settings = (
            self.n_groups * self.ssm_state_size,  # expected model size
            (self.n_groups - n_groups) *
            self.ssm_state_size,  # extra dims assigned
            n_groups == 1,  # if there was only one group
        )
        intermediate_settings = (intermediate_size, 0, False)
        head_setings = (self.num_heads, 0, False)

        # - the weight already has a "weight_loader" attribute
        #   which set_weight_attrs will raise if we do not
        #   delete before trying to override it
        # - ditto for the otther two weights below
        delattr(self.conv1d.bias, "weight_loader")
        set_weight_attrs(
            self.conv1d.bias, {
                "weight_loader":
                mamba_v2_sharded_weight_loader(
                    [
                        intermediate_settings,
                        group_shard_settings,
                        group_shard_settings,
                    ],
                    self.tp_size,
                    tp_rank,
                )
            })

        delattr(self.conv1d.weight, "weight_loader")
        set_weight_attrs(
            self.conv1d.weight, {
                "weight_loader":
                mamba_v2_sharded_weight_loader([
                    intermediate_settings,
                    group_shard_settings,
                    group_shard_settings,
                ], self.tp_size, tp_rank)
            })

        if quant_config is None:
            # - quant layers do not have a weight loader
            delattr(self.in_proj.weight, "weight_loader")
            set_weight_attrs(
                self.in_proj.weight,
                {
                    "weight_loader":
                    mamba_v2_sharded_weight_loader(
                        [
                            intermediate_settings,  # for gate
                            intermediate_settings,
                            group_shard_settings,
                            group_shard_settings,
                            head_setings,  # for dt
                        ],
                        self.tp_size,
                        tp_rank)
                })

        # - these are TPed by heads to reduce the size of the
        #   temporal shape
        self.A = nn.Parameter(
            torch.empty(
                divide(num_heads, self.tp_size),
                dtype=torch.float32,
            ))
        self.D = nn.Parameter(torch.ones(num_heads // self.tp_size))
        self.dt_bias = nn.Parameter(torch.ones(num_heads // self.tp_size))

        set_weight_attrs(self.D, {"weight_loader": sharded_weight_loader(0)})
        a_weight_loader = composed_weight_loader(
            sharded_weight_loader(0), lambda x: -torch.exp(x.float()))
        set_weight_attrs(self.A, {"weight_loader": a_weight_loader})
        set_weight_attrs(self.dt_bias,
                         {"weight_loader": sharded_weight_loader(0)})

        self.out_proj = RowParallelLinear(intermediate_size,
                                          hidden_size,
                                          bias=use_bias,
                                          input_is_parallel=True,
                                          quant_config=quant_config)

        self.norm = Mixer2RMSNormGated(intermediate_size,
                                       n_groups,
                                       eps=rms_norm_eps)

    def forward_native(self, hidden_states: torch.Tensor,
                       conv_state: torch.Tensor, ssm_state: torch.Tensor):
        pass

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        mamba_cache_params: MambaCacheParams,
        mamba2_metadata: Mamba2Metadata,
    ):
        # mamba2_metadata contains metadata necessary for the mamba2 triton
        # kernels to operate in continuous batching and in chunked prefill
        # modes; they are computed at top-level model forward since they
        # are the same and reused for all mamba layers in the same iteration
        attn_metadata: AttentionMetadata = get_forward_context().attn_metadata

        seq_len, _ = hidden_states.shape
        groups_time_state_size = self.n_groups * self.ssm_state_size

        # 1. Gated MLP's linear projection
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

        # 2. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0),
                                               self.conv1d.weight.size(2))

        if mamba2_metadata.has_prefill:
            # |---------- N-1 iteration --------|
            # |---------------- N iteration ---------------------|
            # |- tokenA -|......................|-- newTokens ---|
            # |---------- context_len ----------|
            # |-------------------- seq_len ---------------------|
            #                                   |-- query_len ---|

            # - "cache_indices" updates the conv_state cache in positions
            #   pointed to by "mamba_cache_params.state_indices_tensor"
            hidden_states_B_C = causal_conv1d_fn(
                hidden_states_B_C.transpose(0, 1),
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=mamba_cache_params.conv_state,
                has_initial_state=mamba2_metadata.has_initial_states,
                cache_indices=mamba_cache_params.state_indices_tensor,
                query_start_loc=attn_metadata.query_start_loc).transpose(
                    0, 1)[:seq_len]

            # TODO: Why is this needed?
            hidden_states_B_C = hidden_states_B_C.contiguous()
        else:
            hidden_states_B_C = causal_conv1d_update(
                hidden_states_B_C,
                mamba_cache_params.conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=mamba_cache_params.state_indices_tensor)

        # - get hidden_states, B and C after depthwise convolution.
        hidden_states, B, C = torch.split(
            hidden_states_B_C,
            [
                self.intermediate_size // self.tp_size,
                groups_time_state_size // self.tp_size,
                groups_time_state_size // self.tp_size,
            ],
            dim=-1,
        )

        # 3. State Space Model sequence transformation
        if mamba2_metadata.has_prefill:
            initial_states = None
            if (mamba2_metadata.has_initial_states is not None
                    and mamba2_metadata.prep_initial_states):
                # making a copy of the states
                initial_states = torch.where(
                    mamba2_metadata.has_initial_states[:, None, None, None],
                    mamba_cache_params.ssm_state[
                        mamba_cache_params.state_indices_tensor], 0)

            scan_output, varlen_state = mamba_chunk_scan_combined(
                hidden_states.view(1, seq_len, self.num_heads // self.tp_size,
                                   self.head_dim),
                dt.unsqueeze(0),
                self.A,
                B.view(1, seq_len, self.n_groups // self.tp_size, -1),
                C.view(1, seq_len, self.n_groups // self.tp_size, -1),
                chunk_size=mamba2_metadata.chunk_size,
                D=self.D,
                z=None,
                dt_bias=self.dt_bias,
                seq_idx=mamba2_metadata.seq_idx,
                chunk_indices=mamba2_metadata.chunk_indices,
                chunk_offsets=mamba2_metadata.chunk_offsets,
                cu_seqlens=attn_metadata.query_start_loc,
                initial_states=initial_states,
                return_varlen_states=True,
                return_final_states=False,
                dt_softplus=True,
                dt_limit=(0.0, float("inf")),
            )

            # update ssm states
            # - varlen state is a (batch, nheads, headdim, dstate) tensor
            mamba_cache_params.ssm_state[
                mamba_cache_params.state_indices_tensor] = varlen_state

            # - reshape
            hidden_states = scan_output.view(seq_len, -1)
        else:

            n_groups = self.n_groups // self.tp_size
            A = self.A[:, None, ...][:, :, None].expand(
                -1, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            dt = dt[:, :, None].expand(-1, -1, self.head_dim)
            dt_bias = self.dt_bias[:, None, ...].expand(-1, self.head_dim)
            D = self.D[:, None, ...].expand(-1, self.head_dim)
            B = B.view(-1, n_groups, B.shape[1] // n_groups)
            C = C.view(-1, n_groups, C.shape[1] // n_groups)
            hidden_states_reshaped = hidden_states.view(
                -1, self.num_heads // self.tp_size, self.head_dim)

            # - the hidden is reshaped into number of current batches
            # - in this case there is no more prefill, so the batches gen
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
            hidden_states = hidden_states.view(
                -1, (self.num_heads // self.tp_size) * self.head_dim)

        # # 4. gated MLP
        hidden_states = self.norm(hidden_states, gate)

        # # 5. Final linear projection
        out, _ = self.out_proj(hidden_states)
        return out
