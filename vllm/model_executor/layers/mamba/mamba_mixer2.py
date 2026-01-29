# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
from torch import nn

import vllm.envs as envs
from vllm.config import CacheConfig, ModelConfig, get_current_vllm_config
from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
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
from vllm.model_executor.layers.mamba.ops.layernorm_gated import rms_norm_gated
from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update
from vllm.model_executor.layers.mamba.ops.ssd_combined import (
    mamba_chunk_scan_combined_varlen,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import (
    LoaderFunction,
    composed_weight_loader,
    sharded_weight_loader,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import (
    current_stream,
    direct_register_custom_op,
    mamba_gate_stream,
    mamba_prefill_decode_stream,
    maybe_execute_in_parallel,
)
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadata

# Added by the IBM Team, 2024


# Adapted from transformers.models.mamba2.modeling_mamba2.MambaRMSNormGated
# --8<-- [start:mixer2_gated_rms_norm]
@CustomOp.register("mixer2_gated_rms_norm")
class Mixer2RMSNormGated(CustomOp):
    # --8<-- [end:mixer2_gated_rms_norm]

    def __init__(
        self,
        full_hidden_size: int,
        full_n_groups: int,
        use_rms_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.full_hidden_size = full_hidden_size
        self.group_size = full_hidden_size // full_n_groups
        self.per_rank_hidden_size = full_hidden_size // self.tp_size
        self.n_groups = full_hidden_size // self.group_size

        self.variance_epsilon = eps
        self.use_rms_norm = use_rms_norm
        if self.use_rms_norm:
            # Register norm weight only if we're actually applying RMSNorm
            self.weight = nn.Parameter(torch.ones(self.per_rank_hidden_size))
            set_weight_attrs(self.weight, {"weight_loader": sharded_weight_loader(0)})
        else:
            # Avoid checkpoint mismatch by skipping unused parameter
            self.register_parameter("weight", None)
        assert self.full_hidden_size % self.tp_size == 0, (
            "Tensor parallel world size must divide hidden size."
        )

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
        if not self.use_rms_norm:
            return x.to(input_dtype)

        if self.n_groups == 1:
            if self.tp_size > 1:
                # Compute local sum and then reduce to obtain global sum
                local_sums = x.pow(2).sum(dim=-1, keepdim=True)
                global_sums = tensor_model_parallel_all_reduce(local_sums)
                # Calculate the variance
                count = self.tp_size * x.shape[-1]
                variance = global_sums / count

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
            x_grouped = x_grouped * torch.rsqrt(variance + self.variance_epsilon)
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
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        input_dtype = x.dtype
        if not self.use_rms_norm:
            # Keep gate in float32 for numerical stability during silu
            return x * nn.functional.silu(gate.to(torch.float32)).to(input_dtype)

        if ((self.n_groups % self.tp_size) != 0) or self.n_groups != 1:
            return self.forward_native(x, gate)

        return rms_norm_gated(
            x,
            self.weight.data,
            bias=None,
            z=gate,
            eps=self.variance_epsilon,
            norm_before_gate=False,
        )


def mamba_v2_sharded_weight_loader(
    shard_spec: list[tuple[int, int, float]],
    tp_size: int,
    tp_rank: int,
) -> LoaderFunction:
    """Create a weight loader for mamba v2. This ensures that the projections
    are correctly sharded so that they can be split into x, B, C. It also
    ensures that all the groups corresponding to a head shard is placed
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
                boundary : (boundary + take), ...  # type: ignore[misc]
            ] = loaded_weight[
                loaded_start_idx : (
                    loaded_start_idx + take
                )  # type: ignore[misc]
            ]  # type: ignore[misc]

            # move indexing boundaries
            boundary += shard_size
            loaded_boundary += full_dim - extra

    return loader


# Adapted from transformers.models.mamba.modeling_mamba.MambaMixer
# --8<-- [start:mamba_mixer2]
@CustomOp.register("mamba_mixer2")
class MambaMixer2(MambaBase, CustomOp):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute
    the `contextualized_states`. A, D are input independent
    (see Mamba paper [1] Section 3.5.2 "Interpretation of A"
    for why A isn't selective) ∆, B, C are input-dependent
    (this is a key difference between Mamba and the linear time
    invariant S4, and is why Mamba is called
    **selective** state spaces)
    """

    # --8<-- [end:mamba_mixer2]

    def __init__(
        self,
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
        activation: str = "silu",
        use_rms_norm: bool = True,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
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

        assert num_heads % self.tp_size == 0, (
            "Tensor parallel world size must divide num heads."
        )

        assert (n_groups % self.tp_size) == 0 or n_groups == 1, (
            "If tensor parallel world size does not divide num_groups, "
            "then num_groups must equal 1."
        )

        assert (
            (n_groups % self.tp_size == 0) or self.tp_size == 1 or quant_config is None
        ), (
            "Tensor parallel currently supported for quantized models only "
            "if tensor parallel world size divides num groups."
        )

        self.ssm_state_size = ssm_state_size
        self.conv_kernel_size = conv_kernel_size
        self.activation = activation

        self.intermediate_size = intermediate_size
        self.head_dim = head_dim
        self.num_heads = num_heads

        self.n_groups = n_groups
        if n_groups % self.tp_size != 0:
            # - for TP we shard conv_dim by sharding on n_groups,
            # - but if n_groups cannot divide tp_size, we need to
            #   extend some extra groups
            groups = MambaStateShapeCalculator.extra_groups_for_head_shards(
                n_groups, self.tp_size
            )
            self.n_groups = n_groups + groups

        self.groups_ssm_state_size = self.n_groups * self.ssm_state_size
        self.conv_dim = intermediate_size + 2 * self.groups_ssm_state_size
        # Use fused in_proj when multi-stream is disabled to avoid
        # extra GEMM and kernel-launch overhead.
        self._enable_mamba_multi_stream = not envs.VLLM_DISABLE_MAMBA_MULTI_STREAM
        self._split_gate_proj = self._enable_mamba_multi_stream

        if n_groups % self.tp_size == 0:
            self.conv1d = MergedColumnParallelLinear(
                input_size=conv_kernel_size,
                output_sizes=[
                    intermediate_size,
                    self.groups_ssm_state_size,
                    self.groups_ssm_state_size,
                ],
                bias=use_conv_bias,
                quant_config=None,
                prefix=f"{prefix}.conv1d",
            )

            if self._split_gate_proj:
                # Gate projection is split out so it can run independently
                # and overlap with the conv+SSM path on a separate CUDA stream.
                self.gate_proj = ColumnParallelLinear(
                    input_size=hidden_size,
                    output_size=intermediate_size,
                    bias=use_bias,
                    quant_config=quant_config,
                    prefix=f"{prefix}.gate_proj",
                    return_bias=False,
                )
                in_proj_output_sizes = [
                    intermediate_size,
                    self.groups_ssm_state_size,
                    self.groups_ssm_state_size,
                    self.num_heads,
                ]
            else:
                self.gate_proj = None
                in_proj_output_sizes = [
                    intermediate_size,
                    intermediate_size,
                    self.groups_ssm_state_size,
                    self.groups_ssm_state_size,
                    self.num_heads,
                ]

            self.in_proj = MergedColumnParallelLinear(
                input_size=hidden_size,
                output_sizes=in_proj_output_sizes,
                bias=use_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj",
            )
        else:
            # This is the n_groups == 1 case,
            # where we need to duplicate groups if TP>1.

            self.conv1d = ColumnParallelLinear(
                input_size=conv_kernel_size,
                output_size=self.conv_dim,
                bias=use_conv_bias,
                quant_config=None,
                prefix=f"{prefix}.conv1d",
            )

            if self._split_gate_proj:
                # Gate projection is split out so it can run independently
                # and overlap with the conv+SSM path on a separate CUDA stream.
                self.gate_proj = ColumnParallelLinear(
                    input_size=hidden_size,
                    output_size=intermediate_size,
                    bias=use_bias,
                    quant_config=quant_config,
                    prefix=f"{prefix}.gate_proj",
                    return_bias=False,
                )
                in_proj_output_size = self.conv_dim + self.num_heads
            else:
                self.gate_proj = None
                in_proj_output_size = (
                    self.intermediate_size + self.conv_dim + self.num_heads
                )

            self.in_proj = ColumnParallelLinear(
                input_size=hidden_size,
                output_size=in_proj_output_size,
                bias=use_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj",
            )

            # - because in_proj is a concatenation of multiple weights, we
            #   need to interleave them before sharding
            # - use the custom weight loader mamba_v2_sharded_weight_loader
            #   for conv1d.bias, covn1d.weight and in_proj.weight
            # - need to set these settings, to assign the groups
            #   to the head shards
            group_shard_settings = (
                self.groups_ssm_state_size,  # expected model size
                (self.n_groups - n_groups) * self.ssm_state_size,  # extra dims assigned
                n_groups == 1,  # if there was only one group
            )
            intermediate_settings = (intermediate_size, 0, False)
            head_settings = (self.num_heads, 0, False)

            # - the weight already has a "weight_loader" attribute
            #   which set_weight_attrs will raise if we do not
            #   delete before trying to override it
            # - ditto for the other two weights below
            delattr(self.conv1d.bias, "weight_loader")
            set_weight_attrs(
                self.conv1d.bias,
                {
                    "weight_loader": mamba_v2_sharded_weight_loader(
                        [
                            intermediate_settings,
                            group_shard_settings,
                            group_shard_settings,
                        ],
                        self.tp_size,
                        tp_rank,
                    )
                },
            )

            delattr(self.conv1d.weight, "weight_loader")
            set_weight_attrs(
                self.conv1d.weight,
                {
                    "weight_loader": mamba_v2_sharded_weight_loader(
                        [
                            intermediate_settings,
                            group_shard_settings,
                            group_shard_settings,
                        ],
                        self.tp_size,
                        tp_rank,
                    )
                },
            )

            if quant_config is None:
                # - quant layers do not have a weight loader
                delattr(self.in_proj.weight, "weight_loader")
                in_proj_loader_settings = [intermediate_settings]
                if not self._split_gate_proj:
                    in_proj_loader_settings.append(intermediate_settings)
                in_proj_loader_settings.extend(
                    [
                        group_shard_settings,
                        group_shard_settings,
                        head_settings,  # for dt
                    ]
                )
                set_weight_attrs(
                    self.in_proj.weight,
                    {
                        "weight_loader": mamba_v2_sharded_weight_loader(
                            in_proj_loader_settings,
                            self.tp_size,
                            tp_rank,
                        )
                    },
                )

        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `MergedColumnParallelLinear`,
        # and `set_weight_attrs` doesn't allow to override it
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        self.register_buffer("conv_weights", conv_weights, persistent=False)

        # - these are TPed by heads to reduce the size of the
        #   temporal shape
        self.A = nn.Parameter(
            torch.empty(
                divide(num_heads, self.tp_size),
                dtype=torch.float32,
            )
        )
        self.D = nn.Parameter(torch.ones(num_heads // self.tp_size))
        self.dt_bias = nn.Parameter(torch.ones(num_heads // self.tp_size))
        self.use_rms_norm = use_rms_norm

        set_weight_attrs(self.D, {"weight_loader": sharded_weight_loader(0)})
        a_weight_loader = composed_weight_loader(
            sharded_weight_loader(0), lambda x: -torch.exp(x.float())
        )
        set_weight_attrs(self.A, {"weight_loader": a_weight_loader})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        self.out_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=use_bias,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        self.norm = Mixer2RMSNormGated(
            intermediate_size, n_groups, self.use_rms_norm, eps=rms_norm_eps
        )

        # - get hidden_states, B and C after depthwise convolution.
        self.split_hidden_states_B_C_fn = lambda hidden_states_B_C: torch.split(
            hidden_states_B_C,
            [
                self.intermediate_size // self.tp_size,
                self.groups_ssm_state_size // self.tp_size,
                self.groups_ssm_state_size // self.tp_size,
            ],
            dim=-1,
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        # The tuple is (conv_state, ssm_state)
        self.kv_cache = (torch.tensor([]), torch.tensor([]))

        self.model_config = model_config
        self.cache_config = cache_config
        self.prefix = prefix

        # Pre-compute sizes for forward pass
        self.tped_intermediate_size = self.intermediate_size // self.tp_size
        self.tped_conv_size = self.conv_dim // self.tp_size
        self.tped_dt_size = self.num_heads // self.tp_size
        self._prefill_decode_stream = (
            mamba_prefill_decode_stream()
            if self._enable_mamba_multi_stream
            else None
        )
        self._prefill_decode_events: (
            tuple[torch.cuda.Event, torch.cuda.Event] | None
        ) = None
        if self._prefill_decode_stream is not None:
            self._prefill_decode_events = (
                torch.cuda.Event(),
                torch.cuda.Event(),
            )

        # Dedicated gate stream so the gate projection can overlap
        # with conv+SSM; we only synchronize at the gated norm.
        self._gate_stream = mamba_gate_stream() if self._split_gate_proj else None
        self.split_hidden_states_B_C_fn = lambda hidden_states_B_C: torch.split(
            hidden_states_B_C,
            [
                self.tped_intermediate_size,
                self.groups_ssm_state_size // self.tp_size,
                self.groups_ssm_state_size // self.tp_size,
            ],
            dim=-1,
        )
        if self._split_gate_proj:
            self._wrap_gate_and_in_proj_weight_loaders()

    def _wrap_gate_and_in_proj_weight_loaders(self) -> None:
        """Ensure legacy fused in_proj weights populate gate_proj too."""
        gate_params = dict(self.gate_proj.named_parameters(recurse=True))
        gate_base_loaders: dict[str, LoaderFunction] = {}

        for name, gate_param in gate_params.items():
            base_loader = getattr(gate_param, "weight_loader", None)
            if base_loader is None:
                continue
            gate_base_loaders[name] = base_loader

        for name, in_proj_param in self.in_proj.named_parameters(recurse=True):
            gate_param = gate_params.get(name)
            gate_base_loader = gate_base_loaders.get(name)
            if gate_param is None or gate_base_loader is None:
                continue
            base_loader = getattr(in_proj_param, "weight_loader", None)
            if base_loader is None:
                continue

            def _in_proj_loader(
                param: torch.Tensor,
                loaded_weight: torch.Tensor,
                *args,
                _base_loader: LoaderFunction = base_loader,
                _gate_param: torch.Tensor = gate_param,
                _gate_base_loader: LoaderFunction = gate_base_loader,
                _param_name: str = name,
                **kwargs,
            ) -> None:
                try:
                    gate_weight, in_proj_weight = self._split_fused_in_proj_weight(
                        loaded_weight,
                        _gate_param,
                        param,
                        param_name=_param_name,
                    )
                except ValueError:
                    _base_loader(param, loaded_weight, *args, **kwargs)
                    return
                _gate_base_loader(_gate_param, gate_weight, *args, **kwargs)
                _base_loader(param, in_proj_weight, *args, **kwargs)

            in_proj_param.weight_loader = _in_proj_loader  # type: ignore[assignment]

    @staticmethod
    def _split_fused_in_proj_weight(
        loaded_weight: torch.Tensor,
        gate_param: torch.Tensor,
        in_proj_param: torch.Tensor,
        *,
        param_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            loaded_weight.shape == gate_param.shape
            and loaded_weight.shape == in_proj_param.shape
        ):
            return loaded_weight, loaded_weight

        if (
            loaded_weight.numel() == 1
            and gate_param.numel() == 1
            and in_proj_param.numel() == 1
        ):
            return (
                loaded_weight.reshape(gate_param.shape),
                loaded_weight.reshape(in_proj_param.shape),
            )

        tp_size = getattr(
            gate_param, "tp_size", get_tensor_model_parallel_world_size()
        )
        dims_to_try: list[int] = []
        for attr in ("output_dim", "input_dim"):
            dim = getattr(gate_param, attr, None)
            if dim is not None and dim not in dims_to_try:
                dims_to_try.append(dim)
        dims_to_try += [
            dim for dim in range(loaded_weight.dim()) if dim not in dims_to_try
        ]

        for dim in dims_to_try:
            if (
                dim >= loaded_weight.dim()
                or dim >= gate_param.dim()
                or dim >= in_proj_param.dim()
            ):
                continue
            gate_dim = gate_param.shape[dim]
            in_proj_dim = in_proj_param.shape[dim]
            total_dim = gate_dim + in_proj_dim
            loaded_dim = loaded_weight.shape[dim]
            if loaded_dim == total_dim:
                gate_size = gate_dim
                in_proj_size = in_proj_dim
            elif loaded_dim == total_dim * tp_size:
                gate_size = gate_dim * tp_size
                in_proj_size = in_proj_dim * tp_size
            else:
                continue
            return (
                loaded_weight.narrow(dim, 0, gate_size),
                loaded_weight.narrow(dim, gate_size, in_proj_size),
            )

        raise ValueError(
            "Unable to split fused in_proj parameter "
            f"{param_name!r} with shape {loaded_weight.shape}."
        )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        mup_vector: torch.Tensor | None = None,
    ):
        pass

    def gate_in_proj_forward(
        self,
        hidden_states: torch.Tensor,
        mup_vector: torch.Tensor | None = None,
        *,
        allow_gate_stream: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._split_gate_proj:
            gate_mup = None
            ssm_mup = None
            if mup_vector is not None:
                gate_mup = mup_vector[..., : self.tped_intermediate_size]
                ssm_mup = mup_vector[..., self.tped_intermediate_size :]

            def _apply_gate_proj() -> torch.Tensor:
                gate = self.gate_proj(hidden_states)
                if gate_mup is not None:
                    gate = gate * gate_mup
                return gate

            # Gate projection runs on a dedicated stream so the conv+SSM path
            # can start immediately on the default stream. Synchronize only
            # right before the gated RMSNorm.
            num_tokens = hidden_states.shape[0]
            use_gate_stream = (
                allow_gate_stream
                and self._gate_stream is not None
                and hidden_states.is_cuda
                and num_tokens <= envs.VLLM_MAMBA_GATE_STREAM_TOKEN_THRESHOLD
            )
            if use_gate_stream:
                main_stream = current_stream()
                self._gate_stream.wait_stream(main_stream)
                with torch.cuda.stream(self._gate_stream):
                    gate = _apply_gate_proj()
            else:
                gate = _apply_gate_proj()

            # Linear projection for conv + SSM (gate-free).
            projected_states, _ = self.in_proj(hidden_states)
            if ssm_mup is not None:
                projected_states = projected_states * ssm_mup
        else:
            # Fused in_proj keeps gate + SSM paths together.
            projected_states, _ = self.in_proj(hidden_states)
            if mup_vector is not None:
                projected_states = projected_states * mup_vector
            gate = projected_states[..., : self.tped_intermediate_size]

        return gate, projected_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        mup_vector: torch.Tensor | None = None,
    ):
        gate, projected_states = torch.ops.vllm.mamba_gate_in_proj(
            hidden_states,
            mup_vector,
            self.prefix,
        )

        # 3. Prepare inputs for conv + SSM
        ssm_output = torch.empty(
            [
                hidden_states.shape[0],
                (self.num_heads // self.tp_size) * self.head_dim,
            ],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # 4. conv + SSM
        # (split `projected_states` into hidden_states_B_C, dt in the custom op to
        # ensure it is not treated as an intermediate tensor by torch compile)
        torch.ops.vllm.mamba_mixer2(
            projected_states,
            ssm_output,
            self.prefix,
        )

        torch.ops.vllm.mamba_gate_sync(
            gate,
            self.prefix,
        )

        # 5. gated MLP
        # GatedRMSNorm internally applying SiLU to the gate
        # SiLU is applied internally before normalization, unlike standard
        # norm usage
        hidden_states = self.norm(ssm_output, gate)

        # 6. Final linear projection
        output, _ = self.out_proj(hidden_states)

        return output

    def conv_ssm_forward(
        self,
        projected_states: torch.Tensor,
        output: torch.Tensor,
    ):
        ssm_projected_states = projected_states
        if not self._split_gate_proj:
            ssm_projected_states = projected_states[
                ..., self.tped_intermediate_size :
            ]

        hidden_states_B_C, dt = torch.split(
            ssm_projected_states,
            [self.tped_conv_size, self.tped_dt_size],
            dim=-1,
        )

        forward_context = get_forward_context()
        # attn_metadata contains metadata necessary for the mamba2 triton
        # kernels to operate in continuous batching and in chunked prefill
        # modes; they are computed at top-level model forward since they
        # stay the same and reused for all mamba layers in the same iteration
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        assert self.cache_config is not None
        mamba_block_size = self.cache_config.mamba_block_size
        is_mamba_cache_all = self.cache_config.mamba_cache_mode == "all"
        if attn_metadata is not None:
            assert isinstance(attn_metadata, dict)
            attn_metadata = attn_metadata[self.prefix]
            assert isinstance(attn_metadata, Mamba2AttentionMetadata)
            self_kv_cache = self.kv_cache[forward_context.virtual_engine]
            # conv_state = (..., dim, width-1) yet contiguous along 'dim'
            conv_state = self_kv_cache[0].transpose(-1, -2)
            ssm_state = self_kv_cache[1]
            state_indices_tensor = attn_metadata.state_indices_tensor
            has_initial_states_p = attn_metadata.has_initial_states_p
            prep_initial_states = attn_metadata.prep_initial_states
            chunk_size = attn_metadata.chunk_size
            seq_idx_p = attn_metadata.seq_idx_p
            query_start_loc_p = attn_metadata.query_start_loc_p
            cu_chunk_seqlen_p = attn_metadata.cu_chunk_seqlen_p
            last_chunk_indices_p = attn_metadata.last_chunk_indices_p

        if attn_metadata is None:
            # profile run
            hidden_states_B_C = (
                hidden_states_B_C.transpose(0, 1).clone().transpose(0, 1)
            ).contiguous()
            hidden_states, _B, _C = self.split_hidden_states_B_C_fn(hidden_states_B_C)
            return hidden_states

        num_prefills = attn_metadata.num_prefills  # request count
        num_decodes = attn_metadata.num_decode_tokens  # token count (=request)
        num_prefill_tokens = attn_metadata.num_prefill_tokens  # token count
        has_prefill = num_prefills > 0
        has_decode = num_decodes > 0
        num_actual_tokens = num_prefill_tokens + num_decodes

        # Separate prefill and decode by splitting varlen input
        # Split along token dimension
        hidden_states_B_C_d, hidden_states_B_C_p = torch.split(
            hidden_states_B_C[:num_actual_tokens],
            [num_decodes, num_prefill_tokens],
            dim=0,
        )
        dt_d, dt_p = torch.split(
            dt[:num_actual_tokens],
            [num_decodes, num_prefill_tokens],
            dim=0,
        )
        # Split along batch dimension
        state_indices_tensor_d, state_indices_tensor_p = torch.split(
            state_indices_tensor[:num_actual_tokens],
            [num_decodes, num_prefills],
            dim=0,
        )

        if is_mamba_cache_all:
            # If prefix caching is enabled, retrieve the relevant variables
            # for prefill and decode
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
            num_computed_tokens_p = attn_metadata.num_computed_tokens_p
        else:
            block_idx_last_computed_token_d = None
            block_idx_last_computed_token_p = None
            block_idx_last_scheduled_token_d = None
            block_idx_last_scheduled_token_p = None
            block_idx_first_scheduled_token_p = None
            num_computed_tokens_p = None

        preallocated_ssm_out_d, preallocated_ssm_out_p = torch.split(
            output[:num_actual_tokens],
            [num_decodes, num_prefill_tokens],
            dim=0,
        )

        is_compiling = torch.compiler.is_compiling()

        def _run_prefill() -> torch.Tensor | None:
            if not has_prefill:
                return None

            # 2. Convolution sequence transformation
            # - It will read the initial states for every sequence,
            #   that has "has_initial_states_p" == True,
            #   from "cache_indices", using "state_indices_tensor_p".
            # - It updates the "conv_state" cache in positions pointed
            #   to by "state_indices_tensor_p".
            #   In particular, it will always write the state at the
            #   sequence end.
            #   In addition, "block_idx_first_scheduled_token_p" and
            #   "block_idx_last_scheduled_token_p"
            #   are provided (which are pointers into
            #   "state_indices_tensor_p"), it will write additional cache
            #   states aligned at "block_size_to_align".
            x = hidden_states_B_C_p.transpose(
                0, 1
            )  # this is the form that causal-conv see
            hidden_states_B_C_p_local = causal_conv1d_fn(
                x,
                self.conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_states_p,
                cache_indices=state_indices_tensor_p,
                block_idx_first_scheduled_token=block_idx_first_scheduled_token_p,
                block_idx_last_scheduled_token=block_idx_last_scheduled_token_p,
                initial_state_idx=block_idx_last_computed_token_p,
                num_computed_tokens=num_computed_tokens_p,
                block_size_to_align=mamba_block_size,
                metadata=attn_metadata,
                query_start_loc=query_start_loc_p,
            ).transpose(0, 1)[:num_prefill_tokens]

            hidden_states_p, B_p, C_p = self.split_hidden_states_B_C_fn(
                hidden_states_B_C_p_local
            )

            # 3. State Space Model sequence transformation
            initial_states = None
            if has_initial_states_p is not None and prep_initial_states:
                kernel_ssm_indices = state_indices_tensor_p
                if is_mamba_cache_all:
                    kernel_ssm_indices = state_indices_tensor_p.gather(
                        1, block_idx_last_computed_token_p.unsqueeze(1)
                    ).squeeze(1)
                initial_states = torch.where(
                    has_initial_states_p[:, None, None, None],
                    ssm_state[kernel_ssm_indices],
                    0,
                )

            # NOTE: final output is an in-place update of out tensor
            varlen_states = mamba_chunk_scan_combined_varlen(
                hidden_states_p.view(
                    num_prefill_tokens,
                    self.num_heads // self.tp_size,
                    self.head_dim,
                ),
                dt_p,
                self.A,
                B_p.view(num_prefill_tokens, self.n_groups // self.tp_size, -1),
                C_p.view(num_prefill_tokens, self.n_groups // self.tp_size, -1),
                chunk_size=chunk_size,
                D=self.D,
                z=None,
                dt_bias=self.dt_bias,
                seq_idx=seq_idx_p,
                cu_seqlens=query_start_loc_p,
                cu_chunk_seqlens=cu_chunk_seqlen_p,
                last_chunk_indices=last_chunk_indices_p,
                initial_states=initial_states,
                return_intermediate_states=is_mamba_cache_all,
                dt_softplus=True,
                dt_limit=(0.0, float("inf")),
                out=preallocated_ssm_out_p.view(
                    num_prefill_tokens, -1, self.head_dim
                ),
                state_dtype=ssm_state.dtype,
            )

            if is_mamba_cache_all:
                # The chunk_stride is the number of chunks per mamba block
                # e.g., if mamba_block_size = 512 and chunk_size = 256,
                # then chunk_stride = 2
                chunk_stride = mamba_block_size // chunk_size

                # Save state for sequences with more than just final state
                for seq_idx in range(num_prefills):
                    # Block index for the first scheduled token
                    block_idx_first_scheduled_token = (
                        block_idx_first_scheduled_token_p[seq_idx]
                    )

                    # Block index for the last scheduled token
                    block_idx_last_scheduled_token = (
                        block_idx_last_scheduled_token_p[seq_idx]
                    )

                    # Number of blocks that need to be written
                    n_blocks_to_fill = (
                        block_idx_last_scheduled_token
                        - block_idx_first_scheduled_token
                    )

                    # Skip sequences that don't have any blocks to fill
                    if n_blocks_to_fill == 0:
                        continue

                    # Look up the state indices
                    cache_blocks_to_fill = state_indices_tensor_p[
                        seq_idx,
                        block_idx_first_scheduled_token:block_idx_last_scheduled_token,
                    ]

                    # First chunk index for this sequence
                    if seq_idx == 0:
                        first_chunk = 0
                    else:
                        first_chunk = 1 + last_chunk_indices_p[seq_idx - 1]

                    # First chunk that is aligned on the mamba block boundary
                    first_aligned_chunk = first_chunk + chunk_stride - 1

                    # Calculate the number of computed tokens that were not
                    # already cached
                    num_unaligned_computed_tokens = (
                        num_computed_tokens_p[seq_idx] % mamba_block_size
                    )

                    if num_unaligned_computed_tokens > 0:
                        # If the number of computed tokens is not block aligned,
                        # then we need to shift the index accordingly
                        first_aligned_chunk -= (
                            num_unaligned_computed_tokens // chunk_size
                        )

                    # Get states to write
                    from_where = varlen_states[
                        first_aligned_chunk : first_aligned_chunk
                        + n_blocks_to_fill * chunk_stride : chunk_stride
                    ]

                    # Write the states
                    ssm_state[cache_blocks_to_fill] = from_where

                # For all seqs, store the last state (note: might be partial):
                ssm_state[
                    state_indices_tensor_p.gather(
                        1, block_idx_last_scheduled_token_p.unsqueeze(1)
                    ).squeeze(1)
                ] = varlen_states[last_chunk_indices_p]

            else:
                # update ssm states
                # - varlen state is a (num_prefills, nheads, headdim, dstate)
                #   tensor
                ssm_state[state_indices_tensor_p] = varlen_states

            return preallocated_ssm_out_p

        def _run_decode() -> torch.Tensor | None:
            if not has_decode:
                return None

            if is_mamba_cache_all:
                state_indices_tensor_d_input = state_indices_tensor_d.gather(
                    1, block_idx_last_computed_token_d.unsqueeze(1)
                ).squeeze(1)
                state_indices_tensor_d_output = state_indices_tensor_d.gather(
                    1, block_idx_last_scheduled_token_d.unsqueeze(1)
                ).squeeze(1)
                # for decode:
                #   block_idx_first_scheduled_token_d ==
                #       block_idx_last_scheduled_token_d
                # at block boundaries:
                #   block_idx_first_scheduled_token_d >
                #       block_idx_last_computed_token_d
            else:
                # Without caching, read and write in-place to the same blocks:
                state_indices_tensor_d_input = state_indices_tensor_d
                state_indices_tensor_d_output = state_indices_tensor_d

            # 2. Convolution sequence transformation
            hidden_states_B_C_d_local = causal_conv1d_update(
                hidden_states_B_C_d,
                conv_state,
                self.conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=state_indices_tensor_d,
                block_idx_last_scheduled_token=block_idx_last_scheduled_token_d,
                initial_state_idx=block_idx_last_computed_token_d,
            )

            hidden_states_d, B_d, C_d = self.split_hidden_states_B_C_fn(
                hidden_states_B_C_d_local
            )

            # 3. State Space Model sequence transformation
            n_groups = self.n_groups // self.tp_size
            A_d = (
                self.A[:, None, ...][:, :, None]
                .expand(-1, self.head_dim, self.ssm_state_size)
                .to(dtype=torch.float32)
            )
            dt_d_local = dt_d[:, :, None].expand(-1, -1, self.head_dim)
            dt_bias = self.dt_bias[:, None, ...].expand(-1, self.head_dim)
            D_d = self.D[:, None, ...].expand(-1, self.head_dim)
            B_d = B_d.view(-1, n_groups, B_d.shape[1] // n_groups)
            C_d = C_d.view(-1, n_groups, C_d.shape[1] // n_groups)
            hidden_states_d = hidden_states_d.view(
                -1, self.num_heads // self.tp_size, self.head_dim
            )

            # - the hidden is reshaped into (bs, num_heads, head_dim)
            # - mamba_cache_params.ssm_state's slots will be selected
            #   using state_indices_tensor_d
            # NOTE: final output is an in-place update of out tensor
            selective_state_update(
                ssm_state,
                hidden_states_d,
                dt_d_local,
                A_d,
                B_d,
                C_d,
                D_d,
                z=None,
                dt_bias=dt_bias,
                dt_softplus=True,
                state_batch_indices=state_indices_tensor_d_input,
                dst_state_batch_indices=state_indices_tensor_d_output,
                out=preallocated_ssm_out_d.view(num_decodes, -1, self.head_dim),
            )

            return preallocated_ssm_out_d

        use_multi_stream = (
            has_prefill
            and has_decode
            and self._prefill_decode_stream is not None
            and self._prefill_decode_events is not None
            and not is_compiling
        )
        if use_multi_stream:
            event_main, event_aux = self._prefill_decode_events
            maybe_execute_in_parallel(
                _run_prefill,
                _run_decode,
                event_main,
                event_aux,
                self._prefill_decode_stream,
            )
        else:
            _run_prefill()
            _run_decode()

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        assert self.model_config is not None
        assert self.cache_config is not None
        return MambaStateDtypeCalculator.mamba2_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.mamba2_state_shape(
            intermediate_size=self.intermediate_size,
            tp_world_size=get_tensor_model_parallel_world_size(),
            n_groups=self.n_groups,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            state_size=self.ssm_state_size,
            conv_kernel=self.conv_kernel_size,
        )

    @property
    def mamba_type(self) -> str:
        return "mamba2"


def mamba_gate_in_proj(
    hidden_states: torch.Tensor,
    mup_vector: torch.Tensor | None,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    allow_gate_stream = not torch.cuda.is_current_stream_capturing()
    return self.gate_in_proj_forward(
        hidden_states,
        mup_vector,
        allow_gate_stream=allow_gate_stream,
    )


def mamba_gate_in_proj_fake(
    hidden_states: torch.Tensor,
    mup_vector: torch.Tensor | None,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    num_tokens = hidden_states.shape[0]
    if self._split_gate_proj:
        gate = torch.empty(
            (num_tokens, self.tped_intermediate_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        proj_size = self.tped_conv_size + self.tped_dt_size
        projected_states = torch.empty(
            (num_tokens, proj_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
    else:
        proj_size = (
            self.tped_intermediate_size + self.tped_conv_size + self.tped_dt_size
        )
        projected_states = torch.empty(
            (num_tokens, proj_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        # Match real op: gate is a view into projected_states.
        gate = projected_states[..., : self.tped_intermediate_size]
    return gate, projected_states


direct_register_custom_op(
    op_name="mamba_gate_in_proj",
    op_func=mamba_gate_in_proj,
    fake_impl=mamba_gate_in_proj_fake,
)


def mamba_gate_sync(
    gate: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    if (
        self._gate_stream is None
        or not gate.is_cuda
        or torch.cuda.is_current_stream_capturing()
    ):
        return
    main_stream = current_stream()
    main_stream.wait_stream(self._gate_stream)
    gate.record_stream(main_stream)


def mamba_gate_sync_fake(
    gate: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="mamba_gate_sync",
    op_func=mamba_gate_sync,
    fake_impl=mamba_gate_sync_fake,
)


def mamba_mixer2(
    projected_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self.conv_ssm_forward(projected_states=projected_states, output=output)


def mamba_mixer2_fake(
    projected_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="mamba_mixer2",
    op_func=mamba_mixer2,
    mutates_args=["output"],
    fake_impl=mamba_mixer2_fake,
)
