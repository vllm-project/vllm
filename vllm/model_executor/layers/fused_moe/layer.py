# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import Any

import torch

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import ParallelConfig, get_current_vllm_config
from vllm.distributed import (
    get_dp_group,
    get_pcp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
)
from vllm.model_executor.layers.fused_moe.expert_map_manager import (
    ExpertMapManager,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from vllm.model_executor.layers.fused_moe.router.router_factory import (
    create_fused_moe_router,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
    MoERunner,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)

logger = init_logger(__name__)


def make_parallel_config(
    tp_size: int | None,
    dp_size: int | None,
    pcp_size: int | None,
    is_sequence_parallel: bool,
    parallel_config: ParallelConfig,
) -> FusedMoEParallelConfig:
    tp_size_ = (
        tp_size if tp_size is not None else get_tensor_model_parallel_world_size()
    )
    dp_size_ = dp_size if dp_size is not None else get_dp_group().world_size
    pcp_size_ = pcp_size if pcp_size is not None else get_pcp_group().world_size
    sp_size = tp_size_ if is_sequence_parallel else 1

    moe_parallel_config = FusedMoEParallelConfig.make(
        tp_size_=tp_size_,
        pcp_size_=pcp_size_,
        dp_size_=dp_size_,
        sp_size_=sp_size,
        vllm_parallel_config=parallel_config,
    )

    assert moe_parallel_config.is_sequence_parallel == is_sequence_parallel

    logger.debug("FusedMoEParallelConfig = %s", str(moe_parallel_config))

    return moe_parallel_config


def determine_expert_counts(
    num_experts: int,
    num_redundant_experts: int,
    n_shared_experts: int | None,
    is_act_and_mul: bool,
) -> tuple[int, int, int]:
    global_num_experts = num_experts + num_redundant_experts
    logical_num_experts = num_experts
    # Shared-expert fusion: append the shared expert(s) as routed-expert slots
    # so they run in the same grouped GEMM. Gated by
    # VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS: either the native aiter fused-MoE
    # path (env + master switch, via is_fusion_moe_shared_experts_enabled) or the
    # backend-neutral router-append path (env alone, independent of the master
    # switch; e.g. the MM3 triton/flydsl mxfp8 MoE). Gated activations only.
    fuse_shared_enabled = (
        rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        or envs.VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS
    ) and is_act_and_mul

    num_fused_shared_experts = (
        n_shared_experts if n_shared_experts is not None and fuse_shared_enabled else 0
    )

    return global_num_experts, logical_num_experts, num_fused_shared_experts


# TODO: rename this
def FusedMoE(
    num_experts: int,  # Global number of experts
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    intermediate_pad: int | None = None,
    params_dtype: torch.dtype | None = None,
    renormalize: bool = True,
    use_grouped_topk: bool = False,
    num_expert_group: int | None = None,
    topk_group: int | None = None,
    quant_config: QuantizationConfig | None = None,
    tp_size: int | None = None,
    dp_size: int | None = None,
    pcp_size: int | None = None,
    prefix: str = "",
    custom_routing_function: Callable | None = None,
    router: FusedMoERouter | None = None,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    swiglu_limit: float | None = None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
    e_score_correction_bias: torch.Tensor | None = None,
    apply_router_weight_on_input: bool = False,
    activation: str = "silu",
    enable_eplb: bool = False,
    num_redundant_experts: int = 0,
    has_bias: bool = False,
    is_sequence_parallel: bool = False,
    reduce_results: bool = True,
    ckpt_names: tuple[str, str, str] = ("gate_proj", "down_proj", "up_proj"),
    n_shared_experts: int | None = None,
    router_logits_dtype: torch.dtype | None = None,
    gate: torch.nn.Module | None = None,
    shared_experts: torch.nn.Module | None = None,
    shared_expert_gate: torch.nn.Module | None = None,
    routed_input_transform: torch.nn.Module | None = None,
    routed_output_transform: torch.nn.Module | None = None,
    apply_routed_scale_to_output: bool = False,
    zero_expert_type: str | None = None,
    hash_indices_table: torch.Tensor | None = None,
    runner_cls: type[MoERunner] | None = None,
    runner_args: dict[str, Any] | None = None,
    routed_experts_cls: type[RoutedExperts] | None = None,
    routed_experts_args: dict[str, Any] | None = None,
) -> MoERunner:
    """Factory function for creating MoE execution pipeline.

    Creates and configures a complete MoE execution pipeline including:
    - Router (for token-to-expert assignment)
    - RoutedExperts (containing expert weight parameters)
    - MoERunner (orchestrates the complete forward pass)

    The experts contain both MergedColumnParallel weights (gate_up_proj/w13)
    and RowParallelLinear weights (down_proj/w2).

    Note: Mixtral uses w1, w2, and w3 for gate, up, and down_proj. We
    copy that naming convention here and handle any remapping in the
    load_weights function in each model implementation.

    Args:
        num_experts: Number of experts in the model (global count)
        top_k: Number of experts selected for each token
        hidden_size: Input hidden state size of the transformer
        intermediate_size: Intermediate size of the experts
        params_dtype: Data type for the parameters
        renormalize: Whether to renormalize the logits in the router
        use_grouped_topk: Whether to use grouped top-k routing
        num_expert_group: Number of expert groups for grouped top-k
        topk_group: Top-k value per group for grouped top-k
        quant_config: Quantization configuration
        tp_size: Tensor parallelism size (None = use global default)
        dp_size: Data parallelism size (None = use global default)
        pcp_size: Pipeline context parallelism size (None = use global default)
        prefix: Layer name prefix for weight loading
        custom_routing_function: Custom routing function override
        router: Pre-configured router instance (None = create default)
        scoring_func: Scoring function for routing ("softmax" or others)
        routed_scaling_factor: Scaling factor applied to topk_weights or output
        swiglu_limit: SwiGLU activation limit
        e_score_correction_bias: Expert score correction bias tensor
        apply_router_weight_on_input: Whether to apply router weights on input
        activation: Activation function name ("silu", "gelu", etc.)
        enable_eplb: Whether to enable expert parallelism load balancer
        num_redundant_experts: Number of redundant experts for EPLB
        has_bias: Whether expert layers have bias terms
        is_sequence_parallel: Whether sequence parallelism is enabled
        reduce_results: Whether to all-reduce the final output. Setting this
        to False (to fuse the all-reduce downstream) is only honored on the
        late-AR path.
        expert_mapping: Expert parameter mapping for weight loading
        n_shared_experts: Number of shared experts to fuse into the routed
            grouped GEMM (ROCm; requires aiter FSE or the router-append path)
        router_logits_dtype: Data type for router logits buffers
        gate: Pre-configured gate module
        shared_experts: Pre-configured shared experts module
        shared_expert_gate: Pre-configured shared expert gate module
        routed_input_transform: Input transformation module
        routed_output_transform: Output transformation module
        apply_routed_scale_to_output: Whether to apply routed_scaling_factor to
                                      output instead of topk_weights
        zero_expert_type: Type of zero expert handling
        hash_indices_table: Hash table for expert indices
        runner_cls: Custom MoERunner class (None = use default MoERunner)
        runner_args: Additional arguments for runner constructor
        routed_experts_cls: Custom RoutedExperts class (None = use default)
        routed_experts_args: Additional arguments for routed_experts constructor

    Returns:
        MoERunner: Configured MoE execution pipeline ready for forward passes
    """
    vllm_config = get_current_vllm_config()

    layer_name = prefix

    moe_activation = MoEActivation.from_str(activation)
    is_act_and_mul = moe_activation.is_gated

    moe_parallel_config = make_parallel_config(
        tp_size=tp_size,
        dp_size=dp_size,
        pcp_size=pcp_size,
        is_sequence_parallel=is_sequence_parallel,
        parallel_config=vllm_config.parallel_config,
    )

    # Resolve the deferred all-reduce request against the parallel config.
    skip_final_all_reduce = (
        not reduce_results
        and not moe_parallel_config.use_all2all_kernels
        and not moe_parallel_config.is_sequence_parallel
        and zero_expert_type is None
    )

    global_num_experts, logical_num_experts, num_fused_shared_experts = (
        determine_expert_counts(
            num_experts,
            num_redundant_experts,
            n_shared_experts,
            is_act_and_mul,
        )
    )

    # Initialize EPLB manager (or None?)
    eplb_state: EplbLayerState | None = None
    if enable_eplb:
        use_ep = moe_parallel_config.use_ep
        ep_size = moe_parallel_config.ep_size
        if use_ep and global_num_experts % ep_size != 0:
            raise ValueError(
                f"EPLB currently only supports even distribution of "
                f"experts across ranks. Got {global_num_experts} experts "
                f"and {ep_size} EP ranks."
            )
        eplb_state = EplbLayerState()
    else:
        assert num_redundant_experts == 0, (
            "Redundant experts are only supported with EPLB."
        )

    max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens

    # Create ExpertMapManager to handle expert mapping and placement for EP.
    # See ExpertMapManager for a detailed description of what it does and when
    # it is required.
    expert_map_manager = ExpertMapManager(
        max_num_batched_tokens=max_num_batched_tokens,
        top_k=top_k,
        global_num_experts=global_num_experts,
        num_redundant_experts=num_redundant_experts,
        num_expert_group=num_expert_group,
        moe_parallel_config=moe_parallel_config,
        placement_strategy=vllm_config.parallel_config.expert_placement_strategy,
        enable_eplb=eplb_state is not None,
        num_fused_shared_experts=num_fused_shared_experts,
        rocm_aiter_enabled=rocm_aiter_ops.is_fused_moe_enabled() and is_act_and_mul,
    )

    # TODO(bnell): we should not have to create a router if the kernel is
    # monolithic.
    if router is None:
        router = create_fused_moe_router(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            # When apply_routed_scale_to_output is True, we set the scaling factor
            # to 1.0 so it ends up being a nop. Applying the scale will be handled
            # by the runner in this case.
            # The member variable must be set in the same way as the router since
            # some quantization methods can access it.
            routed_scaling_factor=routed_scaling_factor
            if not apply_routed_scale_to_output
            else 1.0,
            e_score_correction_bias=e_score_correction_bias,
            num_fused_shared_experts=num_fused_shared_experts,
            # Fused shared-expert slot weight. With apply_routed_scale_to_output
            # the runner scales the combined output by routed_scaling_factor, so
            # the shared slot weight must be 1/routed_scaling_factor for its net
            # contribution to be 1.0 (matching the un-scaled separate-MLP add).
            shared_expert_weight=(
                (1.0 / routed_scaling_factor)
                if (
                    apply_routed_scale_to_output
                    and num_fused_shared_experts > 0
                    and routed_scaling_factor
                )
                else 1.0
            ),
            zero_expert_type=zero_expert_type,
            num_logical_experts=logical_num_experts,
            hash_indices_table=hash_indices_table,
        )

    if params_dtype is None:
        params_dtype = torch.get_default_dtype()

    # FIXME (varun): We should have a better way of inferring the activation
    # datatype. This works for now as the tensor datatype entering the MoE
    # operation is typically unquantized (i.e. float16/bfloat16).
    if vllm_config.model_config is not None:
        moe_in_dtype = vllm_config.model_config.dtype
    else:
        # TODO (bnell): This is a hack to get test_mixtral_moe to work
        # since model_config is not set in the pytest test.
        moe_in_dtype = params_dtype

    moe_config = FusedMoEConfig(
        num_experts=global_num_experts,
        experts_per_token=top_k,
        hidden_dim=hidden_size,
        intermediate_size=intermediate_size,
        intermediate_pad=intermediate_pad,
        num_local_experts=expert_map_manager.local_num_experts,
        num_logical_experts=logical_num_experts,
        moe_parallel_config=moe_parallel_config,
        in_dtype=moe_in_dtype,
        moe_backend=vllm_config.kernel_config.moe_backend,
        router_logits_dtype=router_logits_dtype,
        max_num_tokens=max_num_batched_tokens,
        has_bias=has_bias,
        is_lora_enabled=vllm_config.lora_config is not None,
        activation=moe_activation,
        device=vllm_config.device_config.device,
        routing_method=router.routing_method_type,  # Not ideal
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        max_capture_size=vllm_config.compilation_config.max_cudagraph_capture_size,
        skip_final_all_reduce=skip_final_all_reduce,
    )

    logger.debug("FusedMoEConfig = %s", moe_config)

    # Create RoutedExperts instance BEFORE create_weights()
    # This will hold all expert weight parameters
    if routed_experts_cls is None:
        routed_experts_cls = RoutedExperts

    assert params_dtype is not None
    routed_experts = routed_experts_cls(
        layer_name,
        params_dtype,
        moe_config,
        quant_config,
        expert_map_manager=expert_map_manager,
        ckpt_gate_proj_name=ckpt_names[0],
        ckpt_down_proj_name=ckpt_names[1],
        ckpt_up_proj_name=ckpt_names[2],
        # Extra params that are needed by quant_methods, pass along for now
        # Prefer getting these from other sources, e.g. moe_config or
        # router object
        renormalize=renormalize,
        use_grouped_topk=use_grouped_topk,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        custom_routing_function=custom_routing_function,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor
        if not apply_routed_scale_to_output
        else 1.0,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        # TODO get from router? needs to be truncated?
        e_score_correction_bias=e_score_correction_bias,
        apply_router_weight_on_input=apply_router_weight_on_input,
        **routed_experts_args if routed_experts_args is not None else {},
    )

    if runner_cls is None:
        runner_cls = MoERunner

    runner = runner_cls(
        layer_name=layer_name,
        moe_config=moe_config,
        router=router,
        routed_experts=routed_experts,
        enable_dbo=vllm_config.parallel_config.enable_dbo,
        gate=gate,
        shared_expert_gate=shared_expert_gate,
        shared_experts=shared_experts,
        routed_input_transform=routed_input_transform,
        routed_output_transform=routed_output_transform,
        # When apply_routed_scale_to_output is True, we allow
        # the scaling factor to be passed to the runner, otherwise
        # we pass 1.0 so it ends up being a nop.
        routed_scaling_factor=routed_scaling_factor
        if apply_routed_scale_to_output
        else 1.0,
        **runner_args if runner_args is not None else {},
    )

    return runner


def fused_moe_make_expert_params_mapping(
    model: torch.nn.Module,
    ckpt_gate_proj_name: str,
    ckpt_down_proj_name: str,
    ckpt_up_proj_name: str,
    num_experts: int,
    num_redundant_experts: int = 0,
    routed_experts_prefix: str = "routed_experts",
) -> list[tuple[str, str, int, str]]:
    """Delegate to EPLB manager."""
    return RoutedExperts.make_expert_params_mapping(
        model,
        ckpt_gate_proj_name,
        ckpt_down_proj_name,
        ckpt_up_proj_name,
        num_experts,
        num_redundant_experts,
        routed_experts_prefix,
    )
