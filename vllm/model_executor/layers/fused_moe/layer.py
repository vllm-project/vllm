# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import ParallelConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_dp_group,
    get_pcp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
)
from vllm.model_executor.layers.fused_moe.eplb_manager import EplbManager
from vllm.model_executor.layers.fused_moe.expert_map_manager import (
    ExpertMapManager,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.router.router_factory import (
    create_fused_moe_router,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
    MoERunner,
)
from vllm.model_executor.layers.fused_moe.utils import (
    disable_inplace,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)

logger = init_logger(__name__)


def register_layer_for_moe_forward_op(
    vllm_config: VllmConfig,
    layer: MoERunner,
):
    # For smuggling this layer into the fused moe custom op
    prefix = layer.layer_name
    compilation_config = vllm_config.compilation_config
    if prefix in compilation_config.static_forward_context:
        raise ValueError("Duplicate layer name: {}".format(prefix))
    compilation_config.static_forward_context[prefix] = layer
    compilation_config.static_all_moe_layers.append(prefix)


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

    is_sequence_parallel = is_sequence_parallel
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
    # ROCm aiter shared experts fusion
    # AITER only supports gated activations (silu/gelu), so disable it
    # for non-gated MoE (is_act_and_mul=False)
    # rocm_aiter_fmoe_enabled = rocm_aiter_ops.is_fused_moe_enabled() and is_act_and_mul
    aiter_fmoe_shared_expert_enabled = (
        rocm_aiter_ops.is_fusion_moe_shared_experts_enabled() and is_act_and_mul
    )

    num_fused_shared_experts = (
        n_shared_experts
        if n_shared_experts is not None and aiter_fmoe_shared_expert_enabled
        else 0
    )
    if not aiter_fmoe_shared_expert_enabled and num_fused_shared_experts != 0:
        raise ValueError(
            "n_shared_experts is only supported on ROCm aiter when "
            "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS is enabled"
        )

    return global_num_experts, logical_num_experts, num_fused_shared_experts


# TODO: rename this
def FusedMoE(
    num_experts: int,  # Global number of experts
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
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
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    apply_router_weight_on_input: bool = False,
    activation: str = "silu",
    enable_eplb: bool = False,
    num_redundant_experts: int = 0,
    has_bias: bool = False,
    is_sequence_parallel=False,
    expert_mapping: list[tuple[str, str, int, str]] | None = None,
    n_shared_experts: int | None = None,
    router_logits_dtype: torch.dtype | None = None,
    gate: torch.nn.Module | None = None,
    shared_experts: torch.nn.Module | None = None,
    routed_input_transform: torch.nn.Module | None = None,
    routed_output_transform: torch.nn.Module | None = None,
    apply_routed_scale_to_output: bool = False,
    zero_expert_type: str | None = None,
) -> MoERunner:
    # TODO update comment
    """FusedMoE layer builder for MoE models.

    This layer contains both MergedColumnParallel weights (gate_up_proj /
    w13) and RowParallelLinear weights (down_proj/ w2).

    Note: Mixtral uses w1, w2, and w3 for gate, up, and down_proj. We
    copy that naming convention here and handle any remapping in the
    load_weights function in each model implementation.

    Args:
        num_experts: Number of experts in the model
        top_k: Number of experts selected for each token
        hidden_size: Input hidden state size of the transformer
        intermediate_size: Intermediate size of the experts
        params_dtype: Data type for the parameters.
        renormalize: Whether to renormalize the logits in the fused_moe kernel
        quant_config: Quantization configure.
        enable_eplb: Whether to enable expert parallelism load balancer.
        router_logits_dtype: Data type for router logits buffers.
        routed_scaling_factor: A scaling factor that is applied to the topk_weights
                               by the router or the output of the layer depending
                               on the value of `apply_routed_scale_to_output`
        apply_routed_scale_to_output: Determine whether or not `routed_scaling_factor`
                                      is applied to the topk_weights or to the experts
                                      output. It is applied to the experts output
                                      instead of the topk_weights when this feature is
                                      not supported by the router (or the experts).
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

    global_num_experts, logical_num_experts, num_fused_shared_experts = (
        determine_expert_counts(
            num_experts,
            num_redundant_experts,
            n_shared_experts,
            is_act_and_mul,
        )
    )

    # Initialize EPLB manager (or None?)
    eplb_manager: EplbManager | None = None
    if enable_eplb:
        eplb_manager = EplbManager(
            ep_size=moe_parallel_config.ep_size,
            global_num_experts=global_num_experts,
            num_redundant_experts=num_redundant_experts,
        )
    else:
        assert num_redundant_experts == 0, (
            "Redundant experts are only supported with EPLB."
        )

    # Create expert map manager
    expert_map_manager = ExpertMapManager(
        max_num_batched_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
        top_k=top_k,
        global_num_experts=global_num_experts,
        logical_num_experts=logical_num_experts,
        num_redundant_experts=num_redundant_experts,
        num_expert_group=num_expert_group,
        moe_parallel_config=moe_parallel_config,
        placement_strategy=vllm_config.parallel_config.expert_placement_strategy,
        enable_eplb=eplb_manager is not None,
        num_fused_shared_experts=num_fused_shared_experts,
        rocm_aiter_enabled=rocm_aiter_ops.is_fused_moe_enabled() and is_act_and_mul,
        device=vllm_config.device_config.device,
    )

    # TODO(bnell): we should not have to create a router if the kernel is
    # monolithic.
    router = create_fused_moe_router(
        top_k=top_k,
        global_num_experts=global_num_experts,
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
        eplb_manager=eplb_manager,
        zero_expert_type=zero_expert_type,
        num_logical_experts=logical_num_experts,
    )

    # TODO: move this???????????  is this even needed???
    # When using zero experts, slice e_score_correction_bias to cover
    # only real experts, for compatibility with monolithic kernels that
    # read it directly.
    # if False and zero_expert_type is not None and e_score_correction_bias is not None:
    #    e_score_correction_bias = e_score_correction_bias[logical_num_experts]

    # FIXME (varun): We should have a better way of inferring the activation
    # datatype. This works for now as the tensor datatype entering the MoE
    # operation is typically unquantized (i.e. float16/bfloat16).
    if vllm_config.model_config is not None:
        moe_in_dtype = vllm_config.model_config.dtype
    elif params_dtype is not None:
        # TODO (bnell): This is a hack to get test_mixtral_moe to work
        # since model_config is not set in the pytest test.
        moe_in_dtype = params_dtype
    else:
        moe_in_dtype = torch.get_default_dtype()

    moe_config = FusedMoEConfig(
        num_experts=global_num_experts,
        experts_per_token=top_k,
        hidden_dim=hidden_size,
        intermediate_size=intermediate_size,
        num_local_experts=expert_map_manager.local_num_experts,
        num_logical_experts=logical_num_experts,
        moe_parallel_config=moe_parallel_config,
        in_dtype=moe_in_dtype,
        moe_backend=vllm_config.kernel_config.moe_backend,
        router_logits_dtype=router_logits_dtype,
        max_num_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
        has_bias=has_bias,
        is_lora_enabled=vllm_config.lora_config is not None,
        activation=moe_activation,
        device=vllm_config.device_config.device,
        routing_method=router.routing_method_type,  # Not ideal
        # TODO: in_dtype == out_dtype?
        disable_inplace=disable_inplace() or shared_experts is not None,
    )

    logger.debug("FusedMoEConfig = %s", moe_config)

    # Create RoutedExperts instance BEFORE create_weights()
    # This will hold all expert weight parameters
    routed_experts = RoutedExperts(
        layer_name,
        params_dtype,
        moe_config,
        quant_config,
        expert_map_manager=expert_map_manager,
        # Extra params that are needed by quant_methods, pass along for now
        top_k=top_k,
        use_grouped_topk=use_grouped_topk,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        custom_routing_function=custom_routing_function,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        # TODO get from router? needs to be truncated?
        e_score_correction_bias=e_score_correction_bias,
        apply_router_weight_on_input=apply_router_weight_on_input,
        activation=moe_activation,
    )

    runner = MoERunner(
        layer_name=layer_name,
        moe_config=moe_config,
        router=router,
        routed_input_transform=routed_input_transform,
        routed_output_transform=routed_output_transform,
        gate=gate,
        shared_experts=shared_experts,
        routed_experts=routed_experts,
        enable_dbo=vllm_config.parallel_config.enable_dbo,
        # When apply_routed_scale_to_output is True, we allow
        # the scaling factor to be passed to the runner, otherwise
        # we pass 1.0 so it ends up being a nop.
        routed_scaling_factor=routed_scaling_factor
        if apply_routed_scale_to_output
        else 1.0,
    )

    # For smuggling this layer into the fused moe custom op
    register_layer_for_moe_forward_op(vllm_config, runner)

    return runner


def fused_moe_make_expert_params_mapping(
    model: torch.nn.Module,
    ckpt_gate_proj_name: str,
    ckpt_down_proj_name: str,
    ckpt_up_proj_name: str,
    num_experts: int,
    num_redundant_experts: int = 0,
) -> list[tuple[str, str, int, str]]:
    """Delegate to EPLB manager."""
    return RoutedExperts.make_expert_params_mapping(
        model,
        ckpt_gate_proj_name,
        ckpt_down_proj_name,
        ckpt_up_proj_name,
        num_experts,
        num_redundant_experts,
    )
