# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager, nullcontext
from enum import Enum
from typing import Literal, cast, get_args, overload

import torch
import torch.nn.functional as F
from torch.nn.parameter import UninitializedParameter

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.parallel import ExpertPlacementStrategy
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_pcp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.eplb.eplb_state import EplbLayerState, EplbState
from vllm.forward_context import (
    ForwardContext,
    get_forward_context,
    is_forward_context_available,
)
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.fused_moe_modular_method import (
    FusedMoEModularMethod,
)
from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
    init_aiter_topK_meta_data,
)
from vllm.model_executor.layers.fused_moe.router.router_factory import (
    create_fused_moe_router,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.fused_moe.utils import (
    disable_inplace,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.torch_utils import (
    aux_stream,
    current_stream,
    direct_register_custom_op,
)
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

logger = init_logger(__name__)


class FusedMoeWeightScaleSupported(Enum):
    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"
    BLOCK = "block"


def determine_expert_map(
    ep_size: int,
    ep_rank: int,
    global_num_experts: int,
    expert_placement_strategy: ExpertPlacementStrategy = "linear",
    num_fused_shared_experts: int = 0,
    return_expert_mask: bool = False,
) -> tuple[int, torch.Tensor | None, torch.Tensor | None]:
    """
    Calculates how many experts should be assigned to each rank for EP and
    creates a mapping from global to local expert index. Experts are
    distributed evenly across ranks. Any remaining are assigned to the
    last rank.

    Args:
        ep_size: The size of the expert parallel group
        ep_rank: The rank of the current process in the expert parallel
            group
        global_num_experts: The total number of experts in the model.
        expert_placement_strategy: The expert placement strategy.

    Returns:
        tuple[int, Optional[torch.Tensor]]: A tuple containing:
            - local_num_experts (int): The number of experts assigned
                to the current rank.
            - expert_map (Optional[torch.Tensor]): A tensor of shape
                (global_num_experts,) mapping from global to local index.
                Contains -1 for experts not assigned to the current rank.
                Returns None if ep_size is 1.
            - expert_mask (Optional[torch.Tensor]): A tensor of shape
                (global_num_experts + num_fused_shared_experts + 1,)
                containing 1 for experts assigned to the current rank
                and 0 for sentinel.
                Returns None if ep_size is 1.
                Used only when AITER MOE is enabled.
    """
    assert ep_size > 0
    if ep_size == 1:
        return (global_num_experts, None, None)

    # Distribute experts as evenly as possible to each rank.
    base_experts = global_num_experts // ep_size
    remainder = global_num_experts % ep_size
    local_num_experts = base_experts + 1 if ep_rank < remainder else base_experts

    # Create a tensor of size num_experts filled with -1
    expert_map = torch.full((global_num_experts,), -1, dtype=torch.int32)
    # Create an expert map for the local experts
    if expert_placement_strategy == "linear":
        start_idx = ep_rank * base_experts + min(ep_rank, remainder)
        expert_map[start_idx : start_idx + local_num_experts] = torch.arange(
            0, local_num_experts, dtype=torch.int32
        )
    elif expert_placement_strategy == "round_robin":
        local_log_experts = torch.arange(
            ep_rank, global_num_experts, ep_size, dtype=torch.int32
        )

        expert_map[local_log_experts] = torch.arange(
            0, local_num_experts, dtype=torch.int32
        )
    else:
        raise ValueError(
            "Unsupported expert placement strategy "
            f"'{expert_placement_strategy}', expected one of "
            f"{get_args(ExpertPlacementStrategy)}"
        )

    expert_mask = None
    if return_expert_mask:
        expert_mask = torch.ones(
            (global_num_experts + num_fused_shared_experts + 1,), dtype=torch.int32
        )
        expert_mask[-1] = 0
        expert_mask[:global_num_experts] = expert_map > -1
        expert_map = torch.cat(
            (
                expert_map,
                torch.tensor(
                    [local_num_experts + i for i in range(num_fused_shared_experts)],
                    dtype=torch.int32,
                ),
            ),
            dim=0,
        )

    return (local_num_experts, expert_map, expert_mask)


def determine_expert_placement_strategy(
    expert_placement_strategy: ExpertPlacementStrategy,
    moe_parallel_config: FusedMoEParallelConfig,
    num_expert_group: int | None,
    num_redundant_experts: int,
    enable_eplb: bool,
) -> ExpertPlacementStrategy:
    if expert_placement_strategy == "round_robin":
        round_robin_supported = (
            (num_expert_group is not None and num_expert_group > 1)
            and num_redundant_experts == 0
            and not enable_eplb
        )

        if not round_robin_supported:
            logger.warning(
                "Round-robin expert placement is only supported for "
                "models with multiple expert groups and no redundant "
                "experts. Falling back to linear expert placement."
            )
            return "linear"
        if (
            moe_parallel_config.use_all2all_kernels
            and not moe_parallel_config.use_deepep_ll_kernels
        ):
            logger.warning(
                "Round-robin expert placement currently only supports "
                "the DeepEP low-latency backend, but '%s' was configured. "
                "Falling back to linear expert placement.",
                moe_parallel_config.all2all_backend,
            )
            return "linear"

    return expert_placement_strategy


def get_compressed_expert_map(expert_map: torch.Tensor) -> str:
    """
    Compresses the expert map by removing any -1 entries.

    Args:
        expert_map (torch.Tensor): A tensor of shape (global_num_experts,)
            mapping from global to local index. Contains -1 for experts not
            assigned to the current rank.

    Returns:
        str: A string mapping from local to global index.
            Using str to support hashing for logging once only.
    """
    global_indices = torch.where(expert_map != -1)[0]
    local_indices = expert_map[global_indices]
    return ", ".join(
        f"{local_index.item()}->{global_index.item()}"
        for local_index, global_index in zip(local_indices, global_indices)
    )


# TODO(rob): move this down to the kernel.
def maybe_roundup_hidden_size(
    hidden_size: int,
    act_dtype: torch.dtype,
    moe_parallel_config: FusedMoEParallelConfig,
    is_lora_enabled: bool,
    model_type: str | None,
    is_mxfp4_quant: bool,
) -> int:
    """
    Given layer hidden size and MoE configurations, round up hidden_size
    if necessary.

    Args:
        hidden_size: Layer hidden-size
        act_dtype: Data type of the layer activations.
        moe_parallel_config: Fused MoE parallelization strategy configuration.
        is_lora_enabled: True if the engine is enabled with LoRA. This
            is used in the case of mxfp4 quantization in selecting the
            MxFP4Backend.
        model_type: for checking if gpt-oss
        is_mxfp4_quant: whether the layer is quantized with mxfp4

    Return:
        Rounded up hidden_size if rounding up is required based on the configs.
        Original hidden size otherwise.
    """
    from vllm.model_executor.layers.fused_moe.all2all_utils import (
        maybe_roundup_layer_hidden_size,
    )

    hidden_size = maybe_roundup_layer_hidden_size(
        hidden_size, act_dtype, moe_parallel_config
    )

    # we are padding globally so EP buffer allocation works
    if model_type == "gpt_oss" and is_mxfp4_quant:
        from vllm.model_executor.layers.quantization.mxfp4 import (
            Mxfp4Backend,
            get_mxfp4_backend,
        )

        current_mxfp4_backend = get_mxfp4_backend(is_lora_enabled)
        if (
            current_mxfp4_backend == Mxfp4Backend.SM90_FI_MXFP4_BF16
            or current_mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_MXFP8_CUTLASS
        ):
            hidden_size = round_up(hidden_size, 128)
        elif (
            current_platform.is_rocm()
            or current_mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_MXFP8_TRTLLM
            or current_mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_BF16
        ):
            hidden_size = round_up(hidden_size, 256)

    return hidden_size


# --8<-- [start:fused_moe]
@CustomOp.register("fused_moe")
class FusedMoE(CustomOp):
    """FusedMoE layer for MoE models.

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
        reduce_results: Whether to all_reduce on the output of the layer
        renormalize: Whether to renormalize the logits in the fused_moe kernel
        quant_config: Quantization configure.
        enable_eplb: Whether to enable expert parallelism load balancer.
        router_logits_dtype: Data type for router logits buffers.
    """

    # --8<-- [end:fused_moe]

    def __init__(
        self,
        num_experts: int,  # Global number of experts
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype | None = None,
        reduce_results: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: int | None = None,
        topk_group: int | None = None,
        quant_config: QuantizationConfig | None = None,
        tp_size: int | None = None,
        ep_size: int | None = None,
        dp_size: int | None = None,
        pcp_size: int | None = None,
        prefix: str = "",
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        is_act_and_mul: bool = True,
        enable_eplb: bool = False,
        num_redundant_experts: int = 0,
        has_bias: bool = False,
        is_sequence_parallel=False,
        expert_mapping: list[tuple[str, str, int, str]] | None = None,
        n_shared_experts: int | None = None,
        router_logits_dtype: torch.dtype | None = None,
        has_shared_experts: bool = False,
    ):
        super().__init__()

        # Allow disabling of the separate shared experts stream for
        # debug purposes.
        # TODO: Remove this after more extensive testings with TP/DP
        # and other execution modes
        if envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM:
            logger.debug_once("Disabling MoE shared_experts cuda stream", scope="local")
            self.shared_experts_stream = None
        else:
            # TODO(rob): enable shared expert overlap with non-cuda-alike.
            # aux_stream() returns None on non-cuda-alike platforms.
            self.shared_experts_stream = aux_stream()
            if self.shared_experts_stream is not None:
                logger.debug_once(
                    "Enabled separate cuda stream for MoE shared_experts", scope="local"
                )

        # For latent MoE: stores original hidden_states before routed_input_transform
        # so shared_experts can use it for cloning (they need original dimension)
        self._shared_experts_input: torch.Tensor | None = None

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        vllm_config = get_current_vllm_config()
        self.vllm_config = vllm_config

        # FIXME (varun): We should have a better way of inferring the activation
        # datatype. This works for now as the tensor datatype entering the MoE
        # operation is typically unquantized (i.e. float16/bfloat16).
        if vllm_config.model_config is not None:
            moe_in_dtype = vllm_config.model_config.dtype
        else:
            # TODO (bnell): This is a hack to get test_mixtral_moe to work
            # since model_config is not set in the pytest test.
            moe_in_dtype = params_dtype

        tp_size_ = (
            tp_size if tp_size is not None else get_tensor_model_parallel_world_size()
        )
        dp_size_ = dp_size if dp_size is not None else get_dp_group().world_size
        pcp_size_ = pcp_size if pcp_size is not None else get_pcp_group().world_size

        self.is_sequence_parallel = is_sequence_parallel
        self.sp_size = tp_size_ if is_sequence_parallel else 1

        self.moe_parallel_config: FusedMoEParallelConfig = FusedMoEParallelConfig.make(
            tp_size_=tp_size_,
            pcp_size_=pcp_size_,
            dp_size_=dp_size_,
            vllm_parallel_config=vllm_config.parallel_config,
        )

        self.global_num_experts = num_experts + num_redundant_experts
        self.logical_num_experts = num_experts

        # Expert mapping used in self.load_weights
        self.expert_mapping = expert_mapping

        # For smuggling this layer into the fused moe custom op
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError("Duplicate layer name: {}".format(prefix))
        compilation_config.static_forward_context[prefix] = self
        compilation_config.static_all_moe_layers.append(prefix)
        self.layer_name = prefix

        self.enable_eplb = enable_eplb
        self.eplb_state = EplbLayerState()
        self.expert_placement_strategy: ExpertPlacementStrategy = (
            vllm_config.parallel_config.expert_placement_strategy
        )

        # ROCm aiter shared experts fusion
        # AITER only supports gated activations (silu/gelu), so disable it
        # for non-gated MoE (is_act_and_mul=False)
        self.rocm_aiter_fmoe_enabled = (
            rocm_aiter_ops.is_fused_moe_enabled() and is_act_and_mul
        )
        self.aiter_fmoe_shared_expert_enabled = (
            rocm_aiter_ops.is_fusion_moe_shared_experts_enabled() and is_act_and_mul
        )

        self.num_fused_shared_experts = (
            n_shared_experts
            if n_shared_experts is not None and self.aiter_fmoe_shared_expert_enabled
            else 0
        )
        if (
            not self.aiter_fmoe_shared_expert_enabled
            and self.num_fused_shared_experts != 0
        ):
            raise ValueError(
                "n_shared_experts is only supported on ROCm aiter when "
                "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS is enabled"
            )

        # Determine expert maps
        if self.use_ep:
            if self.enable_eplb:
                assert self.global_num_experts % self.ep_size == 0, (
                    "EPLB currently only supports even distribution of "
                    "experts across ranks."
                )
            else:
                assert num_redundant_experts == 0, (
                    "Redundant experts are only supported with EPLB."
                )

            self.expert_placement_strategy = determine_expert_placement_strategy(
                expert_placement_strategy=self.expert_placement_strategy,
                moe_parallel_config=self.moe_parallel_config,
                num_expert_group=num_expert_group,
                num_redundant_experts=num_redundant_experts,
                enable_eplb=self.enable_eplb,
            )

            self._expert_map: torch.Tensor | None
            local_num_experts, expert_map, expert_mask = determine_expert_map(
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
                global_num_experts=self.global_num_experts,
                expert_placement_strategy=self.expert_placement_strategy,
                num_fused_shared_experts=self.num_fused_shared_experts,
                return_expert_mask=self.rocm_aiter_fmoe_enabled,
            )
            self.local_num_experts = local_num_experts
            self.register_buffer("_expert_map", expert_map)
            self.register_buffer("expert_mask", expert_mask)
            self._maybe_init_expert_routing_tables()
            logger.info_once(
                "[EP Rank %s/%s] Expert parallelism is enabled. Expert "
                "placement strategy: %s. Local/global"
                " number of experts: %s/%s. Experts local to global index map:"
                " %s.",
                self.ep_rank,
                self.ep_size,
                self.expert_placement_strategy,
                self.local_num_experts,
                self.global_num_experts,
                get_compressed_expert_map(self._expert_map),
            )
        else:
            self.local_num_experts, self._expert_map, self.expert_mask = (
                self.global_num_experts,
                None,
                None,
            )

        self.top_k = top_k

        self._init_aiter_shared_experts_topK_buffer(
            vllm_config=vllm_config, dp_size=dp_size_
        )
        if self.use_ep and self.rocm_aiter_fmoe_enabled:
            assert self.expert_mask is None or torch.all(
                (expert_mask == 0) | (expert_mask == 1)
            ), "Aiter Fused MoE kernel only supports expert_map with 0 and 1s."

        assert intermediate_size % self.tp_size == 0
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        self.reduce_results = reduce_results
        self.renormalize = renormalize

        # TODO(bnell): these attributes are only used by cpu/xpu/mxfp4
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.custom_routing_function = custom_routing_function
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.e_score_correction_bias = e_score_correction_bias
        # TODO(bnell): end attributes

        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.activation = activation

        self.router = create_fused_moe_router(
            top_k=top_k,
            global_num_experts=self.global_num_experts,
            eplb_state=self.eplb_state,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            num_fused_shared_experts=self.num_fused_shared_experts,
            enable_eplb=enable_eplb,
            # TODO(bnell): once we can construct the MK at init time, we
            # can make this a value.
            indices_type_getter=lambda: self.quant_method.topk_indices_dtype,
        )
        self.routing_method_type: RoutingMethodType = self.router.routing_method_type

        # Round up hidden size before creating moe_config.
        # This way moe_config is created with the correct hidden_size from the start.
        hidden_size = maybe_roundup_hidden_size(
            hidden_size=hidden_size,
            act_dtype=moe_in_dtype,
            moe_parallel_config=self.moe_parallel_config,
            is_lora_enabled=vllm_config.lora_config is not None,
            model_type=(
                self.vllm_config.model_config.hf_config.model_type
                if self.vllm_config.model_config is not None
                else None
            ),
            is_mxfp4_quant=(
                quant_config is not None and quant_config.is_mxfp4_quant(prefix, self)
            ),
        )
        self.hidden_size = hidden_size

        self.moe_config: FusedMoEConfig = FusedMoEConfig(
            num_experts=self.global_num_experts,
            experts_per_token=top_k,
            hidden_dim=hidden_size,
            intermediate_size_per_partition=self.intermediate_size_per_partition,
            num_local_experts=self.local_num_experts,
            moe_parallel_config=self.moe_parallel_config,
            in_dtype=moe_in_dtype,
            router_logits_dtype=router_logits_dtype,
            max_num_tokens=envs.VLLM_MOE_DP_CHUNK_SIZE,
            has_bias=has_bias,
            is_act_and_mul=is_act_and_mul,
            is_lora_enabled=vllm_config.lora_config is not None,
            activation=activation,
            device=vllm_config.device_config.device,
            routing_method=self.routing_method_type,
            # TODO: in_dtype == out_dtype?
            disable_inplace=disable_inplace() or has_shared_experts,
        )
        if self.use_mori_kernels:
            assert self.rocm_aiter_fmoe_enabled, (
                "Mori needs to be used with aiter fused_moe for now."
            )
            assert not self.aiter_fmoe_shared_expert_enabled, (
                "Mori does not support fusion shared expert now. "
                "Turn it off by setting VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0"
            )

        self.quant_config = quant_config

        def _get_quant_method() -> FusedMoEMethodBase:
            """
            Helper method to ensure self.quant_method is never None and
            of the proper type.
            """
            quant_method = None
            if self.quant_config is not None:
                quant_method = self.quant_config.get_quant_method(self, prefix)
            if quant_method is None:
                quant_method = UnquantizedFusedMoEMethod(self.moe_config)
            assert isinstance(quant_method, FusedMoEMethodBase)
            return quant_method

        # Note: get_quant_method will look at the layer's local_num_experts
        # for heuristic purposes, so it must be initialized first.
        self.quant_method: FusedMoEMethodBase = _get_quant_method()

        if not self.moe_config.is_act_and_mul and not current_platform.is_cuda_alike():
            raise NotImplementedError(
                "is_act_and_mul=False is supported only for CUDA and ROCm for now"
            )

        if self.enable_eplb and not self.quant_method.supports_eplb:
            # TODO: Add support for additional quantization methods.
            # The implementation for other quantization methods does not
            # contain essential differences, but the current quant API
            # design causes duplicated work when extending to new
            # quantization methods, so I'm leaving it for now.
            # If you plan to add support for more quantization methods,
            # please refer to the implementation in `Fp8MoEMethod`.
            raise NotImplementedError(
                f"EPLB is not supported {self.quant_method.__class__.__name__}."
            )

        moe_quant_params = {
            "num_experts": self.local_num_experts,
            "hidden_size": hidden_size,
            "intermediate_size_per_partition": self.intermediate_size_per_partition,
            "params_dtype": params_dtype,
            "weight_loader": self.weight_loader,
            "global_num_experts": self.global_num_experts,
        }
        # need full intermediate size pre-sharding for WNA16 act order
        if self.quant_method.__class__.__name__ in (
            "GPTQMarlinMoEMethod",
            "CompressedTensorsWNA16MarlinMoEMethod",
            "CompressedTensorsWNA16MoEMethod",
        ):
            moe_quant_params["intermediate_size_full"] = intermediate_size

        self.quant_method.create_weights(layer=self, **moe_quant_params)

        # Chunked all2all staging tensor
        self.batched_hidden_states: torch.Tensor | None = None
        self.batched_router_logits: torch.Tensor | None = None

    # Note: maybe_init_modular_kernel should only be called by
    # prepare_communication_buffer_for_model.
    # This is called after all weight loading and post-processing, so it
    # should be safe to swap out the quant_method.
    def maybe_init_modular_kernel(self) -> None:
        # NOTE(rob): WIP refactor. For quant methods that own the MK
        # we create the MK during process_weights_after_loading.
        if self.quant_method.supports_internal_mk or self.quant_method.is_monolithic:
            return None

        self.ensure_moe_quant_config_init()
        # routing_tables only needed for round-robin expert placement with
        # DeepEP all2all backend.
        routing_tables = self._maybe_init_expert_routing_tables()
        prepare_finalize = self.quant_method.maybe_make_prepare_finalize(
            routing_tables=routing_tables
        )
        if prepare_finalize is not None:
            logger.debug(
                "%s for %s(%s)", prepare_finalize.__class__.__name__, self, id(self)
            )
            self.quant_method = FusedMoEModularMethod.make(
                self,
                self.quant_method,
                prepare_finalize,
                self.shared_experts,
                inplace=not self.moe_config.disable_inplace,
            )

    @property
    def shared_experts(self) -> torch.nn.Module | None:
        return None

    @property
    def layer_id(self):
        # Delayed import to avoid circular dependency
        from vllm.model_executor.models.utils import extract_layer_index

        return extract_layer_index(self.layer_name)

    @property
    def gate(self) -> torch.nn.Module | None:
        return None

    def apply_routed_input_transform(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Hook to transform hidden_states before passing to routed experts.
        For latent MoE: transforms [S, hidden_size] → [S, moe_latent_size].
        The original hidden_states is saved in _shared_experts_input so
        shared_experts still receive the original [S, hidden_size].

        Override in subclasses (e.g., SharedFusedMoE) for latent MoE.
        """
        return hidden_states

    @contextmanager
    def _set_shared_experts_input(
        self, value: torch.Tensor | None
    ) -> Generator[None, None, None]:
        """Context manager to safely set/clear _shared_experts_input."""
        self._shared_experts_input = value
        try:
            yield
        finally:
            self._shared_experts_input = None

    def _get_shared_experts_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get input for shared experts.

        For latent MoE: shared_experts need original [S, hidden_size],
        not the transformed [S, latent_size] used by routed experts.
        """
        return (
            self._shared_experts_input
            if self._shared_experts_input is not None
            else hidden_states
        )

    @property
    def tp_size(self):
        return self.moe_parallel_config.tp_size

    @property
    def dp_size(self):
        return self.moe_parallel_config.dp_size

    @property
    def pcp_size(self):
        return self.moe_parallel_config.pcp_size

    @property
    def ep_size(self):
        return self.moe_parallel_config.ep_size

    @property
    def tp_rank(self):
        return self.moe_parallel_config.tp_rank

    @property
    def dp_rank(self):
        return self.moe_parallel_config.dp_rank

    @property
    def pcp_rank(self):
        return self.moe_parallel_config.pcp_rank

    @property
    def ep_rank(self):
        return self.moe_parallel_config.ep_rank

    @property
    def use_ep(self):
        return self.moe_parallel_config.use_ep

    @property
    def use_pplx_kernels(self):
        return self.moe_parallel_config.use_pplx_kernels

    @property
    def use_deepep_ht_kernels(self):
        return self.moe_parallel_config.use_deepep_ht_kernels

    @property
    def use_deepep_ll_kernels(self):
        return self.moe_parallel_config.use_deepep_ll_kernels

    @property
    def use_mori_kernels(self):
        return self.moe_parallel_config.use_mori_kernels

    @property
    def use_marlin_kernels(self):
        return getattr(self.quant_method, "use_marlin", False)

    @property
    def use_dp_chunking(self) -> bool:
        return (
            self.moe_parallel_config.use_pplx_kernels
            or self.moe_parallel_config.use_deepep_ll_kernels
            or self.moe_parallel_config.use_mori_kernels
            or self.moe_parallel_config.use_fi_all2allv_kernels
        ) and envs.VLLM_ENABLE_MOE_DP_CHUNK

    @property
    def is_internal_router(self) -> bool:
        # By default, router/gate is called before FusedMoE forward pass
        return False

    def _maybe_init_expert_routing_tables(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        # Currently routing_tables only needed for round-robin expert placement
        # with DeepEP-ll all2all backend.
        if (
            self.expert_placement_strategy != "round_robin"
            or not self.use_deepep_ll_kernels
        ):
            return None

        if hasattr(self, "expert_global_to_physical"):
            return cast(
                tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                (
                    self.expert_global_to_physical,
                    self.expert_physical_to_global,
                    self.expert_local_to_global,
                ),
            )

        if self._expert_map is None:
            return None

        routing_tables = self.ensure_round_robin_expert_routing_tables(
            global_num_experts=self.global_num_experts,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            local_num_experts=self.local_num_experts,
            device=self._expert_map.device,
        )

        global_to_physical, physical_to_global, local_global = routing_tables
        self.register_buffer("expert_global_to_physical", global_to_physical)
        self.register_buffer("expert_physical_to_global", physical_to_global)
        self.register_buffer("expert_local_to_global", local_global)

        return routing_tables

    @staticmethod
    def ensure_round_robin_expert_routing_tables(
        global_num_experts: int,
        ep_size: int,
        ep_rank: int,
        local_num_experts: int,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device_kwargs = {"device": device} if device is not None else {}
        global_indices = torch.arange(
            global_num_experts, dtype=torch.long, **device_kwargs
        )
        owner = torch.remainder(global_indices, ep_size)
        local_index = torch.div(global_indices, ep_size, rounding_mode="floor")
        base = global_num_experts // ep_size
        remainder = global_num_experts % ep_size
        physical_offset = owner * base
        if remainder > 0:
            remainder_tensor = torch.tensor(
                remainder, dtype=torch.long, **device_kwargs
            )
            physical_offset = physical_offset + torch.minimum(owner, remainder_tensor)

        global_to_physical = physical_offset + local_index
        physical_to_global = torch.empty_like(global_to_physical)
        physical_to_global[global_to_physical] = global_indices

        local_global = torch.arange(
            ep_rank,
            global_num_experts,
            ep_size,
            dtype=torch.long,
            **device_kwargs,
        )
        if local_global.numel() != local_num_experts:
            local_global = local_global[:local_num_experts]

        return (global_to_physical, physical_to_global, local_global)

    def update_expert_map(self):
        # ep_size and ep_rank should already be updated
        assert self._expert_map is not None
        with self._expert_map.device:
            local_num_experts, expert_map, expert_mask = determine_expert_map(
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
                global_num_experts=self.global_num_experts,
                expert_placement_strategy=self.expert_placement_strategy,
                num_fused_shared_experts=self.num_fused_shared_experts,
                return_expert_mask=self.rocm_aiter_fmoe_enabled,
            )
            self.local_num_experts = local_num_experts
            self.register_buffer("_expert_map", expert_map)
            self.register_buffer("expert_mask", expert_mask)
            self._maybe_init_expert_routing_tables()
            if self.aiter_fmoe_shared_expert_enabled:
                self._init_aiter_shared_experts_topK_buffer(
                    vllm_config=get_current_vllm_config(),
                    dp_size=get_dp_group().world_size,
                )

    def _maybe_setup_shared_experts_stream(
        self,
        hidden_states: torch.Tensor,
        has_separate_shared_experts: bool,
        use_chunked_impl: bool,
    ) -> tuple[bool, torch.Tensor | None]:
        use_shared_experts_stream = (
            current_platform.is_cuda()
            and has_separate_shared_experts
            and not use_chunked_impl
            and self.shared_experts_stream is not None
            and (
                hidden_states.shape[0]
                <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
            )
        )

        hidden_states_clone: torch.Tensor | None = None
        if use_shared_experts_stream:
            assert self.shared_experts_stream is not None

            shared_experts_input = self._get_shared_experts_input(hidden_states)

            # Clone BEFORE switching streams to avoid race condition
            # where routed_expert kernel may mutate hidden_states.
            hidden_states_clone = shared_experts_input.clone()

            # Record that the clone will be used by shared_experts_stream
            # to avoid gc issue from deallocation of hidden_states_clone
            # For more details: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html # noqa: E501
            # NOTE: We don't need shared_output.record_stream(current_stream())
            # because we synch the streams before using shared_output.
            hidden_states_clone.record_stream(self.shared_experts_stream)

            # Mark sync start point for the separate shared experts
            # stream here since we want to run in parallel with the
            # router/gate (next op below)
            assert self.shared_experts_stream is not None
            self.shared_experts_stream.wait_stream(current_stream())

        return use_shared_experts_stream, hidden_states_clone

    def _load_per_tensor_weight_scale(
        self,
        shard_id: str,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        expert_id: int,
    ):
        param_data = param.data
        # for per tensor weight quantization
        if shard_id in ("w1", "w3"):
            # We have to keep the weight scales of w1 and w3 because
            # we need to re-quantize w1/w3 weights after weight loading.
            idx = 0 if shard_id == "w1" else 1
            param_data[expert_id][idx] = loaded_weight
        # If we are in the row parallel case (down_proj)
        elif shard_id == "w2":
            param_data[expert_id] = loaded_weight

    def _load_combined_w13_weight_scale(
        self,
        shard_dim: int,
        loaded_weight: torch.Tensor,
        param: torch.Tensor,
        tp_rank: int,
    ):
        """
        Load w13 weight scales assuming that w1 weight scales and w3 weight
        scales are stored in the same loaded_weight tensor.
        """
        shard_size = param.shape[shard_dim]
        loaded_weight = loaded_weight.narrow(
            shard_dim, shard_size * tp_rank, shard_size
        )
        param.copy_(loaded_weight)

    def _load_model_weight_or_group_weight_scale(
        self,
        shard_dim: int,
        expert_data: torch.Tensor,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        load_full_w2: bool = False,
    ):
        """
        Load grouped weight scales for group quantization or model weights
            :param shard_dim: dimension to shard
            :param expert_data: parameter for a particular expert
            :param shard_id: either w1, w2, or w3
            :param loaded_weight: checkpoint weight to load into the param
            :param tp_rank: tensor parallel rank
            :param load_full_w2: whether or not the w2 loaded should be sharded.
        """
        if shard_id == "w2":
            # In the case where we have actorder/g_idx, we do not partition the
            # w2 scales, as indicated by `load_full` argument, for all tp cases
            self._load_w2(
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
                load_full=load_full_w2,
            )
        elif shard_id in ("w1", "w3"):
            self._load_w13(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )

    def _load_per_channel_weight_scale(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
    ):
        # for per channel weight quantization
        if shard_id == "w2":
            expert_data.copy_(loaded_weight)
        elif shard_id in ("w1", "w3"):
            self._load_w13(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )

    def _load_w13(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        load_full: bool = False,
    ):
        # Index the loaded weight for tp sharding.
        # gate_up_proj: "MergedColumnParallel", so tp sharding on output_dim
        if self.moe_config.is_act_and_mul:
            shard_size = expert_data.shape[shard_dim] // 2
        else:
            shard_size = expert_data.shape[shard_dim]
        # Only narrow if the loaded_weight is not a scalar (0-dim tensor)
        # and we're not loading the full weight
        if not load_full and loaded_weight.ndim > 0:
            loaded_weight = loaded_weight.narrow(
                shard_dim, shard_size * tp_rank, shard_size
            )
        # Narrow parameter and load.
        # w1, gate_proj: Load into first logical weight of w13.
        if shard_id == "w1":
            expert_data = expert_data.narrow(shard_dim, 0, shard_size)
        # w3, up_proj: Load into second logical weight of w13.
        else:
            assert shard_id == "w3"
            expert_data = expert_data.narrow(shard_dim, shard_size, shard_size)
        expert_data.copy_(loaded_weight)

    def _load_w2(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        load_full: bool = False,
    ):
        # Index the loaded weight for tp sharding.
        # down_proj: "RowParallel" so tp sharding on input_dim
        # Narrow parameter and load.
        shard_size = expert_data.shape[shard_dim]
        # Only narrow if the loaded_weight is not a scalar (0-dim tensor)
        # and we're not loading the full weight
        if not load_full and loaded_weight.ndim > 0:
            loaded_weight = loaded_weight.narrow(
                shard_dim, shard_size * tp_rank, shard_size
            )
        # w2, down_proj: Load into only logical weight of w2.
        expert_data.copy_(loaded_weight)

    def _load_single_value(
        self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int
    ):
        param_data = param.data

        # Input scales can be loaded directly and should be equal.
        param_data[expert_id] = loaded_weight

    def _load_g_idx(
        self,
        shard_id: str,
        expert_data: torch.Tensor,
        shard_dim: int,
        loaded_weight: torch.Tensor,
        tp_rank: int,
    ):
        if shard_id == "w2":
            self._load_w2(
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )
        else:
            assert shard_id in ("w1", "w3")
            expert_data.copy_(loaded_weight)

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        if self._expert_map is None:
            return expert_id
        return self._expert_map[expert_id].item()

    def _init_aiter_shared_experts_topK_buffer(
        self, vllm_config: VllmConfig, dp_size: int
    ):
        if self.num_fused_shared_experts > 0:
            init_aiter_topK_meta_data(
                n_routed_experts=self.global_num_experts,
                n_shared_experts=self.num_fused_shared_experts,
                top_k=self.top_k,
                tp_rank=self.ep_rank if self.use_ep else self.tp_rank,
                tp_size=self.ep_size if self.use_ep else self.tp_size,
                shared_experts_score=1.0,
                max_num_tokens=vllm_config.scheduler_config.max_num_batched_tokens
                * dp_size,
                is_EP=self.use_ep,
            )
        self.local_num_experts += self.num_fused_shared_experts

    @overload
    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: Literal[False],
    ) -> None: ...

    @overload
    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: Literal[True],
    ) -> bool: ...

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        if self.quant_config and self.quant_config.get_name() == "mxfp4":
            # (FIXME) for gpt-oss all experts are combined
            if "bias" in weight_name:
                dim1 = loaded_weight.shape[1]
                param.data[:, :dim1].copy_(loaded_weight)
            else:
                dim1 = loaded_weight.shape[1]
                dim2 = loaded_weight.shape[2]
                param.data[:, :dim1, :dim2].copy_(loaded_weight)
            return True if return_success else None

        quant_method_name = self.quant_method.__class__.__name__
        global_expert_id = expert_id
        expert_id = self._map_global_expert_id_to_local_expert_id(global_expert_id)

        use_global_sf = (
            getattr(self.quant_method, "use_global_sf", False)
            and "input_scale" in weight_name
        )

        if expert_id == -1 and not use_global_sf:
            # Failed to load this param since it's not local to this rank
            return False if return_success else None
        # Hereafter, `expert_id` is local physical id

        # is_transposed: if the dim to shard the weight
        # should be flipped. Required by GPTQ, compressed-tensors
        # should be whatever dimension intermediate_size_per_partition is
        is_transposed = getattr(param, "is_transposed", False)

        # compressed-tensors checkpoints with packed weights are stored flipped
        # TODO (mgoin): check self.quant_method.quant_config.quant_format
        # against known CompressionFormat enum values that have this quality
        if self.quant_method.__class__.__name__ in (
            "CompressedTensorsWNA16MarlinMoEMethod",
            "CompressedTensorsWNA16MoEMethod",
        ):
            if is_transposed:
                loaded_weight = loaded_weight.t().contiguous()
            else:
                loaded_weight = loaded_weight

        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"shard_id must be ['w1','w2','w3'] but got {shard_id}.")

        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size_per_partition is used.
        SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}

        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.weight_type = loaded_weight.item()
            param.data.copy_(loaded_weight)
            return True if return_success else None

        # Case for BitsAndBytes
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
        if use_bitsandbytes_4bit:
            shard_dim = 0

            expert_data = param.data[expert_id]
            if shard_id == "w2":
                expert_data.copy_(loaded_weight)
            elif shard_id in ("w1", "w3"):
                # BNB inflight quantization has already sharded the weights
                full_load = True
                self._load_w13(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank,
                    load_full=full_load,
                )
            return True if return_success else None

        shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
        if is_transposed:
            shard_dim = int(not shard_dim)

        full_load = len(loaded_weight.shape) == 3
        if full_load:
            shard_dim += 1

        # Materialize GGUF UninitializedParameter accounting merged weights
        if is_gguf_weight and isinstance(param, UninitializedParameter):
            # To materialize a tensor, we must have full shape including
            # number of experts, making this portion to require `full_load`.
            assert full_load
            final_shape = list(loaded_weight.shape)
            # w1 and w3 are merged per expert.
            if shard_id in {"w1", "w3"}:
                final_shape[1] *= 2
            final_shape[shard_dim] = final_shape[shard_dim] // self.tp_size
            param.materialize(final_shape, dtype=loaded_weight.dtype)

        expert_data = param.data if full_load else param.data[expert_id]

        # Case input scale: input_scale loading is only supported for fp8
        if "input_scale" in weight_name:
            # this is needed for compressed-tensors only
            loaded_weight = loaded_weight.to(param.data.device)

            if (
                "compressed" in quant_method_name.lower()
                and param.data[expert_id] != 1
                and (param.data[expert_id] - loaded_weight).abs() > 1e-5
            ):
                raise ValueError(
                    "input_scales of w1 and w3 of a layer "
                    f"must be equal. But got {param.data[expert_id]} "
                    f"vs. {loaded_weight}"
                )

            self._load_single_value(
                param=param,
                loaded_weight=loaded_weight,
                expert_id=global_expert_id if use_global_sf else expert_id,
            )
            return True if return_success else None

        # Case g_idx
        if "g_idx" in weight_name:
            self._load_g_idx(
                shard_dim=0,
                shard_id=shard_id,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=self.tp_rank,
            )
            return True if return_success else None

        # TODO @dsikka: ModelOpt should follow the proper MoE loading pattern
        if "ModelOpt" in quant_method_name:
            # Determine per-tensor weight scale patterns based on variant
            # Use the dedicated method instead of brittle string matching
            uses_weight_scale_2 = self.quant_method.uses_weight_scale_2_pattern()

            # Call _load_per_tensor_weight_scale() to load per-tensor (scalar)
            # weights scales.
            # Input scales are always per-tensor.
            # Weight scales: FP4 uses "weight_scale_2" and FP8 uses
            # "weight_scale" for per-tensor scales.
            is_per_tensor = (
                "weight_scale_2" in weight_name
                if uses_weight_scale_2
                else "weight_scale" in weight_name
            ) or "input_scale" in weight_name
            if is_per_tensor:
                self._load_per_tensor_weight_scale(
                    shard_id=shard_id,
                    param=param,
                    loaded_weight=loaded_weight,
                    expert_id=expert_id,
                )
                return True if return_success else None

            # If the weight is w13_weight_scale and w13_weight_scales are
            # combined into single loaded_weight, call
            # _load_combined_w13_weight_scale() to load it.
            # This is checked by comparing the hidden_out dims of the
            # loaded_weight and the param.
            if "w13_weight_scale" in weight_name:
                loaded_weight_hidden_out = loaded_weight.shape[-2]
                param_hidden_out = param.data.shape[-2] * self.tp_size
                if loaded_weight_hidden_out == param_hidden_out:
                    self._load_combined_w13_weight_scale(
                        shard_dim=shard_dim,
                        loaded_weight=loaded_weight,
                        param=expert_data,
                        tp_rank=self.tp_rank,
                    )
                    return True if return_success else None

            # For other weights, call _load_model_weight_or_group_weight_scale()
            # to load it.
            if "weight" in weight_name:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank,
                )
            return True if return_success else None

        # Case weight scales, zero_points and offset, weight/input global scales
        if "scale" in weight_name or "zero" in weight_name or "offset" in weight_name:
            # load the weight scales and zp based on the quantization scheme
            # supported weight scales/zp can be found in
            # FusedMoeWeightScaleSupported
            # TODO @dsikka: once hardened, refactor to use vLLM Parameters
            # specific to each case
            quant_method = getattr(param, "quant_method", None)
            if quant_method == FusedMoeWeightScaleSupported.CHANNEL.value:
                self._load_per_channel_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank,
                )
            elif quant_method in [
                FusedMoeWeightScaleSupported.GROUP.value,
                FusedMoeWeightScaleSupported.BLOCK.value,
            ]:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank,
                    load_full_w2=getattr(param, "load_full_w2", False),
                )
            elif quant_method == FusedMoeWeightScaleSupported.TENSOR.value:
                self._load_per_tensor_weight_scale(
                    shard_id=shard_id,
                    param=param,
                    loaded_weight=loaded_weight,
                    expert_id=expert_id,
                )
            else:
                WEIGHT_SCALE_SUPPORTED = [e.value for e in FusedMoeWeightScaleSupported]
                raise ValueError(
                    f"quant method must be one of {WEIGHT_SCALE_SUPPORTED}"
                )
            return True if return_success else None

        # Case weight_shape
        if "weight_shape" in weight_name:
            # only required by compressed-tensors
            self._load_single_value(
                param=param, loaded_weight=loaded_weight, expert_id=expert_id
            )
            return True if return_success else None

        # Case model weights
        if "weight" in weight_name:
            self._load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=self.tp_rank,
            )
            return True if return_success else None

        return False if return_success else None

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[str]:
        if (expert_mapping := self.expert_mapping) is None:
            raise ValueError(
                "`self.expert_mapping` must be provided to "
                "load weights using `self.load_weights`."
            )
        for expert_name, loaded_weight in weights:
            qual_name = f"{self.layer_name}.{expert_name}"
            for param_name, weight_name, expert_id, shard_id in expert_mapping:
                if weight_name not in qual_name:
                    continue
                weight_name = qual_name.replace(weight_name, param_name)
                param_name = weight_name.removeprefix(f"{self.layer_name}.")
                param = getattr(self, param_name)
                success = self.weight_loader(
                    param=param,
                    loaded_weight=loaded_weight,
                    weight_name=weight_name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                )
                if success:
                    logger.debug(
                        "Loaded %s for expert %d into %s",
                        param_name,
                        expert_id,
                        self.layer_name,
                    )
                    yield param_name

    def get_expert_weights(self) -> Iterable[torch.Tensor]:
        def _maybe_make_contiguous(
            name: str, p: torch.nn.Parameter
        ) -> torch.nn.Parameter:
            """
            In some cases, the last 2 dimensions (the non-expert dimensions)
            of the weight scale tensor are transposed. This function
            transforms the tensor (view update) so the tensor is contiguous().
            Example: A non-contiguous scale tensor,
              `x` of shape (E, 32, 16) and stride (512, 1, 32) is transformed to
              `x_` of shape (E, 16, 32) and stride (512, 32, 1).
              Note that we specifically use torch.transpose() so `x_` refers
              to the same underlying memory. The tensors `x` and `x_`, pointing
              to the same underlying memory make this transformation safe in the
              context of EPLB. i.e. It is the same memory and just the view
              is different.
            Note: This function handles the "weight_scale" tensors specifically.
            This could however be generalized to handle similar tensors.
            """
            if p.ndim != 3:
                return p
            if p.is_contiguous():
                # Already contiguous. do nothing.
                return p
            # p is non-contiguous. We only handle the case where the last 2
            # dimensions of the scales tensor is transposed. We can handle
            # other cases when they become relevant.
            is_transposed_12 = p.stride(1) == 1 and p.stride(2) != 1
            if "weight_scale" not in name or not is_transposed_12:
                # do nothing.
                return p

            # Do not update the layer parameter as the layer's MoE operations would
            # expect the parameter's tensor to the same shape / stride. Instead,
            # make a new torch.nn.Parameter that is used just in the context of
            # EPLB.
            return torch.nn.Parameter(
                torch.transpose(p.data, 1, 2), requires_grad=False
            )

        weights = list(self.named_parameters())
        weights = [(name, _maybe_make_contiguous(name, p)) for name, p in weights]

        assert all(
            weight.is_contiguous()
            for name, weight in weights
            if not name.startswith("_shared_experts.")
        )

        # Filter out the non-expert weights.
        # `e_score_correction_bias` is a bias for each logical expert,
        # with shape (num_logical_experts,), not an expert weight.
        NON_EXPERT_WEIGHTS = {
            "e_score_correction_bias",
        }

        return [
            weight.view(self.local_num_experts, -1)
            for name, weight in weights
            if name not in NON_EXPERT_WEIGHTS
            and weight.shape != torch.Size([])
            and not name.startswith("_shared_experts.")
            # exclude parameters from non-expert submodules (e.g. gate/shared)
            and not name.startswith("_gate.")
        ]

    def set_eplb_state(
        self,
        moe_layer_idx: int,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        """
        Register the EPLB state in this layer.

        This is used later in forward pass, where we get the expert mapping
        and record the load metrics in `expert_load_view`.
        """
        self.eplb_state.expert_load_view = expert_load_view[moe_layer_idx]
        self.eplb_state.logical_to_physical_map = logical_to_physical_map[moe_layer_idx]
        self.eplb_state.logical_replica_count = logical_replica_count[moe_layer_idx]

    def ensure_moe_quant_config_init(self):
        if self.quant_method.moe_quant_config is None:
            # Note: the moe_quant_config can't be constructed until after
            # weight loading post processing.
            self.quant_method.moe_quant_config = (
                self.quant_method.get_fused_moe_quant_config(self)
            )

    @property
    def moe_quant_config(self) -> FusedMoEQuantConfig | None:
        self.ensure_moe_quant_config_init()
        return self.quant_method.moe_quant_config

    def ensure_dp_chunking_init(self):
        if not self.use_dp_chunking or self.batched_hidden_states is not None:
            return

        states_shape: tuple[int, ...]
        logits_shape: tuple[int, ...]

        moe = self.moe_config

        if self.vllm_config.parallel_config.enable_dbo:
            states_shape = (2, moe.max_num_tokens, self.hidden_size)
            logits_shape = (2, moe.max_num_tokens, self.logical_num_experts)
        else:
            states_shape = (moe.max_num_tokens, self.hidden_size)
            logits_shape = (moe.max_num_tokens, self.logical_num_experts)

        self.batched_hidden_states = torch.zeros(
            states_shape, dtype=moe.in_dtype, device=torch.cuda.current_device()
        )

        self.batched_router_logits = torch.zeros(
            logits_shape,
            dtype=moe.router_logits_dtype,
            device=torch.cuda.current_device(),
        )

    def must_reduce_shared_expert_outputs(self) -> bool:
        """
        The shared_experts are typically computed using the RowParallelLinear
        layer. The result of this function is typically used as
        the reduce_results argument to the module.
        When just tensor-parallel is used, it is not required to reduce
        the shared_experts results immediately. Instead we reduce at the
        once at the end of the MoE op. (Refer to DeepSeekV2MoE module)
        With EP and all2all kernels - this is no longer viable as all
        GPU ranks in DP, produce the complete set of hidden_states.
        Therefore it is required that we reduce the shared_experts output
        early.
        """
        assert self.quant_method is not None
        return (
            isinstance(self.quant_method, FusedMoEModularMethod)
            and self.quant_method.moe_mk.output_is_reduced()  # type: ignore[union-attr]
        )

    def maybe_all_reduce_tensor_model_parallel(self, final_hidden_states: torch.Tensor):
        """
        Some combine kernels reduce across GPU ranks by default.
        """
        if self.must_reduce_shared_expert_outputs():
            return final_hidden_states
        else:
            return tensor_model_parallel_all_reduce(final_hidden_states)

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # For latent MoE: save ORIGINAL hidden_states before transform
        # (shared_experts need original dimension, routed experts use transformed)
        original_hidden_states = hidden_states
        original_hidden_dim = hidden_states.shape[-1]

        # Apply transform for routed experts (e.g., latent projection for latent MoE)
        hidden_states = self.apply_routed_input_transform(hidden_states)

        # This is the dimension after transform (for routed expert output slicing)
        transformed_hidden_dim = hidden_states.shape[-1]
        if self.hidden_size != transformed_hidden_dim:
            hidden_states = F.pad(
                hidden_states,
                (0, self.hidden_size - transformed_hidden_dim),
                mode="constant",
                value=0.0,
            )

        def reduce_output(states: torch.Tensor) -> torch.Tensor:
            if (
                not self.is_sequence_parallel
                and not self.use_dp_chunking
                and self.reduce_results
                and (self.tp_size > 1 or self.ep_size > 1)
            ):
                states = self.maybe_all_reduce_tensor_model_parallel(states)
            return states

        def encode_layer_name() -> str:
            # Can be unavailable or None in unittests
            if (
                is_forward_context_available()
                and get_forward_context().all_moe_layers is not None
            ):
                return "from_forward_context"
            return self.layer_name

        if self.shared_experts is None:
            if current_platform.is_tpu() or current_platform.is_cpu():
                # TODO: Once the OOM issue for the TPU backend is resolved, we
                # will switch to using the moe_forward custom op.
                # Note: CPU doesn't require wrapped forward_impl.
                fused_output = self.forward_impl(hidden_states, router_logits)
                assert not isinstance(fused_output, tuple)
            else:
                fused_output = torch.ops.vllm.moe_forward(
                    hidden_states, router_logits, encode_layer_name()
                )
            return reduce_output(fused_output)[..., :transformed_hidden_dim]
        else:
            if current_platform.is_tpu() or current_platform.is_cpu():
                # TODO: Once the OOM issue for the TPU backend is resolved, we
                # will switch to using the moe_forward custom op.
                # Note: CPU doesn't require wrapped forward_impl.
                with self._set_shared_experts_input(original_hidden_states):
                    shared_output, fused_output = self.forward_impl(
                        hidden_states, router_logits
                    )
            else:
                # Custom op handles setting/clearing _shared_experts_input internally
                # We pass original tensor for shared experts (not transformed)
                shared_output, fused_output = torch.ops.vllm.moe_forward_shared(
                    hidden_states,
                    router_logits,
                    encode_layer_name(),
                    original_hidden_states,
                )

            # shared_output uses original dimension (before transform)
            # fused_output uses transformed dimension (after transform)
            return (
                reduce_output(shared_output)[..., :original_hidden_dim],
                reduce_output(fused_output)[..., :transformed_hidden_dim],
            )

    @property
    def expert_map(self) -> torch.Tensor | None:
        return (
            self._expert_map if not self.rocm_aiter_fmoe_enabled else self.expert_mask
        )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.forward_native(hidden_states, router_logits)

    def forward_impl_chunked(
        self,
        full_hidden_states: torch.Tensor,
        full_router_logits: torch.Tensor,
        has_separate_shared_experts: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert self.batched_hidden_states is not None
        assert self.batched_router_logits is not None
        assert self.batched_hidden_states.dtype == full_hidden_states.dtype, (
            f"{self.batched_hidden_states.dtype} == {full_hidden_states.dtype}"
        )
        assert self.batched_router_logits.dtype == full_router_logits.dtype, (
            f"{self.batched_router_logits.dtype} == {full_router_logits.dtype}"
        )
        # Check size compatibility.
        assert self.batched_hidden_states.size(-1) == full_hidden_states.size(-1)
        assert self.batched_router_logits.size(-1) == full_router_logits.size(-1)

        full_fused_final_hidden_states = torch.empty_like(full_hidden_states)
        if self.shared_experts is not None:
            full_shared_final_hidden_states = torch.empty_like(full_hidden_states)

        def process_chunk(chunk_start, chunk_end, skip_result_store=False):
            chunk_size = chunk_end - chunk_start
            hidden_states = full_hidden_states[chunk_start:chunk_end, :]
            router_logits = full_router_logits[chunk_start:chunk_end, :]

            assert self.batched_hidden_states is not None
            assert self.batched_router_logits is not None
            # This is only true when DBO has been enabled in the config.
            # Both tensors will have an outer dimension for the ubatch id
            if self.batched_hidden_states.dim() == 3:
                assert self.batched_router_logits.dim() == 3
                batch_buffer_idx = dbo_current_ubatch_id()
                batched_hidden_states = self.batched_hidden_states[batch_buffer_idx, :]
                batched_router_logits = self.batched_router_logits[batch_buffer_idx, :]
            else:
                batched_hidden_states = self.batched_hidden_states
                batched_router_logits = self.batched_router_logits

            assert (
                batched_hidden_states.size(0)  # type: ignore
                >= chunk_size
            )
            assert (
                batched_router_logits.size(0)  # type: ignore
                >= chunk_size
            )
            staged_hidden_states = batched_hidden_states[:chunk_size, :]  # type: ignore
            staged_router_logits = batched_router_logits[:chunk_size, :]  # type: ignore
            staged_hidden_states.copy_(hidden_states, non_blocking=True)
            staged_router_logits.copy_(router_logits, non_blocking=True)

            # Matrix multiply.
            if self.quant_method.is_monolithic:
                final_hidden_states = self.quant_method.apply_monolithic(
                    layer=self,
                    x=staged_hidden_states,
                    router_logits=staged_router_logits,
                )
            else:
                topk_weights, topk_ids = self.router.select_experts(
                    hidden_states=staged_hidden_states,
                    router_logits=staged_router_logits,
                )

                final_hidden_states = self.quant_method.apply(
                    layer=self,
                    x=staged_hidden_states,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                )

            if has_separate_shared_experts:
                assert not isinstance(final_hidden_states, tuple)
                assert self.shared_experts is not None

                shared_output = self.shared_experts(staged_hidden_states)

                final_hidden_states = (
                    shared_output,
                    final_hidden_states,
                )

            if not skip_result_store:
                if self.shared_experts is None:
                    full_fused_final_hidden_states[chunk_start:chunk_end, :].copy_(
                        final_hidden_states, non_blocking=True
                    )
                else:
                    full_shared_final_hidden_states[chunk_start:chunk_end, :].copy_(
                        final_hidden_states[0], non_blocking=True
                    )
                    full_fused_final_hidden_states[chunk_start:chunk_end, :].copy_(
                        final_hidden_states[1], non_blocking=True
                    )

        ctx = get_forward_context()
        # flashinfer_cutlass_kernels can handle: optional DP + TP/EP
        max_tokens_across_dispatchers = ctx.dp_metadata.max_tokens_across_dp_cpu
        moe_dp_chunk_size_per_rank = self.moe_config.max_num_tokens

        # If the input to the MoE is sequence parallel then divide by sp_size
        # to find the maximum number of tokens for any individual dispatcher.
        if self.is_sequence_parallel:
            max_tokens_across_dispatchers = cdiv(
                max_tokens_across_dispatchers, self.sp_size
            )

        num_tokens = full_hidden_states.size(0)
        for chunk_idx, chunk_start_ in enumerate(
            range(0, max_tokens_across_dispatchers, moe_dp_chunk_size_per_rank)
        ):
            chunk_start = chunk_start_
            chunk_end = min(
                chunk_start + moe_dp_chunk_size_per_rank, max_tokens_across_dispatchers
            )
            # clamp start and end
            chunk_start = min(chunk_start, num_tokens - 1)
            chunk_end = min(chunk_end, num_tokens)
            with ctx.dp_metadata.chunked_sizes(
                self.sp_size, moe_dp_chunk_size_per_rank, chunk_idx
            ):
                process_chunk(
                    chunk_start, chunk_end, skip_result_store=chunk_start_ >= num_tokens
                )

        if self.shared_experts is None:
            return full_fused_final_hidden_states
        else:
            return (full_shared_final_hidden_states, full_fused_final_hidden_states)

    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass implementation for the fused MoE layer."""
        assert self.quant_method is not None

        self.ensure_moe_quant_config_init()
        self.ensure_dp_chunking_init()

        has_separate_shared_experts = (
            not self.quant_method.mk_owns_shared_expert
            and self.shared_experts is not None
        )

        use_chunked_impl = self.use_dp_chunking

        use_shared_experts_stream, hidden_states_clone = (
            self._maybe_setup_shared_experts_stream(
                hidden_states, has_separate_shared_experts, use_chunked_impl
            )
        )

        # If router/gate provided, then apply it here.
        # (Note: This code runs only when "overlapped mode" is on to allow
        #        parallel execution of shared experts with the FusedMoE via
        #        separate cuda stream)
        if self.gate is not None:
            router_logits, _ = self.gate(hidden_states)

        if use_chunked_impl:
            return self.forward_impl_chunked(
                hidden_states, router_logits, has_separate_shared_experts
            )

        # NOTE(rob): once we finish migrating all the quant methods to use
        # MKs, we can remove the naive dispatch/combine path from here.
        do_naive_dispatch_combine = (
            self.dp_size > 1 and not self.quant_method.supports_internal_mk
        )

        ctx = get_forward_context()
        sp_ctx = (
            ctx.dp_metadata.sp_local_sizes(self.sp_size)
            if ctx.dp_metadata
            else nullcontext()
        )

        with sp_ctx:
            extra_tensors = None
            if do_naive_dispatch_combine:
                post_quant_allgather = (
                    self.quant_method is not None
                    and self.dp_size > 1
                    and self.use_ep
                    and getattr(self.quant_method, "do_post_quant_allgather", False)
                )
                if post_quant_allgather:
                    hidden_states_to_dispatch, extra_tensors = (
                        self.quant_method.prepare_dp_allgather_tensor(
                            self, hidden_states, router_logits
                        )
                    )
                else:
                    hidden_states_to_dispatch = hidden_states

                dispatch_res = get_ep_group().dispatch_router_logits(
                    hidden_states_to_dispatch,
                    router_logits,
                    self.is_sequence_parallel,
                    extra_tensors=extra_tensors,
                )
                if extra_tensors is not None:
                    (
                        orig_hidden_states,
                        router_logits,
                        extra_tensors_combined,
                    ) = dispatch_res
                    hidden_states_combined = (
                        orig_hidden_states,
                        extra_tensors_combined[0],
                    )
                else:
                    hidden_states_combined, router_logits = dispatch_res
                    orig_hidden_states = hidden_states_combined
            else:
                orig_hidden_states = hidden_states

            # Run shared experts before matrix multiply.
            # because matrix multiply maybe modify the hidden_states.
            if has_separate_shared_experts and not use_shared_experts_stream:
                assert self.shared_experts is not None
                shared_input = self._get_shared_experts_input(hidden_states)
                shared_output = self.shared_experts(shared_input)

            # NOTE: Similar with DP, PCP also needs dispatch and combine. For
            # simplicity, AgRsAll2All was added separately for PCP here. Maybe
            # we should modify All2AllManager abstract to better support PCP.
            if self.pcp_size > 1:
                hidden_states = get_pcp_group().all_gather(
                    hidden_states,
                    dim=0,
                )
                router_logits = get_pcp_group().all_gather(
                    router_logits,
                    dim=0,
                )

            # Matrix multiply.
            x = hidden_states_combined if do_naive_dispatch_combine else hidden_states

            # TODO(bnell): deal with fp4 flashinfer tuple hidden states hack (#30014).
            # Figure out nicer way to do this.
            x_orig = orig_hidden_states if do_naive_dispatch_combine else hidden_states

            if self.quant_method.is_monolithic:
                final_hidden_states = self.quant_method.apply_monolithic(
                    layer=self,
                    x=x,
                    router_logits=router_logits,
                )
            else:
                topk_weights, topk_ids = self.router.select_experts(
                    hidden_states=x_orig,
                    router_logits=router_logits,
                )

                final_hidden_states = self.quant_method.apply(
                    layer=self,
                    x=x,  # The type signture of this is wrong due to the hack.
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                )

            if has_separate_shared_experts:
                assert self.shared_experts is not None

                if use_shared_experts_stream:
                    # Run shared experts in parallel on a separate stream
                    # NOTE: We start the separate stream here and mark the
                    # sync end point immediately after it is done. This is
                    # important to avoid excessive stream allocations by the cuda
                    # graph replay later.
                    with torch.cuda.stream(self.shared_experts_stream):
                        # Note that hidden_states clone() is necessary here to avoid
                        # conflict with the main stream
                        shared_output = self.shared_experts(hidden_states_clone)
                    current_stream().wait_stream(self.shared_experts_stream)

                final_hidden_states = (
                    shared_output,
                    final_hidden_states,
                )

            def combine_output(states: torch.Tensor) -> torch.Tensor:
                if do_naive_dispatch_combine:
                    states = get_ep_group().combine(states, self.is_sequence_parallel)

                if self.pcp_size > 1:
                    states = get_pcp_group().reduce_scatter(
                        states,
                        dim=0,
                    )

                return states

            if self.shared_experts is not None:
                return (
                    final_hidden_states[0],
                    combine_output(final_hidden_states[1]),
                )
            else:
                return combine_output(final_hidden_states)

    @classmethod
    def make_expert_params_mapping(
        cls,
        model: torch.nn.Module,
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str,
        num_experts: int,
        num_redundant_experts: int = 0,
    ) -> list[tuple[str, str, int, str]]:
        num_physical_experts = num_experts + num_redundant_experts

        # In the returned mapping:
        # - `expert_id` is the physical expert id
        # - `weight_name` contains the weight name of the logical expert
        # So that we should map the expert id to logical in `weight_name`
        physical_to_logical_map = (
            EplbState.build_initial_global_physical_to_logical_map(
                num_experts, num_redundant_experts
            )
        )

        base_layer = (
            "base_layer."
            if any(".base_layer." in name for name, _ in model.named_parameters())
            else ""
        )

        return [
            # (param_name, weight_name, expert_id, shard_id)
            (
                f"experts.{base_layer}w13_"
                if weight_name in [ckpt_gate_proj_name, ckpt_up_proj_name]
                else f"experts.{base_layer}w2_",
                f"experts.{physical_to_logical_map[expert_id]}.{weight_name}.{base_layer}",
                expert_id,
                shard_id,
            )
            for expert_id in range(num_physical_experts)
            for shard_id, weight_name in [
                ("w1", ckpt_gate_proj_name),
                ("w2", ckpt_down_proj_name),
                ("w3", ckpt_up_proj_name),
            ]
        ]

    def extra_repr(self) -> str:
        s = (
            f"global_num_experts={self.global_num_experts}, "
            f"local_num_experts={self.local_num_experts}, "
            f"top_k={self.top_k}, "
            f"intermediate_size_per_partition={self.intermediate_size_per_partition}, "  # noqa: E501
            f"tp_size={self.tp_size},\n"
            f"ep_size={self.ep_size}, "
            f"reduce_results={self.reduce_results}, "
        )

        return s


def get_layer_from_name(layer_name: str) -> FusedMoE:
    forward_context: ForwardContext = get_forward_context()
    if layer_name == "from_forward_context":
        all_moe_layers = forward_context.all_moe_layers
        assert all_moe_layers is not None
        moe_layer_index = forward_context.moe_layer_index
        if moe_layer_index >= len(all_moe_layers):
            raise AssertionError(
                "We expected the number of MOE layers in `all_moe_layers` "
                "to be equal to the number of "
                "{vllm.moe_forward, vllm.moe_forward_shared} calls."
            )
        layer_name = all_moe_layers[moe_layer_index]
        forward_context.moe_layer_index += 1
    self = cast(FusedMoE, forward_context.no_compile_layers[layer_name])
    return self


def moe_forward(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    self = get_layer_from_name(layer_name)
    assert self.shared_experts is None
    return self.forward_impl(hidden_states, router_logits)


def moe_forward_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="moe_forward",
    op_func=moe_forward,
    mutates_args=["hidden_states"],
    fake_impl=moe_forward_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def moe_forward_shared(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
    shared_experts_input: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    self = get_layer_from_name(layer_name)
    assert self.shared_experts is not None

    # Set here because torch.compile skips forward_native() setup code
    # and calls this op directly. forward_impl() reads from this var.
    with self._set_shared_experts_input(shared_experts_input):
        return self.forward_impl(hidden_states, router_logits)


def moe_forward_shared_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
    shared_experts_input: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Output shapes:
    # - fused_out: same as hidden_states (routed experts use transformed size)
    # - shared_out: same as shared_experts_input if provided, else same as hidden_states
    # (For latent MoE: shared experts use original hidden_size, not latent size)
    fused_out = torch.empty_like(hidden_states)

    if shared_experts_input is not None:
        shared_out = torch.empty_like(shared_experts_input)
    else:
        shared_out = torch.empty_like(hidden_states)

    return shared_out, fused_out


direct_register_custom_op(
    op_name="moe_forward_shared",
    op_func=moe_forward_shared,
    mutates_args=["hidden_states"],
    fake_impl=moe_forward_shared_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)

# Mark the FusedMoE weight_loader as supporting MoE-specific parameters
# to avoid expensive runtime reflection in model loading code
FusedMoE.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]
