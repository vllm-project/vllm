# SPDX-License-Identifier: Apache-2.0

import importlib
import threading
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional
from weakref import WeakValueDictionary

import torch
import torch.nn.functional as F
from torch.nn.parameter import UninitializedParameter

import vllm.envs as envs
from vllm.config import ParallelConfig, get_current_vllm_config
from vllm.distributed import (get_dp_group, get_ep_group,
                              get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
    is_rocm_aiter_moe_enabled)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.platforms.interface import CpuArchEnum
from vllm.utils import direct_register_custom_op

has_pplx = importlib.util.find_spec("pplx_kernels") is not None

if current_platform.is_cuda_alike():
    from .fused_batched_moe import (BatchedPrepareAndFinalize,
                                    BatchedTritonExperts)
    from .fused_moe import TritonExperts, fused_experts
    from .modular_kernel import (FusedMoEModularKernel,
                                 FusedMoEPermuteExpertsUnpermute,
                                 FusedMoEPrepareAndFinalize)
    if has_pplx:
        from .pplx_prepare_finalize import PplxPrepareAndFinalize
else:
    fused_experts = None  # type: ignore
if is_rocm_aiter_moe_enabled():
    from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (  # noqa: E501
        rocm_aiter_biased_group_topk as grouped_topk)
else:
    from vllm.model_executor.layers.fused_moe.fused_moe import grouped_topk
if current_platform.is_tpu():
    # the iterative moe implementation is used until the moe_pallas is fixed
    from .moe_torch_iterative import fused_moe as fused_moe_pallas
else:
    fused_moe_pallas = None  # type: ignore
logger = init_logger(__name__)

# Note: this limit is somewhat arbitrary and might be changed later.
# The size of the activations will be E x MOE_DP_CHUNK_SIZE x hidden_dim.
MOE_DP_CHUNK_SIZE = 256


@dataclass
class FusedMoEParallelConfig:
    tp_size: int
    dp_size: int
    ep_size: int
    tp_rank: int
    dp_rank: int
    ep_rank: int

    use_ep: bool  # whether to use EP or not

    @property
    def use_pplx_kernels(self):
        return self.dp_size > 1 and self.use_ep and has_pplx

    @staticmethod
    def make(tp_size_: int, dp_size_: int,
             vllm_parallel_config: ParallelConfig) -> "FusedMoEParallelConfig":
        """
        Determine MoE parallel configuration. Based on the input tp_size_,
        dp_size_, ep_size_ and vllm's parallel config, determine what
        level's of parallelism to use in the fused moe layer.

        Args:
            tp_size_ (int): tp_size passed into the FusedMoE constructor.
            dp_size_ (int): dp_size passed into the FusedMoE constructor.
            ep_size_ (int): ep_size passed into the FusedMoE constructor.
            vllm_parallel_config (ParallelConfig): vllm's parallel config
            object.

        Examples:
        When there is no parallelism requested, i.e. tp_size_ = dp_size_ = 1,
        we simply return the sizes unaltered and the ranks set to 0.

        Expert Parallelism is considered only when either dp_size_ or tp_size_
        is non trivial.

        When TP = 2, DP = 1 and EP = False, the configuration on different
        devices,
            - device 0 : TP = {2, 0} DP = {1, 0} EP = {1, 0} //
                         legend : {size, rank}
            - device 1 : TP = {2, 1} DP = {1, 0} EP = {1, 0}
            - Comment : Tensors are sharded across 2 devices.

        When TP = 1, DP = 2 and EP = False, the configuration on different
        devices,
            - device 0 : TP = {2, 0} DP = {2, 0} EP = {1, 0}
            - device 1 : TP = {2, 1} DP = {2, 1} EP = {1, 0}
            - Comment: There are 2 engine instances and the tensors are sharded
              across 2 decvices.

        When TP = 2, DP = 2 and EP = False, the configuration on different
        devices,
            - device 0: TP = {4, 0} DP = {2, 0} EP = {1, 0}
            - device 1: TP = {4, 1} DP = {2, 0} EP = {1, 0}
            - device 2: TP = {4, 2} DP = {2, 1} EP = {1, 0}
            - device 3: TP = {4, 3} DP = {2, 1} EP = {1, 0}
            - Comment: There are 2 engine instances and the tensors are sharded
              across 4 devices.

        When, TP = 2, DP = 1 and EP = True, the configuration on different
        devices,
            - device 0: TP = {1, 0} DP = {1, 0} EP = {2, 0}
            - device 1: TP = {1, 0} DP = {1, 0} EP = {2, 1}
            - Comment: The experts are split between the 2 devices.

        When, TP = 1, DP = 2 and EP = True, the configuration on different
        devices,
            - device 0: TP = {1, 0} DP = {2, 0} EP = {2, 0}
            - device 1: TP = {1, 0} DP = {2, 1} EP = {2, 1}
            - Comment: There are 2 engine instances and the experts are split
              between the 2 devices.

        When TP = 2, DP = 2 and EP = True, the configuration on different
        devices,
            - device 0: TP = {1, 0} DP = {2, 0} EP = {4, 0}
            - device 1: TP = {1, 0} DP = {2, 0} EP = {4, 1}
            - device 2: TP = {1, 0} DP = {2, 1} EP = {4, 2}
            - device 3: TP = {1, 0} DP = {2, 1} EP = {4, 3}
            - Comment: There are 2 engine instances and the experts are split
              between the 4 devices.
        """

        def flatten_tp_across_dp(dp_rank: int):
            tp_rank = 0 if tp_size_ == 1 else get_tensor_model_parallel_rank()
            # There are actually dp_size_ * tp_size_ devices. Update tp_size
            # and tp_rank so we shard across all devices.
            tp_size = dp_size_ * tp_size_
            tp_rank = dp_rank * tp_size_ + tp_rank
            return tp_size, tp_rank

        use_ep = (dp_size_ * tp_size_ > 1
                  and vllm_parallel_config.enable_expert_parallel)

        dp_size = dp_size_
        dp_rank = get_dp_group().rank_in_group if dp_size > 1 else 0
        tp_size, tp_rank = flatten_tp_across_dp(dp_rank)

        if not use_ep:
            return FusedMoEParallelConfig(tp_size=tp_size,
                                          tp_rank=tp_rank,
                                          dp_size=dp_size,
                                          dp_rank=dp_rank,
                                          ep_size=1,
                                          ep_rank=0,
                                          use_ep=False)
        # DP + EP / TP + EP / DP + TP + EP
        assert use_ep
        # In EP, each device owns a set of experts fully. There is no tensor
        # parallel update tp_size, tp_rank, ep_size and ep_rank to reflect that.
        ep_size = tp_size
        ep_rank = tp_rank
        return FusedMoEParallelConfig(tp_size=1,
                                      tp_rank=0,
                                      dp_size=dp_size,
                                      dp_rank=dp_rank,
                                      ep_size=ep_size,
                                      ep_rank=ep_rank,
                                      use_ep=True)


# Adapted from pplx-kernels tests/all_to_all_utils.py
@dataclass
class MoEConfig:
    num_experts: int
    experts_per_token: int
    hidden_dim: int

    num_local_experts: int
    moe_parallel_config: FusedMoEParallelConfig

    in_dtype: torch.dtype  # The activation type.

    # TODO: add more quantization params, blocked, per-token, etc.
    block_size: int = 128

    @property
    def tp_size(self):
        return self.moe_parallel_config.tp_size

    @property
    def dp_size(self):
        return self.moe_parallel_config.dp_size

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
    def ep_rank(self):
        return self.moe_parallel_config.ep_rank

    @property
    def use_ep(self):
        return self.moe_parallel_config.use_ep

    @property
    def use_pplx_kernels(self):
        return self.moe_parallel_config.use_pplx_kernels


class FusedMoeWeightScaleSupported(Enum):
    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"
    BLOCK = "block"


class FusedMoEMethodBase(QuantizeMethodBase):

    @abstractmethod
    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        raise NotImplementedError

    def set_prepare_finalize(
        self,
        dp_size: int,
        world_size: int,
        prepare_finalize: FusedMoEPrepareAndFinalize,
    ) -> bool:
        return False

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
    ) -> torch.Tensor:
        raise NotImplementedError


class AllToAllCache:

    def __init__(self):
        self._cache: WeakValueDictionary = WeakValueDictionary()
        self._lock = threading.RLock()  # Reentrant lock for thread safety

    def destroy(self):
        with self._lock:
            # TODO: can we do del self._cache?
            for _, a2a in self._cache.items():
                a2a.destroy()

    def get_or_create(self, **kwargs):
        assert has_pplx
        import pplx_kernels as pplx

        # Create a hashable key from the kwargs
        key = tuple(sorted((k, v) for k, v in kwargs.items()))

        with self._lock:
            instance = self._cache.get(key)
            if instance is None:
                # TODO (varun): Add support to switch to intranode
                # when all communications are within the same
                # node.
                logger.debug("Create AllToAll %s", kwargs)
                instance = pplx.AllToAll.internode(**kwargs)
                self._cache[key] = instance
            return instance


# Global singleton
_all_to_all_cache = AllToAllCache()


# Factory function as a cleaner interface
def get_all_to_all(**kwargs):
    return _all_to_all_cache.get_or_create(**kwargs)


@CustomOp.register("unquantized_fused_moe")
class UnquantizedFusedMoEMethod(FusedMoEMethodBase, CustomOp):
    """MoE method without quantization."""

    def __init__(self, moe: MoEConfig):
        super().__init__()
        self.fused_experts = fused_experts
        self.moe = moe

        self.rocm_aiter_moe_enabled = is_rocm_aiter_moe_enabled()
        if self.rocm_aiter_moe_enabled:
            from .rocm_aiter_fused_moe import rocm_aiter_fused_experts
            self.rocm_aiter_fused_experts = rocm_aiter_fused_experts
        else:
            self.rocm_aiter_fused_experts = None  # type: ignore

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
            dtype=params_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            dtype=params_dtype),
                                       requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

    def _maybe_pad_weight(self, weight: torch.Tensor) -> torch.Tensor:
        # Pad the weight tensor. This is an optimization on ROCm platform, which
        # can benefit from tensors located far enough from one another in memory
        if (envs.VLLM_ROCM_MOE_PADDING and current_platform.is_rocm()
                and weight.stride(-1) == 1
                and (weight.stride(-2) * weight.element_size()) % 512 == 0):
            num_pad = 256 // weight.element_size()
            weight = F.pad(weight, (0, num_pad), "constant", 0)[..., :-num_pad]
            torch.cuda.empty_cache()
        return weight

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)

        # Padding the weight for better performance on ROCm
        layer.w13_weight.data = self._maybe_pad_weight(layer.w13_weight.data)
        layer.w2_weight.data = self._maybe_pad_weight(layer.w2_weight.data)
        # Lazy import to avoid importing triton.
        from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
            shuffle_weights)

        if self.rocm_aiter_moe_enabled:
            # use 2stage ck moe layout
            shuffled_w13, shuffled_w2 = shuffle_weights(layer.w13_weight.data,
                                                        layer.w2_weight.data,
                                                        layout=(32, 32))

            layer.w13_weight.data = shuffled_w13
            layer.w2_weight.data = shuffled_w2

        if current_platform.is_cpu():
            if current_platform.get_cpu_architecture() == CpuArchEnum.X86:
                import intel_extension_for_pytorch as ipex
                layer.ipex_fusion = ipex.llm.modules.GatedMLPMOE(
                    layer.w13_weight,
                    layer.w2_weight,
                    use_prepack=envs.VLLM_CPU_MOE_PREPACK,
                )
            else:
                raise NotImplementedError("CPU MOE only supports x86 arch.")

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
    ) -> torch.Tensor:
        return self.forward(
            x=x,
            layer=layer,
            router_logits=router_logits,
            top_k=top_k,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input)

    def set_prepare_finalize(
        self,
        dp_size: int,
        world_size: int,
        prepare_finalize: FusedMoEPrepareAndFinalize,
    ) -> bool:
        assert self.fused_experts == fused_experts

        experts: Optional[FusedMoEPermuteExpertsUnpermute] = None

        if isinstance(prepare_finalize,
                      (BatchedPrepareAndFinalize, PplxPrepareAndFinalize)):
            logger.debug("BatchedTritonExperts %s", self.moe)
            experts = BatchedTritonExperts(
                max_num_tokens=MOE_DP_CHUNK_SIZE,
                world_size=world_size,
                dp_size=dp_size,
                use_fp8_w8a8=False,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                block_shape=None,
            )
        else:
            logger.debug("TritonExperts %s", self.moe)
            experts = TritonExperts(
                use_fp8_w8a8=False,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                block_shape=None,
                per_channel_quant=False,
            )

        self.fused_experts = FusedMoEModularKernel(
            prepare_finalize,
            experts,
        )

        return True

    def forward_cuda(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
    ) -> torch.Tensor:
        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            indices_type=torch.uint32 if self.moe.use_pplx_kernels else None)

        if self.rocm_aiter_moe_enabled:
            assert not apply_router_weight_on_input
            assert expert_map is None
            return self.rocm_aiter_fused_experts(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input)
        else:
            return self.fused_experts(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=True,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
            )

    def forward_cpu(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        **kwargs,
    ):
        assert activation == "silu", f"{activation} is not supported."
        assert apply_router_weight_on_input is False
        return layer.ipex_fusion(
            x,
            use_grouped_topk,
            top_k,
            router_logits,
            renormalize,
            topk_group,
            num_expert_group,
            custom_routing_function,
            scoring_func,
            e_score_correction_bias,
        )

    def forward_hpu(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
    ) -> torch.Tensor:
        assert not use_grouped_topk
        assert num_expert_group is None
        assert topk_group is None
        assert custom_routing_function is None
        assert layer is not None
        assert apply_router_weight_on_input is False
        if scoring_func != "softmax":
            raise NotImplementedError(
                "Only softmax scoring function is supported for HPU.")
        if e_score_correction_bias is not None:
            raise NotImplementedError(
                "Expert score correction bias is not supported for HPU.")
        return layer.hpu_fused_moe(x, layer.w13_weight, layer.w2_weight,
                                   router_logits, top_k)

    def forward_tpu(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
    ) -> torch.Tensor:
        assert not use_grouped_topk
        assert num_expert_group is None
        assert topk_group is None
        assert custom_routing_function is None
        assert apply_router_weight_on_input is False
        if scoring_func != "softmax":
            raise NotImplementedError(
                "Only softmax scoring function is supported for TPU.")
        if e_score_correction_bias is not None:
            raise NotImplementedError(
                "Expert score correction bias is not supported for TPU.")
        assert activation == "silu", f"{activation} is not supported for TPU."
        return fused_moe_pallas(hidden_states=x,
                                w1=layer.w13_weight,
                                w2=layer.w2_weight,
                                topk=top_k,
                                gating_output=router_logits,
                                global_num_experts=global_num_experts,
                                expert_map=expert_map,
                                renormalize=renormalize)

    forward_native = forward_tpu if current_platform.is_tpu() else forward_cuda


def determine_expert_map(
        ep_size: int, ep_rank: int,
        global_num_experts: int) -> tuple[int, Optional[torch.Tensor]]:
    """
        Calculates how many experts should be assigned to each rank for EP and
        creates a mapping from global to local expert index. Experts are
        distributed evenly across ranks. Any remaining are assigned to the
        last rank.

        Args:
            ep_size (int): The size of the expert parallel group
            global_num_experts (int): The total number of experts in the model.

        Returns:
            tuple[int, Optional[torch.Tensor]]: A tuple containing:
                - local_num_experts (int): The number of experts assigned
                    to the current rank.
                - expert_map (Optional[torch.Tensor]): A tensor of shape
                    (global_num_experts,) mapping from global to local index.
                    Contains -1 for experts not assigned to the current rank.
                    Returns None if ep_size is 1.
        """
    assert ep_size > 0
    if ep_size == 1:
        return (global_num_experts, None)

    local_num_experts = global_num_experts // ep_size

    # Create a tensor of size num_experts filled with -1
    expert_map = torch.full((global_num_experts, ), -1, dtype=torch.int32)
    # Create a expert map for the local experts
    if ep_rank < (ep_size - 1):
        # Each non-last rank gets local_num_experts experts.
        expert_map[ep_rank * local_num_experts:
                        (ep_rank + 1) * local_num_experts] = \
            torch.arange(0, local_num_experts, dtype=torch.int32)
    else:
        # All remaining experts are assigned to the last rank.
        local_num_experts = (global_num_experts - ep_rank * local_num_experts)

        expert_map[-local_num_experts:] = \
            torch.arange(0, local_num_experts, dtype=torch.int32)
    return (local_num_experts, expert_map)


def _construct_prepare_finalize(
    moe: MoEConfig, quant_config: Optional[QuantizationConfig]
) -> Optional[FusedMoEPrepareAndFinalize]:
    max_num_tokens = MOE_DP_CHUNK_SIZE
    world_size = moe.ep_size
    dp_size = moe.ep_size // moe.dp_size  # dp_size actually means TP.
    rank = moe.ep_rank

    if moe.use_pplx_kernels:
        logger.debug("using PplxPrepareAndFinalize")

        all_to_all = get_all_to_all(
            max_num_tokens=max_num_tokens,
            num_experts=moe.num_experts,
            experts_per_token=moe.experts_per_token,  # topk
            rank=rank,
            world_size=world_size,
            dp_size=dp_size,
            hidden_dim=moe.hidden_dim,
            hidden_dim_bytes=moe.hidden_dim * moe.in_dtype.itemsize,
            # For blocked per token: set to
            #   ceil_div(hidden_dim, block_size) * sizeof(float32)
            # For per-token: set to sizeof(float32)
            hidden_dim_scale_bytes=(0 if moe.in_dtype.itemsize != 1 else
                                    ((moe.hidden_dim + moe.block_size - 1) //
                                     moe.block_size * torch.float32.itemsize)))

        return PplxPrepareAndFinalize(
            all_to_all,
            max_num_tokens=max_num_tokens,
            world_size=world_size,
            rank=rank,
            dp_size=dp_size,
            quant_dtype=moe.in_dtype,
        )

    return None


class FusedMoE(torch.nn.Module):
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
        reduce_results: Whether to all all_reduce on the output of the layer
        renomalize: Whether to renormalize the logits in the fused_moe kernel
        quant_config: Quantization configure.
    """

    def __init__(
        self,
        num_experts: int,  # Global number of experts
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        ep_size: Optional[int] = None,
        dp_size: Optional[int] = None,
        prefix: str = "",
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
    ):
        super().__init__()

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        vllm_config = get_current_vllm_config()
        self.moe_parallel_config: FusedMoEParallelConfig = (
            FusedMoEParallelConfig.make(
                tp_size_=(tp_size if tp_size is not None else
                          get_tensor_model_parallel_world_size()),
                dp_size_=(dp_size if dp_size is not None else
                          get_dp_group().world_size),
                vllm_parallel_config=vllm_config.parallel_config))

        self.global_num_experts = num_experts

        # For smuggling this layer into the fused moe custom op
        self.use_direct_call = self.dp_size == 1
        if not self.use_direct_call:
            compilation_config = vllm_config.compilation_config
            if prefix in compilation_config.static_forward_context:
                raise ValueError("Duplicate layer name: {}".format(prefix))
            compilation_config.static_forward_context[prefix] = self
            self.layer_name = prefix

        # Determine expert maps
        if self.use_ep:
            self.local_num_experts, self.expert_map = determine_expert_map(
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
                global_num_experts=self.global_num_experts)
        else:
            self.local_num_experts, self.expert_map = (self.global_num_experts,
                                                       None)

        self.top_k = top_k

        assert intermediate_size % self.tp_size == 0
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        self.reduce_results = reduce_results
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.custom_routing_function = custom_routing_function
        self.scoring_func = scoring_func
        self.e_score_correction_bias = e_score_correction_bias
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.activation = activation

        if self.scoring_func != "softmax" and not self.use_grouped_topk:
            raise ValueError("Only softmax scoring function is supported for "
                             "non-grouped topk.")
        if current_platform.is_hpu():
            from vllm_hpu_extension.ops import DynamicFusedMOE
            self.hpu_fused_moe = DynamicFusedMOE(self.global_num_experts)

        moe = MoEConfig(
            num_experts=self.global_num_experts,
            experts_per_token=top_k,
            hidden_dim=hidden_size,
            num_local_experts=self.local_num_experts,
            moe_parallel_config=self.moe_parallel_config,
            # TODO (bnell): this needs to be fixed for quantized types.
            in_dtype=params_dtype,
        )

        # Note: get_quant_method will look at the layer's local_num_experts
        # for heuristic purposes, so it must be initialized first.
        quant_method: Optional[QuantizeMethodBase] = None

        if quant_config is None:
            quant_method = UnquantizedFusedMoEMethod(moe)
            prepare_finalize = _construct_prepare_finalize(moe, quant_config)
        else:
            quant_method = quant_config.get_quant_method(self, prefix)
            # No pplx for quantized types yet.
            prepare_finalize = None

        assert quant_method is not None
        assert isinstance(quant_method, FusedMoEMethodBase)
        self.quant_method = quant_method

        if prepare_finalize is not None:
            world_size = moe.ep_size
            dp_size = int(moe.ep_size // moe.dp_size)
            success = self.quant_method.set_prepare_finalize(
                dp_size, world_size, prepare_finalize)
            if not success:
                logger.warning("DP+EP not supported for %s.",
                               type(self.quant_method))

        moe_quant_params = {
            "num_experts": self.local_num_experts,
            "hidden_size": hidden_size,
            "intermediate_size_per_partition":
            self.intermediate_size_per_partition,
            "params_dtype": params_dtype,
            "weight_loader": self.weight_loader,
        }
        # need full intermediate size pre-sharding for WNA16 act order
        if (self.quant_method.__class__.__name__
                in ("GPTQMarlinMoEMethod",
                    "CompressedTensorsWNA16MarlinMoEMethod",
                    "CompressedTensorsWNA16MoEMethod")):
            moe_quant_params["intermediate_size_full"] = intermediate_size

        self.quant_method.create_weights(layer=self, **moe_quant_params)

    @property
    def tp_size(self):
        return self.moe_parallel_config.tp_size

    @property
    def dp_size(self):
        return self.moe_parallel_config.dp_size

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
    def ep_rank(self):
        return self.moe_parallel_config.ep_rank

    @property
    def use_ep(self):
        return self.moe_parallel_config.use_ep

    @property
    def use_pplx_kernels(self):
        return self.moe_parallel_config.use_pplx_kernels

    def _load_per_tensor_weight_scale(self, shard_id: str,
                                      param: torch.nn.Parameter,
                                      loaded_weight: torch.Tensor,
                                      expert_id: int):
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

    def _load_model_weight_or_group_weight_scale(self,
                                                 shard_dim: int,
                                                 expert_data: torch.Tensor,
                                                 shard_id: str,
                                                 loaded_weight: torch.Tensor,
                                                 tp_rank: int,
                                                 load_full_w2: bool = False):
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
            self._load_w2(shard_dim=shard_dim,
                          loaded_weight=loaded_weight,
                          expert_data=expert_data,
                          tp_rank=tp_rank,
                          load_full=load_full_w2)
        elif shard_id in ("w1", "w3"):
            self._load_w13(shard_id=shard_id,
                           shard_dim=shard_dim,
                           loaded_weight=loaded_weight,
                           expert_data=expert_data,
                           tp_rank=tp_rank)

    def _load_per_channel_weight_scale(self, expert_data: torch.Tensor,
                                       shard_dim: int, shard_id: str,
                                       loaded_weight: torch.Tensor,
                                       tp_rank: int):
        # for per channel weight quantization
        if shard_id == "w2":
            expert_data.copy_(loaded_weight)
        elif shard_id in ("w1", "w3"):
            self._load_w13(shard_id=shard_id,
                           shard_dim=shard_dim,
                           loaded_weight=loaded_weight,
                           expert_data=expert_data,
                           tp_rank=tp_rank)

    def _load_w13(self, expert_data: torch.Tensor, shard_dim: int,
                  shard_id: str, loaded_weight: torch.Tensor, tp_rank: int):

        # Index the loaded weight for tp sharding.
        # gate_up_proj: "MergedColumnParallel", so tp sharding on output_dim
        shard_size = expert_data.shape[shard_dim] // 2
        loaded_weight = loaded_weight.narrow(shard_dim, shard_size * tp_rank,
                                             shard_size)
        # Narrow parameter and load.
        # w1, gate_proj: Load into first logical weight of w13.
        if shard_id == "w1":
            expert_data = expert_data.narrow(shard_dim, 0, shard_size)
        # w3, up_proj: Load into second logical weight of w13.
        else:
            assert shard_id == "w3"
            expert_data = expert_data.narrow(shard_dim, shard_size, shard_size)
        expert_data.copy_(loaded_weight)

    def _load_w2(self,
                 expert_data: torch.Tensor,
                 shard_dim: int,
                 loaded_weight: torch.Tensor,
                 tp_rank: int,
                 load_full: bool = False):

        # Index the loaded weight for tp sharding.
        # down_proj: "RowParallel" so tp sharding on input_dim
        # Narrow parameter and load.
        shard_size = expert_data.shape[shard_dim]
        if not load_full:
            loaded_weight = loaded_weight.narrow(shard_dim,
                                                 shard_size * tp_rank,
                                                 shard_size)
        # w2, down_proj: Load into only logical weight of w2.
        expert_data.copy_(loaded_weight)

    def _load_single_value(self, param: torch.nn.Parameter,
                           loaded_weight: torch.Tensor, expert_id: int):
        param_data = param.data

        # Input scales can be loaded directly and should be equal.
        param_data[expert_id] = loaded_weight

    def _load_g_idx(self, shard_id: str, expert_data: torch.Tensor,
                    shard_dim: int, loaded_weight: torch.Tensor, tp_rank: int):

        if shard_id == "w2":
            self._load_w2(shard_dim=shard_dim,
                          loaded_weight=loaded_weight,
                          expert_data=expert_data,
                          tp_rank=tp_rank)
        else:
            assert shard_id in ("w1", "w3")
            expert_data.copy_(loaded_weight)

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        if self.expert_map is None:
            return expert_id
        return self.expert_map[expert_id].item()

    def weight_loader(self, param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor, weight_name: str,
                      shard_id: str, expert_id: int) -> None:

        expert_id = self._map_global_expert_id_to_local_expert_id(expert_id)
        if expert_id == -1:
            return
        quant_method_name = self.quant_method.__class__.__name__
        # compressed-tensors checkpoints with packed weights are stored flipped
        # TODO (mgoin): check self.quant_method.quant_config.quant_format
        # against known CompressionFormat enum values that have this quality
        if self.quant_method.__class__.__name__ in (
                "CompressedTensorsWNA16MarlinMoEMethod",
                "CompressedTensorsWNA16MoEMethod"):
            loaded_weight = loaded_weight.t().contiguous()

        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"shard_id must be ['w1','w2','w3'] but "
                             f"got {shard_id}.")

        WEIGHT_SCALE_SUPPORTED = [
            e.value for e in FusedMoeWeightScaleSupported
        ]
        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size_per_partition is used.
        SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}

        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.weight_type = loaded_weight.item()
            param.data.copy_(loaded_weight)
            return

        # is_transposed: if the dim to shard the weight
        # should be flipped. Required by GPTQ, compressed-tensors
        # should be whatever dimension intermediate_size_per_partition is
        is_transposed = getattr(param, "is_transposed", False)
        shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
        if is_transposed:
            shard_dim = int(not shard_dim)

        full_load = len(loaded_weight.shape) == 3
        if full_load:
            shard_dim += 1

        # Materialize GGUF UninitializedParameter
        if is_gguf_weight and isinstance(param, UninitializedParameter):
            final_shape = list(loaded_weight.shape)
            if shard_id in ["w1", "w3"]:
                final_shape[1] *= 2
            final_shape[shard_dim] = final_shape[shard_dim] // self.tp_size
            param.materialize(final_shape, dtype=loaded_weight.dtype)

        expert_data = param.data if full_load else param.data[expert_id]
        # Case input scale: input_scale loading is only supported for fp8
        if "input_scale" in weight_name:
            # this is needed for compressed-tensors only
            loaded_weight = loaded_weight.to(param.data.device)

            if ("compressed" in quant_method_name.lower()
                    and param.data[expert_id] != 1
                    and (param.data[expert_id] - loaded_weight).abs() > 1e-5):
                raise ValueError(
                    "input_scales of w1 and w3 of a layer "
                    f"must be equal. But got {param.data[expert_id]} "
                    f"vs. {loaded_weight}")

            self._load_single_value(param=param,
                                    loaded_weight=loaded_weight,
                                    expert_id=expert_id)
            return

        # Case g_idx
        if "g_idx" in weight_name:
            self._load_g_idx(shard_dim=0,
                             shard_id=shard_id,
                             loaded_weight=loaded_weight,
                             expert_data=expert_data,
                             tp_rank=self.tp_rank)
            return

        if "ModelOpt" in quant_method_name:
            if ('weight_scale_2' in weight_name
                    or 'input_scale' in weight_name):
                self._load_per_tensor_weight_scale(shard_id=shard_id,
                                                   param=param,
                                                   loaded_weight=loaded_weight,
                                                   expert_id=expert_id)
            elif "weight" in weight_name:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank)
            return

        # Case weight scales, zero_points and offset
        if ("scale" in weight_name or "zero" in weight_name
                or "offset" in weight_name):
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
                    tp_rank=self.tp_rank)
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
                    load_full_w2=getattr(param, "load_full_w2", False))
            elif quant_method == FusedMoeWeightScaleSupported.TENSOR.value:
                self._load_per_tensor_weight_scale(shard_id=shard_id,
                                                   param=param,
                                                   loaded_weight=loaded_weight,
                                                   expert_id=expert_id)
            else:
                raise ValueError(
                    f"quant method must be one of {WEIGHT_SCALE_SUPPORTED}")
            return

        # Case weight_shape
        if "weight_shape" in weight_name:
            # only required by compressed-tensors
            self._load_single_value(param=param,
                                    loaded_weight=loaded_weight,
                                    expert_id=expert_id)
            return

        # Case model weights
        if "weight" in weight_name:
            self._load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=self.tp_rank)
            return

    @staticmethod
    def select_experts(hidden_states: torch.Tensor,
                       router_logits: torch.Tensor,
                       top_k: int,
                       use_grouped_topk: bool,
                       renormalize: bool,
                       topk_group: Optional[int] = None,
                       num_expert_group: Optional[int] = None,
                       custom_routing_function: Optional[Callable] = None,
                       scoring_func: str = "softmax",
                       e_score_correction_bias: Optional[torch.Tensor] = None,
                       indices_type: Optional[torch.dtype] = None):
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk

        # DeekSeekv2 uses grouped_top_k
        if use_grouped_topk:
            assert topk_group is not None
            assert num_expert_group is not None
            topk_weights, topk_ids = grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                scoring_func=scoring_func,
                e_score_correction_bias=e_score_correction_bias)
            if indices_type is not None:
                topk_ids = topk_ids.to(dtype=indices_type)
        elif custom_routing_function is None:
            topk_weights, topk_ids, token_expert_indices = fused_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                indices_type=indices_type,
            )
        else:
            topk_weights, topk_ids = custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize)
            if indices_type is not None:
                topk_ids = topk_ids.to(dtype=indices_type)

        return topk_weights, topk_ids

    def must_reduce_shared_expert_outputs(self) -> bool:
        """
        The shared_experts are typically computed using the RowParallelLinear
        layer. The result of this function is typically used as
        the reduce_results argument to the module.
        When just tensor-parallel is used, it is not required to reduce
        the shared_experts results immediately. Instead we reduce at the
        once at the end of the MoE op. (Refer to DeepSeekV2MoE module)
        With EP and the pplx kernels - this is no longer viable as all
        GPU ranks in DP, produce the complete set of hidden_states.
        Therefore it is required that we reduce the shared_experts output
        early.
        """
        return self.use_pplx_kernels

    def maybe_all_reduce_tensor_model_parallel(
            self, final_hidden_states: torch.Tensor):
        """
        The pplx combine kernel reduces across GPU ranks by default.
        """
        if self.use_pplx_kernels:
            return final_hidden_states
        else:
            return tensor_model_parallel_all_reduce(final_hidden_states)

    def forward(self, hidden_states: torch.Tensor,
                router_logits: torch.Tensor):
        if self.use_direct_call:
            return self.forward_impl(hidden_states, router_logits)
        else:
            return torch.ops.vllm.moe_forward(hidden_states, router_logits,
                                              self.layer_name)

    def forward_impl_chunked(self, full_hidden_states: torch.Tensor,
                             full_router_logits: torch.Tensor):

        full_final_hidden_states = torch.empty_like(full_hidden_states)

        def process_chunk(chunk_start, chunk_end, skip_result_store=False):
            hidden_states = full_hidden_states[chunk_start:chunk_end, :]
            router_logits = full_router_logits[chunk_start:chunk_end, :]

            # Matrix multiply.
            final_hidden_states = self.quant_method.apply(
                layer=self,
                x=hidden_states,
                router_logits=router_logits,
                top_k=self.top_k,
                renormalize=self.renormalize,
                use_grouped_topk=self.use_grouped_topk,
                global_num_experts=self.global_num_experts,
                expert_map=self.expert_map,
                topk_group=self.topk_group,
                num_expert_group=self.num_expert_group,
                custom_routing_function=self.custom_routing_function,
                scoring_func=self.scoring_func,
                e_score_correction_bias=self.e_score_correction_bias,
                activation=self.activation,
            )

            if not skip_result_store:
                full_final_hidden_states[chunk_start:chunk_end, :].copy_(
                    final_hidden_states)

        ctx = get_forward_context()
        max_tokens_across_dp = ctx.dp_metadata.max_tokens_across_dp_cpu
        moe_dp_chunk_size_per_rank = MOE_DP_CHUNK_SIZE

        num_tokens = full_hidden_states.size(0)
        for chunk_start_ in range(0, max_tokens_across_dp,
                                  moe_dp_chunk_size_per_rank):
            chunk_start = chunk_start_
            chunk_end = min(chunk_start + moe_dp_chunk_size_per_rank,
                            max_tokens_across_dp)
            # clamp start and end
            chunk_start = min(chunk_start, num_tokens - 1)
            chunk_end = min(chunk_end, num_tokens)

            process_chunk(chunk_start,
                          chunk_end,
                          skip_result_store=chunk_start_ >= num_tokens)

        return full_final_hidden_states

    def forward_impl(self, hidden_states: torch.Tensor,
                     router_logits: torch.Tensor):
        assert self.quant_method is not None
        if self.moe_parallel_config.use_pplx_kernels:
            return self.forward_impl_chunked(hidden_states, router_logits)

        if self.dp_size > 1:
            hidden_states, router_logits = get_ep_group().dispatch(
                hidden_states, router_logits)
        # Matrix multiply.
        final_hidden_states = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            global_num_experts=self.global_num_experts,
            expert_map=self.expert_map,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
        )

        if self.dp_size > 1:
            final_hidden_states = get_ep_group().combine(final_hidden_states)

        if self.reduce_results and (self.tp_size > 1 or self.ep_size > 1):
            # Default set to False. (May have to add shared expert outputs.)
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)

        return final_hidden_states

    @classmethod
    def make_expert_params_mapping(
            cls, ckpt_gate_proj_name: str, ckpt_down_proj_name: str,
            ckpt_up_proj_name: str,
            num_experts: int) -> list[tuple[str, str, int, str]]:

        return [
            # (param_name, weight_name, expert_id, shard_id)
            ("experts.w13_" if weight_name
             in [ckpt_gate_proj_name, ckpt_up_proj_name] else "experts.w2_",
             f"experts.{expert_id}.{weight_name}.", expert_id, shard_id)
            for expert_id in range(num_experts) for shard_id, weight_name in [
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
            f"renormalize={self.renormalize}, "
            f"use_grouped_topk={self.use_grouped_topk}")

        if self.use_grouped_topk:
            s += f", num_expert_group={self.num_expert_group}, topk_group={self.topk_group}"  # noqa: E501

        s += f", scoring_func='{self.scoring_func}', activation='{self.activation}'"  # noqa: E501

        return s


def moe_forward(hidden_states: torch.Tensor, router_logits: torch.Tensor,
                layer_name: str) -> torch.Tensor:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    assert self.quant_method is not None

    return self.forward_impl(hidden_states, router_logits)


def moe_forward_fake(hidden_states: torch.Tensor, router_logits: torch.Tensor,
                     layer_name: str) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="moe_forward",
    op_func=moe_forward,
    mutates_args=[],
    fake_impl=moe_forward_fake,
    dispatch_key=current_platform.dispatch_key,
)
