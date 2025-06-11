# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from enum import Enum
from typing import Callable, List, Optional, Tuple

import torch
import os

import torch.distributed
from vllm.config import get_current_vllm_config
from vllm.distributed import (get_dp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.platforms.interface import CpuArchEnum
from vllm.utils import direct_register_custom_op

is_hpu = current_platform.is_hpu()

if current_platform.is_cuda_alike():
    from .fused_moe import fused_experts
else:
    fused_experts = None  # type: ignore
if current_platform.is_tpu():
    # the iterative moe implementation is used until the moe_pallas is fixed
    from .moe_torch_iterative import fused_moe as fused_moe_pallas
else:
    fused_moe_pallas = None  # type: ignore
logger = init_logger(__name__)


VLLM_REQUANT_FP8_INC = os.getenv("VLLM_REQUANT_FP8_INC", "0") in ["1", "true"]


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
        e_score_correction_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        raise NotImplementedError


@CustomOp.register("unquantized_fused_moe")
class UnquantizedFusedMoEMethod(FusedMoEMethodBase, CustomOp):
    """MoE method without quantization."""

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

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)

        if current_platform.is_cpu():
            if current_platform.get_cpu_architecture() == CpuArchEnum.X86:
                import intel_extension_for_pytorch as ipex
                layer.ipex_fusion = ipex.llm.modules.GatedMLPMOE(
                    layer.w13_weight,
                    layer.w2_weight,
                    use_prepack=True,
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
        ep_rank: Optional[int] = None,
    ) -> torch.Tensor:
        return self.forward(x=x,
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
                            ep_rank= ep_rank,
                            )

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
        e_score_correction_bias: Optional[torch.Tensor] = None
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
            e_score_correction_bias=e_score_correction_bias)

        return fused_experts(hidden_states=x,
                             w1=layer.w13_weight,
                             w2=layer.w2_weight,
                             topk_weights=topk_weights,
                             topk_ids=topk_ids,
                             inplace=True,
                             global_num_experts=global_num_experts,
                             expert_map=expert_map)

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
        ep_rank = None,
    ):
        if len(x.shape) == 3:
            bs, seq_len, hidden_size = x.shape
            x = x.reshape(bs * seq_len, hidden_size)
        assert len(x.shape) == 2
        import habana_frameworks.torch as htorch
        htorch.core.mark_step()
        if use_grouped_topk:
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
                e_score_correction_bias=e_score_correction_bias)
        else:
            import torch.nn.functional as F
            topk_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
            topk_weights, topk_ids = torch.topk(topk_weights,
                                                        top_k,
                                                        dim=-1)
            topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.to(x.dtype)

        final_hidden_states = torch.zeros_like(x)
        num_experts = layer.w13_weight.shape[0]
        n_expert_slice = layer.w13_weight.shape[0] // 8
        assert n_expert_slice * 8 == num_experts

        # w13_list = layer.hpu_fused_moe.MoeOp.w13_list
        # w2_list = layer.hpu_fused_moe.MoeOp.w2_list

        for i in range(8):
            min_expert = i * n_expert_slice
            max_expert = (i + 1) * n_expert_slice
            # w13_list_slice = [w13_list[i].weight.squeeze() for i in range(min_expert, max_expert)]
            # w2_list_slice = [w2_list[i].weight.squeeze() for i in range(min_expert, max_expert)]
            w13_list_slice = [layer.w13_weight[j].squeeze().clone() for j in range(min_expert, max_expert)]
            w2_list_slice = [layer.w2_weight[j].squeeze().clone() for j in range(min_expert, max_expert)]
            # print(f"w13_list_slice[0].shape: {w13_list_slice[0].shape}, device: {w13_list_slice[0].device}, dtype: {w13_list_slice[0].dtype}")
            # print(f"w2_list_slice[0].shape: {w2_list_slice[0].shape}, device: {w2_list_slice[0].device}, dtype: {w2_list_slice[0].dtype}")
            # print(f"hidden_states.shape: {x.shape}, device: {x.device}, dtype: {x.dtype}")
            # print(f"topk_ids.shape: {topk_ids.shape}, device: {topk_ids.device}, dtype: {topk_ids.dtype}")
            # print(f"topk_weights.shape: {topk_weights.shape}, device: {topk_weights.device}, dtype: {topk_weights.dtype}")
            # print(f"min_expert: {min_expert}, max_expert: {max_expert}")
            final_hidden_states += torch.ops.hpu.mixture_of_experts(hidden_states=x,
                                         expert_routing_table=topk_ids.to(torch.int64),
                                         router_weights=topk_weights.to(x.dtype),
                                         w12=w13_list_slice,
                                         w3=w2_list_slice,
                                         permuted_weights=True,
                                         activation="silu",
                                         experts_min=min_expert,
                                         experts_max=max_expert - 1)
            # print(f"final_hidden_states.shape: {final_hidden_states.shape}, device: {final_hidden_states.device}, dtype: {final_hidden_states.dtype}")
            htorch.core.mark_step()
            # print(f"done mark step {i}")
        return final_hidden_states.view(-1, x.shape[1])

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
        **kwargs,
    ):
        assert custom_routing_function is None
        return layer.ipex_fusion(
            x,
            use_grouped_topk,
            top_k,
            router_logits,
            renormalize,
            topk_group,
            num_expert_group,
        )

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
        e_score_correction_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        assert not use_grouped_topk
        assert num_expert_group is None
        assert topk_group is None
        assert custom_routing_function is None
        if scoring_func != "softmax":
            raise NotImplementedError(
                "Only softmax scoring function is supported for TPU.")
        if e_score_correction_bias is not None:
            raise NotImplementedError(
                "Expert score correction bias is not supported for TPU.")
        return fused_moe_pallas(hidden_states=x,
                                w1=layer.w13_weight,
                                w2=layer.w2_weight,
                                topk=top_k,
                                gating_output=router_logits,
                                global_num_experts=global_num_experts,
                                expert_map=expert_map,
                                renormalize=renormalize)

    forward_native = forward_cuda


def determine_expert_map(
        ep_size: int, ep_rank: int,
        global_num_experts: int) -> Tuple[int, Optional[torch.Tensor]]:
    """
        Calculates how many experts should be assigned to each rank for EP and
        creates a mapping from global to local expert index. Experts are
        distributed evenly across ranks. Any remaining are assigned to the
        last rank.

        Args:
            ep_size (int): The size of the expert parallel group
            global_num_experts (int): The total number of experts in the model.
        Returns:
            Tuple[int, Optional[torch.Tensor]]: A tuple containing:
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
        num_experts: int,
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
    ):
        super().__init__()

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        # Note: here we guard against accessing the TP and DP groups when
        # uninitialized (this happens when testing)
        self.tp_size = (tp_size if tp_size is not None else
                        get_tensor_model_parallel_world_size())
        tp_rank = 0 if self.tp_size == 1 else get_tensor_model_parallel_rank()
        self.dp_size = (dp_size
                        if dp_size is not None else get_dp_group().world_size)
        self.dp_rank = (0
                        if self.dp_size == 1 else get_dp_group().rank_in_group)
        self.global_num_experts = num_experts

        # Use expert parallelism instead of tensor parallelism?
        vllm_config = get_current_vllm_config()
        use_ep = (vllm_config.parallel_config.enable_expert_parallel
                  and self.tp_size * self.dp_size > 1)

        # For smuggling this layer into the fused moe custom op
        self.use_direct_call = self.dp_size == 1
        if self.use_direct_call:
            compilation_config = vllm_config.compilation_config
            if prefix in compilation_config.static_forward_context:
                raise ValueError("Duplicate layer name: {}".format(prefix))
            compilation_config.static_forward_context[prefix] = self
            self.layer_name = prefix

        if use_ep:
            # Set TP size to 1 to adjust for EP and adjust EP size and rank
            # for DP attention.
            self.ep_rank = tp_rank + self.tp_size * self.dp_rank
            self.tp_rank = 0
            self.ep_size = self.tp_size * self.dp_size
            self.tp_size = 1

            self.local_num_experts, self.expert_map = determine_expert_map(
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
                global_num_experts=self.global_num_experts)
        else:
            # Adjust TP size for DP attention
            self.tp_rank = tp_rank + self.tp_size * self.dp_rank
            self.ep_rank = 0
            self.tp_size = self.tp_size * self.dp_size
            self.ep_size = 1
            self.local_num_experts = self.global_num_experts
            self.expert_map = None

        self.top_k = top_k
        self.global_num_experts = num_experts

        assert intermediate_size % self.tp_size == 0
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        self.reduce_results = reduce_results
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.custom_routing_function = custom_routing_function
        self.multicast_fn = self.hpu_multicast if is_hpu else self.naive_multicast
        if is_hpu:
            if VLLM_REQUANT_FP8_INC:
                ep_shift = self.ep_rank * self.local_num_experts
                from vllm.model_executor.layers.vllm_ext_patch import (
                    VllmMixtureOfExpertsOpFP8,
                )
                moe_n_slice = int(os.environ.get("VLLM_MOE_N_SLICE", 4))
                assert moe_n_slice == 1, (
                    f"moe_n_slice is {moe_n_slice}, expected 1 when using VLLM_REQUANT_FP8_INC"
                )
                num_expert_per_group = self.local_num_experts // moe_n_slice
                experts_min, experts_max = 0, self.local_num_experts
                moe_op = VllmMixtureOfExpertsOpFP8(
                    num_expert_per_group,
                    experts_min + ep_shift,
                    experts_max - 1 + ep_shift,
                )
                self.moe_op = moe_op
            else:
                from vllm_hpu_extension.ops import DynamicFusedMOE

                self.hpu_fused_moe = DynamicFusedMOE(self.local_num_experts)
            

        self.scoring_func = scoring_func
        self.e_score_correction_bias = e_score_correction_bias

        if self.scoring_func != "softmax" and not self.use_grouped_topk:
            raise ValueError("Only softmax scoring function is supported for "
                             "non-grouped topk.")

        if quant_config is None:
            self.quant_method: Optional[QuantizeMethodBase] = (
                UnquantizedFusedMoEMethod())
            self.activation_scheme = 'none'
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix)
            self.activation_scheme = quant_config.activation_scheme
        assert self.quant_method is not None

        local_num_experts = torch.sum(self.expert_map != -1) \
            if self.expert_map is not None else num_experts

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
                in ("GPTQMarlinMoEMethod", "CompressedTensorsWNA16MoEMethod")):
            moe_quant_params["intermediate_size_full"] = intermediate_size

        self.quant_method.create_weights(layer=self, **moe_quant_params)

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
                                                 loaded_weight: torch.tensor,
                                                 tp_rank: int,
                                                 expert_id: int,
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
                          tp_rank=self.tp_rank,
                          expert_id=expert_id,
                          load_full=load_full_w2)
        elif shard_id in ("w1", "w3"):
            self._load_w13(shard_id=shard_id,
                           shard_dim=shard_dim,
                           loaded_weight=loaded_weight,
                           expert_data=expert_data,
                           tp_rank=self.tp_rank,
                           expert_id=expert_id)

    def _load_per_channel_weight_scale(self, expert_data: torch.Tensor,
                                       shard_dim: int, shard_id: str,
                                       loaded_weight: torch.Tensor,
                                       tp_rank: int,
                                       expert_id: int):
        # for per channel weight quantization
        if shard_id == "w2":
            expert_data.copy_(loaded_weight)
        elif shard_id in ("w1", "w3"):
            self._load_w13(shard_id=shard_id,
                           shard_dim=shard_dim,
                           loaded_weight=loaded_weight,
                           expert_data=expert_data,
                           tp_rank=self.tp_rank,
                           expert_id=expert_id)

    def _load_w13(self,
                  expert_data: torch.Tensor,
                  shard_dim: int,
                  shard_id: str,
                  loaded_weight: torch.tensor,
                  tp_rank: int,
                  expert_id: Optional[int] = None):

        orig_exp_data = expert_data.view(expert_data.size())
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

        if is_hpu and not VLLM_REQUANT_FP8_INC:
            self.hpu_fused_moe.MoeOp.w13_list[expert_id].set_weight(
                orig_exp_data)
            # print(f"loaded w13 for hpu for expert_id: {expert_id}, orig_exp_data.shape: {orig_exp_data.shape}")

    def _load_w2(self,
                 expert_data: torch.Tensor,
                 shard_dim: int,
                 loaded_weight: torch.tensor,
                 tp_rank: int,
                 load_full: bool = False,
                 expert_id: Optional[int] = None):

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
        if is_hpu and not VLLM_REQUANT_FP8_INC:
            self.hpu_fused_moe.MoeOp.w2_list[expert_id].set_weight(expert_data)
            # print(f"loaded w2 for hpu for expert_id: {expert_id}, expert_data.shape: {expert_data.shape}")

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
                          tp_rank=self.tp_rank)
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

        # compressed-tensors checkpoints with packed weights are stored flipped
        # TODO (mgoin): check self.quant_method.quant_config.quant_format
        # against known CompressionFormat enum values that have this quality
        loaded_weight = loaded_weight.t().contiguous() if (
            self.quant_method.__class__.__name__
            == "CompressedTensorsWNA16MoEMethod") else loaded_weight

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

        expert_data = param.data[expert_id]

        # is_transposed: if the dim to shard the weight
        # should be flipped. Required by GPTQ, compressed-tensors
        # should be whatever dimension intermediate_size_per_partition is
        is_transposed = getattr(param, "is_transposed", False)
        shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
        if is_transposed:
            shard_dim = int(not shard_dim)

        # Case input scale: input_scale loading is only supported for fp8
        if "input_scale" in weight_name:
            # this is needed for compressed-tensors only
            loaded_weight = loaded_weight.to(param.data.device)

            if param.data[expert_id] != 1 and (param.data[expert_id] -
                                               loaded_weight).abs() > 1e-5:
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

        # Case weight scales and zero_points
        if ("scale" in weight_name or "zero" in weight_name):
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
                    expert_id=expert_id)
            # elif current_platform.is_hpu():
            #     print(f"!!!!!!!!!!!!! HPU load per channel weight scale")
            #     self._load_per_channel_weight_scale(
            #         shard_id=shard_id,
            #         shard_dim=shard_dim,
            #         loaded_weight=loaded_weight,
            #         expert_data=expert_data,
            #         tp_rank=self.tp_rank,
            #         expert_id=expert_id)
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
                    expert_id=expert_id,
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
                tp_rank=self.tp_rank,
                expert_id=expert_id)
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
                       e_score_correction_bias: Optional[torch.Tensor] = None):
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            fused_topk, grouped_topk)

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
        elif custom_routing_function is None:
            topk_weights, topk_ids = fused_topk(hidden_states=hidden_states,
                                                gating_output=router_logits,
                                                topk=top_k,
                                                renormalize=renormalize)
        else:
            topk_weights, topk_ids = custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize)

        return topk_weights, topk_ids

    def hpu_multicast(self, x: torch.Tensor,
                      cu_tokens_across_dp_cpu: torch.Tensor,
                      output_tensor: Optional[torch.Tensor] = None):
        if output_tensor is None:
            world_size = get_dp_group().world_size
            input_size = x.size()
            # Allocate output tensor.
            output_size = list(input_size)
            output_size[0] *= world_size
            output_tensor = torch.empty(output_size,
                                        dtype=x.dtype,
                                        device=x.device)
        else:
            if output_tensor.ndim == 3 and x.ndim == 2:
                output_tensor.view(-1, x.size(1))
        # All-gather.
        torch.distributed.all_gather_into_tensor(output_tensor, x,
                                                 group=get_dp_group().device_group)
        return output_tensor

    def naive_multicast(self, x: torch.Tensor,
                        cu_tokens_across_dp_cpu: torch.Tensor,
                        output_tensor: Optional[torch.Tensor] = None):
        assert (len(x.shape) in [2, 3])
        ori_shape = x.shape
        if len(ori_shape) == 3:
            x = x.view(-1, x.size(2))
        buffer = torch.empty((cu_tokens_across_dp_cpu[-1], x.size(1)),
                             device=x.device,
                             dtype=x.dtype)

        start = 0 if self.dp_rank == 0 else cu_tokens_across_dp_cpu[
            self.dp_rank - 1]
        end = cu_tokens_across_dp_cpu[self.dp_rank]
        buffer[start:end, :].copy_(x)
        for idx in range(get_dp_group().world_size):
            start = 0 if idx == 0 else cu_tokens_across_dp_cpu[idx - 1]
            end = cu_tokens_across_dp_cpu[idx]
            get_dp_group().broadcast(buffer[start:end, :], idx)
        if len(ori_shape) == 3:
            buffer = buffer.view(-1, ori_shape[1], ori_shape[2])

        return buffer

    def forward(self, hidden_states: torch.Tensor,
                router_logits: torch.Tensor):
        if not self.use_direct_call:
            return self.forward_impl(hidden_states, router_logits)
        else:
            return torch.ops.vllm.moe_forward(hidden_states, router_logits,
                                              self.layer_name)

    def forward_impl(self, hidden_states: torch.Tensor,
                     router_logits: torch.Tensor):
        assert self.quant_method is not None

        if self.dp_size > 1:
            cu_tokens_across_dp_cpu = get_forward_context(
            ).dp_metadata.cu_tokens_across_dp_cpu

            if self.activation_scheme != "static":
                hidden_states_across_dp = get_forward_context(
                ).dp_metadata.hidden_states_across_dp
                hidden_states = self.multicast_fn(hidden_states,
                                                  cu_tokens_across_dp_cpu,
                                                  hidden_states_across_dp)


        quant_kwargs = {}
        if (self.quant_method.__class__.__name__ in ("Fp8MoEMethod")):
            quant_kwargs["ep_rank"] = self.ep_rank
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
            **quant_kwargs)

        if self.dp_size > 1:
            start = 0 if self.dp_rank == 0 else cu_tokens_across_dp_cpu[
                self.dp_rank - 1]
            end = cu_tokens_across_dp_cpu[self.dp_rank]

            import habana_frameworks.torch as htorch
            htorch.core.mark_step()
            local_hidden_states = get_forward_context(
            ).dp_metadata.hidden_states
            torch.distributed.reduce_scatter_tensor(local_hidden_states, final_hidden_states, group=get_dp_group().device_group)

            final_hidden_states = local_hidden_states

        if self.reduce_results and (self.tp_size > 1 or self.ep_size > 1):
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)

        return final_hidden_states

    @classmethod
    def make_expert_params_mapping(
            cls, ckpt_gate_proj_name: str, ckpt_down_proj_name: str,
            ckpt_up_proj_name: str,
            num_experts: int) -> List[Tuple[str, str, int, str]]:

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
