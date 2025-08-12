from abc import abstractmethod
from typing import List, Optional, Tuple

import torch

from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)


class FusedMoEMethodBase(QuantizeMethodBase):

    @abstractmethod
    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        raise NotImplementedError

    @abstractmethod
    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              router_logits: torch.Tensor,
              top_k: int,
              renormalize: bool = True,
              use_grouped_topk: bool = False,
              num_expert_group: Optional[int] = None,
              topk_group: Optional[int] = None) -> torch.Tensor:
        raise NotImplementedError


class UnquantizedFusedMoEMethod(FusedMoEMethodBase, CustomOp):
    """MoE method without quantization."""

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(torch.empty(num_experts,
                                                    2 * intermediate_size,
                                                    hidden_size,
                                                    dtype=params_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(torch.empty(num_experts,
                                                   hidden_size,
                                                   intermediate_size,
                                                   dtype=params_dtype),
                                       requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
    ) -> torch.Tensor:
        return self.forward(x, layer.w13_weight, layer.w2_weight,
                            router_logits, top_k, renormalize,
                            use_grouped_topk, num_expert_group, topk_group)

    def forward_cuda(
        self,
        x: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        num_expert_group: Optional[int],
        topk_group: Optional[int],
    ) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe
        return fused_moe(x,
                         w1,
                         w2,
                         router_logits,
                         top_k,
                         renormalize=renormalize,
                         inplace=True,
                         use_grouped_topk=use_grouped_topk,
                         num_expert_group=num_expert_group,
                         topk_group=topk_group)

    def forward_cpu(self, *args, **kwargs):
        raise NotImplementedError(
            "The CPU backend currently does not support MoE.")

    def forward_tpu(
        self,
        x: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        num_expert_group: Optional[int],
        topk_group: Optional[int],
    ) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe.moe_pallas import fused_moe
        assert not use_grouped_topk
        assert num_expert_group is None
        assert topk_group is None
        return fused_moe(x, w1, w2, router_logits, top_k, renormalize)


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
        prefix: str = "",
    ):
        super().__init__()

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.tp_size = (tp_size if tp_size is not None else
                        get_tensor_model_parallel_world_size())
        self.top_k = top_k
        self.num_experts = num_experts
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        self.reduce_results = reduce_results
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group

        if quant_config is None:
            self.quant_method: Optional[QuantizeMethodBase] = (
                UnquantizedFusedMoEMethod())
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix)
        assert self.quant_method is not None

        self.quant_method.create_weights(
            layer=self,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=self.intermediate_size_per_partition,
            params_dtype=params_dtype,
            weight_loader=self.weight_loader)

    def weight_loader(self, param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor, weight_name: str,
                      shard_id: int, expert_id: int):
        param_data = param.data

        # Input scales can be loaded directly and should be equal.
        if "input_scale" in weight_name:
            if param_data[expert_id] != 1 and (param_data[expert_id] -
                                               loaded_weight).abs() > 1e-5:
                raise ValueError(
                    "input_scales of w1 and w3 of a layer "
                    f"must be equal. But got {param_data[expert_id]} "
                    f"vs. {loaded_weight}")
            param_data[expert_id] = loaded_weight
        # Weight scales
        elif "weight_scale" in weight_name:
            # If we are in merged column case (gate_up_proj)
            #   shard_id 0 == gate_proj / w1
            #   shard_id 2 == up_proj / w3
            if shard_id == 0 or shard_id == 2:
                # We have to keep the weight scales of w1 and w3 because
                # we need to re-quantize w1/w3 weights after weight loading.
                idx = 0 if shard_id == 0 else 1
                param_data[expert_id][idx] = loaded_weight
            # If we are in the row parallel case (down_proj)
            #   shard_id 1 == down_proj / w2
            else:
                param_data[expert_id] = loaded_weight
        # Weights
        else:
            tp_rank = get_tensor_model_parallel_rank()
            shard_size = self.intermediate_size_per_partition
            shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)

            # w1, gate_proj case: Load into first shard of w13.
            if shard_id == 0:
                param_data[expert_id,
                           0:shard_size, :] = loaded_weight[shard, :]
            # w3, up_proj case: Load into second shard of w13.
            elif shard_id == 2:
                param_data[expert_id, shard_size:2 *
                           shard_size, :] = loaded_weight[shard, :]
            # w2, down_proj case: Load into only shard of w2.
            elif shard_id == 1:
                param_data[expert_id, :, :] = loaded_weight[:, shard]
            else:
                raise ValueError(
                    f"Shard id must be in [0,1,2] but got {shard_id}")

    def forward(self, hidden_states: torch.Tensor,
                router_logits: torch.Tensor):
        assert self.quant_method is not None

        # Matrix multiply.
        final_hidden_states = self.quant_method.apply(
            self,
            x=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            num_expert_group=self.num_expert_group,
            topk_group=self.topk_group)

        if self.reduce_results and self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)

        return final_hidden_states

    @classmethod
    def make_expert_params_mapping(
            cls, ckpt_gate_proj_name: str, ckpt_down_proj_name: str,
            ckpt_up_proj_name: str,
            num_experts: int) -> List[Tuple[str, str, int, int]]:

        gate_up = [ckpt_gate_proj_name, ckpt_up_proj_name]
        gate_down_up = [
            ckpt_gate_proj_name, ckpt_down_proj_name, ckpt_up_proj_name
        ]

        return [
            # These are the weight scales for the experts
            # (param_name, weight_name, expert_id, shard_id)
            ("experts.w13_scale"
             if weight_name in gate_up else "experts.w2_scale",
             f"experts.{expert_id}.{weight_name}.weight_scale", expert_id,
             shard_id) for expert_id in range(num_experts)
            for shard_id, weight_name in enumerate(gate_down_up)
        ] + [
            # These are the weights for the experts
            # (param_name, weight_name, expert_id, shard_id)
            ("experts.w13_weight"
             if weight_name in gate_up else "experts.w2_weight",
             f"experts.{expert_id}.{weight_name}.weight", expert_id, shard_id)
            for expert_id in range(num_experts)
            for shard_id, weight_name in enumerate(gate_down_up)
        ] + [
            # These are the weight scales for the experts
            # (param_name, weight_name, expert_id, shard_id)
            ("experts.a13_scale"
             if weight_name in gate_up else "experts.a2_scale",
             f"experts.{expert_id}.{weight_name}.input_scale", expert_id,
             shard_id) for expert_id in range(num_experts)
            for shard_id, weight_name in enumerate(gate_down_up)
        ]
