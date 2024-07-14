from abc import abstractmethod
from typing import List, Optional, Tuple

import torch

from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.fused_moe import (fused_experts,
                                                            fused_topk,
                                                            grouped_topk)
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
    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              topk_weights: torch.Tensor,
              topk_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class UnquantizedFusedMoEMethod(FusedMoEMethodBase):
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

    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              topk_weights: torch.Tensor,
              topk_ids: torch.Tensor) -> torch.Tensor:

        return fused_experts(x,
                             layer.w13_weight,
                             layer.w2_weight,
                             topk_weights,
                             topk_ids,
                             inplace=True)


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
            self.quant_method = quant_config.get_quant_method(self)
        assert self.quant_method is not None

        self.quant_method.create_weights(
            layer=self,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=self.intermediate_size_per_partition,
            params_dtype=params_dtype,
            weight_loader=self.weight_loader)

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
            for expert_id in range(num_experts)
            for shard_id, weight_name in [
                ("w1", ckpt_gate_proj_name),
                ("w2", ckpt_down_proj_name),
                ("w3", ckpt_up_proj_name),
            ]
        ]

    def _load_fp8_scale(self, param: torch.nn.Parameter,
                        loaded_weight: torch.Tensor, weight_name: str,
                        shard_id: str, expert_id: int) -> None:
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
            if shard_id == "w1" or shard_id == "w3":
                # We have to keep the weight scales of w1 and w3 because
                # we need to re-quantize w1/w3 weights after weight loading.
                idx = 0 if shard_id == "w1" else 1
                param_data[expert_id][idx] = loaded_weight
            # If we are in the row parallel case (down_proj)
            else:
                param_data[expert_id] = loaded_weight

    def weight_loader(self, param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor, weight_name: str,
                      shard_id: str, expert_id: int) -> None:
        if shard_id not in ["w1", "w2", "w3"]:
            raise ValueError(f"shard_id must be ['w1','w2','w3'] but "
                             f"got {shard_id}.")

        # Special case for fp8 scales.
        if getattr(param, "is_fp8_scale", False):
            self._load_fp8_scale(param.data, loaded_weight, weight_name,
                                 shard_id, expert_id)
            return

        expert_data = param.data[expert_id]
        tp_rank = get_tensor_model_parallel_rank()

        # If transposed, weight is saved as [input_dim, output_dim]
        # Otherwise, weight is saved as     [output_dim, input_dim]
        is_transposed = getattr(param, "is_transposed", False)
        input_dim = 0 if is_transposed else 1
        output_dim = 1 if is_transposed else 0

        # Index the loaded weight for tp sharding.
        # down_proj: "RowParallel" so tp sharding on input_dim
        if (shard_id == "w2"):
            shard_dim = input_dim
            shard_size = expert_data.shape[shard_dim]
        # gate_up_proj: "MergedColumnParallel", so tp sharding on output_dim
        elif (shard_id == "w1" or shard_id == "w3"):
            shard_dim = output_dim
            shard_size = expert_data.shape[output_dim] // 2
        offset = shard_size * tp_rank
        loaded_weight = loaded_weight.narrow(shard_dim, offset, shard_size)

        # Narrow parameter and load.
        # w1, gate_proj: Load into first logical weight of w13.
        if shard_id == "w1":
            expert_data = expert_data.narrow(shard_dim, 0, shard_size)
            expert_data.copy_(loaded_weight)
        # w3, up_proj: Load into second logical weight of w13.
        elif shard_id == "w3":
            expert_data = expert_data.narrow(shard_dim, shard_size, shard_size)
            expert_data.copy_(loaded_weight)
        # w2, down_proj: Load into only logical weight of w2.
        elif shard_id == "w2":
            expert_data.copy_(loaded_weight)
        else:
            raise ValueError

    def _select_experts(self, hidden_states: torch.Tensor,
                        router_logits: torch.Tensor):
        # DeekSeekv2 uses grouped_top_k
        if self.use_grouped_topk:
            assert (self.num_expert_group is not None
                    and self.topk_group is not None)
            topk_weights, topk_ids = grouped_topk(hidden_states, router_logits,
                                                  self.top_k, self.renormalize,
                                                  self.num_expert_group,
                                                  self.topk_group)
        else:
            topk_weights, topk_ids = fused_topk(hidden_states, router_logits,
                                                self.top_k, self.renormalize)

        return topk_weights, topk_ids

    def forward(self, hidden_states: torch.Tensor,
                router_logits: torch.Tensor):
        assert self.quant_method is not None

        # Select experts.
        topk_weights, topk_ids = self._select_experts(hidden_states,
                                                      router_logits)

        # Call fused kernel.
        final_hidden_states = self.quant_method.apply(self, hidden_states,
                                                      topk_weights, topk_ids)

        # Optionally reduce.
        if self.reduce_results and self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)

        return final_hidden_states
