import enum
from abc import abstractmethod
from enum import Enum
from typing import List, Optional, Tuple

import torch

from vllm import _custom_ops as ops
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe.fused_moe import fused_marlin_moe
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.gptq_marlin import (
    GPTQMarlinConfig)
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
              router_logits: torch.Tensor, top_k: int, renormalize: bool,
              use_grouped_topk: bool) -> torch.Tensor:
        raise NotImplementedError


class GPTQMarlinState(Enum):
    REPACK = enum.auto()
    READY = enum.auto()


class MarlinFusedMoEMethod(FusedMoEMethodBase):
    """MoE Marlin method with quantization."""

    def __init__(self, quant_config: GPTQMarlinConfig) -> None:
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        # Currently assuming is_k_full is always True
        # (input size per partition is the same as full input size)
        # Supports only sym for now (no zp)
        if self.quant_config.group_size != -1:
            scales_size13 = hidden_size // self.quant_config.group_size
            scales_size2 = intermediate_size // self.quant_config.group_size
        else:
            scales_size13 = 1
            scales_size2 = 1
        # Fused gate_up_proj (column parallel)
        w13_qweight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size // self.quant_config.pack_factor,
            2 * intermediate_size,
            dtype=torch.int32),
                                         requires_grad=False)
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)
        # down_proj (row parallel)
        w2_qweight = torch.nn.Parameter(torch.empty(
            num_experts,
            intermediate_size // self.quant_config.pack_factor,
            hidden_size,
            dtype=torch.int32),
                                        requires_grad=False)
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)
        # up_proj scales
        w13_scales = torch.nn.Parameter(torch.empty(num_experts,
                                                    scales_size13,
                                                    2 * intermediate_size,
                                                    dtype=params_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)
        # down_proj scales
        w2_scales = torch.nn.Parameter(torch.empty(num_experts,
                                                   scales_size2,
                                                   hidden_size,
                                                   dtype=params_dtype),
                                       requires_grad=False)
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)
        w13_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx", w13_g_idx)
        set_weight_attrs(w13_g_idx, extra_weight_attrs)
        w2_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx", w2_g_idx)
        set_weight_attrs(w2_g_idx, extra_weight_attrs)
        w13_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx_sort_indices",
                                 w13_g_idx_sort_indices)
        set_weight_attrs(w13_g_idx_sort_indices, extra_weight_attrs)
        w2_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx_sort_indices",
                                 w2_g_idx_sort_indices)
        set_weight_attrs(w2_g_idx_sort_indices, extra_weight_attrs)
        layer.marlin_state = GPTQMarlinState.REPACK

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              router_logits: torch.Tensor,
              top_k: int,
              renormalize: bool = True,
              use_grouped_topk: bool = False,
              num_expert_group: Optional[int] = None,
              topk_group: Optional[int] = None) -> torch.Tensor:
        if layer.marlin_state == GPTQMarlinState.REPACK:
            layer.marlin_state = GPTQMarlinState.READY

            # Newly generated tensors need to replace existing tensors that are
            # already registered as parameters by vLLM (and won't be freed)
            def replace_tensor(name, new_t):
                # It is important to use resize_() here since it ensures
                # the same buffer is reused
                getattr(layer, name).resize_(new_t.shape)
                getattr(layer, name).copy_(new_t)
                del new_t

            def get_scale_perms(num_bits: int):
                scale_perm: List[int] = []
                for i in range(8):
                    scale_perm.extend([i + 8 * j for j in range(8)])
                scale_perm_single: List[int] = []
                for i in range(4):
                    scale_perm_single.extend(
                        [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
                return scale_perm, scale_perm_single

            def marlin_permute_scales(s: torch.Tensor, size_k: int,
                                      size_n: int, group_size: int,
                                      num_bits: int):
                scale_perm, scale_perm_single = get_scale_perms(num_bits)
                if group_size < size_k and group_size != -1:
                    s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
                else:
                    s = s.reshape(
                        (-1, len(scale_perm_single)))[:, scale_perm_single]
                s = s.reshape((-1, size_n)).contiguous()
                return s

            def marlin_moe_permute_scales(s: torch.Tensor, size_k: int,
                                          size_n: int, group_size: int,
                                          num_bits: int):
                num_experts = s.shape[0]
                output = torch.empty((num_experts, s.shape[1], s.shape[2]),
                                     device=s.device,
                                     dtype=s.dtype)
                for e in range(num_experts):
                    output[e] = marlin_permute_scales(s[e], size_k, size_n,
                                                      group_size, num_bits)
                return output

            # Process act_order
            if self.quant_config.desc_act:
                # Get sorting based on g_idx
                num_experts = layer.w13_g_idx.shape[0]
                w13_g_idx_sort_indices = torch.empty_like(layer.w13_g_idx)
                w2_g_idx_sort_indices = torch.empty_like(layer.w2_g_idx)
                w13_sorted_g_idx = torch.empty_like(layer.w13_g_idx)
                w2_sorted_g_idx = torch.empty_like(layer.w2_g_idx)
                for e in range(num_experts):
                    w13_g_idx_sort_indices[e] = torch.argsort(
                        layer.w13_g_idx[e]).to(torch.int32)
                    w2_g_idx_sort_indices[e] = torch.argsort(
                        layer.w2_g_idx[e]).to(torch.int32)
                    w13_sorted_g_idx[e] = layer.w13_g_idx[e][
                        w13_g_idx_sort_indices[e]]
                    w2_sorted_g_idx[e] = layer.w2_g_idx[e][
                        w2_g_idx_sort_indices[e]]
                replace_tensor("w13_g_idx", w13_sorted_g_idx)
                replace_tensor("w2_g_idx", w2_sorted_g_idx)
                replace_tensor("w13_g_idx_sort_indices",
                               w13_g_idx_sort_indices)
                replace_tensor("w2_g_idx_sort_indices", w2_g_idx_sort_indices)
            else:
                # Reset g_idx related tensors
                num_experts = layer.w13_g_idx.shape[0]
                device = layer.w13_g_idx.device
                layer.w13_g_idx = torch.nn.Parameter(
                    torch.empty((num_experts, 0),
                                dtype=torch.int32,
                                device=device),
                    requires_grad=False,
                )
                layer.w2_g_idx = torch.nn.Parameter(
                    torch.empty((num_experts, 0),
                                dtype=torch.int32,
                                device=device),
                    requires_grad=False,
                )
                layer.w13_g_idx_sort_indices = torch.nn.Parameter(
                    torch.empty((num_experts, 0),
                                dtype=torch.int32,
                                device=device),
                    requires_grad=False,
                )
                layer.w2_g_idx_sort_indices = torch.nn.Parameter(
                    torch.empty((num_experts, 0),
                                dtype=torch.int32,
                                device=device),
                    requires_grad=False,
                )
            # Repack weights
            marlin_w13_qweight = ops.gptq_marlin_moe_repack(
                layer.w13_qweight,
                layer.w13_g_idx_sort_indices,
                layer.w13_qweight.shape[1] * self.quant_config.pack_factor,
                layer.w13_qweight.shape[2],
                self.quant_config.quant_type.size_bits,
            )
            replace_tensor("w13_qweight", marlin_w13_qweight)
            marlin_w2_qweight = ops.gptq_marlin_moe_repack(
                layer.w2_qweight,
                layer.w2_g_idx_sort_indices,
                layer.w2_qweight.shape[1] * self.quant_config.pack_factor,
                layer.w2_qweight.shape[2],
                self.quant_config.quant_type.size_bits,
            )
            replace_tensor("w2_qweight", marlin_w2_qweight)
            # Repack scales
            marlin_w13_scales = marlin_moe_permute_scales(
                layer.w13_scales,
                x.shape[1],
                layer.w13_scales.shape[2],
                self.quant_config.group_size,
                self.quant_config.quant_type.size_bits,
            )
            replace_tensor("w13_scales", marlin_w13_scales)
            marlin_w2_scales = marlin_moe_permute_scales(
                layer.w2_scales,
                layer.w2_scales.shape[1] * self.quant_config.pack_factor,
                x.shape[1],
                self.quant_config.group_size,
                self.quant_config.quant_type.size_bits,
            )
            replace_tensor("w2_scales", marlin_w2_scales)
        return fused_marlin_moe(x,
                                layer.w13_qweight,
                                layer.w2_qweight,
                                router_logits,
                                layer.w13_g_idx,
                                layer.w2_g_idx,
                                layer.w13_g_idx_sort_indices,
                                layer.w2_g_idx_sort_indices,
                                top_k,
                                renormalize=renormalize,
                                w1_scale=layer.w13_scales,
                                w2_scale=layer.w2_scales)


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

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              router_logits: torch.Tensor,
              top_k: int,
              renormalize: bool,
              use_grouped_topk: bool,
              topk_group: Optional[int] = None,
              num_expert_group: Optional[int] = None) -> torch.Tensor:

        return self.forward(x=x,
                            layer=layer,
                            router_logits=router_logits,
                            top_k=top_k,
                            renormalize=renormalize,
                            use_grouped_topk=use_grouped_topk,
                            topk_group=topk_group,
                            num_expert_group=num_expert_group)

    def forward_cuda(self,
                     layer: torch.nn.Module,
                     x: torch.Tensor,
                     use_grouped_topk: bool,
                     top_k: int,
                     router_logits: torch.Tensor,
                     renormalize: bool,
                     topk_group: Optional[int] = None,
                     num_expert_group: Optional[int] = None) -> torch.Tensor:

        from vllm.model_executor.layers.fused_moe.fused_moe import (
            fused_experts)

        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group)

        return fused_experts(hidden_states=x,
                             w1=layer.w13_weight,
                             w2=layer.w2_weight,
                             topk_weights=topk_weights,
                             topk_ids=topk_ids,
                             inplace=True)

    def forward_cpu(self, *args, **kwargs):
        raise NotImplementedError(
            "The CPU backend currently does not support MoE.")

    def forward_tpu(self,
                    layer: torch.nn.Module,
                    x: torch.Tensor,
                    use_grouped_topk: bool,
                    top_k: int,
                    router_logits: torch.Tensor,
                    renormalize: bool,
                    topk_group: Optional[int] = None,
                    num_expert_group: Optional[int] = None) -> torch.Tensor:

        from vllm.model_executor.layers.fused_moe.moe_pallas import fused_moe
        assert not use_grouped_topk
        assert num_expert_group is None
        assert topk_group is None
        return fused_moe(hidden_states=x,
                         w1=layer.w13_weight,
                         w2=layer.w2_weight,
                         topk=top_k,
                         gating_output=router_logits,
                         renormalize=renormalize)


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

        self.quant_method: Optional[QuantizeMethodBase] = None

        if quant_config is None:
            self.quant_method = UnquantizedFusedMoEMethod()
        elif isinstance(quant_config, GPTQMarlinConfig):
            self.quant_method = MarlinFusedMoEMethod(quant_config)
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

    def weight_loader(self,
                      param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor,
                      weight_name: str,
                      shard_id: str,
                      expert_id: int,
                      is_quantized: bool = False):
        param_data = param.data

        if is_quantized:
            if "_qweight" in weight_name or "_scales" in weight_name:
                if "w13" in weight_name:
                    shard_size = self.intermediate_size_per_partition
                    if shard_id == 0:
                        param_data[expert_id, :, :shard_size] = loaded_weight
                    elif shard_id == 1:
                        param_data[expert_id, :, shard_size:] = loaded_weight
                    else:
                        raise ValueError(f"Invalid shard_id: {shard_id}: "
                                         "must be 0 or 1.")
                elif "w2" in weight_name:
                    param_data[expert_id][:] = loaded_weight
                else:
                    raise ValueError(f"Invalid weight name: {weight_name}: "
                                     "must contain 'w13' or 'w2'.")
            elif "_g_idx" in weight_name:
                if "w13" not in weight_name and "w2" not in weight_name:
                    raise ValueError(f"Invalid weight name: {weight_name}: "
                                     "must contain 'w13' or 'w2'.")
                param_data[expert_id] = loaded_weight
            else:
                raise ValueError(f"Invalid weight name: {weight_name}.")
        else:
            if shard_id not in ("w1", "w2", "w3"):
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
            # Default is not transposed/input dim is dim 1
            input_dim = getattr(param, "input_dim", 1)
            output_dim = getattr(param, "output_dim", 0)

            # Index the loaded weight for tp sharding.
            # down_proj: "RowParallel" so tp sharding on input_dim
            if shard_id == "w2":
                shard_dim = input_dim
                shard_size = expert_data.shape[shard_dim]
            # gate_up_proj: "MergedColumnParallel", so tp sharding on output_dim
            elif shard_id in ("w1", "w3"):
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
                expert_data = expert_data.narrow(shard_dim, shard_size,
                                                 shard_size)
                expert_data.copy_(loaded_weight)
            # w2, down_proj: Load into only logical weight of w2.
            elif shard_id == "w2":
                expert_data.copy_(loaded_weight)
            else:
                raise ValueError(
                    f"Expected shard_id w1, w2 or w3 but got {shard_id}")

    @staticmethod
    def select_experts(hidden_states: torch.Tensor,
                       router_logits: torch.Tensor,
                       top_k: int,
                       use_grouped_topk: bool,
                       renormalize: bool,
                       topk_group: Optional[int] = None,
                       num_expert_group: Optional[int] = None):
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
                topk_group=topk_group)
        else:
            topk_weights, topk_ids = fused_topk(hidden_states=hidden_states,
                                                gating_output=router_logits,
                                                topk=top_k,
                                                renormalize=renormalize)

        return topk_weights, topk_ids

    def forward(self, hidden_states: torch.Tensor,
                router_logits: torch.Tensor):
        assert self.quant_method is not None

        # Matrix multiply.
        final_hidden_states = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group)

        if self.reduce_results and self.tp_size > 1:
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
            if shard_id in ("w1", "w3"):
                # We have to keep the weight scales of w1 and w3 because
                # we need to re-quantize w1/w3 weights after weight loading.
                idx = 0 if shard_id == "w1" else 1
                param_data[expert_id][idx] = loaded_weight
            # If we are in the row parallel case (down_proj)
            else:
                param_data[expert_id] = loaded_weight
