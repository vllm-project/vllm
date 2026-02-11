# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import weakref
from collections.abc import Callable

import torch
from torch.nn import functional as F

from vllm import _custom_ops as ops
from vllm._custom_ops import cpu_fused_moe, cpu_prepack_moe_weight
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.quantization.utils.layer_utils import replace_parameter
from vllm.utils.torch_utils import direct_register_custom_op

_CPU_MOE_LAYER_CACHE = {}


def _swigluoai_forward_native(
    x: torch.Tensor,
    alpha: float = 1.702,
    limit: float = 7.0,
) -> torch.Tensor:
    """PyTorch-native implementation of SwigluOAIAndMul.forward_native.

    Standalone function to avoid instantiating SwigluOAIAndMul (a CustomOp)
    which would trigger get_current_vllm_config() before config is set.
    """
    gate, up = x[..., ::2], x[..., 1::2]
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    gated_output = (up + 1) * glu
    return gated_output


# Map activation names to their native forward functions.
# Uses static methods or standalone functions to avoid instantiating CustomOp
# classes, which would call get_current_vllm_config() before config is set.
_CPU_MOE_ACT_FN: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "silu": SiluAndMul.forward_native,
    "swigluoai": _swigluoai_forward_native,
}


def grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    gating_output = gating_output.float()
    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    num_token = scores.shape[0]
    if e_score_correction_bias is not None:
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores.view(num_token, num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        )
    else:
        group_scores = (
            scores.view(num_token, num_expert_group, -1).max(dim=-1).values
        )  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))  # [n, e]

    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)[1]
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights, topk_ids.to(torch.int32)


def select_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    use_grouped_topk: bool,
    renormalize: bool,
    topk_group: int | None = None,
    num_expert_group: int | None = None,
    custom_routing_function: Callable | None = None,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if use_grouped_topk:
        assert topk_group is not None
        assert num_expert_group is not None
        return grouped_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
        )
    elif custom_routing_function is None:
        assert scoring_func == "softmax"
        topk_logit_vals, topk_idx = torch.topk(
            router_logits, k=top_k, dim=-1, sorted=False
        )
        if renormalize:
            topk_vals = torch.softmax(topk_logit_vals, dim=-1)
        else:
            logZ = torch.logsumexp(router_logits, dim=-1, keepdim=True)
            topk_vals = (topk_logit_vals - logZ).exp()
        return topk_vals.to(torch.float32), topk_idx.to(torch.int32)
    else:
        return custom_routing_function(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
        )


class SGLFusedMOE:
    def __init__(self, layer: torch.nn.Module) -> None:
        pass

    def __call__(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
    ) -> torch.Tensor:
        assert activation == "silu", f"{activation} is not supported."
        assert not apply_router_weight_on_input
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
        )

        torch.ops._C.fused_experts_cpu(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            True,
            False,
            False,
            None,
            None,
            None,
            None,
            None,
            True,
        )
        return x


class CPUFusedMOE:
    """CPU-based fused MoE implementation."""

    def __init__(self, layer: torch.nn.Module) -> None:
        use_grouped_gemm, isa = self.check_grouped_gemm(layer)
        self.isa = isa
        if use_grouped_gemm:
            self.forward_method = self.forward_grouped_gemm
            self.init_moe_grouped_gemm(layer=layer)
        else:
            self.forward_method = self.forward_torch
            self.init_moe_torch(layer=layer)

    def __call__(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
    ) -> torch.Tensor:
        assert activation in _CPU_MOE_ACT_FN, f"{activation} is not supported."
        assert not apply_router_weight_on_input

        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
        )

        return self.forward_method(
            layer,
            x,
            topk_weights,
            topk_ids,
            activation,
            global_num_experts,
        )

    def check_grouped_gemm(
        self,
        layer: torch.nn.Module,
    ) -> tuple[bool, str]:
        if not hasattr(torch.ops._C, "prepack_moe_weight"):
            return False, "none"

        dtype = layer.w13_weight.dtype
        w13_input_size = layer.w13_weight.size(2)
        w13_output_size = layer.w13_weight.size(1)
        w2_input_size = layer.w2_weight.size(2)
        w2_output_size = layer.w2_weight.size(1)

        if not (w13_output_size % 32 == 0 and w2_output_size % 32 == 0):
            return False, "none"

        supports_amx = torch._C._cpu._is_amx_tile_supported()

        if (
            supports_amx
            and dtype == torch.bfloat16
            and w13_input_size % 32 == 0
            and w2_input_size % 32 == 0
        ):
            return True, "amx"

        if supports_amx:
            return False, "none"

        return True, "vec"

    def init_moe_grouped_gemm(
        self,
        layer: torch.nn.Module,
    ) -> None:
        new_w13 = cpu_prepack_moe_weight(layer.w13_weight, self.isa)
        replace_parameter(layer, "w13_weight", new_w13)
        new_w2 = cpu_prepack_moe_weight(layer.w2_weight, self.isa)
        replace_parameter(layer, "w2_weight", new_w2)

    def init_moe_torch(
        self,
        layer: torch.nn.Module,
    ) -> None:
        use_onednn_mm = ops._supports_onednn and ops.is_onednn_acl_supported()
        num_experts = layer.w13_weight.size(0)
        has_w13_bias = hasattr(layer, "w13_bias")
        has_w2_bias = hasattr(layer, "w2_bias")

        layer.gate_up_linear = []
        layer.down_linear = []

        for i in range(num_experts):
            layer_w13_weight = layer.w13_weight[i]
            layer_w13_bias = layer.w13_bias[i] if has_w13_bias else None
            layer_w2_weight = layer.w2_weight[i]
            layer_w2_bias = layer.w2_bias[i] if has_w2_bias else None
            if use_onednn_mm:
                gate_up_handle = ops.create_onednn_mm(layer_w13_weight.t(), 32)
                layer.gate_up_linear.append(
                    lambda x, handle=gate_up_handle, bias=layer_w13_bias: ops.onednn_mm(
                        handle, x, bias
                    )
                )
                down_handle = ops.create_onednn_mm(layer_w2_weight.t(), 32)
                layer.down_linear.append(
                    lambda x, handle=down_handle, bias=layer_w2_bias: ops.onednn_mm(
                        handle, x, bias
                    )
                )
            else:
                layer.gate_up_linear.append(
                    lambda x, w=layer_w13_weight, b=layer_w13_bias: F.linear(x, w, b)
                )
                layer.down_linear.append(
                    lambda x, w=layer_w2_weight, b=layer_w2_bias: F.linear(x, w, b)
                )

        if use_onednn_mm:  # remove weight
            layer.w13_weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)

        _CPU_MOE_LAYER_CACHE[id(layer)] = weakref.ref(layer)

    def forward_grouped_gemm(
        self,
        layer: torch.nn.Module,
        input: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int = -1,
    ) -> torch.Tensor:
        output = cpu_fused_moe(
            input,
            layer.w13_weight,
            layer.w2_weight,
            getattr(layer, "w13_bias", None),
            getattr(layer, "w2_bias", None),
            topk_weights,
            topk_ids,
            activation,
            self.isa,
        )
        return output

    def forward_torch(
        self,
        layer: torch.nn.Module,
        input: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int = -1,
    ) -> torch.Tensor:
        output = torch.empty_like(input)
        layer_id = id(layer)
        torch.ops.vllm.cpu_fused_moe_torch(
            layer_id,
            output,
            input,
            topk_weights,
            topk_ids,
            activation,
            global_num_experts,
        )

        return output


def cpu_fused_moe_torch(
    layer_id: int,
    output: torch.Tensor,
    input: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    global_num_experts: int = -1,
) -> None:
    layer = _CPU_MOE_LAYER_CACHE[layer_id]()

    # Ref code from https://github.com/sgl-project/sglang/blob/716e682721397df103f347d22da8bd46c6016dab/python/sglang/srt/layers/moe/fused_moe_native.py#L53
    len_experts = global_num_experts

    cnts = topk_ids.new_zeros((topk_ids.shape[0], len_experts))
    cnts.scatter_(1, topk_ids.to(torch.int64), 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = topk_ids.view(-1).argsort()

    sorted_tokens = input[idxs // topk_ids.shape[1]]
    tokens_per_expert = tokens_per_expert.cpu().numpy()

    outputs = []
    start_idx = 0

    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        tokens_for_this_expert = sorted_tokens[start_idx:end_idx]

        gate_up = layer.gate_up_linear[i](tokens_for_this_expert)  # type: ignore
        gate_up = _CPU_MOE_ACT_FN[activation](gate_up)
        expert_out = layer.down_linear[i](gate_up)  # type: ignore
        outputs.append(expert_out)
        start_idx = end_idx

    outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
    new_x = torch.empty_like(outs)

    new_x[idxs] = outs
    final_out = (
        new_x.view(*topk_ids.shape, -1)
        .type(topk_weights.dtype)
        .mul_(topk_weights.unsqueeze(dim=-1))
        .sum(dim=1)
        .type(new_x.dtype)
    )
    output.copy_(final_out)


direct_register_custom_op(
    op_name="cpu_fused_moe_torch",
    op_func=cpu_fused_moe_torch,
    mutates_args=["output"],
)
