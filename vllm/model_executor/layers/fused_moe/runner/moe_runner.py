# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Iterable
from contextlib import nullcontext
import os
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.parallel import ExpertPlacementStrategy
from vllm.distributed import (
    get_ep_group,
    get_pcp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.forward_context import (
    ForwardContext,
    get_forward_context,
    is_forward_context_available,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.fused_moe_modular_method import (
    FusedMoEModularMethod,
)
from vllm.model_executor.layers.fused_moe.routed_experts import (
    RoutedExperts,
)
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
    ZeroExpertRouter,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner_interface import (
    MoERunnerInterface,
)
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
    SharedExpertsOrder,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    _USE_LAYERNAME,
    LayerName,
    direct_register_custom_op,
)

logger = init_logger(__name__)

_DSV4_ROUTER_FOCUS_DIMS = (3753, 2404, 5308, 1040, 532, 4265, 5414, 2949)


def _dsv4_debug_runner_tensor(
    counts: dict[str, int],
    layer_name: str,
    label: str,
    tensor: torch.Tensor | None,
    max_logs: int = 8,
) -> None:
    if "layers.60.ffn.experts" not in layer_name:
        return
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return
    if tensor is None:
        logger.warning("[DSV4_MOE_RUNNER_DEBUG] %s:%s None", layer_name, label)
        return
    key = f"{label}:{tuple(tensor.shape)}"
    count = counts.get(key, 0)
    if count >= max_logs:
        return
    counts[key] = count + 1
    with torch.no_grad():
        data = tensor.detach()
        stat = data.float() if data.is_floating_point() else data.to(torch.float32)
        finite = torch.isfinite(stat) if data.is_floating_point() else None
        if finite is not None and not finite.all():
            stat = stat[finite]
        if stat.numel() == 0:
            logger.warning(
                "[DSV4_MOE_RUNNER_DEBUG] %s:%s count=%s shape=%s dtype=%s EMPTY",
                layer_name,
                label,
                count,
                tuple(data.shape),
                data.dtype,
            )
            return
        logger.warning(
            "[DSV4_MOE_RUNNER_DEBUG] %s:%s count=%s shape=%s dtype=%s "
            "finite=%s mean=%.6g std=%.6g max=%.6g min=%.6g nonzero=%.6g",
            layer_name,
            label,
            count,
            tuple(data.shape),
            data.dtype,
            bool(finite.all().item()) if finite is not None else True,
            stat.mean().item(),
            stat.std(unbiased=False).item(),
            stat.abs().max().item(),
            stat.min().item(),
            (data != 0).float().mean().item(),
        )


def _dsv4_debug_router_selection(
    counts: dict[str, int],
    layer_name: str,
    router: FusedMoERouter,
    gate: torch.nn.Module | None,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    input_ids: torch.Tensor | None,
    max_logs: int = 4,
) -> None:
    if "layers.60.ffn.experts" not in layer_name:
        return
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return
    key = f"router_selection:{tuple(router_logits.shape)}"
    count = counts.get(key, 0)
    if count >= max_logs:
        return
    counts[key] = count + 1
    with torch.no_grad():
        hidden = hidden_states.detach().float()
        logits = router_logits.detach().float()
        weights = topk_weights.detach().float()
        ids = topk_ids.detach().to(torch.int64)
        finite = torch.isfinite(logits)
        finite_logits = logits[finite] if finite.numel() else logits
        if finite_logits.numel() == 0:
            finite_logits = logits.reshape(-1)[:0]

        scoring_func = getattr(router, "scoring_func", None)
        routed_scaling_factor = float(
            getattr(router, "routed_scaling_factor", 1.0)
        )
        renormalize = bool(getattr(router, "renormalize", False))
        e_score_bias = getattr(router, "e_score_correction_bias", None)
        bias_124 = None
        top_bias = None
        if e_score_bias is not None and e_score_bias.numel() > 124:
            bias = e_score_bias.detach().float()
            bias_124 = float(bias[124].item())
            top_bias_values, top_bias_ids = torch.topk(
                bias, k=min(12, bias.numel()), dim=-1
            )
            top_bias = [
                (int(expert_id.item()), float(value.item()))
                for expert_id, value in zip(top_bias_ids, top_bias_values)
            ]

        unscaled_weights = (
            weights / routed_scaling_factor
            if routed_scaling_factor != 0.0
            else weights
        )
        row_sums = weights.sum(dim=-1) if weights.ndim >= 2 else weights
        unscaled_row_sums = (
            unscaled_weights.sum(dim=-1)
            if unscaled_weights.ndim >= 2
            else unscaled_weights
        )

        mask_124 = ids == 124
        if mask_124.any():
            weights_124 = weights[mask_124]
            unscaled_124 = unscaled_weights[mask_124]
            token_has_124 = mask_124.any(dim=-1) if mask_124.ndim >= 2 else mask_124
            first_pos = int(torch.nonzero(mask_124.reshape(-1), as_tuple=False)[0].item())
        else:
            weights_124 = weights.reshape(-1)[:0]
            unscaled_124 = unscaled_weights.reshape(-1)[:0]
            token_has_124 = mask_124.reshape(-1)[:0]
            first_pos = -1

        scores_124 = None
        logits_124 = None
        scores = None
        if logits.ndim >= 2 and logits.shape[-1] > 124:
            logits_124 = logits[:, 124]
            if scoring_func == "sigmoid":
                scores = logits.sigmoid()
            elif scoring_func == "softmax":
                scores = torch.softmax(logits, dim=-1)
            elif scoring_func == "sqrtsoftplus":
                scores = torch.sqrt(F.softplus(logits))
            if scores is not None:
                scores_124 = scores[:, 124]

        top_logits_by_mean = None
        top_scores_by_mean = None
        if logits.ndim >= 2 and logits.shape[-1] > 0:
            logits_mean_by_expert = logits.mean(dim=0)
            top_logit_values, top_logit_ids = torch.topk(
                logits_mean_by_expert,
                k=min(12, logits_mean_by_expert.numel()),
                dim=-1,
            )
            top_logits_by_mean = [
                (int(expert_id.item()), float(value.item()))
                for expert_id, value in zip(top_logit_ids, top_logit_values)
            ]
        if scores is not None and scores.ndim >= 2 and scores.shape[-1] > 0:
            scores_mean_by_expert = scores.mean(dim=0)
            top_score_values, top_score_ids = torch.topk(
                scores_mean_by_expert,
                k=min(12, scores_mean_by_expert.numel()),
                dim=-1,
            )
            top_scores_by_mean = [
                (int(expert_id.item()), float(value.item()))
                for expert_id, value in zip(top_score_ids, top_score_values)
            ]

        gate124_stats = None
        gate_top_norms = None
        gate124_recomputed_stats = None
        hidden_norm_stats = None
        hidden_dim_top_signed_124 = None
        hidden_dim_top_abs_124 = None
        expert124_logit_sample = None
        focus_counterfactual = None
        gate_weight = getattr(gate, "weight", None) if gate is not None else None
        if gate_weight is not None and gate_weight.ndim == 2:
            gate_data = gate_weight.detach().float()
            if gate_data.shape[0] > 124:
                gate124 = gate_data[124]
                gate124_stats = (
                    tuple(gate124.shape),
                    float(gate124.mean().item()),
                    float(gate124.std(unbiased=False).item()),
                    float(gate124.abs().max().item()),
                    float(torch.linalg.vector_norm(gate124).item()),
                )
                if hidden.ndim == 2 and hidden.shape[-1] == gate124.shape[-1]:
                    hidden_norm = torch.linalg.vector_norm(hidden, dim=-1)
                    hidden_norm_stats = (
                        float(hidden_norm.mean().item()),
                        float(hidden_norm.std(unbiased=False).item()),
                        float(hidden_norm.min().item()),
                        float(hidden_norm.max().item()),
                    )
                    recomputed_124 = torch.matmul(hidden, gate124)
                    gate_bias = getattr(gate, "bias", None)
                    if gate_bias is not None and gate_bias.numel() > 124:
                        recomputed_124 = recomputed_124 + gate_bias.detach().float()[124]
                    if logits_124 is not None and logits_124.numel():
                        diff = recomputed_124 - logits_124
                        gate124_recomputed_stats = (
                            float(recomputed_124.mean().item()),
                            float(recomputed_124.std(unbiased=False).item()),
                            float(recomputed_124.min().item()),
                            float(recomputed_124.max().item()),
                            float(diff.abs().max().item()),
                            float(diff.abs().mean().item()),
                        )
                        expert124_logit_sample = [
                            float(v.item()) for v in logits_124[:8]
                        ]

                    mean_hidden = hidden.mean(dim=0)
                    mean_abs_hidden = hidden.abs().mean(dim=0)
                    signed_contrib = mean_hidden * gate124
                    abs_contrib = mean_abs_hidden * gate124.abs()
                    top_signed_values, top_signed_ids = torch.topk(
                        signed_contrib.abs(),
                        k=min(16, signed_contrib.numel()),
                        dim=-1,
                    )
                    hidden_dim_top_signed_124 = [
                        (
                            int(dim_id.item()),
                            float(signed_contrib[dim_id].item()),
                            float(mean_hidden[dim_id].item()),
                            float(gate124[dim_id].item()),
                        )
                        for dim_id in top_signed_ids
                    ]
                    top_abs_values, top_abs_ids = torch.topk(
                        abs_contrib,
                        k=min(16, abs_contrib.numel()),
                        dim=-1,
                    )
                    hidden_dim_top_abs_124 = [
                        (
                            int(dim_id.item()),
                            float(top_value.item()),
                            float(mean_abs_hidden[dim_id].item()),
                            float(gate124[dim_id].item()),
                        )
                        for dim_id, top_value in zip(top_abs_ids, top_abs_values)
                    ]
                    focus_dims = [
                        d for d in _DSV4_ROUTER_FOCUS_DIMS if d < hidden.shape[-1]
                    ]
                    if focus_dims and gate_data.shape[-1] == hidden.shape[-1]:
                        focus = torch.tensor(
                            focus_dims, device=hidden.device, dtype=torch.long
                        )
                        hidden_zero = hidden.clone()
                        hidden_zero[:, focus] = 0
                        logits_zero = torch.matmul(hidden_zero, gate_data.T)
                        hidden_centered = hidden.clone()
                        hidden_centered[:, focus] = (
                            hidden_centered[:, focus]
                            - hidden_centered[:, focus].mean(dim=0, keepdim=True)
                        )
                        logits_centered = torch.matmul(hidden_centered, gate_data.T)
                        gate_bias_full = getattr(gate, "bias", None)
                        if gate_bias_full is not None:
                            logits_zero = logits_zero + gate_bias_full.detach().float()
                            logits_centered = (
                                logits_centered + gate_bias_full.detach().float()
                            )
                        zero_mean = logits_zero.mean(dim=0)
                        centered_mean = logits_centered.mean(dim=0)
                        zero_top_vals, zero_top_ids = torch.topk(
                            zero_mean, k=min(8, zero_mean.numel())
                        )
                        centered_top_vals, centered_top_ids = torch.topk(
                            centered_mean, k=min(8, centered_mean.numel())
                        )
                        focus_counterfactual = {
                            "dims": focus_dims,
                            "zero124_mean": float(zero_mean[124].item()),
                            "zero124_rank": int(
                                (zero_mean > zero_mean[124]).sum().item() + 1
                            ),
                            "zero_top": [
                                (int(i.item()), float(v.item()))
                                for i, v in zip(zero_top_ids, zero_top_vals)
                            ],
                            "centered124_mean": float(centered_mean[124].item()),
                            "centered124_rank": int(
                                (centered_mean > centered_mean[124]).sum().item() + 1
                            ),
                            "centered_top": [
                                (int(i.item()), float(v.item()))
                                for i, v in zip(centered_top_ids, centered_top_vals)
                            ],
                        }
            norms = torch.linalg.vector_norm(gate_data, dim=1)
            top_norm_values, top_norm_ids = torch.topk(
                norms, k=min(12, norms.numel()), dim=-1
            )
            gate_top_norms = [
                (int(expert_id.item()), float(value.item()))
                for expert_id, value in zip(top_norm_ids, top_norm_values)
            ]

        logger.warning(
            "[DSV4_ROUTER_SELECTION_DEBUG] layer=%s count=%s "
            "router=%s scoring=%s renormalize=%s routed_scaling_factor=%.6g "
            "bias124=%s input_shape=%s logits_shape=%s logits_dtype=%s "
            "logits_finite=%s logits_mean=%.6g logits_std=%.6g "
            "logits_min=%.6g logits_max=%.6g topk_ids_shape=%s "
            "topk_min=%s topk_max=%s topk_unique=%s weights_shape=%s "
            "weights_mean=%.6g weights_std=%.6g weights_min=%.6g "
            "weights_max=%.6g row_sum_mean=%.6g row_sum_min=%.6g "
            "row_sum_max=%.6g unscaled_row_sum_mean=%.6g "
            "expert124_count=%s expert124_token_frac=%.6g "
            "expert124_weight_mean=%.6g expert124_weight_max=%.6g "
            "expert124_unscaled_mean=%.6g expert124_unscaled_max=%.6g "
            "expert124_logit_mean=%s expert124_logit_max=%s "
            "expert124_score_mean=%s expert124_score_max=%s first124_flat_pos=%s",
            layer_name,
            count,
            type(router).__name__,
            scoring_func,
            renormalize,
            routed_scaling_factor,
            bias_124,
            tuple(input_ids.shape) if input_ids is not None else None,
            tuple(router_logits.shape),
            router_logits.dtype,
            bool(finite.all().item()) if finite.numel() else True,
            finite_logits.mean().item() if finite_logits.numel() else 0.0,
            finite_logits.std(unbiased=False).item()
            if finite_logits.numel()
            else 0.0,
            finite_logits.min().item() if finite_logits.numel() else 0.0,
            finite_logits.max().item() if finite_logits.numel() else 0.0,
            tuple(topk_ids.shape),
            int(ids.min().item()) if ids.numel() else None,
            int(ids.max().item()) if ids.numel() else None,
            int(torch.unique(ids).numel()) if ids.numel() else 0,
            tuple(topk_weights.shape),
            weights.mean().item() if weights.numel() else 0.0,
            weights.std(unbiased=False).item() if weights.numel() else 0.0,
            weights.min().item() if weights.numel() else 0.0,
            weights.max().item() if weights.numel() else 0.0,
            row_sums.mean().item() if row_sums.numel() else 0.0,
            row_sums.min().item() if row_sums.numel() else 0.0,
            row_sums.max().item() if row_sums.numel() else 0.0,
            unscaled_row_sums.mean().item() if unscaled_row_sums.numel() else 0.0,
            int(mask_124.sum().item()) if mask_124.numel() else 0,
            token_has_124.float().mean().item() if token_has_124.numel() else 0.0,
            weights_124.mean().item() if weights_124.numel() else 0.0,
            weights_124.max().item() if weights_124.numel() else 0.0,
            unscaled_124.mean().item() if unscaled_124.numel() else 0.0,
            unscaled_124.max().item() if unscaled_124.numel() else 0.0,
            logits_124.mean().item()
            if logits_124 is not None and logits_124.numel()
            else None,
            logits_124.max().item()
            if logits_124 is not None and logits_124.numel()
            else None,
            scores_124.mean().item()
            if scores_124 is not None and scores_124.numel()
            else None,
            scores_124.max().item()
            if scores_124 is not None and scores_124.numel()
            else None,
            first_pos,
        )

        if weights.numel() and ids.numel():
            flat_ids = ids.reshape(-1)
            flat_weights = weights.reshape(-1)
            unique_ids = torch.unique(flat_ids, sorted=True)
            summaries: list[tuple[int, int, float, float]] = []
            for expert_id in unique_ids[:384]:
                expert_mask = flat_ids == expert_id
                expert_weights = flat_weights[expert_mask]
                summaries.append(
                    (
                        int(expert_id.item()),
                        int(expert_mask.sum().item()),
                        float(expert_weights.mean().item()),
                        float(expert_weights.max().item()),
                    )
                )
            summaries.sort(key=lambda item: item[2], reverse=True)
            logger.warning(
                "[DSV4_ROUTER_SELECTION_DEBUG] layer=%s count=%s "
                "top_experts_by_avg_weight=%s",
                layer_name,
                count,
                summaries[:12],
            )

        logger.warning(
            "[DSV4_ROUTER_GATE_DEBUG] layer=%s count=%s top_bias=%s "
            "top_logits_by_mean=%s top_scores_by_mean=%s "
            "gate124_stats=(shape,mean,std,absmax,l2)=%s "
            "hidden_norm_stats=(mean,std,min,max)=%s "
            "gate124_recomputed=(mean,std,min,max,diff_absmax,diff_absmean)=%s "
            "expert124_logit_sample=%s gate_top_norms=%s",
            layer_name,
            count,
            top_bias,
            top_logits_by_mean,
            top_scores_by_mean,
            gate124_stats,
            hidden_norm_stats,
            gate124_recomputed_stats,
            expert124_logit_sample,
            gate_top_norms,
        )
        logger.warning(
            "[DSV4_ROUTER_INPUT_CONTRIB_DEBUG] layer=%s count=%s "
            "expert124_top_signed_dims=(dim,mean_hidden_x_weight,mean_hidden,weight)=%s "
            "expert124_top_abs_dims=(dim,mean_abs_hidden_x_abs_weight,mean_abs_hidden,weight)=%s "
            "focus_counterfactual=%s",
            layer_name,
            count,
            hidden_dim_top_signed_124,
            hidden_dim_top_abs_124,
            focus_counterfactual,
        )


def register_layer_for_moe_forward_op(
    vllm_config: VllmConfig,
    layer: "MoERunner",
):
    # For smuggling this layer into the fused moe custom op
    prefix = layer.layer_name
    compilation_config = vllm_config.compilation_config
    if prefix in compilation_config.static_forward_context:
        raise ValueError("Duplicate layer name: {}".format(prefix))
    compilation_config.static_forward_context[prefix] = layer
    compilation_config.static_all_moe_layers.append(prefix)


def get_layer_from_name(layer_name: str) -> MoERunnerInterface:
    forward_context: ForwardContext = get_forward_context()
    if not _USE_LAYERNAME and layer_name == "from_forward_context":
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
    layer = forward_context.no_compile_layers[layer_name]
    assert isinstance(layer, MoERunnerInterface)
    return layer


# On torch >= 2.11, layer_name is a hoisted LayerName opaque object;
# on older versions it remains a plain str.
if TYPE_CHECKING:
    from typing import TypeAlias

    _layer_name_type: TypeAlias = str | LayerName
else:
    _layer_name_type = LayerName if _USE_LAYERNAME else str


@torch.compiler.assume_constant_result
def _resolve_layer_name(layer_name: str | LayerName) -> str:
    from torch._library.fake_class_registry import FakeScriptObject

    if isinstance(layer_name, LayerName):
        return layer_name.value
    elif isinstance(layer_name, FakeScriptObject):
        return layer_name.real_obj.value
    return layer_name


# Note: _moe_forward and _moe_forward_shared should not contain any
# implementation details, They should merely pass along control to
# the runner's '_forward_impl' method.
# These functions should never be called directly since they do not
# include all the functionality of the MoE layer.
def _moe_forward(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
) -> torch.Tensor:
    layer = get_layer_from_name(_resolve_layer_name(layer_name))
    return layer._forward_impl(
        hidden_states,
        router_logits,
        shared_experts_input,
        input_ids,
    )


def _moe_forward_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
) -> torch.Tensor:
    # `hidden_dim_unpadded > 0` only on the TRT-LLM MXFP4 path, where the
    # real kernel writes narrower than `hidden_states.shape[-1]`. Plumbed
    # as an op arg (not peeked from the layer registry) to keep the fake
    # a pure shape function of its inputs and preserve subgraph dedup.
    if hidden_dim_unpadded > 0:
        return hidden_states.new_empty((*hidden_states.shape[:-1], hidden_dim_unpadded))
    return torch.empty_like(hidden_states)


def _moe_forward_shared(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    layer = get_layer_from_name(_resolve_layer_name(layer_name))
    return layer._forward_impl(
        hidden_states,
        router_logits,
        shared_experts_input,
        input_ids,
    )


def _moe_forward_shared_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # `fused_out`: see `_moe_forward_fake` for hidden_dim_unpadded semantics.
    # `shared_out`: matches `shared_experts_input` if provided (latent MoE),
    # else `hidden_states`.
    if hidden_dim_unpadded > 0:
        fused_out = hidden_states.new_empty(
            (*hidden_states.shape[:-1], hidden_dim_unpadded)
        )
    else:
        fused_out = torch.empty_like(hidden_states)
    if shared_experts_input is not None:
        shared_out = torch.empty_like(shared_experts_input)
    else:
        shared_out = torch.empty_like(hidden_states)
    return shared_out, fused_out


# NOTE: `moe_forward` and `moe_forward_shared` being opaque custom ops is a
# load-bearing assumption for the MoE-LoRA dual-stream path.
direct_register_custom_op(
    op_name="moe_forward",
    op_func=_moe_forward,
    mutates_args=["hidden_states"],
    fake_impl=_moe_forward_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


direct_register_custom_op(
    op_name="moe_forward_shared",
    op_func=_moe_forward_shared,
    fake_impl=_moe_forward_shared_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def _unpack(
    result: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor | None, torch.Tensor]:
    if isinstance(result, tuple):
        return result
    else:
        return (None, result)


class MoERunner(MoERunnerInterface):
    """
    Standard MoE runner implementation for executing Mixture of Experts layers.

    This is the primary concrete implementation of MoE execution logic, providing
    comprehensive support for standard MoE operations. It handles:
    - Expert routing and token dispatching using various routing strategies
    - Shared experts computation with optional parallel execution using CUDA streams
    - Tensor model parallel and expert parallel operations
    - Multiple quantization methods and optimized kernel selection
    - Both monolithic and decomposed expert execution paths
    - Integration with various parallel execution modes (TP, EP, DP)

    The runner orchestrates the complete MoE forward pass including routing tokens
    to experts, executing expert computations in parallel, and combining results.
    It supports advanced features like overlapped execution of shared experts,
    optimized kernels for different parallel configurations, and seamless
    integration with vLLM's distributed execution framework.

    Eventually, this class may be split into more specialized implementations
    for different configurations (e.g., with/without shared experts, gates, etc.).
    """

    def __init__(
        self,
        layer_name: str,
        moe_config: FusedMoEConfig,
        router: FusedMoERouter,
        routed_experts: RoutedExperts,
        enable_dbo: bool = False,
        gate: torch.nn.Module | None = None,
        shared_experts: torch.nn.Module | None = None,
        shared_expert_gate: torch.nn.Module | None = None,
        routed_input_transform: torch.nn.Module | None = None,
        routed_output_transform: torch.nn.Module | None = None,
        routed_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.moe_config = moe_config
        self.router = router
        self.routed_input_transform = routed_input_transform
        self.routed_output_transform = routed_output_transform
        self.routed_scaling_factor = routed_scaling_factor
        self.gate = gate
        self.shared_expert_gate = shared_expert_gate
        self.routed_experts = routed_experts
        self.enable_dbo = enable_dbo

        # When both gates are present and FSE is enabled, fuse their
        # weight matrices into [num_experts + num_shared, hidden] so one
        # F.linear produces combined logits. The topk kernel can then
        # apply routing softmax and shared expert activation (sigmoid)
        # in a single launch.
        self._fse_fuse_gate = gate is not None and shared_expert_gate is not None
        self._combined_gate_weight: torch.Tensor | None = None

        self._shared_experts: SharedExperts | None = None
        if shared_experts is not None:
            can_overlap = lambda: self._quant_method.mk_can_overlap_shared_experts
            self._shared_experts = SharedExperts(
                shared_experts,
                moe_config=moe_config,
                enable_dbo=enable_dbo,
                mk_can_overlap_shared_experts=can_overlap,
            )

        # Needed for string -> MoERunner layer lookup in custom ops.
        self.layer_name = layer_name
        self._dsv4_debug_counts: dict[str, int] = {}

        self._forward_entry = self._select_forward()

        # For smuggling this layer into the fused moe custom op
        register_layer_for_moe_forward_op(get_current_vllm_config(), self)

    def _select_forward(self) -> Callable:
        if current_platform.is_tpu() or current_platform.is_cpu():
            # TODO: Once the OOM issue for the TPU backend is resolved, we
            # will switch to using the moe_forward custom op.
            # Note: CPU doesn't require wrapped _forward_impl.
            return _moe_forward if self._shared_experts is None else _moe_forward_shared

        return (
            torch.ops.vllm.moe_forward
            if self._shared_experts is None
            else torch.ops.vllm.moe_forward_shared
        )

    @property
    def shared_experts(self) -> SharedExperts | None:
        return self._shared_experts

    @property
    def is_internal_router(self) -> bool:
        return self.gate is not None

    # TODO(bnell): Temporary hack. Get rid of this.
    def _replace_quant_method(self, quant_method: FusedMoEMethodBase):
        self.routed_experts._replace_quant_method(quant_method)

    # TODO(bnell): Hack for elastic_ep. Get rid of this
    def _set_moe_config(self, new_moe_config: FusedMoEConfig):
        self.moe_config = new_moe_config
        self.routed_experts._set_moe_config(new_moe_config)
        if self._shared_experts is not None:
            self._shared_experts._set_moe_config(new_moe_config)

    def _maybe_fuse_gate_weights(self):
        """Fuse router and shared expert gate weights on first call.

        Cannot be done at __init__ because gate weights are loaded after
        module construction (via weight_loader). Called once from
        _forward_impl before the first forward pass.
        """
        if self._combined_gate_weight is None:
            assert self.gate is not None and self.shared_expert_gate is not None
            self._combined_gate_weight = torch.cat(
                [self.gate.weight, self.shared_expert_gate.weight],
                dim=0,
            )

    @property
    def _quant_method(self) -> FusedMoEMethodBase:
        return self.routed_experts.quant_method

    def apply_routed_input_transform(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply transform for routed experts (e.g., latent projection).

        This is called by FusedMoE.forward_native. The original hidden_states
        is saved separately so shared experts get [S, hidden_size] while
        routed experts get the transformed [S, moe_latent_size].

        Returns (possibly transformed) hidden states and the input for shared
        experts (or None if there are no shared experts).
        """
        if self.routed_input_transform is not None:
            result = self.routed_input_transform(hidden_states)
            # ReplicatedLinear returns (output, extra_bias) tuple.
            # We only need the output tensor; extra_bias is not used here.
            if isinstance(result, tuple):
                return result[0], hidden_states
            return result, hidden_states

        return (
            hidden_states,
            hidden_states if self._shared_experts is not None else None,
        )

    def apply_routed_output_transform(
        self,
        fused_output: torch.Tensor,
    ) -> torch.Tensor:
        """Apply transform to routed expert output (e.g., latent to full dim).

        Used by latent MoE models (e.g., NemotronH) where routed experts
        operate in a compressed latent space and need projection back to
        the full hidden dimension before combining with shared expert output.
        """
        if self.routed_output_transform is not None:
            r = self.routed_output_transform(fused_output)
            fused_output = r[0] if isinstance(r, tuple) else r
        return fused_output

    def _maybe_apply_routed_scale_to_output(
        self,
        shared_output: torch.Tensor | None,
        fused_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Apply routed_scaling_factor to the output with FP16 overflow
        protection.

        Scale the fused expert output by routed_scaling_factor. For FP16,
        avoid overflow by dividing shared_output by the scale instead
        (the decoder layer compensates with matching divisions).
        """
        if self.routed_scaling_factor != 1.0:
            if fused_output.dtype != torch.float16 or shared_output is None:
                fused_output *= self.routed_scaling_factor
            elif shared_output is not None:
                shared_output *= 1.0 / self.routed_scaling_factor
        return shared_output, fused_output

    @property
    def _fused_output_is_reduced(self) -> bool:
        return (
            self._quant_method.moe_kernel is not None
            and self._quant_method.moe_kernel.output_is_reduced()
        )

    def _maybe_reduce_shared_expert_output(
        self,
        shared_output: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """All-reduce shared expert output when the combine kernel already
        reduced fused output.

        * If the combine kernel does the reduction for fused_output, reduce
          shared_output separately. O.w, reduce fused_output+shared_output later.
        * If we have SP (TP=N, DP=M, EP), there is a separate AG step handled
          in the model.
        """
        if (
            shared_output is not None
            and not self.moe_config.is_sequence_parallel
            and self._fused_output_is_reduced
        ):
            shared_output = tensor_model_parallel_all_reduce(shared_output)
        return shared_output

    def _maybe_reduce_final_output(
        self,
        states: torch.Tensor,
        trunc_size: int | None,
    ) -> torch.Tensor:
        """All-reduce the combined output if needed.

        This is the "late" all-reduce path. When neither fused nor shared
        output was individually reduced, the combined sum is all-reduced
        here. Skipped when sequence-parallel is active (SP handles its
        own reduction) or when the early path already reduced both outputs.
        """
        # We don't need to reduce the final output if:
        # - We are not running with TP or DP
        # - The MK already reduced the fused output itself.
        if (
            not self.moe_config.is_sequence_parallel
            and (self.moe_config.tp_size > 1 or self.moe_config.ep_size > 1)
            and not self._fused_output_is_reduced
        ):
            states = tensor_model_parallel_all_reduce(states)

        return states[..., :trunc_size] if trunc_size is not None else states

    def _encode_layer_name(self) -> str | LayerName:
        if _USE_LAYERNAME:
            return LayerName(self.layer_name)
        # Can be unavailable or None in unittests
        if (
            is_forward_context_available()
            and get_forward_context().all_moe_layers is not None
        ):
            return "from_forward_context"
        return self.layer_name

    def _maybe_pad_hidden_states(
        self,
        shared_experts_input: torch.Tensor | None,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, int | None, int | None]:
        """Pad hidden_states to moe_config.hidden_dim and compute the
        original dimension for later truncation.

        For latent MoE, the routed hidden_states may be smaller than
        hidden_dim. Padding ensures uniform tensor sizes through the
        fused MoE kernel. The returned trunc_size is used by
        _maybe_reduce_final_output to strip the padding from the result.
        """
        shared_experts_hidden_dim = (
            shared_experts_input.shape[-1] if shared_experts_input is not None else 0
        )
        transformed_hidden_dim: int | None = hidden_states.shape[-1]
        if (
            not self._quant_method.skip_forward_padding
            and self.moe_config.hidden_dim != transformed_hidden_dim
        ):
            assert transformed_hidden_dim is not None
            hidden_states = F.pad(
                hidden_states,
                (0, self.moe_config.hidden_dim - transformed_hidden_dim),
                mode="constant",
                value=0.0,
            )

        # Truncation sizes for stripping kernel padding from the output.
        # None means no truncation needed (no padding was applied).
        #
        # Two truncation points exist in forward():
        #   pre_xform:  applied to fused_output BEFORE routed_output_transform
        #   post_xform: applied to the final result AFTER all-reduce
        #
        # Latent MoE with shared experts (NemotronH):
        #   - pre_xform strips padding from the latent dim so
        #     routed_output_transform receives the correct input size
        #   - post_xform truncates to shared_experts_hidden_dim (full hidden)
        #     after shared + routed outputs are combined and all-reduced
        #
        # Standard MoE / MoE without transforms (GPT-OSS, Mixtral):
        #   - pre_xform is None (no early truncation)
        #   - post_xform strips padding after all-reduce (or None if unpadded)
        if transformed_hidden_dim == hidden_states.shape[-1]:
            transformed_hidden_dim = None

        if self.routed_output_transform is not None and shared_experts_hidden_dim > 0:
            pre_xform_trunc_size = transformed_hidden_dim
            post_xform_trunc_size = shared_experts_hidden_dim
        else:
            pre_xform_trunc_size = None
            post_xform_trunc_size = transformed_hidden_dim

        return hidden_states, pre_xform_trunc_size, post_xform_trunc_size

    def _maybe_apply_shared_experts(
        self,
        shared_experts_input: torch.Tensor | None,
        order: SharedExpertsOrder,
    ):
        if self._shared_experts is not None:
            assert shared_experts_input is not None
            self._shared_experts(shared_experts_input, order)

    def _apply_quant_method(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Run expert routing and the fused MoE kernel via the quant method.

        Orchestrates shared expert execution (before/after), expert selection
        via the router, and the actual fused MoE computation. Returns
        (shared_expert_output, fused_expert_output).
        """
        self._maybe_apply_shared_experts(
            shared_experts_input, SharedExpertsOrder.NO_OVERLAP
        )

        if self.routed_experts.quant_method.is_monolithic:
            # Monolithic kernels: pass router_logits to routed_experts
            fused_out = self.routed_experts.forward_monolithic(
                x=hidden_states,
                router_logits=router_logits,
                input_ids=input_ids,
            )
        else:
            # Modular kernels: select experts first, then call routed_experts
            topk_weights, topk_ids = self.router.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_indices_dtype=self._quant_method.topk_indices_dtype,
                input_ids=input_ids,
            )
            _dsv4_debug_router_selection(
                self._dsv4_debug_counts,
                self.layer_name,
                self.router,
                self.gate,
                hidden_states,
                router_logits,
                topk_weights,
                topk_ids,
                input_ids,
            )

            fused_out = self.routed_experts.forward_modular(
                x=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                shared_experts=self._shared_experts,
                shared_experts_input=shared_experts_input,
            )

        self._maybe_apply_shared_experts(
            shared_experts_input,
            SharedExpertsOrder.MULTI_STREAM_OVERLAPPED,
        )

        return (
            self._shared_experts.output if self._shared_experts is not None else None,
            fused_out,
        )

    def _sequence_parallel_context(self):
        """Return a context manager for sequence-parallel token
        redistribution.

        When sequence parallelism is active, returns a context that handles
        local size tracking for proper token scatter/gather. Otherwise
        returns a no-op context.
        """
        ctx = get_forward_context()
        return (
            ctx.dp_metadata.sp_local_sizes(self.moe_config.sp_size)
            if ctx.dp_metadata
            else nullcontext()
        )

    def _maybe_sync_shared_experts_stream(
        self,
        shared_experts_input: torch.Tensor | None,
    ):
        # If router/gate provided, then apply it here.
        # (Note: This code runs only when "overlapped mode" is on to allow
        #        parallel execution of shared experts with the FusedMoE via
        #        separate cuda stream)
        if self._shared_experts is not None:
            assert shared_experts_input is not None
            self._shared_experts.maybe_sync_shared_experts_stream(shared_experts_input)

    def _maybe_add_zero_expert_output(
        self,
        result: torch.Tensor,
    ) -> torch.Tensor:
        """Add the zero expert's contribution to the final result.

        When a ZeroExpertRouter is used, it computes a bias-like output
        from the "zero expert" that is added to the combined routed+shared
        expert output.
        """
        if isinstance(self.router, ZeroExpertRouter):
            zero_expert_output = self.router.zero_expert_output
            assert zero_expert_output is not None
            result = result + zero_expert_output
        return result

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Invoke the fused moe layer.

        Input:
        - hidden_states
        - router_logits

        Output:
        - The new hidden_states.

        Calling sequence
        - forward
          - self._forward_entry (_moe_forward or _moe_forward_shared custom op)
            - _forward_impl

        Note: The existence of _moe_forward and _moe_forward_shared custom ops are due
        to the following reason:
        1. pytorch cannot handle union types in custom op signatures so
           _moe_forward and _moe_forward_shared must be split.
        """

        # Apply transform for routed experts (e.g., latent projection
        # for latent MoE)
        hidden_states, shared_experts_input = self.apply_routed_input_transform(
            hidden_states
        )

        # Record before `_maybe_pad_hidden_states` pads activations to match
        # `moe_config.hidden_dim`, e.g. after `align_trtllm_fp4_moe_hidden_dim_for_fi`
        # so routed output can be trimmed before
        # shared+routed add / latent up proj if needed.

        hidden_states, og_hidden_dim_pre_xform, og_hidden_dim_post_xform = (
            self._maybe_pad_hidden_states(
                shared_experts_input,
                hidden_states,
            )
        )

        result = self._forward_entry(
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
            self._encode_layer_name(),
            self.moe_config.hidden_dim_unpadded
            if self._quant_method.has_unpadded_output
            else 0,
        )

        #
        # Note: there are two all-reduce points below. They are mutually
        # exclusive, controlled by _fused_output_is_reduced
        #  - When True: the combine kernel already reduced fused_output,
        #    so we reduce shared_output here to match, then skip the
        #    all-reduce in _maybe_reduce_final_output.
        #  - When False: neither output is reduced yet, so we combine
        #    them first and all-reduce the sum in _maybe_reduce_final_output.

        # Extract outputs from result
        shared_output, fused_output = _unpack(result)
        _dsv4_debug_runner_tensor(
            self._dsv4_debug_counts,
            self.layer_name,
            "shared_output_raw",
            shared_output,
        )
        _dsv4_debug_runner_tensor(
            self._dsv4_debug_counts,
            self.layer_name,
            "fused_output_raw",
            fused_output,
        )

        if og_hidden_dim_pre_xform is not None:
            fused_output = fused_output[..., :og_hidden_dim_pre_xform]

        # If combine kernel already reduced fused, reduce shared to match.
        # See note above re: the two all-reduce points.
        shared_output = self._maybe_reduce_shared_expert_output(shared_output)
        _dsv4_debug_runner_tensor(
            self._dsv4_debug_counts,
            self.layer_name,
            "shared_output_after_maybe_reduce",
            shared_output,
        )

        shared_output, fused_output = self._maybe_apply_routed_scale_to_output(
            shared_output, fused_output
        )
        _dsv4_debug_runner_tensor(
            self._dsv4_debug_counts,
            self.layer_name,
            "shared_output_after_routed_scale",
            shared_output,
        )
        _dsv4_debug_runner_tensor(
            self._dsv4_debug_counts,
            self.layer_name,
            "fused_output_after_routed_scale",
            fused_output,
        )

        if (
            shared_output is not None
            and "layers.60.ffn.experts" in self.layer_name
        ):
            shared_scale = os.environ.get("DSV4_DEBUG_LAYER60_SHARED_SCALE")
            if shared_scale is not None:
                try:
                    shared_scale_value = float(shared_scale)
                except ValueError:
                    shared_scale_value = 1.0
                    logger.warning(
                        "[DSV4_MOE_RUNNER_DEBUG] %s invalid "
                        "DSV4_DEBUG_LAYER60_SHARED_SCALE=%r; using 1.0",
                        self.layer_name,
                        shared_scale,
                    )
                logger.warning(
                    "[DSV4_MOE_RUNNER_DEBUG] %s applying "
                    "DSV4_DEBUG_LAYER60_SHARED_SCALE=%s",
                    self.layer_name,
                    shared_scale_value,
                )
                shared_output = shared_output * shared_scale_value
                _dsv4_debug_runner_tensor(
                    self._dsv4_debug_counts,
                    self.layer_name,
                    "shared_output_after_debug_scale",
                    shared_output,
                )

        # Apply output transform (e.g. latent -> full dim)
        fused_output = self.apply_routed_output_transform(fused_output)

        if shared_output is not None:
            result = shared_output + fused_output
        else:
            result = fused_output
        _dsv4_debug_runner_tensor(
            self._dsv4_debug_counts,
            self.layer_name,
            "result_before_final_reduce",
            result,
        )

        result = self._maybe_reduce_final_output(result, og_hidden_dim_post_xform)
        _dsv4_debug_runner_tensor(
            self._dsv4_debug_counts,
            self.layer_name,
            "result_after_final_reduce",
            result,
        )

        return self._maybe_add_zero_expert_output(result)

    @property
    def do_naive_dispatch_combine(self) -> bool:
        return (
            self.moe_config.dp_size > 1 and not self._quant_method.supports_internal_mk
        )

    def _maybe_dispatch(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # For naive dispatch/combine Dp/Ep, dispatch the hidden states and
        # router logits to all experts.
        # NOTE: this will be removed once all kernels are migrated into the
        # MoEKernel framework.
        if self.do_naive_dispatch_combine:
            result = get_ep_group().dispatch_router_logits(
                hidden_states,
                router_logits,
                self.moe_config.is_sequence_parallel,
            )
            assert len(result) == 2
            hidden_states, router_logits = result

        # NOTE: Similar with DP, PCP also needs dispatch and combine. For
        # simplicity, AgRsAll2All was added separately for PCP here. Maybe
        # we should modify All2AllManager abstraction to better support PCP.
        if self.moe_config.pcp_size > 1:
            hidden_states = get_pcp_group().all_gather(
                hidden_states,
                dim=0,
            )
            router_logits = get_pcp_group().all_gather(
                router_logits,
                dim=0,
            )

        return hidden_states, router_logits

    def _maybe_combine(
        self,
        shared_output: torch.Tensor | None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor | None, torch.Tensor]:
        if self.do_naive_dispatch_combine:
            hidden_states = get_ep_group().combine(
                hidden_states, self.moe_config.is_sequence_parallel
            )

        if self.moe_config.pcp_size > 1:
            hidden_states = get_pcp_group().reduce_scatter(
                hidden_states,
                dim=0,
            )

        if self.shared_experts is not None:
            assert shared_output is not None
            return shared_output, hidden_states
        else:
            return hidden_states

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Entry point called by the custom op to run the MoE computation.

        Handles pre-dispatch setup (gate application, external shared expert
        triggering, quant config init) then performs the following steps
        within the sequence-parallel context.

        - Performs expert routing
        - fused MoE kernel execution
        - shared expert computation.

        Returns a single tensor of combined fused and shared output (if present).
        """
        # TODO(bnell): this can be removed after MK migration is complete.
        self.routed_experts._ensure_moe_quant_config_init()

        # Sync aux and main stream for shared expert multi-stream overlap.
        self._maybe_sync_shared_experts_stream(shared_experts_input)

        # If the Runner holds the gate, apply it after the stream sync,
        # so it can run overlapped with the
        # NOTE: in future PR, MoE runner will always hold the gate.
        if self.gate is not None:
            if self._fse_fuse_gate:
                self._maybe_fuse_gate_weights()
                router_logits = F.linear(hidden_states, self._combined_gate_weight)
            else:
                router_logits, _ = self.gate(hidden_states)

        with self._sequence_parallel_context():
            # TODO(bnell): parts of the dispatch/combine steps will go away once
            # #32567 lands and the remaining kernels are made MKs.  The PCP
            # code will probably remain
            hidden_states, router_logits = self._maybe_dispatch(
                hidden_states,
                router_logits,
            )

            shared_output, hidden_states = self._apply_quant_method(
                hidden_states=hidden_states,
                router_logits=router_logits,
                shared_experts_input=shared_experts_input,
                input_ids=input_ids,
            )

            return self._maybe_combine(
                shared_output,
                hidden_states,
            )

    #########################################################
    #
    # Old methods from FusedMoE layer. Remove when possible.
    #
    #########################################################

    # Note: maybe_init_modular_kernel should only be called by
    # prepare_communication_buffer_for_model.
    # This is called after all weight loading and post-processing, so it
    # should be safe to swap out the quant_method.
    def maybe_init_modular_kernel(self) -> None:
        # NOTE(rob): WIP refactor. For quant methods that own the MK
        # we create the MK during process_weights_after_loading.
        if (
            self.routed_experts.quant_method.supports_internal_mk
            or self.routed_experts.quant_method.is_monolithic
        ):
            return None

        self.routed_experts._ensure_moe_quant_config_init()
        # routing_tables only needed for round-robin expert placement with
        # DeepEP all2all backend.
        routing_tables = self._expert_routing_tables()

        if isinstance(self.routed_experts.quant_method, FusedMoEModularMethod):
            base_quant_method = self.routed_experts.quant_method.old_quant_method
        else:
            base_quant_method = self.routed_experts.quant_method

        prepare_finalize = base_quant_method.maybe_make_prepare_finalize(
            routing_tables=routing_tables
        )
        if prepare_finalize is not None:
            logger.debug(
                "%s for %s(%s)", prepare_finalize.__class__.__name__, self, id(self)
            )
            self._replace_quant_method(
                FusedMoEModularMethod.make(
                    self.routed_experts,
                    base_quant_method,
                    prepare_finalize,
                )
            )

    #
    # Properties
    #

    @property
    def layer_id(self):
        # Delayed import to avoid circular dependency
        from vllm.model_executor.models.utils import extract_layer_index

        return extract_layer_index(self.layer_name)

    #
    # Attributes still needed by models
    #

    @property
    def is_monolithic(self) -> bool:
        return self.routed_experts.quant_method.is_monolithic

    @property
    def activation(self) -> MoEActivation:
        return self.routed_experts.activation

    #
    # Expert maps
    #

    @property
    def expert_map_manager(self):
        """Forward to routed_experts.expert_map_manager for backward compatibility."""
        return self.routed_experts.expert_map_manager

    @property
    def expert_placement_strategy(self) -> ExpertPlacementStrategy:
        return self.expert_map_manager.placement_strategy

    @property
    def expert_global_to_physical(self) -> torch.Tensor | None:
        tables = self.expert_map_manager.routing_tables
        return tables[0] if tables else None

    @property
    def expert_physical_to_global(self) -> torch.Tensor | None:
        """Routing table: physical expert ID to global expert ID."""
        tables = self.expert_map_manager.routing_tables
        return tables[1] if tables else None

    @property
    def expert_local_to_global(self) -> torch.Tensor | None:
        """Routing table: local expert ID to global expert ID."""
        tables = self.expert_map_manager.routing_tables
        return tables[2] if tables else None

    @property
    def expert_map(self) -> torch.Tensor | None:
        return self.routed_experts.expert_map

    def _expert_routing_tables(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        return self.routed_experts._expert_routing_tables()

    def update_expert_map(self):
        self.routed_experts.update_expert_map()

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        """Map global expert ID to local expert ID."""
        return self.routed_experts._map_global_expert_id_to_local_expert_id(expert_id)

    def get_expert_weights(self) -> Iterable[torch.Tensor]:
        return self.routed_experts.get_expert_weights()

    #
    # EPLB
    #

    @property
    def eplb_state(self) -> EplbLayerState | None:
        return self.router.eplb_state

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
        if self.router.eplb_state is not None:
            self.router.eplb_state.set_layer_state(
                moe_layer_idx,
                expert_load_view,
                logical_to_physical_map,
                logical_replica_count,
            )
