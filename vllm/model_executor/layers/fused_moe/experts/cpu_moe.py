# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU fused MoE experts."""

import weakref
from collections.abc import Callable

import torch
from torch.nn import functional as F

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm._custom_ops import (
    CPUQuantAlgo,
    CPUQuantMethod,
    convert_weight_packed_scale_zp,
    cpu_fused_moe,
    cpu_fused_moe_int8,
    cpu_prepack_moe_weight,
    cpu_prepack_moe_weight_int8,
    fused_experts_cpu,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
    kInt4Static,
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
    kMxfp4Static,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import CpuArchEnum, current_platform
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import direct_register_custom_op

# ===========================================================================
# Routing
# ===========================================================================


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


# ===========================================================================
# Unquantized (BF16/FP16/FP32) MoE
# ===========================================================================

# The CPU grouped-gemm MoE kernels (AMX and vector) tile the expert
# intermediate ("N") dimension in blocks of this size and have no tail/
# remainder handling, so a shard is only eligible for the fast path when
# its per-partition intermediate size is a multiple of it.
_MOE_GROUPED_GEMM_N_TILE = 32

_CPU_MOE_LAYER_CACHE: dict[int, "weakref.ReferenceType[torch.nn.Module]"] = {}


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


def _gelu_and_mul(
    x: torch.Tensor,
) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.gelu(x[..., :d], approximate="none") * x[..., d:]


# Map activation names to their native forward functions.
# Uses static methods or standalone functions to avoid instantiating CustomOp
# classes, which would call get_current_vllm_config() before config is set.
_CPU_MOE_ACT_FN: dict[MoEActivation, Callable[[torch.Tensor], torch.Tensor]] = {
    MoEActivation.SILU: SiluAndMul.forward_native,
    MoEActivation.SWIGLUOAI: _swigluoai_forward_native,
    MoEActivation.GELU: _gelu_and_mul,
    MoEActivation.GELU_TANH: (
        lambda x: F.gelu(x[..., : x.shape[-1] // 2], approximate="tanh")
        * x[..., x.shape[-1] // 2 :]
    ),
}


def _w13_output_size(moe_config: FusedMoEConfig, intermediate_size: int) -> int:
    return 2 * intermediate_size if moe_config.is_act_and_mul else intermediate_size


def _padded_intermediate_size(moe_config: FusedMoEConfig) -> int:
    """Per-partition MoE intermediate size the grouped-gemm kernels will see.

    Zero-padding lets the kernels be used even when TP sharding lands on an
    unaligned value (e.g. moe_intermediate_size=704 at tp=4 -> 176). It cannot
    be applied to interleaved gate/up layouts (swigluoai).
    """
    intermediate_size = moe_config.intermediate_size_per_partition
    if moe_config.activation == MoEActivation.SWIGLUOAI:
        return intermediate_size
    return round_up(intermediate_size, _MOE_GROUPED_GEMM_N_TILE)


def _grouped_gemm_alignment_error(moe_config: FusedMoEConfig) -> str:
    return (
        "CPU fused-MoE grouped-gemm kernel cannot be used for a layer with "
        f"hidden size {moe_config.hidden_dim} and per-partition MoE "
        f"intermediate size {moe_config.intermediate_size_per_partition}: "
        f"both must be a multiple of {_MOE_GROUPED_GEMM_N_TILE}, and "
        "automatic zero-padding of the intermediate size could not resolve "
        "this (typically because the activation uses an interleaved gate/up "
        f"layout, e.g. {MoEActivation.SWIGLUOAI}). vLLM refuses to silently "
        "fall back to the much slower per-expert torch loop on x86; consider "
        "a different --tensor-parallel-size."
    )


def _x86_grouped_gemm_isa(moe_config: FusedMoEConfig) -> str:
    """AMX only implements bf16 and requires 32-aligned reduction dims; the
    vector kernel covers every other supported dtype and shape."""
    if (
        torch.cpu._is_amx_tile_supported()
        and moe_config.in_dtype == torch.bfloat16
        and moe_config.hidden_dim % 32 == 0
        and _padded_intermediate_size(moe_config) % 32 == 0
    ):
        return "amx"
    return "vec"


def _neon_grouped_gemm_support(
    moe_config: FusedMoEConfig,
) -> tuple[bool, str | None]:
    if moe_config.in_dtype != torch.bfloat16:
        return False, "kernel requires bfloat16 activations"
    intermediate_size = moe_config.intermediate_size_per_partition
    if (
        moe_config.hidden_dim % 32 != 0
        or _w13_output_size(moe_config, intermediate_size) % 32 != 0
    ):
        return False, "kernel requires 32-aligned weight output dims"
    if intermediate_size % 4 != 0:
        return False, "kernel requires 4-aligned weight input dims"
    if (
        moe_config.activation == MoEActivation.SWIGLUOAI
        and intermediate_size % _MOE_GROUPED_GEMM_N_TILE != 0
    ):
        return False, "kernel requires a 32-aligned interleaved gate/up dim"
    return True, None


class CPUUnquantizedExperts(mk.FusedMoEExpertsMonolithic):
    """Base class for the unquantized (bf16/fp16/fp32) CPU MoE experts."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        # Router configuration that the monolithic apply() signature cannot
        # carry. Captured off the layer in process_weights_after_loading.
        self.use_grouped_topk = False
        self.renormalize = False
        self.scoring_func = "softmax"
        self.custom_routing_function: Callable | None = None

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in _CPU_MOE_ACT_FN

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (None, None)

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return True

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        # Routing runs in select_experts(), which covers every routing method
        # a layer can be configured with, including custom routing functions.
        return True

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype | None,
        routing_method: RoutingMethodType,
    ) -> bool:
        return True

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.use_grouped_topk = layer.use_grouped_topk
        self.renormalize = layer.renormalize
        self.scoring_func = layer.scoring_func
        self.custom_routing_function = layer.custom_routing_function

    def _select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        num_expert_group: int | None,
        topk_group: int | None,
        e_score_correction_bias: torch.Tensor | None,
        routed_scaling_factor: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=self.moe_config.experts_per_token,
            use_grouped_topk=self.use_grouped_topk,
            renormalize=self.renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            routed_scaling_factor=(
                routed_scaling_factor if routed_scaling_factor is not None else 1.0
            ),
            e_score_correction_bias=e_score_correction_bias,
        )


class CPUGroupedGemmExperts(CPUUnquantizedExperts):
    """Base class for the prepacked grouped-gemm CPU MoE kernels."""

    isa: str

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        replace_parameter(
            layer, "w13_weight", cpu_prepack_moe_weight(layer.w13_weight, self.isa)
        )
        replace_parameter(
            layer, "w2_weight", cpu_prepack_moe_weight(layer.w2_weight, self.isa)
        )

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        # grouped topk + fused topk bias parameters
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        topk_weights, topk_ids = self._select_experts(
            hidden_states,
            router_logits,
            num_expert_group,
            topk_group,
            e_score_correction_bias,
            routed_scaling_factor,
        )

        if apply_router_weight_on_input:
            assert topk_ids.size(1) == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            hidden_states.mul_(topk_weights.to(hidden_states.dtype))

        return cpu_fused_moe(
            hidden_states,
            w1,
            w2,
            self.w1_bias,
            self.w2_bias,
            topk_weights,
            topk_ids,
            activation.value,
            self.isa,
            apply_router_weight_on_input,
        )


class X86CPUUnquantizedExperts(CPUGroupedGemmExperts):
    """x86 AMX / vector grouped-gemm unquantized MoE experts.

    This is the only unquantized MoE implementation on x86: vLLM refuses to
    silently fall back to the much slower per-expert torch loop.
    """

    @staticmethod
    def _supports_current_device() -> bool:
        return (
            current_platform.is_cpu()
            and current_platform.get_cpu_architecture() == CpuArchEnum.X86
            and hasattr(torch.ops._C, "prepack_moe_weight")
        )

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        supported, reason = mk.FusedMoEExperts.is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )
        if not supported:
            return supported, reason
        # The prepack kernel requires both weights' output dims to be a
        # multiple of 32, and the grouped gemm tiles the intermediate dim in
        # blocks of 32. w2's output dim is the hidden size; w13's output dim
        # and w2's input dim both follow from the (possibly padded)
        # intermediate size.
        if (
            moe_config.hidden_dim % _MOE_GROUPED_GEMM_N_TILE != 0
            or _padded_intermediate_size(moe_config) % _MOE_GROUPED_GEMM_N_TILE != 0
        ):
            raise RuntimeError(_grouped_gemm_alignment_error(moe_config))
        return True, None

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        self.isa = _x86_grouped_gemm_isa(moe_config)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self._pad_moe_intermediate(layer)
        super().process_weights_after_loading(layer)

    def _pad_moe_intermediate(self, layer: torch.nn.Module) -> None:
        """Zero-pad the per-partition MoE intermediate dim of both weights and
        the expert bias, see `_padded_intermediate_size`."""
        intermediate_size = self.moe_config.intermediate_size_per_partition
        padded_size = _padded_intermediate_size(self.moe_config)
        if padded_size == intermediate_size:
            return

        num_experts, _, hidden_size = layer.w13_weight.shape

        new_w13 = layer.w13_weight.new_zeros(num_experts, 2 * padded_size, hidden_size)
        new_w13[:, :intermediate_size] = layer.w13_weight[:, :intermediate_size]
        new_w13[:, padded_size : padded_size + intermediate_size] = layer.w13_weight[
            :, intermediate_size:
        ]
        replace_parameter(layer, "w13_weight", new_w13)

        new_w2 = layer.w2_weight.new_zeros(num_experts, hidden_size, padded_size)
        new_w2[:, :, :intermediate_size] = layer.w2_weight
        replace_parameter(layer, "w2_weight", new_w2)

        if hasattr(layer, "w13_bias"):
            new_bias = layer.w13_bias.new_zeros(num_experts, 2 * padded_size)
            new_bias[:, :intermediate_size] = layer.w13_bias[:, :intermediate_size]
            new_bias[:, padded_size : padded_size + intermediate_size] = layer.w13_bias[
                :, intermediate_size:
            ]
            # Assign through .data rather than replacing the Parameter: the
            # quant config is built before this runs and holds a reference to
            # this very object, which is what feeds self.w1_bias in apply().
            layer.w13_bias.data = new_bias


class ArmCPUUnquantizedExperts(CPUGroupedGemmExperts):
    """Arm NEON grouped-gemm unquantized MoE experts."""

    isa = "neon"

    @staticmethod
    def _supports_current_device() -> bool:
        return (
            current_platform.is_cpu()
            and current_platform.get_cpu_architecture() == CpuArchEnum.ARM
            and hasattr(torch.ops._C, "prepack_moe_weight")
        )

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        supported, reason = mk.FusedMoEExperts.is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )
        if not supported:
            return supported, reason
        return _neon_grouped_gemm_support(moe_config)


class TorchCPUUnquantizedExperts(CPUUnquantizedExperts):
    """Per-expert torch/oneDNN loop, the fallback for non-x86 CPUs."""

    @staticmethod
    def _supports_current_device() -> bool:
        return (
            current_platform.is_cpu()
            and current_platform.get_cpu_architecture() != CpuArchEnum.X86
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)

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
        self.layer_id = id(layer)

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        # grouped topk + fused topk bias parameters
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        topk_weights, topk_ids = self._select_experts(
            hidden_states,
            router_logits,
            num_expert_group,
            topk_group,
            e_score_correction_bias,
            routed_scaling_factor,
        )

        if apply_router_weight_on_input:
            assert topk_ids.size(1) == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            hidden_states.mul_(topk_weights.to(hidden_states.dtype))

        output = torch.empty_like(hidden_states)
        torch.ops.vllm.cpu_fused_moe_torch(
            self.layer_id,
            output,
            hidden_states,
            topk_weights,
            topk_ids,
            activation.value,
            global_num_experts,
            apply_router_weight_on_input,
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
    skip_weighted: bool = False,
) -> None:
    act = MoEActivation.from_str(activation)
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
        gate_up = _CPU_MOE_ACT_FN[act](gate_up)
        expert_out = layer.down_linear[i](gate_up)  # type: ignore
        outputs.append(expert_out)
        start_idx = end_idx

    outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
    new_x = torch.empty_like(outs)

    new_x[idxs] = outs
    if skip_weighted:
        final_out = new_x
    else:
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


# ===========================================================================
# FP8 W8A16 MoE
# ===========================================================================


def prepare_fp8_moe_layer_for_cpu(
    w13: torch.Tensor,
    w2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """VNNI-prepack FP8 MoE weights for CPU kernel."""
    packed_w13 = torch.ops._C.convert_weight_packed(w13)
    packed_w2 = torch.ops._C.convert_weight_packed(w2)
    return packed_w13, packed_w2


class CPUExpertsFp8(mk.FusedMoEExpertsMonolithic):
    """CPU FP8 W8A16 block-quantized monolithic MoE experts."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(
            moe_config,
            quant_config,
        )

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_cpu()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kFp8Static128BlockSym, kFp8Dynamic128Sym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return routing_method in [
            RoutingMethodType.Default,
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
        ]

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype | None,
        routing_method: RoutingMethodType,
    ) -> bool:
        return True

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        # grouped topk + fused topk bias parameters
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            use_grouped_topk=num_expert_group is not None,
            top_k=self.moe_config.experts_per_token,
            renormalize=self.moe_config.routing_method
            in (
                RoutingMethodType.Renormalize,
                RoutingMethodType.RenormalizeNaive,
            ),
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            scoring_func="softmax",
            routed_scaling_factor=(
                routed_scaling_factor if routed_scaling_factor is not None else 1.0
            ),
            e_score_correction_bias=e_score_correction_bias,
        )

        block_shape = (
            list(self.quant_config.block_shape)
            if self.quant_config.block_shape
            else (
                [self.quant_config._w1.shape.row, self.quant_config._w1.shape.col]
                if self.quant_config._w1.shape is not None
                else None
            )
        )

        return fused_experts_cpu(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            False,  # inplace
            CPUQuantMethod.FP8_W8A16,  # moe_comp_method
            self.w1_scale,  # w1_scale
            self.w2_scale,  # w2_scale
            None,  # w1_zero
            None,  # w2_zero
            block_shape,  # block_size
            None,  # w1_bias
            None,  # w2_bias
            None,  # alpha
            None,  # limit
            True,  # is_vnni
        )


# ===========================================================================
# MXFP4 W4A16 MoE
# ===========================================================================


def prepare_mxfp4_moe_layer_for_cpu(
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """VNNI-prepack MXFP4 MoE weights and repack scales for CPU AMX kernel."""
    packed_w13 = torch.ops._C.convert_weight_packed(w13)
    packed_w2 = torch.ops._C.convert_weight_packed(w2)
    packed_w13_scale = torch.ops._C.convert_scale_packed(w13_scale)
    packed_w2_scale = torch.ops._C.convert_scale_packed(w2_scale)
    return packed_w13, packed_w2, packed_w13_scale, packed_w2_scale


class CPUExpertsMxfp4(mk.FusedMoEExpertsMonolithic):
    """CPU MXFP4 W4A16 monolithic MoE experts."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(
            moe_config,
            quant_config,
        )

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_cpu()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (MoEActivation.SILU, MoEActivation.SWIGLUOAI)

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kMxfp4Static, None),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return routing_method in [
            RoutingMethodType.Default,
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
        ]

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype | None,
        routing_method: RoutingMethodType,
    ) -> bool:
        return True

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        # grouped topk + fused topk bias parameters
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            use_grouped_topk=num_expert_group is not None,
            top_k=self.moe_config.experts_per_token,
            renormalize=self.moe_config.routing_method
            in (
                RoutingMethodType.Renormalize,
                RoutingMethodType.RenormalizeNaive,
            ),
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            scoring_func="softmax",
            routed_scaling_factor=(
                routed_scaling_factor if routed_scaling_factor is not None else 1.0
            ),
            e_score_correction_bias=e_score_correction_bias,
        )

        # Get bias and swiglu params from quant config
        w1_bias = self.quant_config.w1_bias
        w2_bias = self.quant_config.w2_bias
        alpha = getattr(self.quant_config, "gemm1_alpha", None)
        limit = getattr(self.quant_config, "gemm1_clamp_limit", None)

        return fused_experts_cpu(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            False,  # inplace
            CPUQuantMethod.MXFP4,  # moe_comp_method
            self.w1_scale,  # w1_scale
            self.w2_scale,  # w2_scale
            None,  # w1_zero
            None,  # w2_zero
            None,  # block_size
            w1_bias,
            w2_bias,
            alpha,
            limit,
            True,  # is_vnni
        )


# ===========================================================================
# INT4 W4A16 MoE
# ===========================================================================


def prepare_int4_moe_layer_for_cpu(
    w13_packed: torch.Tensor,
    w2_packed: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    quant_algo: CPUQuantAlgo = CPUQuantAlgo.GPTQ,
    w13_zeros: torch.Tensor | None = None,
    w2_zeros: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Repack INT4 MoE weights via convert_weight_packed_scale_zp for CPU.

    Args:
        w13_packed: [E, K//8, 2*I] int32 (packed int4)
        w2_packed: [E, I//8, K] int32 (packed int4)
        w13_scale: [E, num_groups, 2*I] float16/bf16
        w2_scale: [E, num_groups, K] float16/bf16
        quant_algo: CPUQuantAlgo.GPTQ or CPUQuantAlgo.AWQ
        w13_zeros: optional [E, num_groups, N//8] int32 packed zeros.
                   If None, synthetic zeros are created for symmetric quant.
        w2_zeros: optional [E, num_groups, N//8] int32 packed zeros.
                  If None, synthetic zeros are created for symmetric quant.

    Returns:
        (blocked_w13, blocked_w2, blocked_s13, blocked_s2, blocked_z13, blocked_z2)
    """
    E = w13_packed.size(0)

    # No qzeros are available in compressed-tensors symmetric checkpoints.
    # The GPTQ unpack kernel (unpack_4bit_to_32bit_signed) adds +1 to stored zeros,
    # so we store 7 per nibble: 0x77777777 → +1 → 8.
    if w13_zeros is None:
        num_groups_w13 = w13_scale.size(1)
        N_w13 = w13_scale.size(2)  # 2*I
        _zp = 0x77777777
        w13_zeros = torch.full(
            (E, num_groups_w13, N_w13 // 8),
            _zp,
            dtype=torch.int32,
        )

    if w2_zeros is None:
        num_groups_w2 = w2_scale.size(1)
        N_w2 = w2_scale.size(2)  # K
        _zp = 0x77777777
        w2_zeros = torch.full(
            (E, num_groups_w2, N_w2 // 8),
            _zp,
            dtype=torch.int32,
        )

    blocked_w13, blocked_z13, blocked_s13 = convert_weight_packed_scale_zp(
        w13_packed, w13_zeros, w13_scale, quant_algo
    )
    blocked_w2, blocked_z2, blocked_s2 = convert_weight_packed_scale_zp(
        w2_packed, w2_zeros, w2_scale, quant_algo
    )
    return (blocked_w13, blocked_w2, blocked_s13, blocked_s2, blocked_z13, blocked_z2)


class CPUExpertsInt4(mk.FusedMoEExpertsMonolithic):
    """CPU INT4 W4A16 group-quantized monolithic MoE experts.

    Weights are int4 (packed), activations are bf16/fp16.
    Internally uses int8 compute via fused_experts_cpu with INT4_W4A8.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config,
            quant_config,
        )

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_cpu()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kInt4Static, None),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return routing_method in [
            RoutingMethodType.Default,
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
        ]

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype | None,
        routing_method: RoutingMethodType,
    ) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return False

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        # grouped topk + fused topk bias parameters
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        if apply_router_weight_on_input:
            raise NotImplementedError(
                "CPUExpertsInt4 (W4A16) does not support "
                "apply_router_weight_on_input=True. "
            )

        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            use_grouped_topk=num_expert_group is not None,
            top_k=self.moe_config.experts_per_token,
            renormalize=self.moe_config.routing_method
            in (
                RoutingMethodType.Renormalize,
                RoutingMethodType.RenormalizeNaive,
            ),
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            scoring_func="softmax",
            routed_scaling_factor=(
                routed_scaling_factor if routed_scaling_factor is not None else 1.0
            ),
            e_score_correction_bias=e_score_correction_bias,
        )

        return fused_experts_cpu(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            False,  # inplace
            CPUQuantMethod.INT4_W4A8,
            self.w1_scale,
            self.w2_scale,
            self.w1_zp,
            self.w2_zp,
            None,  # block_size
            None,  # w1_bias
            None,  # w2_bias
            None,  # alpha
            None,  # limit
            True,  # is_vnni
        )


# ===========================================================================
# INT8 W8A8 MoE
# ===========================================================================


def prepare_int8_moe_layer_for_cpu(
    w13: torch.Tensor,
    w2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepack INT8 MoE weights for the current CPU architecture."""
    # SMMLA packing for AArch64
    if current_platform.get_cpu_architecture() == CpuArchEnum.ARM:
        return (
            cpu_prepack_moe_weight_int8(w13, "neon"),
            cpu_prepack_moe_weight_int8(w2, "neon"),
        )
    # VNNI packing for x86
    packed_w13 = torch.ops._C.convert_weight_packed(w13)
    packed_w2 = torch.ops._C.convert_weight_packed(w2)
    return packed_w13, packed_w2


class CPUExpertsInt8(mk.FusedMoEExpertsMonolithic):
    """CPU INT8 W8A8 per-channel weight / dynamic per-token activation
    monolithic MoE experts."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(
            moe_config,
            quant_config,
        )

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return (
            current_platform.is_cpu()
            and current_platform.get_cpu_architecture() == CpuArchEnum.X86
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kInt8StaticChannelSym, kInt8DynamicTokenSym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return routing_method in [
            RoutingMethodType.Default,
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
        ]

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype | None,
        routing_method: RoutingMethodType,
    ) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """VNNI-prepack INT8 MoE weights for CPU kernel."""

        w13, w2 = prepare_int8_moe_layer_for_cpu(layer.w13_weight, layer.w2_weight)
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        # grouped topk + fused topk bias parameters
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            use_grouped_topk=num_expert_group is not None,
            top_k=self.moe_config.experts_per_token,
            renormalize=self.moe_config.routing_method
            in (
                RoutingMethodType.Renormalize,
                RoutingMethodType.RenormalizeNaive,
            ),
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            scoring_func="softmax",
            routed_scaling_factor=(
                routed_scaling_factor if routed_scaling_factor is not None else 1.0
            ),
            e_score_correction_bias=e_score_correction_bias,
        )

        return fused_experts_cpu(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            False,  # inplace
            CPUQuantMethod.INT8_W8A8,
            self.w1_scale,
            self.w2_scale,
            None,  # w1_zero
            None,  # w2_zero
            None,  # block_size (per-channel, no block)
            None,  # w1_bias
            None,  # w2_bias
            None,  # alpha
            None,  # limit
            True,  # is_vnni
        )


class ArmCPUExpertsInt8(mk.FusedMoEExpertsMonolithic):
    """Arm INT8 MoE with per-token activation and channelwise weight quantization."""

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        supported, reason = mk.FusedMoEExperts.is_supported_config(
            cls,
            moe_config,
            weight_key,
            activation_key,
            activation_format,
        )
        if not supported:
            return supported, reason
        if moe_config.in_dtype not in (
            torch.float32,
            torch.float16,
            torch.bfloat16,
        ):
            return False, "kernel requires float32, float16, or bfloat16 activations"
        if moe_config.hidden_dim % 32 != 0:
            return False, "kernel requires hidden dim divisible by 32"
        if moe_config.intermediate_size_per_partition % 32 != 0:
            return False, "kernel requires intermediate dim divisible by 32"
        return True, None

    @staticmethod
    def _supports_current_device() -> bool:
        return (
            current_platform.is_cpu()
            and current_platform.get_cpu_architecture() == CpuArchEnum.ARM
            and hasattr(torch.ops._C, "cpu_fused_moe_int8")
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (
            MoEActivation.SILU,
            MoEActivation.SWIGLUOAI,
            MoEActivation.GELU,
            MoEActivation.GELU_TANH,
        )

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return not moe_parallel_config.use_ep

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (
            kInt8StaticChannelSym,
            kInt8DynamicTokenSym,
        )

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return routing_method in [
            RoutingMethodType.Default,
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
        ]

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype | None,
        routing_method: RoutingMethodType,
    ) -> bool:
        return True

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w13, w2 = prepare_int8_moe_layer_for_cpu(layer.w13_weight, layer.w2_weight)
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            use_grouped_topk=num_expert_group is not None,
            top_k=self.moe_config.experts_per_token,
            renormalize=self.moe_config.routing_method
            in (
                RoutingMethodType.Renormalize,
                RoutingMethodType.RenormalizeNaive,
            ),
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            scoring_func="softmax",
            routed_scaling_factor=(
                routed_scaling_factor if routed_scaling_factor is not None else 1.0
            ),
            e_score_correction_bias=e_score_correction_bias,
        )

        if apply_router_weight_on_input:
            assert topk_ids.size(1) == 1
            hidden_states.mul_(topk_weights.to(hidden_states.dtype))

        assert self.w1_scale is not None
        assert self.w2_scale is not None
        return cpu_fused_moe_int8(
            hidden_states,
            w1,
            w2,
            self.w1_scale,
            self.w2_scale,
            self.w1_bias,
            self.w2_bias,
            topk_weights,
            topk_ids,
            activation.value,
            "neon",
            skip_weighted=apply_router_weight_on_input,
        )
