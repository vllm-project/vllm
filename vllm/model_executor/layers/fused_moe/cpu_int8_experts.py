# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU FusedMoE experts for W8A8 int8.

Fast path runs the whole FFN via torch.ops.zentorch.zentorch_fused_moe
(needs >=2 active experts); a per-expert zentorch_dynamic_qlinear loop
covers the <2-active-expert case (e.g. M==1, top_k==1 decode).
"""

from collections.abc import Callable

import torch
from torch.nn import functional as F

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.zentorch_utils import has_zentorch_op
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
)
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


def _silu_and_mul_native(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def _gelu_and_mul_native(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.gelu(x[..., :d]) * x[..., d:]


def _swigluoai_and_mul_native(
    x: torch.Tensor,
    *,
    alpha: float = 1.702,  # gpt-oss defaults; match SwigluOAIAndMul.
    limit: float = 7.0,
) -> torch.Tensor:
    # Interleaved SwiGLU-OAI (gate=x[..., 0::2], up=x[..., 1::2]) matching
    # ZenDNN's swiglu_oai_mul; process_weights_after_loading reorders w13 to
    # interleaved so the fused op and this fallback share the layout.
    gate = x[..., 0::2].clamp(max=limit)
    up = x[..., 1::2].clamp(min=-limit, max=limit)
    return (up + 1) * gate * torch.sigmoid(alpha * gate)


# Native (Python) implementations used by the per-expert fallback loop.
_CPU_INT8_MOE_ACT_FN: dict[MoEActivation, Callable[..., torch.Tensor]] = {
    MoEActivation.SILU: _silu_and_mul_native,
    MoEActivation.GELU: _gelu_and_mul_native,
    MoEActivation.SWIGLUOAI: _swigluoai_and_mul_native,
}


class CPUInt8Experts(mk.FusedMoEExpertsModular):
    """CPU FusedMoE experts for W8A8 int8 (channel-weight, per-token-act, sym).

    Runs the [Permute -> GEMM1 -> Activation -> GEMM2 -> Unpermute -> Reduce]
    sequence directly inside ``apply``. Reports ``TopKWeightAndReduceNoOP`` so
    the prepare/finalize layer skips its own reduce.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        assert self.w1_scale is not None and self.w2_scale is not None, (
            "CPUInt8Experts requires per-channel weight scales on the layer."
        )

        # Cast scales/biases to bf16 to match the bf16 activations both
        # zentorch_fused_moe and the per-expert fallback.
        E = self.w1_scale.shape[0]
        self._w13_scale_bf16 = (
            self.w1_scale.detach().to(torch.bfloat16).reshape(E, -1).contiguous()
        )
        self._w2_scale_bf16 = (
            self.w2_scale.detach().to(torch.bfloat16).reshape(E, -1).contiguous()
        )
        self._w13_bias = (
            None
            if self.w1_bias is None
            else self.w1_bias.detach().to(torch.bfloat16).contiguous()
        )
        self._w2_bias = (
            None
            if self.w2_bias is None
            else self.w2_bias.detach().to(torch.bfloat16).contiguous()
        )

        logger.info_once(
            "[zen_cpu] CPUInt8Experts prepared weights "
            "(E=%d, 2I=%d, K=%d, has_w13_bias=%s has_w2_bias=%s)",
            E,
            self._w13_scale_bf16.shape[1],
            self._w2_scale_bf16.shape[1],
            self._w13_bias is not None,
            self._w2_bias is not None,
        )

    # ------------------------------------------------------------------ #
    # Required FusedMoEExperts metadata.
    # ------------------------------------------------------------------ #

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @property
    def expects_unquantized_inputs(self) -> bool:
        # zentorch_dynamic_qlinear quantizes activations itself.
        return True

    @staticmethod
    def _supports_current_device() -> bool:
        return has_zentorch_op(["zentorch_fused_moe", "zentorch_dynamic_qlinear"])

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (
            weight_key == kInt8StaticChannelSym
            and activation_key == kInt8DynamicTokenSym
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in _CPU_INT8_MOE_ACT_FN

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return True

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # Framework uses these only for activation-chunking heuristics; we
        # allocate our own scratch in apply().
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace13 = (M, topk, max(activation_out_dim, K))
        workspace2 = (M, topk, max(N, K))
        output = (M, K)
        return (workspace13, workspace2, output)

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        torch.ops.vllm.cpu_int8_moe_torch(
            output,
            hidden_states,
            w1,
            w2,
            self._w13_scale_bf16,
            self._w2_scale_bf16,
            self._w13_bias,
            self._w2_bias,
            topk_weights,
            topk_ids,
            activation.value,
            global_num_experts if global_num_experts > 0 else -1,
            expert_map,
            apply_router_weight_on_input,
        )


# Custom op wrapping the full int8 MoE so torch.compile/Dynamo treats it as a
# single opaque call (same trick as cpu_fused_moe.py).


def cpu_int8_moe_torch(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    apply_router_weight_on_input: bool,
) -> None:
    act_enum = MoEActivation.from_str(activation)

    E, _, K = w1.shape
    M, top_k = topk_ids.shape
    if global_num_experts <= 0:
        global_num_experts = E

    # Map global -> local expert ids; -1 marks remote-rank tokens.
    local_topk_ids_raw = (
        expert_map[topk_ids.to(torch.long)] if expert_map is not None else topk_ids
    )

    # zentorch_fused_moe needs >=2 active *local* experts. <2 happens for
    # M==1/top_k==1 decode, single-expert top_k==1 prefill, or single-expert
    # EP ranks. Count uniques after remap; -1 (remote) tokens don't count.
    flat_local = local_topk_ids_raw.reshape(-1)
    valid_local = flat_local[flat_local >= 0]
    num_active_local = (
        int(valid_local.unique().numel()) if valid_local.numel() > 0 else 0
    )
    if num_active_local >= 2:
        # The C++ op asserts topk_id values in [0, E). Mask -1 sentinels to
        # expert 0 and zero out their router weights so they contribute zero
        # to the reduce post-op.
        if expert_map is not None and (local_topk_ids_raw < 0).any():
            invalid = local_topk_ids_raw < 0
            local_topk_ids_fast = local_topk_ids_raw.masked_fill(invalid, 0)
            topk_weights_fast = topk_weights.masked_fill(invalid, 0.0)
        else:
            local_topk_ids_fast = local_topk_ids_raw
            topk_weights_fast = topk_weights

        output.zero_()
        input_for_op = hidden_states
        if apply_router_weight_on_input:
            if top_k != 1:
                raise NotImplementedError(
                    "CPUInt8Experts: apply_router_weight_on_input=True is "
                    f"only supported for top_k=1 (got top_k={top_k}); for "
                    "top_k>1 pass apply_router_weight_on_input=False so the "
                    "router weight is applied during the reduce."
                )
            # topk_weights_fast has shape (M, 1) when top_k==1; broadcasts
            # to (M, K) against hidden_states.
            input_for_op = hidden_states.mul(topk_weights_fast.to(hidden_states.dtype))

        logger.info_once(
            "[zen_cpu] CPUInt8Experts dispatching to "
            "torch.ops.zentorch.zentorch_fused_moe "
            "(M=%d, top_k=%d, act=%s, apply_router_weight_on_input=%s)",
            M,
            top_k,
            act_enum.value,
            apply_router_weight_on_input,
        )
        torch.ops.zentorch.zentorch_fused_moe(
            output,
            input_for_op,
            w1,
            w2,
            w13_bias,
            w2_bias,
            topk_weights_fast.to(torch.float32).contiguous(),
            local_topk_ids_fast.to(torch.int32).contiguous(),
            apply_router_weight_on_input,  # skip_weighted
            act_enum.value,
            w13_scale,
            w2_scale,
        )
        return

    # Fallback: per-expert zentorch_dynamic_qlinear loop. Used whenever
    # the active-local-expert count is < 2 (which zentorch_fused_moe
    # rejects); the loop handles arbitrary (M, top_k) shapes correctly.
    logger.info_once(
        "[zen_cpu] CPUInt8Experts dispatching to per-expert "
        "torch.ops.zentorch.zentorch_dynamic_qlinear loop "
        "(M=%d, top_k=%d, act=%s, apply_router_weight_on_input=%s, "
        "num_active_local=%d) [fallback: fewer than 2 active local experts]",
        M,
        top_k,
        act_enum.value,
        apply_router_weight_on_input,
        num_active_local,
    )
    act_fn = _CPU_INT8_MOE_ACT_FN[act_enum]
    local_topk_ids = local_topk_ids_raw

    if num_active_local == 0:
        output.zero_()
        return

    input_for_loop = hidden_states
    if apply_router_weight_on_input:
        input_for_loop = hidden_states.mul(topk_weights.to(hidden_states.dtype))

    # Stable sort by local expert id; remote tokens (-1) sort to the front
    # so the loop can skip them with a single cursor offset.
    flat_ids = local_topk_ids.reshape(-1)
    sort_idx = torch.argsort(flat_ids.to(torch.int64), stable=True)
    sorted_local_ids = flat_ids[sort_idx]
    token_src = sort_idx // top_k
    sorted_tokens = input_for_loop.index_select(0, token_src)

    per_expert_counts = torch.bincount(sorted_local_ids.clamp(min=0), minlength=E)
    num_remote = (sorted_local_ids == -1).sum().item()

    sorted_out = sorted_tokens.new_zeros(sorted_tokens.size(0), K)
    counts_cpu = per_expert_counts.cpu().tolist()
    cursor = int(num_remote)
    for e in range(E):
        n_e = int(counts_cpu[e])
        if n_e == 0:
            continue
        tokens_e = sorted_tokens[cursor : cursor + n_e]
        bias13_e = None if w13_bias is None else w13_bias[e]
        gate_up = torch.ops.zentorch.zentorch_dynamic_qlinear(
            tokens_e, w1[e], w13_scale[e], bias13_e
        )
        act_out = act_fn(gate_up)
        bias2_e = None if w2_bias is None else w2_bias[e]
        sorted_out[cursor : cursor + n_e] = torch.ops.zentorch.zentorch_dynamic_qlinear(
            act_out, w2[e], w2_scale[e], bias2_e
        )
        cursor += n_e

    unsorted_out = torch.empty_like(sorted_out)
    unsorted_out.index_copy_(0, sort_idx, sorted_out)
    per_topk = unsorted_out.view(M, top_k, K)
    if not apply_router_weight_on_input:
        per_topk = per_topk * topk_weights.view(M, top_k, 1).to(per_topk.dtype)
    torch.sum(per_topk, dim=1, out=output)


direct_register_custom_op(
    op_name="cpu_int8_moe_torch",
    op_func=cpu_int8_moe_torch,
    mutates_args=["output"],
)
