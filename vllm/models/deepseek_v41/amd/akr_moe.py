# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 MoE block on amd_kernels: routes `router`, `draft_router` and
`shared_expert`.

router / draft_router: gate GEMM, sqrt-softplus, correction bias, top-k,
    renormalize and routed scaling in one entry, for the target layers (384
    experts, top 6) and the DSpark draft layers (128 experts, top 3).
shared_expert: the shared expert's gate_up and clamped SiLU-mul as MXFP8
    for the `mhc_ffn` epilogue (shared down + routed add + FFN post-mix);
    `SHARED_PLAN` picks per static row count which gate_up runs. Standalone
    the stock chain on the FFN seam's MXFP8 copy wins at every row count, so
    the route declines there. The stock linears' preshuffled weights are read
    as they are; only their scales get a copy (about 0.5 MB per layer).

While any of these routes is on, a MegaMoE layer runs its shared expert after
the routed experts on the same stream: beside MegaMoE's persistent,
spin-waiting stage-1 kernel the overlapped gate_up GEMM stalled both for
1.3-13.7 ms per occurrence on DEP8 decode.
"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.models.deepseek_v41.amd import akr_config, akr_hooks, akr_kernels

logger = init_logger(__name__)


# (largest row count, gate_up plan) for the `mhc_ffn` epilogue (up to its 192
# rows), from a graph-replay benchmark of the shared-expert chain on MI355X
# against dev1133 stock (gate_up on the seam's MXFP8 copy, silu_and_mul_mxfp8,
# down; 12.9-25.6 us at 16-192 rows): "swiglu" 13.0 us at 16 rows, "stock"
# 15.8-26.0 us at 32-192 rows ("swiglu" 16.9-27.0, "linear" 18.3-27.1).
# "swiglu": amd_kernels gate_up with the clamped SiLU-mul + MXFP8 epilogue
# "linear": amd_kernels gate_up, then the amd_kernels clamped SiLU-mul + MXFP8
# "stock":  the stock gate_up, then the amd_kernels clamped SiLU-mul + MXFP8
# The stock down projection reads the MXFP8 activation in every plan.
SHARED_PLAN: tuple[tuple[int, str], ...] = (
    (16, "swiglu"),
    (192, "stock"),
)
# the gate_up epilogue reads MXFP8 input codes only up to this row count
SWIGLU_CODES_MAX_ROWS = 16


def shared_plan(m: int) -> str | None:
    for bound, plan in SHARED_PLAN:
        if m <= bound:
            return plan
    return None


def shared_serves(m: int) -> bool:
    return akr_config.serves("shared_expert", m) and shared_plan(m) is not None


def shared_takes_codes(m: int) -> bool:
    """Whether the shared expert reads the FFN seam's MXFP8 codes at `m` rows."""
    if not shared_serves(m):
        return False
    plan = shared_plan(m)
    return plan == "stock" or (plan == "swiglu" and m <= SWIGLU_CODES_MAX_ROWS)


class Router:
    """`DeepseekV4MoE.akr`: the router route of one MoE layer (`gate` None when
    the layer's router is not eligible: only the serial shared expert)."""

    def __init__(self, route: str, gate: torch.Tensor | None, moe) -> None:
        self.route_name = route
        self.gate = gate
        if gate is not None:
            self.bias = moe.gate.e_score_correction_bias.data
            self.scale = float(moe.routed_scaling_factor)
            self.top_k = moe.n_activated_experts

    def route(self, moe, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        """fp32 top-k weights and int32 expert ids, or None for the stock router."""
        if (
            self.gate is None
            or x.dtype != torch.bfloat16
            or x.dim() != 2
            or x.stride(1) != 1
            or not akr_config.serves(self.route_name, x.shape[0])
        ):
            return None
        from vllm.model_executor.layers.fused_moe.router.base_router import (
            get_moe_padding_mask,
        )

        pad = get_moe_padding_mask(x.shape[0])
        return akr_kernels.router_topk(
            x,
            self.gate,
            self.bias,
            self.scale,
            self.top_k,
            None if pad is None else pad.contiguous(),
        )


class SharedExpert:
    """`DeepseekV4MLP.akr`: the shared-expert route of one MoE layer. The down
    projection stays the stock linear, on the MXFP8 activation."""

    def __init__(self, mlp, gate_up: akr_kernels.Mxfp8Weight) -> None:
        self.gate_up = gate_up
        self.limit = mlp.act_fn.swiglu_limit

    def activation(
        self, mlp, x: torch.Tensor, x_quant: QuantizedActivation | None
    ) -> QuantizedActivation | None:
        """gate_up and the clamped SiLU-mul as MXFP8, or None for the stock
        chain."""
        m = x.shape[0]
        if x.dtype != torch.bfloat16 or x.dim() != 2 or not x.is_contiguous():
            return None
        plan = shared_plan(m) if akr_config.serves("shared_expert", m) else None
        if plan is None:
            return None
        out = None
        if plan == "swiglu":
            codes_in = x_quant is not None and m <= SWIGLU_CODES_MAX_ROWS
            out = akr_kernels.mxfp8_gate_up_swiglu(
                x_quant if codes_in else x, self.gate_up
            )
        if out is None:
            gate_up, _ = mlp.gate_up_proj(x if x_quant is None else x_quant)
            out = akr_kernels.clamp_silu_mul_mxfp8(gate_up, self.limit)
        codes, scale = out
        return akr_config.quant_activation(codes, scale, x.dtype)

    def forward(
        self, mlp, x: torch.Tensor, x_quant: QuantizedActivation | None
    ) -> torch.Tensor | None:
        """None: standalone, the stock chain on the seam's MXFP8 copy wins or
        ties (within 0.2 us) at every row count from 16 to 768 on MI355X
        (dev1133), by up to 3.7 us."""
        return None

    def chain(
        self, mlp, x: torch.Tensor, x_quant: QuantizedActivation | None
    ) -> torch.Tensor | None:
        """`activation` and the stock down projection (the unfused form of the
        `mhc_ffn` epilogue's input; tests and warm-up)."""
        act = self.activation(mlp, x, x_quant)
        if act is None:
            return None
        out, _ = mlp.down_proj(act)
        return out

    def warmup(self, mlp) -> None:
        dev = self.gate_up.weight.device
        hidden = self.gate_up.k
        akr_kernels.warmup_clamp_silu_mul(self.gate_up.n // 2, dev)
        for bound, _ in SHARED_PLAN:
            x = torch.zeros(bound, hidden, dtype=torch.bfloat16, device=dev)
            self.chain(mlp, x, None)
            if bound <= SWIGLU_CODES_MAX_ROWS:
                codes = torch.zeros(
                    bound, hidden, dtype=torch.float8_e4m3fn, device=dev
                )
                scale = torch.full(
                    (bound, hidden // 32), 127, dtype=torch.uint8, device=dev
                )
                self.chain(mlp, x, akr_config.quant_activation(codes, scale, x.dtype))


def wants_x_quant(moe, m: int) -> bool:
    """Whether the FFN seam should also emit MXFP8 codes for `moe` at `m` rows:
    the fused EP8 experts read them, and so does the shared expert where its
    gate_up runs on codes (the route's plan, or the stock linear where the
    route does not serve)."""
    fused = getattr(getattr(moe, "experts", None), "akr", None)
    if fused is not None and fused.serves(m):
        return True
    shared = getattr(moe, "shared_experts", None)
    if shared is None:
        return False
    if getattr(shared, "akr", None) is not None and shared_serves(m):
        return shared_takes_codes(m)
    return _stock_takes_codes(shared)


def _stock_takes_codes(mlp) -> bool:
    """Whether the stock shared expert's gate_up reads an attached MXFP8 copy."""
    takes = getattr(mlp, "_akr_gate_up_codes", None)
    if takes is None:
        takes = mlp._gateup_scale is None and akr_config.takes_mxfp8_codes(
            mlp.gate_up_proj
        )
        mlp._akr_gate_up_codes = takes
    return takes


def forward_parts(
    moe,
    x: torch.Tensor,
    input_ids: torch.Tensor | None,
    x_quant: QuantizedActivation | None,
) -> tuple[torch.Tensor, QuantizedActivation] | None:
    """The routed output and the shared expert's MXFP8 activation, for a caller
    that runs the shared down projection in its own epilogue. None, having done
    nothing, where the shared-expert route does not serve."""
    shared = moe.shared_experts
    if not moe.use_mega_moe or shared is None or shared.akr is None:
        return None
    act = shared.akr.activation(shared, x, x_quant)
    if act is None:
        return None
    return moe._forward_mega_moe_routed(x, input_ids, x_quant), act


def _router_weight(moe) -> torch.Tensor | None:
    gate = moe.gate
    bias = gate.e_score_correction_bias
    if (
        gate.weight.dtype != torch.bfloat16
        or bias is None
        or bias.dtype != torch.float32
        or gate.tid2eid is not None
        or moe.scoring_func != "sqrtsoftplus"
        or not moe.renormalize
        # vision tokens pick experts with bias_vl, which the kernel lacks
        or (gate.bias_vl is not None and not _language_model_only())
    ):
        return None
    return akr_kernels.router_weight(gate.weight.data, moe.n_activated_experts)


def _language_model_only() -> bool:
    from vllm.config import get_current_vllm_config_or_none

    cfg = get_current_vllm_config_or_none()
    mm = None if cfg is None else cfg.model_config.multimodal_config
    return bool(mm is not None and mm.language_model_only)


def _shared_gate_up(mlp) -> akr_kernels.Mxfp8Weight | None:
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp

    # the entries clamp SwiGLU at 10; the stock down reads MXFP8 codes as is
    if (
        not isinstance(mlp.act_fn, SiluAndMulWithClamp)
        or mlp.act_fn.swiglu_limit != 10.0
        or not akr_config.takes_mxfp8_codes(mlp.down_proj)
        or not mlp.gate_up_proj.weight.is_cuda
    ):
        return None
    gate_up = akr_kernels.mxfp8_weight(mlp.gate_up_proj)
    if gate_up is None or not akr_kernels.shared_expert_ok(gate_up, gate_up.n // 2):
        return None
    return gate_up


def post_load(model) -> None:
    from vllm.models.deepseek_v4.amd.model import DeepseekV4MLP, DeepseekV4MoE

    draft = akr_hooks.is_draft(model)
    route = "draft_router" if draft else "router"
    routers = shared = 0
    first_router = first_shared = None
    for mod in model.modules():
        if isinstance(mod, DeepseekV4MoE) and mod.use_mega_moe:
            gate = None
            if akr_config.on(route):
                gate = _router_weight(mod)
            mod.akr = Router(route, gate, mod)
            if gate is not None:
                routers += 1
                first_router = first_router or mod
        elif isinstance(mod, DeepseekV4MLP) and akr_config.on("shared_expert"):
            gate_up = _shared_gate_up(mod)
            if gate_up is not None:
                mod.akr = SharedExpert(mod, gate_up)
                shared += 1
                first_shared = first_shared or mod
    if first_router is not None:
        akr_kernels.router_warmup(first_router.akr.gate, first_router.akr.top_k)
    if first_shared is not None:
        first_shared.akr.warmup(first_shared)
    logger.info(
        "amd_kernels MoE block (%s): %s on %d layers, shared expert on %d",
        "draft" if draft else "target",
        route,
        routers,
        shared,
    )
    if not draft and akr_config.on("router") and routers == 0:
        akr_config.fail("router", "no sqrt-softplus router the entry covers")
    if not draft and akr_config.on("shared_expert") and shared == 0:
        akr_config.fail("shared_expert", "no FP8 clamped-SwiGLU shared expert")
