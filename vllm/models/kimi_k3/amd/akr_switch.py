# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-kernel sub-switches of the kda / moe / gemm / norm routes (kdamoe area).

Every wired kernel is on only when its route is on (`akr_config.on(route)`: config
`disabled_routes` and `VLLM_KIMI_AKR`) and its own env `VLLM_KIMI_AKR_<NAME>` is not
0/false/off (default on), except the sub-switches in `DEFAULT_OFF`, which need `=1`.
"""

import os

from vllm.models.kimi_k3.amd import akr_config

# name -> route
SWITCHES: dict[str, str] = {
    # attention/kda_common.kda_decode_fb_gate_spec (f_b + spec decode)
    "KDA_DECODE": "kda",
    "ROUTER": "moe",  # moe/latent_moe_common.router_sigmoid_bias_topk (+ aux sort)
    # moe/latent_moe_common.mxfp4_moe_experts (own routed experts chain)
    "EXPERTS": "moe",
    "DOWN_QUANT": "gemm",  # gemm/bf16_common.bf16_latent_down_mxfp4_quant
    "LATENT_AR": "moe",  # moe/latent_moe_common.allreduce_latent_rmsnorm_up_proj
    "SLOT_FOLD": "moe",  # 16-slot sum folded into the latent AR (M <= 64)
    "ATTNRES_AR": "norm",  # norm/attnres_rmsnorm_common (allreduce_attnres_rmsnorm)
    # shared experts on the aux stream inside the akr MoE chain (else inline)
    "SHARED_AUX": "overlap",
    # c4 specialized K1 (attention/kda_common.kda_mono_decode), default off
    "K1": "kda",
    # c4 specialized K2 (moe/latent_moe_common.layer_tail_launch) for the KDA / ops
    # layers, default off
    "K2": "moe",
    # gemm/bf16_skinny_* (TODO: no final version on the branch yet)
    "SKINNY_GEMM": "gemm",
}
DEFAULT_OFF = frozenset({"K1", "K2", "SKINNY_GEMM"})
_OFF = ("0", "false", "off", "no", "none")


def env_value(name: str) -> str:
    return os.environ.get(f"VLLM_KIMI_AKR_{name}", "").strip().lower()


def on(name: str) -> bool:
    """Whether sub-switch `name` (key of SWITCHES) is on in this process.

    Batch-invariant runs keep the stock kernels (the fused kernels pick their splits and
    orders from the row count)."""
    from vllm import envs

    if envs.VLLM_BATCH_INVARIANT:
        return False
    v = env_value(name)
    if name in DEFAULT_OFF:
        if v not in ("1", "true", "on", "yes"):
            return False
    elif v in _OFF:
        return False
    return akr_config.on(SWITCHES[name])


def shared_mode() -> str:
    """Where the shared experts start inside the akr MoE chain.

    `experts` (default): on the aux stream right after the down-projection kernel,
    overlapping the long routed-experts kernel. `start`: aux stream from the layer's
    first kernel, overlapping the router and the down projection (forces the down
    projection's XCC-independent slab exchange). `inline`: no aux stream."""
    if not on("SHARED_AUX"):
        return "inline"
    v = os.environ.get("VLLM_KIMI_AKR_SHARED_AT", "experts").strip().lower()
    return v if v in ("experts", "start") else "experts"
