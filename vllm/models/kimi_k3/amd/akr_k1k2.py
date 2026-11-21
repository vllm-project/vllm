# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The c4 latency-specialized decode path (specialized_decode,
`decode_backend=latency_specialized`) on the amd_kernels K1 / K2 ports, behind
sub-switches that default OFF until A/B'd:

  VLLM_KIMI_AKR_K1=1  attention/kda_common.kda_mono_decode replaces the stock
                      `kda_pre` launch of a KDA layer (WholeLayer.launch): 45.4 ->
                      30.5 us at 12 rows (1.49x).
  VLLM_KIMI_AKR_K2=1  moe/latent_moe_common.layer_tail_launch replaces the stock
                      `k2_launch` of the layers whose K2 does not run an MLA/DSA M1b
                      (KDA layers, MLA-ops layers): 116.9 -> 108.5 us at 12 rows
                      (1.08x). K2 layers with `m1b` / `dsa` keep the stock K2.

A workspace / runner or a launch the entry's `*_unsupported_reason` rejects keeps the
stock launch (warned once).

Both need the route (`kda` for K1, `moe` for K2) on and
`decode_backend=latency_specialized`. The stock mailboxes (MonoRunner scratch / peers /
epoch) are NOT shared: K1 has its own workspace (`kda_mono_workspace`, single GPU), K2
its own runner (`layer_tail_runner`, one more PeerBuffer, created collectively inside
`MonoRunner.__init__`); both are bumped by `MonoRunner.step_begin`. The engine-fault /
alarm plumbing of the stock runner does not see their fail-stop words (K1
`ws.failed()`, K2 `health`): a stuck wait gives up after 10 s and voids that step, like
the stock kernels, but is not reported through the runner's fault hooks.
"""

import dataclasses
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.models.kimi_k3.amd import akr_config, akr_switch

logger = init_logger(__name__)

_k1_ws: dict[int, Any] = {}


def k1_on() -> bool:
    return akr_switch.on("K1")


def k2_on() -> bool:
    return akr_switch.on("K2")


def _k1_lib():
    from amd_kernels.attention import kda_common

    return kda_common


def _k2_lib():
    from amd_kernels.moe import latent_moe_common

    return latent_moe_common


def _dims() -> dict[str, int]:
    """The shapes the stock specialized decode kernels are built for."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import kda
    from vllm.models.kimi_k3.amd.specialized_decode.mono.stages import moe

    return dict(
        hidden=kda.HIDDEN,
        heads=kda.NH,
        head_dim=kda.HD,
        o_proj_k=kda.PROJ,
        experts=moe.E,
    )


def k1_workspace(device: torch.device):
    """The K1 workspace of `device`, or None (warned once) when the entry has none."""
    if device.index in _k1_ws:
        return _k1_ws[device.index]
    lib, d = _k1_lib(), _dims()
    kw = dict(hidden=d["hidden"], heads=d["heads"], head_dim=d["head_dim"])
    ws = None
    if not akr_config.unsupported_reason(
        "kda", lib.kda_mono_workspace_unsupported_reason(device, **kw)
    ):
        ws = lib.kda_mono_workspace(device, **kw)
    _k1_ws[device.index] = ws
    return ws


def step_begin(runner) -> None:
    """Once a forward before its first mono launch (MonoRunner.step_begin): bump the
    epochs of the amd_kernels K1 workspace / K2 runner."""
    if getattr(runner, "akr_k1", None) is not None:
        runner.akr_k1.begin_step()
    if getattr(runner, "akr_k2", None) is not None:
        runner.akr_k2.step_begin()


def make_runner_state(runner, device: torch.device, group, rank: int) -> None:
    """At the end of MonoRunner.__init__ (collective)."""
    runner.akr_k1 = k1_workspace(device) if k1_on() else None
    runner.akr_k2 = None
    if k2_on():
        lib, d = _k2_lib(), _dims()
        kw = dict(hidden=d["hidden"], o_proj_k=d["o_proj_k"], experts=d["experts"])
        if not akr_config.unsupported_reason(
            "moe", lib.layer_tail_runner_unsupported_reason(device, group, rank, **kw)
        ):
            runner.akr_k2 = lib.layer_tail_runner(device, group, rank, **kw)
    if runner.akr_k1 is not None or runner.akr_k2 is not None:
        logger.info(
            "amd_kernels specialized decode: K1 %s, K2 %s",
            "on" if runner.akr_k1 is not None else "off",
            "on" if runner.akr_k2 is not None else "off",
        )


def k1_launch(
    whole, step, prefix_sum, block_residual, prefix_delta, attn, m, conv_state, rstate
):
    """K1 of a KDA layer (`WholeLayer.launch`): returns `core` [S, 1536] bf16 or None
    (stock)."""
    layer, runner = whole.layer, whole.runner
    ws = getattr(runner, "akr_k1", None)
    if ws is None:
        return None
    st_idx = m.spec_state_indices_tensor
    lib = _k1_lib()
    args = (
        ws,
        layer.layer_idx,
        prefix_sum,
        prefix_delta,
        block_residual,
        layer.self_attention_res_norm.weight,
        layer.self_attention_res_proj.weight.view(-1),
        layer.input_layernorm.weight,
        attn.in_proj_qkvgfab.weight,
        attn.f_b_proj.weight,
        attn.conv1d.weight.view(attn.conv1d.weight.size(0), -1),
        conv_state,
        attn.A_log,
        attn.dt_bias,
        attn.o_norm.weight,
        rstate,
        st_idx,
        m.num_accepted_tokens,
    )
    kw = dict(
        qlen=st_idx.size(-1),
        nblocks=layer.prev_valid_blocks,
        write_idx=layer.block_write_idx if layer.is_block_write_layer else -1,
        eps=layer.self_attention_res_norm.variance_epsilon,
        out_eps=layer.input_layernorm.variance_epsilon,
        onorm_eps=attn.o_norm.eps,
        lower_bound=attn.gate_lower_bound,
    )
    if akr_config.call_unsupported(
        "kda", lib.kda_mono_decode_unsupported_reason(*args, **kw)
    ):
        return None
    return lib.kda_mono_decode(*args, **kw)


def k2_launch(runner, key, **kw) -> bool:
    """K2 of a layer on the amd_kernels port: True when it ran (kw as the stock
    `k2_launch`, `out` written), False to let the stock K2 run (an MLA / DSA M1b in the
    launch, or a launch the entry does not serve)."""
    tail = getattr(runner, "akr_k2", None)
    if tail is None or key.m1b_qlen or key.m1c or key.dsa:
        return False
    lib = _k2_lib()
    build = {
        f.name: getattr(key, f.name) for f in dataclasses.fields(key) if f.name != "opt"
    }
    for drop in (
        "m1b_in",
        "m1c_in",
        "dsa_in",
        "scratch",
        "epoch",
        "health",
        "alarm",
        "peers",
        "rank",
    ):
        kw.pop(drop, None)
    if akr_config.call_unsupported(
        "moe", lib.layer_tail_launch_unsupported_reason(tail, build, **kw)
    ):
        return False
    lib.layer_tail_launch(tail, build, **kw)
    return True
