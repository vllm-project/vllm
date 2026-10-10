# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host side of the mono MoE launch (``layer.py``): the launcher cache, the
per-device workspace and ``mono_moe``, which takes the tensors vLLM's Kimi-K3
MoE holds and returns its routed and shared outputs.

``mono_moe`` replaces, for M <= ``M_MAX`` tokens,

    aiter.biased_grouped_topk(logits, bias, tw, ti, 1, 1, True, 1.0)
    aiter.fused_moe.fused_moe(x, w13, w2, tw, ti, Situv2, per_1x32, ...)
    KimiMLP(shared_x)                       (vLLM's shared expert, aux stream)

on AITER's a4w4 SiTUv2 path. Callers check ``supported`` first.
"""

import functools

import torch
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from vllm.models.kimi_k3.amd.mono.common.plan import (
    BM,
    ctrl_words,
    max_m_blocks,
    ws_layout,
)
from vllm.models.kimi_k3.amd.mono.layer import compile_mono_moe

# The shared expert's tiles cover one m-block, so the launch takes up to BM
# tokens; the control words and the workspace are sized for that.
M_MAX = BM
G1_BN = 128
# gemm1 switches to this tile width once routing yields many m-blocks.
G1_BN_WIDE = 256
# Wide gemm2 tiles amortise the per-tile wait and setup; 512 needs hidden % 512 == 0.
G2_BN = 512
# Shared expert: K splits per gate/up pair, down tile width.
SH_KS = 4
SH_DN_BN = 64


@functools.cache
def _launcher(ne, topk, hidden, inter, beta, linear_beta, sh, trace=False):
    """sh: (hidden, inter, beta, linear_beta) of the shared expert."""
    g2_bn = G2_BN if hidden % G2_BN == 0 else 256
    g1_wide = G1_BN_WIDE if (2 * inter) % G1_BN_WIDE == 0 else 0
    return compile_mono_moe(
        M_MAX=M_MAX, NE=ne, TOPK=topk, D_HIDDEN=hidden, D_INTER=inter,
        G1_BN=G1_BN, G2_BN=g2_bn, G1_BN_WIDE=g1_wide, situ_beta=beta,
        situ_linear_beta=linear_beta, SH_HIDDEN=sh[0], SH_INTER=sh[1],
        sh_beta=sh[2], sh_linear_beta=sh[3], SH_KS=SH_KS, SH_DN_BN=SH_DN_BN,
        TRACE=trace,
    )  # fmt: skip


class Workspace:
    """Per-device buffers sized for M_MAX tokens.

    The control words must start at zero; the kernel's last workgroup returns
    them to zero, so one zero fill at allocation covers every later launch.
    Sharing is safe while launches are ordered, as on one stream; launches on
    two streams at once need a workspace each.
    """

    def __init__(self, device, topk, inter, sh_inter):
        offs, total = ws_layout(M_MAX, topk, inter, sh_inter, SH_KS)
        self.buf = torch.zeros(total, dtype=torch.uint8, device=device)
        self.ctrl = self.buf[: ctrl_words(M_MAX, topk, sh_inter) * 4].view(torch.int32)
        assert offs["ctrl"] == 0


_WORKSPACES: dict[tuple, Workspace] = {}


def _workspace(device, topk, inter, sh_inter):
    key = (device, topk, inter, sh_inter)
    ws = _WORKSPACES.get(key)
    if ws is None:
        ws = _WORKSPACES[key] = Workspace(device, topk, inter, sh_inter)
    return ws


@functools.cache
def _num_cus(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def supported(
    x: torch.Tensor,
    ne: int,
    topk: int,
    inter: int,
    shared_x: torch.Tensor,
    shared_w_gu: torch.Tensor,
    shared_w_dn: torch.Tensor,
) -> bool:
    """Shapes the launch takes (gfx950, AITER's a4w4 SiTUv2 routed weights,
    bf16 shared expert)."""
    m = x.shape[0]
    sh_hidden, sh_inter = shared_w_dn.shape
    return (
        1 <= m <= M_MAX
        and ne % 64 == 0
        and topk <= 64
        and inter % 128 == 0
        and shared_x.dtype == shared_w_gu.dtype == shared_w_dn.dtype == torch.bfloat16
        and shared_x.is_contiguous()
        and shared_w_gu.is_contiguous()
        and shared_w_dn.is_contiguous()
        and tuple(shared_x.shape) == (m, sh_hidden)
        and tuple(shared_w_gu.shape) == (2 * sh_inter, sh_hidden)
        and sh_inter % 32 == 0
        and sh_hidden % (SH_KS * 4 * 32) == 0
        and sh_hidden % SH_DN_BN == 0
    )


def mono_moe(
    logits: torch.Tensor,
    bias: torch.Tensor,
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    shared_x: torch.Tensor,
    shared_w_gu: torch.Tensor,
    shared_w_dn: torch.Tensor,
    *,
    topk: int,
    situ_beta: float,
    situ_linear_beta: float,
    shared_beta: float,
    shared_linear_beta: float | None,
    out: torch.Tensor | None = None,
    shared_out: torch.Tensor | None = None,
    topk_weights: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    workspace: Workspace | None = None,
    trace: torch.Tensor | None = None,
    grid: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(routed output [M, H], shared output [M, H']), bf16.

    logits [M, E] f32 and bias [E] f32 (biased sigmoid top-k, renormalized,
    unscaled); x [M, H] bf16; w1 [E, 2I, H/2] / w2 [E, H, I/2] MXFP4 with their
    E8M0 scales, [gate; up] rows, as AITER's a16w4 shuffle leaves them. The
    shared expert: shared_x [M, H'], shared_w_gu [2I', H'], shared_w_dn [H', I']
    bf16 (Kimi-K3 runs it at the full hidden size H', the routed experts at
    the latent H); SiTU with no hard clamp, linear_beta None or <= 0 leaves up
    unclipped. topk_weights / topk_ids [M, topk] receive the routing.
    """
    m, hidden = x.shape
    ne = w1.shape[0]
    inter = w1.shape[1] // 2
    device = x.device
    assert supported(x, ne, topk, inter, shared_x, shared_w_gu, shared_w_dn)
    sh_hidden, sh_inter = shared_w_dn.shape
    lb = shared_linear_beta
    lb = float(lb) if lb is not None and lb > 0 else 0.0
    if out is None:
        out = torch.empty((m, hidden), dtype=torch.bfloat16, device=device)
    if shared_out is None:
        shared_out = torch.empty_like(shared_x)
    if topk_weights is None:
        topk_weights = torch.empty((m, topk), dtype=torch.float32, device=device)
    if topk_ids is None:
        topk_ids = torch.empty((m, topk), dtype=torch.int32, device=device)
    launch = _launcher(
        ne, topk, hidden, inter, float(situ_beta), float(situ_linear_beta),
        (sh_hidden, sh_inter, float(shared_beta), lb), trace is not None,
    )  # fmt: skip
    ws = workspace or _workspace(device, topk, inter, sh_inter)
    meta = launch.work_meta
    work = m + meta["SH_T"] + max_m_blocks(m, topk) * (meta["NNB1"] + meta["NNB2"])
    if grid is None:
        grid = min(work, _num_cus(device.index))
    _run_compiled(
        launch,
        logits.data_ptr(),
        bias.data_ptr(),
        x.data_ptr(),
        w1.data_ptr(),
        w1_scale.data_ptr(),
        w2.data_ptr(),
        w2_scale.data_ptr(),
        out.data_ptr(),
        topk_weights.data_ptr(),
        topk_ids.data_ptr(),
        ws.ctrl.data_ptr(),
        shared_x.data_ptr(),
        shared_w_gu.data_ptr(),
        shared_w_dn.data_ptr(),
        shared_out.data_ptr(),
        0 if trace is None else trace.data_ptr(),
        m,
        grid,
        torch.cuda.current_stream(device),
    )
    return out, shared_out
