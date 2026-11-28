# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Float64 references of the gfx942 specialized decode's stages with the stock
path's rounding points (`bf`), for Muse Spark PTPC FP8 at TP = EP."""

import torch

RMS_EPS = 1e-5
ROUTE_EPS = 1e-15


def bf(t):
    return t.to(torch.bfloat16).double()


def scaled_mm(q, qs, w, ws):
    """`bf16(q @ w.T * qs * ws)`: the per-token / per-channel FP8 GEMM."""
    return bf(q @ w.T * qs * ws[None, :])


def rms(x, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def sigmoid(x):
    return torch.sigmoid(x)


def o_gate_norm(attn, og, head_dim: int):
    """`rms_norm_gated`: a weightless RMSNorm a head times `sigmoid(og)`,
    one bf16 rounding. `attn`, `og`: [rows, q_local]."""
    rows = attn.shape[0]
    a = attn.view(rows, -1, head_dim)
    g = og.view(rows, -1, head_dim)
    return bf(rms(a, RMS_EPS) * sigmoid(g)).reshape(rows, -1)


def boundary(branch, stream, post_w, post_input_scale, alpha, beta, pre_w, post_eps):
    """`fused_norm_residual_norm_fwd`: post-norm of `branch` (weightless when
    `post_w` is None; weight-first when `post_input_scale`), the gated
    residual in fp32 (no rounding of the post-norm), the next pre-norm (`w + 1`)
    rounded to bf16. Returns `(x bf16, new stream fp32)`."""
    x = branch
    if post_w is not None and post_input_scale:
        x = x * (post_w + 1.0)
    y = rms(x, post_eps)
    if post_w is not None and not post_input_scale:
        y = y * (post_w + 1.0)
    new = (alpha * stream + beta * y).float().double()
    z = bf(rms(new, RMS_EPS) * (pre_w + 1.0))
    return z, new


def route(logits, bias, topk: int, route_scale: float = 1.0):
    """`fused_muse_spark_routing`: top-k of `sigmoid(l) + bias` (ties: the
    lowest id), weights `score * (route_scale / (sum + 1e-15))`."""
    scores = sigmoid(logits)
    biased = scores + bias[None, :]
    ids = torch.empty(logits.shape[0], topk, dtype=torch.long)
    for t in range(logits.shape[0]):
        v = biased[t].clone()
        for j in range(topk):
            m = v.max()
            i = int((v == m).nonzero()[0])
            ids[t, j] = i
            v[i] = float("-inf")
    picked = scores.gather(1, ids)
    w = picked * (route_scale / (picked.sum(-1, keepdim=True) + ROUTE_EPS))
    return ids, w


def expert_act(h_q, h_s, w13, w13_s, inter: int):
    """`bf16(bf16(silu(g)) * u)` of the gate/up GEMM of one expert."""
    gu = scaled_mm(h_q, h_s, w13, w13_s)
    g, u = gu[:, :inter], gu[:, inter:]
    return bf(bf(g * sigmoid(g)) * u)
