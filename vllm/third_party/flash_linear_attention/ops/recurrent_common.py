# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# Portions of this file are adapted from the flash-linear-attention project,
# which is licensed under the MIT license.
"""Shared gated-delta-rule math for the recurrent GDN kernels.

Every recurrent GDN entry point must use the same operation sequence for the
same state and token, otherwise target-only and speculative decoding can
disagree when logits are tied. Keeping normalization, gating and the
delta-rule update in one place makes that a property of the code instead of a
coincidence between separately written kernels.
"""

from vllm.triton_utils import tl, triton

from .op import exp, log

_L2NORM_EPS = tl.constexpr(1e-6)


@triton.jit
def l2norm_qk(b_q, b_k, AXIS: tl.constexpr):
    b_q_inv = tl.rsqrt(tl.sum(b_q * b_q, axis=AXIS) + _L2NORM_EPS)
    b_k_inv = tl.rsqrt(tl.sum(b_k * b_k, axis=AXIS) + _L2NORM_EPS)
    if AXIS == 1:
        b_q_inv = b_q_inv[:, None]
        b_k_inv = b_k_inv[:, None]
    b_q = b_q * b_q_inv
    b_k = b_k * b_k_inv
    return b_q, b_k


@triton.jit
def sigmoid_gating(
    b_a,
    b_b,
    b_A_log,
    b_dt_bias,
    BETA: tl.constexpr,
    THRESHOLD: tl.constexpr,
):
    x = b_a + b_dt_bias
    if BETA == 1.0:
        softplus_x = tl.where(x <= THRESHOLD, log(1.0 + exp(x)), x)
    else:
        softplus_x = tl.where(
            BETA * x <= THRESHOLD,
            (1.0 / BETA) * log(1.0 + exp(BETA * x)),
            x,
        )
    b_g = -exp(b_A_log) * softplus_x
    b_beta = tl.sigmoid(b_b)
    return b_g, b_beta


@triton.jit
def recurrent_step(b_h, b_q, b_k, b_v, b_g, b_beta, IS_KDA: tl.constexpr):
    if IS_KDA:
        b_h *= exp(b_g[None, :])
    else:
        b_h *= exp(b_g)
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= b_beta
    b_h += b_v[:, None] * b_k[None, :]
    b_o = tl.sum(b_h * b_q[None, :], 1)
    return b_h, b_o
