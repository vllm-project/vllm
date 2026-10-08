# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 mono MoE: the routed experts (top-k, sort, a4w4 gemm1 / gemm2) and
the shared expert of a small decode batch in one persistent FlyDSL launch
(README.md). The host side is ``runner``; vLLM docks it in
``..latent_moe_runner``."""
