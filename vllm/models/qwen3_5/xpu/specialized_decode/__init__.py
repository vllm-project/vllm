# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.5 XPU specialized MoE decode over the eager backbone."""

MAX_DECODE_ROWS = 8
MAX_ROUTED_ROWS = 512
