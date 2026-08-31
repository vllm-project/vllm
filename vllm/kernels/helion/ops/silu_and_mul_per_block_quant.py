# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.kernels/helion/ops/silu_and_mul_per_block_quant -> vllm.backends.compute.kernels.helion.ops.silu_and_mul_per_block_quant (sys.modules alias)."""

import importlib
import sys

_real = importlib.import_module(
    "vllm.backends.compute.kernels.helion.ops.silu_and_mul_per_block_quant"
)
sys.modules[__name__] = _real
