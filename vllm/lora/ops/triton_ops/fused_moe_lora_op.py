# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.lora/ops/triton_ops/fused_moe_lora_op -> vllm.runtime.modeling.lora.ops.triton_ops.fused_moe_lora_op (sys.modules alias)."""

import importlib
import sys

_real = importlib.import_module(
    "vllm.runtime.modeling.lora.ops.triton_ops.fused_moe_lora_op"
)
sys.modules[__name__] = _real
