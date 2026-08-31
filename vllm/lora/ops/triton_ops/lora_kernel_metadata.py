# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.lora/ops/triton_ops/lora_kernel_metadata -> vllm.runtime.modeling.lora.ops.triton_ops.lora_kernel_metadata (sys.modules alias)."""
import importlib
import sys

_real = importlib.import_module("vllm.runtime.modeling.lora.ops.triton_ops.lora_kernel_metadata")
sys.modules[__name__] = _real
