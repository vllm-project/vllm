# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.triton_utils/tensor_descriptor -> vllm.backends.compute.dsl.triton_utils.tensor_descriptor (sys.modules alias)."""
import importlib
import sys

_real = importlib.import_module("vllm.backends.compute.dsl.triton_utils.tensor_descriptor")
sys.modules[__name__] = _real
