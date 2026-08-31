# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.lora/punica_wrapper/punica_cpu -> vllm.runtime.modeling.lora.punica_wrapper.punica_cpu (sys.modules alias)."""
import importlib
import sys

_real = importlib.import_module("vllm.runtime.modeling.lora.punica_wrapper.punica_cpu")
sys.modules[__name__] = _real
