# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.lora/layers/logits_processor -> vllm.runtime.modeling.lora.layers.logits_processor (sys.modules alias)."""
import importlib
import sys

_real = importlib.import_module("vllm.runtime.modeling.lora.layers.logits_processor")
sys.modules[__name__] = _real
