# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.plugins/lora_resolvers/hf_hub_resolver -> vllm.foundation.extensibility.plugins.lora_resolvers.hf_hub_resolver (sys.modules alias)."""
import importlib
import sys

_real = importlib.import_module("vllm.foundation.extensibility.plugins.lora_resolvers.hf_hub_resolver")
sys.modules[__name__] = _real
