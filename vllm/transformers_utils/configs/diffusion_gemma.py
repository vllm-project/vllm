# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.transformers_utils/configs/diffusion_gemma -> vllm.foundation.integrations.transformers_utils.configs.diffusion_gemma (sys.modules alias)."""
import importlib
import sys

_real = importlib.import_module("vllm.foundation.integrations.transformers_utils.configs.diffusion_gemma")
sys.modules[__name__] = _real
