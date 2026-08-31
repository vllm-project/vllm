# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.multimodal/processing/dummy_inputs -> vllm.frontend.processing.multimodal.processing.dummy_inputs (sys.modules alias)."""
import importlib
import sys

_real = importlib.import_module("vllm.frontend.processing.multimodal.processing.dummy_inputs")
sys.modules[__name__] = _real
