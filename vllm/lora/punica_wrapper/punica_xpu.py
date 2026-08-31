# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.lora/punica_wrapper/punica_xpu -> vllm.runtime.modeling.lora.punica_wrapper.punica_xpu (sys.modules alias)."""

import importlib
import sys

_real = importlib.import_module("vllm.runtime.modeling.lora.punica_wrapper.punica_xpu")
sys.modules[__name__] = _real
