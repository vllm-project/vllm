# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.multimodal/cache -> vllm.frontend.processing.multimodal.cache (sys.modules alias)."""

import importlib
import sys

_real = importlib.import_module("vllm.frontend.processing.multimodal.cache")
sys.modules[__name__] = _real
