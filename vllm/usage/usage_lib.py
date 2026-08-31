# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.usage/usage_lib -> vllm.foundation.observability.usage.usage_lib (sys.modules alias)."""

import importlib
import sys

_real = importlib.import_module("vllm.foundation.observability.usage.usage_lib")
sys.modules[__name__] = _real
