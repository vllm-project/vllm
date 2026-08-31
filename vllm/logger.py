# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.logger -> vllm.foundation.observability.logger (sys.modules alias)."""

import importlib
import sys

_real = importlib.import_module("vllm.foundation.observability.logger")
sys.modules[__name__] = _real
