# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.utils/deep_gemm -> vllm.foundation.utilities.deep_gemm (sys.modules alias)."""

import importlib
import sys

_real = importlib.import_module("vllm.foundation.utilities.deep_gemm")
sys.modules[__name__] = _real
