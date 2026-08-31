# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.utils/nvtx_pytorch_hooks -> vllm.foundation.utilities.nvtx_pytorch_hooks (sys.modules alias)."""

import importlib
import sys

_real = importlib.import_module("vllm.foundation.utilities.nvtx_pytorch_hooks")
sys.modules[__name__] = _real
