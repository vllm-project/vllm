# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.distributed/nixl_utils -> vllm.backends.distributed.nixl_utils (sys.modules alias)."""
import importlib
import sys

_real = importlib.import_module("vllm.backends.distributed.nixl_utils")
sys.modules[__name__] = _real
