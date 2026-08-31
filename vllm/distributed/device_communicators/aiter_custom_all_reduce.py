# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.distributed/device_communicators/aiter_custom_all_reduce -> vllm.backends.distributed.device_communicators.aiter_custom_all_reduce (sys.modules alias)."""

import importlib
import sys

_real = importlib.import_module(
    "vllm.backends.distributed.device_communicators.aiter_custom_all_reduce"
)
sys.modules[__name__] = _real
