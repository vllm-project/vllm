# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.multimodal/gpu_ipc_memory -> vllm.frontend.processing.multimodal.gpu_ipc_memory (sys.modules alias)."""

import importlib
import sys

_real = importlib.import_module("vllm.frontend.processing.multimodal.gpu_ipc_memory")
sys.modules[__name__] = _real
