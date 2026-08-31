# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.entrypoints/generate/beam_search/online -> vllm.frontend.entrypoints.generate.beam_search.online (sys.modules alias)."""

import importlib
import sys

_real = importlib.import_module("vllm.frontend.entrypoints.generate.beam_search.online")
sys.modules[__name__] = _real
