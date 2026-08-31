# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.entrypoints/openai/responses/streaming_events -> vllm.frontend.entrypoints.openai.responses.streaming_events (sys.modules alias)."""
import importlib
import sys

_real = importlib.import_module("vllm.frontend.entrypoints.openai.responses.streaming_events")
sys.modules[__name__] = _real
