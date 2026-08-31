# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.parser/engine/parser_engine_config -> vllm.frontend.processing.parser.engine.parser_engine_config (sys.modules alias)."""
import importlib
import sys

_real = importlib.import_module("vllm.frontend.processing.parser.engine.parser_engine_config")
sys.modules[__name__] = _real
