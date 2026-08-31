# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.transformers_utils/processors/muse_glimmer -> vllm.foundation.integrations.transformers_utils.processors.muse_glimmer (sys.modules alias)."""
import importlib
import sys

_real = importlib.import_module("vllm.foundation.integrations.transformers_utils.processors.muse_glimmer")
sys.modules[__name__] = _real
