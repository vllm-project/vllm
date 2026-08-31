# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.distributed/eplb/rebalance_execute -> vllm.backends.distributed.eplb.rebalance_execute (sys.modules alias)."""
import importlib
import sys

_real = importlib.import_module("vllm.backends.distributed.eplb.rebalance_execute")
sys.modules[__name__] = _real
