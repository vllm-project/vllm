# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.distributed/kv_transfer/kv_connector/v1/multi_connector -> vllm.backends.distributed.kv_transfer.kv_connector.v1.multi_connector (sys.modules alias)."""
import importlib
import sys

_real = importlib.import_module("vllm.backends.distributed.kv_transfer.kv_connector.v1.multi_connector")
sys.modules[__name__] = _real
