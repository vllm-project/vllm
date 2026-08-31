# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.distributed/weight_transfer/nccl_common -> vllm.backends.distributed.weight_transfer.nccl_common (sys.modules alias)."""
import importlib
import sys

_real = importlib.import_module("vllm.backends.distributed.weight_transfer.nccl_common")
sys.modules[__name__] = _real
