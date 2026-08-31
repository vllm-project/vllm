# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
# COMPAT SHIM (auto-generated): old path -> canonical new path

"""Compatibility shim: vllm.distributed/ec_transfer/ec_connector/cpu/common -> vllm.backends.distributed.ec_transfer.ec_connector.cpu.common (sys.modules alias)."""

import importlib
import sys

_real = importlib.import_module(
    "vllm.backends.distributed.ec_transfer.ec_connector.cpu.common"
)
sys.modules[__name__] = _real
