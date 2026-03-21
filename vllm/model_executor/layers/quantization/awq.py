# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Backwards-compatibility shim.
# AWQ quantization has been consolidated into awq_marlin.py.
# This module re-exports the public API so that existing imports continue to work.

from vllm.model_executor.layers.quantization.awq_marlin import (  # noqa: F401
    AWQConfig,
    AWQLinearMethod,
)

__all__ = ["AWQConfig", "AWQLinearMethod"]
