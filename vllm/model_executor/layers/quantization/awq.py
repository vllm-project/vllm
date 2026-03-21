# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Backward compatibility: AWQConfig and AWQLinearMethod have been consolidated
# into awq_marlin.py
from vllm.model_executor.layers.quantization.awq_marlin import (  # noqa: F401
    AWQConfig,
    AWQLinearMethod,
)
