# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import warnings
from vllm.model_executor.layers.quantization.utils.flashinfer_moe import *  # noqa: F403

warnings.warn(
    "vllm.model_executor.layers.quantization.utils.flashinfer_utils is deprecated and "
    "will be removed in a future release. Please import from "
    "vllm.model_executor.layers.quantization.utils.flashinfer_moe instead.",
    DeprecationWarning,
    stacklevel=2,
)
