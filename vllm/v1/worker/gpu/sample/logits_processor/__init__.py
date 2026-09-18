# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom logits processors for the V2 model runner.

Kept import-light: the frontend process imports this package to validate
per-request params, so nothing here may pull in model-runner side modules
(torch, triton, worker state) at import time.
"""

from vllm.v1.worker.gpu.sample.logits_processor.interface import (
    LogitsContext,
    LogitsProcessor,
    LogitsProcRequestState,
)
from vllm.v1.worker.gpu.sample.logits_processor.loader import (
    build_custom_logits_processors,
    build_custom_logits_processors_params_validator,
)

__all__ = [
    "LogitsProcRequestState",
    "LogitsContext",
    "LogitsProcessor",
    "build_custom_logits_processors",
    "build_custom_logits_processors_params_validator",
]
