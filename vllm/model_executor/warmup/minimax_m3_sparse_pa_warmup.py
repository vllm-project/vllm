# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Connect a loaded ROCm MiniMax-M3 model to its AITER sparse PA warmup."""

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker


def minimax_m3_sparse_pa_warmup(worker: "Worker") -> None:
    """Warm the AITER sparse PA decode variants of a loaded MiniMax-M3."""
    if "vllm.models.minimax_m3.amd.model" not in sys.modules:
        return
    from vllm.models.minimax_m3.amd.ops.sparse_pa import (
        minimax_m3_sparse_pa_decode_warmup,
    )

    minimax_m3_sparse_pa_decode_warmup(worker.get_model(), worker.model_config.dtype)
