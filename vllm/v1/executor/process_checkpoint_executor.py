# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility import for the original experimental executor name."""

from vllm.v1.executor.cuda_checkpoint_executor import (
    CUDACheckpointExecutor,
    _validate_nccl_checkpoint,  # noqa: F401
)

ProcessCheckpointExecutor = CUDACheckpointExecutor
