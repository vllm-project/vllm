# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Local weight-sharing backends, independent of D2D copy movers."""

from vllm.model_executor.model_loader.weight_cache.ipc_backends.base import (
    IPCBackend,
)
from vllm.model_executor.model_loader.weight_cache.ipc_backends.cuda_ipc import (
    CudaIpcBackend,
)
from vllm.model_executor.model_loader.weight_cache.ipc_backends.vmm import (
    VmmBackend,
)

IPC_BACKENDS: dict[str, type[IPCBackend]] = {
    "cuda_ipc": CudaIpcBackend,
    "vmm": VmmBackend,
}
