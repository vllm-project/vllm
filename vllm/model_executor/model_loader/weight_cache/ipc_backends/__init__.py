# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Local weight-sharing backends, independent of D2D copy movers."""

from vllm.model_executor.model_loader.weight_cache.ipc_backends.base import (
    IPCBackendBase,
)
from vllm.model_executor.model_loader.weight_cache.ipc_backends.torch_ipc import (
    TorchIPCBackend,
)
from vllm.model_executor.model_loader.weight_cache.ipc_backends.vmm_ipc import (
    VmmIPCBackend,
)

IPC_BACKENDS: dict[str, type[IPCBackendBase]] = {
    "torch_ipc": TorchIPCBackend,
    "vmm_ipc": VmmIPCBackend,
}
