# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Local weight-sharing backends, independent of D2D copy movers."""

from vllm.preload.ipc_backends.base import IPCBackendBase
from vllm.preload.ipc_backends.torch_ipc import TorchIPCBackend
from vllm.preload.ipc_backends.vmm_ipc import VmmIPCBackend

IPC_BACKENDS: dict[str, type[IPCBackendBase]] = {
    "torch_ipc": TorchIPCBackend,
    "vmm_ipc": VmmIPCBackend,
}
