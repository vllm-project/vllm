# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pickle

import torch


class CudaIPCWrapper:

    def __init__(self, tensor: torch.Tensor):
        assert tensor.storage_offset() == 0
        assert tensor.is_contiguous()
        storage = tensor.untyped_storage()
        handle = storage._share_cuda_()

        self.handle = handle
        self.dtype = tensor.dtype
        self.shape = tensor.shape
        self.device = tensor.device.index  # Explicit device ordinal

    def to_tensor(self):
        torch.cuda.set_device(self.device)  # Ensure correct device/context
        device = self.handle[0]
        storage = torch.UntypedStorage._new_shared_cuda(*self.handle)
        t = torch.tensor(0, device=device, dtype=self.dtype)
        t.set_(storage)
        return t.view(self.shape)

    def __eq__(self, other):
        if not isinstance(other, CudaIPCWrapper):
            return False
        return (self.handle == other.handle and self.dtype == other.dtype
                and self.shape == other.shape and self.device == other.device)

    def serialize(self) -> bytes:
        return pickle.dumps(self)

    @staticmethod
    def deserialize(data: bytes) -> 'CudaIPCWrapper':
        return pickle.loads(data)


def encode_cuda_ipc_wrapper(wrapper: CudaIPCWrapper) -> bytes:
    return wrapper.serialize()


def decode_cuda_ipc_wrapper(data: bytes) -> CudaIPCWrapper:
    return CudaIPCWrapper.deserialize(data)
