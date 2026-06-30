from multiprocessing import shared_memory
from unittest.mock import patch


import numpy as np
import torch

from vllm.utils.torch_utils import PIN_MEMORY
from vllm import _custom_ops as ops


class PagedSHMStorage:
    def __init__(self, size: int, block_size: int, name: str | None = None, pin: bool = False):
        self.name = name
        self.pin = pin
        self.block_size = block_size
        self.n_block = size // block_size
        self.size = block_size * self.n_block
        self.dtype = np.uint8

        if name is None:
            self.shm = shared_memory.SharedMemory(create=True, size=self.size)
        else:
            with patch("multiprocessing.resource_tracker.register",lambda *args, **kwargs: None,):
                try:
                    self.shm = shared_memory.SharedMemory(name=name)
                    assert self.shm.size >= self.size
                except FileNotFoundError:
                    raise FileNotFoundError(f"Shared memory '{name}' not found")
        assert self.shm.buf is not None, "Buffer was not created"

        self.name = self.shm.name
        self.shm_tensor = torch.from_numpy(
            np.ndarray(self.size, dtype=self.dtype, buffer=self.shm.buf)
        )
        self.shm_np = self.shm_tensor.numpy()

        if pin and PIN_MEMORY:
            from vllm.v1.simple_kv_offload.cuda_mem_ops import pin_tensor
            pin_tensor(self.shm_tensor)

    def write(self, data: bytes | np.ndarray | torch.Tensor, blocks: list[int]):
        if isinstance(data, torch.Tensor):
            if data.device.type != "cpu":
                self._write_gpu_tensor_to_shm(data, blocks)
                return
            data_np = data.contiguous().view(torch.uint8).numpy()
        elif isinstance(data, bytes):
            data_np = np.frombuffer(data, dtype=np.uint8)
        elif isinstance(data, np.ndarray):
            data_np = np.ascontiguousarray(data).view(np.uint8)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        nelement = len(data_np)
        n_blocks = len(blocks)
        if nelement > n_blocks * self.block_size:
            raise ValueError("Data too large for provided blocks")

        full_blocks = nelement // self.block_size
        remainder = nelement % self.block_size

        for i in range(full_blocks):
            blk = blocks[i]
            start = blk * self.block_size
            self.shm_np[start:start + self.block_size] = \
                data_np[i * self.block_size:(i + 1) * self.block_size]

        if remainder > 0:
            blk = blocks[full_blocks]
            start = blk * self.block_size
            self.shm_np[start:start + remainder] = \
                data_np[full_blocks * self.block_size:]

    def _write_gpu_tensor_to_shm(self, data: torch.Tensor, blocks: list[int]):
        # GPU → CPU using cuMemcpyBatchAsync

        data = data.contiguous().view(torch.uint8)
        nelement = data.numel()
        n_blocks = len(blocks)
        if nelement > n_blocks * self.block_size:
            raise ValueError("Data too large for provided blocks")

        full_blocks = nelement // self.block_size
        remainder = nelement % self.block_size

        src_addrs_list = []
        dst_addrs_list = []
        sizes_list = []

        for i in range(full_blocks):
            src_start = i * self.block_size
            dst_start = blocks[i] * self.block_size

            src_addrs_list.append(data.data_ptr() + src_start)
            dst_addrs_list.append(self.shm_tensor.data_ptr() + dst_start)
            sizes_list.append(self.block_size)

        if remainder > 0:
            src_start = full_blocks * self.block_size
            dst_start = blocks[full_blocks] * self.block_size

            src_addrs_list.append(data.data_ptr() + src_start)
            dst_addrs_list.append(self.shm_tensor.data_ptr() + dst_start)
            sizes_list.append(remainder)

        src_addrs = torch.tensor(src_addrs_list)
        dst_addrs = torch.tensor(dst_addrs_list)
        sizes = torch.tensor(sizes_list, dtype=torch.int64)

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            ops.swap_blocks_batch(
                src_addrs,
                dst_addrs,
                sizes,
            )
        torch.cuda.current_stream().wait_stream(stream)

    def read(self, nelement: int, blocks: list[int],
             output: np.ndarray | torch.Tensor | None = None) -> np.ndarray | torch.Tensor:
        if nelement > len(blocks) * self.block_size:
            raise ValueError("Requested data too large for provided blocks")

        full_blocks = nelement // self.block_size
        remainder = nelement % self.block_size

        if output is None:
            output_np = np.empty(nelement, dtype=np.uint8)
            out_is_tensor = False
        elif isinstance(output, torch.Tensor):
            if output.device.type != "cpu":
                raise TypeError("read() only supports CPU tensors")
            if output.numel() < nelement:
                raise ValueError("Output tensor too small")
            output_np = output.numpy()
            out_is_tensor = True
        else:  # numpy array
            if output.size < nelement:
                raise ValueError("Output array too small")
            output_np = output
            out_is_tensor = False

        for i in range(full_blocks):
            blk = blocks[i]
            start = blk * self.block_size
            output_np[i * self.block_size:(i + 1) * self.block_size] = \
                self.shm_np[start:start + self.block_size]

        if remainder > 0:
            blk = blocks[full_blocks]
            start = blk * self.block_size
            output_np[full_blocks * self.block_size:] = \
                self.shm_np[start:start + remainder]

        return output_np if not out_is_tensor else output

    def read_to_device(self, nelement: int, blocks: list[int], output: torch.Tensor | None = None):
        # CPU → GPU using cuMemcpyBatchAsync

        if output is None:
            output_tensor = torch.empty(nelement, dtype=torch.uint8, device="")
        else:
            output_tensor = output

        if output_tensor.device.type == "cpu":
            raise TypeError("read_to_device() requires a GPU tensor")
        if output_tensor.numel() < nelement:
            raise ValueError("Output tensor too small")
        if nelement > len(blocks) * self.block_size:
            raise ValueError("Requested data too large for provided blocks")

        output_tensor = output_tensor.contiguous().view(torch.uint8)
        full_blocks = nelement // self.block_size
        remainder = nelement % self.block_size

        src_addrs_list = []
        dst_addrs_list = []
        sizes_list = []

        for i in range(full_blocks):
            src_start = i * self.block_size
            dst_start = blocks[i] * self.block_size

            src_addrs_list.append(self.shm_tensor.data_ptr() + dst_start)
            dst_addrs_list.append(output.data_ptr() + src_start)
            sizes_list.append(self.block_size)

        if remainder > 0:
            src_start = full_blocks * self.block_size
            dst_start = blocks[full_blocks] * self.block_size

            src_addrs_list.append(self.shm_tensor.data_ptr() + dst_start)
            dst_addrs_list.append(output.data_ptr() + src_start)
            sizes_list.append(remainder)

        src_addrs = torch.tensor(src_addrs_list)
        dst_addrs = torch.tensor(dst_addrs_list)
        sizes = torch.tensor(sizes_list, dtype=torch.int64)

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            ops.swap_blocks_batch(
                src_addrs,
                dst_addrs,
                sizes,
            )
        torch.cuda.current_stream().wait_stream(stream)
        return output

    def close(self):
        print("close")
        if hasattr(self, 'shm'):
            self.shm.close()

    def unlink(self):
        print("unlink")
        if hasattr(self, 'shm'):
            self.shm.unlink()

    def __del__(self):
        print("=" *80)
        print("__del__")
        self.close()
        self.unlink()