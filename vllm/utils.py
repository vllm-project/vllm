import ctypes
import enum
import uuid
from platform import uname

import psutil
import torch


class Device(enum.Enum):
    GPU = enum.auto()
    CPU = enum.auto()


class Counter:

    def __init__(self, start: int = 0) -> None:
        self.counter = start

    def __next__(self) -> int:
        i = self.counter
        self.counter += 1
        return i

    def reset(self) -> None:
        self.counter = 0


def get_gpu_memory(gpu: int = 0) -> int:
    """Returns the total memory of the GPU in bytes."""
    return torch.cuda.get_device_properties(gpu).total_memory


def get_cpu_memory() -> int:
    """Returns the total CPU memory of the node in bytes."""
    return psutil.virtual_memory().total


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


def in_wsl() -> bool:
    # Reference: https://github.com/microsoft/WSL/issues/4071
    return "microsoft" in " ".join(uname()).lower()


def get_max_shared_mem_bytes() -> int:
    libnames = ("libcuda.so", "libcuda.dylib", "nvcuda.dll", "cuda.dll")
    for libname in libnames:
        try:
            cuda = ctypes.CDLL(libname)
        except OSError:
            continue
        else:
            break
    else:
        raise OSError(f"Could not load any of {' '.join(libnames)}")

    smem_size = ctypes.c_size_t()
    device = ctypes.c_size_t()

    cuda.cuDeviceGet(ctypes.byref(device), torch.cuda.current_device())
    cuda.cuInit(0)
    assert not cuda.cuDeviceGetAttribute(ctypes.byref(smem_size), 97, device)
    return smem_size.value


def check_if_can_support_max_seq_len(max_seq_len: int,
                                     block_size: int) -> None:
    # Follows the logic in
    # attention_kernels.cu::single_query_cached_kv_attention_launcher
    max_shared_mem = get_max_shared_mem_bytes()
    float32_bytes = torch.finfo(torch.float).bits // 8
    padded_max_seq_len = (
        (max_seq_len + block_size - 1) / block_size) * block_size
    # 128 as a buffer
    required_shared_mem = (padded_max_seq_len + 128) * float32_bytes
    if padded_max_seq_len * float32_bytes > max_shared_mem:
        raise RuntimeError(
            f"vLLM cannot currently support max_model_len={max_seq_len} "
            f"with block_size={block_size} on GPU with compute "
            f"capability {torch.cuda.get_device_capability()} "
            f"(required shared memory {required_shared_mem} > "
            f"available shared memory {max_shared_mem}). "
            "This will be fixed in a future release.")
