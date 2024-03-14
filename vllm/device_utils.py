import torch
from typing import Optional

from vllm.config import DeviceConfig
from vllm.utils import is_xpu, in_wsl, is_neuron


def get_device_stream(device_config: DeviceConfig):
    if torch.cuda.is_available() and device_config.device.type == "cuda":
        stream = torch.cuda.stream
    elif is_xpu() and device_config.device_type == "xpu":
        stream = torch.xpu.stream
    else:
        stream = None
    return stream


def get_device_cache_events(device_config: DeviceConfig, num_layers):
    if torch.cuda.is_available() and device_config.device.type == "cuda":
        cache_stream = torch.cuda.Stream()
        assert cache_stream != torch.cuda.current_stream()
        # Initialize the events for stream synchronization.
        events = [torch.cuda.Event() for _ in range(num_layers)]
    elif is_xpu() and device_config.device_type == "xpu":
        cache_stream = torch.xpu.Stream()
        # assert cache_stream != torch.xpu.current_stream()
        events = [torch.xpu.Event() for _ in range(num_layers)]
    else:
        cache_stream = None
        events = None
    return cache_stream, events


def device_empty_cache(device_config: DeviceConfig):
    if torch.cuda.is_available() and device_config.device.type == "cuda":
        torch.cuda.empty_cache()
    elif is_xpu() and device_config.device_type == "xpu":
        torch.xpu.empty_cache()
    else:
        pass


def device_synchronize(device_config: DeviceConfig):
    if torch.cuda.is_available() and device_config.device.type == "cuda":
        torch.cuda.synchronize()
    elif is_xpu() and device_config.device_type == "xpu":
        torch.xpu.synchronize()
    else:
        pass


def get_total_xpu_memory(xpu: int = 0) -> int:
    """Returns the total memory of the XPU in bytes."""
    if is_xpu():
        return torch.xpu.get_device_properties(xpu).total_memory
    return 0


def mem_get_info(device_config: DeviceConfig):
    if torch.cuda.is_available() and device_config.device.type == "cuda":
        return torch.cuda.mem_get_info()
    elif is_xpu() and device_config.device_type == "xpu":
        used_memory = torch.xpu.memory_allocated()
        total_xpu_memory = get_total_xpu_memory()
        return total_xpu_memory - used_memory, total_xpu_memory
    else:
        return 0, 0


def get_distribute_backend(device_config: DeviceConfig):
    if torch.cuda.is_available() and device_config.device.type == "cuda":
        backend = "nccl"
    elif is_xpu() and device_config.device_type == "xpu":
        backend = "ccl"
        try:
            import oneccl_bindings_for_pytorch  # noqa: F401
        except ImportError:
            pass
    else:
        backend = ""
    return backend


def could_pin_memory(target_device: Optional[torch.device]):
    if in_wsl() or is_neuron() or is_xpu():
        return False
    if target_device is None:
        return True
    else:
        return target_device.type == "cuda"
