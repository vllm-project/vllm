import torch
from typing import Optional

from vllm.config import DeviceConfig
from vllm.utils import in_wsl, is_neuron


def get_device_stream(device_config: DeviceConfig):
    if torch.cuda.is_available() and device_config.device.type == "cuda":
        stream = torch.cuda.stream
    else:
        stream = None
    return stream


def get_device_cache_events(device_config: DeviceConfig, num_layers):
    if torch.cuda.is_available() and device_config.device.type == "cuda":
        cache_stream = torch.cuda.Stream()
        assert cache_stream != torch.cuda.current_stream()
        # Initialize the events for stream synchronization.
        events = [torch.cuda.Event() for _ in range(num_layers)]
    else:
        cache_stream = None
        events = None
    return cache_stream, events


def device_empty_cache(device_config: DeviceConfig):
    if torch.cuda.is_available() and device_config.device.type == "cuda":
        torch.cuda.empty_cache()
    else:
        pass


def device_synchronize(device_config: DeviceConfig):
    if torch.cuda.is_available() and device_config.device.type == "cuda":
        torch.cuda.synchronize()
    else:
        pass


def mem_get_info(device_config: DeviceConfig):
    if torch.cuda.is_available() and device_config.device.type == "cuda":
        return torch.cuda.mem_get_info()
    else:
        return 0, 0


def get_distribute_backend(device_config: DeviceConfig):
    if torch.cuda.is_available() and device_config.device.type == "cuda":
        backend = "nccl"
    else:
        backend = ""
    return backend


def could_pin_memory(target_device: Optional[torch.device]):
    if in_wsl() or is_neuron():
        return False
    if target_device is None:
        return True
    else:
        return target_device.type == "cuda"
