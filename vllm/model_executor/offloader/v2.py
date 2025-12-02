# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/utils/offloader.py
"""OffloaderV2: Advanced CPU offloading with async prefetching."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Generator

import torch
import torch.nn as nn
from torch.func import functional_call

from vllm.logger import init_logger
from vllm.model_executor.offloader.base import BaseOffloader
from vllm.utils.platform_utils import is_pin_memory_available

logger = init_logger(__name__)

# Type aliases
_SubmoduleAccessor = Callable[[nn.Module], nn.Module]
_WhitelistParamNamesCreator = Callable[[nn.Module], list[str]]


class OffloaderV2(BaseOffloader):
    """Advanced offloader with group-based selection and async prefetching.

    Unlike UVA offloading which provides zero-copy access, V2 explicitly
    manages parameter transfers with prefetching to hide latency.

    Args:
        group_size: Group every N layers together.
        num_in_group: Offload this many layers per group (last N of each group).
        prefetch_step: Number of layers to prefetch ahead.
        mode: Offload mode ("cpu" is currently supported).
    """

    def __init__(
        self,
        group_size: int,
        num_in_group: int,
        prefetch_step: int,
        mode: str = "cpu",
    ):
        self.group_size = group_size
        self.num_in_group = num_in_group
        self.prefetch_step = prefetch_step
        self.mode = mode
        self.alt_stream = torch.cuda.Stream()
        self.module_offloaders: list[_ModuleOffloader] = []
        self.total_offloaded_bytes = 0

    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
        submodule_accessor: _SubmoduleAccessor | None = None,
        whitelist_param_names_creator: _WhitelistParamNamesCreator | None = None,
    ) -> list[nn.Module]:
        """Wrap modules with V2 offloading and prefetching logic."""
        assert len(self.module_offloaders) == 0, (
            "wrap_modules should only be called once"
        )

        all_modules = []
        offload_submodules = []

        for module_index, module in enumerate(modules_generator):
            all_modules.append(module)

            # Select layers to offload based on group pattern
            # Offload last num_in_group layers of each group_size
            if module_index % self.group_size >= self.group_size - self.num_in_group:
                submodule = submodule_accessor(module) if submodule_accessor else module
                whitelist_param_names = (
                    whitelist_param_names_creator(submodule)
                    if whitelist_param_names_creator
                    else [name for name, _ in submodule.named_parameters()]
                )

                offload_submodules.append(submodule)
                self.module_offloaders.append(
                    _ModuleOffloader(
                        mode=self.mode,
                        module=submodule,
                        alt_stream=self.alt_stream,
                        whitelist_param_names=whitelist_param_names,
                    )
                )

        # Hook forward passes for all offloaded submodules
        for index, submodule in enumerate(offload_submodules):
            self._hook_module_forward(index, submodule)

        return all_modules

    def _hook_module_forward(self, index: int, module: nn.Module):
        """Hook module's forward to implement prefetch + execute + offload pattern."""
        original_forward = module.forward

        def forward(*args, **kwargs):
            module.forward = original_forward
            device_tensors = self.module_offloaders[index].wait_and_get_device_tensors()
            output = functional_call(module, device_tensors, args=args, kwargs=kwargs)
            next_index = (index + self.prefetch_step) % len(self.module_offloaders)
            self.module_offloaders[next_index].start_onload()
            self.module_offloaders[index].offload()
            module.forward = forward
            return output

        module.forward = forward

    def post_init(self):
        """Initialize offloaders and start prefetching first N modules."""
        for offloader in self.module_offloaders:
            offloader.post_init()
            self.total_offloaded_bytes += offloader.offloaded_bytes

        logger.info_once(
            f"[OffloaderV2] Initialized {len(self.module_offloaders)} modules. "
            f"Total GPU memory saved: {self.total_offloaded_bytes / 1e9:.4f} GB "
            f"(group_size={self.group_size}, num_in_group={self.num_in_group}, "
            f"prefetch_step={self.prefetch_step}, mode={self.mode})"
        )

        for i in range(min(self.prefetch_step, len(self.module_offloaders))):
            self.module_offloaders[i].start_onload()

    @property
    def forbid_copy_engine_usage(self) -> bool:
        """CPU mode may conflict with copy engine in some scenarios."""
        return self.mode == "cpu"


class _ModuleOffloader:
    """Manages offloading for a single module.

    Responsibilities:
    - Create parameter offloaders for each parameter
    - Coordinate async loading via alternate CUDA stream
    - Provide device tensors when needed
    """

    def __init__(
        self,
        mode: str,
        module: nn.Module,
        alt_stream: torch.cuda.Stream,
        whitelist_param_names: list[str],
    ):
        self.mode = mode
        self.module = module
        self.device = next(module.parameters()).device
        self.alt_stream = alt_stream
        self.offloaded_bytes = 0

        assert self.device != torch.device("cpu"), (
            "Module parameters should not already be on CPU "
            "(offloader handles CPU placement)"
        )

        self._device_tensors: dict[str, torch.Tensor] | None = None
        self._load_event: torch.cuda.Event | None = None

        param_dict = dict(self.module.named_parameters())
        assert all(name in param_dict for name in whitelist_param_names), (
            f"Whitelist params {whitelist_param_names} not found in module params "
            f"{list(param_dict.keys())}"
        )

        self._param_offloaders = {
            name: _BaseParamOffloader.create(mode, module=module, param_name=name)
            for name in whitelist_param_names
        }

    def post_init(self):
        """Collect total offloaded bytes (offloading already done in __init__)."""
        for param_offloader in self._param_offloaders.values():
            param_offloader.post_init()
            self.offloaded_bytes += param_offloader.offloaded_bytes

    def start_onload(self):
        """Start async loading in alternate CUDA stream."""
        # Synchronize with main stream before starting
        self.alt_stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self.alt_stream):
            # Load all parameters to device
            self._device_tensors = {
                name: offloader.create_device_tensor()
                for name, offloader in self._param_offloaders.items()
            }
            # Record completion event
            self._load_event = torch.cuda.Event()
            self._load_event.record()

    def offload(self):
        """Free device tensors (offload from GPU memory)."""
        self._device_tensors = None
        self._load_event = None

    def wait_and_get_device_tensors(self) -> dict[str, torch.Tensor]:
        """Wait for async loading to complete and return device tensors."""
        assert self._device_tensors is not None, (
            "Tensors not loaded (call start_onload first)"
        )
        # Wait for loading event to complete
        if self._load_event is not None:
            self._load_event.wait()
        return self._device_tensors


class _BaseParamOffloader(ABC):
    """Base class for parameter offloading strategies."""

    @staticmethod
    def create(mode: str, **kwargs) -> "_BaseParamOffloader":
        """Factory method to create appropriate offloader for mode."""
        if mode == "cpu":
            return _CpuParamOffloader(**kwargs)
        else:
            raise ValueError(f"Unknown offload mode: {mode}")

    def __init__(self, module: nn.Module, param_name: str):
        self._module = module
        self._param_name = param_name
        self.offloaded_bytes = 0

    @property
    def _param(self) -> nn.Parameter:
        """Get the parameter being offloaded."""
        return getattr(self._module, self._param_name)

    def post_init(self):
        """Initialize offloading (move parameter to storage)."""
        return

    @abstractmethod
    def create_device_tensor(self) -> torch.Tensor:
        """Create device tensor from offloaded storage."""
        pass


class _CpuParamOffloader(_BaseParamOffloader):
    """Offload parameter to pinned CPU memory."""

    def __init__(self, module: nn.Module, param_name: str):
        super().__init__(module, param_name)

        # Offload immediately to free GPU memory by moving param.data to CPU
        self._move_param_to_cpu()

    def _move_param_to_cpu(self):
        """Move parameter data to pinned CPU memory (modify param.data in-place)."""
        param = self._param
        pin_memory = is_pin_memory_available()

        # Calculate memory size
        self.offloaded_bytes = param.data.numel() * param.data.element_size()

        # Create pinned CPU tensor with same layout
        cpu_data = torch.empty_strided(
            size=param.data.size(),
            stride=param.data.stride(),
            dtype=param.data.dtype,
            layout=param.data.layout,
            device="cpu",
            pin_memory=pin_memory,
        )
        cpu_data.copy_(param.data)

        logger.debug_once(
            f"[OffloaderV2] Offloaded parameter '{self._param_name}': "
            f"shape={tuple(param.shape)}, dtype={param.dtype}, "
            f"size={self.offloaded_bytes / 1e9:.6f} GB, pinned={pin_memory}"
        )

        # Modify param.data in-place to point to CPU memory
        # This keeps the parameter in the module but with CPU data
        param.data = cpu_data

    def post_init(self):
        """No-op: offloading already done in __init__."""
        pass

    def create_device_tensor(self) -> torch.Tensor:
        """Load from CPU to GPU (async if pinned).

        Returns a CUDA copy of the parameter (which has CPU data).
        """
        return self._param.to("cuda", non_blocking=True)
