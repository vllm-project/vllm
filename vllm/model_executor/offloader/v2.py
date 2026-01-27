# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/utils/offloader.py
"""OffloaderV2: CPU offloading with async prefetching.

This version uses static buffers and event-based stream forking for
torch.compile + CUDA graph compatibility. Events allow the copy stream
to join CUDA graph captures, ensuring H2D copies are properly captured.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from dataclasses import dataclass

import torch
import torch.nn as nn

# Import v2_ops to register custom ops at module load time
import vllm.model_executor.offloader.v2_ops  # noqa: F401
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import BaseOffloader
from vllm.utils.platform_utils import is_pin_memory_available

logger = init_logger(__name__)

_SubmoduleAccessor = Callable[[nn.Module], nn.Module]
_WhitelistParamNamesCreator = Callable[[nn.Module], list[str]]


@dataclass
class ParamInfo:
    """Metadata about an offloaded parameter."""

    name: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype

    @property
    def key(self) -> tuple[str, tuple[int, ...], tuple[int, ...], torch.dtype]:
        """Unique key for buffer pool grouping.

        Includes parameter name to prevent different parameters with the same
        shape from sharing buffers within the same layer. Parameters with the
        same name across different layers will share buffers (via slots).

        Includes stride because parameters with same shape but different
        strides need separate buffers to preserve memory layout.
        """
        return (self.name, self.shape, self.stride, self.dtype)

    @property
    def num_bytes(self) -> int:
        """Size in bytes."""
        numel = 1
        for dim in self.shape:
            numel *= dim
        return numel * torch.tensor([], dtype=self.dtype).element_size()


class StaticBufferPool:
    """Pre-allocated GPU buffer pool for offloaded parameters.

    Allocates slot_capacity copies of each unique parameter
    (name, shape, stride, dtype), allowing for double/triple buffering
    during prefetch.

    Buffer slots are reused circularly: layer N uses slot (N % slot_capacity).

    The key includes parameter name to prevent different parameters within
    the same layer from sharing buffers. Parameters with the same name
    across different layers share buffers via the slot mechanism.
    """

    def __init__(
        self,
        param_infos: list[ParamInfo],
        slot_capacity: int,
        device: torch.device,
    ):
        self.slot_capacity = slot_capacity
        self.total_bytes = 0
        self._device = device

        # Group by (shape, stride, dtype) - only allocate unique combinations
        unique_params: dict[tuple, ParamInfo] = {}
        for info in param_infos:
            if info.key not in unique_params:
                unique_params[info.key] = info

        # Allocate buffers: key -> list of tensors (one per slot)
        self._buffers: dict[tuple, list[torch.Tensor]] = {}
        for key, info in unique_params.items():
            slot_tensors = []
            for _ in range(slot_capacity):
                # Use empty_strided to preserve parameter's memory layout
                buf = torch.empty_strided(
                    size=info.shape,
                    stride=info.stride,
                    dtype=info.dtype,
                    device=device,
                )
                slot_tensors.append(buf)
                self.total_bytes += info.num_bytes
            self._buffers[key] = slot_tensors

        logger.debug(
            "[StaticBufferPool] Allocated %d unique (name, shape, stride, dtype), "
            "%d slots each, total %.4f GB",
            len(unique_params),
            slot_capacity,
            self.total_bytes / 1e9,
        )

    def get_buffer(
        self,
        name: str,
        shape: tuple[int, ...],
        stride: tuple[int, ...],
        dtype: torch.dtype,
        slot_idx: int,
    ) -> torch.Tensor:
        """Get a static buffer for the given name/shape/stride/dtype/slot."""
        key = (name, shape, stride, dtype)
        return self._buffers[key][slot_idx % self.slot_capacity]


class OffloaderV2(BaseOffloader):
    """Advanced offloader with group-based selection and async prefetching.

    Uses static buffers and stream synchronization for torch.compile and
    CUDA graph compatibility.

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

        # Copy stream for async H2D transfers
        self.copy_stream = torch.cuda.Stream()

        # Module offloaders and buffer pool (populated in wrap_modules/post_init)
        self.module_offloaders: list[_ModuleOffloader] = []
        self.buffer_pool: StaticBufferPool | None = None
        self.total_offloaded_bytes = 0

        # Register this instance for custom ops
        from vllm.model_executor.offloader.v2_ops import set_offloader_instance

        set_offloader_instance(self)

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
                        copy_stream=self.copy_stream,
                        whitelist_param_names=whitelist_param_names,
                        layer_idx=len(self.module_offloaders),
                    )
                )

        for index, submodule in enumerate(offload_submodules):
            self._hook_module_forward(index, submodule)

        return all_modules

    def _hook_module_forward(self, index: int, module: nn.Module):
        """Hook module's forward with torch.compile-compatible sync."""
        original_forward = module.forward

        def forward(*args, **kwargs):
            # Temporarily restore original forward to avoid recursion
            module.forward = original_forward

            # Wait for this layer's prefetch to complete
            input_tensor = args[0] if args else kwargs.get("hidden_states")
            input_tensor = torch.ops.vllm.wait_prefetch(input_tensor, index)

            # Replace the first arg with the returned tensor to maintain dependency
            if args:
                args = (input_tensor,) + args[1:]
            else:
                kwargs["hidden_states"] = input_tensor

            # No parameter swapping needed - parameters already point to
            # GPU static buffers (set in assign_static_buffer)
            output = original_forward(*args, **kwargs)

            # Start prefetch for next layer (circular)
            # Custom op returns output_tensor to create data dependency
            next_index = (index + self.prefetch_step) % len(self.module_offloaders)
            # Handle tuple output (e.g., (hidden_states, residual))
            if isinstance(output, tuple):
                output_tensor = torch.ops.vllm.start_prefetch(output[0], next_index)
                output = (output_tensor,) + output[1:]
            else:
                output = torch.ops.vllm.start_prefetch(output, next_index)

            # No explicit offload needed - static buffers are reused implicitly

            # Restore hooked forward
            module.forward = forward
            return output

        module.forward = forward

    def _wait_for_layer(self, layer_idx: int):
        """Called by custom op - wait for copy to complete.

        Synchronization strategy:
        - During CUDA graph capture: use event-based wait (graph-compatible)
        - Outside capture (warmup/eager): use wait_stream (more robust)

        During capture, we skip wait for pre-capture prefetches because:
        1. sync_before_graph_capture() ensures pre-capture work is complete
        2. We can't wait on pre-capture events during capture (isolation error)
        """
        offloader = self.module_offloaders[layer_idx]

        if torch.cuda.is_current_stream_capturing():
            # During capture, skip wait for pre-capture prefetches.
            # sync_before_graph_capture() ensures pre-capture work is complete.
            if not offloader._prefetch_in_capture:
                return
            # Event-based wait for in-capture prefetches (graph-compatible)
            torch.cuda.current_stream().wait_event(offloader._copy_done_event)
            # Mark that this prefetch has been waited on (joined).
            offloader._prefetch_in_capture = False
        else:
            # Outside capture: use wait_stream for robustness.
            # Events used in previous captures can be in invalid state.
            torch.cuda.current_stream().wait_stream(self.copy_stream)

    def sync_before_graph_capture(self):
        """Sync copy stream before CUDA graph capture or replay.

        Pre-capture prefetches from warmup must complete before capture.
        This method ensures those dependencies are satisfied.

        Call this:
        1. Before capturing a CUDA graph
        2. Before replaying a CUDA graph (if prefetches were issued outside)

        Also resets _prefetch_in_capture flags. This is critical for piecewise
        cudagraph mode where multiple subgraphs are captured sequentially.
        Each subgraph's prefetches must be tracked independently - we can't
        wait on events recorded in a different subgraph's capture.
        """
        torch.cuda.current_stream().wait_stream(self.copy_stream)

        # Reset flags so only prefetches started in THIS capture are tracked.
        # This prevents cross-capture event waits which cause cudaErrorInvalidValue.
        for offloader in self.module_offloaders:
            offloader._prefetch_in_capture = False

    def _start_prefetch(self, layer_idx: int):
        """Called by custom op - start async copy to static buffer."""
        offloader = self.module_offloaders[layer_idx]
        offloader.start_onload_to_static()

    def _join_copy_stream(self, layer_idx: int):
        """Called by custom op - join copy_stream back to compute stream.

        Used after the last layer's start_prefetch to ensure copy_stream is
        rejoined before CUDA graph capture ends. The start_prefetch forks
        copy_stream into the capture, but the corresponding wait_prefetch
        only happens in the next forward pass. During capture, this would
        leave copy_stream unjoined, causing cudaErrorStreamCaptureUnjoined.

        During capture, we check if the prefetch was started during the same
        capture. If not (e.g., from post_init or warmup), we skip the wait
        because:
        1. sync_before_graph_capture() ensures pre-capture work is complete
        2. We can't wait on pre-capture events during capture (would cause
           cudaErrorStreamCaptureIsolation)
        """
        offloader = self.module_offloaders[layer_idx]

        # During capture, skip wait for pre-capture prefetches.
        # sync_before_graph_capture() ensures pre-capture work is complete.
        if (
            torch.cuda.is_current_stream_capturing()
            and not offloader._prefetch_in_capture
        ):
            return

        torch.cuda.current_stream().wait_event(offloader._copy_done_event)

    def join_after_forward(self):
        """Join copy_stream after model forward completes.

        Call this after the model forward pass but before CUDA graph capture
        ends. This ensures copy_stream is rejoined for any prefetches started
        during the forward pass.

        We join ALL layers that have _prefetch_in_capture=True, meaning their
        prefetch was started during capture but not yet waited on (joined).
        This handles both full and piecewise cudagraph modes correctly:
        - Full mode: joins layers 0..prefetch_step-1 (prefetched by last layers)
        - Piecewise mode: joins only layers prefetched by THIS subgraph's layers
        """
        if not self.module_offloaders:
            return
        # Join all layers whose prefetch was started in capture but not waited on
        for offloader in self.module_offloaders:
            if offloader._prefetch_in_capture:
                torch.cuda.current_stream().wait_event(offloader._copy_done_event)
                offloader._prefetch_in_capture = False

    def post_init(self):
        """Allocate static buffer pool and start initial prefetches.

        Note: Parameters have already been offloaded to CPU during wrap_modules()
        (in _CpuParamOffloader.__init__), so GPU memory is available for the
        static buffer pool.
        """
        # Sync CPU storage with current param.data BEFORE collecting param info.
        # This is needed because process_weights_after_loading may have:
        # 1. Transformed weights (quantization, transpose, etc.)
        # 2. Created new CPU tensors via device_loading_context
        # Our _cpu_storage would be stale otherwise.
        for offloader in self.module_offloaders:
            offloader.sync_cpu_storage()

        # Collect parameter info (now using synced CPU storage)
        param_infos: list[ParamInfo] = []
        device: torch.device | None = None

        for offloader in self.module_offloaders:
            param_infos.extend(offloader.get_param_infos())
            if device is None:
                device = offloader.device

        if device is None:
            # No modules to offload
            return

        # Allocate static buffer pool
        self.buffer_pool = StaticBufferPool(
            param_infos=param_infos,
            slot_capacity=self.prefetch_step,
            device=device,
        )

        # Assign buffer slots and point parameters to GPU buffers
        for idx, offloader in enumerate(self.module_offloaders):
            slot_idx = idx % self.prefetch_step
            offloader.assign_buffer_slot(self.buffer_pool, slot_idx)

        # Collect offloaded bytes
        for offloader in self.module_offloaders:
            offloader.post_init()
            self.total_offloaded_bytes += offloader.offloaded_bytes

        logger.info_once(
            f"[OffloaderV2] Initialized {len(self.module_offloaders)} modules. "
            f"Total GPU memory saved: {self.total_offloaded_bytes / 1e9:.4f} GB, "
            f"Static buffer pool: {self.buffer_pool.total_bytes / 1e9:.4f} GB "
            f"(group_size={self.group_size}, num_in_group={self.num_in_group}, "
            f"prefetch_step={self.prefetch_step}, mode={self.mode})"
        )

        # Start initial prefetches
        for i in range(min(self.prefetch_step, len(self.module_offloaders))):
            self.module_offloaders[i].start_onload_to_static()


class _ModuleOffloader:
    """Manages offloading for a single module.

    Uses static buffers from a shared pool instead of dynamic allocation.
    """

    def __init__(
        self,
        mode: str,
        module: nn.Module,
        copy_stream: torch.cuda.Stream,
        whitelist_param_names: list[str],
        layer_idx: int,
    ):
        self.mode = mode
        self.module = module
        self.device = next(module.parameters()).device
        self.copy_stream = copy_stream
        self.layer_idx = layer_idx
        self.offloaded_bytes = 0

        # Event to signal when H2D copy to static buffer is complete.
        # Used for CUDA graph compatible synchronization during capture.
        # Outside capture, we use wait_stream instead (more robust).
        self._copy_done_event = torch.cuda.Event()

        # Track if last prefetch was started during CUDA graph capture.
        # Used to skip wait_event during capture for pre-capture prefetches.
        self._prefetch_in_capture = False

        assert self.device != torch.device("cpu"), (
            "Module parameters should not already be on CPU "
            "(offloader handles CPU placement)"
        )

        # Buffer pool and slot (assigned in assign_buffer_slot)
        self._buffer_pool: StaticBufferPool | None = None
        self._buffer_slot_idx: int = 0

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

    def sync_cpu_storage(self):
        """Sync CPU storage with current param.data.

        Called after process_weights_after_loading to ensure _cpu_storage
        contains the final processed weights, not stale pre-loading data.
        """
        for param_offloader in self._param_offloaders.values():
            param_offloader.sync_cpu_storage()

    def get_param_infos(self) -> list[ParamInfo]:
        """Get parameter metadata for buffer pool allocation.

        Note: sync_cpu_storage() must be called before this method to ensure
        _cpu_storage reflects the final processed weights (after quantization).
        """
        infos = []
        for name, offloader in self._param_offloaders.items():
            cpu_storage = offloader._cpu_storage
            assert cpu_storage is not None, "CPU storage not initialized"
            infos.append(
                ParamInfo(
                    name=name,
                    shape=tuple(cpu_storage.shape),
                    stride=tuple(cpu_storage.stride()),
                    dtype=cpu_storage.dtype,
                )
            )
        return infos

    def assign_buffer_slot(self, pool: StaticBufferPool, slot_idx: int):
        """Assign this module to a buffer slot in the pool.

        Also assigns static GPU buffers to each parameter offloader,
        which moves the parameter data to point to the GPU buffer.
        """
        self._buffer_pool = pool
        self._buffer_slot_idx = slot_idx

        # Assign static buffers to parameters
        # Use CPU storage shape/stride/dtype since param.data is now empty
        for name, offloader in self._param_offloaders.items():
            cpu_storage = offloader._cpu_storage
            assert cpu_storage is not None, "CPU storage not initialized"
            buffer = pool.get_buffer(
                name=name,
                shape=tuple(cpu_storage.shape),
                stride=tuple(cpu_storage.stride()),
                dtype=cpu_storage.dtype,
                slot_idx=slot_idx,
            )
            offloader.assign_static_buffer(buffer)

    def start_onload_to_static(self):
        """Start async copy from CPU storage to GPU buffer.

        Uses event-based forking to join copy_stream to CUDA graph capture.
        This ensures H2D copies are properly captured when recording a graph.

        IMPORTANT: We must wait for the compute stream before copying, because
        the previous layer's forward may still be using the buffer (GPU ops are
        async). Without this sync, we could overwrite the buffer while it's
        being read.
        """
        assert self._buffer_pool is not None, "Buffer pool not assigned"

        # Track if this prefetch is being captured (for _wait_for_layer logic)
        self._prefetch_in_capture = torch.cuda.is_current_stream_capturing()

        # Fork: record event on compute stream, copy_stream waits on it
        # This joins copy_stream to any active CUDA graph capture
        fork_event = torch.cuda.Event()
        torch.cuda.current_stream().record_event(fork_event)
        self.copy_stream.wait_event(fork_event)

        with torch.cuda.stream(self.copy_stream):
            for name, offloader in self._param_offloaders.items():
                cpu_storage = offloader._cpu_storage
                gpu_buffer = offloader._gpu_buffer
                assert cpu_storage is not None, "CPU storage not initialized"
                assert gpu_buffer is not None, "GPU buffer not assigned"
                # Async copy from pinned CPU storage to GPU buffer
                gpu_buffer.copy_(cpu_storage, non_blocking=True)

        # Record completion event for _wait_for_layer to use
        self._copy_done_event.record(self.copy_stream)


class _BaseParamOffloader(ABC):
    """Base class for parameter offloading strategies."""

    # CPU storage for offloaded parameters (set by subclasses)
    _cpu_storage: torch.Tensor | None
    # GPU buffer reference (set by subclasses when using static buffers)
    _gpu_buffer: torch.Tensor | None

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
        self._cpu_storage = None
        self._gpu_buffer = None

    @property
    def _param(self) -> nn.Parameter:
        """Get the parameter being offloaded."""
        return getattr(self._module, self._param_name)

    def post_init(self):
        """Initialize offloading (move parameter to storage)."""
        return

    @abstractmethod
    def sync_cpu_storage(self) -> None:
        """Sync CPU storage with current param.data.

        Called after process_weights_after_loading to update _cpu_storage
        with the final processed weights.
        """
        pass

    @abstractmethod
    def assign_static_buffer(self, gpu_buffer: torch.Tensor) -> None:
        """Point parameter data to GPU static buffer."""
        pass


class _CpuParamOffloader(_BaseParamOffloader):
    """Offload parameter to pinned CPU memory.

    Uses GPU static buffers as the actual parameter, with CPU storage
    kept separately. This ensures torch.compile sees GPU tensors at trace time.

    The offloading happens in two phases:
    1. __init__() - copies GPU data to CPU, frees GPU memory immediately
    2. assign_static_buffer() - points param.data to GPU static buffer
    """

    def __init__(self, module: nn.Module, param_name: str):
        super().__init__(module, param_name)
        self._cpu_storage: torch.Tensor | None = None
        self._gpu_buffer: torch.Tensor | None = None  # Store reference to GPU buffer

        # Offload to CPU immediately to free GPU memory during model loading
        self._offload_to_cpu_internal()

    def _offload_to_cpu_internal(self):
        """Copy parameter data to pinned CPU storage and free GPU memory.

        This replaces param.data with CPU storage, allowing weight loading
        to continue writing to CPU memory. GPU memory is freed when the
        original GPU tensor is garbage collected.
        """
        param = self._param
        pin_memory = is_pin_memory_available()

        # Create pinned CPU storage and copy current GPU data
        self._cpu_storage = torch.empty_strided(
            size=param.data.size(),
            stride=param.data.stride(),
            dtype=param.data.dtype,
            layout=param.data.layout,
            device="cpu",
            pin_memory=pin_memory,
        )
        self._cpu_storage.copy_(param.data)

        self.offloaded_bytes = (
            self._cpu_storage.numel() * self._cpu_storage.element_size()
        )

        # Point param.data to CPU storage - this allows weight loading to work
        # and frees GPU memory when the original GPU tensor is garbage collected
        param.data = self._cpu_storage

    def assign_static_buffer(self, gpu_buffer: torch.Tensor) -> None:
        """Point parameter data to GPU static buffer.

        This is called after weight loading AND process_weights_after_loading
        complete. At this point:
        - param.data may have been replaced by device_loading_context
          (which creates new CPU tensors after quantization processing)
        - We need to update _cpu_storage to point to current param.data
          so that prefetch copies the processed weights, not stale data
        - Then point param.data to the GPU buffer for torch.compile
        """
        assert self._cpu_storage is not None, (
            "_offload_to_cpu_internal() must be called before assign_static_buffer()"
        )

        # Get current parameter (may have been replaced by
        # process_weights_after_loading)
        param = self._param

        # Update _cpu_storage to current param.data. This is critical because:
        # 1. process_weights_after_loading may transform weights (quantization)
        # 2. device_loading_context creates NEW CPU tensors when moving back
        # 3. Our old _cpu_storage would have pre-processed or stale data
        if param.data.device.type == "cpu":
            # param.data is already on CPU - use it as our CPU storage
            self._cpu_storage = param.data
        else:
            # param.data is on GPU - copy to our CPU storage
            self._cpu_storage.copy_(param.data)

        # Store reference to GPU buffer for use in start_onload
        self._gpu_buffer = gpu_buffer

        # Point parameter to static GPU buffer - this is what torch.compile sees
        param.data = gpu_buffer

    def sync_cpu_storage(self) -> None:
        """Sync CPU storage with current param.data.

        Called after process_weights_after_loading to update _cpu_storage
        with the final processed weights. This is critical because:
        1. process_weights_after_loading may transform weights (quantization)
        2. device_loading_context creates NEW CPU tensors when moving back
        3. Our old _cpu_storage would have pre-processed or stale data
        """
        param = self._param

        if param.data.device.type == "cpu":
            # param.data is already on CPU - use it as our CPU storage
            self._cpu_storage = param.data
        else:
            # param.data is on GPU - copy to existing CPU storage
            assert self._cpu_storage is not None
            self._cpu_storage.copy_(param.data)

    def post_init(self):
        """No-op: offloading done in offload_to_cpu/assign_static_buffer."""
        pass
