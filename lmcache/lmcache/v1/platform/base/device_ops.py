# SPDX-License-Identifier: Apache-2.0
"""The unified per-device ops abstraction (the ``lmcache.c_ops`` surface).

:class:`DeviceOps` is a strategy base class whose every op is an instance
method delegating to :mod:`lmcache.v1.platform.torch_ops`.  Accelerator
subclasses override individual methods with native kernels via normal OO
polymorphism, or bulk-rebind via :meth:`DeviceOps.bind_native`.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    # Third Party
    import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform import ops_types, torch_ops
from lmcache.v1.platform.ops_types import (
    EngineKVFormat,
    PageBufferShapeDesc,
    TransferDirection,
    set_shape_desc_dtype,
)

logger = init_logger(__name__)


class DeviceOps:
    """Strategy base: per-device ops resolved via normal instance MRO.

    Every op is an instance method delegating to the torch baseline in
    :mod:`~lmcache.v1.platform.torch_ops`.  Accelerator subclasses either:

    - Override individual methods (e.g. MUSA overrides
      :meth:`multi_layer_block_kv_transfer`).
    - Call :meth:`bind_native` in :meth:`ensure_native` to bulk-rebind
      all ops the compiled extension exports.

    The ``lmcache.c_ops`` shim forwards attribute access to a resolved
    singleton instance so module-level call sites keep working.
    """

    device_type: ClassVar[str] = ""  # base is unregistered

    # ── Shared types (explicit for static analysis) ────────────────────
    TransferDirection = TransferDirection
    EngineKVFormat = EngineKVFormat
    GPUKVFormat = EngineKVFormat  # back-compat alias
    PageBufferShapeDesc = PageBufferShapeDesc
    set_shape_desc_dtype = staticmethod(set_shape_desc_dtype)

    def __init__(self) -> None:
        self._native_bound: bool = False

    # ── Lazy native binding ───────────────────────────────────────────

    def ensure_native(self) -> None:
        """Attempt to load and bind the compiled native extension.

        Subclasses override this to import their compiled module and call
        :meth:`bind_native`.  The base class is a no-op (pure torch).
        Guarded by ``_native_bound`` so it runs at most once per instance.
        """

    def bind_native(self, module: object) -> None:
        """Rebind ops and types from a compiled native module.

        Walks the native module's public symbols.  For each callable or
        type, binds it on this instance — replacing the torch fallback if
        one exists, or adding a native-only symbol that has no pure-Python
        equivalent.
        """
        bound_ops = 0
        bound_types = 0
        for name in dir(module):
            if name.startswith("_"):
                continue
            native_sym = getattr(module, name, None)
            if native_sym is None:
                continue
            if isinstance(native_sym, type):
                setattr(self, name, native_sym)
                bound_types += 1
            elif callable(native_sym):
                setattr(self, name, native_sym)
                bound_ops += 1
        # GPUKVFormat alias → EngineKVFormat on native side
        ekf = getattr(module, "EngineKVFormat", None)
        if ekf is not None:
            self.GPUKVFormat = ekf
        logger.debug(
            "bind_native: %s <- %s (%d ops, %d types)",
            type(self).__name__,
            getattr(module, "__name__", repr(module)),
            bound_ops,
            bound_types,
        )

    # ── Ops: memory alloc / free ─────────────────────────────────────

    def alloc_hugepage_pinned_numa_ptr(self, size, numa_id=0):
        return torch_ops.alloc_hugepage_pinned_numa_ptr(size, numa_id)

    def alloc_hugepage_pinned_ptr(self, size, device_id=0):
        return torch_ops.alloc_hugepage_pinned_ptr(size, device_id)

    def alloc_numa_ptr(self, size, numa_id=0):
        return torch_ops.alloc_numa_ptr(size, numa_id)

    def alloc_pinned_numa_ptr(self, size, numa_id=0):
        return torch_ops.alloc_pinned_numa_ptr(size, numa_id)

    def alloc_pinned_ptr(self, size, device_id=0):
        return torch_ops.alloc_pinned_ptr(size, device_id)

    def alloc_shm_pinned_ptr(self, size, shm_name=""):
        return torch_ops.alloc_shm_pinned_ptr(size, shm_name)

    def free_hugepage_pinned_numa_ptr(self, ptr, size=0):
        return torch_ops.free_hugepage_pinned_numa_ptr(ptr, size)

    def free_hugepage_pinned_ptr(self, ptr, size=0):
        return torch_ops.free_hugepage_pinned_ptr(ptr, size)

    def free_numa_ptr(self, ptr, size=None):
        return torch_ops.free_numa_ptr(ptr, size)

    def free_pinned_numa_ptr(self, ptr, size=None):
        return torch_ops.free_pinned_numa_ptr(ptr, size)

    def free_pinned_ptr(self, ptr):
        return torch_ops.free_pinned_ptr(ptr)

    def free_shm_pinned_ptr(self, ptr, size=0, shm_name=""):
        return torch_ops.free_shm_pinned_ptr(ptr, size, shm_name)

    # ── Ops: KV transfer ─────────────────────────────────────────────

    def multi_layer_block_kv_transfer(
        self,
        paged_buffer_ptrs_tensor: "torch.Tensor | list",
        lmcache_objects_ptrs: "list[int] | list[torch.Tensor]",
        block_ids: "torch.Tensor | list[int]",
        device: "torch.device | str",
        direction: ops_types.TransferDirection,
        shape_desc: ops_types.PageBufferShapeDesc,
        lmcache_chunk_size: int,
        engine_kv_format: ops_types.EngineKVFormat,
        skip_prefix_n_blocks: int,
    ) -> None:
        return torch_ops.multi_layer_block_kv_transfer(
            paged_buffer_ptrs_tensor,
            lmcache_objects_ptrs,
            block_ids,
            device,
            direction,
            shape_desc,
            lmcache_chunk_size,
            engine_kv_format,
            skip_prefix_n_blocks,
        )

    def multi_layer_kv_transfer(self, *args, **kwargs):
        return torch_ops.multi_layer_kv_transfer(*args, **kwargs)

    def multi_layer_kv_transfer_unilateral(self, *args, **kwargs):
        return torch_ops.multi_layer_kv_transfer_unilateral(*args, **kwargs)

    def single_layer_kv_transfer(self, *args, **kwargs):
        return torch_ops.single_layer_kv_transfer(*args, **kwargs)

    def single_layer_kv_transfer_sgl(self, *args, **kwargs):
        return torch_ops.single_layer_kv_transfer_sgl(*args, **kwargs)

    # ── Ops: KV reshape ──────────────────────────────────────────────

    def load_and_reshape_flash(self, *args, **kwargs):
        return torch_ops.load_and_reshape_flash(*args, **kwargs)

    def reshape_and_cache_back_flash(self, *args, **kwargs):
        return torch_ops.reshape_and_cache_back_flash(*args, **kwargs)

    # ── Ops: codec ───────────────────────────────────────────────────

    def calculate_cdf(self, input_tensor, num_bins):
        return torch_ops.calculate_cdf(input_tensor, num_bins)

    def decode_fast_new(self, cdf, bytestreams, lengths, output):
        return torch_ops.decode_fast_new(cdf, bytestreams, lengths, output)

    def decode_fast_prefsum(self, cdf, bytestreams, lengths_prefsum, output):
        return torch_ops.decode_fast_prefsum(cdf, bytestreams, lengths_prefsum, output)

    def encode_fast_new(self, cdf, input_sym, output_buffer, output_lengths):
        return torch_ops.encode_fast_new(cdf, input_sym, output_buffer, output_lengths)

    # ── Ops: format query ────────────────────────────────────────────

    def is_cross_layer(self, engine_kv_format):
        return torch_ops.is_cross_layer(engine_kv_format)

    def is_kv_list(self, engine_kv_format):
        return torch_ops.is_kv_list(engine_kv_format)

    def is_layer_list(self, engine_kv_format):
        return torch_ops.is_layer_list(engine_kv_format)

    def is_mla(self, engine_kv_format):
        return torch_ops.is_mla(engine_kv_format)

    # ── Ops: async / event recording ─────────────────────────────────

    def drain_recorded_completions(self):
        return torch_ops.drain_recorded_completions()

    def drain_recorded_events(self):
        return torch_ops.drain_recorded_events()

    def record_completion_on_stream(self, *args, **kwargs):
        return torch_ops.record_completion_on_stream(*args, **kwargs)

    def record_event_on_stream(self, *args, **kwargs):
        return torch_ops.record_event_on_stream(*args, **kwargs)

    # ── Ops: misc ────────────────────────────────────────────────────

    def batched_memcpy(self, src_ptrs, dst_ptrs, sizes):
        return torch_ops.batched_memcpy(src_ptrs, dst_ptrs, sizes)

    def get_gpu_pci_bus_id(self, device_id=0):
        return torch_ops.get_gpu_pci_bus_id(device_id)

    def lmcache_memcpy_async(self, *args, **kwargs):
        return torch_ops.lmcache_memcpy_async(*args, **kwargs)

    def rotary_embedding_k_fused(self, *args, **kwargs):
        return torch_ops.rotary_embedding_k_fused(*args, **kwargs)
