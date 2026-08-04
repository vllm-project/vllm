# SPDX-License-Identifier: Apache-2.0
"""Abstract base class for platform cache contexts.

Defines the common interface shared by :class:`GPUCacheContext` and
:class:`CPUCacheContext`.  Concrete subclasses provide
device-specific implementations of stream / buffer / copy primitives
while the base class owns layout-agnostic helpers (shape calculation,
status reporting, block-ID staging).
"""

# Future
from __future__ import annotations

# Standard
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar
import array

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.utils import (
    get_attention_backend,
    get_concrete_engine_kv_shape_from_shape_desc,
    get_engine_kv_shape_description,
    is_mla,
)
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager

if TYPE_CHECKING:
    # First Party
    import lmcache.c_ops as lmc_ops


class BaseCacheContext(ABC):
    """Abstract base for GPU and CPU cache contexts.

    Subclasses call :meth:`__init__` after computing the common
    layout parameters and before setting up device-specific state.
    All keyword arguments are required so the contract is explicit.

    Concrete subclasses MUST set :attr:`device_type` to the
    ``torch.device.type`` string they handle (``"cuda"``, ``"cpu"``,
    ...). The platform-agnostic :func:`create_cache_context` factory
    uses this attribute (via the platform registry) to pick the right
    subclass without any ``isinstance`` / ``if-elif`` chain.
    """

    #: ``torch.device.type`` string the subclass handles. Concrete
    #: subclasses MUST override this.
    device_type: ClassVar[str] = ""

    def __init__(
        self,
        *,
        kv_caches: list[torch.Tensor],
        device: torch.device,
        num_layers: int,
        kv_layer_groups_manager: KVLayerGroupsManager,
        block_ids_buffer: torch.Tensor,
        lmcache_tokens_per_chunk: int,
    ) -> None:
        self.kv_caches_ = kv_caches
        self.device_ = device
        self.num_layers_ = num_layers
        self.kv_layer_groups_manager_ = kv_layer_groups_manager
        self.block_ids_buffer_ = block_ids_buffer
        self.lmcache_tokens_per_chunk = lmcache_tokens_per_chunk

    # ------------------------------------------------------------------
    # Abstract -- subclasses MUST implement
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def stream(self) -> Any:
        """Returns the device-specific stream for async operations."""
        ...

    @property
    @abstractmethod
    def cupy_stream(self) -> Any:
        """Returns the cupy ExternalStream wrapping *stream*."""
        ...

    @property
    @abstractmethod
    def max_batch_size(self) -> int:
        """Returns the maximum number of concurrent batches."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release device-specific resources (GDS staging buffers, etc.)."""
        ...

    @abstractmethod
    def get_kernel_group_kv_pointers(self, kernel_group_idx: int) -> torch.Tensor:
        """Returns the KV-cache pointer tensor for *kernel_group_idx*."""
        ...

    @abstractmethod
    def get_temp_kernel_group_buffer(
        self, batch_idx: int, kernel_group_idx: int
    ) -> torch.Tensor:
        """Returns a typed temp-buffer view for a (batch, kernel-group)
        pair."""
        ...

    @abstractmethod
    def get_temp_object_group_buffer(
        self, batch_idx: int, object_group_idx: int
    ) -> torch.Tensor:
        """Returns a flat uint8 temp-buffer view for a (batch, object-group)
        pair."""
        ...

    @abstractmethod
    def get_kernel_group_shape_dtype(
        self,
        num_tokens: int,
        kernel_group_idx: int,
    ) -> tuple[torch.Size, torch.dtype]:
        """Returns ``(shape, dtype)`` for *kernel_group_idx*."""
        ...

    @abstractmethod
    def cache_size_per_token(self) -> int:
        """Returns cache size per logical token in bytes (all groups)."""
        ...

    # ------------------------------------------------------------------
    # Concrete -- shared implementations
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """Returns the device where KV-cache tensors live."""
        return self.device_

    @property
    def kv_tensors(self) -> list[torch.Tensor]:
        """Returns the list of per-layer KV cache tensors."""
        return self.kv_caches_

    @property
    def num_layers(self) -> int:
        """Returns the number of layers in the model."""
        return self.num_layers_

    @property
    def num_blocks(self) -> int:
        """Returns the number of blocks in the KV cache.

        Sourced from the kernel groups (one shared block-id space), not a
        representative-format computation.
        """
        return self.kv_layer_groups_manager_.num_blocks

    @property
    def hidden_dim_sizes(self) -> list[int]:
        """Returns hidden dimension sizes per KV layer group."""
        return [
            group.hidden_dim_size
            for group in self.kv_layer_groups_manager.kernel_groups
        ]

    @property
    def kv_layer_groups_manager(self) -> KVLayerGroupsManager:
        """Returns the KV layer groups manager."""
        return self.kv_layer_groups_manager_

    def calculate_num_blocks(self, num_tokens: int, kernel_group_idx: int) -> int:
        """Calculate the number of blocks for *num_tokens* in a kernel
        group."""
        return self.kv_layer_groups_manager.calculate_num_blocks(
            kernel_group_idx, num_tokens
        )

    def get_shape_desc(self, group_idx: int) -> "lmc_ops.PageBufferShapeDesc":
        """Returns the PageBufferShapeDesc for *group_idx*."""
        return self.kv_layer_groups_manager.get_shape_desc(group_idx)

    def get_engine_kv_format(self, kernel_group_idx: int) -> "lmc_ops.EngineKVFormat":
        """Returns the Engine KV format of kernel *kernel_group_idx*.

        Raises:
            ValueError: If the group has no format (a bookkeeping group built by
                ``parse_kvcache_shape_spec`` should never reach the transfer
                path; detection-built groups always carry one).
        """
        groups = self.kv_layer_groups_manager.kernel_groups
        engine_kv_format = groups[kernel_group_idx].engine_kv_format
        if engine_kv_format is None:
            raise ValueError(
                f"kernel group {kernel_group_idx} has no engine_kv_format; a "
                "formatless bookkeeping group reached the transfer path"
            )
        return engine_kv_format

    def engine_kv_formats(self) -> list["lmc_ops.EngineKVFormat"]:
        """Returns the Engine KV format of each kernel group, in group order."""
        num_groups = len(self.kv_layer_groups_manager.kernel_groups)
        return [self.get_engine_kv_format(idx) for idx in range(num_groups)]

    def engine_kv_format_per_layer(self) -> list["lmc_ops.EngineKVFormat | None"]:
        """Returns each layer's Engine KV format, indexed by layer index.

        Formats differ across layers for a mixed-format model. ``None`` marks a
        layer in no kernel group (a cross-layer KV-sharing layer).
        """
        formats: list["lmc_ops.EngineKVFormat | None"] = [None] * len(self.kv_caches_)
        for kernel_group_idx, group in enumerate(
            self.kv_layer_groups_manager.kernel_groups
        ):
            fmt = self.get_engine_kv_format(kernel_group_idx)
            for layer_idx in group.layer_indices:
                formats[layer_idx] = fmt
        return formats

    def get_slots_per_chunk_in_sw(self, kernel_group_idx: int) -> int:
        """Returns the number of slots per lmcache chunk for D/H
        transfer."""
        return self.kv_layer_groups_manager.get_slots_per_chunk_in_sw(kernel_group_idx)

    def get_kv_buffer_shape(
        self, logical_num_tokens: int, group_idx: int = 0
    ) -> torch.Size:
        """Returns the KV buffer shape for *logical_num_tokens*."""
        group = self.kv_layer_groups_manager.kernel_groups[group_idx]
        compress_ratio = group.tokens_per_block // group.slots_per_block
        if logical_num_tokens % compress_ratio != 0:
            raise ValueError(
                "logical_num_tokens (%d) is not a multiple of "
                "compress_ratio (%d) for group %d"
                % (logical_num_tokens, compress_ratio, group_idx)
            )
        num_slots = logical_num_tokens // compress_ratio
        sd = group.shape_desc
        return torch.Size(
            (sd.kv_size, group.num_layers, num_slots, group.hidden_dim_size)
        )

    def stage_block_ids(
        self, block_ids_per_group: list[list[int]]
    ) -> list[torch.Tensor]:
        """Stage per-group block IDs into the shared staging buffer.

        Returns one non-overlapping view per LMCache group.
        """
        offsets = [0]
        flat: array.array = array.array("q")
        for view_block_ids in block_ids_per_group:
            flat.extend(view_block_ids)
            offsets.append(len(flat))

        total = offsets[-1]
        if total > self.block_ids_buffer_.shape[0]:
            raise ValueError(
                "block ID total %d exceeds the pre-allocated buffer "
                "size %d" % (total, self.block_ids_buffer_.shape[0])
            )
        if total:
            cpu_tensor = torch.frombuffer(flat, dtype=torch.long)
            self.block_ids_buffer_[:total].copy_(cpu_tensor, non_blocking=True)

        return [
            self.block_ids_buffer_[offsets[i] : offsets[i + 1]]
            for i in range(len(block_ids_per_group))
        ]

    # ------------------------------------------------------------------
    # Derived properties (pure helpers)
    # ------------------------------------------------------------------

    @property
    def concrete_engine_kv_shape(self) -> str:
        """Returns the engine KV shape with actual numeric values."""
        group = self.kv_layer_groups_manager.kernel_groups[0]
        return get_concrete_engine_kv_shape_from_shape_desc(
            group.shape_desc, group.engine_kv_format
        )

    # ------------------------------------------------------------------
    # Shared report_status
    # ------------------------------------------------------------------

    def _build_group_report_map(self) -> dict[int, int]:
        """Map each kernel-group index to its owning object-group index."""
        return {
            kg_idx: og_idx
            for og_idx, og in enumerate(self.kv_layer_groups_manager.object_groups)
            for kg_idx in og.kernel_group_indices
        }

    def _build_single_group_report(
        self,
        kernel_group_idx: int,
        group: Any,
        group_map: dict[int, int],
    ) -> dict:
        """Build a status dict for a single kernel group.

        Override this in subclasses to inject extra per-group fields
        without duplicating the whole :meth:`report_status` method.
        """
        engine_kv_format = self.get_engine_kv_format(kernel_group_idx)
        return {
            "kernel_group_idx": kernel_group_idx,
            "engine_group_idx": group.engine_group_idx,
            "object_group_idx": group_map.get(kernel_group_idx, 0),
            "num_layers": group.num_layers,
            "layer_indices": list(group.layer_indices),
            "tokens_per_block": group.tokens_per_block,
            "slots_per_block": group.slots_per_block,
            "dtype": str(group.dtype),
            "engine_kv_concrete_shape": (
                get_concrete_engine_kv_shape_from_shape_desc(
                    group.shape_desc, engine_kv_format
                )
            ),
            "is_mla": is_mla(engine_kv_format),
            "engine_kv_format": engine_kv_format.name,
            "engine_kv_shape": get_engine_kv_shape_description(engine_kv_format),
            "attention_backend": get_attention_backend(engine_kv_format),
        }

    def report_status(self) -> dict:
        """Return this context's KV cache layout metadata."""
        manager = self.kv_layer_groups_manager
        kernel_groups = manager.kernel_groups
        group_map = self._build_group_report_map()

        group_reports = [
            self._build_single_group_report(kernel_group_idx, group, group_map)
            for kernel_group_idx, group in enumerate(kernel_groups)
        ]

        return {
            "num_layers": self.num_layers,
            "num_blocks": self.num_blocks,
            "cache_size_per_token": self.cache_size_per_token(),
            "kernel_groups": group_reports,
        }
