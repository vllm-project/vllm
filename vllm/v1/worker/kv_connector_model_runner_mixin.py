# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Define KV connector functionality mixin for model runners.
"""

import copy
import math
from collections import defaultdict
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_shutdown,
    get_kv_transfer_group,
    has_kv_transfer_group,
)
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    Chunk,
    KVCacheDataReference,
    KVCacheTensorReference,
)
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.logger import init_logger
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    KVConnectorOutput,
    ModelRunnerOutput,
)
from vllm.v1.worker.utils import AttentionGroup

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVCacheTopology
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


@dataclass
class CrossLayerGroup:
    """A contiguous int8 buffer shared by layers with the same page size.

    Layers are packed so that all layers' data for one block is
    contiguous, enabling efficient bulk KV transfers.

    Tensor shape is either:
      - ordered (HND): (num_blocks, *prefix_dims, num_layers,
        remaining_bytes)
      - default (NHD / Mamba): (num_blocks, num_layers, page_size_bytes)
    """

    tensor: torch.Tensor
    layer_names: list[str]
    page_size_bytes: int
    topologies: "list[KVCacheTopology] | None" = None


# Defined as a kv connector functionality mixin for ModelRunner (GPU, TPU)
class KVConnectorModelRunnerMixin:
    @staticmethod
    def ensure_kv_transfer_shutdown() -> None:
        # has_kv_transfer_group can be None during interpreter shutdown.
        if has_kv_transfer_group and has_kv_transfer_group():  # type: ignore[truthy-function]
            ensure_kv_transfer_shutdown()

    @staticmethod
    def kv_connector_no_forward(
        scheduler_output: "SchedulerOutput", vllm_config: VllmConfig
    ) -> ModelRunnerOutput:
        # KV send/recv even if no work to do.
        with (
            set_forward_context(None, vllm_config),
            KVConnectorModelRunnerMixin._get_kv_connector_output(
                scheduler_output, wait_for_save=False
            ) as kv_connector_output,
        ):
            pass

        if kv_connector_output.is_empty():
            return EMPTY_MODEL_RUNNER_OUTPUT

        output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.kv_connector_output = kv_connector_output
        return output

    @staticmethod
    def maybe_get_kv_connector_output(
        scheduler_output: "SchedulerOutput",
        defer_finalize: bool = False,
    ) -> AbstractContextManager[KVConnectorOutput | None]:
        return (
            KVConnectorModelRunnerMixin._get_kv_connector_output(
                scheduler_output, defer_finalize=defer_finalize
            )
            if has_kv_transfer_group()
            else nullcontext()
        )

    @staticmethod
    def finalize_kv_connector() -> None:
        """Finalize the KV connector: wait_for_save and clear metadata.

        Call after draft model forward when defer_finalize=True was used.
        """
        if has_kv_transfer_group():
            kv_connector = get_kv_transfer_group()
            kv_connector.wait_for_save()
            kv_connector.clear_connector_metadata()

    # This context manager must be used within an active forward context.
    # It encapsulates the entire KV connector lifecycle within execute_model
    @staticmethod
    @contextmanager
    def _get_kv_connector_output(
        scheduler_output: "SchedulerOutput",
        wait_for_save: bool = True,
        defer_finalize: bool = False,
    ) -> Generator[KVConnectorOutput, None, None]:
        output = KVConnectorOutput()

        # Update KVConnector with the KVConnector metadata forward().
        kv_connector = get_kv_transfer_group()
        assert isinstance(kv_connector, KVConnectorBase)
        assert scheduler_output.kv_connector_metadata is not None
        kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)

        # Background KV cache transfers happen here.
        # These transfers are designed to be async and the requests
        # involved may be disjoint from the running requests.
        # Do this here to save a collective_rpc.
        kv_connector.start_load_kv(get_forward_context())
        try:
            yield output
        finally:
            if wait_for_save and not defer_finalize:
                kv_connector.wait_for_save()

            output.finished_sending, output.finished_recving = (
                kv_connector.get_finished(scheduler_output.finished_req_ids)
            )
            output.invalid_block_ids = kv_connector.get_block_ids_with_load_errors()

            output.kv_connector_stats = kv_connector.get_kv_connector_stats()
            output.kv_cache_events = kv_connector.get_kv_connector_kv_cache_events()

            if not defer_finalize:
                kv_connector.clear_connector_metadata()

    @staticmethod
    def use_uniform_kv_cache(
        attn_groups: list[list[AttentionGroup]],
        cache_dtype: CacheDType,
    ) -> bool:
        """
        Check if we should use a uniform cross-layer KV layout.

        When enabled, layers sharing the same page geometry are packed into
        a single contiguous tensor.

        Two paths are supported:
        - Hybrid path: connector overrides register_hybrid_kv_caches.
          Supports multiple KV cache groups with AttentionSpec/MambaSpec.
        - Legacy path: connector sets prefer_cross_layer_blocks = True.
          Restricted to a single group of uniform AttentionSpec layers
          whose backend supports a layers dimension in the stride order.
        """

        if not has_kv_transfer_group():
            return False

        if not attn_groups:
            return False

        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorBase_V1,
        )

        connector = get_kv_transfer_group()
        # Check if the connector supports hybrid KV caching, if not fall back to legacy.
        has_hybrid = (
            type(connector).register_hybrid_kv_caches
            is not KVConnectorBase_V1.register_hybrid_kv_caches
        )

        if has_hybrid:
            # Multi-group path: all specs must be AttentionSpec or MambaSpec.
            for subgroups in attn_groups:
                for attn_group in subgroups:
                    if not isinstance(
                        attn_group.kv_cache_spec, (AttentionSpec, MambaSpec)
                    ):
                        logger.warning(
                            "Uniform KV cache layout not supported for "
                            "spec type %s, falling back to per-layer "
                            "allocation",
                            type(attn_group.kv_cache_spec).__name__,
                        )
                        return False
            return True

        if connector.prefer_cross_layer_blocks:
            # Legacy single-group path: one group, AttentionSpec only,
            # and the backend must support a layers dimension.
            if len(attn_groups) != 1 or len(attn_groups[0]) != 1:
                return False

            attn_group = attn_groups[0][0]
            kv_cache_spec = attn_group.kv_cache_spec
            if not isinstance(kv_cache_spec, AttentionSpec):
                return False

            attn_backend = attn_group.backend
            kv_cache_shape = attn_backend.get_kv_cache_shape(
                1234,
                kv_cache_spec.block_size,
                kv_cache_spec.num_kv_heads,
                kv_cache_spec.head_size,
                cache_dtype_str=cache_dtype,
            )

            try:
                kv_cache_stride_order = attn_backend.get_kv_cache_stride_order(
                    include_num_layers_dimension=True
                )
            except (AttributeError, NotImplementedError):
                return False

            # Check that the attention backend includes a layers dimension.
            return len(kv_cache_stride_order) == len(kv_cache_shape) + 1

        return False

    @staticmethod
    def _cross_layer_group_key(
        spec: KVCacheSpec,
        backend: type[AttentionBackend],
        cache_dtype: CacheDType,
    ) -> tuple:
        """Compute the grouping key that determines which layers share
        a cross-layer tensor.

        Examines the backend's stride order (with a prepended layers
        dimension) to classify the layer into one of:
          - ("ordered", prefix_sizes, remaining_bytes) -- HND layout
          - ("default", page_size_bytes) -- NHD layout
          - ("mamba", page_size_bytes) -- Mamba state layers
          - ("isolated",) -- unsupported backend, no sharing
        """
        if isinstance(spec, MambaSpec):
            return ("mamba", spec.page_size_bytes)
        if not isinstance(spec, AttentionSpec):
            return ("isolated",)

        try:
            stride_order_with_layers = backend.get_kv_cache_stride_order(
                include_num_layers_dimension=True,
            )
            # Use sentinel values to locate the heads and blocks
            # dimensions in the base (without-layers) logical shape.
            _SENTINEL_BLOCKS, _SENTINEL_HEADS = 1234, 5678
            base_logical_shape = backend.get_kv_cache_shape(
                _SENTINEL_BLOCKS,
                spec.block_size,
                _SENTINEL_HEADS,
                spec.head_size,
                cache_dtype_str=cache_dtype,
            )
            heads_base_idx = base_logical_shape.index(_SENTINEL_HEADS)
        except (AttributeError, NotImplementedError, ValueError, AssertionError):
            return ("isolated",)

        # Build a mapping from logical dimension index (in the
        # with-layers shape) to physical position.  Logical index 0 is
        # the prepended layers dimension; every base dim shifts by +1.
        logical_to_physical = {
            dim: pos for pos, dim in enumerate(stride_order_with_layers)
        }
        layers_phys_pos = logical_to_physical[0]
        heads_phys_pos = logical_to_physical[heads_base_idx + 1]

        blocks_base_idx = base_logical_shape.index(_SENTINEL_BLOCKS)
        blocks_phys_pos = logical_to_physical[blocks_base_idx + 1]
        if blocks_phys_pos != 0 or layers_phys_pos <= blocks_phys_pos:
            return ("isolated",)

        # Heads after layers → no useful prefix to extract.
        if heads_phys_pos >= layers_phys_pos:
            return ("default", spec.page_size_bytes)

        # Heads come before layers (HND) — figure out the dimension
        # sizes between blocks and layers so we can replicate that
        # prefix in the cross-layer tensor shape.
        real_base_shape = backend.get_kv_cache_shape(
            1,  # single block
            spec.block_size,
            spec.num_kv_heads,
            spec.head_size,
            cache_dtype_str=cache_dtype,
        )
        # Prepend a dummy layers=1 to align indices with stride_order.
        real_shape_with_layers = (1, *real_base_shape)
        prefix_sizes = tuple(
            real_shape_with_layers[stride_order_with_layers[i]]
            for i in range(1, layers_phys_pos)
        )
        remaining_bytes = spec.page_size_bytes // (
            math.prod(prefix_sizes) if prefix_sizes else 1
        )
        return ("ordered", prefix_sizes, remaining_bytes)

    @staticmethod
    def _create_attention_layer_view(
        buffer: torch.Tensor,
        layer_idx: int,
        num_layers: int,
        num_blocks: int,
        attn_spec: AttentionSpec,
        backend: type[AttentionBackend],
        kernel_block_size: int,
        cache_dtype: CacheDType,
    ) -> torch.Tensor:
        """Create one attention layer's KV cache view from the shared
        cross-layer int8 buffer.

        Reinterprets the buffer as the layer's dtype, reshapes to the
        backend's physical layout (with layers dimension), selects
        the requested layer, and permutes back to the logical shape
        the kernel expects.  All operations are zero-copy.
        """
        # The spec block size may be a multiple of the kernel block
        # size.  Convert to kernel-level block counts.
        kernel_blocks_per_spec_block = attn_spec.block_size // kernel_block_size
        kernel_num_blocks = num_blocks * kernel_blocks_per_spec_block

        base_logical_shape = backend.get_kv_cache_shape(
            kernel_num_blocks,
            kernel_block_size,
            attn_spec.num_kv_heads,
            attn_spec.head_size,
            cache_dtype_str=cache_dtype,
        )

        stride_order_with_layers = backend.get_kv_cache_stride_order(
            include_num_layers_dimension=True,
        )

        # Build the physical shape by permuting (num_layers, *base)
        # according to the stride order.
        logical_shape_with_layers = (num_layers, *base_logical_shape)
        physical_shape_with_layers = tuple(
            logical_shape_with_layers[stride_order_with_layers[i]]
            for i in range(len(logical_shape_with_layers))
        )

        # Reinterpret raw bytes as the layer's dtype, then reshape to
        # the physical layout.
        typed = buffer.view(attn_spec.dtype).view(*physical_shape_with_layers)

        # Build the logical-to-physical dimension mapping.
        logical_to_physical = {
            dim: pos for pos, dim in enumerate(stride_order_with_layers)
        }
        layers_phys_pos = logical_to_physical[0]

        # Select the requested layer (removes the layers dimension).
        layer_slice = typed.select(layers_phys_pos, layer_idx)

        # Permute back to the base logical order the kernel expects.
        # After select() removes the layers dim, physical positions
        # above it shift down by 1.
        inverse_perm = tuple(
            logical_to_physical[k + 1]
            - (1 if logical_to_physical[k + 1] > layers_phys_pos else 0)
            for k in range(len(base_logical_shape))
        )
        return layer_slice.permute(*inverse_perm)

    @staticmethod
    def _create_mamba_layer_views(
        buffer: torch.Tensor,
        layer_idx: int,
        num_layers: int,
        mamba_spec: MambaSpec,
        num_blocks: int,
    ) -> list[torch.Tensor]:
        """Create views for one Mamba layer's state tensors from the shared
        cross-layer buffer.

        Each state tensor (conv, SSM, etc.) gets its own
        ``torch.as_strided`` view so that per-block data across all
        layers stays contiguous for efficient transfers.  Returns one
        tensor per entry in ``mamba_spec.shapes``.
        """
        page_bytes = mamba_spec.page_size_bytes
        state_tensors: list[torch.Tensor] = []
        offset_bytes = layer_idx * page_bytes

        for state_shape, state_dtype in zip(mamba_spec.shapes, mamba_spec.dtypes):
            element_size = torch.empty((), dtype=state_dtype).element_size()
            elements_per_page = page_bytes // element_size
            state_elements = math.prod(state_shape)

            target_shape = (num_blocks, *state_shape)

            # Compute row-major (C-contiguous) strides for the state
            # dimensions, then prepend the block stride which skips
            # over all layers' pages to reach the next block.
            inner_strides = []
            acc = 1
            for dim_size in reversed(state_shape):
                inner_strides.append(acc)
                acc *= dim_size
            inner_strides.reverse()
            block_stride = num_layers * elements_per_page
            target_stride = (block_stride, *inner_strides)

            assert offset_bytes % element_size == 0
            flat = buffer.view(torch.int8).view(state_dtype)
            state_tensors.append(
                torch.as_strided(
                    flat,
                    size=target_shape,
                    stride=target_stride,
                    storage_offset=offset_bytes // element_size,
                )
            )
            offset_bytes += state_elements * element_size

        return state_tensors

    @staticmethod
    def allocate_uniform_kv_caches(
        kv_cache_config: KVCacheConfig,
        attn_groups: list[list[AttentionGroup]],
        cache_dtype: CacheDType,
        device: torch.device,
        kernel_block_sizes: list[int],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, type[AttentionBackend]]:
        """Allocate a cross-layer KV cache for the legacy single-group path.

        All layers must share the same AttentionSpec and backend.
        This function assumes ``use_uniform_kv_cache`` returned True.

        Returns:
            A tuple (kv_caches, cross_layers_kv_cache, attn_backend).
        """
        attn_group = attn_groups[0][0]
        kv_cache_spec = attn_group.kv_cache_spec
        assert isinstance(kv_cache_spec, AttentionSpec)

        tensor_sizes = set(
            kv_cache_tensor.size for kv_cache_tensor in kv_cache_config.kv_cache_tensors
        )
        assert len(tensor_sizes) == 1
        tensor_size = tensor_sizes.pop()

        page_size = kv_cache_spec.page_size_bytes
        assert tensor_size % page_size == 0
        num_blocks = tensor_size // page_size
        num_layers = len(kv_cache_config.kv_cache_tensors)
        total_size = tensor_size * num_layers

        assert len(kernel_block_sizes) == 1
        kernel_block_size = kernel_block_sizes[0]
        num_blocks_per_kv_block = kv_cache_spec.block_size // kernel_block_size
        kernel_num_blocks = num_blocks * num_blocks_per_kv_block

        attn_backend = attn_group.backend
        kv_cache_shape = attn_backend.get_kv_cache_shape(
            kernel_num_blocks,
            kernel_block_size,
            kv_cache_spec.num_kv_heads,
            kv_cache_spec.head_size,
            cache_dtype_str=cache_dtype,
        )

        # prepend a num_layers dimension into the shape
        kv_cache_shape = (num_layers,) + kv_cache_shape

        try:
            kv_cache_stride_order = attn_backend.get_kv_cache_stride_order(
                include_num_layers_dimension=True
            )
            assert len(kv_cache_stride_order) == len(kv_cache_shape)
        except (AttributeError, NotImplementedError):
            # Fallback: prepend layers dim to the base stride order.
            base_order = attn_backend.get_kv_cache_stride_order()
            kv_cache_stride_order = (0,) + tuple(x + 1 for x in base_order)

        kv_cache_shape = tuple(kv_cache_shape[i] for i in kv_cache_stride_order)

        logger.info("Allocating a cross layer KV cache of shape %s", kv_cache_shape)

        # allocate one contiguous buffer for all layers
        cross_layers_kv_cache = (
            torch.zeros(total_size, dtype=torch.int8, device=device)
            .view(kv_cache_spec.dtype)
            .view(kv_cache_shape)
        )

        # Maintain original KV shape view.
        inv_order = [
            kv_cache_stride_order.index(i) for i in range(len(kv_cache_stride_order))
        ]
        permuted_kv_cache = cross_layers_kv_cache.permute(*inv_order)

        kv_caches: dict[str, torch.Tensor] = {}
        for i, kv_cache_tensor in enumerate(kv_cache_config.kv_cache_tensors):
            tensor = permuted_kv_cache[i]
            for layer_name in kv_cache_tensor.shared_by:
                kv_caches[layer_name] = tensor

        return kv_caches, cross_layers_kv_cache, attn_backend

    @staticmethod
    def allocate_hybrid_kv_caches(
        kv_cache_config: KVCacheConfig,
        attn_groups: list[list[AttentionGroup]],
        cache_dtype: CacheDType,
        device: torch.device,
        kernel_block_sizes: list[int],
    ) -> tuple[
        dict[str, torch.Tensor | list[torch.Tensor]],
        list[CrossLayerGroup],
    ]:
        """Allocate cross-layer KV caches for hybrid (multi-group) models.

        Layers are classified by ``_cross_layer_group_key`` and packed
        into shared buffers.  One int8 allocation per group, with
        per-layer views carved out via ``_create_attention_layer_view``
        or ``_create_mamba_layer_views``.

        This function assumes ``use_uniform_kv_cache`` returned True.

        Args:
            kv_cache_config (KVCacheConfig): cache config from the
                scheduler.
            attn_groups (list[list[AttentionGroup]]): two-level list
                indexed by [kv_cache_group][attn_backend].
            cache_dtype (CacheDType): the KV cache dtype string.
            device (torch.device): device to allocate on.
            kernel_block_sizes (list[int]): per-group kernel block
                sizes, indexed by ``kv_cache_group_id``.

        Returns:
            A tuple (kv_caches, cross_layer_groups) where kv_caches
            maps layer names to view tensors and cross_layer_groups
            holds the shared buffers with topology metadata.
        """
        # -----------------------------------------------------------------
        # Phase 1: Build a flat lookup from layer name → (spec, backend, gid)
        # -----------------------------------------------------------------
        layer_info: dict[str, tuple[KVCacheSpec, type[AttentionBackend], int]] = {}
        for subgroups in attn_groups:
            for attn_group in subgroups:
                for name in attn_group.layer_names:
                    layer_info[name] = (
                        attn_group.kv_cache_spec,
                        attn_group.backend,
                        attn_group.kv_cache_group_id,
                    )

        # -----------------------------------------------------------------
        # Phase 2: Group KVCacheTensors by their cross-layer group key
        # -----------------------------------------------------------------
        grouped: dict[tuple, list[tuple[int, Any]]] = defaultdict(list)
        for tensor_idx, kv_tensor in enumerate(kv_cache_config.kv_cache_tensors):
            spec, backend, _ = layer_info[kv_tensor.shared_by[0]]
            key = KVConnectorModelRunnerMixin._cross_layer_group_key(
                spec,
                backend,
                cache_dtype,
            )

            # Validate: all layers sharing this tensor must agree on
            # tensor size and group key.
            for name in kv_tensor.shared_by:
                layer_spec, layer_backend, _ = layer_info[name]
                assert (
                    layer_spec.page_size_bytes * kv_cache_config.num_blocks
                    == kv_tensor.size
                ), (
                    f"Layer {name}: expected tensor size "
                    f"{layer_spec.page_size_bytes * kv_cache_config.num_blocks}, "
                    f"got {kv_tensor.size}"
                )
                other_key = KVConnectorModelRunnerMixin._cross_layer_group_key(
                    layer_spec, layer_backend, cache_dtype
                )
                assert other_key == key, (
                    f"Layers sharing tensor disagree on group key: "
                    f"{kv_tensor.shared_by[0]} -> {key}, "
                    f"{name} -> {other_key}"
                )

            # Isolated layers must not share with each other, so give
            # each its own unique key.
            if key == ("isolated",):
                key = ("isolated", tensor_idx)

            grouped[key].append((tensor_idx, kv_tensor))

        # -----------------------------------------------------------------
        # Phase 3: Allocate one buffer per group and create per-layer views
        # -----------------------------------------------------------------
        kv_caches: dict[str, torch.Tensor | list[torch.Tensor]] = {}
        cross_layer_groups: list[CrossLayerGroup] = []

        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVCacheTopology,
        )

        for group_key, members in grouped.items():
            num_group_layers = len(members)

            # All tensors in a group must have the same byte size.
            first_tensor_size = members[0][1].size
            assert all(m[1].size == first_tensor_size for m in members), (
                "All KVCacheTensors in a cross-layer group must have the same size"
            )

            # Use the first member as the representative for spec/backend
            # lookups (all members in the group have matching geometry).
            representative_name = members[0][1].shared_by[0]
            representative_spec, _, _ = layer_info[representative_name]
            page_size = representative_spec.page_size_bytes

            assert first_tensor_size % page_size == 0
            num_blocks = first_tensor_size // page_size

            # Single contiguous allocation for the entire group.
            total_bytes = first_tensor_size * num_group_layers
            buffer = torch.zeros(total_bytes, dtype=torch.int8, device=device)

            # Shape the buffer according to the group type.
            if group_key[0] == "ordered":
                prefix_sizes = group_key[1]
                remaining_bytes = group_key[2]
                cross_layer_tensor = buffer.view(
                    num_blocks,
                    *prefix_sizes,
                    num_group_layers,
                    remaining_bytes,
                )
                layers_dim_idx = 1 + len(prefix_sizes)

                # Probe the backend to identify which prefix dimensions
                # correspond to heads and block_size for the topology.
                _, representative_backend, _ = layer_info[representative_name]
                _SENTINEL_B, _SENTINEL_H, _SENTINEL_BS = 1234, 5678, 9876
                assert isinstance(representative_spec, AttentionSpec)
                probe_shape = representative_backend.get_kv_cache_shape(
                    _SENTINEL_B,
                    _SENTINEL_BS,
                    _SENTINEL_H,
                    representative_spec.head_size,
                    cache_dtype_str=cache_dtype,
                )
                probe_shape_with_layers = (1, *probe_shape)
                stride_order_with_layers = (
                    representative_backend.get_kv_cache_stride_order(
                        include_num_layers_dimension=True,
                    )
                )
                heads_dim_idx = None
                block_size_dim_idx = None
                for phys_pos in range(1, layers_dim_idx):
                    logical_dim = stride_order_with_layers[phys_pos]
                    dim_value = probe_shape_with_layers[logical_dim]
                    if dim_value == _SENTINEL_H:
                        heads_dim_idx = phys_pos
                    elif dim_value == _SENTINEL_BS:
                        block_size_dim_idx = phys_pos

                ordered_topo = KVCacheTopology(
                    num_blocks_dim=0,
                    num_layers_dim=layers_dim_idx,
                    num_heads_dim=heads_dim_idx,
                    block_size_dim=block_size_dim_idx,
                )
            else:
                # Default / Mamba / isolated: flat (blocks, layers, page)
                cross_layer_tensor = buffer.view(
                    num_blocks,
                    num_group_layers,
                    page_size,
                )
                layers_dim_idx = 1

            logger.info(
                "Allocating a cross-layer KV cache of shape %s (group=%s)",
                cross_layer_tensor.shape,
                group_key[0],
            )

            # Create per-layer views into the shared buffer.
            group_layer_names: list[str] = []
            group_topologies: list[KVCacheTopology] = []
            for local_layer_idx, (_, kv_tensor) in enumerate(members):
                spec, backend, group_id = layer_info[kv_tensor.shared_by[0]]

                if isinstance(spec, MambaSpec):
                    view: torch.Tensor | list[torch.Tensor] = (
                        KVConnectorModelRunnerMixin._create_mamba_layer_views(
                            buffer,
                            local_layer_idx,
                            num_group_layers,
                            spec,
                            num_blocks,
                        )
                    )
                elif isinstance(spec, AttentionSpec):
                    view = KVConnectorModelRunnerMixin._create_attention_layer_view(
                        buffer,
                        local_layer_idx,
                        num_group_layers,
                        num_blocks,
                        spec,
                        backend,
                        kernel_block_sizes[group_id],
                        cache_dtype,
                    )
                else:
                    raise NotImplementedError(
                        f"Uniform KV cache layout not implemented "
                        f"for spec type {type(spec).__name__}"
                    )

                # All layers sharing this KVCacheTensor position get
                # the same view (HMA sharing).
                for name in kv_tensor.shared_by:
                    kv_caches[name] = view

                # Assign topology metadata for this layer.
                if group_key[0] == "ordered":
                    layer_topo = ordered_topo
                elif group_key[0] == "isolated":
                    layer_topo = KVCacheTopology(
                        num_blocks_dim=0,
                        num_layers_dim=None,
                    )
                else:
                    layer_topo = KVCacheTopology(
                        num_blocks_dim=0,
                        num_layers_dim=layers_dim_idx,
                    )
                for _ in kv_tensor.shared_by:
                    group_topologies.append(layer_topo)
                group_layer_names.extend(kv_tensor.shared_by)

            cross_layer_groups.append(
                CrossLayerGroup(
                    tensor=cross_layer_tensor,
                    layer_names=group_layer_names,
                    page_size_bytes=page_size,
                    topologies=group_topologies,
                )
            )

        return kv_caches, cross_layer_groups

    @staticmethod
    def build_kv_cache_references(
        cross_layer_groups: list["CrossLayerGroup"],
        kv_cache_config: KVCacheConfig,
        kv_caches: dict[str, torch.Tensor | list[torch.Tensor]],
        attn_groups: list[list[AttentionGroup]],
    ) -> tuple[
        list[KVCacheTensorReference],
        list[list[KVCacheDataReference]],
    ]:
        """
        Convert CrossLayerGroup list into the connector-facing
        KVCacheTensorReference / KVCacheDataReference structures.

        Args:
            cross_layer_groups: cross-layer buffers from
                ``allocate_hybrid_kv_caches``.
            kv_cache_config: KV cache config from the scheduler.
            kv_caches: per-layer KV cache views (used to compute head
                strides).
            attn_groups: two-level list of AttentionGroups.

        Returns:
            (kv_cache_tensors, kv_cache_groups_data_refs)
        """
        _SENTINEL_HEADS = 8

        # layer_name → (spec, backend)
        layer_info: dict[str, tuple[KVCacheSpec, type[AttentionBackend]]] = {}
        for subgroups in attn_groups:
            for attn_group in subgroups:
                for layer_name in attn_group.layer_names:
                    layer_info[layer_name] = (
                        attn_group.kv_cache_spec,
                        attn_group.backend,
                    )

        # layer_name → per-layer KV cache spec
        # (handles UniformTypeKVCacheSpecs where layers in the same
        # group may have different page sizes)
        per_layer_spec: dict[str, KVCacheSpec] = {}
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            group_kv_cache_spec = kv_cache_group.kv_cache_spec
            if isinstance(group_kv_cache_spec, UniformTypeKVCacheSpecs):
                per_layer_specs = group_kv_cache_spec.kv_cache_specs
            else:
                per_layer_specs = {}
            for layer_name in kv_cache_group.layer_names:
                per_layer_spec[layer_name] = per_layer_specs.get(
                    layer_name, group_kv_cache_spec
                )

        # layer_name → head stride in bytes
        heads_stride_bytes: dict[str, int] = {}
        for layer_name, (_, backend) in layer_info.items():
            spec = per_layer_spec.get(layer_name)
            if isinstance(spec, AttentionSpec):
                layer_kv_cache = kv_caches[layer_name]
                assert isinstance(layer_kv_cache, torch.Tensor)
                test_shape = backend.get_kv_cache_shape(
                    num_blocks=1234,
                    block_size=16,
                    num_kv_heads=_SENTINEL_HEADS,
                    head_size=256,
                )
                heads_dim_idx = test_shape.index(_SENTINEL_HEADS)
                heads_stride_bytes[layer_name] = (
                    layer_kv_cache.strides()[heads_dim_idx]
                    * layer_kv_cache.element_size()
                )

        # Build tensor refs and collect per-layer chunks.
        #
        # A CrossLayerGroup's layer_names includes ALL layers sharing
        # the buffer, but the buffer has one slot per KVCacheTensor
        # (= per member in the grouped dict).  Multiple layers in the
        # same KVCacheTensor.shared_by share one slot — they get the
        # same chunks at the same offset.
        kv_cache_tensors: list[KVCacheTensorReference] = []
        # layer_name → (tensor_idx, chunks)
        layer_chunks: dict[str, tuple[int, list[Chunk]]] = {}

        for group in cross_layer_groups:
            group_layer_set = set(group.layer_names)
            per_layer_page_size = group.page_size_bytes

            # Find the KVCacheTensors belonging to this group.
            # Each KVCacheTensor maps to one slot in the buffer.
            group_kv_cache_tensors = [
                kv_cache_tensor
                for kv_cache_tensor in kv_cache_config.kv_cache_tensors
                if kv_cache_tensor.shared_by[0] in group_layer_set
            ]

            num_slots = len(group_kv_cache_tensors)
            full_page_size_bytes = per_layer_page_size * num_slots
            tensor_idx = len(kv_cache_tensors)
            kv_cache_tensors.append(
                KVCacheTensorReference(
                    tensor=group.tensor,
                    page_size_bytes=full_page_size_bytes,
                )
            )

            # Each KVCacheTensor occupies one slot at a fixed offset.
            for slot_idx, kv_cache_tensor in enumerate(group_kv_cache_tensors):
                representative_name = kv_cache_tensor.shared_by[0]
                layer_kv_cache_spec = per_layer_spec[representative_name]
                base_offset = slot_idx * per_layer_page_size

                if isinstance(layer_kv_cache_spec, MambaSpec):
                    chunks = KVConnectorModelRunnerMixin._build_mamba_chunks(
                        representative_name,
                        layer_kv_cache_spec,
                        base_offset,
                    )
                elif isinstance(layer_kv_cache_spec, AttentionSpec):
                    real_page_size = layer_kv_cache_spec.real_page_size_bytes
                    chunks = [
                        Chunk(
                            layer_names=list(kv_cache_tensor.shared_by),
                            tensor_start_offset=base_offset,
                            tensor_length=real_page_size,
                            num_heads_stride=heads_stride_bytes.get(
                                representative_name, 0
                            ),
                        )
                    ]
                else:
                    raise NotImplementedError(
                        f"Unsupported KV cache spec: "
                        f"{type(layer_kv_cache_spec).__name__}"
                    )

                # All layers sharing this KVCacheTensor get the same
                # chunks at the same offset.
                for layer_name in kv_cache_tensor.shared_by:
                    layer_chunks[layer_name] = (tensor_idx, chunks)

        # Build one KVCacheDataReference per scheduler group.
        kv_cache_groups_data_refs: list[list[KVCacheDataReference]] = []
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            all_chunks: list[Chunk] = []
            group_tensor_idx: int | None = None

            for layer_name in kv_cache_group.layer_names:
                layer_tensor_idx, chunks = layer_chunks[layer_name]
                if group_tensor_idx is None:
                    group_tensor_idx = layer_tensor_idx
                all_chunks.extend(chunks)

            if group_tensor_idx is not None:
                unpadded_page_size_bytes = sum(
                    chunk.tensor_length for chunk in all_chunks
                )
                kv_cache_groups_data_refs.append(
                    [
                        KVCacheDataReference(
                            tensor_idx=group_tensor_idx,
                            unpadded_page_size_bytes=unpadded_page_size_bytes,
                            chunks=all_chunks,
                        )
                    ]
                )
            else:
                kv_cache_groups_data_refs.append([])

        return kv_cache_tensors, kv_cache_groups_data_refs

    @staticmethod
    def _build_mamba_chunks(
        layer_name: str,
        mamba_spec: MambaSpec,
        base_offset: int,
    ) -> list[Chunk]:
        """Build one Chunk per Mamba state tensor (e.g. conv, ssm)."""
        _MAMBA_STATE_NAMES = ("conv", "ssm")
        chunks: list[Chunk] = []
        offset = base_offset
        for idx, (state_shape, state_dtype) in enumerate(
            zip(mamba_spec.shapes, mamba_spec.dtypes)
        ):
            state_bytes = math.prod(state_shape) * get_dtype_size(state_dtype)
            state_suffix = (
                _MAMBA_STATE_NAMES[idx]
                if idx < len(_MAMBA_STATE_NAMES)
                else f"state_{idx}"
            )
            chunks.append(
                Chunk(
                    layer_names=[f"{layer_name}.{state_suffix}"],
                    tensor_start_offset=offset,
                    tensor_length=state_bytes,
                    num_heads_stride=0,
                )
            )
            offset += state_bytes
        return chunks
