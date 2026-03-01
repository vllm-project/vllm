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
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
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
    """
    One contiguous int8 tensor shared by layers with the same page size.

    Per-layer views reinterpret the raw bytes as the layer's dtype.
    The tensor shape follows the backend's stride order up to the layers
    dimension:
    - ordered: (num_blocks, *prefix_dims, num_layers, remaining_bytes)
    - default: (num_blocks, num_layers, page_size_bytes)
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
        clear_metadata: bool = True,
    ) -> AbstractContextManager[KVConnectorOutput | None]:
        return (
            KVConnectorModelRunnerMixin._get_kv_connector_output(
                scheduler_output, clear_metadata=clear_metadata
            )
            if has_kv_transfer_group()
            else nullcontext()
        )

    # This context manager must be used within an active forward context.
    # It encapsulates the entire KV connector lifecycle within execute_model
    @staticmethod
    @contextmanager
    def _get_kv_connector_output(
        scheduler_output: "SchedulerOutput",
        wait_for_save: bool = True,
        clear_metadata: bool = True,
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
            if wait_for_save:
                kv_connector.wait_for_save()

            output.finished_sending, output.finished_recving = (
                kv_connector.get_finished(scheduler_output.finished_req_ids)
            )
            output.invalid_block_ids = kv_connector.get_block_ids_with_load_errors()

            output.kv_connector_stats = kv_connector.get_kv_connector_stats()
            output.kv_cache_events = kv_connector.get_kv_connector_kv_cache_events()

            if clear_metadata:
                kv_connector.clear_connector_metadata()

    @staticmethod
    def clear_kv_connector_metadata() -> None:
        """Clear the KV connector metadata. Call after draft model runs."""
        if has_kv_transfer_group():
            kv_connector = get_kv_transfer_group()
            kv_connector.clear_connector_metadata()

    @staticmethod
    def use_uniform_kv_cache(
        attn_groups: list[list[AttentionGroup]],
        cache_dtype: CacheDType,
    ) -> bool:
        """
        Check if we should use a uniform cross-layer KV layout.

        When enabled, layers sharing the same page geometry are packed into
        a single contiguous tensor for efficient per-block transfers.

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
        """
        Compute the grouping key for a layer.

        Examines the backend's stride order (with layers dimension) to
        determine how this layer should be grouped:

        - ("ordered", prefix_sizes, remaining_bytes): blocks is first
          and heads come before layers. prefix_sizes are the dimension
          sizes between blocks and layers in physical order. Layers with
          the same prefix share a tensor shaped
          (num_blocks, *prefix_sizes, num_layers, remaining_bytes).
        - ("default", page_size_bytes): everything else (including
          non-attention specs or heads after layers). Layers share a
          tensor shaped (num_blocks, num_layers, page_size_bytes).
        """
        if not isinstance(spec, AttentionSpec):
            return ("default", spec.page_size_bytes)

        try:
            stride_order_wl = backend.get_kv_cache_stride_order(
                include_num_layers_dimension=True,
            )
            _B, _H = 1234, 5678
            base_shape = backend.get_kv_cache_shape(
                _B,
                spec.block_size,
                _H,
                spec.head_size,
                cache_dtype_str=cache_dtype,
            )
            heads_base = base_shape.index(_H)
        except (AttributeError, NotImplementedError, ValueError, AssertionError):
            return ("default", spec.page_size_bytes)

        # With layers prepended, every base dim index shifts up by 1.
        log_to_phys = {dim: pos for pos, dim in enumerate(stride_order_wl)}
        layers_phys = log_to_phys[0]
        heads_phys = log_to_phys[heads_base + 1]

        # Heads after layers means no useful prefix to extract.
        if heads_phys >= layers_phys:
            return ("default", spec.page_size_bytes)

        # Heads come before layers -- figure out what sits between
        # blocks and layers so we can replicate that prefix exactly.
        actual_base = backend.get_kv_cache_shape(
            1,
            spec.block_size,
            spec.num_kv_heads,
            spec.head_size,
            cache_dtype_str=cache_dtype,
        )
        actual_wl = (1, *actual_base)  # prepend a dummy layers=1
        prefix_sizes = tuple(
            actual_wl[stride_order_wl[i]] for i in range(1, layers_phys)
        )
        remaining = spec.page_size_bytes // (
            math.prod(prefix_sizes) if prefix_sizes else 1
        )
        return ("ordered", prefix_sizes, remaining)

    @staticmethod
    def _create_attention_layer_view(
        raw: torch.Tensor,
        layer_idx: int,
        num_layers: int,
        num_blocks: int,
        spec: AttentionSpec,
        backend: type[AttentionBackend],
        kernel_block_size: int,
        cache_dtype: CacheDType,
    ) -> torch.Tensor:
        """
        Carve one attention layer's view from the raw int8 buffer.

        Views the raw buffer following the backend's with-layers physical
        layout, selects the requested layer, and permutes to the backend's
        logical shape.
        """
        npkb = spec.block_size // kernel_block_size
        knb = num_blocks * npkb

        base_logical = backend.get_kv_cache_shape(
            knb,
            kernel_block_size,
            spec.num_kv_heads,
            spec.head_size,
            cache_dtype_str=cache_dtype,
        )

        try:
            stride_order_wl = backend.get_kv_cache_stride_order(
                include_num_layers_dimension=True,
            )
        except (AttributeError, NotImplementedError):
            stride_order_wl = tuple(range(len(base_logical) + 1))

        logical_wl = (num_layers, *base_logical)
        physical_wl = tuple(
            logical_wl[stride_order_wl[i]] for i in range(len(logical_wl))
        )

        typed = raw.view(spec.dtype).view(*physical_wl)

        # Select the layer and permute back to base logical order.
        log_to_phys = {dim: pos for pos, dim in enumerate(stride_order_wl)}
        layers_phys = log_to_phys[0]
        layer_slice = typed.select(layers_phys, layer_idx)

        perm = tuple(
            log_to_phys[k + 1] - (1 if log_to_phys[k + 1] > layers_phys else 0)
            for k in range(len(base_logical))
        )
        return layer_slice.permute(*perm)

    @staticmethod
    def _create_mamba_layer_views(
        cross_layer_tensor: torch.Tensor,
        layer_idx: int,
        num_layers: int,
        spec: MambaSpec,
        num_blocks: int,
    ) -> list[torch.Tensor]:
        """
        Carve one Mamba layer's state tensors from the cross-layer tensor.

        Mamba packs multiple states (conv, ssm, ...) into one page.
        We use as_strided so each block's data across layers stays
        contiguous for efficient transfers.
        """
        page_bytes = spec.page_size_bytes
        state_tensors: list[torch.Tensor] = []
        offset_bytes = layer_idx * page_bytes

        for shape, dtype in zip(spec.shapes, spec.dtypes):
            el = torch.empty((), dtype=dtype).element_size()
            elements_per_page = page_bytes // el
            state_elements = math.prod(shape)

            target_shape = (num_blocks, *shape)
            inner_strides = []
            acc = 1
            for s in reversed(shape):
                inner_strides.append(acc)
                acc *= s
            inner_strides.reverse()
            target_stride = (num_layers * elements_per_page, *inner_strides)

            assert offset_bytes % el == 0
            flat = cross_layer_tensor.view(torch.int8).view(dtype)
            state_tensors.append(
                torch.as_strided(
                    flat,
                    size=target_shape,
                    stride=target_stride,
                    storage_offset=offset_bytes // el,
                )
            )
            offset_bytes += state_elements * el

        return state_tensors

    @staticmethod
    def allocate_uniform_kv_caches(
        kv_cache_config: KVCacheConfig,
        attn_groups: list[list[AttentionGroup]],
        cache_dtype: CacheDType,
        device: torch.device,
        kernel_block_sizes: list[int],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, type[AttentionBackend]]:
        """
        Initializes and reshapes KV caches for the simple case where all
        layers have the same layout.

        This function assumes use_uniform_kv_cache() returned True.

        Args:
            kv_cache_config: The KV cache config
            attn_groups: The list of attention groups for this model
            cache_dtype: The KV cache dtype
            device: The torch device to allocate on.
            kernel_block_sizes: The kernel block sizes for each KV cache group.
        Returns:
            A tuple (kv_caches, cross_layers_kv_cache, attn_backend) where:
                kv_caches is a dict mapping between layer names to their
                    corresponding memory buffer for KV cache.
                cross_layers_kv_cache is the cross layers kv cache tensor
                attn_backend is the attention backend matching this tensor
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
            kv_cache_stride_order = tuple(range(len(kv_cache_shape)))

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
        """
        Allocate cross-layer KV caches, one tensor per group.

        Each attention layer is classified via _cross_layer_group_key
        into one of two categories:

        - ordered: blocks first and heads before layers. Layers with
          matching stride-order prefix share a tensor shaped
          (num_blocks, *prefix_dims, num_layers, remaining_bytes).
        - default: everything else (including non-attention specs or
          heads after layers). Grouped by page_size_bytes with shape
          (num_blocks, num_layers, page_size_bytes).

        Assumes use_uniform_kv_cache() returned True.
        """
        layer_info: dict[str, tuple[KVCacheSpec, type[AttentionBackend], int]] = {}
        for subgroups in attn_groups:
            for attn_group in subgroups:
                for name in attn_group.layer_names:
                    layer_info[name] = (
                        attn_group.kv_cache_spec,
                        attn_group.backend,
                        attn_group.kv_cache_group_id,
                    )

        grouped: dict[tuple, list[tuple[int, Any]]] = defaultdict(list)
        for tensor_idx, kv_tensor in enumerate(kv_cache_config.kv_cache_tensors):
            spec, backend, _ = layer_info[kv_tensor.shared_by[0]]
            key = KVConnectorModelRunnerMixin._cross_layer_group_key(
                spec,
                backend,
                cache_dtype,
            )
            grouped[key].append((tensor_idx, kv_tensor))

        kv_caches: dict[str, torch.Tensor | list[torch.Tensor]] = {}
        cross_layer_groups: list[CrossLayerGroup] = []

        for group_key, members in grouped.items():
            num_group_layers = len(members)

            first_size = members[0][1].size
            assert all(m[1].size == first_size for m in members), (
                "All KVCacheTensors in a cross-layer group must have the same size"
            )

            rep_name = members[0][1].shared_by[0]
            rep_spec, _, _ = layer_info[rep_name]
            page_size = rep_spec.page_size_bytes

            assert first_size % page_size == 0
            num_blocks = first_size // page_size

            total_bytes = first_size * num_group_layers
            raw = torch.zeros(total_bytes, dtype=torch.int8, device=device)

            from vllm.distributed.kv_transfer.kv_connector.v1.base import (
                KVCacheTopology,
            )

            if group_key[0] == "ordered":
                prefix_sizes = group_key[1]
                remaining = group_key[2]
                cross_layer_tensor = raw.view(
                    num_blocks,
                    *prefix_sizes,
                    num_group_layers,
                    remaining,
                )
                # Topology mirrors the physical shape we just built:
                # dim 0 = num_blocks
                # dims 1..len(prefix_sizes) = prefix (may contain
                #   block_size, num_heads, etc. depending on backend)
                # dim 1+len(prefix_sizes) = num_layers
                base_num_layers_dim = 1 + len(prefix_sizes)
            else:
                cross_layer_tensor = raw.view(
                    num_blocks,
                    num_group_layers,
                    page_size,
                )
                base_num_layers_dim = 1

            logger.info(
                "Allocating a cross-layer KV cache of shape %s (group=%s)",
                cross_layer_tensor.shape,
                group_key[0],
            )

            group_layer_names: list[str] = []
            group_topologies: list[KVCacheTopology] = []
            for local_idx, (_, kv_tensor) in enumerate(members):
                spec, backend, gid = layer_info[kv_tensor.shared_by[0]]

                if isinstance(spec, MambaSpec):
                    view: torch.Tensor | list[torch.Tensor] = (
                        KVConnectorModelRunnerMixin._create_mamba_layer_views(
                            raw,
                            local_idx,
                            num_group_layers,
                            spec,
                            num_blocks,
                        )
                    )
                elif isinstance(spec, AttentionSpec):
                    view = KVConnectorModelRunnerMixin._create_attention_layer_view(
                        raw,
                        local_idx,
                        num_group_layers,
                        num_blocks,
                        spec,
                        backend,
                        kernel_block_sizes[gid],
                        cache_dtype,
                    )
                else:
                    raise NotImplementedError(
                        f"Uniform KV cache layout not implemented "
                        f"for spec type {type(spec).__name__}"
                    )

                for name in kv_tensor.shared_by:
                    kv_caches[name] = view

                # One topology entry per layer in the group.
                layer_topo = KVCacheTopology(
                    num_blocks_dim=0,
                    num_layers_dim=base_num_layers_dim,
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
