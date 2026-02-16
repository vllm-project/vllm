# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Define KV connector functionality mixin for model runners.
"""

import copy
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
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


@dataclass
class CrossLayerGroup:
    """One contiguous int8 tensor shared by layers with the same page size.

    Per-layer views reinterpret the raw bytes as the layer's dtype.
    """

    tensor: torch.Tensor
    layer_names: list[str]
    page_size_bytes: int
    dtype: torch.dtype
    spec: KVCacheSpec
    backend: type[AttentionBackend]


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
    ) -> AbstractContextManager[KVConnectorOutput | None]:
        return (
            KVConnectorModelRunnerMixin._get_kv_connector_output(scheduler_output)
            if has_kv_transfer_group()
            else nullcontext()
        )

    # This context manager must be used within an active forward context.
    # It encapsulates the entire KV connector lifecycle within execute_model
    @staticmethod
    @contextmanager
    def _get_kv_connector_output(
        scheduler_output: "SchedulerOutput", wait_for_save: bool = True
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

            kv_connector.clear_connector_metadata()

    @staticmethod
    def use_uniform_kv_cache(
        attn_groups: list[list[AttentionGroup]],
        cache_dtype: CacheDType,
    ) -> bool:
        """Check if we should use a uniform cross-layer KV layout.

        When enabled, layers are grouped by page_size_bytes and each group
        gets a contiguous (num_blocks, num_layers, page_size) tensor for
        efficient per-block transfers. Requires a KV connector that prefers
        cross-layer blocks and only AttentionSpec/MambaSpec layers.
        """

        if not has_kv_transfer_group():
            return False
        if not get_kv_transfer_group().prefer_cross_layer_blocks:
            return False

        if not attn_groups:
            return False

        for subgroups in attn_groups:
            for attn_group in subgroups:
                if not isinstance(attn_group.kv_cache_spec, (AttentionSpec, MambaSpec)):
                    logger.warning(
                        "Uniform KV cache layout not supported for "
                        "spec type %s, falling back to per-layer "
                        "allocation",
                        type(attn_group.kv_cache_spec).__name__,
                    )
                    return False

        return True

    @staticmethod
    def _find_blocks_dim(
        backend: type[AttentionBackend],
        kv_cache_spec: AttentionSpec,
        cache_dtype: CacheDType,
    ) -> int:
        """Find which dim in get_kv_cache_shape holds num_blocks.

        Probes with a sentinel value (same trick as kv_connector/utils.py).
        """
        _SENTINEL = 1234
        sentinel_shape = backend.get_kv_cache_shape(
            _SENTINEL,
            kv_cache_spec.block_size,
            kv_cache_spec.num_kv_heads,
            kv_cache_spec.head_size,
            cache_dtype_str=cache_dtype,
        )
        return sentinel_shape.index(_SENTINEL)

    @staticmethod
    def _per_layer_permutation(
        stride_order: tuple[int, ...],
        blocks_physical_pos: int,
    ) -> tuple[int, ...]:
        """Permutation to go from (num_blocks, *page_physical_dims) back
        to the backend's logical dim order."""
        n = len(stride_order)
        inv_stride = [0] * n
        for i, j in enumerate(stride_order):
            inv_stride[j] = i

        p = blocks_physical_pos

        def phys_to_our(phys_dim: int) -> int:
            if phys_dim == p:
                return 0
            elif phys_dim < p:
                return phys_dim + 1
            else:
                return phys_dim

        return tuple(phys_to_our(inv_stride[j]) for j in range(n))

    @staticmethod
    def _create_attention_layer_view(
        cross_layer_tensor: torch.Tensor,
        layer_idx: int,
        num_layers: int,
        spec: AttentionSpec,
        backend: type[AttentionBackend],
        kernel_num_blocks: int,
        kernel_block_size: int,
        cache_dtype: CacheDType,
    ) -> torch.Tensor:
        """Carve one attention layer's view from the cross-layer tensor.

        Pipeline: view (typed 3D) -> slice layer dim -> permute to
        the backend's logical shape.
        """
        logical_shape = backend.get_kv_cache_shape(
            kernel_num_blocks,
            kernel_block_size,
            spec.num_kv_heads,
            spec.head_size,
            cache_dtype_str=cache_dtype,
        )

        try:
            stride_order = backend.get_kv_cache_stride_order()
            assert len(stride_order) == len(logical_shape)
        except (AttributeError, NotImplementedError):
            stride_order = tuple(range(len(logical_shape)))

        physical_shape = tuple(logical_shape[j] for j in stride_order)

        blocks_logical_pos = KVConnectorModelRunnerMixin._find_blocks_dim(
            backend, spec, cache_dtype
        )
        blocks_physical_pos = stride_order[blocks_logical_pos]

        page_physical_shape = (
            physical_shape[:blocks_physical_pos]
            + physical_shape[blocks_physical_pos + 1 :]
        )

        layer_view = cross_layer_tensor.view(
            kernel_num_blocks, num_layers, *page_physical_shape
        )
        layer_slice = layer_view[:, layer_idx]
        perm = KVConnectorModelRunnerMixin._per_layer_permutation(
            stride_order, blocks_physical_pos
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
        """Carve one Mamba layer's state tensors from the cross-layer tensor.

        Mamba packs multiple states (conv, ssm, ...) into one page.
        We use as_strided so each block's data across layers stays
        contiguous for efficient transfers.
        """
        page_size_bytes = spec.page_size_bytes
        state_tensors: list[torch.Tensor] = []
        storage_offset_bytes = layer_idx * page_size_bytes

        for shape, dtype in zip(spec.shapes, spec.dtypes):
            dtype_size = torch.tensor([], dtype=dtype).element_size()
            num_elements_per_page = page_size_bytes // dtype_size
            block_stride = num_layers * num_elements_per_page

            target_shape = (num_blocks, *shape)
            inner_stride = torch.empty(target_shape).stride()
            target_stride = (block_stride, *inner_stride[1:])

            assert storage_offset_bytes % dtype_size == 0
            flat = cross_layer_tensor.view(torch.int8).view(dtype)
            tensor = torch.as_strided(
                flat,
                size=target_shape,
                stride=target_stride,
                storage_offset=storage_offset_bytes // dtype_size,
            )
            state_tensors.append(tensor)
            storage_offset_bytes += inner_stride[0] * dtype_size

        return state_tensors

    @staticmethod
    def allocate_uniform_kv_caches(
        kv_cache_config: KVCacheConfig,
        attn_groups: list[list[AttentionGroup]],
        cache_dtype: CacheDType,
        device: torch.device,
        kernel_block_sizes: list[int],
    ) -> tuple[
        dict[str, torch.Tensor | list[torch.Tensor]],
        list[CrossLayerGroup],
    ]:
        """Allocate cross-layer KV caches, one tensor per page_size group.

        Each group is a contiguous int8 tensor of shape
        (num_blocks, num_layers_in_group, page_size_bytes).
        Per-layer views reinterpret the raw bytes for the layer's dtype.

        Assumes use_uniform_kv_cache() returned True.

        Returns (kv_caches, cross_layer_groups) where kv_caches maps
        layer names to views and cross_layer_groups holds the backing
        tensors with metadata.
        """
        # Build layer_name -> (spec, backend, group_id) lookup
        layer_info: dict[str, tuple[KVCacheSpec, type[AttentionBackend], int]] = {}
        for subgroups in attn_groups:
            for attn_group in subgroups:
                for name in attn_group.layer_names:
                    layer_info[name] = (
                        attn_group.kv_cache_spec,
                        attn_group.backend,
                        attn_group.kv_cache_group_id,
                    )

        # Group KVCacheTensors by page_size_bytes
        grouped: dict[int, list[tuple[int, Any]]] = defaultdict(list)
        for tensor_idx, kv_tensor in enumerate(kv_cache_config.kv_cache_tensors):
            spec = layer_info[kv_tensor.shared_by[0]][0]
            grouped[spec.page_size_bytes].append((tensor_idx, kv_tensor))

        # Allocate one cross-layer tensor per group
        kv_caches: dict[str, torch.Tensor | list[torch.Tensor]] = {}
        cross_layer_groups: list[CrossLayerGroup] = []

        for page_size, members in grouped.items():
            num_group_layers = len(members)

            first_size = members[0][1].size
            assert all(m[1].size == first_size for m in members), (
                "All KVCacheTensors in a cross-layer group must have the same size"
            )
            assert first_size % page_size == 0
            num_blocks = first_size // page_size

            rep_name = members[0][1].shared_by[0]
            rep_spec, rep_backend, _ = layer_info[rep_name]

            # Raw int8 buffer; per-layer views reinterpret as needed
            total_bytes = first_size * num_group_layers
            raw = torch.zeros(total_bytes, dtype=torch.int8, device=device)
            cross_layer_tensor = raw.view(num_blocks, num_group_layers, page_size)

            logger.info(
                "Allocating a cross-layer KV cache of shape %s",
                cross_layer_tensor.shape,
            )

            # Create per-layer views, dispatching by spec type
            layer_views: list[torch.Tensor | list[torch.Tensor]] = []
            for local_idx, (_, kv_tensor) in enumerate(members):
                spec, backend, gid = layer_info[kv_tensor.shared_by[0]]

                if isinstance(spec, MambaSpec):
                    layer_views.append(
                        KVConnectorModelRunnerMixin._create_mamba_layer_views(
                            raw,
                            local_idx,
                            num_group_layers,
                            spec,
                            num_blocks,
                        )
                    )
                elif isinstance(spec, AttentionSpec):
                    dtype = spec.dtype
                    el = torch.tensor([], dtype=dtype).element_size()
                    kbs = kernel_block_sizes[gid]
                    npkb = spec.block_size // kbs
                    knb = num_blocks * npkb
                    kpe = page_size // el // npkb
                    typed_3d = raw.view(dtype).view(
                        knb,
                        num_group_layers,
                        kpe,
                    )
                    layer_views.append(
                        KVConnectorModelRunnerMixin._create_attention_layer_view(
                            typed_3d,
                            local_idx,
                            num_group_layers,
                            spec,
                            backend,
                            knb,
                            kbs,
                            cache_dtype,
                        )
                    )
                else:
                    raise NotImplementedError(
                        f"Uniform KV cache layout not implemented "
                        f"for spec type {type(spec).__name__}"
                    )

            group_layer_names: list[str] = []
            for (_, kv_tensor), view in zip(members, layer_views):
                for name in kv_tensor.shared_by:
                    kv_caches[name] = view
                group_layer_names.extend(kv_tensor.shared_by)

            cross_layer_groups.append(
                CrossLayerGroup(
                    tensor=cross_layer_tensor,
                    layer_names=group_layer_names,
                    page_size_bytes=page_size,
                    dtype=torch.int8,
                    spec=rep_spec,
                    backend=rep_backend,
                )
            )

        return kv_caches, cross_layer_groups
