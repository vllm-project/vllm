# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Define KV connector functionality mixin for model runners.
"""

import copy
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_shutdown,
    get_kv_transfer_group,
    has_kv_transfer_group,
)
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.distributed.kv_transfer.kv_connector.v1.base import supports_hma
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    CanonicalKVCaches,
    KVCacheBlockDataRef,
    KVCacheBlockTensor,
    KVCacheConfig,
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
            output.kv_connector_worker_meta = kv_connector.build_connector_worker_meta()

            if not defer_finalize:
                kv_connector.clear_connector_metadata()

    @staticmethod
    def use_uniform_kv_cache(
        attn_groups: list[list[AttentionGroup]],
        cache_dtype: CacheDType,
    ) -> bool:
        """
        Determines whether a uniform KV layout should be used.
        A uniform layout means all layers KV caches will share the same
        underlying tensor, where for a given block number, the respective
        KV data for all layers will be contiguous.
        This will allow efficient KV transfer of per-block KV data for all
        layers at once.
        Note this layout will only be applied given 3 conditions:
        1. The KV Cache config contains just a single group where all layers
            have the same page size.
        2. A KV connector is configured, and the KV connector instance prefers
            to use this layout (prefer_cross_layer_blocks() returns True)
        2. The flash attention backend supports this layout
            (get_kv_cache_stride_order(True) includes a placement for a
            num_layers dimension)

        Note that the actual placement of the num_layers dimensions
        in the unified layers tensors will be determined by the attention
        backend.
        Thus, the layers KV data may still not be contiguous per block
        if the attention backend does not support it.

        Args:
            attn_groups: The list of attention groups for this model
            cache_dtype: The KV cache dtype
        Returns:
            True if we should use a uniform KV cache layout.
        """

        if not has_kv_transfer_group():
            return False
        if not get_kv_transfer_group().prefer_cross_layer_blocks:
            return False

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

        # check that attention backend includes a layers dimension
        if len(kv_cache_stride_order) != len(kv_cache_shape) + 1:
            return False

        # stride_order[0] == 0 means num_layers stays first in physical
        # layout (identity permutation), so cross-layer is unsupported.
        return kv_cache_stride_order[0] != 0

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

        kv_caches = {}
        for i, kv_cache_tensor in enumerate(kv_cache_config.kv_cache_tensors):
            tensor = permuted_kv_cache[i]
            for layer_name in kv_cache_tensor.shared_by:
                kv_caches[layer_name] = tensor

        return kv_caches, cross_layers_kv_cache, attn_backend

    @staticmethod
    def use_canonical_kv_caches(
        kv_cache_config: KVCacheConfig,
        attn_groups: list[list[AttentionGroup]],
        cache_dtype: CacheDType,
    ) -> bool:
        """
        Determines whether a contiguous canonical KV cache should be
        allocated for HMA (Hybrid Multi-Attention) models.

        A canonical layout allocates a single contiguous buffer where,
        for a given block number, the KV data for all layers is
        contiguous. This allows efficient KV transfer of per-block data.

        This layout will only be applied given 5 conditions:
        1. A KV connector is configured and prefers cross-layer blocks.
        2. The connector supports HMA.
        3. The model has multiple KV cache groups (HMA).
        4. All groups use AttentionSpec with uniform page size.
        5. All backends share the same stride order that places
           num_blocks first in the physical layout.

        Args:
            kv_cache_config: The KV cache configuration.
            attn_groups: The attention groups (indexed [group_id][...]).
            cache_dtype: The KV cache dtype.
        Returns:
            True if we should use contiguous canonical allocation.
        """
        if not has_kv_transfer_group():
            return False
        if not get_kv_transfer_group().prefer_cross_layer_blocks:
            return False

        # The connector must support HMA
        if not supports_hma(get_kv_transfer_group()):
            return False
        if len(kv_cache_config.kv_cache_groups) <= 1:
            return False

        # All groups must use AttentionSpec with uniform page size
        page_sizes: set[int] = set()
        for group in kv_cache_config.kv_cache_groups:
            if not isinstance(group.kv_cache_spec, AttentionSpec):
                return False
            page_sizes.add(group.kv_cache_spec.page_size_bytes)
        if len(page_sizes) != 1:
            return False

        # all backends must agree on the same stride order
        spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
        assert isinstance(spec, AttentionSpec)

        common_stride_order: tuple[int, ...] | None = None
        for groups in attn_groups:
            for group in groups:
                attn_backend = group.backend
                kv_cache_shape = attn_backend.get_kv_cache_shape(
                    1234,
                    spec.block_size,
                    spec.num_kv_heads,
                    spec.head_size,
                    cache_dtype_str=cache_dtype,
                )

                try:
                    stride_order = attn_backend.get_kv_cache_stride_order(
                        include_num_layers_dimension=True
                    )
                except (AttributeError, NotImplementedError):
                    return False

                if len(stride_order) != len(kv_cache_shape) + 1:
                    return False

                if common_stride_order is None:
                    common_stride_order = stride_order
                elif stride_order != common_stride_order:
                    return False

        if common_stride_order is None:
            return False

        # stride_order[0] == 0 means the group_size dimension stays first
        # in physical layout, so per-block contiguity is not achieved.
        return common_stride_order[0] != 0

    @staticmethod
    def allocate_canonical_kv_caches(
        kv_cache_config: KVCacheConfig,
        attn_groups: list[list[AttentionGroup]],
        cache_dtype: CacheDType,
        device: torch.device,
        kernel_block_sizes: list[int],
    ) -> tuple[dict[str, torch.Tensor], CanonicalKVCaches]:
        """
        Allocates and reshapes KV caches for HMA models where all
        groups share the same page size.

        This function assumes use_canonical_kv_caches() returned True.

        Args:
            kv_cache_config: The KV cache config.
            attn_groups: The attention groups (indexed [group_id][...]).
            cache_dtype: The KV cache dtype.
            device: The torch device to allocate on.
            kernel_block_sizes: The kernel block sizes per KV cache group.
        Returns:
            A tuple (kv_caches, canonical_kv_caches) where:
                kv_caches is a dict mapping between layer names to their
                    corresponding memory buffer for KV cache.
                canonical_kv_caches is the CanonicalKVCaches wrapping
                    for the connector.
        """
        attn_backend = attn_groups[0][0].backend
        kv_cache_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
        assert isinstance(kv_cache_spec, AttentionSpec)

        tensor_sizes = set(t.size for t in kv_cache_config.kv_cache_tensors)
        assert len(tensor_sizes) == 1
        tensor_size = tensor_sizes.pop()

        page_size = kv_cache_spec.page_size_bytes
        assert tensor_size % page_size == 0
        num_blocks = tensor_size // page_size
        group_size = len(kv_cache_config.kv_cache_tensors)
        total_size = tensor_size * group_size

        unique_kernel_bs = set(kernel_block_sizes)
        assert len(unique_kernel_bs) == 1
        kernel_block_size = unique_kernel_bs.pop()
        num_blocks_per_kv_block = kv_cache_spec.block_size // kernel_block_size
        kernel_num_blocks = num_blocks * num_blocks_per_kv_block

        # prepend a group_size dimension into the shape
        kv_cache_shape = attn_backend.get_kv_cache_shape(
            kernel_num_blocks,
            kernel_block_size,
            kv_cache_spec.num_kv_heads,
            kv_cache_spec.head_size,
            cache_dtype_str=cache_dtype,
        )
        kv_cache_shape = (group_size,) + kv_cache_shape

        try:
            kv_cache_stride_order = attn_backend.get_kv_cache_stride_order(
                include_num_layers_dimension=True
            )
            assert len(kv_cache_stride_order) == len(kv_cache_shape)
        except (AttributeError, NotImplementedError):
            kv_cache_stride_order = tuple(range(len(kv_cache_shape)))

        physical_shape = tuple(kv_cache_shape[i] for i in kv_cache_stride_order)

        logger.info(
            "Allocating canonical KV cache: group_size=%d, physical shape=%s",
            group_size,
            physical_shape,
        )

        # allocate one contiguous buffer for all positions
        contiguous_buffer = (
            torch.zeros(total_size, dtype=torch.int8, device=device)
            .view(kv_cache_spec.dtype)
            .view(physical_shape)
        )

        # maintain original KV shape view
        inv_order = [
            kv_cache_stride_order.index(i) for i in range(len(kv_cache_stride_order))
        ]
        permuted = contiguous_buffer.permute(*inv_order)

        kv_caches: dict[str, torch.Tensor] = {}
        for i, kv_cache_tensor in enumerate(kv_cache_config.kv_cache_tensors):
            tensor = permuted[i]
            for layer_name in kv_cache_tensor.shared_by:
                kv_caches[layer_name] = tensor

        # build CanonicalKVCaches for the connector
        block_dim = attn_backend.get_kv_cache_block_dim(
            kernel_block_size,
            kv_cache_spec.num_kv_heads,
            kv_cache_spec.head_size,
            cache_dtype_str=cache_dtype,
        )

        # split each position's tensor so num_blocks is the leading dim
        block_tensors: list[KVCacheBlockTensor] = []
        sample = permuted[0]
        num_splits = 1
        for d in range(block_dim):
            num_splits *= sample.shape[d]

        for i in range(group_size):
            layer_tensor = permuted[i]
            if block_dim == 0:
                page_bytes = layer_tensor.stride(0) * layer_tensor.element_size()
                block_tensors.append(
                    KVCacheBlockTensor(tensor=layer_tensor, page_size_bytes=page_bytes)
                )
            else:
                for j in range(num_splits):
                    idx = []
                    rem = j
                    for d in range(block_dim - 1, -1, -1):
                        idx.append(rem % layer_tensor.shape[d])
                        rem //= layer_tensor.shape[d]
                    idx.reverse()
                    sub_tensor = layer_tensor[tuple(idx)]
                    page_bytes = sub_tensor.stride(0) * sub_tensor.element_size()
                    block_tensors.append(
                        KVCacheBlockTensor(
                            tensor=sub_tensor, page_size_bytes=page_bytes
                        )
                    )

        layer_to_position: dict[str, int] = {}
        for pos, kv_tensor in enumerate(kv_cache_config.kv_cache_tensors):
            for layer_name in kv_tensor.shared_by:
                layer_to_position[layer_name] = pos

        group_data_refs: list[list[KVCacheBlockDataRef]] = []
        for group in kv_cache_config.kv_cache_groups:
            refs: list[KVCacheBlockDataRef] = []
            per_split_page_size = group.kv_cache_spec.page_size_bytes // num_splits
            for layer_name in group.layer_names:
                pos = layer_to_position[layer_name]
                base = pos * num_splits
                for s in range(num_splits):
                    refs.append(
                        KVCacheBlockDataRef(
                            tensor_idx=base + s,
                            page_size_bytes=per_split_page_size,
                        )
                    )
            group_data_refs.append(refs)

        canonical = CanonicalKVCaches(
            tensors=block_tensors,
            group_data_refs=group_data_refs,
        )
        return kv_caches, canonical
