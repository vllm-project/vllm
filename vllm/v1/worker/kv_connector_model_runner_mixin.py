# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Define KV connector functionality mixin for model runners.
"""

import enum
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
    set_forward_context,
)
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig
from vllm.v1.outputs import (
    KVConnectorOutput,
    ModelRunnerOutput,
)
from vllm.v1.worker.utils import AttentionGroup

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


class KVConnectorStepState(enum.Enum):
    """Lifecycle state of a :class:`KVConnectorStep`."""

    OPEN = enum.auto()
    FINISHED = enum.auto()
    ABORTED = enum.auto()


class KVConnectorStep:
    """Handle for one imperative KV-connector step.

    Returned by :meth:`KVConnectorModelRunnerMixin.begin_kv_connector_step` and
    passed back to :meth:`finish_kv_connector_step` (normal completion) or
    :meth:`abort_kv_connector_step` (exceptional completion). Carries only the
    per-step scalars the finish/abort halves need; holds no device, backend, or
    connector knowledge. Runners that split a step across two calls (e.g. an
    async ``execute_model`` that submits the forward and a later
    ``sample_tokens`` that completes it) keep this handle between the two calls.

    The ``state`` field enforces single, correct finalization: a step is
    ``OPEN`` from ``begin`` until exactly one of ``finish``/``abort`` moves it to
    ``FINISHED``/``ABORTED``. Re-finalizing (double finish/abort, finish after
    abort, etc.) is a programming error and raises.
    """

    __slots__ = (
        "scheduler_output",
        "output",
        "wait_for_save",
        "defer_finalize",
        "state",
    )

    def __init__(
        self,
        scheduler_output: "SchedulerOutput",
        *,
        wait_for_save: bool,
        defer_finalize: bool,
    ) -> None:
        self.scheduler_output = scheduler_output
        self.output = KVConnectorOutput()
        self.wait_for_save = wait_for_save
        self.defer_finalize = defer_finalize
        self.state = KVConnectorStepState.OPEN


# Defined as a kv connector functionality mixin for ModelRunner (GPU, TPU)
class KVConnectorModelRunnerMixin:
    @staticmethod
    def begin_kv_connector_step(
        scheduler_output: "SchedulerOutput",
        *,
        wait_for_save: bool = True,
        defer_finalize: bool = False,
    ) -> KVConnectorStep:
        """Start one KV-connector step: bind metadata and kick off loads.

        Generic imperative counterpart to the first half of
        :meth:`_get_kv_connector_output`. Device- and connector-agnostic: it
        only calls the public ``KVConnectorBase`` SPI. Use this (paired with
        :meth:`finish_kv_connector_step` on success, or
        :meth:`abort_kv_connector_step` on an exception) when a runner cannot
        wrap the forward in the :meth:`_get_kv_connector_output` context manager
        because the step spans two separate engine calls (submit in
        ``execute_model``, complete in ``sample_tokens``).

        Returns an ``OPEN`` step handle. The caller must eventually finish or
        abort it exactly once.

        ``start_load_kv`` receives the active forward context when one is set,
        else ``None`` -- determined explicitly via
        :func:`is_forward_context_available`, never by catching exceptions.
        """
        kv_connector = get_kv_transfer_group()
        assert isinstance(kv_connector, KVConnectorBase)
        assert scheduler_output.kv_connector_metadata is not None
        kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)

        # Background KV cache transfers start here. They are designed to be
        # async and may involve requests disjoint from the running ones.
        forward_context = (
            get_forward_context() if is_forward_context_available() else None
        )
        kv_connector.start_load_kv(forward_context)

        return KVConnectorStep(
            scheduler_output,
            wait_for_save=wait_for_save,
            defer_finalize=defer_finalize,
        )

    @staticmethod
    def finish_kv_connector_step(step: KVConnectorStep) -> KVConnectorOutput:
        """Complete one KV-connector step normally and return its output.

        Generic imperative counterpart to the success path of the ``finally`` in
        :meth:`_get_kv_connector_output`: optional ``wait_for_save``, collect
        finished send/recv, invalid block ids, stats, cache events and worker
        metadata, then clear connector metadata exactly once (unless the step
        deferred finalization). Device- and connector-agnostic. Any connector
        error propagates to the caller; nothing is swallowed here.

        The step must be ``OPEN``; finishing a finished or aborted step raises
        ``RuntimeError``.
        """
        if step.state is not KVConnectorStepState.OPEN:
            raise RuntimeError(
                f"finish_kv_connector_step called on a step in state "
                f"{step.state.name}; a step may be finalized exactly once."
            )
        kv_connector = get_kv_transfer_group()
        assert isinstance(kv_connector, KVConnectorBase)

        if step.wait_for_save and not step.defer_finalize:
            kv_connector.wait_for_save()

        KVConnectorModelRunnerMixin._assemble_kv_connector_output(kv_connector, step)

        if not step.defer_finalize:
            kv_connector.clear_connector_metadata()

        step.state = KVConnectorStepState.FINISHED
        return step.output

    @staticmethod
    def abort_kv_connector_step(step: KVConnectorStep) -> None:
        """Abort one KV-connector step after an exception between begin/finish.

        Clears connector metadata exactly once so the next scheduler step does
        not inherit stale metadata, and marks the step ``ABORTED``. Does NOT
        assemble or return a success ``KVConnectorOutput`` (the step did not
        complete). The caller is responsible for re-raising the original
        exception; this method does not swallow it.

        Note on in-flight transfers: the generic ``KVConnectorBase`` SPI has no
        cancellation primitive for an already-submitted async load. Aborting
        clears this step's bound metadata so a partially-started transfer cannot
        be associated with the next step's metadata; whether the transfer itself
        continues in the background is connector-defined. Connectors that need
        hard cancellation should surface it through their own API.

        The step must be ``OPEN``; aborting a finished or aborted step raises
        ``RuntimeError``.
        """
        if step.state is not KVConnectorStepState.OPEN:
            raise RuntimeError(
                f"abort_kv_connector_step called on a step in state "
                f"{step.state.name}; a step may be finalized exactly once."
            )
        # Clear exactly once (unless finalization was deferred to a later
        # finalize_kv_connector call, matching finish semantics).
        if not step.defer_finalize and has_kv_transfer_group():
            get_kv_transfer_group().clear_connector_metadata()
        step.state = KVConnectorStepState.ABORTED

    @staticmethod
    def _assemble_kv_connector_output(
        kv_connector: KVConnectorBase, step: KVConnectorStep
    ) -> None:
        """Populate ``step.output`` from the connector (single shared impl).

        The one place that builds a ``KVConnectorOutput`` for both the context
        manager and the imperative paths: finished send/recv, invalid block ids,
        stats, cache events and worker metadata. Does not wait_for_save or clear
        metadata (the callers own lifecycle ordering).
        """
        output = step.output
        scheduler_output = step.scheduler_output
        output.finished_sending, output.finished_recving = kv_connector.get_finished(
            scheduler_output.finished_req_ids
        )
        output.invalid_block_ids = kv_connector.get_block_ids_with_load_errors()
        output.kv_connector_stats = kv_connector.get_kv_connector_stats()
        output.kv_cache_events = kv_connector.get_kv_connector_kv_cache_events()
        output.kv_connector_worker_meta = kv_connector.build_connector_worker_meta()

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

        return ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)

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
    # It encapsulates the entire KV connector lifecycle within execute_model.
    # It is a thin wrapper over the generic begin/finish/abort step helpers so
    # the in-forward path and the split (execute_model/sample_tokens) path share
    # one canonical lifecycle implementation. On success it finishes the step;
    # if the body raises, it aborts the step (clear metadata once, no fake
    # output) and re-raises the original exception.
    @staticmethod
    @contextmanager
    def _get_kv_connector_output(
        scheduler_output: "SchedulerOutput",
        wait_for_save: bool = True,
        defer_finalize: bool = False,
    ) -> Generator[KVConnectorOutput, None, None]:
        step = KVConnectorModelRunnerMixin.begin_kv_connector_step(
            scheduler_output,
            wait_for_save=wait_for_save,
            defer_finalize=defer_finalize,
        )
        try:
            yield step.output
        except BaseException:
            KVConnectorModelRunnerMixin.abort_kv_connector_step(step)
            raise
        else:
            KVConnectorModelRunnerMixin.finish_kv_connector_step(step)

    @staticmethod
    def use_uniform_kv_cache(
        attn_groups: list[list[AttentionGroup]],
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
        3. The attention backend indexes KV by the block stride
            (kv_cache_spec.indexes_kv_by_block_stride), i.e. num_blocks is the
            outermost physical dim so per-block all-layers data is contiguous.

        Note that the actual placement of the num_layers dimensions
        in the unified layers tensors will be determined by the attention
        backend.
        Thus, the layers KV data may still not be contiguous per block
        if the attention backend does not support it.

        Args:
            attn_groups: The list of attention groups for this model
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
        return kv_cache_spec.indexes_kv_by_block_stride

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
