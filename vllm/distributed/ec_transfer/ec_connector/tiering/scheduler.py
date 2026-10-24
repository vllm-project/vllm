# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Scheduler orchestration for CPU-primary encoder-cache tiering."""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING

from vllm.distributed.ec_transfer.ec_connector.cpu.common import (
    ECCPUConnectorMetadata,
    ECCPUWorkerMetadata,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler import (
    ECCPUScheduler,
)
from vllm.distributed.ec_transfer.ec_connector.encoder_output_width import (
    get_encoder_output_width,
)
from vllm.distributed.ec_transfer.ec_connector.tiering.config import (
    ECTieringConfig,
)
from vllm.distributed.ec_transfer.ec_connector.tiering.filesystem import (
    ECFileSystemTierManager,
)
from vllm.distributed.ec_transfer.ec_connector.tiering.layout import (
    ECCacheNamespace,
)
from vllm.distributed.ec_transfer.ec_connector.tiering.manager import (
    ECTieringManager,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import ECConnectorOutput
    from vllm.v1.request import Request


class ECTieringScheduler(ECCPUScheduler):
    """Extend the CPU scheduler with asynchronous secondary-tier transfers."""

    def __init__(self, vllm_config: VllmConfig) -> None:
        self._tiering_manager: ECTieringManager | None = None
        super().__init__(vllm_config)
        tiering_config = ECTieringConfig.from_vllm_config(vllm_config)
        fs_config = tiering_config.filesystem_tier
        if fs_config is None:
            return

        filesystem: ECFileSystemTierManager | None = None
        try:
            namespace = self._build_namespace(vllm_config)
            namespace_digest = namespace.digest()
            filesystem = ECFileSystemTierManager(
                root_dir=fs_config.root_dir,
                namespace_digest=namespace_digest,
                capacity_bytes=fs_config.capacity_bytes,
                n_read_threads=fs_config.n_read_threads,
                n_write_threads=fs_config.n_write_threads,
                fsync=fs_config.fsync,
                backpressure=fs_config.backpressure,
            )
            self._tiering_manager = ECTieringManager(
                self._primary,
                filesystem,
                namespace_digest,
            )
        except Exception:
            if filesystem is not None:
                filesystem.shutdown()
            super().shutdown()
            raise

    @staticmethod
    def _build_namespace(vllm_config: VllmConfig) -> ECCacheNamespace:
        model_config = vllm_config.model_config
        ec_config = vllm_config.ec_transfer_config
        assert ec_config is not None
        widths = tuple(sorted(ec_config.mm_encoder_output_widths.items()))
        parallel_config = vllm_config.parallel_config
        return ECCacheNamespace(
            model_id=str(model_config.model),
            model_revision=model_config.revision,
            dtype=str(model_config.dtype),
            hidden_dim=model_config.get_inputs_embeds_size(),
            encoder_output_widths=widths,
            element_size=model_config.dtype.itemsize,
            tensor_parallel_size=parallel_config.tensor_parallel_size,
            prefill_context_parallel_size=(
                parallel_config.prefill_context_parallel_size
            ),
        )

    def update_state_after_alloc(self, request: Request, index: int) -> None:
        super().update_state_after_alloc(request, index)
        if self._tiering_manager is None or not self._is_producer:
            return
        feature = request.mm_features[index]
        if feature.identifier not in self._pending_saves:
            return
        self._tiering_manager.register_cpu_save(
            feature.identifier,
            feature.modality,
            feature.mm_position.get_num_embeds(),
        )

    def update_connector_output(self, connector_output: ECConnectorOutput) -> None:
        metadata = connector_output.ec_connector_worker_meta
        completed_saves = (
            tuple(dict.fromkeys(metadata.completed_saves))
            if isinstance(metadata, ECCPUWorkerMetadata)
            else ()
        )
        super().update_connector_output(connector_output)
        if self._tiering_manager is not None:
            for identifier in completed_saves:
                self._tiering_manager.on_cpu_save_completed(identifier)
            self._tiering_manager.poll()

    def ensure_cache_available(
        self,
        request: Request,
        num_computed_tokens: int,
        local_cache_hashes: Collection[str] | None = None,
    ) -> bool:
        if not super().ensure_cache_available(
            request,
            num_computed_tokens,
            local_cache_hashes,
        ):
            return False
        if self._tiering_manager is None or not self._is_consumer:
            return True

        all_ready = True
        for feature in request.mm_features:
            position = feature.mm_position
            if position.offset + position.length <= num_computed_tokens:
                continue
            num_tokens = position.get_num_embeds()
            shape = (
                num_tokens,
                get_encoder_output_width(self._vllm_config, feature.modality),
            )
            if not self._tiering_manager.ensure_primary_entry(
                feature.identifier,
                feature.modality,
                num_tokens,
                shape,
                self._dtype.itemsize,
            ):
                all_ready = False
        return all_ready

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ECCPUConnectorMetadata:
        if self._tiering_manager is not None:
            self._tiering_manager.poll()
        return super().build_connector_meta(scheduler_output)

    def has_pending_push_work(self) -> bool:
        secondary_pending = (
            self._tiering_manager is not None
            and self._tiering_manager.has_pending()
        )
        return secondary_pending or super().has_pending_push_work()

    def shutdown(self) -> None:
        try:
            if self._tiering_manager is not None:
                self._tiering_manager.shutdown()
                self._tiering_manager = None
        finally:
            super().shutdown()
