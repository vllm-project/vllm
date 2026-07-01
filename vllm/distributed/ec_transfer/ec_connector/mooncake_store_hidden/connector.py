# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EC connector backed by Mooncake Store for hidden-state tensors."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

from vllm.distributed import (
    get_dcp_group,
    get_pcp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_hidden.data import (
    HIDDEN_LAYOUT_VERSION,
    HiddenKeyMetadata,
    HiddenSaveRequest,
    LoadSpec,
    MMMeta,
    MooncakeStoreConnectorMetadata,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_hidden.store_client import (
    MooncakeHiddenStoreClient,
    create_mooncake_hidden_store_client,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_hidden.worker import (
    HiddenLookupClient,
    HiddenLookupServer,
    HiddenStoreWorker,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


class MooncakeStoreECConnector(ECConnectorBase):
    """Hidden-state EC connector that stores tensors in Mooncake Store."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: ECConnectorRole,
        store_client: MooncakeHiddenStoreClient | None = None,
    ):
        super().__init__(vllm_config=vllm_config, role=role)
        self.lookup_client: HiddenLookupClient | None = None
        self.lookup_server: HiddenLookupServer | None = None
        self.store_client: MooncakeHiddenStoreClient | None = None
        self.worker: HiddenStoreWorker | None = None
        assert vllm_config.ec_transfer_config is not None
        extra_config = vllm_config.ec_transfer_config.ec_connector_extra_config
        self.soft_pin_video_hidden = bool(
            extra_config.get("soft_pin_video_hidden", False)
        )

        if role == ECConnectorRole.SCHEDULER:
            if self.is_consumer:
                self.lookup_client = HiddenLookupClient(vllm_config)
        else:
            if not (self.is_producer or self.is_consumer):
                return
            engine_id = vllm_config.ec_transfer_config.engine_id

            hidden_key_metadata = build_hidden_key_metadata(vllm_config)
            self.store_client = store_client or create_mooncake_hidden_store_client()
            self.worker = HiddenStoreWorker(
                store_client=self.store_client,
                producer_engine_id=engine_id,
                key_metadata=hidden_key_metadata,
            )
            if self.is_producer:
                self.worker.start_sending_thread()
            if self.is_consumer and vllm_config.parallel_config.rank == 0:
                self.lookup_server = HiddenLookupServer(self.worker, vllm_config)

        self.load_specs: dict[str, LoadSpec] = {}
        self._load_identifiers_to_schedule: set[str] = set()
        self._save_identifiers_to_schedule: set[str] = set()
        self._load_modalities: dict[str, str | None] = {}
        self._save_modalities: dict[str, str | None] = {}

    def shutdown(self) -> None:
        if self.lookup_client is not None:
            self.lookup_client.close()
        if self.lookup_server is not None:
            self.lookup_server.close()
        if self.worker is not None:
            self.worker.shutdown()

    def has_cache_item(self, identifier: str) -> bool:
        if not self.is_consumer:
            return False
        assert self.lookup_client is not None

        started = time.perf_counter()
        if not self.lookup_client.lookup(identifier):
            self.load_specs.pop(identifier, None)
            logger.info(
                "hidden_store_scheduler_miss identifier=%s "
                "reason=worker_lookup_miss hidden_store_lookup_ms=%.3f",
                identifier,
                (time.perf_counter() - started) * 1000.0,
            )
            return False

        self.load_specs[identifier] = LoadSpec(can_load=False)
        logger.info(
            "hidden_store_scheduler_hit identifier=%s " "hidden_store_lookup_ms=%.3f",
            identifier,
            (time.perf_counter() - started) * 1000.0,
        )
        return True

    def update_state_after_alloc(self, request: Request, index: int) -> None:
        mm_feature = request.mm_features[index]
        identifier = mm_feature.identifier
        modality = mm_feature.modality

        if self.is_consumer and identifier in self.load_specs:
            load_spec = self.load_specs[identifier]
            load_spec.can_load = True
            self._load_modalities[identifier] = modality
            self._load_identifiers_to_schedule.add(identifier)

        if self.is_producer:
            self._save_modalities[identifier] = modality
            self._save_identifiers_to_schedule.add(identifier)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ECConnectorMetadata:
        items_by_identifier: dict[str, MMMeta] = {}

        for identifier in self._load_identifiers_to_schedule:
            load_spec = self.load_specs.pop(identifier, None)
            if load_spec is None:
                continue
            items_by_identifier[identifier] = MMMeta(
                identifier=identifier,
                modality=self._load_modalities.get(identifier),
                load_spec=load_spec,
            )

        for identifier in self._save_identifiers_to_schedule:
            item = items_by_identifier.get(identifier)
            if item is None:
                item = MMMeta(
                    identifier=identifier,
                    modality=self._save_modalities.get(identifier),
                )
                items_by_identifier[identifier] = item
            item.can_save = True
            if item.modality is None:
                item.modality = self._save_modalities.get(identifier)

        metadata = MooncakeStoreConnectorMetadata(
            items=list(items_by_identifier.values()),
            unfinished_identifiers=self._save_identifiers_to_schedule.copy(),
        )

        self._load_identifiers_to_schedule.clear()
        self._save_identifiers_to_schedule.clear()
        self._load_modalities.clear()
        self._save_modalities.clear()
        return metadata

    def start_load_caches(
        self,
        encoder_cache: dict[str, torch.Tensor],
        **kwargs,
    ) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, MooncakeStoreConnectorMetadata)
        assert self.worker is not None
        self.worker.load(
            metadata.items,
            encoder_cache,
            device=kwargs.get("device"),
        )

    def save_caches(
        self,
        encoder_cache: dict[str, torch.Tensor],
        mm_hash: str,
        **kwargs,
    ) -> None:
        if not self.is_producer:
            return
        assert self.worker is not None
        identifier = mm_hash
        if identifier not in encoder_cache:
            logger.warning(
                "Skip hidden store save; identifier %s is missing",
                identifier,
            )
            return
        item = self._find_metadata_item(identifier)
        if item is None or not item.can_save:
            logger.debug(
                "Skip hidden store save; identifier %s has no save plan",
                identifier,
            )
            return
        pool_key = self.worker.make_pool_key(identifier)
        self.worker.enqueue_save(
            HiddenSaveRequest(
                pool_key=pool_key,
                tensor=encoder_cache[identifier],
                now_ms=kwargs.get("now_ms"),
                with_soft_pin=self._should_soft_pin(item),
            )
        )

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        if self.worker is None or not self.is_producer:
            return None, None
        finished_sending = self.worker.get_finished_sending()
        return finished_sending or None, None

    def _find_metadata_item(self, identifier: str) -> MMMeta | None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, MooncakeStoreConnectorMetadata)
        for item in metadata.items:
            if item.identifier == identifier:
                return item
        return None

    def _should_soft_pin(self, item: MMMeta) -> bool:
        return self.soft_pin_video_hidden and item.modality == "video"


def build_hidden_key_metadata(vllm_config: VllmConfig) -> HiddenKeyMetadata:
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config

    multimodal_config = getattr(model_config, "multimodal_config", None)
    compute_hash = getattr(multimodal_config, "compute_hash", None)
    mm_encoder_config_hash = (
        compute_hash() if callable(compute_hash) else "mm_encoder:default"
    )

    tp_size = get_tensor_model_parallel_world_size()
    pp_size = parallel_config.pipeline_parallel_size
    pcp_size = get_pcp_group().world_size
    dcp_size = get_dcp_group().world_size
    mm_encoder_tp_mode = getattr(
        multimodal_config,
        "mm_encoder_tp_mode",
        "unknown",
    )
    hidden_parallel_key = (
        f"tp:{tp_size}"
        f"@pp:{pp_size}"
        f"@pcp:{pcp_size}"
        f"@dcp:{dcp_size}"
        f"@mm_tp:{mm_encoder_tp_mode}"
        "@storage:replicated"
    )

    return HiddenKeyMetadata(
        model_name=model_config.model.rstrip("/").split("/")[-1],
        mm_encoder_config_hash=str(mm_encoder_config_hash),
        hidden_parallel_key=hidden_parallel_key,
        layout=HIDDEN_LAYOUT_VERSION,
    )
