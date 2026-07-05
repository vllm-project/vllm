# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EC connector backed by Mooncake Store for hidden-state tensors."""

from __future__ import annotations

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
    HIDDEN_OBJECT_KIND,
    HIDDEN_STORAGE_LAYOUT,
    HIDDEN_TENSOR_LAYOUT,
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
from vllm.multimodal.utils import get_mm_features_in_window
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
        self.lookup_async = bool(extra_config.get("lookup_async", True))

        if role == ECConnectorRole.SCHEDULER:
            if self.is_consumer:
                self.lookup_client = HiddenLookupClient(vllm_config)
        else:
            if not (self.is_producer or self.is_consumer):
                return
            hidden_key_metadata = build_hidden_key_metadata(vllm_config)
            self.store_client = store_client or create_mooncake_hidden_store_client()
            self.worker = HiddenStoreWorker(
                store_client=self.store_client,
                key_metadata=hidden_key_metadata,
            )
            if self.is_producer:
                self.worker.start_sending_thread()
            if self.is_consumer and vllm_config.parallel_config.rank == 0:
                self.lookup_server = HiddenLookupServer(self.worker, vllm_config)

        self.load_specs: dict[str, LoadSpec] = {}
        self.lookup_result_cache: dict[str, bool] = {}
        self.identifier_waiters: dict[str, set[str]] = {}
        self._candidate_consumes: dict[str, set[str]] = {}
        self._candidate_loads: dict[str, set[str]] = {}
        self._candidate_saves: dict[str, set[str]] = {}
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

        if not self.lookup_result_cache.get(identifier, False):
            self.load_specs.pop(identifier, None)
            logger.info(
                "hidden_store_scheduler_miss identifier=%s "
                "reason=local_lookup_result_miss",
                identifier,
            )
            return False

        self.load_specs.setdefault(identifier, LoadSpec(can_load=False))
        logger.info(
            "hidden_store_scheduler_hit identifier=%s",
            identifier,
        )
        return True

    def ensure_cache_available(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> bool:
        if not self.is_consumer:
            return True
        if not request.mm_features:
            return True
        assert self.lookup_client is not None

        start = num_computed_tokens
        end = request.num_tokens
        lo, hi = get_mm_features_in_window(request.mm_features, start, end)
        identifiers = list(
            dict.fromkeys(
                feature.identifier for feature in request.mm_features[lo:hi]
            )
        )
        if not identifiers:
            return True

        request_id = request.request_id
        for identifier in identifiers:
            self.identifier_waiters.setdefault(identifier, set()).add(request_id)

        unknown_identifiers = [
            identifier
            for identifier in identifiers
            if identifier not in self.lookup_result_cache
        ]
        if not unknown_identifiers:
            return True

        lookup_results = self.lookup_client.lookup_batch(
            unknown_identifiers,
            non_block=self.lookup_async,
        )
        if lookup_results is None:
            return False

        for identifier in unknown_identifiers:
            self.lookup_result_cache[identifier] = lookup_results.get(
                identifier,
                False,
            )
        return True

    def update_state_after_alloc(self, request: Request, index: int) -> None:
        mm_feature = request.mm_features[index]
        identifier = mm_feature.identifier
        modality = mm_feature.modality
        request_id = request.request_id

        self._candidate_consumes.setdefault(request_id, set()).add(identifier)

        if self.is_consumer and identifier in self.load_specs:
            self._candidate_loads.setdefault(request_id, set()).add(identifier)
            self._load_modalities[identifier] = modality

        if self.is_producer:
            self._save_modalities[identifier] = modality
            self._candidate_saves.setdefault(request_id, set()).add(identifier)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ECConnectorMetadata:
        items_by_identifier: dict[str, MMMeta] = {}
        preempted_ids = getattr(scheduler_output, "preempted_req_ids", None) or set()

        for request_id, identifiers in self._candidate_consumes.items():
            if request_id in preempted_ids:
                continue
            for identifier in identifiers:
                waiters = self.identifier_waiters.get(identifier)
                if waiters is not None:
                    waiters.discard(request_id)

        for request_id, identifiers in self._candidate_loads.items():
            if request_id in preempted_ids:
                continue
            for identifier in identifiers:
                load_spec = self.load_specs.pop(identifier, None)
                if load_spec is None:
                    continue
                load_spec.can_load = True
                items_by_identifier[identifier] = MMMeta(
                    identifier=identifier,
                    modality=self._load_modalities.get(identifier),
                    load_spec=load_spec,
                )

        for request_id, identifiers in self._candidate_saves.items():
            if request_id in preempted_ids:
                continue
            for identifier in identifiers:
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

        finished_req_ids = getattr(scheduler_output, "finished_req_ids", set())
        for finished_req_id in finished_req_ids:
            for waiters in self.identifier_waiters.values():
                waiters.discard(finished_req_id)

        self._cleanup_lookup_results_without_waiters()

        metadata = MooncakeStoreConnectorMetadata(
            items=list(items_by_identifier.values()),
        )

        self._candidate_consumes.clear()
        self._candidate_loads.clear()
        self._candidate_saves.clear()
        self._load_modalities.clear()
        self._save_modalities.clear()
        return metadata

    def _cleanup_lookup_results_without_waiters(self) -> None:
        for identifier, waiters in list(self.identifier_waiters.items()):
            if waiters:
                continue
            del self.identifier_waiters[identifier]
            self.lookup_result_cache.pop(identifier, None)
            self.load_specs.pop(identifier, None)
            if self.lookup_client is not None:
                self.lookup_client.discard(identifier)

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
    assert vllm_config.ec_transfer_config is not None
    extra_config = vllm_config.ec_transfer_config.ec_connector_extra_config

    multimodal_config = getattr(model_config, "multimodal_config", None)
    compute_hash = getattr(multimodal_config, "compute_hash", None)
    mm_encoder_config_hash = (
        compute_hash() if callable(compute_hash) else "encoder:default"
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
    parallel = (
        f"tp:{tp_size}"
        f"@pp:{pp_size}"
        f"@pcp:{pcp_size}"
        f"@dcp:{dcp_size}"
        f"@mm_tp:{mm_encoder_tp_mode}"
    )

    return HiddenKeyMetadata(
        cache_prefix=str(
            extra_config.get(
                "hidden_cache_prefix",
                extra_config.get("cache_prefix", ""),
            )
        ),
        kind=HIDDEN_OBJECT_KIND,
        model_name=model_config.model.rstrip("/").split("/")[-1],
        encoder=str(mm_encoder_config_hash),
        storage=HIDDEN_STORAGE_LAYOUT,
        parallel=parallel,
        tensor_layout=HIDDEN_TENSOR_LAYOUT,
    )
