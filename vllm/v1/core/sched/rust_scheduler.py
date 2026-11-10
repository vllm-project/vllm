# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Rust port of `Scheduler` and `AsyncScheduler` (`vllm._rust_core`).

`RustScheduler` and `RustAsyncScheduler` are drop-in replacements with
bitwise-identical behavior, selected with `--scheduler-cls` or by
`VLLM_USE_RUST_CORE` (see `rust_scheduler_cls`). The Python collaborators (KV
connector, structured output manager, KV event publisher) are constructed as
`Scheduler` does and called by the Rust scheduler through the extension.

Unless `VLLM_RUST_CORE_NATIVE_HOOKS=0`, native ports replace the scheduler
half of the NIXL and Mooncake store KV connectors where they support the
configuration (`_native_connector`; the reason is logged when they do not).
The Python connector stays the engine's, for the handshake side channel or the
store lookups.

Not supported (rejected at construction): custom encoder cache managers and
custom KV cache managers; per request: prompt embeds with KV cache events.
"""

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

import vllm.envs as envs
from vllm.config import KVEventsConfig, VllmConfig
from vllm.distributed.aux_output_connector.connector import AuxOutputSchedulerConnector
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
from vllm.distributed.kv_events import EventPublisherFactory
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.platforms import current_platform
from vllm.v1.core.kv_cache_manager import MambaManager
from vllm.v1.core.sched.interface import PauseState, SchedulerInterface
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.metrics.perf import ModelMetrics
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup
from vllm.v1.structured_output import StructuredOutputManager

logger = init_logger(__name__)

_KV_CONNECTOR_V1 = "vllm.distributed.kv_transfer.kv_connector.v1"


def unsupported_reason(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> str | None:
    """Why the Rust scheduler does not support this configuration, if so."""
    try:
        from vllm._rust_core import RustSchedulerCore
    except ImportError:
        return "vllm._rust_core is not built"
    if vllm_config.ec_manager_config.get_encoder_cache_manager_obj() is not None:
        return "custom encoder cache managers"
    for group in kv_cache_config.kv_cache_groups:
        manager_cls = KVCacheSpecRegistry.get_manager_class(
            group.kv_cache_spec, group.role
        )
        if getattr(manager_cls, "__module__", None) != MambaManager.__module__:
            return f"custom KV cache manager {manager_cls}"
    try:
        RustSchedulerCore.check_kv_cache_config(kv_cache_config)
    except NotImplementedError as e:
        return str(e)
    return None


def _core_config(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    block_size: int,
    hash_block_size: int | None,
    mm_budget: MultiModalBudget | None,
    include_finished_set: bool,
    log_stats: bool,
    decode_stagger: int | None,
) -> dict[str, Any]:
    """The values `Scheduler.__init__` derives from `vllm_config`."""
    sched = vllm_config.scheduler_config
    model = vllm_config.model_config
    cache = vllm_config.cache_config
    parallel = vllm_config.parallel_config
    observability = vllm_config.observability_config
    spec = vllm_config.speculative_config
    kv_transfer = vllm_config.kv_transfer_config
    kv_events = vllm_config.kv_events_config
    dynamic_sd_lookup = None
    if spec is not None and spec.num_speculative_tokens_per_batch_size:
        dynamic_sd_lookup = build_dynamic_sd_schedule_lookup(
            spec.num_speculative_tokens_per_batch_size,
            vllm_max_batch_size=sched.max_num_seqs,
            vllm_num_speculative_tokens=vllm_config.num_speculative_tokens,
        )
    return dict(
        max_num_seqs=sched.max_num_seqs,
        max_num_active_seqs=sched.max_num_active_seqs,
        max_num_batched_tokens=sched.max_num_batched_tokens,
        max_num_scheduled_tokens=sched.max_num_scheduled_tokens,
        long_prefill_token_threshold=sched.long_prefill_token_threshold,
        long_prefill_token_threshold_adaptive=(
            sched.long_prefill_token_threshold_adaptive
        ),
        enable_chunked_prefill=sched.enable_chunked_prefill,
        disable_chunked_mm_input=sched.disable_chunked_mm_input,
        policy=sched.policy,
        async_scheduling=bool(sched.async_scheduling),
        scheduler_reserve_full_isl=sched.scheduler_reserve_full_isl,
        cache_aware_admission_window=sched.cache_aware_admission_window,
        cache_aware_admission_threshold=sched.cache_aware_admission_threshold,
        watermark=sched.watermark,
        max_num_inflight_kv_loads=sched.max_num_inflight_kv_loads,
        max_async_kv_loads_per_step=sched.max_async_kv_loads_per_step,
        max_model_len=model.max_model_len,
        # Pooling and diffusion models do not append a sampled token after
        # prefill.
        num_sampled_tokens_per_step=(
            0 if model.runner_type == "pooling" or model.is_diffusion else 1
        ),
        is_encoder_decoder=model.is_encoder_decoder,
        is_mm_encoder_only=vllm_config.is_mm_encoder_only,
        uses_mrope=model.uses_mrope,
        return_sampling_mask=model.return_sampling_mask,
        block_size=block_size,
        hash_block_size=hash_block_size,
        cache_block_size=cache.block_size,
        num_gpu_blocks=cache.num_gpu_blocks,
        enable_prefix_caching=cache.enable_prefix_caching,
        mamba_cache_mode=cache.mamba_cache_mode,
        enable_mamba_shared_prefix_checkpoint=(
            cache.enable_mamba_shared_prefix_checkpoint
        ),
        pipeline_parallel_size=parallel.pipeline_parallel_size,
        decode_context_parallel_size=parallel.decode_context_parallel_size,
        max_concurrent_batches=vllm_config.max_concurrent_batches,
        max_in_flight_tokens=vllm_config.max_in_flight_tokens,
        num_spec_tokens=vllm_config.num_speculative_tokens,
        num_lookahead_tokens=vllm_config.num_lookahead_tokens,
        num_prefill_lookahead_tokens=vllm_config.num_prefill_lookahead_tokens,
        num_hash_lookahead_tokens=vllm_config.num_hash_lookahead_tokens,
        use_eagle_block_drop=spec is not None and spec.use_eagle_block_drop(),
        draft_slots=spec.max_num_new_slots_for_drafting if spec is not None else 0,
        dynamic_sd_lookup=dynamic_sd_lookup,
        max_num_encoder_input_tokens=(
            mm_budget.encoder_compute_budget if mm_budget else 0
        ),
        encoder_cache_size=mm_budget.encoder_cache_size if mm_budget else 0,
        max_loras=(
            vllm_config.lora_config.max_loras if vllm_config.lora_config else None
        ),
        kv_transfer=(
            None
            if kv_transfer is None
            else (
                kv_transfer.is_kv_producer,
                kv_transfer.is_kv_consumer,
                kv_transfer.kv_load_failure_policy == "recompute",
            )
        ),
        log_stats=log_stats,
        enable_iteration_details=(
            log_stats and observability.enable_logging_iteration_details
        )
        or vllm_config.profiler_config.profiler is not None,
        kv_cache_metrics_sample=(
            observability.kv_cache_metrics_sample
            if observability.kv_cache_metrics
            else None
        ),
        spec_decode_metrics_level=observability.per_request_spec_decode_metrics,
        enable_kv_cache_events=(
            kv_events is not None and kv_events.enable_kv_cache_events
        ),
        kv_events_use_int_block_hashes=envs.VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES,
        report_watermarking=bool(vllm_config.watermark_config),
        include_finished_set=include_finished_set,
        decode_stagger=decode_stagger,
        ec_transfer=vllm_config.ec_transfer_config is not None,
        aux_output=vllm_config.aux_output_config.enabled,
        custom_encoder_cache_manager=(
            vllm_config.ec_manager_config.get_encoder_cache_manager_obj() is not None
        ),
    )


def _error_request_order(
    grammar: list[str],
    encoder_cache_mismatch: list[str],
    failed_recving: list[str],
    invalid_async: list[str],
    invalid_sync: list[str],
    ec_unavailable: list[str],
) -> list[str]:
    """The order in which `Scheduler.update_from_output` finishes the requests
    that failed, given each source's ids in the order they were found: it
    iterates a set, whose order follows the string hashes and how the set was
    built, so build it the same way."""
    failed: set[str] = set()
    if failed_recving:
        failed.update(set(failed_recving))
    if invalid_async or invalid_sync:
        failed.update(set(invalid_async) | set(invalid_sync))
    error_req_ids = set(dict.fromkeys(grammar))
    error_req_ids.update(set(encoder_cache_mismatch))
    error_req_ids.update(failed)
    # In the order of the EC connector's set.
    error_req_ids.update(ec_unavailable)
    # `finish_requests` copies the set.
    return list(set(error_req_ids))


def _native_connector(connector: KVConnectorBase_V1) -> tuple[Any, str | None]:
    """The native scheduler half for `connector`: a `RustNixlConnectorScheduler`,
    `"mooncake_store"`, or None (keep the Python hooks) with the reason."""
    from vllm._rust_core import (
        RustNixlConnectorScheduler,
        mooncake_store_native_scheduler_unsupported,
    )

    if not envs.VLLM_RUST_CORE_NATIVE_HOOKS:
        return None, "VLLM_RUST_CORE_NATIVE_HOOKS=0"
    module = type(connector).__module__
    if module.startswith(f"{_KV_CONNECTOR_V1}.nixl."):
        try:
            native = RustNixlConnectorScheduler(connector)
        except NotImplementedError as e:
            return None, str(e)
        # Peers re-handshake when the registration changes; the native half
        # stamps the generation into the `kv_transfer_params` it returns.
        connector_scheduler = connector.connector_scheduler  # type: ignore[attr-defined]
        set_metadata = connector_scheduler.set_xfer_handshake_metadata

        def set_xfer_handshake_metadata(metadata):
            set_metadata(metadata)
            native.set_registration_generation(
                connector_scheduler._registration_generation
            )

        connector_scheduler.set_xfer_handshake_metadata = set_xfer_handshake_metadata
        return native, None
    if module.startswith(f"{_KV_CONNECTOR_V1}.mooncake_store."):
        reason = mooncake_store_native_scheduler_unsupported(connector)
        return (None, reason) if reason else ("mooncake_store", None)
    return None, f"{type(connector).__name__} has no native scheduler half"


class _RequestsView(Mapping[str, Request]):
    """`Scheduler.requests`: the requests the scheduler holds, each brought up
    to date with the Rust scheduler when read."""

    def __init__(self, core: Any):
        self._core = core

    def __getitem__(self, request_id: str) -> Request:
        request = self._core.get_request(request_id)
        if request is None:
            raise KeyError(request_id)
        return request

    def __iter__(self) -> Iterator[str]:
        return iter(self._core.request_ids())

    def __len__(self) -> int:
        return self._core.num_requests()

    def __contains__(self, request_id: object) -> bool:
        return isinstance(request_id, str) and self._core.has_request(request_id)


class RustScheduler(SchedulerInterface):
    """`Scheduler`, implemented in Rust."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        hash_block_size: int | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        if (reason := unsupported_reason(vllm_config, kv_cache_config)) is not None:
            raise NotImplementedError(f"The Rust scheduler does not support {reason}")
        from vllm._rust_core import RustSchedulerCore

        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.kv_cache_config = kv_cache_config
        self.structured_output_manager = structured_output_manager
        self.log_stats = log_stats
        self.block_size = block_size
        self.hash_block_size = hash_block_size or block_size
        self.max_model_len = vllm_config.model_config.max_model_len

        # Encoder-related.
        mm_budget = (
            MultiModalBudget(vllm_config, mm_registry)
            if vllm_config.model_config.supports_multimodal_inputs
            else None
        )
        # NOTE: Text-only encoder-decoder models are implemented as
        # multi-modal models for convenience
        # Example: https://github.com/vllm-project/bart-plugin
        if vllm_config.model_config.is_encoder_decoder:
            assert mm_budget and len(mm_budget.mm_max_toks_per_item) <= 1, (
                "Encoder-decoder models are expected to implement the "
                "multimodal interface with at most one modality."
            )

        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self._connector: KVConnectorBase_V1 | None = None
        if vllm_config.kv_transfer_config is not None:
            assert not vllm_config.model_config.is_encoder_decoder, (
                "Encoder-decoder models are not currently supported with KV connectors"
            )
            self._connector = KVConnectorFactory.create_connector(
                config=vllm_config,
                role=KVConnectorRole.SCHEDULER,
                kv_cache_config=kv_cache_config,
            )
        self._ec_connector: ECConnectorBase | None = None
        if vllm_config.ec_transfer_config is not None:
            self._ec_connector = ECConnectorFactory.create_connector(
                config=vllm_config, role=ECConnectorRole.SCHEDULER
            )
        self.kv_event_publisher = EventPublisherFactory.create(
            vllm_config.kv_events_config,
            vllm_config.parallel_config.data_parallel_index,
        )
        self.perf_metrics: ModelMetrics | None = None
        if log_stats and vllm_config.observability_config.enable_mfu_metrics:
            self.perf_metrics = ModelMetrics(vllm_config)

        native_connector = None
        if self._connector is not None:
            native_connector, reason = _native_connector(self._connector)
            name = type(self._connector).__name__
            if native_connector is None:
                logger.info_once(
                    "The Rust scheduler calls the Python %s hooks: %s", name, reason
                )
            else:
                logger.info_once(
                    "The Rust scheduler uses the native %s scheduler half.", name
                )

        self._core = RustSchedulerCore(
            _core_config(
                vllm_config,
                kv_cache_config,
                block_size,
                hash_block_size,
                mm_budget,
                include_finished_set,
                log_stats,
                self._decode_stagger(),
            ),
            kv_cache_config,
            structured_output_manager,
            self.kv_event_publisher,
            _error_request_order,
            self.connector,
            native_connector,
        )
        logger.info_once("Using the Rust scheduler (%s).", type(self).__name__)
        self.kv_cache_manager = self._core.kv_cache_manager
        # Bind after construction so connectors can access the cache manager.
        if self.connector is not None:
            self.connector.bind_kv_cache_manager(self.kv_cache_manager)
        if self._ec_connector is not None:
            self._core.set_ec_connector(self._ec_connector)
        self._aux_output_connector: AuxOutputSchedulerConnector | None = None
        if vllm_config.aux_output_config.enabled:
            self.aux_output_connector = AuxOutputSchedulerConnector(
                self.kv_cache_manager, self.connector
            )
        self.requests = _RequestsView(self._core)

    def _decode_stagger(self) -> int | None:
        """`AsyncScheduler.decode_stagger`; None for `Scheduler`."""
        return None

    @property
    def connector(self) -> KVConnectorBase_V1 | None:
        return self._connector

    @connector.setter
    def connector(self, connector: KVConnectorBase_V1) -> None:
        self._core.set_connector(connector)
        self._connector = connector

    @property
    def ec_connector(self) -> ECConnectorBase | None:
        return self._ec_connector

    @ec_connector.setter
    def ec_connector(self, ec_connector: ECConnectorBase | None) -> None:
        self._core.set_ec_connector(ec_connector)
        self._ec_connector = ec_connector

    @property
    def aux_output_connector(self) -> AuxOutputSchedulerConnector | None:
        return self._aux_output_connector

    @aux_output_connector.setter
    def aux_output_connector(
        self, connector: AuxOutputSchedulerConnector | None
    ) -> None:
        self._core.set_aux_output_connector(connector)
        self._aux_output_connector = connector

    @property
    def running(self) -> list[Request]:
        return self._core.running()

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        return self._core.schedule(throttle_prefills)

    def get_grammar_bitmask(
        self, scheduler_output: SchedulerOutput
    ) -> GrammarOutput | None:
        return self._core.get_grammar_bitmask(scheduler_output)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        perf_stats = None
        if self.perf_metrics and self.perf_metrics.is_enabled():
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)
        return self._core.update_from_output(
            scheduler_output, model_runner_output, perf_stats
        )

    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        self._core.update_draft_token_ids(draft_token_ids)

    def update_draft_token_ids_in_output(
        self, draft_token_ids: DraftTokenIds, scheduler_output: SchedulerOutput
    ) -> None:
        self._core.update_draft_token_ids_in_output(draft_token_ids, scheduler_output)

    def add_request(self, request: Request) -> None:
        self._core.add_request(request)

    def has_request(self, request_id: str) -> bool:
        return self._core.has_request(request_id)

    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[Request]:
        # In the order `Scheduler` iterates them.
        if isinstance(request_ids, str):
            request_ids = [request_ids]
        elif request_ids is not None:
            request_ids = list(set(request_ids))
        return self._core.finish_requests(request_ids, finished_status)

    def get_num_unfinished_requests(self) -> int:
        return self._core.get_num_unfinished_requests()

    def has_unfinished_requests(self) -> bool:
        return self._core.has_unfinished_requests()

    def has_finished_requests(self) -> bool:
        return self._core.has_finished_requests()

    def has_requests(self) -> bool:
        return self._core.has_requests()

    @property
    def pause_state(self) -> PauseState:
        return self._core.pause_state

    def set_pause_state(self, pause_state: PauseState) -> None:
        self._core.set_pause_state(pause_state)

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return self._core.reset_prefix_cache(reset_running_requests, reset_connector)

    def reset_encoder_cache(self) -> None:
        self._core.reset_encoder_cache()

    def get_request_counts(self) -> tuple[int, int]:
        return self._core.get_request_counts()

    def get_kv_cache_usage(self) -> float:
        return self._core.get_kv_cache_usage()

    def make_stats(self) -> SchedulerStats | None:
        return self._core.make_stats()

    def shutdown(self) -> None:
        if self.kv_event_publisher:
            self.kv_event_publisher.shutdown()
        if self.connector is not None:
            self.connector.shutdown()
        if self.ec_connector is not None:
            self.ec_connector.shutdown()

    def get_kv_connector(self) -> KVConnectorBase_V1 | None:
        return self.connector

    def get_ec_connector(self) -> ECConnectorBase | None:
        return self.ec_connector

    def get_kv_event_publisher_config(self) -> KVEventsConfig | None:
        return self.kv_event_publisher.get_publisher_config()


class RustAsyncScheduler(RustScheduler):
    """`AsyncScheduler`, implemented in Rust."""

    def _decode_stagger(self) -> int | None:
        # Only XPU can disable PP decode microbatching via VLLM_XPU_PP_MICROBATCH.
        if current_platform.is_xpu() and not envs.VLLM_XPU_PP_MICROBATCH:
            return 1
        return self.vllm_config.parallel_config.pipeline_parallel_size


def _python_fallback(rust_cls: type[RustScheduler], python_cls: type[Scheduler]):
    """`rust_cls`, constructing `python_cls` instead for configurations the
    Rust scheduler does not support."""

    class WithFallback(rust_cls):  # type: ignore[valid-type, misc]
        def __new__(cls, vllm_config, kv_cache_config, *args, **kwargs):
            reason = unsupported_reason(vllm_config, kv_cache_config)
            if reason is not None:
                logger.info(
                    "Using the Python scheduler: the Rust one does not support %s.",
                    reason,
                )
                return python_cls(vllm_config, kv_cache_config, *args, **kwargs)
            return super().__new__(cls)

    WithFallback.__name__ = WithFallback.__qualname__ = rust_cls.__name__
    return WithFallback


def rust_scheduler_cls(async_scheduling: bool) -> type[SchedulerInterface]:
    """The scheduler class `VLLM_USE_RUST_CORE` selects."""
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    if async_scheduling:
        return _python_fallback(RustAsyncScheduler, AsyncScheduler)
    return _python_fallback(RustScheduler, Scheduler)
