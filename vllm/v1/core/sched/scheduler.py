# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import math
import time
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import KVEventsConfig, VllmConfig
from vllm.distributed.aux_output_connector.connector import AuxOutputSchedulerConnector
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
from vllm.distributed.ec_transfer.ec_connector.metrics import ECConnectorStats
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.multimodal.inputs import MultiModalKwargsItem
from vllm.multimodal.utils import get_mm_features_in_window
from vllm.outputs import RequestError
from vllm.utils.math_utils import cdiv
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
    EncoderDecoderCacheManager,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.sched.interface import PauseState, SchedulerInterface
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    IterationDetails,
    KVConnectorBlockState,
    NewRequestData,
    RepairWork,
    SchedulerOutput,
)
from vllm.v1.core.sched.request_queue import (
    RequestQueue,
    SchedulingPolicy,
    create_request_queue,
)
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    MambaSpec,
    get_mamba_prefill_checkpoint_position,
    is_mamba_prefill_checkpoint_valid,
)
from vllm.v1.metrics.perf import ModelMetrics, PerfStats
from vllm.v1.metrics.stats import (
    KV_FETCH_COMPLETED_WAITING,
    KV_FETCH_IN_PROGRESS,
    KV_FETCH_STAGES,
    KV_FETCH_WAITING_TO_START,
    PrefixCacheStats,
    RequestSpecDecodeMetrics,
    SchedulerStats,
)
from vllm.v1.outputs import (
    DraftTokenIds,
    KVConnectorOutput,
    KVLoadAttestation,
    KVLoadFailureReport,
    ModelRunnerOutput,
    RepairResultReport,
)
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.token_buffer import extend_from_slice
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)


@dataclass
class RecoveryPlan:
    """Scheduler-owned admission state for one damaged request (the
    hole-recovery `repair` policy). The RequestStatus is never modified
    by the plan; the plan only adds a budget charge at the repair
    admission (see Scheduler._recovery_plan_budget).
    """

    request_id: str
    # [rollback_boundary, corrupt_end): the physical repair span, in token
    # positions. This is what num_scheduled_tokens carries for the repair
    # step -- the worker's execution geometry stays physical.
    repair_span: tuple[int, int]
    # The budget-equivalent compute charge, in prefill-equivalent tokens
    # (E = ceil(T_repair / m(ctx))). NOT a physical token count: the
    # repair's replay is deeper per position than an ordinary prefill, and
    # the charge makes the step budgets pay for that difference.
    est_equivalent_tokens: int
    # Exclusive layer limit: the repair executes layers [0, layer_end) only
    # (the necessary range: below-CL hidden-state replay + the CL KV write).
    layer_end: int = 0
    # The pre-failure external computed count C: num_computed_tokens is
    # restored to this at the repair commit -- ONLY while the report
    # certified the above-span external ranges landed byte-exact
    # (healthy_above_span); otherwise the commit restores to the repair
    # span's end and the resume recomputes the untrusted above-span
    # extent.
    restored_extent: int = 0
    # The report's certification that the above-span external ranges
    # landed byte-exact. When False the commit's extent restoration is
    # CAPPED at the repair span's end (the largest commit-checked
    # boundary): the untrusted [span_end, C) content is never consumed
    # as cached; the ordinary resume recompute rewrites it.
    healthy_above_span: bool = True
    # The report's failed block ids (the abort path's eviction targets).
    failed_block_ids: list[int] = field(default_factory=list)
    # The plan identity: echoed by the repair result; a mismatched echo is
    # a stale/duplicate result and is rejected.
    plan_id: int = -1
    # The evidence identity of the report that built this plan
    # (Scheduler._report_evidence_key): the duplicate-absorption key for
    # same-step re-deliveries of the same fault across TP ranks.
    report_evidence: tuple | None = None
    # The identity of the transfer attempt whose failure built this
    # plan (the generation the admitting report echoed). Reports
    # arriving for a pending plan must match BOTH the evidence key and
    # this generation: a stale-generation report against a pending
    # plan is a late duplicate of the attempt the plan already
    # consumed (absorbed), never fresh divergence evidence.
    transfer_generation: int | None = None
    # The execution epoch: the scheduler step at which the plan was
    # admitted. A result echoing a different epoch is stale (the
    # overlapping-step rejection).
    admitted_step: int | None = None
    state: str = "DETECTED"
    num_deferrals: int = 0

    @property
    def n_eff(self) -> int:
        return self.repair_span[1] - self.repair_span[0]


class Scheduler(SchedulerInterface):
    # Inert defaults for partially-constructed instances: unit
    # tests build Scheduler via object.__new__ (no __init__) and
    # assign only the attributes they exercise, so attributes read
    # on the ordinary schedule()/update_from_output path must also
    # exist at the class level. __init__ assigns the real per-
    # instance state; on a bare instance every access below is an
    # empty or False read and the transfer-evidence machinery stays
    # inert until a report, attestation, or verdict exists.
    repair_kv_load_failures = False
    _evidence_state_initialized = False
    # Read by the ordinary schedule() path (plan-first exclusivity
    # scan); bare instances never write it -- every mutation is
    # gated on _evidence_state_initialized.
    recovery_plans: dict[str, RecoveryPlan] = {}

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
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.model_uses_mrope = vllm_config.model_config.uses_mrope
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        self.observability_config = vllm_config.observability_config
        self.spec_decode_metrics_level = (
            self.observability_config.per_request_spec_decode_metrics
        )
        self.kv_metrics_collector: KVCacheMetricsCollector | None = None
        if self.observability_config.kv_cache_metrics:
            self.kv_metrics_collector = KVCacheMetricsCollector(
                self.observability_config.kv_cache_metrics_sample,
            )
        self.structured_output_manager = structured_output_manager
        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder
        self.is_mm_encoder_only = vllm_config.is_mm_encoder_only

        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        self.finished_req_ids_dict: dict[int, set[str]] | None = (
            defaultdict(set) if include_finished_set else None
        )

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        # Admission into RUNNING. Separate from max_num_running_reqs, which
        # is the model-runner slot count.
        self.max_num_active_reqs = (
            self.scheduler_config.max_num_active_seqs
            if self.scheduler_config.max_num_active_seqs is not None
            else self.max_num_running_reqs
        )
        self.max_num_scheduled_tokens = (
            self.scheduler_config.max_num_scheduled_tokens
            if self.scheduler_config.max_num_scheduled_tokens is not None
            else self.scheduler_config.max_num_batched_tokens
        )
        self.adaptive_long_prefill_threshold = (
            self.scheduler_config.long_prefill_token_threshold_adaptive
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events
        )
        # Pooling and diffusion models do not append a sampled token after
        # prefill, so they should not reserve one context slot while scheduling.
        self.num_sampled_tokens_per_step = (
            0
            if (
                vllm_config.model_config.runner_type == "pooling"
                or vllm_config.model_config.is_diffusion
            )
            else 1
        )

        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        self.connector_prefix_cache_stats: PrefixCacheStats | None = None
        self.recompute_kv_load_failures = True
        self.repair_kv_load_failures = False
        # Delayed publication: request ids whose sync external-claim blocks
        # were left unpublished at admission under the repair policy;
        # published at the step's update once the delivery verdict is known.
        self._delayed_sync_claim_pubs: set[str] = set()
        # Positive transfer evidence (the hole-repair policy's
        # completeness gate): the number of TP ranks whose terminal
        # attestation a recovery authorization requires. A silent rank
        # -- one whose GET 'succeeded' against corrupt or stale content,
        # or one that failed before reporting -- supplies no report at
        # all, so reports alone cannot distinguish 'all ranks verified
        # their portion' from 'one rank verified and the rest said
        # nothing'. With PP>1 the repair policy is TP-scoped:
        # attestations are required from the TP group's ranks only.
        self._evidence_tp_size = vllm_config.parallel_config.tensor_parallel_size
        # Transfer identity (the report/attestation generation): a
        # globally monotonic counter assigned at the moment an external
        # load attempt is SCHEDULED -- early enough to accompany the
        # actual transfer and all its resulting evidence. Never reused,
        # including across request-ID reuse, so a late report cannot
        # alias onto a later attempt. request_id -> the generation of
        # its latest scheduled attempt.
        self._next_transfer_generation = 0
        self._req_transfer_generation: dict[str, int] = {}
        # request_id -> the EXTERNAL token
        # count of its latest scheduled attempt (the claim the evidence
        # vouches for). The report battery bounds restored_extent and
        # the failed span by it, and the completeness gate ties the
        # attestations' expected-object coverage to it: a misreporting
        # connector can no longer push the commit's restoration (and
        # with it publication) beyond the content the attempt actually
        # claimed. Recorded alongside the generation; dies with the
        # request (_free_request).
        self._req_transfer_claim: dict[str, int] = {}
        # The attempt's ABSOLUTE prefix extent (local hit + external
        # delta): the producer's failure-report geometry (token_span,
        # rollback_boundary, restored_extent) is request-absolute, while
        # the claim above is the external DELTA only. Mixed local+external
        # hits need both bounds: the span/restoration checks
        # below are bounded by the absolute extent; the attestation
        # coverage tie is bounded by the delta (the store objects the
        # producer actually GETs cover only the delta).
        self._req_transfer_claim_extent: dict[str, int] = {}
        # request_id -> the accumulated terminal attestations of the
        # request's LATEST scheduled attempt (validated, current-generation
        # only, identical duplicates absorbed). The publication gates read
        # this state at the delivery's decision points -- the sync claim's
        # publication step and the async request's promotion -- which may
        # be a LATER step than any single rank's attestation arrival (the
        # timing contract only bounds the LAST arrival), so evidence folds
        # here step by step instead of being judged per arrival. The list
        # is bounded by the TP group size: a divergent re-attestation from
        # one rank fails the request closed before it can accumulate, and
        # the state dies with the attempt (a new generation resets it),
        # with the request (_free_request), and with every fail-closed
        # rejection or consumed authorization.
        self._req_transfer_attestations: dict[str, list[KVLoadAttestation]] = {}
        # request_id -> failure reports of async attempts that are
        # still IN FLIGHT (no finished_recving yet). Judging their
        # completeness at arrival would reject deliveries whose evidence
        # is merely split across passes (rank A reports its failure before
        # rank B's healthy attestation arrives); the reports are re-examined
        # at the attempt's terminal pass instead (update_from_output
        # re-injects them once finished_recving arrives).
        self._req_pending_async_reports: dict[str, list[KVLoadFailureReport]] = {}
        # Update steps each deferred async report has remained
        # deferred (liveness bound; reset on each fresh deferral).
        self._req_async_report_deferrals: dict[str, int] = {}
        # Requests whose external transfer evidence failed closed at
        # least once: the connector is not re-consulted
        # for them -- the retry is a local recompute, never an unbounded
        # external re-attempt loop.
        self._untrusted_external_reqs: set[str] = set()
        # request_id -> the (transfer_generation, evidence key) pairs
        # of every recovery authorization that reached a terminal
        # disposition (committed, aborted, dissolved): a late
        # re-delivery of the SAME attempt's SAME evidence is a
        # duplicate whose outcome is already known and is absorbed --
        # while a genuine repeated failure carries a NEW generation (a
        # new scheduled attempt) and is honored.
        self._consumed_report_evidence: dict[str, list[tuple[int | None, tuple]]] = {}
        # Transactional publication: request ids whose repair-step
        # admissions were left unpublished at allocate_slots; they publish
        # ONLY on the step's accepted verdict, never on abort/missing/stale.
        self._delayed_repair_txn_pubs: set[str] = set()
        # Marks the transfer-evidence state above as initialized;
        # bare instances (class-level defaults above) stay inert.
        self._evidence_state_initialized = True
        # Requests whose num_computed_tokens moved other than by the
        # ordinary step advance and whose GPU-side count the worker must
        # restage at their next scheduling; drained into
        # SchedulerOutput.rewound_req_ids.
        self._pending_rewound_req_ids: set[str] = set()
        self.defer_block_free = False
        self.defer_async_loads = False
        # Whether a preempted request's in-flight output must be dropped; see
        # KVConnectorBase_V1.requires_kv_delivery.
        self.requires_kv_delivery = False
        kv_transfer_config = self.vllm_config.kv_transfer_config
        if kv_transfer_config is not None:
            assert not self.is_encoder_decoder, (
                "Encoder-decoder models are not currently supported with KV connectors"
            )
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config,
                role=KVConnectorRole.SCHEDULER,
                kv_cache_config=self.kv_cache_config,
            )
            if self.log_stats:
                self.connector_prefix_cache_stats = PrefixCacheStats()
            kv_load_failure_policy = kv_transfer_config.kv_load_failure_policy
            self.recompute_kv_load_failures = kv_load_failure_policy == "recompute"
            self.repair_kv_load_failures = kv_load_failure_policy == "repair"
            if self.repair_kv_load_failures and (
                vllm_config.parallel_config.pipeline_parallel_size > 1
            ):
                # The v1 evidence contract is TP-scoped: the
                # completeness gate demands terminal attestations from
                # exactly the TP group's rank indices, but the output
                # aggregator stamps source_rank over every gathered
                # worker (tp*pp under PP>1) -- a count-satisfying set of
                # the WRONG stage's workers would pass a weaker check
                # while the shard-holding stage stays silent. Fail fast at
                # init instead of shipping a false-commit path; PP support
                # needs the request's actual TP-group rank set derived in
                # the executor.
                raise RuntimeError(
                    "kv_load_failure_policy='repair' requires "
                    "pipeline_parallel_size=1 in v1 (the transfer-evidence "
                    "completeness gate is TP-scoped)"
                )
            if self.repair_kv_load_failures and (
                vllm_config.parallel_config.data_parallel_size > 1
            ):
                # The repair TP consensus gathers over
                # the WORLD process group (the worker-side all_gather has
                # no TP-group handle in v1). Under DP>1 each DP rank
                # schedules independently: a repair on one DP rank would
                # gather foreign ranks' (different-work) summaries --
                # guaranteed divergence -- or peers never enter the
                # collective, deadlocking the engine. Reject at init,
                # exactly like PP>1, until the collective is scoped to
                # the TP group.
                raise RuntimeError(
                    "kv_load_failure_policy='repair' requires "
                    "data_parallel_size=1 in v1 (the repair verdict "
                    "consensus is TP-group-scoped)"
                )
            if self.repair_kv_load_failures and (
                len(self.kv_cache_config.kv_cache_groups) > 1
            ):
                # The report consumer, the write-set
                # derivation, and the block-table geometry are all
                # single-group-shaped (the per-group block-id unpack in
                # _repair_write_set_block_ids would raise ValueError --
                # an engine crash, not a refusal). Reject at init so the
                # documented single-group scope is loud, not latent.
                raise RuntimeError(
                    "kv_load_failure_policy='repair' requires a single "
                    "KV cache group in v1 (the repair write-set and report "
                    "geometry are single-group-shaped)"
                )
            if self.repair_kv_load_failures and (
                vllm_config.parallel_config.prefill_context_parallel_size > 1
                or vllm_config.parallel_config.decode_context_parallel_size > 1
            ):
                # Under context parallelism the positions tensor carries
                # CP-local (not absolute) token positions, so the executor's
                # span validation cannot hold. Reject at init until the
                # repair path learns CP-local geometry.
                raise RuntimeError(
                    "kv_load_failure_policy='repair' is not supported with "
                    "context parallelism in v1 (repair spans are expressed "
                    "in absolute positions)"
                )
            if self.repair_kv_load_failures and (
                self.scheduler_config.async_scheduling
            ):
                # Repair-only steps return their verdicts from execute_model
                # without setting execute_model_state, but the batch-queue
                # step calls sample_tokens unconditionally; sample_tokens
                # then returns None, the engine discards the repair verdicts
                # riding the execution future, and raises at the batch-queue
                # None check. Fail fast at init instead (the async-aware
                # repair step shape is v2 work).
                raise RuntimeError(
                    "kv_load_failure_policy='repair' is not supported with "
                    "async scheduling in v1 (the batch-queue step would "
                    "discard the repair verdicts)"
                )

            # With overlapping batches (async scheduling or PP), a step may
            # still be writing a freed request's KV blocks. A consumer KV
            # Connector can reallocate and fill those blocks via a load that
            # isn't ordered against that write, so defer freeing them.
            multiple_inflight_batches = self.vllm_config.max_concurrent_batches > 1
            if multiple_inflight_batches and kv_transfer_config.is_kv_consumer:
                self.defer_block_free = True
            # HiSparse keeps reading a released page until the pool reuses it,
            # so a load into a reused page must also wait for in-flight steps.
            self.defer_async_loads = (
                self.defer_block_free
                and kv_cache_config.hisparse_host_num_blocks is not None
            )

            self.requires_kv_delivery = self.connector.requires_kv_delivery

        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,
            self.parallel_config.data_parallel_index,
        )
        self.ec_connector = None
        if self.vllm_config.ec_transfer_config is not None:
            self.ec_connector = ECConnectorFactory.create_connector(
                config=self.vllm_config, role=ECConnectorRole.SCHEDULER
            )

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = block_size
        self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        self.pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size

        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Scheduling policy
        try:
            self.policy = SchedulingPolicy(self.scheduler_config.policy)
        except ValueError as e:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}"
            ) from e
        # Priority queues for requests.
        self.waiting = create_request_queue(self.policy)
        # Waiting requests that hold KV blocks are always drained before
        # self.waiting: a block-holder must never sit behind a request whose
        # failed allocation would stop the scheduling scan.
        self.kv_holding_waiting = create_request_queue(self.policy)
        # With either async-load limit, keep requests that may load KV async
        # separate so throttling their loads doesn't block local requests.
        self.async_kv_waiting = create_request_queue(self.policy)
        # Waiting requests that are deferred, i.e. were skipped by the scheduling
        # scan or enqueued in a blocked status.
        self.deferred_waiting: set[Request] = set()
        self.running: list[Request] = []

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()

        # IDs of requests preempted since the last call to schedule().
        self.reset_preempted_req_ids: set[str] = set()

        # Counter for requests waiting for streaming input. Used to calculate
        # number of unfinished requests
        self.num_waiting_for_streaming_input: int = 0

        # KV Connector: requests in process of async KV loading or recving
        self.finished_recving_kv_req_ids: set[str] = set()
        self.failed_recving_kv_req_ids: set[str] = set()

        # Hole recovery: pending repair plans, keyed by request id. A plan
        # adds an equivalent-token budget charge at the repair admission.
        self.recovery_plans: dict[str, RecoveryPlan] = {}
        self._next_repair_plan_id = 0

        # Grammar compilation failures to finish as per-request errors in
        # update_from_output.
        self.grammar_compile_error_reqs: dict[str, RequestError] = {}
        self.encoder_cache_mismatch_reqs: set[str] = set()

        # Encoder-related.
        # Calculate encoder cache size if applicable
        supports_mm_inputs = vllm_config.model_config.supports_multimodal_inputs
        mm_budget = (
            MultiModalBudget(vllm_config, mm_registry) if supports_mm_inputs else None
        )

        # NOTE: Text-only encoder-decoder models are implemented as
        # multi-modal models for convenience
        # Example: https://github.com/vllm-project/bart-plugin
        if self.is_encoder_decoder:
            assert mm_budget and len(mm_budget.mm_max_toks_per_item) <= 1, (
                "Encoder-decoder models are expected to implement the "
                "multimodal interface with at most one modality."
            )

        self.max_num_encoder_input_tokens = (
            mm_budget.encoder_compute_budget if mm_budget else 0
        )
        encoder_cache_size = mm_budget.encoder_cache_size if mm_budget else 0
        manager_cls_obj = vllm_config.ec_manager_config.get_encoder_cache_manager_obj()
        if manager_cls_obj is None:
            manager_cls_obj = (
                EncoderDecoderCacheManager
                if self.is_encoder_decoder
                else EncoderCacheManager
            )
        self.encoder_cache_manager = manager_cls_obj.create_manager(
            cache_size=encoder_cache_size, vllm_config=vllm_config
        )
        speculative_config = vllm_config.speculative_config
        self.use_eagle = False
        self.use_eagle_block_drop = False
        self.num_spec_tokens = vllm_config.num_scheduled_spec_tokens
        self.num_lookahead_tokens = vllm_config.num_lookahead_tokens
        # DSV41 SWA bounded replay: groups that declare a replay window are rebuilt
        # after a prefix hit by recomputing its trailing tokens. One window
        # for all such groups, so the rewind matches every group's allocation.
        self.prefix_replay_group_ids = tuple(
            group_id
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
            if group.kv_cache_spec.prefix_replay_tokens > 0
        )
        replay_windows = {
            kv_cache_config.kv_cache_groups[group_id].kv_cache_spec.prefix_replay_tokens
            for group_id in self.prefix_replay_group_ids
        }
        assert len(replay_windows) <= 1, (
            f"Prefix replay windows should agree: {sorted(replay_windows)}"
        )
        self.prefix_replay_tokens = replay_windows.pop() if replay_windows else 0
        self.num_prefill_lookahead = vllm_config.num_prefill_lookahead_tokens
        # The model runner skips drafting in steps where no scheduled request
        # can use the drafts. Only drafters with their own forward pass reserve
        # lookahead slots.
        self.can_skip_drafts = self.num_lookahead_tokens > 0
        self.dynamic_sd_lookup: list[int] | None = None
        if speculative_config is not None:
            if speculative_config.num_speculative_tokens_per_batch_size:
                self.dynamic_sd_lookup = build_dynamic_sd_schedule_lookup(
                    speculative_config.num_speculative_tokens_per_batch_size,
                    vllm_max_batch_size=self.scheduler_config.max_num_seqs,
                    vllm_num_speculative_tokens=self.num_spec_tokens,
                )
            self.use_eagle = speculative_config.uses_target_hidden_states
            self.use_eagle_block_drop = speculative_config.use_eagle_block_drop()
            if (
                speculative_config.needs_eagle_block_drop()
                and speculative_config.disable_eagle_block_drop
            ):
                logger.warning(
                    "EAGLE trailing prefix-cache block dropping is disabled. "
                    "This is experimental and may affect speculative-token "
                    "acceptance rates."
                )

        self.num_hash_lookahead_tokens = vllm_config.num_hash_lookahead_tokens
        if self.num_hash_lookahead_tokens:
            # Lookahead hashes prove the boundary token, so EAGLE needs no drop.
            self.use_eagle_block_drop = False

        # Create the KV cache manager.
        if hash_block_size is None:
            hash_block_size = block_size
        self.hash_block_size = hash_block_size
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            max_in_flight_tokens=vllm_config.max_in_flight_tokens,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle_block_drop,
            num_hash_lookahead_tokens=self.num_hash_lookahead_tokens,
            num_prefill_lookahead=self.num_prefill_lookahead,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=self.dcp_world_size,
            pcp_world_size=1,
            scheduler_block_size=self.block_size,
            hash_block_size=hash_block_size,
            metrics_collector=self.kv_metrics_collector,
            watermark=self.scheduler_config.watermark,
            enable_mamba_shared_prefix_checkpoint=(
                self.cache_config.enable_mamba_shared_prefix_checkpoint
            ),
        )
        # Mamba managers already account for their speculative state blocks.
        num_spec_decode_tokens = (
            1 + self.num_spec_tokens + self.num_lookahead_tokens
            if self.num_spec_tokens
            else 0
        )
        self._spec_decode_step_blocks = sum(
            cdiv(num_spec_decode_tokens, manager.block_size)
            for manager in self.kv_cache_manager.coordinator.single_type_managers
            if not isinstance(manager.kv_cache_spec, MambaSpec)
        )
        # Bind after construction so connectors can access the cache manager.
        if self.connector is not None:
            self.connector.bind_kv_cache_manager(self.kv_cache_manager)

        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
        self._pending_encoder_input_ids: dict[str, set[int]] = {}
        # Scheduler iteration counter. Drives the PP+async decode-throttle
        # cadence (`next_decode_eligible_step`).
        self.current_step = 0
        # DP prefill balancing: Flag to track whether the last cadence-aligned
        # prefill batch fully drained the waiting queue. Prefill throttling
        # is disabled in this case.
        self.prefill_capacity_bound = False
        self.scheduler_reserve_full_isl = (
            self.scheduler_config.scheduler_reserve_full_isl
        )
        # Cache-aware admission needs a prefix cache to consult, and reorders by
        # position, which only a FCFS queue has.
        self.cache_aware_window = (
            self.scheduler_config.cache_aware_admission_window
            if self.policy == SchedulingPolicy.FCFS
            and self.cache_config.enable_prefix_caching
            else 0
        )
        self.cache_aware_threshold = (
            self.scheduler_config.cache_aware_admission_threshold
        )

        self.has_mamba_layers = kv_cache_config.has_mamba_layers
        self.needs_kv_cache_zeroing = kv_cache_config.needs_kv_cache_zeroing
        self.need_mamba_block_aligned_split = (
            self.has_mamba_layers and self.cache_config.mamba_cache_mode == "align"
        )
        # TODO: Support models with multiple Mamba specs that require different
        # prefill checkpoint alignments instead of selecting the first one.
        self.mamba_prefill_checkpoint_alignment = next(
            (
                group.kv_cache_spec.prefill_checkpoint_alignment
                for group in kv_cache_config.kv_cache_groups
                if isinstance(group.kv_cache_spec, MambaSpec)
            ),
            None,
        )
        self.mamba_has_prefill_checkpoint_blocks = self.has_mamba_layers and all(
            not isinstance(group.kv_cache_spec, MambaSpec)
            or group.kv_cache_spec.num_prefill_checkpoint_blocks > 0
            for group in kv_cache_config.kv_cache_groups
        )
        # Fine-grained hits reuse the prompt's hash-aligned Mamba checkpoint.
        # Split there only if the backend cannot save it within the final chunk.
        self.mamba_partial_cache_hit = (
            self.need_mamba_block_aligned_split
            and self.hash_block_size < self.block_size
            and self.kv_cache_manager.coordinator.enable_partial_hash_hits
        )
        # Opt-in: also stop at the junction, where an eagle sibling resumes. The
        # manager decides whether it can check-point there (per-group eagle bit,
        # no MTP re-prefill tail); splitting for a stop it would refuse costs a
        # forward pass and displaces the block-boundary stop.
        self.mamba_shared_prefix_checkpoint = (
            self.mamba_partial_cache_hit
            and self.kv_cache_manager.mamba_shared_prefix_checkpoint
        )

        # Counts of non-empty steps scheduled / processed. update_from_output
        # is called once per scheduled step in FIFO order, so these stay in sync.
        self.sched_step_seq = 0
        self.processed_step_seq = 0
        # FIFO of (fence_seq, blocks): blocks become safe to free once
        # processed_step_seq >= fence_seq.
        self.deferred_frees: deque[tuple[int, list[KVCacheBlock]]] = deque()
        # Request id -> (fence_seq, num_external_tokens, connector prefix cache
        # queries, hits): async KV loads started once processed_step_seq >=
        # fence_seq.
        self.deferred_loads: dict[str, tuple[int, int, int, int]] = {}

        self.perf_metrics: ModelMetrics | None = None
        if self.log_stats and vllm_config.observability_config.enable_mfu_metrics:
            self.perf_metrics = ModelMetrics(vllm_config)

        self.aux_output_connector = (
            AuxOutputSchedulerConnector(self.kv_cache_manager, self.connector)
            if vllm_config.aux_output_config.enabled
            else None
        )
        self.return_sampling_mask = vllm_config.model_config.return_sampling_mask

        self._pause_state: PauseState = PauseState.UNPAUSED

        # Remaining-block reservations of in-flight prefills gate async loads.
        # Parked async KV loads can number thousands, so theirs are cached until
        # they are scheduled; prefill chunks are few and summed on demand.
        self._inflight_prefills: set[Request] = set()
        self._parked_reservations: dict[Request, int] = {}
        self._parked_reserved_blocks = 0

        # Requests with an async KV load, by stage (KV_FETCH_STAGES).
        # None when not tracked (no KV connector, or no stats or load cap).
        self._kv_fetch_stages: dict[Request, str] | None = (
            {}
            if self.connector is not None
            and (self.log_stats or self.scheduler_config.max_num_inflight_kv_loads)
            else None
        )
        self._kv_fetch_counts: dict[str, int] = dict.fromkeys(KV_FETCH_STAGES, 0)
        self._aborted_kv_loads: set[str] = set()

    def _drafts_unused(
        self, request: Request, num_computed_tokens: int, num_new_tokens: int
    ) -> bool:
        """Whether the drafts proposed for `request` after this step are sure to
        go unused.

        Drafts proposed after a step are verified in the request's next decode
        step. They go unused when no such step follows, in two cases:

        1. This step is a non-final prefill chunk. It samples no token, and the
           scheduler drops drafts proposed after a prefill chunk.
        2. This step samples the request's last output token: even if every
           draft is rejected, the request reaches `max_tokens` and finishes.

        Stop tokens and stop strings also end requests, but they cannot be known
        in advance. This returns True only when it is sure.

        The model runner proposes no drafts (K=0) in a step where this is True
        for every scheduled request on every DP rank.

        Args:
            request: A request scheduled in this step.
            num_computed_tokens: Tokens computed before this step, including
                prefix-cache and KV-connector hits for a newly admitted request.
            num_new_tokens: Tokens scheduled for this step.

        """
        # Case 1: a non-final prefill chunk stops before the request's last
        # known token, so this step samples nothing.
        if num_computed_tokens + num_new_tokens < request.num_tokens:
            return True

        # Case 2: this step samples the request's last output token. Drafts may
        # all be rejected, so count only the output tokens the request is sure
        # to have once this step finishes: the verified ones, plus one from each
        # step that samples. This step samples one.
        min_num_output_tokens = request.num_output_tokens + 1
        if request.num_output_placeholders > 0:
            # Async scheduling: a step that samples is still in flight, and it
            # adds one more.
            min_num_output_tokens += 1
        # Reaching max_tokens finishes the request, so no step verifies the drafts.
        return min_num_output_tokens >= request.max_tokens

    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        """Clip a prefill chunk so it ends where Mamba state must be cached.

        In "align" cache mode reusable SSM states are materialized at block
        boundaries, plus mandatory early stops (the prompt's partial-tail hash
        boundary, a detected shared-prefix junction). If a block is larger
        than the configured prefill chunk limit, intermediate chunks keep
        private running state until they reach the next cacheable position.
        """
        start = (
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        # Split only during prefill: `request.num_tokens - 1` extends this to
        # resumed requests replaying their output tokens.
        prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
        if start >= prefill_end:
            return num_new_tokens

        block_size = self.cache_config.block_size
        # The last block-aligned position whose state can be cached. With
        # Eagle, FullAttn prunes the last matching block, so back off one
        # block to avoid a Mamba cache miss.
        ckpt_block_size = (
            self.hash_block_size if self.mamba_partial_cache_hit else self.block_size
        )
        last_cache_position = get_mamba_prefill_checkpoint_position(
            request.num_tokens,
            block_size,
            self.use_eagle_block_drop,
            num_hash_lookahead_tokens=self.num_hash_lookahead_tokens,
        )

        end = start + num_new_tokens
        checkpoint_position = get_mamba_prefill_checkpoint_position(
            prefill_end,
            ckpt_block_size,
            drop_eagle_block=self.use_eagle_block_drop,
            num_hash_lookahead_tokens=self.num_hash_lookahead_tokens,
        )
        use_internal_checkpoint = (
            self.mamba_has_prefill_checkpoint_blocks
            and end >= prefill_end
            and is_mamba_prefill_checkpoint_valid(
                query_start=start,
                query_end=end,
                checkpoint_position=checkpoint_position,
                hash_block_size=self.hash_block_size,
                mamba_block_size=block_size,
                checkpoint_alignment=self.mamba_prefill_checkpoint_alignment,
            )
        )
        if use_internal_checkpoint:
            last_cache_position = 0
        # Invariant: slot p holds the state after exactly (p + 1) * block_size
        # tokens. State is written at chunk ends, so chunk ends must be block
        # aligned. Exempt: the prompt's last chunk, whose slot decode advances
        # to the boundary. A block too wide for one chunk advances sub-block
        # and re-aligns at the next boundary.
        if end < prefill_end:
            max_prefill_tokens = self.max_num_scheduled_tokens
            long_prefill_threshold = self.scheduler_config.long_prefill_token_threshold
            if long_prefill_threshold > 0:
                max_prefill_tokens = min(max_prefill_tokens, long_prefill_threshold)
            aligned_end = end // block_size * block_size
            if aligned_end > start or block_size <= max_prefill_tokens:
                end = aligned_end

        next_block_boundary = (start // block_size + 1) * block_size
        tail_boundary = (
            get_mamba_prefill_checkpoint_position(
                request.num_prompt_tokens,
                self.hash_block_size,
                self.use_eagle_block_drop,
                num_hash_lookahead_tokens=self.num_hash_lookahead_tokens,
            )
            if self.mamba_partial_cache_hit and not use_internal_checkpoint
            else 0
        )
        junction = request.shared_prefix_boundary
        # Block-floored: a sub-block junction's state is not separately cacheable.
        block_floored = start + (junction - start) // block_size * block_size
        # Past the prompt the manager writes nothing, so fall back to the
        # block-floored stop rather than dropping it: a resumed request replaying
        # output tokens can still observe a junction there.
        junction_stop = (
            junction
            if self.mamba_shared_prefix_checkpoint
            and junction <= request.num_prompt_tokens
            else block_floored
        )
        stops = (
            # Same invariant: a chunk starting mid-block stops at the boundary
            # rather than running past it.
            next_block_boundary
            if start % block_size != 0 and not use_internal_checkpoint
            else 0,
            # Never run past the last cacheable block boundary mid-chunk.
            last_cache_position,
            # Stop at the prompt checkpoint unless the backend saves it internally.
            tail_boundary
            if last_cache_position < tail_boundary < request.num_prompt_tokens
            else 0,
            # Marconi shared-prefix junction: cache its state so sibling
            # requests sharing the prefix can reuse it.
            junction_stop if start < junction < end else 0,
        )
        # Stop at the earliest mandatory position strictly inside the chunk.
        end = min((s for s in stops if start < s < end), default=end)
        return max(end - start, 0)

    def _get_local_prefix_cache_hit(
        self, request: Request, max_cache_hit_length: int | None = None
    ) -> tuple[KVCacheBlocks, int, int, bool]:
        connector = self.connector
        if connector is not None and connector.supports_divergent_local_hybrid_hits:
            return self.kv_cache_manager.get_computed_blocks_for_connector(
                request, max_cache_hit_length
            )

        blocks, num_local, shared_prefix_boundary = (
            self.kv_cache_manager.get_computed_blocks(request, max_cache_hit_length)
        )
        return blocks, num_local, shared_prefix_boundary, False

    def _reserve_prefill_lookahead(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
    ) -> int:
        """Never end a prefill chunk within num_prefill_lookahead of the
        prefill end.

        At a chunked-prefill boundary, the multi-module MTP drafter consumes
        the next num_prefill_lookahead known prefill tokens as draft inputs. A
        boundary closer to the end than that would make it fall back to
        sampled drafts, permanently polluting the trailing modules' KV caches.
        Either finish the prefill or leave at least num_prefill_lookahead for
        the next chunk. No-op for eagle-family drafters (lookahead 1).
        """
        remaining = request.num_tokens - num_computed_tokens - num_new_tokens
        if 0 < remaining < self.num_prefill_lookahead:
            num_new_tokens -= self.num_prefill_lookahead - remaining
        return max(num_new_tokens, 0)

    def _is_full_forward_repair(self, plan: RecoveryPlan) -> bool:
        return (
            self.vllm_config.speculative_config is None
            and len(self.kv_cache_config.kv_cache_groups) == 1
            and isinstance(
                self.kv_cache_config.kv_cache_groups[0].kv_cache_spec, FullAttentionSpec
            )
            and plan.layer_end
            == self.vllm_config.model_config.get_num_layers(self.parallel_config)
        )

    def _recovery_plan_budget(
        self,
        request: Request,
        num_new_tokens: int,
        input_budget: int,
        draft_slots: int,
    ) -> tuple[int, int] | None:
        """Hole-repair budget admission for a pending recovery plan.

        The plan's physical repair span (n_eff) is scheduled as ordinary
        tokens; its equivalent-token compute charge is debited from the
        step's input budget as an additional surcharge beside
        num_new_tokens -- the draft_slots precedent (a per-request debit
        that is NOT part of num_scheduled_tokens, so the worker's
        execution geometry stays physical).

        Returns (num_new_tokens, surcharge), or None when the request must
        be deferred this step (the plan stays pending). A plan no step
        could ever fund is rejected: it dissolves and the ordinary
        rollback-recompute reschedule owns the request (the configured
        failure-policy floor).
        """
        plan = self.recovery_plans.get(request.request_id)
        if plan is None:
            return num_new_tokens, 0
        if plan.state != "DETECTED":
            # ADMITTED: the repair step is scheduled and its verdict is
            # pending. Defer the request until the verdict is consumed:
            # with the batch queue (async scheduling) the next schedule()
            # runs before the verdict, and rows scheduled beside the
            # pending plan would race the commit's num_computed_tokens
            # restoration (double execution, out-of-order samples).
            # The verdict always arrives -- the repair step's output is
            # queued ahead of any later step, and a missing result is
            # synthesized as a failed verdict at that step's update.
            return None
        # Static never-admissible gate. The running arm reserves the
        # plan's full equivalent-token charge before the span cap (the
        # v1 constraint: budget - E >= n_eff, i.e. E + n_eff must fit
        # the input budget beside the draft slots); the waiting arm has
        # no reserve, so only the larger of the two charges must fit.
        # A plan no step could ever fund (spec-decode draft slots
        # reserving head-room, a scheduled-tokens cap below the span)
        # dissolves immediately instead of deferring forever while the
        # pending-plan drain freezes the whole engine.
        if self._is_full_forward_repair(plan):
            # Remaining capacity can defer this request, but cannot make a
            # feasible repair permanently inadmissible.
            never_fits = max(plan.est_equivalent_tokens, plan.n_eff) + draft_slots > (
                self.scheduler_config.max_num_batched_tokens
            )
        elif request.status == RequestStatus.RUNNING:
            never_fits = (
                plan.est_equivalent_tokens + plan.n_eff + draft_slots > input_budget
            )
        else:
            never_fits = (
                max(plan.est_equivalent_tokens, plan.n_eff) + draft_slots > input_budget
            )
        if (
            never_fits
            or plan.est_equivalent_tokens > self.scheduler_config.max_num_batched_tokens
            or plan.n_eff > self.max_num_scheduled_tokens
        ):
            self._dissolve_recovery_plan(plan, "cannot fit any step budget")
            # Skip this step for the owner rather than scheduling an
            # ordinarily-capped chunk: the long-prefill threshold was
            # EXEMPTED for this request while the plan existed, so
            # returning the uncapped catch-up here would let one
            # dissolution step bypass the threshold.
            # The next step schedules ordinarily, threshold applied.
            return 0, 0
        # The repair step schedules exactly the repair span.
        num_new_tokens = min(num_new_tokens, plan.n_eff)
        surcharge = max(0, plan.est_equivalent_tokens - plan.n_eff)
        num_new_tokens = min(num_new_tokens, input_budget - draft_slots - surcharge)
        if num_new_tokens < plan.n_eff:
            # Defer the request this step (the plan stays pending). The
            # deferral COUNT is ticked once per schedule() step for
            # every DETECTED plan that was not admitted (see
            # _tick_recovery_plan_deferrals), NOT here: owners skipped
            # before this call -- blocked remote-KV waits, paused
            # sessions, DP-balancing and PP-cadence skips -- must be
            # bounded identically, and they never reach this branch.
            return None
        return num_new_tokens, surcharge

    def _tick_recovery_plan_deferrals(
        self, num_scheduled_tokens: dict[str, int]
    ) -> None:
        """The step-level deferral tick: every DETECTED plan that was not
        admitted THIS step -- whatever the reason: a budget deferral, an
        owner skipped before its budget call, a blocked remote-KV wait,
        a paused streaming session -- moves one step closer to the
        bound. This is the uniform liveness guard: no plan, and
        therefore no engine-wide drain-first freeze, can outlive
        REPAIR_MAX_ADMISSION_DEFERRALS steps without either admitting
        or dissolving to the recompute fallback (which chunks, and
        therefore always makes progress)."""
        for req_id in list(self.recovery_plans):
            plan = self.recovery_plans[req_id]
            if plan.state != "DETECTED" or req_id in num_scheduled_tokens:
                continue
            plan.num_deferrals += 1
            if plan.num_deferrals >= self.REPAIR_MAX_ADMISSION_DEFERRALS:
                self._dissolve_recovery_plan(plan, "admission deferral bound")

    def _dissolve_recovery_plan(self, plan: RecoveryPlan, reason: str) -> None:
        """Drop a DETECTED plan and hand the request to the ordinary
        rollback-recompute path (num_computed_tokens already sits at the
        rollback boundary). The request keeps running; only the repair
        attempt is abandoned. The dissolved plan's evidence is recorded
        as consumed: a late re-delivery of it is
        absorbed, never re-authorized on top of the fallback."""
        self.recovery_plans.pop(plan.request_id, None)
        self._remember_consumed_evidence(plan)
        logger.warning(
            "Repair plan dissolved: req=%s plan_id=%d span=%s E=%d "
            "deferrals=%d (%s) -> the ordinary recompute path owns the "
            "request",
            plan.request_id,
            plan.plan_id,
            plan.repair_span,
            plan.est_equivalent_tokens,
            plan.num_deferrals,
            reason,
        )

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        self.current_step += 1
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_new_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        iteration_details = None
        if (
            self.log_stats
            and self.observability_config.enable_logging_iteration_details
        ) or self.vllm_config.profiler_config.profiler is not None:
            iteration_details = IterationDetails()

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        repair_charges: dict[str, int] = {}
        repair_admitted = False
        repair_exclusive = False
        # Partial-depth replay still drains an exclusive batch. Full-depth
        # attention-only repair participates in ordinary admission order.
        repair_pending = any(
            plan.state == "DETECTED" and not self._is_full_forward_repair(plan)
            for plan in self.recovery_plans.values()
        )
        token_budget = self.max_num_scheduled_tokens
        spec = self.vllm_config.speculative_config
        draft_slots = spec.max_num_new_slots_for_drafting if spec is not None else 0
        input_budget = self.scheduler_config.max_num_batched_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
            # Do not schedule any requests when paused.
            token_budget = 0

        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        num_scheduled_encoder_inputs = 0
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}
        # Whether a request scheduled this step can use the drafts proposed after
        # it. Stays set if that request is preempted later in the step.
        drafts_needed = not self.can_skip_drafts
        # Whether the running batch contains any prefill requests.
        prefill_scheduled = False
        # Whether any scheduled request has a synchronous connector KV load.
        has_sync_kv_loads = False

        # For logging.
        scheduled_timestamp = time.monotonic()

        self.kv_cache_manager.new_step_starts()

        # DP prefill balancing: on a throttled (non-cadence-aligned) step, defer
        # all prefill compute unless saturated.
        defer_prefills = (
            throttle_prefills and not self.prefill_capacity_bound
        ) and any(not r.is_prefill_chunk for r in self.running)

        # `long_prefill_token_threshold` exists to stop a long prefill from
        # starving other requests of the token budget. When it is the only
        # request there is nobody to starve, so let it use the whole budget.
        num_running, num_waiting = self.get_request_counts()
        num_eligible_reqs = num_running + num_waiting
        long_prefill_token_threshold = (
            self.scheduler_config.long_prefill_token_threshold
            if self.scheduler_config.enable_chunked_prefill and num_eligible_reqs > 1
            else 0
        )
        if long_prefill_token_threshold > 0 and self.adaptive_long_prefill_threshold:
            # Floor the cap at a fair share of the input budget so it never
            # cuts a request below max_num_batched_tokens / num requests.
            long_prefill_token_threshold = max(
                long_prefill_token_threshold, input_budget // num_eligible_reqs
            )

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]
            if input_budget <= draft_slots:
                break
            if repair_admitted and (
                repair_exclusive or request.request_id in self.recovery_plans
            ):
                # One repair per step; only partial replay blocks healthy rows.
                num_new_tokens = 0
                req_index += 1
                continue
            if repair_pending and request.request_id not in self.recovery_plans:
                # The pending plan drains first; ordinary running work
                # waits one step. The damaged request itself must fall
                # through to the budget call below: freezing it here
                # would skip its own admission (deadlock).
                num_new_tokens = 0
                req_index += 1
                continue

            if (
                request.num_output_placeholders > 0
                # This is (num_computed_tokens + 1) - (num_output_placeholders - 1).
                # Since output placeholders are also included in the computed tokens
                # count, we subtract (num_output_placeholders - 1) to remove any draft
                # tokens, so that we can be sure no further steps are needed even if
                # they are all rejected.
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                # Async scheduling: Avoid scheduling an extra step when we are sure that
                # the previous step has reached request.max_tokens. We don't schedule
                # partial draft tokens since this prevents uniform decode optimizations.
                req_index += 1
                continue

            if self.current_step < request.next_decode_eligible_step:
                # PP+async: enforce `pp_size` steps between same-req decodes
                # to match worker-side sampled-tokens broadcast slot ring cadence.
                req_index += 1
                continue

            if defer_prefills and request.is_prefill_chunk:
                # DP prefill balancing: defer this in-progress prefill chunk to a
                # cadence-aligned step; decodes still run to fill this step.
                req_index += 1
                continue

            if (
                self.ec_connector is not None
                and request.mm_features
                and not self.ec_connector.ensure_cache_available(
                    request,
                    request.num_computed_tokens - request.num_output_placeholders,
                )
            ):
                req_index += 1
                continue

            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            _pending_plan = self.recovery_plans.get(request.request_id)
            if (
                0 < long_prefill_token_threshold < num_new_tokens
                # A repair span must fit whole. Its equivalent-token charge
                # still shares the ordinary budget with healthy requests.
                and _pending_plan is None
            ):
                num_new_tokens = long_prefill_token_threshold
            # Reserve the pending plan's equivalent-token charge before the
            # cap, or the later surcharge debit would not fit the budget.
            _repair_reserve = (
                (
                    max(0, _pending_plan.est_equivalent_tokens - _pending_plan.n_eff)
                    if self._is_full_forward_repair(_pending_plan)
                    else _pending_plan.est_equivalent_tokens
                )
                if (_pending_plan is not None and _pending_plan.state == "DETECTED")
                else 0
            )
            num_new_tokens = min(
                num_new_tokens,
                token_budget,
                input_budget - draft_slots - _repair_reserve,
            )

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len
                - request.num_computed_tokens
                - self.num_sampled_tokens_per_step,
            )

            # Apply Mamba alignment before encoder caps.
            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request, num_new_tokens
                )

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            external_load_encoder_input: list[int] = []
            duplicate_encoder_inputs: list[int] = []
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                    external_load_encoder_input,
                    duplicate_encoder_inputs,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                    shift_computed_tokens=self.num_prefill_lookahead,
                )

            # Multi-module MTP: avoid ending a prefill chunk within
            # num_prefill_lookahead of the prefill end.
            num_new_tokens = self._reserve_prefill_lookahead(
                request, request.num_computed_tokens, num_new_tokens
            )

            # Charge the plan's equivalent-token surcharge, or defer the
            # request this step (falling into the skip below).
            recovery_surcharge = 0
            recovery = self._recovery_plan_budget(
                request, num_new_tokens, input_budget, draft_slots
            )
            if recovery is None:
                num_new_tokens = 0
            else:
                num_new_tokens, recovery_surcharge = recovery
            if repair_pending and not self.recovery_plans:
                # The plan was dissolved at the gate: the constraint lifts.
                repair_pending = False

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # 4. Insufficient budget for a block-aligned chunk in hybrid
                #    models with mamba cache mode \"align\".
                # 5. Insufficient budget to keep a multi-module MTP prefill
                #    chunk out of the prefill-lookahead window.
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            # Schedule newly needed KV blocks for the request.
            with record_function_or_nullcontext("schedule: allocate_slots"):
                # A repair transaction's admission must not publish its
                # blocks before the verdict: nothing whose validity depends
                # on an unresolved transaction may be matchable.
                repair_plan = self.recovery_plans.get(request.request_id)
                delay_repair_txn = (
                    self.repair_kv_load_failures
                    and repair_plan is not None
                    and repair_plan.state in ("DETECTED", "ADMITTED")
                )
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                        delay_cache_blocks=delay_repair_txn,
                    )

                    if new_blocks is not None:
                        # The request can be scheduled.
                        if delay_repair_txn:
                            self._delayed_repair_txn_pubs.add(request.request_id)
                        break

                    if (
                        self.connector is not None
                        and self.connector.has_pending_block_frees()
                    ):
                        break

                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                    else:
                        preempted_req = self.running[-1]

                    # A deferred free will not help with immediate allocation.
                    if not self._request_blocks_can_be_freed(preempted_req):
                        break

                    if self.policy == SchedulingPolicy.PRIORITY:
                        victim_index = self.running.index(preempted_req)
                        del self.running[victim_index]
                        if victim_index < req_index:
                            req_index -= 1

                        if preempted_req in scheduled_running_reqs:
                            preempted_req_id = preempted_req.request_id
                            scheduled_running_reqs.remove(preempted_req)
                            restored = num_scheduled_tokens.pop(preempted_req_id)
                            restored_charge = repair_charges.pop(preempted_req_id, 0)
                            token_budget += restored
                            input_budget += restored + draft_slots + restored_charge
                            req_to_new_blocks.pop(preempted_req_id)
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                                preempted_req_id, None
                            )
                            if preempted_encoder_inputs:
                                # Restore encoder compute budget if the preempted
                                # request had encoder inputs scheduled in this step.
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                                if iteration_details is not None:
                                    num_scheduled_encoder_inputs -= len(
                                        preempted_encoder_inputs
                                    )
                    else:
                        preempted_req = self.running.pop()

                    self._preempt_request(
                        preempted_req,
                        scheduled_timestamp,
                        drop_stale_output=self.requires_kv_delivery,
                    )
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt. Cannot schedule this request.
                        break

            if new_blocks is None:
                if (
                    self.policy == SchedulingPolicy.PRIORITY
                    and request.status == RequestStatus.PREEMPTED
                ):
                    # The request was the lowest-priority running request and
                    # preempted itself. The requests after it have higher
                    # priority and may still fit (it just freed its blocks);
                    # it was removed from `self.running`, so `req_index`
                    # already points at the next one.
                    continue
                # Cannot schedule this request.
                break

            # Schedule the request.
            scheduled_running_reqs.append(request)
            prefill_scheduled |= request.is_prefill_chunk
            request_id = request.request_id
            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            input_budget -= num_new_tokens + draft_slots + recovery_surcharge
            if recovery_surcharge:
                repair_charges[request_id] = recovery_surcharge
            recovery_plan = self.recovery_plans.get(request_id)
            if recovery_plan is not None and recovery_plan.state == "DETECTED":
                recovery_plan.state = "ADMITTED"
                recovery_plan.admitted_step = self.current_step
                repair_admitted = True
                repair_exclusive = not self._is_full_forward_repair(recovery_plan)
            req_index += 1
            drafts_needed = drafts_needed or not self._drafts_unused(
                request, request.num_computed_tokens, num_new_tokens
            )

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (
                    num_new_tokens
                    + request.num_computed_tokens
                    - request.num_tokens
                    - request.num_output_placeholders
                )
                if num_scheduled_spec_tokens > 0:
                    spec_token_ids = request.spec_token_ids
                    if len(spec_token_ids) > num_scheduled_spec_tokens:
                        spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
                    scheduled_spec_decode_tokens[request.request_id] = spec_token_ids

                # New spec tokens will be set in `update_draft_token_ids` before the
                # next step when applicable.
                request.spec_token_ids = []

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                if iteration_details is not None:
                    num_scheduled_encoder_inputs += len(encoder_inputs_to_schedule)
                encoder_compute_budget = new_encoder_compute_budget
            if encoder_inputs_to_schedule or external_load_encoder_input:
                self._allocate_encoder_inputs(
                    request,
                    (encoder_inputs_to_schedule or []) + external_load_encoder_input,
                    duplicate_encoder_inputs,
                )

        # Record the LoRAs of all running requests, including those not
        # scheduled in this step, which can join a later batch.
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in self.running
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # Next, schedule the WAITING requests.
        if not preempted_reqs and self._pause_state != PauseState.PAUSED_ALL:
            if self.cache_aware_window and not defer_prefills:
                self._reorder_waiting_by_cached_prefix()

            step_skipped_waiting: deque[Request] = deque()
            step_skipped_kv_holding: deque[Request] = deque()
            step_skipped_async_kv: deque[Request] = deque()
            # Async loads started this step: newer than any block-holder.
            step_started_kv_loads: list[Request] = []

            def skip_request(from_queue: RequestQueue) -> None:
                request_to_skip = from_queue.pop_request()
                self.deferred_waiting.add(request_to_skip)
                holds_blocks = self._holds_kv_blocks(request_to_skip)
                if holds_blocks and from_queue is not self.kv_holding_waiting:
                    step_started_kv_loads.append(request_to_skip)
                elif holds_blocks:
                    step_skipped_kv_holding.appendleft(request_to_skip)
                elif from_queue is self.async_kv_waiting:
                    step_skipped_async_kv.appendleft(request_to_skip)
                else:
                    step_skipped_waiting.appendleft(request_to_skip)

            num_async_kv_loads = 0
            max_async_kv_loads = (
                self.scheduler_config.max_async_kv_loads_per_step or float("inf")
            )
            # Cleared once self.async_kv_waiting's head can't start a load.
            take_async_kv = True
            while True:
                # Paused streaming sessions (WAITING_FOR_STREAMING_REQ) are not
                # in `running` but still hold a model-runner request slot.
                num_running = len(self.running) + self.num_waiting_for_streaming_input
                loads_only = (
                    num_running >= self.max_num_active_reqs or token_budget <= 0
                )
                if not loads_only:
                    # Requests holding KV blocks are always drained first, and are
                    # the only ones admitted under PAUSED_NEW (e.g. async KV loads).
                    request_queue = self.kv_holding_waiting
                    if not request_queue and self._pause_state == PauseState.UNPAUSED:
                        request_queue = self._next_waiting_queue(take_async_kv)
                elif (
                    take_async_kv
                    and self.scheduler_config.max_num_inflight_kv_loads
                    and self._pause_state == PauseState.UNPAUSED
                ):
                    # No running slot or token budget left: start async KV loads
                    # only, which take neither until their KV has arrived.
                    request_queue = self.async_kv_waiting
                else:
                    break
                if not request_queue or (
                    not loads_only and input_budget <= draft_slots
                ):
                    break

                request = request_queue.peek_request()
                request_id = request.request_id

                if repair_admitted and (
                    repair_exclusive or request.request_id in self.recovery_plans
                ):
                    # One repair per step.
                    skip_request(request_queue)
                    continue
                if repair_pending and request.request_id not in self.recovery_plans:
                    # The pending plan drains first.
                    skip_request(request_queue)
                    continue

                ready_to_schedule = self._handle_blocked_waiting_request(request)
                if not ready_to_schedule:
                    skip_request(request_queue)
                    continue

                if request.num_stale_output_tokens and not request.drop_stale_output:
                    # Deliverable stale output still in flight: resuming now
                    # could resample a position that output later delivers.
                    # It drains within the pipeline depth.
                    skip_request(request_queue)
                    continue

                # Check that adding the request still respects the max_loras constraint.
                if (
                    self.lora_config
                    and request.lora_request
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id not in scheduled_loras
                    )
                ):
                    # Scheduling would exceed max_loras, skip.
                    skip_request(request_queue)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False
                connector_prefix_cache_queries, connector_prefix_cache_hits = 0, 0
                did_prefix_cache_lookup = False

                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    if request.resumable and self.num_hash_lookahead_tokens:
                        # Resident KV can retain a discarded streaming token's
                        # hash. Recomputing KV must use the current token sequence.
                        request.block_hashes.clear()
                        request.update_block_hashes()
                    did_prefix_cache_lookup = True
                    max_cache_hit_length = (
                        self.aux_output_connector.max_cache_hit(request)
                        if self.aux_output_connector is not None
                        else None
                    )
                    (
                        new_computed_blocks,
                        num_new_local_computed_tokens,
                        request.shared_prefix_boundary,
                        hit_diverged,
                    ) = self._get_local_prefix_cache_hit(request, max_cache_hit_length)

                    # Get externally-cached tokens if using a KVConnector.
                    if self.connector is not None:
                        # Present a block-aligned local hit to the connector so
                        # a strictly longer remote hit can supersede a local
                        # sub-block tail without racing its copy-on-write.
                        partial_tail = num_new_local_computed_tokens % self.block_size
                        block_aligned_local = (
                            num_new_local_computed_tokens - partial_tail
                        )
                        ext_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, block_aligned_local
                            )
                        )
                        if request.skip_reading_external_kv:
                            ext_tokens, load_kv_async = 0, False
                        if request.request_id in self._untrusted_external_reqs:
                            # The request's transfer evidence already
                            # failed closed once: a
                            # connector that re-offers the same hit
                            # would start an unbounded external
                            # re-attempt loop. The external source is
                            # untrusted for this request's lifetime:
                            # the retry is a local recompute.
                            ext_tokens, load_kv_async = 0, False

                        if ext_tokens is None:
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            skip_request(request_queue)
                            continue

                        if self.prefix_replay_tokens and not (
                            ext_tokens and self._load_restores_replay_window(request)
                        ):
                            # DSV41 SWA bounded replay recomputes the hit's last
                            # window, from the block holding its first
                            # token; the sliding-window groups retire
                            # whole blocks below the window of the hit's
                            # next token. Hits end on a block boundary, or
                            # a hit ending one token short of one would
                            # retire the block the replay starts in. A load
                            # that carries the window replays nothing.
                            ext_tokens -= ext_tokens % self.block_size
                            load_kv_async = load_kv_async and ext_tokens > 0

                        if partial_tail and ext_tokens > partial_tail:
                            # Remote strictly exceeds the full local hit: drop the
                            # sub-block tail so no CoW is needed, and let the load
                            # cover it. Trim the partial block out of the local
                            # computed blocks so it is not adopted from the cache.
                            new_computed_blocks = (
                                self.kv_cache_manager.truncate_computed_blocks(
                                    new_computed_blocks, block_aligned_local
                                )
                            )
                            num_new_local_computed_tokens = block_aligned_local
                            num_external_computed_tokens = ext_tokens
                        elif partial_tail:
                            # Remote does not exceed the full local hit: keep the
                            # local sub-block tail and load nothing external.
                            num_external_computed_tokens = 0
                            # Nothing to load remotely -> not an async-load step;
                            # clearing avoids the `load_kv_async` assert below.
                            load_kv_async = False
                        else:
                            num_external_computed_tokens = ext_tokens

                        if self.aux_output_connector is not None:
                            num_external_computed_tokens = (
                                self.aux_output_connector.cap_external_hit(
                                    request,
                                    num_new_local_computed_tokens,
                                    num_external_computed_tokens,
                                )
                            )
                            load_kv_async &= num_external_computed_tokens > 0

                        if hit_diverged and num_external_computed_tokens == 0:
                            # No external tokens back the deeper local hit, so its
                            # resume boundary would have no valid Mamba state.
                            # Reconcile to the boundary every group agrees on.
                            (
                                new_computed_blocks,
                                num_new_local_computed_tokens,
                                request.shared_prefix_boundary,
                            ) = self.kv_cache_manager.get_computed_blocks(
                                request, max_cache_hit_length
                            )

                        connector_prefix_cache_queries = (
                            request.num_tokens - num_new_local_computed_tokens
                        )
                        connector_prefix_cache_hits = num_external_computed_tokens

                    # Total computed tokens (local + external).
                    num_computed_tokens = (
                        num_new_local_computed_tokens + num_external_computed_tokens
                    )
                    assert num_computed_tokens <= request.num_tokens
                    # DSV41 SWA bounded replay: a KV load that carries the
                    # window is taken as is; any other hit no longer than the
                    # replayed window would be recomputed in full and save
                    # nothing.
                    window_loaded = bool(
                        num_external_computed_tokens
                    ) and self._load_restores_replay_window(request)
                    if (
                        not window_loaded
                        and 0 < num_computed_tokens <= self.prefix_replay_tokens
                    ):
                        new_computed_blocks = (
                            self.kv_cache_manager.empty_kv_cache_blocks
                        )
                        num_new_local_computed_tokens = 0
                        num_external_computed_tokens = 0
                        connector_prefix_cache_hits = 0
                        num_computed_tokens = 0
                        load_kv_async = False

                    # Include encoder inputs needed to replay the prefix tail.
                    ec_computed_tokens = num_computed_tokens
                    if not window_loaded:
                        ec_computed_tokens = max(
                            0, num_computed_tokens - self.prefix_replay_tokens
                        )
                    if self._ec_transfer_pending(request, ec_computed_tokens):
                        skip_request(request_queue)
                        continue

                    # Track first scheduled prefill, not post-preemption repeat prefills
                    if request.prefill_stats and request.num_preemptions <= 0:
                        assert num_computed_tokens <= request.num_prompt_tokens
                        request.prefill_stats.set(
                            num_prompt_tokens=request.num_prompt_tokens,
                            num_local_cached_tokens=num_new_local_computed_tokens,
                            num_external_cached_tokens=num_external_computed_tokens,
                        )
                    self._set_kv_fetch_stage(
                        request, KV_FETCH_WAITING_TO_START if load_kv_async else None
                    )
                else:
                    # KVTransfer: WAITING reqs have num_computed_tokens > 0
                    # after async KV recvs are completed. A streaming-input
                    # session resumes here too, carrying whatever media its
                    # latest chunk added, so this branch needs the same gate.
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                    if self._ec_transfer_pending(request, num_computed_tokens):
                        skip_request(request_queue)
                        continue

                encoder_inputs_to_schedule = None
                external_load_encoder_input = []
                duplicate_encoder_inputs = []
                new_encoder_compute_budget = encoder_compute_budget
                pad_spec_decode = False

                # SWA bounded replay: recompute the tail of the hit without
                # rewriting its cached KV. An async load replays once the KV
                # has arrived (_update_waiting_for_remote_kv).
                num_replay_tokens = 0

                if (
                    load_kv_async
                    and (
                        num_async_kv_loads >= max_async_kv_loads
                        or not self._admits_inflight_kv_load()
                    )
                ) or (loads_only and not load_kv_async):
                    # Past an async KV load limit, or needs a running slot while none
                    # is free. It waits at the head of its queue; others may admit.
                    if request_queue is not self.async_kv_waiting:
                        break
                    take_async_kv = False
                    continue

                if load_kv_async:
                    # KVTransfer: loading remote KV, do not allocate for new work.
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                elif defer_prefills and num_computed_tokens < request.num_tokens - 1:
                    # DP prefill balancing: defer this step's local prefill
                    # compute to a cadence-aligned step.
                    break
                else:
                    request_token_budget = min(token_budget, input_budget - draft_slots)
                    if did_prefix_cache_lookup:
                        # Fresh admission: local and/or sync external hit.
                        num_replay_tokens = self._mark_prefix_replay(
                            request, num_computed_tokens, window_loaded
                        )
                        num_computed_tokens -= num_replay_tokens
                    # Number of tokens to be scheduled.
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens

                    # Pad new decode requests to uniform spec decoding size to
                    # preserve full cudagraph for this step.
                    # Not for diffusion where draft tokens can't be padded.
                    if (
                        (self.num_spec_tokens > 0 and self.dynamic_sd_lookup is None)
                        and self.num_sampled_tokens_per_step > 0
                        and num_new_tokens == 1
                        and not prefill_scheduled
                        and (scheduled_running_reqs or num_computed_tokens > 0)
                    ):
                        padded_num_tokens = 1 + self.num_spec_tokens
                        # Pad only when there is room for the sampled token(s).
                        if (
                            num_computed_tokens
                            + padded_num_tokens
                            + self.num_sampled_tokens_per_step
                            <= self.max_model_len
                        ):
                            if padded_num_tokens > request_token_budget:
                                # Prefer to not schedule than schedule un-padded.
                                break
                            num_new_tokens = padded_num_tokens
                            pad_spec_decode = True

                    if (
                        0 < long_prefill_token_threshold < num_new_tokens
                        # Pending repair plans are exempt from the
                        # long-prefill cap in the waiting arm for the
                        # same reason as the running arm: the repair
                        # step is exclusive and cannot be chunked below
                        # its span.
                        and self.recovery_plans.get(request.request_id) is None
                    ):
                        num_new_tokens = long_prefill_token_threshold

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    if (
                        not self.scheduler_config.enable_chunked_prefill
                        and num_new_tokens > request_token_budget
                        and not self.is_mm_encoder_only
                    ):
                        # If chunked_prefill is disabled,
                        # we can stop the scheduling here.
                        break

                    num_new_tokens = min(num_new_tokens, request_token_budget)
                    assert num_new_tokens > 0

                    # Apply Mamba alignment before encoder caps.
                    if self.need_mamba_block_aligned_split:
                        num_new_tokens = self._mamba_block_aligned_split(
                            request,
                            num_new_tokens,
                            num_new_local_computed_tokens,
                            num_external_computed_tokens,
                        )
                        if num_new_tokens == 0:
                            break
                        if (
                            pad_spec_decode
                            and num_new_tokens != 1 + self.num_spec_tokens
                        ):
                            # Alignment clipped the placeholder rows. The split
                            # aligns prefill chunks, but the padded tail rows are
                            # speculative positions, not prefill tokens. A padded
                            # request must keep all 1 + num_spec rows or the
                            # sampler's row count stops matching its query rows,
                            # so drop the padding instead of shortening it.
                            num_new_tokens = 1
                            pad_spec_decode = False

                    # Schedule encoder inputs.
                    if request.has_encoder_inputs:
                        (
                            encoder_inputs_to_schedule,
                            num_new_tokens,
                            new_encoder_compute_budget,
                            external_load_encoder_input,
                            duplicate_encoder_inputs,
                        ) = self._try_schedule_encoder_inputs(
                            request,
                            num_computed_tokens,
                            num_new_tokens,
                            encoder_compute_budget,
                            shift_computed_tokens=self.num_prefill_lookahead,
                        )

                    # Multi-module MTP: avoid ending a prefill chunk within
                    # num_prefill_lookahead of the prefill end.
                    num_new_tokens = self._reserve_prefill_lookahead(
                        request, num_computed_tokens, num_new_tokens
                    )

                    # Hole repair: charge the plan's equivalent-token
                    # surcharge, or defer the request this step.
                    recovery = self._recovery_plan_budget(
                        request, num_new_tokens, input_budget, draft_slots
                    )
                    if recovery is None:
                        skip_request(request_queue)
                        continue
                    num_new_tokens, recovery_surcharge = recovery
                    if repair_pending and not self.recovery_plans:
                        # The plan was dissolved at the gate: the
                        # constraint lifts.
                        repair_pending = False

                    if (
                        request.has_encoder_inputs
                        and not self.scheduler_config.enable_chunked_prefill
                        and not self.is_mm_encoder_only
                        and num_new_tokens > 0
                        and num_new_tokens < request.num_tokens - num_computed_tokens
                    ):
                        # Encoder cache pressure must not split a prompt that
                        # requires full prefill. Release references acquired by hits.
                        self.encoder_cache_manager.free(request)
                        num_new_tokens = 0

                    if num_new_tokens == 0:
                        # The request cannot be scheduled. Requeue it so it does
                        # not stall the requests behind it.
                        skip_request(request_queue)
                        continue

                # During async KV load, no forward pass is run yet.
                # Allocate speculative lookahead slots later to avoid
                # mismatching local and remote block counts.
                limit_lookahead_tokens = load_kv_async and self.num_lookahead_tokens > 0
                effective_lookahead_tokens = (
                    0 if limit_lookahead_tokens else self.num_lookahead_tokens
                )

                # Determine if we need to allocate cross-attention blocks.
                num_encoder_tokens = 0
                if (
                    self.is_encoder_decoder
                    and request.has_encoder_inputs
                    and encoder_inputs_to_schedule
                ):
                    num_encoder_tokens = sum(
                        request.get_num_encoder_embeds(i)
                        for i in encoder_inputs_to_schedule
                    )

                reserved_blocks = 0
                if load_kv_async:
                    # An async load holds its blocks for the whole transfer with
                    # no forward progress and isn't preemptible here. Admit it
                    # only if it fits in (free - other in-flight reservations)
                    # plus its own spec decode step blocks, to avoid deadlock and
                    # predictable preemptions.
                    reserved_blocks = (
                        self._inflight_prefill_reserved_blocks()
                        + self._spec_decode_step_blocks
                    )

                # Replayed tokens are already counted in the adopted hit; a
                # chunk that ends inside the replayed range needs no new slots.
                num_tokens_past_hit = max(num_new_tokens - num_replay_tokens, 0)
                # Delayed publication: under the repair policy a sync
                # external claim is not content-verified until the step's
                # delivery verdict, while the admission-time publication is
                # content-blind (a hash of token ids). Publishing here would
                # let a same-pass request match and consume unverified KV.
                # Delay like the P/D async precedent; publish at the step's
                # update.
                delay_sync_claim_pub = (
                    self.repair_kv_load_failures
                    and num_external_computed_tokens > 0
                    and not load_kv_async
                )
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_tokens_past_hit,
                    num_new_computed_tokens=num_new_local_computed_tokens,
                    new_computed_blocks=new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    num_external_computed_tokens=num_external_computed_tokens,
                    delay_cache_blocks=load_kv_async or delay_sync_claim_pub,
                    skip_zeroing_group_ids=(
                        self.connector.get_loaded_kv_cache_group_ids(request)
                        if load_kv_async and self.connector is not None
                        else ()
                    ),
                    num_encoder_tokens=num_encoder_tokens,
                    full_sequence_must_fit=self.scheduler_reserve_full_isl,
                    reserved_blocks=reserved_blocks,
                    has_scheduled_reqs=bool(self.running),
                )

                if new_blocks is None:
                    # The request cannot be scheduled.

                    # NOTE: we need to untouch the request from the encode cache
                    # manager
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    break

                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    if num_external_computed_tokens > 0:
                        # A new external load attempt is admitted:
                        # assign its identity HERE, at admission, so every
                        # report and attestation the attempt produces
                        # echoes an identity the scheduler can attribute
                        # to this attempt (and a late report of a
                        # consumed attempt is distinguishable from a new
                        # attempt's failure). When async loads are
                        # deferred (defer_async_loads), the connector
                        # starts the transfer at the fence step; the
                        # identity was still fixed at this admission,
                        # and note_transfer_attempt has already
                        # delivered it to the chosen connector.
                        self._next_transfer_generation += 1
                        self._req_transfer_generation[request_id] = (
                            self._next_transfer_generation
                        )
                        self._req_transfer_claim[request_id] = (
                            num_external_computed_tokens
                        )
                        self._req_transfer_claim_extent[request_id] = (
                            num_new_local_computed_tokens + num_external_computed_tokens
                        )
                        # A new attempt supersedes everything accumulated
                        # for the previous one: its attestations and any
                        # still-deferred failure reports belong to an attempt
                        # whose outcome no longer matters (the partition
                        # would drop their late arrivals as stale anyway).
                        self._req_transfer_attestations.pop(request_id, None)
                        self._req_pending_async_reports.pop(request_id, None)
                        self.connector.note_transfer_attempt(
                            request, self._next_transfer_generation
                        )
                    defer_load = (
                        load_kv_async
                        and self.defer_async_loads
                        and self.sched_step_seq > self.processed_step_seq
                    )
                    if defer_load:
                        request.last_sched_seq = self.sched_step_seq
                        self.deferred_loads[request_id] = (
                            self.sched_step_seq,
                            num_external_computed_tokens,
                            connector_prefix_cache_queries,
                            connector_prefix_cache_hits,
                        )
                    else:
                        self.connector.update_state_after_alloc(
                            request,
                            self.kv_cache_manager.get_blocks(request_id),
                            num_external_computed_tokens,
                        )
                    if (
                        self.connector_prefix_cache_stats is not None
                        and connector_prefix_cache_queries != 0
                        and not defer_load
                    ):
                        preempted = request.num_preemptions > 0
                        self.connector_prefix_cache_stats.record(
                            num_tokens=connector_prefix_cache_queries,
                            num_hits=connector_prefix_cache_hits,
                            preempted=preempted,
                            # The connector knows the tier only after the load
                            # plan above exists.
                            hits_by_source=(
                                self.connector.get_external_cache_hit_sources(
                                    request, connector_prefix_cache_hits
                                )
                                if connector_prefix_cache_hits and not preempted
                                else None
                            ),
                        )

                # Record at admission so unscheduled lookups are not counted.
                if did_prefix_cache_lookup:
                    self.kv_cache_manager.record_prefix_cache_stats(
                        request, num_new_local_computed_tokens
                    )

                if load_kv_async:
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    num_async_kv_loads += 1
                    # Set num_computed_tokens even though KVs are not yet loaded.
                    # request.num_computed_tokens will not be used anywhere until
                    # the request finished the KV transfer.
                    #
                    # If a transfer error is reported by the connector,
                    # request.num_computed_tokens will be re-set accordingly in
                    # _update_requests_with_invalid_blocks.
                    #
                    # When the transfer is finished, either successfully or not,
                    # request.num_computed_tokens will correctly reflect the number
                    # of computed tokens.
                    # _update_waiting_for_remote_kv will then cache
                    # only the successfully loaded tokens.
                    request.num_computed_tokens = num_computed_tokens
                    if self.aux_output_connector is not None:
                        self.aux_output_connector.request_loading(
                            request, num_computed_tokens
                        )
                    self._reserve_parked(request)
                    self._set_kv_fetch_stage(request, KV_FETCH_IN_PROGRESS)
                    skip_request(request_queue)
                    continue

                request = request_queue.pop_request()
                self.deferred_waiting.discard(request)
                self.running.append(request)
                if num_external_computed_tokens > 0:
                    # load_kv_async is False here
                    has_sync_kv_loads = True
                    if delay_sync_claim_pub:
                        self._delayed_sync_claim_pubs.add(request.request_id)
                if self.log_stats:
                    request.record_event(
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp
                    )
                # A resumed request is re-sent in full, like a new one: its
                # blocks were freed and its worker state dropped at preemption.
                if request.status not in (
                    RequestStatus.WAITING,
                    RequestStatus.PREEMPTED,
                ):
                    raise RuntimeError(f"Invalid request status: {request.status}")
                scheduled_new_reqs.append(request)

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(
                    request_id
                )
                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                input_budget -= num_new_tokens + draft_slots + recovery_surcharge
                if recovery_surcharge:
                    repair_charges[request_id] = recovery_surcharge
                recovery_plan = self.recovery_plans.get(request_id)
                if recovery_plan is not None and recovery_plan.state == "DETECTED":
                    recovery_plan.state = "ADMITTED"
                    recovery_plan.admitted_step = self.current_step
                    repair_admitted = True
                    repair_exclusive = not self._is_full_forward_repair(recovery_plan)
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                self._set_kv_fetch_stage(request, None)
                drafts_needed = drafts_needed or not self._drafts_unused(
                    request, num_computed_tokens, num_new_tokens
                )
                if pad_spec_decode:
                    assert num_new_tokens == 1 + self.num_spec_tokens
                    scheduled_spec_decode_tokens[request_id] = [
                        -1
                    ] * self.num_spec_tokens
                # Track requests still prefilling after this chunk. A promoted load
                # also stays tracked until _update_after_schedule drops it.
                if (
                    num_computed_tokens + num_new_tokens < request.num_tokens
                    or request in self._parked_reservations
                ):
                    self._inflight_prefills.add(request)
                self._unreserve_parked(request)
                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                    if iteration_details is not None:
                        num_scheduled_encoder_inputs += len(encoder_inputs_to_schedule)
                    encoder_compute_budget = new_encoder_compute_budget
                if encoder_inputs_to_schedule or external_load_encoder_input:
                    self._allocate_encoder_inputs(
                        request,
                        (encoder_inputs_to_schedule or [])
                        + external_load_encoder_input,
                        duplicate_encoder_inputs,
                    )

            # re-queue requests skipped in this pass ahead of older skipped items.
            if step_skipped_kv_holding:
                self.kv_holding_waiting.prepend_requests(step_skipped_kv_holding)
            for request in step_started_kv_loads:
                self.kv_holding_waiting.add_request(request)
            if step_skipped_waiting:
                self.waiting.prepend_requests(step_skipped_waiting)
            if step_skipped_async_kv:
                self.async_kv_waiting.prepend_requests(step_skipped_async_kv)

            # DP prefill balancing: on a step that admitted prefills (release),
            # record whether it was capacity-bound.
            if not defer_prefills:
                _, num_waiting = self.get_request_counts()
                self.prefill_capacity_bound = num_waiting > len(self.deferred_waiting)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens

        assert token_budget >= 0
        assert input_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert len(scheduled_new_reqs) + len(scheduled_running_reqs) <= len(
            self.running
        )

        # Construct the scheduler output.
        new_reqs_data = [
            NewRequestData.from_request(
                req,
                req_to_new_blocks[req.request_id].get_block_ids(),
                req.prefill_token_ids,
                uses_mrope=self.model_uses_mrope,
            )
            for req in scheduled_new_reqs
        ]

        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_blocks,
                iteration_details=iteration_details,
            )

        if iteration_details is not None:
            # Decode requests are counted in _make_cached_request_data()
            # Other scheduled requests - including new requests - are prefill
            iteration_details.num_prefill_requests = (
                len(num_scheduled_tokens) - iteration_details.num_decode_requests
            )
            iteration_details.num_prefill_tokens = (
                total_num_scheduled_tokens - iteration_details.num_decode_tokens
            )
            iteration_details.num_encoder_inputs = num_scheduled_encoder_inputs
            iteration_details.num_encoder_output_tokens = (
                self.max_num_encoder_input_tokens - encoder_compute_budget
            )

        # Mamba "align" boundary states must be handed off with exact block ids;
        # they cannot be reconstructed from a connector's append-only block
        # table. Drained every step so stale offers cannot accumulate.
        boundary_state_offloads = self.kv_cache_manager.take_boundary_state_offloads()

        kv_connector_block_state = None
        if self.connector is not None:
            # Any request scheduled this step can become a connector job now,
            # not only the ones that were allocated blocks: a store save lands
            # on the step that fills a block, which allocated none.
            block_state_req_ids = set(num_scheduled_tokens)
            block_state_req_ids.update(
                req_id for req_id in boundary_state_offloads if req_id in self.requests
            )
            kv_connector_block_state = KVConnectorBlockState(
                req_ids=block_state_req_ids,
                resolve_block_ids=self.kv_cache_manager.get_block_ids,
                boundary_state_offloads=boundary_state_offloads,
            )

        kv_cache_block_copies, cow_retained_blocks = (
            self.kv_cache_manager.take_kv_cache_block_copies()
        )
        if kv_cache_block_copies:
            # The copies run with this step's execution; the first non-empty
            # step at or after it gets seq `sched_step_seq + 1` (0-token steps
            # do not advance the seq), and its completion implies the copies
            # have run.
            self._free_cow_retained_blocks(cow_retained_blocks, self.sched_step_seq + 1)
        pending_kv_cache_block_copies = kv_cache_block_copies or None

        # Dynamic speculative decoding: compute optimal K
        num_spec_tokens_to_schedule = self.num_spec_tokens
        if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:
            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[
                len(num_scheduled_tokens)
            ]

        # Bounded admission: tick the deferral count of
        # every DETECTED plan this step did not admit, so owners skipped
        # before their budget call are bounded exactly like deferred ones.
        self._tick_recovery_plan_deferrals(num_scheduled_tokens)
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            iteration_details=iteration_details,
            preempted_req_ids=self.reset_preempted_req_ids,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            encoder_input_updates=self._get_encoder_input_updates(
                num_scheduled_tokens, new_reqs_data, scheduled_encoder_inputs
            ),
            new_block_ids_to_zero=self._get_new_block_ids_to_zero(),
            has_sync_kv_loads=has_sync_kv_loads,
            kv_cache_block_copies=pending_kv_cache_block_copies,
            kv_connector_block_state=kv_connector_block_state,
            num_spec_tokens_to_schedule=num_spec_tokens_to_schedule,
            skip_drafts=not drafts_needed,
        )

        if repair_admitted:
            scheduler_output.repair_full_forward = not repair_exclusive
            # Describe only repairs that remain scheduled after preemption.
            scheduler_output.repair_works = {
                req_id: RepairWork(
                    request_id=req_id,
                    repair_span=plan.repair_span,
                    layer_end=plan.layer_end,
                    restored_extent=plan.restored_extent,
                    plan_id=plan.plan_id,
                    epoch=plan.admitted_step if plan.admitted_step is not None else -1,
                    # The ownership authorization: the
                    # scheduler-authoritative physical blocks covering the
                    # repair's write set, snapshotted from KVCacheManager
                    # at this admission (after the step's allocation) --
                    # the worker validates its geometry against THESE,
                    # never against its own metadata.
                    # _repair_write_set_block_ids asserts the single
                    # KV cache group scope.
                    expected_write_block_ids=tuple(
                        self._repair_write_set_block_ids(
                            self.requests[req_id],
                            plan.repair_span[0],
                            plan.repair_span[1],
                        )
                    ),
                )
                for req_id, plan in self.recovery_plans.items()
                if plan.state == "ADMITTED" and req_id in num_scheduled_tokens
            }
            # A repair-only step never carries speculative drafts: the
            # replay's span is exact and no drafter runs. The spec
            # arithmetic already yields none for a rewound request
            # (num_scheduled_spec_tokens goes non-positive); pin it so a
            # future accounting change fails here instead of silently
            # drafting inside a repair.
            assert not (
                set(scheduler_output.repair_works) & set(scheduled_spec_decode_tokens)
            ), "hole repair: a repair-only step must not schedule drafts"
            if not scheduler_output.repair_works:
                scheduler_output.repair_works = None
                scheduler_output.repair_full_forward = False

        if self._pending_rewound_req_ids:
            # Drain the flags of every request scheduled in THIS step:
            # the worker stages the echoed (moved) count for exactly
            # these rows. Flags of requests no longer tracked are
            # dropped -- a preempted or finished request's worker state
            # is rebuilt by add_request, which stages its own init.
            scheduler_output.rewound_req_ids = (
                self._pending_rewound_req_ids & num_scheduled_tokens.keys()
            )
            self._pending_rewound_req_ids -= num_scheduled_tokens.keys()
            self._pending_rewound_req_ids &= self.requests.keys()

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self._build_kv_connector_meta(self.connector, scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        if self.aux_output_connector is not None:
            scheduler_output.aux_output_connector_metadata = (
                self.aux_output_connector.build_connector_meta(
                    scheduler_output, self.requests
                )
            )

        # Build the connector meta for ECConnector
        if self.ec_connector is not None:
            ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.ec_connector_metadata = ec_meta

        # Connector-only block state must not be dispatched to workers.
        scheduler_output.kv_connector_block_state = None

        # Advance the fence only for non-empty steps (those that actually
        # write KV and have their output processed later in update_from_output).
        if self.defer_block_free and total_num_scheduled_tokens > 0:
            self.sched_step_seq += 1

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        return scheduler_output

    def _build_kv_connector_meta(
        self, connector: KVConnectorBase_V1, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        return connector.build_connector_meta(scheduler_output)

    def _get_new_block_ids_to_zero(self) -> list[int] | None:
        # Drain new attention block ids every step so the manager-side list
        # does not grow unbounded; only kv-cache zeroing consumes them.
        new_block_ids_to_zero = self.kv_cache_manager.take_new_block_ids()
        if not self.needs_kv_cache_zeroing:
            return None
        return new_block_ids_to_zero or None

    def _preempt_request(
        self, request: Request, timestamp: float, drop_stale_output: bool = False
    ) -> None:
        """Preempt a request and put it back to the waiting queue.

        NOTE: The request should be popped from the running queue outside of this
        method.

        drop_stale_output: drop (rather than deliver) any in-flight output; used
        by reset_prefix_cache, whose same-step resume would otherwise deliver
        tokens out of order, and for connectors with a pending KV hand-off,
        which the preemption's block free would leave without valid KV.
        """
        assert request.status == RequestStatus.RUNNING, (
            "Only running requests can be preempted"
        )
        if self.aux_output_connector is not None:
            self.aux_output_connector.request_finished(request)
        self._free_request_blocks(request)
        self.encoder_cache_manager.free(request)
        self._pending_encoder_input_ids.pop(request.request_id, None)
        self._inflight_prefills.discard(request)
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        if request.spec_token_ids:
            request.spec_token_ids = []
        # Async scheduling: mark all in-flight output as stale. Its tokens are
        # still delivered on return (dropping them would perturb spec-decode
        # acceptance) but must not mutate the reset counters; each step drains
        # its share in update_from_output. num_in_flight_tokens already
        # includes any undrained stale share, so assign rather than accumulate.
        # An undrained drop-mode share stays dropped: its positions have
        # already been resampled.
        request.drop_stale_output = drop_stale_output or (
            request.drop_stale_output and request.num_stale_output_tokens > 0
        )
        request.num_stale_output_tokens = request.num_in_flight_tokens
        request.num_output_placeholders = 0
        request.num_preemptions += 1
        if self._evidence_state_initialized:
            # The whole prompt is recomputed on re-admission: any pending
            # hole-repair plan is void (its evidence recorded as
            # consumed -- a late re-delivery of the
            # voided plan's evidence is absorbed, never re-authorized).
            preempted_plan = self.recovery_plans.pop(request.request_id, None)
            if preempted_plan is not None:
                self._remember_consumed_evidence(preempted_plan)
            # The preempted request's blocks are freed, so its pending
            # delayed publication must not survive to a later update (the
            # hash entries would map onto freed blocks); and its transfer
            # evidence belongs to an attempt whose claim is gone (a later
            # re-admission with an external hit gets a NEW generation).
            # The attempt identity dies with the claim too: without this,
            # a re-admission that loads nothing external keeps the OLD
            # generation current, and a late report of the freed attempt
            # would still pass the identity partition and authorize a plan
            # against the recycled block table. Dropping
            # the identity makes such a report unattributable -- dropped
            # loudly, fail closed.
            self._delayed_sync_claim_pubs.discard(request.request_id)
            self._req_transfer_generation.pop(request.request_id, None)
            self._req_transfer_claim.pop(request.request_id, None)
            self._req_transfer_claim_extent.pop(request.request_id, None)
            self._req_transfer_attestations.pop(request.request_id, None)
            self._req_pending_async_reports.pop(request.request_id, None)
            self._req_async_report_deferrals.pop(request.request_id, None)
        if self.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)

        # Put the request back to the waiting queue.
        self.waiting.prepend_request(request)
        self.reset_preempted_req_ids.add(request.request_id)

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token
            request.num_in_flight_tokens += num_scheduled_token
            if self.defer_block_free:
                # Record the in-flight step, to fence deferred block freeing.
                request.last_sched_seq = self.sched_step_seq
            request.is_prefill_chunk = request.num_computed_tokens < (
                request.num_tokens + request.num_output_placeholders
            )
            scheduler_output.has_structured_output_requests |= (
                request.use_structured_output and not request.is_prefill_chunk
            )
            # Drop from the in-flight-prefill set once it's no longer prefilling.
            if not request.is_prefill_chunk:
                self._inflight_prefills.discard(request)

        # Clear the finished and preempted request IDs.
        # NOTE: We shouldn't just clear() here because it will also affect
        # the scheduler output.
        self.finished_req_ids = set()
        self.reset_preempted_req_ids = set()

    def _update_request_as_session(
        self, session: Request, update: StreamingUpdate
    ) -> None:
        """Updates the waiting session with the next streaming update.

        Discards the last sampled output token from the prior input chunk.
        """
        # Keep all output tokens but the final sampled one.
        keep_end = session.num_tokens - 1
        assert keep_end >= session.num_computed_tokens
        del session._all_token_ids[keep_end:]
        del session.block_hashes[keep_end // self.hash_block_size :]
        session._output_token_ids.clear()
        assert session.prompt_token_ids is not None
        # Extend prompt with kept output tokens.
        extend_from_slice(
            session.prompt_token_ids,
            session._all_token_ids,
            session.num_prompt_tokens,
            keep_end,
        )

        if update.mm_features:
            base = session.num_tokens
            for mm_feature in update.mm_features:
                mm_feature.mm_position = replace(
                    mm_feature.mm_position, offset=mm_feature.mm_position.offset + base
                )
            session.mm_features.extend(update.mm_features)

        session._all_token_ids.extend(update.prompt_token_ids or ())
        session.prompt_token_ids.extend(update.prompt_token_ids or ())
        # Update block hashes for the new tokens.
        session.update_block_hashes()
        session.num_prompt_tokens = len(session.prompt_token_ids)
        session.arrival_time = update.arrival_time
        session.sampling_params = update.sampling_params
        session.prefix_cache_hit_limit = session.get_prefix_cache_hit_limit()
        session.max_tokens = update.max_tokens
        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            self.num_waiting_for_streaming_input -= 1
        session.status = RequestStatus.WAITING

        if self.log_stats:
            session.record_event(EngineCoreEventType.QUEUED)

    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_blocks: dict[str, KVCacheBlocks],
        iteration_details: IterationDetails | None = None,
    ) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list[tuple[list[int], ...] | None] = []
        num_computed_tokens: list[int] = []
        num_output_tokens: list[int] = []

        for req in running_reqs:
            req_id = req.request_id
            req_ids.append(req_id)
            is_prefill = req.is_prefill_chunk
            # NOTE: In PP+async scheduling, we consume token ids via a direct GPU
            # broadcast path (`input_batch.prev_sampled_token_ids`), so we can
            # omit this payload.
            if self.use_pp and not self.scheduler_config.async_scheduling:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker. Otherwise, we don't
                # need to send the sampled tokens back because the model runner
                # will cache them.
                num_tokens = num_scheduled_tokens[req_id] - len(
                    spec_decode_tokens.get(req_id, ())
                )
                token_ids = req.all_token_ids[
                    req.num_computed_tokens : req.num_computed_tokens + num_tokens
                ]
                new_token_ids.append(token_ids)
            new_block_ids.append(
                req_to_new_blocks[req_id].get_block_ids(allow_none=True)
            )
            num_computed_tokens.append(req.num_computed_tokens)
            num_output_tokens.append(
                req.num_output_tokens + req.num_output_placeholders
            )
            if iteration_details is not None and not is_prefill:
                # Count decode requests here; all other scheduled reqs are prefills
                iteration_details.num_decode_requests += 1
                iteration_details.num_decode_tokens += num_scheduled_tokens[req_id]

        return CachedRequestData(
            req_ids=req_ids,
            new_token_ids=new_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
            num_output_tokens=num_output_tokens,
        )

    def _reject_on_encoder_cache_embed_mismatch(
        self,
        request: Request,
        input_id: int,
        identifier: str,
        cached_num_encoder_embeds: int,
        requested_num_encoder_embeds: int,
    ) -> bool:
        if cached_num_encoder_embeds == requested_num_encoder_embeds:
            return False
        logger.warning(
            "Encoder cache entry for multimodal identifier %r holds %d "
            "embeddings, but request %s input %d expects %d. A client-provided "
            "multimodal UUID may have been reused for different content.",
            identifier,
            cached_num_encoder_embeds,
            request.request_id,
            input_id,
            requested_num_encoder_embeds,
        )
        self.encoder_cache_mismatch_reqs.add(request.request_id)
        return True

    def _allocate_encoder_inputs(
        self,
        request: Request,
        input_ids: list[int],
        duplicate_input_ids: list[int],
    ) -> None:
        for input_id in input_ids:
            self.encoder_cache_manager.allocate(request, input_id)
            if self.ec_connector is not None:
                self.ec_connector.update_state_after_alloc(request, input_id)

        for input_id in duplicate_input_ids:
            self.encoder_cache_manager.check_and_update_cache(request, input_id)

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_compute_budget: int,
        shift_computed_tokens: int = 0,
    ) -> tuple[list[int], int, int, list[int], list[int]]:
        """Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - It is not exist on remote encoder cache (via ECConnector)
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.

        Note that num_computed_tokens includes both locally cached
        blocks and externally cached blocks (via KVConnector).
        """
        if num_new_tokens == 0 or not request.has_encoder_inputs:
            return [], num_new_tokens, encoder_compute_budget, [], []
        encoder_inputs_to_schedule: list[int] = []
        mm_features = request.mm_features
        assert mm_features is not None
        assert len(mm_features) > 0
        external_load_encoder_input = []
        duplicate_encoder_inputs = []

        # NOTE: since scheduler operates on the request level (possibly with
        # multiple encoder inputs per request), we need to create temporary
        # trackers for accounting at the encoder input level.
        mm_hashes_to_schedule: dict[str, int] = {}
        num_embeds_to_schedule = 0

        encoder_window_end = (
            num_computed_tokens + num_new_tokens + shift_computed_tokens
        )
        lo, hi = get_mm_features_in_window(
            mm_features,
            start=num_computed_tokens,
            end=encoder_window_end,
        )
        # For encoder-decoder, all inputs sit at start_pos=0, so lo=0 always.
        if self.is_encoder_decoder:
            lo = 0

        for i in range(lo, hi):
            mm_feature = mm_features[i]
            start_pos = mm_feature.mm_position.offset
            num_encoder_tokens = mm_feature.mm_position.length
            num_encoder_embeds = mm_feature.mm_position.get_num_embeds()
            item_identifier = mm_feature.identifier

            if self.is_encoder_decoder and num_computed_tokens > 0:
                assert start_pos == 0, (
                    "Encoder input should be processed at the beginning of "
                    "the sequence when encoder-decoder models are used."
                )
                # Encoder input has already been computed
                # The calculation here is a bit different. We don't turn encoder
                # output into tokens that get processed by the decoder and
                # reflected in num_computed_tokens. Instead, start_pos reflects
                # the position where we need to ensure we calculate encoder
                # inputs. This should always be 0 to ensure we calculate encoder
                # inputs before running the decoder.  Once we've calculated some
                # decoder tokens (num_computed_tokens > 0), then we know we
                # already calculated encoder inputs and can skip here.
                continue

            # If no encoder input chunking is allowed, we do not want to
            # partially schedule a multimodal item. If the scheduled range would
            # only cover part of the mm input, roll back to before the mm item.
            if (
                self.scheduler_config.disable_chunked_mm_input
                and not self.is_encoder_decoder
                and (num_computed_tokens + num_new_tokens)
                < (start_pos + num_encoder_tokens)
            ):
                if num_computed_tokens < start_pos:
                    # Account for EAGLE shift when rolling back to avoid
                    # encoder cache miss. This ensures the scheduled range
                    # stops before start_pos even with the shift.
                    num_new_tokens = max(
                        0, start_pos - (num_computed_tokens + shift_computed_tokens)
                    )
                else:
                    # The chunk starts on or inside the item (a prefix cache hit
                    # can end inside it), so there is nothing to roll back to.
                    # Schedule nothing until the rest of the item fits.
                    num_new_tokens = 0
                break

            if not self.is_encoder_decoder:
                # We are not using the encoder cache for encoder-decoder models,
                # yet.
                if item_identifier in mm_hashes_to_schedule:
                    scheduled_num_encoder_embeds = mm_hashes_to_schedule[
                        item_identifier
                    ]
                    if self._reject_on_encoder_cache_embed_mismatch(
                        request,
                        i,
                        item_identifier,
                        scheduled_num_encoder_embeds,
                        num_encoder_embeds,
                    ):
                        return [], 0, encoder_compute_budget, [], []
                    # The same encoder input has already been scheduled in the
                    # current step.
                    duplicate_encoder_inputs.append(i)
                    continue

                cached_num_encoder_embeds = (
                    self.encoder_cache_manager.get_cached_num_encoder_embeds(request, i)
                )
                if cached_num_encoder_embeds is not None and (
                    self._reject_on_encoder_cache_embed_mismatch(
                        request,
                        i,
                        item_identifier,
                        cached_num_encoder_embeds,
                        num_encoder_embeds,
                    )
                ):
                    return [], 0, encoder_compute_budget, [], []

                if self.encoder_cache_manager.check_and_update_cache(request, i):
                    # The encoder input is already computed and cached from a
                    # previous step.
                    continue

            if not self.encoder_cache_manager.can_allocate(
                request, i, encoder_compute_budget, num_embeds_to_schedule
            ):
                # The encoder cache is full or the encoder budget is exhausted.
                # NOTE(woosuk): We assume that the encoder input tokens should
                # be processed altogether, as the encoder usually uses
                # bidirectional attention.
                if num_computed_tokens < start_pos:
                    # We only schedule the decoder tokens just before the
                    # encoder input. No drafter shift here: its read-ahead into
                    # this unencoded input falls back to token embeddings.
                    num_new_tokens = min(
                        num_new_tokens, start_pos - num_computed_tokens
                    )
                else:
                    # Because of prefix caching, num_computed_tokens is greater
                    # than start_pos even though its encoder input is not
                    # available. In this case, we can't schedule any token for
                    # the request in this step.
                    num_new_tokens = 0
                break

            # Calculate the number of embeddings to schedule in the current range
            # of scheduled encoder placeholder tokens.
            start_idx_rel = max(0, num_computed_tokens - start_pos)
            end_idx_rel = min(num_encoder_tokens, encoder_window_end - start_pos)
            curr_embeds_start, curr_embeds_end = (
                mm_feature.mm_position.get_embeds_indices_in_range(
                    start_idx_rel, end_idx_rel
                )
            )
            # There's no embeddings in the current range of encoder placeholder tokens
            # so we can skip the encoder input.
            if curr_embeds_end - curr_embeds_start == 0:
                continue

            if self.ec_connector is not None and self.ec_connector.has_cache_item(
                item_identifier
            ):
                mm_hashes_to_schedule[item_identifier] = num_encoder_embeds
                external_load_encoder_input.append(i)
                num_embeds_to_schedule += num_encoder_embeds
                continue

            num_embeds_to_schedule += num_encoder_embeds
            encoder_compute_budget -= num_encoder_embeds
            mm_hashes_to_schedule[item_identifier] = num_encoder_embeds
            encoder_inputs_to_schedule.append(i)

        return (
            encoder_inputs_to_schedule,
            num_new_tokens,
            encoder_compute_budget,
            external_load_encoder_input,
            duplicate_encoder_inputs,
        )

    def get_grammar_bitmask(
        self, scheduler_output: SchedulerOutput
    ) -> GrammarOutput | None:
        # Collect list of scheduled request ids that use structured output.
        # The corresponding rows of the bitmask will be in this order.
        if not scheduler_output.has_structured_output_requests:
            return None

        structured_output_request_ids = [
            req_id
            for req_id in scheduler_output.num_scheduled_tokens
            if (req := self.requests.get(req_id))
            and (req.use_structured_output and not req.is_prefill_chunk)
        ]
        if not structured_output_request_ids:
            return None

        bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            scheduler_output.scheduled_spec_decode_tokens,
        )
        return GrammarOutput(structured_output_request_ids, bitmask)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        prompt_token_id_logprobs_dict = (
            model_runner_output.prompt_token_id_logprobs_dict
        )
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output
        ec_connector_output = model_runner_output.ec_connector_output
        cudagraph_stats = model_runner_output.cudagraph_stats

        # Every GPU write enqueued by this and earlier steps has completed, so it is
        # safe to return deferred-free blocks to the pool.
        if self.defer_block_free and scheduler_output.total_num_scheduled_tokens > 0:
            self.processed_step_seq += 1
            self._drain_deferred_frees()
            self._drain_deferred_loads()

        perf_stats: PerfStats | None = None
        if self.perf_metrics and self.perf_metrics.is_enabled():
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None

        failed_kv_load_req_ids: set[str] = set()
        repair_dispositions: dict[str, str] = {}
        # Re-inject the failure reports of async attempts that were
        # deferred while in flight and have now become terminal: their
        # completeness can be judged (and their recovery authorized or
        # refused) exactly now.
        step_reports = (
            kv_connector_output.load_failure_reports if kv_connector_output else None
        )
        # The set of async requests whose transfer attempt is provably
        # TERMINAL this step -- their finished_recving arrived now or in an
        # earlier update. Only then may missing attestations be judged as a
        # contract violation rather than as ranks still in flight. Computed
        # only while the evidence machinery is live (bare instances stay
        # off it entirely; see the class-level defaults).
        terminal_async_req_ids = set()
        if self._evidence_state_initialized and (
            self._req_pending_async_reports
            or step_reports
            or (
                kv_connector_output is not None
                and kv_connector_output.load_attestations
            )
        ):
            terminal_async_req_ids = self.finished_recving_kv_req_ids | (
                kv_connector_output.finished_recving
                if kv_connector_output is not None
                and kv_connector_output.finished_recving
                else set()
            )
        if self._evidence_state_initialized and self._req_pending_async_reports:
            retrigger = [
                req_id
                for req_id in self._req_pending_async_reports
                if req_id in terminal_async_req_ids
            ]
            if retrigger:
                step_reports = list(step_reports or [])
                for req_id in retrigger:
                    self._req_async_report_deferrals.pop(req_id, None)
                    step_reports.extend(self._req_pending_async_reports.pop(req_id))
            # Liveness: age every still-deferred attempt one update
            # step; past the bound the request is failed closed (the
            # recompute fallback, the claim released) instead of
            # stranding in WAITING_FOR_REMOTE_KVS holding blocks with
            # its failure evidence never judged.
            expired: list[str] = []
            for req_id in list(self._req_pending_async_reports):
                if req_id in terminal_async_req_ids:
                    continue
                deferrals = self._req_async_report_deferrals.get(req_id, 0) + 1
                self._req_async_report_deferrals[req_id] = deferrals
                if deferrals > self.ASYNC_REPORT_MAX_DEFERRAL_STEPS:
                    expired.append(req_id)
            for req_id in expired:
                request = self.requests.get(req_id)
                deferred_reports = self._req_pending_async_reports.pop(req_id, [])
                self._req_async_report_deferrals.pop(req_id, None)
                if request is None or request.is_finished():
                    continue
                logger.warning(
                    "Async transfer evidence for req=%s remained deferred "
                    "beyond %d steps; failing the request closed to the "
                    "recompute fallback",
                    req_id,
                    self.ASYNC_REPORT_MAX_DEFERRAL_STEPS,
                )
                self._reject_uninterpretable_reports(
                    request,
                    deferred_reports,
                    "async transfer evidence remained deferred beyond the "
                    "liveness bound (the transfer attempt never reached "
                    "terminality)",
                    set(),
                    trusted_rollback=0,
                    mark_external_untrusted=True,
                )
                self._req_transfer_attestations.pop(req_id, None)
                self._release_failed_async_claim(request)
                # The attempt can never deliver finished_recving (that is
                # the shape the bound exists for), so the request cannot
                # leave WAITING_FOR_REMOTE_KS through the promotion path:
                # transition it here exactly as the promotion would, or
                # the local recompute could never own it -- and clear the
                # fetch-stage accounting so the inflight-KV-load slot is
                # released instead of leaking.
                request.status = (
                    RequestStatus.PREEMPTED
                    if request.num_preemptions
                    else RequestStatus.WAITING
                )
                self._set_kv_fetch_stage(request, None)
                self._reserve_parked(request)
                failed_kv_load_req_ids.add(req_id)
        if kv_connector_output and (
            step_reports or kv_connector_output.load_attestations
        ):
            failed_kv_load_req_ids.update(
                self._handle_repair_evidence(
                    step_reports,
                    kv_connector_output.load_attestations,
                    terminal_async_req_ids,
                )
            )
        if kv_connector_output and kv_connector_output.repair_results:
            # The repair step's verdicts, one per admitted work item. TP
            # aggregation may repeat a request's verdict across ranks;
            # duplicates are rejected as stale by the identity machinery.
            # The disposition keeps the first verdict that CONSUMED the
            # plan (committed/aborted); a stale echo never overrides a
            # consuming outcome, and a consuming outcome replaces an
            # earlier stale one (the stale verdict consumed nothing).
            for result in kv_connector_output.repair_results:
                disposition = self._consume_repair_result(result)
                if result.request_id not in repair_dispositions or (
                    repair_dispositions[result.request_id] == "stale"
                    and disposition != "stale"
                ):
                    repair_dispositions[result.request_id] = disposition
        if scheduler_output.repair_works:
            # Every work admitted this step must produce its verdict at
            # THIS update: a missing OR ENTIRELY STALE result set would
            # leave the plan ADMITTED indefinitely (the request stuck
            # above the rollback floor, the span's blocks held) -- a
            # stale echo does not consume the plan, so it counts as no
            # verdict. Fail
            # safe: synthesize the missing verdict as a failed result --
            # the abort fallback, never a silent retry loop.
            for req_id, work in scheduler_output.repair_works.items():
                if repair_dispositions.get(req_id) not in (None, "stale"):
                    continue
                plan = self.recovery_plans.get(req_id)
                if plan is None or plan.state != "ADMITTED":
                    continue
                logger.warning(
                    "Repair step admitted req=%s plan_id=%d span=%s but no "
                    "verdict arrived with it; applying the recompute "
                    "fallback",
                    req_id,
                    work.plan_id,
                    work.repair_span,
                )
                missing = RepairResultReport(
                    request_id=req_id,
                    plan_id=work.plan_id,
                    epoch=work.epoch,
                    structural_check=False,
                    executed_layers=0,
                    span=work.repair_span,
                    aborted=True,
                )
                repair_dispositions[req_id] = self._consume_repair_result(missing)
        if kv_connector_output and kv_connector_output.failed_recving:
            failed_kv_load_req_ids.update(
                self._handle_failed_recving(kv_connector_output.failed_recving)
            )
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            # These blocks contain externally computed tokens that failed to
            # load. Identify affected requests and adjust their computed token
            # count to trigger recomputation of the invalid blocks.
            failed_kv_load_req_ids.update(
                self._handle_invalid_blocks(
                    kv_connector_output.invalid_block_ids,
                    num_scheduled_tokens,
                )
            )
        # The sync-claim publication gate. It runs at EVERY update
        # under the repair policy -- whether or not any evidence arrived in
        # the step -- because a pending delayed publication is due exactly
        # now, and the absence of evidence is a contract violation, never
        # success. (Previously the gate ran only when some report or
        # attestation arrived, so a fully silent step -- a worker returning
        # no connector output, or one whose evidence lists were empty --
        # published the claim unverified.)
        if self.repair_kv_load_failures:
            failed_kv_load_req_ids.update(
                self._check_delivery_completeness(failed_kv_load_req_ids)
            )
        if self._evidence_state_initialized:
            # Delayed publication: the delivery verdict for this step's
            # external claims is now known (the failure channels above
            # consumed their reports); publish the verified ones.
            self._publish_delayed_sync_claims(failed_kv_load_req_ids)
            # Transactional publication: every repair-step admission
            # recorded in _delayed_repair_txn_pubs has its verdict in this
            # update (the missing-result fail-safe guarantees a
            # disposition for every admitted work). Publish ONLY on the
            # accepted verdict -- the committed extent is fully written
            # and verified at this point. Every other disposition stays
            # unpublished: the abort path evicts the span, and the
            # fallback recompute republishes through the ordinary path.
            for req_id in list(self._delayed_repair_txn_pubs):
                self._delayed_repair_txn_pubs.discard(req_id)
                if repair_dispositions.get(req_id) != "committed":
                    continue
                request = self.requests.get(req_id)
                if request is None or request.is_finished():
                    continue
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        # Stopped resumable requests to re-enqueue, in output order.
        continuing_reqs: list[Request] = []
        # The worker's output order follows its batch, not num_scheduled_tokens.
        req_indices = {
            req_id: i for i, req_id in enumerate(model_runner_output.req_ids)
        }
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            request = self.requests.get(req_id)
            output_is_stale = False
            if request is not None:
                request.num_in_flight_tokens -= num_tokens_scheduled
                # Drain any stale share (see _preempt_request) in lockstep.
                if request.num_stale_output_tokens > 0:
                    output_is_stale = True
                    request.num_stale_output_tokens -= num_tokens_scheduled
                    assert request.num_stale_output_tokens >= 0
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # skip failed or rescheduled requests from KV load failure
                continue
            if (
                request is not None
                and self._evidence_state_initialized
                and req_id in self.recovery_plans
            ):
                # The request has a pending recovery plan: any output this
                # step produced for it was computed over untrusted state.
                # With async scheduling the batch queue can deliver a
                # schedule-ahead step's output after the failure is known;
                # suppress it until the repair verdict restores the request.
                continue
            if scheduler_output.repair_works is not None and (
                req_id in scheduler_output.repair_works
            ):
                # A repair-only step produces no tokens by contract. Any
                # output arriving for its requests came from a runner
                # that ignored repair_works (possible only for an
                # out-of-tree platform runner); suppress it loudly -- the
                # request's real output continues on the next ordinary
                # step, after the verdict settled the plan.
                if scheduler_output.repair_full_forward:
                    # Only the other rows may emit tokens during repair.
                    continue
                logger.error_once(
                    "Suppressing output delivered for req=%s on its "
                    "repair-only step: a repair step produces no tokens "
                    "(a runner ignored repair_works)",
                    req_id,
                )
                continue
            if request is None or request.is_finished():
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism or in async scheduling).
                # NOTE(Kuntai): When delay_free_blocks=True (for async KV
                # cache transfer in KV connector), the aborted request will not
                # be set to None (in order to finish async KV transfer).
                # In this case, we use is_finished() to check.
                continue

            # Drop-mode stale output (same-step resume) is discarded entirely.
            if output_is_stale and request.drop_stale_output:
                continue

            req_index = req_indices[req_id]
            generated_token_ids = (
                sampled_token_ids[req_index] if sampled_token_ids else []
            )

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            )
            if scheduled_spec_token_ids and (
                generated_token_ids or self.num_sampled_tokens_per_step == 0
            ):
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_sampled = self.num_sampled_tokens_per_step
                num_accepted = max(len(generated_token_ids) - num_sampled, 0)
                num_rejected = num_draft_tokens - num_accepted
                # Rejections roll back num_computed_tokens (and, under async
                # scheduling, num_output_placeholders, which covers the spec
                # tokens). A stale rejection count predates the preemption
                # rollback and must not apply.
                if not output_is_stale:
                    if request.num_computed_tokens > 0:
                        request.num_computed_tokens -= num_rejected
                    if request.num_output_placeholders > 0:
                        request.num_output_placeholders -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                    request_id=req_id,
                )
                if request.spec_decode_metrics is not None:
                    # Exclude grammar-invalidated drafts from the proposed
                    # count, mirroring make_spec_decoding_stats; the accepted
                    # bucket (j) is unaffected.
                    adj_draft_tokens = num_draft_tokens
                    if scheduler_output.num_invalid_spec_tokens:
                        adj_draft_tokens -= (
                            scheduler_output.num_invalid_spec_tokens.get(req_id, 0)
                        )
                    request.spec_decode_metrics.observe(
                        num_draft_tokens=adj_draft_tokens,
                        num_accepted=num_accepted,
                        detailed=self.spec_decode_metrics_level == "detailed",
                    )

            # Free encoder inputs only after the step has actually executed.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

            stopped = False
            new_logprobs = None
            new_sampling_mask = None
            new_token_ids = generated_token_ids
            pooler_output = pooler_outputs[req_index] if pooler_outputs else None
            kv_transfer_params = None
            ec_transfer_params = None
            prefill_stats = None
            watermarking = None
            status_before_stop = request.status

            # Check for stop and update request status.
            if new_token_ids:
                has_output_placeholders = request.num_output_placeholders > 0
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids, is_stale=output_is_stale
                )
                if (
                    not output_is_stale
                    and status_before_stop == RequestStatus.RUNNING
                    and (
                        has_output_placeholders
                        or self.num_spec_tokens > 0
                        or (
                            self.has_mamba_layers
                            and not self.scheduler_config.async_scheduling
                            and self.cache_config.enable_prefix_caching
                        )
                    )
                ):
                    # Publish finalized states before retirement or completion.
                    self.kv_cache_manager.cache_blocks(
                        request,
                        request.num_computed_tokens - request.num_in_flight_tokens,
                    )
            elif request.pooling_params and pooler_output is not None:
                # Pooling stops as soon as there is output.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True
            elif (
                self.is_mm_encoder_only
                and request.num_computed_tokens >= request.num_prompt_tokens
            ):
                # An encoder instance runs the encoder and publishes the
                # embeddings instead of sampling, so it stops as soon as the
                # whole prompt is consumed. Encoder inputs are never scheduled
                # past a multi-modal item the encoder cache could not admit, so
                # a consumed prompt also means every item in it was encoded.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True

            if new_token_ids and not self.structured_output_manager.accept_tokens(
                request, new_token_ids
            ):
                logger.error(
                    "Unexpected: grammar rejected tokens %s for request %s. "
                    "Terminating request.",
                    new_token_ids,
                    req_id,
                )
                request.status = RequestStatus.FINISHED_ERROR
                request.resumable = False
                stopped = True

            routed_experts = None
            should_emit_output = bool(
                new_token_ids or pooler_output is not None or stopped
            )
            if self.aux_output_connector is not None and should_emit_output:
                routed_experts = self.aux_output_connector.take_output(
                    request, model_runner_output.aux_output_connector_output
                )
            if should_emit_output:
                prefill_stats = request.take_prefill_stats()
                if prefill_stats is not None:
                    prefill_stats.finalize(
                        self.kv_cache_manager.estimate_cached_tokens(request)
                    )
                    params = request.sampling_params
                    if params is not None and self.vllm_config.watermark_config:
                        # Report the flag after enforcement; the sampler skips
                        # greedy and trace replay rows.
                        watermarking = bool(
                            params.watermarking
                            and params.temperature != 0
                            and params.trace_decode_token_ids is None
                        )

            # Extract sample logprobs before stop handling can replace the
            # sampling parameters for a streaming continuation.
            if (
                request.sampling_params is not None
                and request.sampling_params.num_logprobs is not None
                and logprobs
            ):
                new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

            finish_reason = None
            if stopped:
                # Capture finish_reason BEFORE _handle_stopped_request, which may
                # reset the status to WAITING for streaming requests that continue.
                finish_reason = request.get_finished_reason()
                finished = self._handle_stopped_request(request)
                if finished:
                    kv_transfer_params, ec_transfer_params = self._free_request(request)
                    if (
                        prefill_stats is not None
                        and kv_transfer_params is not None
                        and kv_transfer_params.get("do_remote_prefill")
                    ):
                        # P-side cache hits, so D can report them in
                        # prompt_tokens_details instead of its own (~100%)
                        # hit rate from the KV transfer.
                        kv_transfer_params["remote_prefill_cached_tokens"] = (
                            prefill_stats.num_cached_tokens
                        )
                else:
                    continuing_reqs.append(request)

                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            if self.return_sampling_mask:
                sampling_masks = model_runner_output.sampling_masks
                if new_token_ids and sampling_masks is not None:
                    new_sampling_mask = sampling_masks.slice_request(
                        req_index, len(new_token_ids)
                    )

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            prompt_token_id_logprobs = prompt_token_id_logprobs_dict.get(req_id)
            if should_emit_output:
                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=finish_reason,
                        new_logprobs=new_logprobs,
                        new_sampling_mask=new_sampling_mask,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        prompt_token_id_logprobs=prompt_token_id_logprobs,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        prefill_stats=prefill_stats,
                        spec_decode_metrics=(
                            request.spec_decode_metrics
                            if finish_reason is not None
                            else None
                        ),
                        kv_transfer_params=kv_transfer_params,
                        ec_transfer_params=ec_transfer_params,
                        trace_headers=request.trace_headers,
                        routed_experts=routed_experts,
                        num_nans_in_logits=request.num_nans_in_logits,
                        watermarking=watermarking,
                    )
                )
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors
                assert prompt_token_id_logprobs is None

        # Remove stopped requests from their queues, then enqueue the ones
        # that continue, so each ends up in exactly one queue.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            self.waiting.remove_requests(stopped_preempted_reqs)
            self.kv_holding_waiting.remove_requests(stopped_preempted_reqs)
            self.async_kv_waiting.remove_requests(stopped_preempted_reqs)
            self.deferred_waiting.difference_update(stopped_preempted_reqs)
        for request in continuing_reqs:
            self._enqueue_waiting_request(request)

        request_errors = self.grammar_compile_error_reqs
        self.grammar_compile_error_reqs = {}
        error_req_ids = set(request_errors)
        error_req_ids.update(self.encoder_cache_mismatch_reqs)
        self.encoder_cache_mismatch_reqs.clear()
        if (
            failed_kv_load_req_ids
            and not self.recompute_kv_load_failures
            and not self.repair_kv_load_failures
        ):
            error_req_ids.update(failed_kv_load_req_ids)
        if self.ec_connector is not None:
            # An encoder input the connector can no longer obtain. Failing is
            # retryable: re-issuing the request re-runs the encode.
            error_req_ids.update(self.ec_connector.take_unavailable_requests())

        if error_req_ids:
            error_reqs = self.finish_requests(
                error_req_ids, RequestStatus.FINISHED_ERROR
            )
            for request in error_reqs:
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[],
                        finish_reason=request.get_finished_reason(),
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                        error=request_errors.get(request.request_id),
                    )
                )

        # KV Connector: update state for finished KV Transfers.
        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)

        # EC Connector: update state from worker-side EC connector output.
        if self.ec_connector is not None and ec_connector_output:
            self.ec_connector.update_connector_output(ec_connector_output)

        # Worker-side KV connector stats from the model runner output.
        kv_connector_stats: KVConnectorStats | None = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )
        if self.connector:
            # Scheduler-side KV connector stats collected after connector update.
            scheduler_kv_connector_stats = self.connector.get_kv_connector_stats()
            if (
                scheduler_kv_connector_stats is not None
                and not scheduler_kv_connector_stats.is_empty()
            ):
                kv_connector_stats = (
                    kv_connector_stats.aggregate(scheduler_kv_connector_stats)
                    if kv_connector_stats is not None
                    else scheduler_kv_connector_stats
                )

        # Worker-side EC connector stats from the model runner output.
        ec_connector_stats: ECConnectorStats | None = (
            ec_connector_output.ec_connector_stats if ec_connector_output else None
        )
        if self.ec_connector:
            # Scheduler-side EC connector stats collected after connector update.
            scheduler_ec_connector_stats = self.ec_connector.get_ec_connector_stats()
            if (
                scheduler_ec_connector_stats is not None
                and not scheduler_ec_connector_stats.is_empty()
            ):
                ec_connector_stats = (
                    ec_connector_stats.aggregate(scheduler_ec_connector_stats)
                    if ec_connector_stats is not None
                    else scheduler_ec_connector_stats
                )

        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set
                    )
            finished_req_ids.clear()

        if (
            stats := self.make_stats(
                spec_decoding_stats,
                kv_connector_stats,
                cudagraph_stats,
                perf_stats,
                ec_connector_stats=ec_connector_stats,
            )
        ) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs

    def _ec_transfer_pending(self, request: Request, num_computed_tokens: int) -> bool:
        """Whether an encoder input this request needs is still in transit."""
        return (
            self.ec_connector is not None
            and bool(request.mm_features)
            and not self.ec_connector.ensure_cache_available(
                request, num_computed_tokens
            )
        )

    @staticmethod
    def _is_blocked_waiting_status(status: RequestStatus) -> bool:
        return status in (
            RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
            RequestStatus.WAITING_FOR_REMOTE_KVS,
            RequestStatus.WAITING_FOR_STREAMING_REQ,
        )

    @staticmethod
    def _holds_kv_blocks(request: Request) -> bool:
        # Whether a request currently has kv blocks allocated.
        # While always the case when status is WAITING_FOR_REMOTE_KVS
        # or WAITING_FOR_STREAMING_REQ, there are such cases where it's
        # WAITING or PREEMPTED.
        return (
            request.num_computed_tokens > 0
            # num_computed_tokens is reset to 0 if kv transfer fails for requests in
            # WAITING_FOR_REMOTE_KVS state, before their allocated blocks are freed.
            or request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        )

    def _enqueue_waiting_request(self, request: Request) -> None:
        if self._is_blocked_waiting_status(request.status):
            self.deferred_waiting.add(request)
        if self._holds_kv_blocks(request):
            self.kv_holding_waiting.add_request(request)
        elif (
            self.connector is not None
            and (
                self.scheduler_config.max_num_inflight_kv_loads
                or self.scheduler_config.max_async_kv_loads_per_step
            )
            and self.connector.may_load_async(request)
        ):
            self.async_kv_waiting.add_request(request)
        else:
            self.waiting.add_request(request)

    def _reorder_waiting_by_cached_prefix(self) -> None:
        """Move waiting requests with a cached prefix to the front of the window.

        Reordering is confined to the slots held by plain waiting requests, so a
        request in a blocked status or one already partially computed keeps its
        exact position and nothing is moved past it. That keeps this a
        reordering among peers rather than a change in priority, and leaves the
        queue beyond the window untouched.
        """
        if len(self.waiting) < 2:
            return
        if self.kv_cache_manager.usage < self.cache_aware_threshold:
            return

        head = list(itertools.islice(self.waiting, self.cache_aware_window))
        slots = [
            i
            for i, request in enumerate(head)
            if request.status == RequestStatus.WAITING
            and not request.num_computed_tokens
        ]
        if len(slots) < 2:
            return

        # Deepest cached prefix first, arrival order breaking ties.
        get_num_cached = self.kv_cache_manager.get_num_cached_tokens
        cached = {i: get_num_cached(head[i]) for i in slots}
        order = sorted(slots, key=lambda i: (-cached[i], i))
        if order == slots:
            return

        reordered = list(head)
        for slot, source in zip(slots, order):
            reordered[slot] = head[source]
        # One pass to drop the whole window, then re-insert in the new order;
        # removing each request individually would be quadratic.
        self.waiting.remove_requests(head)
        for request in reversed(reordered):
            self.waiting.prepend_request(request)

    def _next_waiting_queue(self, take_async_kv: bool) -> RequestQueue:
        """self.waiting or self.async_kv_waiting, whichever head goes first."""
        async_kv = self.async_kv_waiting
        if not take_async_kv or not async_kv:
            return self.waiting
        if not self.waiting:
            return async_kv
        async_head, head = async_kv.peek_request(), self.waiting.peek_request()
        if self.policy == SchedulingPolicy.PRIORITY:
            return async_kv if async_head < head else self.waiting
        return async_kv if async_head.arrival_time < head.arrival_time else self.waiting

    def _admits_inflight_kv_load(self) -> bool:
        """Whether a new async KV load stays within --max-num-inflight-kv-loads."""
        limit = self.scheduler_config.max_num_inflight_kv_loads
        return limit is None or (
            self._kv_fetch_counts[KV_FETCH_IN_PROGRESS]
            + self._kv_fetch_counts[KV_FETCH_COMPLETED_WAITING]
            + len(self._aborted_kv_loads)
            < limit
        )

    def _handle_stopped_request(self, request: Request) -> bool:
        """Return True if finished (can be False for resumable requests)."""
        if not request.resumable:
            return True

        # Drop the finished turn's in-flight work and resume from the
        # materialized frontier.
        safe_frontier = request.num_computed_tokens - request.num_output_placeholders
        assert safe_frontier >= 0
        request.num_computed_tokens = safe_frontier
        request.spec_token_ids = []
        request.drop_stale_output = True
        request.num_stale_output_tokens = request.num_in_flight_tokens
        request.num_output_placeholders = 0

        if request.streaming_queue:
            update = request.streaming_queue.popleft()
            if update is None:
                # Streaming request finished.
                return True
            self._update_request_as_session(request, update)
        else:
            request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
            self.num_waiting_for_streaming_input += 1

        return False

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int], is_stale: bool = False
    ) -> tuple[list[int], bool]:
        num_new_tokens = len(new_token_ids)
        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)

            # Check for stop and update request state.
            # This must be called before we make the EngineCoreOutput.
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                # num_computed_tokens covers all new tokens but the last one.
                # Keep that true after trimming: resumable requests resume from it.
                if not is_stale:
                    request.num_computed_tokens -= len(new_token_ids) - num_new
                del new_token_ids[num_new:]  # Trim new tokens if needed.
                break

        if not is_stale and request.num_output_placeholders > 0:
            # Resolve all returned tokens, including those trimmed at a stop.
            request.num_output_placeholders -= num_new_tokens
            assert request.num_output_placeholders >= 0
        return new_token_ids, stopped

    def _get_encoder_input_updates(
        self,
        scheduled: dict[str, int],
        new_requests: list[NewRequestData],
        scheduled_encoder_inputs: dict[str, list[int]],
    ) -> dict[str, dict[int, MultiModalKwargsItem | None]]:
        # Installations (including resume/streaming) replace worker features.
        # SHM descriptors follow the processor cache's separate ACK protocol;
        # EngineCore cannot replay their resolved tensors.
        for new_request in new_requests:
            if not new_request.mm_features:
                continue
            self._pending_encoder_input_ids[new_request.req_id] = {
                i
                for i, feature in enumerate(new_request.mm_features)
                if feature.data is not None and "address" not in feature.data
            }

        updates: dict[str, dict[int, MultiModalKwargsItem | None]] = {}
        for req_id in scheduled:
            request = self.requests[req_id]
            pending = self._pending_encoder_input_ids.get(req_id)
            if pending is None:
                continue
            encoding = scheduled_encoder_inputs.get(req_id, ())
            if not pending and not encoding:
                continue
            for input_id in encoding:
                data = request.mm_features[input_id].data
                if (
                    input_id not in pending
                    and data is not None
                    and "address" not in data
                ):
                    # A running request may rewind without being reinstalled.
                    updates.setdefault(req_id, {})[input_id] = data
                    pending.add(input_id)

            # schedule() advances progress before execution. Exclude in-flight
            # prefill and speculative placeholders to obtain confirmed progress.
            computed = request.num_computed_tokens - max(
                request.num_output_placeholders, request.num_in_flight_tokens
            )
            for input_id in list(pending):
                if input_id in encoding:
                    continue
                position = request.mm_features[input_id].mm_position
                if position.offset + position.length <= computed or (
                    self.is_encoder_decoder and computed > 0
                ):
                    updates.setdefault(req_id, {})[input_id] = None
                    pending.remove(input_id)
        return updates

    def _free_encoder_inputs(self, request: Request) -> None:
        cached_encoder_input_ids = self.encoder_cache_manager.get_cached_input_ids(
            request
        )
        # OPTIMIZATION: Avoid list(set) if the set is empty.
        if not cached_encoder_input_ids:
            return

        # Here, we use list(set) to avoid modifying the set while iterating
        # over it.
        for input_id in list(cached_encoder_input_ids):
            mm_feature = request.mm_features[input_id]
            start_pos = mm_feature.mm_position.offset
            num_tokens = mm_feature.mm_position.length
            if self.is_encoder_decoder and request.num_computed_tokens > 0:
                # With Whisper, as soon as we've generated a single token,
                # we know we're done with the encoder input. Cross Attention
                # KVs have been calculated and cached already.
                self._free_encoder_input(request, input_id)
            elif (
                start_pos + num_tokens
                <= request.num_computed_tokens - request.num_output_placeholders
            ):
                # Processed, stored in the decoder KV cache, and past the
                # placeholder range with no pending rejection that could roll
                # back into it. The drafter only reads ahead of
                # num_computed_tokens, so it cannot reference this input either.
                self._free_encoder_input(request, input_id)

    def _free_encoder_input(self, request: Request, input_id: int) -> None:
        self.encoder_cache_manager.free_encoder_input(request, input_id)
        if self.ec_connector is not None:
            self.ec_connector.update_state_after_free(request, input_id)

    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            if request.is_prefill_chunk:
                # Ignore draft tokens for prefill chunks.
                if request.spec_token_ids:
                    request.spec_token_ids = []
                continue

            # Add newly generated spec token ids to the request.
            request.spec_token_ids = self.structured_output_manager.trim_spec_tokens(
                request, spec_token_ids
            )

    def update_draft_token_ids_in_output(
        self, draft_token_ids: DraftTokenIds, scheduler_output: SchedulerOutput
    ) -> None:
        num_invalid_spec_tokens: dict[str, int] = {}

        sched_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            placeholder_spec_tokens = sched_spec_tokens.get(req_id)
            if not placeholder_spec_tokens:
                continue

            orig_num_spec_tokens = len(placeholder_spec_tokens)
            # Trim drafts to scheduled number of spec tokens
            # (needed for chunked prefill case for example).
            del spec_token_ids[orig_num_spec_tokens:]
            # Preserve draft slots; rejection sampling handles grammar violations.
            # Pad to original number of spec tokens.
            num_invalid_tokens = orig_num_spec_tokens - len(spec_token_ids)
            if num_invalid_tokens:
                spec_token_ids.extend([-1] * num_invalid_tokens)
                num_invalid_spec_tokens[req_id] = num_invalid_tokens

            sched_spec_tokens[req_id] = spec_token_ids

        scheduler_output.num_invalid_spec_tokens = num_invalid_spec_tokens

    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        return len(self.running), (
            len(self.waiting)
            + len(self.kv_holding_waiting)
            + len(self.async_kv_waiting)
        )

    def get_kv_cache_usage(self) -> float:
        """Returns the fraction of the KV cache currently in use (0.0-1.0)."""
        return self.kv_cache_manager.usage

    def has_request(self, request_id: str) -> bool:
        return request_id in self.requests

    def add_request(self, request: Request) -> None:
        existing = self.requests.get(request.request_id)
        if existing is not None:
            update = StreamingUpdate.from_request(request)
            if existing.status != RequestStatus.WAITING_FOR_STREAMING_REQ:
                assert existing.streaming_queue is not None, "duplicate request id"
                # Queue next input chunk (or finished sentinel).
                existing.streaming_queue.append(update)
            elif update is not None:
                # Commence next input chunk.
                self._update_request_as_session(existing, update)
            else:
                # Streaming-input session finished.
                self.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
        else:
            if request.resumable:
                request.streaming_queue = deque()
            if self.connector is not None:
                # Before enqueueing: may_load_async can use what this records.
                self.connector.on_new_request(request)
            self._enqueue_waiting_request(request)
            self.requests[request.request_id] = request
            if self.spec_decode_metrics_level != "none":
                request.spec_decode_metrics = RequestSpecDecodeMetrics.new(
                    self.num_spec_tokens
                )
            if self.connector is not None:
                kv_transfer_params = request.kv_transfer_params
                if kv_transfer_params and kv_transfer_params.get("do_remote_prefill"):
                    # Count P/D requests on arrival: connector matching is
                    # skipped while all run slots are taken.
                    self._set_kv_fetch_stage(request, KV_FETCH_WAITING_TO_START)
            if self.log_stats:
                request.record_event(EngineCoreEventType.QUEUED)

    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[Request]:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.

        If request_ids is None, all requests will be finished.

        Returns:
            List of requests that were aborted. Will not include any that were
            already finished.

        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        elif request_ids is not None:
            request_ids = set(request_ids)
        else:
            request_ids = self.requests.keys()

        running_requests_to_remove = set()
        waiting_requests_to_remove = []
        valid_requests = []

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # Invalid request ID.
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.add(request)
            else:
                if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                    self.num_waiting_for_streaming_input -= 1
                waiting_requests_to_remove.append(request)

        # Remove all requests from queues at once for better efficiency
        if running_requests_to_remove:
            self.running = remove_all(self.running, running_requests_to_remove)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)
            self.kv_holding_waiting.remove_requests(waiting_requests_to_remove)
            self.async_kv_waiting.remove_requests(waiting_requests_to_remove)
            self.deferred_waiting.difference_update(waiting_requests_to_remove)

        # Second pass: set status and free requests
        for request in valid_requests:
            delay_free_blocks = False
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                # A deferred load never started, so it will not report done.
                delay_free_blocks = (
                    self.deferred_loads.pop(request.request_id, None) is None
                    and request.request_id not in self.finished_recving_kv_req_ids
                )
                self.finished_recving_kv_req_ids.discard(request.request_id)
                self.failed_recving_kv_req_ids.discard(request.request_id)

            request.status = finished_status
            self._free_request(request, delay_free_blocks=delay_free_blocks)
            if delay_free_blocks:
                self._aborted_kv_loads.add(request.request_id)

        return valid_requests

    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        assert request.is_finished()

        if self.aux_output_connector is not None:
            self.aux_output_connector.request_finished(request)
        self._inflight_prefills.discard(request)
        self._unreserve_parked(request)
        self._set_kv_fetch_stage(request, None)
        connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)

        # EC Connector: mirror the KV hook. The contract requires firing
        # before the encoder cache is freed so the connector can inspect
        # per-request state (e.g. which mm_hashes it recorded during
        # save_caches()) and emit ec_transfer_params for the response body.
        ec_xfer_params: dict[str, Any] | None = None
        if self.ec_connector is not None:
            ec_delay_free, ec_xfer_params = self.ec_connector.request_finished(request)
            connector_delay_free_blocks |= ec_delay_free

        self.encoder_cache_manager.free(request)
        self._pending_encoder_input_ids.pop(request.request_id, None)
        request_id = request.request_id
        if self._evidence_state_initialized:
            self.recovery_plans.pop(request_id, None)
            # The transfer-identity bookkeeping dies with the request:
            # a later request reusing the ID cannot inherit (or
            # collide with) its generations -- the global counter
            # never reuses one.
            self._req_transfer_generation.pop(request_id, None)
            self._req_transfer_claim.pop(request_id, None)
            self._req_transfer_claim_extent.pop(request_id, None)
            self._req_transfer_attestations.pop(request_id, None)
            self._req_pending_async_reports.pop(request_id, None)
            self._req_async_report_deferrals.pop(request_id, None)
            self._consumed_report_evidence.pop(request_id, None)
            self._untrusted_external_reqs.discard(request_id)
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        delay_free_blocks |= connector_delay_free_blocks
        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params, ec_xfer_params

    def _free_blocks(self, request: Request):
        assert request.is_finished()
        self._free_request_blocks(request)
        del self.requests[request.request_id]

    @property
    def pause_state(self) -> PauseState:
        return self._pause_state

    def set_pause_state(self, pause_state: PauseState) -> None:
        logger.info("setting pause state to %s", pause_state.name)
        self._pause_state = pause_state

    def _request_blocks_can_be_freed(self, request: Request) -> bool:
        # We must defer freeing blocks if an async kv connector may
        # write to them immediately (not ordered with GPU stream).
        return not self.defer_block_free or (
            request.last_sched_seq <= self.processed_step_seq
        )

    def _free_request_blocks(self, request: Request):
        """Free the request's KV blocks, deferring the return to the block
        pool when an in-flight GPU step may still write them.
        """
        if self._request_blocks_can_be_freed(request):
            self.kv_cache_manager.free(request)
            return
        blocks = self.kv_cache_manager.pop_blocks_for_free(request)
        if blocks:
            self.deferred_frees.append((self.sched_step_seq, blocks))

    def _free_cow_retained_blocks(
        self, blocks: list[KVCacheBlock], fence_seq: int
    ) -> None:
        """Release CoW copy retentions, deferring their return to the block
        pool while the step that runs the copy may still be in flight.
        """
        if not self.defer_block_free or fence_seq <= self.processed_step_seq:
            self.kv_cache_manager.block_pool.free_blocks(blocks)
            return
        self.deferred_frees.append((fence_seq, blocks[::-1]))

    def _drain_deferred_frees(self):
        """Return deferred blocks whose fence step has completed.

        Fences are appended in near-monotonic order (a CoW retention fence
        can lead request-free fences by one step), so stop at the first
        pending one; any satisfied entry behind it is merely freed later.
        """
        while self.deferred_frees:
            fence, _ = self.deferred_frees[0]
            if fence > self.processed_step_seq:
                break
            _, blocks = self.deferred_frees.popleft()
            # Free in reverse order so that the tail blocks are evicted first.
            self.kv_cache_manager.block_pool.free_blocks(reversed(blocks))

    def _drain_deferred_loads(self) -> None:
        """Start deferred async loads whose fence step has completed."""
        for request_id, (fence, num_tokens, queries, hits) in list(
            self.deferred_loads.items()
        ):
            if fence > self.processed_step_seq:
                break
            del self.deferred_loads[request_id]
            assert self.connector is not None
            request = self.requests[request_id]
            self.connector.update_state_after_alloc(
                request, self.kv_cache_manager.get_blocks(request_id), num_tokens
            )
            if self.connector_prefix_cache_stats is not None and queries != 0:
                preempted = request.num_preemptions > 0
                self.connector_prefix_cache_stats.record(
                    num_tokens=queries,
                    num_hits=hits,
                    preempted=preempted,
                    hits_by_source=(
                        self.connector.get_external_cache_hit_sources(request, hits)
                        if hits and not preempted
                        else None
                    ),
                )

    def get_num_unfinished_requests(self) -> int:
        if self._pause_state == PauseState.PAUSED_ALL:
            return 0
        num_running, num_waiting = self.get_request_counts()
        if self._pause_state == PauseState.PAUSED_NEW:
            # Requests holding KV blocks still drain; queued ones stay queued.
            num_waiting = len(self.kv_holding_waiting)
        num_waiting -= self.num_waiting_for_streaming_input
        return num_waiting + num_running

    def has_finished_requests(self) -> bool:
        if self.finished_req_ids:
            return True
        if self.connector is None:
            return False
        # Finished requests waiting on delayed connector cleanup remain in
        # self.requests after they have been removed from scheduling queues.
        num_running, num_waiting = self.get_request_counts()
        return len(self.requests) > (num_running + num_waiting)

    def has_requests(self) -> bool:
        # Override the interface default to also keep the engine alive while a
        # connector still has pending push work (e.g. push-mode WRITE transfers
        # in flight after all "live" requests have finished). Without this hook
        # the engine would quiesce before the connector can drain completions.
        # TODO: replace with a more general mechanism for connectors to keep
        # the scheduler alive.
        return (
            self.has_unfinished_requests()
            or self.has_finished_requests()
            or (self.connector is not None and self.connector.has_pending_push_work())
            or (
                self.ec_connector is not None
                and self.ec_connector.has_pending_push_work()
            )
        )

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Reset the KV prefix cache.

        If reset_running_requests is True, all the running requests will be
        preempted and moved to the waiting queue.
        Otherwise, this method will only reset the KV prefix cache when there
        is no running requests taking KV cache.
        """
        if reset_running_requests:
            # For logging.
            timestamp = time.monotonic()
            # Invalidate all the current running requests KV's by pushing them to
            # the waiting queue. In this case, we can reduce the ref count of all
            # the kv blocks to 0 and thus we can make sure the reset is successful.
            # Preempt in reverse order so the requests will be added back to the
            # running queue in FIFO order.
            while self.running:
                request = self.running.pop()
                self._preempt_request(request, timestamp, drop_stale_output=True)

        reset_successful = self.kv_cache_manager.reset_prefix_cache()
        if reset_running_requests and not reset_successful:
            raise RuntimeError(
                "Failed to reset KV cache even when all the running requests are "
                "preempted and moved to the waiting queue. This is likely due to "
                "the presence of running requests waiting for remote KV transfer, "
                "which is not supported yet."
            )

        if reset_connector:
            reset_successful = self.reset_connector_cache() and reset_successful

        if reset_successful and self.aux_output_connector is not None:
            self.aux_output_connector.reset()

        return reset_successful

    def reset_connector_cache(self) -> bool:
        if self.connector is None:
            # No connector attached -> nothing to reset, treat as success so
            # callers that unconditionally request a connector reset (e.g. as
            # part of a cache-clearing cascade after a weight update) don't
            # see reset_prefix_cache() flip to False purely because they
            # didn't configure a connector.
            logger.debug(
                "reset_connector requested but no KV connector is configured; "
                "treating as no-op success."
            )
            return True

        if self.connector.reset_cache() is False:
            return False

        if self.log_stats:
            assert self.connector_prefix_cache_stats is not None
            self.connector_prefix_cache_stats.reset = True

        return True

    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache to invalidate all cached encoder outputs.

        This should be called when model weights are updated to ensure
        stale vision embeddings are not reused.
        """
        self.encoder_cache_manager.reset()

    def make_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None = None,
        kv_connector_stats: KVConnectorStats | None = None,
        cudagraph_stats: CUDAGraphStat | None = None,
        perf_stats: PerfStats | None = None,
        ec_connector_stats: ECConnectorStats | None = None,
    ) -> SchedulerStats | None:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        connector_prefix_cache_stats: PrefixCacheStats | None = None
        if self.connector_prefix_cache_stats is not None:
            connector_prefix_cache_stats = self.connector_prefix_cache_stats
            self.connector_prefix_cache_stats = PrefixCacheStats()
        eviction_events = (
            self.kv_metrics_collector.drain_events()
            if self.kv_metrics_collector is not None
            else []
        )
        spec_stats = spec_decoding_stats
        connector_stats_payload = (
            kv_connector_stats.to_dict() if kv_connector_stats else None
        )
        ec_connector_stats_payload = (
            ec_connector_stats.data if ec_connector_stats else None
        )
        num_running, num_waiting = self.get_request_counts()
        num_deferred = len(self.deferred_waiting)
        return SchedulerStats(
            num_running_reqs=num_running,
            num_waiting_reqs=num_waiting - num_deferred,
            num_skipped_waiting_reqs=num_deferred,
            num_kv_fetch_reqs_by_stage=self._kv_fetch_counts.copy(),
            kv_cache_usage=self.kv_cache_manager.usage,
            prefix_cache_stats=prefix_cache_stats,
            connector_prefix_cache_stats=connector_prefix_cache_stats,
            kv_cache_eviction_events=eviction_events,
            spec_decoding_stats=spec_stats,
            kv_connector_stats=connector_stats_payload,
            cudagraph_stats=cudagraph_stats,
            perf_stats=perf_stats,
            ec_connector_stats=ec_connector_stats_payload,
        )

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        num_invalid_spec_tokens: dict[str, int] | None,
        request_id: str,
    ) -> SpecDecodingStats | None:
        if not self.log_stats or not num_draft_tokens:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        if num_invalid_spec_tokens:
            num_draft_tokens -= num_invalid_spec_tokens.get(request_id, 0)
        spec_decoding_stats.observe_draft(
            num_draft_tokens=num_draft_tokens, num_accepted_tokens=num_accepted_tokens
        )
        return spec_decoding_stats

    def shutdown(self) -> None:
        logger.debug_once("[shutdown] Scheduler: start")
        if self.kv_event_publisher:
            self.kv_event_publisher.shutdown()
        if self.connector is not None:
            self.connector.shutdown()

        if self.ec_connector is not None:
            self.ec_connector.shutdown()

        logger.debug_once("[shutdown] Scheduler: complete")

    ########################################################################
    # KV Connector Related Methods
    ########################################################################

    def get_kv_connector(self) -> KVConnectorBase_V1 | None:
        return self.connector

    def get_ec_connector(self) -> ECConnectorBase | None:
        return self.ec_connector

    def get_kv_event_publisher_config(self) -> KVEventsConfig | None:
        return self.kv_event_publisher.get_publisher_config()

    def _connector_finished(
        self, request: Request
    ) -> tuple[bool, dict[str, Any] | None]:
        """Invoke the KV connector request_finished() method if applicable.

        Returns optional kv transfer parameters to be included with the
        request outputs.
        """
        if self.connector is None:
            return False, None

        finished_partial_tails: list[tuple[int, int, int]] = []
        kv_transfer_config = self.vllm_config.kv_transfer_config
        if kv_transfer_config is not None and kv_transfer_config.is_kv_producer:
            finished_partial_tails = (
                self.kv_cache_manager.finalize_partial_tail_offloads(request)
            )

        # Free any out-of-window prefix blocks before we hand the block table to
        # the connector, on the processed-token basis (see `allocate_slots`).
        self.kv_cache_manager.remove_skipped_blocks(
            request_id=request.request_id,
            processed_computed_tokens=max(
                0, request.num_computed_tokens - request.num_in_flight_tokens
            ),
            num_prompt_tokens=request.num_prompt_tokens,
        )

        block_ids = self.kv_cache_manager.get_block_ids_for_computed_tokens(
            request_id=request.request_id,
            num_computed_tokens=request.num_computed_tokens,
        )
        partial_tail_delay = False
        if finished_partial_tails:
            partial_tail_delay = self.connector.register_finished_partial_tail(
                request,
                block_ids,
                finished_partial_tails,
            )

        if not isinstance(self.connector, SupportsHMA):
            # NOTE(Kuntai): We should deprecate this code path after we enforce
            # all connectors to support HMA.
            # Hybrid memory allocator should be already turned off for this
            # code path, but let's double-check here.
            assert len(self.kv_cache_config.kv_cache_groups) == 1
            delay_free, kv_xfer_params = self.connector.request_finished(
                request, block_ids[0]
            )
        else:
            delay_free, kv_xfer_params = self.connector.request_finished_all_groups(
                request, block_ids
            )
        return delay_free or partial_tail_delay, kv_xfer_params

    def _request_remaining_blocks(self, request: Request) -> int:
        """Blocks `request` still needs to hold its full sequence and be promoted."""
        full_num_tokens = min(request.num_tokens, self.max_model_len)
        num_tokens = full_num_tokens
        if self.num_spec_tokens:
            num_tokens += 1 + self.num_spec_tokens + self.num_lookahead_tokens
        return self.kv_cache_manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens,
            new_computed_blocks=self.kv_cache_manager.empty_kv_cache_blocks.blocks,
            num_encoder_tokens=0,
            total_computed_tokens=request.num_computed_tokens,
            num_local_computed_tokens=request.num_computed_tokens,
            num_tokens_main_model=full_num_tokens,
            apply_admission_cap=True,
            prefill_end=max(request.num_prompt_tokens, request.num_tokens - 1),
        )

    def _set_kv_fetch_stage(self, request: Request, stage: str | None) -> None:
        kv_fetch_stages = self._kv_fetch_stages
        if kv_fetch_stages is not None:
            prev_stage = kv_fetch_stages.pop(request, None)
            if prev_stage is not None:
                self._kv_fetch_counts[prev_stage] -= 1
            if stage is not None:
                kv_fetch_stages[request] = stage
                self._kv_fetch_counts[stage] += 1

    def _inflight_prefill_reserved_blocks(self) -> int:
        """Num blocks in-flight prefills still need to finish (their reservation)."""
        return self._parked_reserved_blocks + sum(
            self._request_remaining_blocks(req) for req in self._inflight_prefills
        )

    def _reserve_parked(self, request: Request) -> None:
        """Cache a parked async KV load's reservation. Set when parked and refreshed
        at promotion, the only points where the load's held blocks change."""
        if self.kv_cache_config.hisparse_host_num_blocks is not None:
            # HiSparse's depends on free host blocks, so it is summed on demand.
            self._inflight_prefills.add(request)
            return
        blocks = self._request_remaining_blocks(request)
        cached = self._parked_reservations.get(request, 0)
        self._parked_reserved_blocks += blocks - cached
        self._parked_reservations[request] = blocks

    def _unreserve_parked(self, request: Request) -> None:
        self._parked_reserved_blocks -= self._parked_reservations.pop(request, 0)

    def _load_restores_replay_window(self, request: Request) -> bool:
        """Whether the KV connector's load for `request` restores every DSV41
        SWA bounded-replay group, as a P/D transfer of the request's own blocks
        does; a prefix-cache store holds no window."""
        assert self.connector is not None
        return bool(self.prefix_replay_group_ids) and set(
            self.prefix_replay_group_ids
        ).issubset(self.connector.get_loaded_kv_cache_group_ids(request))

    def _mark_prefix_replay(
        self, request: Request, num_hit_tokens: int, window_loaded: bool
    ) -> int:
        """Record where a prefix hit's replay starts and return the number of
        hit tokens to recompute; the caller rewinds the computed count by it.

        The replayed tokens are the hit's last window (see
        `Request.replay_start`): the worker rebuilds their sliding-window KV
        and leaves their cached KV alone. A hit whose KV load carries the
        window replays nothing.
        """
        if not self.prefix_replay_tokens:
            return 0
        if window_loaded:
            request.replay_start = 0
            return 0
        num_replay_tokens = min(self.prefix_replay_tokens, num_hit_tokens)
        request.replay_start = num_hit_tokens - num_replay_tokens
        return num_replay_tokens

    def _publish_delayed_sync_claims(self, failed_req_ids: set[str]) -> None:
        """Publish the sync external-claim blocks whose delivery this
        step verified.

        Requests named by any failure channel (the structured report arm,
        the flat invalid-blocks walk, async recv failures) keep their
        blocks unpublished: their truncation/eviction arms own them, and
        unmatchable is the safe state for unverified content. Aborted or
        finished requests keep theirs unpublished too: their blocks are
        freed, and a hash entry would map to freed blocks.
        """
        if not self._delayed_sync_claim_pubs:
            return
        pending, self._delayed_sync_claim_pubs = (
            self._delayed_sync_claim_pubs,
            set(),
        )
        for req_id in pending:
            if req_id in failed_req_ids:
                continue
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                continue
            # num_computed_tokens already covers the verified claim plus
            # the chunk the completed forward wrote (the schedule-time
            # advance), so this mirrors the async recv precedent's
            # success arm.
            self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
            # The attempt's evidence has now
            # served its purpose -- the delivery is published. A late
            # same-generation artifact must not read the stale state as
            # if the attempt were still verifiable.
            self._req_transfer_attestations.pop(req_id, None)
            self._req_pending_async_reports.pop(req_id, None)

    def _release_failed_async_claim(self, request: Request) -> None:
        """Release an async request whose external claim was failed closed:
        nothing of the claim is trusted, nothing was published, and the
        local recompute reschedule owns the request from zero."""
        self.kv_cache_manager.free(request)
        self.finished_recving_kv_req_ids.discard(request.request_id)
        self.failed_recving_kv_req_ids.discard(request.request_id)

    def _update_waiting_for_remote_kv(self, request: Request) -> None:
        """KV Connector: update request state after async recv is finished.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        assert self.connector is not None

        if (
            self.repair_kv_load_failures
            and request.request_id in self._req_transfer_generation
            # A pending recovery plan IS the delivery verdict: the plan's
            # repair step and commit/abort transaction own the request and
            # its publication from here (its admission already consumed the
            # complete evidence).
            and request.request_id not in self.recovery_plans
        ):
            # The async delivery's publication gate. The
            # request's latest transfer attempt must be covered by
            # terminal attestations from every TP rank, each proving
            # content identity for its loaded objects, before ANY of the
            # claim is trusted -- published or computed over. This runs at
            # promotion, the first point the request can leave
            # WAITING_FOR_REMOTE_KVS, so every arrival up to and including
            # the finished-recving pass is already folded in. A request
            # already failed closed at its delivery update (a report or
            # attestation violation) takes the same path.
            already_failed = request.request_id in self._untrusted_external_reqs
            gate = None if already_failed else self._delivery_evidence_gate(request)
            if already_failed or gate is not None:
                if gate is not None:
                    self._reject_uninterpretable_reports(
                        request,
                        [],
                        "unverified delivery: " + gate,
                        set(),
                        trusted_rollback=0,
                        mark_external_untrusted=True,
                    )
                    self._req_transfer_attestations.pop(request.request_id, None)
                    self._req_pending_async_reports.pop(request.request_id, None)
                self._release_failed_async_claim(request)
                return

        load_failed = request.request_id in self.failed_recving_kv_req_ids
        if load_failed:
            # Request had KV load failures; num_computed_tokens was already
            # updated in _update_requests_with_invalid_blocks
            if request.num_computed_tokens:
                # Cache any valid computed tokens.
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
                if self.needs_kv_cache_zeroing:
                    # The failed load left the blocks beyond the valid
                    # prefix unwritten and their zeroing was skipped; zero
                    # them before they are recomputed locally.
                    self.kv_cache_manager.record_blocks_for_zeroing(
                        request.request_id, request.num_computed_tokens
                    )
            else:
                # No valid computed tokens, release allocated blocks.
                # There may be a local cache hit on retry.
                # (Freed blocks are re-recorded for zeroing when
                # reallocated, so the skipped blocks need no handling.)
                self.kv_cache_manager.free(request)

            self.failed_recving_kv_req_ids.remove(request.request_id)
        else:
            # Now that the blocks are ready, actually cache them.
            # This will cache the blocks iff caching is enabled.
            self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)

        # DSV41 SWA bounded replay recomputes the tail of a hit loaded without
        # its window, or of a failed load, which may have lost it; the replay
        # covers the last token. Otherwise a full prompt hit re-computes that
        # token so the next one can be sampled.
        num_replay_tokens = self._mark_prefix_replay(
            request,
            request.num_computed_tokens,
            window_loaded=not load_failed
            and self._load_restores_replay_window(request),
        )
        if num_replay_tokens > 0:
            request.num_computed_tokens -= num_replay_tokens
        elif request.num_computed_tokens == request.num_tokens:
            request.num_computed_tokens = request.num_tokens - 1
        self.finished_recving_kv_req_ids.remove(request.request_id)

    def _handle_blocked_waiting_request(self, request: Request) -> bool:
        """Returns True if scheduling of this request should proceed, False if
        it should be skipped for this step."""
        if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            # Check whether async loading requests are finished.
            request_id = request.request_id
            if request_id not in self.finished_recving_kv_req_ids:
                logger.debug("%s is still in WAITING_FOR_REMOTE_KVS state.", request_id)
                return False
            # Promote request to waiting or preempted.
            self._update_waiting_for_remote_kv(request)
            self._reserve_parked(request)
            if request.num_preemptions:
                request.status = RequestStatus.PREEMPTED
            else:
                request.status = RequestStatus.WAITING

        elif request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR:
            # Check whether grammar compilation has finished.
            struct_output_req = request.structured_output_request
            if not struct_output_req or struct_output_req.grammar is None:
                return False
            if isinstance(struct_output_req.grammar, Exception):
                self.grammar_compile_error_reqs[request.request_id] = RequestError(
                    code="validation_error",
                    message=f"Invalid structured outputs: {struct_output_req.grammar}",
                )
                return False
            # Grammar is ready, promote to waiting state.
            request.status = RequestStatus.WAITING

        elif request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            # Streaming input request waiting for more input.
            assert not request.streaming_queue
            return False

        return True

    def _update_from_kv_xfer_finished(self, kv_connector_output: KVConnectorOutput):
        """KV Connector: update the scheduler state based on the output.

        The Worker side connectors add finished_recving and
        finished_sending reqs to the output.
        * if finished_sending: free the blocks
        # if finished_recving: add to state so we can
            schedule the request during the next step.
        """
        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)

        # KV Connector:: update recv and send status from last step.
        for req_id in kv_connector_output.finished_recving or ():
            logger.debug("Finished recving KV transfer for request %s", req_id)
            assert req_id in self.requests
            req = self.requests[req_id]
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                self.finished_recving_kv_req_ids.add(req_id)
                self._set_kv_fetch_stage(req, KV_FETCH_COMPLETED_WAITING)
            else:
                assert RequestStatus.is_finished(req.status)
                self._aborted_kv_loads.discard(req_id)
                self._free_blocks(self.requests[req_id])
        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            assert req_id in self.requests
            self._free_blocks(self.requests[req_id])

    def _update_requests_with_invalid_blocks(
        self,
        requests: Iterable[Request],
        invalid_block_ids: set[int],
        num_scheduled_tokens: dict[str, int],
        evict_blocks: bool = True,
    ) -> tuple[set[str], int, set[int]]:
        """Identify and update requests affected by invalid KV cache blocks.

        This method scans the given requests, detects those with invalid blocks
        and adjusts their `num_computed_tokens` to the longest valid prefix.
        For observability, it also accumulates the total number of tokens that
        will need to be recomputed across all affected requests.

        Args:
            requests: The set of requests to scan for invalid blocks.
            invalid_block_ids: IDs of invalid blocks.
            num_scheduled_tokens: req_id -> number of scheduled tokens.
            evict_blocks: Whether to collect blocks for eviction (False for
                async requests which aren't cached yet).

        Returns:
            tuple:
                - affected_req_ids (set[str]): IDs of requests impacted by
                invalid blocks.
                - total_affected_tokens (int): Total number of tokens that must
                be recomputed across all affected requests.
                - blocks_to_evict (set[int]): Block IDs to evict from cache,
                including invalid blocks and downstream dependent blocks.

        """
        affected_req_ids: set[str] = set()
        total_affected_tokens = 0
        blocks_to_evict: set[int] = set()
        # If a block is invalid and shared by multiple requests in the batch,
        # these requests must be rescheduled, but only the first will recompute
        # it. This set tracks blocks already marked for recomputation.
        marked_invalid_block_ids: set[int] = set()
        for request in requests:
            is_affected = False
            marked_invalid_block = False
            req_id = request.request_id
            # One block-id list per KV cache group; single-group layouts are
            # enforced by _handle_invalid_blocks.
            (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)
            # We iterate only over blocks that may contain externally computed
            # tokens
            req_num_computed_tokens = (
                request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0)
            )

            req_num_computed_blocks = (
                req_num_computed_tokens + self.block_size - 1
            ) // self.block_size
            for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):
                if block_id not in invalid_block_ids:
                    continue

                is_affected = True

                if block_id in marked_invalid_block_ids:
                    # This invalid block is shared with a previous request
                    # and was already marked for recomputation.
                    # This means this request can still consider this block
                    # as computed when rescheduled.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    continue

                marked_invalid_block_ids.add(block_id)

                if marked_invalid_block:
                    # This request has already marked an invalid block for
                    # recomputation and updated its num_computed_tokens.
                    continue

                marked_invalid_block = True
                # Truncate the computed tokens at the first failed block
                request.num_computed_tokens = idx * self.block_size
                num_affected_tokens = (
                    req_num_computed_tokens - request.num_computed_tokens
                )
                total_affected_tokens += num_affected_tokens

                # collect invalid block and all downstream dependent blocks
                if evict_blocks:
                    blocks_to_evict.update(req_block_ids[idx:])

            if is_affected:
                if not marked_invalid_block:
                    # All invalid blocks of this request are shared with
                    # previous requests and will be recomputed by them.
                    # Revert to considering only cached tokens as computed.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    total_affected_tokens += (
                        request.num_computed_tokens - req_num_computed_tokens
                    )
                    request.num_computed_tokens = req_num_computed_tokens

                affected_req_ids.add(request.request_id)
                # The flat channel vs a pending plan:
                # this truncation may cut BELOW the plan's rollback and
                # evict blocks inside the repair's write set, while the
                # plan survives and its repair step would execute over
                # the partially-evicted span. The flat channel's
                # evidence and the structured evidence conflict:
                # dissolve the plan to the recompute fallback (the
                # flat truncation already owns the request).
                flat_conflicted_plan = (
                    self.recovery_plans.pop(request.request_id, None)
                    if self._evidence_state_initialized
                    else None
                )
                if flat_conflicted_plan is not None:
                    self._remember_consumed_evidence(flat_conflicted_plan)
                    logger.warning(
                        "Repair plan dissolved: req=%s conflicted with the "
                        "flat invalid-blocks channel -> the flat recompute "
                        "path owns the request",
                        request.request_id,
                    )

        return affected_req_ids, total_affected_tokens, blocks_to_evict

    # Per-layer fixed cost of a repair layer.forward in equivalent
    # prefill tokens: at small spans the launch-bound floor dominates, so
    # each replayed layer charges this floor beside the layer-fraction
    # term (a conservative placeholder).
    REPAIR_LAYER_EQ_TOKENS = 32

    # The admission-deferral bound for a DETECTED recovery plan. A pending
    # plan is the drain-first admission of its steps: every OTHER request
    # is frozen while the plan waits for a budget it can fit, so an
    # unbounded deferral is an engine-wide stall rather than a
    # per-request one. At the bound the plan dissolves and the ordinary
    # rollback-recompute path (which chunks, and therefore always makes
    # progress) owns the request. 32 steps bounds the extra recovery
    # latency to roughly 0.3-3 s at typical engine cadences (10-100 ms
    # per step) while riding out transient pressure -- preemption
    # cascades, connector drains, bursty prefills -- that resolves
    # within a few scheduling rounds. Not a public knob: the fallback it
    # arms (full recompute) is always correct, only more expensive.
    REPAIR_MAX_ADMISSION_DEFERRALS = 32
    # Liveness bound: async transfer evidence may stay deferred (the
    # attempt not yet terminal) for at most this many update steps before
    # the request is failed closed to the recompute fallback -- a
    # connector whose receive thread never returns the request must not
    # strand it in WAITING_FOR_REMOTE_KVS holding blocks indefinitely.
    ASYNC_REPORT_MAX_DEFERRAL_STEPS = 32

    def _repair_estimate(self, n_eff: int, layer_end: int) -> int:
        n_layers = self.vllm_config.model_config.hf_config.num_hidden_layers
        return math.ceil(
            n_eff * layer_end / n_layers + layer_end * self.REPAIR_LAYER_EQ_TOKENS
        )

    @staticmethod
    def _report_evidence_key(report: KVLoadFailureReport) -> tuple:
        """The report's evidence identity: every failure-describing field
        except the diagnostic source_rank tag. Two reports with equal keys
        are the SAME evidence -- the identical duplicate every TP rank
        produces for one fault -- while different keys for one request
        are divergent evidence that no single recovery can represent."""
        return (
            report.request_id,
            tuple(report.failed_block_ids),
            report.token_span,
            report.rollback_boundary,
            tuple(report.observed_failed_layers),
            report.healthy_above_span,
            report.restored_extent,
        )

    def _handle_repair_evidence(
        self,
        reports: list[KVLoadFailureReport] | None,
        attestations: list[KVLoadAttestation] | None,
        terminal_async_req_ids: set[str] | None = None,
    ) -> set[str]:
        """Consume the step's structured transfer evidence under the
        repair policy: the negative half (failure reports) plus the
        positive half (per-rank transfer attestations).

        The attestation pass runs first and INDEPENDENTLY of the
        reports: violations of the positive contract -- malformed
        accounting, divergent attestations from one rank, or a rank
        that attests failed or certification-failed objects without
        the matching report -- fail the request closed to the
        recompute fallback even when the reports look clean. A rank's
        silence about its own corruption is exactly what the
        attestation exists to catch (the silent-rank gap).

        The report pass then requires, for every request it would
        authorize, terminal attestations from EVERY rank of the TP
        group: incomplete evidence never authorizes a recovery.

        Attestations are FOLDED into the per-request accumulated
        state before any gate reads them, so evidence split across
        passes (an async rank attesting before its peers) is judged at
        the delivery's decision points rather than per arrival. The
        reports of async attempts that are still in flight are deferred
        (stored) until the attempt is provably terminal: judging their
        completeness at arrival would reject a delivery whose evidence
        is merely split across passes.
        """
        reports = reports or []
        attestations = attestations or []
        terminal_async_req_ids = terminal_async_req_ids or set()
        if not self.repair_kv_load_failures:
            # A non-repair policy cannot interpret structured
            # transfer evidence. The in-tree producers never emit it
            # under these policies (evidence production is
            # repair-gated), so this is a nonconforming connector --
            # fail the AFFECTED REQUESTS closed, never the engine. A
            # reported failure degrades the request to the recompute
            # floor (the flat channels' conservative treatment for a
            # connector-reported failure); attestations alone assert
            # that nothing failed and are dropped loudly with no state
            # change. Under kv_load_failure_policy='fail' the affected
            # requests are errored by the caller's existing union.
            kv_transfer_config = self.vllm_config.kv_transfer_config
            policy = (
                kv_transfer_config.kv_load_failure_policy
                if kv_transfer_config is not None
                else None
            )
            what = "load_failure_reports" if reports else "load_attestations"
            reports_by_req: dict[str, list[KVLoadFailureReport]] = {}
            for report in reports:
                reports_by_req.setdefault(report.request_id, []).append(report)
            attest_only_reqs = {a.request_id for a in attestations} - set(
                reports_by_req
            )
            failed: set[str] = set()
            for req_id in sorted(set(reports_by_req) | attest_only_reqs):
                request = self.requests.get(req_id)
                if request is None or request.is_finished():
                    continue
                logger.error(
                    "KV connector reported %s for req=%s while "
                    "kv_load_failure_policy=%r; the repair policy is "
                    "required to consume transfer evidence.",
                    what,
                    req_id,
                    policy,
                )
                if req_id in reports_by_req:
                    self._reject_uninterpretable_reports(
                        request,
                        reports_by_req[req_id],
                        "structured transfer evidence under a non-repair "
                        "policy: the repair policy is required to consume "
                        "transfer evidence",
                        set(),
                        trusted_rollback=0,
                        mark_external_untrusted=True,
                    )
                    # An async attempt whose evidence failed closed must
                    # leave WAITING_FOR_REMOTE_KVS through the promotion's
                    # FAILURE arm (the flat channels' treatment): that arm
                    # frees the claim's blocks and applies the failed-load
                    # zeroing bookkeeping.
                    if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                        self.failed_recving_kv_req_ids.add(req_id)
                    failed.add(req_id)
            return failed
        # The identity pass: partition the step's evidence by transfer
        # generation. Only evidence of the request's LATEST scheduled
        # attempt is current; anything else is a late artifact of an
        # older attempt (dropped -- it can neither authorize nor
        # dissolve anything) or a contract violation (no identity, or
        # an identity the scheduler never issued) that fails the
        # request closed.
        (
            reports_by_request,
            attest_by_request,
            violations,
        ) = self._partition_transfer_evidence(reports, attestations)
        skip_req_ids: set[str] = set()
        for req_id, reason, request_reports in violations:
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                continue
            self._reject_uninterpretable_reports(
                request,
                request_reports,
                reason,
                skip_req_ids,
                trusted_rollback=0,
                mark_external_untrusted=True,
            )
        # Defer the reports of async attempts still in flight.
        # Their completeness cannot be judged yet (the attempt's other
        # ranks may not have attested), and deferring BEFORE the
        # attestation reconciliation keeps the deferred reports counted
        # as present for the ranks that already reported.
        for req_id in list(reports_by_request):
            request = self.requests.get(req_id)
            if (
                request is not None
                and request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
                and req_id not in terminal_async_req_ids
            ):
                if req_id not in self._req_pending_async_reports:
                    # A fresh deferral epoch: a stale counter of an
                    # earlier (judged) deferral must not age this one.
                    self._req_async_report_deferrals[req_id] = 0
                self._req_pending_async_reports.setdefault(req_id, []).extend(
                    reports_by_request.pop(req_id)
                )
        # Fold this step's validated current-generation attestations
        # into the per-request accumulated state every gate below (and
        # every publication point) reads.
        self._fold_transfer_attestations(attest_by_request, skip_req_ids)
        skip_req_ids |= self._check_transfer_attestations(reports_by_request)
        skip_req_ids |= self._handle_repair_reports(
            [
                report
                for request_reports in reports_by_request.values()
                for report in request_reports
            ],
            evidence_failed=skip_req_ids,
        )
        return skip_req_ids

    @staticmethod
    def _attestation_evidence_key(attestation: KVLoadAttestation) -> tuple:
        """The attestation's evidence identity: every evidence-describing
        field including the aggregator-stamped rank. Two attestations
        with equal keys are the SAME statement -- the identical duplicate
        a connector may re-deliver -- while different keys for one rank
        and attempt are divergent evidence."""
        return (
            attestation.request_id,
            attestation.source_rank,
            attestation.transfer_generation,
            attestation.expected_objects,
            attestation.loaded_objects,
            attestation.failed_objects,
            attestation.certification,
            attestation.certified_objects,
            attestation.certification_failures,
            attestation.terminal,
        )

    def _fold_transfer_attestations(
        self,
        attest_by_request: dict[str, list[KVLoadAttestation]],
        skip_req_ids: set[str],
    ) -> None:
        """Fold the step's current-generation attestations into the
        per-request accumulated evidence state.

        Only live requests whose evidence was not already failed closed
        this step accumulate; identical duplicates are absorbed, so the
        per-request list stays bounded by the TP group size (a divergent
        re-attestation from one rank fails the request closed in
        _check_transfer_attestations before it can accumulate). The
        accumulated state is reset when a new attempt is scheduled and
        cleared when the request finishes, is preempted, or fails
        closed."""
        for req_id, request_attestations in attest_by_request.items():
            if req_id in skip_req_ids or req_id in self._untrusted_external_reqs:
                continue
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                continue
            accumulated = self._req_transfer_attestations.setdefault(req_id, [])
            for attestation in request_attestations:
                key = self._attestation_evidence_key(attestation)
                if any(
                    self._attestation_evidence_key(existing) == key
                    for existing in accumulated
                ):
                    continue
                accumulated.append(attestation)

    def _partition_transfer_evidence(
        self,
        reports: list[KVLoadFailureReport],
        attestations: list[KVLoadAttestation],
    ) -> tuple[
        dict[str, list[KVLoadFailureReport]],
        dict[str, list[KVLoadAttestation]],
        list[tuple[str, str, list[KVLoadFailureReport]]],
    ]:
        """Attribute the step's evidence to transfer attempts by
        generation.

        Returns (current reports by request, current attestations by
        request, per-request identity violations). Current means: the
        evidence's echoed generation equals the request's latest
        SCHEDULED attempt generation. A report whose generation is
        older is a late duplicate of a consumed attempt -- dropped,
        never able to authorize or dissolve anything, so a genuine new
        attempt's failure is not discarded as a duplicate either (its
        report carries the new attempt's generation). A report with
        no identity, or one naming a generation the scheduler never
        issued, is a contract violation collected for the caller to
        fail closed. Attestations are informational: non-current ones
        simply do not count toward the completeness gate.
        """
        reports_by_request: dict[str, list[KVLoadFailureReport]] = {}
        attest_by_request: dict[str, list[KVLoadAttestation]] = {}
        violations: list[tuple[str, str, list[KVLoadFailureReport]]] = []
        for report in reports:
            req_id = report.request_id
            current = self._req_transfer_generation.get(req_id)
            if current is None:
                # No scheduled external load attempt for this request:
                # the report is unattributable (a very late artifact
                # of a finished request, or a bogus connector). The
                # request holds no external data to distrust; drop.
                logger.error(
                    "KV repair evidence dropped: report for req=%s names "
                    "no scheduled external load attempt (generation=%r)",
                    req_id,
                    report.transfer_generation,
                )
                continue
            if report.transfer_generation is None:
                violations.append(
                    (
                        req_id,
                        (
                            "report without a transfer identity (the connector "
                            "must implement the transfer-identity contract)"
                        ),
                        [report],
                    )
                )
                continue
            if report.transfer_generation > current:
                violations.append(
                    (
                        req_id,
                        (
                            "report names transfer generation "
                            f"{report.transfer_generation} but the latest "
                            f"scheduled attempt is {current} (an identity the "
                            "scheduler never issued)"
                        ),
                        [report],
                    )
                )
                continue
            if report.transfer_generation < current:
                logger.info(
                    "KV repair evidence dropped: late report of the "
                    "consumed attempt %d for req=%s (latest attempt is %d)",
                    report.transfer_generation,
                    req_id,
                    current,
                )
                continue
            reports_by_request.setdefault(req_id, []).append(report)
        for attestation in attestations:
            req_id = attestation.request_id
            current = self._req_transfer_generation.get(req_id)
            if current is None:
                # No scheduled attempt for this request: unattributable
                # (a very late artifact, or a bogus connector). Dropped
                # loudly, symmetric with the reports rule.
                logger.error(
                    "KV repair evidence dropped: attestation for req=%s "
                    "names no scheduled external load attempt "
                    "(generation=%r)",
                    req_id,
                    attestation.transfer_generation,
                )
                continue
            if attestation.transfer_generation is None:
                # The identity contract is mandatory under the repair
                # policy: an attestation without it cannot be pinned to
                # any attempt. Loud, symmetric with the reports rule
                # .
                violations.append(
                    (
                        req_id,
                        (
                            "attestation without a transfer identity (the "
                            "connector must implement the transfer-identity "
                            "contract)"
                        ),
                        [],
                    )
                )
                continue
            if attestation.transfer_generation != current:
                if attestation.transfer_generation > current:
                    # An identity the scheduler never issued for
                    # this request -- symmetric with the reports rule. A
                    # connector must never be able to introduce a transfer
                    # identity (the generations are scheduler-authoritative
                    # and globally monotonic); naming one that was never
                    # scheduled is a contract violation that fails the
                    # request closed, not an ignorable artifact.
                    violations.append(
                        (
                            req_id,
                            (
                                "attestation names transfer generation "
                                f"{attestation.transfer_generation} but the "
                                f"latest scheduled attempt is {current} (an "
                                "identity the scheduler never issued)"
                            ),
                            [],
                        )
                    )
                    continue
                # OLDER than the latest attempt: a late artifact of a
                # consumed attempt -- dropped, never able to authorize
                # or dissolve anything (a genuine new failure carries
                # the new attempt's generation).
                logger.info(
                    "KV repair evidence dropped: late attestation of the "
                    "consumed attempt %d for req=%s (latest attempt is %d)",
                    attestation.transfer_generation,
                    req_id,
                    current,
                )
                continue
            attest_by_request.setdefault(req_id, []).append(attestation)
        return reports_by_request, attest_by_request, violations

    def _delivery_evidence_gate(self, request: Request) -> str | None:
        """Whether the request's latest transfer attempt carries the
        complete, content-certified positive evidence that publication
        requires. Returns a human-readable rejection reason, or None
        when the delivery is verified.

        Two separable requirements: transfer COMPLETENESS -- a
        terminal, well-formed attestation from every rank of the TP
        group, over the attempt's full lifetime (the accumulated
        state, not just one step's arrivals) -- and CONTENT
        CERTIFICATION -- every rank's attestation must independently
        prove content identity for the objects it loaded
        (KVLoadAttestation.certifies_content: a real certification
        level, per-object digest ("sha256") or content-addressed
        immutable keys under the documented store-trust assumption
        ("content"), covering every loaded object with zero
        failures). A successful GET proves only that bytes arrived;
        "size"/"none" are the connector explicitly reporting that it
        could NOT verify content, and under the repair policy such a
        delivery fails closed to the recompute fallback -- symmetric
        with the recovery path, where an uncertified above-span extent
        is recomputed rather than restored."""
        request_attestations = self._req_transfer_attestations.get(
            request.request_id, []
        )
        gate = self._transfer_evidence_complete(request_attestations)
        if gate is not None:
            return gate
        if not all(
            attestation.certifies_content()
            for attestation in request_attestations
            if attestation.terminal
        ):
            return (
                "delivery not content-certified: every TP rank must attest "
                'a real certification level ("sha256" or "content") covering '
                'its loaded objects; "size"/"none" attest that content was '
                "NOT verified"
            )
        return None

    def _check_delivery_completeness(self, skip_req_ids: set[str]) -> set[str]:
        """The publication gate for deliveries with NO failure reports
        : the report path checks
         completeness before authorizing a recovery, but an all-healthy
         delivery publishes and consumes the claim through paths that
         never consult the report gate -- a silent rank's unverified
         shard would be published, shared, and consumed as trusted.

         The gate runs at EVERY update_from_output under the
         repair policy -- whether or not any evidence arrived in the
         step -- because a pending sync-claim publication is due
         exactly now, and the absence of evidence is a contract
         violation, never success. (An earlier gate ran only when some
         report or attestation arrived in the step, so a fully silent
         step -- a worker returning no connector output at all, or one
         whose evidence lists were empty -- published the claim
         unverified.) Async attempts are NOT judged here: their
         publication point is the promotion in
         _update_waiting_for_remote_kv, after the finished-recving pass
         has folded the last arrivals.

         Publication additionally requires content
         certification (see _delivery_evidence_gate): transfer
         completion alone authorizes nothing."""
        if not self._delayed_sync_claim_pubs:
            return set()
        newly_failed: set[str] = set()
        for req_id in list(self._delayed_sync_claim_pubs):
            if req_id in skip_req_ids:
                continue
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                continue
            gate = self._delivery_evidence_gate(request)
            if gate is not None:
                self._reject_uninterpretable_reports(
                    request,
                    [],
                    "unverified delivery: " + gate,
                    newly_failed,
                    trusted_rollback=0,
                    mark_external_untrusted=True,
                )
                self._req_transfer_attestations.pop(req_id, None)
                self._req_pending_async_reports.pop(req_id, None)
        return newly_failed

    def _check_transfer_attestations(
        self,
        reports_by_request: dict[str, list[KVLoadFailureReport]],
    ) -> set[str]:
        """Validate the POSITIVE transfer evidence of every request with
        live accumulated attestations or this-step reports.

        Validation runs over the accumulated per-request state (the
        fold has already merged this step's arrivals), and the report
        reconciliation counts both this step's reports and the deferred
        reports of async attempts still in flight -- a rank whose failure
        report was deferred is not "report-less" for reconciliation
        purposes.

        Returns the request ids whose evidence failed closed; each such
        request truncates to rollback 0 (no attestation vouches for any
        span of the external claim) and the ordinary
        rollback-recompute path owns it. These are connector contract
        violations, not ordinary load failures: loud, monitored, and
        engine-survivable.
        """
        skip_req_ids: set[str] = set()
        candidates = set(reports_by_request) | set(self._req_transfer_attestations)
        for req_id in candidates:
            if req_id in self.recovery_plans:
                # A pending recovery plan owns the request: its admission
                # already consumed and reconciled the attempt's complete
                # evidence (this pass ran before the admission). A LATE
                # re-delivery of the same attempt's attestation -- e.g. a
                # connector re-emitting it after the plan was admitted --
                # carries no report this step and must not be read as a
                # 'failed objects without a report' violation that would
                # dissolve the legitimately admitted plan (a
                # false-positive rejection; the recompute fallback is
                # safe but wasteful). The plan's own verdict machinery
                # owns the request from admission.
                continue
            request_attestations = self._req_transfer_attestations.get(req_id, [])
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                continue
            request_reports = reports_by_request.get(req_id, [])
            # The reconciliation counts the deferred reports of an async
            # attempt still in flight as present: the rank DID report.
            request_reports = request_reports + self._req_pending_async_reports.get(
                req_id, []
            )
            reason = self._attestation_violation(
                request, request_attestations, request_reports
            )
            if reason is not None:
                self._reject_uninterpretable_reports(
                    request,
                    request_reports,
                    reason,
                    skip_req_ids,
                    trusted_rollback=0,
                    mark_external_untrusted=True,
                )
                self._req_transfer_attestations.pop(req_id, None)
                self._req_pending_async_reports.pop(req_id, None)
        return skip_req_ids

    def _attestation_violation(
        self,
        request: Request,
        request_attestations: list[KVLoadAttestation],
        request_reports: list[KVLoadFailureReport],
    ) -> str | None:
        """Whether the request's attestations violate the positive
        evidence contract. Returns a human-readable rejection reason,
        or None when the attestations are well-formed, per-rank
        consistent, and (where terminal) reconciled with the failure
        reports. The caller owns the failure mode."""
        # The structural battery: terminal accounting must cover the
        # complete expected object set (loaded + failed == expected), a
        # rank cannot certify more than it loaded, and a certification
        # failure requires a certification capable of detecting it.
        for attestation in request_attestations:
            if not attestation.well_formed():
                return (
                    "malformed attestation accounting from rank "
                    f"{attestation.source_rank} (expected="
                    f"{attestation.expected_objects} loaded="
                    f"{attestation.loaded_objects} failed="
                    f"{attestation.failed_objects} terminal="
                    f"{attestation.terminal})"
                )
        # Per-rank identity: one rank, one attestation per attempt.
        # Identical duplicates are absorbed; divergent content from one
        # rank is unattributable evidence and fails closed.
        by_rank: dict[int | None, list[KVLoadAttestation]] = {}
        for attestation in request_attestations:
            by_rank.setdefault(attestation.source_rank, []).append(attestation)
        for rank, rank_attestations in by_rank.items():
            first = rank_attestations[0]
            for other in rank_attestations[1:]:
                diverged = (
                    other.transfer_generation,
                    other.expected_objects,
                    other.loaded_objects,
                    other.failed_objects,
                    other.certification,
                    other.certified_objects,
                    other.certification_failures,
                    other.terminal,
                ) != (
                    first.transfer_generation,
                    first.expected_objects,
                    first.loaded_objects,
                    first.failed_objects,
                    first.certification,
                    first.certified_objects,
                    first.certification_failures,
                    first.terminal,
                )
                if diverged:
                    return (
                        f"divergent attestations from rank {rank} for one "
                        "request and attempt"
                    )
        # A terminal attestation that observed failures or
        # certification failures must be accompanied by THAT RANK'S
        # structured failure evidence in the same step: reports are
        # per-rank statements (the aggregator stamps each with its
        # originating worker), and a peer's report does not cover a
        # rank's own corrupt cells. A rank whose connector detected
        # corrupt or missing content and reported nothing -- while a
        # peer's clean-looking report arrives -- is exactly the
        # silent-corruption shape this contract exists to catch.
        reported_ranks = {r.source_rank for r in request_reports}
        reports_by_rank: dict[int | None, list[KVLoadFailureReport]] = {}
        for report in request_reports:
            reports_by_rank.setdefault(report.source_rank, []).append(report)
        for attestation in request_attestations:
            if not attestation.terminal:
                continue
            rank_reports = reports_by_rank.get(attestation.source_rank, [])
            if attestation.source_rank not in reported_ranks:
                if attestation.failed_objects > 0:
                    return (
                        f"rank {attestation.source_rank} attests "
                        f"{attestation.failed_objects} failed objects "
                        "without its own failure report"
                    )
                if attestation.certification_failures > 0:
                    return (
                        f"rank {attestation.source_rank} attests "
                        f"{attestation.certification_failures} certification "
                        "failures without its own failure report"
                    )
                continue
            # The count reconciliation: a rank's attestation must
            # account for exactly the failed objects its own reports
            # describe. The documented object unit is one cell per
            # (rank, layer, block) -- the v1 full-attention pool has a
            # single part per layer -- so the relation is failed
            # blocks x failed layers per report (a
            # raw object-vs-block equality would fail-close every
            # conforming connector whose failed block spans more than
            # one layer). An attestation counting more leaves failures
            # outside the reported span -- un-repaired holes the plan
            # would never cover; counting fewer contradicts its own
            # report.
            reported_failed = sum(
                len(r.failed_block_ids) * len(r.observed_failed_layers)
                for r in rank_reports
            )
            if attestation.failed_objects != reported_failed:
                return (
                    f"rank {attestation.source_rank} attests "
                    f"{attestation.failed_objects} failed objects but its "
                    f"reports describe {reported_failed} object(s) "
                    "(failed blocks x failed layers)"
                )
        return None

    def _remember_consumed_evidence(self, plan: RecoveryPlan) -> None:
        """Record that this plan's report evidence reached a terminal
        disposition (committed, aborted, or dissolved -- EVERY pop of
        the plan without a verdict included). A LATE
        re-delivery of the same evidence -- same transfer generation
        and same evidence key -- is a duplicate of an attempt whose
        outcome is already known: absorbed silently, never
        re-authorized. A genuine repeated failure carries a NEW
        generation (a new scheduled attempt) and is never absorbed by
        this record. Entries of OLDER generations are pruned on
        insert: the partition drops older-generation reports before
        they can reach the absorption check, so only the latest
        generation's pairs are ever matchable (no
        fixed-count cap that a burst of same-generation entries can
        evict)."""
        if plan.report_evidence is None:
            return
        consumed = [
            (gen, key)
            for gen, key in self._consumed_report_evidence.get(plan.request_id, [])
            if gen == plan.transfer_generation
        ]
        entry = (plan.transfer_generation, plan.report_evidence)
        if entry not in consumed:
            consumed.append(entry)
        self._consumed_report_evidence[plan.request_id] = consumed

    def _transfer_evidence_complete(
        self, request_attestations: list[KVLoadAttestation]
    ) -> str | None:
        """The completeness gate: whether every rank of the TP group
         supplied a terminal, well-formed attestation for this request.
         Returns a rejection reason naming the shortfall, or None when
         complete. Reports are negative evidence -- a rank whose GET
         'succeeded' against corrupt or stale content reports nothing at
         all -- so only the positive attestations can prove that every
         participating rank completed and verified its portion.

         The count is an EXACT SET over the TP group's rank indices
        : the aggregator stamps source_rank over the
         gathered worker outputs, which under PP>1 covers the whole
         world -- a count-based check is satisfiable by the wrong PP
         stage's workers while the shard-holding stage stays silent.
         Exact equality against set(range(tp_size)) makes that shape
         fail closed; the repair policy is additionally rejected at
         init when PP>1 (the v1 evidence contract is TP-scoped).
         Vacuous attestations (expected_objects == 0) never count
        : a rank with a real external claim holds at
         least one object."""
        terminal_ranks: set[int] = set()
        for attestation in request_attestations:
            if (
                attestation.terminal
                and attestation.well_formed()
                and attestation.source_rank is not None
                # A vacuous attestation never counts: a rank holding a
                # real external claim has at least one expected object.
                and attestation.expected_objects >= 1
            ):
                terminal_ranks.add(attestation.source_rank)
        expected = set(range(self._evidence_tp_size))
        if terminal_ranks != expected:
            return (
                f"incomplete transfer evidence: {len(terminal_ranks)} of "
                f"{self._evidence_tp_size} TP ranks supplied a terminal "
                f"attestation for the request (ranks={sorted(terminal_ranks)})"
            )
        return None

    def _handle_repair_reports(
        self,
        reports: list[KVLoadFailureReport],
        evidence_failed: set[str] | None = None,
    ) -> set[str]:
        """Consume structured load-failure reports under the repair policy.

        Two passes per request. The consensus pass groups the step's
        reports by request and folds identical duplicates (every TP rank
        reports the same fault; the aggregator concatenates the per-rank
        lists, so one fault arrives N times): a request keeps at most one
        piece of evidence. Divergent evidence -- reports naming different
        regions, or the same region with different metadata -- is never
        merged and never authorizes a recovery: the request falls back to
        the ordinary rollback-recompute path (see
        _reject_uninterpretable_reports) and the engine stays alive.
        Before any plan is admitted, the completeness gate requires
        terminal attestations from every TP rank (the positive half of
        the evidence contract, checked in _transfer_evidence_complete):
        failure reports alone cannot prove that a silent rank holds
        healthy KV. The gate reads the accumulated per-request
        attestation state (the fold has merged this step's arrivals), so
        evidence split across passes still completes. The install pass is
        otherwise unchanged: truncate to the rollback boundary, install
        the RecoveryPlan, and return the ids whose (garbage) outputs from
        the detection step must be skipped.
        """
        evidence_failed = evidence_failed or set()
        skip_req_ids: set[str] = set()
        by_request: dict[str, list[KVLoadFailureReport]] = {}
        for report in reports:
            by_request.setdefault(report.request_id, []).append(report)
        for req_id, req_reports in by_request.items():
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                continue
            if req_id in evidence_failed:
                # The attestation pass already failed this request
                # closed this step: its positive evidence violated the
                # contract. A report arriving in the same step must not
                # resurrect an authorization on top of evidence that
                # was just rejected.
                skip_req_ids.add(req_id)
                continue
            if req_id in self.recovery_plans:
                # A plan already owns this request. Reports matching the
                # plan's evidence AND its transfer generation are
                # same-step duplicates (absorbed); a report of a NEWER
                # attempt (the only non-matching generation that can
                # reach here -- the partition drops older ones) means a
                # NEW scheduled attempt failed while this plan was
                # pending: the plan is stale, it is DISSOLVED and the
                # fresh evidence falls through to admission (the old
                # behavior absorbed the newer attempt's report and let
                # the plan commit over its corruption); anything else of
                # the plan's own generation diverges from the evidence
                # the plan was built on and dissolves it.
                plan = self.recovery_plans[req_id]
                same_attempt = [
                    r
                    for r in req_reports
                    if r.transfer_generation == plan.transfer_generation
                ]
                if len(same_attempt) < len(req_reports):
                    # After the identity partition every report here
                    # carries the LATEST generation, so a report that
                    # does not match the plan's generation is NEWER: a
                    # new scheduled attempt failed while this plan was
                    # pending. The plan is stale -- dissolve it and
                    # fall through to the fresh-admission path with the
                    # newer evidence (it owns the request now).
                    logger.warning(
                        "Repair plan dissolved: req=%s plan_id=%d was built "
                        "on attempt %r but a NEWER attempt's failure "
                        "evidence arrived -> the fresh evidence owns the "
                        "request",
                        req_id,
                        plan.plan_id,
                        plan.transfer_generation,
                    )
                    self._dissolve_recovery_plan(
                        plan, "newer attempt evidence superseded it"
                    )
                    # Fall through: req_reports (the newer evidence) is
                    # admitted fresh below.
                else:
                    divergent = [
                        r
                        for r in same_attempt
                        if self._report_evidence_key(r) != plan.report_evidence
                    ]
                    if divergent:
                        self._reject_uninterpretable_reports(
                            request,
                            divergent,
                            "divergent against the pending plan",
                            skip_req_ids,
                            trusted_rollback=plan.repair_span[0],
                        )
                    continue
            rejection = next(
                (
                    reason
                    for r in req_reports
                    if (reason := self._validate_load_failure_report(r)) is not None
                ),
                None,
            )
            if rejection is not None:
                # Malformed evidence: even the rollback boundary is
                # untrustworthy. Fail closed to a full recompute.
                self._reject_uninterpretable_reports(
                    request, req_reports, "malformed (" + rejection + ")", skip_req_ids
                )
                continue
            unique: list[KVLoadFailureReport] = []
            for report in req_reports:
                key = self._report_evidence_key(report)
                if all(self._report_evidence_key(u) != key for u in unique):
                    unique.append(report)
            if len(unique) > 1:
                self._reject_uninterpretable_reports(
                    request, unique, "divergent", skip_req_ids
                )
                continue
            report = unique[0]
            request = self.requests[report.request_id]
            # The consumed-evidence absorption: this exact (attempt,
            # evidence) pair already reached a terminal disposition
            # (committed, aborted, or dissolved) -- a late
            # re-delivery is a duplicate of an outcome already known,
            # absorbed without re-authorization and without touching
            # the request (its outputs of THIS step are ordinary: the
            # fault was detected in an earlier step).
            if (
                report.transfer_generation,
                self._report_evidence_key(report),
            ) in self._consumed_report_evidence.get(req_id, []):
                logger.info(
                    "KV repair evidence absorbed: late duplicate of the "
                    "consumed attempt %d for req=%s",
                    report.transfer_generation,
                    req_id,
                )
                continue
            # The completeness gate: failure reports are negative
            # evidence -- a rank whose GET 'succeeded' against corrupt
            # or stale content, or one that failed before reporting,
            # supplies no report at all. A recovery may be authorized
            # only when EVERY rank of the TP group supplied a terminal,
            # well-formed attestation for this request. Incomplete
            # evidence fails closed to the recompute fallback (the
            # engine stays alive); it never blocks on waiting for
            # stragglers. The rollback floor is ZERO, not the report's
            # boundary: TP attention reads every rank's shard, so a
            # prefix boundary certified by one rank's report is not
            # certified on a silent rank's shard -- the external claim
            # as a whole demotes to a full recompute.
            request_attestations = self._req_transfer_attestations.get(req_id, [])
            gate = self._transfer_evidence_complete(request_attestations)
            if gate is not None:
                self._reject_uninterpretable_reports(
                    request,
                    req_reports,
                    gate,
                    skip_req_ids,
                    trusted_rollback=0,
                    mark_external_untrusted=True,
                )
                continue
            # Multimodal requests cannot take the repair path: the repair
            # replay embeds the raw placeholder token ids, which is not the
            # embedding the original prefill used (the merged vision
            # features), so the replayed KV would be silently wrong while
            # the geometry-only commit battery passes. Fail closed to the
            # recompute fallback until a correct re-embedding mechanism
            # exists.
            if request.mm_features:
                self._reject_uninterpretable_reports(
                    request,
                    req_reports,
                    "multimodal request (the repair replay cannot re-embed "
                    "vision features; a re-embedding mechanism does not "
                    "exist in v1)",
                    skip_req_ids,
                    trusted_rollback=0,
                )
                continue
            rollback, corrupt_end = (
                report.rollback_boundary,
                report.token_span[1],
            )
            # the rollback prefix is
            # external content too. Its publication -- at the async
            # promotion and at the repair commit -- is authorized by the
            # SAME content certification the all-healthy delivery path
            # requires: without every rank's certifies_content(), the
            # prefix below the failed region is unverified (same-size
            # stale content passes a "size" check), so the repair has
            # nothing safe to offer -- its only value was sparing the
            # prefix -- and the whole claim demotes to the full-recompute
            # fallback, exactly like an uncertified all-healthy delivery.
            # (Truncating the plan's rollback to 0 instead would re-open
            # the connector admission path at num_computed_tokens == 0
            # and start a NEW external attempt under a pending plan.)
            evidence_certifies = all(
                attestation.certifies_content()
                for attestation in request_attestations
                if attestation.terminal
            )
            if not evidence_certifies:
                self._reject_uninterpretable_reports(
                    request,
                    req_reports,
                    "delivery not content-certified (repair path): the "
                    "rollback prefix is external content and requires every "
                    'rank to attest a real certification level ("sha256" or '
                    '"content") covering its loaded objects; "size"/"none" '
                    "attest that content was NOT verified",
                    skip_req_ids,
                    trusted_rollback=0,
                    mark_external_untrusted=True,
                )
                continue
            # The completeness gate proves every rank
            # attested, but nothing yet ties the attested expected set
            # to the claimed extent. Require every terminal rank's
            # expected-object count to COVER the claim's cells (v1's
            # cell unit: (block, layer) of the single full-attention
            # group -- every rank holds every layer): an under-reporting
            # producer can no longer pass a tiny attestation beside a
            # full-claim report and have the commit certify the whole
            # claim as verified.
            claim = self._req_transfer_claim.get(req_id)
            if claim is not None:
                # Unit-agnostic lower bound: the claim's every block
                # must be represented in each terminal rank's expected
                # set (>= one object per claimed block). The attestation
                # contract leaves the object unit to the connector
                # (per block, or per (block, layer) cell as the shipped
                # producers use), so no upper arithmetic is imposed --
                # but no honest producer can cover a 192-block claim
                # with a 2-object expected set.
                min_expected_objects = -(-claim // self.block_size)
                under_covering = [
                    attestation.source_rank
                    for attestation in request_attestations
                    if attestation.terminal
                    and attestation.expected_objects < min_expected_objects
                ]
                if under_covering:
                    self._reject_uninterpretable_reports(
                        request,
                        req_reports,
                        "incomplete extent coverage: the attested "
                        "expected-object sets (min "
                        + str(
                            min(
                                attestation.expected_objects
                                for attestation in request_attestations
                                if attestation.terminal
                            )
                        )
                        + " objects) do not cover the attempt's external "
                        "claim ("
                        + str(min_expected_objects)
                        + " objects for "
                        + str(claim)
                        + " tokens); ranks "
                        + str(under_covering)
                        + " attest fewer objects than the claim contains",
                        skip_req_ids,
                        trusted_rollback=0,
                        mark_external_untrusted=True,
                    )
                    continue
            write_set = self._repair_write_set_block_ids(request, rollback, corrupt_end)
            violating_block = self._write_set_shared_block(write_set)
            if violating_block is not None:
                # Write-set fence: the repair execution writes the span's
                # blocks at every executed layer, so any shared or
                # published block in the write set cannot be repaired
                # without breaking sharer/reader safety. Fail closed: no
                # plan, the ordinary recompute floor owns the request, and
                # the write set and everything downstream are evicted so
                # no future request matches it.
                self._reject_shared_block_repair(request, report, write_set)
                skip_req_ids.add(report.request_id)
                continue
            request.num_computed_tokens = rollback
            self._mark_num_computed_tokens_rewound(request)
            # The async schedule-ahead may have left placeholder accounting
            # for in-flight steps whose outputs are now suppressed.
            request.num_output_placeholders = 0
            layer_end = max(report.observed_failed_layers) + 1
            # The global above-span trust gate: the report's
            # healthy_above_span is ONE RANK's certification of ITS OWN
            # above-span cells, but the commit decision consumes it for
            # every rank's extent. The claim is honored only when every
            # rank's terminal attestation independently proves content
            # identity for its full expected object set
            # (KVLoadAttestation.certifies_content: a real certification
            # level -- per-object digest ("sha256") or content-addressed
            # immutable keys under the store-trust assumption
            # ("content") -- with every object certified and zero
            # failures). Otherwise the plan carries False and the
            # commit caps the restoration at the repair span's end:
            # the untrusted above-span extent is recomputed, never
            # consumed as verified cache.
            healthy_above_span = report.healthy_above_span and evidence_certifies
            self._next_repair_plan_id += 1
            plan = RecoveryPlan(
                request_id=report.request_id,
                repair_span=(rollback, corrupt_end),
                est_equivalent_tokens=self._repair_estimate(
                    corrupt_end - rollback, layer_end
                ),
                layer_end=layer_end,
                restored_extent=report.restored_extent,
                healthy_above_span=healthy_above_span,
                failed_block_ids=list(report.failed_block_ids),
                plan_id=self._next_repair_plan_id,
                report_evidence=self._report_evidence_key(report),
                transfer_generation=report.transfer_generation,
            )
            self.recovery_plans[report.request_id] = plan
            skip_req_ids.add(report.request_id)
            # The attempt's positive evidence is consumed into the plan:
            # its accumulated attestations have served their purpose (the
            # healthy_above_span verdict was derived from them above), and
            # leaving them would let a later re-validation step reconcile
            # them against reports that no longer arrive.
            self._req_transfer_attestations.pop(report.request_id, None)
            logger.info(
                "Repair plan admitted: req=%s plan_id=%d span=[%d,%d) "
                "layer_end=%d E=%d restored_extent=%d "
                "healthy_above_span=%s",
                report.request_id,
                plan.plan_id,
                rollback,
                corrupt_end,
                layer_end,
                plan.est_equivalent_tokens,
                report.restored_extent,
                healthy_above_span,
            )
        return skip_req_ids

    def _mark_num_computed_tokens_rewound(self, request: Request) -> None:
        """Flag a request whose num_computed_tokens just moved other than
        by the ordinary step advance, so the next SchedulerOutput that
        schedules it tells the worker to stage the value over its
        GPU-side count."""
        self._pending_rewound_req_ids.add(request.request_id)

    def _repair_write_set_block_ids(
        self, request: Request, rollback: int, corrupt_end: int
    ) -> list[int]:
        """The COMPLETE physical write set of the repair execution: every
        block covering [rollback, corrupt_end). The repair stream advances
        through the layers below the extent writing each layer's KV
        natively, so the span's blocks are written at EVERY executed layer
        -- the block set is layer-invariant in v1's single kv-cache-group
        geometry, but the fence must cover the whole span: the stream's
        writes go beyond the failed blocks whenever the rollback boundary
        sits below the failed region."""
        (req_block_ids,) = self.kv_cache_manager.get_block_ids(request.request_id)
        first = rollback // self.block_size
        last = -(-corrupt_end // self.block_size)
        return req_block_ids[first:last]

    def _write_set_shared_block(self, write_set: list[int]) -> int | None:
        """The first write-set block that is shared (ref_cnt > 1: another
        request's table references it) or published (block_hash set: any
        future request with the same prefix would match it), or None when
        the whole write set is private and unpublished."""
        block_pool = self.kv_cache_manager.block_pool
        for block_id in write_set:
            block = block_pool.blocks[block_id]
            if block.ref_cnt > 1 or block.block_hash is not None:
                return block_id
        return None

    def _reject_shared_block_repair(
        self,
        request: Request,
        report: KVLoadFailureReport,
        write_set: list[int],
    ) -> None:
        # Fail closed for the shared-block shape: the write set and
        # everything downstream are evicted so no future request matches
        # the untrusted content, and the request truncates to its rollback
        # boundary for the ordinary recompute fallback.
        logger.warning(
            "KV repair refused: the write set spans a shared or published "
            "block; failing closed to the recompute fallback (req=%s, "
            "rollback=%d)",
            request.request_id,
            report.rollback_boundary,
        )
        (req_block_ids,) = self.kv_cache_manager.get_block_ids(request.request_id)
        start_idx = req_block_ids.index(write_set[0])
        blocks_to_evict = set(req_block_ids[start_idx:])
        self.kv_cache_manager.evict_blocks(blocks_to_evict)
        request.num_computed_tokens = report.rollback_boundary
        self._mark_num_computed_tokens_rewound(request)
        # The shared/published write set may already be attached to
        # other requests (ref_cnt > 1 blocks the eviction above only
        # clears hash entries for); rewind every consumer below the
        # failed blocks.
        self._invalidate_failed_blocks_for_consumers(set(report.failed_block_ids))

    def _validate_load_failure_report(self, report: KVLoadFailureReport) -> str | None:
        """The report's geometry battery. Returns None when the report is
        a well-formed single-region failure description for a live
        request, and a human-readable rejection reason otherwise. The
        caller owns the failure mode: a malformed report is a connector
        contract violation, logged at ERROR and answered with the
        request-scoped recompute fallback -- loud enough to monitor,
        never an engine-wide crash."""
        request = self.requests.get(report.request_id)
        if request is None or request.is_finished():
            return "request is unknown or finished"
        if len(report.failed_block_ids) != 1:
            return (
                "names "
                + str(len(report.failed_block_ids))
                + " blocks; v1 repair supports exactly one failed region"
            )
        if not (
            0 <= report.rollback_boundary <= report.token_span[0]
            and report.token_span[1] <= report.restored_extent
            and report.token_span[0] < report.token_span[1]
        ):
            return (
                "malformed geometry: span="
                + str(report.token_span)
                + " rollback="
                + str(report.rollback_boundary)
                + " restored_extent="
                + str(report.restored_extent)
            )
        req_block_ids_groups = self.kv_cache_manager.get_block_ids(report.request_id)
        if len(req_block_ids_groups) != 1:
            # Hybrid/multi-group layouts are outside the single-group
            # v1 evidence contract; refuse loudly per request (mirroring
            # the flat per-block channel's guard) instead of crashing on
            # the unpack.
            return (
                "multi-group KV cache layouts are not supported by the "
                "structured reporting channel (report via failed_recving)"
            )
        (req_block_ids,) = req_block_ids_groups
        for block_id in report.failed_block_ids:
            if block_id not in req_block_ids:
                return (
                    "names block "
                    + str(block_id)
                    + " which is not in the reporting request's block table"
                )
        bs = self.block_size
        if (
            report.rollback_boundary % bs
            or report.token_span[0] % bs
            or report.token_span[1] % bs
        ):
            return (
                "not block-aligned (block_size="
                + str(bs)
                + "): span="
                + str(report.token_span)
                + " rollback="
                + str(report.rollback_boundary)
            )
        # The failed region and the restoration
        # target are properties OF THE ATTEMPT'S EXTERNAL CLAIM -- a
        # failure beyond the claimed extent, or a restoration above it,
        # is a misreporting connector trying to push the commit's
        # frontier (and with it publication) onto content the evidence
        # never vouches for. Both are honest-producer-impossible (the
        # failed object lies inside the claim; the restoration target is
        # the claim's own pre-failure computed count).
        # The report's geometry is request-absolute, so the bound
        # is the attempt's absolute prefix extent (local hit + external
        # delta), not the delta claim alone: a mixed local+external hit
        # legitimately reports spans inside the store prefix that lie
        # beyond the delta. Falls back to the delta claim for attempts
        # recorded before the extent existed (never on this code path).
        extent = self._req_transfer_claim_extent.get(report.request_id) or (
            self._req_transfer_claim.get(report.request_id)
        )
        if extent is not None and report.token_span[1] > extent:
            return (
                "the failed span "
                + str(report.token_span)
                + " lies beyond the attempt's transfer extent ("
                + str(extent)
                + " tokens)"
            )
        if extent is not None and report.restored_extent > extent:
            return (
                "restored_extent="
                + str(report.restored_extent)
                + " exceeds the attempt's transfer extent ("
                + str(extent)
                + " tokens)"
            )
        n_layers = self.vllm_config.model_config.hf_config.num_hidden_layers
        if not report.observed_failed_layers or any(
            not (0 <= layer < n_layers) for layer in report.observed_failed_layers
        ):
            return (
                "invalid failed-layer information "
                + repr(report.observed_failed_layers)
                + " (model has "
                + str(n_layers)
                + " layers)"
            )
        span_first = report.token_span[0] // bs
        span_last = -(-report.token_span[1] // bs)
        if list(req_block_ids[span_first:span_last]) != list(report.failed_block_ids):
            return (
                "the span "
                + str(report.token_span)
                + " maps to blocks "
                + str(list(req_block_ids[span_first:span_last]))
                + " but the report claims "
                + str(report.failed_block_ids)
            )
        return None

    def _invalidate_failed_blocks_for_consumers(
        self, failed_block_ids: set[int]
    ) -> None:
        """Rewind every OTHER request that attached a failed external
        block. When a claim was already published (or its blocks shared)
        before the attempt's failure report was judged, the refusal's
        hash eviction cannot touch blocks other requests still reference
        (ref_cnt > 1): a consumer that matched the published prefix would
        keep attending over the corrupt region. This runs the flat
        channel's consumer-invalidation walk over the same population it
        scans (async waiting + running requests), rewinding each affected
        consumer below the first failed block and evicting the failed
        and downstream blocks so no future request matches them.
        Unpublished-unshared shapes find no consumers -- a no-op."""
        if not failed_block_ids:
            return
        async_load_reqs = [
            req
            for req in self.kv_holding_waiting
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        ]
        _, _, async_evict = self._update_requests_with_invalid_blocks(
            async_load_reqs,
            failed_block_ids,
            {},
            evict_blocks=False,
        )
        # All running requests are scanned (the walk is idempotent for
        # the reporting request: its computed tokens were already
        # rewound to the rollback boundary above, and re-walking the
        # same failed block lands on the same boundary).
        sync_reqs = list(self.running)
        sync_failed, _, sync_evict = self._update_requests_with_invalid_blocks(
            sync_reqs,
            failed_block_ids,
            {},
            evict_blocks=True,
        )
        if sync_evict:
            self.kv_cache_manager.evict_blocks(sync_evict)
        if sync_failed:
            logger.warning(
                "Rewound %d consumer request(s) over failed external "
                "blocks %s (late failure evidence after publication)",
                len(sync_failed),
                sorted(failed_block_ids),
            )

    def _reject_uninterpretable_reports(
        self,
        request: Request,
        reports: list[KVLoadFailureReport],
        reason: str,
        skip_req_ids: set[str],
        trusted_rollback: int | None = None,
        mark_external_untrusted: bool = False,
    ) -> None:
        """Request-scoped fail-closed fallback for uninterpretable report
        evidence (malformed, or divergent across ranks).

        Recovery is REFUSED -- conflicting or malformed evidence is never
        merged into an authorization. The request truncates to the
        largest boundary every well-formed report independently certifies
        as healthy (the minimum rollback boundary; zero when the evidence
        is malformed), everything from that boundary to the request's
        last block is evicted so no future request matches untrusted
        content, and the ordinary rollback-recompute path owns the
        request from here. The engine stays alive: broken evidence
        invalidates the recovery attempt, not the request.
        """
        dissolved = self.recovery_plans.pop(request.request_id, None)
        if dissolved is not None:
            # The plan's evidence was superseded by this rejection: a
            # late re-delivery of it is absorbed, not re-authorized.
            self._remember_consumed_evidence(dissolved)
        # The attempt's accumulated positive evidence dies with the
        # rejection -- it failed closed, and no later gate may read the
        # stale state as if the attempt were still verifiable.
        attempt_attestations = self._req_transfer_attestations.pop(
            request.request_id, None
        )
        self._req_pending_async_reports.pop(request.request_id, None)
        well_formed = [
            r for r in reports if self._validate_load_failure_report(r) is None
        ]
        # The reports' rollback boundaries are claims about
        # content the POSITIVE evidence must back. The single-report path
        # Keep any of the claim only when EVERY TP
        # rank's terminal attestation certifies content -- and the
        # divergent path must not be more permissive. The rollback prefix
        # collapses to the full-recompute floor (0) unless the attempt's
        # terminal attestations are complete (every rank of the TP group
        # supplied one) and each certifies content: an uncertified or
        # silent rank leaves its shard unverified for the kept prefix.
        terminal_atts = (
            [a for a in attempt_attestations if a.terminal]
            if attempt_attestations is not None
            else []
        )
        # EXACT-set completeness, the same rule the primary gate
        # (_transfer_evidence_complete) enforces -- a count-based check
        # would let a forged rank set (a fabricated healthy attestation
        # from a rank outside the TP group) stand in for a silent rank.
        complete = {a.source_rank for a in terminal_atts} == set(
            range(self._evidence_tp_size)
        )
        uncertified_delivery = not (
            complete and all(a.certifies_content() for a in terminal_atts)
        )
        # The truncation boundary: every well-formed report independently
        # certifies [0, its rollback) healthy on its rank, so the minimum
        # rollback is healthy on ALL reporting ranks. A rollback already
        # certified by an installed plan (the divergent-against-plan
        # shape) joins the minimum as trusted evidence -- but only when
        # at least one of the new reports is itself well-formed: with
        # exclusively malformed evidence even the plan's boundary is
        # untrustworthy, and the full-recompute floor (rollback 0)
        # applies.
        certified = [r.rollback_boundary for r in well_formed]
        if trusted_rollback is not None and well_formed:
            certified.append(trusted_rollback)
        rollback = min(certified) if certified else 0
        if uncertified_delivery:
            rollback = 0
        # Flatten across KV cache groups: hybrid layouts reach this
        # refusal too (the battery refuses them per request), and the
        # refusal itself must not crash on the single-group unpack.
        req_block_ids_groups = self.kv_cache_manager.get_block_ids(request.request_id)
        req_block_ids = [
            block_id for group in req_block_ids_groups for block_id in group
        ]
        evict_from = rollback // self.block_size if well_formed else 0
        blocks_to_evict = set(req_block_ids[evict_from:])
        self.kv_cache_manager.evict_blocks(blocks_to_evict)
        request.num_computed_tokens = rollback
        self._mark_num_computed_tokens_rewound(request)
        # The failed blocks may already be attached to OTHER requests
        # (the claim was published, or the blocks shared, before this
        # evidence was judged). evict_blocks only clears hash entries; a
        # consumer's reference keeps the corrupt region matchable and
        # attended-over. Rewind every consumer below the failed blocks
        # and evict them -- including the blocks a MALFORMED report
        # names (bounded to the reporting request's own table, so a
        # fabricated id cannot rewind an unrelated request). The walk
        # rides the single-group upstream helper, so hybrid layouts skip
        # it (the flat channel refuses them outright); their hash entries
        # were already cleared by the direct eviction above.
        if len(req_block_ids_groups) != 1:
            logger.warning(
                "Consumer rewind skipped for req=%s: hybrid (multi-group) KV "
                "layouts are outside the reporting channel's single-group "
                "consumer walk (the flat per-block channel refuses them "
                "outright); the hash entries were cleared by the eviction "
                "above, but already-matched consumers are not rewound.",
                request.request_id,
            )
        else:
            table = set(req_block_ids)
            failed_block_ids = {
                block_id
                for report in reports
                for block_id in report.failed_block_ids
                if block_id in table
            }
            self._invalidate_failed_blocks_for_consumers(failed_block_ids)
        # The async schedule-ahead may have left placeholder accounting
        # for in-flight steps whose outputs are now suppressed.
        request.num_output_placeholders = 0
        skip_req_ids.add(request.request_id)
        if mark_external_untrusted:
            # The transfer-evidence contract itself failed: the external
            # source is untrusted for this request's remaining lifetime
            # (a re-offered hit would loop the re-attempt cycle --
            # source).
            self._untrusted_external_reqs.add(request.request_id)
        logger.error(
            "KV repair refused: %s report evidence for req=%s -- failing "
            "closed to the recompute fallback (rollback=%d, evicted=%d). "
            "Conflicting evidence: %s",
            reason,
            request.request_id,
            rollback,
            len(blocks_to_evict),
            [
                {
                    "source_rank": r.source_rank,
                    "token_span": r.token_span,
                    "rollback_boundary": r.rollback_boundary,
                    "failed_block_ids": r.failed_block_ids,
                    "restored_extent": r.restored_extent,
                    "healthy_above_span": r.healthy_above_span,
                    "observed_failed_layers": r.observed_failed_layers,
                }
                for r in reports
            ],
        )

    def _consume_repair_result(
        self, result: RepairResultReport
    ) -> Literal["committed", "aborted", "stale"]:
        """Consume the repair step's commit verdict: restore the logical
        cached extent on pass, fall back to the ordinary
        rollback-recompute on failure. A result that does not echo the
        pending plan's identity (plan_id, epoch, span) is a stale or
        duplicate report and is rejected without touching the plan.

        Returns the disposition: "committed" (the extent restored),
        "aborted" (the rollback-recompute fallback applied), or "stale"
        (rejected; the plan untouched).
        """
        plan = self.recovery_plans.get(result.request_id)
        if plan is None or plan.state != "ADMITTED":
            return "stale"
        if (
            result.plan_id != plan.plan_id
            or result.epoch != plan.admitted_step
            or result.span != plan.repair_span
        ):
            return "stale"
        plan.state = "REPAIRING"
        request = self.requests.get(result.request_id)
        self.recovery_plans.pop(result.request_id, None)
        # The attempt's evidence reached its terminal disposition: a
        # late re-delivery of the same (generation, evidence) is a
        # duplicate of an outcome already known -- absorbed, never
        # re-authorized. A genuine repeated failure is a NEW scheduled
        # attempt with a NEW generation and is unaffected.
        self._remember_consumed_evidence(plan)
        # Re-examine the folded attestations at the verdict. The
        # admission-time healthy_above_span is one point-in-time
        # certification, and the validation pass skips plan-holders (to
        # absorb late identical duplicates), so a divergent
        # re-attestation folded while the plan was pending may
        # contradict it. Any terminal folded attestation that does not
        # certify content caps the restoration at the repair span's
        # end -- the certification-failed above-span extent is never
        # consumed as verified.
        healthy_above_span = plan.healthy_above_span
        if healthy_above_span:
            for folded in self._req_transfer_attestations.get(result.request_id, []):
                if folded.terminal and not folded.certifies_content():
                    healthy_above_span = False
                    logger.warning(
                        "Repair commit for req=%s plan_id=%d: a folded "
                        "attestation of the attempt does not certify "
                        "content; capping restoration at the repair span "
                        "end %d",
                        result.request_id,
                        plan.plan_id,
                        plan.repair_span[1],
                    )
                    break
        # The attempt's accumulated positive evidence dies with the
        # plan's disposition (nothing may read it as a live attempt's
        # verification anymore).
        self._req_transfer_attestations.pop(result.request_id, None)
        self._req_pending_async_reports.pop(result.request_id, None)
        if request is None:
            # The request object is already gone (it finished before the
            # verdict arrived) -- nothing to restore or truncate.
            return "committed" if result.structural_check else "aborted"
        if result.structural_check:
            # The conditional restoration: the pre-failure extent C is
            # restored ONLY when the report certified the above-span
            # external ranges landed byte-exact. Without the
            # certification the restoration is capped at the repair
            # span's end -- the largest boundary the commit check covers
            # -- and the resume recomputes the untrusted [span_end, C)
            # extent through the ordinary path.
            restored = (
                plan.restored_extent if healthy_above_span else plan.repair_span[1]
            )
            # The result is untrusted input: an aborted verdict reaching
            # the commit arm, or an executed count that does not EXACTLY
            # equal the plan's layer range, must never advance the
            # frontier -- reject loudly with the fail-safe recompute
            # fallback (the worker's structural_check implies layer
            # coverage; this keeps the arm safe against forged
            # summaries too).
            malformed = result.aborted or result.executed_layers != plan.layer_end
            if malformed:
                logger.warning(
                    "Malformed repair result for req=%s plan_id=%d: "
                    "aborted=%s executed_layers=%d -> recompute fallback",
                    result.request_id,
                    plan.plan_id,
                    result.aborted,
                    result.executed_layers,
                )
                self._abort_repair_plan(plan, request, result)
                return "aborted"
            request.num_computed_tokens = restored
            self._mark_num_computed_tokens_rewound(request)
            logger.info(
                "Repair committed: req=%s plan_id=%d restored=%d",
                result.request_id,
                plan.plan_id,
                restored,
            )
            return "committed"
        self._abort_repair_plan(plan, request, result)
        return "aborted"

    def _abort_repair_plan(
        self, plan: RecoveryPlan, request: Request, result: RepairResultReport
    ) -> None:
        """Abort terminal state: the worker's re-zero of partial writes is
        best-effort (a clean full-depth replay that then fails the verdict
        collective leaves its rows un-zeroed); the authoritative
        invalidation happens here -- the failed span is evicted (so no
        future request matches untrustworthy content), the request
        truncates to the rollback boundary, and the ordinary recompute
        fallback owns it. The plan is already purged."""
        if plan.failed_block_ids:
            (req_block_ids,) = self.kv_cache_manager.get_block_ids(request.request_id)
            try:
                failed_idx = req_block_ids.index(plan.failed_block_ids[0])
                blocks_to_evict = set(req_block_ids[failed_idx:])
            except ValueError:
                blocks_to_evict = set(plan.failed_block_ids)
            self.kv_cache_manager.evict_blocks(blocks_to_evict)
        else:
            blocks_to_evict = set()
        request.num_computed_tokens = plan.repair_span[0]
        self._mark_num_computed_tokens_rewound(request)
        logger.warning(
            "Repair aborted: req=%s plan_id=%d span=%s "
            "executed_layers=%d evicted=%d -> recompute fallback",
            result.request_id,
            plan.plan_id,
            plan.repair_span,
            result.executed_layers,
            len(blocks_to_evict),
        )

    def _handle_failed_recving(self, failed_req_ids: set[str]) -> set[str]:
        """Fail closed for layouts whose block IDs are not globally unique.

        Under the repair policy the flat request-level failure channel
        gets the same conservative treatment as under recompute -- an
        earlier revision skipped both the truncation and the arming, so
        the promotion's success arm published the failed claim at its
        full placeholder extent. Truncate to zero and arm the
        promotion's failure arm. A recovery requires structured
        evidence; the flat channel never authorizes one."""
        affected_req_ids: set[str] = set()
        for req_id in failed_req_ids:
            request = self.requests.get(req_id)
            if (
                request is not None
                and request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
            ):
                affected_req_ids.add(req_id)
                if self.recompute_kv_load_failures or self.repair_kv_load_failures:
                    if self._evidence_state_initialized:
                        # The flat request-level channel and the structured
                        # evidence can name the same request in one update
                        # (the aggregator releases failed_recving at the
                        # same terminal pass the reports ride). A recovery
                        # plan the structured arm just admitted is now
                        # detached from the truncation below: void it (its
                        # evidence counts as consumed) and let the flat
                        # recompute path own the request -- the same
                        # conflict rule the per-block channel applies.
                        conflicted_plan = self.recovery_plans.pop(req_id, None)
                        if conflicted_plan is not None:
                            self._remember_consumed_evidence(conflicted_plan)
                            logger.warning(
                                "Repair plan dissolved: req=%s conflicted "
                                "with the flat failed_recving channel -> "
                                "the flat recompute path owns the request",
                                req_id,
                            )
                    request.num_computed_tokens = 0
                    self._mark_num_computed_tokens_rewound(request)
                    self.failed_recving_kv_req_ids.add(req_id)
        return (
            set()
            if (self.recompute_kv_load_failures or self.repair_kv_load_failures)
            else affected_req_ids
        )

    def _handle_invalid_blocks(
        self, invalid_block_ids: set[int], num_scheduled_tokens: dict[str, int]
    ) -> set[str]:
        """Handle requests affected by invalid KV cache blocks.

        Returns:
            Set of affected request IDs to skip in update_from_output main loop.

        """
        if len(self.kv_cache_config.kv_cache_groups) > 1:
            # Block IDs are only unique within a group, so a flat set of
            # invalid block IDs cannot be mapped back to requests.
            raise RuntimeError(
                "A KV connector reported block-level load failures "
                "(invalid_block_ids) which is not supported for models "
                "with multiple KV cache groups. "
                "Connectors must report failed requests via "
                "KVConnectorTransferResults.failed_recving instead."
            )
        if self._evidence_state_initialized and self.recovery_plans:
            # The truncation walk below only sees blocks within the
            # request's already-truncated computed count, so a flat
            # invalid block at or above a plan's rollback boundary (the
            # structured arm runs earlier in this same update and
            # rewinds the frontier to it) is invisible to the walk and
            # its conflict rule. Void any pending plan whose own block
            # table intersects the flat invalid set, independent of the
            # walk's visibility window -- the same conflict rule the
            # failed_recving channel applies.
            for req_id in list(self.recovery_plans):
                (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)
                if invalid_block_ids.intersection(req_block_ids):
                    plan = self.recovery_plans.pop(req_id)
                    self._remember_consumed_evidence(plan)
                    logger.warning(
                        "Repair plan dissolved: req=%s conflicted with "
                        "the flat invalid-blocks channel (the flat "
                        "recompute path owns the request)",
                        req_id,
                    )

        # The repair policy keeps the flat channel's semantics at the
        # recompute floor: a block-level failure without a structured
        # report reschedules, never fails the request.
        should_fail = not (
            self.recompute_kv_load_failures or self.repair_kv_load_failures
        )

        # handle async KV loads (not cached yet, evict_blocks=False)
        async_load_reqs = (
            req
            for req in self.kv_holding_waiting
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        )
        async_failed_req_ids, num_failed_tokens, _ = (
            self._update_requests_with_invalid_blocks(
                async_load_reqs,
                invalid_block_ids,
                num_scheduled_tokens,
                evict_blocks=False,
            )
        )

        total_failed_requests = len(async_failed_req_ids)
        total_failed_tokens = num_failed_tokens

        # handle sync loads (may be cached, collect blocks for eviction)
        sync_failed_req_ids, num_failed_tokens, sync_blocks_to_evict = (
            self._update_requests_with_invalid_blocks(
                self.running, invalid_block_ids, num_scheduled_tokens, evict_blocks=True
            )
        )

        total_failed_requests += len(sync_failed_req_ids)
        total_failed_tokens += num_failed_tokens

        if not total_failed_requests:
            return set()

        # evict invalid blocks and downstream dependent blocks from cache
        # only when not using recompute policy (where blocks will be recomputed
        # and reused by other requests sharing them)
        if sync_blocks_to_evict and not self.recompute_kv_load_failures:
            self.kv_cache_manager.evict_blocks(sync_blocks_to_evict)

        if should_fail:
            all_failed_req_ids = async_failed_req_ids | sync_failed_req_ids
            logger.error(
                "Failing %d request(s) due to KV load failure "
                "(failure_policy=fail, %d tokens affected). Request IDs: %s",
                total_failed_requests,
                total_failed_tokens,
                all_failed_req_ids,
            )
            return all_failed_req_ids

        logger.warning(
            "Recovered from KV load failure: "
            "%d request(s) rescheduled (%d tokens affected).",
            total_failed_requests,
            total_failed_tokens,
        )

        if self.repair_kv_load_failures:
            # Under the repair policy the flat block-level channel
            # cannot vouch for any of the claim. The 'valid prefix' the
            # flat truncation leaves behind is UNVERIFIED external content,
            # and the ordinary path publishes it at the next allocation
            # (every block of the claim becomes prefix-matchable one
            # schedule later). The structured evidence contract is the only
            # authorization to keep any of an external claim: fail the
            # whole claim closed -- full recompute, nothing published, the
            # source marked untrusted -- for every affected request that
            # carries a transfer identity. (Async requests take the same
            # path here; their promotion then only releases the blocks.)
            flat_failed: set[str] = set()
            for req_id in async_failed_req_ids | sync_failed_req_ids:
                request = self.requests.get(req_id)
                if (
                    request is None
                    or request.is_finished()
                    or req_id not in self._req_transfer_generation
                ):
                    continue
                self._reject_uninterpretable_reports(
                    request,
                    [],
                    "flat invalid-block failure under the repair policy: the "
                    "structured evidence contract is required to keep any of "
                    "the external claim",
                    flat_failed,
                    trusted_rollback=0,
                    mark_external_untrusted=True,
                )
            sync_failed_req_ids |= flat_failed

        # Mark async requests with KV load failures for retry once loading completes
        self.failed_recving_kv_req_ids |= async_failed_req_ids
        # Return sync affected IDs to skip in update_from_output
        return sync_failed_req_ids
