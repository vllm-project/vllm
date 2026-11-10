// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Port of `vllm/v1/core/sched/scheduler.py` (`Scheduler`) and
//! `async_scheduler.py` (`AsyncScheduler`).
//!
//! The scheduler owns the requests (in a slab addressed by handles, with an
//! id index), the request queues, the KV cache manager and the encoder cache
//! manager. Python-only collaborators are reached through traits the binding
//! layer implements: [`KvConnector`], [`StructuredOutput`], [`EcConnector`]
//! and [`AuxOutputConnector`].
//!
//! Python objects the scheduler only passes along (sampling params, logprob
//! tensors, pooler outputs, multi-modal payloads, trace headers) stay with
//! the binding layer, keyed by request id or output index.
//!
//! Not supported (rejected at construction): custom encoder cache managers.
//! MFU (`perf_stats`)
//! and CUDA graph stats are attached to [`SchedulerStats`] by the binding
//! layer.

mod config;
mod connector;
mod ec_connector;
mod schedule;
mod structured_output;
mod update;

#[cfg(test)]
mod tests;

use std::collections::VecDeque;
use std::ops::{Index, IndexMut};

use indexmap::{IndexMap, IndexSet};
use rustc_hash::{FxHashMap, FxHashSet};

pub use self::config::{KvTransferConfig, SchedulerClass, SchedulerConfig, SchedulerError};
pub use self::connector::{
    KvConnector, KvConnectorBlockState, KvConnectorOutput, KvTransferParams, RequestFinished,
    Requests,
};
pub use self::ec_connector::{AuxOutputConnector, EcConnector};
pub use self::structured_output::{AcceptAllStructuredOutput, GrammarStatus, StructuredOutput};
pub use self::update::{
    DraftTokenIds, EngineCoreOutput, EngineCoreOutputs, ModelRunnerOutput, RequestError,
    UpdateOutput,
};
use super::encoder_cache_manager::{
    AnyEncoderCacheManager, EncoderCacheManager, EncoderDecoderCacheManager,
};
use super::output::{Opaque, SchedulerOutput};
use super::py_int_set::PyIntSet;
use super::request::{EngineCoreEventType, Request, RequestStatus, StreamingUpdate};
use super::request_queue::{RequestLookup, RequestQueue, SchedulingPolicy};
use super::stats::{
    ConnectorPrefixCacheStats, KvFetchCounts, KvFetchStage, RequestSpecDecodeMetrics,
    SpecDecodeMetricsLevel,
};
use super::utils::remove_all;
use crate::kv_cache::{
    BlockId, KvCacheConfig, KvCacheManager, KvCacheManagerConfig, KvCacheMetricsCollector,
    MambaCacheMode, cdiv,
};

/// Scheduler pause state.
///
/// - UNPAUSED: Normal operation
/// - PAUSE_NEW: No new requests are scheduled, requests already in
///   running state or holding KV blocks are scheduled.
/// - PAUSE_ALL: No requests are scheduled
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PauseState {
    #[default]
    Unpaused,
    PausedNew,
    PausedAll,
}

/// A monotonic clock in seconds (`time.monotonic`).
pub type Clock = Box<dyn Fn() -> f64 + Send>;

/// Request ids whose KV load failed, by how the failure was detected, each
/// in the order the scheduler found them.
#[derive(Clone, Debug, Default)]
pub struct FailedKvLoads {
    /// Reported in `failed_recving`.
    pub recving: IndexSet<String>,
    /// Waiting for an async load that hit an invalid block.
    pub invalid_async: IndexSet<String>,
    /// Running requests whose sync load hit an invalid block.
    pub invalid_sync: IndexSet<String>,
}

/// The requests `update_from_output` finishes with an error, by source.
#[derive(Clone, Debug, Default)]
pub struct ErrorRequestIds {
    /// Grammar compilation failures.
    pub grammar: IndexSet<String>,
    pub encoder_cache_mismatch: IndexSet<String>,
    /// Failed KV loads under the `fail` policy.
    pub failed_kv_loads: FailedKvLoads,
    /// Encoder inputs the EC connector can no longer obtain.
    pub ec_unavailable: IndexSet<String>,
}

impl ErrorRequestIds {
    fn is_empty(&self) -> bool {
        self.grammar.is_empty()
            && self.encoder_cache_mismatch.is_empty()
            && self.failed_kv_loads.recving.is_empty()
            && self.failed_kv_loads.invalid_async.is_empty()
            && self.failed_kv_loads.invalid_sync.is_empty()
            && self.ec_unavailable.is_empty()
    }

    /// All ids, in order of discovery.
    fn all(&self) -> IndexSet<String> {
        let f = &self.failed_kv_loads;
        self.grammar
            .iter()
            .chain(&self.encoder_cache_mismatch)
            .chain(&f.recving)
            .chain(&f.invalid_async)
            .chain(&f.invalid_sync)
            .chain(&self.ec_unavailable)
            .cloned()
            .collect()
    }
}

/// The order in which Python finishes the error requests: it iterates a
/// `set` of their ids, whose order follows the string hashes and how the set
/// was built. The order is observable (which freed blocks are reused first,
/// the order of the outputs), so a binding layer reproduces it.
pub type IdSetOrder = Box<dyn Fn(&ErrorRequestIds) -> IndexSet<String> + Send>;

/// `time.monotonic()`.
pub fn monotonic() -> f64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid, writable timespec.
    let ret = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    assert_eq!(ret, 0, "clock_gettime(CLOCK_MONOTONIC) failed");
    ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
}

/// The scheduler's requests (`Scheduler.requests`), addressed by handle.
///
/// Python keeps a finished request object alive while locals still refer to
/// it after `del self.requests[...]`; here a removed request leaves the id
/// index at once but keeps its slot until the end of the public call
/// ([`Scheduler::flush_removed`]), so handles held for the rest of the call
/// stay valid.
#[derive(Default)]
pub(crate) struct RequestStore {
    slots: Vec<Option<Request>>,
    /// Insertion order of each slot, for Python's dict order.
    seqs: Vec<u64>,
    free_slots: Vec<usize>,
    by_id: FxHashMap<String, usize>,
    next_seq: u64,
    /// Removed from `by_id`; slot freed by `flush`.
    removed: Vec<usize>,
}

impl RequestStore {
    fn insert(&mut self, request: Request) -> usize {
        let seq = self.next_seq;
        self.next_seq += 1;
        let id = request.request_id.clone();
        let h = if let Some(h) = self.free_slots.pop() {
            self.slots[h] = Some(request);
            self.seqs[h] = seq;
            h
        } else {
            self.slots.push(Some(request));
            self.seqs.push(seq);
            self.slots.len() - 1
        };
        self.by_id.insert(id, h);
        h
    }

    /// `del self.requests[request_id]`.
    fn remove(&mut self, h: usize) {
        let request = self.slots[h].as_ref().expect("live request handle");
        let removed = self.by_id.remove(&request.request_id);
        assert_eq!(removed, Some(h), "request is not registered");
        self.removed.push(h);
    }

    /// Free the slots of removed requests, returning the requests.
    fn flush(&mut self) -> impl Iterator<Item = Request> + '_ {
        self.removed.drain(..).map(|h| {
            self.free_slots.push(h);
            self.slots[h].take().expect("live slot")
        })
    }

    /// `self.requests.get(request_id)`.
    pub(crate) fn handle(&self, request_id: &str) -> Option<usize> {
        self.by_id.get(request_id).copied()
    }

    fn len(&self) -> usize {
        self.by_id.len()
    }

    /// Handles in Python's dict order.
    fn handles(&self) -> Vec<usize> {
        let mut handles: Vec<usize> = self.by_id.values().copied().collect();
        handles.sort_unstable_by_key(|&h| self.seqs[h]);
        handles
    }
}

impl Index<usize> for RequestStore {
    type Output = Request;

    fn index(&self, h: usize) -> &Request {
        self.slots[h].as_ref().expect("live request handle")
    }
}

impl IndexMut<usize> for RequestStore {
    fn index_mut(&mut self, h: usize) -> &mut Request {
        self.slots[h].as_mut().expect("live request handle")
    }
}

impl RequestLookup<usize> for RequestStore {
    fn request(&self, handle: usize) -> &Request {
        &self[handle]
    }
}

/// A waiting queue of the scheduler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WaitingQueue {
    Waiting,
    KvHolding,
    AsyncKv,
}

pub struct Scheduler {
    // ---- Configuration ----
    log_stats: bool,
    enable_iteration_details: bool,
    spec_decode_metrics_level: SpecDecodeMetricsLevel,
    is_encoder_decoder: bool,
    is_mm_encoder_only: bool,
    uses_mrope: bool,
    max_num_running_reqs: usize,
    max_num_active_reqs: usize,
    max_num_scheduled_tokens: usize,
    max_num_batched_tokens: usize,
    long_prefill_token_threshold: usize,
    adaptive_long_prefill_threshold: bool,
    enable_chunked_prefill: bool,
    disable_chunked_mm_input: bool,
    async_scheduling: bool,
    max_num_inflight_kv_loads: Option<usize>,
    max_async_kv_loads_per_step: Option<usize>,
    max_model_len: usize,
    num_sampled_tokens_per_step: usize,
    return_sampling_mask: bool,
    report_watermarking: bool,
    enable_prefix_caching: bool,
    cache_block_size: usize,
    block_size: usize,
    hash_block_size: usize,
    policy: SchedulingPolicy,
    max_loras: Option<usize>,
    kv_transfer: Option<KvTransferConfig>,
    recompute_kv_load_failures: bool,
    defer_block_free: bool,
    defer_async_loads: bool,
    /// Whether a preempted request's in-flight output must be dropped; see
    /// `KVConnectorBase_V1.requires_kv_delivery`.
    requires_kv_delivery: bool,
    max_num_encoder_input_tokens: usize,
    num_spec_tokens: usize,
    num_lookahead_tokens: usize,
    draft_slots: usize,
    /// DSV41 SWA bounded replay: groups that declare a replay window are rebuilt
    /// after a prefix hit by recomputing its trailing tokens. One window
    /// for all such groups, so the rewind matches every group's allocation.
    prefix_replay_group_ids: Vec<usize>,
    prefix_replay_tokens: usize,
    num_prefill_lookahead: usize,
    /// The model runner skips drafting in steps where no scheduled request
    /// can use the drafts. Only drafters with their own forward pass reserve
    /// lookahead slots.
    can_skip_drafts: bool,
    dynamic_sd_lookup: Option<Vec<usize>>,
    use_eagle_block_drop: bool,
    num_hash_lookahead_tokens: usize,
    /// Mamba managers already account for their speculative state blocks.
    spec_decode_step_blocks: i64,
    use_pp: bool,
    scheduler_reserve_full_isl: bool,
    /// Cache-aware admission needs a prefix cache to consult, and reorders by
    /// position, which only a FCFS queue has.
    cache_aware_window: usize,
    cache_aware_threshold: f64,
    has_mamba_layers: bool,
    needs_kv_cache_zeroing: bool,
    need_mamba_block_aligned_split: bool,
    // TODO: Support models with multiple Mamba specs that require different
    // prefill checkpoint alignments instead of selecting the first one.
    mamba_prefill_checkpoint_alignment: Option<usize>,
    mamba_has_prefill_checkpoint_blocks: bool,
    /// Fine-grained hits reuse the prompt's hash-aligned Mamba checkpoint.
    /// Split there only if the backend cannot save it within the final chunk.
    mamba_partial_cache_hit: bool,
    /// Opt-in: also stop at the junction, where an eagle sibling resumes. The
    /// manager decides whether it can check-point there (per-group eagle bit,
    /// no MTP re-prefill tail); splitting for a stop it would refuse costs a
    /// forward pass and displaces the block-boundary stop.
    mamba_shared_prefix_checkpoint: bool,
    /// `AsyncScheduler.decode_stagger`; `None` for `Scheduler`.
    decode_stagger: Option<u64>,

    // ---- Collaborators ----
    kv_cache_manager: KvCacheManager,
    encoder_cache_manager: AnyEncoderCacheManager,
    connector: Option<Box<dyn KvConnector>>,
    structured_output: Box<dyn StructuredOutput>,
    /// Set by the binding layer when `ec_transfer_config` is set.
    ec_connector: Option<Box<dyn EcConnector>>,
    /// Set by the binding layer when `aux_output_config.enabled`.
    aux_output_connector: Option<Box<dyn AuxOutputConnector>>,
    clock: Clock,
    /// Insertion order unless the binding layer sets Python's.
    id_set_order: Option<IdSetOrder>,

    // ---- State ----
    /// req_id -> Request
    requests: RequestStore,
    /// Priority queues for requests.
    waiting: RequestQueue<usize>,
    /// Waiting requests that hold KV blocks are always drained before
    /// self.waiting: a block-holder must never sit behind a request whose
    /// failed allocation would stop the scheduling scan.
    kv_holding_waiting: RequestQueue<usize>,
    /// With either async-load limit, keep requests that may load KV async
    /// separate so throttling their loads doesn't block local requests.
    async_kv_waiting: RequestQueue<usize>,
    /// Waiting requests that are deferred, i.e. were skipped by the scheduling
    /// scan or enqueued in a blocked status.
    deferred_waiting: FxHashSet<usize>,
    running: Vec<usize>,

    /// The request IDs that are finished in between the previous and the
    /// current steps. This is used to notify the workers about the finished
    /// requests so that they can free the cached states for those requests.
    /// This is flushed at the end of each scheduling step.
    finished_req_ids: IndexSet<String>,
    /// IDs of requests preempted since the last call to schedule().
    reset_preempted_req_ids: IndexSet<String>,
    /// include_finished_set controls whether a separate set of finished
    /// request ids should be included in the EngineCoreOutputs returned
    /// by update_from_outputs(). This is currently used in the multi-engine
    /// case to track request lifetimes efficiently.
    finished_req_ids_dict: Option<IndexMap<usize, IndexSet<String>>>,
    /// Counter for requests waiting for streaming input. Used to calculate
    /// number of unfinished requests
    num_waiting_for_streaming_input: usize,

    /// KV Connector: requests in process of async KV loading or recving
    finished_recving_kv_req_ids: FxHashSet<String>,
    failed_recving_kv_req_ids: FxHashSet<String>,

    /// Grammar compilation failures to finish as per-request errors in
    /// update_from_output.
    grammar_compile_error_reqs: IndexMap<String, RequestError>,
    encoder_cache_mismatch_reqs: IndexSet<String>,

    connector_prefix_cache_stats: Option<ConnectorPrefixCacheStats>,
    pending_encoder_input_ids: FxHashMap<String, PyIntSet>,
    /// Scheduler iteration counter. Drives the PP+async decode-throttle
    /// cadence (`next_decode_eligible_step`).
    current_step: u64,
    /// DP prefill balancing: Flag to track whether the last cadence-aligned
    /// prefill batch fully drained the waiting queue. Prefill throttling
    /// is disabled in this case.
    prefill_capacity_bound: bool,

    /// Counts of non-empty steps scheduled / processed. update_from_output
    /// is called once per scheduled step in FIFO order, so these stay in sync.
    sched_step_seq: u64,
    processed_step_seq: u64,
    /// FIFO of (fence_seq, blocks): blocks become safe to free once
    /// processed_step_seq >= fence_seq.
    deferred_frees: VecDeque<(u64, Vec<BlockId>)>,
    /// Request id -> (fence_seq, num_external_tokens, connector prefix cache
    /// queries, hits): async KV loads started once processed_step_seq >=
    /// fence_seq.
    deferred_loads: IndexMap<String, (u64, usize, usize, usize)>,

    pause_state: PauseState,

    /// Remaining-block reservations of in-flight prefills gate async loads.
    /// Parked async KV loads can number thousands, so theirs are cached until
    /// they are scheduled; prefill chunks are few and summed on demand.
    inflight_prefills: FxHashSet<usize>,
    parked_reservations: FxHashMap<usize, i64>,
    parked_reserved_blocks: i64,

    /// Requests with an async KV load, by stage (KV_FETCH_STAGES).
    /// None when not tracked (no KV connector, or no stats or load cap).
    kv_fetch_stages: Option<FxHashMap<usize, KvFetchStage>>,
    kv_fetch_counts: KvFetchCounts,
    aborted_kv_loads: FxHashSet<String>,

    /// Requests whose slots were freed, in their final state, for the
    /// binding layer to release (and bring up to date) its Python objects.
    removed_requests: Vec<Request>,
}

impl Scheduler {
    pub fn new(
        config: SchedulerConfig,
        kv_cache_config: KvCacheConfig,
        connector: Option<Box<dyn KvConnector>>,
        structured_output: Box<dyn StructuredOutput>,
    ) -> Result<Self, SchedulerError> {
        let unsupported = |message: &str| {
            Err(SchedulerError::Unsupported {
                message: message.to_owned(),
            })
        };
        if config.custom_encoder_cache_manager {
            return unsupported(
                "Custom encoder cache managers are not supported by the Rust scheduler",
            );
        }
        let invalid = |message: String| Err(SchedulerError::InvalidConfig { message });
        if config.kv_transfer.is_some() != connector.is_some() {
            return invalid(
                "a KV connector must be given exactly when kv_transfer is configured".to_owned(),
            );
        }
        if config.kv_transfer.is_some() && config.is_encoder_decoder {
            return invalid(
                "Encoder-decoder models are not currently supported with KV connectors".to_owned(),
            );
        }
        if config.num_gpu_blocks == 0 {
            return invalid("num_gpu_blocks must be positive".to_owned());
        }

        let mut recompute_kv_load_failures = true;
        let mut defer_block_free = false;
        let mut requires_kv_delivery = false;
        if let (Some(kv_transfer), Some(connector)) = (&config.kv_transfer, &connector) {
            recompute_kv_load_failures = kv_transfer.recompute_kv_load_failures;
            // With overlapping batches (async scheduling or PP), a step may
            // still be writing a freed request's KV blocks. A consumer KV
            // Connector can reallocate and fill those blocks via a load that
            // isn't ordered against that write, so defer freeing them.
            let multiple_inflight_batches = config.max_concurrent_batches > 1;
            if multiple_inflight_batches && kv_transfer.is_kv_consumer {
                defer_block_free = true;
            }
            requires_kv_delivery = connector.requires_kv_delivery();
        }
        // HiSparse keeps reading a released page until the pool reuses it,
        // so a load into a reused page must also wait for in-flight steps.
        let defer_async_loads =
            defer_block_free && kv_cache_config.hisparse_host_num_blocks.is_some();

        let prefix_replay_group_ids: Vec<usize> = kv_cache_config
            .kv_cache_groups
            .iter()
            .enumerate()
            .filter(|(_, g)| g.kv_cache_spec.prefix_replay_tokens() > 0)
            .map(|(i, _)| i)
            .collect();
        let replay_windows: IndexSet<usize> = prefix_replay_group_ids
            .iter()
            .map(|&g| kv_cache_config.kv_cache_groups[g].kv_cache_spec.prefix_replay_tokens())
            .collect();
        if replay_windows.len() > 1 {
            let mut windows: Vec<usize> = replay_windows.into_iter().collect();
            windows.sort_unstable();
            return invalid(format!("Prefix replay windows should agree: {windows:?}"));
        }
        let prefix_replay_tokens = replay_windows.first().copied().unwrap_or(0);

        let mut use_eagle_block_drop = config.use_eagle_block_drop;
        if config.num_hash_lookahead_tokens > 0 {
            // Lookahead hashes prove the boundary token, so EAGLE needs no drop.
            use_eagle_block_drop = false;
        }

        let hash_block_size = config.hash_block_size.unwrap_or(config.block_size);
        let mut kv_config =
            KvCacheManagerConfig::new(config.max_model_len, config.block_size, hash_block_size);
        kv_config.max_in_flight_tokens = config.max_in_flight_tokens;
        kv_config.enable_caching = config.enable_prefix_caching;
        kv_config.use_eagle = use_eagle_block_drop;
        kv_config.num_hash_lookahead_tokens = config.num_hash_lookahead_tokens;
        kv_config.num_prefill_lookahead = config.num_prefill_lookahead_tokens;
        kv_config.log_stats = config.log_stats;
        kv_config.enable_kv_cache_events = config.enable_kv_cache_events;
        kv_config.kv_events_use_int_block_hashes = config.kv_events_use_int_block_hashes;
        kv_config.dcp_world_size = config.decode_context_parallel_size;
        kv_config.pcp_world_size = 1;
        kv_config.metrics_collector =
            config.kv_cache_metrics_sample.map(KvCacheMetricsCollector::new);
        kv_config.watermark = config.watermark;
        kv_config.enable_mamba_shared_prefix_checkpoint =
            config.enable_mamba_shared_prefix_checkpoint;
        let kv_cache_manager = KvCacheManager::new(kv_cache_config.clone(), kv_config)?;

        // Mamba managers already account for their speculative state blocks.
        let num_spec_decode_tokens = if config.num_spec_tokens > 0 {
            1 + config.num_spec_tokens + config.num_lookahead_tokens
        } else {
            0
        };
        let spec_decode_step_blocks = kv_cache_manager
            .coordinator
            .single_type_managers
            .iter()
            .filter(|m| !m.kv_cache_spec.is_mamba())
            .map(|m| cdiv(num_spec_decode_tokens, m.block_size) as i64)
            .sum();

        let has_mamba_layers = kv_cache_config.has_mamba_layers();
        let need_mamba_block_aligned_split =
            has_mamba_layers && config.mamba_cache_mode == MambaCacheMode::Align;
        let mamba_prefill_checkpoint_alignment = kv_cache_config
            .kv_cache_groups
            .iter()
            .find_map(|g| match g.kv_cache_spec.spec_type {
                crate::kv_cache::KvCacheSpecType::Mamba {
                    prefill_checkpoint_alignment,
                    ..
                } => Some(prefill_checkpoint_alignment),
                _ => None,
            })
            .flatten();
        let mamba_has_prefill_checkpoint_blocks = has_mamba_layers
            && kv_cache_config.kv_cache_groups.iter().all(|g| match g.kv_cache_spec.spec_type {
                crate::kv_cache::KvCacheSpecType::Mamba {
                    num_prefill_checkpoint_blocks,
                    ..
                } => num_prefill_checkpoint_blocks > 0,
                _ => true,
            });
        let mamba_partial_cache_hit = need_mamba_block_aligned_split
            && hash_block_size < config.block_size
            && kv_cache_manager.coordinator.enable_partial_hash_hits;
        let mamba_shared_prefix_checkpoint =
            mamba_partial_cache_hit && kv_cache_manager.mamba_shared_prefix_checkpoint;

        let encoder_cache_manager = if config.is_encoder_decoder {
            AnyEncoderCacheManager::EncoderDecoder(EncoderDecoderCacheManager::new(
                config.encoder_cache_size,
            ))
        } else {
            AnyEncoderCacheManager::Mm(EncoderCacheManager::new(config.encoder_cache_size))
        };

        let kv_fetch_stages = (connector.is_some()
            && (config.log_stats || config.max_num_inflight_kv_loads.is_some_and(|n| n > 0)))
        .then(FxHashMap::default);

        let decode_stagger = match config.scheduler_class {
            SchedulerClass::Sync => None,
            SchedulerClass::Async { decode_stagger } => Some(decode_stagger),
        };

        Ok(Self {
            log_stats: config.log_stats,
            enable_iteration_details: config.enable_iteration_details,
            spec_decode_metrics_level: config.spec_decode_metrics_level,
            is_encoder_decoder: config.is_encoder_decoder,
            is_mm_encoder_only: config.is_mm_encoder_only,
            uses_mrope: config.uses_mrope,
            max_num_running_reqs: config.max_num_seqs,
            max_num_active_reqs: config.max_num_active_seqs.unwrap_or(config.max_num_seqs),
            max_num_scheduled_tokens: config
                .max_num_scheduled_tokens
                .unwrap_or(config.max_num_batched_tokens),
            max_num_batched_tokens: config.max_num_batched_tokens,
            long_prefill_token_threshold: config.long_prefill_token_threshold,
            adaptive_long_prefill_threshold: config.long_prefill_token_threshold_adaptive,
            enable_chunked_prefill: config.enable_chunked_prefill,
            disable_chunked_mm_input: config.disable_chunked_mm_input,
            async_scheduling: config.async_scheduling,
            max_num_inflight_kv_loads: config.max_num_inflight_kv_loads,
            max_async_kv_loads_per_step: config.max_async_kv_loads_per_step.filter(|&n| n > 0),
            max_model_len: config.max_model_len,
            num_sampled_tokens_per_step: config.num_sampled_tokens_per_step,
            return_sampling_mask: config.return_sampling_mask,
            report_watermarking: config.report_watermarking,
            enable_prefix_caching: config.enable_prefix_caching,
            cache_block_size: config.cache_block_size,
            block_size: config.block_size,
            hash_block_size,
            policy: config.policy,
            max_loras: config.max_loras,
            kv_transfer: config.kv_transfer,
            recompute_kv_load_failures,
            defer_block_free,
            defer_async_loads,
            requires_kv_delivery,
            max_num_encoder_input_tokens: config.max_num_encoder_input_tokens,
            num_spec_tokens: config.num_spec_tokens,
            num_lookahead_tokens: config.num_lookahead_tokens,
            draft_slots: config.draft_slots,
            prefix_replay_group_ids,
            prefix_replay_tokens,
            num_prefill_lookahead: config.num_prefill_lookahead_tokens,
            can_skip_drafts: config.num_lookahead_tokens > 0,
            dynamic_sd_lookup: config.dynamic_sd_lookup,
            use_eagle_block_drop,
            num_hash_lookahead_tokens: config.num_hash_lookahead_tokens,
            spec_decode_step_blocks,
            use_pp: config.pipeline_parallel_size > 1,
            scheduler_reserve_full_isl: config.scheduler_reserve_full_isl,
            cache_aware_window: if config.policy == SchedulingPolicy::Fcfs
                && config.enable_prefix_caching
            {
                config.cache_aware_admission_window
            } else {
                0
            },
            cache_aware_threshold: config.cache_aware_admission_threshold,
            has_mamba_layers,
            needs_kv_cache_zeroing: kv_cache_config.needs_kv_cache_zeroing,
            need_mamba_block_aligned_split,
            mamba_prefill_checkpoint_alignment,
            mamba_has_prefill_checkpoint_blocks,
            mamba_partial_cache_hit,
            mamba_shared_prefix_checkpoint,
            decode_stagger,
            kv_cache_manager,
            encoder_cache_manager,
            connector_prefix_cache_stats: (connector.is_some() && config.log_stats)
                .then(ConnectorPrefixCacheStats::default),
            connector,
            structured_output,
            ec_connector: None,
            aux_output_connector: None,
            clock: Box::new(monotonic),
            id_set_order: None,
            requests: RequestStore::default(),
            waiting: RequestQueue::new(config.policy),
            kv_holding_waiting: RequestQueue::new(config.policy),
            async_kv_waiting: RequestQueue::new(config.policy),
            deferred_waiting: FxHashSet::default(),
            running: Vec::new(),
            finished_req_ids: IndexSet::new(),
            reset_preempted_req_ids: IndexSet::new(),
            finished_req_ids_dict: config.include_finished_set.then(IndexMap::new),
            num_waiting_for_streaming_input: 0,
            finished_recving_kv_req_ids: FxHashSet::default(),
            failed_recving_kv_req_ids: FxHashSet::default(),
            grammar_compile_error_reqs: IndexMap::new(),
            encoder_cache_mismatch_reqs: IndexSet::new(),
            pending_encoder_input_ids: FxHashMap::default(),
            current_step: 0,
            prefill_capacity_bound: false,
            sched_step_seq: 0,
            processed_step_seq: 0,
            deferred_frees: VecDeque::new(),
            deferred_loads: IndexMap::new(),
            pause_state: PauseState::Unpaused,
            inflight_prefills: FxHashSet::default(),
            parked_reservations: FxHashMap::default(),
            parked_reserved_blocks: 0,
            kv_fetch_stages,
            kv_fetch_counts: KvFetchCounts::default(),
            aborted_kv_loads: FxHashSet::default(),
            removed_requests: Vec::new(),
        })
    }

    /// Replace the clock behind `time.monotonic()` (event timestamps), e.g.
    /// with a deterministic one for differential tests.
    pub fn set_clock(&mut self, clock: Clock) {
        self.clock = clock;
    }

    /// Replace the KV connector (`scheduler.connector = ...`; tests swap in
    /// mocks). The settings read from the connector at construction stay.
    pub fn set_connector(&mut self, connector: Option<Box<dyn KvConnector>>) {
        self.connector = connector;
    }

    /// Set the EC connector (`ec_transfer_config`); the binding layer
    /// constructs it after the scheduler, as `Scheduler.__init__` does.
    pub fn set_ec_connector(&mut self, ec_connector: Option<Box<dyn EcConnector>>) {
        self.ec_connector = ec_connector;
    }

    /// Set the aux output connector (`aux_output_config.enabled`), which
    /// reads the scheduler's KV cache manager.
    pub fn set_aux_output_connector(&mut self, connector: Option<Box<dyn AuxOutputConnector>>) {
        self.aux_output_connector = connector;
    }

    /// The KV connector, if any.
    pub fn connector(&self) -> Option<&dyn KvConnector> {
        self.connector.as_deref()
    }

    /// Finish error requests in Python's order (see [`IdSetOrder`]).
    pub fn set_id_set_order(&mut self, order: IdSetOrder) {
        self.id_set_order = Some(order);
    }

    /// The order in which `update_from_output` finishes `ids`; order of
    /// discovery without an [`IdSetOrder`].
    pub(super) fn error_request_order(&self, ids: &ErrorRequestIds) -> IndexSet<String> {
        match &self.id_set_order {
            Some(order) => order(ids),
            None => ids.all(),
        }
    }

    pub fn kv_cache_manager(&self) -> &KvCacheManager {
        &self.kv_cache_manager
    }

    /// The KV cache manager, for the binding layer to hand KV connectors
    /// (`bind_kv_cache_manager`) between scheduler calls.
    pub fn kv_cache_manager_mut(&mut self) -> &mut KvCacheManager {
        &mut self.kv_cache_manager
    }

    /// `self.requests.get(request_id)`.
    pub fn get_request(&self, request_id: &str) -> Option<&Request> {
        self.requests.handle(request_id).map(|h| &self.requests[h])
    }

    /// Ids of the running requests, in order.
    pub fn running_request_ids(&self) -> impl Iterator<Item = &str> {
        self.running.iter().map(|&h| self.requests[h].request_id.as_str())
    }

    /// Drain the requests the scheduler no longer holds, so the binding
    /// layer can release their Python objects.
    pub fn take_removed_requests(&mut self) -> Vec<Request> {
        std::mem::take(&mut self.removed_requests)
    }

    /// Free the slots of requests removed during the current call.
    fn flush_removed(&mut self) {
        let removed: Vec<Request> = self.requests.flush().collect();
        self.removed_requests.extend(removed);
    }

    /// Returns (num_running_reqs, num_waiting_reqs).
    pub fn get_request_counts(&self) -> (usize, usize) {
        (
            self.running.len(),
            self.waiting.len() + self.kv_holding_waiting.len() + self.async_kv_waiting.len(),
        )
    }

    /// Returns the fraction of the KV cache currently in use (0.0-1.0).
    pub fn get_kv_cache_usage(&self) -> f64 {
        self.kv_cache_manager.usage()
    }

    pub fn has_request(&self, request_id: &str) -> bool {
        self.requests.handle(request_id).is_some()
    }

    pub fn add_request(&mut self, mut request: Request) {
        if let Some(existing) = self.requests.handle(&request.request_id) {
            let update = StreamingUpdate::from_request(&request);
            if self.requests[existing].status != RequestStatus::WaitingForStreamingReq {
                // Queue next input chunk (or finished sentinel).
                self.requests[existing]
                    .streaming_queue
                    .as_mut()
                    .expect("duplicate request id")
                    .push_back(update);
            } else if let Some(update) = update {
                // Commence next input chunk.
                self.update_request_as_session(existing, update);
            } else {
                // Streaming-input session finished.
                self.finish_request_handles(
                    Some([request.request_id].into_iter().collect()),
                    RequestStatus::FinishedAborted,
                );
                self.flush_removed();
            }
            return;
        }
        if request.resumable {
            request.streaming_queue = Some(VecDeque::new());
        }
        if self.spec_decode_metrics_level != SpecDecodeMetricsLevel::None {
            request.spec_decode_metrics = Some(RequestSpecDecodeMetrics::new(self.num_spec_tokens));
        }
        if let Some(connector) = self.connector.as_mut() {
            // Before enqueueing: may_load_async can use what this records.
            connector.on_new_request(&request, &mut self.kv_cache_manager);
        }
        let do_remote_prefill = request.do_remote_prefill;
        let h = self.requests.insert(request);
        self.enqueue_waiting_request(h);
        if self.connector.is_some() && do_remote_prefill {
            // Count P/D requests on arrival: connector matching is
            // skipped while all run slots are taken.
            self.set_kv_fetch_stage(h, Some(KvFetchStage::WaitingToStart));
        }
        if self.log_stats {
            let now = (self.clock)();
            self.requests[h].record_event(EngineCoreEventType::Queued, now);
        }
    }

    /// Handles the finish signal from outside the scheduler.
    ///
    /// For example, the API server can abort a request when the client
    /// disconnects.
    ///
    /// If request_ids is None, all requests will be finished.
    ///
    /// Returns:
    ///     The ids of the requests that were aborted. Will not include any
    ///     that were already finished.
    pub fn finish_requests<S: AsRef<str>>(
        &mut self,
        request_ids: Option<&[S]>,
        finished_status: RequestStatus,
    ) -> Vec<String> {
        let request_ids =
            request_ids.map(|ids| ids.iter().map(|id| id.as_ref().to_owned()).collect());
        let finished = self.finish_request_handles(request_ids, finished_status);
        let ids = finished.iter().map(|&h| self.requests[h].request_id.clone()).collect();
        self.flush_removed();
        ids
    }

    /// `finish_requests` returning handles, which stay valid until
    /// [`Self::flush_removed`].
    fn finish_request_handles(
        &mut self,
        request_ids: Option<IndexSet<String>>,
        finished_status: RequestStatus,
    ) -> Vec<usize> {
        assert!(finished_status.is_finished());
        let handles: Vec<Option<usize>> = match request_ids {
            Some(ids) => ids.iter().map(|id| self.requests.handle(id)).collect(),
            None => self.requests.handles().into_iter().map(Some).collect(),
        };

        let mut running_requests_to_remove = FxHashSet::default();
        let mut waiting_requests_to_remove = Vec::new();
        let mut valid_requests = Vec::new();

        // First pass: collect requests to remove from queues
        for h in handles.into_iter().flatten() {
            let request = &self.requests[h];
            if request.is_finished() {
                // Invalid request ID.
                continue;
            }

            valid_requests.push(h);
            if request.status == RequestStatus::Running {
                running_requests_to_remove.insert(h);
            } else {
                if request.status == RequestStatus::WaitingForStreamingReq {
                    self.num_waiting_for_streaming_input -= 1;
                }
                waiting_requests_to_remove.push(h);
            }
        }

        // Remove all requests from queues at once for better efficiency
        if !running_requests_to_remove.is_empty() {
            remove_all(&mut self.running, &running_requests_to_remove);
        }
        if !waiting_requests_to_remove.is_empty() {
            let reqs = &waiting_requests_to_remove;
            self.waiting.remove_requests(reqs.iter().copied(), &self.requests);
            self.kv_holding_waiting.remove_requests(reqs.iter().copied(), &self.requests);
            self.async_kv_waiting.remove_requests(reqs.iter().copied(), &self.requests);
            for h in reqs {
                self.deferred_waiting.remove(h);
            }
        }

        // Second pass: set status and free requests
        for &h in &valid_requests {
            let mut delay_free_blocks = false;
            let request_id = self.requests[h].request_id.clone();
            if self.requests[h].status == RequestStatus::WaitingForRemoteKvs {
                // A deferred load never started, so it will not report done.
                delay_free_blocks = self.deferred_loads.shift_remove(&request_id).is_none()
                    && !self.finished_recving_kv_req_ids.contains(&request_id);
                self.finished_recving_kv_req_ids.remove(&request_id);
                self.failed_recving_kv_req_ids.remove(&request_id);
            }

            self.requests[h].status = finished_status;
            self.free_request(h, delay_free_blocks);
            if delay_free_blocks {
                self.aborted_kv_loads.insert(request_id);
            }
        }

        valid_requests
    }

    /// Returns the kv transfer params and the EC connector's
    /// `ec_transfer_params`.
    fn free_request(
        &mut self,
        h: usize,
        mut delay_free_blocks: bool,
    ) -> (Option<KvTransferParams>, Option<Opaque>) {
        assert!(self.requests[h].is_finished());

        if let Some(aux) = self.aux_output_connector.as_mut() {
            aux.request_finished(&self.requests[h], &mut self.kv_cache_manager);
        }
        self.inflight_prefills.remove(&h);
        self.unreserve_parked(h);
        self.set_kv_fetch_stage(h, None);
        let (mut connector_delay_free_blocks, kv_xfer_params) = self.connector_finished(h);

        // EC Connector: mirror the KV hook. The contract requires firing
        // before the encoder cache is freed so the connector can inspect
        // per-request state (e.g. which mm_hashes it recorded during
        // save_caches()) and emit ec_transfer_params for the response body.
        let mut ec_xfer_params = None;
        if let Some(ec_connector) = self.ec_connector.as_mut() {
            let ec_delay_free;
            (ec_delay_free, ec_xfer_params) = ec_connector.request_finished(&self.requests[h]);
            connector_delay_free_blocks |= ec_delay_free;
        }

        let request = &self.requests[h];
        self.encoder_cache_manager.free(request);
        self.pending_encoder_input_ids.remove(&request.request_id);
        let request_id = request.request_id.clone();
        if let Some(dict) = &mut self.finished_req_ids_dict {
            dict.entry(request.client_index).or_default().insert(request_id.clone());
        }
        self.finished_req_ids.insert(request_id);

        delay_free_blocks |= connector_delay_free_blocks;
        if !delay_free_blocks {
            self.free_blocks(h);
        }

        (kv_xfer_params, ec_xfer_params)
    }

    fn free_blocks(&mut self, h: usize) {
        assert!(self.requests[h].is_finished());
        self.free_request_blocks(h);
        self.requests.remove(h);
    }

    pub fn pause_state(&self) -> PauseState {
        self.pause_state
    }

    pub fn set_pause_state(&mut self, pause_state: PauseState) {
        self.pause_state = pause_state;
    }

    fn request_blocks_can_be_freed(&self, h: usize) -> bool {
        // We must defer freeing blocks if an async kv connector may
        // write to them immediately (not ordered with GPU stream).
        !self.defer_block_free || self.requests[h].last_sched_seq <= self.processed_step_seq
    }

    /// Free the request's KV blocks, deferring the return to the block
    /// pool when an in-flight GPU step may still write them.
    fn free_request_blocks(&mut self, h: usize) {
        let request = &self.requests[h];
        let (request_id, finished) = (&request.request_id, request.is_finished());
        if self.request_blocks_can_be_freed(h) {
            self.kv_cache_manager.free(request_id, finished);
            return;
        }
        let blocks = self.kv_cache_manager.pop_blocks_for_free(request_id, finished);
        if !blocks.is_empty() {
            self.deferred_frees.push_back((self.sched_step_seq, blocks));
        }
    }

    /// Release CoW copy retentions, deferring their return to the block
    /// pool while the step that runs the copy may still be in flight.
    fn free_cow_retained_blocks(&mut self, mut blocks: Vec<BlockId>, fence_seq: u64) {
        if !self.defer_block_free || fence_seq <= self.processed_step_seq {
            self.kv_cache_manager.free_blocks(blocks, false);
            return;
        }
        blocks.reverse();
        self.deferred_frees.push_back((fence_seq, blocks));
    }

    /// Return deferred blocks whose fence step has completed.
    ///
    /// Fences are appended in near-monotonic order (a CoW retention fence
    /// can lead request-free fences by one step), so stop at the first
    /// pending one; any satisfied entry behind it is merely freed later.
    fn drain_deferred_frees(&mut self) {
        while let Some(&(fence, _)) = self.deferred_frees.front() {
            if fence > self.processed_step_seq {
                break;
            }
            let (_, blocks) = self.deferred_frees.pop_front().unwrap();
            // Free in reverse order so that the tail blocks are evicted first.
            self.kv_cache_manager.free_blocks(blocks.into_iter().rev(), false);
        }
    }

    pub fn get_num_unfinished_requests(&self) -> usize {
        if self.pause_state == PauseState::PausedAll {
            return 0;
        }
        let (num_running, mut num_waiting) = self.get_request_counts();
        if self.pause_state == PauseState::PausedNew {
            // Requests holding KV blocks still drain; queued ones stay queued.
            num_waiting = self.kv_holding_waiting.len();
        }
        // Python can go negative here, which reads as no unfinished requests.
        (num_waiting + num_running).saturating_sub(self.num_waiting_for_streaming_input)
    }

    pub fn has_unfinished_requests(&self) -> bool {
        self.get_num_unfinished_requests() > 0
    }

    pub fn has_finished_requests(&self) -> bool {
        if !self.finished_req_ids.is_empty() {
            return true;
        }
        if self.connector.is_none() {
            return false;
        }
        // Finished requests waiting on delayed connector cleanup remain in
        // self.requests after they have been removed from scheduling queues.
        let (num_running, num_waiting) = self.get_request_counts();
        self.requests.len() > num_running + num_waiting
    }

    pub fn has_requests(&mut self) -> bool {
        // Override the interface default to also keep the engine alive while a
        // connector still has pending push work (e.g. push-mode WRITE transfers
        // in flight after all "live" requests have finished). Without this hook
        // the engine would quiesce before the connector can drain completions.
        // TODO: replace with a more general mechanism for connectors to keep
        // the scheduler alive.
        self.has_unfinished_requests()
            || self.has_finished_requests()
            || self
                .connector
                .as_mut()
                .is_some_and(|c| c.has_pending_push_work(&mut self.kv_cache_manager))
            || self.ec_connector.as_mut().is_some_and(|c| c.has_pending_push_work())
    }

    /// Reset the KV prefix cache.
    ///
    /// If reset_running_requests is True, all the running requests will be
    /// preempted and moved to the waiting queue.
    /// Otherwise, this method will only reset the KV prefix cache when there
    /// is no running requests taking KV cache.
    ///
    /// Panics where Python raises `RuntimeError`.
    pub fn reset_prefix_cache(
        &mut self,
        reset_running_requests: bool,
        reset_connector: bool,
    ) -> bool {
        if reset_running_requests {
            // For logging.
            let timestamp = (self.clock)();
            // Invalidate all the current running requests KV's by pushing them to
            // the waiting queue. In this case, we can reduce the ref count of all
            // the kv blocks to 0 and thus we can make sure the reset is successful.
            // Preempt in reverse order so the requests will be added back to the
            // running queue in FIFO order.
            while let Some(h) = self.running.pop() {
                self.preempt_request(h, timestamp, true);
            }
        }

        let mut reset_successful = self.kv_cache_manager.reset_prefix_cache();
        assert!(
            !reset_running_requests || reset_successful,
            "Failed to reset KV cache even when all the running requests are \
             preempted and moved to the waiting queue. This is likely due to \
             the presence of running requests waiting for remote KV transfer, \
             which is not supported yet."
        );

        if reset_connector {
            reset_successful = self.reset_connector_cache() && reset_successful;
        }

        if reset_successful && let Some(aux) = self.aux_output_connector.as_mut() {
            aux.reset();
        }

        reset_successful
    }

    /// Reset the encoder cache to invalidate all cached encoder outputs.
    ///
    /// This should be called when model weights are updated to ensure
    /// stale vision embeddings are not reused.
    pub fn reset_encoder_cache(&mut self) {
        self.encoder_cache_manager.reset();
    }

    /// The structured output requests among the scheduled ones, in schedule
    /// order: the rows of the grammar bitmask (`get_grammar_bitmask`, whose
    /// bitmask the binding layer computes). `None` when there are none.
    pub fn get_structured_output_request_ids(
        &self,
        scheduler_output: &SchedulerOutput,
    ) -> Option<Vec<String>> {
        // Collect list of scheduled request ids that use structured output.
        // The corresponding rows of the bitmask will be in this order.
        if !scheduler_output.has_structured_output_requests {
            return None;
        }

        let ids: Vec<String> = scheduler_output
            .num_scheduled_tokens
            .keys()
            .filter(|req_id| {
                self.get_request(req_id)
                    .is_some_and(|r| r.use_structured_output() && !r.is_prefill_chunk)
            })
            .cloned()
            .collect();
        (!ids.is_empty()).then_some(ids)
    }

    pub(super) fn is_blocked_waiting_status(status: RequestStatus) -> bool {
        matches!(
            status,
            RequestStatus::WaitingForStructuredOutputGrammar
                | RequestStatus::WaitingForRemoteKvs
                | RequestStatus::WaitingForStreamingReq
        )
    }

    /// Whether a request currently has kv blocks allocated.
    /// While always the case when status is WAITING_FOR_REMOTE_KVS
    /// or WAITING_FOR_STREAMING_REQ, there are such cases where it's
    /// WAITING or PREEMPTED.
    fn holds_kv_blocks(request: &Request) -> bool {
        request.num_computed_tokens > 0
            // num_computed_tokens is reset to 0 if kv transfer fails for requests in
            // WAITING_FOR_REMOTE_KVS state, before their allocated blocks are freed.
            || request.status == RequestStatus::WaitingForRemoteKvs
    }

    fn enqueue_waiting_request(&mut self, h: usize) {
        let request = &self.requests[h];
        if Self::is_blocked_waiting_status(request.status) {
            self.deferred_waiting.insert(h);
        }
        if Self::holds_kv_blocks(request) {
            self.kv_holding_waiting.add_request(h, &self.requests);
        } else if (self.max_num_inflight_kv_loads.is_some_and(|n| n > 0)
            || self.max_async_kv_loads_per_step.is_some())
            && let Some(connector) = self.connector.as_mut()
            && connector.may_load_async(request)
        {
            self.async_kv_waiting.add_request(h, &self.requests);
        } else {
            self.waiting.add_request(h, &self.requests);
        }
    }

    fn queue(&self, queue: WaitingQueue) -> &RequestQueue<usize> {
        match queue {
            WaitingQueue::Waiting => &self.waiting,
            WaitingQueue::KvHolding => &self.kv_holding_waiting,
            WaitingQueue::AsyncKv => &self.async_kv_waiting,
        }
    }

    /// The queue and the request store, borrowed together.
    fn queue_mut(&mut self, queue: WaitingQueue) -> (&mut RequestQueue<usize>, &RequestStore) {
        let q = match queue {
            WaitingQueue::Waiting => &mut self.waiting,
            WaitingQueue::KvHolding => &mut self.kv_holding_waiting,
            WaitingQueue::AsyncKv => &mut self.async_kv_waiting,
        };
        (q, &self.requests)
    }
}

impl std::fmt::Debug for Scheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (num_running, num_waiting) = self.get_request_counts();
        f.debug_struct("Scheduler")
            .field("num_running", &num_running)
            .field("num_waiting", &num_waiting)
            .field("current_step", &self.current_step)
            .finish_non_exhaustive()
    }
}
