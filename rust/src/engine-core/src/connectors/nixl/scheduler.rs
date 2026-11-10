// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `NixlBaseConnectorScheduler` with its pull (`pull_scheduler.py`) and push
//! (`push_scheduler.py`) subclasses.

use std::sync::Arc;

use indexmap::{IndexMap, IndexSet};
use rustc_hash::{FxHashMap, FxHashSet};
use thiserror::Error;

use super::metadata::{
    HeartbeatInfo, NixlConnectorMetadata, PushRegistration, RemoteMeta, ReqMeta,
};
use super::params::{
    BlockIds, NixlFinishedParams, NixlKvTransferParams, RemoteBlockIds, Seq, SeqKind, TransferMode,
};
use crate::kv_cache::BlockId;
use crate::sched::request::RequestStatus;

/// A clock in seconds on the `time.perf_counter()` timeline.
///
/// Workers rebase the send deadlines through `scheduler_clock`, and the
/// handshake listener (which stays in Python) echoes `time.perf_counter()`,
/// so the deadlines must be on that clock: [`perf_counter`] by default.
pub type PerfClock = Box<dyn Fn() -> f64 + Send + Sync>;

/// `time.perf_counter()`: `CLOCK_MONOTONIC` on Linux.
pub fn perf_counter() -> f64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid, writable timespec.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
}

/// The scheduler-side configuration of a NIXL connector, resolved by the
/// caller (the binding layer reads it from the constructed Python
/// connector scheduler, so the Python config parsing stays the reference).
#[derive(Clone, Debug, PartialEq)]
pub struct NixlSchedulerConfig {
    pub mode: TransferMode,
    pub engine_id: Arc<str>,
    pub side_channel_host: Arc<str>,
    /// `VLLM_NIXL_SIDE_CHANNEL_PORT + data_parallel_index`.
    pub side_channel_port: i64,
    /// `max(tensor_parallel_size, decode_context_parallel_size)`.
    pub transfer_tp_size: i64,
    pub tensor_parallel_size: i64,
    pub decode_context_parallel_size: i64,
    pub pipeline_parallel_size: i64,
    /// `kv_lease_duration`, the P-side block lease in seconds.
    pub kv_lease_duration: f64,
    /// `kv_lease_duration // 6`.
    pub heartbeat_interval: f64,
    /// `kv_buffer_device == "cpu"` on a non-CPU platform.
    pub use_host_buffer: bool,
    /// HMA is enabled and some transfer group is not full attention.
    pub is_hma_required: bool,
    pub num_kv_cache_groups: usize,
    /// `KVCacheConfig.transfer_group_ids`.
    pub transfer_group_ids: Vec<usize>,
    /// Per transfer group: blocks kept by the sliding-window clip (0: none).
    pub blocks_per_sw: Vec<usize>,
    /// Per transfer group: `MambaSpec.num_speculative_blocks`, `None` for
    /// non-SSM groups.
    pub ssm_spec_blocks: Vec<Option<usize>>,
    /// `KVCacheConfig.has_mamba_layers`.
    pub has_mamba: bool,
    /// Some group has `prefix_replay_tokens > 0` (DSV41 bounded replay).
    pub bounded_replay: bool,
    pub num_prefill_lookahead_tokens: usize,
    pub kv_recompute_threshold: i64,
    pub bidirectional_kv_xfer: bool,
    pub decoder_kv_blocks_ttl: f64,
    /// `kv_role` is `kv_producer` or `kv_both`.
    pub is_kv_producer: bool,
    /// Push only.
    pub push_registration_timeout: f64,
}

/// A request's params the connector can't use.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum NixlParamsError {
    #[error("invalid kv_transfer_params: {0}")]
    Invalid(String),
}

/// What the connector reads of a request.
#[derive(Clone, Copy, Debug)]
pub struct NixlRequestView<'a> {
    pub request_id: &'a str,
    /// `len(request.prompt_token_ids or [])`.
    pub num_prompt_token_ids: usize,
    pub num_prompt_tokens: usize,
    pub num_computed_tokens: usize,
    pub status: RequestStatus,
}

/// The blocks `update_state_after_alloc` receives (`KVCacheBlocks`).
pub trait AllocatedBlocks {
    /// `get_unhashed_block_ids_all_groups()`.
    fn unhashed_block_ids_all_groups(&self) -> Vec<Vec<BlockId>>;
    /// Per group, the blocks with a hash that are not the null block.
    fn num_computed_blocks(&self) -> Vec<usize>;
}

/// A scheduled request as `_build_save_meta` reads it (`yield_req_data`
/// plus the request state); only read with a host buffer.
pub struct ScheduledReq<'a> {
    pub req_id: &'a str,
    pub new_block_ids: Option<&'a [Vec<BlockId>]>,
    pub num_scheduled_tokens: usize,
    pub num_computed_tokens: usize,
    pub num_prompt_tokens: usize,
}

/// The `kv_transfer_params` fields `ReqMeta` copies (`_add_new_req`).
#[derive(Clone, Copy, Debug)]
struct ReqMetaParams {
    tp_size: i64,
    dcp_size: i64,
    remote_block_size: Option<i64>,
    pp_size: i64,
}

impl ReqMetaParams {
    fn of(params: &NixlKvTransferParams) -> Self {
        Self {
            tp_size: params.tp_size.unwrap_or(1),
            dcp_size: params.dcp_size.unwrap_or(1),
            remote_block_size: params.remote_block_size,
            pp_size: params.pp_size.unwrap_or(1),
        }
    }

    fn req_meta(
        self,
        local_block_ids: Arc<BlockIds>,
        local_num_computed_blocks: Vec<usize>,
        awaiting_kvs: bool,
    ) -> ReqMeta {
        ReqMeta {
            local_block_ids,
            tp_size: self.tp_size,
            dcp_size: self.dcp_size,
            local_num_computed_blocks,
            remote: None,
            remote_block_size: self.remote_block_size,
            pp_size: self.pp_size,
            awaiting_kvs,
        }
    }
}

/// `NixlConnectorMetadata.add_new_req_to_recv`, or `None` when a remote
/// field is missing (Python raises `KeyError` while building the metadata).
fn recv_req_meta(
    params: &NixlKvTransferParams,
    local_block_ids: Arc<BlockIds>,
    local_num_computed_blocks: Vec<usize>,
    awaiting_kvs: bool,
) -> Option<ReqMeta> {
    let mut meta = ReqMetaParams::of(params).req_meta(
        local_block_ids,
        local_num_computed_blocks,
        awaiting_kvs,
    );
    meta.remote = Some(RemoteMeta {
        block_ids: params.remote_block_ids.clone()?,
        engine_id: params.remote_engine_id.clone()?,
        request_id: params.remote_request_id.clone()?,
        host: params.remote_host.clone()?,
        port: params.remote_port?,
        blocks_expiry_time: params.remote_blocks_expiry_time,
        num_tokens: params.remote_num_tokens,
        registration_generation: params.remote_registration_generation.unwrap_or(0),
    });
    Some(meta)
}

/// `clip_ssm_state_blocks`: drop speculative scratch slots and keep only the
/// running-state slot.
fn clip_ssm_state_blocks(blocks: &[BlockId], num_spec_blocks: usize) -> &[BlockId] {
    if blocks.is_empty() {
        return blocks;
    }
    let num_scratch = num_spec_blocks.min(blocks.len() - 1);
    let blocks = &blocks[..blocks.len() - num_scratch];
    &blocks[blocks.len() - 1..]
}

fn params_of<'a>(
    params: &'a FxHashMap<String, Option<NixlKvTransferParams>>,
    request_id: &str,
) -> Option<&'a NixlKvTransferParams> {
    params.get(request_id).and_then(Option::as_ref)
}

fn params_mut<'a>(
    params: &'a mut FxHashMap<String, Option<NixlKvTransferParams>>,
    request_id: &str,
) -> Option<&'a mut NixlKvTransferParams> {
    params.get_mut(request_id).and_then(Option::as_mut)
}

/// Push-mode state (`NixlPushConnectorScheduler`).
#[derive(Default)]
struct PushState {
    /// D side: registrations for the D workers to send on the next step.
    pending_registrations: IndexMap<String, PushRegistration>,
    /// D side: watchdog deadline per registered request, in arming order.
    registration_deadlines: IndexMap<String, f64>,
    /// P side: finished requests whose blocks await the WRITE.
    finished_request_blocks: FxHashSet<String>,
    /// P side: newly finished blocks for the P workers.
    newly_finished_blocks: IndexMap<String, BlockIds>,
}

/// The scheduler half of the NIXL connector (pull or push).
pub struct NixlConnectorScheduler {
    config: NixlSchedulerConfig,
    clock: PerfClock,
    /// Bumped whenever the handshake listener (Python) publishes a changed
    /// registration.
    registration_generation: u64,

    /// Each live request's parsed `kv_transfer_params` (`None`: no or empty
    /// params), registered by [`Self::add_request`].
    params: FxHashMap<String, Option<NixlKvTransferParams>>,

    reqs_need_recv: IndexMap<String, ReqMeta>,
    reqs_need_save: IndexMap<String, ReqMetaParams>,
    reqs_need_send: IndexMap<String, f64>,
    reqs_in_batch: IndexSet<String>,
    reqs_not_processed: IndexSet<String>,

    heartbeat_by_engine: IndexMap<Arc<str>, HeartbeatInfo>,
    /// Local request id -> (engine id, remote request id).
    heartbeat_req_engine: FxHashMap<String, (Arc<str>, String)>,
    last_heartbeat_time: f64,

    push: PushState,
}

impl NixlConnectorScheduler {
    pub fn new(config: NixlSchedulerConfig) -> Result<Self, NixlConfigError> {
        let n = config.transfer_group_ids.len();
        if config.blocks_per_sw.len() != n || config.ssm_spec_blocks.len() != n {
            return Err(NixlConfigError::Unsupported(
                "per-transfer-group settings must match transfer_group_ids".to_owned(),
            ));
        }
        if config.transfer_group_ids.iter().any(|&g| g >= config.num_kv_cache_groups) {
            return Err(NixlConfigError::Unsupported(
                "transfer group id out of range".to_owned(),
            ));
        }
        if config.mode == TransferMode::Push && config.bidirectional_kv_xfer {
            return Err(NixlConfigError::Unsupported(
                "Bidirectional KV transfer is not supported for NIXL push connector.".to_owned(),
            ));
        }
        Ok(Self {
            config,
            clock: Box::new(perf_counter),
            registration_generation: 0,
            params: FxHashMap::default(),
            reqs_need_recv: IndexMap::new(),
            reqs_need_save: IndexMap::new(),
            reqs_need_send: IndexMap::new(),
            reqs_in_batch: IndexSet::new(),
            reqs_not_processed: IndexSet::new(),
            heartbeat_by_engine: IndexMap::new(),
            heartbeat_req_engine: FxHashMap::default(),
            last_heartbeat_time: 0.0,
            push: PushState::default(),
        })
    }

    pub fn config(&self) -> &NixlSchedulerConfig {
        &self.config
    }

    /// Replace the [`perf_counter`] clock (tests).
    pub fn set_clock(&mut self, clock: PerfClock) {
        self.clock = clock;
    }

    /// The handshake listener's `_registration_generation`, reported in the
    /// finished requests' params (pull).
    pub fn set_registration_generation(&mut self, generation: u64) {
        self.registration_generation = generation;
    }

    fn params_of(&self, request_id: &str) -> Option<&NixlKvTransferParams> {
        params_of(&self.params, request_id)
    }

    /// `_prefill_backoff`: trailing prompt tokens the prefiller must not
    /// compute; the decoder recomputes them locally.
    pub fn prefill_backoff(&self) -> usize {
        let base = usize::from(self.config.has_mamba || self.config.bounded_replay);
        base.max(self.config.num_prefill_lookahead_tokens.saturating_sub(1))
    }

    /// `_get_remote_prefill_token_count` (D side).
    fn remote_prefill_token_count(&self, num_prompt_tokens: usize) -> usize {
        let backoff = self.prefill_backoff();
        if backoff > 0 && num_prompt_tokens > backoff {
            num_prompt_tokens - backoff
        } else {
            num_prompt_tokens
        }
    }

    /// The P-side prompt truncation of `on_new_request`
    /// (`_truncate_request_for_prefill`): the number of trailing prompt
    /// tokens to drop, if any. The caller drops them from the prompt, sets
    /// `max_tokens = 1`, truncates the block hashes and then sets
    /// `params.p_side_truncated`; a request with neither prompt token ids nor
    /// prompt embeds is left alone.
    pub fn prefill_truncation(
        &self,
        params: Option<&NixlKvTransferParams>,
        num_prompt_tokens: usize,
        skip_reading_external_kv: bool,
    ) -> Option<usize> {
        let params = params?;
        let backoff = self.prefill_backoff();
        (params.do_remote_decode
            && backoff > 0
            && !params.p_side_truncated
            && num_prompt_tokens > backoff
            && !skip_reading_external_kv)
            .then_some(backoff)
    }

    /// Register a new request's params, before `on_new_request` (pass `None`
    /// for absent or empty params).
    ///
    /// Push-mode D-side requests must carry `remote_engine_id`,
    /// `remote_request_id`, `remote_host`, `remote_port` and `tp_size`:
    /// Python raises `KeyError` without them once the request is scheduled
    /// or aborted, here the request is rejected up front.
    pub fn add_request(
        &mut self,
        request_id: &str,
        params: Option<NixlKvTransferParams>,
    ) -> Result<(), NixlParamsError> {
        if self.config.mode == TransferMode::Push
            && let Some(p) = &params
            && p.do_remote_prefill
            && !(p.has_remote_address() && p.tp_size.is_some())
        {
            return Err(NixlParamsError::Invalid(
                "push-mode do_remote_prefill requires remote_engine_id, remote_request_id, \
                 remote_host, remote_port and tp_size"
                    .to_owned(),
            ));
        }
        self.params.insert(request_id.to_owned(), params);
        Ok(())
    }

    /// `on_new_request`, after the P-side truncation: track the request for
    /// lease-renewal heartbeats.
    pub fn on_new_request(&mut self, request_id: &str) {
        let Some(params) = params_of(&self.params, request_id) else {
            return;
        };
        if !params.do_remote_prefill {
            return;
        }
        let (Some(engine_id), Some(remote_request_id), Some(host), Some(port), Some(tp_size)) = (
            &params.remote_engine_id,
            &params.remote_request_id,
            &params.remote_host,
            params.remote_port,
            params.tp_size,
        ) else {
            return;
        };
        let (engine_id, remote_request_id) = (engine_id.clone(), remote_request_id.clone());
        let info =
            self.heartbeat_by_engine
                .entry(engine_id.clone())
                .or_insert_with(|| HeartbeatInfo {
                    req_ids: IndexSet::new(),
                    host: host.clone(),
                    port,
                    tp_size,
                    dcp_size: params.dcp_size.unwrap_or(1),
                    pp_size: params.pp_size.unwrap_or(1),
                });
        info.req_ids.insert(remote_request_id.clone());
        self.heartbeat_req_engine
            .insert(request_id.to_owned(), (engine_id, remote_request_id));
    }

    /// `_stop_heartbeat`.
    fn stop_heartbeat(&mut self, request_id: &str) {
        if let Some((engine_id, remote_id)) = self.heartbeat_req_engine.remove(request_id)
            && let Some(info) = self.heartbeat_by_engine.get_mut(&engine_id)
        {
            // Python sets are unordered: no order to keep.
            info.req_ids.swap_remove(&remote_id);
            if info.req_ids.is_empty() {
                // Python dicts keep the order of the remaining engines.
                self.heartbeat_by_engine.shift_remove(&engine_id);
            }
        }
    }

    /// `may_load_async` (the base default).
    pub fn may_load_async(&self, request_id: &str) -> bool {
        self.params_of(request_id).is_some_and(|p| p.do_remote_prefill)
    }

    /// `get_exchange_clipped_blocks` on all-group block ids.
    fn clipped_blocks(&self, block_ids: &[Vec<BlockId>], clip_ssm: bool) -> Vec<Vec<BlockId>> {
        assert_eq!(
            block_ids.len(),
            self.config.num_kv_cache_groups,
            "Expected {} KV cache groups, got {}.",
            self.config.num_kv_cache_groups,
            block_ids.len()
        );
        let selected = self.config.transfer_group_ids.iter().map(|&g| &block_ids[g]);
        if !self.config.is_hma_required {
            return selected.cloned().collect();
        }
        selected
            .enumerate()
            .map(|(i, blocks)| {
                let n_sw = self.config.blocks_per_sw[i];
                if n_sw > 0 {
                    blocks[blocks.len().saturating_sub(n_sw)..].to_vec()
                } else if let Some(n_spec) = self.config.ssm_spec_blocks[i]
                    && clip_ssm
                {
                    clip_ssm_state_blocks(blocks, n_spec).to_vec()
                } else {
                    blocks.clone()
                }
            })
            .collect()
    }

    /// `get_exchange_clipped_blocks` of all-group block ids, as the tuple
    /// Python returns.
    fn clipped_tuple(&self, block_ids: &[Vec<BlockId>], clip_ssm: bool) -> BlockIds {
        Seq::new(SeqKind::Tuple, self.clipped_blocks(block_ids, clip_ssm))
    }

    /// `get_num_new_matched_tokens`.
    pub fn get_num_new_matched_tokens(
        &self,
        request: &NixlRequestView<'_>,
        num_computed_tokens: usize,
    ) -> (usize, bool) {
        let Some(params) = self.params_of(request.request_id) else {
            return (0, false);
        };
        if params.do_remote_prefill {
            // Remote prefill: get all prompt blocks from remote.
            let actual = self.remote_prefill_token_count(request.num_prompt_token_ids);
            if actual > num_computed_tokens {
                return (actual - num_computed_tokens, true);
            }
        }
        if self.config.mode == TransferMode::Pull
            && params.do_remote_decode
            && params.has_remote_block_ids()
            && params.has_remote_address()
        {
            // The decode node holds KV blocks for part of this prefill request.
            let remote_num_tokens = params.remote_num_tokens.unwrap_or(0);
            let count = remote_num_tokens.min(request.num_prompt_tokens as i64)
                - num_computed_tokens as i64;
            if count > 0 {
                let threshold = self.config.kv_recompute_threshold;
                if threshold > 0 && count < threshold {
                    // Too few remote tokens to amortize the transfer.
                    return (0, false);
                }
                return (count as usize, true);
            }
        }
        (0, false)
    }

    /// `update_state_after_alloc`.
    pub fn update_state_after_alloc(
        &mut self,
        request_id: &str,
        blocks: &dyn AllocatedBlocks,
        num_external_tokens: usize,
    ) {
        match self.config.mode {
            TransferMode::Pull => {
                self.pull_update_state_after_alloc(request_id, blocks, num_external_tokens)
            }
            TransferMode::Push => {
                self.push_update_state_after_alloc(request_id, blocks, num_external_tokens)
            }
        }
    }

    fn pull_update_state_after_alloc(
        &mut self,
        request_id: &str,
        blocks: &dyn AllocatedBlocks,
        num_external_tokens: usize,
    ) {
        let bidirectional = self.config.bidirectional_kv_xfer;
        let Some(params) = params_of(&self.params, request_id) else {
            return;
        };
        if params.do_remote_decode || (params.do_remote_prefill && bidirectional) {
            self.reqs_in_batch.insert(request_id.to_owned());
        }
        if self.config.use_host_buffer && params.do_remote_decode {
            // Prefilled blocks are saved to host memory before the transfer.
            let meta_params = ReqMetaParams::of(params);
            self.reqs_need_save.insert(request_id.to_owned(), meta_params);
            return;
        }
        if !(params.do_remote_prefill
            || (params.do_remote_decode && bidirectional && !params.remote_blocks_processed))
        {
            return;
        }
        if params.has_remote_block_ids() {
            if params.has_remote_address() {
                // Remote blocks with no external tokens: a full local prefix
                // cache hit; the worker still notifies the remote to free them.
                let local_block_ids = if num_external_tokens > 0 {
                    self.clipped_tuple(&blocks.unhashed_block_ids_all_groups(), true)
                } else {
                    Seq::new(SeqKind::Tuple, Vec::new())
                };
                let meta = recv_req_meta(
                    params,
                    Arc::new(local_block_ids),
                    blocks.num_computed_blocks(),
                    // Parked in WAITING_FOR_REMOTE_KVS only when there is
                    // something to pull.
                    num_external_tokens > 0,
                )
                .expect("remote address checked");
                self.reqs_need_recv.insert(request_id.to_owned(), meta);
            }
            // Python logs a warning for the invalid params and skips the
            // transfer.
        } else {
            assert_eq!(num_external_tokens, 0);
        }
        // Only trigger 1 KV transfer per request.
        let params = params_mut(&mut self.params, request_id).expect("checked above");
        params.do_remote_prefill = false;
        params.remote_blocks_processed = true;
    }

    fn push_update_state_after_alloc(
        &mut self,
        request_id: &str,
        blocks: &dyn AllocatedBlocks,
        num_external_tokens: usize,
    ) {
        let Some(params) = params_of(&self.params, request_id) else {
            return;
        };
        if params.do_remote_decode {
            self.reqs_in_batch.insert(request_id.to_owned());
        }
        if self.config.use_host_buffer && params.do_remote_decode {
            let meta_params = ReqMetaParams::of(params);
            self.reqs_need_save.insert(request_id.to_owned(), meta_params);
            return;
        }
        if !params.do_remote_prefill || num_external_tokens == 0 {
            return;
        }
        let local_block_ids =
            Arc::new(self.clipped_tuple(&blocks.unhashed_block_ids_all_groups(), true));
        // `add_request` checked the remote fields of push D-side requests.
        let registration = PushRegistration {
            request_id: request_id.to_owned(),
            decode_engine_id: self.config.engine_id.clone(),
            decode_host: self.config.side_channel_host.clone(),
            decode_port: self.config.side_channel_port,
            decode_tp_size: self.config.tensor_parallel_size,
            local_block_ids: local_block_ids.clone(),
            remote_engine_id: params.remote_engine_id.clone().expect("checked by add_request"),
            remote_host: params.remote_host.clone().expect("checked by add_request"),
            remote_port: params.remote_port.expect("checked by add_request"),
            remote_tp_size: params.tp_size.expect("checked by add_request"),
            remote_pp_size: params.pp_size.unwrap_or(1),
        };
        self.push.pending_registrations.insert(request_id.to_owned(), registration);
        let deadline = (self.clock)() + self.config.push_registration_timeout;
        self.push.registration_deadlines.insert(request_id.to_owned(), deadline);
        let params = params_mut(&mut self.params, request_id).expect("checked above");
        // P learns its blocks from the registration.
        params.remote_block_ids = Some(Arc::new(RemoteBlockIds::empty_tuple()));
        let meta = recv_req_meta(params, local_block_ids, Vec::new(), false)
            .expect("checked by add_request");
        // Mark as processed so a re-entry doesn't re-stage the registration.
        params.do_remote_prefill = false;
        self.reqs_need_recv.insert(request_id.to_owned(), meta);
    }

    /// `request_finished` (`request_finished_all_groups`): whether the blocks
    /// are freed asynchronously, and the params for the request's output.
    /// Drops the request's params.
    pub fn request_finished(
        &mut self,
        request: &NixlRequestView<'_>,
        block_ids: &[Vec<BlockId>],
    ) -> (bool, Option<NixlFinishedParams>) {
        let Some(mut params) = self.params.remove(request.request_id).flatten() else {
            return (false, None);
        };
        let request_id = request.request_id;
        let push = self.config.mode == TransferMode::Push;
        let is_p_node = params.do_remote_decode;
        let is_d_node = !is_p_node;

        // Aborted requests that never reached finished_recving.
        self.stop_heartbeat(request_id);
        if push {
            self.push.registration_deadlines.shift_remove(request_id);
        }

        if params.do_remote_prefill {
            // update_state_after_alloc never ran: the request was aborted
            // before it was scheduled. An empty recv lets the worker notify
            // the remote to free the prefill blocks.
            if push {
                params.remote_block_ids = Some(Arc::new(RemoteBlockIds::empty_tuple()));
            }
            let local_block_ids = Arc::new(Seq::new(SeqKind::List, Vec::new()));
            // Python raises `KeyError` building the metadata when a remote
            // field is missing; without a remote there is nothing to free.
            if let Some(meta) = recv_req_meta(&params, local_block_ids, Vec::new(), false) {
                self.reqs_need_recv.insert(request_id.to_owned(), meta);
            }
            return (false, None);
        }

        if push && !is_p_node {
            return (false, None);
        }
        if !push && is_d_node && !self.config.bidirectional_kv_xfer {
            return (false, None);
        }

        if !matches!(
            request.status,
            RequestStatus::FinishedLengthCapped | RequestStatus::FinishedStopped
        ) {
            // Aborted (or otherwise not completed): stop tracking.
            self.reqs_not_processed.insert(request_id.to_owned());
            self.reqs_need_save.shift_remove(request_id);
            return (false, None);
        }

        let delay_free_blocks = block_ids.iter().any(|g| !g.is_empty());
        let mut remote_num_tokens = 0;
        let mut blocks_expiry_time = None;
        let remote_block_ids = if delay_free_blocks {
            let ttl = if is_d_node {
                // Blocks pinned on D: a plain timeout, the turn-2 request is
                // client-driven.
                self.config.decoder_kv_blocks_ttl
            } else {
                self.config.kv_lease_duration
            };
            let deadline = (self.clock)() + ttl;
            self.reqs_need_send.insert(request_id.to_owned(), deadline);
            if is_d_node {
                blocks_expiry_time = Some(deadline);
            }
            remote_num_tokens = request.num_computed_tokens;
            let clipped = self.clipped_blocks(block_ids, true);
            if push {
                self.push.finished_request_blocks.insert(request_id.to_owned());
                self.push.newly_finished_blocks.insert(
                    request_id.to_owned(),
                    Seq::new(SeqKind::Tuple, clipped.clone()),
                );
            }
            clipped
        } else {
            block_ids.to_vec()
        };

        let config = &self.config;
        let finished = NixlFinishedParams {
            mode: config.mode,
            do_remote_prefill: if push { true } else { is_p_node },
            do_remote_decode: if push { false } else { is_d_node },
            remote_block_ids,
            remote_engine_id: config.engine_id.clone(),
            remote_request_id: request_id.to_owned(),
            remote_host: config.side_channel_host.clone(),
            remote_port: config.side_channel_port,
            tp_size: config.transfer_tp_size,
            dcp_size: config.decode_context_parallel_size,
            pp_size: config.pipeline_parallel_size,
            remote_num_tokens,
            remote_blocks_expiry_time: blocks_expiry_time,
            remote_registration_generation: self.registration_generation,
        };
        (delay_free_blocks, Some(finished))
    }

    /// `_build_save_meta` (host buffer only).
    fn build_save_meta<'a>(
        &mut self,
        meta: &mut NixlConnectorMetadata,
        scheduled: impl IntoIterator<Item = ScheduledReq<'a>>,
    ) {
        for req in scheduled {
            let Some(new_block_ids) = req.new_block_ids else {
                continue;
            };
            let Some(&meta_params) = self.reqs_need_save.get(req.req_id) else {
                continue;
            };
            let local_block_ids = Arc::new(self.clipped_tuple(new_block_ids, false));
            meta.reqs_to_save.insert(
                req.req_id.to_owned(),
                meta_params.req_meta(local_block_ids, Vec::new(), false),
            );
            // A chunked prefill stays until all its blocks are scheduled.
            if req.num_computed_tokens + req.num_scheduled_tokens >= req.num_prompt_tokens {
                self.reqs_need_save.shift_remove(req.req_id);
            }
        }
    }

    /// `build_connector_meta`. `scheduled` yields the step's scheduled
    /// requests in `yield_req_data` order; it is only consumed with a host
    /// buffer.
    pub fn build_connector_meta<'a>(
        &mut self,
        scheduled: impl IntoIterator<Item = ScheduledReq<'a>>,
    ) -> NixlConnectorMetadata {
        let mut meta = NixlConnectorMetadata {
            reqs_to_recv: std::mem::take(&mut self.reqs_need_recv),
            ..Default::default()
        };
        if self.config.use_host_buffer {
            self.build_save_meta(&mut meta, scheduled);
        }
        meta.reqs_to_send = std::mem::take(&mut self.reqs_need_send);
        // Workers rebase the deadlines above onto their own clocks.
        meta.scheduler_clock = (self.clock)();
        meta.reqs_in_batch = std::mem::take(&mut self.reqs_in_batch);
        meta.reqs_not_processed = std::mem::take(&mut self.reqs_not_processed);

        // Heartbeats, throttled by heartbeat_interval.
        if !self.heartbeat_by_engine.is_empty() {
            let now = (self.clock)();
            if now - self.last_heartbeat_time >= self.config.heartbeat_interval {
                self.last_heartbeat_time = now;
                // Python hands out its live dict; this is the snapshot the
                // (multiprocess) workers receive.
                meta.heartbeat_by_engine = self.heartbeat_by_engine.clone();
            }
        }

        if self.config.mode == TransferMode::Push {
            self.build_push_meta(&mut meta);
        }
        meta
    }

    fn build_push_meta(&mut self, meta: &mut NixlConnectorMetadata) {
        let push = &mut self.push;
        // Watchdog: drop registrations that were never fulfilled. Deadlines
        // are armed in non-decreasing order, so stop at the first live one.
        let now = (self.clock)();
        let num_expired = push
            .registration_deadlines
            .values()
            .take_while(|&&deadline| deadline <= now)
            .count();
        for (rid, _) in push.registration_deadlines.drain(..num_expired) {
            // Avoid resending a registration that already timed out.
            push.pending_registrations.shift_remove(&rid);
        }
        meta.push_registrations = std::mem::take(&mut push.pending_registrations);
        meta.push_finished_blocks = std::mem::take(&mut push.newly_finished_blocks);
    }

    /// `update_connector_output`.
    pub fn update_connector_output(
        &mut self,
        finished_sending: Option<&[String]>,
        finished_recving: Option<&[String]>,
    ) {
        for req_id in finished_recving.unwrap_or_default() {
            self.stop_heartbeat(req_id);
        }
        if self.config.mode == TransferMode::Push {
            for req_id in finished_sending.unwrap_or_default() {
                self.push.finished_request_blocks.remove(req_id);
            }
            // D side: the push completed.
            for req_id in finished_recving.unwrap_or_default() {
                self.push.registration_deadlines.shift_remove(req_id);
            }
        }
    }

    /// `has_pending_push_work`.
    pub fn has_pending_push_work(&self) -> bool {
        self.config.mode == TransferMode::Push
            && !(self.push.finished_request_blocks.is_empty()
                && self.push.pending_registrations.is_empty())
    }
}

/// A NIXL configuration the native scheduler half does not implement.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum NixlConfigError {
    #[error("unsupported NIXL configuration: {0}")]
    Unsupported(String),
}
