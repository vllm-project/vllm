// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The scheduler half of the NIXL KV connector
//! (`vllm/distributed/kv_transfer/kv_connector/v1/nixl/`): the
//! `NixlPullConnectorScheduler` / `NixlPushConnectorScheduler` hooks the
//! scheduler calls, natively, so that scheduling makes no Python calls.
//!
//! - [`NixlConnectorScheduler`] holds the connector state; its hooks take
//!   plain views of the request and its blocks.
//! - [`NixlKvConnector`] adapts it to the scheduler's [`KvConnector`]; the
//!   binding layer keeps a handle to register each request's parsed
//!   `kv_transfer_params` ([`NixlConnectorScheduler::add_request`]) and to
//!   build the Python metadata for the unchanged Python worker half.
//!
//! The Python connector stays the behavioral reference and keeps what runs
//! outside the scheduling loop: config parsing, the handshake side channel
//! (whose `_registration_generation` the binding forwards with
//! [`NixlConnectorScheduler::set_registration_generation`]) and shutdown.

mod metadata;
mod params;
mod scheduler;

#[cfg(test)]
mod tests;

use std::sync::{Arc, Mutex, MutexGuard};

pub use self::metadata::{
    HeartbeatInfo, NixlConnectorMetadata, PushRegistration, RemoteMeta, ReqMeta,
};
pub use self::params::{
    BlockIds, NixlFinishedParams, NixlKvTransferParams, PyNumber, RemoteBlockIds, Seq, SeqKind,
    TransferMode,
};
pub use self::scheduler::{
    AllocatedBlocks, NixlConfigError, NixlConnectorScheduler, NixlParamsError, NixlRequestView,
    NixlSchedulerConfig, PerfClock, ScheduledReq, perf_counter,
};
use crate::kv_cache::{BlockId, BoundaryStateOffload, KvCacheBlocks, KvCacheManager};
use crate::sched::output::{Opaque, SchedulerOutput};
use crate::sched::request::Request;
use crate::sched::scheduler::{
    KvConnector, KvConnectorBlockState, KvConnectorOutput, KvTransferParams, RequestFinished,
};

/// `CacheHitSource.P2P`, NIXL's `_cache_hit_source`.
const CACHE_HIT_SOURCE_P2P: &str = "p2p";

impl<'a> NixlRequestView<'a> {
    pub fn of(request: &'a Request) -> Self {
        Self {
            request_id: &request.request_id,
            num_prompt_token_ids: request.prompt_token_ids.as_ref().map_or(0, Vec::len),
            num_prompt_tokens: request.num_prompt_tokens,
            num_computed_tokens: request.num_computed_tokens,
            status: request.status,
        }
    }
}

/// `KVCacheBlocks` with the manager whose pools its blocks live in.
pub struct ManagerBlocks<'a> {
    pub blocks: &'a KvCacheBlocks,
    pub kv_cache_manager: &'a KvCacheManager,
}

impl AllocatedBlocks for ManagerBlocks<'_> {
    fn unhashed_block_ids_all_groups(&self) -> Vec<Vec<BlockId>> {
        self.blocks
            .get_unhashed_block_ids_all_groups(|g| self.kv_cache_manager.group_pool(g))
    }

    fn num_computed_blocks(&self) -> Vec<usize> {
        self.blocks
            .blocks
            .iter()
            .enumerate()
            .map(|(group_id, group)| {
                let pool = self.kv_cache_manager.group_pool(group_id);
                group
                    .iter()
                    .filter(|&&b| {
                        let block = pool.block(b);
                        block.block_hash.is_some() && !block.is_null
                    })
                    .count()
            })
            .collect()
    }
}

/// [`NixlConnectorScheduler`] as the scheduler's [`KvConnector`], sharing the
/// connector with the binding layer.
#[derive(Clone)]
pub struct NixlKvConnector {
    inner: Arc<Mutex<NixlConnectorScheduler>>,
    requires_kv_delivery: bool,
    transfer_group_ids: Vec<usize>,
}

impl NixlKvConnector {
    pub fn new(scheduler: NixlConnectorScheduler) -> Self {
        let config = scheduler.config();
        Self {
            requires_kv_delivery: config.is_kv_producer,
            transfer_group_ids: config.transfer_group_ids.clone(),
            inner: Arc::new(Mutex::new(scheduler)),
        }
    }

    pub fn lock(&self) -> MutexGuard<'_, NixlConnectorScheduler> {
        self.inner.lock().unwrap()
    }

    /// The `on_new_request` work that precedes scheduling, for a request
    /// whose `kv_transfer_params` came off the wire rather than in a Python
    /// request: truncate a P-side prompt in the Rust request, as
    /// `_truncate_request_for_prefill` does, then register the params.
    pub fn add_wire_request(
        &self,
        request: &mut Request,
        mut params: Option<NixlKvTransferParams>,
        hash_block_size: usize,
        num_hash_lookahead_tokens: usize,
    ) -> Result<(), NixlParamsError> {
        let mut scheduler = self.lock();
        let truncation = scheduler.prefill_truncation(
            params.as_ref(),
            request.num_prompt_tokens,
            request.skip_reading_external_kv(),
        );
        if truncation.is_some() && hash_block_size == 0 {
            return Err(NixlParamsError::Invalid(
                "hash_block_size is not resolved".to_owned(),
            ));
        }
        if let (Some(backoff), Some(params)) = (truncation, params.as_mut())
            && truncate_request_for_prefill(
                request,
                backoff,
                hash_block_size,
                num_hash_lookahead_tokens,
            )
        {
            params.p_side_truncated = true;
        }
        scheduler.add_request(&request.request_id, params)
    }
}

/// `_truncate_request_for_prefill` on a Rust request: drop its trailing
/// `backoff` prompt tokens, so that the prefiller stops short of what the
/// decoder recomputes, and the block hashes that invalidates. `false` for a
/// request without prompt token ids, which is left alone (the binding layer
/// holds prompt embeddings).
pub fn truncate_request_for_prefill(
    request: &mut Request,
    backoff: usize,
    hash_block_size: usize,
    num_hash_lookahead_tokens: usize,
) -> bool {
    let Some(prompt) = request.prompt_token_ids.as_mut() else {
        return false;
    };
    prompt.truncate(prompt.len() - backoff);
    let num_tokens = request.all_token_ids().len() - backoff;
    request.truncate_all_token_ids(num_tokens);
    request.num_prompt_tokens -= backoff;
    request.max_tokens = 1;
    request.truncate_block_hashes(hash_block_size, num_hash_lookahead_tokens);
    true
}

impl KvConnector for NixlKvConnector {
    fn supports_divergent_local_hybrid_hits(&self) -> bool {
        true
    }

    fn requires_kv_delivery(&self) -> bool {
        self.requires_kv_delivery
    }

    /// The P-side prompt truncation runs before the request is built (see
    /// [`NixlConnectorScheduler::prefill_truncation`]).
    fn on_new_request(&mut self, request: &Request, _kv_cache_manager: &mut KvCacheManager) {
        self.lock().on_new_request(&request.request_id);
    }

    fn may_load_async(&mut self, request: &Request) -> bool {
        self.lock().may_load_async(&request.request_id)
    }

    fn get_num_new_matched_tokens(
        &mut self,
        request: &mut Request,
        num_computed_tokens: usize,
        _kv_cache_manager: &mut KvCacheManager,
    ) -> (Option<usize>, bool) {
        let (n, load_async) = self
            .lock()
            .get_num_new_matched_tokens(&NixlRequestView::of(request), num_computed_tokens);
        (Some(n), load_async)
    }

    fn get_loaded_kv_cache_group_ids(&mut self, _request: &Request) -> Vec<usize> {
        self.transfer_group_ids.clone()
    }

    fn update_state_after_alloc(
        &mut self,
        request: &Request,
        blocks: &KvCacheBlocks,
        kv_cache_manager: &mut KvCacheManager,
        num_external_tokens: usize,
    ) {
        let blocks = ManagerBlocks {
            blocks,
            kv_cache_manager,
        };
        self.lock()
            .update_state_after_alloc(&request.request_id, &blocks, num_external_tokens);
    }

    fn get_external_cache_hit_sources(
        &mut self,
        _request: &Request,
        num_external_tokens: usize,
    ) -> Option<Vec<(String, usize)>> {
        // The base default: everything from `_cache_hit_source`.
        Some(if num_external_tokens == 0 {
            Vec::new()
        } else {
            vec![(CACHE_HIT_SOURCE_P2P.to_owned(), num_external_tokens)]
        })
    }

    fn has_pending_block_frees(&mut self, _kv_cache_manager: &mut KvCacheManager) -> bool {
        false
    }

    fn build_connector_meta(
        &mut self,
        scheduler_output: &SchedulerOutput,
        block_state: &mut KvConnectorBlockState<'_>,
    ) -> Opaque {
        let requests = block_state.requests;
        let cached = &scheduler_output.scheduled_cached_reqs;
        // `yield_req_data`: new requests with their whole block table, then
        // cached ones with this step's additions.
        let scheduled = scheduler_output
            .scheduled_new_reqs
            .iter()
            .map(|r| (r.req_id.as_str(), Some(r.block_ids.as_slice())))
            .chain(
                cached
                    .req_ids
                    .iter()
                    .zip(&cached.new_block_ids)
                    .map(|(id, b)| (id.as_str(), b.as_deref())),
            )
            .map(|(req_id, new_block_ids)| {
                let request = requests.get(req_id).expect("scheduled request");
                ScheduledReq {
                    req_id,
                    new_block_ids,
                    num_scheduled_tokens: scheduler_output.num_scheduled_tokens[req_id],
                    num_computed_tokens: request.num_computed_tokens,
                    num_prompt_tokens: request.num_prompt_tokens,
                }
            });
        Opaque::new(self.lock().build_connector_meta(scheduled))
    }

    fn register_finished_partial_tail(
        &mut self,
        _request: &Request,
        _block_ids: &[Vec<BlockId>],
        _partial_tail_offloads: &[BoundaryStateOffload],
        _kv_cache_manager: &mut KvCacheManager,
    ) -> bool {
        false
    }

    fn request_finished(
        &mut self,
        request: &Request,
        block_ids: &[Vec<BlockId>],
        _kv_cache_manager: &mut KvCacheManager,
    ) -> RequestFinished {
        let (delay_free_blocks, params) =
            self.lock().request_finished(&NixlRequestView::of(request), block_ids);
        RequestFinished {
            delay_free_blocks,
            kv_transfer_params: params.map(|params| KvTransferParams {
                do_remote_prefill: params.do_remote_prefill,
                params: Opaque::new(params),
                remote_prefill_cached_tokens: None,
            }),
        }
    }

    fn update_connector_output(
        &mut self,
        kv_connector_output: &KvConnectorOutput,
        _kv_cache_manager: &mut KvCacheManager,
    ) {
        self.lock().update_connector_output(
            kv_connector_output.finished_sending.as_deref(),
            kv_connector_output.finished_recving.as_deref(),
        );
    }

    fn get_kv_connector_stats(&mut self, _kv_cache_manager: &mut KvCacheManager) -> Option<Opaque> {
        // Stats come from the workers.
        None
    }

    fn take_events(&mut self, _kv_cache_manager: &mut KvCacheManager) -> Option<Opaque> {
        None
    }

    fn has_pending_push_work(&mut self, _kv_cache_manager: &mut KvCacheManager) -> bool {
        self.lock().has_pending_push_work()
    }

    fn reset_cache(&mut self, _kv_cache_manager: &mut KvCacheManager) -> Option<bool> {
        None
    }
}
