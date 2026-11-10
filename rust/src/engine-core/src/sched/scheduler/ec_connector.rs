// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The scheduler-side encoder cache (EC) connector and auxiliary output
//! connector interfaces.
//!
//! Both are Python collaborators in `Scheduler` (`ECConnectorBase` with the
//! scheduler role, `AuxOutputSchedulerConnector`); the binding layer
//! implements these traits over them. Their metadata, outputs and stats are
//! Python objects the scheduler passes along as [`Opaque`] values.

use indexmap::IndexSet;

use super::Scheduler;
use super::connector::Requests;
use crate::kv_cache::KvCacheManager;
use crate::sched::output::{Opaque, SchedulerOutput};
use crate::sched::request::Request;

/// The scheduler-side `ECConnectorBase` methods the scheduler calls.
pub trait EcConnector {
    /// Ensure encoder cache items are available for the given request.
    /// May initiate asynchronous transfers for items not yet local.
    ///
    /// Returns false if any items are still in transit (the request should
    /// be deferred).
    fn ensure_cache_available(&mut self, request: &Request, num_computed_tokens: usize) -> bool;

    /// Check if a single encoder cache exists.
    fn has_cache_item(&mut self, identifier: &str) -> bool;

    /// Update connector state after an encoder cache entry was allocated.
    fn update_state_after_alloc(&mut self, request: &Request, index: usize);

    /// Notify the connector that an encoder cache entry was released.
    fn update_state_after_free(&mut self, request: &Request, index: usize);

    /// Build the connector metadata for this step.
    fn build_connector_meta(
        &mut self,
        scheduler_output: &SchedulerOutput,
        requests: Requests<'_>,
    ) -> Opaque;

    /// Request ids whose encoder inputs the connector can no longer obtain,
    /// in the Python set's iteration order.
    fn take_unavailable_requests(&mut self) -> IndexSet<String>;

    /// Update connector state from the (truthy) worker-side
    /// `ECConnectorOutput`.
    fn update_connector_output(&mut self, ec_connector_output: &Opaque);

    /// Scheduler-side connector stats, `None` when absent or empty.
    fn get_ec_connector_stats(&mut self) -> Option<Opaque>;

    /// Called when a request has finished, before its encoder cache is
    /// freed. Returns whether the request's blocks must stay allocated until
    /// the connector reports it done, and the `ec_transfer_params` for the
    /// request's output.
    fn request_finished(&mut self, request: &Request) -> (bool, Option<Opaque>);

    /// Whether push-mode work requires the engine main loop to keep stepping.
    fn has_pending_push_work(&mut self) -> bool;
}

/// The `AuxOutputSchedulerConnector` methods the scheduler calls.
///
/// The connector reads the scheduler's KV cache manager (its block pool's
/// cache) in the hooks that receive it.
pub trait AuxOutputConnector {
    /// The longest prefix-cache hit whose auxiliary outputs are retained.
    fn max_cache_hit(&mut self, request: &Request, kv_cache_manager: &mut KvCacheManager) -> usize;

    /// `num_external` capped to what the retained auxiliary outputs cover.
    fn cap_external_hit(
        &mut self,
        request: &Request,
        num_local: usize,
        num_external: usize,
        kv_cache_manager: &mut KvCacheManager,
    ) -> usize;

    /// The request starts an async KV load of `num_tokens`.
    fn request_loading(&mut self, request: &Request, num_tokens: usize);

    /// Build one step's incremental worker metadata.
    fn build_connector_meta(
        &mut self,
        scheduler_output: &SchedulerOutput,
        requests: Requests<'_>,
        kv_cache_manager: &mut KvCacheManager,
    ) -> Opaque;

    /// Queue a request's terminal event and final block hashes (also on
    /// preemption).
    fn request_finished(&mut self, request: &Request, kv_cache_manager: &mut KvCacheManager);

    /// The accepted rows of one request in the worker's
    /// `aux_output_connector_output` (`routed_experts`).
    fn take_output(&mut self, request: &Request, output: Option<&Opaque>) -> Option<Opaque>;

    /// Start a new namespace after a prefix-cache reset.
    fn reset(&mut self);
}

impl Scheduler {
    /// Whether an encoder input this request needs is still in transit.
    pub(super) fn ec_transfer_pending(&mut self, h: usize, num_computed_tokens: usize) -> bool {
        let request = &self.requests[h];
        self.ec_connector.as_mut().is_some_and(|ec_connector| {
            !request.mm_features.is_empty()
                && !ec_connector.ensure_cache_available(request, num_computed_tokens)
        })
    }
}
