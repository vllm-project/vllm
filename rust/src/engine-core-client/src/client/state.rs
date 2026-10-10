// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::sync::oneshot;
use tracing::trace;

use crate::EngineId;
use crate::client::output_channel::output_channel;
pub(crate) use crate::client::output_channel::{OutputReceiver, OutputSender};
use crate::client::stream::EngineCoreStreamOutput;
use crate::error::{Error, Result};
use crate::protocol::output::{EngineCoreEventType, EngineCoreFinishReason, EngineCoreOutput};
use crate::protocol::stats::SchedulerStats;
use crate::protocol::utility::UtilityOutput;
use crate::transport::ConnectedEngine;

pub type UtilitySender = oneshot::Sender<Result<UtilityOutput>>;
pub type UtilityReceiver = oneshot::Receiver<Result<UtilityOutput>>;

#[derive(Debug)]
struct TrackedRequest {
    sender: OutputSender,
    engine_id: EngineId,
    lora: Option<LoraRequestState>,
}

/// Frontend-side view of one LoRA request's scheduling phase.
///
/// The engine's `SchedulerStats` does not carry adapter names, so
/// `vllm:lora_requests_info` must be derived from per-request lifecycle events
/// observed by this client, mirroring `LoRARequestStates` in the Python
/// frontend (`vllm/v1/engine/output_processor.py`).
#[derive(Debug)]
struct LoraRequestState {
    adapter_name: String,
    phase: LoraPhase,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoraPhase {
    Waiting,
    Running,
}

/// The latest real scheduler-side load snapshot observed from one engine.
///
/// These counters come from `scheduler_stats` on the normal engine output path
/// and are the preferred routing signal once available.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EngineLoadSnapshot {
    /// Requests still counted on the scheduler's waiting side.
    waiting: usize,
    /// Requests currently counted on the scheduler's running side.
    running: usize,
}

#[derive(Debug, Default)]
struct EngineRoutingState {
    /// Requests admitted by this frontend that have not finished yet.
    ///
    /// This is used both as the bootstrap fallback before real scheduler stats
    /// exist and as a lower bound afterwards so asynchronous scheduler
    /// snapshots cannot erase frontend admission history.
    inflight: usize,
    /// The latest real scheduler snapshot received from this engine, if any.
    last_scheduler_stats: Option<EngineLoadSnapshot>,
    /// Requests admitted since the last scheduler snapshot was received.
    ///
    /// Added to the snapshot so each admission raises the routing score before
    /// the next snapshot arrives. Only tracked once a snapshot exists.
    admitted_since_stats: usize,
}

impl EngineRoutingState {
    /// Compute the routing score used to pick the least-loaded engine.
    ///
    /// Scheduler stats, plus admissions not yet reflected in them, can raise
    /// the load estimate above the frontend-local view, but they should not
    /// lower it below requests this frontend has already admitted.
    fn routing_score(&self) -> usize {
        let Some(stats) = self.last_scheduler_stats else {
            return self.inflight;
        };

        self.inflight.max(stats.running + stats.waiting + self.admitted_since_stats)
    }

    /// Record one request admitted to this engine.
    fn record_admission(&mut self) {
        self.inflight += 1;
        if self.last_scheduler_stats.is_some() {
            self.admitted_since_stats += 1;
        }
    }

    /// Replace the local routing view with a fresh real scheduler snapshot,
    /// resetting the admissions counted on top of the previous one.
    fn apply_scheduler_counts(&mut self, next: EngineLoadSnapshot) {
        self.last_scheduler_stats = Some(next);
        self.admitted_since_stats = 0;
    }
}

/// Internal registry for tracking active requests and their output stream
/// senders.
///
/// This is used to route incoming outputs to the correct request stream, and to
/// ensure proper cleanup of senders when requests finish or the client shuts
/// down.
#[derive(Debug)]
pub struct RequestRegistry {
    closed: bool,
    requests: HashMap<String, TrackedRequest>,
    active_lora_requests: usize,
    routing_per_engine: BTreeMap<EngineId, EngineRoutingState>,
}

impl RequestRegistry {
    pub fn new(engines: &[ConnectedEngine]) -> Self {
        Self {
            closed: false,
            requests: HashMap::default(),
            active_lora_requests: 0,
            routing_per_engine: engines
                .iter()
                .map(|engine| (engine.engine_id.clone(), EngineRoutingState::default()))
                .collect(),
        }
    }

    /// Register a newly added request. Create the per-request output channel
    /// bound to its `request_id` and return the selected engine id.
    ///
    /// When `data_parallel_rank` is provided, the request is routed directly to
    /// the engine at that rank index, bypassing load balancing. Otherwise
    /// the engine with the lowest routing score is chosen.
    ///
    /// `stream_interval` sets how many new tokens the channel batches into one
    /// delivery after the first output; see [`OutputSender::send`].
    pub fn register(
        &mut self,
        request_id: String,
        lora_name: Option<String>,
        data_parallel_rank: Option<u32>,
        stream_interval: NonZeroU32,
    ) -> Result<(EngineId, OutputReceiver)> {
        if self.requests.contains_key(&request_id) {
            return Err(Error::DuplicateRequestId { request_id });
        }

        let engine_id = self.choose_engine_for_request(data_parallel_rank)?;
        let (tx, rx) = output_channel(stream_interval);
        let lora = lora_name.map(|adapter_name| LoraRequestState {
            adapter_name,
            phase: LoraPhase::Waiting,
        });
        if lora.is_some() {
            self.active_lora_requests += 1;
        }
        self.requests.insert(
            request_id,
            TrackedRequest {
                sender: tx,
                engine_id: engine_id.clone(),
                lora,
            },
        );

        let state = self
            .routing_per_engine
            .get_mut(&engine_id)
            .expect("request registry must track all known engines");
        state.record_admission();

        Ok((engine_id, rx))
    }

    fn choose_engine_for_request(&mut self, data_parallel_rank: Option<u32>) -> Result<EngineId> {
        if let Some(rank) = data_parallel_rank {
            let engine_id = u16::try_from(rank).ok().map(EngineId::from_engine_index);
            return engine_id
                .filter(|engine_id| self.routing_per_engine.contains_key(engine_id))
                .ok_or_else(|| Error::InvalidDataParallelRank {
                    rank,
                    connected_ranks: self
                        .routing_per_engine
                        .keys()
                        .filter_map(EngineId::engine_index)
                        .collect(),
                });
        }

        Ok(self
            .routing_per_engine
            .iter()
            .min_by_key(|(_, state)| state.routing_score())
            .map(|(engine_id, _)| engine_id.clone())
            .expect("request registry must contain at least one engine"))
    }

    /// Filter the given request IDs to the subset that are still tracked as
    /// active and can be aborted, grouped by engine.
    pub fn abortable_request_ids(&self, request_ids: &[String]) -> BTreeMap<EngineId, Vec<String>> {
        let mut by_engine = BTreeMap::new();
        for request_id in request_ids {
            let Some(tracked) = self.requests.get(request_id.as_str()) else {
                continue;
            };
            by_engine
                .entry(tracked.engine_id.clone())
                .or_insert_with(Vec::new)
                .push(request_id.clone());
        }
        by_engine
    }

    /// Send one output to its request stream. If it indicates the request is
    /// finished, the request will be removed from the registry. Returns the
    /// output back if its request is no longer tracked.
    ///
    /// The send happens under the registry lock: the sender is stateful (see
    /// [`OutputSender::send`]) and owned by the registry entry, and sending on
    /// its unbounded channel never blocks.
    pub fn send_output(
        &mut self,
        output: EngineCoreStreamOutput,
    ) -> Option<EngineCoreStreamOutput> {
        self.apply_lora_events(&output.output);
        if output.finished() {
            match self.remove(output.request_id.as_str()) {
                Some((mut sender, _)) => sender.send(output),
                None => return Some(output),
            }
        } else {
            match self.requests.get_mut(output.request_id.as_str()) {
                Some(tracked) => tracked.sender.send(output),
                None => return Some(output),
            }
        }
        None
    }

    /// Advance the request's LoRA scheduling phase from the engine-core events
    /// attached to one output, mirroring the Python frontend's
    /// `LoRARequestStates.update_from_events`.
    fn apply_lora_events(&mut self, output: &EngineCoreOutput) {
        let Some(events) = output.events.as_ref() else {
            return;
        };
        let Some(lora) = self
            .requests
            .get_mut(output.request_id.as_str())
            .and_then(|tracked| tracked.lora.as_mut())
        else {
            return;
        };
        for event in events {
            lora.phase = match event.r#type {
                EngineCoreEventType::Queued | EngineCoreEventType::Preempted => LoraPhase::Waiting,
                EngineCoreEventType::Scheduled => LoraPhase::Running,
            };
        }
    }

    /// Snapshot the adapter names of tracked LoRA requests as
    /// (running, waiting) sets. Feeds the `vllm:lora_requests_info` gauge.
    pub fn lora_adapter_states(&self) -> (BTreeSet<String>, BTreeSet<String>) {
        if self.active_lora_requests == 0 {
            return (BTreeSet::new(), BTreeSet::new());
        }

        let mut running = BTreeSet::new();
        let mut waiting = BTreeSet::new();
        for lora in self.requests.values().filter_map(|tracked| tracked.lora.as_ref()) {
            let set = match lora.phase {
                LoraPhase::Running => &mut running,
                LoraPhase::Waiting => &mut waiting,
            };
            set.insert(lora.adapter_name.clone());
        }
        (running, waiting)
    }

    /// Remove a batch of requests that have finished or aborted, returning
    /// their stream senders after delivering any held-back outputs.
    pub fn finish_many<'a>(
        &mut self,
        request_ids: impl IntoIterator<Item = &'a String>,
    ) -> Vec<OutputSender> {
        request_ids
            .into_iter()
            .filter_map(|request_id| self.remove(request_id.as_str()).map(|tracked| tracked.0))
            .map(|mut sender| {
                sender.flush();
                sender
            })
            .collect()
    }

    /// Apply one scheduler stats update for the given engine to the local
    /// routing state. Returns `false` if the engine is unknown to the
    /// client.
    pub fn apply_scheduler_stats(&mut self, engine_index: u32, stats: &SchedulerStats) -> bool {
        if stats.sleep_state_only {
            return u16::try_from(engine_index).ok().is_some_and(|index| {
                self.routing_per_engine.contains_key(&EngineId::from_engine_index(index))
            });
        }
        self.apply_scheduler_counts(
            engine_index,
            EngineLoadSnapshot {
                waiting: stats.num_waiting_reqs as usize,
                running: stats.num_running_reqs as usize,
            },
        )
    }

    /// Mark the registry as closed, detach and return all tracked senders.
    pub fn close(&mut self) -> Vec<OutputSender> {
        if self.closed {
            return Vec::new();
        }

        self.closed = true;
        self.active_lora_requests = 0;
        std::mem::take(&mut self.requests)
            .into_values()
            .map(|tracked| tracked.sender)
            .collect()
    }

    /// Finalize client-initiated aborts: remove each request and push a
    /// terminal output with `finish_reason = Abort` down its stream before the
    /// sender drops. Returns the request ids that were still active.
    pub fn abort_many<'a>(
        &mut self,
        request_ids: impl IntoIterator<Item = &'a String>,
        timestamp: f64,
    ) -> Vec<String> {
        let mut aborted = Vec::new();
        for request_id in request_ids {
            let Some((mut sender, engine_id)) = self.remove(request_id) else {
                continue;
            };
            let output = EngineCoreStreamOutput {
                engine_index: engine_id.engine_index().unwrap_or(0),
                timestamp,
                output: EngineCoreOutput {
                    request_id: request_id.clone(),
                    finish_reason: Some(EngineCoreFinishReason::Abort),
                    ..EngineCoreOutput::default()
                },
            };
            sender.send(output);
            aborted.push(request_id.clone());
        }
        aborted
    }

    /// Remove one request from the local registry. Returns the tracked entry if
    /// it exists.
    #[must_use]
    pub fn remove(&mut self, request_id: &str) -> Option<(OutputSender, EngineId)> {
        let tracked = self.requests.remove(request_id)?;
        if tracked.lora.is_some() {
            self.active_lora_requests -= 1;
        }
        self.routing_per_engine
            .get_mut(&tracked.engine_id)
            .expect("request registry must track all known engines")
            .inflight -= 1;
        Some((tracked.sender, tracked.engine_id))
    }

    fn apply_scheduler_counts(&mut self, engine_index: u32, next: EngineLoadSnapshot) -> bool {
        let Ok(engine_index) = u16::try_from(engine_index) else {
            return false;
        };
        let engine_id = EngineId::from_engine_index(engine_index);
        let Some(state) = self.routing_per_engine.get_mut(&engine_id) else {
            return false;
        };

        let previous = state.last_scheduler_stats;
        if previous != Some(next) {
            trace!(
                ?engine_id,
                previous_waiting = previous.map(|stats| stats.waiting),
                previous_running = previous.map(|stats| stats.running),
                waiting = next.waiting,
                running = next.running,
                "updated scheduler routing counts",
            );
        }

        state.apply_scheduler_counts(next);
        true
    }

    #[cfg(test)]
    pub fn contains(&self, request_id: &str) -> bool {
        self.requests.contains_key(request_id)
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }

    #[cfg(test)]
    fn active_lora_requests(&self) -> usize {
        self.active_lora_requests
    }
}

/// Internal registry for tracking active utility calls and their waiting
/// receivers.
#[derive(Debug)]
pub struct UtilityRegistry {
    closed: bool,
    next_call_id: AtomicU64,
    utility_calls: BTreeMap<u64, UtilitySender>,
}

impl Default for UtilityRegistry {
    fn default() -> Self {
        Self {
            closed: false,
            next_call_id: AtomicU64::new(1),
            utility_calls: BTreeMap::default(),
        }
    }
}

impl UtilityRegistry {
    /// Allocate the next utility `call_id` and register a newly added utility
    /// call.
    pub fn allocate_and_register(&mut self) -> (u64, UtilityReceiver) {
        let call_id = self.next_call_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.utility_calls.insert(call_id, tx);
        (call_id, rx)
    }

    /// Resolve a utility output to its waiting receiver.
    pub fn resolve(&mut self, call_id: &u64) -> Option<UtilitySender> {
        self.utility_calls.remove(call_id)
    }

    /// Drop a batch of registered utility calls without delivering a result.
    /// Used to roll back allocations when the dispatch fan-out fails before
    /// every engine could accept the request.
    pub fn unregister_many(&mut self, call_ids: impl IntoIterator<Item = u64>) {
        for call_id in call_ids {
            self.utility_calls.remove(&call_id);
        }
    }

    /// Mark the registry as closed, detach and return all tracked senders.
    pub fn close(&mut self) -> Vec<UtilitySender> {
        if self.closed {
            return Vec::new();
        }

        self.closed = true;
        std::mem::take(&mut self.utility_calls).into_values().collect()
    }

    #[cfg(test)]
    pub fn contains(&self, call_id: u64) -> bool {
        self.utility_calls.contains_key(&call_id)
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.utility_calls.len()
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::num::NonZeroU32;

    use crate::EngineId;
    use crate::client::state::{
        EngineLoadSnapshot, EngineRoutingState, OutputReceiver, RequestRegistry, UtilityRegistry,
    };
    use crate::client::stream::EngineCoreStreamOutput;
    use crate::mock_engine::default_ready_response;
    use crate::protocol::output::{
        EngineCoreEvent, EngineCoreEventType, EngineCoreFinishReason, EngineCoreOutput,
    };
    use crate::transport::ConnectedEngine;

    fn connected_engine(engine_id: EngineId) -> ConnectedEngine {
        ConnectedEngine {
            engine_id,
            ready_response: default_ready_response(),
        }
    }

    #[test]
    fn sleep_snapshot_preserves_routing_load() {
        let engine_id = EngineId::from_engine_index(0);
        let mut registry = RequestRegistry::new(&[connected_engine(engine_id)]);
        let stats = crate::protocol::stats::SchedulerStats {
            num_running_reqs: 7,
            num_waiting_reqs: 3,
            ..Default::default()
        };
        assert!(registry.apply_scheduler_stats(0, &stats));
        let before = registry.routing_per_engine[&engine_id].routing_score();
        let snapshot = crate::protocol::stats::SchedulerStats {
            sleep_state_only: true,
            ..Default::default()
        };
        assert!(registry.apply_scheduler_stats(0, &snapshot));
        assert_eq!(registry.routing_per_engine[&engine_id].routing_score(), before);
        assert!(!registry.apply_scheduler_stats(1, &snapshot));
    }

    fn output_with_events(
        request_id: &str,
        events: &[EngineCoreEventType],
        finish_reason: Option<EngineCoreFinishReason>,
    ) -> EngineCoreOutput {
        EngineCoreOutput {
            request_id: request_id.to_string(),
            events: Some(
                events
                    .iter()
                    .map(|event_type| EngineCoreEvent {
                        r#type: *event_type,
                        timestamp: 0.0,
                    })
                    .collect(),
            ),
            finish_reason,
            ..Default::default()
        }
    }

    fn stream_output(output: EngineCoreOutput) -> EngineCoreStreamOutput {
        EngineCoreStreamOutput {
            engine_index: 0,
            timestamp: 0.0,
            output,
        }
    }

    fn adapter_names(values: &[&str]) -> BTreeSet<String> {
        values.iter().map(|name| (*name).to_string()).collect()
    }

    #[test]
    fn registry_rejects_duplicate_request_ids() {
        let mut registry = RequestRegistry::new(&[connected_engine(EngineId::from(b"engine-0"))]);
        registry.register("req-1".to_string(), None, None, NonZeroU32::MIN).unwrap();
        let error =
            registry.register("req-1".to_string(), None, None, NonZeroU32::MIN).unwrap_err();
        assert!(matches!(
            error,
            crate::error::Error::DuplicateRequestId { request_id } if request_id == "req-1"
        ));
    }

    #[test]
    fn registry_removes_finished_request_on_output() {
        let mut registry = RequestRegistry::new(&[connected_engine(EngineId::from(b"engine-0"))]);
        registry.register("req-1".to_string(), None, None, NonZeroU32::MIN).unwrap();

        let unrouted = registry.send_output(stream_output(EngineCoreOutput {
            request_id: "req-1".to_string(),
            finish_reason: Some(EngineCoreFinishReason::Length),
            ..Default::default()
        }));

        assert!(unrouted.is_none());
        assert!(!registry.contains("req-1"));
    }

    /// Drain the outputs that are ready on `receiver` without waiting,
    /// rendering each as its token ids and finish reason.
    fn ready_outputs(
        receiver: &mut OutputReceiver,
    ) -> Vec<(Vec<u32>, Option<EngineCoreFinishReason>)> {
        let mut outputs = Vec::new();
        while let Ok(delivery) = receiver.try_recv() {
            for output in delivery.unwrap() {
                outputs.push((output.output.new_token_ids, output.output.finish_reason));
            }
        }
        outputs
    }

    #[test]
    fn registry_abort_delivers_held_back_outputs_before_terminal_output() {
        let mut registry = RequestRegistry::new(&[connected_engine(EngineId::from(b"engine-0"))]);
        let (_, mut receiver) = registry
            .register("req-1".to_string(), None, None, NonZeroU32::new(4).unwrap())
            .unwrap();

        for token_id in 1..=3 {
            let unrouted = registry.send_output(stream_output(EngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: vec![token_id],
                ..Default::default()
            }));
            assert!(unrouted.is_none());
        }
        // Only the first output is delivered; the next two are held back
        // below the stream interval.
        expect_test::expect![[r#"
            [
                (
                    [
                        1,
                    ],
                    None,
                ),
            ]
        "#]]
        .assert_debug_eq(&ready_outputs(&mut receiver));

        registry.abort_many(&["req-1".to_string()], 0.0);
        expect_test::expect![[r#"
            [
                (
                    [
                        2,
                    ],
                    None,
                ),
                (
                    [
                        3,
                    ],
                    None,
                ),
                (
                    [],
                    Some(
                        Abort,
                    ),
                ),
            ]
        "#]]
        .assert_debug_eq(&ready_outputs(&mut receiver));
    }

    #[test]
    fn registry_tracks_lora_phases_from_engine_events() {
        let mut registry = RequestRegistry::new(&[connected_engine(EngineId::from(b"engine-0"))]);
        registry
            .register(
                "req-lora".to_string(),
                Some("adapter-a".to_string()),
                None,
                NonZeroU32::MIN,
            )
            .unwrap();
        registry.register("req-plain".to_string(), None, None, NonZeroU32::MIN).unwrap();

        // Registered but not yet scheduled: counted as waiting. The non-LoRA
        // request never shows up.
        assert_eq!(
            registry.lora_adapter_states(),
            (adapter_names(&[]), adapter_names(&["adapter-a"]))
        );

        // Queued then scheduled in one output: running.
        registry.send_output(stream_output(output_with_events(
            "req-lora",
            &[EngineCoreEventType::Queued, EngineCoreEventType::Scheduled],
            None,
        )));
        assert_eq!(
            registry.lora_adapter_states(),
            (adapter_names(&["adapter-a"]), adapter_names(&[]))
        );

        // Preempted: back to waiting.
        registry.send_output(stream_output(output_with_events(
            "req-lora",
            &[EngineCoreEventType::Preempted],
            None,
        )));
        assert_eq!(
            registry.lora_adapter_states(),
            (adapter_names(&[]), adapter_names(&["adapter-a"]))
        );

        // Finished: dropped from tracking entirely.
        registry.send_output(stream_output(output_with_events(
            "req-lora",
            &[EngineCoreEventType::Scheduled],
            Some(EngineCoreFinishReason::Stop),
        )));
        assert_eq!(
            registry.lora_adapter_states(),
            (adapter_names(&[]), adapter_names(&[]))
        );
    }

    #[test]
    fn registry_unions_lora_adapters_across_requests() {
        let mut registry = RequestRegistry::new(&[connected_engine(EngineId::from(b"engine-0"))]);
        registry
            .register(
                "req-a1".to_string(),
                Some("adapter-a".to_string()),
                None,
                NonZeroU32::MIN,
            )
            .unwrap();
        registry
            .register(
                "req-a2".to_string(),
                Some("adapter-a".to_string()),
                None,
                NonZeroU32::MIN,
            )
            .unwrap();
        registry
            .register(
                "req-b".to_string(),
                Some("adapter-b".to_string()),
                None,
                NonZeroU32::MIN,
            )
            .unwrap();

        // One of adapter-a's requests starts running while the other waits:
        // the adapter appears in both sets.
        registry.send_output(stream_output(output_with_events(
            "req-a1",
            &[EngineCoreEventType::Scheduled],
            None,
        )));
        assert_eq!(
            registry.lora_adapter_states(),
            (
                adapter_names(&["adapter-a"]),
                adapter_names(&["adapter-a", "adapter-b"])
            )
        );
    }

    #[test]
    fn registry_counts_only_active_lora_requests() {
        let mut registry = RequestRegistry::new(&[connected_engine(EngineId::from(b"engine-0"))]);

        registry.register("req-plain".to_string(), None, None, NonZeroU32::MIN).unwrap();
        assert_eq!(registry.active_lora_requests(), 0);
        assert_eq!(
            registry.lora_adapter_states(),
            (adapter_names(&[]), adapter_names(&[]))
        );

        registry
            .register(
                "req-lora-a".to_string(),
                Some("adapter-a".to_string()),
                None,
                NonZeroU32::MIN,
            )
            .unwrap();
        registry
            .register(
                "req-lora-b".to_string(),
                Some("adapter-b".to_string()),
                None,
                NonZeroU32::MIN,
            )
            .unwrap();
        assert_eq!(registry.active_lora_requests(), 2);

        drop(registry.remove("req-plain"));
        assert_eq!(registry.active_lora_requests(), 2);

        drop(registry.finish_many(&["req-lora-a".to_string()]));
        assert_eq!(registry.active_lora_requests(), 1);

        drop(registry.abort_many(&["req-lora-b".to_string()], 0.0));
        assert_eq!(registry.active_lora_requests(), 0);
        assert_eq!(
            registry.lora_adapter_states(),
            (adapter_names(&[]), adapter_names(&[]))
        );
    }

    #[test]
    fn registry_clears_lora_count_on_close() {
        let mut registry = RequestRegistry::new(&[connected_engine(EngineId::from(b"engine-0"))]);
        registry
            .register(
                "req-lora".to_string(),
                Some("adapter-a".to_string()),
                None,
                NonZeroU32::MIN,
            )
            .unwrap();

        assert_eq!(registry.active_lora_requests(), 1);
        drop(registry.close());
        assert_eq!(registry.active_lora_requests(), 0);
        assert_eq!(
            registry.lora_adapter_states(),
            (adapter_names(&[]), adapter_names(&[]))
        );
    }

    #[test]
    fn registry_drops_lora_tracking_on_abort() {
        let mut registry = RequestRegistry::new(&[connected_engine(EngineId::from(b"engine-0"))]);
        registry
            .register(
                "req-lora".to_string(),
                Some("adapter-a".to_string()),
                None,
                NonZeroU32::MIN,
            )
            .unwrap();

        drop(registry.finish_many(&["req-lora".to_string()]));

        assert_eq!(
            registry.lora_adapter_states(),
            (adapter_names(&[]), adapter_names(&[]))
        );
    }

    #[test]
    fn registry_closes_all_requests_on_failure() {
        let mut registry = RequestRegistry::new(&[connected_engine(EngineId::from(b"engine-0"))]);
        registry.register("req-1".to_string(), None, None, NonZeroU32::MIN).unwrap();
        registry.register("req-2".to_string(), None, None, NonZeroU32::MIN).unwrap();

        let senders = registry.close();

        assert_eq!(senders.len(), 2);
        assert!(registry.is_closed());
    }

    #[test]
    fn registry_tracks_engine_id_per_request() {
        let engine_0 = EngineId::from_engine_index(0);
        let engine_1 = EngineId::from_engine_index(1);
        let mut registry = RequestRegistry::new(&[
            connected_engine(engine_0.clone()),
            connected_engine(engine_1.clone()),
        ]);
        let (chosen_0, _) =
            registry.register("req-1".to_string(), None, None, NonZeroU32::MIN).unwrap();
        let (chosen_1, _) =
            registry.register("req-2".to_string(), None, None, NonZeroU32::MIN).unwrap();
        let (chosen_0_again, _) =
            registry.register("req-3".to_string(), None, None, NonZeroU32::MIN).unwrap();

        assert_eq!(chosen_0, engine_0);
        assert_eq!(chosen_1, engine_1);
        assert_eq!(chosen_0_again, engine_0);

        let grouped = registry.abortable_request_ids(&[
            "req-1".to_string(),
            "req-2".to_string(),
            "req-3".to_string(),
        ]);
        assert_eq!(
            grouped.get(&engine_0).unwrap(),
            &vec!["req-1".to_string(), "req-3".to_string()]
        );
        assert_eq!(grouped.get(&engine_1).unwrap(), &vec!["req-2".to_string()]);
    }

    #[test]
    fn registry_uses_inflight_as_waiting_fallback_before_stats_arrive() {
        let engine_0 = EngineId::from_engine_index(0);
        let engine_1 = EngineId::from_engine_index(1);
        let mut registry = RequestRegistry::new(&[
            connected_engine(engine_0.clone()),
            connected_engine(engine_1.clone()),
        ]);

        let (chosen_0, _) =
            registry.register("req-1".to_string(), None, None, NonZeroU32::MIN).unwrap();
        let (chosen_1, _) =
            registry.register("req-2".to_string(), None, None, NonZeroU32::MIN).unwrap();
        let (chosen_0_again, _) =
            registry.register("req-3".to_string(), None, None, NonZeroU32::MIN).unwrap();

        assert_eq!(chosen_0, engine_0);
        assert_eq!(chosen_1, engine_1);
        assert_eq!(chosen_0_again, engine_0);
    }

    #[test]
    fn routing_score_uses_inflight_before_stats_arrive() {
        let state = EngineRoutingState {
            inflight: 3,
            last_scheduler_stats: None,
            ..Default::default()
        };

        assert_eq!(state.routing_score(), 3);
    }

    #[test]
    fn routing_score_uses_inflight_as_scheduler_stats_lower_bound() {
        let state = EngineRoutingState {
            inflight: 7,
            last_scheduler_stats: Some(EngineLoadSnapshot {
                waiting: 0,
                running: 2,
            }),
            ..Default::default()
        };

        assert_eq!(state.routing_score(), 7);
    }

    #[test]
    fn routing_score_counts_waiting_without_extra_penalty() {
        let state = EngineRoutingState {
            inflight: 1,
            last_scheduler_stats: Some(EngineLoadSnapshot {
                waiting: 3,
                running: 2,
            }),
            ..Default::default()
        };

        assert_eq!(state.routing_score(), 5);
    }

    #[test]
    fn registry_prefers_real_scheduler_stats_over_inflight() {
        let engine_0 = EngineId::from_engine_index(0);
        let engine_1 = EngineId::from_engine_index(1);
        let mut registry = RequestRegistry::new(&[
            connected_engine(engine_0.clone()),
            connected_engine(engine_1.clone()),
        ]);

        assert!(registry.apply_scheduler_counts(
            0,
            EngineLoadSnapshot {
                waiting: 3,
                running: 2
            }
        ));
        assert!(registry.apply_scheduler_counts(
            1,
            EngineLoadSnapshot {
                waiting: 0,
                running: 1
            }
        ));

        let (chosen, _) =
            registry.register("req-stats".to_string(), None, None, NonZeroU32::MIN).unwrap();
        assert_eq!(chosen, engine_1);
    }

    #[test]
    fn registry_spreads_bursts_after_cancellation_before_stats_refresh() {
        let mut distributions = Vec::new();
        for counts in [[10, 12], [12, 12]] {
            let mut registry = RequestRegistry::new(
                &[0, 1].map(|rank| connected_engine(EngineId::from_engine_index(rank))),
            );
            let mut old_requests = Vec::new();
            for (rank, running) in counts.into_iter().enumerate() {
                for index in 0..running {
                    let request_id = format!("old-{rank}-{index}");
                    registry
                        .register(request_id.clone(), None, Some(rank as u32), NonZeroU32::MIN)
                        .unwrap();
                    old_requests.push(request_id);
                }
                registry.apply_scheduler_counts(
                    rank as u32,
                    EngineLoadSnapshot {
                        waiting: 0,
                        running,
                    },
                );
            }
            drop(registry.abort_many(&old_requests, 0.0));

            let mut distribution = [0; 2];
            for index in 0..14 {
                let (engine, _) =
                    registry.register(format!("new-{index}"), None, None, NonZeroU32::MIN).unwrap();
                distribution[engine.engine_index().unwrap() as usize] += 1;
            }
            distributions.push(distribution);
        }
        expect_test::expect!["[[8, 6], [7, 7]]"].assert_eq(&format!("{distributions:?}"));
    }

    #[test]
    fn registry_preserves_tie_order_across_finished_requests_and_stats_refreshes() {
        let ranks = [2, 5, 9];
        let mut registry = RequestRegistry::new(
            &ranks.map(|rank| connected_engine(EngineId::from_engine_index(rank))),
        );
        let mut chosen = Vec::new();
        for index in 0..6 {
            for rank in ranks {
                registry.apply_scheduler_counts(
                    u32::from(rank),
                    EngineLoadSnapshot {
                        waiting: 0,
                        running: 0,
                    },
                );
            }
            let request_id = format!("req-{index}");
            let (engine, _) =
                registry.register(request_id.clone(), None, None, NonZeroU32::MIN).unwrap();
            chosen.push(engine.engine_index().unwrap());
            drop(registry.finish_many(&[request_id]));
        }
        expect_test::expect!["[2, 2, 2, 2, 2, 2]"].assert_eq(&format!("{chosen:?}"));
    }

    #[test]
    fn registry_refreshes_estimates_without_erasing_inflight() {
        let mut registry = RequestRegistry::new(
            &[0, 1].map(|rank| connected_engine(EngineId::from_engine_index(rank))),
        );
        let snapshot = EngineLoadSnapshot {
            waiting: 0,
            running: 10,
        };
        registry.apply_scheduler_counts(0, snapshot);
        registry.apply_scheduler_counts(
            1,
            EngineLoadSnapshot {
                waiting: 0,
                running: 12,
            },
        );
        for index in 0..4 {
            registry
                .register(format!("pinned-{index}"), None, Some(0), NonZeroU32::MIN)
                .unwrap();
        }

        // Even unchanged counts replace the optimistic estimate for that engine.
        registry.apply_scheduler_counts(0, snapshot);
        let (after_refresh, _) =
            registry.register("after-refresh".into(), None, None, NonZeroU32::MIN).unwrap();
        // An older, empty snapshot must not erase the five admitted requests.
        for rank in [0, 1] {
            registry.apply_scheduler_counts(
                rank,
                EngineLoadSnapshot {
                    waiting: 0,
                    running: 0,
                },
            );
        }
        let (after_empty, _) =
            registry.register("after-empty".into(), None, None, NonZeroU32::MIN).unwrap();
        expect_test::expect!["(Some(0), Some(1))"].assert_eq(&format!(
            "{:?}",
            (after_refresh.engine_index(), after_empty.engine_index())
        ));
    }

    #[test]
    fn registry_explicit_admissions_raise_routing_score() {
        let mut registry = RequestRegistry::new(
            &[2, 5, 9].map(|rank| connected_engine(EngineId::from_engine_index(rank))),
        );
        for rank in [2, 5, 9] {
            registry.apply_scheduler_counts(
                rank,
                EngineLoadSnapshot {
                    waiting: 0,
                    running: 10,
                },
            );
        }
        registry.register("pinned".into(), None, Some(9), NonZeroU32::MIN).unwrap();
        let mut chosen = Vec::new();
        for index in 0..3 {
            let (engine, _) =
                registry.register(format!("auto-{index}"), None, None, NonZeroU32::MIN).unwrap();
            chosen.push(engine.engine_index().unwrap());
        }
        expect_test::expect!["[2, 5, 2]"].assert_eq(&format!("{chosen:?}"));
    }

    #[test]
    fn register_with_data_parallel_rank_routes_to_specified_engine() {
        let engine_0 = EngineId::from_engine_index(0);
        let engine_1 = EngineId::from_engine_index(1);
        let engine_2 = EngineId::from_engine_index(2);
        let mut registry = RequestRegistry::new(&[
            connected_engine(engine_0.clone()),
            connected_engine(engine_1.clone()),
            connected_engine(engine_2.clone()),
        ]);

        // Explicitly target rank 2 (third engine).
        let (chosen, _) =
            registry.register("req-1".to_string(), None, Some(2), NonZeroU32::MIN).unwrap();
        assert_eq!(chosen, engine_2);

        // Explicitly target rank 0 (first engine).
        let (chosen, _) =
            registry.register("req-2".to_string(), None, Some(0), NonZeroU32::MIN).unwrap();
        assert_eq!(chosen, engine_0);

        // Explicitly target rank 1.
        let (chosen, _) =
            registry.register("req-3".to_string(), None, Some(1), NonZeroU32::MIN).unwrap();
        assert_eq!(chosen, engine_1);
    }

    #[test]
    fn register_with_data_parallel_rank_bypasses_load_balancing() {
        let engine_0 = EngineId::from_engine_index(0);
        let engine_1 = EngineId::from_engine_index(1);
        let mut registry = RequestRegistry::new(&[
            connected_engine(engine_0.clone()),
            connected_engine(engine_1.clone()),
        ]);

        // Load-balance: first two go to engine_0 and engine_1.
        registry.register("req-lb-0".to_string(), None, None, NonZeroU32::MIN).unwrap();

        // Now engine_0 has 1 in-flight. Without dp_rank, next would go to engine_1.
        // But with dp_rank=0, it should still go to engine_0.
        let (chosen, _) =
            registry.register("req-dp".to_string(), None, Some(0), NonZeroU32::MIN).unwrap();
        assert_eq!(chosen, engine_0);
    }

    #[test]
    fn register_with_out_of_range_rank_returns_error() {
        let mut registry = RequestRegistry::new(&[
            connected_engine(EngineId::from_engine_index(0)),
            connected_engine(EngineId::from_engine_index(1)),
        ]);

        let error = registry
            .register("req-1".to_string(), None, Some(2), NonZeroU32::MIN)
            .unwrap_err();
        assert!(matches!(
            error,
            crate::error::Error::InvalidDataParallelRank {
                rank: 2,
                connected_ranks,
            } if connected_ranks == vec![0, 1]
        ));
    }

    #[test]
    fn register_with_rank_uses_global_engine_identity() {
        let engine_3 = EngineId::from_engine_index(3);
        let mut registry = RequestRegistry::new(&[connected_engine(engine_3.clone())]);

        let (chosen, _) =
            registry.register("req-ok".to_string(), None, Some(3), NonZeroU32::MIN).unwrap();
        assert_eq!(chosen, engine_3);

        let error = registry
            .register("req-bad".to_string(), None, Some(0), NonZeroU32::MIN)
            .unwrap_err();
        assert!(matches!(
            error,
            crate::error::Error::InvalidDataParallelRank {
                rank: 0,
                connected_ranks,
            } if connected_ranks == vec![3]
        ));
    }

    #[test]
    fn utility_registry_tracks_and_removes_call_ids() {
        let mut registry = UtilityRegistry::default();
        let (call_id_1, _) = registry.allocate_and_register();
        let (call_id_2, _) = registry.allocate_and_register();

        assert_eq!(call_id_1, 1);
        assert_eq!(call_id_2, 2);
        assert!(registry.contains(1));
        assert!(registry.contains(2));
        assert!(registry.resolve(&1).is_some());
        assert!(!registry.contains(1));
        assert!(registry.contains(2));
    }

    #[test]
    fn utility_registry_closes_all_waiters_on_failure() {
        let mut registry = UtilityRegistry::default();
        registry.allocate_and_register();
        registry.allocate_and_register();

        let senders = registry.close();

        assert_eq!(senders.len(), 2);
        assert!(!registry.contains(1));
        assert!(!registry.contains(2));
        assert!(registry.is_closed());
    }

    #[test]
    fn utility_registry_unregister_many_drops_pending_calls() {
        use tokio::sync::oneshot::error::TryRecvError;

        let mut registry = UtilityRegistry::default();
        let (call_id_1, mut rx_1) = registry.allocate_and_register();
        let (call_id_2, mut rx_2) = registry.allocate_and_register();
        let (call_id_3, _rx_3) = registry.allocate_and_register();

        // Drop two of the three allocated calls; the third stays pending.
        registry.unregister_many([call_id_1, call_id_2]);

        assert!(!registry.contains(call_id_1));
        assert!(!registry.contains(call_id_2));
        assert!(registry.contains(call_id_3));
        // The receivers must observe the sender being dropped (channel closed).
        assert!(matches!(rx_1.try_recv(), Err(TryRecvError::Closed)));
        assert!(matches!(rx_2.try_recv(), Err(TryRecvError::Closed)));
    }

    #[test]
    fn utility_registry_unregister_many_ignores_unknown_call_ids() {
        let mut registry = UtilityRegistry::default();
        let (call_id, _rx) = registry.allocate_and_register();

        // Unknown call ids are silently ignored — caller doesn't care which were live.
        registry.unregister_many([call_id, 42, 9999]);

        assert!(!registry.contains(call_id));
    }
}
