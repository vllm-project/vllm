use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::sync::{mpsc, oneshot};
use tracing::trace;

use crate::EngineId;
use crate::client::stream::EngineCoreStreamOutput;
use crate::error::{Error, Result};
use crate::protocol::stats::SchedulerStats;
use crate::protocol::utility::UtilityOutput;
use crate::protocol::{EngineCoreEventType, EngineCoreFinishReason, EngineCoreOutput};
use crate::transport::ConnectedEngine;

pub type OutputSender = mpsc::UnboundedSender<Result<EngineCoreStreamOutput>>;
pub type OutputReceiver = mpsc::UnboundedReceiver<Result<EngineCoreStreamOutput>>;
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
}

impl EngineRoutingState {
    /// Compute the routing score used to pick the least-loaded engine.
    ///
    /// Scheduler stats can raise the load estimate above the frontend-local
    /// view, but they should not lower it below requests this frontend has
    /// already admitted. Waiting requests still get the same extra penalty
    /// as the original `waiting * 4 + running` score.
    fn routing_score(&self) -> usize {
        const WAITING_WEIGHT: usize = 4;

        let Some(stats) = self.last_scheduler_stats else {
            return self.inflight;
        };

        let scheduler_total = stats.running + stats.waiting;
        self.inflight.max(scheduler_total) + stats.waiting * (WAITING_WEIGHT - 1)
    }

    /// Replace the local routing view with a fresh real scheduler snapshot.
    fn apply_scheduler_counts(&mut self, next: EngineLoadSnapshot) {
        self.last_scheduler_stats = Some(next);
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
    routing_per_engine: BTreeMap<EngineId, EngineRoutingState>,
}

impl RequestRegistry {
    pub fn new(engines: &[ConnectedEngine]) -> Self {
        Self {
            closed: false,
            requests: HashMap::default(),
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
    /// the engine with the fewest in-flight requests is chosen.
    pub fn register(
        &mut self,
        request_id: String,
        lora_name: Option<String>,
        data_parallel_rank: Option<u32>,
    ) -> Result<(EngineId, OutputReceiver)> {
        if self.requests.contains_key(&request_id) {
            return Err(Error::DuplicateRequestId { request_id });
        }

        let engine_id = self.choose_engine_for_request(data_parallel_rank)?;
        let (tx, rx) = mpsc::unbounded_channel();
        self.requests.insert(
            request_id,
            TrackedRequest {
                sender: tx,
                engine_id: engine_id.clone(),
                lora: lora_name.map(|adapter_name| LoraRequestState {
                    adapter_name,
                    phase: LoraPhase::Waiting,
                }),
            },
        );

        let state = self
            .routing_per_engine
            .get_mut(&engine_id)
            .expect("request registry must track all known engines");
        state.inflight += 1;

        Ok((engine_id, rx))
    }

    fn choose_engine_for_request(&mut self, data_parallel_rank: Option<u32>) -> Result<EngineId> {
        if let Some(rank) = data_parallel_rank {
            // Route to the engine at the specified rank index.
            let engine_id = EngineId::from_engine_index(rank);
            return self
                .routing_per_engine
                .contains_key(&engine_id)
                .then_some(engine_id)
                .ok_or_else(|| Error::InvalidDataParallelRank {
                    rank,
                    num_engines: self.routing_per_engine.len() as u32,
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

    /// Obtain the stream sender for one output. If it indicates the request is
    /// finished, it will be removed from the registry.
    pub fn sender_for_output(&mut self, output: &EngineCoreOutput) -> Option<OutputSender> {
        self.apply_lora_events(output);
        if output.finished() {
            self.remove(output.request_id.as_str()).map(|tracked| tracked.0)
        } else {
            self.requests
                .get(output.request_id.as_str())
                .map(|tracked| tracked.sender.clone())
        }
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

    /// Obtain stream senders for a whole engine output batch under one
    /// registry lock. Finished outputs are removed before returning.
    pub fn senders_for_outputs<'a>(
        &mut self,
        outputs: impl IntoIterator<Item = &'a EngineCoreOutput>,
    ) -> Vec<Option<OutputSender>> {
        outputs.into_iter().map(|output| self.sender_for_output(output)).collect()
    }

    /// Remove a batch of requests that have finished or aborted, returning
    /// their stream senders.
    pub fn finish_many<'a>(
        &mut self,
        request_ids: impl IntoIterator<Item = &'a String>,
    ) -> Vec<OutputSender> {
        request_ids
            .into_iter()
            .filter_map(|request_id| self.remove(request_id.as_str()).map(|tracked| tracked.0))
            .collect()
    }

    /// Apply one scheduler stats update for the given engine to the local
    /// routing state. Returns `false` if the engine is unknown to the
    /// client.
    pub fn apply_scheduler_stats(&mut self, engine_index: u32, stats: &SchedulerStats) -> bool {
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
            let Some((sender, engine_id)) = self.remove(request_id) else {
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
            let _ = sender.send(Ok(output));
            aborted.push(request_id.clone());
        }
        aborted
    }

    /// Remove one request from the local registry. Returns the tracked entry if
    /// it exists.
    #[must_use]
    pub fn remove(&mut self, request_id: &str) -> Option<(OutputSender, EngineId)> {
        let tracked = self.requests.remove(request_id)?;
        self.routing_per_engine
            .get_mut(&tracked.engine_id)
            .expect("request registry must track all known engines")
            .inflight -= 1;
        Some((tracked.sender, tracked.engine_id))
    }

    fn apply_scheduler_counts(&mut self, engine_index: u32, next: EngineLoadSnapshot) -> bool {
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

    pub fn is_closed(&self) -> bool {
        self.closed
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crate::EngineId;
    use crate::client::state::{
        EngineLoadSnapshot, EngineRoutingState, RequestRegistry, UtilityRegistry,
    };
    use crate::mock_engine::default_ready_response;
    use crate::protocol::{
        EngineCoreEvent, EngineCoreEventType, EngineCoreFinishReason, EngineCoreOutput,
    };
    use crate::transport::ConnectedEngine;

    fn connected_engine(engine_id: EngineId) -> ConnectedEngine {
        ConnectedEngine {
            engine_id,
            ready_response: default_ready_response(),
        }
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

    fn adapter_names(values: &[&str]) -> BTreeSet<String> {
        values.iter().map(|name| (*name).to_string()).collect()
    }

    #[test]
    fn registry_rejects_duplicate_request_ids() {
        let mut registry = RequestRegistry::new(&[connected_engine(EngineId::from(b"engine-0"))]);
        registry.register("req-1".to_string(), None, None).unwrap();
        let error = registry.register("req-1".to_string(), None, None).unwrap_err();
        assert!(matches!(
            error,
            crate::error::Error::DuplicateRequestId { request_id } if request_id == "req-1"
        ));
    }

    #[test]
    fn registry_removes_finished_request_on_output() {
        let mut registry = RequestRegistry::new(&[connected_engine(EngineId::from(b"engine-0"))]);
        registry.register("req-1".to_string(), None, None).unwrap();

        let sender = registry.sender_for_output(&EngineCoreOutput {
            request_id: "req-1".to_string(),
            finish_reason: Some(EngineCoreFinishReason::Length),
            ..Default::default()
        });

        assert!(sender.is_some());
        assert!(!registry.contains("req-1"));
    }

    #[test]
    fn registry_tracks_lora_phases_from_engine_events() {
        let mut registry = RequestRegistry::new(&[connected_engine(EngineId::from(b"engine-0"))]);
        registry
            .register("req-lora".to_string(), Some("adapter-a".to_string()), None)
            .unwrap();
        registry.register("req-plain".to_string(), None, None).unwrap();

        // Registered but not yet scheduled: counted as waiting. The non-LoRA
        // request never shows up.
        assert_eq!(
            registry.lora_adapter_states(),
            (adapter_names(&[]), adapter_names(&["adapter-a"]))
        );

        // Queued then scheduled in one output: running.
        drop(registry.sender_for_output(&output_with_events(
            "req-lora",
            &[EngineCoreEventType::Queued, EngineCoreEventType::Scheduled],
            None,
        )));
        assert_eq!(
            registry.lora_adapter_states(),
            (adapter_names(&["adapter-a"]), adapter_names(&[]))
        );

        // Preempted: back to waiting.
        drop(registry.sender_for_output(&output_with_events(
            "req-lora",
            &[EngineCoreEventType::Preempted],
            None,
        )));
        assert_eq!(
            registry.lora_adapter_states(),
            (adapter_names(&[]), adapter_names(&["adapter-a"]))
        );

        // Finished: dropped from tracking entirely.
        drop(registry.sender_for_output(&output_with_events(
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
            .register("req-a1".to_string(), Some("adapter-a".to_string()), None)
            .unwrap();
        registry
            .register("req-a2".to_string(), Some("adapter-a".to_string()), None)
            .unwrap();
        registry
            .register("req-b".to_string(), Some("adapter-b".to_string()), None)
            .unwrap();

        // One of adapter-a's requests starts running while the other waits:
        // the adapter appears in both sets.
        drop(registry.sender_for_output(&output_with_events(
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
    fn registry_drops_lora_tracking_on_abort() {
        let mut registry = RequestRegistry::new(&[connected_engine(EngineId::from(b"engine-0"))]);
        registry
            .register("req-lora".to_string(), Some("adapter-a".to_string()), None)
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
        registry.register("req-1".to_string(), None, None).unwrap();
        registry.register("req-2".to_string(), None, None).unwrap();

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
        let (chosen_0, _) = registry.register("req-1".to_string(), None, None).unwrap();
        let (chosen_1, _) = registry.register("req-2".to_string(), None, None).unwrap();
        let (chosen_0_again, _) = registry.register("req-3".to_string(), None, None).unwrap();

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

        let (chosen_0, _) = registry.register("req-1".to_string(), None, None).unwrap();
        let (chosen_1, _) = registry.register("req-2".to_string(), None, None).unwrap();
        let (chosen_0_again, _) = registry.register("req-3".to_string(), None, None).unwrap();

        assert_eq!(chosen_0, engine_0);
        assert_eq!(chosen_1, engine_1);
        assert_eq!(chosen_0_again, engine_0);
    }

    #[test]
    fn routing_score_uses_inflight_before_stats_arrive() {
        let state = EngineRoutingState {
            inflight: 3,
            last_scheduler_stats: None,
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
        };

        assert_eq!(state.routing_score(), 7);
    }

    #[test]
    fn routing_score_keeps_extra_waiting_penalty() {
        let state = EngineRoutingState {
            inflight: 1,
            last_scheduler_stats: Some(EngineLoadSnapshot {
                waiting: 3,
                running: 2,
            }),
        };

        assert_eq!(state.routing_score(), 14);
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

        let (chosen, _) = registry.register("req-stats".to_string(), None, None).unwrap();
        assert_eq!(chosen, engine_1);
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
        let (chosen, _) = registry.register("req-1".to_string(), None, Some(2)).unwrap();
        assert_eq!(chosen, engine_2);

        // Explicitly target rank 0 (first engine).
        let (chosen, _) = registry.register("req-2".to_string(), None, Some(0)).unwrap();
        assert_eq!(chosen, engine_0);

        // Explicitly target rank 1.
        let (chosen, _) = registry.register("req-3".to_string(), None, Some(1)).unwrap();
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
        registry.register("req-lb-0".to_string(), None, None).unwrap();

        // Now engine_0 has 1 in-flight. Without dp_rank, next would go to engine_1.
        // But with dp_rank=0, it should still go to engine_0.
        let (chosen, _) = registry.register("req-dp".to_string(), None, Some(0)).unwrap();
        assert_eq!(chosen, engine_0);
    }

    #[test]
    fn register_with_out_of_range_rank_returns_error() {
        let mut registry = RequestRegistry::new(&[
            connected_engine(EngineId::from_engine_index(0)),
            connected_engine(EngineId::from_engine_index(1)),
        ]);

        let error = registry.register("req-1".to_string(), None, Some(2)).unwrap_err();
        assert!(matches!(
            error,
            crate::error::Error::InvalidDataParallelRank {
                rank: 2,
                num_engines: 2,
            }
        ));
    }

    #[test]
    fn register_with_rank_on_single_engine_only_accepts_zero() {
        let engine_0 = EngineId::from_engine_index(0);
        let mut registry = RequestRegistry::new(&[connected_engine(engine_0.clone())]);

        let (chosen, _) = registry.register("req-ok".to_string(), None, Some(0)).unwrap();
        assert_eq!(chosen, engine_0);

        let error = registry.register("req-bad".to_string(), None, Some(1)).unwrap_err();
        assert!(matches!(
            error,
            crate::error::Error::InvalidDataParallelRank {
                rank: 1,
                num_engines: 1,
            }
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
