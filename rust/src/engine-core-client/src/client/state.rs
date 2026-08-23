// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::sync::{mpsc, oneshot};
use tracing::trace;

use crate::EngineId;
use crate::client::stream::EngineCoreStreamOutput;
use crate::error::{Error, Result};
use crate::protocol::output::{EngineCoreEventType, EngineCoreFinishReason, EngineCoreOutput};
use crate::protocol::stats::SchedulerStats;
use crate::protocol::utility::UtilityOutput;
use crate::transport::ConnectedEngine;

/// Internal stream message: `Ok(Some(_))` carries output and `Ok(None)` marks
/// explicit clean completion.
pub type OutputMessage = Result<Option<EngineCoreStreamOutput>>;
pub type OutputSender = mpsc::UnboundedSender<OutputMessage>;
pub type OutputReceiver = mpsc::UnboundedReceiver<OutputMessage>;
pub type UtilitySender = oneshot::Sender<Result<UtilityOutput>>;
pub type UtilityReceiver = oneshot::Receiver<Result<UtilityOutput>>;

/// Where one engine output goes and any resumable accounting deferred until
/// after that output is enqueued.
#[derive(Debug)]
pub struct OutputRoute {
    pub sender: OutputSender,
    pub stop_action: Option<ResumableStopAction>,
}

/// What a terminal output means for a resumable session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResumableStopAction {
    Continue,
    End,
}

/// How a tracked request reaches its end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequestMode {
    /// An ordinary call, retired by its first terminal output.
    OneShot,
    Resumable(ResumableSession),
}

/// Segment accounting for one resumable request.
///
/// The engine stops the request once per submitted segment, so the session is
/// over when the closing sentinel ADD has been sent and every segment has
/// stopped. This mirrors the Python frontend's `input_chunk_queue` bookkeeping
/// in `OutputProcessor.process_outputs`. `EngineCoreOutputs.finished_requests`
/// cannot drive completion instead: the engine only populates it under
/// data-parallel internal load balancing (`include_finished_set`), so a
/// single-engine deployment never reports it for a normal finish.
/// [`RequestRegistry::finish_many`] therefore leaves resumable IDs registered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InFlightAdd {
    Content,
    Sentinel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ResumableSession {
    outstanding: u32,
    sentinel_sent: bool,
    /// At most one content or sentinel ADD can be in flight at a time.
    in_flight: Option<InFlightAdd>,
}

impl ResumableSession {
    fn open() -> Self {
        Self {
            outstanding: 1,
            sentinel_sent: false,
            in_flight: None,
        }
    }

    fn is_complete(&self) -> bool {
        self.sentinel_sent && self.outstanding == 0
    }
}

fn resumable_stop_action(output: &EngineCoreOutput) -> ResumableStopAction {
    if matches!(
        output.finish_reason,
        Some(EngineCoreFinishReason::Abort | EngineCoreFinishReason::Error)
    ) {
        ResumableStopAction::End
    } else {
        ResumableStopAction::Continue
    }
}

#[derive(Debug)]
struct TrackedRequest {
    sender: OutputSender,
    engine_id: EngineId,
    lora: Option<LoraRequestState>,
    mode: RequestMode,
}

impl TrackedRequest {
    fn is_resumable(&self) -> bool {
        matches!(self.mode, RequestMode::Resumable(_))
    }

    fn session_mut(&mut self) -> Option<&mut ResumableSession> {
        match &mut self.mode {
            RequestMode::OneShot => None,
            RequestMode::Resumable(session) => Some(session),
        }
    }
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
    /// already admitted.
    fn routing_score(&self) -> usize {
        let Some(stats) = self.last_scheduler_stats else {
            return self.inflight;
        };

        self.inflight.max(stats.running + stats.waiting)
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
    /// the engine with the fewest in-flight requests is chosen.
    ///
    /// Set `resumable` so later [`Self::continue_resumable`] can reuse this
    /// entry. Duplicate `request_id`s are rejected — continuations join an
    /// existing session, they do not register a new one.
    pub fn register(
        &mut self,
        request_id: String,
        lora_name: Option<String>,
        data_parallel_rank: Option<u32>,
        resumable: bool,
    ) -> Result<(EngineId, OutputReceiver)> {
        if self.requests.contains_key(&request_id) {
            return Err(Error::DuplicateRequestId { request_id });
        }

        let engine_id = self.choose_engine_for_request(data_parallel_rank)?;
        let (tx, rx) = mpsc::unbounded_channel();
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
                mode: if resumable {
                    RequestMode::Resumable(ResumableSession::open())
                } else {
                    RequestMode::OneShot
                },
            },
        );

        let state = self
            .routing_per_engine
            .get_mut(&engine_id)
            .expect("request registry must track all known engines");
        state.inflight += 1;

        Ok((engine_id, rx))
    }

    /// Reuse an open resumable session for one continuation ADD.
    ///
    /// A content ADD counts one more segment the engine still has to stop. A
    /// sentinel ADD does not. Concurrent ADDs are rejected while this one is
    /// in flight.
    pub fn continue_resumable(&mut self, request_id: &str, sentinel: bool) -> Result<EngineId> {
        let unknown = || Error::UnknownResumableRequestId {
            request_id: request_id.to_string(),
        };
        let existing = self.requests.get_mut(request_id).ok_or_else(unknown)?;
        let engine_id = existing.engine_id.clone();
        let session = existing.session_mut().ok_or_else(unknown)?;
        if session.in_flight.is_some() {
            return Err(Error::ContinuationInProgress {
                request_id: request_id.to_string(),
            });
        }
        if session.sentinel_sent {
            return Err(unknown());
        }

        if sentinel {
            session.in_flight = Some(InFlightAdd::Sentinel);
        } else {
            session.outstanding += 1;
            session.in_flight = Some(InFlightAdd::Content);
        }
        Ok(engine_id)
    }

    /// Undo the accounting for a continuation whose ADD could not be sent.
    pub fn rollback_continuation(&mut self, request_id: &str) {
        let Some(session) = self.requests.get_mut(request_id).and_then(TrackedRequest::session_mut)
        else {
            return;
        };
        if session.in_flight.take() == Some(InFlightAdd::Content) {
            session.outstanding = session.outstanding.saturating_sub(1);
        }
    }

    /// Mark a continuation ADD as sent. A sentinel ADD also commits
    /// `sentinel_sent` and, if every segment has already stopped, completes
    /// the session.
    ///
    /// Returns `false` if the session was already removed.
    pub fn commit_continuation(&mut self, request_id: &str) -> bool {
        let complete = {
            let Some(session) =
                self.requests.get_mut(request_id).and_then(TrackedRequest::session_mut)
            else {
                return false;
            };
            match session.in_flight.take() {
                None => return true,
                Some(InFlightAdd::Sentinel) => session.sentinel_sent = true,
                Some(InFlightAdd::Content) => {}
            }
            session.is_complete()
        };
        if !complete {
            return true;
        }
        if let Some((sender, _)) = self.remove(request_id) {
            let _ = sender.send(Ok(None));
        }
        true
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

    /// Route one output to its stream.
    ///
    /// Resumable terminal accounting is deferred until
    /// [`Self::finish_resumable_output`] so the dispatcher can enqueue this
    /// output before a concurrent sentinel completes its stream.
    pub fn route_for_output(&mut self, output: &EngineCoreOutput) -> Option<OutputRoute> {
        self.apply_lora_events(output);

        let request_id = output.request_id.as_str();
        let resumable = self.requests.get(request_id)?.is_resumable();

        if output.finished() && !resumable {
            self.remove(request_id).map(|(sender, _)| OutputRoute {
                sender,
                stop_action: None,
            })
        } else {
            self.requests.get(request_id).map(|tracked| OutputRoute {
                sender: tracked.sender.clone(),
                stop_action: (resumable && output.finished())
                    .then(|| resumable_stop_action(output)),
            })
        }
    }

    /// Account for a terminal resumable output after it has been enqueued.
    ///
    /// Returns the sender that should receive explicit completion when this was
    /// the session's last output.
    pub fn finish_resumable_output(
        &mut self,
        request_id: &str,
        stop_action: ResumableStopAction,
    ) -> Option<(OutputSender, EngineId)> {
        let retire = {
            let session =
                self.requests.get_mut(request_id).and_then(TrackedRequest::session_mut)?;
            session.outstanding = session.outstanding.saturating_sub(1);
            session.is_complete() || stop_action == ResumableStopAction::End
        };
        if !retire {
            return None;
        }
        self.remove(request_id)
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

    /// Obtain stream routes for a whole engine output batch under one registry
    /// lock. Finished one-shot outputs are removed before returning.
    pub fn routes_for_outputs<'a>(
        &mut self,
        outputs: impl IntoIterator<Item = &'a EngineCoreOutput>,
    ) -> Vec<Option<OutputRoute>> {
        outputs.into_iter().map(|output| self.route_for_output(output)).collect()
    }

    /// Remove one-shot requests the engine marked finished, returning their
    /// stream senders.
    ///
    /// Skips resumable IDs; see [`ResumableSession`].
    pub fn finish_many<'a>(
        &mut self,
        request_ids: impl IntoIterator<Item = &'a String>,
    ) -> Vec<OutputSender> {
        request_ids
            .into_iter()
            .filter_map(|request_id| {
                let tracked = self.requests.get(request_id.as_str())?;
                if tracked.is_resumable() {
                    return None;
                }
                self.remove(request_id.as_str()).map(|(sender, _)| sender)
            })
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
            let _ = sender.send(Ok(Some(output)));
            let _ = sender.send(Ok(None));
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

    pub fn is_closed(&self) -> bool {
        self.closed
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crate::EngineId;
    use crate::client::state::{
        EngineLoadSnapshot, EngineRoutingState, RequestRegistry, ResumableStopAction,
        UtilityRegistry,
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

    fn segment_stop(request_id: &str) -> EngineCoreOutput {
        EngineCoreOutput {
            request_id: request_id.to_string(),
            finish_reason: Some(EngineCoreFinishReason::Stop),
            ..Default::default()
        }
    }

    fn admit_content(registry: &mut RequestRegistry, request_id: &str) -> EngineId {
        let engine_id = registry.continue_resumable(request_id, false).unwrap();
        assert!(registry.commit_continuation(request_id));
        engine_id
    }

    fn admit_close(registry: &mut RequestRegistry, request_id: &str) {
        registry.continue_resumable(request_id, true).unwrap();
        assert!(registry.commit_continuation(request_id));
    }

    #[test]
    fn registry_rejects_duplicate_request_ids() {
        let mut registry = RequestRegistry::new(&[connected_engine(EngineId::from(b"engine-0"))]);
        registry.register("req-1".to_string(), None, None, false).unwrap();
        let error = registry.register("req-1".to_string(), None, None, false).unwrap_err();
        assert!(matches!(
            error,
            crate::error::Error::DuplicateRequestId { request_id } if request_id == "req-1"
        ));
    }

    #[test]
    fn resumable_entry_survives_a_segment_finish_reason() {
        let mut registry = RequestRegistry::new(&[connected_engine(EngineId::from(b"engine-0"))]);
        registry.register("rt-abc".to_string(), None, None, true).unwrap();

        let route = registry
            .route_for_output(&segment_stop("rt-abc"))
            .expect("a segment stop still routes to the open stream");
        assert_eq!(route.stop_action, Some(ResumableStopAction::Continue));
        assert!(registry.finish_resumable_output("rt-abc", route.stop_action.unwrap()).is_none());
        assert!(
            registry.contains("rt-abc"),
            "a segment finish reason is not the session's end"
        );
        drop(route);
    }

    #[test]
    fn finish_many_skips_resumable_requests() {
        let mut registry = RequestRegistry::new(&[connected_engine(EngineId::from(b"engine-0"))]);
        registry.register("rt-abc".to_string(), None, None, true).unwrap();
        registry.register("one-shot".to_string(), None, None, false).unwrap();

        let finished = registry.finish_many(&["rt-abc".to_string(), "one-shot".to_string()]);
        assert_eq!(finished.len(), 1);
        assert!(
            registry.contains("rt-abc"),
            "finished_requests must not retire a resumable session"
        );
        assert!(!registry.contains("one-shot"));
    }

    #[test]
    fn completion_does_not_overtake_an_already_routed_output() {
        let mut registry = RequestRegistry::new(&[connected_engine(EngineId::from(b"engine-0"))]);
        let (_, mut rx) = registry.register("rt-abc".to_string(), None, None, true).unwrap();
        let output = segment_stop("rt-abc");

        // The dispatcher has decided where the output goes but has not put it
        // on the channel yet.
        let route = registry
            .route_for_output(&output)
            .expect("the segment stop routes to the open stream");
        assert_eq!(route.stop_action, Some(ResumableStopAction::Continue));

        registry.continue_resumable("rt-abc", true).unwrap();
        assert!(registry.commit_continuation("rt-abc"));
        route
            .sender
            .send(Ok(Some(EngineCoreStreamOutput {
                engine_index: 0,
                timestamp: 0.0,
                output,
            })))
            .unwrap();
        let completion = registry
            .finish_resumable_output("rt-abc", route.stop_action.unwrap())
            .expect("the committed sentinel completes the session");
        completion.0.send(Ok(None)).unwrap();

        assert!(
            matches!(rx.try_recv(), Ok(Ok(Some(_)))),
            "the routed output must precede completion"
        );
        assert!(matches!(rx.try_recv(), Ok(Ok(None))));
    }

    #[test]
    fn resumable_session_ends_on_a_stop_it_cannot_resume_from() {
        let mut registry = RequestRegistry::new(&[connected_engine(EngineId::from(b"engine-0"))]);
        registry.register("rt-abc".to_string(), None, None, true).unwrap();

        let route = registry
            .route_for_output(&EngineCoreOutput {
                request_id: "rt-abc".to_string(),
                finish_reason: Some(EngineCoreFinishReason::Error),
                ..Default::default()
            })
            .expect("the error still reaches the stream");
        assert_eq!(route.stop_action, Some(ResumableStopAction::End));
        assert!(registry.finish_resumable_output("rt-abc", route.stop_action.unwrap()).is_some());
        assert!(!registry.contains("rt-abc"));
    }

    #[test]
    fn continuation_without_an_open_session_is_rejected() {
        let mut registry = RequestRegistry::new(&[connected_engine(EngineId::from(b"engine-0"))]);

        let error = registry.continue_resumable("rt-abc", false).unwrap_err();
        assert!(matches!(
            error,
            crate::error::Error::UnknownResumableRequestId { request_id } if request_id == "rt-abc"
        ));

        // One-shot requests are not continuable.
        registry.register("one-shot".to_string(), None, None, false).unwrap();
        let error = registry.continue_resumable("one-shot", false).unwrap_err();
        assert!(matches!(
            error,
            crate::error::Error::UnknownResumableRequestId { .. }
        ));
    }

    #[test]
    fn closing_session_rejects_later_continuations() {
        let mut registry = RequestRegistry::new(&[connected_engine(EngineId::from(b"engine-0"))]);
        registry.register("rt-abc".to_string(), None, None, true).unwrap();

        admit_close(&mut registry, "rt-abc");
        let error = registry.continue_resumable("rt-abc", false).unwrap_err();
        assert!(matches!(
            error,
            crate::error::Error::UnknownResumableRequestId { .. }
        ));
    }

    #[test]
    fn concurrent_continuation_is_rejected_until_the_first_finishes() {
        let mut registry = RequestRegistry::new(&[connected_engine(EngineId::from(b"engine-0"))]);
        registry.register("rt-abc".to_string(), None, None, true).unwrap();

        registry.continue_resumable("rt-abc", false).unwrap();
        let error = registry.continue_resumable("rt-abc", false).unwrap_err();
        assert!(matches!(
            error,
            crate::error::Error::ContinuationInProgress { request_id }
                if request_id == "rt-abc"
        ));

        registry.rollback_continuation("rt-abc");
        admit_content(&mut registry, "rt-abc");
    }

    #[test]
    fn commit_reports_a_session_retired_by_abort() {
        let mut registry = RequestRegistry::new(&[connected_engine(EngineId::from(b"engine-0"))]);
        registry.register("rt-abc".to_string(), None, None, true).unwrap();
        registry.continue_resumable("rt-abc", true).unwrap();

        registry.abort_many(&["rt-abc".to_string()], 0.0);

        assert!(!registry.commit_continuation("rt-abc"));
    }

    /// A rolled-back sentinel reopens the session even when nothing is left
    /// outstanding, which is the case a premature completion would have closed.
    #[test]
    fn failed_closing_send_after_last_stop_reopens_session() {
        let mut registry = RequestRegistry::new(&[connected_engine(EngineId::from(b"engine-0"))]);
        registry.register("rt-abc".to_string(), None, None, true).unwrap();

        registry.continue_resumable("rt-abc", true).unwrap();
        let route = registry
            .route_for_output(&segment_stop("rt-abc"))
            .expect("the in-flight stop still routes to the stream");
        assert_eq!(route.stop_action, Some(ResumableStopAction::Continue));
        assert!(registry.finish_resumable_output("rt-abc", route.stop_action.unwrap()).is_none());

        registry.rollback_continuation("rt-abc");
        assert!(registry.contains("rt-abc"));
        admit_content(&mut registry, "rt-abc");
    }

    #[test]
    fn failed_continuation_send_does_not_consume_a_segment() {
        let mut registry = RequestRegistry::new(&[connected_engine(EngineId::from(b"engine-0"))]);
        registry.register("rt-abc".to_string(), None, None, true).unwrap();

        registry.continue_resumable("rt-abc", false).unwrap();
        registry.rollback_continuation("rt-abc");
        admit_close(&mut registry, "rt-abc");

        let route = registry
            .route_for_output(&segment_stop("rt-abc"))
            .expect("the remaining segment routes to the stream it closes");
        assert_eq!(route.stop_action, Some(ResumableStopAction::Continue));
        assert!(
            registry.finish_resumable_output("rt-abc", route.stop_action.unwrap()).is_some(),
            "a segment that was never sent must not hold the session open"
        );
    }

    #[test]
    fn registry_removes_finished_request_on_output() {
        let mut registry = RequestRegistry::new(&[connected_engine(EngineId::from(b"engine-0"))]);
        registry.register("req-1".to_string(), None, None, false).unwrap();

        let route = registry.route_for_output(&EngineCoreOutput {
            request_id: "req-1".to_string(),
            finish_reason: Some(EngineCoreFinishReason::Length),
            ..Default::default()
        });

        assert!(route.is_some());
        assert!(!registry.contains("req-1"));
    }

    #[test]
    fn registry_tracks_lora_phases_from_engine_events() {
        let mut registry = RequestRegistry::new(&[connected_engine(EngineId::from(b"engine-0"))]);
        registry
            .register(
                "req-lora".to_string(),
                Some("adapter-a".to_string()),
                None,
                false,
            )
            .unwrap();
        registry.register("req-plain".to_string(), None, None, false).unwrap();

        // Registered but not yet scheduled: counted as waiting. The non-LoRA
        // request never shows up.
        assert_eq!(
            registry.lora_adapter_states(),
            (adapter_names(&[]), adapter_names(&["adapter-a"]))
        );

        // Queued then scheduled in one output: running.
        drop(registry.route_for_output(&output_with_events(
            "req-lora",
            &[EngineCoreEventType::Queued, EngineCoreEventType::Scheduled],
            None,
        )));
        assert_eq!(
            registry.lora_adapter_states(),
            (adapter_names(&["adapter-a"]), adapter_names(&[]))
        );

        // Preempted: back to waiting.
        drop(registry.route_for_output(&output_with_events(
            "req-lora",
            &[EngineCoreEventType::Preempted],
            None,
        )));
        assert_eq!(
            registry.lora_adapter_states(),
            (adapter_names(&[]), adapter_names(&["adapter-a"]))
        );

        // Finished: dropped from tracking entirely.
        drop(registry.route_for_output(&output_with_events(
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
                false,
            )
            .unwrap();
        registry
            .register(
                "req-a2".to_string(),
                Some("adapter-a".to_string()),
                None,
                false,
            )
            .unwrap();
        registry
            .register(
                "req-b".to_string(),
                Some("adapter-b".to_string()),
                None,
                false,
            )
            .unwrap();

        // One of adapter-a's requests starts running while the other waits:
        // the adapter appears in both sets.
        drop(registry.route_for_output(&output_with_events(
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

        registry.register("req-plain".to_string(), None, None, false).unwrap();
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
                false,
            )
            .unwrap();
        registry
            .register(
                "req-lora-b".to_string(),
                Some("adapter-b".to_string()),
                None,
                false,
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
                false,
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
                false,
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
        registry.register("req-1".to_string(), None, None, false).unwrap();
        registry.register("req-2".to_string(), None, None, false).unwrap();

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
        let (chosen_0, _) = registry.register("req-1".to_string(), None, None, false).unwrap();
        let (chosen_1, _) = registry.register("req-2".to_string(), None, None, false).unwrap();
        let (chosen_0_again, _) =
            registry.register("req-3".to_string(), None, None, false).unwrap();

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

        let (chosen_0, _) = registry.register("req-1".to_string(), None, None, false).unwrap();
        let (chosen_1, _) = registry.register("req-2".to_string(), None, None, false).unwrap();
        let (chosen_0_again, _) =
            registry.register("req-3".to_string(), None, None, false).unwrap();

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
    fn routing_score_counts_waiting_without_extra_penalty() {
        let state = EngineRoutingState {
            inflight: 1,
            last_scheduler_stats: Some(EngineLoadSnapshot {
                waiting: 3,
                running: 2,
            }),
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

        let (chosen, _) = registry.register("req-stats".to_string(), None, None, false).unwrap();
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
        let (chosen, _) = registry.register("req-1".to_string(), None, Some(2), false).unwrap();
        assert_eq!(chosen, engine_2);

        // Explicitly target rank 0 (first engine).
        let (chosen, _) = registry.register("req-2".to_string(), None, Some(0), false).unwrap();
        assert_eq!(chosen, engine_0);

        // Explicitly target rank 1.
        let (chosen, _) = registry.register("req-3".to_string(), None, Some(1), false).unwrap();
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
        registry.register("req-lb-0".to_string(), None, None, false).unwrap();

        // Now engine_0 has 1 in-flight. Without dp_rank, next would go to engine_1.
        // But with dp_rank=0, it should still go to engine_0.
        let (chosen, _) = registry.register("req-dp".to_string(), None, Some(0), false).unwrap();
        assert_eq!(chosen, engine_0);
    }

    #[test]
    fn register_with_out_of_range_rank_returns_error() {
        let mut registry = RequestRegistry::new(&[
            connected_engine(EngineId::from_engine_index(0)),
            connected_engine(EngineId::from_engine_index(1)),
        ]);

        let error = registry.register("req-1".to_string(), None, Some(2), false).unwrap_err();
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

        let (chosen, _) = registry.register("req-ok".to_string(), None, Some(3), false).unwrap();
        assert_eq!(chosen, engine_3);

        let error = registry.register("req-bad".to_string(), None, Some(0), false).unwrap_err();
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
