// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use arc_swap::ArcSwapOption;
use bytes::Bytes;
use futures::future::join_all;
use parking_lot::Mutex;
use thiserror_ext::AsReport as _;
use tokio::runtime::Handle;
use tokio::sync::{mpsc, watch};
use tracing::{debug, error, info, trace, warn};
use vllm_metrics::{LoraInfoExporter, METRICS};
use zeromq::RouterSendHalf;

use crate::client::readiness::{Readiness, RecheckGuard};
use crate::client::state::{
    OutputReceiver, OutputSender, RequestObserver, RequestRegistry, UtilityReceiver,
    UtilityRegistry,
};
use crate::client::stream::{EngineCoreOutputStream, EngineCoreStreamOutput};
use crate::client::{AbortCause, AbortRequest};
use crate::error::{client_closed, dispatcher_closed, unexpected_dispatcher_output};
use crate::metrics::{EngineSleepState, IterationMetricHandles, SchedulerStatsRecorder};
use crate::protocol::output::{EngineCoreOutput, EngineCoreOutputs};
use crate::protocol::request::{EngineCoreRequest, EngineCoreRequestType};
use crate::protocol::stats::SchedulerStats;
use crate::protocol::utility::{EngineCoreUtilityRequest, UtilityOutput};
use crate::protocol::{encode_msgpack, msgpack_zero_copy_threshold};
use crate::transport::{ConnectedEngine, EngineId};
use crate::{Error, Result, transport};

const INPUT_SEND_TIMEOUT_ENV: &str = "VLLM_RS_ENGINE_SEND_TIMEOUT_SECS";
/// Default upper bound on a single input-socket send before treating the
/// engine transport as stalled. Matches `EngineCoreClientConfig::new_single`'s
/// 30s `ready_timeout` so transient backpressure completes without false positives.
const DEFAULT_INPUT_SEND_TIMEOUT: Duration = Duration::from_secs(30);

/// Parse `VLLM_RS_ENGINE_SEND_TIMEOUT_SECS` into a positive finite `Duration`,
/// falling back to [`DEFAULT_INPUT_SEND_TIMEOUT`] when unset, invalid, zero,
/// sub-nanosecond (rounds to zero), or overflowing `Duration::MAX`.
fn parse_input_send_timeout(raw: Option<&str>) -> Duration {
    raw.and_then(|value| value.parse::<f64>().ok())
        .and_then(|value| Duration::try_from_secs_f64(value).ok())
        .filter(|duration| !duration.is_zero())
        .unwrap_or(DEFAULT_INPUT_SEND_TIMEOUT)
}

fn input_send_timeout() -> Duration {
    parse_input_send_timeout(std::env::var(INPUT_SEND_TIMEOUT_ENV).ok().as_deref())
}

pub(crate) struct ClientInner {
    input_send: RouterSendHalf,
    /// Upper bound for a single send on `input_send`, loaded from
    /// `VLLM_RS_ENGINE_SEND_TIMEOUT_SECS` when this inner client is created.
    input_send_timeout: Duration,
    /// The runtime handle used for sending messages to the engine.
    handle: Handle,
    model_name: String,
    /// Engines that utility calls go to, and this client's index on them.
    engine_ids: Vec<EngineId>,
    client_index: u32,
    /// Per-tensor byte threshold loaded from env variable
    /// `VLLM_MSGPACK_ZERO_COPY_THRESHOLD` when this inner client is created.
    msgpack_zero_copy_threshold: usize,
    /// Whether the connected engines record stats. See
    /// [`crate::EngineCoreClientConfig::engine_stats_enabled`].
    engine_stats_enabled: bool,
    /// The engines' `max_loras`, reported as the `max_lora` label of
    /// `vllm:lora_requests_info`.
    max_loras: u32,
    /// Scheduler-stats metric handles, present only when engines record stats
    /// so that gauges are not exported with values no engine ever reports.
    scheduler_stats_recorder: Option<SchedulerStatsRecorder>,
    iteration_metrics: BTreeMap<u32, IterationMetricHandles>,
    request_reg: Mutex<RequestRegistry>,
    utility_reg: Mutex<UtilityRegistry>,
    health_error: ArcSwapOption<Error>,
    health_tx: watch::Sender<bool>,
    /// Whether the engines can accept and process new requests, maintained
    /// around the control calls this client sends.
    readiness: Arc<Readiness>,
}

impl ClientInner {
    /// Create a new instance with the given input send half after the startup
    /// handshake completes.
    pub fn new(
        input_send: RouterSendHalf,
        handle: Handle,
        model_name: String,
        client_index: u32,
        engine_stats_enabled: bool,
        engines: &[ConnectedEngine],
    ) -> Self {
        let scheduler_stats_recorder = engine_stats_enabled
            .then(|| SchedulerStatsRecorder::new(&METRICS.scheduler, &model_name, engines));
        let iteration_metrics = engines
            .iter()
            .filter_map(|engine| {
                let engine = engine.engine_id.engine_index()?;
                let handles = IterationMetricHandles::new(&METRICS.request, &model_name, engine);
                Some((engine, handles))
            })
            .collect();
        // Every engine reports the same `max_loras`, validated before the
        // client is built.
        let max_loras = engines[0].ready_response.max_loras;
        Self {
            input_send,
            input_send_timeout: input_send_timeout(),
            handle,
            model_name,
            engine_ids: engines.iter().map(|engine| engine.engine_id.clone()).collect(),
            client_index,
            msgpack_zero_copy_threshold: msgpack_zero_copy_threshold(),
            engine_stats_enabled,
            max_loras,
            scheduler_stats_recorder,
            iteration_metrics,
            request_reg: Mutex::new(RequestRegistry::new(engines)),
            utility_reg: Mutex::new(UtilityRegistry::default()),
            health_error: ArcSwapOption::empty(),
            health_tx: watch::Sender::new(true),
            readiness: Arc::new(Readiness::new()),
        }
    }

    /// Get the model name associated with this client used for metrics
    /// labeling.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Record a completed sleep or wake transition in
    /// `vllm:engine_sleep_state`, exported only when engines record stats.
    pub fn record_sleep_state(&self, state: EngineSleepState) {
        if let Some(recorder) = &self.scheduler_stats_recorder {
            recorder.record_sleep_state(state);
        }
    }

    /// Register a newly added request. Return the selected engine id and the
    /// per-request output channel bound to its `request_id`.
    ///
    /// When `data_parallel_rank` is provided, the request is routed to that
    /// specific engine rank, bypassing load balancing. `lora_name` is the
    /// request's LoRA adapter, tracked for `vllm:lora_requests_info`.
    pub fn register_request(
        &self,
        request_id: String,
        lora_name: Option<String>,
        data_parallel_rank: Option<u32>,
        stream_interval: NonZeroU32,
        observer: Option<Box<dyn RequestObserver>>,
    ) -> Result<(EngineId, OutputReceiver)> {
        let mut registry = self.request_reg.lock();
        if registry.is_closed() {
            return Err(self.closed_error());
        }
        let registered = registry.register(
            request_id.clone(),
            lora_name,
            data_parallel_rank,
            stream_interval,
        )?;
        if let Some(observer) = observer {
            registry.observe(&request_id, observer);
        }
        Ok(registered)
    }

    /// Allocate the next utility `call_id` and register its waiting receiver.
    pub fn allocate_and_register_utility_call(&self) -> Result<(u64, UtilityReceiver)> {
        let mut registry = self.utility_reg.lock();
        if registry.is_closed() {
            return Err(self.closed_error());
        }
        Ok(registry.allocate_and_register())
    }

    /// Undo a batch of utility call allocations when the fan-out send fails
    /// partway through. Silently ignores unknown call ids so callers can pass
    /// the full set without first filtering successful sends.
    pub fn unregister_utility_calls(&self, call_ids: impl IntoIterator<Item = u64>) {
        self.utility_reg.lock().unregister_many(call_ids);
    }

    #[cfg(test)]
    pub fn pending_utility_call_count(&self) -> usize {
        self.utility_reg.lock().len()
    }

    /// Undo a request registration when `add_request()` fails.
    pub fn rollback_request(&self, request_id: &str) {
        let _ = self.request_reg.lock().remove(request_id);
    }

    /// Filter the given request IDs to the subset that are still tracked as
    /// active and can be aborted, grouped by the engine that originally
    /// accepted them.
    pub fn abortable_request_ids(
        &self,
        request_ids: &[String],
    ) -> Result<BTreeMap<EngineId, Vec<String>>> {
        let registry = self.request_reg.lock();
        if registry.is_closed() {
            return Err(self.closed_error());
        }
        Ok(registry.abortable_request_ids(request_ids))
    }

    /// Send a whole engine output batch to the request streams with one
    /// registry lock acquisition. Returns the outputs whose requests are no
    /// longer tracked.
    pub fn send_outputs(
        &self,
        outputs: impl IntoIterator<Item = EngineCoreStreamOutput>,
    ) -> Vec<EngineCoreStreamOutput> {
        let mut registry = self.request_reg.lock();
        outputs.into_iter().filter_map(|output| registry.send_output(output)).collect()
    }

    /// Remove a batch of requests that have finished or aborted, returning
    /// their stream senders.
    pub fn finish_requests<'a>(
        &self,
        request_ids: impl IntoIterator<Item = &'a String>,
    ) -> Vec<OutputSender> {
        self.request_reg.lock().finish_many(request_ids)
    }

    /// Finalize client-initiated aborts by pushing a terminal `Abort` output
    /// down each request's stream and removing it from the registry. Returns
    /// the request ids that were still active. See [`RequestRegistry::abort_many`].
    pub fn abort_requests_locally<'a>(
        &self,
        request_ids: impl IntoIterator<Item = &'a String>,
    ) -> Vec<String> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        self.request_reg.lock().abort_many(request_ids, timestamp)
    }

    /// Apply one scheduler stats update for the given engine to the local
    /// routing state. Returns `false` if the engine is unknown to the
    /// client.
    pub fn apply_scheduler_stats(&self, engine_index: u32, stats: &SchedulerStats) -> bool {
        self.request_reg.lock().apply_scheduler_stats(engine_index, stats)
    }

    /// Snapshot the adapter names of tracked LoRA requests as
    /// (running, waiting) sets.
    pub fn lora_adapter_states(&self) -> (BTreeSet<String>, BTreeSet<String>) {
        self.request_reg.lock().lora_adapter_states()
    }

    /// Close all active request streams and utility calls with the first
    /// persistent health error.
    pub fn close_registries(&self, error: Arc<Error>) {
        let persistent_error = self.record_health_error(error);
        self.publish_unhealthy();
        let request_senders = self.request_reg.lock().close();
        let utility_senders = self.utility_reg.lock().close();

        // Notify all ongoing requests that the client is closed.
        for sender in request_senders {
            sender.send_error(Error::Shared(persistent_error.clone()));
        }
        for sender in utility_senders {
            let _ = sender.send(Err(Error::Shared(persistent_error.clone())));
        }
    }

    /// Return the first persistent health error observed by the client, if any.
    pub fn health_error(&self) -> Option<Arc<Error>> {
        self.health_error.load_full()
    }

    /// Return whether the client still considers the engine healthy.
    pub fn is_healthy(&self) -> bool {
        self.health_error.load().is_none()
    }

    /// Subscribe to engine health changes. The current value is `true` while
    /// the client is healthy and changes permanently to `false` on failure.
    pub fn subscribe_health(&self) -> watch::Receiver<bool> {
        self.health_tx.subscribe()
    }

    pub fn readiness(&self) -> &Arc<Readiness> {
        &self.readiness
    }

    /// Call a typed utility method on every engine. See
    /// [`crate::EngineCoreClient::call_utility`].
    ///
    /// A control call passes its `recheck` guard, which the call holds until
    /// the engines reply.
    pub async fn call_utility<T, A>(
        self: &Arc<Self>,
        method: &str,
        args: A,
        recheck: Option<RecheckGuard>,
    ) -> Result<Vec<T>>
    where
        T: serde::de::DeserializeOwned,
        A: serde::Serialize + std::fmt::Debug,
    {
        trace!(
            method,
            client_index = self.client_index,
            engine_count = self.engine_ids.len(),
            "sending utility request"
        );

        let mut call_ids = Vec::with_capacity(self.engine_ids.len());
        let mut calls = Vec::with_capacity(self.engine_ids.len());
        let prepared = self.engine_ids.iter().try_for_each(|engine_id| {
            let (call_id, rx) = self.allocate_and_register_utility_call()?;
            call_ids.push(call_id);
            let request = EngineCoreUtilityRequest::new(self.client_index, call_id, method, &args)?;
            let payload = Bytes::from(encode_msgpack(&request)?);
            calls.push((engine_id.clone(), call_id, rx, payload));
            Ok(())
        });
        if let Err(error) = prepared {
            self.unregister_utility_calls(call_ids);
            return Err(error);
        }

        // The engines execute a sent request whatever happens to the caller,
        // so the call runs to completion in a task of its own on this client's
        // runtime, and dropping this future only stops waiting for it.
        let inner = self.clone();
        let method_name = method.to_string();
        let call = self.handle.spawn(async move {
            let _recheck = recheck;
            let outputs = join_all(calls.into_iter().map(|(engine_id, call_id, rx, payload)| {
                let inner = &inner;
                let method_name = &method_name;
                async move {
                    inner
                        .send_input_message(
                            &engine_id,
                            EngineCoreRequestType::Utility,
                            payload,
                            Vec::new(),
                        )
                        .await?;
                    rx.await.map_err(|_| Error::UtilityCallClosed {
                        method: method_name.clone(),
                        call_id,
                    })?
                }
            }))
            .await;
            // A request that failed to send leaves its waiter without a reply.
            inner.unregister_utility_calls(call_ids);
            outputs
        });
        let outputs = match call.await {
            Ok(outputs) => outputs,
            Err(error) if error.is_panic() => std::panic::resume_unwind(error.into_panic()),
            Err(_) => {
                self.close_registries(Arc::new(client_closed!(
                    "engine-core client shut down during utility call `{method}`"
                )));
                return Err(self.closed_error());
            }
        };
        outputs.into_iter().map(|output| output?.into_typed_result(method)).collect()
    }

    /// Ask every engine whether it can accept and process new requests.
    pub async fn is_inference_ready(self: &Arc<Self>) -> Result<bool> {
        let ready: Vec<bool> = self.call_utility("is_inference_ready", (), None).await?;
        Ok(ready.into_iter().all(|ready| ready))
    }

    /// Resolve one utility output to the waiting caller. Returns `true` if a
    /// waiting caller existed.
    pub fn resolve_utility_output(&self, output: UtilityOutput) -> bool {
        let Some(call_id) = output.call_id.as_u64() else {
            // Currently, all utility call issued by the client should have unsigned call IDs.
            return false;
        };

        match self.utility_reg.lock().resolve(&call_id) {
            Some(sender) => {
                sender.send(Ok(output)).unwrap_or_default();
                true
            }
            None => false,
        }
    }

    #[cfg(any(test, feature = "test-util"))]
    pub(crate) fn with_input_send_timeout(mut self, timeout: Duration) -> Self {
        self.input_send_timeout = timeout;
        self
    }

    /// Send one encoded message on the shared input socket under a bounded
    /// timeout. If the transport stalls or fails, transition the client to its
    /// sticky unhealthy state (`close_registries`) so `/health` immediately
    /// returns 503, in-flight request/utility waiters fail fast, and
    /// `run_abort_loop` stops sending aborts onto the unhealthy transport.
    ///
    /// Note on concurrent health transitions: a send that passes the initial
    /// `health_error()` check just before another task closes the registries
    /// may still finish enqueueing onto the socket without a global send lock;
    /// its corresponding output stream or utility waiter has already been
    /// terminated by `close_registries`.
    async fn send_input_message(
        &self,
        engine_id: &EngineId,
        request_type: EngineCoreRequestType,
        payload: Bytes,
        aux_frames: Vec<Bytes>,
    ) -> Result<()> {
        if let Some(existing) = self.health_error() {
            return Err(Error::Shared(existing));
        }

        let mut input_send = self.input_send.clone();
        let send_timeout = self.input_send_timeout;
        let outcome = tokio::time::timeout(
            send_timeout,
            transport::send_message(
                &mut input_send,
                engine_id,
                request_type.to_frame(),
                payload,
                aux_frames,
            ),
        )
        .await;

        match outcome {
            Ok(Ok(())) => Ok(()),
            Ok(Err(error)) => Err(self.fail_on_input_send_error(engine_id, request_type, error)),
            Err(_) => Err(self.fail_on_input_send_error(
                engine_id,
                request_type,
                Error::InputSendTimeout {
                    engine_id: engine_id.clone(),
                    timeout: send_timeout,
                },
            )),
        }
    }

    fn fail_on_input_send_error(
        &self,
        engine_id: &EngineId,
        request_type: EngineCoreRequestType,
        error: Error,
    ) -> Error {
        error!(
            ?engine_id,
            ?request_type,
            error = %error.as_report(),
            "engine input send failed; marking client unhealthy"
        );
        self.close_registries(Arc::new(error));
        self.closed_error()
    }

    /// Send the given message to the engine. The request should be first
    /// registered via `register_request()` to ensure the request stream is
    /// tracked.
    pub async fn send_to_engine<T>(
        self: &Arc<Self>,
        engine_id: &EngineId,
        request_type: EngineCoreRequestType,
        payload: &T,
    ) -> Result<()>
    where
        T: serde::Serialize + std::fmt::Debug,
    {
        let payload = Bytes::from(encode_msgpack(payload)?);
        self.send_encoded_to_engine(engine_id, request_type, payload, Vec::new()).await
    }

    /// Send an add request, moving large tensor buffers into auxiliary frames.
    pub async fn send_request_to_engine(
        self: &Arc<Self>,
        engine_id: &EngineId,
        mut payload: EngineCoreRequest,
        stream: EngineCoreOutputStream,
    ) -> Result<EngineCoreOutputStream> {
        let aux_frames = payload.extract_aux_frames(self.msgpack_zero_copy_threshold);
        let payload = Bytes::from(encode_msgpack(&payload)?);
        let inner = self.clone();
        let engine_id_owned = engine_id.clone();

        let join_result = self
            .handle
            .spawn(async move {
                // If the caller cancels, dropping the stream must enqueue Abort
                // after Add, not while this detached send is still pending.
                // If sending fails or times out, `send_input_message` closes the
                // registries BEFORE `stream` drops here, so `run_abort_loop`
                // never attempts to send an Abort onto the stalled/broken socket.
                inner
                    .send_input_message(
                        &engine_id_owned,
                        EngineCoreRequestType::Add,
                        payload,
                        aux_frames,
                    )
                    .await?;
                Ok(stream)
            })
            .await;

        match join_result {
            Ok(result) => result,
            Err(join_error) if join_error.is_panic() => {
                std::panic::resume_unwind(join_error.into_panic())
            }
            Err(join_error) => Err(self.fail_on_input_send_error(
                engine_id,
                EngineCoreRequestType::Add,
                Error::ZmqRuntimeTask(join_error),
            )),
        }
    }

    async fn send_encoded_to_engine(
        self: &Arc<Self>,
        engine_id: &EngineId,
        request_type: EngineCoreRequestType,
        payload: Bytes,
        aux_frames: Vec<Bytes>,
    ) -> Result<()> {
        let inner = self.clone();
        let engine_id_owned = engine_id.clone();

        let join_result = self
            .handle
            .spawn(async move {
                inner
                    .send_input_message(&engine_id_owned, request_type, payload, aux_frames)
                    .await
            })
            .await;

        match join_result {
            Ok(result) => result,
            Err(join_error) if join_error.is_panic() => {
                std::panic::resume_unwind(join_error.into_panic())
            }
            Err(join_error) => Err(self.fail_on_input_send_error(
                engine_id,
                request_type,
                Error::ZmqRuntimeTask(join_error),
            )),
        }
    }

    /// Handle an abort request by sending the abort message to the engine.
    pub async fn do_abort_requests(
        self: &Arc<Self>,
        engine_id: &EngineId,
        request_ids: &[String],
    ) -> Result<()> {
        self.send_to_engine(engine_id, EngineCoreRequestType::Abort, &request_ids).await
    }

    /// Shut down by closing all active request streams and utility calls with a
    /// sticky client closed error.
    pub fn shutdown(&self) {
        self.close_registries(Arc::new(client_closed!("engine-core client shut down")));
    }

    /// Remove the request from the active registry for auto-abort and return
    /// the engine that the request was originally routed to, if it is still
    /// active.
    pub fn take_auto_abort_target(&self, request_id: &str) -> Option<EngineId> {
        let mut registry = self.request_reg.lock();
        let (_, engine_id) = registry.remove(request_id)?;
        if registry.is_closed() {
            return None;
        }
        Some(engine_id)
    }

    /// Publish the first persistent health error and return the sticky error
    /// recorded for this client. Later failures do not overwrite the first
    /// one so `/health` and post-close callers observe a stable cause.
    fn record_health_error(&self, error: Arc<Error>) -> Arc<Error> {
        if let Some(existing) = self.health_error.load_full() {
            return existing;
        }
        self.health_error
            .rcu(|current| current.clone().unwrap_or_else(|| error.clone()));
        self.health_error
            .load_full()
            .expect("health error must be recorded before registries close")
    }

    /// Publish the sticky healthy-to-unhealthy transition.
    fn publish_unhealthy(&self) {
        self.health_tx.send_if_modified(|healthy| std::mem::replace(healthy, false));
    }

    /// Assert there is a recorded health error and return a `Shared` variant
    /// wrapping it for error returns when the client is already closed.
    fn closed_error(&self) -> Error {
        Error::Shared(self.health_error.load_full().expect(
            "closed registry must have a recorded health error before rejecting new operations",
        ))
    }
}

/// Background loop that listens for request IDs to abort and sends abort
/// messages to the engine. This is used to implement the auto-abort behavior
/// when a request stream is dropped without being properly terminated.
pub(crate) async fn run_abort_loop(
    inner: Arc<ClientInner>,
    mut abort_rx: mpsc::UnboundedReceiver<AbortRequest>,
) {
    // Coalesce bursts of auto-aborts into a single Abort message per engine.
    // A dropped-stream storm (e.g. many clients disconnecting at once under
    // high concurrency) would otherwise issue one engine round-trip per
    // request. `recv_many` returns as soon as at least one item is ready, so a
    // lone abort is still forwarded promptly.
    const MAX_DRAIN: usize = 1024;
    let mut batch: Vec<AbortRequest> = Vec::new();

    while abort_rx.recv_many(&mut batch, MAX_DRAIN).await > 0 {
        if !inner.is_healthy() {
            batch.clear();
            continue;
        }
        let mut by_engine: BTreeMap<EngineId, Vec<String>> = BTreeMap::new();

        for AbortRequest { request_id, cause } in batch.drain(..) {
            let Some(engine_id) = inner.take_auto_abort_target(&request_id) else {
                debug!(request_id, "skip auto-abort for inactive request");
                continue;
            };

            match cause {
                AbortCause::DroppedStream => {
                    info!(request_id, "auto-aborting request due to dropped stream")
                }
                AbortCause::StopStringMatched => {
                    debug!(
                        request_id,
                        "auto-aborting request due to stop string matched"
                    )
                }
            }

            by_engine.entry(engine_id).or_default().push(request_id);
        }

        for (engine_id, request_ids) in by_engine {
            if let Err(error) = inner.do_abort_requests(&engine_id, &request_ids).await {
                warn!(
                    ?engine_id,
                    ?request_ids,
                    error = %error.as_report(),
                    "failed to auto-abort request streams"
                );
                if !inner.is_healthy() {
                    break;
                }
            }
        }
    }
}

/// Number of tokens one output accounts for in `vllm:iteration_tokens_total`.
fn output_iteration_tokens(output: &EngineCoreOutput) -> u64 {
    let mut tokens = output.new_token_ids.len() as u64;
    if let Some(prefill) = &output.prefill_stats {
        // The engine emits prefill_stats once, on the first output.
        tokens += u64::from(prefill.num_computed_tokens);
    }
    tokens
}

/// Background loop that listens for engine-core outputs and dispatches them to
/// the corresponding request streams based on their `request_id`.
pub(crate) async fn run_output_dispatcher_loop(
    inner: Arc<ClientInner>,
    mut output_rx: mpsc::Receiver<Result<EngineCoreOutputs>>,
) {
    // LoRA phases come from request lifecycle events, which engines only emit
    // when they record stats.
    let mut lora_info = inner.engine_stats_enabled.then(|| LoraInfoExporter::new(inner.max_loras));

    let result: Result<()> = async {
        loop {
            let outputs = match output_rx.recv().await {
                Some(outputs) => outputs,
                None => Err(dispatcher_closed!(
                    "engine-core output dispatcher channel closed"
                )),
            }?;

            match outputs {
                EngineCoreOutputs::RequestBatch(batch) => {
                    let has_outputs = !batch.outputs.is_empty();
                    let mut iteration_tokens: u64 =
                        batch.outputs.iter().map(output_iteration_tokens).sum();
                    let (engine_index, timestamp, weight_version) =
                        (batch.engine_index, batch.timestamp, batch.weight_version);
                    let outputs = batch.outputs.into_iter().map(|output| EngineCoreStreamOutput {
                        engine_index,
                        timestamp,
                        weight_version: weight_version.clone(),
                        output,
                    });
                    for output in inner.send_outputs(outputs) {
                        debug!(
                            request_id = output.request_id,
                            "dropping output for inactive request"
                        );
                        iteration_tokens -= output_iteration_tokens(&output);
                    }

                    if has_outputs
                        && let Some(handles) = inner.iteration_metrics.get(&batch.engine_index)
                    {
                        handles.iteration_tokens_total.observe(iteration_tokens as f64);
                    }

                    // The sender for normally-finished requests should have already been removed
                    // from the registry when their final output was dispatched
                    // above. This serves as a safety net to capture any
                    // requests marked as finished by the engine.
                    if let Some(finished_requests) = batch.finished_requests.as_ref() {
                        for request_id in finished_requests {
                            trace!(request_id, "request completed via finished_requests");
                        }
                        drop(inner.finish_requests(finished_requests));
                    }

                    if let Some(scheduler_stats) = batch.scheduler_stats.as_ref() {
                        if !inner.apply_scheduler_stats(batch.engine_index, scheduler_stats) {
                            debug!(
                                engine_index = batch.engine_index,
                                "dropping scheduler stats for unknown engine"
                            );
                        }
                        if let Some(recorder) = &inner.scheduler_stats_recorder {
                            recorder.record(batch.engine_index, scheduler_stats);
                        }
                    }

                    // The engine's scheduler stats never carry adapter names;
                    // the gauge is derived from the registry's frontend-side
                    // request tracking instead.
                    if let Some(lora_info) = &mut lora_info {
                        let (running, waiting) = inner.lora_adapter_states();
                        lora_info.update(&METRICS.scheduler, running, waiting);
                    }
                }
                EngineCoreOutputs::Utility(utility) => {
                    let call_id = utility.output.call_id;
                    if inner.resolve_utility_output(utility.output) {
                        trace!(
                            %call_id,
                            engine_index = utility.engine_index,
                            "resolved utility output"
                        );
                    } else {
                        warn!(
                            %call_id,
                            engine_index = utility.engine_index,
                            "dropping output for unexpected utility call"
                        );
                    }
                }
                other => {
                    Err::<(), _>(unexpected_dispatcher_output!(
                        "received unexpected output on main dispatcher path: {other:?}"
                    ))?;
                }
            }
        }
    }
    .await;
    let Err(error) = result else { return };

    warn!(error = %error.as_report(), "output dispatcher exiting with error");
    inner.close_registries(Arc::new(error));
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tokio::runtime::Builder;
    use tokio::time::timeout;
    use zeromq::{DealerSocket, RouterSocket, Socket, SocketRecv, SocketSend};

    use super::*;
    use crate::client::output_channel::output_channel;
    use crate::mock_engine::default_ready_response;
    use crate::protocol::sampling::EngineCoreSamplingParams;

    async fn test_inner() -> ClientInner {
        let mut socket = RouterSocket::new();
        socket.bind("tcp://127.0.0.1:0").await.unwrap();
        let (send, _) = socket.split();
        ClientInner::new(
            send,
            Handle::current(),
            "test-model".to_string(),
            0,
            true,
            &[ConnectedEngine {
                engine_id: EngineId::from(b"engine-0"),
                ready_response: default_ready_response(),
            }],
        )
    }

    #[test]
    fn cancelling_add_defers_abort_until_send_completes() {
        let runtime = Builder::new_current_thread().enable_all().build().unwrap();
        let send_runtime = Builder::new_current_thread().enable_all().build().unwrap();
        runtime.block_on(async {
            let mut router = RouterSocket::new();
            let endpoint = router.bind("tcp://127.0.0.1:0").await.unwrap();
            let mut dealer = DealerSocket::new();
            dealer.connect(&endpoint.to_string()).await.unwrap();
            dealer.send("ready".into()).await.unwrap();
            let ready = router.recv().await.unwrap().into_vec();
            let engine_id = EngineId::from(ready[0].to_vec());
            let (input_send, _) = router.split();
            let inner = Arc::new(ClientInner::new(
                input_send,
                send_runtime.handle().clone(),
                "test-model".to_string(),
                0,
                false,
                &[ConnectedEngine {
                    engine_id: engine_id.clone(),
                    ready_response: default_ready_response(),
                }],
            ));
            let (abort_tx, mut abort_rx) = mpsc::unbounded_channel();
            let (_output_tx, output_rx) = output_channel(NonZeroU32::MIN);
            let stream = EngineCoreOutputStream::new("req-1".to_string(), 0, abort_tx, output_rx);
            let request = EngineCoreRequest {
                request_id: "req-1".to_string(),
                prompt_token_ids: Some(vec![1].into()),
                sampling_params: Some(EngineCoreSamplingParams::for_test()),
                ..Default::default()
            };
            let mut send = Box::pin(inner.send_request_to_engine(&engine_id, request, stream));

            // The send runtime is not running yet, so cancellation happens
            // after spawning the send but before Add can enter the socket queue.
            assert!(futures::poll!(send.as_mut()).is_pending());
            drop(send);
            assert!(matches!(
                abort_rx.try_recv(),
                Err(mpsc::error::TryRecvError::Empty)
            ));

            let abort = std::thread::scope(|scope| {
                scope
                    .spawn(|| {
                        send_runtime.block_on(async {
                            timeout(Duration::from_secs(5), abort_rx.recv()).await.unwrap().unwrap()
                        })
                    })
                    .join()
                    .unwrap()
            });
            assert_eq!(abort.request_id, "req-1");
            let add = timeout(Duration::from_secs(5), dealer.recv())
                .await
                .unwrap()
                .unwrap()
                .into_vec();
            assert_eq!(add[0], EngineCoreRequestType::Add.to_frame());
            let request: EngineCoreRequest = rmp_serde::from_slice(&add[1]).unwrap();
            assert_eq!(request.request_id, abort.request_id);
        });
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn close_registries_records_first_health_error_only() {
        let inner = test_inner().await;
        let mut health = inner.subscribe_health();
        assert!(*health.borrow());

        inner.close_registries(Arc::new(Error::EngineCoreDead));
        health.changed().await.expect("health sender remains open");
        assert!(!inner.is_healthy());
        assert!(!*health.borrow());
        assert!(matches!(
            inner.health_error().as_deref(),
            Some(Error::EngineCoreDead)
        ));
        assert!(!*inner.subscribe_health().borrow());

        inner.close_registries(Arc::new(client_closed!("shutdown")));
        assert!(matches!(
            inner.health_error().as_deref(),
            Some(Error::EngineCoreDead)
        ));
    }

    async fn saturate_peer_send_queue(input_send: &mut RouterSendHalf, engine_id: &EngineId) {
        let big_payload = Bytes::from(vec![b'x'; 4096]);
        let small_payload = Bytes::from_static(b"x");
        for i in 0..115_000usize {
            let payload = if i < 4_000 {
                big_payload.clone()
            } else {
                small_payload.clone()
            };
            let fut = transport::send_message(
                input_send,
                engine_id,
                EngineCoreRequestType::Abort.to_frame(),
                payload,
                Vec::new(),
            );
            let wait = if i < 100_050 {
                Duration::from_millis(50)
            } else {
                Duration::from_millis(20)
            };
            if timeout(wait, fut).await.is_err() {
                return;
            }
        }
        panic!("expected peer send queue to saturate and block");
    }

    #[test]
    fn parse_input_send_timeout_handles_edge_cases() {
        assert_eq!(parse_input_send_timeout(None), DEFAULT_INPUT_SEND_TIMEOUT);
        assert_eq!(
            parse_input_send_timeout(Some("12.5")),
            Duration::from_millis(12_500)
        );
        assert_eq!(parse_input_send_timeout(Some("1")), Duration::from_secs(1));
        for bad in [
            "", "abc", "0", "-5", "NaN", "inf", "-inf", "1e-300", "1e308",
        ] {
            assert_eq!(
                parse_input_send_timeout(Some(bad)),
                DEFAULT_INPUT_SEND_TIMEOUT,
                "expected fallback for {bad:?}"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn input_send_transport_error_marks_client_unhealthy_and_fails_inflight() {
        let inner = Arc::new(test_inner().await);
        let engine_id = EngineId::from(b"engine-0");

        let (_routed, mut inflight_rx) = inner
            .register_request(
                "req-inflight".to_string(),
                None,
                None,
                NonZeroU32::MIN,
                None,
            )
            .unwrap();
        let (_call_id, utility_rx) = inner.allocate_and_register_utility_call().unwrap();

        let err = inner
            .do_abort_requests(&engine_id, &["req-other".to_string()])
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Shared(_)), "unexpected error: {err:?}");
        assert!(
            matches!(inner.health_error().as_deref(), Some(Error::Transport(_))),
            "expected recorded Transport health error, got {:?}",
            inner.health_error()
        );

        assert!(
            !inner.is_healthy(),
            "input send transport error must flip client unhealthy"
        );
        assert!(
            matches!(inflight_rx.try_recv(), Ok(Err(Error::Shared(_)))),
            "in-flight request stream must receive sticky health error"
        );
        assert!(
            matches!(
                timeout(Duration::from_secs(1), utility_rx).await,
                Ok(Ok(Err(Error::Shared(_))))
            ),
            "in-flight utility waiter must receive sticky health error"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stalled_abort_send_times_out_marks_unhealthy_and_unblocks_abort_loop() {
        let mut router = RouterSocket::new();
        let endpoint = router.bind("tcp://127.0.0.1:0").await.unwrap();
        let mut dealer = DealerSocket::new();
        dealer.connect(&endpoint.to_string()).await.unwrap();
        dealer.send("ready".into()).await.unwrap();
        let ready = router.recv().await.unwrap().into_vec();
        let engine_id = EngineId::from(ready[0].to_vec());
        let (mut input_send_fill, _input_recv) = router.split();

        let inner = Arc::new(
            ClientInner::new(
                input_send_fill.clone(),
                Handle::current(),
                "test-model".to_string(),
                0,
                false,
                &[ConnectedEngine {
                    engine_id: engine_id.clone(),
                    ready_response: default_ready_response(),
                }],
            )
            .with_input_send_timeout(Duration::from_millis(80)),
        );

        saturate_peer_send_queue(&mut input_send_fill, &engine_id).await;

        let (_routed, mut inflight_rx) = inner
            .register_request(
                "req-inflight".to_string(),
                None,
                None,
                NonZeroU32::MIN,
                None,
            )
            .unwrap();
        let (_routed1, _rx1) = inner
            .register_request("req-drop-1".to_string(), None, None, NonZeroU32::MIN, None)
            .unwrap();
        let (_routed2, _rx2) = inner
            .register_request("req-drop-2".to_string(), None, None, NonZeroU32::MIN, None)
            .unwrap();

        let mut health_rx = inner.subscribe_health();
        let (abort_tx, abort_rx) = mpsc::unbounded_channel();
        let abort_task = tokio::spawn(run_abort_loop(inner.clone(), abort_rx));

        // Send req-drop-1 FIRST while client is healthy so `run_abort_loop` actually
        // enters and blocks inside `do_abort_requests()` on the saturated socket.
        abort_tx
            .send(AbortRequest {
                request_id: "req-drop-1".to_string(),
                cause: AbortCause::DroppedStream,
            })
            .unwrap();

        timeout(Duration::from_secs(1), health_rx.changed())
            .await
            .expect("stalled abort send must time out and flip health")
            .unwrap();
        assert!(!*health_rx.borrow());
        assert!(!inner.is_healthy());
        assert!(matches!(
            inner.health_error().as_deref(),
            Some(Error::InputSendTimeout { .. })
        ));
        assert!(
            matches!(inflight_rx.try_recv(), Ok(Err(Error::Shared(_)))),
            "in-flight stream must fail fast after stalled abort send"
        );

        // Subsequent aborts are skipped without blocking on the unhealthy transport.
        abort_tx
            .send(AbortRequest {
                request_id: "req-drop-2".to_string(),
                cause: AbortCause::DroppedStream,
            })
            .unwrap();
        drop(abort_tx);
        timeout(Duration::from_secs(1), abort_task)
            .await
            .expect("abort loop must drain remaining events and exit once sender closes")
            .unwrap();
        drop(dealer);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stalled_add_closes_registries_before_stream_drop_auto_abort() {
        let mut router = RouterSocket::new();
        let endpoint = router.bind("tcp://127.0.0.1:0").await.unwrap();
        let mut dealer = DealerSocket::new();
        dealer.connect(&endpoint.to_string()).await.unwrap();
        dealer.send("ready".into()).await.unwrap();
        let engine_id = EngineId::from(router.recv().await.unwrap().into_vec()[0].to_vec());
        let (mut input_send_fill, _input_recv) = router.split();

        let inner = Arc::new(
            ClientInner::new(
                input_send_fill.clone(),
                Handle::current(),
                "test-model".to_string(),
                0,
                false,
                &[ConnectedEngine {
                    engine_id: engine_id.clone(),
                    ready_response: default_ready_response(),
                }],
            )
            .with_input_send_timeout(Duration::from_millis(80)),
        );
        saturate_peer_send_queue(&mut input_send_fill, &engine_id).await;

        let (abort_tx, abort_rx) = mpsc::unbounded_channel();
        let abort_task = tokio::spawn(run_abort_loop(inner.clone(), abort_rx));

        let (routed, rx) = inner
            .register_request("req-new".to_string(), None, None, NonZeroU32::MIN, None)
            .unwrap();
        assert_eq!(routed, engine_id);
        let stream = EngineCoreOutputStream::new("req-new".to_string(), 0, abort_tx.clone(), rx);
        let req = EngineCoreRequest {
            request_id: "req-new".to_string(),
            prompt_token_ids: Some(vec![1].into()),
            sampling_params: Some(EngineCoreSamplingParams::for_test()),
            ..Default::default()
        };

        let send_res = timeout(
            Duration::from_secs(1),
            inner.send_request_to_engine(&engine_id, req, stream),
        )
        .await
        .expect("stalled add must time out boundedly");
        assert!(send_res.is_err());
        assert!(!inner.is_healthy());
        assert!(matches!(
            inner.health_error().as_deref(),
            Some(Error::InputSendTimeout { .. })
        ));

        drop(abort_tx);
        timeout(Duration::from_secs(1), abort_task)
            .await
            .expect("abort loop must not block on stream drop after add timeout")
            .unwrap();
        drop(dealer);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn all_three_input_send_paths_propagate_failure_consistently() {
        // 1. call_utility path
        let inner_util = Arc::new(test_inner().await);
        let res: Result<Vec<bool>> = inner_util.call_utility("is_sleeping", (), None).await;
        assert!(res.is_err());
        assert!(!inner_util.is_healthy());
        assert_eq!(inner_util.pending_utility_call_count(), 0);

        // 2. send_request_to_engine path
        let inner_add = Arc::new(test_inner().await);
        let engine_id = EngineId::from(b"engine-0");
        let (abort_tx, _abort_rx) = mpsc::unbounded_channel();
        let (_tx, rx) = output_channel(NonZeroU32::MIN);
        let stream = EngineCoreOutputStream::new("req-add".to_string(), 0, abort_tx, rx);
        let req = EngineCoreRequest {
            request_id: "req-add".to_string(),
            prompt_token_ids: Some(vec![1].into()),
            sampling_params: Some(EngineCoreSamplingParams::for_test()),
            ..Default::default()
        };
        assert!(inner_add.send_request_to_engine(&engine_id, req, stream).await.is_err());
        assert!(!inner_add.is_healthy());

        // 3. send_encoded_to_engine / do_abort_requests path
        let inner_abort = Arc::new(test_inner().await);
        assert!(inner_abort.do_abort_requests(&engine_id, &["req-1".to_string()]).await.is_err());
        assert!(!inner_abort.is_healthy());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_sends_fail_with_shared_first_error_and_reject_new_registrations() {
        let mut router = RouterSocket::new();
        let endpoint = router.bind("tcp://127.0.0.1:0").await.unwrap();
        let mut dealer = DealerSocket::new();
        dealer.connect(&endpoint.to_string()).await.unwrap();
        dealer.send("ready".into()).await.unwrap();
        let ready = router.recv().await.unwrap().into_vec();
        let engine_id = EngineId::from(ready[0].to_vec());
        let (mut input_send_fill, _input_recv) = router.split();

        let inner = Arc::new(
            ClientInner::new(
                input_send_fill.clone(),
                Handle::current(),
                "test-model".to_string(),
                0,
                false,
                &[ConnectedEngine {
                    engine_id: engine_id.clone(),
                    ready_response: default_ready_response(),
                }],
            )
            .with_input_send_timeout(Duration::from_millis(80)),
        );
        saturate_peer_send_queue(&mut input_send_fill, &engine_id).await;

        let (abort_tx, abort_rx) = mpsc::unbounded_channel();
        let abort_task = tokio::spawn(run_abort_loop(inner.clone(), abort_rx));

        let barrier = Arc::new(tokio::sync::Barrier::new(17));
        let mut tasks = Vec::new();
        for i in 0..16usize {
            let inner = inner.clone();
            let engine_id = engine_id.clone();
            let abort_tx = abort_tx.clone();
            let barrier = barrier.clone();
            let req_id = format!("req-conc-{i}");
            let (_routed, rx) = inner
                .register_request(req_id.clone(), None, None, NonZeroU32::MIN, None)
                .unwrap();
            let stream = EngineCoreOutputStream::new(req_id.clone(), 0, abort_tx, rx);
            let req = EngineCoreRequest {
                request_id: req_id,
                prompt_token_ids: Some(vec![1].into()),
                sampling_params: Some(EngineCoreSamplingParams::for_test()),
                ..Default::default()
            };
            tasks.push(tokio::spawn(async move {
                barrier.wait().await;
                inner.send_request_to_engine(&engine_id, req, stream).await.map(|_| ())
            }));
        }

        let util_inner = inner.clone();
        let util_barrier = barrier.clone();
        let util_task = tokio::spawn(async move {
            util_barrier.wait().await;
            util_inner.call_utility::<bool, _>("is_sleeping", (), None).await
        });

        let results = timeout(Duration::from_secs(2), join_all(tasks))
            .await
            .expect("concurrent sends must finish boundedly without deadlock");
        let util_res = timeout(Duration::from_secs(1), util_task)
            .await
            .expect("utility call must finish boundedly")
            .unwrap();

        let first_error =
            inner.health_error().expect("first persistent health error must be recorded");
        assert!(matches!(
            first_error.as_ref(),
            Error::InputSendTimeout { .. }
        ));

        for r in results {
            match r.unwrap() {
                Err(Error::Shared(shared)) => {
                    assert!(
                        Arc::ptr_eq(&shared, &first_error),
                        "all concurrent sends must observe the exact same first health error"
                    );
                }
                other => panic!("expected Err(Error::Shared), got {other:?}"),
            }
        }
        match util_res {
            Err(Error::Shared(shared)) => {
                assert!(Arc::ptr_eq(&shared, &first_error));
            }
            other => panic!("expected Err(Error::Shared) from utility call, got {other:?}"),
        }

        assert!(!inner.is_healthy());
        assert!(matches!(
            inner.register_request(
                "req-after-close".to_string(),
                None,
                None,
                NonZeroU32::MIN,
                None
            ),
            Err(Error::Shared(_))
        ));
        assert!(matches!(
            inner.allocate_and_register_utility_call(),
            Err(Error::Shared(_))
        ));

        drop(abort_tx);
        timeout(Duration::from_secs(1), abort_task)
            .await
            .expect("abort loop must exit cleanly")
            .unwrap();
        drop(dealer);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn transient_backpressure_within_timeout_keeps_client_healthy() {
        let mut router = RouterSocket::new();
        let endpoint = router.bind("tcp://127.0.0.1:0").await.unwrap();
        let mut dealer = DealerSocket::new();
        dealer.connect(&endpoint.to_string()).await.unwrap();
        dealer.send("ready".into()).await.unwrap();
        let ready = router.recv().await.unwrap().into_vec();
        let engine_id = EngineId::from(ready[0].to_vec());
        let (mut input_send_fill, _input_recv) = router.split();

        let inner = Arc::new(
            ClientInner::new(
                input_send_fill.clone(),
                Handle::current(),
                "test-model".to_string(),
                0,
                false,
                &[ConnectedEngine {
                    engine_id: engine_id.clone(),
                    ready_response: default_ready_response(),
                }],
            )
            .with_input_send_timeout(Duration::from_secs(2)),
        );
        saturate_peer_send_queue(&mut input_send_fill, &engine_id).await;

        let drainer = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(30)).await;
            for _ in 0..4_500 {
                if timeout(Duration::from_millis(50), dealer.recv()).await.is_err() {
                    break;
                }
            }
            dealer
        });

        let (abort_tx, _abort_rx) = mpsc::unbounded_channel();
        let (_tx, rx) = output_channel(NonZeroU32::MIN);
        let stream = EngineCoreOutputStream::new("req-ok".to_string(), 0, abort_tx, rx);
        let req = EngineCoreRequest {
            request_id: "req-ok".to_string(),
            prompt_token_ids: Some(vec![1].into()),
            sampling_params: Some(EngineCoreSamplingParams::for_test()),
            ..Default::default()
        };

        let stream = inner
            .send_request_to_engine(&engine_id, req, stream)
            .await
            .expect("transient backpressure within timeout must succeed");
        assert!(
            inner.is_healthy(),
            "client must remain healthy after transient backpressure"
        );
        drop(stream);
        let _dealer = drainer.await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dp_multi_engine_single_peer_stall_fails_client_cleanly() {
        let mut router = RouterSocket::new();
        let endpoint = router.bind("tcp://127.0.0.1:0").await.unwrap();

        let id0 = EngineId::from_engine_index(0);
        let mut opts0 = zeromq::SocketOptions::default();
        opts0.peer_identity(zeromq::util::PeerIdentity::try_from(id0.clone()).unwrap());
        let mut dealer0 = DealerSocket::with_options(opts0);
        dealer0.connect(&endpoint.to_string()).await.unwrap();
        dealer0.send("ready-0".into()).await.unwrap();
        assert_eq!(
            EngineId::from(router.recv().await.unwrap().into_vec()[0].to_vec()),
            id0
        );

        let id1 = EngineId::from_engine_index(1);
        let mut opts1 = zeromq::SocketOptions::default();
        opts1.peer_identity(zeromq::util::PeerIdentity::try_from(id1.clone()).unwrap());
        let mut dealer1 = DealerSocket::with_options(opts1);
        dealer1.connect(&endpoint.to_string()).await.unwrap();
        dealer1.send("ready-1".into()).await.unwrap();
        assert_eq!(
            EngineId::from(router.recv().await.unwrap().into_vec()[0].to_vec()),
            id1
        );

        let (mut input_send_fill, _input_recv) = router.split();
        let inner = Arc::new(
            ClientInner::new(
                input_send_fill.clone(),
                Handle::current(),
                "test-model".to_string(),
                0,
                false,
                &[
                    ConnectedEngine {
                        engine_id: id0.clone(),
                        ready_response: default_ready_response(),
                    },
                    ConnectedEngine {
                        engine_id: id1.clone(),
                        ready_response: default_ready_response(),
                    },
                ],
            )
            .with_input_send_timeout(Duration::from_millis(80)),
        );

        // Saturate only DP rank 1 while DP rank 0 stays healthy.
        saturate_peer_send_queue(&mut input_send_fill, &id1).await;

        let (routed0, mut rx0) = inner
            .register_request("req-e0".to_string(), None, Some(0), NonZeroU32::MIN, None)
            .unwrap();
        let (routed1, mut rx1) = inner
            .register_request("req-e1".to_string(), None, Some(1), NonZeroU32::MIN, None)
            .unwrap();
        assert_eq!(routed0, id0, "req-e0 must route explicitly to DP rank 0");
        assert_eq!(routed1, id1, "req-e1 must route explicitly to DP rank 1");

        let err = inner.do_abort_requests(&id1, &["req-e1".to_string()]).await.unwrap_err();
        assert!(matches!(err, Error::Shared(_)));
        assert!(!inner.is_healthy());
        assert!(matches!(
            inner.health_error().as_deref(),
            Some(Error::InputSendTimeout { engine_id, .. }) if *engine_id == id1
        ));
        assert!(matches!(rx0.try_recv(), Ok(Err(Error::Shared(_)))));
        assert!(matches!(rx1.try_recv(), Ok(Err(Error::Shared(_)))));
        drop(dealer0);
        drop(dealer1);
    }
}
