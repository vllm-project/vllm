use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use arc_swap::ArcSwapOption;
use parking_lot::Mutex;
use thiserror_ext::AsReport as _;
use tokio::sync::mpsc;
use tracing::{debug, info, trace, warn};
use vllm_metrics::METRICS;
use zeromq::RouterSendHalf;

use crate::client::state::{OutputReceiver, RequestRegistry, UtilityReceiver, UtilityRegistry};
use crate::client::stream::EngineCoreStreamOutput;
use crate::client::{AbortCause, AbortRequest};
use crate::error::{client_closed, dispatcher_closed, unexpected_dispatcher_output};
use crate::metrics::{LoraInfoExporter, record_scheduler_stats};
use crate::protocol::stats::SchedulerStats;
use crate::protocol::utility::UtilityOutput;
use crate::protocol::{
    ClassifiedEngineCoreOutputs, EngineCoreOutput, EngineCoreOutputs, EngineCoreRequestType,
    encode_msgpack,
};
use crate::transport::{ConnectedEngine, EngineId};
use crate::{Error, Result, transport};

pub(crate) struct ClientInner {
    input_send: RouterSendHalf,
    model_name: String,
    request_reg: Mutex<RequestRegistry>,
    utility_reg: Mutex<UtilityRegistry>,
    health_error: ArcSwapOption<Error>,
}

impl ClientInner {
    /// Create a new instance with the given input send half after the startup
    /// handshake completes.
    pub fn new(
        input_send: RouterSendHalf,
        model_name: String,
        engines: &[ConnectedEngine],
    ) -> Self {
        Self {
            input_send,
            model_name,
            request_reg: Mutex::new(RequestRegistry::new(engines)),
            utility_reg: Mutex::new(UtilityRegistry::default()),
            health_error: ArcSwapOption::empty(),
        }
    }

    /// Get the model name associated with this client used for metrics
    /// labeling.
    pub fn model_name(&self) -> &str {
        &self.model_name
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
    ) -> Result<(EngineId, OutputReceiver)> {
        let mut registry = self.request_reg.lock();
        if registry.is_closed() {
            return Err(self.closed_error());
        }
        registry.register(request_id, lora_name, data_parallel_rank)
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

    /// Obtain stream senders for a whole engine output batch with one registry
    /// lock acquisition.
    pub fn take_senders_for_outputs<'a>(
        &self,
        outputs: impl IntoIterator<Item = &'a EngineCoreOutput>,
    ) -> Vec<Option<mpsc::UnboundedSender<Result<EngineCoreStreamOutput>>>> {
        self.request_reg.lock().senders_for_outputs(outputs)
    }

    /// Remove a batch of requests that have finished or aborted, returning
    /// their stream senders.
    pub fn finish_requests<'a>(
        &self,
        request_ids: impl IntoIterator<Item = &'a String>,
    ) -> Vec<mpsc::UnboundedSender<Result<EngineCoreStreamOutput>>> {
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
        let request_senders = self.request_reg.lock().close();
        let utility_senders = self.utility_reg.lock().close();

        // Notify all ongoing requests that the client is closed.
        for sender in request_senders {
            let _ = sender.send(Err(Error::Shared(persistent_error.clone())));
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

    /// Send the given message to the engine. The request should be first
    /// registered via `register_request()` to ensure the request stream is
    /// tracked.
    pub async fn send_to_engine<T>(
        &self,
        engine_id: &EngineId,
        request_type: EngineCoreRequestType,
        payload: &T,
    ) -> Result<()>
    where
        T: serde::Serialize + std::fmt::Debug,
    {
        // TODO: for `EngineCoreRequest`, split outbound tensor raw views into aux
        // frames instead of always producing a single msgpack frame.
        let payload = encode_msgpack(payload)?;
        let mut input_send = self.input_send.clone();
        transport::send_message(&mut input_send, engine_id, request_type.to_frame(), payload)
            .await?;
        Ok(())
    }

    /// Handle an abort request by sending the abort message to the engine.
    pub async fn do_abort_requests(
        &self,
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
            }
        }
    }
}

/// Background loop that listens for engine-core outputs and dispatches them to
/// the corresponding request streams based on their `request_id`.
pub(crate) async fn run_output_dispatcher_loop(
    inner: Arc<ClientInner>,
    mut output_rx: mpsc::Receiver<Result<EngineCoreOutputs>>,
) {
    let mut lora_info = LoraInfoExporter::default();

    let result: Result<()> = async {
        loop {
            let outputs = match output_rx.recv().await {
                Some(outputs) => outputs,
                None => Err(dispatcher_closed!(
                    "engine-core output dispatcher channel closed"
                )),
            }?;

            match outputs.classify() {
                ClassifiedEngineCoreOutputs::RequestBatch(batch) => {
                    let senders = inner.take_senders_for_outputs(&batch.outputs);
                    for (output, sender) in batch.outputs.into_iter().zip(senders) {
                        let request_id = output.request_id.clone();
                        let Some(sender) = sender else {
                            debug!(request_id, "dropping output for inactive request");
                            continue;
                        };

                        let wrapped_output = EngineCoreStreamOutput {
                            engine_index: batch.engine_index,
                            timestamp: batch.timestamp,
                            output,
                        };
                        if sender.send(Ok(wrapped_output)).is_err() {
                            debug!(request_id, "request output stream receiver dropped");
                        }
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
                        record_scheduler_stats(
                            &METRICS.scheduler,
                            inner.model_name(),
                            batch.engine_index,
                            scheduler_stats,
                        );
                    }

                    // The engine's scheduler stats never carry adapter names;
                    // the gauge is derived from the registry's frontend-side
                    // request tracking instead.
                    let (running, waiting) = inner.lora_adapter_states();
                    lora_info.update(&METRICS.scheduler, running, waiting);
                }
                ClassifiedEngineCoreOutputs::Utility(utility) => {
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
                other @ (ClassifiedEngineCoreOutputs::DpControl { .. }
                | ClassifiedEngineCoreOutputs::Other(_)) => {
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
    use zeromq::{RouterSocket, Socket};

    use super::*;
    use crate::mock_engine::default_ready_response;

    async fn test_inner() -> ClientInner {
        let mut socket = RouterSocket::new();
        socket.bind("tcp://127.0.0.1:0").await.unwrap();
        let (send, _) = socket.split();
        ClientInner::new(
            send,
            "test-model".to_string(),
            &[ConnectedEngine {
                engine_id: EngineId::from(b"engine-0"),
                ready_response: default_ready_response(),
            }],
        )
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn close_registries_records_first_health_error_only() {
        let inner = test_inner().await;

        inner.close_registries(Arc::new(Error::EngineCoreDead));
        assert!(!inner.is_healthy());
        assert!(matches!(
            inner.health_error().as_deref(),
            Some(Error::EngineCoreDead)
        ));

        inner.close_registries(Arc::new(client_closed!("shutdown")));
        assert!(matches!(
            inner.health_error().as_deref(),
            Some(Error::EngineCoreDead)
        ));
    }
}
