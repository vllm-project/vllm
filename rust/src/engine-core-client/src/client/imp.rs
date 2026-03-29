use std::collections::BTreeMap;
use std::slice;
use std::sync::Arc;

use arc_swap::ArcSwapOption;
use parking_lot::Mutex;
use thiserror_ext::AsReport as _;
use tokio::sync::{Mutex as AsyncMutex, mpsc};
use tracing::{debug, info, trace, warn};
use vllm_metrics::METRICS;
use zeromq::RouterSendHalf;

use crate::client::state::{OutputReceiver, RequestRegistry, UtilityReceiver, UtilityRegistry};
use crate::client::stream::EngineCoreStreamOutput;
use crate::error::{
    client_closed, control_closed, dispatcher_closed, unexpected_dispatcher_output,
};
use crate::metrics::record_scheduler_stats;
use crate::protocol::{
    ClassifiedEngineCoreOutputs, EngineCoreOutput, EngineCoreOutputs, EngineCoreRequestType,
    UtilityOutput, encode_msgpack,
};
use crate::transport::{ConnectedEngine, EngineId};
use crate::{Error, Result, transport};

pub(crate) struct ClientInner {
    input_send: AsyncMutex<Option<RouterSendHalf>>,
    model_name: String,
    request_reg: Mutex<RequestRegistry>,
    utility_reg: Mutex<UtilityRegistry>,
    health_error: ArcSwapOption<Error>,
}

impl ClientInner {
    /// Create a new instance with the given input send half after the startup handshake completes.
    pub fn new(
        input_send: RouterSendHalf,
        model_name: String,
        engines: &[ConnectedEngine],
    ) -> Self {
        Self {
            input_send: AsyncMutex::new(Some(input_send)),
            model_name,
            request_reg: Mutex::new(RequestRegistry::new(engines)),
            utility_reg: Mutex::new(UtilityRegistry::default()),
            health_error: ArcSwapOption::empty(),
        }
    }

    /// Get the model name associated with this client used for metrics labeling.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Register a newly added request. Return the selected engine id and the per-request
    /// output channel bound to its `request_id`.
    pub fn register_request(&self, request_id: String) -> Result<(EngineId, OutputReceiver)> {
        let mut registry = self.request_reg.lock();
        if registry.is_closed() {
            return Err(self.closed_error());
        }
        registry.register(request_id)
    }

    /// Allocate the next utility `call_id` and register its waiting receiver.
    pub fn allocate_and_register_utility_call(&self) -> Result<(i64, UtilityReceiver)> {
        let mut registry = self.utility_reg.lock();
        if registry.is_closed() {
            return Err(self.closed_error());
        }
        Ok(registry.allocate_and_register())
    }

    /// Undo a request registration when `add_request()` fails.
    pub fn rollback_request(&self, request_id: &str) {
        let _ = self.request_reg.lock().remove(request_id);
    }

    /// Filter the given request IDs to the subset that are still tracked as active and can be
    /// aborted, grouped by the engine that originally accepted them.
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

    /// Obtain the stream sender for one output. If it indicates the request is finished, it will be
    /// removed from the registry.
    pub fn take_sender_for_output(
        &self,
        output: &EngineCoreOutput,
    ) -> Option<mpsc::UnboundedSender<Result<EngineCoreStreamOutput>>> {
        self.request_reg.lock().sender_for_output(output)
    }

    /// Remove a batch of requests that have finished or aborted, returning their stream senders.
    pub fn finish_requests<'a>(
        &self,
        request_ids: impl IntoIterator<Item = &'a String>,
    ) -> Vec<mpsc::UnboundedSender<Result<EngineCoreStreamOutput>>> {
        self.request_reg.lock().finish_many(request_ids)
    }

    /// Close all active request streams and utility calls with the first persistent health error.
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

    /// Resolve one utility output to the waiting caller. Returns `true` if a waiting caller
    /// existed.
    pub fn resolve_utility_output(&self, output: UtilityOutput) -> bool {
        let Some(sender) = self.utility_reg.lock().resolve(output.clone()) else {
            return false;
        };
        sender.send(Ok(output)).is_ok()
    }

    /// Send the given message to the engine. The request should be first registered via
    /// `register_request()` to ensure the request stream is tracked.
    pub async fn send_to_engine<T>(
        &self,
        engine_id: &EngineId,
        request_type: EngineCoreRequestType,
        payload: &T,
    ) -> Result<()>
    where
        T: serde::Serialize,
    {
        let payload = encode_msgpack(payload)?;
        let mut guard = self.input_send.lock().await;
        let input_send = guard
            .as_mut()
            .ok_or_else(|| control_closed!("input sender already shut down"))?;
        transport::send_message(input_send, engine_id, request_type.to_frame(), payload).await?;
        Ok(())
    }

    /// Handle an abort request by sending the abort message to the engine.
    pub async fn do_abort_requests(
        &self,
        engine_id: &EngineId,
        request_ids: &[String],
    ) -> Result<()> {
        self.send_to_engine(engine_id, EngineCoreRequestType::Abort, &request_ids)
            .await
    }

    /// Shut down by closing all active request streams and then closing the input socket to signal
    /// the engine that no more messages will be sent.
    pub async fn shutdown(&self) {
        self.close_registries(Arc::new(client_closed!("engine-core client shut down")));
        self.input_send.lock().await.take();
    }

    /// Remove the request from the active registry for auto-abort and return the engine that the
    /// request was originally routed to, if it is still active.
    pub fn take_auto_abort_target(&self, request_id: &str) -> Option<EngineId> {
        let mut registry = self.request_reg.lock();
        let (_, engine_id) = registry.remove(request_id)?;
        if registry.is_closed() {
            return None;
        }
        Some(engine_id)
    }

    /// Publish the first persistent health error and return the sticky error recorded for this
    /// client. Later failures do not overwrite the first one so `/health` and post-close callers
    /// observe a stable cause.
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

    /// Assert there is a recorded health error and return a `Shared` variant wrapping it for error
    /// returns when the client is already closed.
    fn closed_error(&self) -> Error {
        Error::Shared(self.health_error.load_full().expect(
            "closed registry must have a recorded health error before rejecting new operations",
        ))
    }
}

/// Background loop that listens for request IDs to abort and sends abort messages to the engine.
/// This is used to implement the auto-abort behavior when a request stream is dropped without being
/// properly terminated.
pub(crate) async fn run_abort_loop(
    inner: Arc<ClientInner>,
    mut abort_rx: mpsc::UnboundedReceiver<String>,
) {
    // TODO: receive and abort requests in batch
    while let Some(request_id) = abort_rx.recv().await {
        let Some(engine_id) = inner.take_auto_abort_target(&request_id) else {
            debug!(request_id, "skip auto-abort for inactive request");
            continue;
        };
        info!(request_id, "auto-aborting request due to dropped stream");

        if let Err(error) = inner
            .do_abort_requests(&engine_id, slice::from_ref(&request_id))
            .await
        {
            warn!(
                request_id,
                ?engine_id,
                error = %error.as_report(),
                "failed to auto-abort dropped request stream"
            );
        }
    }
}

/// Background loop that listens for engine-core outputs and dispatches them to the corresponding
/// request streams based on their `request_id`.
pub(crate) async fn run_output_dispatcher_loop(
    inner: Arc<ClientInner>,
    mut output_rx: mpsc::Receiver<Result<EngineCoreOutputs>>,
) {
    let Err(error) = try {
        loop {
            let outputs = match output_rx.recv().await {
                Some(outputs) => outputs,
                None => Err(dispatcher_closed!(
                    "engine-core output dispatcher channel closed"
                )),
            }?;

            match outputs.classify() {
                ClassifiedEngineCoreOutputs::RequestBatch(batch) => {
                    for output in batch.outputs {
                        let request_id = output.request_id.clone();
                        let Some(sender) = inner.take_sender_for_output(&output) else {
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
                        record_scheduler_stats(
                            &METRICS.scheduler,
                            inner.model_name(),
                            batch.engine_index,
                            scheduler_stats,
                        );
                    }
                }
                ClassifiedEngineCoreOutputs::Utility(utility) => {
                    let call_id = utility.output.call_id;
                    if inner.resolve_utility_output(utility.output) {
                        trace!(
                            call_id,
                            engine_index = utility.engine_index,
                            "resolved utility output"
                        );
                    } else {
                        warn!(
                            call_id,
                            engine_index = utility.engine_index,
                            "dropping output for inactive utility call"
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
    };

    warn!(error = %error.as_report(), "output dispatcher exiting with error");
    inner.close_registries(Arc::new(error));
}

#[cfg(test)]
mod tests {
    use zeromq::{RouterSocket, Socket};

    use super::*;

    async fn test_inner() -> ClientInner {
        let mut socket = RouterSocket::new();
        socket.bind("tcp://127.0.0.1:0").await.unwrap();
        let (send, _) = socket.split();
        ClientInner::new(
            send,
            "test-model".to_string(),
            &[ConnectedEngine {
                engine_id: EngineId::from(b"engine-0"),
                ready_message: Default::default(),
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
