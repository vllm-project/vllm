use std::slice;
use std::sync::Arc;

use parking_lot::Mutex;
use thiserror_ext::AsReport as _;
use tokio::sync::{Mutex as AsyncMutex, mpsc};
use tracing::{debug, info, trace, warn};
use vllm_metrics::METRICS;
use zeromq::RouterSendHalf;

use crate::client::state::{ClientClosedState, OutputReceiver, RequestRegistry};
use crate::client::stream::EngineCoreStreamOutput;
use crate::metrics::record_scheduler_stats;
use crate::protocol::{
    ClassifiedEngineCoreOutputs, EngineCoreOutput, EngineCoreOutputs, EngineCoreRequestType,
    encode_msgpack,
};
use crate::{Error, Result, transport};

pub(crate) struct ClientInner {
    input_send: AsyncMutex<Option<RouterSendHalf>>,
    model_name: String,
    request_reg: Mutex<RequestRegistry>,
}

impl ClientInner {
    /// Create a new instance with the given input send half after the startup handshake completes.
    pub fn new(input_send: RouterSendHalf, model_name: String) -> Self {
        Self {
            input_send: AsyncMutex::new(Some(input_send)),
            model_name,
            request_reg: Mutex::new(RequestRegistry::default()),
        }
    }

    /// Get the model name associated with this client used for metrics labeling.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Register a newly added request. Return the per-request output channel bound to its
    /// `request_id`.
    pub fn register_request(&self, request_id: String) -> Result<OutputReceiver> {
        self.request_reg.lock().register(request_id)
    }

    /// Undo a request registration when `add_request()` fails.
    pub fn rollback_request(&self, request_id: &str) {
        let _ = self.request_reg.lock().remove(request_id);
    }

    /// Filter the given request IDs to the subset that are still tracked as active and can be
    /// aborted.
    pub fn abortable_request_ids(&self, request_ids: &[String]) -> Result<Vec<String>> {
        self.request_reg.lock().abortable_request_ids(request_ids)
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

    /// Close all active request streams with an error message based on the given closed state.
    pub fn close_requests(&self, closed_state: ClientClosedState) {
        let error = Arc::new(closed_state.error());
        let senders = self.request_reg.lock().close(closed_state);

        // Notify all ongoing requests that the client is closed.
        for sender in senders {
            let _ = sender.send(Err(Error::Shared(error.clone())));
        }
    }

    /// Send the given message to the engine. The request should be first registered via
    /// `register_request()` to ensure the request stream is tracked.
    pub async fn send_to_engine<T>(
        &self,
        engine_identity: &[u8],
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
            .ok_or_else(|| Error::ControlClosed("input sender already shut down".to_string()))?;
        transport::send_message(
            input_send,
            engine_identity,
            request_type.as_frame(),
            payload,
        )
        .await?;
        Ok(())
    }

    /// Handle an abort request by sending the abort message to the engine.
    pub async fn do_abort_requests(
        &self,
        engine_identity: &[u8],
        request_ids: &[String],
    ) -> Result<()> {
        self.send_to_engine(engine_identity, EngineCoreRequestType::Abort, &request_ids)
            .await
    }

    /// Shut down by closing all active request streams and then closing the input socket to signal
    /// the engine that no more messages will be sent.
    pub async fn shutdown(&self) {
        self.close_requests(ClientClosedState::ClientShutdown {
            reason: "engine-core client shut down".to_string(),
        });
        self.input_send.lock().await.take();
    }
}

/// Background loop that listens for request IDs to abort and sends abort messages to the engine.
/// This is used to implement the auto-abort behavior when a request stream is dropped without being
/// properly terminated.
pub(crate) async fn run_abort_loop(
    inner: Arc<ClientInner>,
    engine_identity: Vec<u8>,
    mut abort_rx: mpsc::UnboundedReceiver<String>,
) {
    // TODO: receive and abort requests in batch
    while let Some(request_id) = abort_rx.recv().await {
        let should_abort = {
            let mut registry = inner.request_reg.lock();
            let removed = registry.remove(&request_id).is_some();
            removed && registry.is_running()
        };
        if !should_abort {
            debug!(request_id, "skip auto-abort for inactive request");
            continue;
        }
        info!(request_id, "auto-aborting request due to dropped stream");

        if let Err(error) = inner
            .do_abort_requests(&engine_identity, slice::from_ref(&request_id))
            .await
        {
            warn!(
                request_id,
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
    while let Some(outputs) = output_rx.recv().await {
        let outputs = match outputs {
            Ok(outputs) => outputs,
            Err(error) => {
                let reason = error.to_report_string();
                warn!(reason, "engine-core output loop failed");
                inner.close_requests(ClientClosedState::DispatcherFailed { reason });
                return;
            }
        };

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

                // The sender for normally-finished requests should have already been removed from
                // the registry when their final output was dispatched above. This serves as a
                // safety net to capture any requests marked as finished by the engine.
                if let Some(finished_requests) = batch.finished_requests.as_ref() {
                    for request_id in finished_requests {
                        trace!(request_id, "request completed via finished_requests");
                    }
                    drop(inner.finish_requests(finished_requests));
                }

                if let Some(scheduler_stats) = batch.scheduler_stats.as_ref() {
                    record_scheduler_stats(
                        &METRICS,
                        inner.model_name(),
                        batch.engine_index,
                        scheduler_stats,
                    );
                }
            }
            ClassifiedEngineCoreOutputs::Other(other) => {
                warn!(outputs = ?other, "ignoring non-request engine-core output");
                // TODO: handle other outputs, like utility call
            }
        }
    }

    inner.close_requests(ClientClosedState::DispatcherFailed {
        reason: "engine-core output dispatcher channel closed".to_string(),
    });
}
