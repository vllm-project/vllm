use std::pin::Pin;
use std::sync::Arc;
use std::sync::Weak;
use std::task::Context;
use std::task::Poll;

use futures::Stream;
use parking_lot::Mutex;
use thiserror_ext::AsReport as _;
use tokio::sync::{Mutex as AsyncMutex, mpsc};
use tracing::debug;
use tracing::warn;

use crate::Error;
use crate::Result;
use crate::protocol::ClassifiedEngineCoreOutputs;
use crate::protocol::EngineCoreOutputs;
use crate::protocol::EngineCoreRequestType;
use crate::protocol::encode_msgpack;
use crate::transport;
use crate::{
    client::state::{ClientClosedState, RequestRegistry, RequestStreamReceiver},
    protocol::EngineCoreOutput,
};

pub(super) struct ClientInner {
    pub(super) input_send: AsyncMutex<Option<zeromq::RouterSendHalf>>,
    pub(super) state: Mutex<RequestRegistry>,
}

impl ClientInner {
    /// Install local request lifecycle state before the `Add` message is sent,
    /// so output dispatch can start immediately if the engine replies fast.
    pub fn register_request(&self, request_id: String) -> Result<RequestStreamReceiver> {
        self.state.lock().register(request_id)
    }

    /// Undo local lifecycle state when the outbound `Add` send fails after
    /// registration.
    pub fn rollback_request(&self, request_id: &str) {
        self.state.lock().rollback(request_id);
    }

    /// Keep explicit aborts limited to requests that are still locally tracked
    /// as in flight.
    pub fn abortable_request_ids(&self, request_ids: &[String]) -> Result<Vec<String>> {
        self.state.lock().abortable_request_ids(request_ids)
    }

    /// Resolve the per-request stream sender for one output, removing the
    /// request first if this output already completes its lifecycle.
    pub fn take_sender_for_output(
        &self,
        output: &EngineCoreOutput,
    ) -> Option<mpsc::UnboundedSender<Result<EngineCoreOutput>>> {
        self.state.lock().sender_for_output(output)
    }

    /// Remove requests that finished through the batched `finished_requests`
    /// side channel rather than an inline final output.
    pub fn finish_requests<'a>(
        &self,
        request_ids: impl IntoIterator<Item = &'a String>,
    ) -> Vec<mpsc::UnboundedSender<Result<EngineCoreOutput>>> {
        self.state.lock().finish_requests(request_ids)
    }

    /// Close all active request streams with the same terminal lifecycle error.
    pub fn close_requests(&self, closed_state: ClientClosedState) {
        let error = Arc::new(closed_state.error());
        let senders = self.state.lock().close(closed_state);
        for sender in senders {
            let _ = sender.send(Err(Error::Shared(error.clone())));
        }
    }

    /// Remove a request whose consumer disappeared and report whether the
    /// engine still needs a best-effort abort for it.
    pub fn drop_request_stream(&self, request_id: &str) -> bool {
        let mut state = self.state.lock();
        let removed = state.remove_request(request_id).is_some();
        removed && state.is_running()
    }

    /// Send one lifecycle control message after local request bookkeeping has
    /// already been updated.
    pub async fn send_to_engine<T>(
        &self,
        engine_identity: &[u8],
        request_type: EngineCoreRequestType,
        payload: &T,
    ) -> Result<()>
    where
        T: serde::Serialize + std::fmt::Debug,
    {
        debug!(?request_type, request = ?payload, "encoding engine-core message");
        let payload = encode_msgpack(payload)?;
        debug!(
            ?request_type,
            payload_bytes = payload.len(),
            "sending engine-core message"
        );
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
        debug!(?request_type, "sent engine-core message");
        Ok(())
    }
}

/// Stream of raw engine-core outputs for one request.
///
/// The stream yields only `EngineCoreOutput` values whose `request_id` matches
/// the originating `add_request()` call. If the engine finishes the request via
/// `finished_requests` without emitting a final output object, the stream ends
/// with EOF and does not synthesize an extra item.
pub struct RequestOutputStream {
    pub(super) request_id: String,
    pub(super) auto_abort_tx: mpsc::UnboundedSender<String>,
    pub(super) inner: Weak<ClientInner>,
    pub(super) rx: RequestStreamReceiver,
}

impl RequestOutputStream {
    /// Return the engine-core `request_id` bound to this stream.
    pub fn request_id(&self) -> &str {
        &self.request_id
    }
}

impl Stream for RequestOutputStream {
    type Item = Result<EngineCoreOutput>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.rx).poll_recv(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Some(item)) => Poll::Ready(Some(item)),
            Poll::Ready(None) => {
                let request_id = std::mem::take(&mut self.request_id);
                if let Some(inner) = self.inner.upgrade()
                    && inner.drop_request_stream(&request_id)
                {
                    return Poll::Ready(Some(Err(Error::RequestStreamClosed { request_id })));
                }
                Poll::Ready(None)
            }
        }
    }
}

impl Drop for RequestOutputStream {
    fn drop(&mut self) {
        if self.request_id.is_empty() {
            return;
        }

        let Some(inner) = self.inner.upgrade() else {
            return;
        };

        let request_id = std::mem::take(&mut self.request_id);
        if !inner.drop_request_stream(&request_id) {
            return;
        }

        if self.auto_abort_tx.send(request_id.clone()).is_err() {
            warn!(
                request_id,
                "auto-abort worker already shut down; skip auto-abort"
            );
        }
    }
}

/// Serialize auto-aborts for request streams that were dropped before the
/// engine declared completion.
pub(super) async fn run_auto_abort_loop(
    inner: Arc<ClientInner>,
    engine_identity: Vec<u8>,
    mut auto_abort_rx: mpsc::UnboundedReceiver<String>,
) {
    while let Some(request_id) = auto_abort_rx.recv().await {
        if let Err(error) = inner
            .send_to_engine(
                &engine_identity,
                EngineCoreRequestType::Abort,
                &vec![request_id.clone()],
            )
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

/// Consume raw engine outputs and dispatch only request-scoped lifecycle
/// events into the matching per-request streams.
pub(super) async fn run_output_dispatcher_loop(
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
                    let finished = output.finished();
                    let Some(sender) = inner.take_sender_for_output(&output) else {
                        debug!(request_id, "dropping output for inactive request");
                        continue;
                    };

                    if sender.send(Ok(output)).is_err() {
                        debug!(request_id, "request output stream receiver dropped");
                    }

                    if finished {
                        debug!(request_id, "request completed via final output");
                    }
                }

                if let Some(finished_requests) = batch.finished_requests.as_ref() {
                    for request_id in finished_requests {
                        debug!(request_id, "request completed via finished_requests");
                    }
                    drop(inner.finish_requests(finished_requests));
                }

                if batch.scheduler_stats.is_some() {
                    debug!("ignoring scheduler stats in request batch");
                }
            }
            ClassifiedEngineCoreOutputs::Other(other) => {
                debug!(outputs = ?other, "ignoring non-request engine-core output");
            }
        }
    }

    inner.close_requests(ClientClosedState::DispatcherFailed {
        reason: "engine-core output dispatcher channel closed".to_string(),
    });
}
