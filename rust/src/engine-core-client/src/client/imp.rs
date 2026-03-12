use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use futures::Stream;
use futures::stream::FusedStream;
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

    /// Remove requests that the engine declared finished through the batched
    /// `finished_requests` side channel.
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RequestStreamState {
    Running,
    Finished,
    ClosedWithError,
    UnexpectedClose,
}

/// Stream of raw engine-core outputs for one request.
///
/// The stream yields only `EngineCoreOutput` values whose `request_id` matches
/// the originating `add_request()` call. Normal request completion is expected
/// to include a final output object whose `finish_reason` is non-`None`.
pub struct RequestOutputStream {
    pub(super) request_id: String,
    pub(super) auto_abort_tx: mpsc::UnboundedSender<String>,
    pub(super) state: RequestStreamState,
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
        if self.is_terminated() {
            return Poll::Ready(None);
        }

        match Pin::new(&mut self.rx).poll_recv(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Some(item)) => {
                match &item {
                    Ok(output) => {
                        if output.finished() {
                            self.state = RequestStreamState::Finished;
                        }
                    }
                    Err(_) => {
                        self.state = RequestStreamState::ClosedWithError;
                    }
                }
                Poll::Ready(Some(item))
            }
            Poll::Ready(None) => {
                self.state = RequestStreamState::UnexpectedClose;

                Poll::Ready(Some(Err(Error::RequestStreamClosed {
                    request_id: self.request_id.clone(),
                })))
            }
        }
    }
}

impl FusedStream for RequestOutputStream {
    fn is_terminated(&self) -> bool {
        !matches!(self.state, RequestStreamState::Running)
    }
}

impl Drop for RequestOutputStream {
    fn drop(&mut self) {
        if self.is_terminated() {
            return;
        }

        let request_id = self.request_id.clone();
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
        let should_abort = {
            let mut state = inner.state.lock();
            let removed = state.remove_request(&request_id).is_some();
            removed && state.is_running()
        };
        if !should_abort {
            debug!(request_id, "skip auto-abort for inactive request");
            continue;
        }

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
