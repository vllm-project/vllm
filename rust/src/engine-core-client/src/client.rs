use std::pin::Pin;
use std::sync::{Arc, Weak};
use std::task::{Context, Poll};
use std::time::Duration;

use futures::Stream;
use parking_lot::Mutex;
use thiserror_ext::AsReport;
use tokio::sync::{Mutex as AsyncMutex, mpsc};
use tokio_util::task::AbortOnDropHandle;
use tracing::{debug, warn};

use crate::error::{Error, Result};
use crate::protocol::handshake::ReadyMessage;
use crate::protocol::{
    EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest, EngineCoreRequestType, encode_msgpack,
};
use crate::state::{ClientClosedState, RequestRegistry, RequestStreamReceiver};
use crate::transport;

const CLIENT_SHUTDOWN_REASON: &str = "engine-core client shut down";

/// Configuration for connecting a Rust frontend client to an already running
/// Python `EngineCoreProc`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineCoreClientConfig {
    /// Startup handshake address that the Python engine connects to first.
    pub handshake_address: String,
    /// Local host/interface used when allocating the frontend input/output addresses.
    pub local_host: String,
    /// Timeout while waiting for each step of the startup handshake.
    pub ready_timeout: Duration,
    /// Frontend client index stamped onto every request.
    pub client_index: u32,
}

impl EngineCoreClientConfig {
    pub fn new(handshake_address: impl Into<String>) -> Self {
        Self {
            handshake_address: handshake_address.into(),
            local_host: "127.0.0.1".to_string(),
            ready_timeout: Duration::from_secs(30),
            client_index: 0,
        }
    }
}

struct ClientInner {
    input_send: AsyncMutex<Option<zeromq::RouterSendHalf>>,
    state: Mutex<RequestRegistry>,
}

impl ClientInner {
    fn register_request(&self, request_id: String) -> Result<RequestStreamReceiver> {
        self.state.lock().register(request_id)
    }

    fn rollback_request(&self, request_id: &str) {
        self.state.lock().rollback(request_id);
    }

    fn abortable_request_ids(&self, request_ids: &[String]) -> Result<Vec<String>> {
        self.state.lock().abortable_request_ids(request_ids)
    }

    fn take_sender_for_output(
        &self,
        output: &EngineCoreOutput,
    ) -> Option<mpsc::UnboundedSender<Result<EngineCoreOutput>>> {
        self.state.lock().sender_for_output(output)
    }

    fn finish_requests<'a>(
        &self,
        request_ids: impl IntoIterator<Item = &'a String>,
    ) -> Vec<mpsc::UnboundedSender<Result<EngineCoreOutput>>> {
        self.state.lock().finish_requests(request_ids)
    }

    fn close_requests(&self, closed_state: ClientClosedState) {
        let error = Arc::new(closed_state.error());
        let senders = self.state.lock().close(closed_state);
        for sender in senders {
            let _ = sender.send(Err(Error::Shared(error.clone())));
        }
    }

    fn drop_request_stream(&self, request_id: &str) -> bool {
        let mut state = self.state.lock();
        let removed = state.remove_request(request_id).is_some();
        removed && state.is_running()
    }

    async fn send_to_engine<T>(
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
        let input_send = guard.as_mut().ok_or_else(|| {
            Error::ControlClosed(
                std::io::Error::other("input sender already shut down").to_report_string(),
            )
        })?;
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

    async fn abort_dropped_request(self: Arc<Self>, engine_identity: Vec<u8>, request_id: String) {
        if let Err(error) = self
            .send_to_engine(
                &engine_identity,
                EngineCoreRequestType::Abort,
                &vec![request_id.clone()],
            )
            .await
        {
            warn!(
                request_id,
                error = %error.to_report_string(),
                "failed to abort dropped request stream"
            );
        }
    }
}

/// Stream of raw engine-core outputs for one request.
///
/// The stream yields only `EngineCoreOutput` values whose `request_id` matches
/// the originating `add_request()` call. If the engine finishes the request via
/// `finished_requests` without emitting a final output object, the stream ends
/// with EOF and does not synthesize an extra item.
pub struct RequestOutputStream {
    request_id: String,
    engine_identity: Vec<u8>,
    inner: Weak<ClientInner>,
    rx: RequestStreamReceiver,
}

impl RequestOutputStream {
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

        let engine_identity = self.engine_identity.clone();
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                inner
                    .abort_dropped_request(engine_identity, request_id)
                    .await;
            });
        } else {
            warn!(
                request_id,
                "dropping request stream outside Tokio runtime; skip auto-abort"
            );
        }
    }
}

/// Default ZMQ-based implementation that talks directly to a Python
/// `EngineCoreProc`.
pub struct EngineCoreClient {
    config: EngineCoreClientConfig,
    input_address: String,
    output_address: String,
    engine_identity: Vec<u8>,
    inner: Arc<ClientInner>,
    output_task: Option<AbortOnDropHandle<()>>,
    dispatcher_task: Option<AbortOnDropHandle<()>>,
    pub ready_message: Option<ReadyMessage>,
}

impl EngineCoreClient {
    /// Connect to an already running Python engine and complete the startup handshake.
    pub async fn connect(config: EngineCoreClientConfig) -> Result<Self> {
        let connected = transport::connect(
            &config.handshake_address,
            &config.local_host,
            config.ready_timeout,
        )
        .await?;
        Self::from_connected(config, connected)
    }

    fn from_connected(
        config: EngineCoreClientConfig,
        connected: transport::ConnectedTransport,
    ) -> Result<Self> {
        let (tx, rx) = mpsc::channel(64);
        let inner = Arc::new(ClientInner {
            input_send: AsyncMutex::new(Some(connected.input_send)),
            state: Mutex::new(RequestRegistry::default()),
        });
        let output_task =
            AbortOnDropHandle::new(transport::spawn_output_loop(connected.output_socket, tx));
        let dispatcher_task = AbortOnDropHandle::new(spawn_dispatcher_loop(inner.clone(), rx));

        Ok(Self {
            config,
            input_address: connected.input_address,
            output_address: connected.output_address,
            engine_identity: connected.engine_identity,
            inner,
            output_task: Some(output_task),
            dispatcher_task: Some(dispatcher_task),
            ready_message: Some(connected.ready_message),
        })
    }

    pub fn input_address(&self) -> &str {
        &self.input_address
    }

    pub fn output_address(&self) -> &str {
        &self.output_address
    }

    pub fn engine_identity(&self) -> &[u8] {
        &self.engine_identity
    }
}

// Client API implementation.
impl EngineCoreClient {
    /// Add a new request to the engine and return a per-request raw output
    /// stream.
    pub async fn add_request(&self, mut req: EngineCoreRequest) -> Result<RequestOutputStream> {
        req.client_index = self.config.client_index;
        req.validate()?;
        debug!(
            request_id = %req.request_id,
            client_index = req.client_index,
            request = ?req,
            "sending add request"
        );

        let request_id = req.request_id.clone();
        let rx = self.inner.register_request(request_id.clone())?;
        if let Err(error) = self
            .inner
            .send_to_engine(&self.engine_identity, EngineCoreRequestType::Add, &req)
            .await
        {
            self.inner.rollback_request(&request_id);
            return Err(error);
        }

        Ok(RequestOutputStream {
            request_id,
            engine_identity: self.engine_identity.clone(),
            inner: Arc::downgrade(&self.inner),
            rx,
        })
    }

    /// Abort currently in-flight requests by request ID.
    pub async fn abort_requests(&self, ids: &[String]) -> Result<()> {
        let abortable = self.inner.abortable_request_ids(ids)?;

        debug!(request_ids = ?ids, abortable_request_ids = ?abortable, "sending abort request ids");

        if abortable.is_empty() {
            return Ok(());
        }

        self.inner
            .send_to_engine(
                &self.engine_identity,
                EngineCoreRequestType::Abort,
                &abortable,
            )
            .await
    }

    /// Shut down local client tasks and close transport state.
    pub async fn shutdown(&mut self) -> Result<()> {
        debug!("shutting down engine-core client");
        self.inner
            .close_requests(ClientClosedState::ClientShutdown {
                reason: CLIENT_SHUTDOWN_REASON.to_string(),
            });
        self.inner.input_send.lock().await.take();

        if let Some(task) = self.dispatcher_task.take() {
            task.abort();
        }
        if let Some(task) = self.output_task.take() {
            task.abort();
        }
        debug!("engine-core client shut down");
        Ok(())
    }
}

fn spawn_dispatcher_loop(
    inner: Arc<ClientInner>,
    mut output_rx: mpsc::Receiver<Result<EngineCoreOutputs>>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
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
                crate::protocol::ClassifiedEngineCoreOutputs::RequestBatch(batch) => {
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
                crate::protocol::ClassifiedEngineCoreOutputs::Other(other) => {
                    debug!(outputs = ?other, "ignoring non-request engine-core output");
                }
            }
        }

        inner.close_requests(ClientClosedState::DispatcherFailed {
            reason: "engine-core output dispatcher channel closed".to_string(),
        });
    })
}
