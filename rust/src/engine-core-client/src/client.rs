use std::sync::Arc;
use std::time::Duration;

use thiserror_ext::AsReport;
use tokio::sync::{Mutex, mpsc};
use tokio_util::task::AbortOnDropHandle;
use tracing::debug;

use crate::error::{Error, Result};
use crate::protocol::handshake::ReadyMessage;
use crate::protocol::{
    ClassifiedEngineCoreOutputs, EngineCoreOutputs, EngineCoreRequest, EngineCoreRequestType,
    encode_msgpack,
};
use crate::state::RequestTracker;
use crate::transport;

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

/// Default ZMQ-based implementation that talks directly to a Python
/// `EngineCoreProc`.
pub struct EngineCoreClient {
    config: EngineCoreClientConfig,
    input_address: String,
    output_address: String,
    engine_identity: Vec<u8>,
    input_send: Arc<Mutex<Option<zeromq::RouterSendHalf>>>,
    output_rx: mpsc::Receiver<Result<EngineCoreOutputs>>,
    output_task: Option<AbortOnDropHandle<()>>,
    state: Arc<Mutex<RequestTracker>>,
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
        let output_task =
            AbortOnDropHandle::new(transport::spawn_output_loop(connected.output_socket, tx));

        Ok(Self {
            config,
            input_address: connected.input_address,
            output_address: connected.output_address,
            engine_identity: connected.engine_identity,
            input_send: Arc::new(Mutex::new(Some(connected.input_send))),
            output_rx: rx,
            output_task: Some(output_task),
            state: Arc::new(Mutex::new(RequestTracker::default())),
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

    async fn send<T>(&self, request_type: EngineCoreRequestType, payload: &T) -> Result<()>
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
            &self.engine_identity,
            request_type.as_frame(),
            payload,
        )
        .await?;
        debug!(?request_type, "sent engine-core message");
        Ok(())
    }
}

// Client API implementation.
impl EngineCoreClient {
    /// Add a new request to the engine.
    pub async fn add_request(&self, mut req: EngineCoreRequest) -> Result<()> {
        req.client_index = self.config.client_index;
        req.validate()?;
        debug!(
            request_id = %req.request_id,
            client_index = req.client_index,
            request = ?req,
            "sending add request"
        );

        {
            let mut state = self.state.lock().await;
            state.insert(req.request_id.clone());
        }

        self.send(EngineCoreRequestType::Add, &req).await
    }

    /// Abort currently in-flight requests by request ID.
    pub async fn abort_requests(&self, ids: &[String]) -> Result<()> {
        let abortable = {
            let state = self.state.lock().await;
            state.retain_abortable(ids)
        };

        debug!(request_ids = ?ids, abortable_request_ids = ?abortable, "sending abort request ids");

        if abortable.is_empty() {
            return Ok(());
        }

        self.send(EngineCoreRequestType::Abort, &abortable).await
    }

    /// Wait for the next batch of outputs from the engine.
    pub async fn next_output(&mut self) -> Result<EngineCoreOutputs> {
        let outputs = self.output_rx.recv().await.ok_or(Error::OutputClosed)??;
        {
            let mut state = self.state.lock().await;
            state.observe_outputs(&outputs);
        }
        debug!(outputs = ?outputs, "delivering engine-core outputs");
        Ok(outputs)
    }

    /// Wait for the next batch of outputs and immediately classify the raw
    /// wire message into a more semantic Rust enum.
    pub async fn next_classified_output(&mut self) -> Result<ClassifiedEngineCoreOutputs> {
        self.next_output().await.map(EngineCoreOutputs::classify)
    }

    /// Shut down local client tasks and close transport state.
    pub async fn shutdown(&mut self) -> Result<()> {
        debug!("shutting down engine-core client");
        self.input_send.lock().await.take();

        if let Some(task) = self.output_task.take() {
            task.abort();
        }
        debug!("engine-core client shut down");
        Ok(())
    }
}
