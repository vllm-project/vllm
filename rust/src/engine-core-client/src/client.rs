use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, mpsc};

use crate::error::{Error, Result};
use crate::protocol::{
    EngineCoreOutputs, EngineCoreRequest, EngineCoreRequestType, encode_msgpack,
};
use crate::state::RequestTracker;
use crate::transport;

/// Configuration for connecting a Rust frontend client to an already running
/// Python `EngineCoreProc`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZmqEngineCoreClientConfig {
    /// ROUTER bind address used for frontend -> engine requests.
    pub input_address: String,
    /// PULL connect address used for engine -> frontend outputs.
    pub output_address: String,
    /// Expected engine DEALER identity used for ready-handshake validation and routing.
    pub engine_identity: Vec<u8>,
    /// Timeout while waiting for the engine ready message.
    pub ready_timeout: Duration,
    /// Frontend client index stamped onto every request.
    pub client_index: u32,
}

impl ZmqEngineCoreClientConfig {
    pub fn new(
        input_address: impl Into<String>,
        output_address: impl Into<String>,
        engine_identity: Vec<u8>,
    ) -> Self {
        Self {
            input_address: input_address.into(),
            output_address: output_address.into(),
            engine_identity,
            ready_timeout: Duration::from_secs(30),
            client_index: 0,
        }
    }
}

/// Decoded engine ready-handshake payload.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReadyMessage {
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub local: Option<bool>,
    #[serde(default)]
    pub headless: Option<bool>,
    #[serde(default)]
    pub num_gpu_blocks: Option<u64>,
    #[serde(default)]
    pub dp_stats_address: Option<String>,
    #[serde(default)]
    pub parallel_config_hash: Option<u64>,
}

/// Minimal async engine-core client surface for the first-stage Rust frontend.
#[async_trait]
pub trait EngineCoreClient {
    /// Add a new request to the engine.
    async fn add_request(&self, req: EngineCoreRequest) -> Result<()>;
    /// Abort currently in-flight requests by request ID.
    async fn abort_requests(&self, ids: &[String]) -> Result<()>;
    /// Wait for the next batch of outputs from the engine.
    async fn next_output(&mut self) -> Result<EngineCoreOutputs>;
    /// Shut down local client tasks and close transport state.
    async fn shutdown(&mut self) -> Result<()>;
}

/// Default ZMQ-based implementation that talks directly to a Python
/// `EngineCoreProc`.
pub struct ZmqEngineCoreClient {
    config: ZmqEngineCoreClientConfig,
    input_send: Arc<Mutex<Option<zeromq::RouterSendHalf>>>,
    output_rx: mpsc::Receiver<Result<EngineCoreOutputs>>,
    output_task: Option<tokio::task::JoinHandle<()>>,
    input_monitor_task: Option<tokio::task::JoinHandle<()>>,
    state: Arc<Mutex<RequestTracker>>,
    pub ready_message: Option<ReadyMessage>,
}

impl ZmqEngineCoreClient {
    /// Connect to an already running Python engine and complete the ready handshake.
    pub async fn connect(config: ZmqEngineCoreClientConfig) -> Result<Self> {
        let connected = transport::connect(
            &config.input_address,
            &config.output_address,
            &config.engine_identity,
            config.ready_timeout,
        )
        .await?;

        let (tx, rx) = mpsc::channel(64);
        let output_task = transport::spawn_output_loop(connected.output_socket, tx.clone());
        let input_monitor_task = transport::spawn_input_monitor(
            connected.input_recv,
            config.engine_identity.clone(),
            tx,
        );

        Ok(Self {
            config,
            input_send: Arc::new(Mutex::new(Some(connected.input_send))),
            output_rx: rx,
            output_task: Some(output_task),
            input_monitor_task: Some(input_monitor_task),
            state: Arc::new(Mutex::new(RequestTracker::default())),
            ready_message: connected.ready_message,
        })
    }

    async fn send<T>(&self, request_type: EngineCoreRequestType, payload: &T) -> Result<()>
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
            &self.config.engine_identity,
            request_type.as_frame(),
            payload,
        )
        .await
    }
}

#[async_trait]
impl EngineCoreClient for ZmqEngineCoreClient {
    async fn add_request(&self, mut req: EngineCoreRequest) -> Result<()> {
        req.client_index = self.config.client_index;
        req.validate()?;

        {
            let mut state = self.state.lock().await;
            state.insert(req.request_id.clone());
        }

        self.send(EngineCoreRequestType::Add, &req).await
    }

    async fn abort_requests(&self, ids: &[String]) -> Result<()> {
        let abortable = {
            let state = self.state.lock().await;
            state.retain_abortable(ids)
        };

        if abortable.is_empty() {
            return Ok(());
        }

        self.send(EngineCoreRequestType::Abort, &abortable).await
    }

    async fn next_output(&mut self) -> Result<EngineCoreOutputs> {
        let outputs = self.output_rx.recv().await.ok_or(Error::OutputClosed)??;
        {
            let mut state = self.state.lock().await;
            state.observe_outputs(&outputs);
        }
        Ok(outputs)
    }

    async fn shutdown(&mut self) -> Result<()> {
        self.input_send.lock().await.take();

        if let Some(task) = self.input_monitor_task.take() {
            task.abort();
        }
        if let Some(task) = self.output_task.take() {
            task.abort();
        }
        Ok(())
    }
}

impl Drop for ZmqEngineCoreClient {
    fn drop(&mut self) {
        if let Some(task) = self.input_monitor_task.take() {
            task.abort();
        }
        if let Some(task) = self.output_task.take() {
            task.abort();
        }
    }
}
