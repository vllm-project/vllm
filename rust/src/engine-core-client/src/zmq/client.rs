use std::sync::Arc;

use async_trait::async_trait;
use thiserror_ext::AsReport;
use tokio::sync::{Mutex, mpsc};
use tokio_util::task::AbortOnDropHandle;

use crate::client::{EngineCoreClient, ReadyMessage, ZmqEngineCoreClientConfig};
use crate::error::{Error, Result};
use crate::protocol::{
    EngineCoreOutputs, EngineCoreRequest, EngineCoreRequestType, encode_msgpack,
};
use crate::state::RequestTracker;

use super::transport;

/// Default ZMQ-based implementation that talks directly to a Python
/// `EngineCoreProc`.
pub struct ZmqEngineCoreClient {
    config: ZmqEngineCoreClientConfig,
    input_address: String,
    output_address: String,
    engine_identity: Vec<u8>,
    input_send: Arc<Mutex<Option<zeromq::RouterSendHalf>>>,
    output_rx: mpsc::Receiver<Result<EngineCoreOutputs>>,
    output_task: Option<AbortOnDropHandle<()>>,
    input_monitor_task: Option<AbortOnDropHandle<()>>,
    state: Arc<Mutex<RequestTracker>>,
    pub ready_message: Option<ReadyMessage>,
}

impl ZmqEngineCoreClient {
    /// Connect to an already running Python engine and complete the startup handshake.
    pub async fn connect(config: ZmqEngineCoreClientConfig) -> Result<Self> {
        let connected = transport::connect(
            &config.handshake_address,
            &config.local_host,
            config.ready_timeout,
        )
        .await?;
        Self::from_connected(config, connected)
    }

    fn from_connected(
        config: ZmqEngineCoreClientConfig,
        connected: transport::ConnectedTransport,
    ) -> Result<Self> {
        let (tx, rx) = mpsc::channel(64);
        let engine_identity = connected.engine_identity.clone();
        let output_task = AbortOnDropHandle::new(transport::spawn_output_loop(
            connected.output_socket,
            tx.clone(),
        ));
        let input_monitor_task = AbortOnDropHandle::new(transport::spawn_input_monitor(
            connected.input_recv,
            connected.engine_identity,
            tx,
        ));

        Ok(Self {
            config,
            input_address: connected.input_address,
            output_address: connected.output_address,
            engine_identity,
            input_send: Arc::new(Mutex::new(Some(connected.input_send))),
            output_rx: rx,
            output_task: Some(output_task),
            input_monitor_task: Some(input_monitor_task),
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
        T: serde::Serialize,
    {
        let payload = encode_msgpack(payload)?;
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
