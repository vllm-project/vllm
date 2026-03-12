use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use tokio::sync::{Mutex as AsyncMutex, mpsc};
use tokio_util::task::AbortOnDropHandle;
use tracing::debug;

use crate::client::imp::{
    ClientInner, RequestStreamState, run_auto_abort_loop, run_output_dispatcher_loop,
};
use crate::client::state::{ClientClosedState, RequestRegistry};
use crate::error::Result;
use crate::protocol::handshake::ReadyMessage;
use crate::protocol::{EngineCoreRequest, EngineCoreRequestType};
use crate::transport;

mod imp;
mod state;

pub use imp::RequestOutputStream;

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

/// Default ZMQ-based implementation that talks directly to a Python
/// `EngineCoreProc`.
pub struct EngineCoreClient {
    config: EngineCoreClientConfig,
    input_address: String,
    output_address: String,
    engine_identity: Vec<u8>,
    inner: Arc<ClientInner>,
    auto_abort_tx: mpsc::UnboundedSender<String>,
    output_task: AbortOnDropHandle<()>,
    dispatcher_task: AbortOnDropHandle<()>,
    auto_abort_task: AbortOnDropHandle<()>,
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
        let (output_tx, rx) = mpsc::channel(64);
        let (auto_abort_tx, auto_abort_rx) = mpsc::unbounded_channel();
        let inner = Arc::new(ClientInner {
            input_send: AsyncMutex::new(Some(connected.input_send)),
            state: Mutex::new(RequestRegistry::default()),
        });
        let engine_identity = connected.engine_identity;
        let output_task = AbortOnDropHandle::new(tokio::spawn(transport::run_output_loop(
            connected.output_socket,
            output_tx,
        )));
        let dispatcher_task =
            AbortOnDropHandle::new(tokio::spawn(run_output_dispatcher_loop(inner.clone(), rx)));
        let auto_abort_task = AbortOnDropHandle::new(tokio::spawn(run_auto_abort_loop(
            inner.clone(),
            engine_identity.clone(),
            auto_abort_rx,
        )));

        Ok(Self {
            config,
            input_address: connected.input_address,
            output_address: connected.output_address,
            engine_identity,
            inner,
            auto_abort_tx,
            output_task,
            dispatcher_task,
            auto_abort_task,
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
    pub async fn call(&self, mut req: EngineCoreRequest) -> Result<RequestOutputStream> {
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
            auto_abort_tx: self.auto_abort_tx.clone(),
            state: RequestStreamState::Running,
            rx,
        })
    }

    /// Abort currently in-flight requests by request ID.
    pub async fn abort(&self, ids: &[String]) -> Result<()> {
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
    pub async fn shutdown(self) -> Result<()> {
        let Self {
            inner,
            auto_abort_tx,
            output_task,
            dispatcher_task,
            auto_abort_task,
            ..
        } = self;

        debug!("shutting down engine-core client");
        inner.close_requests(ClientClosedState::ClientShutdown {
            reason: CLIENT_SHUTDOWN_REASON.to_string(),
        });
        drop(auto_abort_tx);
        inner.input_send.lock().await.take();
        auto_abort_task.abort();
        dispatcher_task.abort();
        output_task.abort();
        debug!("engine-core client shut down");

        Ok(())
    }
}
