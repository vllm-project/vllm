use std::sync::Arc;
use std::time::Duration;

use futures::future::{join_all, try_join_all};
use tokio::sync::mpsc;
use tokio_util::task::AbortOnDropHandle;
use tracing::{debug, info, trace};

use crate::client::imp::{ClientInner, run_abort_loop, run_output_dispatcher_loop};
use crate::coordinator::CoordinatorHandle;
use crate::error::{Error, Result};
use crate::protocol::handshake::ReadyMessage;
use crate::protocol::{EngineCoreRequest, EngineCoreRequestType, EngineCoreUtilityRequest};
use crate::transport::{self, ConnectedEngine};

pub(crate) mod imp;
mod state;
mod stream;

pub use stream::{EngineCoreOutputStream, EngineCoreStreamOutput};

/// Configuration for connecting a Rust frontend client to an already running Python
/// `EngineCoreProc`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineCoreClientConfig {
    /// Startup handshake address used to bootstrap one or more Python engines.
    pub handshake_address: String,
    /// Number of engines expected to connect on the shared handshake socket.
    pub engine_count: usize,
    /// Model name used for frontend-side metrics labels.
    pub model_name: String,
    /// Local host/interface used when allocating the frontend input/output addresses.
    pub local_host: String,
    /// Timeout while waiting for each step of the startup handshake.
    pub ready_timeout: Duration,
    /// Frontend client index stamped onto every request.
    pub client_index: u32,
    /// Enable the in-process wave coordinator for single-frontend deployments.
    pub enable_inproc_coordinator: bool,
}

impl EngineCoreClientConfig {
    /// Create a new client config with the given handshake address, expecting a single engine, and
    /// default values for all other fields.
    pub fn new_single(handshake_address: impl Into<String>) -> Self {
        Self {
            handshake_address: handshake_address.into(),
            engine_count: 1,
            model_name: String::new(),
            local_host: "127.0.0.1".to_string(),
            ready_timeout: Duration::from_secs(30),
            client_index: 0,
            enable_inproc_coordinator: false,
        }
    }
}

/// Default ZMQ-based implementation that talks directly to a Python `EngineCoreProc`.
pub struct EngineCoreClient {
    config: EngineCoreClientConfig,
    input_address: String,
    output_address: String,
    engines: Vec<ConnectedEngine>,
    inner: Arc<ClientInner>,
    coordinator: Option<CoordinatorHandle>,
    abort_tx: mpsc::UnboundedSender<String>,

    // Background tasks
    output_task: AbortOnDropHandle<()>,
    dispatcher_task: AbortOnDropHandle<()>,
    abort_task: AbortOnDropHandle<()>,
    coordinator_output_task: Option<AbortOnDropHandle<()>>,
    coordinator_task: Option<AbortOnDropHandle<()>>,
}

impl EngineCoreClient {
    /// Connect to an already running Python engine and complete the startup handshake.
    pub async fn connect(config: EngineCoreClientConfig) -> Result<Self> {
        Self::connect_with_input_output_addresses(config, None, None).await
    }

    /// Connect to an already running Python engine and complete the startup handshake, while
    /// allowing the caller to specify explicit local input/output addresses instead of allocating
    /// TCP ports on `local_host`.
    pub async fn connect_with_input_output_addresses(
        config: EngineCoreClientConfig,
        local_input_address: Option<String>,
        local_output_address: Option<String>,
    ) -> Result<Self> {
        let connected = transport::connect(
            &config.handshake_address,
            config.engine_count,
            &config.local_host,
            local_input_address.as_deref(),
            local_output_address.as_deref(),
            config.enable_inproc_coordinator,
            config.ready_timeout,
        )
        .await?;

        Self::from_connected(config, connected)
    }

    /// Create a new client instance from the connected transport state after the startup handshake
    /// completes.
    fn from_connected(
        config: EngineCoreClientConfig,
        connected: transport::ConnectedTransport,
    ) -> Result<Self> {
        let (output_tx, output_rx) = mpsc::channel(64);
        let (abort_tx, abort_rx) = mpsc::unbounded_channel();
        let engines = connected.engines;
        let inner = Arc::new(ClientInner::new(
            connected.input_send,
            config.model_name.clone(),
            &engines,
        ));
        let output_task = AbortOnDropHandle::new(tokio::spawn(transport::run_output_loop(
            connected.output_socket,
            output_tx,
        )));
        let dispatcher_task = AbortOnDropHandle::new(tokio::spawn(run_output_dispatcher_loop(
            inner.clone(),
            output_rx,
        )));
        let abort_task =
            AbortOnDropHandle::new(tokio::spawn(run_abort_loop(inner.clone(), abort_rx)));

        let (coordinator, coordinator_output_task, coordinator_task) =
            if let Some(coordinator_transport) = connected.coordinator {
                let (handle, runner) = CoordinatorHandle::new(coordinator_transport.input_socket);
                let (coordinator_output_tx, coordinator_output_rx) = mpsc::channel(64);
                let coordinator_output_task =
                    AbortOnDropHandle::new(tokio::spawn(transport::run_output_loop(
                        coordinator_transport.output_socket,
                        coordinator_output_tx,
                    )));
                let coordinator_task = AbortOnDropHandle::new(tokio::spawn(
                    runner.run(coordinator_output_rx, inner.clone()),
                ));
                (
                    Some(handle),
                    Some(coordinator_output_task),
                    Some(coordinator_task),
                )
            } else {
                (None, None, None)
            };

        Ok(Self {
            config,
            input_address: connected.input_address,
            output_address: connected.output_address,
            engines,
            inner,
            coordinator,
            abort_tx,
            output_task,
            dispatcher_task,
            abort_task,
            coordinator_output_task,
            coordinator_task,
        })
    }

    /// Return the address of the input socket that the client uses to send requests to the engine.
    pub fn input_address(&self) -> &str {
        &self.input_address
    }

    /// Return the address of the output socket that the client listens on for engine responses.
    pub fn output_address(&self) -> &str {
        &self.output_address
    }

    /// Return the number of engines connected to this client.
    pub fn engine_count(&self) -> usize {
        self.engines.len()
    }

    /// Return the engine identities of all engines connected to this client.
    pub fn engine_identities(&self) -> Vec<&[u8]> {
        self.engines
            .iter()
            .map(|engine| &*engine.engine_id)
            .collect()
    }

    /// Return the READY messages received from all engines during the startup handshake.
    pub fn ready_messages(&self) -> Vec<&ReadyMessage> {
        self.engines
            .iter()
            .map(|engine| &engine.ready_message)
            .collect()
    }

    /// Get the model name associated with this client used for metrics labeling.
    pub fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    /// Return whether the client still considers the engine healthy.
    pub fn is_healthy(&self) -> bool {
        self.inner.is_healthy()
    }

    /// Return the first persistent health error observed by the client, if any.
    pub fn health_error(&self) -> Option<Arc<Error>> {
        self.inner.health_error()
    }
}

// Client API implementation.
impl EngineCoreClient {
    /// Add a new request to the engine and return a per-request raw output stream.
    pub async fn call(&self, mut req: EngineCoreRequest) -> Result<EngineCoreOutputStream> {
        req.client_index = self.config.client_index;
        req.validate()?;
        trace!(
            request_id = %req.request_id,
            client_index = req.client_index,
            current_wave = req.current_wave,
            request = ?req,
            "sending add request"
        );

        let request_id = req.request_id.clone();
        let (engine_id, rx) = self.inner.register_request(request_id.clone())?;

        let result = try {
            if let Some(coordinator) = self.coordinator.as_ref() {
                let snapshot = coordinator.snapshot();
                req.current_wave = snapshot.current_wave;
                if !snapshot.engines_running {
                    coordinator.notify_first_request(engine_id.clone())?;
                }
            }

            debug!(
                request_id = req.request_id,
                ?engine_id,
                "registered request to engine"
            );

            self.inner
                .send_to_engine(&engine_id, EngineCoreRequestType::Add, &req)
                .await?;
        };

        // Failed to send the request to the engine, roll back the registration.
        if let Err(error) = result {
            self.inner.rollback_request(&request_id);
            return Err(error);
        }

        Ok(EngineCoreOutputStream::new(
            request_id,
            self.abort_tx.clone(),
            rx,
        ))
    }

    /// Abort currently in-flight requests by request ID.
    pub async fn abort(&self, ids: &[String]) -> Result<()> {
        let abortable = self.inner.abortable_request_ids(ids)?;

        trace!(request_ids = ?ids, abortable_request_ids = ?abortable, "sending abort request ids");

        if abortable.is_empty() {
            return Ok(());
        }

        for (engine_id, request_ids) in abortable {
            self.inner
                .do_abort_requests(&engine_id, &request_ids)
                .await?;
        }
        Ok(())
    }

    /// Call a typed utility method on all connected engines, returning the first result if all
    /// calls succeed or an error if any call fails.
    ///
    /// Callers should pass utility arguments using Rust tuple semantics so the encoded payload
    /// matches Python's `(client_index, call_id, method_name, args)` contract:
    /// `()`, `(arg,)`, `(arg1, arg2)`, etc.
    pub async fn call_utility<T, A>(&self, method: &str, args: A) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
        A: serde::Serialize,
    {
        trace!(
            method,
            client_index = self.config.client_index,
            engine_count = self.engines.len(),
            "sending utility request"
        );

        let mut pending_calls = Vec::with_capacity(self.engines.len());
        for engine in &self.engines {
            let (call_id, rx) = self.inner.allocate_and_register_utility_call()?;
            let request =
                EngineCoreUtilityRequest::new(self.config.client_index, call_id, method, &args)?;

            // Return error immediately once we fail to send to any engine.
            // TODO: this operation is not atomic.
            self.inner
                .send_to_engine(&engine.engine_id, EngineCoreRequestType::Utility, &request)
                .await?;
            pending_calls.push((call_id, rx));
        }

        // Wait for all engines to respond and return the first successful result.
        // TODO: shall we check if all results match?
        let futures = pending_calls.into_iter().map(|(call_id, rx)| async move {
            rx.await
                .map_err(|_| Error::UtilityCallClosed {
                    method: method.to_string(),
                    call_id,
                })??
                .into_typed_result(method)
        });
        let results = try_join_all(futures).await?;

        Ok(results
            .into_iter()
            .next()
            .expect("utility fanout must include at least one engine"))
    }

    /// Return whether the engine is currently sleeping at any level.
    pub async fn is_sleeping(&self) -> Result<bool> {
        self.call_utility("is_sleeping", ()).await
    }

    /// Reset the multi-modal cache.
    pub async fn reset_mm_cache(&self) -> Result<()> {
        self.call_utility("reset_mm_cache", ()).await
    }

    /// Reset the encoder cache.
    pub async fn reset_encoder_cache(&self) -> Result<()> {
        self.call_utility("reset_encoder_cache", ()).await
    }

    /// Reset the prefix cache and optionally the external connector cache.
    pub async fn reset_prefix_cache(
        &self,
        reset_running_requests: bool,
        reset_connector: bool,
    ) -> Result<bool> {
        self.call_utility(
            "reset_prefix_cache",
            (reset_running_requests, reset_connector),
        )
        .await
    }

    /// Put the engine to sleep.
    pub async fn sleep(&self, level: u32, mode: &str) -> Result<()> {
        self.call_utility("sleep", (level, mode)).await
    }

    /// Wake the engine from sleep, optionally limiting the wake-up to specific tags.
    pub async fn wake_up(&self, tags: Option<Vec<String>>) -> Result<()> {
        self.call_utility("wake_up", (tags,)).await
    }

    /// Shut down local client tasks and close transport state.
    pub async fn shutdown(self) -> Result<()> {
        let Self {
            inner,
            abort_tx,
            output_task,
            dispatcher_task,
            abort_task,
            coordinator_output_task,
            coordinator_task,
            ..
        } = self;

        info!("shutting down engine-core client");
        inner.shutdown().await;
        drop(abort_tx);

        // Abort all client tasks first, then await them.
        // Note the aborting orders here.
        let mut tasks = vec![abort_task, dispatcher_task, output_task];
        tasks.extend(coordinator_task);
        tasks.extend(coordinator_output_task);

        tasks.iter().for_each(|t| t.abort());
        join_all(tasks).await;

        info!("engine-core client shut down");
        Ok(())
    }
}
