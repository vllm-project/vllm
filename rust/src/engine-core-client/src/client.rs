use std::sync::Arc;
use std::time::Duration;

use futures::future::{join_all, try_join_all};
use tokio::sync::mpsc;
use tokio_util::task::AbortOnDropHandle;
use tracing::{debug, info, trace};

use crate::client::imp::{ClientInner, run_abort_loop, run_output_dispatcher_loop};
use crate::coordinator::CoordinatorHandle;
use crate::error::{Error, Result};
use crate::protocol::handshake::EngineCoreReadyResponse;
use crate::protocol::{EngineCoreRequest, EngineCoreRequestType, EngineCoreUtilityRequest};
use crate::transport::{self, ConnectedEngine};

pub(crate) mod imp;
mod state;
mod stream;

pub use stream::{EngineCoreOutputStream, EngineCoreStreamOutput};

/// How the frontend acquires its request/response transport with Python `EngineCoreProc`s.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransportMode {
    /// The Rust process owns the startup handshake and allocates or binds the frontend transport
    /// addresses itself before replying to engine `HELLO` messages.
    HandshakeOwner {
        /// Shared handshake endpoint that engines dial during startup.
        handshake_address: String,
        /// Host/IP that engines should use to connect back to the frontend transport sockets.
        advertised_host: String,
        /// Total number of engines expected to join this transport.
        engine_count: usize,
        /// Maximum time to wait for each startup phase to complete.
        ready_timeout: Duration,
        /// Optional explicit bind address for the input ROUTER socket.
        local_input_address: Option<String>,
        /// Optional explicit bind address for the output PULL socket.
        local_output_address: Option<String>,
    },

    /// The Python supervisor has already chosen the frontend transport addresses, and the Rust
    /// process only needs to bind them and wait for engine registration frames.
    Bootstrapped {
        /// Input ROUTER socket address that engines will connect to for requests.
        input_address: String,
        /// Output PULL socket address that engines will connect to for responses.
        output_address: String,
        /// Total number of engines expected to register on this transport.
        engine_count: usize,
        /// Maximum time to wait for all expected engines to register.
        ready_timeout: Duration,
    },
}

/// Which coordinator implementation should be active when one is present for a frontend client.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoordinatorMode {
    /// Run the Rust in-process coordinator for managed `serve` deployments.
    InProc,
    /// Connect to an external coordinator owned by another process.
    External { address: String },
}

/// Configuration for connecting a Rust frontend client to an already running Python
/// `EngineCoreProc`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineCoreClientConfig {
    /// Frontend-to-engine transport setup.
    pub transport_mode: TransportMode,
    /// Frontend-side coordinator behavior, or `None` when requests should flow directly to engines
    /// without any coordinator involvement.
    pub coordinator_mode: Option<CoordinatorMode>,
    /// Model name used for frontend-side metrics labels.
    pub model_name: String,
    /// Frontend client index stamped onto every request.
    pub client_index: u32,
}

impl EngineCoreClientConfig {
    /// Create a new client config with the given handshake address, expecting a single engine, and
    /// default values for all other fields.
    pub fn new_single(handshake_address: impl Into<String>) -> Self {
        Self {
            transport_mode: TransportMode::HandshakeOwner {
                handshake_address: handshake_address.into(),
                advertised_host: "127.0.0.1".to_string(),
                engine_count: 1,
                ready_timeout: Duration::from_secs(30),
                local_input_address: None,
                local_output_address: None,
            },
            coordinator_mode: None,
            model_name: String::new(),
            client_index: 0,
        }
    }

    /// Set the model name used by frontend-side metrics and diagnostics.
    pub fn with_model_name(mut self, model_name: impl Into<String>) -> Self {
        self.model_name = model_name.into();
        self
    }

    /// Override the client index stamped onto every outgoing request.
    pub fn with_client_index(mut self, client_index: u32) -> Self {
        self.client_index = client_index;
        self
    }

    /// Override the optional coordinator mode for this client config.
    pub fn with_coordinator_mode(mut self, coordinator_mode: Option<CoordinatorMode>) -> Self {
        self.coordinator_mode = coordinator_mode;
        self
    }

    /// Override the locally bound input/output addresses for handshake-owned transport mode.
    ///
    /// This is primarily used by tests that want deterministic IPC endpoints while still exercising
    /// the handshake-owned startup path.
    pub fn with_local_input_output_addresses(
        mut self,
        local_input_address: Option<String>,
        local_output_address: Option<String>,
    ) -> Self {
        let TransportMode::HandshakeOwner {
            local_input_address: current_input,
            local_output_address: current_output,
            ..
        } = &mut self.transport_mode
        else {
            panic!("local input/output overrides are only valid in handshake-owned mode");
        };
        *current_input = local_input_address;
        *current_output = local_output_address;
        self
    }
}

/// The reason a request stream is being aborted when its output stream is dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AbortCause {
    /// The consumer dropped the stream before the request reached a terminal engine output.
    #[default]
    DroppedStream,
    /// The frontend matched a stop string locally and intentionally stopped consuming the stream.
    StopStringMatched,
}

task_local::task_local! {
    static ABORT_CAUSE: AbortCause;
}

impl AbortCause {
    /// Return the abort cause currently associated with this task, or [`AbortCause::DroppedStream`]
    /// by default.
    pub fn current() -> Self {
        ABORT_CAUSE.try_get().unwrap_or_default()
    }

    /// Drop one value while marking the drop as happening for this abort cause.
    pub fn drop_as<T>(self, value: T) {
        ABORT_CAUSE.sync_scope(self, move || drop(value));
    }
}

/// Internal auto-abort work item sent from stream `Drop` handlers to the abort worker.
#[derive(Debug, Clone)]
pub(crate) struct AbortRequest {
    request_id: String,
    cause: AbortCause,
}

/// Default ZMQ-based implementation that talks directly to a Python `EngineCoreProc`.
pub struct EngineCoreClient {
    config: EngineCoreClientConfig,
    input_address: String,
    output_address: String,
    engines: Vec<ConnectedEngine>,
    inner: Arc<ClientInner>,
    coordinator: Option<CoordinatorHandle>,
    abort_tx: mpsc::UnboundedSender<AbortRequest>,

    // Background tasks
    output_task: AbortOnDropHandle<()>,
    dispatcher_task: AbortOnDropHandle<()>,
    abort_task: AbortOnDropHandle<()>,
    coordinator_output_task: Option<AbortOnDropHandle<()>>,
    coordinator_task: Option<AbortOnDropHandle<()>>,
}

impl EngineCoreClient {
    /// Connect to Python `EngineCoreProc`s using the configured transport/coordinator modes.
    ///
    /// In handshake-owned mode this method drives the full engine startup handshake. In
    /// bootstrapped mode it binds the provided frontend sockets and waits for the expected engine
    /// registration frames.
    pub async fn connect(config: EngineCoreClientConfig) -> Result<Self> {
        let connected = match &config.transport_mode {
            TransportMode::HandshakeOwner {
                handshake_address,
                advertised_host,
                engine_count,
                ready_timeout,
                local_input_address,
                local_output_address,
            } => {
                let enable_inproc_coordinator = match config.coordinator_mode {
                    None => false,
                    Some(CoordinatorMode::InProc) => true,
                    Some(CoordinatorMode::External { .. }) => {
                        return Err(Error::UnsupportedExternalCoordinator);
                    }
                };

                transport::connect_handshake(
                    handshake_address,
                    *engine_count,
                    advertised_host,
                    local_input_address.as_deref(),
                    local_output_address.as_deref(),
                    enable_inproc_coordinator,
                    *ready_timeout,
                )
                .await?
            }

            TransportMode::Bootstrapped {
                input_address,
                output_address,
                engine_count,
                ready_timeout,
            } => {
                if let Some(CoordinatorMode::InProc) = config.coordinator_mode {
                    panic!("cannot use in-process coordinator with bootstrapped transport mode")
                }

                transport::connect_bootstrapped(
                    input_address,
                    output_address,
                    *engine_count,
                    *ready_timeout,
                )
                .await?
            }
        };

        Self::from_connected(config, connected).await
    }

    /// Connect using handshake-owned transport mode while overriding the frontend input/output bind
    /// addresses.
    ///
    /// This helper preserves the previous test-facing API shape. It is only valid when
    /// `config.transport_mode` is `TransportMode::HandshakeOwner`.
    // TODO: inline this
    pub async fn connect_with_input_output_addresses(
        config: EngineCoreClientConfig,
        local_input_address: Option<String>,
        local_output_address: Option<String>,
    ) -> Result<Self> {
        let config =
            config.with_local_input_output_addresses(local_input_address, local_output_address);
        Self::connect(config).await
    }

    /// Create a new client instance from the connected transport state after the startup handshake
    /// completes.
    async fn from_connected(
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

        // If any engine reported a dp_stats_address in its ready response, use it
        // as the external coordinator address.
        let dp_stats_address: Option<String> = engines
            .iter()
            .filter_map(|e| e.ready_response.as_ref())
            .find_map(|r| r.dp_stats_address.clone());

        let (coordinator, coordinator_output_task, coordinator_task) =
            if let Some(coordinator_transport) = connected.coordinator {
                let (handle, runner) =
                    CoordinatorHandle::new_inproc(coordinator_transport.input_socket);
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
            } else if let Some(address) =
                dp_stats_address
                    .as_deref()
                    .or(match config.coordinator_mode.as_ref() {
                        Some(CoordinatorMode::External { address }) => Some(address.as_str()),
                        _ => None,
                    })
            {
                let (handle, service) = CoordinatorHandle::connect_external(address).await?;
                let coordinator_task =
                    AbortOnDropHandle::new(tokio::spawn(service.run(inner.clone())));
                (Some(handle), None, Some(coordinator_task))
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

    /// Return the ready responses received from all engines on the input socket.
    pub fn ready_responses(&self) -> Vec<&EngineCoreReadyResponse> {
        self.engines
            .iter()
            .filter_map(|engine| engine.ready_response.as_ref())
            .collect()
    }

    /// Return the total number of GPU blocks summed across all connected engines.
    pub fn total_num_gpu_blocks(&self) -> u64 {
        self.engines
            .iter()
            .filter_map(|engine| engine.ready_response.as_ref())
            .map(|r| r.num_gpu_blocks)
            .sum()
    }

    /// Return the minimum engine-reported `max_model_len` across all engines.
    ///
    /// This is the auto-fitted value after KV cache profiling and may differ from
    /// the originally configured value.
    pub fn max_model_len(&self) -> Option<u64> {
        self.engines
            .iter()
            .filter_map(|e| e.ready_response.as_ref())
            .map(|r| r.max_model_len)
            .min()
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
        let data_parallel_rank = req.data_parallel_rank;
        let (engine_id, rx) = self
            .inner
            .register_request(request_id.clone(), data_parallel_rank)?;

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
        inner.shutdown();
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
