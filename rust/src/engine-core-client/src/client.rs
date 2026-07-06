use std::sync::Arc;
use std::time::Duration;

use futures::future::{join_all, try_join_all};
use itertools::Itertools;
use serde::Serialize;
use tokio::sync::mpsc;
use tokio_util::task::AbortOnDropHandle;
use tracing::{debug, info, trace};

use crate::client::imp::{ClientInner, run_abort_loop, run_output_dispatcher_loop};
use crate::coordinator::CoordinatorHandle;
use crate::error::{Error, Result};
use crate::protocol::dtype::ModelDtype;
use crate::protocol::handshake::EngineCoreReadyResponse;
use crate::protocol::lora::LoraRequest;
use crate::protocol::request::{EngineCoreRequest, EngineCoreRequestType};
use crate::protocol::utility::{EngineCoreUtilityRequest, PauseMode};
use crate::runtime::{BackgroundShutdownRuntime, build_zmq_runtime};
use crate::transport::{self, ConnectedEngine};

pub(crate) mod imp;
mod state;
mod stream;

pub use stream::{EngineCoreOutputStream, EngineCoreStreamOutput};

/// How the frontend acquires its request/response transport with Python
/// `EngineCoreProc`s.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum TransportMode {
    /// The Rust process owns the startup handshake and allocates or binds the
    /// frontend transport addresses itself before replying to engine
    /// `HELLO` messages.
    HandshakeOwner {
        /// Shared handshake endpoint that engines dial during startup.
        handshake_address: String,
        /// Host/IP that engines should use to connect back to the frontend
        /// transport sockets.
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

    /// The Python supervisor has already chosen the frontend transport
    /// addresses, and the Rust process only needs to bind them and wait for
    /// engine registration frames.
    Bootstrapped {
        /// Input ROUTER socket address that engines will connect to for
        /// requests.
        input_address: String,
        /// Output PULL socket address that engines will connect to for
        /// responses.
        output_address: String,
        /// First data-parallel engine rank expected to register on this
        /// transport.
        engine_start_index: u32,
        /// Total number of engines expected to register on this transport.
        engine_count: usize,
        /// Maximum time to wait for all expected engines to register.
        ready_timeout: Duration,
    },
}

/// Which coordinator implementation should be active when one is present for a
/// frontend client.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoordinatorMode {
    /// Run the Rust in-process coordinator for managed `serve` deployments.
    InProc,
    /// Connect to an external coordinator owned by another process.
    External { address: String },
}

/// Configuration for connecting a Rust frontend client to an already running
/// Python `EngineCoreProc`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineCoreClientConfig {
    /// Frontend-to-engine transport setup.
    pub transport_mode: TransportMode,
    /// Frontend-side coordinator behavior, or `None` when requests should flow
    /// directly to engines without any coordinator involvement.
    pub coordinator_mode: Option<CoordinatorMode>,
    /// Model name used for frontend-side metrics labels.
    pub model_name: String,
    /// Frontend client index stamped onto every request.
    pub client_index: u32,
}

impl EngineCoreClientConfig {
    /// Create a new client config with the given handshake address, expecting a
    /// single engine, and default values for all other fields.
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

    /// Override the locally bound input/output addresses for handshake-owned
    /// transport mode.
    ///
    /// This is primarily used by tests that want deterministic IPC endpoints
    /// while still exercising the handshake-owned startup path.
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

/// The reason a request stream is being aborted when its output stream is
/// dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AbortCause {
    /// The consumer dropped the stream before the request reached a terminal
    /// engine output.
    #[default]
    DroppedStream,
    /// The frontend matched a stop string locally and intentionally stopped
    /// consuming the stream.
    StopStringMatched,
}

task_local::task_local! {
    static ABORT_CAUSE: AbortCause;
}

impl AbortCause {
    /// Return the abort cause currently associated with this task, or
    /// [`AbortCause::DroppedStream`] by default.
    pub fn current() -> Self {
        ABORT_CAUSE.try_get().unwrap_or_default()
    }

    /// Drop one value while marking the drop as happening for this abort cause.
    pub fn drop_as<T>(self, value: T) {
        ABORT_CAUSE.sync_scope(self, move || drop(value));
    }
}

/// Internal auto-abort work item sent from stream `Drop` handlers to the abort
/// worker.
#[derive(Debug, Clone)]
pub(crate) struct AbortRequest {
    request_id: String,
    cause: AbortCause,
}

/// Default ZMQ-based implementation that talks directly to a Python
/// `EngineCoreProc`.
pub struct EngineCoreClient {
    config: EngineCoreClientConfig,
    input_address: String,
    output_address: String,
    engines: Vec<ConnectedEngine>,
    inner: Arc<ClientInner>,
    coordinator: Option<CoordinatorHandle>,
    abort_tx: mpsc::UnboundedSender<AbortRequest>,

    /// Runtime used to send messages to the engine and drive all background tasks.
    runtime: BackgroundShutdownRuntime,
    // Background tasks
    output_task: AbortOnDropHandle<()>,
    dispatcher_task: AbortOnDropHandle<()>,
    abort_task: AbortOnDropHandle<()>,
    coordinator_output_task: Option<AbortOnDropHandle<()>>,
    coordinator_task: Option<AbortOnDropHandle<()>>,
}

impl EngineCoreClient {
    /// Connect to Python `EngineCoreProc`s using the configured
    /// transport/coordinator modes.
    ///
    /// In handshake-owned mode this method drives the full engine startup
    /// handshake. In bootstrapped mode it binds the provided frontend
    /// sockets and waits for the expected engine registration frames.
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
                engine_start_index,
                engine_count,
                ready_timeout,
            } => {
                if let Some(CoordinatorMode::InProc) = config.coordinator_mode {
                    panic!("cannot use in-process coordinator with bootstrapped transport mode")
                }

                transport::connect_bootstrapped(
                    input_address,
                    output_address,
                    *engine_start_index,
                    *engine_count,
                    *ready_timeout,
                )
                .await?
            }
        };

        Self::from_connected(config, connected).await
    }

    /// Create a new client instance from the connected transport state after
    /// the startup handshake completes.
    async fn from_connected(
        config: EngineCoreClientConfig,
        connected: transport::ConnectedTransport,
    ) -> Result<Self> {
        let (output_tx, output_rx) = mpsc::channel(64);
        let (abort_tx, abort_rx) = mpsc::unbounded_channel();
        let engines = connected.engines;
        let runtime = build_zmq_runtime();
        let inner = Arc::new(ClientInner::new(
            connected.input_send,
            runtime.handle().clone(),
            config.model_name.clone(),
            &engines,
        ));
        let output_task = AbortOnDropHandle::new(runtime.spawn(transport::run_output_loop(
            connected.output_socket,
            output_tx,
        )));
        let dispatcher_task = AbortOnDropHandle::new(
            runtime.spawn(run_output_dispatcher_loop(inner.clone(), output_rx)),
        );
        let abort_task =
            AbortOnDropHandle::new(runtime.spawn(run_abort_loop(inner.clone(), abort_rx)));

        // If any engine reported a dp_stats_address in its ready response, use it
        // as the external coordinator address.
        let dp_stats_address: Option<String> =
            engines.iter().find_map(|engine| engine.ready_response.dp_stats_address.clone());

        let (coordinator, coordinator_output_task, coordinator_task) =
            if let Some(coordinator_transport) = connected.coordinator {
                let (handle, runner) =
                    CoordinatorHandle::new_inproc(coordinator_transport.input_socket);
                let (coordinator_output_tx, coordinator_output_rx) = mpsc::channel(64);
                let coordinator_output_task =
                    AbortOnDropHandle::new(runtime.spawn(transport::run_output_loop(
                        coordinator_transport.output_socket,
                        coordinator_output_tx,
                    )));
                let coordinator_task = AbortOnDropHandle::new(
                    runtime.spawn(runner.run(coordinator_output_rx, inner.clone())),
                );
                (
                    Some(handle),
                    Some(coordinator_output_task),
                    Some(coordinator_task),
                )
            } else if let Some(address) =
                dp_stats_address.as_deref().or(match config.coordinator_mode.as_ref() {
                    Some(CoordinatorMode::External { address }) => Some(address.as_str()),
                    _ => None,
                })
            {
                let (handle, service) = CoordinatorHandle::connect_external(address).await?;
                let coordinator_task =
                    AbortOnDropHandle::new(runtime.spawn(service.run(inner.clone())));
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
            runtime,
            output_task,
            dispatcher_task,
            abort_task,
            coordinator_output_task,
            coordinator_task,
        })
    }

    /// Return the address of the input socket that the client uses to send
    /// requests to the engine.
    pub fn input_address(&self) -> &str {
        &self.input_address
    }

    /// Return the address of the output socket that the client listens on for
    /// engine responses.
    pub fn output_address(&self) -> &str {
        &self.output_address
    }

    /// Return the number of engines connected to this client.
    pub fn engine_count(&self) -> usize {
        self.engines.len()
    }

    /// Return the engine-side indices connected to this client.
    ///
    /// # Panics
    ///
    /// Panics if any connected engine uses an opaque identity that does not
    /// encode an index. Use [`Self::known_engine_indices`] for a lossy,
    /// non-panicking variant.
    pub fn engine_indices(&self) -> Vec<u32> {
        self.engines
            .iter()
            .map(|engine| engine.engine_id.engine_index().expect("engine id must encode as u16"))
            .collect()
    }

    /// Return the engine-side indices connected to this client, skipping
    /// engines with opaque identities that do not encode an index (e.g. mock
    /// engines in tests).
    pub fn known_engine_indices(&self) -> Vec<u32> {
        self.engines
            .iter()
            .filter_map(|engine| engine.engine_id.engine_index())
            .collect()
    }

    /// Return the engine identities of all engines connected to this client.
    pub fn engine_identities(&self) -> Vec<&[u8]> {
        self.engines.iter().map(|engine| &*engine.engine_id).collect()
    }

    /// Return the ready responses received from all engines on the input
    /// socket.
    pub fn ready_responses(&self) -> Vec<&EngineCoreReadyResponse> {
        self.engines.iter().map(|engine| &engine.ready_response).collect()
    }

    /// Return the engine-reported effective model dtype.
    pub fn model_dtype(&self) -> ModelDtype {
        self.engines
            .first()
            .expect("engine core client requires at least one engine")
            .ready_response
            .dtype
    }

    /// Return the engine-reported Python vLLM version.
    pub fn vllm_version(&self) -> &str {
        self.engines
            .first()
            .expect("engine core client requires at least one engine")
            .ready_response
            .vllm_version
            .as_str()
    }

    /// Return the total number of GPU blocks summed across all connected
    /// engines.
    pub fn total_num_gpu_blocks(&self) -> u64 {
        self.engines.iter().map(|engine| engine.ready_response.num_gpu_blocks).sum()
    }

    /// Return the minimum engine-reported `max_model_len` across all engines.
    ///
    /// This is the auto-fitted value after KV cache profiling and may differ
    /// from the originally configured value.
    pub fn max_model_len(&self) -> u32 {
        self.engines
            .iter()
            .map(|engine| engine.ready_response.max_model_len as u32)
            .min()
            .expect("engine core client requires at least one engine")
    }

    /// Return the world size (TP * PP) from the parallel config, if available.
    pub fn world_size(&self) -> u64 {
        self.engines
            .first()
            .expect("engine core client requires at least one engine")
            .ready_response
            .world_size
    }

    /// Return the data parallel size from the parallel config, if available.
    pub fn data_parallel_size(&self) -> u64 {
        self.engines
            .first()
            .expect("engine core client requires at least one engine")
            .ready_response
            .data_parallel_size
    }

    /// Get the model name associated with this client used for metrics
    /// labeling.
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
    /// Add a new request to the engine and return a per-request raw output
    /// stream.
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
        let lora_name = req.lora_request.as_ref().map(|lora| lora.lora_name.clone());
        let data_parallel_rank = req.data_parallel_rank;
        let (engine_id, rx) =
            self.inner.register_request(request_id.clone(), lora_name, data_parallel_rank)?;

        let result: Result<()> = async {
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

            self.inner.send_to_engine(&engine_id, EngineCoreRequestType::Add, &req).await?;
            Ok(())
        }
        .await;

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

        // Finalize the consumer streams first, before the engine round-trip.
        let all_request_ids: Vec<String> = abortable.values().flatten().cloned().collect();
        self.inner.abort_requests_locally(&all_request_ids);

        for (engine_id, request_ids) in abortable {
            self.inner.do_abort_requests(&engine_id, &request_ids).await?;
        }
        Ok(())
    }

    /// Call a typed utility method on all connected engines, returning one
    /// decoded result per connected engine if all calls succeed or an error
    /// if any call fails.
    ///
    /// Callers should pass utility arguments using Rust tuple semantics so the
    /// encoded payload matches Python's `(client_index, call_id,
    /// method_name, args)` contract: `()`, `(arg,)`, `(arg1, arg2)`, etc.
    pub async fn call_utility<T, A>(&self, method: &str, args: A) -> Result<Vec<T>>
    where
        T: serde::de::DeserializeOwned,
        A: serde::Serialize + std::fmt::Debug,
    {
        trace!(
            method,
            client_index = self.config.client_index,
            engine_count = self.engines.len(),
            "sending utility request"
        );

        // Phase 1: allocate one call id per engine and build the per-engine
        // request payloads up-front. Any failure here (registry closed, encode
        // error) must roll back the call ids already allocated so they do not
        // leak in the utility registry until shutdown.
        let mut pending_calls = Vec::with_capacity(self.engines.len());
        let mut prepared_sends = Vec::with_capacity(self.engines.len());
        for engine in &self.engines {
            let (call_id, rx) = match self.inner.allocate_and_register_utility_call() {
                Ok(pair) => pair,
                Err(err) => {
                    self.inner.unregister_utility_calls(pending_calls.iter().map(|(id, _)| *id));
                    return Err(err);
                }
            };
            let request = match EngineCoreUtilityRequest::new(
                self.config.client_index,
                call_id,
                method,
                &args,
            ) {
                Ok(request) => request,
                Err(err) => {
                    self.inner.unregister_utility_calls(
                        pending_calls.iter().map(|(id, _)| *id).chain(std::iter::once(call_id)),
                    );
                    return Err(err);
                }
            };
            pending_calls.push((call_id, rx));
            prepared_sends.push((&engine.engine_id, request));
        }

        // Phase 2: dispatch every utility request concurrently. `try_join_all`
        // fails fast on the first transport error and drops the remaining send
        // futures; any engines that already received the request will reply,
        // but those replies are simply dropped because we roll back the call
        // ids below.
        let send_futures = prepared_sends.iter().map(|(engine_id, request)| {
            self.inner.send_to_engine(engine_id, EngineCoreRequestType::Utility, request)
        });
        if let Err(err) = try_join_all(send_futures).await {
            self.inner.unregister_utility_calls(pending_calls.iter().map(|(id, _)| *id));
            return Err(err);
        }

        // Phase 3: wait for all engines to respond and preserve the per-engine
        // result list.
        let futures = pending_calls.into_iter().map(|(call_id, rx)| async move {
            rx.await
                .map_err(|_| Error::UtilityCallClosed {
                    method: method.to_string(),
                    call_id,
                })??
                .into_typed_result(method)
        });
        try_join_all(futures).await
    }

    /// Call a utility method on all connected engines and return the shared
    /// result if every engine agrees.
    pub async fn call_utility_consensus<T, A>(&self, method: &str, args: A) -> Result<T>
    where
        T: serde::de::DeserializeOwned + std::fmt::Debug + PartialEq,
        A: serde::Serialize + std::fmt::Debug,
    {
        let results: Vec<T> = self.call_utility(method, args).await?;

        if results.iter().all_equal() {
            // `engine_count >= 1` is enforced during startup handshake so `results` must be
            // non-empty.
            Ok(results.into_iter().next().unwrap())
        } else {
            Err(Error::InconsistentUtilityResults {
                method: method.to_string(),
                values: format!("{results:?}"),
            })
        }
    }

    /// Execute `collective_rpc` on all engines and flatten all engine results
    /// into one list.
    pub async fn collective_rpc<A, K>(
        &self,
        method: &str,
        timeout: Option<f64>,
        args: A,
        kwargs: K,
    ) -> Result<Vec<rmpv::Value>>
    where
        A: serde::Serialize + std::fmt::Debug,
        K: serde::Serialize + std::fmt::Debug,
    {
        let results = self
            .call_utility::<rmpv::Value, _>("collective_rpc", (method, timeout, args, kwargs))
            .await?;

        Ok(results
            .into_iter()
            .flat_map(|result| match result {
                // Each engine's `collective_rpc` result is itself the worker-level result list.
                rmpv::Value::Array(results) => results,
                other => vec![other],
            })
            .collect())
    }

    /// Return whether the engine is currently sleeping at any level.
    pub async fn is_sleeping(&self) -> Result<bool> {
        self.call_utility_consensus("is_sleeping", ()).await
    }

    /// Reset the multi-modal cache.
    pub async fn reset_mm_cache(&self) -> Result<()> {
        self.call_utility::<(), _>("reset_mm_cache", ()).await?;
        Ok(())
    }

    /// Reset the encoder cache.
    pub async fn reset_encoder_cache(&self) -> Result<()> {
        self.call_utility::<(), _>("reset_encoder_cache", ()).await?;
        Ok(())
    }

    /// Reset the prefix cache and optionally the external connector cache.
    ///
    /// Under data parallel, returns `true` only when every engine confirms the
    /// reset (AND aggregation).
    pub async fn reset_prefix_cache(
        &self,
        reset_running_requests: bool,
        reset_connector: bool,
    ) -> Result<bool> {
        Ok(self
            .call_utility(
                "reset_prefix_cache",
                (reset_running_requests, reset_connector),
            )
            .await?
            .into_iter()
            .all(|reset| reset))
    }

    /// Load or refresh one LoRA adapter on every connected engine.
    pub async fn add_lora(&self, lora_request: &LoraRequest) -> Result<bool> {
        Ok(self
            .call_utility::<bool, _>("add_lora", (lora_request,))
            .await?
            .into_iter()
            .all(|loaded| loaded))
    }

    /// Remove one LoRA adapter from every connected engine.
    pub async fn remove_lora(&self, lora_id: u64) -> Result<bool> {
        Ok(self
            .call_utility::<bool, _>("remove_lora", (lora_id,))
            .await?
            .into_iter()
            .all(|removed| removed))
    }

    /// Put the engine to sleep.
    pub async fn sleep(&self, level: u32, mode: PauseMode) -> Result<()> {
        self.call_utility::<(), _>("sleep", (level, mode)).await?;
        Ok(())
    }

    /// Wake the engine from sleep, optionally limiting the wake-up to specific
    /// tags.
    pub async fn wake_up(&self, tags: Option<Vec<String>>) -> Result<()> {
        self.call_utility::<(), _>("wake_up", (tags,)).await?;
        Ok(())
    }

    /// Pause the scheduler so generation can be halted
    pub async fn pause_scheduler(&self, mode: PauseMode, clear_cache: bool) -> Result<()> {
        self.call_utility::<(), _>("pause_scheduler", (mode, clear_cache)).await?;
        Ok(())
    }

    /// Resume the scheduler after a pause
    pub async fn resume_scheduler(&self) -> Result<()> {
        self.call_utility::<(), _>("resume_scheduler", ()).await?;
        Ok(())
    }

    /// Return whether the scheduler is currently in any pause state.
    pub async fn is_scheduler_paused(&self) -> Result<bool> {
        self.call_utility_consensus("is_scheduler_paused", ()).await
    }

    /// Start profiling the engine.
    pub async fn start_profile(&self, profile_prefix: Option<&str>) -> Result<()> {
        self.call_utility::<(), _>("profile", (true, profile_prefix)).await?;
        Ok(())
    }

    /// Stop profiling the engine.
    pub async fn stop_profile(&self, profile_prefix: Option<&str>) -> Result<()> {
        self.call_utility::<(), _>("profile", (false, profile_prefix)).await?;
        Ok(())
    }

    /// Shut down local client tasks and close transport state.
    pub async fn shutdown(self) -> Result<()> {
        let Self {
            inner,
            abort_tx,
            runtime,
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
        drop(inner);
        drop(runtime);

        info!("engine-core client shut down");
        Ok(())
    }
}
