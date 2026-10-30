// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Minimal OpenAI-compatible HTTP server above [`vllm_chat`].

mod config;
mod error;
mod grpc;
mod listener;
mod lora;
mod middleware;
mod model_names;
mod render;
mod routes;
mod runtime;
mod server_info;
mod state;
mod tls;
#[cfg(test)]
mod tls_tests;
mod utils;

use std::any::Any;
use std::future::Future;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use anyhow::{Context as _, Result, ensure};
use axum::Router;
use axum::body::Body;
use axum::extract::ConnectInfo;
use axum::http::Request;
pub use config::{
    AccessLogConfig, ApiServerOptions, CoordinatorMode, CorsConfig, DEFAULT_KEEP_ALIVE_TIMEOUT,
    DEFAULT_LOG_STATS_INTERVAL, DEFAULT_REQUEST_BODY_LIMIT_BYTES, EngineCoreConfig,
    HttpListenerMode, LoraModulePath, OutputLogConfig, OutputLogging, ServerConfig, TlsConfig,
};
use hyper::body::Incoming;
use hyper::server::conn::http1;
use hyper_util::rt::{TokioIo, TokioTimer};
use hyper_util::server::graceful::GracefulShutdown;
use hyper_util::service::TowerToHyperService;
pub use render::{RenderConfig, serve_render};
pub use routes::openai::utils::types::ReasoningEffort;
use tokio::time::{Instant, sleep_until};
use tokio_util::sync::CancellationToken;
use tonic::server::NamedService as _;
use tonic::transport::Server as TonicServer;
use tonic_health::ServingStatus;
use tonic_health::server::health_reporter;
use tower::ServiceExt as _;
use tracing::{info, trace, warn};
use vllm_chat::{ChatLlm, LoadModelBackendsOptions, load_model_backends};
pub use vllm_chat::{
    ChatTemplateContentFormatOption, GenerationConfigMode, ParserSelection, RendererSelection,
};
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};
use vllm_llm::{EngineCoreLlm, Llm};
pub use vllm_metrics::CustomHistogramBuckets;
use vllm_metrics::METRICS;
use vllm_text::TextLlm;

use crate::listener::{Listener, ListenerAddr, MaybeTlsListener};
use crate::routes::build_router;
use crate::server_info::ServerInfoSnapshot;
use crate::state::AppState;

/// How often the server PINGs an idle gRPC connection to reap a dead peer;
/// tonic enables no keepalive by default. 2h matches the gRPC-core default.
const GRPC_KEEPALIVE_INTERVAL: Duration = Duration::from_secs(7200);
/// How long the server waits for a keepalive PING reply before dropping the gRPC
/// connection. 20s matches the gRPC-core default.
const GRPC_KEEPALIVE_TIMEOUT: Duration = Duration::from_secs(20);

/// Resolve the public model names accepted by the frontend.
fn effective_served_model_names(model: &str, served_model_name: &[String]) -> Vec<String> {
    if served_model_name.is_empty() {
        vec![model.to_string()]
    } else {
        served_model_name.to_vec()
    }
}

/// Build the shared application state for one configured model and its
/// [`Llm`]. When the server drives engine-core itself, connect to it here and
/// load any static LoRA adapters from `--lora-modules`.
async fn build_state(config: &ServerConfig, llm: LlmSource) -> Result<Arc<AppState>> {
    // Fix the histogram buckets before the engine connection records metrics.
    METRICS.set_custom_histogram_buckets(config.custom_histogram_buckets.clone());

    // If no served names are specified, fall back to the backend model path so
    // that the API always has at least one valid model ID. Use the same primary
    // public name for frontend-side metrics labels.
    let served_model_names = effective_served_model_names(&config.model, &config.served_model_name);
    let metrics_model_name = served_model_names[0].clone();

    // Load both backends from the same model metadata so they stay in sync.
    let loaded = load_model_backends(
        &config.model,
        LoadModelBackendsOptions {
            revision: config.revision.clone(),
            tokenizer: config.tokenizer.clone(),
            hf_config: config.hf_config.clone(),
            hf_overrides: config.hf_overrides.clone(),
            generation_config: config.generation_config,
            renderer: config.renderer,
            language_model_only: config.language_model_only,
            chat_template: config.chat_template.clone(),
            trust_request_chat_template: config.trust_request_chat_template,
            chat_template_content_format: config.chat_template_content_format,
            default_chat_template_kwargs: config
                .default_chat_template_kwargs
                .clone()
                .unwrap_or_default(),
            limit_mm_per_prompt: config.limit_mm_per_prompt.clone(),
        },
    )
    .await
    .context("failed to create chat/text backends")?;
    let text_backend = loaded.text_backend;
    let chat_backend = loaded.chat_backend;

    let (llm, engine_core) = match llm {
        LlmSource::EngineCore(engine_core) => {
            let (llm, server_info) = connect_engine_core(
                config,
                &engine_core,
                text_backend.is_moe(),
                metrics_model_name,
            )
            .await?;
            (
                Box::new(llm) as Box<dyn Llm>,
                Some((engine_core, server_info)),
            )
        }
        LlmSource::Provided(llm) => (llm, None),
    };
    let text = TextLlm::from_boxed(llm, text_backend)
        .with_max_logprobs(config.max_logprobs)
        .with_structured_outputs_backend(config.structured_outputs_backend);

    let chat = ChatLlm::new(text, chat_backend)
        .with_tool_call_parser(config.tool_call_parser.clone())
        .with_reasoning_parser(config.reasoning_parser.clone())
        .with_tool_strict_level(config.tool_strict_level)
        .with_reasoning_effort_map(
            config
                .reasoning_effort_map
                .iter()
                .map(|(effort, value)| (effort.as_str().to_owned(), value.clone()))
                .collect(),
        )
        .with_exclude_tools_when_tool_choice_none(config.exclude_tools_when_tool_choice_none);

    let mut state = AppState::new(served_model_names, chat)
        .with_model_path(config.model.clone())
        .with_api_server_options(config.api_server_options)
        .with_api_keys(config.api_keys.clone())
        .with_cors(config.cors.clone())
        .with_access_log(config.access_log.clone());
    if let Some((engine_core, server_info)) = &engine_core {
        state = (state.with_server_info(server_info.clone()))
            .with_profiler(engine_core.profiler.clone());
    }
    let state = Arc::new(state);

    if let Some((engine_core, _)) = &engine_core {
        // Load operator-configured static LoRA adapters before serving, failing
        // startup on any error like Python's `init_static_loras` does.
        load_static_loras(&state, &engine_core.lora_modules).await?;
    }

    Ok(state)
}

/// Connect to engine-core as `engine_core` describes and drive it through an
/// [`EngineCoreLlm`].
async fn connect_engine_core(
    config: &ServerConfig,
    engine_core: &EngineCoreConfig,
    model_is_moe: bool,
    metrics_model_name: String,
) -> Result<(EngineCoreLlm, ServerInfoSnapshot)> {
    let coordinator_mode = engine_core.effective_coordinator_mode(model_is_moe);
    info!(
        engine_count = engine_core.engine_count(),
        model_is_moe,
        ?coordinator_mode,
        "resolved coordinator mode"
    );

    let client = EngineCoreClient::connect(EngineCoreClientConfig {
        transport_mode: engine_core.transport_mode.clone(),
        coordinator_mode,
        model_name: metrics_model_name,
        client_index: 0,
        engine_stats_enabled: !engine_core.disable_log_stats,
    })
    .await
    .context("failed to connect to engine core")?;
    let server_info = ServerInfoSnapshot::new(config, engine_core, &client);

    let llm = EngineCoreLlm::new(client)
        .with_log_stats(engine_core.log_stats_interval.filter(|_| !engine_core.disable_log_stats))
        .with_request_id_randomization(!engine_core.disable_request_id_randomization)
        .with_stream_interval(engine_core.stream_interval);
    Ok((llm, server_info))
}

async fn load_static_loras(state: &AppState, modules: &[LoraModulePath]) -> Result<()> {
    for module in modules {
        let request = state.load_static_lora(module).await.with_context(|| {
            format!(
                "failed to load LoRA adapter `{}` from --lora-modules",
                module.name
            )
        })?;
        info!(
            lora_name = %request.lora_name,
            lora_int_id = request.lora_int_id,
            lora_path = %request.lora_path,
            "loaded static LoRA adapter"
        );
    }
    Ok(())
}

/// Extra routes merged into the finalized vLLM router.
type RouterExtension = Box<dyn FnOnce(Router) -> Router + Send>;

/// Where the server's [`Llm`] comes from.
#[allow(
    clippy::large_enum_variant,
    reason = "built once per server, so boxing the config would only burden callers"
)]
pub enum LlmSource {
    /// The server connects to engine-core and serves it through an
    /// [`EngineCoreLlm`].
    EngineCore(EngineCoreConfig),
    /// A caller-built `Llm`, such as a router over several engines or a client
    /// for a remote one. The server treats it as opaque: routes and gRPC
    /// services that drive the engine-core client directly are not served.
    Provided(Box<dyn Llm>),
}

/// The OpenAI-compatible HTTP server, with an optional gRPC server.
///
/// The server owns one `vllm-chat` facade, which in turn owns the lower
/// `vllm-text` and `vllm-llm` layers, and shuts them down before returning.
pub struct Server {
    config: ServerConfig,
    llm: LlmSource,
    extend_router: RouterExtension,
}

impl Server {
    /// Serve the `Llm` that `llm` describes, configured by `config`.
    pub fn new(config: ServerConfig, llm: LlmSource) -> Self {
        Self {
            config,
            llm,
            extend_router: Box::new(|router| router),
        }
    }

    /// Add an opt-in router extension.
    ///
    /// The extension receives the finalized vLLM router and can merge additional
    /// routes before the server starts accepting requests.
    pub fn router_extension(
        mut self,
        extend_router: impl FnOnce(Router) -> Router + Send + 'static,
    ) -> Self {
        self.extend_router = Box::new(extend_router);
        self
    }

    /// Run the server until the supplied shutdown token is cancelled.
    pub async fn serve(self, shutdown: CancellationToken) -> Result<()> {
        serve(self.config, self.llm, shutdown, self.extend_router).await
    }
}

/// Reject an [`EngineCoreLlm`] passed as [`LlmSource::Provided`]. The server
/// would serve it without its engine-core settings, and its request histograms
/// would already use the default buckets, since the server sets the custom
/// ones while building its state.
fn reject_engine_core_llm(llm: &dyn Llm) -> Result<()> {
    ensure!(
        !(llm as &dyn Any).is::<EngineCoreLlm>(),
        "an `EngineCoreLlm` must be served through `LlmSource::EngineCore`"
    );
    Ok(())
}

async fn serve(
    config: ServerConfig,
    llm: LlmSource,
    shutdown: CancellationToken,
    extend_router: RouterExtension,
) -> Result<()> {
    config.validate().context("invalid OpenAI frontend configuration")?;
    match &llm {
        LlmSource::EngineCore(engine_core) => {
            (engine_core.validate(&config)).context("invalid engine-core configuration")?;
        }
        LlmSource::Provided(llm) => reject_engine_core_llm(llm.as_ref())?,
    }
    let manages_engine =
        matches!(&llm, LlmSource::EngineCore(engine_core) if engine_core.manages_engine);

    // Build the TLS server config once, up front, so a bad cert/key fails fast
    // before the (potentially long) engine handshake.
    let tls_config = config
        .tls
        .as_ref()
        .map(tls::build_server_config)
        .transpose()
        .context("invalid TLS configuration")?;

    // Also check shutdown during the (potentially long) startup handshake and
    // static LoRA loading.
    let state = tokio::select! {
        result = build_state(&config, llm) => result?,
        _ = shutdown.cancelled() => return Ok(()),
    };
    let model = state.primary_model_name().to_owned();
    let mut app = extend_router(build_router(state.clone()));
    if let Some(root_path) = config.root_path.clone() {
        app = routes::with_root_path(app, root_path);
    }

    info!(model, "starting vLLM server");

    let listener = Listener::bind(&config.listener_mode)
        .await
        .context("failed to bind listener for OpenAI server")?;
    let bind_address = listener.local_addr_display()?;

    // Optionally bind the gRPC Inference server on a separate port. Bind
    // synchronously here so bind errors (port in use, permission denied, ...)
    // surface before serving rather than being deferred until shutdown.
    let grpc_setup = if let Some(grpc_listener_mode) = &config.grpc_listener_mode {
        let grpc_listener = Listener::bind(grpc_listener_mode)
            .await
            .with_context(|| format!("failed to bind gRPC listener for {grpc_listener_mode:?}"))?;
        let addr = grpc_listener.local_addr_display()?;
        // gRPC reuses the HTTP TLS config (same SslContext) plus ALPN h2.
        let grpc_tls = config
            .tls
            .as_ref()
            .map(tls::build_grpc_server_config)
            .transpose()
            .context("invalid gRPC TLS configuration")?;
        let (health_reporter, health_service) = health_reporter();
        let engine_health = state.llm().subscribe_health();
        // Control drives the engine-core client directly, so it is only served
        // over an `EngineCoreLlm`.
        let control_service = state.has_engine_core_client().then(|| {
            grpc::ControlGrpcService::new(
                grpc::ControlServiceImpl::new(state.clone())
                    .with_engine_shutdown(manages_engine.then(|| shutdown.clone())),
            )
            .max_decoding_message_size(DEFAULT_REQUEST_BODY_LIMIT_BYTES)
        });
        let mut health_services = vec![grpc::InferenceGrpcService::NAME];
        if control_service.is_some() {
            health_services.push(grpc::ControlGrpcService::NAME);
        }
        for service in &health_services {
            (health_reporter.set_service_status(service, ServingStatus::Serving)).await;
        }
        let inference_service =
            grpc::InferenceGrpcService::new(grpc::InferenceServiceImpl::new(state.clone()))
                .max_decoding_message_size(DEFAULT_REQUEST_BODY_LIMIT_BYTES);
        let svc = TonicServer::builder()
            .http2_keepalive_interval(Some(GRPC_KEEPALIVE_INTERVAL))
            .http2_keepalive_timeout(Some(GRPC_KEEPALIVE_TIMEOUT))
            .layer(middleware::request_runtime_layer(state.clone()))
            .add_service(health_service)
            .add_optional_service(control_service)
            .add_service(inference_service);
        Some((
            addr,
            grpc_listener,
            svc,
            grpc_tls,
            health_reporter,
            health_services,
            engine_health,
        ))
    } else {
        None
    };

    let scheme = if tls_config.is_some() {
        "https"
    } else {
        "http"
    };
    let model = model.as_str();

    // Run HTTP and gRPC concurrently under a child token of the caller's shutdown
    // token. Caller cancellation propagates into both protocols; if either
    // protocol exits first, we cancel this child token so its sibling also
    // begins a graceful drain.
    let server_shutdown = shutdown.child_token();
    let force_shutdown = CancellationToken::new();
    let shutdown_deadline = Arc::new(OnceLock::new());

    // Spawn a task to trigger `force_shutdown` after shutdown deadline elapses.
    tokio::spawn({
        let shutdown = server_shutdown.clone();
        let force_shutdown = force_shutdown.clone();
        let shutdown_deadline = shutdown_deadline.clone();
        let shutdown_timeout = config.shutdown_timeout;

        async move {
            shutdown.cancelled().await;
            let deadline = Instant::now() + shutdown_timeout;
            let _ = shutdown_deadline.set(deadline);

            if shutdown_timeout.is_zero() {
                force_shutdown.cancel();
            } else {
                sleep_until(deadline).await;
                force_shutdown.cancel();
            }
        }
    });

    // 0 disables keep-alive but still bounds the head read (default), so a
    // silent client cannot hold the connection open.
    let keep_alive_timeout = config.keep_alive_timeout;
    let timeouts = ConnectionTimeouts {
        header_read: if keep_alive_timeout.is_zero() {
            DEFAULT_KEEP_ALIVE_TIMEOUT
        } else {
            keep_alive_timeout
        },
        keep_alive_enabled: !keep_alive_timeout.is_zero(),
    };

    let http_fut = {
        let shutdown = server_shutdown.child_token();
        let server_shutdown = server_shutdown.clone();
        let force_shutdown = force_shutdown.clone();
        async move {
            let listener = match tls_config {
                Some(context) => MaybeTlsListener::tls(listener, context),
                None => MaybeTlsListener::plain(listener),
            };
            let server = serve_connections(listener, app, shutdown.cancelled_owned(), timeouts);

            info!(
                bind_address,
                scheme, model, "OpenAI server is ready to accept requests"
            );

            let result = tokio::select! {
                result = server => {
                    result.context("HTTP server failed")
                }
                _ = force_shutdown.cancelled() => {
                    warn!("HTTP graceful shutdown deadline elapsed; aborting server");
                    Ok(())
                }
            };

            server_shutdown.cancel();
            result
        }
    };

    let grpc_fut = {
        let shutdown = server_shutdown.child_token();
        let server_shutdown = server_shutdown.clone();
        let force_shutdown = force_shutdown.clone();
        async move {
            let Some((
                addr,
                grpc_listener,
                svc,
                grpc_tls,
                health_reporter,
                health_services,
                engine_health,
            )) = grpc_setup
            else {
                // No gRPC configured: just wait for shutdown so we do not race the
                // join! by resolving early and tripping the cancellation token.
                shutdown.cancelled().await;
                return Ok(());
            };
            let tls = grpc_tls.is_some();
            let incoming = match grpc_tls {
                Some(context) => MaybeTlsListener::tls(grpc_listener, context),
                None => MaybeTlsListener::plain(grpc_listener),
            };
            let server =
                svc.serve_with_incoming_shutdown(incoming, shutdown.clone().cancelled_owned());
            let health_monitor =
                grpc::monitor_health(health_reporter, health_services, engine_health, shutdown);

            info!(%addr, tls, model, "gRPC server is ready to accept requests");

            let server = async move {
                let result = tokio::select! {
                    result = server => {
                        result.context("gRPC server failed")
                    }
                    _ = force_shutdown.cancelled() => {
                        warn!("gRPC graceful shutdown deadline elapsed; aborting server");
                        Ok(())
                    }
                };

                server_shutdown.cancel();
                result
            };

            let (result, ()) = tokio::join!(server, health_monitor);
            result
        }
    };

    let (http_res, grpc_res) = tokio::join!(http_fut, grpc_fut);
    http_res.and(grpc_res)?;

    let shutdown_deadline = shutdown_deadline
        .get()
        .copied()
        .unwrap_or_else(|| Instant::now() + config.shutdown_timeout);
    state.shutdown(shutdown_deadline).await
}

/// Per-connection timeouts applied while serving HTTP/HTTPS.
#[derive(Clone, Copy)]
pub(crate) struct ConnectionTimeouts {
    /// HTTP/1 header-read timeout (bounds idle keep-alive and the head read).
    pub(crate) header_read: Duration,
    /// Whether HTTP/1 keep-alive is enabled; `false` closes after each response.
    pub(crate) keep_alive_enabled: bool,
}

/// Serve `app` per connection (HTTP/1) with a keep-alive idle timeout and
/// graceful drain. Hand-rolled on hyper because [`axum::serve()`] takes no config.
async fn serve_connections<L>(
    mut listener: L,
    app: Router,
    shutdown: impl Future<Output = ()> + Send,
    timeouts: ConnectionTimeouts,
) -> Result<()>
where
    L: axum::serve::Listener<Addr = ListenerAddr>,
{
    let graceful = GracefulShutdown::new();
    let mut shutdown = std::pin::pin!(shutdown);
    loop {
        let (io, addr) = tokio::select! {
            conn = listener.accept() => conn,
            () = &mut shutdown => break,
        };

        // Expose the TCP peer to handlers, as `ConnectInfo` does for `axum::serve`.
        let peer = match addr {
            ListenerAddr::Tcp(addr) => Some(ConnectInfo(addr)),
            ListenerAddr::Unix(_) => None,
        };
        let service =
            TowerToHyperService::new(app.clone().map_request(move |req: Request<Incoming>| {
                let mut req = req.map(Body::new);
                if let Some(peer) = peer {
                    req.extensions_mut().insert(peer);
                }
                req
            }));
        let mut builder = http1::Builder::new();
        builder.timer(TokioTimer::new()).header_read_timeout(timeouts.header_read);
        if !timeouts.keep_alive_enabled {
            builder.keep_alive(false);
        }
        let connection = builder.serve_connection(TokioIo::new(io), service);
        let connection = graceful.watch(connection);

        tokio::spawn(async move {
            if let Err(err) = connection.await {
                trace!(error = %err, "failed to serve connection");
            }
        });
    }

    drop(listener);
    graceful.shutdown().await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_served_model_names_falls_back_to_backend_model() {
        assert_eq!(
            effective_served_model_names("backend-model", &[]),
            vec!["backend-model"]
        );
    }

    #[test]
    fn effective_served_model_names_preserves_public_names() {
        let served_names = vec!["public-model".to_string(), "public-alias".to_string()];

        assert_eq!(
            effective_served_model_names("backend-model", &served_names),
            served_names
        );
    }
}
