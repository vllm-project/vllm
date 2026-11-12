// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Minimal OpenAI-compatible HTTP server above [`vllm_chat`].

mod config;
mod error;
mod fingerprint;
mod grpc;
mod listener;
mod lora;
mod middleware;
mod mm_wire;
mod model_names;
mod render;
mod routes;
mod runtime;
mod server_info;
mod serving;
mod state;
mod tls;
#[cfg(test)]
mod tls_tests;
mod utils;

use std::sync::Arc;

use anyhow::{Context as _, Result, bail};
use axum::Router;
pub use config::{
    AccessLogConfig, ApiServerOptions, Config, CoordinatorMode, CorsConfig,
    DEFAULT_KEEP_ALIVE_TIMEOUT, DEFAULT_LOG_STATS_INTERVAL, DEFAULT_MAX_N_SEQUENCES,
    DEFAULT_REQUEST_BODY_LIMIT_BYTES, HttpListenerMode, LoraModulePath, OutputLogConfig,
    OutputLogging, TlsConfig,
};
pub use fingerprint::FingerprintMode;
use futures::FutureExt as _;
pub use render::{RenderConfig, serve_render};
pub use routes::openai::utils::types::ReasoningEffort;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use tonic_health::server::health_reporter;
use tracing::info;
use vllm_chat::{ChatLlm, LoadModelBackendsOptions, load_model_backends};
pub use vllm_chat::{
    ChatTemplateContentFormatOption, GenerationConfigMode, ParserSelection, RendererSelection,
};
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};
use vllm_llm::EngineCoreLlm;
pub use vllm_metrics::CustomHistogramBuckets;
use vllm_metrics::METRICS;
use vllm_text::TextLlm;

use crate::routes::build_router;
use crate::server_info::ServerInfoSnapshot;
use crate::serving::{BoundListener, ConnectionTimeouts, GrpcServer, HttpServer, ListenerTls};
use crate::state::AppState;

async fn monitor_llm_health(
    mut health: watch::Receiver<bool>,
    shutdown: CancellationToken,
    keep_alive: bool,
) -> Result<()> {
    if keep_alive {
        shutdown.cancelled().await;
        return Ok(());
    }

    let health_result = tokio::select! {
        biased;

        result = health.wait_for(|healthy| !*healthy) => result,
        _ = shutdown.cancelled() => return Ok(()),
    };
    let health_channel_error = health_result.err();

    shutdown.cancel();
    match health_channel_error {
        None => bail!("LLM became permanently unhealthy"),
        Some(error) => Err(error).context("LLM health channel closed"),
    }
}

/// Resolve the public model names accepted by the frontend.
fn effective_served_model_names(model: &str, served_model_name: &[String]) -> Vec<String> {
    if served_model_name.is_empty() {
        vec![model.to_string()]
    } else {
        served_model_name.to_vec()
    }
}

/// Build the shared application state for one configured model and one engine
/// client, including any static LoRA adapters from `--lora-modules`.
async fn build_state(config: &Config) -> Result<Arc<AppState>> {
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
            skip_tokenizer_init: config.skip_tokenizer_init,
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

    let coordinator_mode = config.effective_coordinator_mode(text_backend.is_moe());
    info!(
        engine_count = config.engine_count(),
        model_is_moe = text_backend.is_moe(),
        ?coordinator_mode,
        "resolved coordinator mode"
    );

    let client = EngineCoreClient::connect(EngineCoreClientConfig {
        transport_mode: config.transport_mode.clone(),
        coordinator_mode,
        model_name: metrics_model_name.clone(),
        client_index: 0,
        engine_stats_enabled: !config.disable_log_stats,
    })
    .await
    .context("failed to connect to engine core")?;
    let server_info = ServerInfoSnapshot::new(config, &client);
    let system_fingerprint = config
        .fingerprint_mode
        .fingerprint(&client.ready_responses(), client.data_parallel_size());

    let llm = EngineCoreLlm::new(client)
        .with_log_stats(config.log_stats_interval.filter(|_| !config.disable_log_stats))
        .with_request_id_randomization(!config.disable_request_id_randomization)
        .with_stream_interval(config.stream_interval);
    let text = TextLlm::new(llm, text_backend)
        .with_served_model_name(metrics_model_name)
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

    let state = Arc::new(
        AppState::new(served_model_names, chat)
            .with_model_path(config.model.clone())
            .with_api_server_options(config.api_server_options)
            .with_system_fingerprint(system_fingerprint)
            .with_server_info(server_info)
            .with_api_keys(config.api_keys.clone())
            .with_cors(config.cors.clone())
            .with_access_log(config.access_log.clone())
            .with_profiler(config.profiler.clone()),
    );

    // Load operator-configured static LoRA adapters before serving, failing
    // startup on any error like Python's `init_static_loras` does.
    load_static_loras(&state, &config.lora_modules).await?;

    Ok(state)
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

/// Run the OpenAI-compatible HTTP server until the supplied shutdown token is
/// cancelled.
///
/// The server owns one `vllm-chat` facade, which in turn owns the lower
/// `vllm-text` and `vllm-llm` layers, and shuts them down before returning.
pub async fn serve(config: Config, shutdown: CancellationToken) -> Result<()> {
    serve_with_router_extension(config, shutdown, |router| router).await
}

/// Run the OpenAI-compatible HTTP server with an opt-in router extension.
///
/// The extension receives the finalized vLLM router and can merge additional
/// routes before the server starts accepting requests.
pub async fn serve_with_router_extension<F>(
    config: Config,
    shutdown: CancellationToken,
    extend_router: F,
) -> Result<()>
where
    F: FnOnce(Router) -> Router,
{
    config.validate().context("invalid OpenAI frontend configuration")?;

    // Build the TLS server configs once, up front, so a bad cert/key fails fast
    // before the (potentially long) engine handshake.
    let tls = ListenerTls::build(config.tls.as_ref(), config.grpc_listener_mode.is_some())?;

    // Also check shutdown during the (potentially long) startup handshake and
    // static LoRA loading.
    let state = tokio::select! {
        result = build_state(&config) => result?,
        _ = shutdown.cancelled() => return Ok(()),
    };
    let model = state.primary_model_name().to_owned();
    let mut app = extend_router(build_router(state.clone()));
    if let Some(root_path) = config.root_path.clone() {
        app = routes::with_root_path(app, root_path);
    }

    info!(model, "starting vLLM server");

    let listener = BoundListener::bind(&config.listener_mode, tls.http)
        .await
        .context("failed to bind listener for OpenAI server")?;

    // Optionally bind the gRPC Inference server on a separate port. Bind
    // synchronously here so bind errors (port in use, permission denied, ...)
    // surface before serving rather than being deferred until shutdown.
    let grpc = if let Some(grpc_listener_mode) = &config.grpc_listener_mode {
        let grpc_listener = BoundListener::bind(grpc_listener_mode, tls.grpc)
            .await
            .with_context(|| format!("failed to bind gRPC listener for {grpc_listener_mode:?}"))?;
        let (health_reporter, health_service) = health_reporter();
        let engine_health = grpc::EngineHealth::subscribe(state.llm());
        let health_monitor = grpc::HealthMonitor::register(
            health_reporter,
            grpc::HealthSource::Engines(engine_health),
        )
        .await;
        let control_service = grpc::ControlGrpcService::new(
            grpc::ControlServiceImpl::new(state.clone())
                .with_engine_shutdown(config.manages_engine.then(|| shutdown.clone())),
        )
        .max_decoding_message_size(DEFAULT_REQUEST_BODY_LIMIT_BYTES);
        let inference_service =
            grpc::InferenceGrpcService::new(grpc::InferenceServiceImpl::new(state.clone()))
                .max_decoding_message_size(DEFAULT_REQUEST_BODY_LIMIT_BYTES);
        let render_service =
            grpc::RenderGrpcService::new(grpc::RenderServiceImpl::new(state.clone()))
                .max_decoding_message_size(DEFAULT_REQUEST_BODY_LIMIT_BYTES);
        let router = serving::grpc_server_builder()
            .layer(middleware::request_runtime_layer(state.clone()))
            .add_service(health_service)
            .add_service(control_service)
            .add_service(inference_service)
            .add_service(render_service);
        Some(GrpcServer {
            listener: grpc_listener,
            health: health_monitor,
            serve: Box::new(move |incoming, stop| {
                router.serve_with_incoming_shutdown(incoming, stop).boxed()
            }),
        })
    } else {
        None
    };

    let http = HttpServer {
        listener,
        app,
        timeouts: ConnectionTimeouts::for_keep_alive(config.keep_alive_timeout),
    };
    // The LLM health monitor cancels `server_shutdown` on a permanent engine
    // failure; ending the servers in turn ends the monitor.
    let server_shutdown = shutdown.child_token();
    let health_fut = monitor_llm_health(
        state.llm().subscribe_health(),
        server_shutdown.clone(),
        config.keep_alive_on_engine_death,
    );
    let serve_fut = async {
        let result = serving::serve(
            "OpenAI server",
            &model,
            http,
            grpc,
            config.shutdown_timeout,
            server_shutdown.clone(),
        )
        .await;
        server_shutdown.cancel();
        result
    };
    let (serve_res, health_res) = tokio::join!(serve_fut, health_fut);
    let shutdown_deadline = serve_res?;

    let shutdown_result = state.shutdown(shutdown_deadline).await;
    health_res?;
    shutdown_result
}

#[cfg(test)]
mod tests {
    use std::task::Poll;

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

    #[tokio::test]
    async fn unhealthy_llm_cancels_server_shutdown_and_returns_error() {
        let (health_tx, health) = watch::channel(true);
        let shutdown = CancellationToken::new();
        let mut monitor = Box::pin(monitor_llm_health(health, shutdown.clone(), false));

        assert!(futures::poll!(&mut monitor).is_pending());
        health_tx.send(false).expect("publish unhealthy LLM");

        let Poll::Ready(result) = futures::poll!(&mut monitor) else {
            panic!("health monitor did not react to an unhealthy LLM");
        };
        assert_eq!(
            result.unwrap_err().to_string(),
            "LLM became permanently unhealthy"
        );
        assert!(shutdown.is_cancelled());
    }

    #[tokio::test]
    async fn unhealthy_llm_remains_fatal_when_shutdown_is_already_requested() {
        let (health_tx, health) = watch::channel(true);
        let shutdown = CancellationToken::new();

        health_tx.send(false).expect("publish unhealthy LLM");
        shutdown.cancel();

        let result = monitor_llm_health(health, shutdown, false).await;
        assert_eq!(
            result.unwrap_err().to_string(),
            "LLM became permanently unhealthy"
        );
    }

    #[tokio::test]
    async fn keep_alive_ignores_unhealthy_llm_until_shutdown() {
        let (health_tx, health) = watch::channel(true);
        let shutdown = CancellationToken::new();
        let mut monitor = Box::pin(monitor_llm_health(health, shutdown.clone(), true));

        assert!(futures::poll!(&mut monitor).is_pending());
        health_tx.send(false).expect("publish unhealthy LLM");
        assert!(futures::poll!(&mut monitor).is_pending());
        assert!(!shutdown.is_cancelled());

        shutdown.cancel();
        let Poll::Ready(result) = futures::poll!(&mut monitor) else {
            panic!("health monitor did not finish during shutdown");
        };
        result.expect("keep-alive shutdown is graceful");
    }
}
