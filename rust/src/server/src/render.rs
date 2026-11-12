// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::{Context as _, Result, bail};
use serde_json::Value;
use tokio_util::sync::CancellationToken;
use tonic::transport::Server as TonicServer;
use tonic_health::server::health_reporter;
use tracing::info;
use vllm_chat::multimodal::MmLimitPerPrompt;
use vllm_chat::{
    ChatRequestProcessor, ChatTemplateContentFormatOption, LoadModelBackendsOptions,
    ParserSelection, RendererSelection, ToolStrictLevel, load_model_backends,
};
use vllm_engine_core_client::protocol::dtype::ModelDtype;
use vllm_text::TextRequestProcessor;
use vllm_text::backend::hf::{HfOverrides, ModelSource};

use crate::lora::check_lora_module;
use crate::model_names::ServedModelNames;
use crate::serving::{GRPC_KEEPALIVE_INTERVAL, GRPC_KEEPALIVE_TIMEOUT};
use crate::{
    DEFAULT_REQUEST_BODY_LIMIT_BYTES, HttpListenerMode, LoraModulePath, ReasoningEffort, TlsConfig,
    grpc,
    listener::{Listener, MaybeTlsListener},
    tls,
};

/// Configuration for the engine-free text rendering server.
#[derive(Debug)]
pub struct RenderConfig {
    pub model: String,
    pub revision: Option<String>,
    /// Source of the tokenizer files and the chat template, or `None` for `model`.
    pub tokenizer: Option<ModelSource>,
    /// Source of `config.json` and `generation_config.json`, or `None` for `model`.
    pub hf_config: Option<ModelSource>,
    /// Load no tokenizer and serve token IDs only (`--skip-tokenizer-init`).
    pub skip_tokenizer_init: bool,
    pub hf_overrides: HfOverrides,
    pub served_model_name: Vec<String>,
    /// Accept any requested model name (`VLLM_SKIP_MODEL_NAME_VALIDATION`).
    pub skip_model_name_validation: bool,
    /// Upper bound on a request's `n` (`VLLM_MAX_N_SEQUENCES`).
    pub max_n_sequences: i64,
    pub host: String,
    pub port: u16,
    /// Port for the gRPC Render service on `host`; `None` disables gRPC.
    pub grpc_port: Option<u16>,
    pub tool_call_parser: ParserSelection,
    pub reasoning_parser: ParserSelection,
    pub tool_strict_level: Option<ToolStrictLevel>,
    pub renderer: RendererSelection,
    pub chat_template: Option<String>,
    pub default_chat_template_kwargs: HashMap<String, Value>,
    pub reasoning_effort_map: HashMap<ReasoningEffort, String>,
    pub trust_request_chat_template: bool,
    pub exclude_tools_when_tool_choice_none: bool,
    pub chat_template_content_format: ChatTemplateContentFormatOption,
    pub max_model_len: Option<u32>,
    pub max_logprobs: Option<i32>,
    /// LoRA adapters whose names are accepted as the request model. Rendering
    /// does not depend on adapter weights, so they are not loaded.
    pub lora_modules: Vec<LoraModulePath>,
    /// Disable multimodal preprocessing and treat the model as language-only.
    pub language_model_only: bool,
    /// Dtype multimodal tensors are produced in; must match the engine's.
    /// `None` follows the model config, like vLLM's `--dtype auto`.
    pub dtype: Option<ModelDtype>,
    /// Maximum multimodal items per prompt for each modality.
    pub limit_mm_per_prompt: MmLimitPerPrompt,
    pub tls: Option<TlsConfig>,
}

impl RenderConfig {
    /// Validate configuration before initializing renderer/tokenizer backends
    /// or binding a listener.
    pub fn validate(&self) -> Result<()> {
        vllm_chat::validate_parser_overrides(&self.tool_call_parser, &self.reasoning_parser)?;
        if self.max_logprobs.is_some_and(|value| value < -1) {
            bail!("max_logprobs must be non-negative or -1");
        }
        if let Some(tls) = &self.tls {
            tls.validate()?;
        }
        Ok(())
    }
}

/// Resolve the model dtype from the checkpoint's config dtype the way vLLM's
/// `--dtype auto` does on GPUs with bfloat16: float32, missing, and
/// unrecognized config dtypes run in bfloat16.
fn auto_model_dtype(config_dtype: Option<ModelDtype>) -> ModelDtype {
    match config_dtype {
        Some(ModelDtype::Float16) => ModelDtype::Float16,
        Some(ModelDtype::BFloat16 | ModelDtype::Float32) | None => ModelDtype::BFloat16,
    }
}

pub(crate) struct RenderState {
    pub(crate) model: String,
    pub(crate) served_model_names: ServedModelNames,
    /// Unlike `text.max_model_len()` this stays `None` when unset,
    /// so model cards advertise `null` instead of the `u32::MAX`.
    pub(crate) max_model_len: Option<u32>,
    /// Upper bound on a request's `n` (`VLLM_MAX_N_SEQUENCES`).
    pub(crate) max_n_sequences: i64,
    pub(crate) text: TextRequestProcessor,
    pub(crate) chat: ChatRequestProcessor,
    pub(crate) lora_modules: Vec<LoraModulePath>,
}

impl RenderState {
    /// Served model names followed by LoRA adapter names.
    pub(crate) fn model_names(&self) -> ServedModelNames {
        (self.served_model_names.clone())
            .with_names(self.lora_modules.iter().map(|module| module.name.clone()))
    }
}

/// Check `--lora-modules` the way `vllm-rs serve` does when it loads them.
fn validate_lora_modules(
    served_model_names: &ServedModelNames,
    lora_modules: &[LoraModulePath],
) -> Result<()> {
    let mut seen = HashSet::new();
    for module in lora_modules {
        let name = &module.name;
        check_lora_module(served_model_names.names(), module)
            .with_context(|| format!("invalid --lora-modules entry `{name}`"))?;
        if !seen.insert(name) {
            bail!("LoRA adapter `{name}` is given more than once");
        }
    }
    Ok(())
}

async fn build_state(config: &RenderConfig) -> Result<Arc<RenderState>> {
    let loaded = load_model_backends(
        &config.model,
        LoadModelBackendsOptions {
            revision: config.revision.clone(),
            tokenizer: config.tokenizer.clone(),
            hf_config: config.hf_config.clone(),
            skip_tokenizer_init: config.skip_tokenizer_init,
            hf_overrides: config.hf_overrides.clone(),
            generation_config: Default::default(),
            renderer: config.renderer,
            language_model_only: config.language_model_only,
            chat_template: config.chat_template.clone(),
            trust_request_chat_template: config.trust_request_chat_template,
            chat_template_content_format: config.chat_template_content_format,
            default_chat_template_kwargs: config.default_chat_template_kwargs.clone(),
            limit_mm_per_prompt: config.limit_mm_per_prompt.clone(),
        },
    )
    .await
    .context("failed to load renderer/tokenizer backends")?;
    let served_model_names = ServedModelNames::new(crate::effective_served_model_names(
        &config.model,
        &config.served_model_name,
    ))
    .with_accept_any(config.skip_model_name_validation);
    validate_lora_modules(&served_model_names, &config.lora_modules)?;
    // Only multimodal preprocessing reads the dtype.
    let model_dtype = (loaded.chat_backend.multimodal_model_info().is_some()).then(|| {
        (config.dtype).unwrap_or_else(|| auto_model_dtype(loaded.text_backend.config_dtype()))
    });
    if let Some(model_dtype) = model_dtype {
        info!(
            dtype = model_dtype.as_str(),
            "multimodal preprocessing enabled"
        );
    }
    let max_model_len = config.max_model_len.unwrap_or(u32::MAX);
    let text = TextRequestProcessor::new(loaded.text_backend, max_model_len)
        .with_max_logprobs(config.max_logprobs);
    let chat = ChatRequestProcessor::new(loaded.chat_backend, model_dtype, config.max_model_len)
        .with_parser_selections(
            config.tool_call_parser.clone(),
            config.reasoning_parser.clone(),
        )
        .with_tool_strict_level(config.tool_strict_level)
        .with_reasoning_effort_map(
            config
                .reasoning_effort_map
                .iter()
                .map(|(effort, value)| (effort.as_str().to_owned(), value.clone()))
                .collect(),
        )
        .with_exclude_tools_when_tool_choice_none(config.exclude_tools_when_tool_choice_none);
    Ok(Arc::new(RenderState {
        model: config.model.clone(),
        served_model_names,
        max_model_len: config.max_model_len,
        max_n_sequences: config.max_n_sequences,
        text,
        chat,
        lora_modules: config.lora_modules.clone(),
    }))
}

/// Run the text-only preprocessing server without starting or connecting
/// to an inference engine.
pub async fn serve_render(config: RenderConfig, shutdown: CancellationToken) -> Result<()> {
    config.validate().context("invalid render server configuration")?;
    let tls_config = config
        .tls
        .as_ref()
        .map(tls::build_server_config)
        .transpose()
        .context("invalid TLS configuration")?;
    // gRPC reuses the HTTP TLS config plus ALPN h2.
    let grpc_tls_config = match (config.grpc_port, &config.tls) {
        (Some(_), Some(tls)) => {
            Some(tls::build_grpc_server_config(tls).context("invalid gRPC TLS configuration")?)
        }
        _ => None,
    };
    let state = tokio::select! {
        result = build_state(&config) => result?,
        _ = shutdown.cancelled() => return Ok(()),
    };
    let listener_mode = HttpListenerMode::BindTcp {
        host: config.host.clone(),
        port: config.port,
    };
    let listener = Listener::bind(&listener_mode)
        .await
        .with_context(|| format!("failed to bind {}:{}", config.host, config.port))?;
    let address = listener.local_addr_display()?;
    let scheme = if tls_config.is_some() {
        "https"
    } else {
        "http"
    };
    let listener = match tls_config {
        Some(context) => MaybeTlsListener::tls(listener, context),
        None => MaybeTlsListener::plain(listener),
    };
    let grpc_listener = match config.grpc_port {
        Some(port) => {
            let mode = HttpListenerMode::BindTcp {
                host: config.host.clone(),
                port,
            };
            let listener = Listener::bind(&mode)
                .await
                .with_context(|| format!("failed to bind gRPC {}:{port}", config.host))?;
            let address = listener.local_addr_display()?;
            let listener = match grpc_tls_config {
                Some(context) => MaybeTlsListener::tls(listener, context),
                None => MaybeTlsListener::plain(listener),
            };
            Some((address, listener))
        }
        None => None,
    };
    info!(
        %address,
        scheme,
        model = %config.model,
        "starting engine-free Rust render server"
    );

    // If either protocol exits first, cancel the other one too.
    let server_shutdown = shutdown.child_token();
    let http_fut = async {
        let result = axum::serve(listener, crate::routes::render::build_router(state.clone()))
            .with_graceful_shutdown(server_shutdown.clone().cancelled_owned())
            .await
            .context("render server failed");
        server_shutdown.cancel();
        result
    };
    let grpc_fut = async {
        let Some((address, grpc_listener)) = grpc_listener else {
            return Ok(());
        };
        info!(%address, "starting engine-free Rust render gRPC server");
        let result = serve_render_grpc(state.clone(), grpc_listener, server_shutdown.clone()).await;
        server_shutdown.cancel();
        result
    };
    let (http_result, grpc_result) = tokio::join!(http_fut, grpc_fut);
    http_result.and(grpc_result)
}

/// Serve the gRPC Render service and its health service until `shutdown`.
pub(crate) async fn serve_render_grpc(
    state: Arc<RenderState>,
    listener: MaybeTlsListener,
    shutdown: CancellationToken,
) -> Result<()> {
    let (health_reporter, health_service) = health_reporter();
    let health_monitor =
        grpc::HealthMonitor::register(health_reporter, grpc::HealthSource::EngineFree).await;
    let render_service = grpc::RenderGrpcService::new(grpc::RenderServiceImpl::new(state))
        .max_decoding_message_size(DEFAULT_REQUEST_BODY_LIMIT_BYTES);
    let server = TonicServer::builder()
        .http2_keepalive_interval(Some(GRPC_KEEPALIVE_INTERVAL))
        .http2_keepalive_timeout(Some(GRPC_KEEPALIVE_TIMEOUT))
        .add_service(health_service)
        .add_service(render_service);
    health_monitor
        .serve(shutdown, std::future::pending(), |stop| {
            server.serve_with_incoming_shutdown(listener, stop)
        })
        .await
        .expect("never forced to abort")
        .context("render gRPC server failed")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lora_modules_are_checked_like_serve() {
        let served = ServedModelNames::new(vec!["qwen".to_string()]);
        let modules = |specs: &[&str]| -> Vec<LoraModulePath> {
            specs.iter().map(|spec| spec.parse().unwrap()).collect()
        };
        assert!(validate_lora_modules(&served, &modules(&["a=org/a", "b=org/b"])).is_ok());
        assert!(validate_lora_modules(&served, &modules(&["qwen=org/a"])).is_err());
        assert!(validate_lora_modules(&served, &modules(&["a=org/a", "a=org/b"])).is_err());
    }

    #[test]
    fn auto_model_dtype_follows_config_like_vllm() {
        for (config_dtype, expected) in [
            (Some(ModelDtype::Float16), ModelDtype::Float16),
            (Some(ModelDtype::BFloat16), ModelDtype::BFloat16),
            (Some(ModelDtype::Float32), ModelDtype::BFloat16),
            (None, ModelDtype::BFloat16),
        ] {
            assert_eq!(auto_model_dtype(config_dtype), expected);
        }
    }

    #[tokio::test]
    async fn serve_render_rejects_unknown_parser_before_startup() {
        let shutdown = CancellationToken::new();
        shutdown.cancel();
        let error = serve_render(
            RenderConfig {
                tool_strict_level: None,
                model: "test-model".to_string(),
                revision: None,
                tokenizer: None,
                hf_config: None,
                skip_tokenizer_init: false,
                hf_overrides: Default::default(),
                served_model_name: Vec::new(),
                skip_model_name_validation: false,
                max_n_sequences: crate::DEFAULT_MAX_N_SEQUENCES,
                host: "127.0.0.1".to_string(),
                port: 8000,
                grpc_port: None,
                tool_call_parser: ParserSelection::Auto,
                reasoning_parser: ParserSelection::Explicit("typo".to_string()),
                renderer: RendererSelection::Auto,
                chat_template: None,
                default_chat_template_kwargs: HashMap::new(),
                reasoning_effort_map: HashMap::new(),
                trust_request_chat_template: false,
                exclude_tools_when_tool_choice_none: false,
                chat_template_content_format: ChatTemplateContentFormatOption::Auto,
                max_model_len: Some(128),
                max_logprobs: None,
                lora_modules: Vec::new(),
                language_model_only: false,
                dtype: None,
                limit_mm_per_prompt: Default::default(),
                tls: None,
            },
            shutdown,
        )
        .await
        .unwrap_err();
        let report = format!("{error:#}");

        assert!(
            report.contains("reasoning parser `typo` is not registered"),
            "{report}"
        );
    }
}
