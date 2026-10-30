// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;
use std::str::FromStr;
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

use crate::{
    DEFAULT_REQUEST_BODY_LIMIT_BYTES, GRPC_KEEPALIVE_INTERVAL, GRPC_KEEPALIVE_TIMEOUT,
    HttpListenerMode, LoraModulePath, ReasoningEffort, TlsConfig, grpc,
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
    pub hf_overrides: HfOverrides,
    pub served_model_name: Vec<String>,
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
    pub dtype: RenderDtype,
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

/// Model dtype selection for the render server, which has no engine to report
/// it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RenderDtype {
    /// Follow the model config, like vLLM's `--dtype auto`.
    #[default]
    Auto,
    Fixed(ModelDtype),
}

impl FromStr for RenderDtype {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Ok(match value {
            "auto" => Self::Auto,
            "half" | "float16" => Self::Fixed(ModelDtype::Float16),
            "bfloat16" => Self::Fixed(ModelDtype::BFloat16),
            "float" | "float32" => Self::Fixed(ModelDtype::Float32),
            _ => {
                return Err(format!(
                    "unsupported dtype `{value}`; expected auto, half, float16, bfloat16, float, or float32"
                ));
            }
        })
    }
}

impl RenderDtype {
    /// Resolve the dtype from the checkpoint's config dtype the way vLLM's
    /// `auto` does: float32 or missing dtypes run in bfloat16.
    fn resolve(self, config_dtype: Option<&str>) -> Result<ModelDtype> {
        match self {
            Self::Fixed(dtype) => Ok(dtype),
            Self::Auto => match config_dtype.unwrap_or("float32") {
                "float16" => Ok(ModelDtype::Float16),
                "bfloat16" | "float32" => Ok(ModelDtype::BFloat16),
                other => {
                    bail!("cannot infer the model dtype from config dtype `{other}`; pass --dtype")
                }
            },
        }
    }
}

pub(crate) struct RenderState {
    pub(crate) model: String,
    pub(crate) served_model_names: Vec<String>,
    /// Unlike `text.max_model_len()` this stays `None` when unset,
    /// so model cards advertise `null` instead of the `u32::MAX`.
    pub(crate) max_model_len: Option<u32>,
    pub(crate) text: TextRequestProcessor,
    pub(crate) chat: ChatRequestProcessor,
    pub(crate) lora_modules: Vec<LoraModulePath>,
}

impl RenderState {
    /// Served model names followed by LoRA adapter names.
    pub(crate) fn model_names(&self) -> Vec<String> {
        (self.served_model_names.iter())
            .chain(self.lora_modules.iter().map(|module| &module.name))
            .cloned()
            .collect()
    }
}

async fn build_state(config: &RenderConfig) -> Result<Arc<RenderState>> {
    let loaded = load_model_backends(
        &config.model,
        LoadModelBackendsOptions {
            revision: config.revision.clone(),
            tokenizer: config.tokenizer.clone(),
            hf_config: config.hf_config.clone(),
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
    let served_model_names =
        crate::effective_served_model_names(&config.model, &config.served_model_name);
    // Only multimodal preprocessing reads the dtype, so text-only models never
    // fail on an unusual config dtype.
    let model_dtype = loaded
        .chat_backend
        .multimodal_model_info()
        .is_some()
        .then(|| config.dtype.resolve(loaded.text_backend.config_dtype()))
        .transpose()?;
    let max_model_len = config.max_model_len.unwrap_or(u32::MAX);
    let text = TextRequestProcessor::new(loaded.text_backend, max_model_len)
        .with_max_logprobs(config.max_logprobs);
    let mut chat = ChatRequestProcessor::render_only(loaded.chat_backend)
        .with_max_model_len(config.max_model_len)
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
    if let Some(model_dtype) = model_dtype {
        info!(
            dtype = model_dtype.as_str(),
            "multimodal preprocessing enabled"
        );
        chat = chat.with_model_dtype(model_dtype);
    }
    Ok(Arc::new(RenderState {
        model: config.model.clone(),
        served_model_names,
        max_model_len: config.max_model_len,
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
        let (health_reporter, health_service) = health_reporter();
        health_reporter.set_serving::<grpc::RenderGrpcService<RenderState>>().await;
        let render_service =
            grpc::RenderGrpcService::new(grpc::RenderServiceImpl::new(state.clone()))
                .max_decoding_message_size(DEFAULT_REQUEST_BODY_LIMIT_BYTES);
        info!(%address, "starting engine-free Rust render gRPC server");
        let result = TonicServer::builder()
            .http2_keepalive_interval(Some(GRPC_KEEPALIVE_INTERVAL))
            .http2_keepalive_timeout(Some(GRPC_KEEPALIVE_TIMEOUT))
            .add_service(health_service)
            .add_service(render_service)
            .serve_with_incoming_shutdown(grpc_listener, server_shutdown.clone().cancelled_owned())
            .await
            .context("render gRPC server failed");
        server_shutdown.cancel();
        result
    };
    let (http_result, grpc_result) = tokio::join!(http_fut, grpc_fut);
    http_result.and(grpc_result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_dtype_auto_follows_config_like_vllm() {
        let auto = RenderDtype::Auto;
        assert_eq!(auto.resolve(Some("float16")).unwrap(), ModelDtype::Float16);
        assert_eq!(
            auto.resolve(Some("bfloat16")).unwrap(),
            ModelDtype::BFloat16
        );
        assert_eq!(auto.resolve(Some("float32")).unwrap(), ModelDtype::BFloat16);
        assert_eq!(auto.resolve(None).unwrap(), ModelDtype::BFloat16);
        assert!(auto.resolve(Some("float8_e4m3fn")).is_err());

        let fixed: RenderDtype = "half".parse().unwrap();
        assert_eq!(
            fixed.resolve(Some("bfloat16")).unwrap(),
            ModelDtype::Float16
        );
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
                hf_overrides: Default::default(),
                served_model_name: Vec::new(),
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
                dtype: RenderDtype::Auto,
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
