// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context as _, Result, bail};
use serde_json::Value;
use tokio_util::sync::CancellationToken;
use tracing::info;
use vllm_chat::{
    ChatRequestProcessor, ChatTemplateContentFormatOption, LoadModelBackendsOptions,
    ParserSelection, RendererSelection, load_model_backends,
};
use vllm_text::TextRequestProcessor;

use crate::{HttpListenerMode, listener::Listener};

/// Configuration for the engine-free text rendering server.
#[derive(Debug)]
pub struct RenderConfig {
    pub model: String,
    pub served_model_name: Vec<String>,
    pub host: String,
    pub port: u16,
    pub tool_call_parser: ParserSelection,
    pub reasoning_parser: ParserSelection,
    pub renderer: RendererSelection,
    pub chat_template: Option<String>,
    pub default_chat_template_kwargs: HashMap<String, Value>,
    pub chat_template_content_format: ChatTemplateContentFormatOption,
    pub max_model_len: u32,
    pub max_logprobs: Option<i32>,
}

impl RenderConfig {
    /// Validate configuration before initializing renderer/tokenizer backends
    /// or binding a listener.
    pub fn validate(&self) -> Result<()> {
        vllm_chat::validate_parser_overrides(&self.tool_call_parser, &self.reasoning_parser)?;
        if self.max_logprobs.is_some_and(|value| value < -1) {
            bail!("max_logprobs must be non-negative or -1");
        }
        Ok(())
    }
}

pub(crate) struct RenderState {
    pub(crate) model: String,
    pub(crate) served_model_names: Vec<String>,
    pub(crate) tool_call_parser: ParserSelection,
    pub(crate) reasoning_parser: ParserSelection,
    pub(crate) text: TextRequestProcessor,
    pub(crate) chat: ChatRequestProcessor,
}

async fn build_state(config: &RenderConfig) -> Result<Arc<RenderState>> {
    let loaded = load_model_backends(
        &config.model,
        LoadModelBackendsOptions {
            renderer: config.renderer,
            language_model_only: true,
            chat_template: config.chat_template.clone(),
            chat_template_content_format: config.chat_template_content_format,
            default_chat_template_kwargs: config.default_chat_template_kwargs.clone(),
            limit_mm_per_prompt: Default::default(),
        },
    )
    .await
    .context("failed to load renderer/tokenizer backends")?;
    let served_model_names =
        crate::effective_served_model_names(&config.model, &config.served_model_name);
    let text = TextRequestProcessor::new(loaded.text_backend, config.max_model_len)
        .with_max_logprobs(config.max_logprobs);
    let chat = ChatRequestProcessor::render_only(loaded.chat_backend);
    Ok(Arc::new(RenderState {
        model: config.model.clone(),
        served_model_names,
        tool_call_parser: config.tool_call_parser.clone(),
        reasoning_parser: config.reasoning_parser.clone(),
        text,
        chat,
    }))
}

/// Run the text-only preprocessing server without starting or connecting
/// to an inference engine.
pub async fn serve_render(config: RenderConfig, shutdown: CancellationToken) -> Result<()> {
    config.validate().context("invalid render server configuration")?;
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
    info!(
        %address,
        model = %config.model,
        "starting engine-free Rust render server"
    );
    axum::serve(listener, crate::routes::render::build_router(state))
        .with_graceful_shutdown(shutdown.cancelled_owned())
        .await
        .context("render server failed")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn serve_render_rejects_unknown_parser_before_startup() {
        let shutdown = CancellationToken::new();
        shutdown.cancel();
        let error = serve_render(
            RenderConfig {
                model: "test-model".to_string(),
                served_model_name: Vec::new(),
                host: "127.0.0.1".to_string(),
                port: 8000,
                tool_call_parser: ParserSelection::Auto,
                reasoning_parser: ParserSelection::Explicit("typo".to_string()),
                renderer: RendererSelection::Auto,
                chat_template: None,
                default_chat_template_kwargs: HashMap::new(),
                chat_template_content_format: ChatTemplateContentFormatOption::Auto,
                max_model_len: 128,
                max_logprobs: None,
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
