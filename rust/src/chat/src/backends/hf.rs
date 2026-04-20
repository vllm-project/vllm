use std::sync::Arc;

use tracing::info;
use vllm_text::DynTextBackend;
use vllm_text::backends::hf::{HfTextBackend, ResolvedModelFiles, load_model_config};

use crate::RendererSelection;
use crate::backend::{ChatBackend, DynChatBackend};
use crate::backends::{LoadModelBackendsOptions, LoadedModelBackends};
use crate::error::Result;
use crate::renderers::hf::HfChatRenderer;
use crate::renderers::{DeepSeekV32ChatRenderer, DynChatRenderer};

/// [`ChatBackend`] implementation built on Hugging Face model files.
pub struct HfChatBackend {
    chat_renderer: DynChatRenderer,
}

impl HfChatBackend {
    /// Load the chat backend with the given model id.
    pub async fn from_model(model_id: &str) -> Result<Self> {
        let files = ResolvedModelFiles::new(model_id).await?;
        Self::from_resolved_model_files(files, model_id.to_string(), Default::default())
    }

    /// Load the chat backend from resolved Hugging Face model files.
    pub fn from_resolved_model_files(
        files: ResolvedModelFiles,
        model_id: String,
        options: LoadModelBackendsOptions,
    ) -> Result<Self> {
        let model_config = load_model_config(files.config_path.as_deref())?;
        let model_type = model_config.model_type().unwrap_or_default();

        let renderer = options.renderer.resolve(model_type);
        let chat_renderer: DynChatRenderer = match renderer {
            RendererSelection::Auto => unreachable!("renderer auto should be resolved above"),
            RendererSelection::Hf => Arc::new(HfChatRenderer::load(&files, options)?),
            RendererSelection::DeepSeekV32 => Arc::new(DeepSeekV32ChatRenderer::new()),
        };

        info!(
            model_id,
            model_type,
            %renderer,
            "loaded chat backend with Hugging Face model files"
        );

        Ok(Self { chat_renderer })
    }
}

impl ChatBackend for HfChatBackend {
    fn chat_renderer(&self) -> DynChatRenderer {
        self.chat_renderer.clone()
    }
}

/// Load the Hugging Face text and chat backends for the given model id.
pub(super) async fn load_model_backends(
    model_id: &str,
    options: LoadModelBackendsOptions,
) -> Result<LoadedModelBackends> {
    let files = ResolvedModelFiles::new(model_id).await?;
    let text_backend: DynTextBackend = Arc::new(HfTextBackend::from_resolved_model_files(
        files.clone(),
        model_id.to_string(),
    )?);
    let chat_backend: DynChatBackend = Arc::new(HfChatBackend::from_resolved_model_files(
        files,
        model_id.to_string(),
        options,
    )?);

    Ok(LoadedModelBackends {
        text_backend,
        chat_backend,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;

    use tempfile::tempdir;
    use vllm_text::backends::hf::TokenizerSource;

    use super::HfChatBackend;
    use crate::RendererSelection;
    use crate::backend::ChatBackend;
    use crate::backends::LoadModelBackendsOptions;
    use crate::request::{ChatContent, ChatMessage, ChatRequest};

    fn request_with_user_text(text: &str) -> ChatRequest {
        ChatRequest {
            request_id: "renderer-selection-test".to_string(),
            messages: vec![ChatMessage::User {
                content: ChatContent::Text(text.to_string()),
            }],
            ..ChatRequest::for_test()
        }
    }

    fn write_json(path: &std::path::Path, content: &str) {
        std::fs::write(path, content).unwrap();
    }

    fn resolved_files(
        config_json: &str,
        tokenizer_config_json: &str,
    ) -> vllm_text::backends::hf::ResolvedModelFiles {
        let dir = tempdir().unwrap();
        let root = dir.keep();
        let config_path = root.join("config.json");
        let tokenizer_config_path = root.join("tokenizer_config.json");
        write_json(&config_path, config_json);
        write_json(&tokenizer_config_path, tokenizer_config_json);

        vllm_text::backends::hf::ResolvedModelFiles {
            tokenizer: TokenizerSource::HuggingFace(PathBuf::from("/tmp/unused-tokenizer.json")),
            tokenizer_config_path: Some(tokenizer_config_path),
            generation_config_path: None,
            chat_template_path: None,
            config_path: Some(config_path),
        }
    }

    fn render_prompt(
        renderer: RendererSelection,
        config_json: &str,
        tokenizer_config_json: &str,
    ) -> String {
        let backend = HfChatBackend::from_resolved_model_files(
            resolved_files(config_json, tokenizer_config_json),
            "test-model".to_string(),
            LoadModelBackendsOptions {
                renderer,
                chat_template_content_format: Default::default(),
                chat_template: None,
                default_chat_template_kwargs: HashMap::new(),
            },
        )
        .unwrap();

        backend
            .chat_renderer()
            .render(&request_with_user_text("hello"))
            .unwrap()
            .prompt
    }

    #[test]
    fn auto_uses_deepseek_renderer_for_deepseek_v32_model_type() {
        let prompt = render_prompt(
            RendererSelection::Auto,
            r#"{"model_type":"deepseek_v32"}"#,
            r#"{}"#,
        );

        assert_eq!(
            prompt,
            "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>"
        );
    }

    #[test]
    fn auto_uses_hf_renderer_for_other_model_types() {
        let prompt = render_prompt(
            RendererSelection::Auto,
            r#"{"model_type":"qwen2"}"#,
            r#"{"chat_template":"{{ messages[0].content }}"}"#,
        );

        assert_eq!(prompt, "hello");
    }

    #[test]
    fn explicit_deepseek_renderer_overrides_generic_model_type() {
        let prompt = render_prompt(
            RendererSelection::DeepSeekV32,
            r#"{"model_type":"qwen2"}"#,
            r#"{"chat_template":"{{ messages[0].content }}"}"#,
        );

        assert_eq!(
            prompt,
            "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>"
        );
    }

    #[test]
    fn explicit_hf_renderer_overrides_deepseek_v32_model_type() {
        let prompt = render_prompt(
            RendererSelection::Hf,
            r#"{"model_type":"deepseek_v32"}"#,
            r#"{"chat_template":"{{ messages[0].content }}"}"#,
        );

        assert_eq!(prompt, "hello");
    }

    #[test]
    fn auto_uses_nested_text_config_model_type() {
        let prompt = render_prompt(
            RendererSelection::Auto,
            r#"{"text_config":{"model_type":"deepseek_v32","num_attention_heads":32}}"#,
            r#"{}"#,
        );

        assert_eq!(
            prompt,
            "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>"
        );
    }
}
