use std::sync::Arc;

use tracing::info;
use vllm_text::backend::hf::{HfTextBackend, ResolvedModelFiles, load_model_config};
use vllm_text::tokenizer::DynTokenizer;
use vllm_text::{DynTextBackend, TextBackend as _};

use crate::backend::{
    ChatBackend, DynChatBackend, LoadModelBackendsOptions, LoadedModelBackends,
    NewChatOutputProcessorOptions,
};
use crate::error::Result;
use crate::multimodal::MultimodalModelInfo;
use crate::output::{
    DefaultChatOutputProcessor, HarmonyChatOutputProcessor, validate_harmony_parser_overrides,
};
use crate::renderer::hf::{HfChatRenderer, MultimodalRenderInfo};
use crate::renderer::{
    DeepSeekV4ChatRenderer, DeepSeekV32ChatRenderer, DynChatRenderer, HarmonyChatRenderer,
};
use crate::request::ChatRequest;
use crate::{DynChatOutputProcessor, RendererSelection};

/// [`ChatBackend`] implementation built on Hugging Face model files.
pub struct HfChatBackend {
    model_id: String,
    model_type: String,
    tokenizer: DynTokenizer,
    chat_renderer: DynChatRenderer,
    multimodal_model_info: Option<MultimodalModelInfo>,
}

impl HfChatBackend {
    /// Load the chat backend from resolved Hugging Face model files.
    pub fn from_resolved_model_files(
        files: ResolvedModelFiles,
        model_id: String,
        options: LoadModelBackendsOptions,
        tokenizer: DynTokenizer,
    ) -> Result<Self> {
        let model_config = load_model_config(files.config_path.as_deref())?;
        let model_type = model_config.model_type().unwrap_or_default();
        let multimodal_model_info = if options.language_model_only {
            None
        } else {
            MultimodalModelInfo::from_paths(
                model_id.clone(),
                (!model_type.is_empty()).then_some(model_type.to_string()),
                files.config_path.as_deref(),
                files.preprocessor_config_path.as_deref(),
                tokenizer.clone(),
            )?
        };
        let multimodal_render_info = resolve_multimodal_render_info(multimodal_model_info.as_ref());

        let renderer = options.renderer.resolve(model_type);
        let chat_renderer: DynChatRenderer = match renderer {
            RendererSelection::Auto => unreachable!("renderer auto should be resolved above"),
            RendererSelection::Hf => Arc::new(HfChatRenderer::load(
                &files,
                options,
                multimodal_render_info,
            )?),
            RendererSelection::DeepSeekV32 => Arc::new(DeepSeekV32ChatRenderer::new()),
            RendererSelection::DeepSeekV4 => Arc::new(DeepSeekV4ChatRenderer::new()),
            RendererSelection::Harmony => Arc::new(HarmonyChatRenderer::new()?),
        };

        info!(
            model_id,
            model_type,
            %renderer,
            "loaded chat backend with Hugging Face model files"
        );

        Ok(Self {
            model_id,
            model_type: model_type.to_string(),
            tokenizer,
            chat_renderer,
            multimodal_model_info,
        })
    }
}

impl ChatBackend for HfChatBackend {
    fn chat_renderer(&self) -> DynChatRenderer {
        self.chat_renderer.clone()
    }

    fn multimodal_model_info(&self) -> Option<&MultimodalModelInfo> {
        self.multimodal_model_info.as_ref()
    }

    fn new_chat_output_processor(
        &self,
        request: &mut ChatRequest,
        options: NewChatOutputProcessorOptions<'_>,
    ) -> Result<DynChatOutputProcessor> {
        if self.model_type == "gpt_oss" {
            validate_harmony_parser_overrides(options.tool_call_parser, options.reasoning_parser)?;
            return Ok(Box::new(HarmonyChatOutputProcessor::new(request)?));
        }

        Ok(Box::new(DefaultChatOutputProcessor::new(
            request,
            &self.model_id,
            self.tokenizer.clone(),
            options.tool_call_parser,
            options.reasoning_parser,
        )?))
    }
}

/// Load the Hugging Face text and chat backends for the given model id.
pub(super) async fn load_model_backends(
    model_id: &str,
    options: LoadModelBackendsOptions,
) -> Result<LoadedModelBackends> {
    let files = ResolvedModelFiles::new(model_id).await?;
    let text_backend =
        HfTextBackend::from_resolved_model_files(files.clone(), model_id.to_string())?;
    let tokenizer = text_backend.tokenizer();
    let text_backend: DynTextBackend = Arc::new(text_backend);

    let chat_backend: DynChatBackend = Arc::new(HfChatBackend::from_resolved_model_files(
        files,
        model_id.to_string(),
        options,
        tokenizer,
    )?);

    Ok(LoadedModelBackends {
        text_backend,
        chat_backend,
    })
}

fn resolve_multimodal_render_info(
    info: Option<&MultimodalModelInfo>,
) -> Option<MultimodalRenderInfo> {
    info.map(|info| MultimodalRenderInfo {
        placeholder_token: info.placeholder_token().to_string(),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::Arc;

    use tempfile::tempdir;
    use thiserror_ext::AsReport as _;
    use vllm_text::Prompt;
    use vllm_text::backend::hf::TokenizerSource;
    use vllm_text::tokenizer::DynTokenizer;
    use vllm_tokenizer::test_utils::TestTokenizer;

    use super::HfChatBackend;
    use crate::backend::{ChatBackend, LoadModelBackendsOptions, NewChatOutputProcessorOptions};
    use crate::request::{ChatContent, ChatMessage, ChatRequest};
    use crate::{ParserSelection, RendererSelection};

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
    ) -> vllm_text::backend::hf::ResolvedModelFiles {
        let dir = tempdir().unwrap();
        let root = dir.keep();
        let config_path = root.join("config.json");
        let tokenizer_config_path = root.join("tokenizer_config.json");
        write_json(&config_path, config_json);
        write_json(&tokenizer_config_path, tokenizer_config_json);

        vllm_text::backend::hf::ResolvedModelFiles {
            tokenizer: TokenizerSource::HuggingFace(PathBuf::from("/tmp/unused-tokenizer.json")),
            tokenizer_config_path: Some(tokenizer_config_path),
            generation_config_path: None,
            preprocessor_config_path: None,
            chat_template_path: None,
            config_path: Some(config_path),
        }
    }

    fn test_tokenizer() -> DynTokenizer {
        Arc::new(TestTokenizer::new())
    }

    fn backend_for_selection(
        renderer: RendererSelection,
        config_json: &str,
        tokenizer_config_json: &str,
    ) -> HfChatBackend {
        HfChatBackend::from_resolved_model_files(
            resolved_files(config_json, tokenizer_config_json),
            "test-model".to_string(),
            LoadModelBackendsOptions {
                renderer,
                language_model_only: false,
                chat_template_content_format: Default::default(),
                chat_template: None,
                default_chat_template_kwargs: HashMap::new(),
            },
            test_tokenizer(),
        )
        .unwrap()
    }

    fn render_prompt(
        renderer: RendererSelection,
        config_json: &str,
        tokenizer_config_json: &str,
    ) -> String {
        backend_for_selection(renderer, config_json, tokenizer_config_json)
            .chat_renderer()
            .render(&request_with_user_text("hello"))
            .unwrap()
            .prompt
            .into_text()
            .expect("renderer should return text prompt")
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
    fn auto_uses_harmony_renderer_and_output_processor_for_gpt_oss_model_type() {
        let backend = backend_for_selection(
            RendererSelection::Auto,
            r#"{"model_type":"gpt_oss"}"#,
            r#"{"chat_template":"{{ messages[0].content }}"}"#,
        );

        let prompt =
            backend.chat_renderer().render(&request_with_user_text("hello")).unwrap().prompt;
        assert!(matches!(prompt, Prompt::TokenIds(_)));

        let mut request = request_with_user_text("hello");
        let error = match backend.new_chat_output_processor(
            &mut request,
            NewChatOutputProcessorOptions {
                tool_call_parser: &ParserSelection::Explicit("json".to_string()),
                reasoning_parser: &ParserSelection::Auto,
            },
        ) {
            Ok(_) => panic!("gpt_oss should reject generic parser overrides"),
            Err(error) => error,
        };
        assert_eq!(
            error.to_report_string(),
            "gpt_oss uses native Harmony output parsing; generic tool parser override `json` is not supported"
        );
    }

    #[test]
    fn language_model_only_skips_multimodal_preprocessor_config() {
        let mut files = resolved_files(
            r#"{"model_type":"deepseek_v0_vl"}"#,
            r#"{"chat_template":"{{ messages[0].content }}"}"#,
        );
        let preprocessor_config_path = files
            .config_path
            .as_ref()
            .unwrap()
            .parent()
            .unwrap()
            .join("preprocessor_config.json");
        write_json(&preprocessor_config_path, r#"{"size":[672,672]}"#);
        files.preprocessor_config_path = Some(preprocessor_config_path);

        let backend = HfChatBackend::from_resolved_model_files(
            files.clone(),
            "test-model".to_string(),
            LoadModelBackendsOptions {
                language_model_only: true,
                chat_template_content_format: Default::default(),
                chat_template: None,
                default_chat_template_kwargs: HashMap::new(),
                ..Default::default()
            },
            test_tokenizer(),
        )
        .unwrap();

        assert!(backend.multimodal_model_info().is_none());

        let error = HfChatBackend::from_resolved_model_files(
            files,
            "test-model".to_string(),
            LoadModelBackendsOptions {
                chat_template_content_format: Default::default(),
                chat_template: None,
                default_chat_template_kwargs: HashMap::new(),
                ..Default::default()
            },
            test_tokenizer(),
        )
        .err()
        .expect("invalid preprocessor config should fail without language_model_only");
        assert!(error.to_string().contains("failed to parse preprocessor_config.json"));
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
