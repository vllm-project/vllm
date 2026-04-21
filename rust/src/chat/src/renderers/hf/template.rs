//! Chat template support for tokenizers using Jinja2 templates.
//!
//! This module is inlined from SMG's tokenizer crate with local adaptations:
//! - thinking-related detection/state is removed
//! - special tokens are wired to `vllm_text::backends::hf::HfSpecialTokens`

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use minijinja::Environment;
use serde::{Deserialize, Serialize};
use serde_json::{self};
use vllm_text::backends::hf::HfSpecialTokens;

use super::error::TemplateError;
use super::format::{
    ChatTemplateContentFormat, ChatTemplateContentFormatOption, detect_chat_template_content_format,
};
use super::tojson::hf_tojson_filter;
use crate::renderers::hf::{TemplateMessage, TemplateTool};

type Result<T> = std::result::Result<T, TemplateError>;

/// Build a pre-configured environment with the given template string.
fn build_environment(template: String) -> Result<Environment<'static>> {
    let mut env = Environment::new();

    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);

    env.add_template_owned("chat".to_owned(), template)?;

    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_filter("tojson", hf_tojson_filter);

    Ok(env)
}

#[serde_with::skip_serializing_none]
#[derive(Default, Serialize)]
pub(super) struct TemplateContext<'a> {
    pub(super) messages: &'a [TemplateMessage],
    pub(super) add_generation_prompt: bool,
    pub(super) continue_final_message: bool,
    pub(super) tools: Option<&'a [TemplateTool]>,
    pub(super) documents: Option<&'a [serde_json::Value]>,
    #[serde(flatten)]
    pub(super) special_tokens: Option<&'a HfSpecialTokens>,
    #[serde(flatten)]
    pub(super) template_kwargs: Option<&'a HashMap<String, serde_json::Value>>,
}

/// Load chat template from a file (`.jinja` or `.json` containing Jinja).
pub fn load_chat_template(template_path: &Path) -> Result<Option<String>> {
    let content = fs::read_to_string(template_path).map_err(TemplateError::ReadTemplateFile)?;

    if template_path.extension().is_some_and(|ext| ext == "json") {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum ChatTemplateFile {
            String(String),
            Object { chat_template: String },
        }

        let json_value =
            serde_json::from_str(&content).map_err(TemplateError::ParseTemplateJson)?;
        let json_template =
            serde_json::from_value(json_value).map_err(|_| TemplateError::InvalidTemplateJson)?;

        return Ok(Some(match json_template {
            ChatTemplateFile::String(template) => template,
            ChatTemplateFile::Object { chat_template } => chat_template,
        }));
    }

    let template = content.trim().replace("\\n", "\n");
    Ok(Some(template))
}

/// Resolve a configured chat template value into a template string.
pub fn resolve_chat_template(chat_template: &str) -> Result<String> {
    let path = Path::new(chat_template);
    if path.exists() {
        return load_chat_template(path).map(|template| template.unwrap_or_default());
    }

    const JINJA_CHARS: [char; 3] = ['{', '}', '\n'];
    if chat_template.chars().any(|c| JINJA_CHARS.contains(&c)) {
        return Ok(chat_template.to_string());
    }

    Err(TemplateError::MissingTemplatePath)
}

/// One compiled chat template with its Jinja environment and detected content format.
pub(super) struct CompiledChatTemplate {
    /// Cached, fully-configured environment for one compiled template.
    env: Environment<'static>,
    content_format: ChatTemplateContentFormat,
}

impl CompiledChatTemplate {
    /// Compile the given chat template string into a [`CompiledChatTemplate`].
    pub fn new(template: String, content_format: ChatTemplateContentFormatOption) -> Result<Self> {
        let content_format = match content_format {
            ChatTemplateContentFormatOption::Auto => detect_chat_template_content_format(&template),
            ChatTemplateContentFormatOption::String => ChatTemplateContentFormat::String,
            ChatTemplateContentFormatOption::OpenAi => ChatTemplateContentFormat::OpenAi,
        };
        let env = build_environment(template)?;
        Ok(Self {
            env,
            content_format,
        })
    }

    /// Apply the compiled template to the given context and return the rendered prompt.
    pub fn apply(&self, ctx: TemplateContext<'_>) -> Result<String> {
        let tmpl = self.env.get_template("chat")?;
        tmpl.render(ctx).map_err(TemplateError::from)
    }

    pub fn content_format(&self) -> ChatTemplateContentFormat {
        self.content_format
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::TempDir;
    use vllm_text::backends::hf::{HfSpecialTokens, NamedSpecialToken};

    use super::*;

    #[test]
    fn test_chat_template_state_valid_template() {
        let template = CompiledChatTemplate::new(
            "{{ messages }}".to_string(),
            ChatTemplateContentFormatOption::Auto,
        )
        .unwrap();
        assert_eq!(template.content_format(), ChatTemplateContentFormat::String);
        let result = template.apply(TemplateContext::default()).unwrap();
        assert_eq!(result, "[]");
    }

    #[test]
    fn test_chat_template_state_invalid_template() {
        let result = CompiledChatTemplate::new(
            "{% invalid".to_string(),
            ChatTemplateContentFormatOption::Auto,
        );
        assert!(result.is_err());
        let err = result.err().unwrap().to_string();
        assert!(
            err.contains("failed to render jinja template"),
            "Error should explain parse failure, got: {err}"
        );
    }

    #[test]
    fn test_special_tokens_injected_into_context() {
        let template = "{{ bos_token }}hello{{ eos_token }}";
        let template =
            CompiledChatTemplate::new(template.to_string(), ChatTemplateContentFormatOption::Auto)
                .unwrap();

        let special_tokens = HfSpecialTokens {
            bos_token: Some(NamedSpecialToken::Text("<s>".to_string())),
            eos_token: Some(NamedSpecialToken::Text("</s>".to_string())),
            ..Default::default()
        };

        let result = template
            .apply(TemplateContext {
                special_tokens: Some(&special_tokens),
                ..Default::default()
            })
            .unwrap();

        assert_eq!(result, "<s>hello</s>");
    }

    #[test]
    fn test_special_tokens_undefined_when_not_provided() {
        let template = "{% if bos_token is defined %}{{ bos_token }}{% endif %}hello";
        let template =
            CompiledChatTemplate::new(template.to_string(), ChatTemplateContentFormatOption::Auto)
                .unwrap();

        let result = template.apply(TemplateContext::default()).unwrap();
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_special_tokens_partial() {
        let template =
            "{{ bos_token }}hello{% if eos_token is defined %}{{ eos_token }}{% endif %}";
        let template =
            CompiledChatTemplate::new(template.to_string(), ChatTemplateContentFormatOption::Auto)
                .unwrap();

        let special_tokens = HfSpecialTokens {
            bos_token: Some(NamedSpecialToken::Text("<s>".to_string())),
            eos_token: None,
            ..Default::default()
        };

        let result = template
            .apply(TemplateContext {
                special_tokens: Some(&special_tokens),
                ..Default::default()
            })
            .unwrap();

        assert_eq!(result, "<s>hello");
    }

    #[test]
    fn test_tojson_filter_supports_indent_and_sort_keys() {
        let template = CompiledChatTemplate::new(
            "{{ payload | tojson(indent=2, sort_keys=true) }}".to_string(),
            ChatTemplateContentFormatOption::Auto,
        )
        .unwrap();
        let mut kwargs = HashMap::new();
        kwargs.insert("payload".to_string(), serde_json::json!({"b": 1, "a": 2}));

        let result = template
            .apply(TemplateContext {
                template_kwargs: Some(&kwargs),
                ..Default::default()
            })
            .unwrap();

        assert_eq!(result, "{\n  \"a\": 2,\n  \"b\": 1\n}");
    }

    #[test]
    fn test_load_chat_template_from_file_jinja() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("chat_template.jinja");
        fs::write(&path, "{{ messages }}").unwrap();

        let template = load_chat_template(&path).unwrap();

        assert_eq!(template.as_deref(), Some("{{ messages }}"));
    }

    #[test]
    fn test_resolve_chat_template_from_inline_literal() {
        let template = resolve_chat_template("{{ messages }}").unwrap();

        assert_eq!(template, "{{ messages }}");
    }

    #[test]
    fn test_resolve_chat_template_from_existing_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("chat_template.jinja");
        fs::write(&path, "{{ messages }}").unwrap();

        let template = resolve_chat_template(path.to_str().unwrap()).unwrap();

        assert_eq!(template, "{{ messages }}");
    }

    #[test]
    fn test_resolve_chat_template_rejects_missing_path_like_value() {
        let error = resolve_chat_template("missing_template.jinja").unwrap_err();

        assert!(matches!(error, TemplateError::MissingTemplatePath));
    }

    #[test]
    fn test_chat_template_state_respects_explicit_content_format_override() {
        let template = CompiledChatTemplate::new(
            "{% for item in messages[0].content %}{{ item.text }}{% endfor %}".to_string(),
            ChatTemplateContentFormatOption::String,
        )
        .unwrap();

        assert_eq!(template.content_format(), ChatTemplateContentFormat::String);
    }

    #[test]
    fn test_load_chat_template_from_file_json_string() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("chat_template.json");
        fs::write(&path, "\"{{ messages }}\"").unwrap();

        let template = load_chat_template(&path).unwrap();

        assert_eq!(template.as_deref(), Some("{{ messages }}"));
    }

    #[test]
    fn test_load_chat_template_from_file_json_object() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("chat_template.json");
        fs::write(&path, r#"{"chat_template":"{{ messages }}"}"#).unwrap();

        let template = load_chat_template(&path).unwrap();

        assert_eq!(template.as_deref(), Some("{{ messages }}"));
    }
}
