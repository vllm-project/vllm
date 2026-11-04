// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Chat template support for tokenizers using Jinja2 templates.
//!
//! This module is inlined from SMG's tokenizer crate with local adaptations:
//! - thinking-related detection/state is removed
//! - special tokens are wired to `vllm_text::backends::hf::HfSpecialTokens`

use std::collections::HashMap;
use std::error::Error as _;
use std::ffi::OsStr;
use std::fs;
use std::iter;
use std::path::Path;

use minijinja::{Environment, Error as MinijinjaError, ErrorKind, Value};
use serde::{Deserialize, Serialize};
use serde_json::{self};
use thiserror::Error as ThisError;
use vllm_text::backend::hf::HfSpecialTokens;

use super::error::TemplateError;
use super::format::{
    ChatTemplateContentFormat, ChatTemplateContentFormatOption, detect_chat_template_content_format,
};
use super::generation::{render_generation, rewrite_generation_blocks};
use super::tojson::hf_tojson_filter;
use crate::renderer::hf::{TemplateMessage, TemplateTool};

type Result<T> = std::result::Result<T, TemplateError>;

/// Default fuel budget for rendering request-supplied templates. Prevents
/// unbounded loops and string operations from exhausting CPU. Each template
/// instruction consumes one unit of fuel; when depleted, rendering aborts with
/// an error.
const DEFAULT_TEMPLATE_FUEL: u64 = 500_000;

/// Fuel budget for request-supplied templates from the value of
/// `VLLM_CHAT_TEMPLATE_FUEL`, if set. `0` disables the limit.
pub(super) fn request_template_fuel(value: Option<&OsStr>) -> Result<Option<u64>> {
    let fuel = match value {
        None => DEFAULT_TEMPLATE_FUEL,
        Some(value) => value
            .to_str()
            .and_then(|value| value.parse().ok())
            .ok_or_else(|| TemplateError::InvalidFuel(value.to_owned()))?,
    };
    Ok((fuel > 0).then_some(fuel))
}

fn supports_developer_role(template: &str) -> bool {
    template.contains("\"developer\"") || template.contains("'developer'")
}

/// Build a pre-configured environment with the given template string.
fn build_environment(template: String) -> Result<Environment<'static>> {
    let mut env = Environment::new();

    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);

    env.add_template_owned("chat".to_owned(), template)?;

    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_filter("tojson", hf_tojson_filter);
    env.add_function("__hf_generation", render_generation);
    env.add_function("strftime_now", |format: String| {
        chrono::Local::now().format(&format).to_string()
    });
    env.add_function("raise_exception", raise_exception);

    Ok(env)
}

/// Error source attached by [`raise_exception`], so that an exception thrown
/// by the template can be told apart from a template that fails to render.
#[derive(Debug, ThisError)]
#[error("{message}")]
struct ThrownTemplateException {
    message: String,
}

/// Throw an exception with the template's own message, aborting rendering.
fn raise_exception(message: Value) -> std::result::Result<Value, MinijinjaError> {
    let thrown = ThrownTemplateException {
        message: message.to_string(),
    };
    let error = MinijinjaError::new(
        ErrorKind::InvalidOperation,
        "chat template threw an exception",
    );
    Err(error.with_source(thrown))
}

impl TemplateError {
    /// Return the template's own message if rendering was aborted by
    /// [`raise_exception`].
    pub(super) fn thrown_message(&self) -> Option<&str> {
        let Self::Jinja(error) = self else {
            return None;
        };
        iter::successors(error.source(), |&error| error.source())
            .find_map(|error| error.downcast_ref::<ThrownTemplateException>())
            .map(|thrown| thrown.message.as_str())
    }
}

#[serde_with::skip_serializing_none]
#[derive(Default, Serialize)]
pub(super) struct TemplateContext<'a> {
    pub(super) messages: &'a [TemplateMessage<'a>],
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

/// One compiled chat template with its Jinja environment and detected content
/// format.
pub(super) struct CompiledChatTemplate {
    /// Cached, fully-configured environment for one compiled template.
    env: Environment<'static>,
    content_format: ChatTemplateContentFormat,
    supports_developer_role: bool,
}

impl CompiledChatTemplate {
    /// Compile the given chat template string into a [`CompiledChatTemplate`].
    pub fn new(template: String, content_format: ChatTemplateContentFormatOption) -> Result<Self> {
        let template = rewrite_generation_blocks(template)?;
        let supports_developer_role = supports_developer_role(&template);
        let content_format = match content_format {
            ChatTemplateContentFormatOption::Auto => detect_chat_template_content_format(&template),
            ChatTemplateContentFormatOption::String => ChatTemplateContentFormat::String,
            ChatTemplateContentFormatOption::OpenAi => ChatTemplateContentFormat::OpenAi,
        };
        let env = build_environment(template)?;
        Ok(Self {
            env,
            content_format,
            supports_developer_role,
        })
    }

    /// Limit each render to `fuel` template instructions; `None` removes the
    /// limit.
    pub fn with_fuel(mut self, fuel: Option<u64>) -> Self {
        self.env.set_fuel(fuel);
        self
    }

    /// Apply the compiled template to the given context and return the rendered
    /// prompt.
    pub fn apply(&self, ctx: TemplateContext<'_>) -> Result<String> {
        let tmpl = self.env.get_template("chat")?;
        tmpl.render(ctx).map_err(|error| match error.kind() {
            ErrorKind::OutOfFuel => TemplateError::OutOfFuel(error),
            _ => TemplateError::Jinja(error),
        })
    }

    pub fn content_format(&self) -> ChatTemplateContentFormat {
        self.content_format
    }

    pub fn supports_developer_role(&self) -> bool {
        self.supports_developer_role
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use expect_test::expect;
    use tempfile::TempDir;
    use thiserror_ext::AsReport as _;
    use vllm_text::backend::hf::{HfSpecialTokens, NamedSpecialToken};

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
    fn test_midchain_dotted_integer_lookup() {
        let template = CompiledChatTemplate::new(
            "{{ values.0.name }}".to_string(),
            ChatTemplateContentFormatOption::Auto,
        )
        .unwrap();
        let mut kwargs = HashMap::new();
        kwargs.insert("values".to_string(), serde_json::json!([{"name": "first"}]));

        let result = template
            .apply(TemplateContext {
                template_kwargs: Some(&kwargs),
                ..Default::default()
            })
            .unwrap();

        assert_eq!(result, "first");
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
    fn test_strftime_now_matches_transformers_contract() {
        // `strftime_now` renders local wall-clock time with strftime-style
        // formats, like the transformers Jinja environment.
        let template = CompiledChatTemplate::new(
            "{{ strftime_now('literal') }}|{{ strftime_now('%Y-%m-%d') }}".to_string(),
            ChatTemplateContentFormatOption::Auto,
        )
        .unwrap();

        let result = template.apply(TemplateContext::default()).unwrap();

        let (literal, date) = result.split_once('|').unwrap();
        assert_eq!(literal, "literal");
        let is_digit_at =
            |range: std::ops::Range<usize>| date[range].chars().all(|c| c.is_ascii_digit());
        assert!(
            date.len() == 10 && is_digit_at(0..4) && is_digit_at(5..7) && is_digit_at(8..10),
            "expected %Y-%m-%d date, got: {date}"
        );
    }

    #[test]
    fn test_raise_exception_returns_template_message() {
        let template = CompiledChatTemplate::new(
            "{{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort ~ '.') }}"
                .to_string(),
            ChatTemplateContentFormatOption::Auto,
        )
        .unwrap();
        let mut kwargs = HashMap::new();
        kwargs.insert("reasoning_effort".to_string(), serde_json::json!("high"));

        let error = template
            .apply(TemplateContext {
                template_kwargs: Some(&kwargs),
                ..Default::default()
            })
            .unwrap_err();

        assert_eq!(
            error.thrown_message(),
            Some("Unexpected reasoning effort high."),
            "expected an exception thrown by the template, got: {error:?}"
        );
    }

    #[test]
    fn test_raise_exception_in_untaken_branch_renders() {
        let template = CompiledChatTemplate::new(
            "{% if messages %}{{ raise_exception('unreachable') }}{% endif %}ok".to_string(),
            ChatTemplateContentFormatOption::Auto,
        )
        .unwrap();

        assert_eq!(template.apply(TemplateContext::default()).unwrap(), "ok");
    }

    #[test]
    fn test_unknown_function_is_not_reported_as_thrown() {
        let template = CompiledChatTemplate::new(
            "{{ undefined_helper('not registered') }}".to_string(),
            ChatTemplateContentFormatOption::Auto,
        )
        .unwrap();

        let error = template.apply(TemplateContext::default()).unwrap_err();

        assert_eq!(
            error.thrown_message(),
            None,
            "expected a jinja render error, got: {error:?}"
        );
    }

    /// Each turn costs a fixed number of instructions, like a chat template's message loop.
    fn render_turns(turns: usize, fuel: Option<u64>) -> Result<String> {
        let template =
            "{% for turn in turns %}{% for _ in range(10) %}{{ turn }}{% endfor %}{% endfor %}";
        let template = CompiledChatTemplate::new(
            template.to_string(),
            ChatTemplateContentFormatOption::String,
        )?
        .with_fuel(fuel);
        let mut kwargs = HashMap::new();
        kwargs.insert("turns".to_string(), serde_json::json!(vec![0; turns]));
        template.apply(TemplateContext {
            template_kwargs: Some(&kwargs),
            ..Default::default()
        })
    }

    #[test]
    fn test_long_conversation_exceeds_default_fuel() {
        let error = render_turns(20_000, Some(DEFAULT_TEMPLATE_FUEL)).unwrap_err();
        expect!["chat template exceeded its fuel budget; adjust VLLM_CHAT_TEMPLATE_FUEL to override: engine ran out of fuel (in chat:1)"].assert_eq(&error.to_report_string());
    }

    #[test]
    fn test_raised_or_disabled_fuel_renders_long_conversation() {
        assert!(render_turns(20_000, Some(5_000_000)).is_ok());
        assert!(render_turns(20_000, None).is_ok());
    }

    #[test]
    fn test_request_template_fuel_rejects_invalid_value() {
        assert_eq!(
            request_template_fuel(None).unwrap(),
            Some(DEFAULT_TEMPLATE_FUEL)
        );
        assert_eq!(
            request_template_fuel(Some("2000000".as_ref())).unwrap(),
            Some(2_000_000)
        );
        assert_eq!(request_template_fuel(Some("0".as_ref())).unwrap(), None);
        let error = request_template_fuel(Some("-1".as_ref())).unwrap_err();
        expect![[r#"VLLM_CHAT_TEMPLATE_FUEL must be a non-negative integer, got "-1""#]]
            .assert_eq(&error.to_report_string());
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
