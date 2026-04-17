#![feature(coroutines)]
#![feature(trait_alias)]

//! Minimal chat facade above [`vllm_text`].
//!
//! This crate keeps the northbound boundary intentionally small:
//! `messages -> rendered prompt -> tokenized prompt -> engine request -> streamed structured
//! assistant events`. The request side remains text-first, while the response side can emit
//! structured reasoning and final-answer blocks. It is closer to vLLM's internal chat-rendering
//! flow than to a full OpenAI-compatible surface.

pub use backend::{
    ChatBackend, ChatTextBackend, DynChatBackend, DynChatTextBackend, SamplingHints,
};
pub use backends::{LoadModelBackendsOptions, LoadedModelBackends, load_model_backends};
pub use error::{Error, Result};
pub use event::{
    AssistantBlockKind, AssistantContentBlock, AssistantMessage, AssistantMessageExt,
    AssistantToolCall, ChatEvent,
};
use futures::{StreamExt, TryStreamExt as _};
pub use parser::ParserSelection;
pub use reasoning::{ReasoningDelta, ReasoningError, ReasoningParser, ReasoningParserFactory};
pub use renderers::hf::ChatTemplateContentFormatOption;
pub use renderers::{ChatRenderer, DynChatRenderer, RenderedPrompt};
pub use request::{
    ChatContent, ChatContentPart, ChatMessage, ChatOptions, ChatRequest, ChatRole, ChatTool,
    ChatToolChoice, SamplingParams,
};
pub use stream::{ChatEventStream, ChatEventStreamTrait, CollectedAssistantMessage};
pub use tool::{ToolParser, ToolParserError, ToolParserFactory};
use tracing::info;
pub use vllm_llm::FinishReason;

mod backend;
pub mod backends;
mod error;
mod event;
mod output;
mod parser;
mod reasoning;
mod renderers;
mod request;
mod stream;
mod tool;

use vllm_engine_core_client::EngineCoreClient;
use vllm_llm::Llm;
use vllm_text::{Prompt, TextLlm, TextRequest};

/// Validate explicit parser override names without starting request processing.
pub fn validate_parser_overrides(
    tool_call_parser: &ParserSelection,
    reasoning_parser: &ParserSelection,
) -> Result<()> {
    let tool_parser_factory = ToolParserFactory::new();
    if let ParserSelection::Explicit(name) = tool_call_parser
        && !tool_parser_factory.contains(name)
    {
        return Err(Error::ParserUnavailableByName {
            kind: "tool",
            name: name.clone(),
            available_names: tool_parser_factory.list(),
        });
    }

    let reasoning_parser_factory = ReasoningParserFactory::new();
    if let ParserSelection::Explicit(name) = reasoning_parser
        && !reasoning_parser_factory.contains(name)
    {
        return Err(Error::ParserUnavailableByName {
            kind: "reasoning",
            name: name.clone(),
            available_names: reasoning_parser_factory.list(),
        });
    }

    Ok(())
}

/// Structured chat facade above [`TextLlm`].
///
/// This layer stays above raw text semantics: it takes care of chat-template rendering, exposes
/// structured assistant events, and adds chat-specific request semantics such as tool calls.
pub struct ChatLlm {
    text: TextLlm,
    backend: DynChatBackend,
    reasoning_parser_factory: ReasoningParserFactory,
    tool_parser_factory: ToolParserFactory,
    /// Tool-call parser selection.
    tool_call_parser: ParserSelection,
    /// Reasoning parser selection.
    reasoning_parser: ParserSelection,
}

impl ChatLlm {
    /// Create a new chat facade from a text-generation facade plus a chat backend.
    pub fn new(text: TextLlm, backend: DynChatBackend) -> Self {
        Self {
            text,
            backend,
            reasoning_parser_factory: ReasoningParserFactory::new(),
            tool_parser_factory: ToolParserFactory::new(),
            tool_call_parser: ParserSelection::Auto,
            reasoning_parser: ParserSelection::Auto,
        }
    }

    /// Convenience constructor for one shared backend object that implements both text and chat
    /// responsibilities.
    pub fn from_shared_backend(llm: Llm, backend: DynChatTextBackend) -> Self {
        let text = TextLlm::new(llm, backend.clone());
        Self::new(text, backend)
    }

    /// Set tool-call parser selection.
    pub fn with_tool_call_parser(mut self, selection: ParserSelection) -> Self {
        self.tool_call_parser = selection;
        self
    }

    /// Set reasoning parser selection.
    pub fn with_reasoning_parser(mut self, selection: ParserSelection) -> Self {
        self.reasoning_parser = selection;
        self
    }

    /// Expose the underlying text facade for raw text-generation routes such as `/v1/completions`.
    pub fn text(&self) -> &TextLlm {
        &self.text
    }

    /// Return the model ID reported by the underlying text backend.
    pub fn model_id(&self) -> &str {
        self.text.model_id()
    }

    /// Expose the underlying engine-core client for low-level utility/admin calls.
    pub fn engine_core_client(&self) -> &EngineCoreClient {
        self.text.engine_core_client()
    }

    /// Render, tokenize, and submit one chat request.
    pub async fn chat(&self, mut request: ChatRequest) -> Result<ChatEventStream> {
        request.validate()?;

        let output_processors = self.prepare_output_processors(&mut request)?;
        let rendered = self.backend.chat_renderer().render(&request)?;

        let text_request = TextRequest {
            request_id: request.request_id.clone(),
            prompt: Prompt::Text(rendered.prompt),
            sampling_params: request.sampling_params,
            decode_options: request.decode_options,
            intermediate: request.intermediate,
            priority: request.priority,
            cache_salt: request.cache_salt,
            add_special_tokens: request.add_special_tokens,
            data_parallel_rank: request.data_parallel_rank,
        };
        let decoded_stream = self.text.generate(text_request).await?.map_err(Error::from);

        let structured_stream =
            output::output_stream(request.intermediate, decoded_stream, output_processors)?;

        Ok(ChatEventStream::new(
            request.request_id,
            structured_stream.boxed(),
        ))
    }

    /// Shut down the underlying LLM client and its background tasks.
    pub async fn shutdown(self) -> Result<()> {
        self.text.shutdown().await?;
        Ok(())
    }
}

impl ChatLlm {
    fn prepare_output_processors(
        &self,
        request: &mut ChatRequest,
    ) -> Result<output::OutputProcessors> {
        let tool_parsing_enabled =
            matches!(request.tool_choice, ChatToolChoice::Auto) && !request.tools.is_empty();
        let tool_parser = if tool_parsing_enabled {
            Some(self.resolve_tool_parser(request)?)
        } else {
            None
        };
        let reasoning_parser = self.resolve_optional_reasoning_parser(request)?;

        Ok(output::OutputProcessors {
            reasoning_parser,
            tool_parser,
        })
    }

    fn resolve_tool_parser(&self, request: &mut ChatRequest) -> Result<Box<dyn ToolParser>> {
        let parser_name = match &self.tool_call_parser {
            ParserSelection::Auto => self
                .tool_parser_factory
                .resolve_name_for_model(self.text.model_id())
                .ok_or_else(|| Error::ParserUnavailableForModel {
                    kind: "tool",
                    model_id: self.text.model_id().to_string(),
                })?,
            ParserSelection::None => return Err(Error::ParserDisabled { kind: "tool" }),
            ParserSelection::Explicit(name) => name.as_str(),
        };

        let parser = self
            .tool_parser_factory
            .create(parser_name, &request.tools)?;

        parser
            .adjust_request(request)
            .map_err(|error| Error::ParserInitialization {
                kind: "tool",
                name: parser_name.to_string(),
                error: error.into(),
            })?;

        TOOL_PARSER_LOG_ONCE.call_once(|| info!(parser_name, "using tool parser"));
        Ok(parser)
    }

    fn resolve_optional_reasoning_parser(
        &self,
        request: &mut ChatRequest,
    ) -> Result<Option<Box<dyn ReasoningParser>>> {
        let parser_name = match &self.reasoning_parser {
            ParserSelection::Auto => self
                .reasoning_parser_factory
                .resolve_name_for_model(self.text.model_id()),
            ParserSelection::None => None,
            ParserSelection::Explicit(name) => Some(name.as_str()),
        };

        let Some(parser_name) = parser_name else {
            REASONING_PARSER_LOG_ONCE.call_once(|| info!("reasoning parsing disabled"));
            return Ok(None);
        };

        let parser = self
            .reasoning_parser_factory
            .create(parser_name, self.text.tokenizer())?;

        parser
            .adjust_request(request)
            .map_err(|error| Error::ParserInitialization {
                kind: "reasoning",
                name: parser_name.to_string(),
                error: error.into(),
            })?;

        REASONING_PARSER_LOG_ONCE.call_once(|| info!(parser_name, "using reasoning parser"));
        Ok(Some(parser))
    }
}

static TOOL_PARSER_LOG_ONCE: std::sync::Once = std::sync::Once::new();
static REASONING_PARSER_LOG_ONCE: std::sync::Once = std::sync::Once::new();

#[cfg(test)]
mod tests {
    use thiserror_ext::AsReport;

    use super::{ParserSelection, validate_parser_overrides};
    use crate::reasoning::names;

    #[test]
    fn validate_parser_overrides_accepts_registered_names() {
        validate_parser_overrides(
            &ParserSelection::Explicit("json".to_string()),
            &ParserSelection::Explicit(names::QWEN3.to_string()),
        )
        .unwrap();
    }

    #[test]
    fn validate_parser_overrides_accepts_auto_and_none() {
        validate_parser_overrides(&ParserSelection::Auto, &ParserSelection::None).unwrap();
    }

    #[test]
    fn validate_parser_overrides_rejects_unknown_tool_parser() {
        let error = validate_parser_overrides(
            &ParserSelection::Explicit("definitely_missing_tool_parser".to_string()),
            &ParserSelection::Auto,
        )
        .unwrap_err();

        expect_test::expect!["tool parser `definitely_missing_tool_parser` is not registered (choose from: cohere, deepseek_v3, deepseek_v31, glm45, glm47, json, kimi_k2, llama3_json, llama4_json, llama4_pythonic, minimax_m2, mistral, pythonic, qwen3_coder, qwen3_xml, step3)"].assert_eq(&error.to_report_string());
    }

    #[test]
    fn validate_parser_overrides_rejects_unknown_reasoning_parser() {
        let error = validate_parser_overrides(
            &ParserSelection::Auto,
            &ParserSelection::Explicit("definitely_missing_reasoning_parser".to_string()),
        )
        .unwrap_err();

        expect_test::expect!["reasoning parser `definitely_missing_reasoning_parser` is not registered (choose from: cohere_cmd, deepseek_r1, deepseek_v3, gemma4, glm45, kimi, kimi_k2, minimax_m2, nemotron_v3, qwen3, step3)"].assert_eq(&error.to_report_string());
    }
}
