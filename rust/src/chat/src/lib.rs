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
pub use backends::{LoadedModelBackends, load_model_backends};
pub use error::{Error, Result};
pub use event::{
    AssistantBlockKind, AssistantContentBlock, AssistantMessage, AssistantMessageExt,
    AssistantToolCall, ChatEvent,
};
use futures::{StreamExt, TryStreamExt as _};
pub use renderers::{ChatRenderer, DynChatRenderer, ReasoningParserInit, RenderedPrompt};
pub use request::{
    ChatContent, ChatContentPart, ChatMessage, ChatOptions, ChatRequest, ChatRole, ChatTool,
    ChatToolChoice, SamplingParams,
};
pub use stream::{ChatEventStream, ChatEventStreamTrait, CollectedAssistantMessage};
pub use vllm_llm::FinishReason;

mod backend;
pub mod backends;
mod error;
mod event;
mod output;
mod renderers;
mod request;
mod stream;

use reasoning_parser::{ParserFactory as ReasoningParserFactory, ReasoningParser};
use tool_parser::{ParserFactory as ToolParserFactory, ToolParser};
use tracing::info;
use vllm_engine_core_client::EngineCoreClient;
use vllm_llm::Llm;
use vllm_text::{Prompt, TextLlm, TextRequest};

fn available_tool_parser_names(tool_parser_factory: &ToolParserFactory) -> Vec<String> {
    let mut available_names = tool_parser_factory.list_parsers();
    available_names.sort_unstable();
    available_names
}

fn available_reasoning_parser_names(
    reasoning_parser_factory: &ReasoningParserFactory,
) -> Vec<String> {
    reasoning_parser_factory.list_parsers()
}

/// Validate explicit parser override names without starting request processing.
pub fn validate_parser_overrides(
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
) -> Result<()> {
    let tool_parser_factory = ToolParserFactory::new();
    if let Some(name) = tool_call_parser
        && !tool_parser_factory.registry().has_parser(name)
    {
        let available_names = available_tool_parser_names(&tool_parser_factory);
        return Err(Error::ToolParserUnavailableByName {
            name: name.to_string(),
            available_names,
        });
    }

    let reasoning_parser_factory = ReasoningParserFactory::new();
    if let Some(name) = reasoning_parser
        && !reasoning_parser_factory.registry().has_parser(name)
    {
        let available_names = available_reasoning_parser_names(&reasoning_parser_factory);
        return Err(Error::ReasoningParserUnavailableByName {
            name: name.to_string(),
            available_names,
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
    /// Explicit tool call parser name override (bypasses model-based auto-detection).
    tool_call_parser: Option<String>,
    /// Explicit reasoning parser name override (bypasses model-based auto-detection).
    reasoning_parser: Option<String>,
}

impl ChatLlm {
    /// Create a new chat facade from a text-generation facade plus a chat backend.
    pub fn new(text: TextLlm, backend: DynChatBackend) -> Self {
        Self {
            text,
            backend,
            reasoning_parser_factory: ReasoningParserFactory::new(),
            tool_parser_factory: ToolParserFactory::new(),
            tool_call_parser: None,
            reasoning_parser: None,
        }
    }

    /// Convenience constructor for one shared backend object that implements both text and chat
    /// responsibilities.
    pub fn from_shared_backend(llm: Llm, backend: DynChatTextBackend) -> Self {
        let text = TextLlm::new(llm, backend.clone());
        Self::new(text, backend)
    }

    /// Set an explicit tool call parser name, bypassing model-based auto-detection.
    pub fn with_tool_call_parser(mut self, name: impl Into<String>) -> Self {
        self.tool_call_parser = Some(name.into());
        self
    }

    /// Set an explicit reasoning parser name, bypassing model-based auto-detection.
    pub fn with_reasoning_parser(mut self, name: impl Into<String>) -> Self {
        self.reasoning_parser = Some(name.into());
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
    pub async fn chat(&self, request: ChatRequest) -> Result<ChatEventStream> {
        request.validate()?;

        let rendered = self.backend.chat_renderer().render(&request)?;
        let output_processors =
            self.prepare_output_processors(&request, rendered.reasoning_parser_init)?;

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
        request: &ChatRequest,
        reasoning_parser_init: ReasoningParserInit,
    ) -> Result<output::OutputProcessors> {
        let tool_parsing_enabled =
            matches!(request.tool_choice, ChatToolChoice::Auto) && !request.tools.is_empty();
        let tool_parser = if tool_parsing_enabled {
            Some(self.resolve_tool_parser()?)
        } else {
            None
        };
        let parser_tools = if tool_parsing_enabled {
            request.tools.iter().map(ChatTool::to_openai_tool).collect()
        } else {
            Vec::new()
        };
        let reasoning_parser = self.resolve_reasoning_parser(reasoning_parser_init)?;

        LOG_ONCE.call_once(|| {
            // TODO: tool-parser doesn't expose its model type.
            if let Some(reasoning_parser) = &reasoning_parser {
                let model_type = reasoning_parser.model_type();
                info!(model_type, "using reasoning parser");
            }
        });

        Ok(output::OutputProcessors {
            reasoning_parser,
            parser_tools,
            tool_parser,
        })
    }

    fn resolve_tool_parser(&self) -> Result<Box<dyn ToolParser>> {
        let registry = self.tool_parser_factory.registry();

        if let Some(name) = self.tool_call_parser.as_deref() {
            // Explicit parser name takes precedence.
            return registry.create_parser(name).ok_or_else(|| {
                Error::ToolParserUnavailableByName {
                    name: name.to_string(),
                    available_names: available_tool_parser_names(&self.tool_parser_factory),
                }
            });
        }

        let model_id = self.text.model_id();
        registry
            .create_for_model(model_id)
            .ok_or_else(|| Error::ToolParserUnavailableForModel {
                model_id: model_id.to_string(),
            })
    }

    fn resolve_reasoning_parser(
        &self,
        reasoning_parser_init: ReasoningParserInit,
    ) -> Result<Option<Box<dyn ReasoningParser>>> {
        let registry = self.reasoning_parser_factory.registry();

        let mut reasoning_parser = if let Some(name) = self.reasoning_parser.as_deref() {
            // Explicit parser name takes precedence.
            registry.create_parser(name).map(Some).ok_or_else(|| {
                Error::ReasoningParserUnavailableByName {
                    name: name.to_string(),
                    available_names: available_reasoning_parser_names(
                        &self.reasoning_parser_factory,
                    ),
                }
            })?
        } else {
            registry.create_for_model(self.text.model_id())
        };

        // Apply initialization hints from the rendering result.
        if let Some(parser) = reasoning_parser.as_mut() {
            if reasoning_parser_init.mark_reasoning_started {
                parser.mark_reasoning_started();
            }
            if reasoning_parser_init.mark_think_start_stripped {
                parser.mark_think_start_stripped();
            }
        }

        Ok(reasoning_parser)
    }
}

static LOG_ONCE: std::sync::Once = std::sync::Once::new();

#[cfg(test)]
mod tests {
    use thiserror_ext::AsReport;

    use super::validate_parser_overrides;

    #[test]
    fn validate_parser_overrides_accepts_registered_names() {
        validate_parser_overrides(Some("json"), Some("qwen3")).unwrap();
    }

    #[test]
    fn validate_parser_overrides_rejects_unknown_tool_parser() {
        let error =
            validate_parser_overrides(Some("definitely_missing_tool_parser"), None).unwrap_err();

        expect_test::expect!["tool call parser `definitely_missing_tool_parser` is not registered (choose from: cohere, deepseek, deepseek31, glm45_moe, glm47_moe, json, kimik2, llama, minimax_m2, mistral, passthrough, pythonic, qwen, qwen_coder, step3)"].assert_eq(&error.to_report_string());
    }

    #[test]
    fn validate_parser_overrides_rejects_unknown_reasoning_parser() {
        let error = validate_parser_overrides(None, Some("definitely_missing_reasoning_parser"))
            .unwrap_err();

        expect_test::expect!["reasoning parser `definitely_missing_reasoning_parser` is not registered (choose from: base, cohere_cmd, deepseek_r1, deepseek_v31, glm45, kimi, kimi_k25, kimi_thinking, minimax, nano_v3, qwen3, qwen3_thinking, step3)"].assert_eq(&error.to_report_string());
    }
}
