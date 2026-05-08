//! Streaming tool parsers for chat completions.
//!
//! This module intentionally starts as a local ownership boundary for tool
//! parser registration and selection, without yet taking over the concrete
//! parsing implementation from the external `tool-parser` crate. The goal is
//! to establish the northbound trait and factory shape inside `vllm-chat`
//! first, so later steps can attach adaptor-based implementations and then
//! gradually replace them with native parsers as needed.

mod deepseek_dsml;
mod deepseek_json;
mod external;
mod gemma4;
mod glm_xml;
mod kimi_k2;
mod minimax_m2;
mod parameters;
mod qwen_coder;
#[cfg(any(test, feature = "test-util"))]
pub mod test_utils;
mod utils;

use std::collections::{BTreeMap, btree_map};
use std::sync::LazyLock;

use thiserror::Error;
use thiserror_ext::Macro;

use crate::parser::ParserFactory;
use crate::request::{ChatRequest, ChatTool};

/// Result alias for tool parser operations.
pub type Result<T> = std::result::Result<T, ToolParserError>;

pub use deepseek_dsml::{DeepSeekV4ToolParser, DeepSeekV32ToolParser};
pub use deepseek_json::{DeepSeekV3ToolParser, DeepSeekV31ToolParser};
pub use external::*;
pub use gemma4::Gemma4ToolParser;
pub use glm_xml::{Glm45MoeToolParser, Glm47MoeToolParser};
pub use kimi_k2::KimiK2ToolParser;
pub use minimax_m2::MinimaxM2ToolParser;
pub use qwen_coder::Qwen3CoderToolParser;

/// Canonical public names for registered tool parsers.
pub mod names {
    pub const COHERE: &str = "cohere";
    pub const DEEPSEEK_V3: &str = "deepseek_v3";
    pub const DEEPSEEK_V31: &str = "deepseek_v31";
    pub const DEEPSEEK_V32: &str = "deepseek_v32";
    pub const DEEPSEEK_V4: &str = "deepseek_v4";
    pub const GLM45: &str = "glm45";
    pub const GLM47: &str = "glm47";
    pub const GEMMA4: &str = "gemma4";
    pub const JSON: &str = "json";
    pub const KIMI_K2: &str = "kimi_k2";
    pub const LLAMA3_JSON: &str = "llama3_json";
    pub const LLAMA4_JSON: &str = "llama4_json";
    pub const LLAMA4_PYTHONIC: &str = "llama4_pythonic";
    pub const MINIMAX_M2: &str = "minimax_m2";
    pub const MISTRAL: &str = "mistral";
    pub const PYTHONIC: &str = "pythonic";
    pub const QWEN3_CODER: &str = "qwen3_coder";
    pub const QWEN3_XML: &str = "qwen3_xml";
    pub const STEP3: &str = "step3";
}

/// One tool-call update emitted while parsing assistant text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallDelta {
    /// Stable parser-local tool index for this call within one assistant turn.
    pub tool_index: usize,
    /// Function name, present on the first update for one tool call.
    pub name: Option<String>,
    /// Arguments text contributed by this update.
    pub arguments: String,
}

/// Result of advancing tool parsing with one assistant-text input.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolParseResult {
    /// Plain assistant text that is not part of any tool call.
    pub normal_text: String,
    /// Tool-call updates extracted from this input.
    pub calls: Vec<ToolCallDelta>,
}

impl ToolParseResult {
    /// Append another parser result onto this one.
    ///
    /// Note that this does not attempt to merge multiple deltas for the same
    /// tool call into one complete item. Call `coalesce_calls()` after if
    /// that behavior is desired.
    pub(crate) fn append(&mut self, mut other: Self) {
        self.normal_text.push_str(&other.normal_text);
        self.calls.append(&mut other.calls);
    }

    /// Merge multiple deltas for the same tool call into one complete item.
    ///
    /// This is primarily used by the default `parse_complete()` implementation,
    /// which delegates through the incremental parser lifecycle and then
    /// needs to collapse streaming-style argument fragments into one final
    /// tool call.
    pub(crate) fn coalesce_calls(mut self) -> Self {
        let mut merged = BTreeMap::<usize, ToolCallDelta>::new();
        let mut order = Vec::new();

        for call in self.calls {
            match merged.entry(call.tool_index) {
                btree_map::Entry::Vacant(entry) => {
                    order.push(call.tool_index);
                    entry.insert(call);
                }
                btree_map::Entry::Occupied(mut entry) => {
                    let existing = entry.get_mut();
                    if existing.name.is_none() {
                        existing.name = call.name;
                    }
                    existing.arguments.push_str(&call.arguments);
                }
            }
        }

        self.calls =
            order.into_iter().filter_map(|tool_index| merged.remove(&tool_index)).collect();
        self
    }
}

/// Incremental parser that extracts tool calls from assistant output.
pub trait ToolParser: Send {
    /// Construct a boxed parser instance for one request stream.
    fn create(tools: &[ChatTool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static;

    /// Adjust request-level settings before rendering and decoding start.
    ///
    /// Parsers may use this hook to request decode behavior that matches their
    /// output protocol, such as retaining special tokens in streamed text.
    fn adjust_request(&self, _request: &mut ChatRequest) -> Result<()> {
        Ok(())
    }

    /// Feed one decoded text delta into the parser.
    fn push(&mut self, chunk: &str) -> Result<ToolParseResult>;

    /// Flush any buffered partial state at end of stream.
    fn finish(&mut self) -> Result<ToolParseResult> {
        Ok(ToolParseResult::default())
    }

    /// Parse complete tool calls from final output.
    ///
    /// The default implementation reuses the incremental parser lifecycle by
    /// feeding the full output through `push()` and then calling `finish()`.
    /// This keeps one source of truth for robust parsers whose incremental
    /// state machine is equivalent across arbitrary chunking.
    fn parse_complete(&mut self, output: &str) -> Result<ToolParseResult> {
        let mut result = self.push(output)?;
        result.append(self.finish()?);
        Ok(result.coalesce_calls())
    }
}

/// Errors produced while creating or running tool parsers.
#[derive(Debug, Error, Macro)]
pub enum ToolParserError {
    #[error("tool parser parsing failed: {message}")]
    ParsingFailed { message: String },
    #[error(transparent)]
    External(#[from] tool_parser::errors::ParserError),
}

/// Constructor signature for one registered tool parser implementation.
type ToolParserCreator = fn(&[ChatTool]) -> Result<Box<dyn ToolParser>>;

/// Registry and model matcher for tool parsers.
pub type ToolParserFactory = ParserFactory<ToolParserCreator>;

impl ToolParserFactory {
    /// Get the global tool parser factory with built-in registrations and model
    /// mappings.
    pub fn global() -> &'static Self {
        static INSTANCE: LazyLock<ToolParserFactory> = LazyLock::new(ToolParserFactory::new);
        &INSTANCE
    }

    /// Create the default registry with built-in parser names and model
    /// mappings.
    pub fn new() -> Self {
        let mut factory = Self::default();

        factory
            .register_parser::<CohereToolParser>(names::COHERE)
            .register_parser::<DeepSeekV3ToolParser>(names::DEEPSEEK_V3)
            .register_parser::<DeepSeekV31ToolParser>(names::DEEPSEEK_V31)
            .register_parser::<DeepSeekV32ToolParser>(names::DEEPSEEK_V32)
            .register_parser::<DeepSeekV4ToolParser>(names::DEEPSEEK_V4)
            .register_parser::<Glm45MoeToolParser>(names::GLM45)
            .register_parser::<Glm47MoeToolParser>(names::GLM47)
            .register_parser::<Gemma4ToolParser>(names::GEMMA4)
            .register_parser::<JsonToolParser>(names::JSON)
            .register_parser::<KimiK2ToolParser>(names::KIMI_K2)
            .register_parser::<Llama3JsonToolParser>(names::LLAMA3_JSON)
            .register_parser::<Llama3JsonToolParser>(names::LLAMA4_JSON)
            .register_parser::<PythonicToolParser>(names::LLAMA4_PYTHONIC)
            .register_parser::<MinimaxM2ToolParser>(names::MINIMAX_M2)
            .register_parser::<MistralToolParser>(names::MISTRAL)
            .register_parser::<PythonicToolParser>(names::PYTHONIC)
            .register_parser::<Qwen3XmlToolParser>(names::QWEN3_XML)
            .register_parser::<Qwen3CoderToolParser>(names::QWEN3_CODER)
            .register_parser::<Step3ToolParser>(names::STEP3);

        factory
            .register_pattern("mistral-", names::MISTRAL)
            .register_pattern("mixtral-", names::MISTRAL)
            .register_pattern("qwen3-coder", names::QWEN3_CODER)
            .register_pattern("qwen2.5-coder", names::QWEN3_CODER)
            .register_pattern("qwen3.5", names::QWEN3_CODER)
            .register_pattern("qwen", names::QWEN3_XML)
            .register_pattern("llama-4", names::LLAMA4_PYTHONIC)
            .register_pattern("llama-3.2", names::LLAMA3_JSON)
            .register_pattern("llama-3.1", names::LLAMA3_JSON)
            .register_pattern("llama-", names::JSON)
            .register_pattern("deepseek-r1", names::DEEPSEEK_V3)
            .register_pattern("deepseek-v4", names::DEEPSEEK_V4)
            .register_pattern("deepseek_v4", names::DEEPSEEK_V4)
            .register_pattern("deepseek-v3.2", names::DEEPSEEK_V32)
            .register_pattern("deepseek-v3.1", names::DEEPSEEK_V31)
            .register_pattern("deepseek-v3", names::DEEPSEEK_V3)
            .register_pattern("deepseek-", names::PYTHONIC)
            .register_pattern("glm-5", names::GLM47)
            .register_pattern("glm-4.7", names::GLM47)
            .register_pattern("glm-4.6", names::GLM45)
            .register_pattern("glm-4.5", names::GLM45)
            .register_pattern("glm-", names::JSON)
            .register_pattern("gemma4", names::GEMMA4)
            .register_pattern("gemma-4", names::GEMMA4)
            .register_pattern("step-3", names::STEP3)
            .register_pattern("step3", names::STEP3)
            .register_pattern("kimi-k2", names::KIMI_K2)
            .register_pattern("minimax", names::MINIMAX_M2)
            .register_pattern("command-", names::COHERE)
            .register_pattern("c4ai-command", names::COHERE)
            .register_pattern("cohere", names::COHERE)
            .register_pattern("gemma-", names::JSON);

        factory
    }

    /// Register one parser type that exposes a static `create()` constructor.
    pub fn register_parser<T>(&mut self, name: &str) -> &mut Self
    where
        T: ToolParser + 'static,
    {
        self.register_creator(name, T::create)
    }

    /// Construct a parser from an exact name.
    pub fn create(&self, name: &str, tools: &[ChatTool]) -> crate::Result<Box<dyn ToolParser>> {
        let creator = self.creator(name).ok_or_else(|| crate::Error::ParserUnavailableByName {
            kind: "tool",
            name: name.to_string(),
            available_names: self.list(),
        })?;

        creator(tools).map_err(|error| crate::Error::ParserInitialization {
            kind: "tool",
            name: name.to_string(),
            error: error.into(),
        })
    }

    /// Resolve a parser from model ID and then construct it.
    pub fn create_for_model(
        &self,
        model_id: &str,
        tools: &[ChatTool],
    ) -> crate::Result<Box<dyn ToolParser>> {
        let name = self.resolve_name_for_model(model_id).ok_or_else(|| {
            crate::Error::ParserUnavailableForModel {
                kind: "tool",
                model_id: model_id.to_string(),
            }
        })?;
        self.create(name, tools)
    }
}

#[cfg(test)]
mod tests;
