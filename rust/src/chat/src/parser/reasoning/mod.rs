//! Streaming reasoning parsers for chat completions.
//!
//! The key design choice here is that parser initialization prefers the
//! *actual rendered prompt state* over model-family conventions. When a stream
//! starts, each parser receives the prompt token IDs and inspects the last
//! reasoning boundary that is already present in the prompt. In practice this
//! is a more faithful signal than hardcoding assumptions such as "this model
//! always starts in reasoning" or "this model always emits `<think>` itself".
//!
//! That prompt-first initialization lets multiple model families share the
//! same incremental parser implementation even when older Python parsers split
//! them apart. If two families use the same textual delimiters and differ
//! mostly in how their chat templates prefill `<think>` / `</think>`, they can
//! usually reuse one parser here because the prompt token IDs already tell us
//! which state the stream is entering with.

mod cohere_cmd;
mod deepseek_r1;
mod delimited;
mod gemma4;
mod kimi;
mod qwen3;

use std::sync::LazyLock;

use thiserror::Error;
use vllm_text::tokenizer::DynTokenizer;

pub use self::cohere_cmd::CohereCmdReasoningParser;
pub use self::deepseek_r1::DeepSeekR1ReasoningParser;
pub(crate) use self::delimited::DelimitedReasoningParser;
pub use self::gemma4::Gemma4ReasoningParser;
pub use self::kimi::KimiReasoningParser;
pub use self::qwen3::Qwen3ReasoningParser;
use crate::parser::ParserFactory;
use crate::request::ChatRequest;

/// Canonical public names for registered reasoning parsers.
pub mod names {
    pub const COHERE_CMD: &str = "cohere_cmd";
    pub const DEEPSEEK_R1: &str = "deepseek_r1";
    pub const DEEPSEEK_V3: &str = "deepseek_v3";
    pub const DEEPSEEK_V4: &str = "deepseek_v4";
    pub const GEMMA4: &str = "gemma4";
    pub const GLM45: &str = "glm45";
    pub const KIMI: &str = "kimi";
    pub const KIMI_K2: &str = "kimi_k2";
    pub const MINIMAX_M2: &str = "minimax_m2";
    pub const NEMOTRON_V3: &str = "nemotron_v3";
    pub const QWEN3: &str = "qwen3";
    pub const STEP3: &str = "step3";
}

/// DeepSeek V3 currently shares the standard `<think>...</think>` parser.
pub type DeepSeekV3ReasoningParser = Qwen3ReasoningParser;
/// DeepSeek V4 currently shares the standard `<think>...</think>` parser.
pub type DeepSeekV4ReasoningParser = Qwen3ReasoningParser;
/// GLM45 currently shares the standard `<think>...</think>` parser.
pub type Glm45ReasoningParser = Qwen3ReasoningParser;
/// Kimi K2 currently shares the standard `<think>...</think>` parser.
// TODO: kimi k2 may implicitly end reasoning by starting a tool call section
// using <|tool_calls_section_begin|>, we should support that.
pub type KimiK2ReasoningParser = Qwen3ReasoningParser;
/// MiniMax M2 currently shares the standard `<think>...</think>` parser.
pub type MiniMaxM2ReasoningParser = Qwen3ReasoningParser;
/// Nemotron V3 currently shares the standard `<think>...</think>` parser.
pub type NemotronV3ReasoningParser = Qwen3ReasoningParser;
/// Step3 currently shares the standard `<think>...</think>` parser.
pub type Step3ReasoningParser = Qwen3ReasoningParser;

/// Result alias for reasoning parser operations.
pub type Result<T> = std::result::Result<T, ReasoningError>;

/// One parsed streaming delta split into reasoning and visible content.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ReasoningDelta {
    pub reasoning: Option<String>,
    pub content: Option<String>,
}

impl ReasoningDelta {
    /// Return true when this delta carries neither reasoning nor content text.
    pub fn is_empty(&self) -> bool {
        self.reasoning.is_none() && self.content.is_none()
    }

    /// Append text to the reasoning portion, creating it on first use.
    pub(crate) fn push_reasoning(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        match &mut self.reasoning {
            Some(existing) => existing.push_str(text),
            None => self.reasoning = Some(text.to_string()),
        }
    }

    /// Append text to the visible content portion, creating it on first use.
    pub(crate) fn push_content(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        match &mut self.content {
            Some(existing) => existing.push_str(text),
            None => self.content = Some(text.to_string()),
        }
    }
}

/// Incremental parser that splits decoded text deltas into reasoning and
/// content.
pub trait ReasoningParser: Send {
    /// Construct a boxed parser instance for one request stream.
    fn create(tokenizer: DynTokenizer) -> Result<Box<dyn ReasoningParser>>
    where
        Self: Sized + 'static;

    /// Initialize parser state from prompt token IDs before output deltas
    /// arrive.
    fn initialize(&mut self, _prompt_token_ids: &[u32]) -> Result<()> {
        Ok(())
    }

    /// Adjust request-level settings before rendering and decoding start.
    ///
    /// Parsers may use this hook to request decode behavior that matches their
    /// output protocol, such as retaining special tokens in streamed text.
    fn adjust_request(&self, _request: &mut ChatRequest) -> Result<()> {
        Ok(())
    }

    /// Feed one decoded text delta into the parser.
    fn push(&mut self, delta: &str) -> Result<ReasoningDelta>;

    /// Flush any buffered partial delimiter state at end of stream.
    fn finish(&mut self) -> Result<ReasoningDelta> {
        Ok(ReasoningDelta::default())
    }
}

/// Errors produced while creating or running reasoning parsers.
#[derive(Debug, Error)]
pub enum ReasoningError {
    #[error("tokenizer is missing reasoning delimiter token `{token}`")]
    MissingToken { token: String },
}

/// Constructor signature for one registered reasoning parser implementation.
type ReasoningParserCreator = fn(DynTokenizer) -> Result<Box<dyn ReasoningParser>>;

/// Registry and model matcher for reasoning parsers.
pub type ReasoningParserFactory = ParserFactory<ReasoningParserCreator>;

impl ReasoningParserFactory {
    /// Get the global reasoning parser factory with built-in registrations and
    /// model mappings.
    pub fn global() -> &'static Self {
        static INSTANCE: LazyLock<ReasoningParserFactory> =
            LazyLock::new(ReasoningParserFactory::new);
        &INSTANCE
    }

    /// Create the default registry with built-in parser names and model
    /// mappings.
    pub fn new() -> Self {
        let mut factory = Self::default();

        factory
            .register_parser::<CohereCmdReasoningParser>(names::COHERE_CMD)
            .register_parser::<DeepSeekR1ReasoningParser>(names::DEEPSEEK_R1)
            .register_parser::<DeepSeekV3ReasoningParser>(names::DEEPSEEK_V3)
            .register_parser::<DeepSeekV4ReasoningParser>(names::DEEPSEEK_V4)
            .register_parser::<Gemma4ReasoningParser>(names::GEMMA4)
            .register_parser::<Glm45ReasoningParser>(names::GLM45)
            .register_parser::<KimiReasoningParser>(names::KIMI)
            .register_parser::<KimiK2ReasoningParser>(names::KIMI_K2)
            .register_parser::<MiniMaxM2ReasoningParser>(names::MINIMAX_M2)
            .register_parser::<NemotronV3ReasoningParser>(names::NEMOTRON_V3)
            .register_parser::<Qwen3ReasoningParser>(names::QWEN3)
            .register_parser::<Step3ReasoningParser>(names::STEP3);

        factory
            .register_pattern("deepseek-r1", names::DEEPSEEK_R1)
            .register_pattern("deepseek-v4", names::DEEPSEEK_V4)
            .register_pattern("deepseek_v4", names::DEEPSEEK_V4)
            .register_pattern("deepseek-v3", names::DEEPSEEK_V3)
            .register_pattern("gemma-4", names::GEMMA4)
            .register_pattern("gemma4", names::GEMMA4)
            .register_pattern("qwen", names::QWEN3)
            .register_pattern("glm-5", names::GLM45)
            .register_pattern("glm-4.7", names::GLM45)
            .register_pattern("glm-4.6", names::GLM45)
            .register_pattern("glm-4.5", names::GLM45)
            .register_pattern("kimi-k2", names::KIMI_K2)
            .register_pattern("kimi", names::KIMI)
            .register_pattern("step3", names::STEP3)
            .register_pattern("minimax", names::MINIMAX_M2)
            .register_pattern("mm-m2", names::MINIMAX_M2)
            .register_pattern("cohere", names::COHERE_CMD)
            .register_pattern("command", names::COHERE_CMD)
            .register_pattern("nano", names::NEMOTRON_V3)
            .register_pattern("nemotron", names::NEMOTRON_V3);

        factory
    }

    /// Register one parser type that exposes a static `create()` constructor.
    pub fn register_parser<T>(&mut self, name: &str) -> &mut Self
    where
        T: ReasoningParser + 'static,
    {
        self.register_creator(name, T::create)
    }

    /// Construct a parser from an exact name.
    pub fn create(
        &self,
        name: &str,
        tokenizer: DynTokenizer,
    ) -> crate::Result<Box<dyn ReasoningParser>> {
        let creator = self.creator(name).ok_or_else(|| crate::Error::ParserUnavailableByName {
            kind: "reasoning",
            name: name.to_string(),
            available_names: self.list(),
        })?;

        creator(tokenizer).map_err(|error| crate::Error::ParserInitialization {
            kind: "reasoning",
            name: name.to_string(),
            error: error.into(),
        })
    }
}

#[cfg(test)]
mod tests;
