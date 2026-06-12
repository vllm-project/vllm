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

use thiserror::Error;
use vllm_tokenizer::DynTokenizer;

pub use self::cohere_cmd::CohereCmdReasoningParser;
pub use self::deepseek_r1::DeepSeekR1ReasoningParser;
pub(crate) use self::delimited::DelimitedReasoningParser;
pub use self::gemma4::Gemma4ReasoningParser;
pub use self::kimi::KimiReasoningParser;
pub use self::qwen3::Qwen3ReasoningParser;

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

    /// Return whether decoded output must preserve tokenizer special tokens.
    ///
    /// Some model families emit reasoning sentinels as special tokens. Those
    /// parsers need `skip_special_tokens = false` while parsing is enabled.
    fn preserve_special_tokens(&self) -> bool {
        false
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

#[cfg(test)]
mod tests;
