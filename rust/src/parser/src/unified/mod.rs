// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Unified parser interface for reasoning and tool-call deltas.

mod combined;
mod gemma4;
mod inkling;

pub use combined::CombinedParser;
pub use gemma4::Gemma4UnifiedParser;
pub use inkling::InklingUnifiedParser;
use thiserror::Error;
use thiserror_ext::Macro;
use vllm_tokenizer::DynTokenizer;

use crate::reasoning::ReasoningError;
use crate::tool::{
    StructuralTagModel, Tool, ToolCallDelta, ToolParserError, ToolParserEvent, ToolParserOutput,
};

/// Result alias for unified parser operations.
pub type Result<T> = std::result::Result<T, UnifiedParserError>;

/// One parsed event emitted by a unified parser.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnifiedParserEvent {
    /// Normal assistant-visible text.
    Text(String),
    /// Reasoning text hidden from the normal content stream.
    Reasoning(String),
    /// A tool-call update extracted from visible assistant text.
    ToolCall(ToolCallDelta),
}

/// Result of advancing unified parsing with one assistant-text input.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UnifiedParserOutput {
    /// Ordered parser events committed by this input.
    pub events: Vec<UnifiedParserEvent>,
}

impl UnifiedParserOutput {
    /// Append one visible text event if `delta` is non-empty.
    pub fn push_text(&mut self, delta: impl AsRef<str> + Into<String>) {
        if delta.as_ref().is_empty() {
            return;
        }
        if let Some(UnifiedParserEvent::Text(last_text)) = self.events.last_mut() {
            last_text.push_str(delta.as_ref());
            return;
        }
        self.events.push(UnifiedParserEvent::Text(delta.into()));
    }

    /// Append one reasoning text event if `delta` is non-empty.
    pub fn push_reasoning(&mut self, delta: impl AsRef<str> + Into<String>) {
        if delta.as_ref().is_empty() {
            return;
        }
        if let Some(UnifiedParserEvent::Reasoning(last_text)) = self.events.last_mut() {
            last_text.push_str(delta.as_ref());
            return;
        }
        self.events.push(UnifiedParserEvent::Reasoning(delta.into()));
    }

    /// Append one tool-call event.
    pub fn push_call(&mut self, call: ToolCallDelta) {
        self.events.push(UnifiedParserEvent::ToolCall(call));
    }

    /// Append parsed tool parser output as unified events.
    pub fn append_tool_output(&mut self, output: ToolParserOutput) {
        for event in output.events {
            match event {
                ToolParserEvent::Text(text) => self.push_text(text),
                ToolParserEvent::ToolCall(call) => self.push_call(call),
            }
        }
    }

    /// Append another parser output onto this one.
    pub fn append(&mut self, other: Self) {
        for event in other.events {
            match event {
                UnifiedParserEvent::Text(text) => self.push_text(text),
                UnifiedParserEvent::Reasoning(reasoning) => self.push_reasoning(reasoning),
                UnifiedParserEvent::ToolCall(call) => self.push_call(call),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{UnifiedParserEvent, UnifiedParserOutput};
    use crate::tool::ToolCallDelta;

    #[test]
    fn unified_parser_output_coalesces_adjacent_text_events() {
        let mut output = UnifiedParserOutput::default();
        output.push_text("hello");
        output.push_text(" ");
        output.push_text("world");
        output.push_reasoning("think");
        output.push_reasoning("ing");
        output.push_call(ToolCallDelta {
            tool_index: 0,
            name: Some("lookup".to_string()),
            arguments: "{}".to_string(),
        });
        output.push_text("!");

        assert_eq!(
            output.events,
            vec![
                UnifiedParserEvent::Text("hello world".to_string()),
                UnifiedParserEvent::Reasoning("thinking".to_string()),
                UnifiedParserEvent::ToolCall(ToolCallDelta {
                    tool_index: 0,
                    name: Some("lookup".to_string()),
                    arguments: "{}".to_string(),
                }),
                UnifiedParserEvent::Text("!".to_string()),
            ]
        );
    }

    #[test]
    fn unified_parser_output_append_coalesces_adjacent_events() {
        let mut output = UnifiedParserOutput::default();
        output.push_text("hello");

        let mut other = UnifiedParserOutput::default();
        other.push_text(" ");
        other.push_text("world");
        other.push_reasoning("think");
        output.append(other);

        let mut after_reasoning = UnifiedParserOutput::default();
        after_reasoning.push_reasoning("ing");
        after_reasoning.push_text("!");
        output.append(after_reasoning);

        assert_eq!(
            output.events,
            vec![
                UnifiedParserEvent::Text("hello world".to_string()),
                UnifiedParserEvent::Reasoning("thinking".to_string()),
                UnifiedParserEvent::Text("!".to_string()),
            ]
        );
    }
}

/// Incremental parser that extracts reasoning and tool-call events from assistant output.
pub trait UnifiedParser: Send {
    /// Construct a boxed parser instance for one request stream.
    fn create(tools: &[Tool], tokenizer: DynTokenizer) -> Result<Box<dyn UnifiedParser>>
    where
        Self: Sized + 'static;

    /// Initialize parser state from prompt token IDs before output deltas arrive.
    fn initialize(&mut self, _prompt_token_ids: &[u32]) -> Result<()> {
        Ok(())
    }

    /// Return whether decoded output must preserve tokenizer special tokens.
    fn preserve_special_tokens(&self) -> bool {
        false
    }

    /// Return the xgrammar structural-tag model used for strict tool calling.
    fn structural_tag_model(&self) -> Option<StructuralTagModel> {
        None
    }

    /// Return the parser-provided ID for a tool call by index, if the model emitted one.
    fn tool_call_id(&self, _tool_index: usize) -> Option<&str> {
        None
    }

    /// Feed one decoded text delta into the parser, appending committed output into `output`.
    fn parse_into(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> Result<()>;

    /// Flush any buffered parser state at end of stream.
    fn finish(&mut self) -> Result<UnifiedParserOutput> {
        Ok(UnifiedParserOutput::default())
    }

    /// Clear parser state and return currently uncommitted buffered text.
    fn reset(&mut self) -> String {
        String::new()
    }
}

/// Errors produced while creating or running unified parsers.
#[derive(Debug, Error, Macro)]
#[thiserror_ext(macro(path = "crate::unified", mangle))]
pub enum UnifiedParserError {
    #[error("combined parser is constructed from split parser instances")]
    CombinedParserConstructor,
    #[error("tokenizer is missing unified parser token `{token}`")]
    MissingToken { token: String },
    #[error("unified parser parsing failed: {message}")]
    ParsingFailed { message: String },
    #[error(transparent)]
    Reasoning(#[from] ReasoningError),
    #[error(transparent)]
    Tool(#[from] ToolParserError),
}

/// Returns the ID for the given token, or an error if it's not found.
fn token_id(tokenizer: &dyn vllm_tokenizer::Tokenizer, token: &str) -> Result<u32> {
    tokenizer.token_to_id(token).ok_or_else(|| UnifiedParserError::MissingToken {
        token: token.to_string(),
    })
}
