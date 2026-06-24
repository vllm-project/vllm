//! Unified parser interface for reasoning and tool-call deltas.

mod combined;

use thiserror::Error;

pub use combined::CombinedParser;

use crate::reasoning::ReasoningError;
use crate::tool::{StructuralTagModel, ToolCallDelta, ToolParserError, ToolParserOutput};

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
    pub fn push_text(&mut self, delta: String) {
        if delta.is_empty() {
            return;
        }
        self.events.push(UnifiedParserEvent::Text(delta));
    }

    /// Append one reasoning text event if `delta` is non-empty.
    pub fn push_reasoning(&mut self, delta: String) {
        if delta.is_empty() {
            return;
        }
        self.events.push(UnifiedParserEvent::Reasoning(delta));
    }

    /// Append parsed tool parser output as unified events.
    pub fn append_tool_output(&mut self, output: ToolParserOutput) {
        // TODO: make ToolParserOutput carry ordered events and remove this text-first flattening.
        self.push_text(output.normal_text);
        self.events.extend(output.calls.into_iter().map(UnifiedParserEvent::ToolCall));
    }

    /// Append another parser output onto this one.
    pub fn append(&mut self, mut other: Self) {
        self.events.append(&mut other.events);
    }
}

/// Incremental parser that extracts reasoning and tool-call events from assistant output.
pub trait UnifiedParser: Send {
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
#[derive(Debug, Error)]
pub enum UnifiedParserError {
    #[error(transparent)]
    Reasoning(#[from] ReasoningError),
    #[error(transparent)]
    Tool(#[from] ToolParserError),
}
