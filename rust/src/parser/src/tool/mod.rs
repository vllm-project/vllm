//! Streaming tool parsers for chat completions.

#[macro_use]
pub(crate) mod error;
mod deepseek_dsml;
pub(crate) mod deepseek_json;
mod gemma4;
mod glm_xml;
mod hy_v3;
mod json;
mod kimi_k2;
mod minimax_m2;
mod minimax_m3;
mod parameters;
mod qwen_coder;
#[cfg(any(test, feature = "test-util"))]
pub mod test_utils;
pub(crate) mod utils;

use std::collections::{BTreeMap, btree_map};

pub use deepseek_dsml::{DeepSeekV4ToolParser, DeepSeekV32ToolParser};
pub use deepseek_json::{DeepSeekV3ToolParser, DeepSeekV31ToolParser};
pub use error::{Result, ToolParserError};
pub use gemma4::Gemma4ToolParser;
pub use glm_xml::{Glm45MoeToolParser, Glm47MoeToolParser};
pub use hy_v3::HyV3ToolParser;
pub use json::{
    Granite4ToolParser, HermesToolParser, Internlm2ToolParser, Llama3JsonToolParser,
    MistralToolParser, Phi4MiniJsonToolParser, Qwen3XmlToolParser,
};
pub use kimi_k2::KimiK2ToolParser;
pub use minimax_m2::MinimaxM2ToolParser;
pub use minimax_m3::MinimaxM3ToolParser;
pub use qwen_coder::Qwen3CoderToolParser;
use serde::{Deserialize, Serialize};
use serde_json::Value;
pub use xgrammar_structural_tag::Model as StructuralTagModel;

/// One function-style tool made available to the model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Value,
    pub strict: Option<bool>,
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

/// One ordered event emitted while parsing assistant text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolParserEvent {
    /// Plain assistant text that is not part of any tool call.
    Text(String),
    /// A tool-call update extracted from assistant text.
    ToolCall(ToolCallDelta),
}

/// Result of advancing tool parsing with one assistant-text input.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolParserOutput {
    /// Ordered parser events committed by this input.
    pub events: Vec<ToolParserEvent>,
}

impl ToolParserOutput {
    /// Append one visible text event if `text` is non-empty.
    pub fn push_text(&mut self, text: impl Into<String>) {
        let text = text.into();
        if text.is_empty() {
            return;
        }
        self.events.push(ToolParserEvent::Text(text));
    }

    /// Append one tool-call update event.
    pub fn push_call(&mut self, call: ToolCallDelta) {
        self.events.push(ToolParserEvent::ToolCall(call));
    }

    /// Return all plain assistant text committed by this output.
    pub fn normal_text(&self) -> String {
        self.events
            .iter()
            .filter_map(|event| match event {
                ToolParserEvent::Text(text) => Some(text.as_str()),
                ToolParserEvent::ToolCall(_) => None,
            })
            .collect()
    }

    /// Return all tool-call updates committed by this output.
    pub fn calls(&self) -> Vec<&ToolCallDelta> {
        self.events
            .iter()
            .filter_map(|event| match event {
                ToolParserEvent::Text(_) => None,
                ToolParserEvent::ToolCall(call) => Some(call),
            })
            .collect()
    }

    /// Append another parser output onto this one.
    ///
    /// Note that this keeps events exactly as they arrive. Call `coalesce()`
    /// after if final text and tool-call fragments should be flattened.
    pub fn append(&mut self, mut other: Self) {
        self.events.append(&mut other.events);
    }

    /// Flatten text and merge deltas for the same tool call.
    ///
    /// All text events are concatenated into one leading text event. Tool-call
    /// events follow that text event in first-seen tool index order, with
    /// argument fragments for the same tool call concatenated together.
    ///
    /// This is primarily used by the default `parse_complete()` implementation,
    /// which delegates through the incremental parser lifecycle and then
    /// needs to collapse streaming-style argument fragments into one final
    /// tool call.
    pub fn coalesce(self) -> Self {
        let mut merged = BTreeMap::<usize, ToolCallDelta>::new();
        let mut order = Vec::new();
        let normal_text = self.normal_text();

        for call in self.calls() {
            match merged.entry(call.tool_index) {
                btree_map::Entry::Vacant(entry) => {
                    order.push(call.tool_index);
                    entry.insert(call.clone());
                }
                btree_map::Entry::Occupied(mut entry) => {
                    let existing = entry.get_mut();
                    if existing.name.is_none() {
                        existing.name = call.name.clone();
                    }
                    existing.arguments.push_str(&call.arguments);
                }
            }
        }

        let mut output = Self::default();
        output.push_text(normal_text);
        for call in order.into_iter().filter_map(|tool_index| merged.remove(&tool_index)) {
            output.push_call(call);
        }
        output
    }
}

/// Incremental parser that extracts tool calls from assistant output.
pub trait ToolParser: Send {
    /// Construct a boxed parser instance for one request stream.
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static;

    /// Return whether decoded output must preserve tokenizer special tokens.
    ///
    /// Some model families emit tool-call sentinels as special tokens. Those
    /// parsers need `skip_special_tokens = false` while parsing is enabled.
    fn preserve_special_tokens(&self) -> bool {
        false
    }

    /// Return the xgrammar structural-tag model used for strict tool calling.
    fn structural_tag_model(&self) -> Option<StructuralTagModel> {
        None
    }

    /// Return the parser-provided ID for a tool call by index, if the model
    /// emitted one.
    fn tool_call_id(&self, _tool_index: usize) -> Option<&str> {
        None
    }

    /// Feed one decoded text delta into the parser, appending committed output
    /// into `output`.
    ///
    /// If this returns an error, any output already appended to `output`
    /// remains committed parser output. The parser must keep its uncommitted
    /// buffer intact so callers may recover it with `reset()`.
    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()>;

    /// Flush any buffered partial state at end of stream.
    ///
    /// This operation is atomic: on error no partial output is returned and the
    /// parser's buffered state is left intact.
    fn finish(&mut self) -> Result<ToolParserOutput>;

    /// Clear parser state and return currently uncommitted buffered text.
    ///
    /// Callers may use this to recover any text that failed to parse after an error
    /// and output it as normal text.
    fn reset(&mut self) -> String;
}

/// Extension methods for easily testing `ToolParser` implementations.
///
/// These helpers do not handle partial parsing or error recovery, so they are
/// not intended for use in production code paths.
#[cfg(any(test, feature = "test-util"))]
#[easy_ext::ext(ToolParserTestExt)]
impl<T: ToolParser + ?Sized> T {
    /// Feed one decoded text delta and return only if the whole chunk parses.
    ///
    /// If parsing fails, partial committed output is discarded by this helper.
    /// Prefer `parse_into` for more fine-grained control in error recovery.
    pub fn parse_chunk(&mut self, chunk: &str) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        self.parse_into(chunk, &mut output)?;
        Ok(output)
    }

    /// Parse complete tool calls from final output.
    ///
    /// This default implementation reuses the incremental parser lifecycle by
    /// feeding the full output through `parse_chunk()` and then calling `finish()`.
    ///
    /// If parsing fails, partial committed output is discarded by this helper.
    /// Prefer `parse_into` for more fine-grained control in error recovery.
    pub fn parse_complete(&mut self, text: &str) -> Result<ToolParserOutput> {
        let mut output = self.parse_chunk(text)?;
        output.append(self.finish()?);
        Ok(output.coalesce())
    }
}

#[cfg(test)]
mod tests;
