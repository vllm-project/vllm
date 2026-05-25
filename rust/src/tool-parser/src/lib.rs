//! Streaming tool parsers for chat completions.

#[macro_use]
mod error;
mod deepseek_dsml;
mod deepseek_json;
mod gemma4;
mod glm_xml;
mod json;
mod kimi_k2;
mod minimax_m2;
mod parameters;
mod qwen_coder;
#[cfg(any(test, feature = "test-util"))]
pub mod test_utils;
mod utils;

use std::collections::{BTreeMap, btree_map};

pub use deepseek_dsml::{DeepSeekV4ToolParser, DeepSeekV32ToolParser};
pub use deepseek_json::{DeepSeekV3ToolParser, DeepSeekV31ToolParser};
pub use error::{Result, ToolParserError};
pub use gemma4::Gemma4ToolParser;
pub use glm_xml::{Glm45MoeToolParser, Glm47MoeToolParser};
pub use json::{HermesToolParser, Llama3JsonToolParser, MistralToolParser, Qwen3XmlToolParser};
pub use kimi_k2::KimiK2ToolParser;
pub use minimax_m2::MinimaxM2ToolParser;
pub use qwen_coder::Qwen3CoderToolParser;
use serde::{Deserialize, Serialize};
use serde_json::Value;

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

#[cfg(test)]
mod tests;
