// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_tokenizer::DynTokenizer;

use super::{DelimitedReasoningParser, ReasoningDelta, ReasoningParser, Result};

const M3_THINK_START: &str = "<mm:think>";
const M3_THINK_END: &str = "</mm:think>";

/// Reasoning parser for MiniMax M3 style outputs.
///
/// MiniMax M3 uses `<mm:think>...</mm:think>` delimiters. Its chat template may
/// prefill either delimiter depending on the requested thinking mode, so the
/// shared delimited parser derives the starting state from the rendered prompt.
pub struct MiniMaxM3ReasoningParser {
    inner: DelimitedReasoningParser,
    /// True until the first response text is classified. Only this position may
    /// drop a stray `</mm:think>` emitted at the start of a response.
    at_response_start: bool,
    /// Holds an initial suffix like `</mm` while it may still complete into the
    /// leading closer on a later chunk.
    leading_end_buffer: String,
}

impl MiniMaxM3ReasoningParser {
    /// Create a MiniMax M3 parser backed by the shared delimited state machine.
    pub fn new(tokenizer: DynTokenizer) -> Result<Self> {
        Ok(Self {
            inner: DelimitedReasoningParser::new(tokenizer, M3_THINK_START, M3_THINK_END, false)?,
            at_response_start: true,
            leading_end_buffer: String::new(),
        })
    }

    /// Drop a response-leading `</mm:think>` while preserving later unmatched
    /// closers as ordinary content.
    fn push_inner(&mut self, delta: &str) -> ReasoningDelta {
        if self.at_response_start && !self.inner.in_reasoning() {
            self.leading_end_buffer.push_str(delta);
            let buffered = std::mem::take(&mut self.leading_end_buffer);

            if buffered.is_empty() {
                return ReasoningDelta::default();
            }
            if let Some(rest) = buffered.strip_prefix(M3_THINK_END) {
                self.at_response_start = false;
                return self.inner.push(rest);
            }
            if M3_THINK_END.starts_with(buffered.as_str()) {
                self.leading_end_buffer = buffered;
                return ReasoningDelta::default();
            }

            self.at_response_start = false;
            return self.inner.push(&buffered);
        }

        self.inner.push(delta)
    }
}

fn append_delta(target: &mut ReasoningDelta, delta: ReasoningDelta) {
    if let Some(reasoning) = delta.reasoning {
        target.push_reasoning(&reasoning);
    }
    if let Some(content) = delta.content {
        target.push_content(&content);
    }
}

impl ReasoningParser for MiniMaxM3ReasoningParser {
    fn create(tokenizer: DynTokenizer) -> Result<Box<dyn ReasoningParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tokenizer)?))
    }

    fn initialize(&mut self, prompt_token_ids: &[u32]) -> Result<()> {
        self.inner.initialize(prompt_token_ids);
        self.at_response_start = true;
        self.leading_end_buffer.clear();
        Ok(())
    }

    fn push(&mut self, delta: &str) -> Result<ReasoningDelta> {
        Ok(self.push_inner(delta))
    }

    fn finish(&mut self) -> Result<ReasoningDelta> {
        let mut delta = ReasoningDelta::default();
        if !self.leading_end_buffer.is_empty() {
            let pending = std::mem::take(&mut self.leading_end_buffer);
            self.at_response_start = false;
            append_delta(&mut delta, self.inner.push(&pending));
        }
        append_delta(&mut delta, self.inner.finish());
        Ok(delta)
    }
}
