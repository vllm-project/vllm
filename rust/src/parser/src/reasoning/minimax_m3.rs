// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_tokenizer::{DecodedText, DynTokenizer};

use super::{
    DelimitedReasoningParser, DelimitedReasoningParserBuilder, ReasoningDelta, ReasoningParser,
    Result,
};

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
    leading_end_buffer: DecodedText,
}

impl MiniMaxM3ReasoningParser {
    /// Create a MiniMax M3 parser backed by the shared delimited state machine.
    pub fn new(tokenizer: DynTokenizer) -> Result<Self> {
        Ok(Self {
            inner: DelimitedReasoningParserBuilder::new(tokenizer, M3_THINK_START, M3_THINK_END)
                .build()?,
            at_response_start: true,
            leading_end_buffer: DecodedText::default(),
        })
    }

    /// Drop a response-leading `</mm:think>` while preserving later unmatched
    /// closers as ordinary content.
    fn push_inner(&mut self, delta: DecodedText) -> ReasoningDelta {
        if self.at_response_start && !self.inner.in_reasoning() {
            self.leading_end_buffer.append(delta);
            let mut buffered = self.leading_end_buffer.take();

            if buffered.text.starts_with(M3_THINK_END) {
                self.at_response_start = false;
                // The dropped marker span keeps its tokens out of any count.
                let _ = buffered.drain_prefix(M3_THINK_END.len());
                return self.inner.push(buffered);
            }
            if M3_THINK_END.starts_with(buffered.text.as_str()) {
                self.leading_end_buffer = buffered;
                return ReasoningDelta::default();
            }

            self.at_response_start = false;
            return self.inner.push(buffered);
        }

        self.inner.push(delta)
    }
}

fn append_delta(target: &mut ReasoningDelta, delta: ReasoningDelta) {
    if let Some(reasoning) = delta.reasoning {
        target.push_reasoning(reasoning);
    }
    if let Some(content) = delta.content {
        target.push_content(content);
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
        self.inner.initialize(prompt_token_ids)?;
        self.at_response_start = true;
        self.leading_end_buffer.clear();
        Ok(())
    }

    fn push(&mut self, delta: DecodedText) -> Result<ReasoningDelta> {
        Ok(self.push_inner(delta))
    }

    fn finish(&mut self) -> Result<ReasoningDelta> {
        let mut delta = ReasoningDelta::default();
        if !self.leading_end_buffer.is_empty() {
            let pending = self.leading_end_buffer.take();
            self.at_response_start = false;
            append_delta(&mut delta, self.inner.push(pending));
        }
        append_delta(&mut delta, self.inner.finish());
        Ok(delta)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::MiniMaxM3ReasoningParser;
    use crate::reasoning::ReasoningParser;
    use crate::reasoning::tests::{
        MM_THINK_END_ID, MM_THINK_START_ID, content_str, fake_tokenizer, push_str, reasoning_str,
    };

    #[test]
    fn minimax_m3_handles_explicit_think_delimiters() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = MiniMaxM3ReasoningParser::new(tokenizer).unwrap();

        let delta = push_str(&mut parser, "<mm:think>reason</mm:think>answer");
        assert_eq!(reasoning_str(&delta), Some("reason"));
        assert_eq!(content_str(&delta), Some("answer"));
    }

    #[test]
    fn minimax_m3_drops_leading_end_marker() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = MiniMaxM3ReasoningParser::new(tokenizer).unwrap();

        let delta = push_str(&mut parser, "</mm:think>answer");
        assert_eq!(delta.reasoning, None);
        assert_eq!(content_str(&delta), Some("answer"));
    }

    #[test]
    fn minimax_m3_preserves_non_leading_end_marker() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = MiniMaxM3ReasoningParser::new(tokenizer).unwrap();

        let delta = push_str(&mut parser, "XXX</mm:think>YYY");
        assert_eq!(delta.reasoning, None);
        assert_eq!(content_str(&delta), Some("XXX</mm:think>YYY"));
    }

    #[test]
    fn minimax_m3_drops_split_leading_end_marker() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = MiniMaxM3ReasoningParser::new(tokenizer).unwrap();

        assert!(push_str(&mut parser, "</mm").is_empty());
        let delta = push_str(&mut parser, ":think>answer");
        assert_eq!(delta.reasoning, None);
        assert_eq!(content_str(&delta), Some("answer"));
    }

    #[test]
    fn minimax_m3_uses_prompt_prefilled_start_marker() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = MiniMaxM3ReasoningParser::new(tokenizer).unwrap();
        parser.initialize(&[MM_THINK_START_ID]).unwrap();

        let delta = push_str(&mut parser, "reason</mm:think>answer");
        assert_eq!(reasoning_str(&delta), Some("reason"));
        assert_eq!(content_str(&delta), Some("answer"));
    }

    #[test]
    fn minimax_m3_uses_prompt_prefilled_end_marker() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = MiniMaxM3ReasoningParser::new(tokenizer).unwrap();
        parser.initialize(&[MM_THINK_END_ID]).unwrap();

        let delta = push_str(&mut parser, "answer");
        assert_eq!(delta.reasoning, None);
        assert_eq!(content_str(&delta), Some("answer"));
    }
}
