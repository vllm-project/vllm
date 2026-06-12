use vllm_tokenizer::DynTokenizer;

use super::{DelimitedReasoningParser, ReasoningDelta, ReasoningParser, Result};

/// Reasoning parser for Mistral models that wrap reasoning in `[THINK]` /
/// `[/THINK]`, such as Magistral.
///
/// `[THINK]` is optional, so the no-boundary fallback starts in content
/// (`in_reasoning = false`). The delimiters are Mistral special tokens, so
/// `preserve_special_tokens` returns `true` to keep them in the decoded text.
///
/// Python reference:
/// <https://github.com/vllm-project/vllm/blob/6f573f486/vllm/reasoning/mistral_reasoning_parser.py>
pub struct MistralReasoningParser {
    inner: DelimitedReasoningParser,
}

impl MistralReasoningParser {
    /// Create a Mistral parser backed by the shared delimited state machine.
    pub fn new(tokenizer: DynTokenizer) -> Result<Self> {
        Ok(Self {
            inner: DelimitedReasoningParser::new(tokenizer, "[THINK]", "[/THINK]", false)?,
        })
    }
}

impl ReasoningParser for MistralReasoningParser {
    fn create(tokenizer: DynTokenizer) -> Result<Box<dyn ReasoningParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tokenizer)?))
    }

    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn initialize(&mut self, prompt_token_ids: &[u32]) -> Result<()> {
        self.inner.initialize(prompt_token_ids);
        Ok(())
    }

    fn push(&mut self, delta: &str) -> Result<ReasoningDelta> {
        Ok(self.inner.push(delta))
    }

    fn finish(&mut self) -> Result<ReasoningDelta> {
        Ok(self.inner.finish())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::MistralReasoningParser;
    use crate::{
        ReasoningParser,
        tests::{FakeTokenizer, run_streaming},
    };

    #[test]
    fn streaming_handles_think_delimited_outputs() {
        let cases = [
            (
                "no_think_token",
                vec!["implicit reasoning[/THINK]answer"],
                None,
                Some("implicit reasoning[/THINK]answer"),
            ),
            (
                "reasoning_and_content",
                vec!["[THINK]reason[/THINK]answer"],
                Some("reason"),
                Some("answer"),
            ),
            (
                "streamed_across_deltas",
                vec![
                    "[THINK]",
                    "Some ",
                    "reasoning ",
                    "content",
                    "[/THINK]",
                    "Final ",
                    "answer",
                ],
                Some("Some reasoning content"),
                Some("Final answer"),
            ),
            (
                "no_end_delimiter",
                vec!["[THINK]reason without end"],
                Some("reason without end"),
                None,
            ),
            ("empty", vec![""], None, None),
            (
                "empty_reasoning",
                vec!["[THINK][/THINK]answer"],
                None,
                Some("answer"),
            ),
            (
                "content_around_reasoning",
                vec!["pre[THINK]reason[/THINK]post"],
                Some("reason"),
                Some("prepost"),
            ),
        ];

        for (name, output, expected_reasoning, expected_content) in cases {
            let mut parser = MistralReasoningParser::new(Arc::new(FakeTokenizer)).unwrap();
            let (reasoning, content) = run_streaming(&mut parser, &output);
            assert_eq!(reasoning.as_deref(), expected_reasoning, "{name}");
            assert_eq!(content.as_deref(), expected_content, "{name}");
        }
    }

    #[test]
    fn picks_up_prompt_start_boundary() {
        let tokenizer = Arc::new(FakeTokenizer);
        let mut parser = MistralReasoningParser::new(tokenizer).unwrap();
        // Prompt prefills `[THINK]` (id 12), opening reasoning before the stream.
        parser.initialize(&[12]).unwrap();

        let delta = parser.push("reason[/THINK]answer").unwrap();
        assert_eq!(delta.reasoning.as_deref(), Some("reason"));
        assert_eq!(delta.content.as_deref(), Some("answer"));
    }

    #[test]
    fn respects_prompt_end_boundary() {
        let tokenizer = Arc::new(FakeTokenizer);
        let mut parser = MistralReasoningParser::new(tokenizer).unwrap();
        // Prompt already closed reasoning with `[/THINK]` (id 13).
        parser.initialize(&[13]).unwrap();

        let delta = parser.push("answer").unwrap();
        assert_eq!(delta.reasoning, None);
        assert_eq!(delta.content.as_deref(), Some("answer"));
    }

    #[test]
    fn handles_partial_delimiters_across_pushes() {
        let tokenizer = Arc::new(FakeTokenizer);
        let mut parser = MistralReasoningParser::new(tokenizer).unwrap();
        parser.initialize(&[12]).unwrap();

        // Closing delimiter `[/THINK]` arrives in two halves.
        let first = parser.push("reason[/TH").unwrap();
        assert_eq!(first.reasoning.as_deref(), Some("reason"));
        assert_eq!(first.content, None);

        let second = parser.push("INK]answer").unwrap();
        assert_eq!(second.reasoning, None);
        assert_eq!(second.content.as_deref(), Some("answer"));
    }

    #[test]
    fn preserves_special_tokens_for_decoding() {
        let tokenizer = Arc::new(FakeTokenizer);
        let parser = MistralReasoningParser::new(tokenizer).unwrap();
        assert!(parser.preserve_special_tokens());
    }
}
