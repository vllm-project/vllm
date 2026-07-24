use vllm_tokenizer::DynTokenizer;

use super::{DelimitedReasoningParser, ReasoningDelta, ReasoningParser, Result};

/// Reasoning parser for Mistral models using `[THINK]`/`[/THINK]` delimiters.
pub struct MistralReasoningParser {
    inner: DelimitedReasoningParser,
}

impl MistralReasoningParser {
    /// Create a Mistral reasoning parser backed by the shared delimited state machine.
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
    use crate::reasoning::ReasoningParser;
    use crate::reasoning::tests::{MISTRAL_THINK_END_ID, MISTRAL_THINK_START_ID, fake_tokenizer};

    #[test]
    fn no_delimiters_is_content() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = MistralReasoningParser::new(tokenizer).unwrap();

        let delta = parser.push("This is content").unwrap();
        assert_eq!(delta.reasoning, None);
        assert_eq!(delta.content.as_deref(), Some("This is content"));
    }

    #[test]
    fn valid_reasoning_and_content() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = MistralReasoningParser::new(tokenizer).unwrap();

        let delta = parser
            .push("[THINK]This is a reasoning section[/THINK]This is the rest")
            .unwrap();
        assert_eq!(
            delta.reasoning.as_deref(),
            Some("This is a reasoning section")
        );
        assert_eq!(delta.content.as_deref(), Some("This is the rest"));
    }

    #[test]
    fn reasoning_without_end_tag() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = MistralReasoningParser::new(tokenizer).unwrap();

        let delta = parser.push("[THINK]This is reasoning").unwrap();
        assert_eq!(delta.reasoning.as_deref(), Some("This is reasoning"));
        assert_eq!(delta.content, None);
    }

    #[test]
    fn reasoning_without_end_tag_finish_flushes() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = MistralReasoningParser::new(tokenizer).unwrap();

        let delta = parser.push("[THINK]partial reason").unwrap();
        assert_eq!(delta.reasoning.as_deref(), Some("partial reason"));

        let final_delta = parser.finish().unwrap();
        assert_eq!(final_delta.reasoning, None);
        assert_eq!(final_delta.content, None);
    }

    #[test]
    fn complete_reasoning_no_trailing_content() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = MistralReasoningParser::new(tokenizer).unwrap();

        let delta = parser.push("[THINK]This is a reasoning section[/THINK]").unwrap();
        assert_eq!(
            delta.reasoning.as_deref(),
            Some("This is a reasoning section")
        );
        assert_eq!(delta.content, None);
    }

    #[test]
    fn multiline_reasoning() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = MistralReasoningParser::new(tokenizer).unwrap();

        let delta = parser.push("[THINK]This\nThat[/THINK]This is the rest\nThat").unwrap();
        assert_eq!(delta.reasoning.as_deref(), Some("This\nThat"));
        assert_eq!(delta.content.as_deref(), Some("This is the rest\nThat"));
    }

    #[test]
    fn empty_input() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = MistralReasoningParser::new(tokenizer).unwrap();

        let delta = parser.push("").unwrap();
        assert!(delta.is_empty());
    }

    #[test]
    fn without_prompt_markers_expects_start_token() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = MistralReasoningParser::new(tokenizer).unwrap();

        let delta = parser.push("implicit reasoning[/THINK]answer").unwrap();
        assert_eq!(delta.reasoning, None);
        assert_eq!(
            delta.content.as_deref(),
            Some("implicit reasoning[/THINK]answer")
        );
    }

    #[test]
    fn picks_up_prompt_start_boundary() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = MistralReasoningParser::new(tokenizer).unwrap();
        parser.initialize(&[MISTRAL_THINK_START_ID]).unwrap();

        let delta = parser.push("reason[/THINK]answer").unwrap();
        assert_eq!(delta.reasoning.as_deref(), Some("reason"));
        assert_eq!(delta.content.as_deref(), Some("answer"));
    }

    #[test]
    fn respects_prompt_end_boundary() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = MistralReasoningParser::new(tokenizer).unwrap();
        parser.initialize(&[MISTRAL_THINK_END_ID]).unwrap();

        let delta = parser.push("answer").unwrap();
        assert_eq!(delta.reasoning, None);
        assert_eq!(delta.content.as_deref(), Some("answer"));
    }

    #[test]
    fn streams_across_pushes() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = MistralReasoningParser::new(tokenizer).unwrap();

        let mut reasoning = String::new();
        let mut content = String::new();
        for delta_str in [
            "[THINK]",
            "Some ",
            "reasoning ",
            "content",
            "[/THINK]",
            "Final ",
            "answer",
        ] {
            let delta = parser.push(delta_str).unwrap();
            if let Some(r) = delta.reasoning {
                reasoning.push_str(&r);
            }
            if let Some(c) = delta.content {
                content.push_str(&c);
            }
        }
        assert_eq!(reasoning, "Some reasoning content");
        assert_eq!(content, "Final answer");
    }

    #[test]
    fn handles_partial_delimiters_across_pushes() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = MistralReasoningParser::new(tokenizer).unwrap();
        parser.initialize(&[MISTRAL_THINK_START_ID]).unwrap();

        let first = parser.push("reason[/THI").unwrap();
        assert_eq!(first.reasoning.as_deref(), Some("reason"));
        assert_eq!(first.content, None);

        let second = parser.push("NK]answer").unwrap();
        assert_eq!(second.reasoning, None);
        assert_eq!(second.content.as_deref(), Some("answer"));
    }
}
