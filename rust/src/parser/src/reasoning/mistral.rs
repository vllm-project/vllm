use vllm_tokenizer::{DecodedText, DynTokenizer};

use super::{
    DelimitedReasoningParser, DelimitedReasoningParserBuilder, ReasoningDelta, ReasoningParser,
    Result,
};

/// Reasoning parser for Mistral models using `[THINK]`/`[/THINK]` delimiters.
pub struct MistralReasoningParser {
    inner: DelimitedReasoningParser,
}

impl MistralReasoningParser {
    /// Create a Mistral reasoning parser backed by the shared delimited state machine.
    pub fn new(tokenizer: DynTokenizer) -> Result<Self> {
        Ok(Self {
            inner: DelimitedReasoningParserBuilder::new(tokenizer, "[THINK]", "[/THINK]")
                .build()?,
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
        self.inner.initialize(prompt_token_ids)
    }

    fn push(&mut self, delta: DecodedText) -> Result<ReasoningDelta> {
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
    use crate::reasoning::tests::{
        MISTRAL_THINK_END_ID, MISTRAL_THINK_START_ID, content_str, fake_tokenizer, push_str,
        reasoning_str,
    };

    #[test]
    fn no_delimiters_is_content() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = MistralReasoningParser::new(tokenizer).unwrap();

        let delta = push_str(&mut parser, "This is content");
        assert_eq!(delta.reasoning, None);
        assert_eq!(content_str(&delta), Some("This is content"));
    }

    #[test]
    fn valid_reasoning_and_content() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = MistralReasoningParser::new(tokenizer).unwrap();

        let delta = push_str(
            &mut parser,
            "[THINK]This is a reasoning section[/THINK]This is the rest",
        );
        assert_eq!(reasoning_str(&delta), Some("This is a reasoning section"));
        assert_eq!(content_str(&delta), Some("This is the rest"));
    }

    #[test]
    fn reasoning_without_end_tag() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = MistralReasoningParser::new(tokenizer).unwrap();

        let delta = push_str(&mut parser, "[THINK]This is reasoning");
        assert_eq!(reasoning_str(&delta), Some("This is reasoning"));
        assert_eq!(delta.content, None);
    }

    #[test]
    fn reasoning_without_end_tag_finish_flushes() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = MistralReasoningParser::new(tokenizer).unwrap();

        let delta = push_str(&mut parser, "[THINK]partial reason");
        assert_eq!(reasoning_str(&delta), Some("partial reason"));

        let final_delta = parser.finish().unwrap();
        assert_eq!(final_delta.reasoning, None);
        assert_eq!(final_delta.content, None);
    }

    #[test]
    fn complete_reasoning_no_trailing_content() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = MistralReasoningParser::new(tokenizer).unwrap();

        let delta = push_str(&mut parser, "[THINK]This is a reasoning section[/THINK]");
        assert_eq!(reasoning_str(&delta), Some("This is a reasoning section"));
        assert_eq!(delta.content, None);
    }

    #[test]
    fn multiline_reasoning() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = MistralReasoningParser::new(tokenizer).unwrap();

        let delta = push_str(
            &mut parser,
            "[THINK]This\nThat[/THINK]This is the rest\nThat",
        );
        assert_eq!(reasoning_str(&delta), Some("This\nThat"));
        assert_eq!(content_str(&delta), Some("This is the rest\nThat"));
    }

    #[test]
    fn empty_input() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = MistralReasoningParser::new(tokenizer).unwrap();

        let delta = push_str(&mut parser, "");
        assert!(delta.is_empty());
    }

    #[test]
    fn without_prompt_markers_expects_start_token() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = MistralReasoningParser::new(tokenizer).unwrap();

        let delta = push_str(&mut parser, "implicit reasoning[/THINK]answer");
        assert_eq!(delta.reasoning, None);
        assert_eq!(
            content_str(&delta),
            Some("implicit reasoning[/THINK]answer")
        );
    }

    #[test]
    fn picks_up_prompt_start_boundary() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = MistralReasoningParser::new(tokenizer).unwrap();
        parser.initialize(&[MISTRAL_THINK_START_ID]).unwrap();

        let delta = push_str(&mut parser, "reason[/THINK]answer");
        assert_eq!(reasoning_str(&delta), Some("reason"));
        assert_eq!(content_str(&delta), Some("answer"));
    }

    #[test]
    fn respects_prompt_end_boundary() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = MistralReasoningParser::new(tokenizer).unwrap();
        parser.initialize(&[MISTRAL_THINK_END_ID]).unwrap();

        let delta = push_str(&mut parser, "answer");
        assert_eq!(delta.reasoning, None);
        assert_eq!(content_str(&delta), Some("answer"));
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
            let delta = push_str(&mut parser, delta_str);
            if let Some(r) = delta.reasoning {
                reasoning.push_str(&r.text);
            }
            if let Some(c) = delta.content {
                content.push_str(&c.text);
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

        let first = push_str(&mut parser, "reason[/THI");
        assert_eq!(reasoning_str(&first), Some("reason"));
        assert_eq!(first.content, None);

        let second = push_str(&mut parser, "NK]answer");
        assert_eq!(second.reasoning, None);
        assert_eq!(content_str(&second), Some("answer"));
    }
}
