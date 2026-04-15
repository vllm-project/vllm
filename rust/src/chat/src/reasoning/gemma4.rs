use vllm_text::tokenizers::DynTokenizer;

use super::{DelimitedReasoningParser, ReasoningDelta, ReasoningParser, Result};
use crate::request::ChatRequest;

const THOUGHT_PREFIX: &str = "thought\n";

/// Reasoning parser for Google Gemma4 thinking models.
///
/// Gemma4 emits reasoning inside `<|channel> ... <channel|>` spans and adds a
/// structural `thought\n` label at the beginning of the reasoning channel.
/// This parser keeps the delimiter handling in the shared delimited parser and
/// only layers on Gemma4-specific request adjustment plus prefix stripping.
///
/// Original Python implementation:
/// <https://github.com/vllm-project/vllm/blob/18b1c77211d8f6fe800bcfb89524d2b598708032/vllm/reasoning/gemma4_reasoning_parser.py#L23>
pub struct Gemma4ReasoningParser {
    inner: DelimitedReasoningParser,
    reasoning_text: String,
    prefix_stripped: bool,
}

impl Gemma4ReasoningParser {
    /// Create a Gemma4 parser.
    pub fn new(tokenizer: DynTokenizer) -> Result<Self> {
        Ok(Self {
            inner: DelimitedReasoningParser::new(tokenizer, "<|channel>", "<channel|>", false)?,
            reasoning_text: String::new(),
            prefix_stripped: false,
        })
    }

    /// Apply Gemma4's `thought\n` stripping rule to one reasoning delta.
    ///
    /// Early reasoning text is buffered until we can decide whether it begins
    /// with the structural channel label.
    fn strip_thought_prefix(&mut self, reasoning: &str) -> Option<String> {
        if self.prefix_stripped {
            return Some(reasoning.to_string());
        }

        self.reasoning_text.push_str(reasoning);

        if self.reasoning_text.starts_with(THOUGHT_PREFIX) {
            let prefix_len = THOUGHT_PREFIX.len();
            let previous_len = self.reasoning_text.len() - reasoning.len();
            if previous_len >= prefix_len {
                self.reasoning_text.clear();
                self.prefix_stripped = true;
                return Some(reasoning.to_string());
            }

            let prefix_chars_in_delta = prefix_len - previous_len;
            let stripped = &reasoning[prefix_chars_in_delta.min(reasoning.len())..];
            if stripped.is_empty() {
                if self.reasoning_text.len() >= prefix_len {
                    self.reasoning_text.clear();
                    self.prefix_stripped = true;
                }
                return None;
            }

            self.reasoning_text.clear();
            self.prefix_stripped = true;
            return Some(stripped.to_string());
        }

        if THOUGHT_PREFIX.starts_with(&self.reasoning_text) {
            return None;
        }

        self.prefix_stripped = true;
        Some(std::mem::take(&mut self.reasoning_text))
    }

    /// Apply Gemma4-specific reasoning post-processing to one parsed delta.
    fn post_process(&mut self, mut result: ReasoningDelta) -> ReasoningDelta {
        if let Some(reasoning) = result.reasoning.take() {
            result.reasoning = self
                .strip_thought_prefix(&reasoning)
                .filter(|text| !text.is_empty());
        }
        result
    }
}

impl ReasoningParser for Gemma4ReasoningParser {
    fn create(tokenizer: DynTokenizer) -> Result<Box<dyn ReasoningParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tokenizer)?))
    }

    fn adjust_request(&self, request: &mut ChatRequest) -> Result<()> {
        // Gemma4's reasoning delimiters are marked as special tokens, so we need to ensure they are
        // not stripped during decoding.
        request.decode_options.skip_special_tokens = false;
        Ok(())
    }

    fn initialize(&mut self, prompt_token_ids: &[u32]) -> Result<()> {
        self.inner.initialize(prompt_token_ids);
        self.reasoning_text.clear();
        self.prefix_stripped = false;
        Ok(())
    }

    fn push(&mut self, delta: &str) -> Result<ReasoningDelta> {
        let result = self.inner.push(delta);
        Ok(self.post_process(result))
    }

    fn finish(&mut self) -> Result<ReasoningDelta> {
        let result = self.inner.finish();
        Ok(self.post_process(result))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vllm_text::tokenizers::Tokenizer;

    use super::Gemma4ReasoningParser;
    use crate::reasoning::ReasoningParser;
    use crate::request::ChatRequest;

    struct FakeTokenizer;

    impl Tokenizer for FakeTokenizer {
        fn encode(&self, text: &str, _add_special_tokens: bool) -> vllm_text::Result<Vec<u32>> {
            Ok(text.chars().map(u32::from).collect())
        }

        fn decode(
            &self,
            token_ids: &[u32],
            _skip_special_tokens: bool,
        ) -> vllm_text::Result<String> {
            Ok(token_ids
                .iter()
                .map(|token_id| char::from_u32(*token_id).unwrap_or('\u{FFFD}'))
                .collect())
        }

        fn token_to_id(&self, token: &str) -> Option<u32> {
            match token {
                "<|channel>" => Some(1000),
                "<channel|>" => Some(1001),
                _ => None,
            }
        }
    }

    fn run_streaming(output: &[&str]) -> (Option<String>, Option<String>) {
        let tokenizer = Arc::new(FakeTokenizer);
        let mut parser = Gemma4ReasoningParser::new(tokenizer).unwrap();
        let mut reasoning = String::new();
        let mut content = String::new();

        for delta in output {
            let result = parser.push(delta).unwrap();
            if let Some(next) = result.reasoning {
                reasoning.push_str(&next);
            }
            if let Some(next) = result.content {
                content.push_str(&next);
            }
        }

        let final_delta = parser.finish().unwrap();
        if let Some(next) = final_delta.reasoning {
            reasoning.push_str(&next);
        }
        if let Some(next) = final_delta.content {
            content.push_str(&next);
        }

        (
            (!reasoning.is_empty()).then_some(reasoning),
            (!content.is_empty()).then_some(content),
        )
    }

    #[test]
    fn gemma4_reasoning_streaming_handles_channel_delimited_outputs() {
        let cases = [
            (
                "no_reasoning",
                vec!["This is content"],
                None,
                Some("This is content"),
            ),
            (
                "reasoning_and_content",
                vec!["<|channel>This is a reasoning section<channel|>This is the rest"],
                Some("This is a reasoning section"),
                Some("This is the rest"),
            ),
            (
                "complete_reasoning",
                vec!["<|channel>This is a reasoning section<channel|>"],
                Some("This is a reasoning section"),
                None,
            ),
            (
                "multiple_lines",
                vec!["<|channel>This\nThat<channel|>This is the rest\nThat"],
                Some("This\nThat"),
                Some("This is the rest\nThat"),
            ),
            (
                "no_end",
                vec!["<|channel>This is a reasoning section"],
                Some("This is a reasoning section"),
                None,
            ),
            ("empty", vec![""], None, None),
            (
                "newline_around_reasoning",
                vec!["Before\n<|channel>This is a reasoning section<channel|>\nThis is the rest"],
                Some("This is a reasoning section"),
                Some("Before\n\nThis is the rest"),
            ),
            (
                "thought_prefix",
                vec!["<|channel>thought\nActual reasoning here<channel|>Final answer"],
                Some("Actual reasoning here"),
                Some("Final answer"),
            ),
            (
                "thought_prefix_only",
                vec!["<|channel>thought\n<channel|>"],
                None,
                None,
            ),
            (
                "thought_prefix_multiline",
                vec!["<|channel>thought\nLine1\nLine2<channel|>Answer"],
                Some("Line1\nLine2"),
                Some("Answer"),
            ),
            (
                "thought_prefix_diverge",
                vec!["<|channel>thousand reasons<channel|>Done"],
                Some("thousand reasons"),
                Some("Done"),
            ),
        ];

        for (name, output, expected_reasoning, expected_content) in cases {
            let (reasoning, content) = run_streaming(&output);
            assert_eq!(reasoning.as_deref(), expected_reasoning, "{name}");
            assert_eq!(content.as_deref(), expected_content, "{name}");
        }
    }

    #[test]
    fn gemma4_strips_thought_prefix_even_when_split_across_deltas() {
        let (reasoning, content) =
            run_streaming(&["<|channel>thou", "ght", "\nabc", "<channel|>done"]);
        assert_eq!(reasoning.as_deref(), Some("abc"));
        assert_eq!(content.as_deref(), Some("done"));
    }

    #[test]
    fn gemma4_adjust_request_keeps_special_tokens() {
        let tokenizer = Arc::new(FakeTokenizer);
        let parser = Gemma4ReasoningParser::new(tokenizer).unwrap();
        let mut request = ChatRequest::for_test();

        assert!(request.decode_options.skip_special_tokens);
        parser.adjust_request(&mut request).unwrap();
        assert!(!request.decode_options.skip_special_tokens);
    }
}
