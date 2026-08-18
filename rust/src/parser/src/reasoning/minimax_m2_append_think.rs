// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_tokenizer::DynTokenizer;

use super::{ReasoningDelta, ReasoningParser, Result};

const THINK_START: &str = "<think>";

/// Compatibility parser that preserves MiniMax M2 reasoning markup as content.
///
/// MiniMax M2 emits `</think>` without a matching opening token. This parser
/// prepends `<think>` to the first non-empty output delta and otherwise leaves
/// the generated text untouched, matching the Python
/// `minimax_m2_append_think` parser behavior.
pub struct MiniMaxM2AppendThinkReasoningParser {
    prepended_start: bool,
}

impl Default for MiniMaxM2AppendThinkReasoningParser {
    fn default() -> Self {
        Self::new()
    }
}

impl MiniMaxM2AppendThinkReasoningParser {
    /// Create a parser with no request-scoped output state.
    pub fn new() -> Self {
        Self {
            prepended_start: false,
        }
    }
}

impl ReasoningParser for MiniMaxM2AppendThinkReasoningParser {
    fn create(_tokenizer: DynTokenizer) -> Result<Box<dyn ReasoningParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new()))
    }

    fn initialize(&mut self, _prompt_token_ids: &[u32]) -> Result<()> {
        self.prepended_start = false;
        Ok(())
    }

    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn push(&mut self, delta: &str) -> Result<ReasoningDelta> {
        if delta.is_empty() {
            return Ok(ReasoningDelta::default());
        }

        let mut output = ReasoningDelta::default();
        if !self.prepended_start {
            output.push_content(THINK_START);
            self.prepended_start = true;
        }
        output.push_content(delta);
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use expect_test::expect;
    use vllm_tokenizer::test_utils::TestTokenizer;

    use super::MiniMaxM2AppendThinkReasoningParser;
    use crate::reasoning::ReasoningParser;

    #[test]
    fn prepends_think_once_and_preserves_output_as_content() {
        let mut parser = MiniMaxM2AppendThinkReasoningParser::new();

        let first = parser.push("reasoning</think>").unwrap();
        let second = parser.push("answer").unwrap();

        expect![[r#"
            (
                ReasoningDelta {
                    reasoning: None,
                    content: Some(
                        "<think>reasoning</think>",
                    ),
                },
                ReasoningDelta {
                    reasoning: None,
                    content: Some(
                        "answer",
                    ),
                },
            )
        "#]]
        .assert_debug_eq(&(first, second));
    }

    #[test]
    fn ignores_empty_deltas_until_output_arrives() {
        let mut parser = MiniMaxM2AppendThinkReasoningParser::new();

        assert!(parser.push("").unwrap().is_empty());
        assert_eq!(
            parser.push("answer").unwrap().content.as_deref(),
            Some("<think>answer")
        );
    }

    #[test]
    fn initialize_resets_request_scoped_prefix_state() {
        let mut parser = MiniMaxM2AppendThinkReasoningParser::new();
        assert_eq!(
            parser.push("first").unwrap().content.as_deref(),
            Some("<think>first")
        );

        parser.initialize(&[]).unwrap();

        assert_eq!(
            parser.push("second").unwrap().content.as_deref(),
            Some("<think>second")
        );
    }

    #[test]
    fn factory_creation_preserves_special_tokens() {
        let tokenizer = Arc::new(TestTokenizer::new());
        let parser = MiniMaxM2AppendThinkReasoningParser::create(tokenizer).unwrap();
        assert!(parser.preserve_special_tokens());
    }
}
