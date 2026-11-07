// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_tokenizer::{DecodedText, DynTokenizer};
use xgrammar_structural_tag::format::Format;

use crate::output_grammar::{self, OutputGrammarContext};
use crate::utils::Marker;

use super::{
    DelimitedReasoningParser, DelimitedReasoningParserBuilder, ReasoningDelta, ReasoningParser,
    Result,
};

/// Reasoning parser for MiniMax M2 models, framed like Qwen3.
///
/// A `<minimax:tool_call>` emitted before `</think>` ends reasoning and starts
/// the tool call.
pub struct MiniMaxM2ReasoningParser {
    inner: DelimitedReasoningParser,
}

impl MiniMaxM2ReasoningParser {
    /// Create a MiniMax M2 parser backed by the shared delimited state machine.
    pub fn new(tokenizer: DynTokenizer) -> Result<Self> {
        let tool_call_start = Marker::special_in(tokenizer.as_ref(), "<minimax:tool_call>");
        Ok(Self {
            inner: DelimitedReasoningParserBuilder::new(tokenizer, "<think>", "</think>")
                .with_after_start("\n")
                .with_before_end("\n")
                .with_after_end("\n\n")
                .with_implicit_ends(tool_call_start)
                .build()?,
        })
    }
}

impl ReasoningParser for MiniMaxM2ReasoningParser {
    fn create(tokenizer: DynTokenizer) -> Result<Box<dyn ReasoningParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tokenizer)?))
    }

    fn initialize(&mut self, prompt_token_ids: &[u32]) -> Result<()> {
        self.inner.initialize(prompt_token_ids)
    }

    fn wrap_visible_format(
        &self,
        _ctx: &OutputGrammarContext<'_>,
        visible: &Format,
    ) -> output_grammar::Result<Option<Format>> {
        Ok(Some(self.inner.wrap_visible_format(visible)))
    }

    fn push(&mut self, delta: DecodedText) -> Result<ReasoningDelta> {
        Ok(self.inner.push(delta))
    }

    fn finish(&mut self) -> Result<ReasoningDelta> {
        Ok(self.inner.finish())
    }
}
