// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The token-id view of a reasoning parser that the structured output manager
//! uses to gate the grammar while the model reasons.
//!
//! Python's reasoning parsers (`vllm/reasoning`) are model specific and keep
//! their own state, so the manager takes them through this trait: a binding
//! can wrap a Python parser instance, and [`ThinkTokenReasoner`] is a native
//! port of `BaseThinkingReasoningParser`'s token-id methods.

/// `vllm.parser.engine.parser_engine.ReasoningEnd`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReasoningEnd {
    /// Offset of the reasoning-end token.
    pub offset: usize,
    /// Whether the end token is also content (e.g. a tool-call start).
    pub implicit: bool,
}

/// The token-id methods of `vllm.reasoning.ReasoningParser` that
/// `StructuredOutputManager` calls. One instance per request.
pub trait Reasoner: Send + Sync {
    /// Every token id that alone ends reasoning.
    ///
    /// Empty if reasoning can also end in a way that is not a single token.
    fn reasoning_end_token_ids(&self) -> &[i32] {
        &[]
    }

    /// Reasoning-end tokens that are also content, e.g. a tool-call start.
    fn implicit_reasoning_end_token_ids(&self) -> &[i32] {
        &[]
    }

    /// Find the first reasoning end in `token_ids`.
    ///
    /// Returns its offset and whether its token is content, or
    /// `ReasoningEnd(len(token_ids), False)` if there is none. Returns `None`
    /// if the parser cannot locate reasoning ends by token id.
    fn find_reasoning_end(&mut self, _token_ids: &[i32]) -> Option<ReasoningEnd> {
        None
    }

    /// Check if the reasoning content ends in the input_ids.
    fn is_reasoning_end(&mut self, input_ids: &[i32]) -> bool;

    /// Check if the reasoning content ends in the input_ids on a decode step.
    ///
    /// `input_ids` the entire model output and `delta_ids` are the last few
    /// computed tokens of the model output (like during a decode step).
    fn is_reasoning_end_streaming(&mut self, input_ids: &[i32], _delta_ids: &[i32]) -> bool {
        self.is_reasoning_end(input_ids)
    }
}

/// `BaseThinkingReasoningParser`'s token-id methods: reasoning is delimited
/// by single start/end tokens such as `<think>` and `</think>`.
#[derive(Clone, Debug)]
pub struct ThinkTokenReasoner {
    pub start_token_id: i32,
    pub end_token_id: i32,
}

impl Reasoner for ThinkTokenReasoner {
    fn is_reasoning_end(&mut self, input_ids: &[i32]) -> bool {
        input_ids
            .iter()
            .rev()
            .find(|&&t| t == self.start_token_id || t == self.end_token_id)
            .is_some_and(|&t| t == self.end_token_id)
    }

    fn is_reasoning_end_streaming(&mut self, _input_ids: &[i32], delta_ids: &[i32]) -> bool {
        delta_ids.contains(&self.end_token_id)
    }
}
