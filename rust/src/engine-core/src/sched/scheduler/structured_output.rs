// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The scheduler's view of the `StructuredOutputManager`.

use crate::sched::request::Request;

/// State of a structured output request's grammar
/// (`request.structured_output_request.grammar`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GrammarStatus {
    /// Compilation has not finished (`grammar is None`).
    Pending,
    /// The grammar is compiled.
    Ready,
    /// Compilation failed; `message` is `str(exception)`.
    Failed { message: String },
}

/// The `StructuredOutputManager` methods the scheduler calls.
///
/// The scheduler only calls these for requests with
/// `use_structured_output`, for which Python's manager does not take its
/// unstructured fast path.
pub trait StructuredOutput {
    /// The grammar state, read by the scheduler while the request is in
    /// `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`.
    fn grammar_status(&mut self, request: &Request) -> GrammarStatus;

    /// Advance grammar with accepted tokens. Returns False on rejection.
    fn accept_tokens(&mut self, request: &Request, new_token_ids: &[i32]) -> bool;

    /// Trim structured-output padding and grammar-invalid suffixes.
    fn trim_spec_tokens(&mut self, request: &Request, spec_token_ids: Vec<i32>) -> Vec<i32>;
}

/// A manager whose grammars are always ready and accept every token.
#[derive(Clone, Copy, Debug, Default)]
pub struct AcceptAllStructuredOutput;

impl StructuredOutput for AcceptAllStructuredOutput {
    fn grammar_status(&mut self, _request: &Request) -> GrammarStatus {
        GrammarStatus::Ready
    }

    fn accept_tokens(&mut self, _request: &Request, _new_token_ids: &[i32]) -> bool {
        true
    }

    fn trim_spec_tokens(&mut self, _request: &Request, spec_token_ids: Vec<i32>) -> Vec<i32> {
        spec_token_ids
    }
}
