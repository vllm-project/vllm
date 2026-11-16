// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

/// Public error type for the Rust `llm` facade.
#[derive(Debug, Error)]
pub enum Error {
    #[error("generate request `{request_id}` has an empty prompt_token_ids")]
    EmptyPromptTokenIds { request_id: String },
    #[error("engine-core error")]
    EngineCoreClient(#[from] vllm_engine_core_client::Error),
    #[error(
        "sampling mask for request `{request_id}` has {row_count} rows for {token_count} generated tokens"
    )]
    SamplingMaskTokenCountMismatch {
        request_id: String,
        token_count: usize,
        row_count: usize,
    },
    #[error("invalid auxiliary output rows")]
    AuxOutputRows(#[from] vllm_engine_core_client::protocol::tensor::ArrayRowsError),
    /// The request asks for prompt logprobs, which the engine does not compute,
    /// for example under `--kv-sharing-fast-prefill`.
    #[error(
        "`{parameter}` is not supported: the engine does not compute prompt logprobs \
         (for example, it runs with --kv-sharing-fast-prefill)"
    )]
    PromptLogprobsUnsupported { parameter: &'static str },
    /// The prompt does not fit in one scheduling step while the engine does
    /// not chunk prefills, so it could never be scheduled.
    #[error(
        "The decoder prompt (length {prompt_len}) is longer than the {max_prompt_len} tokens \
         this model can schedule in one step. Chunked prefill is disabled for this model, so \
         longer prompts can never be scheduled. Increase `max_num_batched_tokens` or shorten \
         the prompt."
    )]
    PromptExceedsUnchunkedLimit {
        prompt_len: usize,
        max_prompt_len: u32,
    },
    /// The request uses a field this implementation does not carry out.
    #[error("`{field}` is not supported")]
    Unsupported { field: &'static str },
    /// The request cannot be taken now; retrying later may succeed.
    #[error("unavailable: {message}")]
    Unavailable { message: String },
    /// A remote implementation rejected the request as invalid.
    #[error("{message}")]
    RemoteInvalidRequest { message: String },
    /// A remote implementation failed the request.
    #[error("remote error: {message}")]
    Remote { message: String },
}
