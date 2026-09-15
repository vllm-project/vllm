// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use thiserror::Error;
use vllm_engine_core_client::Error as EngineCoreError;
use vllm_llm::Error as LlmError;

use crate::embedding::EmbeddingError;
pub use crate::lower::logprobs::LogprobsError;
pub use crate::lower::sampling::SamplingParamsError;
pub use crate::lower::token_ids::TokenIdsError;

#[derive(Debug, Error)]
pub enum Error {
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("text request `{request_id}` must contain at least one prompt token ID")]
    EmptyPromptTokenIds { request_id: String },
    #[error("text request `{request_id}` stop strings cannot be empty")]
    EmptyStopString { request_id: String },
    #[error(
        "this model's maximum context length is {max_model_len} tokens, \
         but the prompt contains {prompt_len} input tokens"
    )]
    PromptTooLong { max_model_len: u32, prompt_len: u32 },
    #[error(transparent)]
    Logprobs(#[from] LogprobsError),
    #[error(transparent)]
    TokenIds(#[from] TokenIdsError),
    #[error(transparent)]
    SamplingParams(#[from] SamplingParamsError),
    #[error(transparent)]
    Embedding(#[from] EmbeddingError),
    #[error(
        "`min_tokens` must be less than or equal to `max_tokens`, \
         got min_tokens={min_tokens}, max_tokens={max_tokens}"
    )]
    MinTokensExceedsMaxTokens { min_tokens: u32, max_tokens: u32 },
    #[error("`thinking_token_budget` must be a non-negative integer or -1 for unlimited.")]
    InvalidThinkingTokenBudget,
    #[error("truncate_prompt_tokens={value} exceeds the available input budget of {budget} tokens")]
    TruncatePromptTokensExceedsBudget { value: u64, budget: u32 },
    #[error("invalid truncate_prompt_tokens={value}; must be >= -1")]
    InvalidTruncatePromptTokens { value: i64 },
    #[error("truncate_prompt_tokens is not supported for multimodal requests")]
    TruncateUnsupportedWithMultimodal,
    #[error("invalid repetition detection params: {message}")]
    InvalidRepetitionDetection { message: String },
    #[error("text request stream `{request_id}` closed before terminal output")]
    StreamClosedBeforeTerminalOutput { request_id: String },
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error(transparent)]
    EngineCore(#[from] EngineCoreError),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Whether this error represents invalid user request parameters.
    pub fn is_request_validation_error(&self) -> bool {
        match self {
            Self::Embedding(error) => error.is_request_validation_error(),
            Self::PromptTooLong { .. }
            | Self::EmptyPromptTokenIds { .. }
            | Self::EmptyStopString { .. }
            | Self::Logprobs(_)
            | Self::TokenIds(_)
            | Self::SamplingParams(_)
            | Self::MinTokensExceedsMaxTokens { .. }
            | Self::InvalidThinkingTokenBudget
            | Self::TruncatePromptTokensExceedsBudget { .. }
            | Self::InvalidTruncatePromptTokens { .. }
            | Self::TruncateUnsupportedWithMultimodal
            | Self::InvalidRepetitionDetection { .. }
            // An empty tokenized prompt detected later, at request prepare
            // time, surfaces through the transparent Llm wrapper.
            | Self::Llm(
                LlmError::EmptyPromptTokenIds { .. } | LlmError::UnsupportedTask { .. },
            )
            | Self::Llm(LlmError::EngineCoreClient(
                EngineCoreError::InvalidDataParallelRank { .. },
            )) => true,
            _ => false,
        }
    }
}

impl From<vllm_tokenizer::TokenizerError> for Error {
    fn from(error: vllm_tokenizer::TokenizerError) -> Self {
        Self::Tokenizer(error.0)
    }
}
