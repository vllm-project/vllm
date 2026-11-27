// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use thiserror::Error;
use vllm_engine_core_client::Error as EngineCoreError;
use vllm_llm::Error as LlmError;

pub use crate::lower::logprobs::LogprobsError;
pub use crate::lower::sampling::SamplingParamsError;
pub use crate::lower::token_ids::TokenIdsError;

#[derive(Debug, Error)]
pub enum Error {
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    /// No tokenizer is loaded (`--skip-tokenizer-init`), so the request's
    /// text cannot be encoded or decoded.
    #[error(
        "this request needs a tokenizer, which is not loaded when `--skip-tokenizer-init` is set"
    )]
    TokenizerUnavailable,
    #[error("text request `{request_id}` must contain at least one prompt token ID")]
    EmptyPromptTokenIds { request_id: String },
    #[error("text request `{request_id}` stop strings cannot be empty")]
    EmptyStopString { request_id: String },
    #[error("Too many stop strings: {count}. The max number is {max}.")]
    TooManyStopStrings { count: usize, max: usize },
    #[error(
        "text request `{request_id}` stop strings require detokenization; enable it or use stop \
         token IDs"
    )]
    StopStringsRequireDetokenize { request_id: String },
    #[error(
        "Parameter 'cache_salt' must be a non-empty string of at most 128 characters \
         and must not contain '@', '/', '\\', or NUL."
    )]
    InvalidCacheSalt,
    #[error(
        "This model's maximum context length is {max_model_len} tokens. However, your \
         prompt contains {prompt_len} input tokens, which exceeds this limit even without \
         any output tokens. Please reduce the length of the input prompt."
    )]
    PromptTooLong { max_model_len: u32, prompt_len: u32 },
    #[error(
        "This model's maximum context length is {max_model_len} tokens. However, your \
         prompt fills it with {max_model_len} input tokens, leaving no room for output \
         tokens. Please reduce the length of the input prompt."
    )]
    PromptFillsContext { max_model_len: u32 },
    #[error(transparent)]
    Logprobs(#[from] LogprobsError),
    #[error(transparent)]
    TokenIds(#[from] TokenIdsError),
    #[error(transparent)]
    SamplingParams(#[from] SamplingParamsError),
    #[error(
        "`min_tokens` must be less than or equal to `max_tokens`, \
         got min_tokens={min_tokens}, max_tokens={max_tokens}"
    )]
    MinTokensExceedsMaxTokens { min_tokens: u32, max_tokens: u32 },
    #[error("n must be 1 when using greedy sampling, got {n}.")]
    GreedyParallelSampling { n: u32 },
    #[error("`thinking_token_budget` must be a non-negative integer or -1 for unlimited.")]
    InvalidThinkingTokenBudget,
    #[error(
        "truncate_prompt_tokens={value} cannot be greater than max_model_len = \
         {max_model_len}. Please request a smaller truncation size."
    )]
    TruncatePromptTokensExceedsMaxModelLen { value: u64, max_model_len: u32 },
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

/// High-level fault classification of a request error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// Invalid user request parameters (HTTP 400 / gRPC `INVALID_ARGUMENT`),
    /// with the canonical offending parameter name when attributable to a
    /// single field.
    InvalidRequest { param: Option<&'static str> },
    /// Server-side or internal failure (HTTP 500 / gRPC `INTERNAL`).
    Internal,
}

impl ErrorKind {
    /// Construct an [`ErrorKind::InvalidRequest`] referencing `param`.
    pub const fn invalid_param(param: &'static str) -> Self {
        Self::InvalidRequest { param: Some(param) }
    }

    /// Construct an [`ErrorKind::InvalidRequest`] not tied to a single field.
    pub const fn invalid_request() -> Self {
        Self::InvalidRequest { param: None }
    }

    /// Whether this error represents invalid user request parameters.
    pub const fn is_invalid_request(self) -> bool {
        matches!(self, Self::InvalidRequest { .. })
    }
}

impl Error {
    /// Classify this error as a client request-validation failure (with its
    /// canonical OpenAI `param`, if any) or an internal server failure.
    pub fn kind(&self) -> ErrorKind {
        match self {
            Self::PromptTooLong { .. } | Self::PromptFillsContext { .. } => {
                ErrorKind::invalid_param("input_tokens")
            }
            Self::EmptyPromptTokenIds { .. } => ErrorKind::invalid_param("prompt"),
            Self::EmptyStopString { .. }
            | Self::TooManyStopStrings { .. }
            | Self::StopStringsRequireDetokenize { .. } => ErrorKind::invalid_param("stop"),
            Self::InvalidCacheSalt => ErrorKind::invalid_param("cache_salt"),
            Self::Logprobs(error) => ErrorKind::invalid_param(error.param()),
            Self::TokenIds(error) => ErrorKind::InvalidRequest {
                param: error.param(),
            },
            Self::SamplingParams(error) => ErrorKind::invalid_param(error.param()),
            Self::InvalidThinkingTokenBudget => ErrorKind::invalid_param("thinking_token_budget"),
            Self::TruncatePromptTokensExceedsMaxModelLen { .. }
            | Self::InvalidTruncatePromptTokens { .. }
            | Self::TruncateUnsupportedWithMultimodal => {
                ErrorKind::invalid_param("truncate_prompt_tokens")
            }
            Self::InvalidRepetitionDetection { .. } => {
                ErrorKind::invalid_param("repetition_detection")
            }
            Self::TokenizerUnavailable
            | Self::MinTokensExceedsMaxTokens { .. }
            | Self::GreedyParallelSampling { .. } => ErrorKind::invalid_request(),
            Self::Tokenizer(_) | Self::StreamClosedBeforeTerminalOutput { .. } => {
                ErrorKind::Internal
            }
            Self::Llm(error) => match error {
                // An empty tokenized prompt detected later, at request prepare
                // time, surfaces through the transparent Llm wrapper.
                LlmError::EmptyPromptTokenIds { .. } => ErrorKind::invalid_param("prompt"),
                LlmError::PromptLogprobsUnsupported { parameter } => {
                    ErrorKind::invalid_param(parameter)
                }
                LlmError::Unsupported { field } => ErrorKind::invalid_param(field),
                LlmError::MmEmbedsExceedEncoderCache { .. }
                | LlmError::InvalidMultimodalInput { .. }
                | LlmError::PromptExceedsUnchunkedLimit { .. }
                | LlmError::RemoteInvalidRequest { .. } => ErrorKind::invalid_request(),
                LlmError::EngineCoreClient(error) => engine_core_error_kind(error),
                LlmError::SamplingMaskTokenCountMismatch { .. }
                | LlmError::AuxOutputRows(_)
                | LlmError::Unavailable { .. }
                | LlmError::QueueFull
                | LlmError::PrefillBacklogFull
                | LlmError::Remote { .. } => ErrorKind::Internal,
            },
            Self::EngineCore(error) => engine_core_error_kind(error),
        }
    }

    /// Whether the request could not be taken now, so retrying later may
    /// succeed.
    pub fn is_unavailable(&self) -> bool {
        matches!(
            self,
            Self::Llm(
                LlmError::Unavailable { .. } | LlmError::QueueFull | LlmError::PrefillBacklogFull
            )
        )
    }
}

fn engine_core_error_kind(error: &EngineCoreError) -> ErrorKind {
    match error {
        EngineCoreError::InvalidDataParallelRank { .. } => {
            ErrorKind::invalid_param("data_parallel_rank")
        }
        EngineCoreError::InvalidStructuredOutputsParams { .. } => {
            ErrorKind::invalid_param("structured_outputs")
        }
        EngineCoreError::Shared(inner) => engine_core_error_kind(inner),
        _ => ErrorKind::Internal,
    }
}

impl From<vllm_tokenizer::TokenizerError> for Error {
    fn from(error: vllm_tokenizer::TokenizerError) -> Self {
        match error {
            vllm_tokenizer::TokenizerError::Backend { message } => Self::Tokenizer(message),
            vllm_tokenizer::TokenizerError::Unavailable => Self::TokenizerUnavailable,
        }
    }
}
