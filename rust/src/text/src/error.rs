use thiserror::Error;
use vllm_engine_core_client::Error as EngineCoreError;
use vllm_llm::Error as LlmError;

pub use crate::lower::logprobs::LogprobsError;
pub use crate::lower::token_ids::TokenIdsError;

#[derive(Debug, Error)]
pub enum Error {
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("text request `{request_id}` must contain at least one prompt token ID")]
    EmptyPromptTokenIds { request_id: String },
    #[error(
        "this model's maximum context length is {max_model_len} tokens, \
         but the prompt contains {prompt_len} input tokens"
    )]
    PromptTooLong { max_model_len: u32, prompt_len: u32 },
    #[error(transparent)]
    Logprobs(#[from] LogprobsError),
    #[error(transparent)]
    TokenIds(#[from] TokenIdsError),
    #[error(
        "`min_tokens` must be less than or equal to `max_tokens`, \
         got min_tokens={min_tokens}, max_tokens={max_tokens}"
    )]
    MinTokensExceedsMaxTokens { min_tokens: u32, max_tokens: u32 },
    #[error("`thinking_token_budget` must be a non-negative integer or -1 for unlimited.")]
    InvalidThinkingTokenBudget,
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
            Self::PromptTooLong { .. }
            | Self::EmptyPromptTokenIds { .. }
            | Self::Logprobs(_)
            | Self::TokenIds(_)
            | Self::MinTokensExceedsMaxTokens { .. }
            | Self::InvalidThinkingTokenBudget
            | Self::InvalidRepetitionDetection { .. }
            // An empty tokenized prompt detected later, at request prepare
            // time, surfaces through the transparent Llm wrapper.
            | Self::Llm(LlmError::EmptyPromptTokenIds { .. }) => true,
            _ => false,
        }
    }
}

impl From<vllm_tokenizer::TokenizerError> for Error {
    fn from(error: vllm_tokenizer::TokenizerError) -> Self {
        Self::Tokenizer(error.0)
    }
}
