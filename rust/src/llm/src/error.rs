use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

/// Public error type for the Rust `llm` facade.
#[derive(Debug, Error)]
pub enum Error {
    #[error("generate request `{request_id}` has an empty prompt_token_ids")]
    EmptyPromptTokenIds { request_id: String },
    #[error("generate request `{request_id}` only supports sampling_params.n == 1, got {n}")]
    UnsupportedSamplingCount { request_id: String, n: u32 },
    #[error(transparent)]
    EngineCoreClient(#[from] vllm_engine_core_client::Error),
}
