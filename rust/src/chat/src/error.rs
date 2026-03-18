use thiserror::Error;
use vllm_llm::Error as LlmError;

#[derive(Debug, Error)]
pub enum Error {
    #[error("chat request must contain at least one message")]
    EmptyMessages,
    #[error(
        "cannot enable both `add_generation_prompt` and `continue_final_message` at the same time"
    )]
    ConflictingGenerationPromptMode,
    #[error("chat template is required but none was configured")]
    MissingChatTemplate,
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("tool parsing requires a backend model ID")]
    ToolParserRequiresModelId,
    #[error("tool parsing is not available for model `{model_id}`")]
    ToolParserUnavailableForModel { model_id: String },
    #[error("chat request stream `{request_id}` closed before terminal output")]
    StreamClosedBeforeTerminalOutput { request_id: String },
    #[error("llm request failed: {0}")]
    Llm(#[from] LlmError),
}

pub type Result<T> = std::result::Result<T, Error>;
