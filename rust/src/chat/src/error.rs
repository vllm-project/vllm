use thiserror::Error;
use vllm_llm::Error as LlmError;

use crate::request::ChatTemplateContentFormat;

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
    #[error("chat template content format `{0}` is not supported yet")]
    UnsupportedChatTemplateContentFormat(ChatTemplateContentFormat),
    #[error("rendered token prompts are not supported by this chat renderer yet")]
    UnsupportedRenderedTokens,
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("llm request failed: {0}")]
    Llm(#[from] LlmError),
}

pub type Result<T> = std::result::Result<T, Error>;
