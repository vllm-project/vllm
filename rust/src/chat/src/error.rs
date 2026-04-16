use thiserror::Error;

use crate::parser::available_parser_hint;
use crate::{ReasoningError, ToolParserError};

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
    #[error("chat template error: {0}")]
    ChatTemplate(String),
    #[error("tool parsing is not available for model `{model_id}`")]
    ToolParserUnavailableForModel { model_id: String },
    #[error(
        "tool call parser `{name}` is not registered{}",
        available_parser_hint(.available_names)
    )]
    ToolParserUnavailableByName {
        name: String,
        available_names: Vec<String>,
    },
    #[error(
        "reasoning parser `{name}` is not registered{}",
        available_parser_hint(.available_names)
    )]
    ReasoningParserUnavailableByName {
        name: String,
        available_names: Vec<String>,
    },
    #[error("failed to initialize tool parser `{name}`")]
    ToolParserInitialization {
        name: String,
        #[source]
        error: ToolParserError,
    },
    #[error("failed to initialize reasoning parser `{name}`")]
    ReasoningParserInitialization {
        name: String,
        #[source]
        error: ReasoningError,
    },
    #[error(
        "this model's maximum context length is {max_model_len} tokens, \
         but the prompt contains {prompt_len} input tokens"
    )]
    PromptTooLong { max_model_len: u32, prompt_len: u32 },
    #[error("chat request stream `{request_id}` closed before terminal output")]
    StreamClosedBeforeTerminalOutput { request_id: String },
    #[error(transparent)]
    Text(#[from] vllm_text::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
