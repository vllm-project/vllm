use thiserror::Error;

type BoxedError = Box<dyn std::error::Error + Send + Sync>;

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
    #[error("{kind} parsing is not available for model `{model_id}`")]
    ParserUnavailableForModel {
        kind: &'static str,
        model_id: String,
    },
    #[error("{kind} parsing is disabled by frontend configuration")]
    ParserDisabled { kind: &'static str },
    #[error(
        "{kind} parser `{name}` is not registered{}",
        available_parser_hint(.available_names)
    )]
    ParserUnavailableByName {
        kind: &'static str,
        name: String,
        available_names: Vec<String>,
    },
    #[error("failed to initialize {kind} parser `{name}`")]
    ParserInitialization {
        kind: &'static str,
        name: String,
        #[source]
        error: BoxedError,
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

/// Format the available-parser suffix used in user-facing error messages.
fn available_parser_hint(available_names: &[String]) -> String {
    if available_names.is_empty() {
        String::new()
    } else {
        format!(" (choose from: {})", available_names.join(", "))
    }
}
