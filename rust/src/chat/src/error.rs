// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use thiserror::Error;
use thiserror_ext::{AsReport as _, Construct, Macro};
pub use vllm_text::ErrorKind;

use crate::RendererSelection;

pub(crate) type BoxedError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Debug, Error, Construct, Macro)]
#[thiserror_ext(macro(path = "crate::error"))]
pub enum Error {
    #[error("chat request must contain at least one message")]
    EmptyMessages,
    #[error("{renderer} renderer does not support continue_final_message")]
    UnsupportedContinueFinalMessage { renderer: &'static str },
    #[error("chat template is required but none was configured")]
    MissingChatTemplate,
    /// The chat template could not be loaded or compiled, or this renderer
    /// failed; request-caused refusals use the typed variants below.
    #[error("chat template error: {0}")]
    ChatTemplate(String),
    /// The chat template failed while rendering this request, including its
    /// `raise_exception`. The request's values usually cause it (a template
    /// filter given a list instead of a string), and a template that is broken
    /// for every request fails every request anyway, so it is the request's
    /// error, as in Python.
    #[error("{message}")]
    ChatTemplateRender { message: String },
    /// The request's own `chat_template` failed to compile.
    #[error("{message}")]
    InvalidRequestChatTemplate { message: String },
    /// `continue_final_message` without a final assistant message to
    /// continue, or the template dropped the final message.
    #[error("{message}")]
    InvalidContinueFinalMessage { message: String },
    /// The messages break a rule of the model's prompt format, such as a tool
    /// result without a matching tool call.
    #[error("{message}")]
    InvalidMessages { message: String },
    #[error("{message}")]
    InvalidChatTemplateKwargs { message: String },
    #[error("{0}")]
    InvalidReasoningEffort(String),
    #[error("{message}")]
    InvalidReasoningControl { message: String },
    #[error("chat role `{role}` is not supported by this chat renderer")]
    UnsupportedChatRole { role: String },
    #[error(
        "Chat template is passed with request, but --trust-request-chat-template is not set. \
         Refused request with untrusted chat template."
    )]
    UntrustedRequestChatTemplate,
    #[error("chat_template is not supported by this chat renderer")]
    UnsupportedRequestChatTemplate,
    #[error(
        "--chat-template is not supported by the `{renderer}` renderer; use --tokenizer-mode hf to render a Jinja chat template"
    )]
    UnsupportedChatTemplateOption { renderer: RendererSelection },
    #[error("multimodal input is not supported by this chat renderer")]
    UnsupportedMultimodalRenderer,
    #[error("unsupported multimodal content: {0}")]
    UnsupportedMultimodalContent(&'static str),
    #[error("`{0}` content parts are not supported by this chat renderer")]
    UnsupportedContentPart(&'static str),
    #[error("unsupported input_audio format `{format}`; expected `wav` or `mp3`")]
    UnsupportedInputAudioFormat { format: String },
    #[error("`{modality}` input is not supported by this model")]
    UnsupportedModality { modality: String },
    #[error(
        "`{modality}` input is not supported by the Rust frontend for this model; set \
         VLLM_USE_RUST_FRONTEND=0 to serve it with the Python frontend, or pass \
         --limit-mm-per-prompt '{{\"{modality}\": 0}}' to disable it"
    )]
    RustFrontendUnsupportedModality { modality: &'static str },
    #[error("At most {limit} {modality}(s) may be provided in one prompt.")]
    MmLimitExceeded { modality: String, limit: usize },
    #[error("invalid inline multimodal features: {message}")]
    InvalidPreprocessedMultimodal { message: String },
    #[error(
        "failed to prepare media after the last preprocessed feature range, ending at offset {offset}"
    )]
    MediaAfterPreprocessed {
        offset: usize,
        #[source]
        source: Box<Error>,
    },
    #[error(
        "the Rust frontend cannot serve multimodal input for model `{model_id}` \
         (model_type `{model_type}`); pass --language-model-only to serve it as a text-only \
         model, or set VLLM_USE_RUST_FRONTEND=0 to use the Python frontend"
    )]
    MultimodalModelUnsupported {
        model_id: String,
        model_type: String,
    },
    #[error("multimodal preprocessing error: {0}")]
    Multimodal(#[message] String),
    #[error(transparent)]
    Media(#[from] llm_multimodal::MultiModalError),
    #[error(transparent)]
    MultimodalTransform(#[from] llm_multimodal::TransformError),
    #[error(transparent)]
    MultimodalSpec(#[from] llm_multimodal::registry::ModelRegistryError),
    #[error("{kind} parsing is not available for model `{model_id}`")]
    ParserUnavailableForModel {
        kind: &'static str,
        model_id: String,
    },
    #[error("{kind} parsing is disabled by frontend configuration")]
    ParserDisabled { kind: &'static str },
    #[error(
        "unified parsing requires the tool and reasoning selections to resolve to the same parser; resolved tool={tool}, reasoning={reasoning}"
    )]
    IncompatibleParserSelections { tool: String, reasoning: String },
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
    /// A parser that `auto` matched by model name but cannot build for the
    /// model. This refuses the start rather than serving without the parser:
    /// a managed engine gets the same selection, and would still wait for the
    /// end of reasoning that the frontend no longer parses before applying
    /// structured outputs.
    #[error(
        "{kind} parser `{name}` was selected automatically for model `{model_id}` but cannot be used with it; select a parser explicitly, or pass `{disable}` to disable {kind} parsing"
    )]
    UnusableAutoParser {
        kind: &'static str,
        name: String,
        model_id: String,
        disable: &'static str,
        #[source]
        error: Box<Error>,
    },
    #[error("failed to initialize request output parser")]
    OutputParserInitialization {
        #[source]
        error: BoxedError,
    },
    #[error(
        "gpt_oss uses native Harmony output parsing; generic {kind} parser override `{selection}` is not supported"
    )]
    HarmonyParserOverrideUnsupported {
        kind: &'static str,
        selection: String,
    },
    #[error("harmony output parsing failed")]
    HarmonyOutputParsing {
        #[source]
        error: BoxedError,
    },
    #[error("chat request stream `{request_id}` closed before terminal output")]
    StreamClosedBeforeTerminalOutput { request_id: String },
    #[error("tool call stream state is inconsistent: {message}")]
    ToolCallStreamInvariant { message: String },
    #[error("duplicate tool name `{name}`")]
    DuplicateToolName { name: String },
    #[error("tool_choice requires at least one available tool")]
    ToolChoiceRequiresTools,
    #[error("tool_choice function `{name}` was not found in the available tools")]
    ToolChoiceFunctionNotFound { name: String },
    #[error("tool_reference `{name}` was not found in the available tools")]
    ToolReferenceNotFound { name: String },
    #[error("at least one tool must have defer_loading=false; all tools cannot be deferred")]
    AllToolsDeferred,
    #[error("failed to build output grammar")]
    OutputGrammar {
        #[source]
        error: BoxedError,
    },
    #[error(transparent)]
    Text(#[from] vllm_text::Error),
    #[error(transparent)]
    Tokenizer(#[from] vllm_tokenizer::TokenizerError),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Classify this error as a client request-validation failure (with its
    /// canonical OpenAI `param`, if any) or an internal server failure.
    pub fn kind(&self) -> ErrorKind {
        match self {
            Self::Text(error) => error.kind(),
            Self::MediaAfterPreprocessed { source, .. } => source.kind(),
            Self::EmptyMessages
            | Self::UnsupportedChatRole { .. }
            | Self::ToolReferenceNotFound { .. }
            | Self::InvalidMessages { .. } => ErrorKind::invalid_param("messages"),
            Self::UnsupportedContinueFinalMessage { .. }
            | Self::InvalidContinueFinalMessage { .. } => {
                ErrorKind::invalid_param("continue_final_message")
            }
            Self::InvalidReasoningEffort(_) => ErrorKind::invalid_param("reasoning_effort"),
            Self::DuplicateToolName { .. } | Self::AllToolsDeferred => {
                ErrorKind::invalid_param("tools")
            }
            Self::ToolChoiceRequiresTools | Self::ToolChoiceFunctionNotFound { .. } => {
                ErrorKind::invalid_param("tool_choice")
            }
            Self::UntrustedRequestChatTemplate
            | Self::UnsupportedRequestChatTemplate
            | Self::InvalidRequestChatTemplate { .. } => ErrorKind::invalid_param("chat_template"),
            Self::InvalidChatTemplateKwargs { .. } => {
                ErrorKind::invalid_param("chat_template_kwargs")
            }
            Self::ChatTemplateRender { .. }
            | Self::InvalidReasoningControl { .. }
            | Self::ParserUnavailableForModel { .. }
            | Self::ParserDisabled { .. }
            | Self::UnsupportedMultimodalRenderer
            | Self::UnsupportedMultimodalContent(_)
            | Self::UnsupportedContentPart(_)
            | Self::UnsupportedInputAudioFormat { .. }
            | Self::UnsupportedModality { .. }
            | Self::RustFrontendUnsupportedModality { .. }
            | Self::MmLimitExceeded { .. }
            | Self::InvalidPreprocessedMultimodal { .. }
            | Self::Tokenizer(vllm_tokenizer::TokenizerError::Unavailable) => {
                ErrorKind::invalid_request()
            }
            Self::OutputGrammar { error } => match error.downcast_ref() {
                Some(vllm_parser::output_grammar::OutputGrammarError::UnsupportedRequest(_)) => {
                    ErrorKind::invalid_request()
                }
                _ => ErrorKind::Internal,
            },
            Self::Media(error) => media_error_kind(error),
            Self::MultimodalTransform(error) => transform_error_kind(error),
            Self::MultimodalSpec(error) => spec_error_kind(error),
            Self::MissingChatTemplate
            | Self::ChatTemplate(_)
            | Self::UnsupportedChatTemplateOption { .. }
            | Self::MultimodalModelUnsupported { .. }
            | Self::Multimodal(_)
            | Self::IncompatibleParserSelections { .. }
            | Self::ParserUnavailableByName { .. }
            | Self::ParserInitialization { .. }
            | Self::UnusableAutoParser { .. }
            | Self::OutputParserInitialization { .. }
            | Self::HarmonyParserOverrideUnsupported { .. }
            | Self::HarmonyOutputParsing { .. }
            | Self::StreamClosedBeforeTerminalOutput { .. }
            | Self::ToolCallStreamInvariant { .. }
            | Self::Tokenizer(vllm_tokenizer::TokenizerError::Backend { .. }) => {
                ErrorKind::Internal
            }
        }
    }

    /// Whether the request could not be taken now, so retrying later may
    /// succeed.
    pub fn is_unavailable(&self) -> bool {
        matches!(self, Self::Text(error) if error.is_unavailable())
    }
}

impl From<llm_multimodal::MediaConnectorError> for Error {
    fn from(error: llm_multimodal::MediaConnectorError) -> Self {
        Self::Media(error.into())
    }
}

/// Classify a media tracker error: media the request got wrong and a content
/// part it may not send are the request's. A media host that timed out or
/// failed, and a decoder that failed on this side, stay server errors, as in
/// Python; so do a failed tracker task and a broken tracker invariant.
fn media_error_kind(error: &llm_multimodal::MultiModalError) -> ErrorKind {
    use llm_multimodal::MultiModalError;
    match error {
        MultiModalError::Media(error) => match error.fault() {
            llm_multimodal::MediaFault::Client => ErrorKind::invalid_request(),
            llm_multimodal::MediaFault::Transient | llm_multimodal::MediaFault::Internal => {
                ErrorKind::Internal
            }
        },
        MultiModalError::UnsupportedContent(_) => ErrorKind::invalid_request(),
        MultiModalError::Join(_) | MultiModalError::Validation(_) => ErrorKind::Internal,
    }
}

/// Classify a processor error on already decoded media: only a payload over
/// its decoded-size cap is the request's. The shape errors describe this
/// process's preprocessing output or the checkpoint's processor config.
// TODO: `ShapeError` also carries refusals of the request's media geometry
// (Step3-VL collapsing a 100000x32 image to 3024x0), which Python answers
// with 400; classify those once their producers report them apart.
fn transform_error_kind(error: &llm_multimodal::TransformError) -> ErrorKind {
    use llm_multimodal::TransformError;
    match error {
        TransformError::DecodedTooLarge { .. } => ErrorKind::invalid_request(),
        TransformError::InvalidShape { .. }
        | TransformError::EmptyBatch
        | TransformError::InconsistentShapes
        | TransformError::ShapeError(_) => ErrorKind::Internal,
    }
}

/// Classify a model spec error: more items of a modality, or a modality, than
/// the model takes are the request's. The rest describe the model's files and
/// config, or this process's preprocessing output.
fn spec_error_kind(error: &llm_multimodal::registry::ModelRegistryError) -> ErrorKind {
    use llm_multimodal::registry::ModelRegistryError;
    match error {
        ModelRegistryError::ModalityLimitExceeded { .. }
        | ModelRegistryError::UnsupportedModality { .. } => ErrorKind::invalid_request(),
        ModelRegistryError::UnsupportedModel(_)
        | ModelRegistryError::TokenNotFound { .. }
        | ModelRegistryError::MissingConfigField { .. }
        | ModelRegistryError::InvalidPreprocessedField { .. }
        | ModelRegistryError::InvalidConfigField { .. }
        | ModelRegistryError::TextEncodingFailed { .. }
        | ModelRegistryError::InvalidPreprocessorConfig { .. }
        | ModelRegistryError::DuplicateModality { .. } => ErrorKind::Internal,
    }
}

/// Format the available-parser suffix used in user-facing error messages.
fn available_parser_hint(available_names: &[String]) -> String {
    if available_names.is_empty() {
        String::new()
    } else {
        format!(" (choose from: {})", available_names.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use expect_test::expect;
    use itertools::Itertools as _;
    use llm_multimodal::registry::ModelRegistryError;
    use llm_multimodal::{MediaConnectorError, Modality, MultiModalError, TransformError};

    use super::*;

    /// Only what the request got wrong is a validation error: its media, a
    /// content part it may not send, a payload over its decoded-size cap, too
    /// many items or a modality the model does not take. A media host that
    /// timed out, a decoder or preprocessing step that failed on this side,
    /// and the model's own files are not.
    #[test]
    fn multimodal_error_is_request_validation_error_only_when_the_request_is_at_fault() {
        let errors = [
            Error::from(MediaConnectorError::DataUrl("missing comma".to_string())),
            Error::from(MediaConnectorError::Timeout {
                url: "https://example.com/a.png".to_string(),
                budget: Duration::from_secs(5),
            }),
            Error::from(MediaConnectorError::DecodeIo(std::io::Error::other(
                "broken pipe",
            ))),
            Error::from(MultiModalError::UnsupportedContent("image_embeds")),
            Error::from(MultiModalError::Validation(
                "image slot refers to a slot that was never fetched".to_string(),
            )),
            Error::from(TransformError::DecodedTooLarge {
                bytes: 2048,
                limit: 1024,
                knob: "SMG_AUDIO_MAX_DECODED_BYTES",
            }),
            Error::from(TransformError::ShapeError(
                "invalid resize inputs: image 28x28, token patch 0x0".to_string(),
            )),
            Error::from(ModelRegistryError::ModalityLimitExceeded {
                spec: "test",
                modality: Modality::Audio,
                limit: 1,
                requested: 2,
            }),
            Error::from(ModelRegistryError::UnsupportedModality {
                spec: "test",
                modality: Modality::Video,
            }),
            Error::from(ModelRegistryError::MissingConfigField {
                field: "image_token_id".to_string(),
            }),
        ];
        let rendered = errors
            .into_iter()
            .map(|error| format!("{}: {error}", error.kind().is_invalid_request()))
            .join("\n");
        expect![[r#"
            true: data URL parse error: missing comma
            false: media fetch of https://example.com/a.png timed out after 5s
            false: I/O error while decoding media: broken pipe
            true: unsupported content part: image_embeds
            false: tracker validation error: image slot refers to a slot that was never fetched
            true: decoded audio payload is 2048 bytes, exceeding SMG_AUDIO_MAX_DECODED_BYTES=1024
            false: Shape error: invalid resize inputs: image 28x28, token patch 0x0
            true: model spec test supports at most 1 audio inputs; got 2
            true: modality video is not supported by model spec test
            false: missing config field 'image_token_id'"#]]
        .assert_eq(&rendered);
    }
}
