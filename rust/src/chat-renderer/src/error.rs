// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use thiserror::Error;

type BoxedError = Box<dyn std::error::Error + Send + Sync>;

/// Error returned while constructing or applying a chat renderer.
#[derive(Debug, Error)]
pub enum Error {
    /// Rendering requires at least one chat message.
    #[error("chat request must contain at least one message")]
    EmptyMessages,
    /// Continuation mode requires the final message to be an assistant turn.
    #[error("cannot continue the final message when the last message is not from the assistant")]
    ContinueFinalAssistantWithoutFinalAssistant,
    /// The selected renderer requires a chat template and none was configured.
    #[error("chat template is required but none was configured")]
    MissingChatTemplate,
    /// A chat template could not be compiled or applied.
    #[error("chat template error: {0}")]
    ChatTemplate(String),
    /// The selected renderer cannot represent the given multimodal part.
    #[error("unsupported multimodal content: {0}")]
    UnsupportedMultimodalContent(&'static str),
    /// The process-wide GPT-OSS Harmony encoding could not be initialized.
    #[error("failed to initialize the Harmony encoding")]
    HarmonyEncoding {
        /// Underlying Harmony initialization failure.
        #[source]
        error: BoxedError,
    },
    /// Tokenizer construction or encoding failed.
    #[error(transparent)]
    Tokenizer(#[from] vllm_tokenizer::TokenizerError),
}

impl Error {
    /// Whether this error should be reported as invalid request input when
    /// raised while rendering.
    pub fn is_request_validation_error(&self) -> bool {
        matches!(
            self,
            Self::EmptyMessages
                | Self::ContinueFinalAssistantWithoutFinalAssistant
                | Self::MissingChatTemplate
                | Self::ChatTemplate(_)
                | Self::UnsupportedMultimodalContent(_)
        )
    }
}

/// Result returned by chat renderer operations.
pub type Result<T> = std::result::Result<T, Error>;
