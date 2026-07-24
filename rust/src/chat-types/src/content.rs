// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde::{Deserialize, Serialize};

/// Detail level requested for an OpenAI-style image input.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageDetail {
    /// Let the model-specific multimodal processor select the detail level.
    #[default]
    Auto,
    /// Request low-detail image processing.
    Low,
    /// Request high-detail image processing.
    High,
}

/// One chat content part in OpenAI-style block format.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatContentPart {
    /// One plain-text content block.
    Text {
        /// Plain-text content.
        text: String,
    },
    /// One image URL or data URL content block.
    ImageUrl {
        /// Image URL or data URL.
        image_url: String,
        /// Requested image detail level.
        detail: Option<ImageDetail>,
        /// Optional caller-provided media identifier.
        uuid: Option<String>,
    },
    /// One video URL or data URL content block.
    VideoUrl {
        /// Video URL or data URL.
        video_url: String,
        /// Optional caller-provided media identifier.
        uuid: Option<String>,
    },
    /// One `input_audio` content block carrying base64-encoded audio bytes.
    InputAudio {
        /// Base64-encoded audio bytes.
        data: String,
        /// Optional audio format such as `wav` or `mp3`.
        format: Option<String>,
        /// Optional caller-provided media identifier.
        uuid: Option<String>,
    },
    /// One audio URL or data URL content block.
    AudioUrl {
        /// Audio URL or data URL.
        audio_url: String,
        /// Optional caller-provided media identifier.
        uuid: Option<String>,
    },
}

impl ChatContentPart {
    /// Construct one text content part with plain string content.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// Construct one image URL content part with the given URL string.
    pub fn image_url(image_url: impl Into<String>) -> Self {
        Self::ImageUrl {
            image_url: image_url.into(),
            detail: None,
            uuid: None,
        }
    }

    /// Construct one video URL content part with the given URL string.
    pub fn video_url(video_url: impl Into<String>) -> Self {
        Self::VideoUrl {
            video_url: video_url.into(),
            uuid: None,
        }
    }

    /// Construct one base64-encoded input-audio content part.
    pub fn input_audio(data: impl Into<String>, format: Option<String>) -> Self {
        Self::InputAudio {
            data: data.into(),
            format,
            uuid: None,
        }
    }

    /// Construct one audio URL content part with the given URL string.
    pub fn audio_url(audio_url: impl Into<String>) -> Self {
        Self::AudioUrl {
            audio_url: audio_url.into(),
            uuid: None,
        }
    }

    /// Return the text content of this part.
    ///
    /// Returns the static content-part type for multimodal content.
    pub fn as_text(&self) -> Result<&str, &'static str> {
        match self {
            Self::Text { text } => Ok(text),
            Self::ImageUrl { .. } => Err("image_url"),
            Self::VideoUrl { .. } => Err("video_url"),
            Self::InputAudio { .. } => Err("input_audio"),
            Self::AudioUrl { .. } => Err("audio_url"),
        }
    }

    /// Return whether this part is a text block with empty content.
    fn is_empty_text(&self) -> bool {
        matches!(self, Self::Text { text } if text.is_empty())
    }

    /// Return whether this part contains any multimodal content.
    fn is_multimodal(&self) -> bool {
        match self {
            Self::Text { .. } => false,
            Self::ImageUrl { .. }
            | Self::VideoUrl { .. }
            | Self::InputAudio { .. }
            | Self::AudioUrl { .. } => true,
        }
    }
}

/// Chat content represented as a string or OpenAI-style content parts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatContent {
    /// Simple text content.
    Text(String),
    /// OpenAI-style content parts.
    Parts(Vec<ChatContentPart>),
}

impl ChatContent {
    /// Flatten text parts into one string without adding separators.
    ///
    /// Returns the static content-part type when the content is multimodal.
    pub fn try_flatten_to_text(&self) -> Result<String, &'static str> {
        Ok(match self {
            Self::Text(text) => text.clone(),
            Self::Parts(parts) => parts
                .iter()
                .map(ChatContentPart::as_text)
                .collect::<Result<Vec<_>, _>>()?
                .concat(),
        })
    }

    /// Return whether the content has no text or only empty text blocks.
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Text(text) => text.is_empty(),
            Self::Parts(parts) => parts.iter().all(ChatContentPart::is_empty_text),
        }
    }

    /// Return whether this content contains any multimodal parts.
    pub fn has_multimodal(&self) -> bool {
        match self {
            Self::Text(_) => false,
            Self::Parts(parts) => parts.iter().any(ChatContentPart::is_multimodal),
        }
    }
}

impl From<String> for ChatContent {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for ChatContent {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

impl From<Vec<ChatContentPart>> for ChatContent {
    fn from(value: Vec<ChatContentPart>) -> Self {
        Self::Parts(value)
    }
}
