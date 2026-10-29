// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use llm_multimodal::MediaContentPart;
use vllm_engine_core_client::protocol::multimodal::MmFeatures;

/// Media preparation stage supplied by a frontend handler.
#[derive(Debug)]
pub enum MultimodalInput {
    /// Raw media to fetch and preprocess, expanding prompt placeholders as needed.
    Raw(Vec<MediaPart>),
    /// Preprocessed features to validate against storage, batching, model, and prompt constraints.
    Preprocessed(MmFeatures),
}

impl MultimodalInput {
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Raw(parts) => parts.is_empty(),
            Self::Preprocessed(features) => features.is_empty(),
        }
    }
}

impl From<Vec<MediaContentPart>> for MultimodalInput {
    fn from(parts: Vec<MediaContentPart>) -> Self {
        Self::Raw(parts.into_iter().map(MediaPart::Content).collect())
    }
}

/// One raw media input, in request order.
#[derive(Debug)]
pub enum MediaPart {
    /// Media that the `llm-multimodal` connector fetches and decodes.
    Content(MediaContentPart),
}

impl From<MediaContentPart> for MediaPart {
    fn from(part: MediaContentPart) -> Self {
        Self::Content(part)
    }
}
