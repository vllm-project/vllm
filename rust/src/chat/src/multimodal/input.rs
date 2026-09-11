// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use llm_multimodal::MediaContentPart;
use vllm_engine_core_client::protocol::multimodal::InlineMmFeatures;

/// Media preparation stage supplied by a frontend handler.
#[derive(Debug)]
pub enum MultimodalInput {
    /// Raw media to fetch and preprocess, expanding prompt placeholders as needed.
    Raw(Vec<MediaContentPart>),
    /// Inline features with validated storage and batching metadata.
    /// Media preparation checks model support, item limits, and prompt bounds.
    Preprocessed(InlineMmFeatures),
}

impl MultimodalInput {
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Raw(parts) => parts.is_empty(),
            Self::Preprocessed(features) => features.as_slice().is_empty(),
        }
    }
}

impl From<Vec<MediaContentPart>> for MultimodalInput {
    fn from(parts: Vec<MediaContentPart>) -> Self {
        Self::Raw(parts)
    }
}
