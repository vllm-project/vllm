// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use llm_multimodal::MediaContentPart;
use vllm_engine_core_client::protocol::multimodal::MmFeatures;

/// Media supplied by a frontend handler for an already-tokenized prompt.
#[derive(Debug, Default)]
pub struct MultimodalInput {
    /// Preprocessed features, whose placeholders are already expanded in the
    /// prompt; validated against storage, batching, model, and prompt
    /// constraints.
    pub preprocessed: MmFeatures,
    /// Raw media to fetch and preprocess. Their placeholders are expanded only
    /// in the prompt tokens after the last preprocessed range, which are left
    /// as they are.
    pub raw: Vec<MediaContentPart>,
}

impl MultimodalInput {
    pub fn is_empty(&self) -> bool {
        self.preprocessed.is_empty() && self.raw.is_empty()
    }
}

impl From<Vec<MediaContentPart>> for MultimodalInput {
    fn from(raw: Vec<MediaContentPart>) -> Self {
        Self {
            raw,
            ..Default::default()
        }
    }
}

impl From<MmFeatures> for MultimodalInput {
    fn from(preprocessed: MmFeatures) -> Self {
        Self {
            preprocessed,
            ..Default::default()
        }
    }
}
