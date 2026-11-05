// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use llm_multimodal::MediaContentPart;
use serde_json::{Map, Value};
use vllm_engine_core_client::protocol::multimodal::MmFeatures;

/// Media preparation stage supplied by a frontend handler.
#[derive(Debug)]
pub enum MultimodalInput {
    /// Raw media to fetch and preprocess, expanding prompt placeholders as needed.
    Raw(Vec<MediaContentPart>),
    /// Preprocessed features to validate against storage, batching, model, and prompt constraints.
    Preprocessed(MmFeatures),
    /// Features already placed in the prompt plus raw items still sitting at
    /// their single placeholder marker. Raw items expand in place; feature
    /// offsets after them shift by the growth.
    Mixed {
        raw: Vec<PlacedMedia>,
        features: MmFeatures,
    },
}

/// One raw media item whose placeholder marker is at `marker_offset`.
#[derive(Debug, Clone)]
pub struct PlacedMedia {
    pub part: MediaContentPart,
    pub marker_offset: usize,
}

impl MultimodalInput {
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Raw(parts) => parts.is_empty(),
            Self::Preprocessed(features) => features.is_empty(),
            Self::Mixed { raw, features } => raw.is_empty() && features.is_empty(),
        }
    }
}

impl From<Vec<MediaContentPart>> for MultimodalInput {
    fn from(parts: Vec<MediaContentPart>) -> Self {
        Self::Raw(parts)
    }
}

/// Per-request preprocessing options, as in the Python frontend.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MediaOptions {
    /// Overrides merged into the image and video preprocessor configs.
    pub processor_kwargs: Map<String, Value>,
    /// Media loading options keyed by modality; only `video.num_frames` and
    /// `video.fps` are understood.
    pub media_io_kwargs: Map<String, Value>,
}

impl MediaOptions {
    pub fn is_empty(&self) -> bool {
        self.processor_kwargs.is_empty() && self.media_io_kwargs.is_empty()
    }
}
