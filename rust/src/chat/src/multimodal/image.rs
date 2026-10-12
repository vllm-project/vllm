// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Image-modality preparation: batch preprocessing and per-item feature
//! build.

use std::sync::Arc;

use llm_multimodal::{ImageFrame, Modality, PreprocessedEncoderInputs, VisionPreprocessingContext};
use vllm_engine_core_client::protocol::dtype::ModelDtype;

use super::timing::MM_STAGE_TARGET;
use super::{MultimodalModelInfo, PreparedMedia, VisionModalitySupport, item};
use crate::error::{Error, Result, bail_multimodal, multimodal};

/// Forward-kwargs name of the primary image encoder input.
pub(super) const IMAGE_PRIMARY_KEY: &str = "pixel_values";

impl MultimodalModelInfo {
    /// Preprocess all fetched image frames as one batch and build per-item
    /// features.
    #[tracing::instrument(
        name = "mm_stage",
        target = MM_STAGE_TARGET,
        skip_all,
        fields(stage = "preprocess_image")
    )]
    pub(super) async fn prepare_images(
        &self,
        frames: Vec<Arc<ImageFrame>>,
        uuids: Vec<Option<String>>,
        model_dtype: ModelDtype,
        preprocessing_context: VisionPreprocessingContext,
    ) -> Result<PreparedMedia> {
        let support = self.image.as_ref().ok_or_else(|| Error::UnsupportedModality {
            modality: Modality::Image.to_string(),
        })?;
        let preprocessed = self.preprocess_images(support, &frames, preprocessing_context).await?;
        let replacements = support.spec.prompt_replacements_for(&self.context, &preprocessed)?;
        if replacements.len() != frames.len() {
            bail_multimodal!(
                "number of image prompt replacements {} does not match number of images {}",
                replacements.len(),
                frames.len()
            );
        }
        let hashes = frames.iter().map(|frame| frame.hash.clone()).collect();
        let items =
            item::build_batched_items(&support.spec, preprocessed, hashes, uuids, model_dtype)?;

        Ok(PreparedMedia {
            modality: Modality::Image,
            placeholder: support.placeholder.clone(),
            replacements,
            items,
        })
    }

    /// Preprocess fetched image frames with the model's resolved vision
    /// processor.
    ///
    /// The processor work is CPU-heavy relative to request wiring, so it runs
    /// in a blocking task and returns owned tensors ready for wire
    /// conversion.
    async fn preprocess_images(
        &self,
        support: &VisionModalitySupport,
        image_frames: &[Arc<ImageFrame>],
        preprocessing_context: VisionPreprocessingContext,
    ) -> Result<PreprocessedEncoderInputs> {
        let processor = Arc::clone(&support.processor);
        let images = image_frames.iter().map(|frame| frame.data().clone()).collect::<Vec<_>>();

        // TODO: is it still necessary given that we've already in a dedicated runtime?
        tokio::task::spawn_blocking(move || {
            Ok(processor.preprocess_with_context(&images, &preprocessing_context)?)
        })
        .await
        .map_err(|error| multimodal!("image preprocessing task failed: {error}"))?
    }
}
