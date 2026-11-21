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
    /// Preprocess all fetched image frames in one call and build per-item
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
        // A processor may split the images into consecutive batches (one per
        // image when their tensors cannot share a shape); items keep order.
        let batches = self.preprocess_images(support, &frames, preprocessing_context).await?;
        let mut replacements = Vec::with_capacity(frames.len());
        let mut items = Vec::with_capacity(frames.len());
        let mut hashes = frames.iter().map(|frame| frame.hash.clone());
        let mut uuids = uuids.into_iter();
        for preprocessed in batches {
            let batch = support.spec.prompt_replacements_for(&self.context, &preprocessed)?;
            let len = batch.len();
            replacements.extend(batch);
            items.extend(item::build_batched_items(
                &support.spec,
                preprocessed,
                hashes.by_ref().take(len).collect(),
                uuids.by_ref().take(len).collect(),
                model_dtype,
            )?);
        }
        if replacements.len() != frames.len() {
            bail_multimodal!(
                "number of image prompt replacements {} does not match number of images {}",
                replacements.len(),
                frames.len()
            );
        }

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
    ) -> Result<Vec<PreprocessedEncoderInputs>> {
        let processor = Arc::clone(&support.processor);
        let images = image_frames
            .iter()
            .map(|frame| frame.image().cloned())
            .collect::<std::result::Result<Vec<_>, _>>()?;

        // TODO: is it still necessary given that we've already in a dedicated runtime?
        tokio::task::spawn_blocking(move || {
            let images = images.iter().collect::<Vec<_>>();
            Ok(processor.preprocess_items_with_context(&images, &preprocessing_context)?)
        })
        .await
        .map_err(|error| multimodal!("image preprocessing task failed: {error}"))?
    }
}
