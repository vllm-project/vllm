// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Image-modality preparation: batch preprocessing and per-item feature
//! build.

use std::sync::Arc;

use image::DynamicImage;
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
        let limit = std::env::var("VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(715_827_880);
        let images = collect_images(image_frames, limit)?;

        // TODO: is it still necessary given that we've already in a dedicated runtime?
        tokio::task::spawn_blocking(move || {
            Ok(processor.preprocess_with_context(&images, &preprocessing_context)?)
        })
        .await
        .map_err(|error| multimodal!("image preprocessing task failed: {error}"))?
    }
}

/// Charge every logical image slot, including aliases, before decoding or cloning.
fn collect_images(frames: &[Arc<ImageFrame>], limit: u64) -> Result<Vec<DynamicImage>> {
    let charge = |total: u64, bytes: u64| {
        total
            .checked_add(bytes)
            .filter(|&total| limit == 0 || total <= limit)
            .ok_or(Error::ImageDecodeBudgetExceeded { limit })
    };
    let mut estimated = 0;
    for frame in frames {
        let size = frame.size();
        // Match the Python image admission estimate of four bytes per pixel.
        let bytes = u64::from(size.width)
            .checked_mul(u64::from(size.height))
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or(Error::ImageDecodeBudgetExceeded { limit })?;
        estimated = charge(estimated, bytes)?;
    }

    let mut decoded = 0;
    frames
        .iter()
        .map(|frame| {
            let image = frame.image()?;
            decoded = charge(decoded, image.as_bytes().len() as u64)?;
            Ok(image.clone())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use llm_multimodal::{ImageDetail, ImageSize, ImageSource};

    use super::*;

    #[test]
    fn repeated_images_are_rejected_before_decode() {
        let frame = Arc::new(ImageFrame::encoded(
            Bytes::new(),
            ImageDetail::default(),
            ImageSource::InlineBytes,
            "image".into(),
            ImageSize::new(12_000, 12_000),
            None,
        ));
        let frames = vec![frame; 8];
        let error = collect_images(&frames, 715_827_880).unwrap_err();
        assert!(matches!(error, Error::ImageDecodeBudgetExceeded { .. }));
        assert!(error.is_request_validation_error());
    }

    #[test]
    fn image_budget_counts_aliases_and_accepts_the_boundary() {
        let frame = Arc::new(ImageFrame::new(
            DynamicImage::new_rgba8(2, 2),
            Bytes::new(),
            ImageDetail::default(),
            ImageSource::InlineBytes,
            "image".into(),
        ));
        let frames = vec![frame; 2];
        assert_eq!(collect_images(&frames, 32).unwrap().len(), 2);
        assert!(matches!(
            collect_images(&frames, 31),
            Err(Error::ImageDecodeBudgetExceeded { .. })
        ));
        assert_eq!(collect_images(&frames, 0).unwrap().len(), 2);
    }

    #[test]
    fn image_budget_checks_actual_high_bit_depth_bytes_before_cloning() {
        let frame = Arc::new(ImageFrame::new(
            DynamicImage::new_rgba32f(2, 2),
            Bytes::new(),
            ImageDetail::default(),
            ImageSource::InlineBytes,
            "image".into(),
        ));
        assert!(matches!(
            collect_images(&[frame], 16),
            Err(Error::ImageDecodeBudgetExceeded { .. })
        ));
    }
}
