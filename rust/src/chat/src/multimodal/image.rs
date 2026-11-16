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
        let images = image_frames
            .iter()
            .map(|frame| frame.image().cloned())
            .collect::<std::result::Result<Vec<_>, _>>()?;

        // TODO: is it still necessary given that we've already in a dedicated runtime?
        tokio::task::spawn_blocking(move || {
            let images = images.iter().collect::<Vec<_>>();
            Ok(processor.preprocess_with_context(&images, &preprocessing_context)?)
        })
        .await
        .map_err(|error| multimodal!("image preprocessing task failed: {error}"))?
    }
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, Rgb, RgbImage};
    use llm_multimodal::{ImageDetail, ImageSource};
    use vllm_engine_core_client::protocol::dtype::TensorDtype;
    use vllm_engine_core_client::protocol::multimodal::MmKwargValue;

    use super::super::tests::{glm53_flash_info, qwen3_vl_info};
    use super::*;

    fn solid_frame(pixel: [u8; 3]) -> Arc<ImageFrame> {
        Arc::new(ImageFrame::new(
            DynamicImage::ImageRgb8(RgbImage::from_pixel(28, 28, Rgb(pixel))),
            bytes::Bytes::new(),
            ImageDetail::Auto,
            ImageSource::InlineBytes,
            "image".to_string(),
        ))
    }

    #[tokio::test]
    async fn engine_capability_selects_raw_u8_or_float_images() {
        let mut glm = glm53_flash_info(Default::default());
        glm.set_mm_device_do_normalize(true).unwrap();
        let prepared = glm
            .prepare_images(
                vec![solid_frame([255, 128, 7])],
                vec![None],
                ModelDtype::Float32,
                VisionPreprocessingContext::default(),
            )
            .await
            .unwrap();
        let Some(MmKwargValue::Tensor(primary)) =
            prepared.items[0].data[IMAGE_PRIMARY_KEY].data.as_ref()
        else {
            panic!("GLM image primary input should be a tensor");
        };
        assert_eq!(primary.dtype, TensorDtype::U8);
        assert_eq!(primary.shape, [64, 3 * 2 * 14 * 14]);
        let row = [
            vec![255; 2 * 14 * 14],
            vec![128; 2 * 14 * 14],
            vec![7; 2 * 14 * 14],
        ]
        .concat();
        assert_eq!(
            primary.data.as_raw_view().unwrap().as_ref(),
            row.repeat(64).as_slice()
        );

        let mut qwen = qwen3_vl_info();
        qwen.set_mm_device_do_normalize(false).unwrap();
        let prepared = qwen
            .prepare_images(
                vec![solid_frame([255, 128, 7])],
                vec![None],
                ModelDtype::Float32,
                VisionPreprocessingContext::default(),
            )
            .await
            .unwrap();
        let Some(MmKwargValue::Tensor(primary)) =
            prepared.items[0].data[IMAGE_PRIMARY_KEY].data.as_ref()
        else {
            panic!("Qwen image primary input should be a tensor");
        };
        assert_eq!(primary.dtype, TensorDtype::F32);
    }

    #[tokio::test]
    async fn disabling_device_normalization_restores_frontend_float_images() {
        let mut glm = glm53_flash_info(Default::default());
        glm.set_mm_device_do_normalize(true).unwrap();
        glm.set_mm_device_do_normalize(true).unwrap();
        glm.set_mm_device_do_normalize(false).unwrap();
        glm.set_mm_device_do_normalize(false).unwrap();

        for support in [glm.image.as_ref().unwrap(), glm.video.as_ref().unwrap()] {
            assert_eq!(support.config.do_normalize, None);
            assert_eq!(support.config.do_rescale, None);
        }

        let prepared = glm
            .prepare_images(
                vec![solid_frame([255, 128, 7])],
                vec![None],
                ModelDtype::Float32,
                VisionPreprocessingContext::default(),
            )
            .await
            .unwrap();
        let Some(MmKwargValue::Tensor(primary)) =
            prepared.items[0].data[IMAGE_PRIMARY_KEY].data.as_ref()
        else {
            panic!("GLM image primary input should be a tensor");
        };
        assert_eq!(primary.dtype, TensorDtype::F32);
        let bytes = primary.data.as_raw_view().unwrap();
        let first = f32::from_ne_bytes(bytes[..4].try_into().unwrap());
        let expected = (1.0_f32 - 0.48145466) / 0.26862954;
        assert!((first - expected).abs() < 1e-6, "first pixel was {first}");
    }
}
