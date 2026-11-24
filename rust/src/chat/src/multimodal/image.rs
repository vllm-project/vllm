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

    use super::super::tests::{glm53_flash_info, glm53_flash_info_from_paths, qwen3_vl_info};
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
    async fn device_normalization_rebuilds_preserve_startup_processor_kwargs() {
        const PIXELS_PER_TOKEN: usize = 2 * 14 * 14;

        let mut glm = glm53_flash_info_from_paths(
            serde_json::json!({
                "do_resize": false,
                "patch_size": 14,
                "merge_size": 2,
                "temporal_patch_size": 2,
                "patch_expand_factor": 1,
                "min_pixels": 16 * PIXELS_PER_TOKEN,
                "max_pixels": 128 * PIXELS_PER_TOKEN,
                "resample": 3,
            }),
            serde_json::from_value(serde_json::json!({
                "do_resize": true,
                "do_normalize": true,
                "do_rescale": true,
                "min_pixels": 8 * PIXELS_PER_TOKEN,
                "max_pixels": 32 * PIXELS_PER_TOKEN,
                "resample": 2,
                "images_kwargs": {
                    "min_pixels": PIXELS_PER_TOKEN,
                    "max_pixels": 4 * PIXELS_PER_TOKEN,
                    "image_mean": [0.0, 0.0, 0.0],
                    "image_std": [1.0, 1.0, 1.0],
                },
                "videos_kwargs": {"max_pixels": 8 * PIXELS_PER_TOKEN},
            }))
            .unwrap(),
        );
        let image = glm.image.as_ref().unwrap();
        assert_eq!(image.config.do_resize, Some(true));
        assert_eq!(image.config.resampling, Some(2));
        assert_eq!(image.config.min_pixels, Some(PIXELS_PER_TOKEN));
        assert_eq!(image.config.max_pixels, Some(4 * PIXELS_PER_TOKEN));
        let video = glm.video.as_ref().unwrap();
        assert_eq!(video.config.min_pixels, Some(8 * PIXELS_PER_TOKEN));
        assert_eq!(video.config.max_pixels, Some(8 * PIXELS_PER_TOKEN));
        let startup_configs =
            [image, video].map(|support| serde_json::to_value(&support.config).unwrap());

        glm.set_mm_device_do_normalize(true).unwrap();
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
        assert_eq!(primary.shape, [4, 3 * PIXELS_PER_TOKEN]);

        glm.set_mm_device_do_normalize(false).unwrap();
        glm.set_mm_device_do_normalize(false).unwrap();
        for (support, startup) in [glm.image.as_ref().unwrap(), glm.video.as_ref().unwrap()]
            .into_iter()
            .zip(startup_configs)
        {
            assert_eq!(serde_json::to_value(&support.config).unwrap(), startup);
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
        assert_eq!(primary.shape, [4, 3 * PIXELS_PER_TOKEN]);
        let bytes = primary.data.as_raw_view().unwrap();
        let pixel = |index| {
            let offset = index * size_of::<f32>();
            f32::from_ne_bytes(bytes[offset..offset + size_of::<f32>()].try_into().unwrap())
        };
        assert_eq!(pixel(0), 1.0);
        assert!((pixel(PIXELS_PER_TOKEN) - 128.0 / 255.0).abs() < 1e-6);
        assert!((pixel(2 * PIXELS_PER_TOKEN) - 7.0 / 255.0).abs() < 1e-6);
    }
}
