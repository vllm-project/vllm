// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Inkling2 image support, mirroring `vllm/models/inkling2/processing.py`.
//!
//! The renderer emits one `<|content_image|>` token per image. It is replaced
//! with `<|content_image|>` followed by `rows * cols` copies of the image pad
//! token; only the pad positions receive vision embeddings.

use std::collections::HashMap;

use image::{DynamicImage, GenericImageView as _, Rgb, RgbImage};
use llm_multimodal::registry::{ModelRegistryError, RegistryResult};
use llm_multimodal::{
    EncoderFieldLayouts, FieldLayout, Modality, ModelMetadata, ModelProcessorSpec,
    ModelSpecificValue, PreProcessorConfig, PreprocessedEncoderInputs, PromptReplacement, TokenId,
    TransformError, VisionPreProcessor,
};
use ndarray::ArrayD;
use serde_json::Value;

use crate::RendererSelection;

const MODEL_TYPE: &str = RendererSelection::INKLING2_MODEL_TYPE;
const IMAGE_PLACEHOLDER: &str = "<|content_image|>";
/// `<|unused_200054|>`: one per decoder image token, overwritten by embeddings.
const IMAGE_PAD_ID: TokenId = 200054;

const VISION_PIXELS_KEY: &str = "vision_pixels";
const VISION_GRID_THW_KEY: &str = "vision_grid_thw";

const PATCH_SIZE: usize = 40;
/// The encoder's CLIP normalization (applied on the GPU to raw pixels).
const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const CLIP_STD: [f32; 3] = [0.26862954, 0.2613026, 0.2757771];
/// Raw value of pixels padding an image to the patch grid.
const PAD_RAW_VALUE: f32 = -1.0;
const CHECKERBOARD_CELL: u32 = 8;
const CHECKERBOARD_COLORS: [u16; 2] = [204, 153];

/// In-repo model spec for Inkling2 (`inkling2`) images.
pub(super) struct Inkling2Spec;

impl Inkling2Spec {
    fn unsupported(&self, modality: Modality) -> ModelRegistryError {
        ModelRegistryError::UnsupportedModality {
            spec: self.name(),
            modality,
        }
    }
}

impl ModelProcessorSpec for Inkling2Spec {
    fn name(&self) -> &'static str {
        "inkling2"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        metadata.config_model_type() == Some(MODEL_TYPE)
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok(IMAGE_PLACEHOLDER.to_string())
    }

    fn placeholder_token_id(&self, _metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        Ok(IMAGE_PAD_ID)
    }

    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([(Modality::Image, usize::MAX)]))
    }

    fn vision_processor(
        &self,
        _model_config: &Value,
        _preprocessor_config: &PreProcessorConfig,
        modality: Modality,
    ) -> RegistryResult<Box<dyn VisionPreProcessor>> {
        match modality {
            Modality::Image => Ok(Box::new(Inkling2ImageProcessor)),
            _ => Err(self.unsupported(modality)),
        }
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let marker_id = metadata.token_id(IMAGE_PLACEHOLDER)?;
        Ok(preprocessed
            .feature_token_counts
            .iter()
            .map(|&num_tokens| {
                let tokens = std::iter::once(marker_id)
                    .chain(std::iter::repeat_n(IMAGE_PAD_ID, num_tokens))
                    .collect();
                PromptReplacement::sequence(Modality::Image, IMAGE_PLACEHOLDER, tokens)
            })
            .collect())
    }

    fn encoder_field_layouts_for(&self, modality: Modality) -> EncoderFieldLayouts {
        match modality {
            // Patches are flat over `vision_grid_thw.prod(-1)` per image.
            Modality::Image => EncoderFieldLayouts::new(
                FieldLayout::flat(VISION_GRID_THW_KEY),
                HashMap::from([(VISION_GRID_THW_KEY.to_string(), FieldLayout::Batched)]),
            ),
            _ => EncoderFieldLayouts::default(),
        }
    }

    fn encoder_input_key_for(&self, modality: Modality) -> Option<String> {
        (modality == Modality::Image).then(|| VISION_PIXELS_KEY.to_string())
    }

    fn keep_on_cpu_keys(&self) -> Vec<String> {
        vec![VISION_GRID_THW_KEY.to_string()]
    }
}

/// Pads each image to a 40-pixel grid and emits one patch per cell.
///
/// Produces `vision_pixels` (`[sum(rows*cols), 40, 40, 3]` raw RGB values,
/// padding pixels at -1; the encoder normalizes them) and `vision_grid_thw`
/// (int64 `[num_images, 3]`). Images are not resized.
#[derive(Debug, Clone, Copy)]
pub(super) struct Inkling2ImageProcessor;

impl Inkling2ImageProcessor {
    fn grid(width: usize, height: usize) -> (usize, usize) {
        (height.div_ceil(PATCH_SIZE), width.div_ceil(PATCH_SIZE))
    }

    /// Append one image's patches and return its `(rows, cols)` grid.
    fn fill_patches(image: &DynamicImage, out: &mut Vec<f32>) -> [usize; 2] {
        let pixels = to_rgb(image);
        let (width, height) = (pixels.width() as usize, pixels.height() as usize);
        let raw = pixels.as_raw();
        let (rows, cols) = Self::grid(width, height);

        out.reserve(rows * cols * PATCH_SIZE * PATCH_SIZE * 3);
        for row in 0..rows {
            for col in 0..cols {
                for y in row * PATCH_SIZE..(row + 1) * PATCH_SIZE {
                    let x0 = col * PATCH_SIZE;
                    let valid = if y < height {
                        width.saturating_sub(x0).min(PATCH_SIZE)
                    } else {
                        0
                    };
                    if valid > 0 {
                        let start = (y * width + x0) * 3;
                        out.extend(
                            raw[start..start + valid * 3].iter().map(|&value| f32::from(value)),
                        );
                    }
                    out.extend(std::iter::repeat_n(PAD_RAW_VALUE, (PATCH_SIZE - valid) * 3));
                }
            }
        }
        [rows, cols]
    }
}

impl VisionPreProcessor for Inkling2ImageProcessor {
    fn default_mean(&self) -> [f64; 3] {
        CLIP_MEAN.map(f64::from)
    }

    fn default_std(&self) -> [f64; 3] {
        CLIP_STD.map(f64::from)
    }

    fn preprocess(
        &self,
        images: &[DynamicImage],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if images.is_empty() {
            return Err(TransformError::EmptyBatch);
        }

        let mut patches = Vec::new();
        let mut grid_thw = Vec::with_capacity(images.len() * 3);
        let mut num_tokens = Vec::with_capacity(images.len());
        for image in images {
            let [rows, cols] = Self::fill_patches(image, &mut patches);
            grid_thw.extend([1, rows as i64, cols as i64]);
            num_tokens.push(rows * cols);
        }

        let total_patches = num_tokens.iter().sum::<usize>();
        let shape = vec![total_patches, PATCH_SIZE, PATCH_SIZE, 3];
        let encoder_input = ArrayD::from_shape_vec(shape, patches).map_err(|error| {
            TransformError::ShapeError(format!("failed to build {VISION_PIXELS_KEY}: {error}"))
        })?;
        let item_sizes = images.iter().map(|image| image.dimensions()).collect();

        Ok(
            PreprocessedEncoderInputs::new(encoder_input, num_tokens, item_sizes).with_extra(
                VISION_GRID_THW_KEY,
                ModelSpecificValue::int_2d(grid_thw, images.len(), 3),
            ),
        )
    }

    fn calculate_num_tokens(&self, width: u32, height: u32) -> Result<usize, TransformError> {
        let (rows, cols) = Self::grid(width as usize, height as usize);
        Ok(rows * cols)
    }

    fn model_name(&self) -> &'static str {
        "inkling2"
    }
}

/// Convert to RGB, compositing images with alpha onto an 8px 204/153
/// checkerboard.
fn to_rgb(image: &DynamicImage) -> RgbImage {
    if !image.color().has_alpha() {
        return image.to_rgb8();
    }
    let rgba = image.to_rgba8();
    RgbImage::from_fn(rgba.width(), rgba.height(), |x, y| {
        let [r, g, b, alpha] = rgba.get_pixel(x, y).0.map(u16::from);
        let background =
            CHECKERBOARD_COLORS[((x / CHECKERBOARD_CELL + y / CHECKERBOARD_CELL) % 2) as usize];
        Rgb([r, g, b].map(|value| ((value * alpha + background * (255 - alpha) + 127) / 255) as u8))
    })
}

#[cfg(test)]
mod tests;
