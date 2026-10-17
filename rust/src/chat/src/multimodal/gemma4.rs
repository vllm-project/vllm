// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Gemma 4 multimodal model specification and vision preprocessor.
//!
//! Gemma 4 models (e.g., `google/gemma-4-26B-A4B-it`, `google/gemma-4-E4B-it`,
//! and unified variants) use an aspect-ratio-preserving resize driven by
//! `patch_size`, `pooling_kernel_size`, and `max_soft_tokens`.
//!
//! Input images are resized to a resolution that fits within the patch budget,
//! then patchified into `(max_patches, patch_pixels)` where:
//! - `max_patches = max_soft_tokens * pooling_kernel_size^2`
//! - `patch_pixels = patch_size * patch_size * 3`
//!
//! Any unused patch slots up to `max_patches` are padded with 0 in `pixel_values`,
//! while `pixel_position_ids` records the `[col, row]` grid coordinates of real
//! patches and `[-1, -1]` for padding patches.
//!
//! In the prompt token stream, each `<|image|>` placeholder expands to:
//! `[boi_token_id] + [image_token_id; num_soft_tokens] + [eoi_token_id]`
//! where only the `image_token_id` soft tokens are marked as embeddings (`is_embed`).

use std::collections::HashMap;

use image::DynamicImage;
use llm_multimodal::registry::{
    ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult,
};
use llm_multimodal::types::{FieldLayout, Modality, PromptReplacement, TokenId};
use llm_multimodal::vision::{
    ModelSpecificValue, PreProcessorConfig, PreprocessedEncoderInputs, TransformError,
    VisionPreProcessor,
};
use ndarray::ArrayD;

/// Supported soft token budgets matching `_SUPPORTED_SOFT_TOKENS`.
#[allow(dead_code)]
pub const SUPPORTED_SOFT_TOKENS: &[usize] = &[70, 140, 280, 560, 1120];

/// Calculate the aspect-ratio preserving dimensions that fit within `max_patches`.
pub fn get_aspect_ratio_preserving_size(
    width: u32,
    height: u32,
    patch_size: u32,
    max_patches: usize,
    pooling_kernel_size: u32,
) -> (u32, u32) {
    if width == 0 || height == 0 || patch_size == 0 || pooling_kernel_size == 0 {
        return (patch_size.max(1), patch_size.max(1));
    }

    let total_px = (height as f64) * (width as f64);
    let target_px = (max_patches as f64) * ((patch_size * patch_size) as f64);
    let factor = (target_px / total_px).sqrt();
    let ideal_height = factor * (height as f64);
    let ideal_width = factor * (width as f64);
    let side_mult = (pooling_kernel_size * patch_size) as f64;

    let mut target_height = (ideal_height / side_mult).floor() * side_mult;
    let mut target_width = (ideal_width / side_mult).floor() * side_mult;

    let pool_sq = (pooling_kernel_size * pooling_kernel_size) as usize;
    let max_side_len = ((max_patches / pool_sq) as f64) * side_mult;

    if target_height == 0.0 {
        target_height = side_mult;
        target_width = (((width as f64) / (height as f64)).floor() * side_mult).min(max_side_len);
    } else if target_width == 0.0 {
        target_width = side_mult;
        target_height = (((height as f64) / (width as f64)).floor() * side_mult).min(max_side_len);
    }

    let mut th = target_height as u32;
    let mut tw = target_width as u32;
    let side_mult_u = (pooling_kernel_size * patch_size).max(1);
    if th == 0 {
        th = side_mult_u;
    }
    if tw == 0 {
        tw = side_mult_u;
    }

    // Guard against rounding overflow on extreme aspect ratios
    while (th as u64) * (tw as u64) > (target_px as u64) && (th > side_mult_u || tw > side_mult_u) {
        if tw > th && tw > side_mult_u {
            tw -= side_mult_u;
        } else if th > side_mult_u {
            th -= side_mult_u;
        } else {
            break;
        }
    }

    (tw, th)
}

/// Compute the number of soft tokens produced for an image after stripping padding.
pub fn compute_num_soft_tokens(
    width: u32,
    height: u32,
    patch_size: u32,
    max_soft_tokens: usize,
    pooling_kernel_size: u32,
) -> usize {
    if width == 0 || height == 0 || patch_size == 0 || pooling_kernel_size == 0 {
        return 0;
    }
    let max_patches = max_soft_tokens * (pooling_kernel_size as usize).pow(2);
    let (target_w, target_h) = get_aspect_ratio_preserving_size(
        width,
        height,
        patch_size,
        max_patches,
        pooling_kernel_size,
    );
    let num_patches = (target_h / patch_size) as usize * (target_w / patch_size) as usize;
    let pool_sq = (pooling_kernel_size as usize).pow(2);
    let num_soft_tokens = num_patches / pool_sq;
    num_soft_tokens.min(max_soft_tokens)
}

/// Gemma 4 model processor specification.
pub struct Gemma4Spec;

impl Gemma4Spec {
    pub fn patch_size(metadata: &ModelMetadata<'_>) -> u32 {
        metadata
            .config_u32(&["vision_config", "patch_size"])
            .or_else(|| metadata.config_u32(&["patch_size"]))
            .unwrap_or(16)
    }

    pub fn pooling_kernel_size(metadata: &ModelMetadata<'_>) -> u32 {
        metadata
            .config_u32(&["vision_config", "pooling_kernel_size"])
            .or_else(|| metadata.config_u32(&["pooling_kernel_size"]))
            .unwrap_or(3)
    }

    pub fn max_soft_tokens(metadata: &ModelMetadata<'_>) -> usize {
        metadata
            .config_u32(&["vision_config", "default_output_length"])
            .or_else(|| metadata.config_u32(&["vision_config", "num_soft_tokens"]))
            .or_else(|| metadata.config_u32(&["vision_soft_tokens_per_image"]))
            .or_else(|| metadata.config_u32(&["max_soft_tokens"]))
            .unwrap_or(280) as usize
    }
}

impl ModelProcessorSpec for Gemma4Spec {
    fn name(&self) -> &'static str {
        "gemma4"
    }

    fn matches(&self, metadata: &ModelMetadata<'_>) -> bool {
        let id = metadata.model_id.to_ascii_lowercase();
        id.contains("gemma-4")
            || id.contains("gemma4")
            || metadata
                .config_model_type()
                .is_some_and(|mt| mt == "gemma4" || mt == "gemma4_unified")
            || metadata.config.get("architectures").and_then(|a| a.as_array()).is_some_and(
                |archs| {
                    archs.iter().any(|a| {
                        a.as_str().is_some_and(|s| {
                            s == "Gemma4ForConditionalGeneration"
                                || s == "Gemma4UnifiedForConditionalGeneration"
                        })
                    })
                },
            )
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata<'_>) -> RegistryResult<String> {
        Ok("<|image|>".to_string())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata<'_>) -> RegistryResult<TokenId> {
        if let Some(id) = metadata.config_u32(&["image_token_id"]) {
            return Ok(id as TokenId);
        }
        if let Some(id) = metadata.config_u32(&["image_token_index"]) {
            return Ok(id as TokenId);
        }
        metadata.token_id("<|image|>")
    }

    fn modality_limits(
        &self,
        _metadata: &ModelMetadata<'_>,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([(Modality::Image, usize::MAX)]))
    }

    fn processor_kwargs(&self, _metadata: &ModelMetadata<'_>) -> RegistryResult<serde_json::Value> {
        Ok(serde_json::json!({}))
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata<'_>,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let image_token_id = self.placeholder_token_id(metadata)?;
        let boi_token_id = metadata
            .config_u32(&["boi_token_id"])
            .map(|id| id as TokenId)
            .or_else(|| metadata.token_id("<|image_start|>").ok())
            .or_else(|| metadata.token_id("<boi>").ok())
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: "boi_token_id".to_string(),
            })?;
        let eoi_token_id = metadata
            .config_u32(&["eoi_token_id"])
            .map(|id| id as TokenId)
            .or_else(|| metadata.token_id("<|image_end|>").ok())
            .or_else(|| metadata.token_id("<eoi>").ok())
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: "eoi_token_id".to_string(),
            })?;

        let patch_size = Self::patch_size(metadata);
        let pooling_kernel_size = Self::pooling_kernel_size(metadata);
        let max_soft_tokens = Self::max_soft_tokens(metadata);

        let soft_tokens_per_item: Vec<usize> = if !preprocessed.feature_token_counts.is_empty() {
            preprocessed.feature_token_counts.clone()
        } else {
            preprocessed
                .item_sizes
                .iter()
                .map(|&(width, height)| {
                    compute_num_soft_tokens(
                        width,
                        height,
                        patch_size,
                        max_soft_tokens,
                        pooling_kernel_size,
                    )
                })
                .collect()
        };

        Ok(soft_tokens_per_item
            .into_iter()
            .map(|num_soft| {
                let mut tokens = Vec::with_capacity(num_soft + 2);
                tokens.push(boi_token_id);
                tokens.extend(std::iter::repeat_n(image_token_id, num_soft));
                tokens.push(eoi_token_id);
                PromptReplacement::sequence(Modality::Image, "<|image|>", tokens)
            })
            .collect())
    }

    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        HashMap::from([
            ("pixel_values".to_string(), FieldLayout::Batched),
            ("pixel_position_ids".to_string(), FieldLayout::Batched),
        ])
    }

    fn vision_processor(
        &self,
        metadata: &ModelMetadata<'_>,
        config: &PreProcessorConfig,
        modality: Modality,
    ) -> RegistryResult<Box<dyn VisionPreProcessor>> {
        match modality {
            Modality::Image => Ok(Box::new(Gemma4VisionProcessor::from_configs(
                metadata, config,
            ))),
            _ => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            }),
        }
    }
}

/// Vision preprocessor for Gemma 4 models.
#[derive(Debug, Clone)]
pub struct Gemma4VisionProcessor {
    pub patch_size: u32,
    pub pooling_kernel_size: u32,
    pub max_soft_tokens: usize,
    pub do_normalize: bool,
    pub rescale_factor: f64,
    pub mean: [f64; 3],
    pub std: [f64; 3],
}

impl Gemma4VisionProcessor {
    pub fn from_configs(metadata: &ModelMetadata<'_>, config: &PreProcessorConfig) -> Self {
        let patch_size = config
            .patch_size
            .as_ref()
            .and_then(|p| p.height)
            .or_else(|| metadata.config_u32(&["vision_config", "patch_size"]))
            .or_else(|| metadata.config_u32(&["patch_size"]))
            .unwrap_or(16);

        let pooling_kernel_size = config
            .extra
            .get("pooling_kernel_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .or_else(|| metadata.config_u32(&["vision_config", "pooling_kernel_size"]))
            .or_else(|| metadata.config_u32(&["pooling_kernel_size"]))
            .unwrap_or(3);

        let max_soft_tokens = config
            .extra
            .get("max_soft_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .or_else(|| {
                config
                    .extra
                    .get("image_seq_length")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
            })
            .or_else(|| {
                metadata
                    .config_u32(&["vision_config", "default_output_length"])
                    .map(|v| v as usize)
            })
            .or_else(|| {
                metadata.config_u32(&["vision_config", "num_soft_tokens"]).map(|v| v as usize)
            })
            .unwrap_or(280);

        let do_normalize = config.do_normalize.unwrap_or(false);
        let rescale_factor = config.rescale_factor.unwrap_or(1.0 / 255.0);

        let mean = match &config.image_mean {
            Some(m) if m.len() >= 3 => [m[0], m[1], m[2]],
            _ => [0.0, 0.0, 0.0],
        };

        let std = match &config.image_std {
            Some(s) if s.len() >= 3 => [s[0], s[1], s[2]],
            _ => [1.0, 1.0, 1.0],
        };

        Self {
            patch_size,
            pooling_kernel_size,
            max_soft_tokens,
            do_normalize,
            rescale_factor,
            mean,
            std,
        }
    }

    fn patchify_and_pad(
        &self,
        image: &DynamicImage,
        target_w: u32,
        target_h: u32,
        max_patches: usize,
    ) -> (Vec<f32>, Vec<i64>, usize) {
        let p = self.patch_size as usize;
        let patch_pixels = p * p * 3;
        let num_patches_w = (target_w / self.patch_size) as usize;
        let num_patches_h = (target_h / self.patch_size) as usize;
        let num_real_patches = num_patches_w * num_patches_h;

        let resized = if image.width() == target_w && image.height() == target_h {
            image.to_rgb8()
        } else {
            image
                .resize_exact(target_w, target_h, image::imageops::FilterType::CatmullRom)
                .to_rgb8()
        };

        let raw = resized.as_raw();
        let row_stride = (target_w as usize) * 3;

        let mut pixel_values = vec![0.0f32; max_patches * patch_pixels];
        let mut position_ids = vec![-1i64; max_patches * 2];

        for py in 0..num_patches_h {
            for px in 0..num_patches_w {
                let patch_idx = py * num_patches_w + px;
                let patch_out =
                    &mut pixel_values[patch_idx * patch_pixels..(patch_idx + 1) * patch_pixels];

                for dy in 0..p {
                    for dx in 0..p {
                        let y = py * p + dy;
                        let x = px * p + dx;
                        let pixel_offset = y * row_stride + x * 3;
                        let dest_offset = (dy * p + dx) * 3;

                        for c in 0..3 {
                            let val = raw[pixel_offset + c] as f64;
                            let normalized = if self.do_normalize {
                                (val * self.rescale_factor - self.mean[c]) / self.std[c]
                            } else {
                                val * self.rescale_factor
                            };
                            patch_out[dest_offset + c] = normalized as f32;
                        }
                    }
                }

                position_ids[patch_idx * 2] = px as i64;
                position_ids[patch_idx * 2 + 1] = py as i64;
            }
        }

        let pool_sq = (self.pooling_kernel_size as usize).pow(2);
        let num_soft = (num_real_patches / pool_sq).min(self.max_soft_tokens);

        (pixel_values, position_ids, num_soft)
    }
}

impl VisionPreProcessor for Gemma4VisionProcessor {
    fn default_mean(&self) -> [f64; 3] {
        self.mean
    }

    fn default_std(&self) -> [f64; 3] {
        self.std
    }

    fn calculate_num_tokens(&self, width: u32, height: u32) -> usize {
        compute_num_soft_tokens(
            width,
            height,
            self.patch_size,
            self.max_soft_tokens,
            self.pooling_kernel_size,
        )
    }

    fn model_name(&self) -> &'static str {
        "gemma4"
    }

    fn preprocess(
        &self,
        images: &[DynamicImage],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if images.is_empty() {
            return Err(TransformError::InvalidShape {
                expected: "non-empty image batch".to_string(),
                actual: vec![0],
            });
        }

        let batch_size = images.len();
        let max_patches = self.max_soft_tokens * (self.pooling_kernel_size as usize).pow(2);
        let patch_pixels = (self.patch_size * self.patch_size * 3) as usize;

        let mut all_pixel_values = Vec::with_capacity(batch_size * max_patches * patch_pixels);
        let mut all_position_ids = Vec::with_capacity(batch_size * max_patches * 2);
        let mut feature_token_counts = Vec::with_capacity(batch_size);
        let mut item_sizes = Vec::with_capacity(batch_size);

        for image in images {
            let (orig_w, orig_h) = (image.width(), image.height());
            let (target_w, target_h) = get_aspect_ratio_preserving_size(
                orig_w,
                orig_h,
                self.patch_size,
                max_patches,
                self.pooling_kernel_size,
            );

            let (pixel_values, position_ids, num_soft) =
                self.patchify_and_pad(image, target_w, target_h, max_patches);

            all_pixel_values.extend(pixel_values);
            all_position_ids.extend(position_ids);
            feature_token_counts.push(num_soft);
            item_sizes.push((orig_w, orig_h));
        }

        let encoder_input = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[batch_size, max_patches, patch_pixels]),
            all_pixel_values,
        )
        .map_err(|e| TransformError::ShapeError(e.to_string()))?;

        let position_tensor = ModelSpecificValue::IntTensor {
            data: all_position_ids,
            shape: vec![batch_size, max_patches, 2],
        };

        Ok(
            PreprocessedEncoderInputs::new(encoder_input, feature_token_counts, item_sizes)
                .with_extra("pixel_position_ids", position_tensor),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};
    use serde_json::json;

    struct DummyTokenizer;

    impl llm_multimodal::Tokenizer for DummyTokenizer {
        fn token_to_id(&self, token: &str) -> Option<u32> {
            match token {
                "<|image|>" => Some(258880),
                "<|image_start|>" => Some(255999),
                "<|image_end|>" => Some(258882),
                _ => None,
            }
        }

        fn id_to_token(&self, id: u32) -> Option<String> {
            match id {
                258880 => Some("<|image|>".to_string()),
                255999 => Some("<|image_start|>".to_string()),
                258882 => Some("<|image_end|>".to_string()),
                _ => None,
            }
        }

        fn encode_text(&self, _text: &str) -> Option<Vec<u32>> {
            None
        }
    }

    #[test]
    fn test_compute_num_soft_tokens_standard_resolutions() {
        // (224, 224) -> 256 tokens
        assert_eq!(compute_num_soft_tokens(224, 224, 16, 280, 3), 256);
        // (1000, 1000) -> 256 tokens
        assert_eq!(compute_num_soft_tokens(1000, 1000, 16, 280, 3), 256);
        // (1920, 1080) -> 264 tokens
        assert_eq!(compute_num_soft_tokens(1920, 1080, 16, 280, 3), 264);
        // (800, 600) -> 266 tokens
        assert_eq!(compute_num_soft_tokens(800, 600, 16, 280, 3), 266);
    }

    #[test]
    fn test_compute_num_soft_tokens_extreme_aspect_ratios() {
        // Extreme aspect ratios should be clamped and never exceed max_soft_tokens
        let tokens_wide = compute_num_soft_tokens(900, 3, 16, 280, 3);
        assert!(tokens_wide <= 280);
        assert_eq!(tokens_wide, 280);

        let tokens_tall = compute_num_soft_tokens(3, 900, 16, 280, 3);
        assert!(tokens_tall <= 280);
        assert_eq!(tokens_tall, 280);

        // Budget 70
        let tokens_video = compute_num_soft_tokens(900, 3, 16, 70, 3);
        assert!(tokens_video <= 70);
    }

    #[test]
    fn test_gemma4_spec_matches() {
        let tokenizer = DummyTokenizer;
        let config = json!({
            "model_type": "gemma4",
            "architectures": ["Gemma4ForConditionalGeneration"]
        });
        let metadata = ModelMetadata {
            model_id: "google/gemma-4-26B-A4B-it",
            tokenizer: &tokenizer,
            config: &config,
        };
        assert!(Gemma4Spec.matches(&metadata));

        let config_unified = json!({
            "model_type": "gemma4_unified",
            "architectures": ["Gemma4UnifiedForConditionalGeneration"]
        });
        let metadata_unified = ModelMetadata {
            model_id: "google/gemma-4-unified-mock",
            tokenizer: &tokenizer,
            config: &config_unified,
        };
        assert!(Gemma4Spec.matches(&metadata_unified));

        let config_other = json!({
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"]
        });
        let metadata_other = ModelMetadata {
            model_id: "meta-llama/Llama-3.1-8B-Instruct",
            tokenizer: &tokenizer,
            config: &config_other,
        };
        assert!(!Gemma4Spec.matches(&metadata_other));
    }

    #[test]
    fn test_gemma4_prompt_replacements() {
        let tokenizer = DummyTokenizer;
        let config = json!({
            "model_type": "gemma4",
            "image_token_id": 258880,
            "boi_token_id": 255999,
            "eoi_token_id": 258882,
            "vision_config": {
                "patch_size": 16,
                "pooling_kernel_size": 3,
                "default_output_length": 280
            }
        });
        let metadata = ModelMetadata {
            model_id: "google/gemma-4-26B-A4B-it",
            tokenizer: &tokenizer,
            config: &config,
        };

        let dummy_input = ArrayD::zeros(ndarray::IxDyn(&[1, 2520, 768]));
        let preprocessed = PreprocessedEncoderInputs::new(dummy_input, vec![256], vec![(224, 224)]);

        let replacements = Gemma4Spec.prompt_replacements(&metadata, &preprocessed).unwrap();
        assert_eq!(replacements.len(), 1);
        let repl = &replacements[0];
        assert_eq!(repl.modality, Modality::Image);
        assert_eq!(repl.placeholder_token, "<|image|>");
        assert_eq!(repl.tokens.len(), 258); // 1 boi + 256 image_tokens + 1 eoi
        assert_eq!(repl.tokens[0], 255999);
        assert_eq!(repl.tokens[257], 258882);
        for &tok in &repl.tokens[1..257] {
            assert_eq!(tok, 258880);
        }
    }

    #[test]
    fn test_gemma4_vision_processor_preprocessing() {
        let tokenizer = DummyTokenizer;
        let config_json = json!({
            "model_type": "gemma4",
            "vision_config": {
                "patch_size": 16,
                "pooling_kernel_size": 3,
                "default_output_length": 280
            }
        });
        let metadata = ModelMetadata {
            model_id: "google/gemma-4-26B-A4B-it",
            tokenizer: &tokenizer,
            config: &config_json,
        };
        let preprocessor_cfg = PreProcessorConfig::default();
        let processor = Gemma4VisionProcessor::from_configs(&metadata, &preprocessor_cfg);

        // Create a 224x224 RGB image
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(224, 224, Rgb([128, 64, 32])));
        let inputs = processor.preprocess(&[img]).unwrap();

        assert_eq!(inputs.encoder_input.ndim(), 3);
        assert_eq!(inputs.encoder_input.shape(), &[1, 2520, 768]);
        assert_eq!(inputs.feature_token_counts, vec![256]);
        assert_eq!(inputs.item_sizes, vec![(224, 224)]);

        // Check pixel_position_ids
        let pos_val = inputs.model_specific.get("pixel_position_ids").unwrap();
        if let ModelSpecificValue::IntTensor { data, shape } = pos_val {
            assert_eq!(shape, &[1, 2520, 2]);
            // First patch should be at [0, 0]
            assert_eq!(data[0], 0);
            assert_eq!(data[1], 0);
            // Second patch in row 0 should be at [1, 0]
            assert_eq!(data[2], 1);
            assert_eq!(data[3], 0);
            // Patch 2304 is the start of padding, should have [-1, -1]
            assert_eq!(data[2304 * 2], -1);
            assert_eq!(data[2304 * 2 + 1], -1);
        } else {
            panic!("expected IntTensor for pixel_position_ids");
        }
    }
}
