// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Step3-VL vision preprocessor.
//!
//! Port of vLLM's `transformers_utils/processors/step3_vl.py` (also the
//! frontend Step5-VL reuses): a 728x728 global view plus a
//! sliding-window grid of 504x504 patches, both CLIP-normalized, with a
//! per-patch `<patch_newline>` mask marking row ends in reading order.
//!
//! Each `<im_patch>` placeholder expands to
//! `<im_start> <im_patch>*169 <im_end>` for the global view plus, per patch,
//! `<patch_start> <im_patch>*81 <patch_end>` and a `<patch_newline>` wherever
//! the mask is set. The integer/truncation semantics of the reference
//! (`int()`, `//`, float ratios) are replicated exactly, including the quirk
//! where the crop-resize decision compares the crop size against the post-pad
//! dimensions rather than the post-preprocess dimensions.

use image::{DynamicImage, Rgb, RgbImage, imageops::FilterType};

use llm_multimodal::vision::transforms::{resize, stack_batch, to_tensor_and_normalize};
use llm_multimodal::{
    ModelSpecificValue, PreProcessorConfig, PreprocessedEncoderInputs, TransformError,
    VisionPreProcessor,
};

/// Reference `MAX_IMAGE_SIZE`: longer side is downscaled below this.
const MAX_IMAGE_SIZE: u32 = 3024;
/// Hard-coded threshold inside the reference `determine_window_size`.
const WINDOW_LONG_SIDE: u32 = 728;
/// Hard-coded default window inside the reference `determine_window_size`.
const WINDOW_DEFAULT: u32 = 504;

const DEFAULT_IMAGE_SIZE: u32 = 728;
const DEFAULT_PATCH_SIZE: u32 = 504;
const DEFAULT_NUM_IMAGE_FEATURE_SIZE: usize = 169;
const DEFAULT_NUM_PATCH_FEATURE_SIZE: usize = 81;

/// Long/short side -> sliding window size (0 disables patching).
fn determine_window_size(long: u32, short: u32) -> u32 {
    let ratio = long as f64 / short as f64;
    if long < WINDOW_LONG_SIDE {
        return if ratio > 1.5 { short } else { 0 };
    }
    if ratio > 4.0 {
        short.min(WINDOW_DEFAULT)
    } else {
        WINDOW_DEFAULT
    }
}

/// Extreme-aspect tiny images are square-padded before anything else.
fn get_image_size_for_padding(width: u32, height: u32) -> (u32, u32) {
    let ratio = width as f64 / height as f64;
    if width.min(height) < 32 && !(0.25..=4.0).contains(&ratio) {
        let size = width.max(height);
        (size, size)
    } else {
        (width, height)
    }
}

/// Cap the longer side at `MAX_IMAGE_SIZE` (Python `int()` truncation).
fn get_image_size_for_preprocess(width: u32, height: u32) -> (u32, u32) {
    let max_side = width.max(height);
    if max_side > MAX_IMAGE_SIZE {
        let scale = MAX_IMAGE_SIZE as f64 / max_side as f64;
        (
            (width as f64 * scale) as u32,
            (height as f64 * scale) as u32,
        )
    } else {
        (width, height)
    }
}

/// Round one dimension up to a window multiple; a fractional part above 0.2
/// adds a whole extra window.
fn crop_dim(dim: u32, window_size: u32) -> u32 {
    let ratio = dim as f64 / window_size as f64;
    if ratio < 1.0 {
        return dim;
    }
    let decimal = ratio - (dim / window_size) as f64;
    let n = if decimal > 0.2 {
        ratio as u32 + 1
    } else {
        ratio as u32
    };
    window_size * n
}

fn get_image_size_for_crop(width: u32, height: u32, window_size: u32) -> (u32, u32) {
    (crop_dim(width, window_size), crop_dim(height, window_size))
}

/// Window starts along one axis; the last start is clamped so the window
/// stays inside the extent.
fn window_starts(extent: u32, window_size: u32) -> Vec<u32> {
    let num = if extent <= window_size {
        1
    } else {
        (((extent - window_size) as f64 / window_size as f64) + 1.0).ceil() as u32
    };
    let mut starts: Vec<u32> = (0..num).map(|i| window_size * i).collect();
    if num > 1 && starts[num as usize - 1] + window_size > extent {
        starts[num as usize - 1] = extent - window_size;
    }
    starts
}

/// A sliding-window box as (x, y, width, height).
type Window = (u32, u32, u32, u32);

/// Sliding windows over a (width, height) extent, y-outer x-inner order.
/// Returns (windows, x_num, y_num).
fn slide_window(width: u32, height: u32, window_size: u32) -> (Vec<Window>, u32, u32) {
    let x_start = window_starts(width, window_size);
    let y_start = window_starts(height, window_size);
    let mut windows = Vec::with_capacity(x_start.len() * y_start.len());
    for &y in &y_start {
        for &x in &x_start {
            windows.push((x, y, window_size, window_size));
        }
    }
    (windows, x_start.len() as u32, y_start.len() as u32)
}

/// Reference `ImageOps` square pad: black canvas, paste at top-left.
fn square_pad(img: &RgbImage) -> RgbImage {
    let (w, h) = img.dimensions();
    let size = w.max(h);
    let mut padded = RgbImage::from_pixel(size, size, Rgb([0, 0, 0]));
    image::imageops::overlay(&mut padded, img, 0, 0);
    padded
}

/// One image split into a global view and sliding-window patches.
struct PatchedImage {
    /// Post-preprocess-size image (before the final 728x728 resize).
    global: RgbImage,
    /// Window crops in reading order (before the final 504x504 resize).
    patches: Vec<RgbImage>,
    /// Per-patch 0/1 flags: 1 = a `<patch_newline>` token follows this patch.
    newline_mask: Vec<u32>,
}

/// Step3-VL per-image transform (also the Step5-VL frontend in vLLM).
///
/// Stateless: the preprocessor config's knobs travel per call, matching the
/// `llm-multimodal` processor contract.
#[derive(Debug, Clone, Copy, Default)]
pub struct Step3VLProcessor;

/// The processor's knobs, resolved from the preprocessor config; absent
/// fields fall back to the reference `Step3VLImageProcessor` defaults.
#[derive(Debug, Clone, Copy)]
struct Step3VLParams {
    image_size: u32,
    patch_size: u32,
    num_image_feature_size: usize,
    num_patch_feature_size: usize,
    enable_patch: bool,
}

impl Default for Step3VLParams {
    /// The reference `Step3VLImageProcessor` defaults.
    fn default() -> Self {
        Self {
            image_size: DEFAULT_IMAGE_SIZE,
            patch_size: DEFAULT_PATCH_SIZE,
            num_image_feature_size: DEFAULT_NUM_IMAGE_FEATURE_SIZE,
            num_patch_feature_size: DEFAULT_NUM_PATCH_FEATURE_SIZE,
            enable_patch: true,
        }
    }
}

impl Step3VLParams {
    /// Resolve from a preprocessor config, falling back to the reference
    /// defaults for absent fields.
    fn from_config(config: &PreProcessorConfig) -> Self {
        Self {
            image_size: config.get_extra("image_size").unwrap_or(DEFAULT_IMAGE_SIZE),
            patch_size: config.get_patch_size(DEFAULT_PATCH_SIZE as usize) as u32,
            num_image_feature_size: config
                .get_extra("num_image_feature_size")
                .unwrap_or(DEFAULT_NUM_IMAGE_FEATURE_SIZE),
            num_patch_feature_size: config
                .get_extra("num_patch_feature_size")
                .unwrap_or(DEFAULT_NUM_PATCH_FEATURE_SIZE),
            enable_patch: config.get_extra("enable_patch").unwrap_or(true),
        }
    }

    /// Reference `ImagePatcher.get_num_patches`: (patch count, newline count).
    fn get_num_patches(&self, width: u32, height: u32) -> (usize, usize) {
        let (width, height) = get_image_size_for_padding(width, height);
        let (width, height) = get_image_size_for_preprocess(width, height);
        let window_size = determine_window_size(width.max(height), width.min(height));
        if window_size == 0 || !self.enable_patch {
            return (0, 0);
        }
        let (width, height) = get_image_size_for_crop(width, height, window_size);
        let (windows, x_num, _y_num) = slide_window(width, height, window_size);
        let n = windows.len();
        let mut full_rows = (n - 1) / x_num as usize + 1;
        if n % x_num as usize == 0 {
            full_rows -= 1;
        }
        (n, full_rows)
    }

    /// Reference `ImagePatcher.__call__`: split one decoded image into its
    /// global view and window patches with the newline mask.
    fn patch_one(&self, image: &DynamicImage) -> Result<PatchedImage, TransformError> {
        let rgb = image.to_rgb8();
        let (mut width, mut height) = rgb.dimensions();

        let rgb = match get_image_size_for_padding(width, height) {
            (pw, ph) if (pw, ph) != (width, height) => {
                width = pw;
                height = ph;
                square_pad(&rgb)
            }
            _ => rgb,
        };

        let (new_width, new_height) = get_image_size_for_preprocess(width, height);
        if new_width == 0 || new_height == 0 {
            return Err(TransformError::ShapeError(format!(
                "step3_vl preprocess size collapsed to {new_width}x{new_height} \
                 for input {width}x{height}"
            )));
        }
        let global = resize(
            &DynamicImage::ImageRgb8(rgb),
            new_width,
            new_height,
            FilterType::Triangle,
        )
        .to_rgb8();

        let window_size =
            determine_window_size(new_width.max(new_height), new_width.min(new_height));
        if window_size == 0 || !self.enable_patch {
            return Ok(PatchedImage {
                global,
                patches: Vec::new(),
                newline_mask: Vec::new(),
            });
        }

        let (crop_width, crop_height) = get_image_size_for_crop(new_width, new_height, window_size);
        // Reference quirk: the resize decision compares against the post-pad
        // dimensions, not the post-preprocess ones.
        let img_for_crop = if (crop_width, crop_height) != (width, height) {
            resize(
                &DynamicImage::ImageRgb8(global.clone()),
                crop_width,
                crop_height,
                FilterType::Triangle,
            )
            .to_rgb8()
        } else {
            global.clone()
        };

        let (windows, x_num, _y_num) = slide_window(crop_width, crop_height, window_size);
        let mut patches = Vec::with_capacity(windows.len());
        let mut newline_mask = vec![0u32; windows.len()];
        for (patch_id, &(x, y, pw, ph)) in windows.iter().enumerate() {
            patches.push(image::imageops::crop_imm(&img_for_crop, x, y, pw, ph).to_image());
            if (patch_id as u32 + 1).is_multiple_of(x_num) {
                newline_mask[patch_id] = 1;
            }
        }
        if let Some(last) = newline_mask.last_mut() {
            *last = 0;
        }

        Ok(PatchedImage {
            global,
            patches,
            newline_mask,
        })
    }
}

impl VisionPreProcessor for Step3VLProcessor {
    fn default_mean(&self) -> [f64; 3] {
        PreProcessorConfig::CLIP_MEAN
    }

    fn default_std(&self) -> [f64; 3] {
        PreProcessorConfig::CLIP_STD
    }

    fn preprocess(
        &self,
        images: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if images.is_empty() {
            return Err(TransformError::EmptyBatch);
        }
        let params = Step3VLParams::from_config(config);
        let mean = self.default_mean();
        let std = self.default_std();
        let image_size = params.image_size;
        let patch_size = params.patch_size;

        let mut globals = Vec::with_capacity(images.len());
        let mut patch_data: Vec<f32> = Vec::new();
        let mut newline_mask: Vec<u32> = Vec::new();
        let mut num_patches: Vec<i64> = Vec::with_capacity(images.len());
        let mut feature_token_counts = Vec::with_capacity(images.len());
        let mut item_sizes = Vec::with_capacity(images.len());

        for image in images {
            let patched = params.patch_one(image)?;
            let global = resize(
                &DynamicImage::ImageRgb8(patched.global),
                image_size,
                image_size,
                FilterType::Triangle,
            );
            globals.push(to_tensor_and_normalize(&global, &mean, &std));

            for patch in &patched.patches {
                let patch = resize(
                    &DynamicImage::ImageRgb8(patch.clone()),
                    patch_size,
                    patch_size,
                    FilterType::Triangle,
                );
                let tensor = to_tensor_and_normalize(&patch, &mean, &std);
                patch_data.extend_from_slice(
                    tensor.as_slice().expect("planar tensor is standard-layout by construction"),
                );
            }

            num_patches.push(patched.patches.len() as i64);
            newline_mask.extend_from_slice(&patched.newline_mask);
            feature_token_counts.push(
                params.num_image_feature_size
                    + params.num_patch_feature_size * patched.patches.len(),
            );
            item_sizes.push((image.width(), image.height()));
        }

        let encoder_input = stack_batch(&globals)?;
        let total_patches = patch_data.len() / (3 * patch_size as usize * patch_size as usize);
        let patch_pixel_values = ModelSpecificValue::Tensor {
            data: patch_data,
            shape: vec![total_patches, 3, patch_size as usize, patch_size as usize],
        };

        Ok(
            PreprocessedEncoderInputs::new(encoder_input, feature_token_counts, item_sizes)
                .with_extra("patch_pixel_values", patch_pixel_values)
                .with_extra("num_patches", ModelSpecificValue::int_1d(num_patches))
                .with_extra(
                    "patch_newline_mask",
                    ModelSpecificValue::uint_1d(newline_mask),
                ),
        )
    }

    fn calculate_num_tokens(
        &self,
        width: u32,
        height: u32,
        config: &PreProcessorConfig,
    ) -> Result<usize, TransformError> {
        let params = Step3VLParams::from_config(config);
        let (num_patches, num_newlines) = params.get_num_patches(width, height);
        Ok(num_patches * (params.num_patch_feature_size + 2)
            + params.num_image_feature_size
            + 2
            + num_newlines)
    }

    fn model_name(&self) -> &'static str {
        "step3_vl"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(width: u32, height: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_pixel(width, height, Rgb([120, 30, 200])))
    }

    /// Preprocessor config whose `enable_patch` is false.
    fn no_patch_config() -> PreProcessorConfig {
        serde_json::from_value(serde_json::json!({"enable_patch": false})).unwrap()
    }

    #[test]
    fn determine_window_size_boundaries() {
        // Long side below 728: window only for ratios above 1.5.
        assert_eq!(determine_window_size(700, 400), 400); // 1.75 > 1.5
        assert_eq!(determine_window_size(700, 500), 0); // 1.4 <= 1.5
        assert_eq!(determine_window_size(727, 484), 484); // ratio ~1.502 > 1.5
        assert_eq!(determine_window_size(727, 485), 0); // ratio ~1.499 <= 1.5
        assert_eq!(determine_window_size(727, 400), 400);
        // Long side at/above 728: 504 unless the ratio exceeds 4.
        assert_eq!(determine_window_size(728, 728), 504);
        assert_eq!(determine_window_size(1000, 300), 504); // 3.33 <= 4
        assert_eq!(determine_window_size(800, 150), 150); // 5.33 > 4 -> min(short, 504)
        assert_eq!(determine_window_size(800, 600), 504);
        assert_eq!(determine_window_size(4000, 900), 504); // 4.44 > 4 -> min(900, 504)
    }

    #[test]
    fn crop_size_rounding_uses_02_threshold() {
        // ratio < 1 keeps the dimension.
        assert_eq!(crop_dim(500, 504), 500);
        // Exact multiples are unchanged.
        assert_eq!(crop_dim(1008, 504), 1008);
        // Fractional part <= 0.2 rounds down.
        assert_eq!(crop_dim(554, 504), 504); // 1.099...
        assert_eq!(crop_dim(604, 504), 504); // decimal = 0.1984...
        // Fractional part > 0.2 adds a whole window.
        assert_eq!(crop_dim(605, 504), 1008); // decimal = 0.2004...
        assert_eq!(crop_dim(1500, 504), 1512); // 2.976 -> 3 windows
        assert_eq!(get_image_size_for_crop(605, 554, 504), (1008, 504));
    }

    #[test]
    fn slide_window_counts_and_last_window_clamping() {
        // Extent smaller than the window: a single full-size window at 0.
        let (windows, x_num, y_num) = slide_window(500, 300, 504);
        assert_eq!(windows, vec![(0, 0, 504, 504)]);
        assert_eq!((x_num, y_num), (1, 1));

        // Exact multiple: no clamping needed.
        let (windows, x_num, y_num) = slide_window(1008, 504, 504);
        assert_eq!(windows, vec![(0, 0, 504, 504), (504, 0, 504, 504)]);
        assert_eq!((x_num, y_num), (2, 1));

        // Overhang: the last window is pulled back to end at the extent.
        let (windows, x_num, _y_num) = slide_window(1010, 504, 504);
        assert_eq!(x_num, 3);
        assert_eq!(
            windows,
            vec![(0, 0, 504, 504), (504, 0, 504, 504), (506, 0, 504, 504)]
        );

        // y-outer, x-inner ordering.
        let (windows, _, _) = slide_window(1008, 1008, 504);
        assert_eq!(
            windows,
            vec![
                (0, 0, 504, 504),
                (504, 0, 504, 504),
                (0, 504, 504, 504),
                (504, 504, 504, 504),
            ]
        );
    }

    #[test]
    fn padding_and_preprocess_sizing() {
        // Tiny + extreme aspect -> square pad to the long side.
        assert_eq!(get_image_size_for_padding(20, 100), (100, 100));
        assert_eq!(get_image_size_for_padding(31, 200), (200, 200));
        // Extreme aspect but not tiny -> no pad.
        assert_eq!(get_image_size_for_padding(32, 1000), (32, 1000));
        // Ordinary sizes pass through.
        assert_eq!(get_image_size_for_padding(640, 480), (640, 480));

        // Longer side capped at 3024 with truncating scale.
        assert_eq!(get_image_size_for_preprocess(4000, 2000), (3024, 1512));
        assert_eq!(get_image_size_for_preprocess(3024, 100), (3024, 100));
        assert_eq!(get_image_size_for_preprocess(640, 480), (640, 480));
    }

    #[test]
    fn square_pad_pastes_top_left_on_black() {
        let img = RgbImage::from_pixel(3, 2, Rgb([9, 8, 7]));
        let padded = square_pad(&img);
        assert_eq!(padded.dimensions(), (3, 3));
        assert_eq!(*padded.get_pixel(0, 0), Rgb([9, 8, 7]));
        assert_eq!(*padded.get_pixel(2, 1), Rgb([9, 8, 7]));
        assert_eq!(*padded.get_pixel(0, 2), Rgb([0, 0, 0]));
    }

    #[test]
    fn newline_mask_marks_row_ends_and_unflags_trailing() {
        // Single column, two rows: the first patch ends a row, the last patch's
        // trailing newline flag is popped.
        let patched = Step3VLParams::default().patch_one(&solid(504, 1008)).unwrap();
        assert_eq!(patched.patches.len(), 2);
        assert_eq!(patched.newline_mask, vec![1, 0]);

        // 2x2 grid: only the end of the first row keeps its flag.
        let patched = Step3VLParams::default().patch_one(&solid(1008, 1008)).unwrap();
        assert_eq!(patched.patches.len(), 4);
        assert_eq!(patched.newline_mask, vec![0, 1, 0, 0]);

        // Small image: window size 0, no patches at all.
        let patched = Step3VLParams::default().patch_one(&solid(400, 300)).unwrap();
        assert!(patched.patches.is_empty());
        assert!(patched.newline_mask.is_empty());
    }

    #[test]
    fn calculate_num_tokens_matches_reference_math() {
        let processor = Step3VLProcessor;

        // No patches: just the global span (169 + 2).
        assert_eq!(
            processor
                .calculate_num_tokens(504, 504, &PreProcessorConfig::default())
                .unwrap(),
            171
        );
        assert_eq!(
            processor
                .calculate_num_tokens(400, 300, &PreProcessorConfig::default())
                .unwrap(),
            171
        );

        // 2x2 grid: 4 * (81 + 2) + 1 newline + (169 + 2).
        assert_eq!(
            processor
                .calculate_num_tokens(1008, 1008, &PreProcessorConfig::default())
                .unwrap(),
            504
        );

        // Single column, two rows: 2 * 83 + 1 newline + 171.
        assert_eq!(
            processor
                .calculate_num_tokens(504, 1008, &PreProcessorConfig::default())
                .unwrap(),
            338
        );

        // Wide strip: ratio > 4 -> window = short side (300), crop rounds
        // 2000/300 = 6.67 up to 7 windows; trailing newline popped -> 0 rows.
        assert_eq!(
            processor
                .calculate_num_tokens(2000, 300, &PreProcessorConfig::default())
                .unwrap(),
            752
        );

        // Consistent with the per-image patch counts.
        let (n, rows) = Step3VLParams::default().get_num_patches(1008, 1008);
        assert_eq!((n, rows), (4, 1));
    }

    #[test]
    fn enable_patch_false_disables_patching() {
        let params = Step3VLParams::from_config(&no_patch_config());
        assert_eq!(
            Step3VLProcessor.calculate_num_tokens(1008, 1008, &no_patch_config()).unwrap(),
            171
        );
        let patched = params.patch_one(&solid(1008, 1008)).unwrap();
        assert!(patched.patches.is_empty());
    }

    #[test]
    fn preprocess_emits_engine_contract() {
        let processor = Step3VLProcessor;
        let images = [solid(1008, 1008), solid(400, 300)];
        let out = processor.preprocess(&images, &PreProcessorConfig::default()).unwrap();

        // Global views stack to [B, 3, 728, 728].
        assert_eq!(out.encoder_input.shape(), &[2, 3, 728, 728]);
        // Embed positions only: 169 + 81 * num_patches per image.
        assert_eq!(out.feature_token_counts, vec![169 + 81 * 4, 169]);
        assert_eq!(out.item_sizes, vec![(1008, 1008), (400, 300)]);

        // Flat patch tensor across images.
        let ModelSpecificValue::Tensor {
            data: patch_data,
            shape,
        } = &out.model_specific["patch_pixel_values"]
        else {
            panic!("patch_pixel_values must be a float tensor");
        };
        assert_eq!(shape, &[4, 3, 504, 504]);
        assert_eq!(patch_data.len(), 4 * 3 * 504 * 504);

        let ModelSpecificValue::IntTensor { data, shape } = &out.model_specific["num_patches"]
        else {
            panic!("num_patches must be an int tensor");
        };
        assert_eq!(shape, &[2]);
        assert_eq!(data, &[4, 0]);

        let ModelSpecificValue::UintTensor { data, shape } =
            &out.model_specific["patch_newline_mask"]
        else {
            panic!("patch_newline_mask must be a uint tensor");
        };
        assert_eq!(shape, &[4]);
        assert_eq!(data, &[0, 1, 0, 0]);

        // CLIP-normalized: a solid (120, 30, 200) view lands at the normalized
        // value everywhere (resize preserves the flat field).
        let expected_r = (120.0f32 / 255.0 - 0.481_454_66) / 0.268_629_55;
        assert!((out.encoder_input.as_f32()[[0, 0, 0, 0]] - expected_r).abs() < 1e-3);
        assert!((patch_data[0] - expected_r).abs() < 1e-3);
    }

    #[test]
    fn preprocess_without_patches_emits_empty_patch_tensor() {
        let processor = Step3VLProcessor;
        let images = [solid(400, 300)];
        let out = processor.preprocess(&images, &PreProcessorConfig::default()).unwrap();

        assert_eq!(out.encoder_input.shape(), &[1, 3, 728, 728]);
        let ModelSpecificValue::Tensor { data, shape } = &out.model_specific["patch_pixel_values"]
        else {
            panic!("patch_pixel_values must be a float tensor");
        };
        assert_eq!(shape, &[0, 3, 504, 504]);
        assert!(data.is_empty());
        let ModelSpecificValue::UintTensor { data, .. } = &out.model_specific["patch_newline_mask"]
        else {
            panic!("patch_newline_mask must be a uint tensor");
        };
        assert!(data.is_empty());
    }

    #[test]
    fn tiny_extreme_aspect_image_is_square_padded_first() {
        let processor = Step3VLProcessor;
        // 20x100 pads to 100x100, then ratio 1 -> window size 0.
        let out = processor
            .preprocess(
                std::slice::from_ref(&solid(20, 100)),
                &PreProcessorConfig::default(),
            )
            .unwrap();
        assert_eq!(out.encoder_input.shape(), &[1, 3, 728, 728]);
        assert_eq!(out.feature_token_counts, vec![169]);
        assert_eq!(out.item_sizes, vec![(20, 100)]);
        let ModelSpecificValue::Tensor { shape, .. } = &out.model_specific["patch_pixel_values"]
        else {
            panic!("patch_pixel_values must be a float tensor");
        };
        assert_eq!(shape, &[0, 3, 504, 504]);
    }

    #[test]
    fn config_overrides_feature_sizes_and_patch_flag() {
        let config: PreProcessorConfig = serde_json::from_value(serde_json::json!({
            "image_size": 336,
            "num_image_feature_size": 144,
            "num_patch_feature_size": 64,
            "enable_patch": false
        }))
        .unwrap();
        let processor = Step3VLProcessor;

        assert_eq!(
            processor.calculate_num_tokens(1008, 1008, &config).unwrap(),
            146
        );
        let out = processor.preprocess(std::slice::from_ref(&solid(1008, 1008)), &config).unwrap();
        assert_eq!(out.encoder_input.shape(), &[1, 3, 336, 336]);
        assert_eq!(out.feature_token_counts, vec![144]);
    }
}
