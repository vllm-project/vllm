// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! MuseSpark1x Omni vision preprocessing for the NaViT encoder.
//!
//! Ports the native preprocessing from vLLM's
//! `models/muse_spark/common/processor.py`:
//!
//! - Images are converted to RGB (alpha composited over white), fit to the
//!   least-distorted patch grid within the token budget (`find_resize_target`
//!   over *token* patches of `patch_size * downsample_factor`), resized with
//!   Pillow-exact LANCZOS, and normalized as `(x / 255 - mean) / std`.
//! - Videos sample frames following the model contract (fps-based count,
//!   `num_frames_per_video` cap, frame-factor alignment, then temporal
//!   downsample alignment with last-frame padding), share one resize grid
//!   across all frames, and carry per-clip timestamps/positions for the
//!   placeholder expansion in [`super::MuseSparkSpec`].
//!
//! Because resized images/videos are variable-shape, the normalized fp32
//! pixels are emitted as one flat `[sum(3 * H' * W')]` buffer; per-item
//! element counts (`*_pixel_counts`) drive flat slicing and `*_target_hw`
//! lets the consumer reshape each item back to `[3, H', W']`.

use image::{DynamicImage, GenericImageView, RgbImage};

use llm_multimodal::{
    ModelSpecificValue, PreProcessorConfig, PreprocessedEncoderInputs, RgbFrameRef, TransformError,
    VideoClipOptions, VisionPreProcessor,
};

use super::transform::resize_lanczos_pil_rgb;

/// 80ms @ 24kHz per latent token (mirrors vLLM `WAVEFORM_PER_TOKEN`).
pub(crate) const WAVEFORM_PER_TOKEN: usize = 1920;
/// MuseSpark audio is raw 24kHz waveform (mirrors vLLM `AUDIO_TARGET_SAMPLE_RATE`).
pub(crate) const AUDIO_TARGET_SAMPLE_RATE: usize = 24_000;

pub const MUSE_SPARK_IMAGE_MEAN: [f64; 3] = [0.5; 3];
pub const MUSE_SPARK_IMAGE_STD: [f64; 3] = [0.5; 3];
/// Background an alpha channel is composited over, matching vLLM's
/// `RGBA_BACKGROUND_COLOR` (`ImageMediaIO` default).
pub const RGBA_BACKGROUND_COLOR: [u8; 3] = [255, 255, 255];

const DEFAULT_PATCH_SIZE: usize = 14;
pub(super) const DEFAULT_PATCH_TEMPORAL: usize = 2;
const DEFAULT_DOWNSAMPLE_FACTOR: usize = 2;
const DEFAULT_MIN_NUM_TOKENS: usize = 1;
const DEFAULT_MAX_NUM_IMAGE_TOKENS: usize = 4096;
const DEFAULT_MAX_NUM_VIDEO_TOKENS: usize = 4096;
const DEFAULT_VIDEO_DOWNSAMPLE_FACTOR: usize = 2;
pub(super) const DEFAULT_NUM_FRAMES_PER_VIDEO: usize = 600;
pub(super) const DEFAULT_SAMPLING_FPS: f64 = 2.0;
const DEFAULT_FRAME_FACTOR: usize = 2;
const DEFAULT_MAX_NUM_TOKENS_PER_FRAME: usize = 256;

/// Configuration for image/video preprocessing.
///
/// Mirrors vLLM's `PreprocessConfig`, populated from the image/video sections
/// of the current checkpoint's `processor_config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct MuseSparkVisionParams {
    pub patch_height: usize,
    pub patch_width: usize,
    pub patch_temporal: usize,
    pub downsample_factor: usize,
    pub min_num_tokens: usize,
    pub max_num_image_tokens: usize,
    pub max_num_video_tokens: usize,
    pub use_per_video_token_budget: bool,
    pub image_mean: [f64; 3],
    pub image_std: [f64; 3],
    pub video_downsample_factor: usize,
    pub num_frames_per_video: usize,
    pub sampling_fps: Option<f64>,
    pub frame_factor: Option<usize>,
    pub max_num_tokens_per_frame: usize,
}

impl Default for MuseSparkVisionParams {
    fn default() -> Self {
        Self {
            patch_height: DEFAULT_PATCH_SIZE,
            patch_width: DEFAULT_PATCH_SIZE,
            patch_temporal: DEFAULT_PATCH_TEMPORAL,
            downsample_factor: DEFAULT_DOWNSAMPLE_FACTOR,
            min_num_tokens: DEFAULT_MIN_NUM_TOKENS,
            max_num_image_tokens: DEFAULT_MAX_NUM_IMAGE_TOKENS,
            max_num_video_tokens: DEFAULT_MAX_NUM_VIDEO_TOKENS,
            use_per_video_token_budget: false,
            image_mean: MUSE_SPARK_IMAGE_MEAN,
            image_std: MUSE_SPARK_IMAGE_STD,
            video_downsample_factor: DEFAULT_VIDEO_DOWNSAMPLE_FACTOR,
            num_frames_per_video: DEFAULT_NUM_FRAMES_PER_VIDEO,
            sampling_fps: Some(DEFAULT_SAMPLING_FPS),
            frame_factor: Some(DEFAULT_FRAME_FACTOR),
            max_num_tokens_per_frame: DEFAULT_MAX_NUM_TOKENS_PER_FRAME,
        }
    }
}

impl MuseSparkVisionParams {
    /// Spatial resolution of each token after downsampling.
    pub fn token_patch_height(&self) -> usize {
        self.patch_height * self.downsample_factor
    }

    /// Spatial resolution of each token after downsampling.
    pub fn token_patch_width(&self) -> usize {
        self.patch_width * self.downsample_factor
    }

    fn apply_preprocessor_config(&self, config: &PreProcessorConfig) -> Self {
        let mut params = self.clone();
        let patch = config.get_patch_size(self.patch_height);
        if config.patch_size.is_some() {
            params.patch_height = patch;
            params.patch_width = patch;
        }
        if let Some(mean) = three(config.image_mean.as_deref()) {
            params.image_mean = mean;
        }
        if let Some(std) = three(config.image_std.as_deref()) {
            params.image_std = std;
        }
        if let Some(value) = config.temporal_patch_size {
            params.patch_temporal = value;
            params.video_downsample_factor = value;
            params.frame_factor = Some(value);
        }
        if let Some(value) = config.merge_size {
            params.downsample_factor = value;
        }
        if let Some(value) = config.get_extra::<usize>("max_image_tokens") {
            params.max_num_image_tokens = value;
        }
        if let Some(value) = config.get_extra::<usize>("num_frames") {
            params.num_frames_per_video = value;
        }
        if let Some(value) = config.get_extra::<f64>("fps") {
            params.sampling_fps = Some(value);
        }
        if let Some(value) = config.get_extra::<usize>("max_video_frame_tokens") {
            params.max_num_tokens_per_frame = value;
        }
        params
    }
}

fn three(values: Option<&[f64]>) -> Option<[f64; 3]> {
    let [a, b, c] = values? else { return None };
    Some([*a, *b, *c])
}

/// Python `round()`: round half to even (banker's rounding).
fn py_round(x: f64) -> f64 {
    let floor = x.floor();
    let frac = x - floor;
    if frac < 0.5 {
        floor
    } else if frac > 0.5 {
        floor + 1.0
    } else if (floor as i64) % 2 == 0 {
        floor
    } else {
        floor + 1.0
    }
}

/// Find the patch grid (nph, npw) minimizing aspect-ratio distortion.
///
/// Mirrors vLLM `_least_distortion_fit`; returns `None` when no candidate
/// grid satisfies the token bounds (the caller falls back to
/// [`least_distance_fit`]).
#[expect(
    clippy::too_many_arguments,
    reason = "direct port of the reference signature"
)]
fn least_distortion_fit(
    i_nph: f64,
    i_npw: f64,
    img_h: usize,
    img_w: usize,
    patch_h: usize,
    patch_w: usize,
    min_num_tokens: usize,
    max_num_tokens: usize,
) -> Option<(usize, usize)> {
    let mut candidates: Vec<(i64, i64)> = Vec::with_capacity(4);
    for nph in [i_nph.floor() as i64, i_nph.ceil() as i64] {
        for npw in [i_npw.floor() as i64, i_npw.ceil() as i64] {
            if !candidates.contains(&(nph, npw)) {
                candidates.push((nph, npw));
            }
        }
    }
    candidates.sort_unstable();
    // Deviation from the reference: on exact distortion ties the Python
    // implementation picks the first element in CPython *set iteration order*,
    // which is hash-dependent and not stable across interpreter versions.
    // We deterministically pick the smallest (nph, npw) in sorted order; tied
    // candidates distort the aspect ratio identically, so this only changes
    // the token count within the budget (e.g. 3x3 vs 4x4 for a 100x100 image).
    candidates
        .into_iter()
        .filter(|&(nph, npw)| {
            nph > 0
                && npw > 0
                && (min_num_tokens..=max_num_tokens).contains(&((nph as usize) * (npw as usize)))
        })
        .map(|(nph, npw)| (nph as usize, npw as usize))
        .min_by(|&(nph_a, npw_a), &(nph_b, npw_b)| {
            let distortion = |nph: usize, npw: usize| {
                ((nph * patch_h) as f64 / (npw * patch_w) as f64 - img_h as f64 / img_w as f64)
                    .abs()
            };
            distortion(nph_a, npw_a)
                .partial_cmp(&distortion(nph_b, npw_b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

/// Fallback: closest grid point between the min/max token hyperbolas.
///
/// Mirrors vLLM `_least_distance_fit`. The ideal width is rounded (Python
/// `round`, i.e. half-to-even) *before* clamping: truncating after the clip
/// is off-by-one vs the reference for fractional parts > 0.5.
fn least_distance_fit(
    i_nph: f64,
    i_npw: f64,
    min_num_tokens: usize,
    max_num_tokens: usize,
    patch_h: usize,
    patch_w: usize,
) -> Option<(usize, usize)> {
    let rounded_npw = py_round(i_npw);
    let mut best: Option<(usize, usize, f64)> = None;
    for nph in 1..=max_num_tokens {
        let npw_upper = (max_num_tokens / nph) as f64;
        let npw_lower = min_num_tokens.div_ceil(nph) as f64;
        if npw_upper < npw_lower {
            continue;
        }
        let npw = rounded_npw.clamp(npw_lower, npw_upper);
        let l2 = ((nph as f64 - i_nph) * patch_h as f64).powi(2)
            + ((npw - i_npw) * patch_w as f64).powi(2);
        // `np.argmin` semantics: keep the first minimal element.
        if best.is_none_or(|(_, _, best_l2)| l2 < best_l2) {
            best = Some((nph, npw as usize, l2));
        }
    }
    best.map(|(nph, npw, _)| (nph, npw))
}

/// Resize target `(target_h, target_w)` and the patch grid `(nph, npw)`,
/// where `nph * npw` is the post-downsample token count.
pub type ResizeTarget = ((usize, usize), (usize, usize));

/// Find the target resolution best preserving the image aspect ratio.
///
/// Mirrors vLLM `find_resize_target`.
pub fn find_resize_target(
    img_h: usize,
    img_w: usize,
    token_patch_h: usize,
    token_patch_w: usize,
    min_num_tokens: usize,
    max_num_tokens: usize,
    resize_to_max_canvas: bool,
) -> Result<ResizeTarget, TransformError> {
    if img_h == 0 || img_w == 0 || token_patch_h == 0 || token_patch_w == 0 {
        return Err(TransformError::ShapeError(format!(
            "invalid resize inputs: image {img_w}x{img_h}, token patch {token_patch_w}x{token_patch_h}"
        )));
    }
    let i_nph = img_h as f64 / token_patch_h as f64;
    let i_npw = img_w as f64 / token_patch_w as f64;
    let ideal_ratio = i_npw / i_nph;

    let (i_nph, i_npw) = if resize_to_max_canvas || i_nph * i_npw > max_num_tokens as f64 {
        let nph = (max_num_tokens as f64 / ideal_ratio).sqrt();
        (nph, nph * ideal_ratio)
    } else if i_nph * i_npw < min_num_tokens as f64 {
        let nph = (min_num_tokens as f64 / ideal_ratio).sqrt();
        (nph, nph * ideal_ratio)
    } else {
        (i_nph, i_npw)
    };

    let (nph, npw) = least_distortion_fit(
        i_nph,
        i_npw,
        img_h,
        img_w,
        token_patch_h,
        token_patch_w,
        min_num_tokens,
        max_num_tokens,
    )
    .or_else(|| {
        least_distance_fit(
            i_nph,
            i_npw,
            min_num_tokens,
            max_num_tokens,
            token_patch_h,
            token_patch_w,
        )
    })
    .ok_or_else(|| {
        TransformError::ShapeError(format!(
            "resolution fitting failed for {img_w}x{img_h} image within [{min_num_tokens}, {max_num_tokens}] tokens"
        ))
    })?;

    debug_assert!((min_num_tokens..=max_num_tokens).contains(&(nph * npw)));
    Ok(((nph * token_patch_h, npw * token_patch_w), (nph, npw)))
}

/// Convert a frame to 3-channel RGB, compositing alpha over white.
///
/// Mirrors vLLM `_to_rgb`: RGBA alpha is composited over
/// [`RGBA_BACKGROUND_COLOR`] (Pillow `paste` mask blending,
/// `DIV255(v + 128)` rounding); every other mode takes `.convert("RGB")`.
/// Compositing happens *before* resize so transparent pixels don't bleed
/// into LANCZOS interpolation.
pub fn to_rgb_composited(image: &DynamicImage) -> RgbImage {
    match image {
        DynamicImage::ImageRgba8(rgba) => {
            let (w, h) = rgba.dimensions();
            let mut out = RgbImage::new(w, h);
            let [bg_r, bg_g, bg_b] = RGBA_BACKGROUND_COLOR;
            for (px, out_px) in rgba.pixels().zip(out.pixels_mut()) {
                let a = px[3] as u32;
                for c in 0..3 {
                    let bg = [bg_r, bg_g, bg_b][c] as u32;
                    let v = bg * (255 - a) + px[c] as u32 * a + 128;
                    out_px[c] = (((v >> 8) + v) >> 8) as u8;
                }
            }
            out
        }
        _ => image.to_rgb8(),
    }
}

/// Normalize resized uint8 RGB pixels into `[C, H, W]` planar fp32 with
/// `(x / 255 - mean) / std`, matching the reference's op order.
fn append_normalized_planes(rgb: &RgbImage, mean: [f64; 3], std: [f64; 3], out: &mut Vec<f32>) {
    let pixels = (rgb.width() as usize) * (rgb.height() as usize);
    let base = out.len();
    out.resize(base + 3 * pixels, 0.0);
    for (i, px) in rgb.as_raw().chunks_exact(3).enumerate() {
        for c in 0..3 {
            let scaled = px[c] as f32 / 255.0;
            out[base + c * pixels + i] = (scaled - mean[c] as f32) / std[c] as f32;
        }
    }
}

/// `np.linspace(0, last, num, dtype=int)`: float64 samples truncated to int,
/// with the final sample pinned to `last`.
fn linspace_indices(last: usize, num: usize) -> Vec<usize> {
    if num == 0 {
        return Vec::new();
    }
    if num == 1 {
        return vec![0];
    }
    let step = last as f64 / (num - 1) as f64;
    (0..num)
        .map(|i| {
            if i == num - 1 {
                last
            } else {
                (i as f64 * step) as usize
            }
        })
        .collect()
}

/// MuseSpark1x Omni NaViT image/video preprocessor.
///
/// Owns the parameters resolved from the checkpoint's processor config at
/// construction; request methods take no per-call config.
#[derive(Debug, Clone, Default)]
pub struct MuseSparkVisionProcessor {
    params: MuseSparkVisionParams,
}

impl MuseSparkVisionProcessor {
    pub fn with_params(params: MuseSparkVisionParams) -> Self {
        Self { params }
    }

    /// Build a processor with the checkpoint's image/video processor config
    /// baked in.
    pub fn from_config(config: &PreProcessorConfig) -> Self {
        Self::with_params(MuseSparkVisionParams::default().apply_preprocessor_config(config))
    }

    /// Preprocess one image for the NaViT encoder.
    ///
    /// Returns the normalized `[3, H', W']` planar pixels and the
    /// post-downsample token count, mirroring vLLM `preprocess_image`.
    fn preprocess_image(&self, image: &DynamicImage) -> Result<(RgbImage, usize), TransformError> {
        let params = &self.params;
        let rgb = to_rgb_composited(image);
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        let ((target_h, target_w), _) = find_resize_target(
            h,
            w,
            params.token_patch_height(),
            params.token_patch_width(),
            params.min_num_tokens,
            params.max_num_image_tokens,
            false,
        )?;
        let resized = resize_lanczos_pil_rgb(
            rgb.as_raw(),
            w as u32,
            h as u32,
            target_w as u32,
            target_h as u32,
        )?;
        let num_tokens =
            (target_h / params.token_patch_height()) * (target_w / params.token_patch_width());
        Ok((resized, num_tokens))
    }

    /// Shared video pipeline over RGB frames, mirroring vLLM
    /// `_preprocess_single_video` + `preprocess_video` +
    /// `compute_video_clip_metadata`.
    ///
    /// Frames arrive pre-sampled by the media connector at the model's
    /// `sampling_fps` rate, so the connector output count already equals the
    /// model's FPS-based target; here we only apply the
    /// `num_frames_per_video` cap and the frame-factor / temporal-downsample
    /// alignment.
    ///
    /// Frame `i` (index into the incoming, connector-sampled frames) is
    /// stamped at `i / sampling_fps`; this assumes uniform sampling at the
    /// model's configured fps (the connector default is 2 fps, matching the
    /// model's `video_processor.fps: 2.0`), like Qwen3-VL's timestamp
    /// synthesis.
    fn preprocess_video_frames(
        &self,
        frames: &[RgbImage],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        let params = &self.params;
        if frames.is_empty() {
            return Err(TransformError::EmptyBatch);
        }
        let sample_fps = params.sampling_fps.unwrap_or(DEFAULT_SAMPLING_FPS);
        if !sample_fps.is_finite() || sample_fps <= 0.0 {
            return Err(TransformError::ShapeError(format!(
                "MuseSpark video sampling_fps must be finite and positive, got {sample_fps}"
            )));
        }

        // Frame counts must stay divisible by both the frame factor and the
        // temporal patch size.
        let ff = params.frame_factor.unwrap_or(DEFAULT_FRAME_FACTOR).max(params.patch_temporal);
        let mut target = frames.len().min(params.num_frames_per_video);
        target = ff.max(target / ff * ff);

        // Subsample only when the connector produced more frames than the
        // model contract allows; otherwise keep the frames as-is.
        let mut indices: Vec<usize> = if frames.len() > target {
            linspace_indices(frames.len() - 1, target)
        } else {
            (0..frames.len()).collect()
        };
        let mut timestamps: Vec<f64> = indices.iter().map(|&i| i as f64 / sample_fps).collect();

        // Align to the temporal downsample factor: truncate, or pad with the
        // last frame when fewer frames than the factor remain.
        let vds = params.video_downsample_factor;
        let mut aligned = indices.len() / vds * vds;
        if aligned == 0 {
            aligned = vds;
            let (&last_index, &last_ts) =
                indices.last().zip(timestamps.last()).ok_or(TransformError::EmptyBatch)?;
            indices.resize(aligned, last_index);
            timestamps.resize(aligned, last_ts);
        }
        indices.truncate(aligned);
        timestamps.truncate(aligned);

        // Effective per-frame token budget.
        let max_tokens_per_frame = if params.max_num_tokens_per_frame > 0 {
            params.max_num_tokens_per_frame
        } else if params.use_per_video_token_budget {
            let num_temporal_groups = aligned / params.patch_temporal;
            params
                .max_num_video_tokens
                .checked_div(num_temporal_groups)
                .unwrap_or(params.max_num_video_tokens)
        } else {
            params.max_num_video_tokens
        };

        // One shared resize grid for all frames, from frame 0's dimensions.
        let first = &frames[indices[0]];
        let (frame_w, frame_h) = (first.width() as usize, first.height() as usize);
        let ((target_h, target_w), (nph, npw)) = find_resize_target(
            frame_h,
            frame_w,
            params.token_patch_height(),
            params.token_patch_width(),
            params.min_num_tokens,
            max_tokens_per_frame,
            false,
        )?;
        let tokens_per_frame = nph * npw;

        let mut pixels = Vec::new();
        for &index in &indices {
            let frame = &frames[index];
            let resized = resize_lanczos_pil_rgb(
                frame.as_raw(),
                frame.width(),
                frame.height(),
                target_w as u32,
                target_h as u32,
            )?;
            append_normalized_planes(&resized, params.image_mean, params.image_std, &mut pixels);
        }

        let num_clips = aligned / params.patch_temporal;
        // Equal to `(target_h / patch_h) * (target_w / patch_w) / ds^2` from
        // the reference `compute_video_clip_metadata`, since the resize grid
        // is computed in token patches of `patch * downsample_factor`.
        let patches_per_clip = tokens_per_frame;
        let patch_count = patches_per_clip * num_clips;
        // First-frame timestamp of each clip, emitted as f32 (`FloatVec`):
        // with the uniform-fps synthesis all values are multiples of 0.5 and
        // exactly representable in f32, so the `{timestamp:.2f}` prompt text
        // still rounds exactly like the reference.
        let clip_timestamps: Vec<f32> =
            (0..num_clips).map(|i| timestamps[i * params.patch_temporal] as f32).collect();

        let pixel_count = 3 * target_h * target_w * aligned;
        let encoder_input =
            ndarray::ArrayD::from_shape_vec(vec![pixel_count], pixels).map_err(|error| {
                TransformError::ShapeError(format!(
                    "failed to create MuseSpark video buffer [{pixel_count}]: {error}"
                ))
            })?;

        Ok(PreprocessedEncoderInputs::new(
            encoder_input,
            vec![patch_count],
            vec![(frame_h as u32, frame_w as u32)],
        )
        .with_extra(
            "video_pixel_counts",
            ModelSpecificValue::int_1d(vec![pixel_count as i64]),
        )
        .with_extra(
            "video_frame_counts",
            ModelSpecificValue::int_1d(vec![aligned as i64]),
        )
        .with_extra(
            "video_target_hw",
            ModelSpecificValue::int_2d(vec![target_h as i64, target_w as i64], 1, 2),
        )
        .with_extra(
            "video_patch_counts",
            ModelSpecificValue::int_1d(vec![patch_count as i64]),
        )
        .with_extra(
            "video_clip_num_clips",
            ModelSpecificValue::int_1d(vec![num_clips as i64]),
        )
        .with_extra(
            "video_clip_patches_per_clip",
            ModelSpecificValue::int_1d(vec![patches_per_clip as i64]),
        )
        .with_extra(
            "video_clip_timestamps",
            ModelSpecificValue::FloatVec(clip_timestamps),
        ))
    }
}

impl VisionPreProcessor for MuseSparkVisionProcessor {
    fn default_mean(&self) -> [f64; 3] {
        MUSE_SPARK_IMAGE_MEAN
    }

    fn default_std(&self) -> [f64; 3] {
        MUSE_SPARK_IMAGE_STD
    }

    fn preprocess(
        &self,
        images: &[DynamicImage],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if images.is_empty() {
            return Err(TransformError::EmptyBatch);
        }
        let params = &self.params;

        let mut pixels = Vec::new();
        let mut pixel_counts = Vec::with_capacity(images.len());
        let mut patch_counts = Vec::with_capacity(images.len());
        let mut target_hw = Vec::with_capacity(images.len());
        let mut item_sizes = Vec::with_capacity(images.len());

        for image in images {
            let (resized, num_tokens) = self.preprocess_image(image)?;
            let target_h = resized.height() as usize;
            let target_w = resized.width() as usize;
            append_normalized_planes(&resized, params.image_mean, params.image_std, &mut pixels);
            pixel_counts.push(3 * target_h * target_w);
            patch_counts.push(num_tokens);
            target_hw.push((target_h as i64, target_w as i64));
            item_sizes.push(image.dimensions());
        }

        let total: usize = pixel_counts.iter().sum();
        let encoder_input =
            ndarray::ArrayD::from_shape_vec(vec![total], pixels).map_err(|error| {
                TransformError::ShapeError(format!(
                    "failed to create MuseSpark image buffer [{total}]: {error}"
                ))
            })?;
        let batch = images.len();

        Ok(
            PreprocessedEncoderInputs::new(encoder_input, patch_counts.clone(), item_sizes)
                .with_extra(
                    "image_pixel_counts",
                    ModelSpecificValue::int_1d(pixel_counts.iter().map(|&v| v as i64).collect()),
                )
                .with_extra(
                    "image_patch_counts",
                    ModelSpecificValue::int_1d(patch_counts.iter().map(|&v| v as i64).collect()),
                )
                .with_extra(
                    "image_target_hw",
                    ModelSpecificValue::int_2d(
                        target_hw.into_iter().flat_map(|(h, w)| [h, w]).collect(),
                        batch,
                        2,
                    ),
                ),
        )
    }

    fn preprocess_video(
        &self,
        frames: &[DynamicImage],
        _clip: &VideoClipOptions,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        let mut rgb_frames = Vec::with_capacity(frames.len());
        for frame in frames {
            if matches!(frame, DynamicImage::ImageRgba8(_)) {
                return Err(TransformError::ShapeError(
                    "MuseSpark video frames must be 3-channel RGB, got 4 channels (alpha): \
                     whether the alpha is straight or premultiplied is not recoverable from a \
                     bare frame, so it cannot be composited correctly here"
                        .to_string(),
                ));
            }
            rgb_frames.push(frame.to_rgb8());
        }
        self.preprocess_video_frames(&rgb_frames)
    }

    fn preprocess_video_rgb(
        &self,
        frames: &[RgbFrameRef<'_>],
        _clip: &VideoClipOptions,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        let mut rgb_frames = Vec::with_capacity(frames.len());
        for frame in frames {
            let rgb = RgbImage::from_raw(frame.width, frame.height, frame.data.to_vec())
                .ok_or_else(|| {
                    TransformError::ShapeError(format!(
                        "invalid RGB frame buffer: {} bytes for {}x{}",
                        frame.data.len(),
                        frame.width,
                        frame.height
                    ))
                })?;
            rgb_frames.push(rgb);
        }
        self.preprocess_video_frames(&rgb_frames)
    }

    fn calculate_num_tokens(&self, width: u32, height: u32) -> Result<usize, TransformError> {
        let params = &self.params;
        find_resize_target(
            height as usize,
            width as usize,
            params.token_patch_height(),
            params.token_patch_width(),
            params.min_num_tokens,
            params.max_num_image_tokens,
            false,
        )
        .map(|(_, (nph, npw))| nph * npw)
    }

    fn model_name(&self) -> &'static str {
        "muse_spark"
    }

    fn get_processed_size(&self) -> Option<(u32, u32)> {
        None
    }
}

#[cfg(test)]
mod tests {
    use image::Rgb;

    use super::*;

    fn clip() -> VideoClipOptions {
        VideoClipOptions::sampled_at(2.0)
    }

    fn image(width: u32, height: u32, color: Rgb<u8>) -> DynamicImage {
        DynamicImage::from(RgbImage::from_pixel(width, height, color))
    }

    #[test]
    fn hf_processor_sections_control_image_and_video_budgets() {
        let image_config = serde_json::from_value(serde_json::json!({
            "patch_size": 14, "merge_size": 2, "max_image_tokens": 4,
        }))
        .unwrap();
        let processor = MuseSparkVisionProcessor::from_config(&image_config);
        let output = processor.preprocess(&[image(112, 112, Rgb([128; 3]))]).unwrap();
        assert_eq!(output.feature_token_counts, vec![4]);

        let video_config = serde_json::from_value(serde_json::json!({
            "patch_size": 14, "merge_size": 2, "temporal_patch_size": 2,
            "num_frames": 600, "fps": 2.0, "max_video_frame_tokens": 256,
        }))
        .unwrap();
        let processor = MuseSparkVisionProcessor::from_config(&video_config);
        assert_eq!(processor.params.num_frames_per_video, 600);
        assert_eq!(processor.params.max_num_tokens_per_frame, 256);
        // A 448px frame fits the configured 256-patch budget.
        let frames = vec![image(448, 448, Rgb([128; 3])); 2];
        let output = processor.preprocess_video(&frames, &clip()).unwrap();
        assert_eq!(output.feature_token_counts, vec![256]);

        let video_config = serde_json::from_value(serde_json::json!({
            "patch_size": 14, "merge_size": 1, "temporal_patch_size": 4,
            "num_frames": 4, "fps": 4.0, "max_video_frame_tokens": 4,
        }))
        .unwrap();
        let processor = MuseSparkVisionProcessor::from_config(&video_config);
        let frames = vec![image(28, 28, Rgb([128; 3])); 8];
        let output = processor.preprocess_video(&frames, &clip()).unwrap();
        assert_eq!(output.feature_token_counts, vec![4]);
        assert!(
            matches!(&output.model_specific["video_frame_counts"], ModelSpecificValue::IntTensor { data, .. } if data == &[4])
        );
    }

    #[test]
    fn grid_fit_matches_reference_token_budget() {
        // 448x448 with 28px token patches fits exactly: 16x16 = 256 tokens.
        let ((h, w), (nph, npw)) = find_resize_target(448, 448, 28, 28, 1, 4096, false).unwrap();
        assert_eq!(((h, w), (nph, npw)), ((448, 448), (16, 16)));

        // A tiny image is scaled up to the minimum token count.
        let ((h, w), (nph, npw)) = find_resize_target(1, 1, 28, 28, 1, 4096, false).unwrap();
        assert_eq!((h, w), (28, 28));
        assert_eq!(nph * npw, 1);

        // A huge image is clamped to the max token budget, aspect preserved.
        let ((_, _), (nph, npw)) = find_resize_target(4096, 8192, 28, 28, 1, 4096, false).unwrap();
        assert!(nph * npw <= 4096);
        let distortion = ((nph * 28) as f64 / (npw * 28) as f64 - 4096.0 / 8192.0).abs();
        assert!(distortion < 0.1);
    }

    /// Golden grid-fit values produced by the vLLM reference
    /// (`models/muse_spark/common/processor.py::find_resize_target`) with
    /// 28px token patches, min 1 token, max 4096 (image) / 144 (video frame)
    /// tokens. Exact distortion ties resolve in CPython hash order there; the
    /// `100x100` case is such a tie and asserts this port's deterministic
    /// (sorted-order) choice instead.
    #[test]
    fn grid_fit_matches_vllm_reference() {
        // (img_h, img_w), (target_h, target_w), (nph, npw)
        type Case = ((usize, usize), (usize, usize), (usize, usize));
        let cases: &[Case] = &[
            ((1, 1), (28, 28), (1, 1)),
            ((28, 28), (28, 28), (1, 1)),
            ((112, 56), (112, 56), (4, 2)),
            ((448, 448), (448, 448), (16, 16)),
            // Tie between (3,3) and (4,4): reference picks (4,4) via CPython
            // hash order; we deterministically pick (3,3).
            ((100, 100), (84, 84), (3, 3)),
            ((1000, 337), (1008, 336), (36, 12)),
            ((3333, 1111), (3080, 1036), (110, 37)),
            ((4096, 8192), (1260, 2520), (45, 90)),
            ((55, 3000), (56, 3024), (2, 108)),
            ((719, 1280), (728, 1288), (26, 46)),
            ((2, 2000), (28, 2016), (1, 72)),
            ((2000, 2), (2016, 28), (72, 1)),
        ];
        for &((h, w), target, grid) in cases {
            assert_eq!(
                find_resize_target(h, w, 28, 28, 1, 4096, false).unwrap(),
                (target, grid),
                "image budget mismatch for {h}x{w}"
            );
        }

        let video_cases: &[Case] = &[
            ((448, 448), (336, 336), (12, 12)),
            // Tie between (3,3) and (4,4), see above.
            ((100, 100), (84, 84), (3, 3)),
            ((1000, 337), (560, 196), (20, 7)),
            ((3333, 1111), (560, 196), (20, 7)),
            ((4096, 8192), (224, 448), (8, 16)),
            ((55, 3000), (28, 2464), (1, 88)),
            ((719, 1280), (252, 448), (9, 16)),
        ];
        for &((h, w), target, grid) in video_cases {
            assert_eq!(
                find_resize_target(h, w, 28, 28, 1, 144, false).unwrap(),
                (target, grid),
                "video budget mismatch for {h}x{w}"
            );
        }
    }

    #[test]
    fn py_round_is_bankers_rounding() {
        assert_eq!(py_round(2.5), 2.0);
        assert_eq!(py_round(3.5), 4.0);
        assert_eq!(py_round(-2.5), -2.0);
        assert_eq!(py_round(2.4), 2.0);
        assert_eq!(py_round(2.6), 3.0);
    }

    #[test]
    fn linspace_indices_match_numpy() {
        // np.linspace(0, 99, 8, dtype=int) — numpy 2.3.
        assert_eq!(linspace_indices(99, 8), vec![0, 14, 28, 42, 56, 70, 84, 99]);
        // np.linspace(0, 4, 4, dtype=int)
        assert_eq!(linspace_indices(4, 4), vec![0, 1, 2, 4]);
        assert_eq!(linspace_indices(9, 1), vec![0]);
    }

    #[test]
    fn alpha_is_composited_over_white_with_pil_rounding() {
        // Fully transparent black becomes white; 50% alpha blends per
        // Pillow's DIV255(v + 128).
        let mut rgba = image::RgbaImage::new(2, 1);
        rgba.put_pixel(0, 0, image::Rgba([0, 0, 0, 0]));
        rgba.put_pixel(1, 0, image::Rgba([0, 0, 0, 128]));
        let rgb = to_rgb_composited(&DynamicImage::ImageRgba8(rgba));
        assert_eq!(rgb.get_pixel(0, 0).0, [255, 255, 255]);
        // Pillow DIV255: tmp = bg*(255-a) + fg*a + 128; ((tmp>>8)+tmp)>>8.
        // Verified against Pillow 12.3: paste(mask=128) of black on white -> 127.
        let v = 255u32 * (255 - 128) + 128;
        let expected = (((v >> 8) + v) >> 8) as u8;
        assert_eq!(rgb.get_pixel(1, 0).0, [expected, expected, expected]);
        assert_eq!(expected, 127);
    }

    #[test]
    fn preprocess_outputs_flat_pixels_with_patch_counts() {
        let processor = MuseSparkVisionProcessor::default();
        let result = processor.preprocess(&[image(112, 56, Rgb([255, 255, 255]))]).unwrap();

        // 112x56 -> grid 4x2 of 28px token patches -> 112x56 unchanged.
        assert_eq!(result.feature_token_counts, vec![8]);
        assert_eq!(result.encoder_input.shape(), &[3 * 112 * 56]);
        // White pixels normalize to (1.0 - 0.5) / 0.5 = 1.0.
        assert!(result.encoder_input.as_f32().iter().all(|&v| (v - 1.0).abs() < 1e-6));
        assert!(matches!(
            result.model_specific.get("image_patch_counts"),
            Some(ModelSpecificValue::IntTensor { data, .. }) if data == &vec![8]
        ));
        assert!(matches!(
            result.model_specific.get("image_pixel_counts"),
            Some(ModelSpecificValue::IntTensor { data, .. }) if data == &vec![3 * 112 * 56]
        ));
        assert!(matches!(
            result.model_specific.get("image_target_hw"),
            Some(ModelSpecificValue::IntTensor { data, shape }) if data == &vec![56, 112] && shape == &vec![1, 2]
        ));
    }

    #[test]
    fn calculate_num_tokens_matches_preprocess() {
        let processor = MuseSparkVisionProcessor::default();
        let tokens = processor.calculate_num_tokens(112, 56).unwrap();
        assert_eq!(tokens, 8);
    }

    #[test]
    fn video_aligns_frames_and_reports_clip_metadata() {
        let processor = MuseSparkVisionProcessor::default();
        // 5 frames of 112x56 -> aligned down to 4 frames (vds=2), 2 clips.
        let frames: Vec<DynamicImage> = (0..5).map(|_| image(112, 56, Rgb([0, 0, 0]))).collect();
        let result = processor.preprocess_video(&frames, &clip()).unwrap();

        // tokens_per_frame: 256-token budget keeps 4x2 grid = 8; 2 clips.
        assert_eq!(result.feature_token_counts, vec![16]);
        assert_eq!(result.encoder_input.shape(), &[4 * 3 * 56 * 112]);
        assert!(matches!(
            result.model_specific.get("video_clip_num_clips"),
            Some(ModelSpecificValue::IntTensor { data, .. }) if data == &vec![2]
        ));
        assert!(matches!(
            result.model_specific.get("video_clip_patches_per_clip"),
            Some(ModelSpecificValue::IntTensor { data, .. }) if data == &vec![8]
        ));
        assert!(matches!(
            result.model_specific.get("video_clip_timestamps"),
            Some(ModelSpecificValue::FloatVec(ts)) if ts == &vec![0.0, 1.0]
        ));
    }

    #[test]
    fn video_pads_single_frame_to_downsample_factor() {
        let processor = MuseSparkVisionProcessor::default();
        let frames = vec![image(112, 56, Rgb([0, 0, 0]))];
        let result = processor.preprocess_video(&frames, &clip()).unwrap();

        // One frame is padded (last frame repeated) to vds=2 -> one clip.
        assert_eq!(result.encoder_input.shape(), &[2 * 3 * 56 * 112]);
        assert!(matches!(
            result.model_specific.get("video_clip_num_clips"),
            Some(ModelSpecificValue::IntTensor { data, .. }) if data == &vec![1]
        ));
        assert!(matches!(
            result.model_specific.get("video_clip_timestamps"),
            Some(ModelSpecificValue::FloatVec(ts)) if ts == &vec![0.0]
        ));
    }

    #[test]
    fn video_rejects_rgba_frames() {
        let processor = MuseSparkVisionProcessor::default();
        let frames = vec![DynamicImage::ImageRgba8(image::RgbaImage::new(56, 56))];
        assert!(processor.preprocess_video(&frames, &clip()).is_err());
    }
}
