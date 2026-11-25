// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! HY4-VL vision preprocessors (image + video).
//!
//! Port of vLLM's `transformers_utils/processors/hy_v4.py`:
//!
//! - [`HyV4VLImageProcessor`] mirrors `HYV4VLImageProcessor._preprocess`:
//!   `smart_resize` to a `patch_size * merge_size` multiple within the
//!   512^2..2048^2 pixel budget, a torchvision-exact bicubic resize, and
//!   `patchify` into row-major (h, w), channel-first 768-vectors.
//! - [`HyV4VLVideoProcessor`] mirrors `HYV4VLVideoProcessor._preprocess`:
//!   frames arrive pre-sampled by the media connector (the reference
//!   `sample_frames` rule — fps 2, 4..100 frames, midpoint-of-segment
//!   indices — is ported in `sampling.rs` for the upcoming
//!   `fetch_video_frames` integration; until it lands the connector keeps
//!   the spec-declared sampling, not re-run here); each clip is
//!   resized once per the `adaptive_resize` token budget, patchified per
//!   frame, and stamped with one timestamp per merged frame (the default
//!   branch of `HYV4VLProcessor.__call__`, `sum(indices[i:i+tau])/tau/fps`).
//!   Those synthesized stamps are a fallback: the spec's media-aware
//!   replacement path overrides the stamp VALUES with the actually-served
//!   frames' seconds whenever the decoder reported the clip's sampling,
//!   while this emission keeps shipping as the engine's `video_timestamps`
//!   kwarg and as the fallback when no sampling info is available.
//!
//! Pixels are emitted NORMALIZED (rescale by 1/255, then CLIP mean/std) as
//! f32: the Rust frontend always launches the engine with
//! `VLLM_USE_RUST_FRONTEND=1`, which forces `mm_device_do_normalize=False`, so
//! the model builds `IdentityInputNorm` and pixels would otherwise pass
//! through unnormalized. Both lanes use the CLIP stats (OPENAI_CLIP_MEAN /
//! OPENAI_CLIP_STD): in the Python serving path the on-device
//! `FusedMMInputNorm` is built from the image processor config and applies to
//! `pixel_values` and `pixel_values_videos` alike
//! (`models/hy_v4/common/vision.py`, `vl_model.py`), and the video
//! processor's own 0.5/0.5 class defaults are never exercised in serving
//! (`do_normalize` is forced off, `config/multimodal.py`). A checkpoint's
//! `image_mean`/`image_std`/`rescale_factor` still override the defaults,
//! read from the preprocessor config.
//!
//! Resampling note: the reference image processor is torchvision-backed and
//! its resize is `torch.nn.functional.interpolate(mode="bicubic",
//! antialias=True)` on a `uint8` tensor, so this port uses
//! [`resize_bicubic_torch_aa`] / [`resize_bicubic_torch_aa_rgb`], which mirror
//! that resampler bit-exactly, rather than the Pillow-exact bicubic variant.

use image::{DynamicImage, RgbImage};

use llm_multimodal::vision::transforms::{
    resize_bicubic_torch_aa, resize_bicubic_torch_aa_rgb, round_half_to_even,
};
use llm_multimodal::{
    ModelSpecificValue, PixelNorm, PreProcessorConfig, PreprocessedEncoderInputs, RgbFrameRef,
    TransformError, VideoClipOptions, VisionPreProcessor,
};

const DEFAULT_PATCH_SIZE: usize = 16;
const DEFAULT_MERGE_SIZE: usize = 2;
const DEFAULT_TEMPORAL_PATCH_SIZE: usize = 1;
const DEFAULT_SPATIAL_PATCH_SIZE: usize = 1;
/// Reference image processor `size = {"shortest_edge": 512 * 512, ...}`.
const DEFAULT_IMAGE_MIN_PIXELS: usize = 512 * 512;
/// Reference image processor `size = {..., "longest_edge": 2048 * 2048}`.
const DEFAULT_IMAGE_MAX_PIXELS: usize = 2048 * 2048;
/// Reference video processor `size = {"shortest_edge": 4096, ...}` (unused by
/// `adaptive_resize`; kept for the image-style `size` override path).
const DEFAULT_VIDEO_MIN_PIXELS: usize = 4096;
/// Reference video processor `size = {..., "longest_edge": 33554432}`.
const DEFAULT_VIDEO_MAX_PIXELS: usize = 33_554_432;
/// Reference video processor `fps = 2` (`sample_frames`); the spec declares
/// this as the media connector's sampling rate.
pub(super) const DEFAULT_SAMPLING_FPS: f64 = 2.0;
/// Reference video processor `max_frames = 100`.
const DEFAULT_MAX_FRAMES: usize = 100;
/// Reference video processor `timestamp_token_per_frame = 4`.
const DEFAULT_TIMESTAMP_TOKEN_PER_FRAME: usize = 4;
/// Reference video processor `timestamp_token_reserve = 2000`.
const DEFAULT_TIMESTAMP_TOKEN_RESERVE: usize = 2000;
/// Reference video processor `video_max_image_size = 1024`.
const DEFAULT_VIDEO_MAX_IMAGE_SIZE: usize = 1024;

/// Vision preprocessing knobs, resolved from the checkpoint's preprocessor
/// config at construction. One struct serves both processors; the video-only
/// knobs are read only by [`HyV4VLVideoProcessor`].
///
/// `use_image_newline_token` (+1 learned newline column) and
/// `cat_extra_token` belong to the model's `vision_config` in the reference
/// (`vision_token_grid`), so the spec threads them in from the model config;
/// the reference defaults apply when `vision_config` is absent.
#[derive(Debug, Clone, PartialEq)]
pub struct HyV4VisionParams {
    pub patch_size: usize,
    pub merge_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_patch_size: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub max_frames: usize,
    pub timestamp_token_per_frame: usize,
    pub timestamp_token_reserve: usize,
    pub video_max_image_size: usize,
    pub use_image_newline_token: bool,
    pub cat_extra_token: usize,
    pub image_mean: [f64; 3],
    pub image_std: [f64; 3],
    pub rescale_factor: f64,
}

impl HyV4VisionParams {
    /// Defaults of the reference `HYV4VLImageProcessor`.
    pub(super) fn image() -> Self {
        Self {
            patch_size: DEFAULT_PATCH_SIZE,
            merge_size: DEFAULT_MERGE_SIZE,
            temporal_patch_size: DEFAULT_TEMPORAL_PATCH_SIZE,
            spatial_patch_size: DEFAULT_SPATIAL_PATCH_SIZE,
            min_pixels: DEFAULT_IMAGE_MIN_PIXELS,
            max_pixels: DEFAULT_IMAGE_MAX_PIXELS,
            max_frames: DEFAULT_MAX_FRAMES,
            timestamp_token_per_frame: DEFAULT_TIMESTAMP_TOKEN_PER_FRAME,
            timestamp_token_reserve: DEFAULT_TIMESTAMP_TOKEN_RESERVE,
            video_max_image_size: DEFAULT_VIDEO_MAX_IMAGE_SIZE,
            use_image_newline_token: true,
            cat_extra_token: 0,
            image_mean: PreProcessorConfig::CLIP_MEAN,
            image_std: PreProcessorConfig::CLIP_STD,
            rescale_factor: 1.0 / 255.0,
        }
    }

    /// Defaults of the reference `HYV4VLVideoProcessor`, except the
    /// normalization stats: the reference class carries 0.5/0.5, but serving
    /// never exercises them (see the module docs), so the video lane shares
    /// the image lane's CLIP stats.
    pub(super) fn video() -> Self {
        Self {
            min_pixels: DEFAULT_VIDEO_MIN_PIXELS,
            max_pixels: DEFAULT_VIDEO_MAX_PIXELS,
            ..Self::image()
        }
    }

    pub(super) fn apply_preprocessor_config(&self, config: &PreProcessorConfig) -> Self {
        let mut params = self.clone();
        params.patch_size = config.get_patch_size(params.patch_size);
        if let Some(value) = config.merge_size {
            params.merge_size = value;
        }
        if let Some(value) = config.temporal_patch_size {
            params.temporal_patch_size = value;
        }
        if let Some(value) = config.get_extra::<usize>("spatial_patch_size") {
            params.spatial_patch_size = value;
        }
        // The reference reads the pixel budgets from its `size` dict
        // (`shortest_edge`/`longest_edge`); the flat `min_pixels`/`max_pixels`
        // kwargs override them (`_standardize_kwargs`).
        let size = config.size.as_ref();
        if let Some(value) = config
            .min_pixels
            .or_else(|| size.and_then(|size| size.get("shortest_edge")).map(|&v| v as usize))
        {
            params.min_pixels = value;
        }
        if let Some(value) = config
            .max_pixels
            .or_else(|| size.and_then(|size| size.get("longest_edge")).map(|&v| v as usize))
        {
            params.max_pixels = value;
        }
        if let Some(value) = config.get_extra::<usize>("max_frames") {
            params.max_frames = value;
        }
        if let Some(value) = config.get_extra::<usize>("timestamp_token_per_frame") {
            params.timestamp_token_per_frame = value;
        }
        if let Some(value) = config.get_extra::<usize>("timestamp_token_reserve") {
            params.timestamp_token_reserve = value;
        }
        if let Some(value) = config.get_extra::<usize>("video_max_image_size") {
            params.video_max_image_size = value;
        }
        params.image_mean = config.get_image_mean();
        params.image_std = config.get_image_std();
        params.rescale_factor = config.get_rescale_factor();
        params
    }
}

/// The per-channel map from a pixel byte to its normalized f32, with rescale
/// and normalize fused by scaling the mean and std into byte units — the
/// arithmetic of llm-multimodal's `pixel_norm` (`pub(super)` there, so
/// rebuilt here from the same public pieces) and of the reference processors'
/// `rescale_and_normalize`, so a byte maps to the same `f32` everywhere.
fn pixel_norm(params: &HyV4VisionParams) -> PixelNorm {
    let into_bytes = (1.0 / params.rescale_factor) as f32;
    PixelNorm {
        scale: 1.0,
        offset: std::array::from_fn(|channel| params.image_mean[channel] as f32 * into_bytes),
        divisor: std::array::from_fn(|channel| params.image_std[channel] as f32 * into_bytes),
        channel_block: params.patch_size * params.patch_size,
    }
}

/// Mirrors `smart_resize` in `transformers_utils/processors/hy_v4.py`: round
/// both dimensions to a `factor` multiple (Python `round`, half to even),
/// then clamp the area into the `min_pixels`..`max_pixels` budget, rejecting
/// an aspect ratio over 200 and a result that still overflows `max_pixels`.
/// Returns `(height, width)`, the reference's order.
fn smart_resize(
    height: u32,
    width: u32,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(u32, u32), TransformError> {
    if height == 0 || width == 0 || factor == 0 || min_pixels == 0 || max_pixels < min_pixels {
        return Err(TransformError::ShapeError(
            "image dimensions and pixel limits must be positive and ordered".to_string(),
        ));
    }
    if height.max(width) as f64 / height.min(width) as f64 > 200.0 {
        return Err(TransformError::ShapeError(format!(
            "image aspect ratio must not exceed 200, got {width}x{height}"
        )));
    }
    let factor = factor as u64;
    let mut h = round_half_to_even(height as f64 / factor as f64) as u64 * factor;
    let mut w = round_half_to_even(width as f64 / factor as f64) as u64 * factor;
    if h * w > max_pixels as u64 {
        let scale = (height as f64 * width as f64 / max_pixels as f64).sqrt();
        h = ((height as f64 / scale / factor as f64).floor() as u64 * factor).max(factor);
        w = ((width as f64 / scale / factor as f64).floor() as u64 * factor).max(factor);
    } else if h * w < min_pixels as u64 {
        let scale = (min_pixels as f64 / (height as f64 * width as f64)).sqrt();
        h = (height as f64 * scale / factor as f64).ceil() as u64 * factor;
        w = (width as f64 * scale / factor as f64).ceil() as u64 * factor;
    }
    h = h.max(factor);
    w = w.max(factor);
    if h * w > max_pixels as u64 {
        return Err(TransformError::ShapeError(
            "image aspect ratio and patch rounding exceed max_pixels".to_string(),
        ));
    }
    Ok((h as u32, w as u32))
}

/// Mirrors `adaptive_resize` in `transformers_utils/processors/hy_v4.py`: a
/// per-frame token estimate (`h*w/factor^2 + max(h,w)/factor`) against the
/// whole-clip budget (`max_pixels/factor^2` minus the timestamp-token reserve
/// and the per-frame timestamp tokens, floored at 1), then a
/// `video_max_image_size` cap; dimensions truncate (`int()`) to `factor`
/// multiples with a one-factor floor. Returns `(height, width)`.
#[expect(
    clippy::too_many_arguments,
    reason = "direct port of the reference signature"
)]
fn adaptive_resize(
    height: u32,
    width: u32,
    num_frames: usize,
    patch_size: usize,
    merge_size: usize,
    max_pixels: usize,
    video_max_image_size: usize,
    timestamp_token_per_frame: usize,
    timestamp_token_reserve: usize,
) -> Result<(u32, u32), TransformError> {
    let factor = patch_size * merge_size;
    if height == 0 || width == 0 || num_frames == 0 || factor == 0 || max_pixels == 0 {
        return Err(TransformError::ShapeError(
            "video dimensions, frame count and pixel budget must be positive".to_string(),
        ));
    }
    let factor_sq = factor * factor;
    let (h, w) = (height as usize, width as usize);
    let frame_tokens = h * w / factor_sq + h.max(w) / factor;
    // Python `max(1, a - b - c)` on ints; the saturating subtraction only
    // differs below 1, where the floor lands anyway.
    let budget = (max_pixels / factor_sq)
        .saturating_sub(timestamp_token_reserve)
        .saturating_sub(timestamp_token_per_frame * num_frames)
        .max(1);
    let scale = if frame_tokens * num_frames > budget {
        (budget as f64 / (frame_tokens * num_frames) as f64).sqrt()
    } else if h * w > video_max_image_size * video_max_image_size {
        video_max_image_size as f64 / ((h * w) as f64).sqrt()
    } else {
        1.0
    };
    let truncate_to_factor =
        |dim: usize| ((dim as f64 * scale) as usize / factor * factor).max(factor) as u32;
    Ok((truncate_to_factor(h), truncate_to_factor(w)))
}

/// Mirrors `patchify` in `transformers_utils/processors/hy_v4.py`: patches in
/// row-major (h, w) order, channel-first within each `patch_size` square —
/// row `ph * (w/p) + pw` holds `frame[c][ph*p + y][pw*p + x]` at
/// `c*p*p + y*p + x`. Bytes map to normalized f32 through `table` (see the
/// module docs: the frontend normalizes, the engine's norm is an identity).
fn append_patchified(
    rgb: &RgbImage,
    patch_size: usize,
    table: &[[f32; 256]; 3],
    out: &mut Vec<f32>,
) {
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    debug_assert!(
        h % patch_size == 0 && w % patch_size == 0,
        "hy_v4_vl patchify requires {patch_size}-aligned dimensions, got {w}x{h}"
    );
    let raw = rgb.as_raw();
    out.reserve((h / patch_size) * (w / patch_size) * 3 * patch_size * patch_size);
    for py in 0..h / patch_size {
        for px in 0..w / patch_size {
            for (c, table) in table.iter().enumerate() {
                for y in 0..patch_size {
                    let row = (py * patch_size + y) * w + px * patch_size;
                    for x in 0..patch_size {
                        out.push(table[raw[(row + x) * 3 + c] as usize]);
                    }
                }
            }
        }
    }
}

/// Mirrors `vision_token_grid` + `vision_token_count` in
/// `transformers_utils/processors/hy_v4.py`: `frames = ceil(t /
/// temporal_patch_size)` for video (1 for an image), `rows = grid_h / merge`,
/// `cols = grid_w / merge + use_image_newline_token`, and
/// `frames * (rows * cols + 2 * bool(cat_extra_token))` tokens — the
/// structural start/end and timestamp tokens are not counted. Video merges
/// `spatial_merge_size * spatial_patch_size`.
fn vision_token_count(
    params: &HyV4VisionParams,
    t: usize,
    grid_h: usize,
    grid_w: usize,
    is_video: bool,
) -> Result<usize, TransformError> {
    if t == 0 || grid_h == 0 || grid_w == 0 {
        return Err(TransformError::ShapeError(
            "vision grid dimensions must be positive".to_string(),
        ));
    }
    let mut merge = params.merge_size;
    let frames = if is_video {
        merge *= params.spatial_patch_size;
        if t == 1 && params.spatial_patch_size != 1 {
            return Err(TransformError::ShapeError(
                "single-frame spatial video compression is unsupported".to_string(),
            ));
        }
        t.div_ceil(params.temporal_patch_size)
    } else {
        1
    };
    if !grid_h.is_multiple_of(merge) || !grid_w.is_multiple_of(merge) {
        return Err(TransformError::ShapeError(format!(
            "vision grid {grid_h}x{grid_w} must be divisible by its spatial merge factor {merge}"
        )));
    }
    let rows = grid_h / merge;
    let cols = grid_w / merge + usize::from(params.use_image_newline_token);
    // The reference adds `2 * int(bool(cat_extra_token))`: two structural
    // tokens whenever the extra token is configured, regardless of its count.
    Ok(frames * (rows * cols + 2 * usize::from(params.cat_extra_token > 0)))
}

/// HY4-VL image preprocessor; mirrors `HYV4VLImageProcessor._preprocess` in
/// `transformers_utils/processors/hy_v4.py`.
///
/// Per image: `smart_resize` within the pixel budgets, torchvision-exact
/// bicubic resize, then `patchify` into a flat `(n_patches, 3 * patch^2)`
/// normalized f32 tensor. Emits `image_grid_thw` rows `[1, H/patch, W/patch]`,
/// `image_patch_counts` (per-image patch rows, the flat-slicing sizes of the
/// spec's `FieldLayout::Flat` primary), and the per-image embed-token count
/// of [`vision_token_count`].
#[derive(Debug, Clone)]
pub struct HyV4VLImageProcessor {
    params: HyV4VisionParams,
    norm: PixelNorm,
}

impl Default for HyV4VLImageProcessor {
    fn default() -> Self {
        Self::with_params(HyV4VisionParams::image())
    }
}

impl HyV4VLImageProcessor {
    /// Build from resolved parameters; the spec resolves the preprocessor
    /// config, the model config's `vision_config` knobs, and runs the
    /// geometry check first.
    pub(super) fn with_params(params: HyV4VisionParams) -> Self {
        let norm = pixel_norm(&params);
        Self { params, norm }
    }
}

impl VisionPreProcessor for HyV4VLImageProcessor {
    fn default_mean(&self) -> [f64; 3] {
        // Reference `HYV4VLImageProcessor.image_mean` (OPENAI_CLIP_MEAN), the
        // stats `preprocess` normalizes with unless the checkpoint overrides
        // them (see the module docs).
        PreProcessorConfig::CLIP_MEAN
    }

    fn default_std(&self) -> [f64; 3] {
        PreProcessorConfig::CLIP_STD
    }

    fn preprocess(
        &self,
        images: &[&DynamicImage],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if images.is_empty() {
            return Err(TransformError::EmptyBatch);
        }
        let params = &self.params;
        let factor = params.patch_size * params.merge_size;
        let patch_elems = 3 * params.patch_size * params.patch_size;
        let table = self.norm.table_f32();

        let mut pixels = Vec::new();
        let mut grids = Vec::with_capacity(images.len() * 3);
        let mut feature_token_counts = Vec::with_capacity(images.len());
        let mut item_sizes = Vec::with_capacity(images.len());
        let mut patch_counts = Vec::with_capacity(images.len());
        let mut total_patches = 0usize;

        for image in images {
            // Reference `do_convert_rgb`: alpha is dropped, not composited.
            let rgb = image.to_rgb8();
            let (width, height) = rgb.dimensions();
            let (h, w) = smart_resize(height, width, factor, params.min_pixels, params.max_pixels)?;
            let resized = resize_bicubic_torch_aa(&DynamicImage::ImageRgb8(rgb), w, h).to_rgb8();
            append_patchified(&resized, params.patch_size, &table, &mut pixels);

            let grid_h = h as usize / params.patch_size;
            let grid_w = w as usize / params.patch_size;
            grids.extend_from_slice(&[1, grid_h as i64, grid_w as i64]);
            total_patches += grid_h * grid_w;
            patch_counts.push((grid_h * grid_w) as i64);
            feature_token_counts.push(vision_token_count(params, 1, grid_h, grid_w, false)?);
            item_sizes.push((width, height));
        }

        let encoder_input = ndarray::Array2::from_shape_vec((total_patches, patch_elems), pixels)
            .map_err(|error| {
            TransformError::ShapeError(format!(
                "failed to build HY4-VL image patches [{total_patches}, {patch_elems}]: {error}"
            ))
        })?;

        Ok(
            PreprocessedEncoderInputs::new(encoder_input, feature_token_counts, item_sizes)
                .with_extra(
                    "image_grid_thw",
                    ModelSpecificValue::int_2d(grids, images.len(), 3),
                )
                .with_extra(
                    "image_patch_counts",
                    ModelSpecificValue::int_1d(patch_counts),
                ),
        )
    }

    fn calculate_num_tokens(&self, width: u32, height: u32) -> Result<usize, TransformError> {
        let params = &self.params;
        let factor = params.patch_size * params.merge_size;
        let (h, w) = smart_resize(height, width, factor, params.min_pixels, params.max_pixels)?;
        vision_token_count(
            params,
            1,
            h as usize / params.patch_size,
            w as usize / params.patch_size,
            false,
        )
    }

    fn model_name(&self) -> &'static str {
        "hy_v4_vl"
    }

    fn get_processed_size(&self) -> Option<(u32, u32)> {
        None
    }
}

/// HY4-VL video preprocessor; mirrors `HYV4VLVideoProcessor._preprocess` in
/// `transformers_utils/processors/hy_v4.py`.
///
/// Frames arrive pre-sampled by the media connector at the spec-declared rate
/// (the reference `sample_frames`: `floor(total/fps * self.fps)` frames
/// clamped to `min_frames..=max_frames`, midpoint-of-segment indices), so
/// here a clip only gets the `max_frames` bound check, one `adaptive_resize`
/// grid shared by all frames (computed from the first frame's dimensions,
/// like the reference's single `(h, w)` per video), per-frame
/// `patchify`, and per-merged-frame timestamps. Emits `video_grid_thw`
/// `[t, H/patch, W/patch]`, `video_patch_counts` / `video_timestamp_counts`
/// (the flat-slicing sizes of the spec's `FieldLayout::Flat` fields),
/// `video_timestamps` (one f32 per merged frame, as a 1-D tensor so the spec
/// can declare it flat), and the clip's embed-token count of
/// [`vision_token_count`].
#[derive(Debug, Clone)]
pub struct HyV4VLVideoProcessor {
    params: HyV4VisionParams,
    norm: PixelNorm,
}

impl Default for HyV4VLVideoProcessor {
    fn default() -> Self {
        Self::with_params(HyV4VisionParams::video())
    }
}

impl HyV4VLVideoProcessor {
    /// Build from resolved parameters; the spec resolves the preprocessor
    /// config, the model config's `vision_config` knobs, and runs the
    /// geometry check first.
    pub(super) fn with_params(params: HyV4VisionParams) -> Self {
        let norm = pixel_norm(&params);
        Self { params, norm }
    }

    /// Mirrors the default timestamp branch of `HYV4VLProcessor.__call__`
    /// (`transformers_utils/processors/hy_v4.py`): frame indices padded with
    /// their last entry to a `temporal_patch_size` multiple, then
    /// `offset + sum(indices[i:i+tau]) / tau / fps` per merged frame. The
    /// frames arrive pre-sampled at `clip.sample_fps`, so the reference's
    /// container-frame indices become the uniform `0..t` and the offset 0 —
    /// the same uniform-grid synthesis Qwen3-VL's frontend uses. Values at
    /// the default 2 fps are multiples of `1 / (2*tau)` and exactly
    /// representable in the emitted f32. These are the spec's fallback
    /// stamps: its media-aware replacement path stamps from the
    /// decoder-reported sampled frame indices when available, and only the
    /// stamp values change — this emission always ships as the engine's
    /// `video_timestamps` kwarg.
    fn merged_frame_timestamps(&self, t: usize, sample_fps: f64) -> Vec<f32> {
        let tau = self.params.temporal_patch_size.max(1);
        let groups = t.div_ceil(tau);
        (0..groups)
            .map(|group| {
                let sum: usize =
                    (group * tau..group * tau + tau).map(|index| index.min(t - 1)).sum();
                (sum as f64 / tau as f64 / sample_fps) as f32
            })
            .collect()
    }

    /// Shared clip pipeline over RGB frames, mirroring
    /// `HYV4VLVideoProcessor._preprocess` for one video.
    fn preprocess_clip(
        &self,
        frames: &[RgbImage],
        clip: &VideoClipOptions,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        let params = &self.params;
        if frames.is_empty() {
            return Err(TransformError::EmptyBatch);
        }
        let t = frames.len();
        if t > params.max_frames {
            return Err(TransformError::ShapeError(format!(
                "video has {t} frames, exceeding max_frames {}; sample frames before encoding",
                params.max_frames
            )));
        }
        if !clip.sample_fps.is_finite() || clip.sample_fps <= 0.0 {
            return Err(TransformError::ShapeError(format!(
                "HY4-VL video sample_fps must be finite and positive, got {}",
                clip.sample_fps
            )));
        }

        let first = &frames[0];
        let (h, w) = adaptive_resize(
            first.height(),
            first.width(),
            t,
            params.patch_size,
            params.merge_size,
            params.max_pixels,
            params.video_max_image_size,
            params.timestamp_token_per_frame,
            params.timestamp_token_reserve,
        )?;
        if t * h as usize * w as usize > params.max_pixels {
            return Err(TransformError::ShapeError(format!(
                "processed video of {t} {w}x{h} frames exceeds its total max_pixels budget {}",
                params.max_pixels
            )));
        }

        let mut pixels = Vec::new();
        let table = self.norm.table_f32();
        for frame in frames {
            let resized =
                resize_bicubic_torch_aa_rgb(frame.as_raw(), frame.width(), frame.height(), w, h)?;
            append_patchified(&resized, params.patch_size, &table, &mut pixels);
        }

        let grid_h = h as usize / params.patch_size;
        let grid_w = w as usize / params.patch_size;
        let patch_elems = 3 * params.patch_size * params.patch_size;
        let total_patches = t * grid_h * grid_w;
        let encoder_input = ndarray::Array2::from_shape_vec((total_patches, patch_elems), pixels)
            .map_err(|error| {
            TransformError::ShapeError(format!(
                "failed to build HY4-VL video patches [{total_patches}, {patch_elems}]: {error}"
            ))
        })?;

        // `video_timestamps` ships as a real 1-D tensor (not a `FloatVec`
        // passthrough) so the spec can declare it `FieldLayout::Flat`,
        // matching the reference's `flat_from_sizes` field config
        // (`mm_preprocess._get_mm_fields_config`); a passthrough list cannot
        // ride the wire's flat-field shape.
        let timestamps = self.merged_frame_timestamps(t, clip.sample_fps);
        let timestamp_count = timestamps.len();
        Ok(PreprocessedEncoderInputs::new(
            encoder_input,
            vec![vision_token_count(params, t, grid_h, grid_w, true)?],
            vec![(first.width(), first.height())],
        )
        .with_extra(
            "video_grid_thw",
            ModelSpecificValue::int_2d(vec![t as i64, grid_h as i64, grid_w as i64], 1, 3),
        )
        .with_extra(
            "video_patch_counts",
            ModelSpecificValue::int_1d(vec![(t * grid_h * grid_w) as i64]),
        )
        .with_extra(
            "video_timestamps",
            ModelSpecificValue::Tensor {
                data: timestamps,
                shape: vec![timestamp_count],
            },
        )
        .with_extra(
            "video_timestamp_counts",
            ModelSpecificValue::int_1d(vec![timestamp_count as i64]),
        ))
    }
}

impl VisionPreProcessor for HyV4VLVideoProcessor {
    fn default_mean(&self) -> [f64; 3] {
        // The reference `HYV4VLVideoProcessor.image_mean` is 0.5/0.5, but
        // serving normalizes both lanes with the image processor's CLIP stats
        // (see the module docs), which is what `preprocess_video` applies.
        PreProcessorConfig::CLIP_MEAN
    }

    fn default_std(&self) -> [f64; 3] {
        PreProcessorConfig::CLIP_STD
    }

    fn preprocess(
        &self,
        _images: &[&DynamicImage],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        Err(TransformError::ShapeError(
            "hy_v4_vl video processor handles video clips only; route images to the image processor"
                .to_string(),
        ))
    }

    fn preprocess_video(
        &self,
        frames: &[DynamicImage],
        clip: &VideoClipOptions,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        // Reference `do_convert_rgb`: alpha is dropped, not composited.
        let rgb_frames: Vec<RgbImage> = frames.iter().map(DynamicImage::to_rgb8).collect();
        self.preprocess_clip(&rgb_frames, clip)
    }

    fn preprocess_video_rgb(
        &self,
        frames: &[RgbFrameRef<'_>],
        clip: &VideoClipOptions,
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
        self.preprocess_clip(&rgb_frames, clip)
    }

    fn calculate_num_tokens(&self, _width: u32, _height: u32) -> Result<usize, TransformError> {
        Err(TransformError::ShapeError(
            "hy_v4_vl video token counts come from each clip's video_grid_thw, not a frame size"
                .to_string(),
        ))
    }

    fn model_name(&self) -> &'static str {
        "hy_v4_vl"
    }

    fn get_processed_size(&self) -> Option<(u32, u32)> {
        None
    }
}

#[cfg(test)]
mod tests {
    use image::Rgb;

    use super::*;

    fn solid(width: u32, height: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_pixel(width, height, Rgb([120, 30, 200])))
    }

    fn solid_frame(width: u32, height: u32, color: [u8; 3]) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_pixel(width, height, Rgb(color)))
    }

    /// Build a processor from a preprocessor config the way the spec does
    /// (minus the `vision_config` reconciliation).
    fn image_processor_from(config: &PreProcessorConfig) -> HyV4VLImageProcessor {
        HyV4VLImageProcessor::with_params(
            HyV4VisionParams::image().apply_preprocessor_config(config),
        )
    }

    fn video_processor_from(config: &PreProcessorConfig) -> HyV4VLVideoProcessor {
        HyV4VLVideoProcessor::with_params(
            HyV4VisionParams::video().apply_preprocessor_config(config),
        )
    }

    #[test]
    fn smart_resize_rounds_to_factor_and_keeps_in_budget_sizes() {
        // Already a 32-multiple inside the budget: unchanged.
        assert_eq!(
            smart_resize(512, 512, 32, 512 * 512, 2048 * 2048).unwrap(),
            (512, 512)
        );
        // round(31.25) = 31, round(15.625) = 16.
        assert_eq!(
            smart_resize(1000, 500, 32, 512 * 512, 2048 * 2048).unwrap(),
            (992, 512)
        );
        // Python round() is half-to-even: round(2.5) = 2, round(1.5) = 2.
        assert_eq!(
            smart_resize(80, 48, 32, 4096, 2048 * 2048).unwrap(),
            (64, 64)
        );
    }

    #[test]
    fn smart_resize_clamps_to_min_and_max_pixel_budgets() {
        // 80x48 rounds to 64x64 (4096 px), below min_pixels: scaled up by
        // sqrt(262144/3840) and ceiled to 32-multiples.
        assert_eq!(
            smart_resize(80, 48, 32, 512 * 512, 2048 * 2048).unwrap(),
            (672, 416)
        );
        // 4096x2048 exceeds max_pixels: scaled down by sqrt(2) and floored to
        // 32-multiples.
        assert_eq!(
            smart_resize(4096, 2048, 32, 512 * 512, 2048 * 2048).unwrap(),
            (2880, 1440)
        );
        // Aspect 200 exactly is accepted; the 1px side rounds to 0 and the
        // min-pixels upscale rebuilds both sides.
        assert_eq!(
            smart_resize(200, 1, 32, 512 * 512, 2048 * 2048).unwrap(),
            (7264, 64)
        );
    }

    #[test]
    fn smart_resize_rejects_extreme_aspect_and_bad_inputs() {
        assert!(smart_resize(1, 201, 32, 512 * 512, 2048 * 2048).is_err());
        assert!(smart_resize(201, 1, 32, 512 * 512, 2048 * 2048).is_err());
        assert!(smart_resize(0, 512, 32, 512 * 512, 2048 * 2048).is_err());
        assert!(smart_resize(512, 512, 32, 2048 * 2048, 512 * 512).is_err());
    }

    #[test]
    fn adaptive_resize_budget_cases() {
        // Within budget and under video_max_image_size: unchanged.
        // frame_tokens = 256 + 16 = 272, budget = 32768 - 2000 - 40 = 30728.
        assert_eq!(
            adaptive_resize(512, 512, 10, 16, 2, 33_554_432, 1024, 4, 2000).unwrap(),
            (512, 512)
        );
        // Over budget: frame_tokens = 1024 + 32 = 1056, budget = 30368,
        // scale = sqrt(30368/105600) ~ 0.5363 -> int(549.13)//32*32 = 544.
        assert_eq!(
            adaptive_resize(1024, 1024, 100, 16, 2, 33_554_432, 1024, 4, 2000).unwrap(),
            (544, 544)
        );
        // Within the token budget but over video_max_image_size: capped by
        // scale = 1024/sqrt(1024*2048) -> int truncations 724 and 1448.
        assert_eq!(
            adaptive_resize(1024, 2048, 4, 16, 2, 33_554_432, 1024, 4, 2000).unwrap(),
            (704, 1440)
        );
        // Sub-factor dimensions floor to one factor.
        assert_eq!(
            adaptive_resize(16, 16, 4, 16, 2, 33_554_432, 1024, 4, 2000).unwrap(),
            (32, 32)
        );
        assert!(adaptive_resize(512, 512, 0, 16, 2, 33_554_432, 1024, 4, 2000).is_err());
    }

    #[test]
    fn calculate_num_tokens_matches_the_documented_formula() {
        let processor = HyV4VLImageProcessor::default();
        // 512x512 -> rows 512/32 = 16, cols 512/32 + 1 = 17 -> 272.
        assert_eq!(processor.calculate_num_tokens(512, 512).unwrap(), 272);
        // smart_resize(500, 1000) -> (512, 992): 16 * (31 + 1) = 512.
        assert_eq!(processor.calculate_num_tokens(1000, 500).unwrap(), 512);
        // smart_resize(2048, 4096) -> (1440, 2880): 45 * (90 + 1) = 4095.
        assert_eq!(processor.calculate_num_tokens(4096, 2048).unwrap(), 4095);
        // Extreme aspect is refused, not silently clamped to a count.
        assert!(processor.calculate_num_tokens(201, 1).is_err());
    }

    #[test]
    fn calculate_num_tokens_honors_newline_and_cat_extra_knobs() {
        // The knobs come from the model config's `vision_config`, threaded in
        // by the spec (tested in `hy_v4_vl.rs`).
        let processor = HyV4VLImageProcessor::with_params(HyV4VisionParams {
            use_image_newline_token: false,
            ..HyV4VisionParams::image()
        });
        assert_eq!(processor.calculate_num_tokens(512, 512).unwrap(), 256);

        // The reference adds 2 * bool(cat_extra_token) structural tokens.
        let processor = HyV4VLImageProcessor::with_params(HyV4VisionParams {
            cat_extra_token: 1,
            ..HyV4VisionParams::image()
        });
        assert_eq!(processor.calculate_num_tokens(512, 512).unwrap(), 274);
    }

    /// The normalized value a patch byte carries: the processor's fused
    /// rescale (1/255) + CLIP normalize in byte units.
    fn normalized(byte: u8, channel: usize) -> f32 {
        let mean = PreProcessorConfig::CLIP_MEAN[channel] as f32;
        let std = PreProcessorConfig::CLIP_STD[channel] as f32;
        (f32::from(byte) - mean * 255.0) / (std * 255.0)
    }

    #[test]
    fn preprocess_image_emits_normalized_patch_rows_and_grid() {
        let processor = HyV4VLImageProcessor::default();
        let out = processor.preprocess(&[&solid(512, 512)]).unwrap();

        // 512x512 stays 512x512: 32x32 patches of 3*16*16 = 768 values.
        assert_eq!(out.encoder_input.shape(), &[1024, 768]);
        assert_eq!(out.feature_token_counts, vec![272]);
        assert_eq!(out.item_sizes, vec![(512, 512)]);
        let ModelSpecificValue::IntTensor { data, shape } = &out.model_specific["image_grid_thw"]
        else {
            panic!("image_grid_thw must be an int tensor");
        };
        assert_eq!(shape, &[1, 3]);
        assert_eq!(data, &[1, 32, 32]);
        let ModelSpecificValue::IntTensor { data, shape } =
            &out.model_specific["image_patch_counts"]
        else {
            panic!("image_patch_counts must be an int tensor");
        };
        assert_eq!(shape, &[1]);
        assert_eq!(data, &[1024]);

        // Solid (120, 30, 200): every patch row is channel-first constants —
        // 256 R values, then 256 G, then 256 B, each normalized (rescale
        // 1/255 + CLIP stats) in the processor.
        let flat = out.encoder_input.flat_f32();
        let row = &flat[..768];
        assert!(row[..256].iter().all(|&v| v == normalized(120, 0)));
        assert!(row[256..512].iter().all(|&v| v == normalized(30, 1)));
        assert!(row[512..].iter().all(|&v| v == normalized(200, 2)));
    }

    #[test]
    fn preprocess_normalizes_with_the_configured_mean_and_std() {
        let config: PreProcessorConfig = serde_json::from_value(serde_json::json!({
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.25, 0.25, 0.25],
        }))
        .unwrap();
        let processor = image_processor_from(&config);
        let out = processor.preprocess(&[&solid(512, 512)]).unwrap();

        // Fused in byte units: (120 - 0.5 * 255) / (0.25 * 255).
        let expected = (120.0_f32 - 0.5 * 255.0) / (0.25 * 255.0);
        let flat = out.encoder_input.flat_f32();
        let row = &flat[..768];
        assert!(row[..256].iter().all(|&v| v == expected));
        assert!(row[256..512].iter().all(|&v| v == (30.0_f32 - 0.5 * 255.0) / (0.25 * 255.0)));
    }

    #[test]
    fn preprocess_image_lays_patches_out_row_major_channel_first() {
        // One white pixel at (x=17, y=33) on black: patch (py=2, px=1) of the
        // 32x32 grid, local (y=1, x=1). No resize: 512x512 is in budget.
        let mut img = RgbImage::from_pixel(512, 512, Rgb([0, 0, 0]));
        img.put_pixel(17, 33, Rgb([255, 255, 255]));
        let processor = HyV4VLImageProcessor::default();
        let out = processor.preprocess(&[&DynamicImage::ImageRgb8(img)]).unwrap();

        assert_eq!(out.encoder_input.shape(), &[1024, 768]);
        let flat = out.encoder_input.flat_f32();
        let white = [normalized(255, 0), normalized(255, 1), normalized(255, 2)];
        let black = [normalized(0, 0), normalized(0, 1), normalized(0, 2)];
        let row = (2 * 32 + 1) * 768;
        assert_eq!(flat[row + 17], white[0]); // R at c=0, y=1, x=1
        assert_eq!(flat[row + 256 + 17], white[1]); // G
        assert_eq!(flat[row + 512 + 17], white[2]); // B
        assert_eq!(flat.iter().filter(|&&v| white.contains(&v)).count(), 3);
        assert!(flat.iter().all(|&v| black.contains(&v) || white.contains(&v)));
    }

    #[test]
    fn image_config_overrides_pixel_budgets() {
        let config: PreProcessorConfig = serde_json::from_value(serde_json::json!({
            "min_pixels": 4096,
            "max_pixels": 131072,
        }))
        .unwrap();
        let processor = image_processor_from(&config);

        // 512x512 exceeds 131072: scale sqrt(2) -> floor(11.31)*32 = 352.
        assert_eq!(processor.calculate_num_tokens(512, 512).unwrap(), 11 * 12);
        let out = processor.preprocess(&[&solid(512, 512)]).unwrap();
        assert_eq!(out.encoder_input.shape(), &[22 * 22, 768]);
        let ModelSpecificValue::IntTensor { data, .. } = &out.model_specific["image_grid_thw"]
        else {
            panic!("image_grid_thw must be an int tensor");
        };
        assert_eq!(data, &[1, 22, 22]);
        let ModelSpecificValue::IntTensor { data, .. } = &out.model_specific["image_patch_counts"]
        else {
            panic!("image_patch_counts must be an int tensor");
        };
        assert_eq!(data, &[22 * 22]);
    }

    #[test]
    fn preprocess_video_emits_frames_grid_timestamps_and_normalized_patches() {
        let processor = HyV4VLVideoProcessor::default();
        let frames = vec![solid_frame(512, 512, [10, 20, 30]); 4];
        let out = processor.preprocess_video(&frames, &VideoClipOptions::sampled_at(2.0)).unwrap();

        // 4 frames of 512x512 (adaptive_resize keeps the size): 4*32*32 rows.
        assert_eq!(out.encoder_input.shape(), &[4096, 768]);
        assert_eq!(out.feature_token_counts, vec![4 * 16 * 17]);
        assert_eq!(out.item_sizes, vec![(512, 512)]);
        let ModelSpecificValue::IntTensor { data, shape } = &out.model_specific["video_grid_thw"]
        else {
            panic!("video_grid_thw must be an int tensor");
        };
        assert_eq!(shape, &[1, 3]);
        assert_eq!(data, &[4, 32, 32]);
        let ModelSpecificValue::IntTensor { data, shape } =
            &out.model_specific["video_patch_counts"]
        else {
            panic!("video_patch_counts must be an int tensor");
        };
        assert_eq!(shape, &[1]);
        assert_eq!(data, &[4 * 32 * 32]);
        let ModelSpecificValue::Tensor { data, shape } = &out.model_specific["video_timestamps"]
        else {
            panic!("video_timestamps must be a float tensor");
        };
        assert_eq!(shape, &[4]);
        assert_eq!(data, &[0.0, 0.5, 1.0, 1.5]);
        let ModelSpecificValue::IntTensor { data, shape } =
            &out.model_specific["video_timestamp_counts"]
        else {
            panic!("video_timestamp_counts must be an int tensor");
        };
        assert_eq!(shape, &[1]);
        assert_eq!(data, &[4]);

        let flat = out.encoder_input.flat_f32();
        let row = &flat[..768];
        assert!(row[..256].iter().all(|&v| v == normalized(10, 0)));
        assert!(row[256..512].iter().all(|&v| v == normalized(20, 1)));
        assert!(row[512..].iter().all(|&v| v == normalized(30, 2)));
    }

    #[test]
    fn video_timestamps_merge_frames_by_temporal_patch_size() {
        let config: PreProcessorConfig = serde_json::from_value(serde_json::json!({
            "temporal_patch_size": 2,
        }))
        .unwrap();
        let processor = video_processor_from(&config);
        let frames = vec![solid_frame(512, 512, [0, 0, 0]); 5];
        let out = processor.preprocess_video(&frames, &VideoClipOptions::sampled_at(2.0)).unwrap();

        // 5 frames, tau 2: indices padded to [0,1,2,3,4,4]; merged groups
        // (0,1), (2,3), (4,4) stamp at sum/2/2s.
        let ModelSpecificValue::Tensor { data, shape } = &out.model_specific["video_timestamps"]
        else {
            panic!("video_timestamps must be a float tensor");
        };
        assert_eq!(shape, &[3]);
        assert_eq!(data, &[0.25, 1.25, 2.0]);
        let ModelSpecificValue::IntTensor { data, .. } =
            &out.model_specific["video_timestamp_counts"]
        else {
            panic!("video_timestamp_counts must be an int tensor");
        };
        assert_eq!(data, &[3]);
        // Token count uses the merged frame count; the grid keeps raw t.
        assert_eq!(out.feature_token_counts, vec![3 * 16 * 17]);
        let ModelSpecificValue::IntTensor { data, .. } = &out.model_specific["video_grid_thw"]
        else {
            panic!("video_grid_thw must be an int tensor");
        };
        assert_eq!(data, &[5, 32, 32]);
    }

    #[test]
    fn video_config_overrides_frames_and_size_budgets() {
        let config: PreProcessorConfig = serde_json::from_value(serde_json::json!({
            "max_frames": 4,
            "timestamp_token_per_frame": 6,
            "timestamp_token_reserve": 1000,
            "video_max_image_size": 512,
        }))
        .unwrap();
        let processor = video_processor_from(&config);
        assert_eq!(processor.params.max_frames, 4);
        assert_eq!(processor.params.timestamp_token_per_frame, 6);
        assert_eq!(processor.params.timestamp_token_reserve, 1000);
        assert_eq!(processor.params.video_max_image_size, 512);

        // A clip longer than max_frames is refused (sampling is upstream).
        let frames = vec![solid_frame(512, 512, [0, 0, 0]); 5];
        assert!(processor.preprocess_video(&frames, &VideoClipOptions::sampled_at(2.0)).is_err());

        // 600x600 exceeds the 512 video_max_image_size: scale 512/600 ->
        // int(512.0)//32*32 = 512; the configured budget leaves room.
        let frames = vec![solid_frame(600, 600, [5, 6, 7]); 4];
        let out = processor.preprocess_video(&frames, &VideoClipOptions::sampled_at(2.0)).unwrap();
        assert_eq!(out.encoder_input.shape(), &[4 * 32 * 32, 768]);
        let ModelSpecificValue::IntTensor { data, .. } = &out.model_specific["video_grid_thw"]
        else {
            panic!("video_grid_thw must be an int tensor");
        };
        assert_eq!(data, &[4, 32, 32]);
    }

    #[test]
    fn preprocess_video_rgb_matches_the_dynamic_image_path() {
        let processor = HyV4VLVideoProcessor::default();
        let raw = vec![7u8; 512 * 512 * 3];
        let mut raw_frame = raw.clone();
        for chunk in raw_frame.as_chunks_mut::<3>().0.iter_mut() {
            *chunk = [7, 8, 9];
        }
        let frames = [
            RgbFrameRef {
                width: 512,
                height: 512,
                data: &raw_frame,
            },
            RgbFrameRef {
                width: 512,
                height: 512,
                data: &raw_frame,
            },
        ];
        let out = processor
            .preprocess_video_rgb(&frames, &VideoClipOptions::sampled_at(2.0))
            .unwrap();

        assert_eq!(out.encoder_input.shape(), &[2048, 768]);
        assert_eq!(out.feature_token_counts, vec![2 * 16 * 17]);
        let ModelSpecificValue::Tensor { data, shape } = &out.model_specific["video_timestamps"]
        else {
            panic!("video_timestamps must be a float tensor");
        };
        assert_eq!(shape, &[2]);
        assert_eq!(data, &[0.0, 0.5]);
        let flat = out.encoder_input.flat_f32();
        let row = &flat[..768];
        assert!(row[..256].iter().all(|&v| v == normalized(7, 0)));
        assert!(row[256..512].iter().all(|&v| v == normalized(8, 1)));
        assert!(row[512..].iter().all(|&v| v == normalized(9, 2)));
    }

    #[test]
    fn video_rejects_empty_clips_and_bad_sample_rates() {
        let processor = HyV4VLVideoProcessor::default();
        assert!(processor.preprocess_video(&[], &VideoClipOptions::sampled_at(2.0)).is_err());
        let frames = vec![solid_frame(512, 512, [0, 0, 0]); 2];
        assert!(processor.preprocess_video(&frames, &VideoClipOptions::sampled_at(0.0)).is_err());
        assert!(
            processor
                .preprocess_video(&frames, &VideoClipOptions::sampled_at(f64::NAN))
                .is_err()
        );
    }
}
