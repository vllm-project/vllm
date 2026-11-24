// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! HY4-VL (HunYuan HYV4 VL) vision model contract: image and video lanes.
//!
//! Mirrors vLLM's `transformers_utils/processors/hy_v4.py` (`HYV4VLProcessor`)
//! prompt expansion and `models/hy_v4/common/mm_preprocess.py`
//! (`HYV4VLMultiModalProcessor`) placeholder targets and engine fields:
//!
//! - The prompt markers join the config-declared start, embed and end token
//!   strings, matching `HYV4VLProcessor.get_placeholder` and the checkpoint
//!   chat template. Matching the complete marker is essential: replacing only
//!   the embed token with the sequence below would duplicate the structural
//!   boundaries. The plain-text `<image>` / `<video>` markers are fallbacks
//!   when any structural token is absent from the tokenizer vocabulary.
//! - Image expansion: `[image_start] + [image_token] * count + [image_end]`
//!   with `count` the processor's per-item feature-token count
//!   (`vision_token_count(grid, config, "image")`).
//! - Video expansion: `[video_start]`, then per merged frame the stamp text
//!   `"<{ts:.1f} seconds>"` encoded with the model tokenizer, followed by
//!   `[image_start] + [video_token] * per_frame + [image_end]`, closed by
//!   `[video_end]`. `per_frame` is `vision_token_count(grid) // frames`,
//!   taken from the processor's feature-token count divided by the merged
//!   frame count so checkpoint overrides of `use_image_newline_token` /
//!   `cat_extra_token` stay in one place (the processor). The stamp values
//!   are exact for the frames actually served: computed from the clip's
//!   decoder-reported sampled frame indices when the media path reports
//!   them (`prompt_replacements_with_media`), falling back to the
//!   processor-synthesized `video_timestamps` otherwise.
//!
//! The engine consumes flat normalized patch rows (`pixel_values`,
//! `pixel_values_videos`) with per-item grids kept on CPU, mirroring
//! `_get_mm_fields_config`. Normalization happens in the processor: the Rust
//! frontend always launches the engine with `VLLM_USE_RUST_FRONTEND=1`, which
//! forces `mm_device_do_normalize=False`, so the model builds
//! `IdentityInputNorm` and pixels would otherwise pass through unnormalized.
//! Both lanes use the image processor's CLIP stats — what the Python serving
//! path's on-device `FusedMMInputNorm` applies to `pixel_values` and
//! `pixel_values_videos` alike. The frontend's flat
//! slicing reads per-item sizes from a companion tensor (the
//! `FieldLayout::Flat` contract, as in `muse_spark`), so the processor emits
//! per-item patch-count (`*_patch_counts`) and timestamp-count tensors the
//! layouts name for that purpose; they ride the wire as keep-on-CPU kwargs
//! the model's `embed_multimodal` ignores (it only parses `pixel_values` /
//! `pixel_values_videos` and their grids, skipping every other key).
//! `video_timestamps` ships as a declared flat field — a per-clip 1-D f32
//! tensor, matching the reference's
//! `flat_from_sizes("video", ceil(t / tau), keep_on_cpu=True)` field config:
//! engine-side batching concatenates flat fields across items without
//! constraining grouping, while a shared field with per-clip (almost always
//! unequal) values would make `_can_batch_mm_items`
//! (`vllm/multimodal/utils.py`) split encoder groups on every multi-video
//! request. Nothing model-side reads the value — prompt expansion is
//! frontend-side and `get_mrope_input_positions` gathers only the grids —
//! so it is carried for field-contract parity.

use std::collections::HashMap;

use serde_json::Value;

use llm_multimodal::registry::{ModelRegistryError, RegistryResult};
use llm_multimodal::{
    EncoderFieldLayouts, FieldLayout, FrameSampling, MediaItemInfo, Modality, ModelMetadata,
    ModelProcessorSpec, ModelSpecificValue, PreProcessorConfig, PreprocessedEncoderInputs,
    PromptReplacement, TokenId, VideoSamplingInfo, VisionPreProcessor,
};

use self::processor::{HyV4VLImageProcessor, HyV4VLVideoProcessor, HyV4VisionParams};

mod processor;
mod sampling;

pub(super) struct Hy4VLSpec;

impl Hy4VLSpec {
    const IMAGE_PLACEHOLDER: &'static str = "<image>";
    const VIDEO_PLACEHOLDER: &'static str = "<video>";

    // `HYV4VLConfig` defaults (transformers_utils/configs/hy_v4.py), used when
    // the model config does not carry the field.
    const IMAGE_START_TOKEN_ID: TokenId = 120118;
    const IMAGE_END_TOKEN_ID: TokenId = 120119;
    const IMAGE_TOKEN_ID: TokenId = 120120;
    const VIDEO_START_TOKEN_ID: TokenId = 120122;
    const VIDEO_END_TOKEN_ID: TokenId = 120123;
    const VIDEO_TOKEN_ID: TokenId = 120124;

    /// The structural token ids framing image and video expansions, resolved
    /// from the model config. Mirrors the id resolution in
    /// `HYV4VLProcessor.__init__`.
    fn structural_tokens(metadata: &ModelMetadata) -> RegistryResult<StructuralTokens> {
        Ok(StructuralTokens {
            image_start: Self::config_token_id(
                metadata,
                "image_start_token_id",
                Self::IMAGE_START_TOKEN_ID,
            )?,
            image_end: Self::config_token_id(
                metadata,
                "image_end_token_id",
                Self::IMAGE_END_TOKEN_ID,
            )?,
            image: Self::config_token_id(metadata, "image_token_id", Self::IMAGE_TOKEN_ID)?,
            video_start: Self::config_token_id(
                metadata,
                "video_start_token_id",
                Self::VIDEO_START_TOKEN_ID,
            )?,
            video_end: Self::config_token_id(
                metadata,
                "video_end_token_id",
                Self::VIDEO_END_TOKEN_ID,
            )?,
            video: Self::config_token_id(metadata, "video_token_id", Self::VIDEO_TOKEN_ID)?,
        })
    }

    /// A structural token id from the model config: an integer id directly, a
    /// token string resolved against the tokenizer vocabulary (checkpoints
    /// spell their structural tokens either way), or the `HYV4VLConfig` class
    /// default when the field is absent.
    fn config_token_id(
        metadata: &ModelMetadata,
        field: &str,
        default: TokenId,
    ) -> RegistryResult<TokenId> {
        match metadata.config.get(field) {
            None | Some(Value::Null) => Ok(default),
            Some(Value::Number(id)) => id
                .as_i64()
                .and_then(|id| TokenId::try_from(id).ok())
                .ok_or_else(|| ModelRegistryError::InvalidConfigField {
                    field: field.to_string(),
                    reason: format!("token id {id} does not fit the wire's token id type"),
                }),
            Some(Value::String(token)) => metadata.token_id(token),
            Some(_) => Err(ModelRegistryError::InvalidConfigField {
                field: field.to_string(),
                reason: "expected an integer token id or a token string".to_string(),
            }),
        }
    }

    /// Rows of an `(n, 3)` integer model-specific tensor, the shape of the HF
    /// processor's `<modality>_grid_thw` (`[t, H / 16, W / 16]` per item).
    fn grid_rows(
        preprocessed: &PreprocessedEncoderInputs,
        key: &str,
    ) -> RegistryResult<Vec<[i64; 3]>> {
        match preprocessed.model_specific.get(key) {
            Some(ModelSpecificValue::IntTensor { data, shape })
                if shape.len() == 2 && shape[1] == 3 && data.len() == shape[0] * 3 =>
            {
                Ok(data.chunks_exact(3).map(|row| [row[0], row[1], row[2]]).collect())
            }
            _ => Err(ModelRegistryError::InvalidPreprocessedField {
                field: key.to_string(),
            }),
        }
    }

    /// `vision_config.temporal_patch_size` from the model config, defaulting
    /// to the `HYV4VLVisionConfig` class default.
    fn temporal_patch_size(metadata: &ModelMetadata) -> i64 {
        i64::from(
            metadata
                .config_u32(&["vision_config", "temporal_patch_size"])
                .unwrap_or(1)
                .max(1),
        )
    }

    /// The prompt marker for one modality: the structural start, embed and end
    /// token strings joined exactly as in `HYV4VLProcessor.get_placeholder`
    /// and the checkpoint chat template. If any structural token is absent
    /// from the tokenizer, falls back to the plain-text reference marker.
    fn framed_placeholder(
        metadata: &ModelMetadata,
        fields: [(&'static str, TokenId); 3],
        fallback: &'static str,
    ) -> RegistryResult<String> {
        let mut marker = String::new();
        for (field, default) in fields {
            let id = Self::config_token_id(metadata, field, default)?;
            let Some(token) = metadata.tokenizer.id_to_token(id as u32) else {
                return Ok(fallback.to_string());
            };
            marker.push_str(&token);
        }
        Ok(marker)
    }

    /// Mirrors `mm_preprocess._get_prompt_updates` wrapping
    /// `HYV4VLProcessor.get_replacement("image", grid)`:
    /// `[image_start] + [image_token] * count + [image_end]`, with `count`
    /// the processor's `vision_token_count` for the item.
    fn image_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let tokens = Self::structural_tokens(metadata)?;
        let placeholder = self.placeholder_token_for(metadata, Modality::Image)?;

        Ok(preprocessed
            .feature_token_counts
            .iter()
            .map(|&count| {
                let mut sequence = Vec::with_capacity(count + 2);
                sequence.push(tokens.image_start);
                sequence.extend(std::iter::repeat_n(tokens.image, count));
                sequence.push(tokens.image_end);
                PromptReplacement::sequence(Modality::Image, &placeholder, sequence)
            })
            .collect())
    }

    /// The exact stamp of each merged group of a clip whose decoder reported
    /// its sampling, mirroring the non-interval timestamp branch of
    /// `HYV4VLProcessor.__call__` (`transformers_utils/processors/hy_v4.py`):
    /// the picked frame indices padded by repeating the last index to a
    /// `temporal_patch_size` multiple (`indices += [indices[-1]] *
    /// (-len(indices) % tau)`), then `sum(indices[i:i+tau]) / tau /
    /// source_fps` per group. The reference's offset is 0 and its interval
    /// branch is unreachable here — the Rust frontend rejects
    /// `mm_processor_kwargs`. `None` when the sampling is unusable (no
    /// indices or a non-positive/non-finite source fps); the caller then
    /// keeps the processor-synthesized stamps.
    fn sampled_frame_seconds(sampling: &VideoSamplingInfo, tau: usize) -> Option<Vec<f64>> {
        if !(sampling.source_fps.is_finite() && sampling.source_fps > 0.0) {
            return None;
        }
        let &last = sampling.frame_indices.last()?;
        let tau = tau.max(1);
        Some(
            sampling
                .frame_indices
                .chunks(tau)
                .map(|group| {
                    let sum = group.iter().sum::<usize>() + last * (tau - group.len());
                    sum as f64 / tau as f64 / sampling.source_fps
                })
                .collect(),
        )
    }

    /// Mirrors `mm_preprocess._get_prompt_updates` wrapping
    /// `HYV4VLProcessor.get_replacement("video", grid, timestamps)`:
    /// `[video_start]`, then per merged frame the encoded stamp text
    /// `"<{ts:.1f} seconds>"`, `[image_start]`, `[video_token] * per_frame`,
    /// `[image_end]`, closed by `[video_end]`.
    ///
    /// `sampled_seconds` carries one optional per-group stamp list per clip
    /// ([`Self::sampled_frame_seconds`], empty when the caller has no media
    /// info): it overrides only the stamp VALUES of the groups it covers —
    /// the reference's `sum(sampled_indices) / tau / source_fps` for the
    /// frames actually served — while groups and clips it does not cover
    /// keep the processor-synthesized `video_timestamps` fallback. The
    /// replacement token structure is identical either way.
    fn video_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
        sampled_seconds: &[Option<Vec<f64>>],
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let tokens = Self::structural_tokens(metadata)?;
        let placeholder = self.placeholder_token_for(metadata, Modality::Video)?;
        let grids = Self::grid_rows(preprocessed, "video_grid_thw")?;
        let timestamps = match preprocessed.model_specific.get("video_timestamps") {
            Some(ModelSpecificValue::Tensor { data, shape }) if shape.len() == 1 => data,
            _ => {
                return Err(ModelRegistryError::InvalidPreprocessedField {
                    field: "video_timestamps".to_string(),
                });
            }
        };
        if grids.len() != preprocessed.feature_token_counts.len() {
            return Err(ModelRegistryError::InvalidPreprocessedField {
                field: "video_grid_thw".to_string(),
            });
        }

        let temporal = Self::temporal_patch_size(metadata);
        let mut timestamp_offset = 0usize;
        grids
            .iter()
            .zip(&preprocessed.feature_token_counts)
            .enumerate()
            .map(|(item, ([t, _, _], &count))| {
                // `vision_token_grid`: merged frames, and `get_replacement`'s
                // `vision_token_count(grid) // frames` per-frame token count.
                let frames = (t + temporal - 1) / temporal;
                let per_frame = (frames > 0 && count % frames as usize == 0)
                    .then_some(count / frames as usize)
                    .ok_or_else(|| ModelRegistryError::InvalidPreprocessedField {
                        field: "video_grid_thw".to_string(),
                    })?;
                let frames = frames as usize;
                let stamps = timestamps
                    .get(timestamp_offset..timestamp_offset + frames)
                    .ok_or_else(|| ModelRegistryError::InvalidPreprocessedField {
                        field: "video_timestamps".to_string(),
                    })?;
                timestamp_offset += frames;
                let sampled = sampled_seconds.get(item).and_then(Option::as_deref);

                let mut sequence = vec![tokens.video_start];
                for (group, &fallback) in stamps.iter().enumerate() {
                    let ts = sampled
                        .and_then(|seconds| seconds.get(group).copied())
                        .unwrap_or(f64::from(fallback));
                    // `f"<{float(timestamp):.1f} seconds>"`: Rust's `{:.1}`
                    // rounds the exact binary value with ties to even, as
                    // Python's `:.1f` does.
                    sequence
                        .extend(metadata.encode_plain(self.name(), &format!("<{ts:.1} seconds>"))?);
                    sequence.push(tokens.image_start);
                    sequence.extend(std::iter::repeat_n(tokens.video, per_frame));
                    sequence.push(tokens.image_end);
                }
                sequence.push(tokens.video_end);
                Ok(PromptReplacement::sequence(
                    Modality::Video,
                    &placeholder,
                    sequence,
                ))
            })
            .collect()
    }

    /// The preprocessing knobs for one modality: the preprocessor config's
    /// values over the reference defaults, then the token knobs and the
    /// geometry check against the model config's `vision_config`, which the
    /// reference reads there (`vision_token_grid`,
    /// `HYV4VLProcessor.__init__`).
    fn vision_params(
        &self,
        model_config: &Value,
        config: &PreProcessorConfig,
        modality: Modality,
        defaults: HyV4VisionParams,
    ) -> RegistryResult<HyV4VisionParams> {
        let mut params = defaults.apply_preprocessor_config(config);
        let vision = model_config.get("vision_config");

        // Mirrors `HYV4VLProcessor.__init__`: the preprocessor config's patch
        // geometry must agree with `vision_config`, or the checkpoint is
        // inconsistent and serving would fail later at request time.
        for (field, actual) in [
            ("patch_size", params.patch_size),
            ("spatial_merge_size", params.merge_size),
            ("spatial_patch_size", params.spatial_patch_size),
            ("temporal_patch_size", params.temporal_patch_size),
        ] {
            if let Some(expected) =
                vision.and_then(|vision| vision.get(field)).and_then(Value::as_u64)
                && expected as usize != actual
            {
                return Err(ModelRegistryError::InvalidPreprocessorConfig {
                    spec: self.name(),
                    modality,
                    reason: format!(
                        "preprocessor config {field} {actual} disagrees with \
                         vision_config's {expected}"
                    ),
                });
            }
        }

        // `vision_token_grid` reads these from `vision_config`
        // (`HYV4VLVisionConfig` defaults: true / 0).
        if let Some(vision) = vision {
            if let Some(value) = vision.get("use_image_newline_token").and_then(Value::as_bool) {
                params.use_image_newline_token = value;
            }
            if let Some(value) = vision.get("cat_extra_token").and_then(Value::as_u64) {
                params.cat_extra_token = value as usize;
            }
        }
        Ok(params)
    }

    fn unsupported(&self, modality: Modality) -> ModelRegistryError {
        ModelRegistryError::UnsupportedModality {
            spec: self.name(),
            modality,
        }
    }
}

/// The structural token ids of one loaded model; see
/// [`Hy4VLSpec::structural_tokens`].
struct StructuralTokens {
    image_start: TokenId,
    image_end: TokenId,
    image: TokenId,
    video_start: TokenId,
    video_end: TokenId,
    video: TokenId,
}

impl ModelProcessorSpec for Hy4VLSpec {
    fn name(&self) -> &'static str {
        "hy_v4_vl"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        if metadata
            .config_model_type()
            .is_some_and(|mt| matches!(mt, "hy_v4_vl" | "hy_v4_flash"))
        {
            return true;
        }
        let id = metadata.model_id.to_ascii_lowercase();
        id.contains("hy4-vl") || id.contains("hy4_vl") || id.contains("hy_v4_vl")
    }

    fn placeholder_token(&self, metadata: &ModelMetadata) -> RegistryResult<String> {
        self.placeholder_token_for(metadata, Modality::Image)
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        self.placeholder_token_id_for(metadata, Modality::Image)
    }

    fn placeholder_token_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<String> {
        match modality {
            Modality::Image => Self::framed_placeholder(
                metadata,
                [
                    ("image_start_token_id", Self::IMAGE_START_TOKEN_ID),
                    ("image_token_id", Self::IMAGE_TOKEN_ID),
                    ("image_end_token_id", Self::IMAGE_END_TOKEN_ID),
                ],
                Self::IMAGE_PLACEHOLDER,
            ),
            Modality::Video => Self::framed_placeholder(
                metadata,
                [
                    ("video_start_token_id", Self::VIDEO_START_TOKEN_ID),
                    ("video_token_id", Self::VIDEO_TOKEN_ID),
                    ("video_end_token_id", Self::VIDEO_END_TOKEN_ID),
                ],
                Self::VIDEO_PLACEHOLDER,
            ),
            _ => Err(self.unsupported(modality)),
        }
    }

    fn placeholder_token_id_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<TokenId> {
        match modality {
            Modality::Image => {
                Self::config_token_id(metadata, "image_token_id", Self::IMAGE_TOKEN_ID)
            }
            Modality::Video => {
                Self::config_token_id(metadata, "video_token_id", Self::VIDEO_TOKEN_ID)
            }
            _ => Err(self.unsupported(modality)),
        }
    }

    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        // Mirrors the reference processor: it accepts any number of image and
        // video items.
        Ok(HashMap::from([
            (Modality::Image, usize::MAX),
            (Modality::Video, usize::MAX),
        ]))
    }

    fn vision_processor(
        &self,
        model_config: &Value,
        config: &PreProcessorConfig,
        modality: Modality,
    ) -> RegistryResult<Box<dyn VisionPreProcessor>> {
        match modality {
            Modality::Image => Ok(Box::new(HyV4VLImageProcessor::with_params(
                self.vision_params(model_config, config, modality, HyV4VisionParams::image())?,
            ))),
            Modality::Video => Ok(Box::new(HyV4VLVideoProcessor::with_params(
                self.vision_params(model_config, config, modality, HyV4VisionParams::video())?,
            ))),
            _ => Err(self.unsupported(modality)),
        }
    }

    /// The reference video processor's `fps`: clips arrive pre-sampled by the
    /// media connector at this rate.
    fn default_video_sample_fps(&self) -> Option<f64> {
        Some(processor::DEFAULT_SAMPLING_FPS)
    }

    fn default_video_sample_fps_with(&self, config: &PreProcessorConfig) -> Option<f64> {
        Some(
            config
                .get_extra::<f64>("fps")
                .filter(|fps| fps.is_finite() && *fps > 0.0)
                .unwrap_or(processor::DEFAULT_SAMPLING_FPS),
        )
    }

    /// Closest available rule to the reference's `sample_frames`
    /// (`HYV4VLVideoProcessor`): `floor(duration x fps)` frames clamped to
    /// `[min_frames, max_frames]` and the clip length. `EvenRounded`
    /// reproduces that count rule exactly but places picks at the rounded
    /// `linspace(0, n - 1, count)` positions, while the reference takes each
    /// segment's midpoint (`(round(segment * i) + round(segment * (i + 1)))
    /// // 2`), so the selected frames differ for most clips.
    // TODO(video-sampling parity): exact frame picks await the crate's
    // upcoming `fetch_video_frames(source, cfg, indices)` integration;
    // `sampling.rs` already ports the reference's index rule
    // (`sample_frame_indices`), so the integration only needs to drive the
    // connector with those indices. The accepted deviations until then:
    // (a) `EvenRounded` places picks at the rounded `linspace(0, n - 1,
    // count)` endpoints while the reference takes segment midpoints —
    // different frames on essentially every clip;
    // (b) the connector clamps to its `VideoFetchConfig` bounds [4, 768]
    // while the reference clamps to [min_frames=4, max_frames=100] from the
    // video preprocessor config, and clips where `floor(duration * fps)` >
    // 100 are REJECTED by the processor (`preprocess_clip`'s max_frames
    // check, fail-closed) where Python serves 100 frames.
    // The timestamps are already exact for the frames actually served:
    // `prompt_replacements_with_media` stamps each merged group from the
    // decoder-reported sampled frame indices when available
    // (`sampled_frame_seconds`), falling back to the processor-synthesized
    // uniform-grid stamps otherwise.
    fn video_frame_sampling(&self) -> FrameSampling {
        FrameSampling::EvenRounded
    }

    /// The HY4 engine reads its primary tensors under the reference
    /// processor's kwarg names (`mm_preprocess._get_mm_fields_config`).
    fn encoder_input_key_for(&self, modality: Modality) -> Option<String> {
        match modality {
            Modality::Image => Some("pixel_values".to_owned()),
            Modality::Video => Some("pixel_values_videos".to_owned()),
            _ => None,
        }
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        self.image_replacements(metadata, preprocessed)
    }

    fn prompt_replacements_for(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
        modality: Modality,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        match modality {
            Modality::Image => self.image_replacements(metadata, preprocessed),
            Modality::Video => self.video_replacements(metadata, preprocessed, &[]),
            _ => Err(self.unsupported(modality)),
        }
    }

    /// Video stamps are exact for the frames actually served when the
    /// decoder reported each clip's sampling (`sampled_frame_seconds`);
    /// clips without usable sampling info keep the processor-synthesized
    /// `video_timestamps` fallback. Other modalities are unaffected.
    fn prompt_replacements_with_media(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
        modality: Modality,
        media: &[MediaItemInfo],
        _preprocessor_config: &PreProcessorConfig,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        if modality != Modality::Video {
            return self.prompt_replacements_for(metadata, preprocessed, modality);
        }
        let temporal = Self::temporal_patch_size(metadata) as usize;
        let seconds: Vec<Option<Vec<f64>>> = media
            .iter()
            .map(|item| {
                item.video_sampling
                    .as_ref()
                    .and_then(|sampling| Self::sampled_frame_seconds(sampling, temporal))
            })
            .collect();
        self.video_replacements(metadata, preprocessed, &seconds)
    }

    /// Mirrors `mm_preprocess._get_mm_fields_config`: patch rows are flat over
    /// items and grids are batched (one `[t, H / 16, W / 16]` row per item).
    /// The flat sizes keys name companion per-item count tensors (per-item
    /// patch counts are the grid row products `t * (H / 16) * (W / 16)`;
    /// per-clip timestamp counts are `ceil(t / temporal_patch_size)`): the
    /// frontend's batched item splitting reads per-item flat sizes from the
    /// companion tensor named here (`FieldLayout::Flat`), it cannot derive
    /// them from the grid. The video lane is built one clip at a time and
    /// ships flat fields full-range, so its sizes keys are declarative only;
    /// the processor still emits them, keeping every declared sizes key
    /// resolvable. `video_timestamps` is declared flat like the reference's
    /// `flat_from_sizes` config: the processor emits a per-clip 1-D f32
    /// tensor, so each item carries exactly its own clip's stamps and
    /// engine-side batching concatenates them across items.
    fn encoder_field_layouts_for(&self, modality: Modality) -> EncoderFieldLayouts {
        match modality {
            Modality::Image => EncoderFieldLayouts::new(
                FieldLayout::flat("image_patch_counts"),
                HashMap::from([
                    ("image_grid_thw".to_string(), FieldLayout::Batched),
                    ("image_patch_counts".to_string(), FieldLayout::Batched),
                ]),
            ),
            Modality::Video => EncoderFieldLayouts::new(
                FieldLayout::flat("video_patch_counts"),
                HashMap::from([
                    ("video_grid_thw".to_string(), FieldLayout::Batched),
                    ("video_patch_counts".to_string(), FieldLayout::Batched),
                    (
                        "video_timestamps".to_string(),
                        FieldLayout::flat("video_timestamp_counts"),
                    ),
                    ("video_timestamp_counts".to_string(), FieldLayout::Batched),
                ]),
            ),
            _ => EncoderFieldLayouts::default(),
        }
    }

    /// Mirrors the reference's `keep_on_cpu=True` field configs: grids,
    /// timestamps and the flat-slicing counts are prompt-expansion metadata
    /// only.
    fn keep_on_cpu_keys_for(&self, modality: Modality) -> Vec<String> {
        let keys: &[&str] = match modality {
            Modality::Image => &["image_grid_thw", "image_patch_counts"],
            Modality::Video => &[
                "video_grid_thw",
                "video_patch_counts",
                "video_timestamps",
                "video_timestamp_counts",
            ],
            _ => &[],
        };
        keys.iter().map(|key| key.to_string()).collect()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use image::{DynamicImage, Rgb, RgbImage};
    use serde_json::{Value, json};

    use llm_multimodal::registry::Tokenizer;
    use llm_multimodal::{ImageSize, ModelSpecificValue, VideoClip, VideoSource};
    use vllm_engine_core_client::protocol::dtype::ModelDtype;
    use vllm_engine_core_client::protocol::multimodal::{
        MmField, MmFlatField, MmKwargValue, MmSlice, SliceSpec,
    };
    use vllm_tokenizer::test_utils::TestTokenizer as WireTestTokenizer;

    use super::super::super::{
        MultimodalModelContext, MultimodalModelInfo, ResolvedMultimodalSpec, TokenizerResolver,
        item,
    };
    use super::super::{lookup_spec, test_helpers::*};
    use super::*;

    fn test_metadata<'a>(tokenizer: &'a dyn Tokenizer, config: &'a Value) -> ModelMetadata<'a> {
        ModelMetadata {
            model_id: "test-model",
            tokenizer,
            config,
        }
    }

    fn hy4_config() -> Value {
        json!({"model_type": "hy_v4_vl"})
    }

    /// Test tokenizer that encodes stamp text one byte per token, so tests
    /// can assert the exact `"<{ts:.1f} seconds>"` text in replacements.
    struct ByteTokenizer(TestTokenizer);

    impl ByteTokenizer {
        fn stamp(text: &str) -> Vec<TokenId> {
            text.bytes().map(|byte| byte as TokenId + 1000).collect()
        }
    }

    impl Tokenizer for ByteTokenizer {
        fn token_to_id(&self, token: &str) -> Option<u32> {
            self.0.token_to_id(token)
        }

        fn id_to_token(&self, id: u32) -> Option<String> {
            self.0.id_to_token(id)
        }

        fn encode_text(&self, text: &str) -> Option<Vec<u32>> {
            Some(text.bytes().map(|byte| byte as u32 + 1000).collect())
        }
    }

    fn byte_tokenizer() -> ByteTokenizer {
        ByteTokenizer(TestTokenizer::new(&[]))
    }

    /// Per-clip `video_timestamps` as the processor emits it: a 1-D f32
    /// tensor with one value per merged frame.
    fn timestamps_1d(data: &[f32]) -> ModelSpecificValue {
        ModelSpecificValue::Tensor {
            data: data.to_vec(),
            shape: vec![data.len()],
        }
    }

    #[test]
    fn matches_model_type_and_model_id() {
        let tokenizer = TestTokenizer::new(&[]);

        for model_type in ["hy_v4_vl", "hy_v4_flash"] {
            let config = json!({"model_type": model_type});
            let spec = lookup_spec(&test_metadata(&tokenizer, &config)).unwrap();
            assert_eq!(spec.name(), "hy_v4_vl");
        }

        let config = json!({});
        for model_id in ["nvidia/HY4-VL", "example/hy4_vl", "HY_V4_VL-preview"] {
            let id_metadata = ModelMetadata {
                model_id,
                ..test_metadata(&tokenizer, &config)
            };
            assert!(Hy4VLSpec.matches(&id_metadata), "{model_id}");
        }

        // An unrelated model_type with an unrelated model id must not match.
        let config = json!({"model_type": "hy_v4"});
        let other = ModelMetadata {
            model_id: "nvidia/HY4-text",
            ..test_metadata(&tokenizer, &config)
        };
        assert!(!Hy4VLSpec.matches(&other));
    }

    #[test]
    fn placeholders_and_embed_ids_come_from_config_with_defaults() {
        let tokenizer = TestTokenizer::new(&[
            ("<image_start>", 120118),
            ("<image_pad>", 120120),
            ("<image_end>", 120119),
            ("<video_start>", 120122),
            ("<video_pad>", 120124),
            ("<video_end>", 120123),
        ]);

        let config = hy4_config();
        let metadata = test_metadata(&tokenizer, &config);
        let spec = lookup_spec(&metadata).unwrap();
        assert_eq!(
            spec.placeholder_token_for(&metadata, Modality::Image).unwrap(),
            "<image_start><image_pad><image_end>"
        );
        assert_eq!(
            spec.placeholder_token_for(&metadata, Modality::Video).unwrap(),
            "<video_start><video_pad><video_end>"
        );
        assert_eq!(
            spec.placeholder_token_id_for(&metadata, Modality::Image).unwrap(),
            120120
        );
        assert_eq!(
            spec.placeholder_token_id_for(&metadata, Modality::Video).unwrap(),
            120124
        );
        assert!(matches!(
            spec.placeholder_token_for(&metadata, Modality::Audio),
            Err(ModelRegistryError::UnsupportedModality {
                spec: "hy_v4_vl",
                ..
            })
        ));

        // Without the structural tokens in the vocabulary, the markers fall
        // back to the plain-text reference markers.
        let tokenizer = TestTokenizer::new(&[]);
        let fallback = test_metadata(&tokenizer, &config);
        assert_eq!(
            spec.placeholder_token_for(&fallback, Modality::Image).unwrap(),
            "<image>"
        );
        assert_eq!(
            spec.placeholder_token_for(&fallback, Modality::Video).unwrap(),
            "<video>"
        );

        // Integer config fields override the class defaults.
        let config = json!({"model_type": "hy_v4_vl", "image_token_id": 42, "video_token_id": 43});
        let overridden = test_metadata(&tokenizer, &config);
        assert_eq!(
            Hy4VLSpec.placeholder_token_id_for(&overridden, Modality::Image).unwrap(),
            42
        );
        assert_eq!(
            Hy4VLSpec.placeholder_token_id_for(&overridden, Modality::Video).unwrap(),
            43
        );

        // A config that carries token strings resolves them via the tokenizer.
        let tokenizer = TestTokenizer::new(&[
            ("<|image_start|>", 120118),
            ("<|img|>", 7),
            ("<|image_end|>", 120119),
            ("<|vid|>", 8),
        ]);
        let config = json!({"model_type": "hy_v4_vl", "image_token_id": "<|img|>", "video_token_id": "<|vid|>"});
        let named = test_metadata(&tokenizer, &config);
        assert_eq!(
            Hy4VLSpec.placeholder_token_id_for(&named, Modality::Image).unwrap(),
            7
        );
        assert_eq!(
            Hy4VLSpec.placeholder_token_id_for(&named, Modality::Video).unwrap(),
            8
        );
        assert_eq!(
            Hy4VLSpec.placeholder_token_for(&named, Modality::Image).unwrap(),
            "<|image_start|><|img|><|image_end|>"
        );
    }

    #[test]
    fn image_replacement_wraps_feature_tokens_with_markers() {
        let tokenizer = TestTokenizer::new(&[
            ("<image_start>", 120118),
            ("<image_pad>", 120120),
            ("<image_end>", 120119),
        ]);
        let config = hy4_config();
        let metadata = test_metadata(&tokenizer, &config);
        let spec = lookup_spec(&metadata).unwrap();

        let replacements = spec
            .prompt_replacements_for(
                &metadata,
                &test_preprocessed_with_tokens(&[ImageSize::new(1024, 512)], &[3]),
                Modality::Image,
            )
            .unwrap();

        assert_eq!(replacements.len(), 1);
        assert_eq!(
            replacements[0].tokens,
            vec![120118, 120120, 120120, 120120, 120119]
        );
        assert_eq!(replacements[0].modality, Modality::Image);
        assert_eq!(
            replacements[0].placeholder_token,
            "<image_start><image_pad><image_end>"
        );
    }

    #[test]
    fn video_replacement_interleaves_stamps_and_frame_spans() {
        let tokenizer = byte_tokenizer();
        let config = hy4_config();
        let metadata = test_metadata(&tokenizer, &config);
        let spec = lookup_spec(&metadata).unwrap();

        // One video, grid [2, 64, 96]: 2 merged frames, rows = 64 / 2 = 32,
        // cols = 96 / 2 + 1 = 49, so 1568 embed tokens per frame.
        let per_frame = 32 * 49;
        let preprocessed =
            test_preprocessed_with_tokens(&[ImageSize::new(1536, 1024)], &[2 * per_frame])
                .with_extra(
                    "video_grid_thw",
                    ModelSpecificValue::int_2d(vec![2, 64, 96], 1, 3),
                )
                .with_extra("video_timestamps", timestamps_1d(&[0.0, 2.0]));

        let replacements =
            spec.prompt_replacements_for(&metadata, &preprocessed, Modality::Video).unwrap();

        assert_eq!(replacements.len(), 1);
        let mut expected = vec![120122];
        expected.extend(ByteTokenizer::stamp("<0.0 seconds>"));
        expected.push(120118);
        expected.extend(std::iter::repeat_n(120124, per_frame));
        expected.push(120119);
        expected.extend(ByteTokenizer::stamp("<2.0 seconds>"));
        expected.push(120118);
        expected.extend(std::iter::repeat_n(120124, per_frame));
        expected.push(120119);
        expected.push(120123);
        assert_eq!(replacements[0].tokens, expected);
        assert_eq!(replacements[0].placeholder_token, "<video>");
    }

    #[test]
    fn video_replacement_reads_temporal_patch_size_from_config() {
        let tokenizer = byte_tokenizer();
        let config = json!({
            "model_type": "hy_v4_vl",
            "vision_config": {"temporal_patch_size": 2},
        });
        let metadata = test_metadata(&tokenizer, &config);

        // 3 sampled frames merge into ceil(3 / 2) = 2 stamped frames.
        let preprocessed = test_preprocessed_with_tokens(&[ImageSize::new(64, 64)], &[8])
            .with_extra(
                "video_grid_thw",
                ModelSpecificValue::int_2d(vec![3, 4, 4], 1, 3),
            )
            .with_extra("video_timestamps", timestamps_1d(&[0.0, 1.0]));

        let replacements = Hy4VLSpec
            .prompt_replacements_for(&metadata, &preprocessed, Modality::Video)
            .unwrap();

        let mut expected = vec![120122];
        for text in ["<0.0 seconds>", "<1.0 seconds>"] {
            expected.extend(ByteTokenizer::stamp(text));
            expected.push(120118);
            expected.extend(std::iter::repeat_n(120124, 4));
            expected.push(120119);
        }
        expected.push(120123);
        assert_eq!(replacements[0].tokens, expected);
    }

    /// With the decoder-reported sampling the stamps are the reference's
    /// `sum(group) / tau / source_fps` of the actually-served frames, not
    /// the processor-synthesized fallback values.
    #[test]
    fn video_replacements_stamp_from_the_sampled_frame_indices() {
        let tokenizer = byte_tokenizer();
        let config = hy4_config();
        let metadata = test_metadata(&tokenizer, &config);

        // One video, grid [4, 4, 4]: 4 merged frames, 2 embed tokens each.
        // The fallback stamps (9.x) must not leak into the output.
        let preprocessed = test_preprocessed_with_tokens(&[ImageSize::new(64, 64)], &[8])
            .with_extra(
                "video_grid_thw",
                ModelSpecificValue::int_2d(vec![4, 4, 4], 1, 3),
            )
            .with_extra("video_timestamps", timestamps_1d(&[9.0, 9.1, 9.2, 9.3]));
        let media = [MediaItemInfo {
            video_sampling: Some(VideoSamplingInfo {
                source_fps: 24.0,
                frame_indices: vec![6, 18, 30, 42],
            }),
        }];

        let replacements = Hy4VLSpec
            .prompt_replacements_with_media(
                &metadata,
                &preprocessed,
                Modality::Video,
                &media,
                &PreProcessorConfig::default(),
            )
            .unwrap();

        // tau = 1: each frame stamps at index / 24 — 6/24 = 0.25 rounds
        // ties-to-even to "<0.2 seconds>", 18/24 = 0.75 to "<0.8 seconds>".
        let mut expected = vec![120122];
        for text in [
            "<0.2 seconds>",
            "<0.8 seconds>",
            "<1.2 seconds>",
            "<1.8 seconds>",
        ] {
            expected.extend(ByteTokenizer::stamp(text));
            expected.push(120118);
            expected.extend(std::iter::repeat_n(120124, 2));
            expected.push(120119);
        }
        expected.push(120123);
        assert_eq!(replacements[0].tokens, expected);
    }

    /// The reference's merged-group stamp (`HYV4VLProcessor.__call__`):
    /// indices padded by repeating the last to a `temporal_patch_size`
    /// multiple, then `sum(group) / tau / source_fps` per group.
    #[test]
    fn video_replacements_stamp_merged_groups_from_the_sampled_indices() {
        let tokenizer = byte_tokenizer();
        let config = json!({
            "model_type": "hy_v4_vl",
            "vision_config": {"temporal_patch_size": 2},
        });
        let metadata = test_metadata(&tokenizer, &config);

        // 3 sampled frames merge into ceil(3 / 2) = 2 stamped frames.
        let preprocessed = test_preprocessed_with_tokens(&[ImageSize::new(64, 64)], &[8])
            .with_extra(
                "video_grid_thw",
                ModelSpecificValue::int_2d(vec![3, 4, 4], 1, 3),
            )
            .with_extra("video_timestamps", timestamps_1d(&[9.0, 9.5]));
        let media = [MediaItemInfo {
            video_sampling: Some(VideoSamplingInfo {
                source_fps: 24.0,
                frame_indices: vec![6, 18, 30],
            }),
        }];

        let replacements = Hy4VLSpec
            .prompt_replacements_with_media(
                &metadata,
                &preprocessed,
                Modality::Video,
                &media,
                &PreProcessorConfig::default(),
            )
            .unwrap();

        // tau = 2: indices pad to [6, 18, 30, 30]; groups (6, 18) and
        // (30, 30) stamp at 24/2/24 = 0.5 and 60/2/24 = 1.25 ("<1.2
        // seconds>", ties to even).
        let mut expected = vec![120122];
        for text in ["<0.5 seconds>", "<1.2 seconds>"] {
            expected.extend(ByteTokenizer::stamp(text));
            expected.push(120118);
            expected.extend(std::iter::repeat_n(120124, 4));
            expected.push(120119);
        }
        expected.push(120123);
        assert_eq!(replacements[0].tokens, expected);
    }

    /// Clips without decoder-reported sampling keep the
    /// processor-synthesized `video_timestamps`; a mixed batch overrides
    /// per clip and advances the flat fallback offsets across clips.
    #[test]
    fn video_replacements_fall_back_to_synthesized_stamps_per_clip() {
        let tokenizer = byte_tokenizer();
        let config = hy4_config();
        let metadata = test_metadata(&tokenizer, &config);

        // Two clips, grid [2, 4, 4] each: 2 merged frames, 2 embed tokens
        // per frame; the flat fallback holds both clips' stamps.
        let preprocessed = test_preprocessed_with_tokens(
            &[ImageSize::new(64, 64), ImageSize::new(64, 64)],
            &[4, 4],
        )
        .with_extra(
            "video_grid_thw",
            ModelSpecificValue::int_2d(vec![2, 4, 4, 2, 4, 4], 2, 3),
        )
        .with_extra("video_timestamps", timestamps_1d(&[1.0, 3.0, 7.0, 9.0]));

        // Clip 1 reports its sampling (indices 0 and 12 of a 24 fps source
        // stamp at 0.0 and 0.5); clip 2 does not and keeps its synthesized
        // fallback slice [7.0, 9.0].
        let media = [
            MediaItemInfo {
                video_sampling: Some(VideoSamplingInfo {
                    source_fps: 24.0,
                    frame_indices: vec![0, 12],
                }),
            },
            MediaItemInfo::default(),
        ];
        let replacements = Hy4VLSpec
            .prompt_replacements_with_media(
                &metadata,
                &preprocessed,
                Modality::Video,
                &media,
                &PreProcessorConfig::default(),
            )
            .unwrap();

        assert_eq!(replacements.len(), 2);
        let expected_tokens = |texts: [&str; 2]| {
            let mut expected = vec![120122];
            for text in texts {
                expected.extend(ByteTokenizer::stamp(text));
                expected.push(120118);
                expected.extend(std::iter::repeat_n(120124, 2));
                expected.push(120119);
            }
            expected.push(120123);
            expected
        };
        assert_eq!(
            replacements[0].tokens,
            expected_tokens(["<0.0 seconds>", "<0.5 seconds>"])
        );
        assert_eq!(
            replacements[1].tokens,
            expected_tokens(["<7.0 seconds>", "<9.0 seconds>"])
        );

        // No media at all behaves exactly like prompt_replacements_for.
        let plain = Hy4VLSpec
            .prompt_replacements_for(&metadata, &preprocessed, Modality::Video)
            .unwrap();
        let with_empty = Hy4VLSpec
            .prompt_replacements_with_media(
                &metadata,
                &preprocessed,
                Modality::Video,
                &[],
                &PreProcessorConfig::default(),
            )
            .unwrap();
        assert_eq!(
            plain.iter().map(|r| &r.tokens).collect::<Vec<_>>(),
            with_empty.iter().map(|r| &r.tokens).collect::<Vec<_>>()
        );
    }

    /// Unusable sampling reports (no indices, a non-positive or non-finite
    /// source fps) fall back to the synthesized stamps like absent media.
    #[test]
    fn video_replacements_fall_back_when_sampling_is_unusable() {
        let tokenizer = byte_tokenizer();
        let config = hy4_config();
        let metadata = test_metadata(&tokenizer, &config);

        let preprocessed = test_preprocessed_with_tokens(&[ImageSize::new(64, 64)], &[4])
            .with_extra(
                "video_grid_thw",
                ModelSpecificValue::int_2d(vec![2, 4, 4], 1, 3),
            )
            .with_extra("video_timestamps", timestamps_1d(&[1.0, 3.0]));

        for sampling in [
            VideoSamplingInfo {
                source_fps: 24.0,
                frame_indices: vec![],
            },
            VideoSamplingInfo {
                source_fps: 0.0,
                frame_indices: vec![0, 12],
            },
            VideoSamplingInfo {
                source_fps: f64::NAN,
                frame_indices: vec![0, 12],
            },
        ] {
            let media = [MediaItemInfo {
                video_sampling: Some(sampling),
            }];
            let replacements = Hy4VLSpec
                .prompt_replacements_with_media(
                    &metadata,
                    &preprocessed,
                    Modality::Video,
                    &media,
                    &PreProcessorConfig::default(),
                )
                .unwrap();
            let mut expected = vec![120122];
            for text in ["<1.0 seconds>", "<3.0 seconds>"] {
                expected.extend(ByteTokenizer::stamp(text));
                expected.push(120118);
                expected.extend(std::iter::repeat_n(120124, 2));
                expected.push(120119);
            }
            expected.push(120123);
            assert_eq!(replacements[0].tokens, expected);
        }
    }

    #[test]
    fn video_replacements_error_cleanly_on_missing_or_mismatched_fields() {
        let tokenizer = TestTokenizer::new(&[]);
        let config = hy4_config();
        let metadata = test_metadata(&tokenizer, &config);

        let base = || test_preprocessed_with_tokens(&[ImageSize::new(64, 64)], &[4]);

        // Missing grid.
        let result = Hy4VLSpec.prompt_replacements_for(&metadata, &base(), Modality::Video);
        assert!(matches!(
            result,
            Err(ModelRegistryError::InvalidPreprocessedField { field }) if field == "video_grid_thw"
        ));

        // Missing timestamps.
        let preprocessed = base().with_extra(
            "video_grid_thw",
            ModelSpecificValue::int_2d(vec![1, 4, 4], 1, 3),
        );
        let result = Hy4VLSpec.prompt_replacements_for(&metadata, &preprocessed, Modality::Video);
        assert!(matches!(
            result,
            Err(ModelRegistryError::InvalidPreprocessedField { field }) if field == "video_timestamps"
        ));

        // Fewer timestamps than merged frames.
        let preprocessed = preprocessed.with_extra("video_timestamps", timestamps_1d(&[]));
        let result = Hy4VLSpec.prompt_replacements_for(&metadata, &preprocessed, Modality::Video);
        assert!(matches!(
            result,
            Err(ModelRegistryError::InvalidPreprocessedField { field }) if field == "video_timestamps"
        ));

        // A feature count that does not divide into whole frames.
        let preprocessed = test_preprocessed_with_tokens(&[ImageSize::new(64, 64)], &[5])
            .with_extra(
                "video_grid_thw",
                ModelSpecificValue::int_2d(vec![2, 4, 4], 1, 3),
            )
            .with_extra("video_timestamps", timestamps_1d(&[0.0, 1.0]));
        let result = Hy4VLSpec.prompt_replacements_for(&metadata, &preprocessed, Modality::Video);
        assert!(matches!(
            result,
            Err(ModelRegistryError::InvalidPreprocessedField { field }) if field == "video_grid_thw"
        ));
    }

    #[test]
    fn layouts_cpu_keys_and_primary_keys_match_the_engine_contract() {
        let image = Hy4VLSpec.encoder_field_layouts_for(Modality::Image);
        assert_eq!(image.encoder_input, FieldLayout::flat("image_patch_counts"));
        assert_eq!(
            image.model_specific.get("image_grid_thw"),
            Some(&FieldLayout::Batched)
        );

        let video = Hy4VLSpec.encoder_field_layouts_for(Modality::Video);
        assert_eq!(video.encoder_input, FieldLayout::flat("video_patch_counts"));
        assert_eq!(
            video.model_specific.get("video_grid_thw"),
            Some(&FieldLayout::Batched)
        );
        // `video_timestamps` is a declared flat field like the reference's
        // `flat_from_sizes` config: the processor emits a per-clip 1-D f32
        // tensor, so engine-side batching concatenates stamps across clips.
        assert_eq!(
            video.model_specific.get("video_timestamps"),
            Some(&FieldLayout::flat("video_timestamp_counts"))
        );

        for key in ["image_grid_thw", "image_patch_counts"] {
            assert!(Hy4VLSpec.keep_on_cpu_keys_for(Modality::Image).contains(&key.to_string()));
        }
        for key in [
            "video_grid_thw",
            "video_patch_counts",
            "video_timestamps",
            "video_timestamp_counts",
        ] {
            assert!(Hy4VLSpec.keep_on_cpu_keys_for(Modality::Video).contains(&key.to_string()));
        }

        assert_eq!(
            Hy4VLSpec.encoder_input_key_for(Modality::Image).as_deref(),
            Some("pixel_values")
        );
        assert_eq!(
            Hy4VLSpec.encoder_input_key_for(Modality::Video).as_deref(),
            Some("pixel_values_videos")
        );
    }

    #[test]
    fn video_sampling_follows_the_reference_processor_and_its_config() {
        assert_eq!(Hy4VLSpec.default_video_sample_fps(), Some(2.0));
        assert_eq!(Hy4VLSpec.video_frame_sampling(), FrameSampling::EvenRounded);

        let config = PreProcessorConfig::from_value(json!({"fps": 1.5})).unwrap();
        assert_eq!(Hy4VLSpec.default_video_sample_fps_with(&config), Some(1.5));
        assert_eq!(
            Hy4VLSpec.video_frame_sampling_with(&config),
            FrameSampling::EvenRounded
        );
    }

    #[test]
    fn hy4_spec_builds_image_and_video_processors() {
        let config = hy4_config();
        for modality in [Modality::Image, Modality::Video] {
            assert!(
                Hy4VLSpec
                    .vision_processor(&config, &PreProcessorConfig::default(), modality)
                    .is_ok()
            );
        }
        assert!(matches!(
            Hy4VLSpec.vision_processor(&config, &PreProcessorConfig::default(), Modality::Audio),
            Err(ModelRegistryError::UnsupportedModality {
                spec: "hy_v4_vl",
                modality: Modality::Audio,
            })
        ));
    }

    /// Mirrors the reference's construction check
    /// (`HYV4VLProcessor.__init__`): the preprocessor config's patch geometry
    /// must agree with the model config's `vision_config`, or building the
    /// processor fails instead of serving wrong-shaped requests later.
    #[test]
    fn vision_config_geometry_mismatch_fails_processor_construction() {
        for modality in [Modality::Image, Modality::Video] {
            // vision_config.patch_size 14 vs the preprocessor default 16.
            let config = json!({"model_type": "hy_v4_vl", "vision_config": {"patch_size": 14}});
            assert!(matches!(
                Hy4VLSpec.vision_processor(&config, &PreProcessorConfig::default(), modality),
                Err(ModelRegistryError::InvalidPreprocessorConfig {
                    spec: "hy_v4_vl",
                    modality: m,
                    ..
                }) if m == modality
            ));

            // vision_config.spatial_merge_size 1 vs the preprocessor default
            // merge_size 2 (the reference's merge_size/spatial_merge_size
            // field-name mapping).
            let config =
                json!({"model_type": "hy_v4_vl", "vision_config": {"spatial_merge_size": 1}});
            assert!(matches!(
                Hy4VLSpec.vision_processor(&config, &PreProcessorConfig::default(), modality),
                Err(ModelRegistryError::InvalidPreprocessorConfig { .. })
            ));

            // vision_config.temporal_patch_size 2 vs the preprocessor default
            // 1; agreeing values build fine.
            let config =
                json!({"model_type": "hy_v4_vl", "vision_config": {"temporal_patch_size": 2}});
            assert!(matches!(
                Hy4VLSpec.vision_processor(&config, &PreProcessorConfig::default(), modality),
                Err(ModelRegistryError::InvalidPreprocessorConfig { .. })
            ));
            let preprocessor =
                PreProcessorConfig::from_value(json!({"temporal_patch_size": 2})).unwrap();
            assert!(Hy4VLSpec.vision_processor(&config, &preprocessor, modality).is_ok());
        }

        // An absent vision_config skips the check entirely.
        let config = hy4_config();
        assert!(
            Hy4VLSpec
                .vision_processor(&config, &PreProcessorConfig::default(), Modality::Image)
                .is_ok()
        );
    }

    /// The reference reads `use_image_newline_token` / `cat_extra_token` from
    /// the model config's `vision_config` (`vision_token_grid`), not the
    /// preprocessor config; the spec threads them into the processors.
    #[test]
    fn vision_config_token_knobs_drive_token_counts() {
        // 512x512 -> 16 rows x 16 cols; the newline column adds 1.
        let config = json!({
            "model_type": "hy_v4_vl",
            "vision_config": {"use_image_newline_token": false},
        });
        let processor = Hy4VLSpec
            .vision_processor(&config, &PreProcessorConfig::default(), Modality::Image)
            .unwrap();
        assert_eq!(processor.calculate_num_tokens(512, 512).unwrap(), 256);

        // The reference adds 2 * bool(cat_extra_token) structural tokens.
        let config = json!({
            "model_type": "hy_v4_vl",
            "vision_config": {"cat_extra_token": 1},
        });
        let processor = Hy4VLSpec
            .vision_processor(&config, &PreProcessorConfig::default(), Modality::Image)
            .unwrap();
        assert_eq!(processor.calculate_num_tokens(512, 512).unwrap(), 274);

        // Defaults (absent vision_config): 16 * 17 = 272.
        let processor = Hy4VLSpec
            .vision_processor(
                &hy4_config(),
                &PreProcessorConfig::default(),
                Modality::Image,
            )
            .unwrap();
        assert_eq!(processor.calculate_num_tokens(512, 512).unwrap(), 272);
    }

    #[test]
    fn timestamp_format_matches_python_one_decimal_spec() {
        // Python `f"<{ts:.1f} seconds>"` and Rust `format!("<{ts:.1} seconds>")`
        // both round the exact binary value with ties to even.
        for (ts, expected) in [
            (0.0, "<0.0 seconds>"),
            (2.0, "<2.0 seconds>"),
            (0.25, "<0.2 seconds>"),
            (0.75, "<0.8 seconds>"),
            (0.05, "<0.1 seconds>"),
            (1.05, "<1.1 seconds>"),
            (3.45, "<3.5 seconds>"),
            (7.35, "<7.3 seconds>"),
            (12.5, "<12.5 seconds>"),
            (100.0, "<100.0 seconds>"),
        ] {
            assert_eq!(format!("<{ts:.1} seconds>"), expected, "ts={ts}");
        }
    }

    /// A flat field's full-range slice `0..len` (`dim = 0`).
    fn flat_full_range(len: usize, keep_on_cpu: bool) -> MmField {
        MmField::Flat(MmFlatField {
            slices: vec![MmSlice::Slice(SliceSpec {
                start: Some(0),
                stop: Some(len as isize),
                step: None,
            })],
            dim: 0,
            keep_on_cpu,
        })
    }

    fn solid_rgb(width: u32, height: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_pixel(width, height, Rgb([120, 30, 200])))
    }

    /// The image lane's primary is declared `Flat` with sizes key
    /// `image_patch_counts`: per-item slices of a multi-image batch must come
    /// from the processor-emitted counts tensor (the grid row products), or
    /// `build_batched_items` fails on the missing sizes key.
    #[test]
    fn image_items_slice_flat_pixels_by_emitted_patch_counts() {
        let processor = HyV4VLImageProcessor::default();
        // 512x512 -> 32x32 grid (1024 rows); 1024x512 -> 32x64 (2048 rows).
        let images = [solid_rgb(512, 512), solid_rgb(1024, 512)];
        let preprocessed = processor.preprocess(&images.each_ref()).unwrap();

        let spec = ResolvedMultimodalSpec::new(&Hy4VLSpec, Modality::Image);
        let items = item::build_batched_items(
            &spec,
            preprocessed,
            vec!["first".to_string(), "second".to_string()],
            vec![None, None],
            ModelDtype::Float32,
        )
        .unwrap();

        assert_eq!(items.len(), 2);
        for (item, rows) in items.iter().zip([1024usize, 2048]) {
            let pixels = &item.data["pixel_values"];
            assert_eq!(pixels.field, flat_full_range(rows, false));
            let Some(MmKwargValue::Tensor(tensor)) = &pixels.data else {
                panic!("pixel_values must be a tensor");
            };
            assert_eq!(tensor.shape, vec![rows, 768]);

            let grid = &item.data["image_grid_thw"];
            assert!(matches!(&grid.field, MmField::Batched(_)));
            let Some(MmKwargValue::Tensor(tensor)) = &grid.data else {
                panic!("image_grid_thw must be a tensor");
            };
            assert_eq!(tensor.shape, vec![3]);

            let counts = &item.data["image_patch_counts"];
            assert!(matches!(&counts.field, MmField::Batched(_)));
        }
        let Some(MmKwargValue::Tensor(tensor)) = &items[1].data["image_grid_thw"].data else {
            panic!("image_grid_thw must be a tensor");
        };
        assert_eq!(
            tensor.data.as_raw_view().unwrap().as_ref(),
            [1_i64, 32, 64].into_iter().flat_map(i64::to_ne_bytes).collect::<Vec<_>>()
        );
    }

    /// Multi-video wire shape: each clip's item carries its own flat
    /// `video_timestamps` tensor (one f32 per merged frame) and the
    /// declarative count tensors, so engine-side batching concatenates flat
    /// fields across clips instead of splitting encoder groups on unequal
    /// shared values (`_can_batch_mm_items`).
    #[tokio::test]
    async fn video_items_ship_per_clip_flat_timestamps_and_counts() {
        let context = MultimodalModelContext {
            model_id: "hy4-vl-test".to_string(),
            model_type: Some("hy_v4_vl".to_string()),
            config: hy4_config(),
            tokenizer: TokenizerResolver(Arc::new(WireTestTokenizer::new())),
        };
        let model_spec = context.resolve_model_spec().expect("hy_v4_vl spec should resolve");
        let info = MultimodalModelInfo::from_loaded(
            context,
            model_spec,
            PreProcessorConfig::default(),
            PreProcessorConfig::default(),
            PreProcessorConfig::default(),
            HashMap::new(),
        )
        .unwrap()
        .expect("hy_v4_vl multimodal support should resolve");

        let clip = |frames: usize, hash: &str| {
            let frames = vec![solid_rgb(512, 512); frames];
            Arc::new(VideoClip::new_with_sample_fps(
                frames,
                bytes::Bytes::from_static(b"video"),
                VideoSource::InlineBytes,
                hash.to_string(),
                2.0,
            ))
        };
        let prepared = info
            .prepare_videos(
                vec![clip(2, "two"), clip(5, "five")],
                vec![None, None],
                ModelDtype::Float32,
            )
            .await
            .unwrap();

        assert_eq!(prepared.items.len(), 2);
        for (item, (frames, stamps)) in prepared.items.iter().zip([
            (2usize, &[0.0_f32, 0.5][..]),
            (5, &[0.0, 0.5, 1.0, 1.5, 2.0][..]),
        ]) {
            let pixels = &item.data["pixel_values_videos"];
            assert_eq!(pixels.field, flat_full_range(frames * 32 * 32, false));

            let timestamps = &item.data["video_timestamps"];
            assert_eq!(timestamps.field, flat_full_range(stamps.len(), true));
            let Some(MmKwargValue::Tensor(tensor)) = &timestamps.data else {
                panic!("video_timestamps must be a tensor");
            };
            assert_eq!(tensor.shape, vec![stamps.len()]);
            assert_eq!(
                tensor.data.as_raw_view().unwrap().as_ref(),
                stamps.iter().flat_map(|ts| ts.to_ne_bytes()).collect::<Vec<_>>()
            );

            let timestamp_counts = &item.data["video_timestamp_counts"];
            assert!(matches!(&timestamp_counts.field, MmField::Batched(_)));
            let patch_counts = &item.data["video_patch_counts"];
            assert!(matches!(&patch_counts.field, MmField::Batched(_)));
        }
    }
}
