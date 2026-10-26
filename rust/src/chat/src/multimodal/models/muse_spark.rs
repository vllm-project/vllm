// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! MuseSpark1x Omni model spec: NaViT vision (image + video) and raw-waveform
//! audio on the MuseSpark1x stack.
//!
//! Mirrors vLLM `models/muse_spark/common/processor.py`
//! (`MuseSpark1xOmniMultiModalProcessor`):
//!
//! - Placeholders are the vocabulary tokens `<|patch|>`, `<|video|>`, `<|audio|>`;
//!   structural/embedding token ids are resolved from the tokenizer vocabulary
//!   at runtime (`<|image_start|>`, `<|patch|>`, `<|vid_start|>`, ...), never
//!   hardcoded.
//! - Image expansion: `[<|image_start|>] + [<|patch|>] * n + [<|image_end|>]`
//!   with `n` post-downsample patch tokens.
//! - Video expansion: `[<|vid_start|>]`, then per temporal clip a
//!   `"Time: {timestamp:.2f}s"` timestamp text (encoded with the model tokenizer)
//!   followed by `[<|video|>] * patches_per_clip`, clips joined by
//!   `<|vid_frame_separator|>`, closed by `[<|vid_end|>]`.
//! - Audio expansion: `[<|soft_audio_start|>] + [<|audio|>] * n +
//!   [|soft_audio_end|>]` with `n = ceil(num_samples / 1920)`.
//!
//! The engine consumes primary tensors under the model's own kwarg names
//! (`images`, `videos`, `input_features`), declared via
//! [`ModelProcessorSpec::encoder_input_key_for`].

use std::collections::HashMap;

use serde_json::Value;

use llm_multimodal::registry::{
    ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult,
};
use llm_multimodal::{
    AudioPreProcessor, EncoderFieldLayouts, FieldLayout, FrameSampling, Modality,
    ModelSpecificValue, PreProcessorConfig, PreprocessedEncoderInputs, PromptReplacement, TokenId,
    VisionPreProcessor,
};

use self::audio::MuseSparkAudioProcessor;
use self::processor::MuseSparkVisionProcessor;

mod audio;
mod processor;
mod resample;
mod transform;

const PATCH_TOKEN: &str = "<|patch|>";
const VIDEO_TOKEN: &str = "<|video|>";
const AUDIO_TOKEN: &str = "<|audio|>";
const IMAGE_START_TOKEN: &str = "<|image_start|>";
const IMAGE_END_TOKEN: &str = "<|image_end|>";
const VID_START_TOKEN: &str = "<|vid_start|>";
const VID_END_TOKEN: &str = "<|vid_end|>";
const VID_FRAME_SEPARATOR_TOKEN: &str = "<|vid_frame_separator|>";
const SOFT_AUDIO_START_TOKEN: &str = "<|soft_audio_start|>";
const SOFT_AUDIO_END_TOKEN: &str = "<|soft_audio_end|>";

pub(super) struct MuseSparkSpec;

impl MuseSparkSpec {
    /// Whether the checkpoint includes an audio encoder.
    fn audio_enabled(metadata: &ModelMetadata) -> bool {
        metadata.config.get("audio_config").is_some_and(|config| !config.is_null())
    }

    fn unsupported(&self, modality: Modality) -> ModelRegistryError {
        ModelRegistryError::UnsupportedModality {
            spec: self.name(),
            modality,
        }
    }

    fn encode_text(metadata: &ModelMetadata, text: &str) -> Vec<TokenId> {
        metadata
            .tokenizer
            .encode_text(text)
            .unwrap_or_default()
            .into_iter()
            .map(|token| token as TokenId)
            .collect()
    }

    fn int_1d<'a>(
        preprocessed: &'a PreprocessedEncoderInputs,
        key: &str,
    ) -> RegistryResult<&'a [i64]> {
        match preprocessed.model_specific.get(key) {
            Some(ModelSpecificValue::IntTensor { data, .. }) => Ok(data),
            _ => Err(ModelRegistryError::MissingConfigField {
                field: key.to_string(),
            }),
        }
    }

    fn image_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let image_start = metadata.token_id(IMAGE_START_TOKEN)?;
        let image_end = metadata.token_id(IMAGE_END_TOKEN)?;
        let patch = metadata.token_id(PATCH_TOKEN)?;

        Ok(preprocessed
            .feature_token_counts
            .iter()
            .map(|&count| {
                let mut tokens = Vec::with_capacity(count + 2);
                tokens.push(image_start);
                tokens.extend(std::iter::repeat_n(patch, count));
                tokens.push(image_end);
                PromptReplacement::sequence(Modality::Image, PATCH_TOKEN, tokens)
            })
            .collect())
    }

    fn audio_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let soft_start = metadata.token_id(SOFT_AUDIO_START_TOKEN)?;
        let soft_end = metadata.token_id(SOFT_AUDIO_END_TOKEN)?;
        let audio = metadata.token_id(AUDIO_TOKEN)?;

        Ok(preprocessed
            .feature_token_counts
            .iter()
            .map(|&count| {
                let mut tokens = Vec::with_capacity(count + 2);
                tokens.push(soft_start);
                tokens.extend(std::iter::repeat_n(audio, count));
                tokens.push(soft_end);
                PromptReplacement::sequence(Modality::Audio, AUDIO_TOKEN, tokens)
            })
            .collect())
    }

    fn video_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let vid_start = metadata.token_id(VID_START_TOKEN)?;
        let vid_end = metadata.token_id(VID_END_TOKEN)?;
        let vid_separator = metadata.token_id(VID_FRAME_SEPARATOR_TOKEN)?;
        let video = metadata.token_id(VIDEO_TOKEN)?;

        let num_clips = Self::int_1d(preprocessed, "video_clip_num_clips")?;
        let patches_per_clip = Self::int_1d(preprocessed, "video_clip_patches_per_clip")?;
        // Per-clip first-frame timestamps, matching the reference's
        // `float(frames_indices[i]) / fps` rounding. The processor synthesizes
        // them as `i / sampling_fps` on a uniform 2 fps grid, so all values
        // are multiples of 0.5 and exactly representable in f32.
        let clip_timestamps = match preprocessed.model_specific.get("video_clip_timestamps") {
            Some(ModelSpecificValue::FloatVec(timestamps)) => timestamps,
            _ => {
                return Err(ModelRegistryError::MissingConfigField {
                    field: "video_clip_timestamps".to_string(),
                });
            }
        };

        if num_clips.len() != patches_per_clip.len() {
            return Err(ModelRegistryError::MissingConfigField {
                field: "video_clip_num_clips/video_clip_patches_per_clip length mismatch"
                    .to_string(),
            });
        }

        num_clips
            .iter()
            .zip(patches_per_clip)
            .map(|(&clips, &patches)| {
                let clips = usize::try_from(clips).unwrap_or(0);
                let patches = usize::try_from(patches).unwrap_or(0);
                if clip_timestamps.len() < clips {
                    return Err(ModelRegistryError::MissingConfigField {
                        field: "video_clip_timestamps".to_string(),
                    });
                }

                // Per clip: timestamp text tokens followed by patches_per_clip
                // <|video|> tokens, joined by <|vid_frame_separator|>:
                // <|vid_start|> Time: 0.00s <|video|>*M
                //   <|vid_frame_separator|> Time: X.XXs <|video|>*M ...
                // <|vid_end|>
                let mut tokens = vec![vid_start];
                for (clip, &seconds) in clip_timestamps.iter().enumerate().take(clips) {
                    if clip > 0 {
                        tokens.push(vid_separator);
                    }
                    // No trailing newline, matching training.
                    tokens.extend(Self::encode_text(metadata, &format!("Time: {seconds:.2}s")));
                    tokens.extend(std::iter::repeat_n(video, patches));
                }
                tokens.push(vid_end);
                Ok(PromptReplacement::sequence(
                    Modality::Video,
                    VIDEO_TOKEN,
                    tokens,
                ))
            })
            .collect()
    }
}

impl ModelProcessorSpec for MuseSparkSpec {
    fn name(&self) -> &'static str {
        "muse_spark"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        metadata
            .config_model_type()
            .is_some_and(|model_type| model_type == "muse_spark")
            || metadata.model_id.to_ascii_lowercase().contains("muse-spark")
            || metadata.model_id.to_ascii_lowercase().contains("muse_spark")
    }

    fn placeholder_token(&self, metadata: &ModelMetadata) -> RegistryResult<String> {
        self.placeholder_token_for(metadata, Modality::Image)
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        self.placeholder_token_id_for(metadata, Modality::Image)
    }

    fn placeholder_token_for(
        &self,
        _metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<String> {
        match modality {
            Modality::Image => Ok(PATCH_TOKEN.to_string()),
            Modality::Video => Ok(VIDEO_TOKEN.to_string()),
            Modality::Audio => Ok(AUDIO_TOKEN.to_string()),
            Modality::ImageEmbeds => Err(self.unsupported(modality)),
        }
    }

    fn placeholder_token_id_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<TokenId> {
        metadata.token_id(&self.placeholder_token_for(metadata, modality)?)
    }

    fn modality_limits(
        &self,
        metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        // vLLM `get_supported_mm_limits`: unlimited image/video, audio only
        // when an audio encoder is configured.
        let mut limits =
            HashMap::from([(Modality::Image, usize::MAX), (Modality::Video, usize::MAX)]);
        if Self::audio_enabled(metadata) {
            limits.insert(Modality::Audio, usize::MAX);
        }
        Ok(limits)
    }

    fn audio_processor(
        &self,
        _model_config: &Value,
        preprocessor_config: &PreProcessorConfig,
    ) -> RegistryResult<Option<Box<dyn AudioPreProcessor>>> {
        Ok(Some(Box::new(MuseSparkAudioProcessor::from_config(
            preprocessor_config,
        ))))
    }

    fn vision_processor(
        &self,
        _model_config: &Value,
        preprocessor_config: &PreProcessorConfig,
        modality: Modality,
    ) -> RegistryResult<Box<dyn VisionPreProcessor>> {
        match modality {
            // One processor type covers both lanes; the caller passes the
            // modality's own processor config section at construction.
            Modality::Image | Modality::Video => Ok(Box::new(
                MuseSparkVisionProcessor::from_config(preprocessor_config),
            )),
            _ => Err(self.unsupported(modality)),
        }
    }

    /// The reference video processor's `fps`: clips arrive pre-sampled by the
    /// media connector at this rate.
    fn default_video_sample_fps(&self) -> Option<f64> {
        Some(processor::DEFAULT_SAMPLING_FPS)
    }

    fn default_video_sample_fps_with(&self, config: Option<&PreProcessorConfig>) -> Option<f64> {
        Some(
            config
                .and_then(|config| config.get_extra::<f64>("fps"))
                .filter(|fps| fps.is_finite() && *fps > 0.0)
                .unwrap_or(processor::DEFAULT_SAMPLING_FPS),
        )
    }

    /// The reference video processor's `sample_frames`: `trunc(duration x
    /// fps)` frames, at most `num_frames`, in temporal groups, at torch's
    /// linspace positions.
    fn video_frame_sampling(&self) -> FrameSampling {
        FrameSampling::TemporalGroups {
            frame_factor: processor::DEFAULT_PATCH_TEMPORAL,
            max_frames: processor::DEFAULT_NUM_FRAMES_PER_VIDEO,
        }
    }

    fn video_frame_sampling_with(&self, config: Option<&PreProcessorConfig>) -> FrameSampling {
        FrameSampling::TemporalGroups {
            frame_factor: config
                .and_then(|config| config.temporal_patch_size)
                .unwrap_or(processor::DEFAULT_PATCH_TEMPORAL)
                .max(1),
            max_frames: config
                .and_then(|config| config.get_extra::<usize>("num_frames"))
                .filter(|frames| *frames > 0)
                .unwrap_or(processor::DEFAULT_NUM_FRAMES_PER_VIDEO),
        }
    }

    /// The MuseSpark engine reads its primary tensors under model-specific
    /// kwarg names rather than the frontend's per-modality defaults.
    fn encoder_input_key_for(&self, modality: Modality) -> Option<String> {
        match modality {
            Modality::Image => Some("images".to_owned()),
            Modality::Video => Some("videos".to_owned()),
            Modality::Audio => Some("input_features".to_owned()),
            Modality::ImageEmbeds => None,
        }
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        self.prompt_replacements_for(metadata, preprocessed, Modality::Image)
    }

    fn prompt_replacements_for(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
        modality: Modality,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        match modality {
            Modality::Image => self.image_replacements(metadata, preprocessed),
            Modality::Video => self.video_replacements(metadata, preprocessed),
            Modality::Audio => self.audio_replacements(metadata, preprocessed),
            Modality::ImageEmbeds => Err(self.unsupported(modality)),
        }
    }

    fn encoder_field_layouts_for(&self, modality: Modality) -> EncoderFieldLayouts {
        match modality {
            // Pixels are variable-shape per item, so the primary tensor ships
            // flattened and is sliced by element counts; `image_target_hw`
            // carries each item's `[H', W']` for reshaping.
            Modality::Image => EncoderFieldLayouts::new(
                FieldLayout::flat("image_pixel_counts"),
                HashMap::from([
                    ("image_pixel_counts".to_string(), FieldLayout::Batched),
                    ("image_patch_counts".to_string(), FieldLayout::Batched),
                    ("image_target_hw".to_string(), FieldLayout::Batched),
                ]),
            ),
            Modality::Video => EncoderFieldLayouts::new(
                FieldLayout::flat("video_pixel_counts"),
                HashMap::from([
                    ("video_pixel_counts".to_string(), FieldLayout::Batched),
                    ("video_frame_counts".to_string(), FieldLayout::Batched),
                    ("video_target_hw".to_string(), FieldLayout::Batched),
                    ("video_patch_counts".to_string(), FieldLayout::Batched),
                    ("video_clip_num_clips".to_string(), FieldLayout::Batched),
                    (
                        "video_clip_patches_per_clip".to_string(),
                        FieldLayout::Batched,
                    ),
                ]),
            ),
            Modality::Audio => EncoderFieldLayouts::new(
                FieldLayout::flat("audio_sample_counts"),
                HashMap::from([
                    ("audio_sample_counts".to_string(), FieldLayout::Batched),
                    ("audio_token_counts".to_string(), FieldLayout::Batched),
                ]),
            ),
            Modality::ImageEmbeds => EncoderFieldLayouts::default(),
        }
    }

    fn keep_on_cpu_keys_for(&self, modality: Modality) -> Vec<String> {
        let keys: &[&str] = match modality {
            // Mirrors the reference's `keep_on_cpu=True` field configs: counts
            // and per-item geometry are prompt-expansion metadata only.
            Modality::Image => &[
                "image_pixel_counts",
                "image_patch_counts",
                "image_target_hw",
            ],
            Modality::Video => &[
                "video_pixel_counts",
                "video_frame_counts",
                "video_target_hw",
                "video_patch_counts",
                "video_clip_num_clips",
                "video_clip_patches_per_clip",
                "video_clip_timestamps",
            ],
            Modality::Audio => &["audio_sample_counts", "audio_token_counts"],
            Modality::ImageEmbeds => &[],
        };
        keys.iter().map(|key| key.to_string()).collect()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::super::{lookup_spec, test_helpers::*};
    use super::*;
    use llm_multimodal::{ImageSize, ModelSpecificValue};

    // Token IDs from the Muse-Spark-1.2 checkpoint's `tokenizer_config.json`.
    fn muse_spark_tokenizer() -> TestTokenizer {
        TestTokenizer::new(&[
            (AUDIO_TOKEN, 200038),
            (IMAGE_START_TOKEN, 200080),
            (IMAGE_END_TOKEN, 200081),
            (VID_START_TOKEN, 200082),
            (VID_END_TOKEN, 200083),
            (VID_FRAME_SEPARATOR_TOKEN, 200087),
            (VIDEO_TOKEN, 200091),
            (PATCH_TOKEN, 200092),
            (SOFT_AUDIO_START_TOKEN, 201463),
            (SOFT_AUDIO_END_TOKEN, 201464),
        ])
    }

    fn metadata<'a>(tokenizer: &'a TestTokenizer, config: &'a Value) -> ModelMetadata<'a> {
        ModelMetadata {
            model_id: "test-model",
            tokenizer,
            config,
        }
    }

    fn omni_config() -> Value {
        json!({
            "model_type": "muse_spark",
            "audio_config": {"patch_sizes": [16, 6, 5, 4]},
        })
    }

    #[test]
    fn muse_spark_matches_model_type_and_model_id() {
        let tokenizer = muse_spark_tokenizer();

        let config = omni_config();
        let spec = lookup_spec(&metadata(&tokenizer, &config)).unwrap();
        assert_eq!(spec.name(), "muse_spark");

        let config = json!({"model_type": "unknown"});
        let id_metadata = ModelMetadata {
            model_id: "example/Muse-Spark-1x-Omni",
            ..metadata(&tokenizer, &config)
        };
        let spec = lookup_spec(&id_metadata).unwrap();
        assert_eq!(spec.name(), "muse_spark");

        // An unrelated model_type with an unrelated model id must not match.
        let config = json!({"model_type": "llava"});
        let other_metadata = metadata(&tokenizer, &config);
        let spec = lookup_spec(&other_metadata).unwrap();
        assert_ne!(spec.name(), "muse_spark");
    }

    #[test]
    fn audio_capability_follows_encoder_config() {
        let tokenizer = muse_spark_tokenizer();

        let config = omni_config();
        let wav_metadata = metadata(&tokenizer, &config);
        let limits = lookup_spec(&wav_metadata).unwrap().modality_limits(&wav_metadata).unwrap();
        assert!(limits.contains_key(&Modality::Audio));

        // A vision-only checkpoint has no audio encoder.
        let config = json!({
            "model_type": "muse_spark",
            "audio_config": null,
        });
        let mel_metadata = metadata(&tokenizer, &config);
        let limits = lookup_spec(&mel_metadata).unwrap().modality_limits(&mel_metadata).unwrap();
        assert!(!limits.contains_key(&Modality::Audio));
    }

    #[test]
    fn image_replacement_wraps_patches_with_markers() {
        let tokenizer = muse_spark_tokenizer();
        let config = omni_config();
        let metadata = metadata(&tokenizer, &config);
        let spec = lookup_spec(&metadata).unwrap();

        let replacements = spec
            .prompt_replacements_for(
                &metadata,
                &test_preprocessed_with_tokens(&[ImageSize::new(112, 56)], &[3]),
                Modality::Image,
            )
            .unwrap();

        assert_eq!(replacements.len(), 1);
        assert_eq!(
            replacements[0].tokens,
            vec![200080, 200092, 200092, 200092, 200081]
        );
        assert_eq!(replacements[0].placeholder_token, PATCH_TOKEN);
    }

    #[test]
    fn audio_replacement_wraps_tokens_with_soft_markers() {
        let tokenizer = muse_spark_tokenizer();
        let config = omni_config();
        let metadata = metadata(&tokenizer, &config);
        let spec = lookup_spec(&metadata).unwrap();

        let replacements = spec
            .prompt_replacements_for(
                &metadata,
                &test_preprocessed_with_tokens(&[ImageSize::new(24000, 24000)], &[2]),
                Modality::Audio,
            )
            .unwrap();

        assert_eq!(replacements[0].tokens, vec![201463, 200038, 200038, 201464]);
    }

    #[test]
    fn video_replacement_interleaves_timestamps_and_clip_tokens() {
        let tokenizer = muse_spark_tokenizer();
        let config = omni_config();
        let metadata = metadata(&tokenizer, &config);
        let spec = lookup_spec(&metadata).unwrap();

        // Two clips of 8 patches each, clip timestamps 0.0s and 1.0s
        // (the test tokenizer encodes text to zero tokens).
        let mut preprocessed = test_preprocessed_with_tokens(&[ImageSize::new(56, 112)], &[16]);
        preprocessed.model_specific.extend([
            (
                "video_clip_num_clips".to_string(),
                ModelSpecificValue::int_1d(vec![2]),
            ),
            (
                "video_clip_patches_per_clip".to_string(),
                ModelSpecificValue::int_1d(vec![8]),
            ),
            (
                "video_clip_timestamps".to_string(),
                ModelSpecificValue::FloatVec(vec![0.0, 1.0]),
            ),
        ]);

        let replacements =
            spec.prompt_replacements_for(&metadata, &preprocessed, Modality::Video).unwrap();

        assert_eq!(replacements.len(), 1);
        let mut expected = vec![200082];
        expected.extend(std::iter::repeat_n(200091, 8));
        expected.push(200087);
        expected.extend(std::iter::repeat_n(200091, 8));
        expected.push(200083);
        assert_eq!(replacements[0].tokens, expected);
    }

    #[test]
    fn layouts_and_cpu_keys_are_modality_specific() {
        let image = MuseSparkSpec.encoder_field_layouts_for(Modality::Image);
        assert_eq!(image.encoder_input, FieldLayout::flat("image_pixel_counts"));
        assert!(image.model_specific.contains_key("image_patch_counts"));

        let video = MuseSparkSpec.encoder_field_layouts_for(Modality::Video);
        assert_eq!(video.encoder_input, FieldLayout::flat("video_pixel_counts"));
        assert!(video.model_specific.contains_key("video_clip_num_clips"));

        let audio = MuseSparkSpec.encoder_field_layouts_for(Modality::Audio);
        assert_eq!(
            audio.encoder_input,
            FieldLayout::flat("audio_sample_counts")
        );

        assert!(
            MuseSparkSpec
                .keep_on_cpu_keys_for(Modality::Video)
                .contains(&"video_clip_timestamps".to_string())
        );
        assert_eq!(
            MuseSparkSpec.encoder_input_key_for(Modality::Image).as_deref(),
            Some("images")
        );
        assert_eq!(
            MuseSparkSpec.encoder_input_key_for(Modality::Audio).as_deref(),
            Some("input_features")
        );
    }

    #[test]
    fn video_sampling_follows_the_reference_processor_and_its_config() {
        assert_eq!(MuseSparkSpec.default_video_sample_fps(), Some(2.0));
        assert_eq!(
            MuseSparkSpec.video_frame_sampling(),
            FrameSampling::TemporalGroups {
                frame_factor: 2,
                max_frames: 600
            }
        );

        let config = PreProcessorConfig::from_value(json!({
            "temporal_patch_size": 4,
            "num_frames": 64,
            "fps": 1.0
        }))
        .unwrap();
        assert_eq!(
            MuseSparkSpec.default_video_sample_fps_with(Some(&config)),
            Some(1.0)
        );
        assert_eq!(
            MuseSparkSpec.video_frame_sampling_with(Some(&config)),
            FrameSampling::TemporalGroups {
                frame_factor: 4,
                max_frames: 64
            }
        );
    }

    #[test]
    fn muse_spark_spec_builds_vision_and_audio_processors() {
        let tokenizer = muse_spark_tokenizer();
        let config = omni_config();
        let metadata = metadata(&tokenizer, &config);
        let spec = lookup_spec(&metadata).unwrap();

        for modality in [Modality::Image, Modality::Video] {
            let processor = spec
                .vision_processor(&config, &PreProcessorConfig::default(), modality)
                .unwrap();
            assert_eq!(processor.model_name(), "muse_spark");
        }
        assert!(matches!(
            spec.vision_processor(&config, &PreProcessorConfig::default(), Modality::Audio),
            Err(ModelRegistryError::UnsupportedModality {
                spec: "muse_spark",
                modality: Modality::Audio,
            })
        ));

        let processor = spec
            .audio_processor(&config, &PreProcessorConfig::default())
            .expect("muse_spark audio config is servable")
            .expect("muse_spark spec must provide an audio processor");

        use std::sync::Arc;

        use bytes::Bytes;

        use llm_multimodal::audio::DecodedAudio;
        use llm_multimodal::{AudioClip, AudioSource};

        let clip = Arc::new(AudioClip::new(
            Bytes::from_static(b"audio"),
            DecodedAudio {
                samples: vec![0.0; 24000],
                sample_rate: 24_000,
            },
            AudioSource::InlineBytes,
            "audio-hash".to_string(),
        ));
        let result = processor.preprocess(&[clip]).unwrap();
        assert_eq!(result.feature_token_counts, vec![13]);
    }
}
