// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! MuseSpark1x Omni audio preprocessing.
//!
//! The model consumes *raw waveforms* resampled to 24 kHz — there is no mel
//! spectrogram frontend. Each clip is emitted as-is into one flat
//! `[sum(num_samples)]` fp32 buffer (`input_features` on the vLLM side), with
//! one latent token per 1920 samples (80 ms): `ceil(num_samples / 1920)`.
//!
//! Mirrors vLLM `models/muse_spark/common/processor.py`
//! (`_apply_hf_processor_main` audio branch) and the `AutoEncoder` input
//! contract (`models/muse_spark/common/audio.py`).

use std::sync::Arc;

use llm_multimodal::audio::DecodedAudio;
use llm_multimodal::{
    AudioClip, AudioPreProcessor, ModelSpecificValue, PreProcessorConfig,
    PreprocessedEncoderInputs, TransformError,
};

use super::processor::{AUDIO_TARGET_SAMPLE_RATE, WAVEFORM_PER_TOKEN};
use super::resample::bandlimited_resample;

/// Parameters used by the MuseSpark audio frontend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MuseSparkAudioParams {
    /// Target sample rate; the encoder contract is 24 kHz.
    pub sample_rate: usize,
    /// Samples per latent token (80 ms @ 24 kHz).
    pub waveform_per_token: usize,
}

impl Default for MuseSparkAudioParams {
    fn default() -> Self {
        Self {
            sample_rate: AUDIO_TARGET_SAMPLE_RATE,
            waveform_per_token: WAVEFORM_PER_TOKEN,
        }
    }
}

impl MuseSparkAudioParams {
    pub fn from_config(preprocessor_config: &PreProcessorConfig) -> Self {
        let mut params = Self::default();
        // The combined processor config supplies the feature-extractor section.
        if let Some(value) = preprocessor_config.get_extra::<usize>("sampling_rate") {
            params.sample_rate = value;
        }
        if let Some(value) = preprocessor_config.get_extra::<usize>("samples_per_token") {
            params.waveform_per_token = value;
        }
        params
    }
}

/// MuseSpark1x audio preprocessor: resample to 24 kHz, pass the waveform
/// through unchanged.
#[derive(Debug, Clone, Default)]
pub struct MuseSparkAudioProcessor {
    params: MuseSparkAudioParams,
}

impl MuseSparkAudioProcessor {
    pub fn with_params(params: MuseSparkAudioParams) -> Self {
        Self { params }
    }

    pub fn from_config(preprocessor_config: &PreProcessorConfig) -> Self {
        Self::with_params(MuseSparkAudioParams::from_config(preprocessor_config))
    }

    pub fn preprocess_decoded_clips(
        &self,
        clips: Vec<DecodedAudio>,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if self.params.sample_rate == 0 || self.params.waveform_per_token == 0 {
            return Err(TransformError::ShapeError(
                "MuseSpark audio sample rate and waveform-per-token must be positive".to_string(),
            ));
        }
        if clips.is_empty() {
            return Err(TransformError::EmptyBatch);
        }

        let mut waveforms = Vec::with_capacity(clips.len());
        for clip in &clips {
            if clip.sample_rate == 0 {
                return Err(TransformError::ShapeError(
                    "decoded audio sample rate must be positive".to_string(),
                ));
            }
            if clip.samples.is_empty() {
                return Err(TransformError::ShapeError(
                    "decoded audio contains no samples".to_string(),
                ));
            }
            if clip.samples.iter().any(|sample| !sample.is_finite()) {
                return Err(TransformError::ShapeError(
                    "decoded audio contains a non-finite sample".to_string(),
                ));
            }
            // Decoded clips are already mono (`decode_audio_mono_f32`),
            // matching the reference's multi-channel mean downmix.
            let samples = if clip.sample_rate == self.params.sample_rate {
                clip.samples.clone()
            } else {
                bandlimited_resample(&clip.samples, clip.sample_rate, self.params.sample_rate)?
            };
            waveforms.push(samples);
        }

        let total_samples: usize = waveforms.iter().map(Vec::len).sum();
        let mut samples = Vec::with_capacity(total_samples);
        let mut sample_counts = Vec::with_capacity(waveforms.len());
        let mut token_counts = Vec::with_capacity(waveforms.len());
        let mut item_sizes = Vec::with_capacity(waveforms.len());
        for waveform in waveforms {
            sample_counts.push(waveform.len() as i64);
            token_counts.push(waveform.len().div_ceil(self.params.waveform_per_token));
            item_sizes.push((waveform.len() as u32, self.params.sample_rate as u32));
            samples.extend(waveform);
        }

        let encoder_input =
            ndarray::ArrayD::from_shape_vec(vec![total_samples], samples).map_err(|error| {
                TransformError::ShapeError(format!(
                    "failed to create MuseSpark audio buffer [{total_samples}]: {error}"
                ))
            })?;

        Ok(
            PreprocessedEncoderInputs::new(encoder_input, token_counts.clone(), item_sizes)
                .with_extra(
                    "audio_sample_counts",
                    ModelSpecificValue::int_1d(sample_counts),
                )
                .with_extra(
                    "audio_token_counts",
                    ModelSpecificValue::int_1d(token_counts.iter().map(|&v| v as i64).collect()),
                ),
        )
    }
}

impl AudioPreProcessor for MuseSparkAudioProcessor {
    fn preprocess(
        &self,
        clips: &[Arc<AudioClip>],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        self.preprocess_decoded_clips(clips.iter().map(|clip| clip.decoded().clone()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decoded(samples: usize, sample_rate: usize) -> DecodedAudio {
        DecodedAudio {
            samples: vec![0.25; samples],
            sample_rate,
        }
    }

    #[test]
    fn hf_feature_extractor_controls_resampling_and_token_count() {
        let feature_extractor = serde_json::from_value(serde_json::json!({
            "sampling_rate": 8000, "samples_per_token": 80,
        }))
        .unwrap();
        let processor = MuseSparkAudioProcessor::from_config(&feature_extractor);
        let output = processor.preprocess_decoded_clips(vec![decoded(160, 16000)]).unwrap();
        assert_eq!(output.encoder_input.len(), 80);
        assert_eq!(output.feature_token_counts, vec![1]);
    }

    #[test]
    fn waveform_passthrough_counts_tokens_per_1920_samples() {
        let processor = MuseSparkAudioProcessor::default();
        let output = processor
            .preprocess_decoded_clips(vec![decoded(24000, 24_000), decoded(1921, 24_000)])
            .unwrap();

        assert_eq!(output.encoder_input.shape(), &[24000 + 1921]);
        // ceil(24000/1920) = 13, ceil(1921/1920) = 2
        assert_eq!(output.feature_token_counts, vec![13, 2]);
        assert!(matches!(
            output.model_specific.get("audio_token_counts"),
            Some(ModelSpecificValue::IntTensor { data, .. }) if data == &vec![13, 2]
        ));
        assert!(matches!(
            output.model_specific.get("audio_sample_counts"),
            Some(ModelSpecificValue::IntTensor { data, .. }) if data == &vec![24000, 1921]
        ));
        assert!(output.encoder_input.as_f32().iter().all(|&v| (v - 0.25).abs() < 1e-6));
    }

    #[test]
    fn resamples_to_target_rate() {
        let processor = MuseSparkAudioProcessor::default();
        let output = processor.preprocess_decoded_clips(vec![decoded(16000, 16_000)]).unwrap();

        // 1s at 16kHz -> 1s at 24kHz -> 24000 samples -> 13 tokens.
        assert_eq!(output.feature_token_counts, vec![13]);
        assert_eq!(output.item_sizes, vec![(24000, 24000)]);
    }

    #[test]
    fn rejects_empty_and_nonfinite() {
        let processor = MuseSparkAudioProcessor::default();
        assert!(processor.preprocess_decoded_clips(vec![decoded(0, 24_000)]).is_err());
        let bad = DecodedAudio {
            samples: vec![f32::NAN],
            sample_rate: 24_000,
        };
        assert!(processor.preprocess_decoded_clips(vec![bad]).is_err());
    }
}
