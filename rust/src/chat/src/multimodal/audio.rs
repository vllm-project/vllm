// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Audio-modality preparation through `llm-multimodal`.

use std::sync::Arc;

use llm_multimodal::{AudioClip, Modality, PreprocessedEncoderInputs};
use vllm_engine_core_client::protocol::dtype::ModelDtype;

use super::{AudioModalitySupport, MultimodalModelInfo, PreparedMedia, item};
use crate::error::{Error, Result, bail_multimodal, multimodal};

/// Forward-kwargs name of the primary audio encoder input.
pub(super) const AUDIO_PRIMARY_KEY: &str = "input_audio_features";

impl MultimodalModelInfo {
    /// Preprocess fetched audio clips as one batch and build per-item features.
    pub(super) async fn prepare_audios(
        &self,
        clips: Vec<Arc<AudioClip>>,
        uuids: Vec<Option<String>>,
    ) -> Result<PreparedMedia> {
        let support = self.audio.as_ref().ok_or_else(|| Error::UnsupportedModality {
            modality: Modality::Audio.to_string(),
        })?;
        let preprocessed = self.preprocess_audios(support, &clips).await?;
        let replacements = support.spec.prompt_replacements_for(&self.context, &preprocessed)?;
        if replacements.len() != clips.len() {
            bail_multimodal!(
                "number of audio prompt replacements {} does not match number of audio clips {}",
                replacements.len(),
                clips.len()
            );
        }

        let hashes = clips.iter().map(|clip| clip.hash.clone()).collect();
        let items = item::build_batched_items(
            &support.spec,
            preprocessed,
            hashes,
            uuids,
            ModelDtype::Float32,
        )?;

        Ok(PreparedMedia {
            modality: Modality::Audio,
            placeholder: support.placeholder.clone(),
            replacements,
            items,
        })
    }

    /// Run CPU-heavy audio preprocessing in a blocking task.
    async fn preprocess_audios(
        &self,
        support: &AudioModalitySupport,
        clips: &[Arc<AudioClip>],
    ) -> Result<PreprocessedEncoderInputs> {
        let processor = Arc::clone(&support.processor);
        let clips = clips.to_vec();
        tokio::task::spawn_blocking(move || Ok(processor.preprocess(&clips)?))
            .await
            .map_err(|error| multimodal!("audio preprocessing task failed: {error}"))?
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use llm_multimodal::{MediaContentPart, ModelSpecificValue, PreProcessorConfig};
    use ndarray::ArrayD;
    use vllm_engine_core_client::protocol::multimodal::{MmField, MmKwargValue};
    use vllm_tokenizer::test_utils::TestTokenizer;

    use super::super::{MultimodalModelContext, TokenizerResolver};
    use super::*;

    const AUDIO_PAD_ID: u32 = 151_676;

    fn qwen3_asr_info() -> MultimodalModelInfo {
        let context = MultimodalModelContext {
            model_id: "Qwen/Qwen3-ASR-1.7B".to_string(),
            model_type: Some("qwen3_asr".to_string()),
            config: serde_json::json!({"model_type": "qwen3_asr"}),
            tokenizer: TokenizerResolver(Arc::new(
                TestTokenizer::new().with_regular_token("<|audio_pad|>", AUDIO_PAD_ID),
            )),
        };

        MultimodalModelInfo::from_loaded(
            context,
            PreProcessorConfig::default(),
            PreProcessorConfig::default(),
        )
        .unwrap()
        .expect("Qwen3-ASR multimodal support")
    }

    fn wav_i16_mono(sample_rate: u32, samples: &[i16]) -> Vec<u8> {
        let data_bytes = samples.len() as u32 * 2;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_bytes).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        bytes.extend_from_slice(&(sample_rate * 2).to_le_bytes());
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        bytes.extend_from_slice(&16_u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_bytes.to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn resolves_qwen_audio_from_model_spec() {
        let info = qwen3_asr_info();
        let support = info.audio.as_ref().expect("audio support");

        assert_eq!(
            info.placeholder_token(Modality::Audio),
            Some("<|audio_pad|>")
        );
        assert_eq!(support.placeholder.marker_token_id, AUDIO_PAD_ID);
        assert_eq!(support.placeholder.embed_token_id, AUDIO_PAD_ID);
        assert!(matches!(
            support.spec.field_layouts.encoder_input,
            llm_multimodal::FieldLayout::Batched
        ));

        let preprocessed = PreprocessedEncoderInputs {
            encoder_input: ArrayD::zeros(vec![1, 128, 4]),
            feature_token_counts: vec![1],
            item_sizes: vec![(128, 4)],
            model_specific: HashMap::from([
                (
                    "feature_attention_mask".to_string(),
                    ModelSpecificValue::int_2d(vec![1; 4], 1, 4),
                ),
                (
                    "audio_feature_lengths".to_string(),
                    ModelSpecificValue::int_1d(vec![4]),
                ),
            ]),
        };
        let item = item::build_batched_items(
            &support.spec,
            preprocessed,
            vec!["<hash>".to_string()],
            vec![None],
            ModelDtype::Float32,
        )
        .unwrap()
        .pop()
        .unwrap();

        assert!(matches!(
            &item.data[AUDIO_PRIMARY_KEY].field,
            MmField::Batched(_)
        ));
    }

    #[tokio::test]
    async fn tracker_processor_and_lowering_preserve_audio_contract() {
        let info = qwen3_asr_info();
        let wav = wav_i16_mono(16_000, &[0; 1_600]);
        let expected_hash = llm_multimodal::hasher::hash_audio(&wav);
        let fetched = info
            .fetch_media(vec![MediaContentPart::AudioData {
                data: wav,
                mime_type: Some("audio/wav".to_string()),
                uuid: Some("audio-1".to_string()),
            }])
            .await
            .unwrap();

        let prepared = info.prepare_audios(fetched.audios, fetched.audio_uuids).await.unwrap();

        assert_eq!(prepared.replacements.len(), 1);
        assert!(
            prepared.replacements[0]
                .tokens
                .iter()
                .all(|token| *token == AUDIO_PAD_ID as i32)
        );
        let item = &prepared.items[0];
        assert_eq!(item.hash, expected_hash);
        assert_eq!(item.uuid.as_deref(), Some("audio-1"));

        let features = &item.data[AUDIO_PRIMARY_KEY];
        assert!(matches!(&features.field, MmField::Batched(_)));
        assert!(matches!(
            features.data.as_ref(),
            Some(MmKwargValue::Tensor(tensor))
                if tensor.dtype == "float32" && tensor.shape.first() == Some(&128)
        ));
        let lengths = &item.data["audio_feature_lengths"];
        assert!(matches!(&lengths.field, MmField::Batched(_)));
        assert!(matches!(
            lengths.data.as_ref(),
            Some(MmKwargValue::Tensor(tensor))
                if tensor.dtype == "int64" && tensor.shape.is_empty()
        ));
    }
}
