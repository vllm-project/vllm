// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Audio-modality preparation through `llm-multimodal`.

use std::sync::Arc;

use llm_multimodal::{AudioClip, Modality, PreprocessedEncoderInputs};
use vllm_engine_core_client::protocol::dtype::ModelDtype;

use super::timing::MM_STAGE_TARGET;
use super::{AudioModalitySupport, MmModality, MultimodalModelInfo, PreparedMedia};
use crate::error::{Result, multimodal};

/// Forward-kwargs name of the primary audio encoder input.
pub(super) const AUDIO_PRIMARY_KEY: &str = "input_audio_features";

impl MultimodalModelInfo {
    /// Preprocess fetched audio clips as one batch and build per-item features.
    #[tracing::instrument(
        name = "mm_stage",
        target = MM_STAGE_TARGET,
        skip_all,
        fields(stage = "preprocess_audio")
    )]
    pub(super) async fn prepare_audios(
        &self,
        clips: Vec<Arc<AudioClip>>,
        uuids: Vec<Option<String>>,
    ) -> Result<PreparedMedia> {
        let support = self
            .audio
            .as_ref()
            .ok_or_else(|| self.unsupported_modality(MmModality::Audio))?;

        let sources = super::keyed_sources(clips, uuids, |media| &media.hash);
        self.prepare_media(
            Modality::Audio,
            &support.placeholder,
            sources,
            async |misses| {
                let (clips, hashes): (Vec<_>, Vec<_>) = misses.into_iter().unzip();
                let preprocessed = self.preprocess_audios(support, &clips).await?;
                support
                    .spec
                    .build_batch(&self.context, preprocessed, hashes, ModelDtype::Float32)
            },
        )
        .await
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

    use super::super::tests::from_context;
    use super::super::{MultimodalModelContext, TokenizerResolver, item};
    use super::*;

    const AUDIO_PAD_ID: u32 = 151_676;
    const INKLING_AUDIO_MARKER_ID: u32 = 200_020;
    const INKLING_AUDIO_EMBED_ID: i32 = 200_053;
    const INKLING_AUDIO_SOFT_PLACEHOLDER: &str = "<|unused_200053|>";
    const INKLING_IMAGE_SOFT_PLACEHOLDER: &str = "<|unused_200054|>";

    fn inkling_info(decoder_dmodel: serde_json::Value) -> MultimodalModelInfo {
        let config = serde_json::json!({
            "model_type": "inkling_mm_model",
            // Keep the vision tower enabled so audio gating alone decides
            // whether the model still resolves multimodal support.
            "vision_config": {"decoder_dmodel": 6144},
            "audio_config": {
                "decoder_dmodel": decoder_dmodel,
                "n_mel_bins": 80,
                "mel_vocab_size": 16,
                "dmel_min_value": -7.0,
                "dmel_max_value": 2.0
            }
        });
        let tokenizer = TestTokenizer::new()
            .with_regular_token("<|content_image|>", 200_005)
            .with_regular_token("<|content_audio_input|>", INKLING_AUDIO_MARKER_ID)
            .with_regular_token(
                INKLING_AUDIO_SOFT_PLACEHOLDER,
                INKLING_AUDIO_EMBED_ID as u32,
            )
            .with_regular_token(INKLING_IMAGE_SOFT_PLACEHOLDER, 200_054);
        let context = MultimodalModelContext {
            model_id: "inkling-test".to_string(),
            model_type: Some("inkling_mm_model".to_string()),
            config,
            tokenizer: TokenizerResolver(Arc::new(tokenizer)),
        };

        from_context(
            context,
            PreProcessorConfig::default(),
            PreProcessorConfig::default(),
            PreProcessorConfig::default(),
            HashMap::new(),
        )
        .unwrap()
        .expect("Inkling multimodal support")
    }

    fn qwen3_asr_info() -> MultimodalModelInfo {
        let context = MultimodalModelContext {
            model_id: "Qwen/Qwen3-ASR-1.7B".to_string(),
            model_type: Some("qwen3_asr".to_string()),
            config: serde_json::json!({"model_type": "qwen3_asr"}),
            tokenizer: TokenizerResolver(Arc::new(
                TestTokenizer::new().with_regular_token("<|audio_pad|>", AUDIO_PAD_ID),
            )),
        };

        from_context(
            context,
            PreProcessorConfig::default(),
            PreProcessorConfig::default(),
            PreProcessorConfig::default(),
            HashMap::new(),
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
    fn resolves_inkling_audio_from_model_spec() {
        let info = inkling_info(serde_json::json!(1024));
        let support = info.audio.as_ref().expect("audio support");

        // The spec renders the checkpoint's soft placeholder; the template's
        // `<|content_audio_input|>` marker rides in the structural prefix.
        assert_eq!(
            info.placeholder_token(Modality::Audio),
            Some(INKLING_AUDIO_SOFT_PLACEHOLDER)
        );
        assert_eq!(
            support.placeholder.marker_token_ids,
            vec![INKLING_AUDIO_EMBED_ID as u32]
        );
        assert_eq!(
            support.placeholder.embed_token_id,
            INKLING_AUDIO_EMBED_ID as u32
        );
        assert_eq!(support.spec.primary_key(), AUDIO_PRIMARY_KEY);
        assert!(matches!(
            &support.spec.field_layouts.encoder_input,
            llm_multimodal::FieldLayout::Flat { sizes_key }
                if sizes_key == "tokens_per_item"
        ));
        assert!(matches!(
            support.spec.field_layouts.model_specific.get("tokens_per_item"),
            Some(llm_multimodal::FieldLayout::Batched)
        ));
    }

    #[test]
    fn inkling_decoder_config_gates_audio_capability() {
        let info = inkling_info(serde_json::Value::Null);

        assert!(info.audio.is_none());
        assert_eq!(info.placeholder_token(Modality::Audio), None);
    }

    #[test]
    fn resolves_qwen_audio_from_model_spec() {
        let info = qwen3_asr_info();
        let support = info.audio.as_ref().expect("audio support");

        assert_eq!(
            info.placeholder_token(Modality::Audio),
            Some("<|audio_pad|>")
        );
        assert_eq!(support.placeholder.marker_token_ids, vec![AUDIO_PAD_ID]);
        assert_eq!(support.placeholder.embed_token_id, AUDIO_PAD_ID);
        assert!(matches!(
            support.spec.field_layouts.encoder_input,
            llm_multimodal::FieldLayout::Batched
        ));

        let preprocessed = PreprocessedEncoderInputs {
            encoder_input: ArrayD::zeros(vec![1, 128, 4]).into(),
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
    async fn audio_pipeline_attributes_stage_timings_to_request() {
        use std::sync::LazyLock;

        use tracing::Instrument as _;
        use tracing_subscriber::layer::SubscriberExt as _;
        use vllm_tracing::timing::RequestTimingStats;

        use crate::{mm_request_span, mm_timing_layer};

        // Installed globally, like `vllm-bench mm-processor` does: callsite
        // interest is cached process-wide, so a scoped subscriber misses the
        // stages whose callsites a concurrent test registers first without it.
        static TIMINGS: LazyLock<RequestTimingStats> = LazyLock::new(|| {
            let (layer, timings) = mm_timing_layer();
            tracing::subscriber::set_global_default(tracing_subscriber::registry().with(layer))
                .expect("install global subscriber");
            timings
        });
        // Multimodal requests of other tests also reach the global layer.
        const REQUEST_ID: &str = "audio-stage-timings";

        let info = qwen3_asr_info();
        LazyLock::force(&TIMINGS);
        let parts = vec![MediaContentPart::AudioData {
            data: wav_i16_mono(16_000, &[0; 1_600]),
            mime_type: Some("audio/wav".to_string()),
            uuid: Some("audio-1".to_string()),
        }];
        info.prepare_multimodal(parts, &mut vec![AUDIO_PAD_ID], ModelDtype::Float32, None)
            .instrument(mm_request_span(REQUEST_ID))
            .await
            .unwrap();

        let mut stages = TIMINGS
            .stat()
            .remove(REQUEST_ID)
            .expect("request timings")
            .into_keys()
            .collect::<Vec<_>>();
        stages.sort();
        expect_test::expect![[r#"
            [
                "media_fetch_secs",
                "preprocess_audio_secs",
                "preprocessor_total_secs",
                "prompt_expansion_secs",
            ]
        "#]]
        .assert_debug_eq(&stages);
    }

    #[tokio::test]
    async fn tracker_processor_and_lowering_preserve_audio_contract() {
        let info = qwen3_asr_info();
        let wav = wav_i16_mono(16_000, &[0; 1_600]);
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
        assert_eq!(item.mm_hash, "uuid:audio-1");

        let features = &item.data[AUDIO_PRIMARY_KEY];
        assert!(matches!(&features.field, MmField::Batched(_)));
        assert!(matches!(
            features.data.as_ref(),
            Some(MmKwargValue::Tensor(tensor))
                if tensor.dtype.as_str() == "float32" && tensor.shape.first() == Some(&128)
        ));
        let lengths = &item.data["audio_feature_lengths"];
        assert!(matches!(&lengths.field, MmField::Batched(_)));
        assert!(matches!(
            lengths.data.as_ref(),
            Some(MmKwargValue::Tensor(tensor))
                if tensor.dtype.as_str() == "int64" && tensor.shape.is_empty()
        ));
    }

    #[tokio::test]
    async fn inkling_tracker_and_processor_use_standard_audio_key() {
        let info = inkling_info(serde_json::json!(1024));
        let wav = wav_i16_mono(16_000, &[0; 1_600]);
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
        // Every replacement token is the soft placeholder; the typed
        // `<|content_audio_input|>` marker is the structural prefix owned by
        // the chat template.
        assert!(
            prepared.replacements[0]
                .tokens
                .iter()
                .all(|token| *token == INKLING_AUDIO_EMBED_ID)
        );
        let item = &prepared.items[0];
        assert_eq!(item.mm_hash, "uuid:audio-1");

        let features = &item.data[AUDIO_PRIMARY_KEY];
        assert!(matches!(&features.field, MmField::Flat(_)));
        assert!(matches!(
            features.data.as_ref(),
            Some(MmKwargValue::Tensor(tensor))
                if tensor.dtype.as_str() == "float32" && tensor.shape.get(1) == Some(&80)
        ));
        let count = &item.data["tokens_per_item"];
        assert!(matches!(&count.field, MmField::Batched(_)));
        assert!(matches!(
            count.data.as_ref(),
            Some(MmKwargValue::Tensor(tensor))
                if tensor.dtype.as_str() == "int64" && tensor.shape.is_empty()
        ));
    }
}
