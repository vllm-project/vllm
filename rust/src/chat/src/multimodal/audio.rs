// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Audio-modality preparation through `llm-multimodal`.

use std::sync::Arc;

use itertools::izip;
use llm_multimodal::{AudioClip, Modality, PreprocessedEncoderInputs, PromptReplacement};
use vllm_engine_core_client::protocol::dtype::ModelDtype;

use super::cache::{CacheLookup, merge_slots, put_or_inline_and_track};
use super::timing::MM_STAGE_TARGET;
use super::{AudioModalitySupport, MultimodalModelInfo, PreparedItem, PreparedMedia, item};
use crate::error::{Error, Result, bail_multimodal, multimodal};

/// Forward-kwargs name of the primary audio encoder input.
pub(super) const AUDIO_PRIMARY_KEY: &str = "input_audio_features";

impl MultimodalModelInfo {
    /// Preprocess fetched audio clips as one batch and build per-item features.
    ///
    /// With the SHM processor cache enabled, cached clips skip preprocessing
    /// entirely: their prompt replacements come from the cache shadow and
    /// their feature data is the shm address item. Cache misses are
    /// preprocessed as one batch.
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
        let support = self.audio.as_ref().ok_or_else(|| Error::UnsupportedModality {
            modality: Modality::Audio.to_string(),
        })?;

        let mut slots = Vec::with_capacity(clips.len());
        let mut miss_clips = Vec::new();
        let mut miss_uuids = Vec::new();
        for (clip, uuid) in clips.into_iter().zip(uuids) {
            let lookup = match &self.mm_processor_cache {
                Some(cache) => cache.lookup(&clip.hash, &clip.hash, uuid),
                None => CacheLookup::Miss(uuid),
            };
            match lookup {
                CacheLookup::Hit(item, replacement) => slots.push(Some((item, replacement))),
                CacheLookup::Miss(uuid) => {
                    slots.push(None);
                    miss_clips.push(clip);
                    miss_uuids.push(uuid);
                }
            }
        }

        let mut misses = Vec::new();
        if !miss_clips.is_empty() {
            let (replacements, mut items) =
                self.preprocess_audio_batch(support, miss_clips, miss_uuids).await?;
            for (item, replacement) in izip!(&mut items, &replacements) {
                item.data = put_or_inline_and_track(
                    self.mm_processor_cache.as_ref(),
                    &item.hash,
                    replacement,
                    std::mem::take(&mut item.data),
                );
            }
            misses = izip!(items, replacements).collect();
        }
        let (replacements, items) = merge_slots(slots, misses);

        Ok(PreparedMedia {
            modality: Modality::Audio,
            placeholder: support.placeholder.clone(),
            replacements,
            items,
        })
    }

    /// Preprocess fetched audio clips as one batch and build per-item
    /// replacements and engine kwargs.
    async fn preprocess_audio_batch(
        &self,
        support: &AudioModalitySupport,
        clips: Vec<Arc<AudioClip>>,
        uuids: Vec<Option<String>>,
    ) -> Result<(Vec<PromptReplacement>, Vec<PreparedItem>)> {
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
        Ok((replacements, items))
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
    const INKLING_AUDIO_MARKER_ID: u32 = 200_020;
    const INKLING_AUDIO_EMBED_ID: i32 = 200_053;

    fn inkling_info(decoder_dmodel: serde_json::Value) -> MultimodalModelInfo {
        let config = serde_json::json!({
            "model_type": "inkling_mm_model",
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
            .with_regular_token("<|content_audio_input|>", INKLING_AUDIO_MARKER_ID);
        let context = MultimodalModelContext {
            model_id: "inkling-test".to_string(),
            model_type: Some("inkling_mm_model".to_string()),
            config,
            tokenizer: TokenizerResolver(Arc::new(tokenizer)),
        };

        MultimodalModelInfo::from_loaded(
            context,
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

        MultimodalModelInfo::from_loaded(
            context,
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

        assert_eq!(
            info.placeholder_token(Modality::Audio),
            Some("<|content_audio_input|>")
        );
        assert_eq!(support.placeholder.marker_token_id, INKLING_AUDIO_MARKER_ID);
        assert_eq!(
            support.placeholder.embed_token_id,
            INKLING_AUDIO_EMBED_ID as u32
        );
        assert_eq!(support.spec.primary_key(), AUDIO_PRIMARY_KEY);
        assert!(matches!(
            &support.spec.field_layouts.encoder_input,
            llm_multimodal::FieldLayout::Flat { sizes_key }
                if sizes_key == "num_audio_tokens"
        ));
        assert!(matches!(
            support.spec.field_layouts.model_specific.get("num_audio_tokens"),
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
    async fn audio_pipeline_attributes_stage_timings_to_request() {
        use tracing::Instrument as _;
        use tracing::instrument::WithSubscriber as _;
        use tracing_subscriber::layer::SubscriberExt as _;

        use crate::{mm_request_span, mm_timing_layer};

        // Other tests exercise the same instrumented callsites under
        // NoSubscriber, which caches `Interest::never` globally; rebuild the
        // callsite cache so this test's thread-local subscriber sees them.
        tracing::callsite::rebuild_interest_cache();

        let info = qwen3_asr_info();
        let (layer, timings) = mm_timing_layer();
        let parts = vec![MediaContentPart::AudioData {
            data: wav_i16_mono(16_000, &[0; 1_600]),
            mime_type: Some("audio/wav".to_string()),
            uuid: Some("audio-1".to_string()),
        }];
        async {
            info.prepare_multimodal(parts, &mut vec![AUDIO_PAD_ID], ModelDtype::Float32, None)
                .instrument(mm_request_span("request-1"))
                .await
                .unwrap();
        }
        .with_subscriber(tracing_subscriber::registry().with(layer))
        .await;

        let records = timings.stat();
        assert_eq!(records.len(), 1);
        let mut stages = records["request-1"].keys().collect::<Vec<_>>();
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
    async fn cache_hit_skips_audio_preprocessing() {
        use super::super::cache::tests::{address_of, is_address_item, test_shm_cache};

        let info = qwen3_asr_info().with_mm_processor_cache(Some(test_shm_cache()));
        let media_part = || MediaContentPart::AudioData {
            data: wav_i16_mono(16_000, &[0; 1_600]),
            mime_type: Some("audio/wav".to_string()),
            uuid: Some("audio-1".to_string()),
        };

        let mut tokens_first = vec![1, AUDIO_PAD_ID, 2];
        let first = info
            .prepare_multimodal(
                vec![media_part()],
                &mut tokens_first,
                ModelDtype::Float32,
                None,
            )
            .await
            .unwrap();
        assert_eq!(first.len(), 1);
        let data_first = first[0].data.as_ref().expect("feature data");
        assert!(
            is_address_item(data_first),
            "first prepare should put the item into shm and send the address item"
        );

        let mut tokens_second = vec![1, AUDIO_PAD_ID, 2];
        let second = info
            .prepare_multimodal(
                vec![media_part()],
                &mut tokens_second,
                ModelDtype::Float32,
                None,
            )
            .await
            .unwrap();
        assert_eq!(second.len(), 1);
        let data_second = second[0].data.as_ref().expect("feature data");
        // A cache hit reuses the same shm slot; a re-put of the same hash
        // would hit the duplicate-key inline fallback instead.
        assert!(is_address_item(data_second));
        assert_eq!(address_of(data_second), address_of(data_first));

        // The shadowed prompt replacement reproduces the expansion exactly.
        assert_eq!(tokens_second, tokens_first);
        assert_eq!(first[0].mm_hash, second[0].mm_hash);
        assert_eq!(first[0].mm_position.offset, second[0].mm_position.offset);
        assert_eq!(first[0].mm_position.length, second[0].mm_position.length);
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
        assert_eq!(
            prepared.replacements[0].tokens[0],
            INKLING_AUDIO_MARKER_ID as i32
        );
        assert!(
            prepared.replacements[0].tokens[1..]
                .iter()
                .all(|token| *token == INKLING_AUDIO_EMBED_ID)
        );
        let item = &prepared.items[0];
        assert_eq!(item.hash, expected_hash);
        assert_eq!(item.uuid.as_deref(), Some("audio-1"));

        let features = &item.data[AUDIO_PRIMARY_KEY];
        assert!(matches!(&features.field, MmField::Flat(_)));
        assert!(matches!(
            features.data.as_ref(),
            Some(MmKwargValue::Tensor(tensor))
                if tensor.dtype.as_str() == "float32" && tensor.shape.get(1) == Some(&80)
        ));
        let count = &item.data["num_audio_tokens"];
        assert!(matches!(&count.field, MmField::Batched(_)));
        assert!(matches!(
            count.data.as_ref(),
            Some(MmKwargValue::Tensor(tensor))
                if tensor.dtype.as_str() == "int64" && tensor.shape.is_empty()
        ));
    }
}
