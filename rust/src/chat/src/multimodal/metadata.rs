// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Metadata-only preparation using the model's existing prompt and field rules.

use serde_json::Value;
use vllm_engine_core_client::protocol::dtype::ModelDtype;

use super::{ModalitySupport, MultimodalModelInfo, PreparedMedia, item, tensor};
use crate::error::{Result, bail_multimodal, multimodal};

impl MultimodalModelInfo {
    /// Expand metadata-only references without fetching or preprocessing media.
    pub(super) fn prepare_metadata_only(
        &self,
        support: &ModalitySupport,
        references: Vec<(Value, Option<String>)>,
        model_dtype: ModelDtype,
    ) -> Result<PreparedMedia> {
        if !self.enable_mm_embeds {
            bail_multimodal!("metadata-only inputs require --enable-mm-embeds");
        }
        let spec = &support.spec;
        let config = &self.context.config;
        let max_tokens = config
            .pointer("/text_config/max_position_embeddings")
            .or_else(|| config.pointer("/thinker_config/text_config/max_position_embeddings"))
            .or_else(|| config.pointer("/thinker_config/max_position_embeddings"))
            .or_else(|| config.get("max_position_embeddings"))
            .and_then(Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| multimodal!("model has no metadata token limit"))?;
        let (payloads, uuids): (Vec<_>, Vec<_>) = references.into_iter().unzip();
        let uuids = uuids
            .into_iter()
            .map(|uuid| {
                uuid.filter(|id| !id.is_empty())
                    .ok_or_else(|| multimodal!("metadata-only inputs require a nonempty uuid"))
            })
            .collect::<Result<Vec<_>>>()?;
        let prepared = spec.raw.prepare_metadata_only(
            &self.context.metadata(),
            &support.config,
            spec.modality,
            &payloads,
            max_tokens,
        )?;
        let tensors =
            tensor::collect_metadata_tensors(prepared.metadata.model_specific, model_dtype)?;
        let hashes = uuids.clone();
        let items =
            item::build_items(spec, tensors, hashes, uuids.into_iter().map(Some).collect())?;
        Ok(PreparedMedia {
            modality: spec.modality,
            placeholder: support.placeholder.clone(),
            replacements: prepared.replacements,
            items,
        })
    }
}

#[cfg(test)]
mod tests {
    use llm_multimodal::MediaContentPart;
    use serde_json::json;
    use std::collections::HashMap;
    use vllm_engine_core_client::protocol::dtype::ModelDtype;
    use vllm_engine_core_client::protocol::multimodal::{
        MmBatchedField, MmField, MmFieldElem, MmKwargValue,
    };
    use vllm_engine_core_client::protocol::tensor::WireTensor;

    use super::*;

    fn info() -> MultimodalModelInfo {
        let mut info = super::super::tests::qwen3_vl_info();
        info.enable_mm_embeds = true;
        info.context.config["max_position_embeddings"] = json!(128);
        info
    }

    fn prepare(
        info: &MultimodalModelInfo,
        payload: Value,
        uuid: Option<String>,
    ) -> Result<PreparedMedia> {
        info.prepare_metadata_only(
            info.image.as_ref().unwrap(),
            vec![(payload, uuid)],
            ModelDtype::BFloat16,
        )
    }

    #[tokio::test]
    async fn metadata_images_preserve_duplicate_ids_and_placeholder_positions() {
        let info = info();
        let marker = info.image.as_ref().unwrap().placeholder.marker_token_id;
        let mut tokens = vec![7, marker, 8, marker];
        let images =
            [json!([1, 4, 8]), json!([1, 4, 8])].map(|grid| MediaContentPart::ImageEmbeds {
                payload: json!({"image_grid_thw": grid}),
                uuid: Some("same-image".to_owned()),
            });
        let features = info
            .prepare_multimodal(images.into(), &mut tokens, ModelDtype::BFloat16)
            .await
            .unwrap();
        assert_eq!(features.len(), 2);
        assert_eq!(tokens.len(), 18);
        assert_eq!(features[0].mm_position.offset, 1);
        assert_eq!(features[1].mm_position.offset, 10);
        for feature in features {
            assert_eq!(feature.identifier, "same-image");
            assert_eq!(feature.mm_position.length, 8);
            let data = feature.data.unwrap();
            let expected = HashMap::from([(
                "image_grid_thw".to_owned(),
                MmFieldElem {
                    data: Some(MmKwargValue::Tensor(
                        WireTensor::from_i64(vec![3], vec![1, 4, 8]).unwrap(),
                    )),
                    field: MmField::Batched(MmBatchedField { keep_on_cpu: true }),
                },
            )]);
            assert_eq!(
                serde_json::to_value(data).unwrap(),
                serde_json::to_value(expected).unwrap()
            );
        }
    }

    #[test]
    fn metadata_rejects_invalid_grids_before_expansion() {
        for grid in [
            json!([1, 0, 8]),
            json!([1, 3, 8]),
            json!([1, 32, 32]),
            json!([1, -2, 8]),
            json!([true, 2, 8]),
            json!([1, 2]),
        ] {
            assert!(
                prepare(
                    &info(),
                    json!({"image_grid_thw": grid}),
                    Some("image".to_owned())
                )
                .is_err()
            );
        }
    }

    #[test]
    fn metadata_requires_opt_in_and_uuid() {
        let payload = json!({"image_grid_thw": [1, 2, 2]});
        assert!(prepare(&info(), payload.clone(), None).is_err());
        let mut info = info();
        info.enable_mm_embeds = false;
        assert!(prepare(&info, payload, Some("image".to_owned())).is_err());
    }

    #[test]
    fn metadata_rejects_undeclared_fields_and_embedding_values() {
        for key in ["unknown", "pixel_values", "image_embeds"] {
            let mut payload = json!({"image_grid_thw": [1, 2, 2]});
            payload[key] = json!([1, 2, 3]);
            assert!(prepare(&info(), payload, Some("image".to_owned())).is_err());
        }
    }

    #[test]
    fn metadata_budget_applies_to_the_whole_batch() {
        let info = info();
        let references = (0..2)
            .map(|i| (json!({"image_grid_thw": [1, 16, 32]}), Some(i.to_string())))
            .collect();
        assert!(
            info.prepare_metadata_only(
                info.image.as_ref().unwrap(),
                references,
                ModelDtype::BFloat16
            )
            .is_err()
        );
    }

    #[test]
    fn metadata_uses_loaded_preprocessor_merge_size() {
        let mut info = info();
        info.image.as_mut().unwrap().config.merge_size = Some(4);
        let prepared = prepare(
            &info,
            json!({"image_grid_thw": [1, 4, 8]}),
            Some("image".into()),
        )
        .unwrap();
        expect_test::expect![[r#"
            [
                151655,
                151655,
            ]
        "#]]
        .assert_debug_eq(&prepared.replacements[0].tokens);
    }

    #[tokio::test]
    async fn token_count_metadata_builds_features_without_auxiliary_tensors() {
        let mut info = super::super::tests::test_info(
            "llava-1.5",
            json!({"model_type": "llava", "max_position_embeddings": 128}),
            vllm_tokenizer::test_utils::TestTokenizer::new().with_regular_token("<image>", 32000),
        );
        info.enable_mm_embeds = true;
        let mut tokens = vec![32000];
        let features = info
            .prepare_multimodal(
                vec![MediaContentPart::ImageEmbeds {
                    payload: json!({"num_image_tokens": [2]}),
                    uuid: Some("image".into()),
                }],
                &mut tokens,
                ModelDtype::BFloat16,
            )
            .await
            .unwrap();
        expect_test::expect![[r#"
            (
                [
                    32000,
                    32000,
                ],
                "image",
                2,
                0,
            )
        "#]]
        .assert_debug_eq(&(
            tokens,
            &features[0].identifier,
            features[0].mm_position.length,
            features[0].data.as_ref().unwrap().len(),
        ));
    }

    #[tokio::test]
    async fn model_rejects_foreign_metadata_before_prompt_expansion() {
        let mut info = super::super::tests::llama4_info();
        info.enable_mm_embeds = true;
        info.context.config["max_position_embeddings"] = json!(128);
        let mut tokens = vec![super::super::tests::LLAMA4_IMAGE_ID];
        let result = info
            .prepare_multimodal(
                vec![MediaContentPart::ImageEmbeds {
                    payload: json!({"image_grid_thw": [1, 2, 2]}),
                    uuid: Some("image".into()),
                }],
                &mut tokens,
                ModelDtype::BFloat16,
            )
            .await;
        assert!(
            matches!(result, Err(crate::error::Error::Multimodal(message))
            if message.contains("invalid metadata-only input"))
        );
        assert_eq!(tokens, [super::super::tests::LLAMA4_IMAGE_ID]);
    }

    #[tokio::test]
    async fn metadata_adapter_reuses_non_qwen_structural_prompt_rules() {
        let mut info = super::super::tests::llama4_info();
        info.enable_mm_embeds = true;
        info.context.config["max_position_embeddings"] = json!(128);
        info.context.config["vision_config"] = json!({"image_size": 4, "patch_size": 1});
        let mut tokens = vec![super::super::tests::LLAMA4_IMAGE_ID];
        let features = info
            .prepare_multimodal(
                vec![MediaContentPart::ImageEmbeds {
                    payload: json!({"aspect_ratios": [1, 1]}),
                    uuid: Some("image".into()),
                }],
                &mut tokens,
                ModelDtype::BFloat16,
            )
            .await
            .unwrap();
        let feature = &features[0];
        expect_test::expect![[r#"
            (
                [
                    200088,
                    200090,
                    200092,
                    200092,
                    200092,
                    200092,
                    200089,
                ],
                0,
                7,
                [
                    "aspect_ratios",
                ],
            )
        "#]]
        .assert_debug_eq(&(
            tokens,
            feature.mm_position.offset,
            feature.mm_position.length,
            feature.data.as_ref().unwrap().keys().collect::<Vec<_>>(),
        ));
    }
}
