// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use llm_multimodal::{Modality, PromptReplacement};
use serde::Deserialize;
use serde_json::Value;
use vllm_engine_core_client::protocol::multimodal::{
    MmBatchedField, MmField, MmFieldElem, MmKwargValue,
};
use vllm_engine_core_client::protocol::tensor::WireTensor;

use super::{MultimodalModelInfo, PreparedItem, PreparedMedia};
use crate::error::{Error, Result, bail_multimodal, multimodal};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ImageMetadata {
    image_grid_thw: [u32; 3],
}

impl MultimodalModelInfo {
    /// Expand Qwen image references without fetching or preprocessing pixels.
    pub(super) fn prepare_image_metadata(
        &self,
        images: Vec<(Value, Option<String>)>,
    ) -> Result<PreparedMedia> {
        if !self.enable_mm_embeds {
            bail_multimodal!("image_embeds requires --enable-mm-embeds");
        }
        let support = self
            .image
            .as_ref()
            .ok_or_else(|| multimodal!("image_embeds requires a supported image model"))?;
        if !matches!(support.spec.raw.name(), "qwen_vl" | "qwen3_vl") {
            bail_multimodal!("metadata-only image_embeds is only supported for Qwen VL models");
        }
        let config = &self.context.config;
        let merge = config
            .pointer("/vision_config/spatial_merge_size")
            .and_then(Value::as_u64)
            .unwrap_or(2);
        let max_tokens = config
            .pointer("/text_config/max_position_embeddings")
            .or_else(|| config.get("max_position_embeddings"))
            .and_then(Value::as_u64)
            .ok_or_else(|| multimodal!("model has no metadata token limit"))?;
        let mut prepared = PreparedMedia {
            modality: Modality::Image,
            placeholder: support.placeholder.clone(),
            replacements: Vec::with_capacity(images.len()),
            items: Vec::with_capacity(images.len()),
        };
        let mut total_tokens = 0u64;
        for (payload, uuid) in images {
            let uuid = uuid.filter(|id| !id.is_empty()).ok_or_else(|| {
                multimodal!("metadata-only image_embeds requires a nonempty uuid")
            })?;
            let metadata: ImageMetadata = serde_json::from_value(payload)
                .map_err(|error| multimodal!("invalid image metadata: {error}"))?;
            let [t, h, w] = metadata.image_grid_thw.map(u64::from);
            if t != 1 || merge == 0 || h == 0 || w == 0 || h % merge != 0 || w % merge != 0 {
                bail_multimodal!("invalid image_grid_thw for the model's spatial merge size");
            }
            let tokens = (h / merge)
                .checked_mul(w / merge)
                .ok_or_else(|| multimodal!("image_grid_thw token count overflow"))?;
            total_tokens = total_tokens
                .checked_add(tokens)
                .filter(|&total| total <= max_tokens)
                .ok_or_else(|| multimodal!("image metadata exceeds model context length"))?;
            prepared.replacements.push(PromptReplacement::sequence(
                Modality::Image,
                &support.placeholder.token,
                vec![support.placeholder.embed_token_id as i32; tokens as usize],
            ));
            let grid =
                WireTensor::from_i64(vec![3], metadata.image_grid_thw.map(i64::from).to_vec())
                    .map_err(Error::Multimodal)?;
            prepared.items.push(PreparedItem {
                data: [(
                    "image_grid_thw".to_owned(),
                    MmFieldElem {
                        data: Some(MmKwargValue::Tensor(grid)),
                        field: MmField::Batched(MmBatchedField {
                            keep_on_cpu: support.spec.keep_on_cpu_keys.contains("image_grid_thw"),
                        }),
                    },
                )]
                .into(),
                hash: uuid.clone(),
                uuid: Some(uuid),
            });
        }
        Ok(prepared)
    }
}

#[cfg(test)]
mod tests {
    use llm_multimodal::MediaContentPart;
    use serde_json::json;
    use vllm_engine_core_client::protocol::dtype::ModelDtype;

    use super::*;

    fn info() -> MultimodalModelInfo {
        let mut info = super::super::tests::qwen3_vl_info();
        info.enable_mm_embeds = true;
        info.context.config["max_position_embeddings"] = json!(128);
        info
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
            assert_eq!(
                data.keys().map(String::as_str).collect::<Vec<_>>(),
                ["image_grid_thw"]
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
                info()
                    .prepare_image_metadata(vec![(
                        json!({"image_grid_thw": grid}),
                        Some("image".to_owned())
                    )])
                    .is_err()
            );
        }
    }

    #[test]
    fn metadata_requires_opt_in_and_uuid() {
        let payload = json!({"image_grid_thw": [1, 2, 2]});
        assert!(info().prepare_image_metadata(vec![(payload.clone(), None)]).is_err());
        let mut info = info();
        info.enable_mm_embeds = false;
        assert!(info.prepare_image_metadata(vec![(payload, Some("image".to_owned()))]).is_err());
    }
}
