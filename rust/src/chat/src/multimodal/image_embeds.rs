// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Metadata-only `image_embeds` content parts.
//!
//! The Python disaggregated encoder proxy sends these to the prefill and
//! decode instances, which take each image's encoder output from an EC or KV
//! transfer: the part carries a `uuid` and only the model-specific metadata
//! that sizes the image's placeholder, such as Qwen-VL's patch grid. This is
//! a compatibility path for that proxy; a token-level client sends the same
//! items as preprocessed features.
//!
//! The model knowledge is one row per model spec in [`size_image`], local
//! until `llm-multimodal` supports metadata-only image embeddings; delete this
//! module then.

use std::collections::HashMap;
use std::num::NonZeroU32;

use llm_multimodal::{Modality, PreprocessedEncoderInputs};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use thiserror_ext::AsReport as _;
use vllm_engine_core_client::protocol::multimodal::{
    MmBatchedField, MmField, MmFieldElem, MmKwargValue, MmKwargsItem,
};
use vllm_engine_core_client::protocol::tensor::WireTensor;

use super::{MultimodalModelInfo, PreparedItem, PreparedMedia, VisionModalitySupport};
use crate::error::{
    Error, Result, bail_invalid_image_embeds, bail_multimodal, invalid_image_embeds,
};

/// Qwen2-VL family and Qwen3-VL metadata: the image's patch grid.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct QwenVlMetadata {
    /// `[t, h, w]`, in patches.
    image_grid_thw: [NonZeroU32; 3],
}

/// DeepSeek-V4.1 metadata: the image's token grid.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DeepseekV41Metadata {
    /// `[h, w]`, in tokens.
    llm_grid: [NonZeroU32; 2],
}

/// Kimi-K3 metadata: the image's size before resizing.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct KimiK3Metadata {
    /// `[width, height]`, in pixels.
    image_sizes: [NonZeroU32; 2],
}

/// One image as its metadata sizes it.
struct SizedImage {
    /// `None` when the count overflows.
    num_tokens: Option<usize>,
    /// The spec's `item_sizes` entry, for a spec whose prompt replacement
    /// reads it.
    item_size: Option<(u32, u32)>,
    /// The kwargs the engine reads besides the encoder, as Python builds them
    /// from the same metadata: one int64 row per key.
    cpu_kwargs: Vec<(&'static str, Vec<i64>)>,
}

/// Size one image from its metadata by the model's own rule.
fn size_image(support: &VisionModalitySupport, metadata: Value) -> Result<SizedImage> {
    let rows = |dims: &[NonZeroU32]| dims.iter().map(|dim| i64::from(dim.get())).collect();
    match support.spec.raw.name() {
        "qwen_vl" | "qwen3_vl" => {
            let QwenVlMetadata { image_grid_thw } = decode(metadata)?;
            // Not the processor's default: the grid is only as meaningful as
            // the merge size the encoder side used.
            let Some(merge_size) = support.config.merge_size else {
                bail_multimodal!(
                    "the image preprocessor config sets no `merge_size` to size metadata-only image_embeds by"
                );
            };
            // `t * h * w / merge_size^2`, as Python's prompt update counts it.
            let num_tokens = image_grid_thw
                .iter()
                .try_fold(1_usize, |product, dim| {
                    product.checked_mul(dim.get() as usize)
                })
                .zip(merge_size.checked_pow(2))
                .and_then(|(patches, merge_area)| patches.checked_div(merge_area));
            Ok(SizedImage {
                num_tokens,
                item_size: None,
                cpu_kwargs: vec![("image_grid_thw", rows(&image_grid_thw))],
            })
        }
        "deepseek_v41" => {
            let DeepseekV41Metadata { llm_grid } = decode(metadata)?;
            let [height, width] = llm_grid.map(|dim| dim.get() as usize);
            // A row of `w` tokens plus a newline per grid row, and the two
            // span delimiters.
            let num_tokens = width
                .checked_add(1)
                .and_then(|row| row.checked_mul(height))
                .and_then(|tokens| tokens.checked_add(2));
            Ok(SizedImage {
                num_tokens,
                item_size: None,
                cpu_kwargs: vec![("llm_grid", rows(&llm_grid))],
            })
        }
        "kimi_k3" => {
            let KimiK3Metadata { image_sizes } = decode(metadata)?;
            let [width, height] = image_sizes.map(NonZeroU32::get);
            let num_tokens =
                support.processor.calculate_num_tokens(width, height).map_err(|error| {
                    invalid_image_embeds!(
                        "invalid metadata-only image_embeds for this model: {}",
                        error.as_report()
                    )
                })?;
            Ok(SizedImage {
                num_tokens: Some(num_tokens),
                // The `image {width}x{height}` text of the replacement.
                item_size: Some((width, height)),
                cpu_kwargs: vec![
                    ("image_sizes", rows(&image_sizes)),
                    ("image_num_tokens", vec![num_tokens as i64]),
                ],
            })
        }
        _ => bail_invalid_image_embeds!("this model does not accept metadata-only image_embeds"),
    }
}

/// Decode one model's metadata.
fn decode<T: DeserializeOwned>(metadata: Value) -> Result<T> {
    serde_json::from_value(metadata).map_err(|error| {
        invalid_image_embeds!(
            "invalid metadata-only image_embeds for this model: {}",
            error.as_report()
        )
    })
}

impl MultimodalModelInfo {
    /// Expand metadata-only image embeddings into features without encoder
    /// input, each identified by its required `uuid`.
    ///
    /// `token_budget` is the room the prompt's text leaves, if known: unlike
    /// a decoded image, the metadata can claim any number of tokens.
    pub(super) fn prepare_image_embeds(
        &self,
        parts: Vec<(Value, Option<String>)>,
        token_budget: Option<usize>,
    ) -> Result<PreparedMedia> {
        let support = self.image.as_ref().ok_or_else(|| Error::UnsupportedModality {
            modality: Modality::Image.to_string(),
        })?;

        let mut feature_token_counts = Vec::with_capacity(parts.len());
        let mut item_sizes = Vec::new();
        let mut items = Vec::with_capacity(parts.len());
        for (metadata, uuid) in parts {
            let Some(uuid) = uuid.filter(|uuid| !uuid.is_empty()) else {
                bail_invalid_image_embeds!("metadata-only image_embeds require a nonempty `uuid`");
            };
            // The whole payload, or a value of its fields, as a base64 tensor.
            let base64 = match &metadata {
                Value::String(_) => true,
                Value::Object(fields) => fields.values().any(Value::is_string),
                _ => false,
            };
            if base64 {
                bail_invalid_image_embeds!(
                    "base64 image_embeds tensors are not supported by the Rust frontend"
                );
            }
            let image = size_image(support, metadata)?;
            match image.num_tokens {
                Some(0) => bail_invalid_image_embeds!(
                    "metadata-only image_embeds size an image of no tokens"
                ),
                Some(num_tokens) => feature_token_counts.push(num_tokens),
                None => {
                    bail_invalid_image_embeds!("metadata-only image_embeds dimensions overflow")
                }
            }
            item_sizes.extend(image.item_size);
            items.push(PreparedItem {
                data: cpu_kwargs(image.cpu_kwargs)?,
                // No content to hash: the required `uuid` is the cache key.
                hash: String::new(),
                uuid: Some(uuid),
            });
        }
        if let Some(budget) = token_budget {
            let total = feature_token_counts
                .iter()
                .try_fold(0_usize, |total, &count| total.checked_add(count));
            if total.is_none_or(|total| total > budget) {
                bail_invalid_image_embeds!(
                    "metadata-only image_embeds size more image tokens than the {budget} the prompt has room for"
                );
            }
        }

        // The specs' prompt replacements read only the token counts and item
        // sizes, never the encoder input this leaves empty.
        let preprocessed = PreprocessedEncoderInputs {
            encoder_input: ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&[0])).into(),
            feature_token_counts,
            item_sizes,
            model_specific: HashMap::new(),
        };
        let replacements = support.spec.prompt_replacements_for(&self.context, &preprocessed)?;
        if replacements.len() != items.len() {
            bail_multimodal!(
                "number of image prompt replacements {} does not match number of image embeddings {}",
                replacements.len(),
                items.len()
            );
        }

        Ok(PreparedMedia {
            modality: Modality::Image,
            placeholder: support.placeholder.clone(),
            replacements,
            items,
        })
    }
}

/// Lower one item's CPU kwargs: each batched per item and kept on the CPU, as
/// the models' field configs declare them.
fn cpu_kwargs(kwargs: Vec<(&'static str, Vec<i64>)>) -> Result<MmKwargsItem> {
    kwargs
        .into_iter()
        .map(|(key, row)| {
            let tensor = WireTensor::from_i64(vec![row.len()], row).map_err(Error::Multimodal)?;
            let elem = MmFieldElem {
                data: Some(MmKwargValue::Tensor(tensor)),
                field: MmField::Batched(MmBatchedField { keep_on_cpu: true }),
            };
            Ok((key.to_owned(), elem))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use llm_multimodal::{MediaContentPart, PreProcessorConfig};
    use serde_json::json;
    use thiserror_ext::AsReport as _;
    use vllm_engine_core_client::protocol::dtype::ModelDtype;
    use vllm_engine_core_client::protocol::multimodal::{
        MmFeatureSpec, MmFeatures, MmKwargValue, MmKwargsItem,
    };
    use vllm_tokenizer::test_utils::TestTokenizer;

    use super::super::tests::{
        DEEPSEEK_V41_IMAGE_ID, DEEPSEEK_V41_IMAGE_PAD_ID, QWEN3_IMAGE_PAD_ID, qwen3_vl_tokenizer,
    };
    use super::super::{MultimodalModelContext, MultimodalModelInfo, TokenizerResolver};
    use super::cpu_kwargs;

    const KIMI_MEDIA_PAD_ID: u32 = 163605;

    /// Two images whose width and height differ, so a swapped order shows.
    const IMAGE_SIZES: [(u32, u32); 2] = [(96, 48), (40, 72)];

    fn info(
        model_type: &str,
        config: serde_json::Value,
        tokenizer: TestTokenizer,
        image_config: serde_json::Value,
    ) -> MultimodalModelInfo {
        let context = MultimodalModelContext {
            model_id: format!("{model_type}-test"),
            model_type: Some(model_type.to_owned()),
            config,
            tokenizer: TokenizerResolver(Arc::new(tokenizer)),
        };
        MultimodalModelInfo::from_loaded(
            context,
            PreProcessorConfig::from_value(image_config).unwrap(),
            PreProcessorConfig::default(),
            PreProcessorConfig::default(),
            HashMap::new(),
        )
        .unwrap()
        .unwrap()
    }

    /// Qwen3-VL with its checkpoint's image preprocessor geometry.
    fn qwen3_vl_info() -> MultimodalModelInfo {
        info(
            "qwen3_vl",
            json!({
                "model_type": "qwen3_vl",
                "image_token_id": QWEN3_IMAGE_PAD_ID,
                "vision_config": {"patch_size": 16}
            }),
            qwen3_vl_tokenizer(),
            json!({"patch_size": 16, "merge_size": 2, "temporal_patch_size": 2}),
        )
    }

    fn deepseek_v41_info() -> MultimodalModelInfo {
        info(
            "deepseek_v41",
            json!({"model_type": "deepseek_v41", "image_token_id": DEEPSEEK_V41_IMAGE_ID}),
            TestTokenizer::new()
                .with_regular_token("<｜deepseek_image｜>", DEEPSEEK_V41_IMAGE_ID)
                .with_regular_token("<|place_holder_mm_span_0436|>", DEEPSEEK_V41_IMAGE_PAD_ID),
            json!({}),
        )
    }

    fn kimi_k3_info() -> MultimodalModelInfo {
        info(
            "kimi_k3",
            json!({"model_type": "kimi_k3", "media_placeholder_token_id": KIMI_MEDIA_PAD_ID}),
            TestTokenizer::new()
                .with_regular_token("<|media_begin|>", 163602)
                .with_regular_token("<|media_content|>", 163603)
                .with_regular_token("<|media_end|>", 163604)
                .with_regular_token("<|media_pad|>", KIMI_MEDIA_PAD_ID),
            json!({}),
        )
    }

    fn png(width: u32, height: u32) -> Vec<u8> {
        let mut png = Vec::new();
        ::image::RgbImage::from_fn(width, height, |x, y| {
            ::image::Rgb([(x * 7) as u8, (y * 5) as u8, 128])
        })
        .write_to(
            &mut std::io::Cursor::new(&mut png),
            ::image::ImageFormat::Png,
        )
        .unwrap();
        png
    }

    fn uuid(index: usize) -> Option<String> {
        Some(format!("image-{index}"))
    }

    /// Prepare `parts` in a prompt with one `marker` per part between text.
    async fn prepare(
        info: &MultimodalModelInfo,
        marker: u32,
        parts: Vec<MediaContentPart>,
        max_model_len: Option<usize>,
    ) -> crate::Result<(Vec<u32>, MmFeatures)> {
        let mut tokens = vec![11];
        for _ in &parts {
            tokens.extend([marker, 12]);
        }
        let features = info
            .prepare_multimodal(parts, &mut tokens, ModelDtype::BFloat16, max_model_len)
            .await?;
        Ok((tokens, features))
    }

    /// Prepare [`IMAGE_SIZES`] as raw images.
    async fn prepare_raw(info: &MultimodalModelInfo, marker: u32) -> (Vec<u32>, MmFeatures) {
        let parts = IMAGE_SIZES
            .iter()
            .enumerate()
            .map(|(index, &(width, height))| MediaContentPart::ImageData {
                data: png(width, height),
                mime_type: Some("image/png".to_owned()),
                uuid: uuid(index),
                detail: None,
            })
            .collect();
        prepare(info, marker, parts, None).await.unwrap()
    }

    /// Prepare one metadata-only part per payload, as the proxy sends them.
    async fn prepare_metadata(
        info: &MultimodalModelInfo,
        marker: u32,
        payloads: Vec<serde_json::Value>,
    ) -> (Vec<u32>, MmFeatures) {
        let parts = payloads
            .into_iter()
            .enumerate()
            .map(|(index, payload)| MediaContentPart::ImageEmbeds {
                payload,
                uuid: uuid(index),
            })
            .collect();
        prepare(info, marker, parts, None).await.unwrap()
    }

    /// The flat integer row of `key` in `feature`, as the encoder side
    /// echoes it as metadata.
    fn echoed(feature: &MmFeatureSpec, key: &str) -> serde_json::Value {
        let Some(MmKwargValue::Tensor(tensor)) = &feature.data.as_ref().unwrap()[key].data else {
            panic!("`{key}` is not a tensor");
        };
        let row: Vec<i64> = tensor
            .data
            .as_raw_view()
            .unwrap()
            .chunks_exact(8)
            .map(|bytes| i64::from_ne_bytes(bytes.try_into().unwrap()))
            .collect();
        json!(row)
    }

    /// `features` with their kwargs replaced by `kwargs`, one map per feature.
    fn with_kwargs(features: &MmFeatures, kwargs: Vec<MmKwargsItem>) -> MmFeatures {
        features
            .iter()
            .zip(kwargs)
            .map(|(feature, kwargs)| MmFeatureSpec {
                data: Some(kwargs),
                ..feature.clone()
            })
            .collect()
    }

    /// The image's grid echoed from the raw path sizes the same prompt, and
    /// the features are the raw ones' CPU subset: the grid the engine reads
    /// for M-RoPE positions, under the same identity and ranges.
    #[tokio::test]
    async fn qwen3_vl_metadata_matches_the_raw_cpu_subset() {
        let info = qwen3_vl_info();
        let (raw_tokens, raw_features) = prepare_raw(&info, QWEN3_IMAGE_PAD_ID).await;
        let payloads = raw_features
            .iter()
            .map(|feature| json!({"image_grid_thw": echoed(feature, "image_grid_thw")}))
            .collect();
        let (tokens, features) = prepare_metadata(&info, QWEN3_IMAGE_PAD_ID, payloads).await;

        assert_eq!(tokens, raw_tokens);
        let cpu_subset: MmFeatures = raw_features.iter().map(MmFeatureSpec::cpu_subset).collect();
        assert_eq!(features, cpu_subset);
        assert!(features.iter().all(|feature| info.lacks_encoder_input(feature)));
        // `[t, h, w]`, scaled up to the processor's minimum pixels.
        assert_eq!(
            raw_features
                .iter()
                .map(|feature| echoed(feature, "image_grid_thw"))
                .collect::<Vec<_>>(),
            [json!([1, 12, 24]), json!([1, 22, 12])]
        );
    }

    /// DeepSeek-V4.1 sizes the span from its token grid alone; the features
    /// carry that grid, as Python's metadata-only items do, not the raw
    /// path's patch grid and token types.
    #[tokio::test]
    async fn deepseek_v41_metadata_matches_the_raw_path() {
        let info = deepseek_v41_info();
        let (raw_tokens, raw_features) = prepare_raw(&info, DEEPSEEK_V41_IMAGE_ID).await;
        let payloads = raw_features
            .iter()
            .map(|feature| json!({"llm_grid": echoed(feature, "llm_grid")}))
            .collect();
        let (tokens, features) = prepare_metadata(&info, DEEPSEEK_V41_IMAGE_ID, payloads).await;

        assert_eq!(tokens, raw_tokens);
        let llm_grids = raw_features
            .iter()
            .map(|feature| {
                let mut kwargs = feature.cpu_subset().data.unwrap();
                kwargs.retain(|key, _| key == "llm_grid");
                kwargs
            })
            .collect();
        assert_eq!(features, with_kwargs(&raw_features, llm_grids));
        assert!(features.iter().all(|feature| info.lacks_encoder_input(feature)));
    }

    /// Kimi-K3 sizes the image from its `[width, height]` before resizing,
    /// which also spells the `image {width}x{height}` prompt text; the
    /// features carry those sizes and the token count, as Python's
    /// metadata-only items do.
    #[tokio::test]
    async fn kimi_k3_metadata_matches_the_raw_path() {
        let info = kimi_k3_info();
        let (raw_tokens, raw_features) = prepare_raw(&info, KIMI_MEDIA_PAD_ID).await;
        let payloads = IMAGE_SIZES
            .iter()
            .map(|&(width, height)| json!({"image_sizes": [width, height]}))
            .collect();
        let (tokens, features) = prepare_metadata(&info, KIMI_MEDIA_PAD_ID, payloads).await;

        assert_eq!(tokens, raw_tokens);
        let kwargs = IMAGE_SIZES
            .iter()
            .zip(&raw_features)
            .map(|(&(width, height), feature)| {
                let num_tokens = tokens[feature.mm_position.offset..][..feature.mm_position.length]
                    .iter()
                    .filter(|&&token| token == KIMI_MEDIA_PAD_ID)
                    .count();
                cpu_kwargs(vec![
                    ("image_sizes", vec![width.into(), height.into()]),
                    ("image_num_tokens", vec![num_tokens as i64]),
                ])
                .unwrap()
            })
            .collect();
        assert_eq!(features, with_kwargs(&raw_features, kwargs));
        assert!(features.iter().all(|feature| info.lacks_encoder_input(feature)));
    }

    #[tokio::test]
    async fn metadata_only_image_embeds_reject_unsupported_payloads() {
        let qwen3_vl = qwen3_vl_info();
        let embeds =
            |payload: serde_json::Value, uuid: Option<&str>| MediaContentPart::ImageEmbeds {
                payload,
                uuid: uuid.map(str::to_owned),
            };
        let grid = json!({"image_grid_thw": [1, 4, 4]});
        let image = MediaContentPart::ImageData {
            data: png(32, 32),
            mime_type: Some("image/png".to_owned()),
            uuid: None,
            detail: None,
        };
        let cases: Vec<(&str, Vec<MediaContentPart>, Option<usize>)> = vec![
            ("no uuid", vec![embeds(grid.clone(), None)], None),
            ("empty uuid", vec![embeds(grid.clone(), Some(""))], None),
            (
                "base64 tensor",
                vec![embeds(json!("AAAA"), Some("a"))],
                None,
            ),
            (
                "base64 field",
                vec![embeds(
                    json!({"image_embeds": "AAAA", "image_grid_thw": [1, 4, 4]}),
                    Some("a"),
                )],
                None,
            ),
            (
                "unknown key",
                vec![embeds(json!({"image_sizes": [4, 4]}), Some("a"))],
                None,
            ),
            ("missing key", vec![embeds(json!({}), Some("a"))], None),
            (
                "short grid",
                vec![embeds(json!({"image_grid_thw": [4, 4]}), Some("a"))],
                None,
            ),
            (
                "zero dim",
                vec![embeds(json!({"image_grid_thw": [1, 0, 4]}), Some("a"))],
                None,
            ),
            (
                "negative dim",
                vec![embeds(json!({"image_grid_thw": [1, -4, 4]}), Some("a"))],
                None,
            ),
            (
                "u32 overflow",
                vec![embeds(
                    json!({"image_grid_thw": [1, 4_294_967_296_u64, 4]}),
                    Some("a"),
                )],
                None,
            ),
            (
                "token overflow",
                vec![embeds(
                    json!({"image_grid_thw": [u32::MAX, u32::MAX, u32::MAX]}),
                    Some("a"),
                )],
                None,
            ),
            (
                "no tokens",
                vec![embeds(json!({"image_grid_thw": [1, 1, 1]}), Some("a"))],
                None,
            ),
            // Two text tokens leave room for 3 of the 4 image tokens.
            (
                "over budget",
                vec![embeds(grid.clone(), Some("a"))],
                Some(5),
            ),
            (
                "mixed with raw",
                vec![embeds(grid.clone(), Some("a")), image],
                None,
            ),
        ];
        let mut errors = Vec::new();
        for (name, parts, max_model_len) in cases {
            let error =
                prepare(&qwen3_vl, QWEN3_IMAGE_PAD_ID, parts, max_model_len).await.unwrap_err();
            assert!(error.is_request_validation_error(), "{name}");
            errors.push(format!("{name}: {}", error.to_report_string()));
        }
        // A model without a row.
        let step5_vl = super::super::tests::test_info(
            "step5_vl",
            json!({"model_type": "step5_vl"}),
            super::super::tests::step3_vl_tokenizer(),
        );
        let error = prepare(
            &step5_vl,
            super::super::tests::STEP3_IM_PATCH_ID,
            vec![embeds(grid.clone(), Some("a"))],
            None,
        )
        .await
        .unwrap_err();
        assert!(error.is_request_validation_error());
        errors.push(format!("no row: {}", error.to_report_string()));

        expect_test::expect![[r#"
            [
                "no uuid: metadata-only image_embeds require a nonempty `uuid`",
                "empty uuid: metadata-only image_embeds require a nonempty `uuid`",
                "base64 tensor: base64 image_embeds tensors are not supported by the Rust frontend",
                "base64 field: base64 image_embeds tensors are not supported by the Rust frontend",
                "unknown key: invalid metadata-only image_embeds for this model: unknown field `image_sizes`, expected `image_grid_thw`",
                "missing key: invalid metadata-only image_embeds for this model: missing field `image_grid_thw`",
                "short grid: invalid metadata-only image_embeds for this model: invalid length 2, expected an array of length 3",
                "zero dim: invalid metadata-only image_embeds for this model: invalid value: integer `0`, expected a nonzero u32",
                "negative dim: invalid metadata-only image_embeds for this model: invalid value: integer `-4`, expected a nonzero u32",
                "u32 overflow: invalid metadata-only image_embeds for this model: invalid value: integer `4294967296`, expected a nonzero u32",
                "token overflow: metadata-only image_embeds dimensions overflow",
                "no tokens: metadata-only image_embeds size an image of no tokens",
                "over budget: metadata-only image_embeds size more image tokens than the 3 the prompt has room for",
                "mixed with raw: Mixing raw image and embedding inputs is not allowed",
                "no row: this model does not accept metadata-only image_embeds",
            ]
        "#]].assert_debug_eq(&errors);
    }

    /// Token-level requests send images without encoder input as features,
    /// not as metadata-only content parts.
    #[tokio::test]
    async fn appended_media_rejects_image_embeds() {
        let info = qwen3_vl_info();
        let mut tokens = vec![11, QWEN3_IMAGE_PAD_ID];
        let error = info
            .append_multimodal(
                &mut Vec::new(),
                vec![MediaContentPart::ImageEmbeds {
                    payload: json!({"image_grid_thw": [1, 4, 4]}),
                    uuid: uuid(0),
                }],
                &mut tokens,
                ModelDtype::BFloat16,
                None,
            )
            .await
            .unwrap_err();
        assert_eq!(
            error.to_report_string(),
            "unsupported multimodal content: image_embeds"
        );
        assert_eq!(tokens, [11, QWEN3_IMAGE_PAD_ID]);
    }
}
