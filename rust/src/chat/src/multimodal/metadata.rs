// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Rebuild image placeholders from Python's metadata-only EPD payloads.

use llm_multimodal::{Modality, PreprocessedEncoderInputs};
use serde::{Deserialize, Serialize};
use vllm_engine_core_client::protocol::multimodal::{
    MmBatchedField, MmField, MmFieldElem, MmKwargValue, MmKwargsItem,
};
use vllm_engine_core_client::protocol::tensor::WireTensor;

use super::{MultimodalModelInfo, PreparedItem, PreparedMedia};
use crate::error::{
    Error, Result, bail_invalid_preprocessed_multimodal, invalid_preprocessed_multimodal,
};

/// Numeric JSON arrays published by the encoder, without embedding values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
pub enum ImageMetadata {
    Grid { image_grid_thw: [i64; 3] },
    LlmGrid { llm_grid: [i64; 2] },
    ImageSize { image_sizes: [u32; 2] },
}

impl ImageMetadata {
    pub(super) fn field_for_model(spec: &str) -> Option<&'static str> {
        match spec {
            "qwen_vl" | "qwen3_vl" => Some("image_grid_thw"),
            "deepseek_v41" => Some("llm_grid"),
            "kimi_k3" => Some("image_sizes"),
            _ => None,
        }
    }

    fn field(&self) -> (&'static str, Vec<i64>) {
        match self {
            Self::Grid { image_grid_thw } => ("image_grid_thw", image_grid_thw.to_vec()),
            Self::LlmGrid { llm_grid } => ("llm_grid", llm_grid.to_vec()),
            Self::ImageSize { image_sizes } => ("image_sizes", image_sizes.map(i64::from).to_vec()),
        }
    }
}

pub(super) struct ImageReference {
    pub metadata: ImageMetadata,
    pub uuid: String,
}

fn batched_metadata(values: Vec<i64>) -> Result<MmFieldElem> {
    Ok(MmFieldElem {
        data: Some(MmKwargValue::Tensor(
            WireTensor::from_i64(vec![values.len()], values).map_err(Error::Multimodal)?,
        )),
        field: MmField::Batched(MmBatchedField { keep_on_cpu: true }),
    })
}

impl MultimodalModelInfo {
    pub(super) fn prepare_image_references(
        &self,
        references: Vec<ImageReference>,
        max_model_len: Option<usize>,
    ) -> Result<PreparedMedia> {
        let support = self.image.as_ref().ok_or_else(|| Error::UnsupportedModality {
            modality: "image".to_owned(),
        })?;
        let mut remaining = max_model_len.ok_or_else(|| {
            invalid_preprocessed_multimodal!(
                "metadata-only images require a configured max_model_len"
            )
        })?;
        let mut replacements = Vec::with_capacity(references.len());
        let mut items = Vec::with_capacity(references.len());
        for ImageReference { metadata, uuid } in references {
            let (key, values) = metadata.field();
            if uuid.is_empty() {
                bail_invalid_preprocessed_multimodal!(
                    "metadata-only image_embeds requires a nonempty uuid"
                );
            }
            if support.spec.image_metadata != Some(key) || values.iter().any(|v| *v <= 0) {
                bail_invalid_preprocessed_multimodal!(
                    "unsupported image metadata or nonpositive dimensions"
                );
            }
            let mut image_size = (0, 0);
            let count = match metadata {
                ImageMetadata::Grid {
                    image_grid_thw: [t, h, w],
                } => {
                    let merge = support
                        .config
                        .merge_size
                        .or_else(|| {
                            self.context.config["vision_config"]["spatial_merge_size"]
                                .as_u64()
                                .and_then(|v| usize::try_from(v).ok())
                        })
                        .unwrap_or(2) as u64;
                    let (h, w) = (h as u64, w as u64);
                    if t != 1 || merge == 0 || h % merge != 0 || w % merge != 0 {
                        bail_invalid_preprocessed_multimodal!(
                            "image_grid_thw must have t=1 and spatial dimensions divisible by merge_size"
                        );
                    }
                    (h / merge).checked_mul(w / merge)
                }
                ImageMetadata::LlmGrid { llm_grid: [h, w] } => (w as u64)
                    .checked_add(1)
                    .and_then(|w| (h as u64).checked_mul(w))
                    .and_then(|n| n.checked_add(2)),
                ImageMetadata::ImageSize {
                    image_sizes: [width, height],
                } => {
                    image_size = (width, height);
                    Some(support.processor.calculate_num_tokens(image_size.0, image_size.1)? as u64)
                }
            };
            let count = count
                .and_then(|n| usize::try_from(n).ok())
                .filter(|count| *count > 0 && *count <= remaining)
                .ok_or_else(|| {
                    invalid_preprocessed_multimodal!("image metadata exceeds max_model_len")
                })?;
            remaining -= count;

            // The model spec owns the placeholder layout; no pixel buffer or
            // image preprocessing is needed to construct it.
            let preprocessed = PreprocessedEncoderInputs::new(
                ndarray::Array1::<f32>::zeros(0),
                vec![count],
                vec![image_size],
            );
            replacements
                .extend(support.spec.prompt_replacements_for(&self.context, &preprocessed)?);
            let mut data: MmKwargsItem =
                [(key.to_owned(), batched_metadata(values)?)].into_iter().collect();
            if matches!(metadata, ImageMetadata::ImageSize { .. }) {
                data.insert(
                    "image_num_tokens".to_owned(),
                    batched_metadata(vec![count as i64])?,
                );
            }
            items.push(PreparedItem {
                data,
                hash: uuid.clone(),
                uuid: Some(uuid),
            });
        }
        Ok(PreparedMedia {
            modality: Modality::Image,
            placeholder: support.placeholder.clone(),
            replacements,
            items,
        })
    }
}
