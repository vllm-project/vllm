// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Rebuild image placeholders from Python's metadata-only EPD payloads.

use std::collections::BTreeMap;

use llm_multimodal::{MediaContentPart, Modality, PreprocessedEncoderInputs};
use vllm_engine_core_client::protocol::multimodal::{
    MmBatchedField, MmField, MmFieldElem, MmKwargValue, MmKwargsItem,
};
use vllm_engine_core_client::protocol::tensor::WireTensor;

use super::{MultimodalModelInfo, PreparedItem, PreparedMedia};
use crate::error::{
    Error, Result, bail_invalid_preprocessed_multimodal, invalid_preprocessed_multimodal,
};

/// Numeric JSON arrays published by the encoder, without embedding values.
pub type ImageMetadata = BTreeMap<String, Vec<i64>>;

#[derive(Clone, Copy)]
pub(super) enum ImageMetadataKind {
    Grid,
    LlmGrid,
    ImageSize,
}

impl ImageMetadataKind {
    pub(super) fn for_model(spec: &str) -> Option<Self> {
        match spec {
            "qwen_vl" | "qwen3_vl" => Some(Self::Grid),
            "deepseek_v41" => Some(Self::LlmGrid),
            "kimi_k3" => Some(Self::ImageSize),
            _ => None,
        }
    }

    fn field(self) -> (&'static str, usize) {
        match self {
            Self::Grid => ("image_grid_thw", 3),
            Self::LlmGrid => ("llm_grid", 2),
            Self::ImageSize => ("image_sizes", 2),
        }
    }
}

impl MultimodalModelInfo {
    pub(super) fn prepare_image_references(
        &self,
        references: Vec<MediaContentPart>,
        max_model_len: Option<usize>,
    ) -> Result<PreparedMedia> {
        let support = self.image.as_ref().ok_or_else(|| Error::UnsupportedModality {
            modality: "image".to_owned(),
        })?;
        let kind = support.spec.image_metadata.ok_or_else(|| {
            invalid_preprocessed_multimodal!(
                "metadata-only images are not supported for this model"
            )
        })?;
        let (key, size) = kind.field();
        let mut remaining = max_model_len.ok_or_else(|| {
            invalid_preprocessed_multimodal!(
                "metadata-only images require a configured max_model_len"
            )
        })?;
        let mut replacements = Vec::with_capacity(references.len());
        let mut items = Vec::with_capacity(references.len());
        for reference in references {
            let MediaContentPart::ImageEmbeds { payload, uuid } = reference else {
                unreachable!("only image references are partitioned here")
            };
            let uuid = uuid.filter(|uuid| !uuid.is_empty()).ok_or_else(|| {
                invalid_preprocessed_multimodal!(
                    "metadata-only image_embeds requires a nonempty uuid"
                )
            })?;
            let metadata: ImageMetadata = serde_json::from_value(payload).map_err(|error| {
                invalid_preprocessed_multimodal!("invalid image metadata: {error}")
            })?;
            let values = metadata
                .get(key)
                .filter(|values| {
                    metadata.len() == 1 && values.len() == size && values.iter().all(|v| *v > 0)
                })
                .ok_or_else(|| invalid_preprocessed_multimodal!(
                    "image_embeds must contain only {key}, a positive integer array of length {size}"
                ))?;
            let dimensions = values
                .iter()
                .map(|&v| usize::try_from(v))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|error| {
                    invalid_preprocessed_multimodal!("invalid image dimensions: {error}")
                })?;
            let mut image_size = (0, 0);
            let count = match kind {
                ImageMetadataKind::Grid => {
                    let merge = support.config.merge_size
                        .or_else(|| {
                            self.context.config["vision_config"]["spatial_merge_size"]
                                .as_u64().and_then(|v| usize::try_from(v).ok())
                        })
                        .unwrap_or(2);
                    let [t, h, w] = dimensions.as_slice() else {
                        unreachable!("grid length was validated")
                    };
                    if *t != 1 || merge == 0 || h % merge != 0 || w % merge != 0 {
                        bail_invalid_preprocessed_multimodal!(
                            "image_grid_thw must have t=1 and spatial dimensions divisible by merge_size"
                        );
                    }
                    (h / merge).checked_mul(w / merge)
                }
                ImageMetadataKind::LlmGrid => dimensions[1].checked_add(1)
                    .and_then(|w| dimensions[0].checked_mul(w))
                    .and_then(|n| n.checked_add(2)),
                ImageMetadataKind::ImageSize => {
                    let width = u32::try_from(dimensions[0])
                        .map_err(|error| invalid_preprocessed_multimodal!("invalid image width: {error}"))?;
                    let height = u32::try_from(dimensions[1])
                        .map_err(|error| invalid_preprocessed_multimodal!("invalid image height: {error}"))?;
                    image_size = (width, height);
                    Some(support.processor.calculate_num_tokens(width, height)?)
                }
            }
                .filter(|count| *count > 0 && *count <= remaining)
                .ok_or_else(|| invalid_preprocessed_multimodal!("image metadata exceeds max_model_len"))?;
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
            let mut data = MmKwargsItem::from([(
                key.to_owned(),
                MmFieldElem {
                    data: Some(MmKwargValue::Tensor(
                        WireTensor::from_i64(vec![size], values.clone())
                            .map_err(Error::Multimodal)?,
                    )),
                    field: MmField::Batched(MmBatchedField { keep_on_cpu: true }),
                },
            )]);
            if matches!(kind, ImageMetadataKind::ImageSize) {
                data.insert(
                    "image_num_tokens".to_owned(),
                    MmFieldElem {
                        data: Some(MmKwargValue::Tensor(
                            WireTensor::from_i64(vec![1], vec![count as i64])
                                .map_err(Error::Multimodal)?,
                        )),
                        field: MmField::Batched(MmBatchedField { keep_on_cpu: true }),
                    },
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
