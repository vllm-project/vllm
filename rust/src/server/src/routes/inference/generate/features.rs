// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Render-time multimodal features passed from render to generate.

use std::collections::HashMap;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde::{Deserialize, Serialize};
use thiserror_ext::AsReport as _;
use vllm_engine_core_client::protocol::multimodal::{
    MmFeatureSpec, MmFeatures, MmModality, PlaceholderRange,
};
use vllm_engine_core_client::protocol::tensor::WireTensor;

use super::types::PlaceholderRangeInfo;
use crate::error::{ApiError, bail_invalid_request, server_error};
use crate::mm_wire;

/// Mirrors the Python vLLM `MultiModalFeatures` class.
///
/// Each modality maps to parallel lists, one entry per item in placeholder
/// order.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiModalFeatures {
    /// Item identifiers: the request's `uuid` for the item, else its content
    /// hash.
    pub mm_hashes: HashMap<MmModality, Vec<String>>,
    pub mm_placeholders: HashMap<MmModality, Vec<PlaceholderRangeInfo>>,
    /// Base64-encoded MessagePack `MultiModalKwargsItem`s. Unset when render
    /// was asked for `mm_payload: "none"`.
    pub kwargs_data: Option<HashMap<MmModality, Vec<Option<String>>>>,
    /// Metadata-only items Python's render adds for EC-transfer prefill. Never
    /// emitted; ignored on input because generate requires `kwargs_data` for
    /// every item, which already contains these fields.
    pub mm_metadata: Option<HashMap<MmModality, Vec<Option<String>>>>,
}

/// Which multimodal payload render returns, mirroring Python's `MMPayload`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MmPayload {
    /// Hashes, placeholders and processed tensors.
    #[default]
    Tensors,
    /// Hashes and placeholders only, for callers that route on the prompt
    /// layout. Such features cannot be sent to generate.
    None,
}

impl MultiModalFeatures {
    /// Group prepared features by modality, keeping their order.
    pub(crate) fn from_prepared(
        features: MmFeatures,
        payload: MmPayload,
    ) -> Result<Self, ApiError> {
        let mut mm_hashes = HashMap::<_, Vec<_>>::new();
        let mut mm_placeholders = HashMap::<_, Vec<_>>::new();
        let mut kwargs_data = (payload == MmPayload::Tensors).then(HashMap::<_, Vec<_>>::new);
        for feature in features {
            let modality = feature.modality;
            let position = feature.mm_position;
            mm_hashes.entry(modality).or_default().push(feature.identifier);
            mm_placeholders.entry(modality).or_default().push(PlaceholderRangeInfo {
                offset: position.offset,
                length: position.length,
                is_embed: position
                    .is_embed
                    .as_ref()
                    .map(mm_wire::mask_to_bools)
                    .transpose()
                    .map_err(|error| server_error!("{error}"))?,
            });
            if let Some(kwargs_data) = &mut kwargs_data {
                let data = feature
                    .data
                    .map(|data| mm_wire::encode_kwargs(&data).map(|bytes| BASE64.encode(bytes)))
                    .transpose()
                    .map_err(|error| server_error!("{error}"))?;
                kwargs_data.entry(modality).or_default().push(data);
            }
        }
        Ok(Self {
            mm_hashes,
            mm_placeholders,
            kwargs_data,
            mm_metadata: None,
        })
    }

    /// Rebuild the prepared features, ordered by placeholder offset.
    ///
    /// Every item must carry inline kwargs: the Rust frontend has no
    /// multimodal processor cache to resolve a missing item from.
    pub(crate) fn into_prepared(self) -> Result<MmFeatures, ApiError> {
        let Some(mut kwargs_data) = self.kwargs_data else {
            bail_invalid_request!(
                param = "features",
                "features.kwargs_data is required; cache-only multimodal features are not supported."
            );
        };
        let mut mm_placeholders = self.mm_placeholders;
        if mm_placeholders.len() != self.mm_hashes.len()
            || kwargs_data.len() != self.mm_hashes.len()
        {
            bail_invalid_request!(
                param = "features",
                "features.mm_hashes, mm_placeholders and kwargs_data must use the same modalities."
            );
        }

        let mut features = Vec::new();
        for (modality, hashes) in self.mm_hashes {
            let (Some(placeholders), Some(items)) = (
                mm_placeholders.remove(&modality),
                kwargs_data.remove(&modality),
            ) else {
                bail_invalid_request!(
                    param = "features",
                    "features.mm_hashes, mm_placeholders and kwargs_data must use the same modalities."
                );
            };
            if placeholders.len() != hashes.len() || items.len() != hashes.len() {
                bail_invalid_request!(
                    param = "features",
                    "features.{} lists must have the same length.",
                    modality.as_str()
                );
            }
            for ((hash, placeholder), item) in hashes.into_iter().zip(placeholders).zip(items) {
                let Some(item) = item else {
                    bail_invalid_request!(
                        param = "features",
                        "features.kwargs_data items must be set; cache-only multimodal features are not supported."
                    );
                };
                let data = BASE64
                    .decode(item)
                    .map_err(|error| error.to_report_string())
                    .and_then(|bytes| mm_wire::decode_kwargs(&bytes))
                    .map_err(|error| {
                        ApiError::invalid_request(
                            format!("invalid features.kwargs_data: {error}"),
                            Some("features"),
                        )
                    })?;
                let is_embed = placeholder
                    .is_embed
                    .map(|mask| WireTensor::from_bool(vec![placeholder.length], mask))
                    .transpose()
                    .map_err(|error| {
                        ApiError::invalid_request(
                            format!("invalid features.mm_placeholders is_embed: {error}"),
                            Some("features"),
                        )
                    })?;
                features.push(MmFeatureSpec {
                    data: Some(data),
                    modality,
                    // Like Python, the hash also identifies the item.
                    identifier: hash.clone(),
                    mm_hash: Some(hash),
                    mm_position: PlaceholderRange {
                        offset: placeholder.offset,
                        length: placeholder.length,
                        is_embed,
                    },
                });
            }
        }
        features.sort_by_key(|feature| feature.mm_position.offset);
        Ok(features)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use vllm_engine_core_client::protocol::multimodal::{
        MmBatchedField, MmField, MmFieldElem, MmKwargValue,
    };

    use super::*;

    fn feature(modality: MmModality, offset: usize, is_embed: Option<Vec<bool>>) -> MmFeatureSpec {
        let length = is_embed.as_ref().map_or(2, Vec::len);
        MmFeatureSpec {
            data: Some(BTreeMap::from([(
                "pixel_values".to_owned(),
                MmFieldElem {
                    data: Some(MmKwargValue::Tensor(
                        WireTensor::from_f32(vec![2], vec![1.0, 2.0]).unwrap(),
                    )),
                    field: MmField::Batched(MmBatchedField { keep_on_cpu: false }),
                },
            )])),
            modality,
            identifier: format!("{}-{offset}", modality.as_str()),
            mm_hash: Some(format!("{}-{offset}", modality.as_str())),
            mm_position: PlaceholderRange {
                offset,
                length,
                is_embed: is_embed.map(|mask| WireTensor::from_bool(vec![length], mask).unwrap()),
            },
        }
    }

    /// Render's JSON features rebuild the same prepared features in generate.
    #[test]
    fn features_round_trip_through_json() {
        let prepared = vec![
            feature(MmModality::Image, 1, None),
            feature(MmModality::Audio, 4, None),
            feature(MmModality::Image, 8, Some(vec![true, false, true])),
        ];

        let features =
            MultiModalFeatures::from_prepared(prepared.clone(), MmPayload::Tensors).unwrap();
        let json = serde_json::to_value(&features).unwrap();
        assert_eq!(
            json["mm_hashes"]["image"],
            serde_json::json!(["image-1", "image-8"])
        );
        assert!(json["mm_placeholders"]["image"][0].get("is_embed").is_none());
        let decoded: MultiModalFeatures = serde_json::from_value(json).unwrap();

        assert_eq!(decoded.into_prepared().unwrap(), prepared);
    }

    #[test]
    fn hash_only_features_are_rejected_by_generate() {
        let features = MultiModalFeatures::from_prepared(
            vec![feature(MmModality::Image, 1, None)],
            MmPayload::None,
        )
        .unwrap();
        assert_eq!(features.kwargs_data, None);

        let error = features.into_prepared().unwrap_err();
        assert!(
            error.to_error_response().error.message.contains("cache-only"),
            "{error:?}"
        );
    }

    #[test]
    fn features_with_mismatched_lists_are_rejected() {
        let mut features = MultiModalFeatures::from_prepared(
            vec![
                feature(MmModality::Image, 1, None),
                feature(MmModality::Image, 4, None),
            ],
            MmPayload::Tensors,
        )
        .unwrap();
        features.mm_hashes.get_mut(&MmModality::Image).unwrap().pop();

        assert!(features.into_prepared().is_err());
    }
}
