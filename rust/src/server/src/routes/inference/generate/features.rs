// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Render-time multimodal features passed from render to generate.

use std::collections::HashMap;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde::{Deserialize, Serialize};
use thiserror_ext::AsReport as _;
use vllm_engine_core_client::protocol::multimodal::{
    MmFeatureSpec, MmFeatures, MmKwargsItem, MmModality, PlaceholderRange,
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
    /// Item identifiers: the request's `uuid` for the item tagged `uuid:`,
    /// else its content hash tagged `content:`.
    pub mm_hashes: HashMap<MmModality, Vec<String>>,
    pub mm_placeholders: HashMap<MmModality, Vec<PlaceholderRangeInfo>>,
    /// Base64-encoded MessagePack `MultiModalKwargsItem`s. Unset when render
    /// was asked for `mm_payload: "none"`.
    #[serialize_always]
    pub kwargs_data: Option<HashMap<MmModality, Vec<Option<String>>>>,
    #[serialize_always]
    pub mm_metadata: Option<HashMap<MmModality, Vec<Option<String>>>>,
}

/// Which multimodal payload render returns, mirroring Python's `MMPayload`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MmPayload {
    /// Hashes, placeholders and processed tensors.
    #[default]
    Tensors,
    Metadata,
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
        let mut mm_metadata = (payload != MmPayload::None).then(HashMap::<_, Vec<_>>::new);
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
            if let Some(mm_metadata) = &mut mm_metadata {
                let metadata = feature
                    .data
                    .as_ref()
                    .and_then(metadata_kwargs)
                    .map(|metadata| encode_kwargs(&metadata))
                    .transpose()?;
                mm_metadata.entry(modality).or_default().push(metadata);
            }
            if let Some(kwargs_data) = &mut kwargs_data {
                let data = feature.data.map(|data| encode_kwargs(&data)).transpose()?;
                kwargs_data.entry(modality).or_default().push(data);
            }
        }
        let mm_metadata = mm_metadata
            .filter(|metadata| metadata.values().all(|items| items.iter().any(Option::is_some)));
        if payload == MmPayload::Metadata && mm_metadata.is_none() {
            bail_invalid_request!(
                param = "mm_payload",
                "mm_payload='metadata' requires metadata for every modality in the request. Render with mm_payload='tensors' instead."
            );
        }
        Ok(Self {
            mm_hashes,
            mm_placeholders,
            kwargs_data,
            mm_metadata,
        })
    }

    pub(crate) fn has_metadata_only_items(&self) -> bool {
        let Some(mm_metadata) = &self.mm_metadata else {
            return false;
        };
        mm_metadata.iter().any(|(modality, items)| {
            let kwargs = self.kwargs_data.as_ref().and_then(|data| data.get(modality));
            items.iter().enumerate().any(|(index, item)| {
                item.is_some()
                    && kwargs.and_then(|kwargs| kwargs.get(index)).is_none_or(Option::is_none)
            })
        })
    }

    /// Rebuild the prepared features, ordered by placeholder offset.
    pub(crate) fn into_prepared(self) -> Result<MmFeatures, ApiError> {
        let Self {
            mm_hashes,
            mut mm_placeholders,
            mut kwargs_data,
            mut mm_metadata,
        } = self;
        if kwargs_data.is_none() && mm_metadata.is_none() {
            bail_invalid_request!(
                param = "features",
                "features.kwargs_data or mm_metadata is required; cache-only multimodal features are not supported."
            );
        }
        let covers_hashes = |items: &Option<HashMap<MmModality, Vec<Option<String>>>>| {
            items.as_ref().is_none_or(|items| items.len() == mm_hashes.len())
        };
        if mm_placeholders.len() != mm_hashes.len()
            || !covers_hashes(&kwargs_data)
            || !covers_hashes(&mm_metadata)
        {
            bail_invalid_request!(
                param = "features",
                "features.mm_hashes, mm_placeholders, kwargs_data and mm_metadata must use the same modalities."
            );
        }

        let mut features = Vec::new();
        for (modality, hashes) in mm_hashes {
            let placeholders = mm_placeholders.remove(&modality);
            let kwargs_items = take_items(&mut kwargs_data, modality, hashes.len());
            let metadata_items = take_items(&mut mm_metadata, modality, hashes.len());
            let (Some(placeholders), Some(kwargs_items), Some(metadata_items)) =
                (placeholders, kwargs_items, metadata_items)
            else {
                bail_invalid_request!(
                    param = "features",
                    "features.mm_hashes, mm_placeholders, kwargs_data and mm_metadata must use the same modalities."
                );
            };
            if placeholders.len() != hashes.len()
                || kwargs_items.len() != hashes.len()
                || metadata_items.len() != hashes.len()
            {
                bail_invalid_request!(
                    param = "features",
                    "features.{} lists must have the same length.",
                    modality.as_str()
                );
            }
            for (((hash, placeholder), kwargs), metadata) in
                hashes.into_iter().zip(placeholders).zip(kwargs_items).zip(metadata_items)
            {
                if kwargs.is_none() && metadata.is_none() {
                    bail_invalid_request!(
                        param = "features",
                        "features.kwargs_data or mm_metadata items must be set; cache-only multimodal features are not supported."
                    );
                }
                let mut data = kwargs
                    .map(|item| decode_kwargs(&item, "kwargs_data"))
                    .transpose()?
                    .unwrap_or_default();
                if let Some(metadata) = metadata {
                    data.extend(decode_kwargs(&metadata, "mm_metadata")?);
                }
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

fn take_items(
    items: &mut Option<HashMap<MmModality, Vec<Option<String>>>>,
    modality: MmModality,
    n: usize,
) -> Option<Vec<Option<String>>> {
    match items {
        Some(items) => items.remove(&modality),
        None => Some(vec![None; n]),
    }
}

fn metadata_kwargs(item: &MmKwargsItem) -> Option<MmKwargsItem> {
    let metadata: MmKwargsItem = item
        .iter()
        .filter(|(_, elem)| elem.field.keep_on_cpu())
        .map(|(key, elem)| (key.clone(), elem.clone()))
        .collect();
    (!metadata.is_empty()).then_some(metadata)
}

fn encode_kwargs(item: &MmKwargsItem) -> Result<String, ApiError> {
    mm_wire::encode_kwargs(item)
        .map(|bytes| BASE64.encode(bytes))
        .map_err(|error| server_error!("{error}"))
}

fn decode_kwargs(item: &str, field: &str) -> Result<MmKwargsItem, ApiError> {
    BASE64
        .decode(item)
        .map_err(|error| error.to_report_string())
        .and_then(|bytes| mm_wire::decode_kwargs(&bytes))
        .map_err(|error| {
            ApiError::invalid_request(
                format!("invalid features.{field}: {error}"),
                Some("features"),
            )
        })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use vllm_engine_core_client::protocol::multimodal::{
        MmBatchedField, MmField, MmFieldElem, MmKwargValue,
    };

    use super::*;

    fn field(keep_on_cpu: bool, values: Vec<f32>) -> MmFieldElem {
        MmFieldElem {
            data: Some(MmKwargValue::Tensor(
                WireTensor::from_f32(vec![values.len()], values).unwrap(),
            )),
            field: MmField::Batched(MmBatchedField { keep_on_cpu }),
        }
    }

    fn metadata_feature(modality: MmModality, offset: usize, with_grid: bool) -> MmFeatureSpec {
        let mut feature = feature(modality, offset, None);
        let data = feature.data.as_mut().unwrap();
        if with_grid {
            data.insert(
                "image_grid_thw".to_owned(),
                field(true, vec![1.0, 2.0, 2.0]),
            );
        }
        feature
    }

    fn wire_keys(features: &MultiModalFeatures) -> (Vec<String>, Vec<String>) {
        let json = serde_json::to_value(features).unwrap();
        let object = json.as_object().unwrap();
        let mut keys = object.keys().cloned().collect::<Vec<_>>();
        let mut nulls = object
            .iter()
            .filter(|(_, value)| value.is_null())
            .map(|(key, _)| key.clone())
            .collect::<Vec<_>>();
        keys.sort();
        nulls.sort();
        (keys, nulls)
    }

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

    #[test]
    fn features_wire_keys_match_python_for_each_payload() {
        let keys = [MmPayload::Tensors, MmPayload::Metadata, MmPayload::None].map(|payload| {
            let prepared = vec![metadata_feature(MmModality::Image, 1, true)];
            (
                payload,
                wire_keys(&MultiModalFeatures::from_prepared(prepared, payload).unwrap()),
            )
        });
        expect_test::expect![[r#"
            [
                (
                    Tensors,
                    (
                        [
                            "kwargs_data",
                            "mm_hashes",
                            "mm_metadata",
                            "mm_placeholders",
                        ],
                        [],
                    ),
                ),
                (
                    Metadata,
                    (
                        [
                            "kwargs_data",
                            "mm_hashes",
                            "mm_metadata",
                            "mm_placeholders",
                        ],
                        [
                            "kwargs_data",
                        ],
                    ),
                ),
                (
                    None,
                    (
                        [
                            "kwargs_data",
                            "mm_hashes",
                            "mm_metadata",
                            "mm_placeholders",
                        ],
                        [
                            "kwargs_data",
                            "mm_metadata",
                        ],
                    ),
                ),
            ]
        "#]]
        .assert_debug_eq(&keys);
    }

    #[test]
    fn metadata_only_features_round_trip_through_json() {
        let prepared = vec![metadata_feature(MmModality::Image, 1, true)];
        let features =
            MultiModalFeatures::from_prepared(prepared.clone(), MmPayload::Metadata).unwrap();
        assert_eq!(features.kwargs_data, None);
        assert!(features.has_metadata_only_items());

        let decoded: MultiModalFeatures =
            serde_json::from_value(serde_json::to_value(&features).unwrap()).unwrap();
        let rebuilt = decoded.into_prepared().unwrap();
        let mut expected = prepared;
        expected[0].data.as_mut().unwrap().remove("pixel_values");
        assert_eq!(rebuilt, expected);
    }

    #[test]
    fn tensors_features_carry_metadata_without_needing_ec() {
        let prepared = vec![metadata_feature(MmModality::Image, 1, true)];
        let features =
            MultiModalFeatures::from_prepared(prepared.clone(), MmPayload::Tensors).unwrap();
        assert!(features.mm_metadata.is_some());
        assert!(!features.has_metadata_only_items());
        assert_eq!(features.into_prepared().unwrap(), prepared);
    }

    #[test]
    fn modality_without_metadata_nulls_mm_metadata() {
        let prepared = || {
            vec![
                metadata_feature(MmModality::Image, 1, true),
                metadata_feature(MmModality::Video, 4, false),
            ]
        };
        let tensors = MultiModalFeatures::from_prepared(prepared(), MmPayload::Tensors).unwrap();
        assert_eq!(tensors.mm_metadata, None);
        assert!(tensors.kwargs_data.is_some());

        let error = MultiModalFeatures::from_prepared(prepared(), MmPayload::Metadata).unwrap_err();
        let response = error.to_error_response();
        assert_eq!(response.error.param.as_deref(), Some("mm_payload"));
        assert!(
            response.error.message.contains("requires metadata for every modality"),
            "{error:?}"
        );
    }
}
