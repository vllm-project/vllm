// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;

use vllm_engine_core_client::protocol::dtype::TensorDtype;
use vllm_engine_core_client::protocol::multimodal::{
    MmFeatureSpec, MmField, MmKwargValue, MmKwargsItem,
};

use crate::error::{
    Error, Result, bail_invalid_preprocessed_multimodal, invalid_preprocessed_multimodal,
};

const MAX_DEPTH: usize = 32;

/// Validate inline storage, placeholder ranges, and batching compatibility,
/// then stably sort features by their prompt offsets.
/// Encoder cache identifiers are supplied by the producer, which must bind
/// them to the feature content.
pub(super) fn validate_features(features: &mut [MmFeatureSpec], prompt_len: usize) -> Result<()> {
    let mut fields = HashMap::new();
    for feature in features.iter() {
        if feature.identifier.is_empty() {
            bail_invalid_preprocessed_multimodal!("feature identifier must be nonempty");
        }
        let data = feature.data.as_ref().ok_or_else(|| {
            invalid_preprocessed_multimodal!(
                "inline data is required; cache-only features are unsupported"
            )
        })?;
        validate_kwargs(data)?;
        for (key, elem) in data {
            if let Some(previous) = fields.insert((feature.modality, key), &elem.field)
                && !same_batching(previous, &elem.field)
            {
                bail_invalid_preprocessed_multimodal!(
                    "incompatible batching for {} field {key:?}",
                    feature.modality.as_str()
                );
            }
        }
        let position = &feature.mm_position;
        if position.length == 0 || position.offset.checked_add(position.length).is_none() {
            bail_invalid_preprocessed_multimodal!(
                "placeholder range must have a positive length and fit in usize"
            );
        }
        if position.offset + position.length > prompt_len {
            bail_invalid_preprocessed_multimodal!(
                "{} placeholder range exceeds prompt token IDs",
                feature.modality.as_str()
            );
        }
        if let Some(mask) = &position.is_embed {
            if mask.dtype != TensorDtype::Bool || mask.shape != [position.length] {
                bail_invalid_preprocessed_multimodal!(
                    "is_embed must be a boolean tensor of shape [placeholder length]"
                );
            }
            mask.validate_inline()
                .map_err(|message| Error::InvalidPreprocessedMultimodal { message })?;
        }
    }
    // Mirror the Python frontend (`argsort_mm_positions`): features are
    // ordered by their placeholder position in the prompt.
    features.sort_by_key(|feature| feature.mm_position.offset);
    Ok(())
}

fn validate_kwargs(kwargs: &MmKwargsItem) -> Result<()> {
    if kwargs.is_empty() {
        bail_invalid_preprocessed_multimodal!("kwargs must contain at least one field");
    }
    for (key, elem) in kwargs {
        if key.is_empty() {
            bail_invalid_preprocessed_multimodal!("kwargs field name must be nonempty");
        }
        let data = elem.data.as_ref().ok_or_else(|| {
            invalid_preprocessed_multimodal!("field {key:?} requires inline data")
        })?;
        validate_value(data, 0)?;
        if let MmField::Flat(field) = &elem.field {
            match data {
                MmKwargValue::Tensor(tensor) => {
                    let rank = tensor.shape.len() as i32;
                    if rank == 0 || field.dim < -rank || field.dim >= rank {
                        bail_invalid_preprocessed_multimodal!(
                            "flat field {key:?} dim is outside the tensor rank"
                        );
                    }
                }
                MmKwargValue::List(_) if field.dim == 0 => {}
                _ => bail_invalid_preprocessed_multimodal!(
                    "flat field {key:?} requires a tensor or a list with dim=0"
                ),
            }
        }
    }
    Ok(())
}

fn validate_value(value: &MmKwargValue, depth: usize) -> Result<()> {
    if depth > MAX_DEPTH {
        bail_invalid_preprocessed_multimodal!("kwargs nesting exceeds {MAX_DEPTH} levels");
    }
    match value {
        MmKwargValue::Tensor(tensor) => tensor
            .validate_inline()
            .map_err(|message| Error::InvalidPreprocessedMultimodal { message }),
        MmKwargValue::List(values) => {
            for value in values {
                validate_value(value, depth + 1)?;
            }
            Ok(())
        }
        MmKwargValue::Int(_) | MmKwargValue::Float(_) => Ok(()),
    }
}

fn same_batching(left: &MmField, right: &MmField) -> bool {
    // Slices and batch_size describe the producer's original batch. Python
    // reduce_data combines the already-split item data without using them.
    match (left, right) {
        (MmField::Batched(a), MmField::Batched(b)) => a == b,
        (MmField::Flat(a), MmField::Flat(b)) => a.dim == b.dim && a.keep_on_cpu == b.keep_on_cpu,
        (MmField::Shared(a), MmField::Shared(b)) => a.keep_on_cpu == b.keep_on_cpu,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use vllm_engine_core_client::protocol::multimodal::{
        MmBatchedField, MmFieldElem, MmFlatField, MmModality, MmSharedField, MmSlice,
        PlaceholderRange, SliceSpec,
    };
    use vllm_engine_core_client::protocol::tensor::{WireArrayData, WireTensor};

    use super::*;
    use crate::multimodal::tests::qwen3_vl_info;

    fn feature(offset: usize, field: MmField) -> MmFeatureSpec {
        MmFeatureSpec {
            data: Some(BTreeMap::from([(
                "pixels".to_owned(),
                MmFieldElem {
                    data: Some(MmKwargValue::Tensor(
                        WireTensor::from_f32(vec![2], vec![1., 2.]).unwrap(),
                    )),
                    field,
                },
            )])),
            modality: MmModality::Image,
            identifier: format!("image-{offset}"),
            mm_position: PlaceholderRange {
                offset,
                length: 2,
                is_embed: None,
            },
            mm_hash: Some("processor-hash".to_owned()),
        }
    }

    fn batched() -> MmField {
        MmField::Batched(MmBatchedField { keep_on_cpu: false })
    }

    #[test]
    fn inline_features_preserve_overlaps_and_stably_sort_prompt_positions() {
        let info = qwen3_vl_info();
        let mut a = feature(2, batched());
        a.mm_position.is_embed = Some(WireTensor::from_bool(vec![2], vec![true, false]).unwrap());
        let b = feature(1, batched());
        let mut c = feature(2, batched());
        c.modality = MmModality::Video;
        let expected = vec![b.clone(), a.clone(), c.clone()];
        let features = info.prepare_preprocessed(vec![a, b, c], 4).unwrap();
        assert_eq!(features, expected);
    }

    #[test]
    fn independently_preprocessed_items_keep_their_original_batch_metadata() {
        let info = qwen3_vl_info();
        let flat = |length| {
            MmField::Flat(MmFlatField {
                slices: vec![MmSlice::Slice(SliceSpec {
                    start: Some(0),
                    stop: Some(length),
                    step: None,
                })],
                dim: 0,
                keep_on_cpu: false,
            })
        };
        let shared = |batch_size| {
            MmField::Shared(MmSharedField {
                batch_size,
                keep_on_cpu: true,
            })
        };
        for (a, b) in [(flat(2), flat(7)), (shared(1), shared(3))] {
            let features = vec![feature(0, a), feature(2, b)];
            let result = info.prepare_preprocessed(features.clone(), 4).unwrap();
            assert_eq!(result, features);
        }
    }

    #[test]
    fn inline_features_reject_cache_references_and_invalid_tensor_storage() {
        let info = qwen3_vl_info();
        let original = feature(0, batched());
        let mut cached = original.clone();
        cached.data = None;
        assert!(info.prepare_preprocessed(vec![cached], 4).is_err());
        let mut field_cached = original.clone();
        field_cached.data.as_mut().unwrap().get_mut("pixels").unwrap().data = None;
        assert!(info.prepare_preprocessed(vec![field_cached], 4).is_err());
        for tensor in [
            WireTensor {
                dtype: TensorDtype::F32,
                shape: vec![2],
                data: WireArrayData::AuxIndex(1),
            },
            WireTensor::from_raw(TensorDtype::F32, vec![2], vec![0; 4]),
            WireTensor::from_raw(TensorDtype::F32, vec![usize::MAX, 2], vec![]),
        ] {
            let mut item = original.clone();
            item.data.as_mut().unwrap().get_mut("pixels").unwrap().data =
                Some(MmKwargValue::Tensor(tensor));
            assert!(info.prepare_preprocessed(vec![item], 4).is_err());
        }
    }

    #[test]
    fn inline_features_validate_placeholder_and_mask_bounds() {
        let info = qwen3_vl_info();
        for (offset, length) in [(usize::MAX, 2), (0, 0)] {
            let mut item = feature(offset, batched());
            item.mm_position.length = length;
            assert!(info.prepare_preprocessed(vec![item], 4).is_err());
        }
        assert!(info.prepare_preprocessed(vec![feature(2, batched())], 3).is_err());
        for mask in [
            WireTensor::from_bool(vec![1], vec![true]).unwrap(),
            WireTensor::from_u32(vec![2], vec![1, 0]).unwrap(),
        ] {
            let mut item = feature(0, batched());
            item.mm_position.is_embed = Some(mask);
            assert!(info.prepare_preprocessed(vec![item], 4).is_err());
        }
    }

    #[test]
    fn inline_features_reject_incompatible_batching_and_flat_axes() {
        let info = qwen3_vl_info();
        let flat = MmField::Flat(MmFlatField {
            slices: vec![],
            dim: 2,
            keep_on_cpu: false,
        });
        assert!(info.prepare_preprocessed(vec![feature(0, flat)], 4).is_err());
        let shared = MmField::Shared(MmSharedField {
            batch_size: 1,
            keep_on_cpu: false,
        });
        assert!(
            info.prepare_preprocessed(vec![feature(0, batched()), feature(2, shared)], 4)
                .is_err()
        );
    }
}
