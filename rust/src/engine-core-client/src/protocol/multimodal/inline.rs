// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;
use std::io::Cursor;

use serde::Deserialize as _;
use thiserror_ext::Macro;

use super::{MmFeatureSpec, MmFeatures, MmField, MmKwargValue, MmKwargsItem};
use crate::protocol::dtype::TensorDtype;

/// Maximum serialized kwargs size accepted at an inline feature boundary.
pub const MAX_INLINE_MM_BYTES: usize = 16 * 1024 * 1024;
const MAX_DEPTH: usize = 32;

#[derive(Debug, thiserror::Error, Macro)]
#[thiserror_ext(macro(path = "crate::protocol::multimodal::inline"))]
pub enum InlineMmError {
    #[error("invalid inline multimodal features: {message}")]
    Invalid { message: String },
    #[error("inline multimodal payload exceeds {limit} bytes")]
    PayloadTooLarge { limit: usize },
    #[error("invalid multimodal kwargs MessagePack")]
    Decode(#[from] rmp_serde::decode::Error),
}

type Result<T> = std::result::Result<T, InlineMmError>;

/// Decode one inline item using the engine protocol's typed serde schema.
pub fn decode_inline_mm_kwargs(bytes: &[u8]) -> Result<MmKwargsItem> {
    if bytes.len() > MAX_INLINE_MM_BYTES {
        return Err(InlineMmError::PayloadTooLarge {
            limit: MAX_INLINE_MM_BYTES,
        });
    }
    let mut decoder = rmp_serde::Deserializer::new(Cursor::new(bytes));
    decoder.set_max_depth(MAX_DEPTH);
    let kwargs = MmKwargsItem::deserialize(&mut decoder)?;
    if decoder.position() != bytes.len() as u64 {
        bail_invalid!("kwargs contains trailing MessagePack data");
    }
    validate_kwargs(&kwargs)?;
    Ok(kwargs)
}

/// Fully inline features validated for submission by an external frontend.
///
/// Construction checks storage and batching metadata; lowering checks the
/// placeholder positions against the final prompt. Encoder cache identifiers
/// are supplied by the producer, which must bind them to the feature content.
#[derive(Debug, Clone)]
pub struct InlineMmFeatures(MmFeatures);

impl InlineMmFeatures {
    /// Validate inline storage, placeholder metadata, and batching compatibility,
    /// then stably sort features by their prompt offsets.
    pub fn new(mut features: MmFeatures) -> Result<Self> {
        let mut fields = HashMap::new();
        for feature in &features {
            if feature.identifier.is_empty() {
                bail_invalid!("feature identifier must be nonempty");
            }
            let data = feature.data.as_ref().ok_or_else(|| {
                invalid!("inline data is required; cache-only features are unsupported")
            })?;
            validate_kwargs(data)?;
            for (key, elem) in data {
                if let Some(previous) = fields.insert((feature.modality, key), &elem.field)
                    && !same_batching(previous, &elem.field)
                {
                    bail_invalid!(
                        "incompatible batching for {} field {key:?}",
                        feature.modality.as_str()
                    );
                }
            }
            let position = &feature.mm_position;
            if position.length == 0 || position.offset.checked_add(position.length).is_none() {
                bail_invalid!("placeholder range must have a positive length and fit in usize");
            }
            if let Some(mask) = &position.is_embed {
                if mask.dtype != TensorDtype::Bool || mask.shape != [position.length] {
                    bail_invalid!(
                        "is_embed must be a boolean tensor of shape [placeholder length]"
                    );
                }
                mask.validate_inline().map_err(|message| InlineMmError::Invalid { message })?;
            }
        }
        // Mirror the Python frontend (`argsort_mm_positions`): features are
        // ordered by their placeholder position in the prompt.
        features.sort_by_key(|feature| feature.mm_position.offset);
        Ok(Self(features))
    }

    pub fn as_slice(&self) -> &[MmFeatureSpec] {
        &self.0
    }

    /// Return the features after checking placeholder ranges against the final
    /// prompt length, including expanded multimodal placeholders.
    pub fn into_features(self, prompt_len: usize) -> Result<MmFeatures> {
        for feature in &self.0 {
            let position = &feature.mm_position;
            if position.offset + position.length > prompt_len {
                bail_invalid!(
                    "{} placeholder range exceeds prompt token IDs",
                    feature.modality.as_str()
                );
            }
        }
        Ok(self.0)
    }
}

fn validate_kwargs(kwargs: &MmKwargsItem) -> Result<()> {
    if kwargs.is_empty() {
        bail_invalid!("kwargs must contain at least one field");
    }
    for (key, elem) in kwargs {
        if key.is_empty() {
            bail_invalid!("kwargs field name must be nonempty");
        }
        let data = elem
            .data
            .as_ref()
            .ok_or_else(|| invalid!("field {key:?} requires inline data"))?;
        validate_value(data, 0)?;
        if let MmField::Flat(field) = &elem.field {
            match data {
                MmKwargValue::Tensor(tensor) => {
                    let rank = tensor.shape.len() as i32;
                    if rank == 0 || field.dim < -rank || field.dim >= rank {
                        bail_invalid!("flat field {key:?} dim is outside the tensor rank");
                    }
                }
                MmKwargValue::List(_) if field.dim == 0 => {}
                _ => bail_invalid!("flat field {key:?} requires a tensor or a list with dim=0"),
            }
        }
    }
    Ok(())
}

fn validate_value(value: &MmKwargValue, depth: usize) -> Result<()> {
    if depth > MAX_DEPTH {
        bail_invalid!("kwargs nesting exceeds {MAX_DEPTH} levels");
    }
    match value {
        MmKwargValue::Tensor(tensor) => {
            tensor.validate_inline().map_err(|message| InlineMmError::Invalid { message })
        }
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

    use super::*;
    use crate::protocol::multimodal::{
        MmBatchedField, MmFieldElem, MmFlatField, MmModality, MmSharedField, MmSlice,
        PlaceholderRange, SliceSpec,
    };
    use crate::protocol::tensor::{WireArrayData, WireTensor};

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
        let mut a = feature(2, batched());
        a.mm_position.is_embed = Some(WireTensor::from_bool(vec![2], vec![true, false]).unwrap());
        let b = feature(1, batched());
        let mut c = feature(2, batched());
        c.modality = MmModality::Video;
        let expected = vec![b.clone(), a.clone(), c.clone()];
        let features = InlineMmFeatures::new(vec![a, b, c]).unwrap().into_features(4).unwrap();
        assert_eq!(features, expected);
    }

    #[test]
    fn independently_preprocessed_items_keep_their_original_batch_metadata() {
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
            let result = InlineMmFeatures::new(features.clone()).unwrap().into_features(4).unwrap();
            assert_eq!(result, features);
        }
    }

    #[test]
    fn inline_features_reject_cache_references_and_invalid_tensor_storage() {
        let original = feature(0, batched());
        let mut cached = original.clone();
        cached.data = None;
        assert!(InlineMmFeatures::new(vec![cached]).is_err());
        let mut field_cached = original.clone();
        field_cached.data.as_mut().unwrap().get_mut("pixels").unwrap().data = None;
        assert!(InlineMmFeatures::new(vec![field_cached]).is_err());
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
            assert!(InlineMmFeatures::new(vec![item]).is_err());
        }
    }

    #[test]
    fn inline_features_validate_placeholder_and_mask_bounds() {
        for (offset, length) in [(usize::MAX, 2), (0, 0)] {
            let mut item = feature(offset, batched());
            item.mm_position.length = length;
            assert!(InlineMmFeatures::new(vec![item]).is_err());
        }
        assert!(
            InlineMmFeatures::new(vec![feature(2, batched())])
                .unwrap()
                .into_features(3)
                .is_err()
        );
        for mask in [
            WireTensor::from_bool(vec![1], vec![true]).unwrap(),
            WireTensor::from_u32(vec![2], vec![1, 0]).unwrap(),
        ] {
            let mut item = feature(0, batched());
            item.mm_position.is_embed = Some(mask);
            assert!(InlineMmFeatures::new(vec![item]).is_err());
        }
    }

    #[test]
    fn inline_features_reject_incompatible_batching_and_flat_axes() {
        let flat = MmField::Flat(MmFlatField {
            slices: vec![],
            dim: 2,
            keep_on_cpu: false,
        });
        assert!(InlineMmFeatures::new(vec![feature(0, flat)]).is_err());
        let shared = MmField::Shared(MmSharedField {
            batch_size: 1,
            keep_on_cpu: false,
        });
        assert!(InlineMmFeatures::new(vec![feature(0, batched()), feature(2, shared)]).is_err());
    }

    #[test]
    fn kwargs_decoder_rejects_trailing_truncated_and_deeply_nested_input() {
        let kwargs = feature(0, batched()).data.unwrap();
        let valid = rmp_serde::to_vec_named(&kwargs).unwrap();
        assert_eq!(decode_inline_mm_kwargs(&valid).unwrap(), kwargs);
        let mut trailing = valid.clone();
        trailing.push(0);
        let mut nested_kwargs = kwargs.clone();
        let mut value = MmKwargValue::Int(1);
        for _ in 0..64 {
            value = MmKwargValue::List(vec![value]);
        }
        nested_kwargs.get_mut("pixels").unwrap().data = Some(value);
        let nested = rmp_serde::to_vec_named(&nested_kwargs).unwrap();
        // A declared array length must not drive a proportional allocation.
        let huge_array = vec![
            0x81, 0xa1, b'x', 0x81, 0xa4, b'd', b'a', b't', b'a', 0xdd, 0xff, 0xff, 0xff, 0xff,
        ];
        for bytes in [&valid[..valid.len() - 1], &trailing, &nested, &huge_array] {
            assert!(decode_inline_mm_kwargs(bytes).is_err());
        }
    }

    #[test]
    fn kwargs_decode_python_encoder_tensor_and_factory_tuple() {
        // MsgpackEncoder(size_threshold=1 << 30), vLLM 6cbb3c154e:
        // pixel_values = tensor([1., 2.]), MultiModalFlatField(slices=[slice(0, 2)]).
        let bytes = hex::decode(concat!(
            "81ac706978656c5f76616c75657382a46461746193a7666c6f617433329102d7030000803f00000040",
            "a56669656c6492a4666c617483ab6b6565705f6f6e5f637075c2a6736c6963657391930002c0a364696d00"
        ))
        .unwrap();
        let kwargs = decode_inline_mm_kwargs(&bytes).unwrap();
        let expected = MmFieldElem {
            data: Some(MmKwargValue::Tensor(
                WireTensor::from_f32(vec![2], vec![1., 2.]).unwrap(),
            )),
            field: MmField::Flat(MmFlatField {
                slices: vec![MmSlice::Slice(SliceSpec {
                    start: Some(0),
                    stop: Some(2),
                    step: None,
                })],
                dim: 0,
                keep_on_cpu: false,
            }),
        };
        assert_eq!(kwargs["pixel_values"], expected);
    }
}
