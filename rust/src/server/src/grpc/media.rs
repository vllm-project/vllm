// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Protobuf media adaptation into the shared multimodal preparation boundary.

use std::collections::BTreeMap;
use std::io::Cursor;

use llm_multimodal::ImageDetail;
use serde::Deserialize as _;
use thiserror_ext::AsReport as _;
use tonic::Status;
use url::Url;
use vllm_chat::MediaContentPart;
use vllm_chat::multimodal::{MediaOptions, MultimodalInput, PlacedMedia};
use vllm_engine_core_client::protocol::dtype::{Endianness, NumpyDtype, TensorDtype};
use vllm_engine_core_client::protocol::multimodal::{
    MmBatchedField, MmFeatureSpec, MmField, MmFieldElem, MmFlatField, MmKwargValue, MmKwargsItem,
    MmModality, MmSlice, PlaceholderRange, SliceSpec,
};
use vllm_engine_core_client::protocol::tensor::WireTensor;

use super::pb;

/// Maximum total serialized kwargs size accepted in one gRPC request.
const MAX_INLINE_MM_BYTES: usize = 16 * 1024 * 1024;
const MAX_MSGPACK_DEPTH: usize = 32;

/// Adapt a request containing either raw media or preprocessed features.
///
/// Require a single input kind and bound the total kwargs size before decoding.
/// Shared media preparation validates features, model support, item limits, and prompt bounds.
pub(super) fn from_proto(
    media: Vec<pb::MediaItem>,
    ec_transfer_present: bool,
) -> Result<MultimodalInput, Status> {
    let is_feature =
        |item: &pb::MediaItem| matches!(item.source, Some(pb::media_item::Source::Features(_)));
    if !media.iter().any(is_feature) {
        return raw_parts_from_proto(media).map(MultimodalInput::Raw);
    }
    let mut encoded_bytes = 0usize;
    for item in &media {
        match &item.source {
            Some(pb::media_item::Source::Features(feature)) => {
                encoded_bytes =
                    encoded_bytes.saturating_add(feature.kwargs.as_ref().map_or(0, Vec::len));
            }
            _ if item.marker_offset.is_none() => return Err(mixed_media_error()),
            _ => {}
        }
    }
    if encoded_bytes > MAX_INLINE_MM_BYTES {
        return Err(Status::resource_exhausted(format!(
            "inline multimodal payload exceeds {MAX_INLINE_MM_BYTES} bytes"
        )));
    }
    let mut features = Vec::new();
    let mut raw = Vec::new();
    for (index, item) in media.into_iter().enumerate() {
        if is_feature(&item) {
            features.push(
                feature_from_proto(item, ec_transfer_present).map_err(|error| {
                    Status::new(error.code(), format!("media[{index}]: {}", error.message()))
                })?,
            );
        } else {
            let marker_offset = usize::try_from(item.marker_offset.unwrap_or_default())
                .map_err(|_| Status::invalid_argument("marker_offset exceeds platform limits"))?;
            raw.push(PlacedMedia {
                part: raw_part_from_proto(index, item)?,
                marker_offset,
            });
        }
    }
    if raw.is_empty() {
        Ok(MultimodalInput::Preprocessed(features))
    } else {
        Ok(MultimodalInput::Mixed { raw, features })
    }
}

/// Request-level preprocessing options; absent structs mean none.
pub(super) fn options_from_proto(
    processor_kwargs: Option<prost_types::Struct>,
    media_io_kwargs: Option<prost_types::Struct>,
) -> MediaOptions {
    let to_map = |value: Option<prost_types::Struct>| {
        value
            .map(|value| {
                value
                    .fields
                    .into_iter()
                    .map(|(key, value)| (key, prost_to_json(value)))
                    .collect()
            })
            .unwrap_or_default()
    };
    MediaOptions {
        processor_kwargs: to_map(processor_kwargs),
        media_io_kwargs: to_map(media_io_kwargs),
    }
}

fn prost_to_json(value: prost_types::Value) -> serde_json::Value {
    use prost_types::value::Kind;
    match value.kind {
        Some(Kind::NullValue(_)) | None => serde_json::Value::Null,
        Some(Kind::BoolValue(flag)) => serde_json::Value::Bool(flag),
        Some(Kind::NumberValue(number)) => {
            // Integral numbers stay integers so config fields like `max_pixels` parse.
            if number.fract() == 0.0 && number.abs() < 9.0e15 {
                serde_json::Value::from(number as i64)
            } else {
                serde_json::Number::from_f64(number)
                    .map_or(serde_json::Value::Null, serde_json::Value::Number)
            }
        }
        Some(Kind::StringValue(text)) => serde_json::Value::String(text),
        Some(Kind::ListValue(list)) => {
            serde_json::Value::Array(list.values.into_iter().map(prost_to_json).collect())
        }
        Some(Kind::StructValue(nested)) => serde_json::Value::Object(
            nested
                .fields
                .into_iter()
                .map(|(key, value)| (key, prost_to_json(value)))
                .collect(),
        ),
    }
}

fn feature_from_proto(
    item: pb::MediaItem,
    ec_transfer_present: bool,
) -> Result<MmFeatureSpec, Status> {
    let modality = match item.modality() {
        pb::Modality::Image => MmModality::Image,
        pb::Modality::Video => MmModality::Video,
        pb::Modality::Audio => MmModality::Audio,
        pb::Modality::Unspecified => return Err(Status::invalid_argument("modality is required")),
    };
    let Some(pb::media_item::Source::Features(feature)) = item.source else {
        return Err(mixed_media_error());
    };
    let metadata = feature.metadata;
    let (data, identifier) = match feature.kwargs {
        Some(bytes) => (decode_inline_mm_kwargs(&bytes)?, feature.identifier),
        None => {
            let has_tensors = !feature.tensors.is_empty();
            if !ec_transfer_present && !has_tensors {
                return Err(Status::invalid_argument(
                    "metadata-only multimodal items require ec_transfer_params",
                ));
            }
            let hash =
                feature.mm_hash.clone().filter(|hash| !hash.is_empty()).ok_or_else(|| {
                    Status::invalid_argument("metadata-only features require mm_hash")
                })?;
            let identifier = if feature.identifier.is_empty() {
                hash
            } else {
                feature.identifier
            };
            let mut kwargs = match metadata {
                Some(metadata) if !(has_tensors && metadata.fields.is_empty()) => {
                    kwargs_from_metadata(&metadata)?
                }
                _ if has_tensors => BTreeMap::new(),
                _ => {
                    return Err(Status::invalid_argument(
                        "features.kwargs is required; cache-only features are unsupported",
                    ));
                }
            };
            for (key, bytes) in &feature.tensors {
                let field = tensor_field(key, bytes).map_err(|error| {
                    Status::invalid_argument(format!("features.tensors.{key}: {error}"))
                })?;
                kwargs.insert(key.clone(), field);
            }
            (kwargs, identifier)
        }
    };
    let offset = usize::try_from(feature.offset)
        .map_err(|_| Status::invalid_argument("features.offset exceeds platform limits"))?;
    let length = usize::try_from(feature.length)
        .map_err(|_| Status::invalid_argument("features.length exceeds platform limits"))?;
    let is_embed = if feature.is_embed.is_empty() {
        None
    } else {
        Some(
            WireTensor::from_bool(vec![length], feature.is_embed)
                .map_err(Status::invalid_argument)?,
        )
    };
    Ok(MmFeatureSpec {
        modality,
        data: Some(data),
        identifier,
        mm_hash: feature.mm_hash,
        mm_position: PlaceholderRange {
            offset,
            length,
            is_embed,
        },
    })
}

/// Turn HTTP-style placeholder metadata into the keep-on-CPU kwargs the engine
/// already accepts. Pixel tensors stay out of this path.
fn kwargs_from_metadata(metadata: &prost_types::Struct) -> Result<MmKwargsItem, Status> {
    if metadata.fields.is_empty() {
        return Err(Status::invalid_argument(
            "features.metadata must contain at least one field",
        ));
    }
    let mut kwargs = BTreeMap::new();
    for (key, value) in &metadata.fields {
        let mut numbers = Vec::new();
        flatten_metadata_numbers(value, &mut numbers)?;
        if numbers.is_empty() {
            return Err(Status::invalid_argument(format!(
                "features.metadata.{key} must contain numbers"
            )));
        }
        // Qwen mRoPE reads one feature as a 1-D `[t, h, w]`. A leading
        // batch axis (`[1, 3]`) makes `t, h, w = grid.tolist()` fail.
        // Several rows stay `[n, 3]` for a single item that owns n grids.
        let shape = if key.contains("grid_thw") && numbers.len() == 3 {
            vec![3]
        } else if key.contains("grid_thw") && numbers.len().is_multiple_of(3) {
            vec![numbers.len() / 3, 3]
        } else {
            vec![numbers.len()]
        };
        // Timestamps stay float even when every stamp is whole.
        let integral =
            !key.contains("timestamp") && numbers.iter().all(|number| number.fract() == 0.0);
        let data = if integral {
            let ints = numbers.into_iter().map(|number| number as i64).collect();
            MmKwargValue::Tensor(
                WireTensor::from_i64(shape, ints).map_err(Status::invalid_argument)?,
            )
        } else {
            MmKwargValue::Tensor(
                WireTensor::from_f32(
                    shape,
                    numbers.into_iter().map(|number| number as f32).collect(),
                )
                .map_err(Status::invalid_argument)?,
            )
        };
        kwargs.insert(
            key.clone(),
            MmFieldElem {
                data: Some(data),
                field: MmField::Batched(MmBatchedField { keep_on_cpu: true }),
            },
        );
    }
    Ok(kwargs)
}

/// One `.npy` tensor as a kwarg: embeddings are flat over the item (one row
/// per placeholder token), grids follow the metadata shape rule.
fn tensor_field(key: &str, bytes: &[u8]) -> Result<MmFieldElem, String> {
    let (dtype, mut shape, data) = parse_npy(bytes)?;
    let field = if key.ends_with("_embeds") {
        let rows = *shape.first().ok_or("embeddings need at least one dimension")?;
        MmField::Flat(MmFlatField {
            slices: vec![MmSlice::Slice(SliceSpec {
                start: Some(0),
                stop: Some(isize::try_from(rows).map_err(|_| "too many rows")?),
                step: None,
            })],
            dim: 0,
            keep_on_cpu: false,
        })
    } else {
        if key.contains("grid_thw") && shape == [1, 3] {
            shape = vec![3];
        }
        MmField::Batched(MmBatchedField { keep_on_cpu: true })
    };
    Ok(MmFieldElem {
        data: Some(MmKwargValue::Tensor(WireTensor::from_raw(
            dtype, shape, data,
        ))),
        field,
    })
}

/// Parse a C-order little-endian NumPy `.npy` buffer (format 1.0–3.0).
fn parse_npy(bytes: &[u8]) -> Result<(TensorDtype, Vec<usize>, Vec<u8>), String> {
    let rest = bytes
        .strip_prefix(b"\x93NUMPY")
        .ok_or("not a NumPy .npy buffer; torch.save payloads are unsupported on gRPC")?;
    let (header_len, rest) = match rest {
        [1, _, a, b, rest @ ..] => (usize::from(u16::from_le_bytes([*a, *b])), rest),
        [2 | 3, _, a, b, c, d, rest @ ..] => (u32::from_le_bytes([*a, *b, *c, *d]) as usize, rest),
        _ => return Err("unsupported .npy version".to_string()),
    };
    if rest.len() < header_len {
        return Err("truncated .npy header".to_string());
    }
    let (header, data) = rest.split_at(header_len);
    let header = std::str::from_utf8(header).map_err(|_| "non-UTF-8 .npy header")?;
    let value_of = |key: &str| {
        let start = header.find(&format!("'{key}'"))? + key.len() + 2;
        let tail = header[start..].trim_start().strip_prefix(':')?.trim_start();
        Some(tail)
    };
    let descr = value_of("descr")
        .and_then(|tail| tail.strip_prefix('\'')?.split('\'').next())
        .ok_or(".npy header has no descr")?;
    let dtype: NumpyDtype = descr.parse()?;
    if dtype.endianness == Endianness::Big {
        return Err("big-endian .npy data is unsupported".to_string());
    }
    if value_of("fortran_order").is_none_or(|tail| !tail.starts_with("False")) {
        return Err("only C-order .npy data is supported".to_string());
    }
    let shape = value_of("shape")
        .and_then(|tail| tail.strip_prefix('(')?.split(')').next())
        .ok_or(".npy header has no shape")?
        .split(',')
        .map(str::trim)
        .filter(|dim| !dim.is_empty())
        .map(|dim| dim.parse::<usize>().map_err(|_| format!("bad .npy dimension {dim:?}")))
        .collect::<Result<Vec<_>, _>>()?;
    let expected = shape
        .iter()
        .try_fold(dtype.scalar.element_size(), |size, dim| {
            size.checked_mul(*dim)
        })
        .ok_or(".npy shape overflows")?;
    if data.len() != expected {
        return Err(format!(
            ".npy holds {} data bytes, shape needs {expected}",
            data.len()
        ));
    }
    Ok((dtype.scalar, shape, data.to_vec()))
}

fn flatten_metadata_numbers(value: &prost_types::Value, out: &mut Vec<f64>) -> Result<(), Status> {
    match value.kind.as_ref() {
        Some(prost_types::value::Kind::NumberValue(number)) => {
            if !number.is_finite() {
                return Err(Status::invalid_argument(
                    "features.metadata numbers must be finite",
                ));
            }
            out.push(*number);
            Ok(())
        }
        Some(prost_types::value::Kind::ListValue(list)) => {
            for item in &list.values {
                flatten_metadata_numbers(item, out)?;
            }
            Ok(())
        }
        _ => Err(Status::invalid_argument(
            "features.metadata values must be numbers or arrays of numbers",
        )),
    }
}

/// Decode one inline item using the engine protocol's typed serde schema.
fn decode_inline_mm_kwargs(bytes: &[u8]) -> Result<MmKwargsItem, Status> {
    let mut decoder = rmp_serde::Deserializer::new(Cursor::new(bytes));
    decoder.set_max_depth(MAX_MSGPACK_DEPTH);
    let kwargs = MmKwargsItem::deserialize(&mut decoder).map_err(|error| {
        Status::invalid_argument(format!(
            "invalid multimodal kwargs MessagePack: {}",
            error.as_report()
        ))
    })?;
    if decoder.position() != bytes.len() as u64 {
        return Err(Status::invalid_argument(
            "kwargs contains trailing MessagePack data",
        ));
    }
    Ok(kwargs)
}

fn mixed_media_error() -> Status {
    Status::invalid_argument(
        "raw media mixed with preprocessed media features must set marker_offset",
    )
}

fn raw_parts_from_proto(media: Vec<pb::MediaItem>) -> Result<Vec<MediaContentPart>, Status> {
    media
        .into_iter()
        .enumerate()
        .map(|(index, item)| raw_part_from_proto(index, item))
        .collect()
}

fn raw_part_from_proto(index: usize, item: pb::MediaItem) -> Result<MediaContentPart, Status> {
    {
        let modality = item.modality();
        if modality == pb::Modality::Unspecified {
            return Err(Status::invalid_argument(format!(
                "media[{index}].modality is required"
            )));
        }
        let uuid = (!item.uuid.is_empty()).then_some(item.uuid);
        let mime_type = (!item.mime_type.is_empty()).then_some(item.mime_type);
        let detail = item
            .detail
            .as_deref()
            .map(|detail| {
                serde_json::from_value::<ImageDetail>(serde_json::Value::String(detail.to_owned()))
                    .map_err(|_| {
                        Status::invalid_argument(format!(
                            "media[{index}].detail must be auto, low or high"
                        ))
                    })
            })
            .transpose()?;
        let fps = item.fps;
        let max_long_side_pixel = item.max_long_side_pixel;
        let source = item.source.ok_or_else(|| {
            Status::invalid_argument(format!("media[{index}].source is required"))
        })?;
        match &source {
            pb::media_item::Source::Url(url) => {
                validate_media_uri(index, "url", url, &["http", "https", "file"])?;
            }
            pb::media_item::Source::DataUri(uri) => {
                validate_media_uri(index, "data_uri", uri, &["data"])?;
            }
            pb::media_item::Source::RawBytes(_) => {}
            pb::media_item::Source::Features(_) => return Err(mixed_media_error()),
        }
        let part = match (modality, source) {
            (
                pb::Modality::Image,
                pb::media_item::Source::Url(url) | pb::media_item::Source::DataUri(url),
            ) => MediaContentPart::ImageUrl {
                url,
                detail,
                uuid,
                max_long_side_pixel,
            },
            (pb::Modality::Image, pb::media_item::Source::RawBytes(data)) => {
                MediaContentPart::ImageData {
                    data,
                    mime_type,
                    uuid,
                    detail,
                }
            }
            (
                pb::Modality::Video,
                pb::media_item::Source::Url(url) | pb::media_item::Source::DataUri(url),
            ) => MediaContentPart::VideoUrl {
                url,
                uuid,
                fps,
                max_long_side_pixel,
            },
            (pb::Modality::Video, pb::media_item::Source::RawBytes(data)) => {
                MediaContentPart::VideoData {
                    data,
                    mime_type,
                    uuid,
                }
            }
            (
                pb::Modality::Audio,
                pb::media_item::Source::Url(url) | pb::media_item::Source::DataUri(url),
            ) => MediaContentPart::AudioUrl { url, uuid },
            (pb::Modality::Audio, pb::media_item::Source::RawBytes(data)) => {
                MediaContentPart::AudioData {
                    data,
                    mime_type,
                    uuid,
                }
            }
            (_, pb::media_item::Source::Features(_)) => return Err(mixed_media_error()),
            (pb::Modality::Unspecified, _) => unreachable!("modality validated above"),
        };
        Ok(part)
    }
}

fn validate_media_uri(
    index: usize,
    field: &str,
    value: &str,
    allowed_schemes: &[&str],
) -> Result<(), Status> {
    let uri = Url::parse(value).map_err(|_| {
        Status::invalid_argument(format!("media[{index}].{field} is not a valid URI"))
    })?;
    if !allowed_schemes.contains(&uri.scheme()) {
        return Err(Status::invalid_argument(format!(
            "media[{index}].{field} must use the {} scheme",
            allowed_schemes.join(" or ")
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use vllm_engine_core_client::protocol::multimodal::{
        MmBatchedField, MmField, MmFieldElem, MmFlatField, MmKwargValue, MmSlice, SliceSpec,
    };

    fn preprocessed() -> pb::MediaItem {
        let kwargs = BTreeMap::from([(
            "x",
            MmFieldElem {
                data: Some(MmKwargValue::Int(1)),
                field: MmField::Batched(MmBatchedField { keep_on_cpu: false }),
            },
        )]);
        pb::MediaItem {
            modality: pb::Modality::Image as i32,
            source: Some(pb::media_item::Source::Features(
                pb::PreprocessedMediaFeatures {
                    kwargs: Some(rmp_serde::to_vec_named(&kwargs).unwrap()),
                    identifier: "image-id".to_owned(),
                    length: 1,
                    ..Default::default()
                },
            )),
            ..Default::default()
        }
    }

    #[test]
    fn mixed_media_needs_marker_offsets_in_both_orders() {
        let raw = pb::MediaItem {
            modality: pb::Modality::Image as i32,
            source: Some(pb::media_item::Source::RawBytes(vec![1])),
            ..Default::default()
        };
        for media in [
            vec![raw.clone(), preprocessed()],
            vec![preprocessed(), raw.clone()],
        ] {
            assert_eq!(
                from_proto(media, false).unwrap_err().code(),
                tonic::Code::InvalidArgument
            );
        }
        let placed = pb::MediaItem {
            marker_offset: Some(7),
            ..raw
        };
        let MultimodalInput::Mixed { raw, features } =
            from_proto(vec![preprocessed(), placed], false).unwrap()
        else {
            panic!("expected mixed input");
        };
        assert_eq!(raw.len(), 1);
        assert_eq!(raw[0].marker_offset, 7);
        assert_eq!(features.len(), 1);
    }

    fn npy(descr: &str, shape: &str, data: &[u8]) -> Vec<u8> {
        let mut header =
            format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
        while (10 + header.len() + 1) % 64 != 0 {
            header.push(' ');
        }
        header.push('\n');
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend((header.len() as u16).to_le_bytes());
        bytes.extend(header.as_bytes());
        bytes.extend(data);
        bytes
    }

    #[test]
    fn npy_tensors_become_flat_embeds_and_batched_grids() {
        let embeds = npy("<f4", "(2, 2)", &[0; 16]);
        let grid = npy(
            "<i8",
            "(1, 3)",
            &[
                1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0,
            ],
        );
        let mut item = preprocessed();
        let Some(pb::media_item::Source::Features(feature)) = &mut item.source else {
            unreachable!()
        };
        feature.kwargs = None;
        feature.mm_hash = Some("cached".into());
        feature.tensors = [
            ("image_embeds".to_owned(), embeds),
            ("image_grid_thw".to_owned(), grid),
        ]
        .into();
        let MultimodalInput::Preprocessed(features) = from_proto(vec![item], false).unwrap() else {
            panic!("expected features");
        };
        let data = features[0].data.as_ref().unwrap();
        assert!(matches!(&data["image_embeds"].field, MmField::Flat(flat)
            if flat.slices == [MmSlice::Slice(SliceSpec { start: Some(0), stop: Some(2), step: None })]));
        let Some(MmKwargValue::Tensor(grid)) = &data["image_grid_thw"].data else {
            panic!("grid tensor");
        };
        assert_eq!(grid.shape, [3]);
        assert!(parse_npy(&npy("<f4", "(3,)", &[0; 8])).is_err());
        assert!(parse_npy(&npy(">f4", "(2,)", &[0; 8])).is_err());
        assert!(parse_npy(b"PK\x03\x04torch").unwrap_err().contains("torch.save"));
    }

    #[test]
    fn request_options_convert_struct_values_to_json() {
        use prost_types::value::Kind;
        let number = |value| prost_types::Value {
            kind: Some(Kind::NumberValue(value)),
        };
        let processor = prost_types::Struct {
            fields: BTreeMap::from([
                ("min_pixels".to_owned(), number(1024.0)),
                ("scale".to_owned(), number(0.5)),
            ]),
        };
        let options = options_from_proto(Some(processor), None);
        assert_eq!(
            options.processor_kwargs["min_pixels"],
            serde_json::json!(1024)
        );
        assert_eq!(options.processor_kwargs["scale"], serde_json::json!(0.5));
        assert!(options.media_io_kwargs.is_empty());
        assert!(options_from_proto(None, None).is_empty());
    }

    #[test]
    fn protobuf_features_require_modality_inline_kwargs_and_a_matching_mask() {
        let mut missing_modality = preprocessed();
        missing_modality.modality = 99;
        let mut missing_kwargs = preprocessed();
        let Some(pb::media_item::Source::Features(feature)) = &mut missing_kwargs.source else {
            unreachable!()
        };
        feature.kwargs = None;
        let mut invalid_mask = preprocessed();
        let Some(pb::media_item::Source::Features(feature)) = &mut invalid_mask.source else {
            unreachable!()
        };
        feature.is_embed = vec![true, false];
        for media in [missing_modality, missing_kwargs, invalid_mask] {
            let error = from_proto(vec![media], false).unwrap_err();
            assert_eq!(error.code(), tonic::Code::InvalidArgument);
            assert!(error.message().starts_with("media[0]:"));
        }
    }

    #[test]
    fn aggregate_payload_limit_is_checked_before_decoding_items() {
        let mut item = preprocessed();
        let Some(pb::media_item::Source::Features(feature)) = &mut item.source else {
            unreachable!()
        };
        feature.kwargs = Some(vec![0; MAX_INLINE_MM_BYTES / 2 + 1]);
        let error = from_proto(vec![item.clone(), item], false).unwrap_err();
        assert_eq!(error.code(), tonic::Code::ResourceExhausted);
    }

    #[test]
    fn kwargs_decoder_rejects_trailing_truncated_and_deeply_nested_input() {
        let kwargs = BTreeMap::from([(
            "pixels".to_owned(),
            MmFieldElem {
                data: Some(MmKwargValue::Int(1)),
                field: MmField::Batched(MmBatchedField { keep_on_cpu: false }),
            },
        )]);
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
        let bytes = b"\x81\xacpixel_values\x82\xa4data\x93\xa7float32\x91\x02\xd7\x03\
            \x00\x00\x80\x3f\x00\x00\x00\x40\
            \xa5field\x92\xa4flat\x83\xabkeep_on_cpu\xc2\
            \xa6slices\x91\x93\x00\x02\xc0\xa3dim\x00";
        let kwargs = decode_inline_mm_kwargs(bytes).unwrap();
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

    #[test]
    fn metadata_only_features_become_keep_on_cpu_kwargs() {
        let item = pb::MediaItem {
            modality: pb::Modality::Image as i32,
            source: Some(pb::media_item::Source::Features(
                pb::PreprocessedMediaFeatures {
                    kwargs: None,
                    identifier: String::new(),
                    offset: 4,
                    length: 8,
                    mm_hash: Some("hash-a".to_owned()),
                    is_embed: Vec::new(),
                    metadata: Some(prost_types::Struct {
                        fields: std::iter::once((
                            "image_grid_thw".to_owned(),
                            prost_types::Value {
                                kind: Some(prost_types::value::Kind::ListValue(
                                    prost_types::ListValue {
                                        values: [1.0, 2.0, 2.0]
                                            .into_iter()
                                            .map(|number| prost_types::Value {
                                                kind: Some(prost_types::value::Kind::NumberValue(
                                                    number,
                                                )),
                                            })
                                            .collect(),
                                    },
                                )),
                            },
                        ))
                        .collect(),
                    }),
                    tensors: Default::default(),
                },
            )),
            ..Default::default()
        };
        let MultimodalInput::Preprocessed(features) =
            from_proto(vec![item.clone()], true).expect("metadata-only feature")
        else {
            panic!("expected preprocessed features");
        };
        assert_eq!(features[0].identifier, "hash-a");
        assert_eq!(features[0].mm_position.offset, 4);
        assert_eq!(features[0].mm_position.length, 8);
        let data = features[0].data.as_ref().unwrap();
        assert!(data.contains_key("image_grid_thw"));
        assert!(!data.contains_key("pixel_values"));
        let denied = from_proto(vec![item], false).unwrap_err();
        assert!(denied.message().contains("require ec_transfer_params"));
    }

    fn numbers(values: &[f64]) -> prost_types::Value {
        prost_types::Value {
            kind: Some(prost_types::value::Kind::ListValue(
                prost_types::ListValue {
                    values: values
                        .iter()
                        .map(|number| prost_types::Value {
                            kind: Some(prost_types::value::Kind::NumberValue(*number)),
                        })
                        .collect(),
                },
            )),
        }
    }

    #[test]
    fn video_metadata_keeps_one_grid_flat_and_timestamps_float() {
        let metadata = prost_types::Struct {
            fields: [
                ("video_grid_thw".to_owned(), numbers(&[2.0, 4.0, 4.0])),
                ("timestamps".to_owned(), numbers(&[0.0, 1.0])),
                (
                    "image_grid_thw".to_owned(),
                    numbers(&[1.0, 2.0, 2.0, 1.0, 4.0, 4.0]),
                ),
            ]
            .into_iter()
            .collect(),
        };
        let kwargs = kwargs_from_metadata(&metadata).unwrap();
        let tensor = |key: &str| match kwargs[key].data.as_ref() {
            Some(MmKwargValue::Tensor(tensor)) => tensor.clone(),
            other => panic!("{key}: {other:?}"),
        };
        assert_eq!(
            tensor("video_grid_thw"),
            WireTensor::from_i64(vec![3], vec![2, 4, 4]).unwrap()
        );
        assert_eq!(
            tensor("timestamps"),
            WireTensor::from_f32(vec![2], vec![0.0, 1.0]).unwrap()
        );
        assert_eq!(tensor("image_grid_thw").shape, vec![2, 3]);
    }

    #[test]
    fn raw_items_carry_per_item_processing_options() {
        let image = pb::MediaItem {
            modality: pb::Modality::Image as i32,
            source: Some(pb::media_item::Source::Url("https://x/a.png".into())),
            detail: Some("low".into()),
            max_long_side_pixel: Some(512),
            ..Default::default()
        };
        let video = pb::MediaItem {
            modality: pb::Modality::Video as i32,
            source: Some(pb::media_item::Source::Url("https://x/v.mp4".into())),
            fps: Some(2.0),
            ..Default::default()
        };
        let parts = raw_parts_from_proto(vec![image.clone(), video]).unwrap();
        assert!(matches!(
            &parts[0],
            MediaContentPart::ImageUrl {
                detail: Some(ImageDetail::Low),
                max_long_side_pixel: Some(512),
                ..
            }
        ));
        assert!(
            matches!(&parts[1], MediaContentPart::VideoUrl { fps: Some(fps), .. } if *fps == 2.0)
        );
        let bad = pb::MediaItem {
            detail: Some("ultra".into()),
            ..image.clone()
        };
        let error = raw_parts_from_proto(vec![bad]).unwrap_err();
        assert!(error.message().contains("auto, low or high"));

        let with_url = |url: &str| pb::MediaItem {
            source: Some(pb::media_item::Source::Url(url.into())),
            ..image.clone()
        };
        raw_parts_from_proto(vec![with_url("file:///media/a.png")]).unwrap();
        let error = raw_parts_from_proto(vec![with_url("ftp://x/a.png")]).unwrap_err();
        assert!(error.message().contains("scheme"), "{}", error.message());
    }
}
