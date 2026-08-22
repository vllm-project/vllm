// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Conversion between gRPC protobuf types and internal `vllm-text`
//! request/response types.

use std::collections::HashMap;
use std::io::Cursor;

use serde::Deserialize as _;
use sha2::{Digest, Sha256};
use tonic::Status;
use url::Url;
use uuid::Uuid;
use vllm_chat::MediaContentPart;
use vllm_engine_core_client::protocol::multimodal::{
    MmFeatureSpec, MmFeatures, MmField, MmKwargValue, MmKwargsItem, MmSlice, PlaceholderRange,
};
use vllm_engine_core_client::protocol::output::StopReason;
use vllm_engine_core_client::protocol::structured_outputs::StructuredOutputsParams;
use vllm_engine_core_client::protocol::tensor::{WireArrayData, WireTensor};
use vllm_text::{
    DecodedLogprobs, DecodedPromptLogprobs, FinishReason, Finished, Prompt, SamplingParams,
    TextDecodeOptions, TextRequest,
};

use super::pb;

const MAX_MM_FEATURE_BYTES: usize = 16 * 1024 * 1024;
const MAX_MM_FEATURES: usize = 64;
const MAX_MM_DEPTH: usize = 32;
const MAX_MM_NODES: usize = 65_536;
const MAX_MM_FIELDS_PER_ITEM: usize = 256;
const MAX_MM_KEY_BYTES: usize = 256;
const MAX_MM_TENSOR_RANK: usize = 32;

pub enum GrpcMultimodalInput {
    None,
    Raw(Vec<MediaContentPart>),
    Preprocessed(MmFeatures),
}

pub fn multimodal_input_from_request(
    media: Vec<pb::MediaItem>,
) -> Result<GrpcMultimodalInput, Status> {
    let mut parts = Vec::with_capacity(media.len());
    let mut features = Vec::with_capacity(media.len());
    let mut encoded_feature_bytes = 0usize;
    let mut wire_nodes = 0usize;
    for (index, item) in media.into_iter().enumerate() {
        let modality = match item.modality() {
            pb::Modality::Image => "image",
            pb::Modality::Video => "video",
            pb::Modality::Audio => "audio",
            pb::Modality::Unspecified => {
                return Err(Status::invalid_argument(format!(
                    "media[{index}].modality is required"
                )));
            }
        };
        let uuid = (!item.uuid.is_empty()).then_some(item.uuid);
        let mime_type = (!item.mime_type.is_empty()).then_some(item.mime_type);
        let part = match item.source {
            Some(pb::media_item::Source::Url(url)) => {
                ensure_raw_image(index, modality)?;
                validate_media_uri(index, "url", &url, &["http", "https"])?;
                MediaContentPart::ImageUrl {
                    url,
                    detail: None,
                    uuid,
                }
            }
            Some(pb::media_item::Source::DataUri(uri)) => {
                ensure_raw_image(index, modality)?;
                validate_media_uri(index, "data_uri", &uri, &["data"])?;
                MediaContentPart::ImageUrl {
                    url: uri,
                    detail: None,
                    uuid,
                }
            }
            Some(pb::media_item::Source::RawBytes(bytes)) => {
                ensure_raw_image(index, modality)?;
                MediaContentPart::ImageData {
                    data: bytes,
                    mime_type,
                    uuid,
                    detail: None,
                }
            }
            Some(pb::media_item::Source::Features(feature)) => {
                if !parts.is_empty() {
                    return Err(mixed_media_error());
                }
                if features.len() >= MAX_MM_FEATURES {
                    return Err(Status::resource_exhausted(
                        "too many preprocessed media features",
                    ));
                }
                features.push(preprocessed_feature(
                    index,
                    modality,
                    feature,
                    &mut encoded_feature_bytes,
                    &mut wire_nodes,
                )?);
                continue;
            }
            None => {
                return Err(Status::invalid_argument(format!(
                    "media[{index}].source is required"
                )));
            }
        };
        if !features.is_empty() {
            return Err(mixed_media_error());
        }
        parts.push(part);
    }
    if !features.is_empty() {
        features.sort_unstable_by_key(|feature| feature.mm_position.offset);
        for pair in features.windows(2) {
            let previous_end =
                pair[0].mm_position.offset.checked_add(pair[0].mm_position.length).ok_or_else(
                    || Status::invalid_argument("preprocessed media placeholder range overflows"),
                )?;
            if previous_end > pair[1].mm_position.offset {
                return Err(Status::invalid_argument(
                    "preprocessed media placeholder ranges cannot overlap",
                ));
            }
        }
        validate_mm_field_metadata(&features)?;
        Ok(GrpcMultimodalInput::Preprocessed(features))
    } else if !parts.is_empty() {
        Ok(GrpcMultimodalInput::Raw(parts))
    } else {
        Ok(GrpcMultimodalInput::None)
    }
}

fn ensure_raw_image(index: usize, modality: &str) -> Result<(), Status> {
    if modality == "image" {
        Ok(())
    } else {
        Err(Status::unimplemented(format!(
            "media[{index}] raw {modality} input is not supported by the gRPC service"
        )))
    }
}

fn mixed_media_error() -> Status {
    Status::invalid_argument("raw media and preprocessed media features cannot be mixed")
}

fn preprocessed_feature(
    index: usize,
    modality: &str,
    feature: pb::PreprocessedMediaFeatures,
    encoded_feature_bytes: &mut usize,
    wire_nodes: &mut usize,
) -> Result<MmFeatureSpec, Status> {
    let offset = usize::try_from(feature.offset).map_err(|_| {
        Status::invalid_argument(format!(
            "media[{index}].features.offset exceeds platform limits"
        ))
    })?;
    let length = usize::try_from(feature.length).map_err(|_| {
        Status::invalid_argument(format!(
            "media[{index}].features.length exceeds platform limits"
        ))
    })?;
    if length == 0 {
        return Err(Status::invalid_argument(format!(
            "media[{index}].features.length must be positive"
        )));
    }
    offset.checked_add(length).ok_or_else(|| {
        Status::invalid_argument(format!(
            "media[{index}].features placeholder range overflows"
        ))
    })?;
    let is_embed = if feature.is_embed.is_empty() {
        None
    } else {
        if feature.is_embed.len() != length {
            return Err(Status::invalid_argument(format!(
                "media[{index}].features.is_embed length must equal placeholder length"
            )));
        }
        Some(
            WireTensor::from_bool(vec![length], feature.is_embed).map_err(|error| {
                Status::invalid_argument(format!(
                    "media[{index}].features.is_embed is invalid: {error}"
                ))
            })?,
        )
    };
    let raw = feature.kwargs.ok_or_else(|| {
        Status::invalid_argument(format!(
            "media[{index}].features.kwargs is required; cache-only features are unsupported"
        ))
    })?;
    *encoded_feature_bytes = encoded_feature_bytes.checked_add(raw.len()).ok_or_else(|| {
        Status::resource_exhausted("preprocessed media feature payload is too large")
    })?;
    if *encoded_feature_bytes > MAX_MM_FEATURE_BYTES {
        return Err(Status::resource_exhausted(
            "preprocessed media feature payload exceeds 16 MiB",
        ));
    }
    let data = decode_mm_kwargs(index, &raw, wire_nodes)?;
    validate_mm_kwargs_item(index, &data)?;
    let identifier = mm_cache_identifier(modality, &raw);

    Ok(MmFeatureSpec {
        data: Some(data),
        modality: modality.to_string(),
        identifier: identifier.clone(),
        mm_position: PlaceholderRange {
            offset,
            length,
            is_embed,
        },
        mm_hash: Some(identifier),
    })
}

fn validate_mm_kwargs_item(index: usize, kwargs: &MmKwargsItem) -> Result<(), Status> {
    if kwargs.is_empty() || kwargs.len() > MAX_MM_FIELDS_PER_ITEM {
        return Err(Status::invalid_argument(format!(
            "media[{index}].features.kwargs must contain between 1 and 256 fields"
        )));
    }
    for (name, element) in kwargs {
        if name.is_empty() || name.len() > MAX_MM_KEY_BYTES {
            return Err(Status::invalid_argument(format!(
                "media[{index}].features.kwargs keys must contain between 1 and 256 bytes"
            )));
        }
        let value = element.data.as_ref().ok_or_else(|| {
            Status::invalid_argument(format!(
                "media[{index}].features.kwargs[{name:?}] must contain inline data"
            ))
        })?;
        validate_mm_kwarg_value(index, name, value, 0)?;
    }
    Ok(())
}

fn validate_mm_kwarg_value(
    index: usize,
    name: &str,
    value: &MmKwargValue,
    depth: usize,
) -> Result<(), Status> {
    if depth > MAX_MM_DEPTH {
        return Err(Status::resource_exhausted(format!(
            "media[{index}].features.kwargs nesting exceeds 32 levels"
        )));
    }
    match value {
        MmKwargValue::Tensor(tensor) => validate_mm_tensor(index, name, tensor),
        MmKwargValue::List(values) => {
            for value in values {
                validate_mm_kwarg_value(index, name, value, depth + 1)?;
            }
            Ok(())
        }
        MmKwargValue::Int(_) | MmKwargValue::Float(_) => Ok(()),
    }
}

fn validate_mm_tensor(index: usize, name: &str, tensor: &WireTensor) -> Result<(), Status> {
    if tensor.shape.len() > MAX_MM_TENSOR_RANK {
        return Err(Status::invalid_argument(format!(
            "media[{index}].features.kwargs[{name:?}] tensor rank exceeds 32"
        )));
    }
    let width = match tensor.dtype.as_str() {
        "bool" | "uint8" | "int8" => 1,
        "float16" | "bfloat16" | "uint16" | "int16" => 2,
        "float32" | "uint32" | "int32" => 4,
        "float64" | "uint64" | "int64" => 8,
        dtype => {
            return Err(Status::invalid_argument(format!(
                "media[{index}].features.kwargs[{name:?}] has unsupported tensor dtype {dtype:?}"
            )));
        }
    };
    let numel = tensor
        .shape
        .iter()
        .try_fold(1usize, |count, dim| count.checked_mul(*dim))
        .ok_or_else(|| {
            Status::invalid_argument(format!(
                "media[{index}].features.kwargs[{name:?}] tensor shape overflows"
            ))
        })?;
    let expected = numel.checked_mul(width).ok_or_else(|| {
        Status::invalid_argument(format!(
            "media[{index}].features.kwargs[{name:?}] tensor byte length overflows"
        ))
    })?;
    match &tensor.data {
        WireArrayData::RawView(bytes) if bytes.len() == expected => Ok(()),
        WireArrayData::RawView(bytes) => Err(Status::invalid_argument(format!(
            "media[{index}].features.kwargs[{name:?}] tensor byte length {} does not match expected {expected}",
            bytes.len()
        ))),
        WireArrayData::AuxIndex(aux_index) => Err(Status::invalid_argument(format!(
            "media[{index}].features.kwargs[{name:?}] references auxiliary frame {aux_index}, but gRPC features must contain inline tensor data"
        ))),
    }
}

fn mm_cache_identifier(modality: &str, raw: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"vllm.grpc.preprocessed-mm.v1");
    hasher.update((modality.len() as u64).to_be_bytes());
    hasher.update(modality.as_bytes());
    hasher.update((raw.len() as u64).to_be_bytes());
    hasher.update(raw);
    format!("grpc-mm:{:x}", hasher.finalize())
}

fn decode_mm_kwargs(
    index: usize,
    raw: &[u8],
    wire_nodes: &mut usize,
) -> Result<MmKwargsItem, Status> {
    preflight_msgpack(raw, wire_nodes)?;
    let mut deserializer = rmp_serde::Deserializer::new(Cursor::new(raw));
    deserializer.set_max_depth(MAX_MM_DEPTH);
    let item = MmKwargsItem::deserialize(&mut deserializer).map_err(|error| {
        Status::invalid_argument(format!(
            "media[{index}].features.kwargs is invalid MessagePack: {error}"
        ))
    })?;
    if deserializer.position() != raw.len() as u64 {
        return Err(Status::invalid_argument(format!(
            "media[{index}].features.kwargs contains trailing MessagePack data"
        )));
    }
    Ok(item)
}

fn preflight_msgpack(raw: &[u8], wire_nodes: &mut usize) -> Result<(), Status> {
    let mut cursor = 0usize;
    scan_msgpack_value(raw, &mut cursor, 0, wire_nodes)?;
    if cursor != raw.len() {
        return Err(Status::invalid_argument(
            "preprocessed media kwargs contains trailing MessagePack data",
        ));
    }
    Ok(())
}

fn scan_msgpack_value(
    raw: &[u8],
    cursor: &mut usize,
    depth: usize,
    wire_nodes: &mut usize,
) -> Result<(), Status> {
    if depth > MAX_MM_DEPTH {
        return Err(Status::resource_exhausted(
            "preprocessed media kwargs nesting exceeds 32 levels",
        ));
    }
    *wire_nodes = wire_nodes
        .checked_add(1)
        .ok_or_else(|| Status::resource_exhausted("preprocessed media kwargs is too complex"))?;
    if *wire_nodes > MAX_MM_NODES {
        return Err(Status::resource_exhausted(
            "preprocessed media kwargs contains too many values",
        ));
    }
    let marker = take_msgpack(raw, cursor, 1)?[0];
    match marker {
        0x00..=0x7f | 0xc0 | 0xc2 | 0xc3 | 0xe0..=0xff => Ok(()),
        0x80..=0x8f => {
            scan_msgpack_children(raw, cursor, depth, wire_nodes, (marker & 0x0f) as usize * 2)
        }
        0x90..=0x9f => {
            scan_msgpack_children(raw, cursor, depth, wire_nodes, (marker & 0x0f) as usize)
        }
        0xa0..=0xbf => skip_msgpack(raw, cursor, (marker & 0x1f) as usize),
        0xc1 => Err(Status::invalid_argument("reserved MessagePack marker")),
        0xc4 => {
            let len = read_msgpack_u8(raw, cursor)? as usize;
            skip_msgpack(raw, cursor, len)
        }
        0xc5 => {
            let len = read_msgpack_u16(raw, cursor)? as usize;
            skip_msgpack(raw, cursor, len)
        }
        0xc6 => {
            let len = read_msgpack_u32(raw, cursor)? as usize;
            skip_msgpack(raw, cursor, len)
        }
        0xc7 => {
            let len = read_msgpack_u8(raw, cursor)? as usize;
            skip_msgpack(raw, cursor, len + 1)
        }
        0xc8 => {
            let len = read_msgpack_u16(raw, cursor)? as usize;
            skip_msgpack(raw, cursor, len + 1)
        }
        0xc9 => {
            let len = read_msgpack_u32(raw, cursor)? as usize;
            skip_msgpack(raw, cursor, len + 1)
        }
        0xca => skip_msgpack(raw, cursor, 4),
        0xcb => skip_msgpack(raw, cursor, 8),
        0xcc | 0xd0 => skip_msgpack(raw, cursor, 1),
        0xcd | 0xd1 => skip_msgpack(raw, cursor, 2),
        0xce | 0xd2 => skip_msgpack(raw, cursor, 4),
        0xcf | 0xd3 => skip_msgpack(raw, cursor, 8),
        0xd4 => skip_msgpack(raw, cursor, 2),
        0xd5 => skip_msgpack(raw, cursor, 3),
        0xd6 => skip_msgpack(raw, cursor, 5),
        0xd7 => skip_msgpack(raw, cursor, 9),
        0xd8 => skip_msgpack(raw, cursor, 17),
        0xd9 => {
            let len = read_msgpack_u8(raw, cursor)? as usize;
            skip_msgpack(raw, cursor, len)
        }
        0xda => {
            let len = read_msgpack_u16(raw, cursor)? as usize;
            skip_msgpack(raw, cursor, len)
        }
        0xdb => {
            let len = read_msgpack_u32(raw, cursor)? as usize;
            skip_msgpack(raw, cursor, len)
        }
        0xdc => {
            let count = read_msgpack_u16(raw, cursor)? as usize;
            scan_msgpack_children(raw, cursor, depth, wire_nodes, count)
        }
        0xdd => {
            let count = read_msgpack_u32(raw, cursor)? as usize;
            scan_msgpack_children(raw, cursor, depth, wire_nodes, count)
        }
        0xde => {
            let count = (read_msgpack_u16(raw, cursor)? as usize)
                .checked_mul(2)
                .ok_or_else(|| Status::resource_exhausted("MessagePack map is too large"))?;
            scan_msgpack_children(raw, cursor, depth, wire_nodes, count)
        }
        0xdf => {
            let count = (read_msgpack_u32(raw, cursor)? as usize)
                .checked_mul(2)
                .ok_or_else(|| Status::resource_exhausted("MessagePack map is too large"))?;
            scan_msgpack_children(raw, cursor, depth, wire_nodes, count)
        }
    }
}

fn scan_msgpack_children(
    raw: &[u8],
    cursor: &mut usize,
    depth: usize,
    wire_nodes: &mut usize,
    count: usize,
) -> Result<(), Status> {
    if count > MAX_MM_NODES.saturating_sub(*wire_nodes) {
        return Err(Status::resource_exhausted(
            "preprocessed media kwargs contains too many values",
        ));
    }
    for _ in 0..count {
        scan_msgpack_value(raw, cursor, depth + 1, wire_nodes)?;
    }
    Ok(())
}

fn take_msgpack<'a>(raw: &'a [u8], cursor: &mut usize, len: usize) -> Result<&'a [u8], Status> {
    let end = cursor.checked_add(len).filter(|end| *end <= raw.len()).ok_or_else(|| {
        Status::invalid_argument("truncated preprocessed media kwargs MessagePack")
    })?;
    let bytes = &raw[*cursor..end];
    *cursor = end;
    Ok(bytes)
}

fn skip_msgpack(raw: &[u8], cursor: &mut usize, len: usize) -> Result<(), Status> {
    take_msgpack(raw, cursor, len).map(|_| ())
}

fn read_msgpack_u8(raw: &[u8], cursor: &mut usize) -> Result<u8, Status> {
    Ok(take_msgpack(raw, cursor, 1)?[0])
}

fn read_msgpack_u16(raw: &[u8], cursor: &mut usize) -> Result<u16, Status> {
    Ok(u16::from_be_bytes(
        take_msgpack(raw, cursor, 2)?.try_into().expect("fixed two-byte slice"),
    ))
}

fn read_msgpack_u32(raw: &[u8], cursor: &mut usize) -> Result<u32, Status> {
    Ok(u32::from_be_bytes(
        take_msgpack(raw, cursor, 4)?.try_into().expect("fixed four-byte slice"),
    ))
}

fn validate_mm_field_metadata(features: &[MmFeatureSpec]) -> Result<(), Status> {
    let mut occurrences: HashMap<(String, String), usize> = HashMap::new();
    let mut fields: HashMap<(String, String), MmField> = HashMap::new();
    for feature in features {
        let item = feature.data.as_ref().expect("inline multimodal data is required above");
        for (key, element) in item {
            let identity = (feature.modality.clone(), key.clone());
            *occurrences.entry(identity.clone()).or_default() += 1;
            if let Some(previous) = fields.get(&identity) {
                if previous != &element.field {
                    return Err(Status::invalid_argument(
                        "multimodal field configuration differs across items",
                    ));
                }
            } else {
                fields.insert(identity, element.field.clone());
            }
        }
    }
    for feature in features {
        let item = feature.data.as_ref().expect("inline multimodal data is required above");
        for (key, element) in item {
            let count = occurrences[&(feature.modality.clone(), key.clone())];
            validate_mm_field(
                &element.field,
                element.data.as_ref().expect("inline multimodal field data is required above"),
                count,
            )?;
        }
    }
    Ok(())
}

fn validate_mm_field(
    field: &MmField,
    data: &MmKwargValue,
    occurrences: usize,
) -> Result<(), Status> {
    match field {
        MmField::Batched(_) => Ok(()),
        MmField::Shared(shared) => {
            if shared.batch_size == 0 || shared.batch_size != occurrences {
                return Err(Status::invalid_argument(
                    "multimodal shared-field batch_size must match item count",
                ));
            }
            Ok(())
        }
        MmField::Flat(flat) => {
            if flat.slices.is_empty() || flat.slices.len() != occurrences {
                return Err(Status::invalid_argument(
                    "multimodal flat-field slices must match item count",
                ));
            }
            for slice in &flat.slices {
                match slice {
                    MmSlice::Slice(slice) => validate_slice_step(slice.step)?,
                    MmSlice::Slices(slices) => {
                        if slices.is_empty() || slices.len() > MAX_MM_TENSOR_RANK {
                            return Err(Status::invalid_argument(
                                "multimodal flat-field slice tuple must contain 1 to 32 slices",
                            ));
                        }
                        for slice in slices {
                            validate_slice_step(slice.step)?;
                        }
                    }
                }
            }
            match data {
                MmKwargValue::Tensor(tensor) => {
                    let rank = i32::try_from(tensor.shape.len()).unwrap_or(i32::MAX);
                    if rank == 0 || flat.dim < -rank || flat.dim >= rank {
                        return Err(Status::invalid_argument(
                            "multimodal flat-field dim is outside the tensor rank",
                        ));
                    }
                }
                _ if flat.dim != 0 => {
                    return Err(Status::invalid_argument(
                        "multimodal non-tensor flat fields require dim=0",
                    ));
                }
                _ => {}
            }
            Ok(())
        }
    }
}

fn validate_slice_step(step: Option<isize>) -> Result<(), Status> {
    if step == Some(0) {
        return Err(Status::invalid_argument(
            "multimodal slice step must not be zero",
        ));
    }
    Ok(())
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

// ========================================================================================
// Request conversion
// ========================================================================================

/// Convert a gRPC `GenerateRequest` into the internal `TextRequest`.
///
/// If `req.model` is non-empty, it must match one of `served_model_names`;
/// otherwise the request is rejected with `NotFound`. An empty string is
/// treated as "unset" (proto3 default) and accepted.
pub fn to_text_request(
    req: pb::GenerateRequest,
    stream: bool,
    served_model_names: &[String],
) -> Result<TextRequest, Status> {
    if !req.model.is_empty() && !served_model_names.iter().any(|n| n == &req.model) {
        return Err(Status::not_found(format!(
            "model `{}` not found",
            req.model
        )));
    }

    if req.truncate_prompt_tokens != 0 {
        return Err(Status::invalid_argument(
            "truncate_prompt_tokens is not supported",
        ));
    }

    let prompt = match req.prompt {
        Some(pb::generate_request::Prompt::Text(text)) => Prompt::Text(text),
        Some(pb::generate_request::Prompt::TokenIds(ids)) => Prompt::TokenIds(ids.ids),
        None => return Err(Status::invalid_argument("prompt is required")),
    };

    let request_id = if req.request_id.is_empty() {
        Uuid::new_v4().to_string()
    } else {
        req.request_id
    };
    let session_id = req.session_id.filter(|s| !s.is_empty());

    let sampling = req.sampling.as_ref();
    let decoding = req.decoding.as_ref();
    let stopping = req.stopping.as_ref();
    let response = req.response.as_ref();
    let kv = req.kv.as_ref();

    let mut sampling_params =
        build_sampling_params(req.temperature, sampling, decoding, stopping, response)?;

    // Thread KVCacheParameters → SamplingParams fields.
    if let Some(kv) = kv {
        // Thread kv_transfer_params through vllm_xargs, matching the HTTP route
        // convention.
        if let Some(kv_struct) = kv.kv_transfer_params.as_ref() {
            let kv_json = proto_struct_to_json(kv_struct);
            let map = sampling_params.vllm_xargs.get_or_insert_with(Default::default);
            map.insert("kv_transfer_params".to_string(), kv_json);
        }
        if let Some(ec_struct) = kv.ec_transfer_params.as_ref() {
            let ec_json = proto_struct_to_json(ec_struct);
            let map = sampling_params.vllm_xargs.get_or_insert_with(Default::default);
            map.insert("ec_transfer_params".to_string(), ec_json);
        }
        if kv.bypass_prefix_cache {
            sampling_params.skip_reading_prefix_cache = Some(true);
        }
    }

    let decode_options = TextDecodeOptions {
        skip_special_tokens: response
            .and_then(|options| options.skip_special_tokens)
            .unwrap_or(true),
        include_stop_str_in_output: stopping.is_some_and(|s| s.include_stop_strings),
        stop_strings: stopping.map(|s| &s.stop_strings).filter(|ss| !ss.is_empty()).cloned(),
        min_tokens: stopping.map_or(0, |s| s.min_new_tokens),
    };

    Ok(TextRequest {
        request_id,
        prompt,
        mm_features: None,
        sampling_params,
        decode_options,
        intermediate: stream,
        priority: req.priority,
        cache_salt: kv.map(|k| &k.cache_salt).filter(|s| !s.is_empty()).cloned(),
        add_special_tokens: true,
        data_parallel_rank: None,
        session_id,
        reasoning_parser_kwargs: None,
        lora_request: None,
        arrival_time: None,
    })
}

fn build_sampling_params(
    temperature: Option<f32>,
    sampling: Option<&pb::RandomSampling>,
    decoding: Option<&pb::DecodingParameters>,
    stopping: Option<&pb::StoppingCriteria>,
    response: Option<&pb::ResponseOptions>,
) -> Result<SamplingParams, Status> {
    // Temperature is a top-level GenerateRequest field. Default to greedy (0.0) for
    // the gRPC API when the caller does not specify a value. This differs from
    // the HTTP/OpenAI API (which defaults to 1.0) and matches the convention of
    // programmatic generation APIs.
    let temperature = temperature.or(Some(0.0));
    let mut params = SamplingParams {
        temperature,
        ..SamplingParams::default()
    };

    // RandomSampling: for every remaining sampling field the protobuf default (`0`)
    // is treated as "unset" and leaves the resolved value to the lowering
    // stage, which falls back to the model-provided default or a
    // neutral/disabled value otherwise.
    if let Some(s) = sampling {
        // num_sequences (n > 1) is not supported yet by the TextLlm layer; the response
        // path also hardcodes SequenceOutput.index = 0, so accepting >1 would silently
        // truncate output cardinality. Reject explicitly.
        if s.num_sequences > 1 {
            return Err(Status::invalid_argument(
                "num_sequences > 1 is not supported",
            ));
        }
        if s.top_k != 0 {
            params.top_k = Some(s.top_k);
        }
        if s.top_p != 0.0 {
            params.top_p = Some(s.top_p);
        }
        if s.min_p != 0.0 {
            params.min_p = Some(s.min_p);
        }
        params.seed = s.seed;
    }

    // DecodingParameters
    if let Some(d) = decoding {
        if d.presence_penalty != 0.0 {
            params.presence_penalty = Some(d.presence_penalty);
        }
        if d.frequency_penalty != 0.0 {
            params.frequency_penalty = Some(d.frequency_penalty);
        }
        if d.repetition_penalty != 0.0 {
            params.repetition_penalty = Some(d.repetition_penalty);
        }
        if !d.logit_bias.is_empty() {
            params.logit_bias = Some(d.logit_bias.clone());
        }
        if !d.allowed_token_ids.is_empty() {
            params.allowed_token_ids = Some(d.allowed_token_ids.clone());
        }
        params.structured_outputs = convert_structured_output(d)?;
    }

    // StoppingCriteria
    if let Some(s) = stopping {
        if s.max_new_tokens != 0 {
            params.max_tokens = Some(s.max_new_tokens);
        }
        if s.min_new_tokens != 0 {
            params.min_tokens = Some(s.min_new_tokens);
        }
        if !s.stop_token_ids.is_empty() {
            params.stop_token_ids = Some(s.stop_token_ids.clone());
        }
        params.ignore_eos = s.ignore_eos;
    }

    // ResponseOptions → logprobs
    if let Some(r) = response {
        if r.output_logprobs {
            let (count, token_ids) = candidate_logprob_spec(r.output_candidates.as_ref());
            params.logprobs = Some(count);
            params.logprob_token_ids = token_ids;
        }
        if r.prompt_logprobs {
            // The engine-core protocol has only one shared `logprob_token_ids` field
            // for output and prompt logprobs, so a per-token-id selector for prompt
            // candidates can't be honored independently. Reject it instead of silently
            // dropping the list.
            if matches!(
                r.prompt_candidates.as_ref().and_then(|c| c.select.as_ref()),
                Some(pb::candidate_tokens::Select::TokenIds(_))
            ) {
                return Err(Status::invalid_argument(
                    "prompt_candidates token_ids selector is not supported",
                ));
            }
            let (count, _) = candidate_logprob_spec(r.prompt_candidates.as_ref());
            params.prompt_logprobs = Some(count);
        }
    }

    Ok(params)
}

/// Map the proto `CandidateTokens` selector to a `(logprobs_count,
/// logprob_token_ids)` pair.
///
/// - `top_n(k)` → `(k, None)` — return top-k candidates by probability
/// - `all` → `(-1, None)` — return the full vocabulary
/// - `token_ids(n)` → `(1, Some(vec of n token ids))` — return logprobs for specific tokens (the
///   count `n` is stored in the proto as the number of token IDs that follow, but the actual IDs
///   are carried via `logprob_token_ids` on `SamplingParams`)
/// - absent → `(1, None)` — just the sampled/scored token
fn candidate_logprob_spec(candidates: Option<&pb::CandidateTokens>) -> (i32, Option<Vec<u32>>) {
    match candidates.and_then(|c| c.select.as_ref()) {
        Some(pb::candidate_tokens::Select::TopN(n)) => (*n as i32, None),
        Some(pb::candidate_tokens::Select::All(true)) => (-1, None),
        Some(pb::candidate_tokens::Select::TokenIds(ids)) => (1, Some(ids.ids.clone())),
        _ => (1, None),
    }
}

fn convert_structured_output(
    d: &pb::DecodingParameters,
) -> Result<Option<StructuredOutputsParams>, Status> {
    let so = match d.structured_output.as_ref() {
        None => return Ok(None),
        Some(so) => so,
    };
    use pb::decoding_parameters::StructuredOutput;
    let params = match so {
        StructuredOutput::Json(schema) => {
            let json: serde_json::Value = serde_json::from_str(schema)
                .map_err(|e| Status::invalid_argument(format!("invalid json schema: {e}")))?;
            StructuredOutputsParams::json(json)
        }
        StructuredOutput::Regex(regex) => StructuredOutputsParams::regex(regex.clone()),
        StructuredOutput::Choice(choices) => {
            StructuredOutputsParams::choice(choices.choices.clone())
        }
        StructuredOutput::Grammar(grammar) => StructuredOutputsParams::grammar(grammar.clone()),
        StructuredOutput::JsonObject(true) => StructuredOutputsParams::json_object(),
        StructuredOutput::JsonObject(false) => return Ok(None),
        StructuredOutput::StructuralTag(tag) => {
            StructuredOutputsParams::structural_tag(tag.clone())
        }
    };
    Ok(Some(params))
}

// ========================================================================================
// Response conversion
// ========================================================================================

/// Convert a `DecodedTextEvent::Start` into the prompt info portion of a gRPC
/// response.
pub fn to_prompt_info(
    prompt_token_ids: &[u32],
    prompt_logprobs: Option<&DecodedPromptLogprobs>,
    opts: &ResponseOpts,
) -> pb::PromptInfo {
    let token_ids = if opts.prompt_token_ids {
        prompt_token_ids.to_vec()
    } else {
        vec![]
    };

    let (logprobs, ranks, candidate_tokens) = match prompt_logprobs {
        Some(plp) if opts.prompt_logprobs => prompt_logprobs_to_proto(plp),
        _ => (vec![], vec![], vec![]),
    };

    pb::PromptInfo {
        num_prompt_tokens: prompt_token_ids.len() as u32,
        token_ids,
        logprobs,
        ranks,
        candidate_tokens,
    }
}

/// Convert a `DecodedTextEvent::TextDelta` into a gRPC `SequenceOutput`.
pub fn to_sequence_output(
    delta: &str,
    token_ids: &[u32],
    logprobs: Option<&DecodedLogprobs>,
    finished: Option<&Finished>,
    opts: &ResponseOpts,
) -> pb::SequenceOutput {
    let (lp_values, rank_values, candidates) = match logprobs {
        Some(lp) if opts.output_logprobs => output_logprobs_to_proto(lp),
        _ => (vec![], vec![], vec![]),
    };

    pb::SequenceOutput {
        index: 0, // TODO: multi-sequence (n > 1) not supported
        text: if opts.output_text {
            delta.to_string()
        } else {
            String::new()
        },
        num_tokens: token_ids.len() as u32,
        token_ids: if opts.output_token_ids {
            token_ids.to_vec()
        } else {
            vec![]
        },
        logprobs: lp_values,
        ranks: rank_values,
        candidate_tokens: candidates,
        finish_info: finished.map(|f| to_finish_info(f, token_ids)),
    }
}

fn to_finish_info(finished: &Finished, token_ids: &[u32]) -> pb::FinishInfo {
    use pb::finish_info::FinishReason as PbFinishReason;

    let (finish_reason, stop_reason) = match &finished.finish_reason {
        FinishReason::Stop(reason) => {
            let sr = match reason {
                Some(StopReason::TokenId(id)) => {
                    Some(pb::finish_info::StopReason::StopTokenId(*id))
                }
                Some(StopReason::Text(s)) => {
                    Some(pb::finish_info::StopReason::StopString(s.clone()))
                }
                // EOS-driven stop: engine-core matched the primary EOS token id but did not
                // echo it back as a `stop_reason`. The matched token is, by construction, the
                // last token of the terminal output batch (see vllm's `check_stop` in
                // vllm/v1/core/sched/utils.py), so we recover it from there.
                None => token_ids.last().copied().map(pb::finish_info::StopReason::EosTokenId),
            };
            (PbFinishReason::Stop as i32, sr)
        }
        FinishReason::Length => (PbFinishReason::Length as i32, None),
        FinishReason::Abort | FinishReason::Error | FinishReason::Repetition(_) => {
            (PbFinishReason::Aborted as i32, None)
        }
    };

    pb::FinishInfo {
        num_output_tokens: finished.usage.output_token_count as u32,
        finish_reason,
        stop_reason,
        kv_transfer_params: finished.kv_transfer_params.as_ref().and_then(json_to_proto_struct),
        ec_transfer_params: finished.ec_transfer_params.as_ref().and_then(json_to_proto_struct),
    }
}

// ========================================================================================
// Logprobs helpers
// ========================================================================================

/// Convert output logprobs to the flat proto representation.
///
/// Returns (logprob_values, ranks, candidate_tokens) — all parallel arrays
/// indexed by position.
fn output_logprobs_to_proto(
    lp: &DecodedLogprobs,
) -> (Vec<f32>, Vec<u32>, Vec<pb::CandidateTokenInfo>) {
    positions_to_proto(&lp.positions)
}

/// Convert prompt logprobs to the flat proto representation.
fn prompt_logprobs_to_proto(
    plp: &DecodedPromptLogprobs,
) -> (Vec<f32>, Vec<u32>, Vec<pb::CandidateTokenInfo>) {
    // The proto PromptInfo has flat parallel arrays covering all prompt positions.
    // DecodedPromptLogprobs has first_token separately + scored_positions for the
    // rest. The first prompt position has no scores, so we emit zeros for it.
    let (mut logprobs, mut ranks, mut candidates) = positions_to_proto(&plp.scored_positions);
    logprobs.insert(0, 0.0);
    ranks.insert(0, 0);
    candidates.insert(0, pb::CandidateTokenInfo { tokens: vec![] });
    (logprobs, ranks, candidates)
}

/// Shared helper: convert a slice of decoded position logprobs to flat proto
/// arrays.
fn positions_to_proto(
    positions: &[vllm_text::DecodedPositionLogprobs],
) -> (Vec<f32>, Vec<u32>, Vec<pb::CandidateTokenInfo>) {
    let mut logprobs = Vec::with_capacity(positions.len());
    let mut ranks = Vec::with_capacity(positions.len());
    let mut candidates = Vec::with_capacity(positions.len());

    for pos in positions {
        // First entry is the sampled/scored token.
        if let Some(first) = pos.entries.first() {
            logprobs.push(first.logprob);
            ranks.push(first.rank);
        }

        // Extra candidates beyond the first.
        let entries = pos.entries.iter().skip(1);
        candidates.push(pb::CandidateTokenInfo {
            tokens: entries
                .map(|e| pb::candidate_token_info::TokenInfo {
                    id: e.token_id,
                    logprob: e.logprob,
                    rank: e.rank,
                })
                .collect(),
        });
    }

    (logprobs, ranks, candidates)
}

// ========================================================================================
// KV transfer params conversion (serde_json::Value ↔ prost_types::Struct)
// ========================================================================================

fn proto_struct_to_json(s: &prost_types::Struct) -> serde_json::Value {
    serde_json::Value::Object(
        s.fields.iter().map(|(k, v)| (k.clone(), proto_value_to_json(v))).collect(),
    )
}

fn proto_value_to_json(v: &prost_types::Value) -> serde_json::Value {
    use prost_types::value::Kind;
    match v.kind.as_ref() {
        None | Some(Kind::NullValue(_)) => serde_json::Value::Null,
        Some(Kind::BoolValue(b)) => serde_json::Value::Bool(*b),
        Some(Kind::NumberValue(n)) => serde_json::json!(*n),
        Some(Kind::StringValue(s)) => serde_json::Value::String(s.clone()),
        Some(Kind::ListValue(list)) => {
            serde_json::Value::Array(list.values.iter().map(proto_value_to_json).collect())
        }
        Some(Kind::StructValue(s)) => proto_struct_to_json(s),
    }
}

fn json_to_proto_struct(value: &serde_json::Value) -> Option<prost_types::Struct> {
    match value {
        serde_json::Value::Object(map) => Some(prost_types::Struct {
            fields: map.iter().map(|(k, v)| (k.clone(), json_to_proto_value(v))).collect(),
        }),
        _ => None,
    }
}

fn json_to_proto_value(v: &serde_json::Value) -> prost_types::Value {
    use prost_types::value::Kind;
    let kind = match v {
        serde_json::Value::Null => Kind::NullValue(0),
        serde_json::Value::Bool(b) => Kind::BoolValue(*b),
        serde_json::Value::Number(n) => Kind::NumberValue(n.as_f64().unwrap_or(0.0)),
        serde_json::Value::String(s) => Kind::StringValue(s.clone()),
        serde_json::Value::Array(arr) => Kind::ListValue(prost_types::ListValue {
            values: arr.iter().map(json_to_proto_value).collect(),
        }),
        serde_json::Value::Object(map) => Kind::StructValue(prost_types::Struct {
            fields: map.iter().map(|(k, v)| (k.clone(), json_to_proto_value(v))).collect(),
        }),
    };
    prost_types::Value { kind: Some(kind) }
}

// ========================================================================================
// Options extracted from the request for response building
// ========================================================================================

/// Response-shaping options extracted from the proto `ResponseOptions`.
#[derive(Default)]
pub struct ResponseOpts {
    pub prompt_token_ids: bool,
    pub prompt_logprobs: bool,
    pub output_text: bool,
    pub output_token_ids: bool,
    pub output_logprobs: bool,
}

impl ResponseOpts {
    pub fn from_proto(r: Option<&pb::ResponseOptions>) -> Self {
        match r {
            Some(r) => Self {
                prompt_token_ids: r.prompt_token_ids,
                prompt_logprobs: r.prompt_logprobs,
                output_text: r.output_text.unwrap_or(true),
                output_token_ids: r.output_token_ids,
                output_logprobs: r.output_logprobs,
            },
            None => Self {
                output_text: true,
                ..Default::default()
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use bytes::Bytes;
    use vllm_engine_core_client::protocol::multimodal::{
        MmBatchedField, MmField, MmFieldElem, MmFlatField, MmKwargValue, MmSlice, SliceSpec,
    };
    use vllm_engine_core_client::protocol::output::StopReason;
    use vllm_engine_core_client::protocol::tensor::{WireArrayData, WireTensor};
    use vllm_text::{FinishReason, Finished, Prompt};

    use super::pb::finish_info::{FinishReason as PbFinishReason, StopReason as PbStopReason};
    use super::{
        ResponseOpts, multimodal_input_from_request, pb, to_finish_info, to_sequence_output,
        to_text_request,
    };

    fn base_request() -> pb::GenerateRequest {
        pb::GenerateRequest {
            request_id: "req".to_string(),
            model: "test-model".to_string(),
            prompt: Some(pb::generate_request::Prompt::Text("hi".to_string())),
            ..Default::default()
        }
    }

    fn preprocessed_media(
        identifier: &str,
        offset: u64,
        length: u64,
        kwargs: Option<Vec<u8>>,
    ) -> pb::MediaItem {
        pb::MediaItem {
            modality: pb::Modality::Image as i32,
            source: Some(pb::media_item::Source::Features(
                pb::PreprocessedMediaFeatures {
                    kwargs,
                    identifier: identifier.to_string(),
                    offset,
                    length,
                    mm_hash: None,
                    is_embed: Vec::new(),
                },
            )),
            mime_type: String::new(),
            uuid: String::new(),
        }
    }

    fn encoded_inline_kwargs() -> Vec<u8> {
        let kwargs = BTreeMap::from([(
            "pixel_values".to_string(),
            MmFieldElem {
                data: Some(MmKwargValue::Tensor(WireTensor::from_raw(
                    "uint8",
                    vec![1],
                    vec![7],
                ))),
                field: MmField::Batched(MmBatchedField { keep_on_cpu: false }),
            },
        )]);
        rmp_serde::to_vec_named(&kwargs).expect("encode multimodal kwargs")
    }

    #[test]
    fn preprocessed_features_require_inline_kwargs() {
        let error =
            match multimodal_input_from_request(vec![preprocessed_media("image-a", 0, 1, None)]) {
                Err(error) => error,
                Ok(_) => panic!("cache-only feature must be rejected"),
            };

        assert!(error.message().contains("cache-only features are unsupported"));
    }

    #[test]
    fn preprocessed_features_reject_auxiliary_frame_references() {
        let kwargs = BTreeMap::from([(
            "pixel_values".to_string(),
            MmFieldElem {
                data: Some(MmKwargValue::Tensor(WireTensor {
                    dtype: "uint8".to_string(),
                    shape: vec![1],
                    data: WireArrayData::AuxIndex(1),
                })),
                field: MmField::Batched(MmBatchedField { keep_on_cpu: false }),
            },
        )]);
        let encoded = rmp_serde::to_vec_named(&kwargs).expect("encode multimodal kwargs");

        let error = match multimodal_input_from_request(vec![preprocessed_media(
            "image-a",
            0,
            1,
            Some(encoded),
        )]) {
            Err(error) => error,
            Ok(_) => panic!("auxiliary frame reference must be rejected"),
        };

        assert!(error.message().contains("references auxiliary frame 1"));
    }

    #[test]
    fn preprocessed_features_reject_overlapping_placeholder_ranges() {
        let media = vec![
            preprocessed_media("image-a", 0, 2, Some(encoded_inline_kwargs())),
            preprocessed_media("image-b", 1, 2, Some(encoded_inline_kwargs())),
        ];

        let error = match multimodal_input_from_request(media) {
            Err(error) => error,
            Ok(_) => panic!("overlapping placeholder ranges must be rejected"),
        };

        assert!(error.message().contains("cannot overlap"));
    }

    #[test]
    fn preprocessed_features_reject_malformed_tensor_lengths() {
        let kwargs = BTreeMap::from([(
            "pixel_values".to_string(),
            MmFieldElem {
                data: Some(MmKwargValue::Tensor(WireTensor {
                    dtype: "uint16".to_string(),
                    shape: vec![2],
                    data: WireArrayData::RawView(Bytes::from_static(&[1])),
                })),
                field: MmField::Batched(MmBatchedField { keep_on_cpu: false }),
            },
        )]);
        let encoded = rmp_serde::to_vec_named(&kwargs).expect("encode multimodal kwargs");

        let error = match multimodal_input_from_request(vec![preprocessed_media(
            "image-a",
            0,
            1,
            Some(encoded),
        )]) {
            Err(error) => error,
            Ok(_) => panic!("malformed tensor length must be rejected"),
        };

        assert!(error.message().contains("tensor byte length 1 does not match expected 4"));
    }

    #[test]
    fn preprocessed_features_reject_excessive_nesting() {
        let mut value = MmKwargValue::Int(1);
        for _ in 0..40 {
            value = MmKwargValue::List(vec![value]);
        }
        let kwargs = BTreeMap::from([(
            "pixel_values".to_string(),
            MmFieldElem {
                data: Some(value),
                field: MmField::Batched(MmBatchedField { keep_on_cpu: false }),
            },
        )]);
        let encoded = rmp_serde::to_vec_named(&kwargs).expect("encode multimodal kwargs");

        let error = match multimodal_input_from_request(vec![preprocessed_media(
            "image-a",
            0,
            1,
            Some(encoded),
        )]) {
            Err(error) => error,
            Ok(_) => panic!("excessive MessagePack nesting must be rejected"),
        };

        assert_eq!(error.code(), tonic::Code::ResourceExhausted);
    }

    #[test]
    fn preprocessed_features_reject_invalid_flat_field_metadata() {
        let kwargs = BTreeMap::from([(
            "pixel_values".to_string(),
            MmFieldElem {
                data: Some(MmKwargValue::Tensor(WireTensor::from_raw(
                    "uint8",
                    vec![1],
                    vec![7],
                ))),
                field: MmField::Flat(MmFlatField {
                    slices: vec![MmSlice::Slice(SliceSpec {
                        start: Some(0),
                        stop: Some(1),
                        step: Some(0),
                    })],
                    dim: 2,
                    keep_on_cpu: false,
                }),
            },
        )]);
        let encoded = rmp_serde::to_vec_named(&kwargs).expect("encode multimodal kwargs");

        let error = match multimodal_input_from_request(vec![preprocessed_media(
            "image-a",
            0,
            1,
            Some(encoded),
        )]) {
            Err(error) => error,
            Ok(_) => panic!("invalid flat-field metadata must be rejected"),
        };

        assert!(error.message().contains("slice step must not be zero"));
    }

    #[test]
    fn temperature_propagates_from_top_level_request_field() {
        let req = pb::GenerateRequest {
            temperature: Some(0.7),
            ..base_request()
        };
        let text = to_text_request(req, false, &["test-model".to_string()]).expect("convert ok");
        assert_eq!(text.sampling_params.temperature, Some(0.7));
    }

    #[test]
    fn unset_temperature_defaults_to_greedy() {
        let text = to_text_request(base_request(), false, &["test-model".to_string()])
            .expect("convert ok");
        // The gRPC API defaults to greedy (0.0) when temperature is not specified.
        assert_eq!(text.sampling_params.temperature, Some(0.0));
    }

    #[test]
    fn absent_seed_is_none() {
        let req = pb::GenerateRequest {
            sampling: Some(pb::RandomSampling {
                seed: None,
                ..Default::default()
            }),
            ..base_request()
        };
        let text = to_text_request(req, false, &["test-model".to_string()]).expect("convert ok");
        assert_eq!(text.sampling_params.seed, None);
    }

    #[test]
    fn zero_seed_is_valid() {
        let req = pb::GenerateRequest {
            sampling: Some(pb::RandomSampling {
                seed: Some(0),
                ..Default::default()
            }),
            ..base_request()
        };
        let text = to_text_request(req, false, &["test-model".to_string()]).expect("convert ok");
        assert_eq!(text.sampling_params.seed, Some(0));
    }

    #[test]
    fn bypass_prefix_cache_maps_to_skip_reading_prefix_cache() {
        let req = pb::GenerateRequest {
            kv: Some(pb::KvCacheParameters {
                bypass_prefix_cache: true,
                ..Default::default()
            }),
            ..base_request()
        };
        let text = to_text_request(req, false, &["test-model".to_string()]).expect("convert ok");
        assert_eq!(text.sampling_params.skip_reading_prefix_cache, Some(true));
    }

    #[test]
    fn bypass_prefix_cache_false_leaves_field_unset() {
        let req = pb::GenerateRequest {
            kv: Some(pb::KvCacheParameters {
                bypass_prefix_cache: false,
                ..Default::default()
            }),
            ..base_request()
        };
        let text = to_text_request(req, false, &["test-model".to_string()]).expect("convert ok");
        assert_eq!(text.sampling_params.skip_reading_prefix_cache, None);
        // Prompt conversion still succeeds and reaches the expected variant.
        assert!(matches!(text.prompt, Prompt::Text(s) if s == "hi"));
    }

    fn finished(reason: FinishReason) -> Finished {
        Finished {
            usage: vllm_llm::TokenUsage {
                prompt_token_count: 0,
                output_token_count: 0,
                cached_token_count: 0,
            },
            finish_reason: reason,
            kv_transfer_params: None,
            ec_transfer_params: None,
        }
    }

    #[test]
    fn eos_stop_reports_last_output_token_as_eos_id() {
        let fin = finished(FinishReason::Stop(None));
        let token_ids = [1_u32, 2, 3, 151643];

        let info = to_finish_info(&fin, &token_ids);

        assert_eq!(info.finish_reason, PbFinishReason::Stop as i32);
        assert_eq!(info.stop_reason, Some(PbStopReason::EosTokenId(151643)));
    }

    #[test]
    fn eos_stop_with_empty_token_ids_leaves_stop_reason_unset() {
        let fin = finished(FinishReason::Stop(None));

        let info = to_finish_info(&fin, &[]);

        assert_eq!(info.finish_reason, PbFinishReason::Stop as i32);
        assert_eq!(info.stop_reason, None);
    }

    #[test]
    fn explicit_stop_token_id_is_preserved() {
        let fin = finished(FinishReason::Stop(Some(StopReason::TokenId(42))));
        // Terminal token list should be ignored when an explicit stop reason is
        // present.
        let info = to_finish_info(&fin, &[7, 42]);

        assert_eq!(info.finish_reason, PbFinishReason::Stop as i32);
        assert_eq!(info.stop_reason, Some(PbStopReason::StopTokenId(42)));
    }

    #[test]
    fn explicit_stop_string_is_preserved() {
        let fin = finished(FinishReason::Stop(Some(StopReason::Text("</stop>".into()))));

        let info = to_finish_info(&fin, &[1, 2, 3]);

        assert_eq!(info.finish_reason, PbFinishReason::Stop as i32);
        assert_eq!(
            info.stop_reason,
            Some(PbStopReason::StopString("</stop>".into()))
        );
    }

    #[test]
    fn length_finish_has_no_stop_reason() {
        let fin = finished(FinishReason::Length);

        let info = to_finish_info(&fin, &[1, 2, 3]);

        assert_eq!(info.finish_reason, PbFinishReason::Length as i32);
        assert_eq!(info.stop_reason, None);
    }

    #[test]
    fn abort_finish_is_mapped_to_aborted() {
        let fin = finished(FinishReason::Abort);

        let info = to_finish_info(&fin, &[]);

        assert_eq!(info.finish_reason, PbFinishReason::Aborted as i32);
        assert_eq!(info.stop_reason, None);
    }

    #[test]
    fn to_sequence_output_threads_token_ids_into_eos_id() {
        let fin = finished(FinishReason::Stop(None));
        let opts = ResponseOpts {
            output_text: true,
            output_token_ids: true,
            ..Default::default()
        };

        let out = to_sequence_output("hello", &[10, 20, 30], None, Some(&fin), &opts);

        let finish = out.finish_info.expect("finish_info should be present");
        assert_eq!(finish.finish_reason, PbFinishReason::Stop as i32);
        assert_eq!(finish.stop_reason, Some(PbStopReason::EosTokenId(30)));
    }
}
