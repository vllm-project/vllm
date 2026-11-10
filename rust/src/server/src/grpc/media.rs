// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Protobuf media adaptation into the shared multimodal preparation boundary.

use std::io::Cursor;

use llm_multimodal::tracker::{MAX_SAMPLE_FPS, MIN_SAMPLE_FPS};
use serde::Deserialize;
use thiserror_ext::AsReport as _;
use tonic::Status;
use url::Url;
use vllm_chat::MediaContentPart;
use vllm_chat::multimodal::MultimodalInput;
use vllm_engine_core_client::protocol::multimodal::{
    MmFeatureSpec, MmKwargsItem, MmModality, PlaceholderRange,
};
use vllm_engine_core_client::protocol::tensor::WireTensor;

use super::convert::proto_struct_to_json;
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
    media_io_kwargs: Option<&prost_types::Struct>,
) -> Result<MultimodalInput, Status> {
    let video_fps = video_fps(media_io_kwargs)?;
    if !media
        .iter()
        .any(|item| matches!(item.source, Some(pb::media_item::Source::Features(_))))
    {
        return raw_parts_from_proto(media, video_fps).map(MultimodalInput::Raw);
    }
    if video_fps.is_some() {
        return Err(Status::invalid_argument(
            "media_io_kwargs cannot change preprocessed media features",
        ));
    }
    let mut encoded_bytes = 0usize;
    for item in &media {
        let Some(pb::media_item::Source::Features(feature)) = &item.source else {
            return Err(mixed_media_error());
        };
        encoded_bytes = encoded_bytes.saturating_add(feature.kwargs.as_ref().map_or(0, Vec::len));
    }
    if encoded_bytes > MAX_INLINE_MM_BYTES {
        return Err(Status::resource_exhausted(format!(
            "inline multimodal payload exceeds {MAX_INLINE_MM_BYTES} bytes"
        )));
    }
    let features = media
        .into_iter()
        .enumerate()
        .map(|(index, item)| {
            feature_from_proto(item).map_err(|error| {
                Status::new(error.code(), format!("media[{index}]: {}", error.message()))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(MultimodalInput::Preprocessed(features))
}

fn feature_from_proto(item: pb::MediaItem) -> Result<MmFeatureSpec, Status> {
    let modality = match item.modality() {
        pb::Modality::Image => MmModality::Image,
        pb::Modality::Video => MmModality::Video,
        pb::Modality::Audio => MmModality::Audio,
        pb::Modality::Unspecified => return Err(Status::invalid_argument("modality is required")),
    };
    let Some(pb::media_item::Source::Features(feature)) = item.source else {
        return Err(mixed_media_error());
    };
    let bytes = feature.kwargs.ok_or_else(|| {
        Status::invalid_argument("features.kwargs is required; cache-only features are unsupported")
    })?;
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
        data: Some(decode_inline_mm_kwargs(&bytes)?),
        identifier: feature.identifier,
        mm_hash: feature.mm_hash,
        mm_position: PlaceholderRange {
            offset,
            length,
            is_embed,
        },
    })
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

/// The `media_io_kwargs` options the Rust media loader applies; any other
/// modality or option is rejected rather than ignored.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MediaIoKwargs {
    #[serde(default)]
    video: VideoIoKwargs,
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct VideoIoKwargs {
    /// Frames sampled per second.
    fps: Option<f64>,
}

fn video_fps(media_io_kwargs: Option<&prost_types::Struct>) -> Result<Option<f64>, Status> {
    let Some(kwargs) = media_io_kwargs else {
        return Ok(None);
    };
    let kwargs = proto_struct_to_json(kwargs);
    // A derived struct also deserializes from a positional array, which would
    // read `{"video": [1]}` as fps 1.
    if kwargs.get("video").is_some_and(|video| !video.is_object()) {
        return Err(Status::invalid_argument(
            "invalid media_io_kwargs: video must be a map of options",
        ));
    }
    let kwargs = MediaIoKwargs::deserialize(kwargs).map_err(|error| {
        Status::invalid_argument(format!(
            "invalid media_io_kwargs: {}; only video.fps is supported",
            error.as_report()
        ))
    })?;
    let fps = kwargs.video.fps;
    if let Some(fps) = fps
        && !(MIN_SAMPLE_FPS..=MAX_SAMPLE_FPS).contains(&fps)
    {
        return Err(Status::invalid_argument(format!(
            "invalid media_io_kwargs.video.fps: {fps} is outside [{MIN_SAMPLE_FPS}, {MAX_SAMPLE_FPS}]"
        )));
    }
    Ok(fps)
}

fn mixed_media_error() -> Status {
    Status::invalid_argument("raw media and preprocessed media features cannot be mixed")
}

fn raw_parts_from_proto(
    media: Vec<pb::MediaItem>,
    video_fps: Option<f64>,
) -> Result<Vec<MediaContentPart>, Status> {
    let mut parts = Vec::with_capacity(media.len());
    for (index, item) in media.into_iter().enumerate() {
        let modality = item.modality();
        if modality == pb::Modality::Unspecified {
            return Err(Status::invalid_argument(format!(
                "media[{index}].modality is required"
            )));
        }
        let uuid = (!item.uuid.is_empty()).then_some(item.uuid);
        let mime_type = (!item.mime_type.is_empty()).then_some(item.mime_type);
        let source = item.source.ok_or_else(|| {
            Status::invalid_argument(format!("media[{index}].source is required"))
        })?;
        match &source {
            pb::media_item::Source::Url(url) => {
                validate_media_uri(index, "url", url, &["http", "https"])?;
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
                detail: None,
                uuid,
                max_long_side_pixel: None,
            },
            (pb::Modality::Image, pb::media_item::Source::RawBytes(data)) => {
                MediaContentPart::ImageData {
                    data,
                    mime_type,
                    uuid,
                    detail: None,
                }
            }
            (
                pb::Modality::Video,
                pb::media_item::Source::Url(url) | pb::media_item::Source::DataUri(url),
            ) => MediaContentPart::VideoUrl {
                url,
                uuid,
                fps: video_fps,
                max_long_side_pixel: None,
            },
            (pb::Modality::Video, pb::media_item::Source::RawBytes(_)) if video_fps.is_some() => {
                return Err(Status::invalid_argument(format!(
                    "media[{index}]: media_io_kwargs.video.fps requires a url or data_uri source"
                )));
            }
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
        parts.push(part);
    }
    Ok(parts)
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

    use super::super::convert::json_to_proto_struct;
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
    fn mixed_media_is_rejected_in_both_orders() {
        let raw = pb::MediaItem {
            modality: pb::Modality::Image as i32,
            source: Some(pb::media_item::Source::RawBytes(vec![1])),
            ..Default::default()
        };
        for media in [vec![raw.clone(), preprocessed()], vec![preprocessed(), raw]] {
            assert_eq!(
                from_proto(media, None).unwrap_err().code(),
                tonic::Code::InvalidArgument
            );
        }
    }

    fn video(source: pb::media_item::Source) -> pb::MediaItem {
        pb::MediaItem {
            modality: pb::Modality::Video as i32,
            source: Some(source),
            ..Default::default()
        }
    }

    #[test]
    fn media_io_kwargs_video_fps_sets_the_video_url_sampling_rate() {
        let url = video(pb::media_item::Source::Url(
            "https://example.com/clip.mp4".to_owned(),
        ));
        let kwargs = json_to_proto_struct(&serde_json::json!({"video": {"fps": 1}}));
        let MultimodalInput::Raw(parts) = from_proto(vec![url], kwargs.as_ref()).unwrap() else {
            unreachable!()
        };
        expect_test::expect![[r#"
            [
                VideoUrl {
                    url: "https://example.com/clip.mp4",
                    uuid: None,
                    fps: Some(
                        1.0,
                    ),
                    max_long_side_pixel: None,
                },
            ]
        "#]]
        .assert_debug_eq(&parts);
    }

    #[test]
    fn media_io_kwargs_that_cannot_apply_are_rejected() {
        let url = video(pb::media_item::Source::Url(
            "https://example.com/clip.mp4".to_owned(),
        ));
        let cases = [
            (url.clone(), serde_json::json!({"video": {"num_frames": 8}})),
            (url.clone(), serde_json::json!({"image": {}})),
            (url.clone(), serde_json::json!({"video": {"fps": 30}})),
            (url.clone(), serde_json::json!({"video": {"fps": "1"}})),
            (url, serde_json::json!({"video": [1]})),
            (
                video(pb::media_item::Source::RawBytes(vec![1])),
                serde_json::json!({"video": {"fps": 1}}),
            ),
            (preprocessed(), serde_json::json!({"video": {"fps": 1}})),
        ];
        let messages = cases
            .into_iter()
            .map(|(item, kwargs)| {
                let kwargs = json_to_proto_struct(&kwargs);
                let error = from_proto(vec![item], kwargs.as_ref()).unwrap_err();
                assert_eq!(error.code(), tonic::Code::InvalidArgument);
                error.message().to_owned()
            })
            .collect::<Vec<_>>();
        expect_test::expect![[r#"
            [
                "invalid media_io_kwargs: unknown field `num_frames`, expected `fps`; only video.fps is supported",
                "invalid media_io_kwargs: unknown field `image`, expected `video`; only video.fps is supported",
                "invalid media_io_kwargs.video.fps: 30 is outside [0.2, 5]",
                "invalid media_io_kwargs: invalid type: string \"1\", expected f64; only video.fps is supported",
                "invalid media_io_kwargs: video must be a map of options",
                "media[0]: media_io_kwargs.video.fps requires a url or data_uri source",
                "media_io_kwargs cannot change preprocessed media features",
            ]
        "#]].assert_debug_eq(&messages);
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
            let error = from_proto(vec![media], None).unwrap_err();
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
        let error = from_proto(vec![item.clone(), item], None).unwrap_err();
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
}
