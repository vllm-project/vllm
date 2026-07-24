// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::{BTreeMap, HashMap};

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_default::DefaultFromSerde;
use serde_tuple::{Deserialize_tuple, Serialize_tuple};

use crate::protocol::multimodal::MmFeatures;
use crate::protocol::sampling::EngineCoreSamplingParams;
use crate::protocol::{OpaqueValue, lora};
use crate::{Error, Result};

/// Match Python's default threshold for moving tensor data out of msgpack.
pub const DEFAULT_AUX_FRAME_THRESHOLD: usize = 256;

/// Request types are encoded as single-byte protocol constants so they can be
/// sent over the ZMQ socket without an extra encoding step.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/f22d6e026798a74e6542a52ef776c054f2de572a/vllm/v1/engine/__init__.py#L217-L228>
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EngineCoreRequestType {
    Add = 0,
    Abort = 1,
    StartDpWave = 2,
    Utility = 3,
}

impl EngineCoreRequestType {
    /// Decode the single-byte request type frame used on the engine input
    /// socket. Returns `None` for unrecognized values.
    pub fn from_frame(frame: &[u8]) -> Option<Self> {
        let [value] = frame else {
            return None;
        };

        match value {
            0 => Some(Self::Add),
            1 => Some(Self::Abort),
            2 => Some(Self::StartDpWave),
            3 => Some(Self::Utility),
            _ => None,
        }
    }

    /// Encode the request type as the single-byte frame used on the engine
    /// input socket.
    pub fn to_frame(self) -> Bytes {
        Bytes::from_static(match self {
            Self::Add => b"\x00",
            Self::Abort => b"\x01",
            Self::StartDpWave => b"\x02",
            Self::Utility => b"\x03",
        })
    }
}

/// Extra kwargs consumed by engine-side reasoning parsers.
///
/// Original Python construction point:
/// <https://github.com/vllm-project/vllm/blob/cec2ec11760f9f3beabd4c90451936078bf91533/vllm/entrypoints/openai/chat_completion/serving.py#L367-L369>
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReasoningParserKwargs {
    /// Effective kwargs visible to the chat template for this request.
    pub chat_template_kwargs: HashMap<String, serde_json::Value>,
}

/// Engine-core add-request payload sent from frontend to engine.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/3f5bd482f5c1a5dbdffbbf68d624e20bb7032013/vllm/v1/engine/__init__.py#L80-L129>
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple, DefaultFromSerde)]
pub struct EngineCoreRequest {
    pub request_id: String,
    pub prompt_token_ids: Option<Vec<u32>>,
    /// Multimodal features attached to the request.
    pub mm_features: Option<MmFeatures>,
    pub sampling_params: Option<EngineCoreSamplingParams>,
    /// Pooling parameters are preserved in the schema but not yet strongly
    /// typed.
    pub pooling_params: Option<OpaqueValue>,
    pub arrival_time: f64,
    #[serde(default)]
    pub lora_request: Option<lora::LoraRequest>,
    #[serde(default)]
    pub cache_salt: Option<String>,
    #[serde(default)]
    pub data_parallel_rank: Option<u32>,
    /// Unsupported in the first-stage Rust client because Python uses a custom
    /// tensor/aux-frame encoding path for this field.
    #[serde(default)]
    pub prompt_embeds: Option<OpaqueValue>,
    /// Per-position mask for mixed-mode inputs (e.g. chat completion with
    /// `prompt_embeds` content parts). `Some(true)` means real token id;
    /// `Some(false)` means the position uses a pre-computed entry from
    /// `prompt_embeds`. `None` for pure-tokens and pure-embeds requests.
    #[serde(default)]
    pub prompt_is_token_ids: Option<Vec<bool>>,
    /// Index of the client, used to ensure outputs are sent back to the same
    /// client when scaling out the frontend.
    #[serde(default)]
    pub client_index: u32,
    /// In DP mode, indicates which wave this request is expected to belong to.
    #[serde(default)]
    pub current_wave: u32,
    #[serde(default)]
    pub priority: i32,
    #[serde(default)]
    pub trace_headers: Option<BTreeMap<String, String>>,
    #[serde(default)]
    pub resumable: bool,
    /// Original user-provided request ID, used for output reporting and aborts.
    #[serde(default)]
    pub external_req_id: Option<String>,
    #[serde(default)]
    pub reasoning_ended: Option<bool>,
    /// Reasoning-parser kwargs forwarded from the frontend to the
    /// structured-output backend.
    #[serde(default)]
    pub reasoning_parser_kwargs: Option<ReasoningParserKwargs>,
    /// If `true`, the request should be added to the scheduler's waiting queue
    /// and immediately aborted, so connector-side cleanup runs via the
    /// standard `request_finished` hook.
    #[serde(default)]
    pub abort_immediately: bool,
}

impl EngineCoreRequest {
    /// Validate fields intentionally not supported in the first-stage client.
    pub fn validate(&self) -> Result<()> {
        if self.prompt_embeds.is_some() {
            return Err(Error::UnsupportedField {
                context: "EngineCoreRequest",
                field: "prompt_embeds",
            });
        }
        Ok(())
    }

    pub(crate) fn extract_aux_frames(&mut self, threshold: usize) -> Vec<Bytes> {
        let mut aux_frames = Vec::new();
        if let Some(features) = &mut self.mm_features {
            for feature in features {
                feature.extract_aux_frames(&mut aux_frames, threshold);
            }
        }
        aux_frames
    }
}

#[cfg(test)]
mod tests {
    use rmpv::Value;

    use super::*;
    use crate::protocol::multimodal::{
        MmBatchedField, MmFeatureSpec, MmField, MmFieldElem, MmKwargValue, PlaceholderRange,
    };
    use crate::protocol::sampling::EngineCoreSamplingParams;
    use crate::protocol::tensor::{WireArrayData, WireTensor};
    use crate::protocol::{decode_value, encode_msgpack};

    #[test]
    fn engine_core_request_serializes_as_full_array() {
        let request = EngineCoreRequest {
            request_id: "req-1".to_string(),
            prompt_token_ids: Some(vec![1, 2, 3]),
            sampling_params: Some(EngineCoreSamplingParams {
                max_tokens: 8,
                ..EngineCoreSamplingParams::for_test()
            }),
            arrival_time: 1234.5,
            client_index: 7,
            ..EngineCoreRequest::default()
        };

        let encoded = encode_msgpack(&request).unwrap();
        let value = decode_value(&encoded).unwrap();
        let array = match value {
            Value::Array(array) => array,
            other => panic!("expected array, got {other:?}"),
        };

        assert_eq!(array.len(), 20);
        assert_eq!(array[0], Value::from("req-1"));
        assert_eq!(array[2], Value::Nil);
        assert_eq!(array[4], Value::Nil);
        assert_eq!(array[10], Value::Nil);
        assert_eq!(array[11], Value::from(7));
    }

    #[test]
    fn engine_core_request_extracts_large_nested_tensors_in_wire_order() {
        let inline = vec![1_u8; DEFAULT_AUX_FRAME_THRESHOLD - 1];
        let first_aux = vec![2_u8; DEFAULT_AUX_FRAME_THRESHOLD];
        let second_aux = vec![3_u8; DEFAULT_AUX_FRAME_THRESHOLD + 1];
        let first_aux_ptr = first_aux.as_ptr();
        let second_aux_ptr = second_aux.as_ptr();
        let mut request = EngineCoreRequest {
            mm_features: Some(vec![MmFeatureSpec {
                data: Some(BTreeMap::from([
                    (
                        "inline".to_string(),
                        MmFieldElem {
                            data: Some(MmKwargValue::Tensor(WireTensor::from_raw(
                                "uint8",
                                vec![inline.len()],
                                inline,
                            ))),
                            field: MmField::Batched(MmBatchedField { keep_on_cpu: false }),
                        },
                    ),
                    (
                        "nested".to_string(),
                        MmFieldElem {
                            data: Some(MmKwargValue::List(vec![
                                MmKwargValue::Int(7),
                                MmKwargValue::Tensor(WireTensor::from_raw(
                                    "uint8",
                                    vec![first_aux.len()],
                                    first_aux,
                                )),
                            ])),
                            field: MmField::Batched(MmBatchedField { keep_on_cpu: false }),
                        },
                    ),
                ])),
                modality: "image".to_string(),
                identifier: "id".to_string(),
                mm_position: PlaceholderRange {
                    offset: 0,
                    length: second_aux.len(),
                    is_embed: Some(WireTensor::from_raw(
                        "bool",
                        vec![second_aux.len()],
                        second_aux,
                    )),
                },
                mm_hash: None,
            }]),
            ..EngineCoreRequest::default()
        };

        let aux_frames = request.extract_aux_frames(DEFAULT_AUX_FRAME_THRESHOLD);

        assert_eq!(aux_frames.len(), 2);
        assert_eq!(aux_frames[0].as_ptr(), first_aux_ptr);
        assert_eq!(aux_frames[1].as_ptr(), second_aux_ptr);
        let feature = &request.mm_features.as_ref().unwrap()[0];
        let MmKwargValue::Tensor(inline) =
            feature.data.as_ref().unwrap()["inline"].data.as_ref().unwrap()
        else {
            panic!("expected inline tensor");
        };
        assert!(matches!(inline.data, WireArrayData::RawView(_)));
        let MmKwargValue::List(nested) =
            feature.data.as_ref().unwrap()["nested"].data.as_ref().unwrap()
        else {
            panic!("expected nested tensor list");
        };
        let MmKwargValue::Tensor(nested_tensor) = &nested[1] else {
            panic!("expected nested tensor");
        };
        assert_eq!(nested_tensor.data, WireArrayData::AuxIndex(1));
        assert_eq!(
            feature.mm_position.is_embed.as_ref().unwrap().data,
            WireArrayData::AuxIndex(2)
        );
        assert!(request.extract_aux_frames(DEFAULT_AUX_FRAME_THRESHOLD).is_empty());
    }
}
