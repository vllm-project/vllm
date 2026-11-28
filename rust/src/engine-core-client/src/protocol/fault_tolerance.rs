// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Sentinel `UtilityOutput.call_id` used for unsolicited fault-tolerance
/// engine status pushes from `EngineCoreSentinel`.
///
/// Original Python definition:
/// `vllm/v1/engine/__init__.py` (`FT_STATUS_CALL_ID = -2`).
pub const FT_STATUS_CALL_ID: i64 = -2;

/// Engine-core utility method name for fault-tolerance recovery instructions.
///
/// Original Python definition:
/// `vllm/v1/fault_tolerance/engine_core_sentinel.py` (`FT_UTILITY_METHOD`).
pub const FT_UTILITY_METHOD: &str = "handle_fault_tolerance";

/// Supported fault-tolerance instruction names.
///
/// Original Python definition:
/// `vllm/v1/fault_tolerance/utils.py` (`ALLOWED_FT_INSTRUCTIONS`).
pub const ALLOWED_FT_INSTRUCTIONS: &[&str] = &["retry", "scale_down"];

/// Return whether `instruction` is an allowed fault-tolerance instruction.
pub fn is_allowed_ft_instruction(instruction: &str) -> bool {
    ALLOWED_FT_INSTRUCTIONS.contains(&instruction)
}

/// Fault-tolerance recovery instruction sent via the `handle_fault_tolerance`
/// utility RPC.
///
/// Serialized as a MessagePack map to match Python's non-`array_like`
/// `msgspec.Struct` `FaultToleranceRequest` (which `EngineCoreSentinel`
/// unpacks with `FaultToleranceRequest(**ft_args)`).
///
/// `#[serde(into = "Map<String, Value>")]` ensures `rmpv::ext::to_value`
/// encodes a MessagePack map rather than a positional struct array.
///
/// Original Python definition:
/// `vllm/v1/fault_tolerance/utils.py` (`FaultToleranceRequest`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(into = "Map<String, Value>")]
pub struct FaultToleranceRequest {
    pub instruction: String,
    #[serde(default)]
    pub params: Map<String, Value>,
    #[serde(default)]
    pub request_id: String,
}

impl From<FaultToleranceRequest> for Map<String, Value> {
    fn from(request: FaultToleranceRequest) -> Self {
        let mut map = Map::with_capacity(3);
        map.insert(
            "instruction".to_string(),
            Value::String(request.instruction),
        );
        map.insert("params".to_string(), Value::Object(request.params));
        map.insert("request_id".to_string(), Value::String(request.request_id));
        map
    }
}

/// Result returned by the engine's `handle_fault_tolerance` utility RPC.
///
/// Original Python definition:
/// `vllm/v1/fault_tolerance/utils.py` (`FaultToleranceResult`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(into = "Map<String, Value>")]
pub struct FaultToleranceResult {
    pub request_id: String,
    pub success: bool,
    #[serde(default)]
    pub reason: Option<String>,
}

impl From<FaultToleranceResult> for Map<String, Value> {
    fn from(result: FaultToleranceResult) -> Self {
        let mut map = Map::with_capacity(3);
        map.insert("request_id".to_string(), Value::String(result.request_id));
        map.insert("success".to_string(), Value::Bool(result.success));
        map.insert(
            "reason".to_string(),
            match result.reason {
                Some(reason) => Value::String(reason),
                None => Value::Null,
            },
        );
        map
    }
}

/// JSON payload returned by `GET /v1/fault_tolerance/status`.
///
/// Original Python definition:
/// `vllm/v1/engine/core_client.py` (`AsyncMPClient.get_status`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FaultToleranceStatusResponse {
    pub schema_version: u32,
    pub total_engines: usize,
    pub engines: Vec<Map<String, Value>>,
}

#[cfg(test)]
mod tests {
    use rmpv::Value as MsgpackValue;
    use serde_json::{Map, json};

    use super::{
        FT_STATUS_CALL_ID, FT_UTILITY_METHOD, FaultToleranceRequest, FaultToleranceResult,
        is_allowed_ft_instruction,
    };
    use crate::protocol::utility::{
        EngineCoreUtilityRequest, UtilityOutput, UtilityResultEnvelope,
    };
    use crate::protocol::{decode_msgpack, decode_value, encode_msgpack};

    #[test]
    fn allowed_instructions_match_python() {
        assert!(is_allowed_ft_instruction("retry"));
        assert!(is_allowed_ft_instruction("scale_down"));
        assert!(!is_allowed_ft_instruction("restart"));
        assert!(!is_allowed_ft_instruction(""));
    }

    #[test]
    fn fault_tolerance_request_serializes_as_named_map_inside_utility_args() {
        let mut params = Map::new();
        params.insert("abort".to_string(), json!(true));
        let ft_request = FaultToleranceRequest {
            instruction: "scale_down".to_string(),
            params,
            request_id: "ft-req-1".to_string(),
        };

        let request =
            EngineCoreUtilityRequest::new(0, 11, FT_UTILITY_METHOD, (&ft_request,)).unwrap();
        let encoded = encode_msgpack(&request).unwrap();
        let value = decode_value(&encoded).unwrap();

        let MsgpackValue::Array(frame) = value else {
            panic!("expected utility request array, got {value:?}");
        };
        assert_eq!(frame.len(), 4);
        assert_eq!(frame[0], MsgpackValue::from(0));
        assert_eq!(frame[1], MsgpackValue::from(11));
        assert_eq!(frame[2], MsgpackValue::from(FT_UTILITY_METHOD));

        let MsgpackValue::Array(args) = &frame[3] else {
            panic!("expected 1-element utility args array, got {:?}", frame[3]);
        };
        assert_eq!(args.len(), 1);

        let decoded: FaultToleranceRequest = rmpv::ext::from_value(args[0].clone()).unwrap();
        assert_eq!(decoded, ft_request);
        assert!(
            matches!(args[0], MsgpackValue::Map(_)),
            "expected FaultToleranceRequest to encode as a msgpack map, got {:?}",
            args[0]
        );
    }

    #[test]
    fn fault_tolerance_result_decodes_from_utility_output() {
        let payload = rmpv::ext::to_value(json!({
            "request_id": "ft-req-2",
            "success": false,
            "reason": "recovery timed out",
        }))
        .unwrap();
        let output = UtilityOutput {
            call_id: 12_u64.into(),
            failure_message: None,
            result: Some(UtilityResultEnvelope::without_type_info(payload)),
        };

        let decoded: FaultToleranceResult = output.into_typed_result(FT_UTILITY_METHOD).unwrap();
        assert_eq!(
            decoded,
            FaultToleranceResult {
                request_id: "ft-req-2".to_string(),
                success: false,
                reason: Some("recovery timed out".to_string()),
            }
        );
    }

    #[test]
    fn fault_tolerance_status_push_decodes_negative_call_id() {
        let payload = rmpv::ext::to_value(json!({
            "id": 0,
            "status": "unhealthy",
            "fault_info": "NCCL error",
            "mask": [1, 0],
        }))
        .unwrap();
        let output = UtilityOutput {
            call_id: FT_STATUS_CALL_ID.into(),
            failure_message: None,
            result: Some(UtilityResultEnvelope::without_type_info(payload)),
        };
        let encoded = encode_msgpack(&output).unwrap();
        let decoded: UtilityOutput = decode_msgpack(&encoded).unwrap();
        assert_eq!(decoded.call_id.as_i64(), Some(FT_STATUS_CALL_ID));
        let status: Option<Map<String, serde_json::Value>> =
            decoded.into_typed_result("fault_tolerance_status").unwrap();
        assert_eq!(
            serde_json::Value::Object(status.unwrap()),
            json!({
                "id": 0,
                "status": "unhealthy",
                "fault_info": "NCCL error",
                "mask": [1, 0],
            })
        );
    }
}
