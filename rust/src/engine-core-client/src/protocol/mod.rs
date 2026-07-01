use std::any::type_name;
use std::io::Cursor;

use rmpv::Value;
use serde::{Deserialize, Serialize};
use thiserror_ext::AsReport;

use crate::error::{Error, Result};

// TODO: This module currently mixes reusable frontend-facing semantic types
// (for example `FinishReason`, `StopReason`, `RequestOutputKind`, and future
// cleaned-up frontend sampling types) with engine-core-specific wire DTOs and
// handshake/control messages. While the Rust frontend is still evolving
// quickly, keep them co-located here for iteration speed. Once the higher-level
// API boundary stabilizes, move the truly reusable semantic types into a
// lower-level common crate and keep the engine transport/wire messages here.

/// Dynamic msgpack value used for schema positions that are preserved but not
/// yet strongly typed in the early-stage Rust client.
pub type OpaqueValue = Value;

fn default_opaque_value_nil() -> OpaqueValue {
    Value::Nil
}

fn is_false(v: &bool) -> bool {
    !v
}

pub mod dtype;
pub mod handshake;
pub mod logprobs;
pub mod lora;
pub mod multimodal;
mod output;
mod request;
mod sampling;
pub mod stats;
mod structured_outputs;
pub mod tensor;
pub mod utility;
pub use dtype::ModelDtype;
pub use logprobs::decode_engine_core_outputs;
pub use output::*;
pub use request::*;
pub use sampling::*;
pub use structured_outputs::*;

/// Encode a Rust value into msgpack using the protocol crate's serde model.
pub fn encode_msgpack<T>(value: &T) -> Result<Vec<u8>>
where
    T: Serialize + std::fmt::Debug,
{
    rmp_serde::to_vec_named(value).map_err(|error| Error::Encode {
        target_type: type_name::<T>(),
        message: format!(
            "failed to encode value `{:?}`: {}",
            value,
            error.to_report_string()
        ),
    })
}

/// Decode a msgpack payload into a strongly typed protocol value, with enhanced
/// error reporting.
pub fn decode_msgpack<T>(bytes: &[u8]) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    fn decode_value_preview(bytes: &[u8]) -> String {
        match decode_value(bytes) {
            Ok(value) => format!("{value}"),
            Err(error) => format!("<value decode failed: {error}>"),
        }
    }

    rmp_serde::from_slice(bytes).map_err(|error| Error::Decode {
        target_type: type_name::<T>(),
        message: format!("{error}; value fallback: {}", decode_value_preview(bytes)),
    })
}

pub fn decode_value(bytes: &[u8]) -> Result<Value> {
    Ok(rmpv::decode::read_value(&mut Cursor::new(bytes))?)
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use crate::protocol::structured_outputs::{StructuredOutputBackend, StructuredOutputsParams};

    use super::*;

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
    fn engine_core_outputs_roundtrip_finished_fields() {
        let outputs = EngineCoreOutputs {
            outputs: vec![EngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: vec![42],
                new_logprobs: None,
                new_prompt_logprobs_tensors: None,
                pooling_output: None,
                finish_reason: Some(EngineCoreFinishReason::Length),
                stop_reason: Some(StopReason::Text("stop".to_string())),
                events: None,
                kv_transfer_params: None,
                trace_headers: None,
                prefill_stats: None,
                routed_experts: None,
                num_nans_in_logits: 0,
            }],
            finished_requests: Some(BTreeSet::from(["req-1".to_string()])),
            ..Default::default()
        };

        let encoded = encode_msgpack(&outputs).unwrap();
        let decoded: EngineCoreOutputs = decode_msgpack(&encoded).unwrap();

        assert_eq!(decoded.outputs.len(), 1);
        assert_eq!(
            decoded.outputs[0].finish_reason,
            Some(EngineCoreFinishReason::Length)
        );
        assert_eq!(
            decoded.finished_requests,
            Some(BTreeSet::from(["req-1".to_string()]))
        );
    }

    #[test]
    fn decode_msgpack_includes_type_name_and_value_fallback() {
        let error = decode_msgpack::<u64>(
            &rmp_serde::to_vec_named(&BTreeMap::from([("status", "READY")])).unwrap(),
        )
        .unwrap_err();

        expect_test::expect![[r#"messagepack decode failed for u64: wrong msgpack marker FixMap(1); value fallback: {"status": "READY"}"#]].assert_eq(&error.to_report_string());
    }

    #[test]
    fn structured_outputs_backend_ignores_deserialized_value() {
        let params: StructuredOutputsParams = serde_json::from_value(serde_json::json!({
            "json_object": true,
            "_backend": "xgrammar",
        }))
        .unwrap();

        assert_eq!(params.backend, StructuredOutputBackend::Guidance);

        let value = serde_json::to_value(params).unwrap();
        assert_eq!(value["_backend"], "guidance");
    }

    /// A real `sampling_params` is a sparse `omit_defaults` map; absent fields
    /// must fall back to defaults. `python_compat` can't catch this since Rust
    /// encodes full maps (see `engine_core_request_serializes_as_full_array`).
    #[test]
    fn decodes_sampling_params_with_omitted_defaults() {
        let sampling_params = Value::Map(vec![
            (
                Value::from("stop_token_ids"),
                Value::Array(vec![Value::from(151643u32)]),
            ),
            (Value::from("skip_reading_prefix_cache"), Value::from(false)),
        ]);
        let request = Value::Array(vec![
            Value::from("req-omit-defaults"),
            Value::Array(vec![
                Value::from(1u32),
                Value::from(2u32),
                Value::from(3u32),
            ]),
            Value::Nil,
            sampling_params,
            Value::Nil,
            Value::from(1.0f64),
        ]);

        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &request).unwrap();

        let decoded: EngineCoreRequest = decode_msgpack(&bytes)
            .expect("a real omit_defaults request must decode (regression: missing field)");

        assert_eq!(decoded.request_id, "req-omit-defaults");
        let sampling = decoded.sampling_params.expect("sampling params present");

        assert_eq!(sampling.stop_token_ids, vec![151643]);
        assert_eq!(sampling.skip_reading_prefix_cache, Some(false));

        // Omitted fields -> Python defaults.
        assert_eq!(sampling.temperature, 1.0);
        assert_eq!(sampling.top_p, 1.0);
        assert_eq!(sampling.top_k, 0);
        assert_eq!(sampling.seed, None);
        assert_eq!(sampling.max_tokens, 16);
        assert_eq!(sampling.min_tokens, 0);
        assert_eq!(sampling.min_p, 0.0);
        assert_eq!(sampling.frequency_penalty, 0.0);
        assert_eq!(sampling.presence_penalty, 0.0);
        assert_eq!(sampling.repetition_penalty, 1.0);
        assert_eq!(sampling.logprobs, None);
        assert_eq!(sampling.prompt_logprobs, None);
        assert_eq!(sampling.eos_token_id, None);
        assert!(sampling.all_stop_token_ids.is_empty());
    }
}
