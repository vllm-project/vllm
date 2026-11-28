// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Conversion from internal token requests to gRPC protobuf types.

use serde_json::Value;
use thiserror_ext::AsReport as _;
use uuid::Uuid;
use vllm_llm::{Error as LlmError, GenerateRequest};
use vllm_proto as pb;

/// Encode an already lowered token request for a remote generation RPC.
pub(crate) fn to_grpc_request(
    model: &str,
    rpc_request_id: Uuid,
    request: GenerateRequest,
) -> Result<pb::GenerateRequest, LlmError> {
    let sampling_params = request.sampling_params;
    let kwargs = request
        .reasoning_parser_kwargs
        .filter(|k| !k.chat_template_kwargs.is_empty())
        .map(|kwargs| {
            let kwargs = kwargs.chat_template_kwargs;
            if !kwargs.values().all(numbers_fit_protobuf_struct) {
                return Err(LlmError::Unsupported {
                    field: "chat_template_kwargs number outside protobuf Struct precision",
                });
            }
            Ok(prost_types::Struct {
                fields: kwargs
                    .into_iter()
                    .map(|(key, value)| (key, json_to_proto_value(&value)))
                    .collect(),
            })
        })
        .transpose()?;
    Ok(pb::GenerateRequest {
        model: model.into(),
        request_id: rpc_request_id.to_string(),
        prompt: Some(pb::generate_request::Prompt::TokenIds(pb::TokenIds {
            ids: request.prompt_token_ids,
        })),
        temperature: Some(sampling_params.temperature),
        sampling: Some(pb::RandomSampling {
            num_sequences: 1,
            top_p: Some(sampling_params.top_p),
            top_k: Some(sampling_params.top_k),
            min_p: Some(sampling_params.min_p),
            seed: sampling_params.seed,
        }),
        decoding: Some(pb::DecodingParameters {
            presence_penalty: sampling_params.presence_penalty,
            frequency_penalty: sampling_params.frequency_penalty,
            repetition_penalty: sampling_params.repetition_penalty,
            allowed_token_ids: sampling_params.allowed_token_ids.unwrap_or_default(),
            logit_bias: sampling_params.logit_bias.unwrap_or_default(),
            bad_words_token_ids: sampling_params
                .bad_words_token_ids
                .unwrap_or_default()
                .into_iter()
                .map(|ids| pb::TokenIds { ids })
                .collect(),
            ..Default::default()
        }),
        stopping: Some(pb::StoppingCriteria {
            max_new_tokens: sampling_params.max_tokens,
            min_new_tokens: sampling_params.min_tokens,
            stop_token_ids: sampling_params.stop_token_ids,
            ignore_eos: sampling_params.eos_token_id.is_none(),
            thinking_token_budget: sampling_params
                .thinking_token_budget
                .map(i64::try_from)
                .transpose()
                .map_err(|error| LlmError::Remote {
                    message: error.to_report_string(),
                })?,
            ..Default::default()
        }),
        response: Some(pb::ResponseOptions {
            output_text: Some(false),
            output_token_ids: true,
            prompt_token_ids: true,
            ..Default::default()
        }),
        priority: request.priority,
        session_id: request.session_id,
        watermarking: Some(sampling_params.watermarking),
        kv: Some(pb::KvCacheParameters {
            cache_salt: request.cache_salt.unwrap_or_default(),
            bypass_prefix_cache: sampling_params.skip_reading_prefix_cache.unwrap_or(false),
            ..Default::default()
        }),
        engine_reasoning_gate: (request.reasoning_ended.is_some() || kwargs.is_some()).then_some(
            pb::EngineReasoningGate {
                reasoning_ended: request.reasoning_ended,
                chat_template_kwargs: kwargs,
            },
        ),
        ..Default::default()
    })
}

fn numbers_fit_protobuf_struct(value: &Value) -> bool {
    match value {
        // Struct stores numbers as f64. Compare integer values in i128 so even
        // a u64::MAX rounded up to 2^64 is detected.
        Value::Number(n) => n.as_i128().is_none_or(|i| i as f64 as i128 == i),
        Value::Array(items) => items.iter().all(numbers_fit_protobuf_struct),
        Value::Object(fields) => fields.values().all(numbers_fit_protobuf_struct),
        _ => true,
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

#[cfg(test)]
mod tests {
    use super::numbers_fit_protobuf_struct;

    #[test]
    fn template_kwargs_require_exact_numbers_in_protobuf_struct() {
        use serde_json::json;
        assert!(numbers_fit_protobuf_struct(
            &json!({"nested": [1.0, 0.25, 1u64 << 63]})
        ));
        for integer in [
            json!(9_007_199_254_740_993u64),
            json!(-9_007_199_254_740_993i64),
            json!(u64::MAX),
        ] {
            assert!(!numbers_fit_protobuf_struct(&json!({"nested": [integer]})));
        }
    }
}
