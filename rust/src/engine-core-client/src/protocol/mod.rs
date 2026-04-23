use std::any::type_name;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::Cursor;

use bytes::Bytes;
use rmpv::Value;
use serde::{Deserialize, Serialize};
use serde_default::DefaultFromSerde;
use serde_repr::{Deserialize_repr, Serialize_repr};
use serde_tuple::{Deserialize_tuple, Serialize_tuple};
use thiserror_ext::AsReport;

use crate::error::{Error, Result, value_encode_ext};
use crate::protocol::stats::SchedulerStats;

// TODO: This module currently mixes reusable frontend-facing semantic types
// (for example `FinishReason`, `StopReason`, `RequestOutputKind`, and future
// cleaned-up frontend sampling types) with engine-core-specific wire DTOs and
// handshake/control messages. While the Rust frontend is still evolving quickly,
// keep them co-located here for iteration speed. Once the higher-level API
// boundary stabilizes, move the truly reusable semantic types into a lower-level
// common crate and keep the engine transport/wire messages here.

/// Dynamic msgpack value used for schema positions that are preserved but not
/// yet strongly typed in the early-stage Rust client.
pub type OpaqueValue = Value;

fn default_opaque_value_nil() -> OpaqueValue {
    Value::Nil
}

fn is_false(v: &bool) -> bool {
    !v
}

mod classfied_outputs;
pub mod handshake;
mod logprobs;
pub mod stats;
pub use classfied_outputs::{
    ClassifiedEngineCoreOutputs, DpControlMessage, RequestBatchOutputs, UtilityCallOutput,
};
pub use logprobs::{
    Logprobs, MaybeWireLogprobs, PositionLogprobs, TokenLogprob, decode_engine_core_outputs,
};

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
    pub fn to_frame(self) -> Bytes {
        Bytes::from_static(match self {
            Self::Add => b"\x00",
            Self::Abort => b"\x01",
            Self::StartDpWave => b"\x02",
            Self::Utility => b"\x03",
        })
    }
}

/// Reason a request finished: stop, length, abort, error, or repetition.
///
/// This mirrors the Python enum and uses integer encoding for compact wire
/// representation.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/f22d6e026798a74e6542a52ef776c054f2de572a/vllm/v1/engine/__init__.py#L41-L63>
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize_repr, Deserialize_repr)]
#[repr(u8)]
pub enum EngineCoreFinishReason {
    /// A stop string was emitted.
    Stop = 0,
    /// `max_tokens` or `max_model_len` was reached.
    Length = 1,
    /// The request was aborted by the client.
    Abort = 2,
    /// A retryable request-level internal error occurred.
    Error = 3,
    /// A repetitive token pattern was detected.
    Repetition = 4,
}

/// Event types emitted by engine-core for one request.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/f22d6e026798a74e6542a52ef776c054f2de572a/vllm/v1/engine/__init__.py#L113-L118>
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize_repr, Deserialize_repr)]
#[repr(u8)]
pub enum EngineCoreEventType {
    Queued = 1,
    Scheduled = 2,
    Preempted = 3,
}

/// A timestamped engine-core event associated with one request.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/f22d6e026798a74e6542a52ef776c054f2de572a/vllm/v1/engine/__init__.py#L121-L130>
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineCoreEvent {
    pub r#type: EngineCoreEventType,
    pub timestamp: f64,
}

/// Controls how intermediate outputs are returned to the frontend.
///
/// `Cumulative = 0` is intentionally not supported in Rust frontend.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/f22d6e026798a74e6542a52ef776c054f2de572a/vllm/sampling_params.py#L146-L152>
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize_repr, Deserialize_repr)]
#[repr(u8)]
pub enum RequestOutputKind {
    /// Return only token deltas in each update.
    #[default]
    Delta = 1,
    /// Suppress intermediate updates and return only the final output.
    FinalOnly = 2,
}

/// The stop reason associated with a finished output.
///
/// Python models this as the union-typed `stop_reason: int | str | None`
/// field on `EngineCoreOutput`; the Rust client narrows it into a tagged enum.
///
/// Original Python field:
/// <https://github.com/vllm-project/vllm/blob/f22d6e026798a74e6542a52ef776c054f2de572a/vllm/v1/engine/__init__.py#L155>
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StopReason {
    TokenId(u32),
    Text(String),
}

/// Parameters for configuring structured outputs (guided decoding).
///
/// Exactly one constraint field (`json`, `regex`, `choice`, `grammar`,
/// `json_object`, or `structural_tag`) should be set. The engine-core
/// backend selects the appropriate grammar compiler based on which field
/// is present.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/f22d6e026798a74e6542a52ef776c054f2de572a/vllm/sampling_params.py#L36-L107>
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StructuredOutputsParams {
    /// JSON schema (as a dict/object or JSON string) constraining the output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub json: Option<serde_json::Value>,
    /// Regular expression the output must match.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regex: Option<String>,
    /// List of allowed output strings (the model must produce one of these).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub choice: Option<Vec<String>>,
    /// Context-free grammar (in EBNF-like notation) the output must conform to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grammar: Option<String>,
    /// When `true`, output must be valid JSON (free-form, no schema).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub json_object: Option<bool>,
    /// Disable any additional whitespace in guided JSON output.
    #[serde(default, skip_serializing_if = "crate::protocol::is_false")]
    pub disable_any_whitespace: bool,
    /// Disable `additionalProperties` in JSON schema output.
    #[serde(default, skip_serializing_if = "crate::protocol::is_false")]
    pub disable_additional_properties: bool,
    /// Custom whitespace pattern for guided JSON output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub whitespace_pattern: Option<String>,
    /// Structural tag configuration (JSON-encoded string).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structural_tag: Option<String>,
}

/// Engine-core-facing sampling parameters for text generation.
///
/// This is the normalized southbound subset used by the Rust frontend when it
/// talks to Python engine-core over the wire. User-facing request semantics
/// such as `stop` strings, `n`, `ignore_eos`, and output aggregation mode are
/// intentionally handled by higher layers before values reach this DTO.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/f22d6e026798a74e6542a52ef776c054f2de572a/vllm/sampling_params.py#L155-L291>
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineCoreSamplingParams {
    /// Controls randomness. Lower values are more deterministic; zero means
    /// greedy sampling.
    pub temperature: f32,
    /// Cumulative probability threshold for nucleus sampling.
    pub top_p: f32,
    /// Maximum number of top tokens to consider. `0` means all tokens.
    pub top_k: u32,
    /// Random seed used by the sampler when present.
    pub seed: Option<i64>,
    /// Maximum number of tokens to generate per output sequence.
    pub max_tokens: u32,
    /// Minimum number of tokens to generate before EOS or stop-token handling.
    pub min_tokens: u32,
    /// Number of log probabilities to return per generated token.
    ///
    /// `None` disables sample logprobs. `-1` requests the full vocabulary.
    pub logprobs: Option<i32>,
    /// Number of log probabilities to return per prompt token.
    ///
    /// `None` disables prompt logprobs. `-1` requests the full vocabulary.
    pub prompt_logprobs: Option<i32>,
    /// Minimum probability threshold for token sampling.
    pub min_p: f32,
    /// Frequency penalty applied by the sampler.
    pub frequency_penalty: f32,
    /// Presence penalty applied by the sampler.
    pub presence_penalty: f32,
    /// Repetition penalty applied by the sampler.
    pub repetition_penalty: f32,
    /// Token IDs that stop generation.
    pub stop_token_ids: Vec<u32>,
    /// Primary EOS token ID used by engine-core's dedicated EOS stop path.
    ///
    /// This mirrors Python's internal `_eos_token_id` field and is derived by
    /// the frontend from tokenizer/model metadata rather than supplied directly
    /// by end users.
    #[serde(rename = "_eos_token_id")]
    pub eos_token_id: Option<u32>,
    /// Complete stop-token set used by engine-core for `min_tokens` masking.
    ///
    /// This mirrors Python's internal `_all_stop_token_ids` field and should
    /// contain explicit `stop_token_ids` plus any frontend-derived EOS token IDs.
    #[serde(rename = "_all_stop_token_ids")]
    pub all_stop_token_ids: BTreeSet<u32>,
    /// Logit biases to apply during sampling.
    /// Keys are token IDs
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<u32, f32>>,
    /// Restrict output to these token IDs only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_token_ids: Option<Vec<u32>>,
    /// Tokenized bad words to avoid during generation.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "_bad_words_token_ids"
    )]
    pub bad_words_token_ids: Option<Vec<Vec<u32>>>,
    /// Parameters for configuring structured outputs (guided decoding).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structured_outputs: Option<StructuredOutputsParams>,
    /// Specific token IDs for which log probabilities should be returned at each position.
    ///
    /// When set, the engine returns logprobs for exactly these tokens in addition to the
    /// sampled/scored token. Mutually exclusive with the `logprobs` count field in practice.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprob_token_ids: Option<Vec<u32>>,
    /// If `Some(true)`, the request will not attempt to read from the prefix cache; newly
    /// computed blocks may still populate the cache. `None` defers to engine-core defaults.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skip_reading_prefix_cache: Option<bool>,
    /// Additional request parameters for custom extensions (from `vllm_xargs`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_args: Option<HashMap<String, serde_json::Value>>,
}

impl EngineCoreSamplingParams {
    /// Constructs a default sampling params for testing purposes only.
    pub fn for_test() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            seed: None,
            max_tokens: 65536,
            min_tokens: 0,
            logprobs: None,
            prompt_logprobs: None,
            min_p: 0.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            stop_token_ids: Vec::new(),
            eos_token_id: None,
            all_stop_token_ids: BTreeSet::new(),
            logit_bias: None,
            allowed_token_ids: None,
            bad_words_token_ids: None,
            structured_outputs: None,
            logprob_token_ids: None,
            skip_reading_prefix_cache: None,
            extra_args: None,
        }
    }
}

/// Engine-core add-request payload sent from frontend to engine.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/f22d6e026798a74e6542a52ef776c054f2de572a/vllm/v1/engine/__init__.py#L66-L110>
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple, DefaultFromSerde)]
pub struct EngineCoreRequest {
    pub request_id: String,
    pub prompt_token_ids: Option<Vec<u32>>,
    /// Multimodal features are preserved in the schema but not yet strongly typed.
    pub mm_features: Option<OpaqueValue>,
    pub sampling_params: Option<EngineCoreSamplingParams>,
    /// Pooling parameters are preserved in the schema but not yet strongly typed.
    pub pooling_params: Option<OpaqueValue>,
    pub arrival_time: f64,
    #[serde(default)]
    pub lora_request: Option<OpaqueValue>,
    #[serde(default)]
    pub cache_salt: Option<String>,
    #[serde(default)]
    pub data_parallel_rank: Option<u32>,
    /// Unsupported in the first-stage Rust client because Python uses a custom
    /// tensor/aux-frame encoding path for this field.
    #[serde(default)]
    pub prompt_embeds: Option<OpaqueValue>,
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
}

/// Engine-core utility call payload sent from frontend to engine.
///
/// Original Python payload shape:
/// `(client_index, call_id, method_name, args)`
#[derive(Debug, Clone, PartialEq, Serialize_tuple)]
pub struct EngineCoreUtilityRequest {
    pub client_index: u32,
    pub call_id: i64,
    pub method_name: String,
    pub args: OpaqueValue,
}

impl EngineCoreUtilityRequest {
    /// Create a new utility request with the given strongly typed arguments, encoding them into the
    /// expected msgpack value format.
    pub fn new<T>(
        client_index: u32,
        call_id: i64,
        method_name: impl Into<String>,
        args: T,
    ) -> Result<Self>
    where
        T: Serialize,
    {
        let args = rmpv::ext::to_value(args).map_err(|error| {
            value_encode_ext!("failed to encode utility args: {}", error.as_report())
        })?;
        let args = match args {
            Value::Nil => Value::Array(Vec::new()),
            other => other,
        };

        Ok(Self {
            client_index,
            call_id,
            method_name: method_name.into(),
            args,
        })
    }
}

/// Engine-core output for a single request.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/f22d6e026798a74e6542a52ef776c054f2de572a/vllm/v1/engine/__init__.py#L140-L171>
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple, DefaultFromSerde)]
pub struct EngineCoreOutput {
    pub request_id: String,
    pub new_token_ids: Vec<u32>,
    /// Decoded sample logprobs for the newly generated positions in this output.
    #[serde(default)]
    pub new_logprobs: Option<MaybeWireLogprobs>,
    /// Decoded prompt logprobs for the scored prompt positions emitted in this output.
    #[serde(default)]
    pub new_prompt_logprobs_tensors: Option<MaybeWireLogprobs>,
    #[serde(default)]
    pub pooling_output: Option<OpaqueValue>,
    #[serde(default)]
    pub finish_reason: Option<EngineCoreFinishReason>,
    #[serde(default)]
    pub stop_reason: Option<StopReason>,
    #[serde(default)]
    pub events: Option<Vec<EngineCoreEvent>>,
    #[serde(default)]
    pub kv_transfer_params: Option<serde_json::Value>,
    #[serde(default)]
    pub trace_headers: Option<OpaqueValue>,
    /// Number of tokens with prefix-cache hits, local plus external.
    #[serde(default)]
    pub num_cached_tokens: u32,
    /// Number of tokens computed remotely, preserving the original connector count.
    #[serde(default)]
    pub num_external_computed_tokens: u32,
    #[serde(default)]
    pub routed_experts: Option<OpaqueValue>,
    /// Number of NaNs seen in logits. Values above zero indicate corruption.
    #[serde(default)]
    pub num_nans_in_logits: u32,
}

impl EngineCoreOutput {
    /// Returns whether this output is terminal for the request.
    pub fn finished(&self) -> bool {
        self.finish_reason.is_some()
    }
}

/// Result of a utility call.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/f22d6e026798a74e6542a52ef776c054f2de572a/vllm/v1/engine/__init__.py#L174-L183>
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple, DefaultFromSerde)]
pub struct UtilityOutput {
    pub call_id: i64,
    /// Non-`None` implies the call failed and `result` should be ignored.
    #[serde(default)]
    pub failure_message: Option<String>,
    #[serde(default)]
    pub result: Option<UtilityResultEnvelope>,
}

/// Python `UtilityResult` wrapper carried inside `UtilityOutput.result`.
///
/// Upstream reference:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/v1/serial_utils.py#L178-L185>
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple)]
pub struct UtilityResultEnvelope {
    /// Recursive type information encoded on Python side, serving as the hint for deserialization.
    /// We don't care it here as in Rust frontend all utility calls are strongly-typed.
    #[serde(default)]
    type_info: Option<OpaqueValue>,
    /// The actual utility result.
    #[serde(default = "default_opaque_value_nil")]
    result: OpaqueValue,
}

impl UtilityResultEnvelope {
    /// Create a utility result envelope without type information.
    pub fn without_type_info(result: OpaqueValue) -> Self {
        Self {
            type_info: None,
            result,
        }
    }
}

impl UtilityOutput {
    /// Decode the typed result of a utility call.
    pub fn into_typed_result<T>(self, method: &str) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
    {
        if let Some(message) = self.failure_message {
            return Err(Error::UtilityCallFailed {
                method: method.to_string(),
                call_id: self.call_id,
                message,
            });
        }

        let result = self.result.map(|e| e.result).unwrap_or(Value::Nil);

        rmpv::ext::from_value(result).map_err(|error| Error::UtilityResultDecode {
            method: method.to_string(),
            call_id: self.call_id,
            message: error.to_report_string(),
        })
    }
}

/// Batch of engine-core outputs returned to a frontend client.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/f22d6e026798a74e6542a52ef776c054f2de572a/vllm/v1/engine/__init__.py#L186-L214>
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple, DefaultFromSerde)]
pub struct EngineCoreOutputs {
    #[serde(default)]
    pub engine_index: u32,
    /// Outputs grouped for this client in the current engine tick.
    #[serde(default)]
    pub outputs: Vec<EngineCoreOutput>,
    #[serde(default)]
    pub scheduler_stats: Option<Box<SchedulerStats>>,
    #[serde(default)]
    pub timestamp: f64,
    #[serde(default)]
    pub utility_output: Option<UtilityOutput>,
    #[serde(default)]
    pub finished_requests: Option<BTreeSet<String>>,
    /// In DP mode, signals that the current wave finished and engines are paused.
    #[serde(default)]
    pub wave_complete: Option<u32>,
    /// In DP mode, signals that a request arrived for an old wave and the next
    /// wave needs to start in other engines.
    #[serde(default)]
    pub start_wave: Option<u32>,
}

/// Encode a Rust value into msgpack using the protocol crate's serde model.
pub fn encode_msgpack<T>(value: &T) -> Result<Vec<u8>>
where
    T: Serialize,
{
    Ok(rmp_serde::to_vec_named(value)?)
}

/// Decode a msgpack payload into a strongly typed protocol value.
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

    rmp_serde::from_slice(bytes).map_err(|error| Error::DecodeWithMessage {
        target_type: type_name::<T>(),
        message: format!("{error}; value fallback: {}", decode_value_preview(bytes)),
    })
}

pub fn decode_value(bytes: &[u8]) -> Result<Value> {
    Ok(rmpv::decode::read_value(&mut Cursor::new(bytes))?)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    fn utility_result_value<T>(value: T) -> UtilityResultEnvelope
    where
        T: Serialize,
    {
        UtilityResultEnvelope::without_type_info(rmpv::ext::to_value(value).unwrap())
    }

    #[test]
    fn engine_core_request_serializes_as_full_array() {
        let request = EngineCoreRequest {
            request_id: "req-1".to_string(),
            prompt_token_ids: Some(vec![1, 2, 3]),
            mm_features: None,
            sampling_params: Some(EngineCoreSamplingParams {
                max_tokens: 8,
                ..EngineCoreSamplingParams::for_test()
            }),
            pooling_params: None,
            arrival_time: 1234.5,
            lora_request: None,
            cache_salt: None,
            data_parallel_rank: None,
            prompt_embeds: None,
            client_index: 7,
            current_wave: 0,
            priority: 0,
            trace_headers: None,
            resumable: false,
            external_req_id: None,
            reasoning_ended: None,
        };

        let encoded = encode_msgpack(&request).unwrap();
        let value = decode_value(&encoded).unwrap();
        let array = match value {
            Value::Array(array) => array,
            other => panic!("expected array, got {other:?}"),
        };

        assert_eq!(array.len(), 17);
        assert_eq!(array[0], Value::from("req-1"));
        assert_eq!(array[2], Value::Nil);
        assert_eq!(array[4], Value::Nil);
        assert_eq!(array[10], Value::from(7));
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
                num_cached_tokens: 0,
                num_external_computed_tokens: 0,
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
    fn utility_request_serializes_as_tuple_payload() {
        let request = EngineCoreUtilityRequest::new(7, 42, "is_sleeping", ()).unwrap();

        let encoded = encode_msgpack(&request).unwrap();
        let value = decode_value(&encoded).unwrap();
        let array = match value {
            Value::Array(array) => array,
            other => panic!("expected utility request array, got {other:?}"),
        };

        assert_eq!(array.len(), 4);
        assert_eq!(array[0], Value::from(7));
        assert_eq!(array[1], Value::from(42));
        assert_eq!(array[2], Value::from("is_sleeping"));
        assert_eq!(array[3], Value::Array(Vec::new()));
    }

    #[test]
    fn utility_output_decodes_typed_result() {
        let output = UtilityOutput {
            call_id: 9,
            failure_message: None,
            result: Some(utility_result_value(true)),
        };

        assert!(output.into_typed_result::<bool>("is_sleeping").unwrap());
    }

    #[test]
    fn utility_output_reports_failure_message() {
        let error = UtilityOutput {
            call_id: 9,
            failure_message: Some("boom".to_string()),
            result: None,
        }
        .into_typed_result::<bool>("is_sleeping")
        .unwrap_err();

        assert!(matches!(
            error,
            Error::UtilityCallFailed {
                method,
                call_id,
                message
            } if method == "is_sleeping" && call_id == 9 && message == "boom"
        ));
    }

    #[test]
    fn utility_output_decodes_missing_result_as_unit() {
        UtilityOutput {
            call_id: 3,
            failure_message: None,
            result: None,
        }
        .into_typed_result::<()>("reset_mm_cache")
        .unwrap();
    }

    #[test]
    fn utility_output_decodes_nil_result_as_unit() {
        UtilityOutput {
            call_id: 4,
            failure_message: None,
            result: Some(UtilityResultEnvelope::without_type_info(Value::Nil)),
        }
        .into_typed_result::<()>("sleep")
        .unwrap();
    }

    #[test]
    fn decode_msgpack_includes_type_name_and_value_fallback() {
        let error = decode_msgpack::<u64>(
            &rmp_serde::to_vec_named(&BTreeMap::from([("status", "READY")])).unwrap(),
        )
        .unwrap_err();

        expect_test::expect![[r#"messagepack decode failed for u64: wrong msgpack marker FixMap(1); value fallback: {"status": "READY"}"#]].assert_eq(&error.to_report_string());
    }
}
