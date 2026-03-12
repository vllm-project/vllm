use std::collections::{BTreeMap, BTreeSet};
use std::io::Cursor;

use bytes::Bytes;
use rmpv::Value;
use serde::{Deserialize, Serialize};
use serde_default::DefaultFromSerde;
use serde_repr::{Deserialize_repr, Serialize_repr};
use serde_tuple::{Deserialize_tuple, Serialize_tuple};

use crate::error::{Error, Result};

/// Dynamic msgpack value used for schema positions that are preserved but not
/// yet strongly typed in the early-stage Rust client.
pub type OpaqueValue = Value;

pub mod handshake;

mod defaults {
    pub fn sampling_n() -> u32 {
        1
    }

    pub fn temperature() -> f32 {
        1.0
    }

    pub fn top_p() -> f32 {
        1.0
    }

    pub fn max_tokens() -> Option<u32> {
        Some(16)
    }
}

/// Request types are encoded as single-byte protocol constants so they can be
/// sent over the ZMQ socket without an extra encoding step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EngineCoreRequestType {
    Add = 0,
    Abort = 1,
    StartDpWave = 2,
    Utility = 3,
}

impl EngineCoreRequestType {
    pub fn as_frame(self) -> Bytes {
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize_repr, Deserialize_repr)]
#[repr(u8)]
pub enum FinishReason {
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

/// Controls how intermediate outputs are returned to the frontend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize_repr, Deserialize_repr)]
#[repr(u8)]
pub enum RequestOutputKind {
    /// Return the entire output-so-far in every update.
    #[default]
    Cumulative = 0,
    /// Return only token deltas in each update.
    Delta = 1,
    /// Suppress intermediate updates and return only the final output.
    FinalOnly = 2,
}

/// The stop reason associated with a finished output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StopReason {
    TokenId(u32),
    Text(String),
}

/// Sampling parameters for text generation.
///
/// This is the first-stage strongly typed subset of Python `SamplingParams`.
/// Complex or less stable fields remain dynamic values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, DefaultFromSerde)]
pub struct SamplingParams {
    /// Number of outputs to return for the given prompt request.
    #[serde(default = "defaults::sampling_n")]
    pub n: u32,
    /// Controls randomness. Lower values are more deterministic; zero means
    /// greedy sampling.
    #[serde(default = "defaults::temperature")]
    pub temperature: f32,
    /// Cumulative probability threshold for nucleus sampling.
    #[serde(default = "defaults::top_p")]
    pub top_p: f32,
    /// Maximum number of top tokens to consider. `0` means all tokens.
    #[serde(default)]
    pub top_k: i32,
    /// Maximum number of tokens to generate per output sequence.
    #[serde(default = "defaults::max_tokens")]
    pub max_tokens: Option<u32>,
    /// Minimum number of tokens to generate before EOS or stop-token handling.
    #[serde(default)]
    pub min_tokens: u32,
    /// Stop strings. Returned output does not include them.
    #[serde(default)]
    pub stop: Vec<String>,
    /// Token IDs that stop generation.
    #[serde(default)]
    pub stop_token_ids: Vec<u32>,
    /// Continue generation after EOS if set.
    #[serde(default)]
    pub ignore_eos: bool,
    /// Whether updates are cumulative, delta-based, or final-only.
    #[serde(default)]
    pub output_kind: RequestOutputKind,
    /// Structured output configuration carried through as an opaque msgpack value.
    #[serde(default)]
    pub structured_outputs: Option<OpaqueValue>,
    /// Logit bias configuration carried through as an opaque msgpack value.
    #[serde(default)]
    pub logit_bias: Option<OpaqueValue>,
    /// Optional allow-list of token IDs.
    #[serde(default)]
    pub allowed_token_ids: Option<Vec<u32>>,
    /// Arbitrary engine/plugin-specific args.
    #[serde(default)]
    pub extra_args: Option<OpaqueValue>,
    /// Repetition detection config carried through as an opaque msgpack value.
    #[serde(default)]
    pub repetition_detection: Option<OpaqueValue>,
}

/// Engine-core add-request payload sent from frontend to engine.
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple, DefaultFromSerde)]
pub struct EngineCoreRequest {
    pub request_id: String,
    pub prompt_token_ids: Option<Vec<u32>>,
    /// Multimodal features are preserved in the schema but not yet strongly typed.
    pub mm_features: Option<OpaqueValue>,
    pub sampling_params: Option<SamplingParams>,
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

/// Engine-core output for a single request.
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple, DefaultFromSerde)]
pub struct EngineCoreOutput {
    pub request_id: String,
    pub new_token_ids: Vec<u32>,
    #[serde(default)]
    pub new_logprobs: Option<OpaqueValue>,
    #[serde(default)]
    pub new_prompt_logprobs_tensors: Option<OpaqueValue>,
    #[serde(default)]
    pub pooling_output: Option<OpaqueValue>,
    #[serde(default)]
    pub finish_reason: Option<FinishReason>,
    #[serde(default)]
    pub stop_reason: Option<StopReason>,
    #[serde(default)]
    pub events: Option<OpaqueValue>,
    #[serde(default)]
    pub kv_transfer_params: Option<OpaqueValue>,
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
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple, DefaultFromSerde)]
pub struct UtilityOutput {
    pub call_id: i64,
    /// Non-`None` implies the call failed and `result` should be ignored.
    #[serde(default)]
    pub failure_message: Option<String>,
    #[serde(default)]
    pub result: Option<OpaqueValue>,
}

/// Batch of engine-core outputs returned to a frontend client.
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple, DefaultFromSerde)]
pub struct EngineCoreOutputs {
    #[serde(default)]
    pub engine_index: u32,
    /// Outputs grouped for this client in the current engine tick.
    #[serde(default)]
    pub outputs: Vec<EngineCoreOutput>,
    #[serde(default)]
    pub scheduler_stats: Option<OpaqueValue>,
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
    Ok(rmp_serde::from_slice(bytes)?)
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn decode_value(bytes: &[u8]) -> Result<Value> {
    Ok(rmpv::decode::read_value(&mut Cursor::new(bytes))?)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn engine_core_request_serializes_as_full_array() {
        let request = EngineCoreRequest {
            request_id: "req-1".to_string(),
            prompt_token_ids: Some(vec![1, 2, 3]),
            mm_features: None,
            sampling_params: Some(SamplingParams {
                max_tokens: Some(8),
                ..Default::default()
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
                finish_reason: Some(FinishReason::Length),
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
        assert_eq!(decoded.outputs[0].finish_reason, Some(FinishReason::Length));
        assert_eq!(
            decoded.finished_requests,
            Some(BTreeSet::from(["req-1".to_string()]))
        );
    }
}
