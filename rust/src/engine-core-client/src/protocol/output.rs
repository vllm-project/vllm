use std::collections::BTreeSet;

use enum_as_inner::EnumAsInner;
use serde::{Deserialize, Serialize};
use serde_default::DefaultFromSerde;
use serde_repr::{Deserialize_repr, Serialize_repr};
use serde_tuple::{Deserialize_tuple, Serialize_tuple};

use super::utility::UtilityOutput;
use crate::error::{Error, Result, ext_value_decode};
use crate::protocol::logprobs::MaybeWireLogprobs;
use crate::protocol::stats::{PrefillStats, SchedulerStats};
use crate::protocol::{OpaqueValue, decode_msgpack};

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

/// Engine-core output for a single request.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/d3af8c18317c0dc008d42e4367fbb9045cfb7bf6/vllm/v1/engine/__init__.py#L154-L184>
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple, DefaultFromSerde)]
pub struct EngineCoreOutput {
    pub request_id: String,
    pub new_token_ids: Vec<u32>,
    /// Decoded sample logprobs for the newly generated positions in this
    /// output.
    #[serde(default)]
    pub new_logprobs: Option<MaybeWireLogprobs>,
    /// Decoded prompt logprobs for the scored prompt positions emitted in this
    /// output.
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
    /// Breakdown of the scheduled prefill computation, set on the first output
    /// of a newly scheduled prefill and elided for subsequent decode outputs.
    #[serde(default)]
    pub prefill_stats: Option<PrefillStats>,
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

    /// Resolve all wire-format fields in-place by looking up aux frames and
    /// decoding raw-view payloads as needed.
    fn resolve_in_place<Frame>(&mut self, frames: &[Frame]) -> Result<()>
    where
        Frame: AsRef<[u8]>,
    {
        self.new_logprobs = (self.new_logprobs.take())
            .map(|value| value.resolve(frames, "new_logprobs"))
            .transpose()?;
        self.new_prompt_logprobs_tensors = (self.new_prompt_logprobs_tensors.take())
            .map(|value| value.resolve(frames, "new_prompt_logprobs_tensors"))
            .transpose()?;
        Ok(())
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
    /// In DP mode, signals that the current wave finished and engines are
    /// paused.
    #[serde(default)]
    pub wave_complete: Option<u32>,
    /// In DP mode, signals that a request arrived for an old wave and the next
    /// wave needs to start in other engines.
    #[serde(default)]
    pub start_wave: Option<u32>,
}

/// Data-parallel control notifications multiplexed through `EngineCoreOutputs`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DpControlMessage {
    WaveComplete(u32),
    StartWave(u32),
}

#[derive(Debug, Clone, PartialEq)]
pub struct RequestBatchOutputs {
    pub engine_index: u32,
    pub outputs: Vec<EngineCoreOutput>,
    pub scheduler_stats: Option<Box<SchedulerStats>>,
    pub timestamp: f64,
    pub finished_requests: Option<BTreeSet<String>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UtilityCallOutput {
    pub engine_index: u32,
    pub timestamp: f64,
    pub output: UtilityOutput,
}

/// Semantic classification of a raw `EngineCoreOutputs` message.
///
/// Python currently uses one product-shaped wire struct for several distinct
/// output families. This enum exposes those families more explicitly without
/// changing the wire format.
#[derive(Debug, Clone, PartialEq, EnumAsInner)]
pub enum ClassifiedEngineCoreOutputs {
    RequestBatch(RequestBatchOutputs),
    Utility(UtilityCallOutput),
    DpControl {
        engine_index: u32,
        timestamp: f64,
        control: DpControlMessage,
    },
    /// Fallback for wire-shape combinations that do not map cleanly onto the
    /// current semantic families.
    Other(EngineCoreOutputs),
}

impl EngineCoreOutputs {
    /// Resolve all wire-format fields in-place by looking up aux frames and
    /// decoding raw-view payloads as needed.
    fn resolve_in_place<Frame>(&mut self, frames: &[Frame]) -> Result<()>
    where
        Frame: AsRef<[u8]>,
    {
        for output in &mut self.outputs {
            output.resolve_in_place(frames)?;
        }
        Ok(())
    }

    /// Classify the raw wire message into a more semantic Rust enum.
    pub fn classify(self) -> ClassifiedEngineCoreOutputs {
        let has_request_payload = !self.outputs.is_empty()
            || self.scheduler_stats.is_some()
            || self.finished_requests.is_some();

        match (
            has_request_payload,
            &self.utility_output,
            &self.wave_complete,
            &self.start_wave,
        ) {
            (true, None, None, None) => {
                ClassifiedEngineCoreOutputs::RequestBatch(RequestBatchOutputs {
                    engine_index: self.engine_index,
                    outputs: self.outputs,
                    scheduler_stats: self.scheduler_stats,
                    timestamp: self.timestamp,
                    finished_requests: self.finished_requests,
                })
            }
            (false, Some(_), None, None) => {
                ClassifiedEngineCoreOutputs::Utility(UtilityCallOutput {
                    engine_index: self.engine_index,
                    timestamp: self.timestamp,
                    output: self.utility_output.unwrap(),
                })
            }
            (false, None, Some(_), None) => ClassifiedEngineCoreOutputs::DpControl {
                engine_index: self.engine_index,
                timestamp: self.timestamp,
                control: DpControlMessage::WaveComplete(self.wave_complete.unwrap()),
            },
            (false, None, None, Some(_)) => ClassifiedEngineCoreOutputs::DpControl {
                engine_index: self.engine_index,
                timestamp: self.timestamp,
                control: DpControlMessage::StartWave(self.start_wave.unwrap()),
            },
            _ => ClassifiedEngineCoreOutputs::Other(self),
        }
    }
}

/// Decode one ordinary or multipart engine-core output message into the strong
/// typed public protocol shape.
pub fn decode_engine_core_outputs<Frame>(frames: &[Frame]) -> Result<EngineCoreOutputs>
where
    Frame: AsRef<[u8]>,
{
    let first_frame = frames.first().ok_or_else(|| ext_value_decode!("missing output frame"))?;

    let mut outputs: EngineCoreOutputs = decode_msgpack(first_frame.as_ref())?;
    outputs.resolve_in_place(frames)?;
    Ok(outputs)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::protocol::output::EngineCoreOutput;
    use crate::protocol::{decode_msgpack, encode_msgpack};

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
    fn engine_core_outputs_classify_request_batch() {
        let outputs = EngineCoreOutputs {
            outputs: vec![EngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: vec![7],
                ..Default::default()
            }],
            finished_requests: Some(BTreeSet::from(["req-1".to_string()])),
            ..Default::default()
        };

        expect_test::expect![[r#"
            RequestBatch(
                RequestBatchOutputs {
                    engine_index: 0,
                    outputs: [
                        EngineCoreOutput {
                            request_id: "req-1",
                            new_token_ids: [
                                7,
                            ],
                            new_logprobs: None,
                            new_prompt_logprobs_tensors: None,
                            pooling_output: None,
                            finish_reason: None,
                            stop_reason: None,
                            events: None,
                            kv_transfer_params: None,
                            trace_headers: None,
                            prefill_stats: None,
                            routed_experts: None,
                            num_nans_in_logits: 0,
                        },
                    ],
                    scheduler_stats: None,
                    timestamp: 0.0,
                    finished_requests: Some(
                        {
                            "req-1",
                        },
                    ),
                },
            )
        "#]]
        .assert_debug_eq(&outputs.classify());
    }

    #[test]
    fn engine_core_outputs_classify_utility() {
        let outputs = EngineCoreOutputs {
            utility_output: Some(UtilityOutput {
                call_id: 42_u64.into(),
                failure_message: None,
                result: None,
            }),
            ..Default::default()
        };

        expect_test::expect![[r#"
            Utility(
                UtilityCallOutput {
                    engine_index: 0,
                    timestamp: 0.0,
                    output: UtilityOutput {
                        call_id: 42,
                        failure_message: None,
                        result: None,
                    },
                },
            )
        "#]]
        .assert_debug_eq(&outputs.classify());
    }

    #[test]
    fn engine_core_outputs_classify_control() {
        let outputs = EngineCoreOutputs {
            start_wave: Some(3),
            ..Default::default()
        };

        expect_test::expect![[r#"
            DpControl {
                engine_index: 0,
                timestamp: 0.0,
                control: StartWave(
                    3,
                ),
            }
        "#]]
        .assert_debug_eq(&outputs.classify());
    }

    #[test]
    fn engine_core_outputs_classify_mixed_shape_as_raw() {
        let outputs = EngineCoreOutputs {
            outputs: vec![EngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: vec![7],
                ..Default::default()
            }],
            utility_output: Some(UtilityOutput {
                call_id: 1_u64.into(),
                failure_message: None,
                result: None,
            }),
            ..Default::default()
        };

        expect_test::expect![[r#"
            Other(
                EngineCoreOutputs {
                    engine_index: 0,
                    outputs: [
                        EngineCoreOutput {
                            request_id: "req-1",
                            new_token_ids: [
                                7,
                            ],
                            new_logprobs: None,
                            new_prompt_logprobs_tensors: None,
                            pooling_output: None,
                            finish_reason: None,
                            stop_reason: None,
                            events: None,
                            kv_transfer_params: None,
                            trace_headers: None,
                            prefill_stats: None,
                            routed_experts: None,
                            num_nans_in_logits: 0,
                        },
                    ],
                    scheduler_stats: None,
                    timestamp: 0.0,
                    utility_output: Some(
                        UtilityOutput {
                            call_id: 1,
                            failure_message: None,
                            result: None,
                        },
                    ),
                    finished_requests: None,
                    wave_complete: None,
                    start_wave: None,
                },
            )
        "#]]
        .assert_debug_eq(&outputs.classify());
    }
}
