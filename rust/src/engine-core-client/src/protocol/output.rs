use std::collections::BTreeSet;

use enum_as_inner::EnumAsInner;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_default::DefaultFromSerde;
use serde_repr::{Deserialize_repr, Serialize_repr};
use serde_tuple::{Deserialize_tuple, Serialize_tuple};

use super::utility::UtilityOutput;
use crate::error::{Error, Result, ext_value_decode};
use crate::protocol::logprobs::MaybeWireLogprobs;
use crate::protocol::notifications::EngineNotification;
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

/// Raw Python/msgpack engine-core output envelope.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/f22d6e026798a74e6542a52ef776c054f2de572a/vllm/v1/engine/__init__.py#L186-L214>
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple, DefaultFromSerde)]
struct WireEngineCoreOutputs {
    #[serde(default)]
    engine_index: u32,
    /// Outputs grouped for this client in the current engine tick.
    #[serde(default)]
    outputs: Vec<EngineCoreOutput>,
    #[serde(default)]
    scheduler_stats: Option<Box<SchedulerStats>>,
    #[serde(default)]
    timestamp: f64,
    #[serde(default)]
    utility_output: Option<UtilityOutput>,
    #[serde(default)]
    finished_requests: Option<BTreeSet<String>>,
    /// In DP mode, signals that the current wave finished and engines are
    /// paused.
    #[serde(default)]
    wave_complete: Option<u32>,
    /// In DP mode, signals that a request arrived for an old wave and the next
    /// wave needs to start in other engines.
    #[serde(default)]
    start_wave: Option<u32>,
    /// Rare engine-level event notifications (see `notifications.rs`).
    #[serde(default)]
    engine_notifications: Option<Vec<EngineNotification>>,
}

/// Data-parallel control notifications multiplexed through `EngineCoreOutputs`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DpControlMessage {
    WaveComplete(u32),
    StartWave(u32),
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct RequestBatchOutputs {
    pub engine_index: u32,
    pub outputs: Vec<EngineCoreOutput>,
    pub scheduler_stats: Option<Box<SchedulerStats>>,
    pub timestamp: f64,
    pub finished_requests: Option<BTreeSet<String>>,
    pub engine_notifications: Option<Vec<EngineNotification>>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct UtilityCallOutput {
    pub engine_index: u32,
    pub timestamp: f64,
    pub output: UtilityOutput,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DpControlOutput {
    pub engine_index: u32,
    pub timestamp: f64,
    pub control: DpControlMessage,
}

/// Semantic engine-core output families.
///
/// Python currently uses one product-shaped wire struct. The Rust protocol
/// exposes the finite semantic families while preserving the same msgpack shape
/// for serialization.
#[derive(Debug, Clone, PartialEq, EnumAsInner)]
pub enum EngineCoreOutputs {
    RequestBatch(RequestBatchOutputs),
    Utility(UtilityCallOutput),
    DpControl(DpControlOutput),
}

impl From<RequestBatchOutputs> for EngineCoreOutputs {
    fn from(outputs: RequestBatchOutputs) -> Self {
        Self::RequestBatch(outputs)
    }
}

impl From<UtilityCallOutput> for EngineCoreOutputs {
    fn from(output: UtilityCallOutput) -> Self {
        Self::Utility(output)
    }
}

impl From<DpControlOutput> for EngineCoreOutputs {
    fn from(output: DpControlOutput) -> Self {
        Self::DpControl(output)
    }
}

impl EngineCoreOutputs {
    /// Resolve all wire-format fields in-place by looking up aux frames and
    /// decoding raw-view payloads as needed.
    fn resolve_in_place<Frame>(&mut self, frames: &[Frame]) -> Result<()>
    where
        Frame: AsRef<[u8]>,
    {
        if let Self::RequestBatch(batch) = self {
            for output in &mut batch.outputs {
                output.resolve_in_place(frames)?;
            }
        }
        Ok(())
    }
}

/// Classify the raw wire message into a more semantic Rust enum.
impl TryFrom<WireEngineCoreOutputs> for EngineCoreOutputs {
    type Error = Error;

    fn try_from(value: WireEngineCoreOutputs) -> Result<Self> {
        let has_request_payload = !value.outputs.is_empty()
            || value.scheduler_stats.is_some()
            || value.finished_requests.is_some()
            || value.engine_notifications.is_some();

        match (
            has_request_payload,
            &value.utility_output,
            &value.wave_complete,
            &value.start_wave,
        ) {
            (true, None, None, None) => Ok(RequestBatchOutputs {
                engine_index: value.engine_index,
                outputs: value.outputs,
                scheduler_stats: value.scheduler_stats,
                timestamp: value.timestamp,
                finished_requests: value.finished_requests,
                engine_notifications: value.engine_notifications,
            }
            .into()),
            (false, Some(_), None, None) => Ok(UtilityCallOutput {
                engine_index: value.engine_index,
                timestamp: value.timestamp,
                output: value.utility_output.unwrap(),
            }
            .into()),
            (false, None, Some(_), None) => Ok(DpControlOutput {
                engine_index: value.engine_index,
                timestamp: value.timestamp,
                control: DpControlMessage::WaveComplete(value.wave_complete.unwrap()),
            }
            .into()),
            (false, None, None, Some(_)) => Ok(DpControlOutput {
                engine_index: value.engine_index,
                timestamp: value.timestamp,
                control: DpControlMessage::StartWave(value.start_wave.unwrap()),
            }
            .into()),

            _ => Err(Error::Decode {
                target_type: "EngineCoreOutputs",
                message: "invalid wire shape".to_string(),
            }),
        }
    }
}

impl From<EngineCoreOutputs> for WireEngineCoreOutputs {
    fn from(value: EngineCoreOutputs) -> Self {
        match value {
            EngineCoreOutputs::RequestBatch(batch) => Self {
                engine_index: batch.engine_index,
                outputs: batch.outputs,
                scheduler_stats: batch.scheduler_stats,
                timestamp: batch.timestamp,
                finished_requests: batch.finished_requests,
                engine_notifications: batch.engine_notifications,
                ..Default::default()
            },
            EngineCoreOutputs::Utility(utility) => Self {
                engine_index: utility.engine_index,
                timestamp: utility.timestamp,
                utility_output: Some(utility.output),
                ..Default::default()
            },
            EngineCoreOutputs::DpControl(control) => {
                let (wave_complete, start_wave) = match control.control {
                    DpControlMessage::WaveComplete(wave) => (Some(wave), None),
                    DpControlMessage::StartWave(wave) => (None, Some(wave)),
                };
                Self {
                    engine_index: control.engine_index,
                    timestamp: control.timestamp,
                    wave_complete,
                    start_wave,
                    ..Default::default()
                }
            }
        }
    }
}

impl Serialize for EngineCoreOutputs {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        WireEngineCoreOutputs::from(self.clone()).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for EngineCoreOutputs {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        WireEngineCoreOutputs::deserialize(deserializer)?
            .try_into()
            .map_err(serde::de::Error::custom)
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
    use crate::protocol::notifications::LoraLoadEvent;
    use crate::protocol::output::EngineCoreOutput;
    use crate::protocol::{decode_msgpack, encode_msgpack};

    #[test]
    fn engine_core_outputs_roundtrip_finished_fields() {
        let outputs = WireEngineCoreOutputs {
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
        let decoded: WireEngineCoreOutputs = decode_msgpack(&encoded).unwrap();

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
        let outputs = WireEngineCoreOutputs {
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
                    engine_notifications: None,
                },
            )
        "#]]
        .assert_debug_eq(&EngineCoreOutputs::try_from(outputs).unwrap());
    }

    #[test]
    fn engine_core_outputs_classify_event_only_as_request_batch() {
        let outputs = WireEngineCoreOutputs {
            engine_notifications: Some(vec![EngineNotification::LoraLoadEvent(
                LoraLoadEvent::default(),
            )]),
            ..Default::default()
        };

        expect_test::expect![[r#"
            RequestBatch(
                RequestBatchOutputs {
                    engine_index: 0,
                    outputs: [],
                    scheduler_stats: None,
                    timestamp: 0.0,
                    finished_requests: None,
                    engine_notifications: Some(
                        [
                            LoraLoadEvent(
                                LoraLoadEvent {
                                    gpu_adapters: [],
                                    cpu_adapters: [],
                                    pinned_adapters: [],
                                },
                            ),
                        ],
                    ),
                },
            )
        "#]]
        .assert_debug_eq(&EngineCoreOutputs::try_from(outputs).unwrap());
    }

    #[test]
    fn engine_core_outputs_classify_utility() {
        let outputs = WireEngineCoreOutputs {
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
        .assert_debug_eq(&EngineCoreOutputs::try_from(outputs).unwrap());
    }

    #[test]
    fn engine_core_outputs_classify_control() {
        let outputs = WireEngineCoreOutputs {
            start_wave: Some(3),
            ..Default::default()
        };

        expect_test::expect![[r#"
            DpControl(
                DpControlOutput {
                    engine_index: 0,
                    timestamp: 0.0,
                    control: StartWave(
                        3,
                    ),
                },
            )
        "#]]
        .assert_debug_eq(&EngineCoreOutputs::try_from(outputs).unwrap());
    }

    #[test]
    fn engine_core_outputs_rejects_mixed_shape() {
        let outputs = WireEngineCoreOutputs {
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

        let error = EngineCoreOutputs::try_from(outputs).unwrap_err();
        expect_test::expect![[
            r#"messagepack decode failed for EngineCoreOutputs: invalid wire shape"#
        ]]
        .assert_eq(&error.to_string());
    }
}
