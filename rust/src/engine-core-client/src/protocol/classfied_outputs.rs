use std::collections::BTreeSet;

use enum_as_inner::EnumAsInner;

use super::{EngineCoreOutput, EngineCoreOutputs, UtilityOutput};
use crate::protocol::stats::SchedulerStats;

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

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::protocol::EngineCoreOutput;

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
                call_id: 42,
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
                call_id: 1,
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
