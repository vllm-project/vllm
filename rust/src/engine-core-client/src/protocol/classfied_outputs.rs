use std::collections::BTreeSet;

use enum_as_inner::EnumAsInner;

use super::{EngineCoreOutput, EngineCoreOutputs, OpaqueValue, UtilityOutput};

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
    pub scheduler_stats: Option<OpaqueValue>,
    pub timestamp: f64,
    pub finished_requests: Option<BTreeSet<String>>,
}

#[derive(Debug, Clone, PartialEq, EnumAsInner)]
pub enum OtherEngineCoreOutputs {
    Utility {
        engine_index: u32,
        timestamp: f64,
        utility_output: UtilityOutput,
    },
    DpControl {
        engine_index: u32,
        timestamp: f64,
        control: DpControlMessage,
    },
    /// Fallback for wire-shape combinations that do not map cleanly onto the
    /// current semantic families.
    Raw(EngineCoreOutputs),
}

/// Semantic classification of a raw `EngineCoreOutputs` message.
///
/// Python currently uses one product-shaped wire struct for several distinct
/// output families. This enum exposes those families more explicitly without
/// changing the wire format.
#[derive(Debug, Clone, PartialEq, EnumAsInner)]
pub enum ClassifiedEngineCoreOutputs {
    RequestBatch(RequestBatchOutputs),
    Other(OtherEngineCoreOutputs),
}

impl EngineCoreOutputs {
    /// Classify the raw wire message into a more semantic Rust enum.
    pub fn classify(self) -> ClassifiedEngineCoreOutputs {
        let raw = self.clone();
        let has_request_payload = !self.outputs.is_empty()
            || self.scheduler_stats.is_some()
            || self.finished_requests.is_some();

        match (
            has_request_payload,
            self.utility_output,
            self.wave_complete,
            self.start_wave,
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
            (false, Some(utility_output), None, None) => {
                ClassifiedEngineCoreOutputs::Other(OtherEngineCoreOutputs::Utility {
                    engine_index: self.engine_index,
                    timestamp: self.timestamp,
                    utility_output,
                })
            }
            (false, None, Some(wave), None) => {
                ClassifiedEngineCoreOutputs::Other(OtherEngineCoreOutputs::DpControl {
                    engine_index: self.engine_index,
                    timestamp: self.timestamp,
                    control: DpControlMessage::WaveComplete(wave),
                })
            }
            (false, None, None, Some(wave)) => {
                ClassifiedEngineCoreOutputs::Other(OtherEngineCoreOutputs::DpControl {
                    engine_index: self.engine_index,
                    timestamp: self.timestamp,
                    control: DpControlMessage::StartWave(wave),
                })
            }
            _ => ClassifiedEngineCoreOutputs::Other(OtherEngineCoreOutputs::Raw(raw)),
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
                            num_cached_tokens: 0,
                            num_external_computed_tokens: 0,
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
            Other(
                Utility {
                    engine_index: 0,
                    timestamp: 0.0,
                    utility_output: UtilityOutput {
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
            Other(
                DpControl {
                    engine_index: 0,
                    timestamp: 0.0,
                    control: StartWave(
                        3,
                    ),
                },
            )
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
                Raw(
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
                                num_cached_tokens: 0,
                                num_external_computed_tokens: 0,
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
                ),
            )
        "#]]
        .assert_debug_eq(&outputs.classify());
    }
}
