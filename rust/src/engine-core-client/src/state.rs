use std::collections::BTreeSet;

use crate::protocol::EngineCoreOutputs;

#[derive(Debug, Default)]
pub struct RequestTracker {
    in_flight: BTreeSet<String>,
}

impl RequestTracker {
    pub fn insert(&mut self, request_id: String) {
        self.in_flight.insert(request_id);
    }

    pub fn retain_abortable(&self, request_ids: &[String]) -> Vec<String> {
        request_ids
            .iter()
            .filter(|request_id| self.in_flight.contains(request_id.as_str()))
            .cloned()
            .collect()
    }

    pub fn observe_outputs(&mut self, outputs: &EngineCoreOutputs) {
        if let Some(finished_requests) = &outputs.finished_requests {
            for request_id in finished_requests {
                self.in_flight.remove(request_id);
            }
        }

        for output in &outputs.outputs {
            if output.finished() {
                self.in_flight.remove(output.request_id.as_str());
            }
        }
    }

    #[cfg(test)]
    pub fn contains(&self, request_id: &str) -> bool {
        self.in_flight.contains(request_id)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crate::protocol::{EngineCoreOutput, EngineCoreOutputs, FinishReason};

    use super::RequestTracker;

    #[test]
    fn tracker_removes_finished_requests_from_both_sources() {
        let mut tracker = RequestTracker::default();
        tracker.insert("req-1".to_string());
        tracker.insert("req-2".to_string());

        tracker.observe_outputs(&EngineCoreOutputs {
            outputs: vec![EngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: vec![1],
                new_logprobs: None,
                new_prompt_logprobs_tensors: None,
                pooling_output: None,
                finish_reason: Some(FinishReason::Abort),
                stop_reason: None,
                events: None,
                kv_transfer_params: None,
                trace_headers: None,
                num_cached_tokens: 0,
                num_external_computed_tokens: 0,
                routed_experts: None,
                num_nans_in_logits: 0,
            }],
            finished_requests: Some(BTreeSet::from(["req-2".to_string()])),
            ..Default::default()
        });

        assert!(!tracker.contains("req-1"));
        assert!(!tracker.contains("req-2"));
    }
}
