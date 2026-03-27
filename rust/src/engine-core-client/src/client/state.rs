use std::collections::BTreeMap;
use std::sync::atomic::{AtomicI64, Ordering};

use itertools::Itertools;
use tokio::sync::{mpsc, oneshot};

use crate::client::stream::EngineCoreStreamOutput;
use crate::error::{Error, Result};
use crate::protocol::{EngineCoreOutput, UtilityOutput};

pub type OutputSender = mpsc::UnboundedSender<Result<EngineCoreStreamOutput>>;
pub type OutputReceiver = mpsc::UnboundedReceiver<Result<EngineCoreStreamOutput>>;
pub type UtilitySender = oneshot::Sender<Result<UtilityOutput>>;
pub type UtilityReceiver = oneshot::Receiver<Result<UtilityOutput>>;

#[derive(Debug)]
struct TrackedRequest {
    sender: OutputSender,
    engine_idx: usize,
}

/// Internal registry for tracking active requests and their output stream senders.
///
/// This is used to route incoming outputs to the correct request stream, and to ensure proper
/// cleanup of senders when requests finish or the client shuts down.
#[derive(Debug)]
pub struct RequestRegistry {
    closed: bool,
    requests: BTreeMap<String, TrackedRequest>,
    in_flight_per_engine: Vec<usize>,
}

impl RequestRegistry {
    pub fn new(engine_count: usize) -> Self {
        Self {
            closed: false,
            requests: BTreeMap::default(),
            in_flight_per_engine: vec![0; engine_count],
        }
    }

    /// Register a newly added request. Create the per-request output channel bound to its
    /// `request_id` and return the selected engine index.
    pub fn register(&mut self, request_id: String) -> Result<(usize, OutputReceiver)> {
        if self.requests.contains_key(&request_id) {
            return Err(Error::DuplicateRequestId { request_id });
        }

        // Simple routing strategy: assign to the engine with the least in-flight requests.
        let engine_idx = self
            .in_flight_per_engine
            .iter()
            .position_min()
            .expect("request registry must contain at least one engine");

        let (tx, rx) = mpsc::unbounded_channel();
        self.requests.insert(
            request_id,
            TrackedRequest {
                sender: tx,
                engine_idx,
            },
        );
        self.in_flight_per_engine[engine_idx] += 1;

        Ok((engine_idx, rx))
    }

    /// Filter the given request IDs to the subset that are still tracked as active and can be
    /// aborted, grouped by engine.
    pub fn abortable_request_ids(&self, request_ids: &[String]) -> BTreeMap<usize, Vec<String>> {
        let mut by_engine = BTreeMap::new();
        for request_id in request_ids {
            let Some(tracked) = self.requests.get(request_id.as_str()) else {
                continue;
            };
            by_engine
                .entry(tracked.engine_idx)
                .or_insert_with(Vec::new)
                .push(request_id.clone());
        }
        by_engine
    }

    /// Obtain the stream sender for one output. If it indicates the request is finished, it will be
    /// removed from the registry.
    pub fn sender_for_output(&mut self, output: &EngineCoreOutput) -> Option<OutputSender> {
        if output.finished() {
            self.remove(output.request_id.as_str())
                .map(|tracked| tracked.0)
        } else {
            self.requests
                .get(output.request_id.as_str())
                .map(|tracked| tracked.sender.clone())
        }
    }

    /// Remove a batch of requests that have finished or aborted, returning their stream senders.
    pub fn finish_many<'a>(
        &mut self,
        request_ids: impl IntoIterator<Item = &'a String>,
    ) -> Vec<OutputSender> {
        request_ids
            .into_iter()
            .filter_map(|request_id| self.remove(request_id.as_str()).map(|tracked| tracked.0))
            .collect()
    }

    /// Mark the registry as closed, detach and return all tracked senders.
    pub fn close(&mut self) -> Vec<OutputSender> {
        if self.closed {
            return Vec::new();
        }

        self.closed = true;
        std::mem::take(&mut self.requests)
            .into_values()
            .map(|tracked| tracked.sender)
            .collect()
    }

    /// Remove one request from the local registry. Returns the tracked entry if it exists.
    #[must_use]
    pub fn remove(&mut self, request_id: &str) -> Option<(OutputSender, usize)> {
        let tracked = self.requests.remove(request_id)?;
        self.in_flight_per_engine[tracked.engine_idx] = self.in_flight_per_engine
            [tracked.engine_idx]
            .checked_sub(1)
            .expect("request registry must not underflow engine in-flight counts");
        Some((tracked.sender, tracked.engine_idx))
    }

    #[cfg(test)]
    pub fn contains(&self, request_id: &str) -> bool {
        self.requests.contains_key(request_id)
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }
}

/// Internal registry for tracking active utility calls and their waiting receivers.
#[derive(Debug)]
pub struct UtilityRegistry {
    closed: bool,
    next_call_id: AtomicI64,
    utility_calls: BTreeMap<i64, UtilitySender>,
}

impl Default for UtilityRegistry {
    fn default() -> Self {
        Self {
            closed: false,
            next_call_id: AtomicI64::new(1),
            utility_calls: BTreeMap::default(),
        }
    }
}

impl UtilityRegistry {
    /// Allocate the next utility `call_id` and register a newly added utility call.
    pub fn allocate_and_register(&mut self) -> (i64, UtilityReceiver) {
        let call_id = self.next_call_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.utility_calls.insert(call_id, tx);
        (call_id, rx)
    }

    /// Resolve a utility output to its waiting receiver.
    pub fn resolve(&mut self, output: UtilityOutput) -> Option<UtilitySender> {
        self.utility_calls.remove(&output.call_id)
    }

    /// Mark the registry as closed, detach and return all tracked senders.
    pub fn close(&mut self) -> Vec<UtilitySender> {
        if self.closed {
            return Vec::new();
        }

        self.closed = true;
        std::mem::take(&mut self.utility_calls)
            .into_values()
            .collect()
    }

    #[cfg(test)]
    pub fn contains(&self, call_id: i64) -> bool {
        self.utility_calls.contains_key(&call_id)
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }
}

#[cfg(test)]
mod tests {
    use super::{RequestRegistry, UtilityRegistry};
    use crate::protocol::{EngineCoreFinishReason, EngineCoreOutput, UtilityOutput};

    #[test]
    fn registry_rejects_duplicate_request_ids() {
        let mut registry = RequestRegistry::new(1);
        registry.register("req-1".to_string()).unwrap();
        let error = registry.register("req-1".to_string()).unwrap_err();
        assert!(matches!(
            error,
            crate::error::Error::DuplicateRequestId { request_id } if request_id == "req-1"
        ));
    }

    #[test]
    fn registry_removes_finished_request_on_output() {
        let mut registry = RequestRegistry::new(1);
        registry.register("req-1".to_string()).unwrap();

        let sender = registry.sender_for_output(&EngineCoreOutput {
            request_id: "req-1".to_string(),
            finish_reason: Some(EngineCoreFinishReason::Length),
            ..Default::default()
        });

        assert!(sender.is_some());
        assert!(!registry.contains("req-1"));
    }

    #[test]
    fn registry_closes_all_requests_on_failure() {
        let mut registry = RequestRegistry::new(1);
        registry.register("req-1".to_string()).unwrap();
        registry.register("req-2".to_string()).unwrap();

        let senders = registry.close();

        assert_eq!(senders.len(), 2);
        assert!(registry.is_closed());
    }

    #[test]
    fn registry_tracks_engine_idx_per_request() {
        let mut registry = RequestRegistry::new(2);
        let (engine_0, _) = registry.register("req-1".to_string()).unwrap();
        let (engine_1, _) = registry.register("req-2".to_string()).unwrap();
        let (engine_0_again, _) = registry.register("req-3".to_string()).unwrap();

        assert_eq!(engine_0, 0);
        assert_eq!(engine_1, 1);
        assert_eq!(engine_0_again, 0);

        let grouped = registry.abortable_request_ids(&[
            "req-1".to_string(),
            "req-2".to_string(),
            "req-3".to_string(),
        ]);
        assert_eq!(
            grouped.get(&0).unwrap(),
            &vec!["req-1".to_string(), "req-3".to_string()]
        );
        assert_eq!(grouped.get(&1).unwrap(), &vec!["req-2".to_string()]);
    }

    #[test]
    fn utility_registry_tracks_and_removes_call_ids() {
        let mut registry = UtilityRegistry::default();
        let (call_id_1, _) = registry.allocate_and_register();
        let (call_id_2, _) = registry.allocate_and_register();

        assert_eq!(call_id_1, 1);
        assert_eq!(call_id_2, 2);
        assert!(registry.contains(1));
        assert!(registry.contains(2));
        assert!(
            registry
                .resolve(UtilityOutput {
                    call_id: 1,
                    failure_message: None,
                    result: None,
                })
                .is_some()
        );
        assert!(!registry.contains(1));
        assert!(registry.contains(2));
    }

    #[test]
    fn utility_registry_closes_all_waiters_on_failure() {
        let mut registry = UtilityRegistry::default();
        registry.allocate_and_register();
        registry.allocate_and_register();

        let senders = registry.close();

        assert_eq!(senders.len(), 2);
        assert!(!registry.contains(1));
        assert!(!registry.contains(2));
        assert!(registry.is_closed());
    }
}
