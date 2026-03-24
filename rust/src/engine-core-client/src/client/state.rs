use std::collections::BTreeMap;
use std::sync::atomic::{AtomicI64, Ordering};

use tokio::sync::{mpsc, oneshot};

use crate::client::stream::EngineCoreStreamOutput;
use crate::error::{Error, Result};
use crate::protocol::{EngineCoreOutput, UtilityOutput};

pub type OutputSender = mpsc::UnboundedSender<Result<EngineCoreStreamOutput>>;
pub type OutputReceiver = mpsc::UnboundedReceiver<Result<EngineCoreStreamOutput>>;
pub type UtilitySender = oneshot::Sender<Result<UtilityOutput>>;
pub type UtilityReceiver = oneshot::Receiver<Result<UtilityOutput>>;

/// Internal registry for tracking active requests and their output stream senders.
///
/// This is used to route incoming outputs to the correct request stream, and to ensure proper
/// cleanup of senders when requests finish or the client shuts down.
#[derive(Debug, Default)]
pub struct RequestRegistry {
    closed: bool,
    requests: BTreeMap<String, OutputSender>,
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

    /// Remove one utility call from the local registry. Returns the corresponding sender if exists.
    #[must_use]
    pub fn remove(&mut self, call_id: i64) -> Option<UtilitySender> {
        self.utility_calls.remove(&call_id)
    }

    #[cfg(test)]
    pub fn contains(&self, call_id: i64) -> bool {
        self.utility_calls.contains_key(&call_id)
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }
}

impl RequestRegistry {
    /// Register a newly added request. Create the per-request output channel bound to its
    /// `request_id`.
    pub fn register(&mut self, request_id: String) -> Result<OutputReceiver> {
        if self.requests.contains_key(&request_id) {
            return Err(Error::DuplicateRequestId { request_id });
        }

        let (tx, rx) = mpsc::unbounded_channel();
        self.requests.insert(request_id, tx);
        Ok(rx)
    }

    /// Filter the given request IDs to the subset that are still tracked as active and can be
    /// aborted.
    pub fn abortable_request_ids(&self, request_ids: &[String]) -> Vec<String> {
        request_ids
            .iter()
            .filter(|request_id| self.requests.contains_key(request_id.as_str()))
            .cloned()
            .collect()
    }

    /// Obtain the stream sender for one output. If it indicates the request is finished, it will be
    /// removed from the registry.
    pub fn sender_for_output(&mut self, output: &EngineCoreOutput) -> Option<OutputSender> {
        if output.finished() {
            self.requests.remove(output.request_id.as_str())
        } else {
            self.requests.get(output.request_id.as_str()).cloned()
        }
    }

    /// Remove a batch of requests that have finished or aborted, returning their stream senders.
    pub fn finish_many<'a>(
        &mut self,
        request_ids: impl IntoIterator<Item = &'a String>,
    ) -> Vec<OutputSender> {
        request_ids
            .into_iter()
            .filter_map(|request_id| self.requests.remove(request_id.as_str()))
            .collect()
    }

    /// Mark the registry as closed, detach and return all tracked senders.
    pub fn close(&mut self) -> Vec<OutputSender> {
        if self.closed {
            return Vec::new();
        }

        self.closed = true;
        std::mem::take(&mut self.requests).into_values().collect()
    }

    /// Remove one request from the local registry. Returns the corresponding sender if exists.
    #[must_use]
    pub fn remove(&mut self, request_id: &str) -> Option<OutputSender> {
        self.requests.remove(request_id)
    }

    #[cfg(test)]
    pub fn contains(&self, request_id: &str) -> bool {
        self.requests.contains_key(request_id)
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }
}

#[cfg(test)]
mod tests {
    use super::{RequestRegistry, UtilityRegistry};
    use crate::protocol::{EngineCoreOutput, FinishReason};

    #[test]
    fn registry_rejects_duplicate_request_ids() {
        let mut registry = RequestRegistry::default();
        registry.register("req-1".to_string()).unwrap();
        let error = registry.register("req-1".to_string()).unwrap_err();
        assert!(matches!(
            error,
            crate::error::Error::DuplicateRequestId { request_id } if request_id == "req-1"
        ));
    }

    #[test]
    fn registry_removes_finished_request_on_output() {
        let mut registry = RequestRegistry::default();
        registry.register("req-1".to_string()).unwrap();

        let sender = registry.sender_for_output(&EngineCoreOutput {
            request_id: "req-1".to_string(),
            finish_reason: Some(FinishReason::Length),
            ..Default::default()
        });

        assert!(sender.is_some());
        assert!(!registry.contains("req-1"));
    }

    #[test]
    fn registry_closes_all_requests_on_failure() {
        let mut registry = RequestRegistry::default();
        registry.register("req-1".to_string()).unwrap();
        registry.register("req-2".to_string()).unwrap();

        let senders = registry.close();

        assert_eq!(senders.len(), 2);
        assert!(registry.is_closed());
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
        assert!(registry.remove(1).is_some());
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
