use std::collections::BTreeMap;

use tokio::sync::mpsc;

use crate::client::stream::EngineCoreStreamOutput;
use crate::error::{Error, Result};
use crate::protocol::EngineCoreOutput;

pub type OutputSender = mpsc::UnboundedSender<Result<EngineCoreStreamOutput>>;
pub type OutputReceiver = mpsc::UnboundedReceiver<Result<EngineCoreStreamOutput>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClientClosedState {
    DispatcherFailed { reason: String },
    ClientShutdown { reason: String },
}

impl ClientClosedState {
    /// Convert the closed state into a corresponding error for reporting to active request streams.
    pub fn error(&self) -> Error {
        match self {
            Self::DispatcherFailed { reason } => Error::DispatcherClosed {
                reason: reason.clone(),
            },
            Self::ClientShutdown { reason } => Error::ClientClosed {
                reason: reason.clone(),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
enum ClientState {
    #[default]
    Running,
    Closed(ClientClosedState),
}

/// Internal registry for tracking active requests and their output stream senders.
///
/// This is used to route incoming outputs to the correct request stream, and to ensure proper
/// cleanup of senders when requests finish or the client shuts down.
#[derive(Debug, Default)]
pub struct RequestRegistry {
    state: ClientState,
    requests: BTreeMap<String, OutputSender>,
}

impl RequestRegistry {
    /// Register a newly added request. Create the per-request output channel bound to its
    /// `request_id`.
    pub fn register(&mut self, request_id: String) -> Result<OutputReceiver> {
        self.ensure_running()?;
        if self.requests.contains_key(&request_id) {
            return Err(Error::DuplicateRequestId { request_id });
        }

        let (tx, rx) = mpsc::unbounded_channel();
        self.requests.insert(request_id, tx);
        Ok(rx)
    }

    /// Filter the given request IDs to the subset that are still tracked as active and can be
    /// aborted.
    pub fn abortable_request_ids(&self, request_ids: &[String]) -> Result<Vec<String>> {
        self.ensure_running()?;
        Ok(request_ids
            .iter()
            .filter(|request_id| self.requests.contains_key(request_id.as_str()))
            .cloned()
            .collect())
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

    /// Mark the registry as closed with the given state, detach and return all tracked senders.
    /// This will reject any future operations on the registry.
    pub fn close(&mut self, closed_state: ClientClosedState) -> Vec<OutputSender> {
        if matches!(self.state, ClientState::Closed(_)) {
            // Already closed, no-op.
            return Vec::new();
        }

        self.state = ClientState::Closed(closed_state);
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

    /// Returns true if the registry is still running.
    pub fn is_running(&self) -> bool {
        matches!(self.state, ClientState::Running)
    }

    /// Ensure the state of the registry is still running, returning an error if it has been closed.
    fn ensure_running(&self) -> Result<()> {
        match &self.state {
            ClientState::Running => Ok(()),
            ClientState::Closed(closed_state) => Err(closed_state.error()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ClientClosedState, RequestRegistry};
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

        let senders = registry.close(ClientClosedState::DispatcherFailed {
            reason: "boom".to_string(),
        });

        assert_eq!(senders.len(), 2);
        assert!(!registry.is_running());
    }
}
