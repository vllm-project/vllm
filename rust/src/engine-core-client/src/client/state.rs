use std::collections::BTreeMap;

use tokio::sync::mpsc;

use crate::error::{Error, Result};
use crate::protocol::EngineCoreOutput;

pub type RequestStreamSender = mpsc::UnboundedSender<Result<EngineCoreOutput>>;
pub type RequestStreamReceiver = mpsc::UnboundedReceiver<Result<EngineCoreOutput>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClientClosedState {
    DispatcherFailed { reason: String },
    ClientShutdown { reason: String },
}

impl ClientClosedState {
    /// Convert the internal closed-state marker into the public client error
    /// that callers should observe after request lifecycle processing stops.
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

#[derive(Debug, Default)]
pub struct RequestRegistry {
    state: ClientState,
    requests: BTreeMap<String, RequestStreamSender>,
}

impl RequestRegistry {
    /// Register a newly added request before it is sent to the engine and
    /// create the per-request output channel bound to its `request_id`.
    pub fn register(&mut self, request_id: String) -> Result<RequestStreamReceiver> {
        self.ensure_running()?;
        if self.requests.contains_key(&request_id) {
            return Err(Error::DuplicateRequestId { request_id });
        }

        let (tx, rx) = mpsc::unbounded_channel();
        self.requests.insert(request_id, tx);
        Ok(rx)
    }

    /// Remove a request registration when `add_request()` fails after local
    /// state has already been installed.
    pub fn rollback(&mut self, request_id: &str) {
        self.requests.remove(request_id);
    }

    /// Filter a caller-provided abort list down to requests that are still
    /// locally tracked as in flight.
    pub fn abortable_request_ids(&self, request_ids: &[String]) -> Result<Vec<String>> {
        self.ensure_running()?;
        Ok(request_ids
            .iter()
            .filter(|request_id| self.requests.contains_key(request_id.as_str()))
            .cloned()
            .collect())
    }

    /// Look up the stream sender for one output and detach it immediately if
    /// this output marks the request finished.
    pub fn sender_for_output(&mut self, output: &EngineCoreOutput) -> Option<RequestStreamSender> {
        if output.finished() {
            self.requests.remove(output.request_id.as_str())
        } else {
            self.requests.get(output.request_id.as_str()).cloned()
        }
    }

    /// Close and remove requests that finished via the batched
    /// `finished_requests` signal rather than an inline final output object.
    pub fn finish_requests<'a>(
        &mut self,
        request_ids: impl IntoIterator<Item = &'a String>,
    ) -> Vec<RequestStreamSender> {
        request_ids
            .into_iter()
            .filter_map(|request_id| self.requests.remove(request_id.as_str()))
            .collect()
    }

    /// Transition the registry into a closed state and detach every active
    /// request stream for terminal error propagation.
    pub fn close(&mut self, closed_state: ClientClosedState) -> Vec<RequestStreamSender> {
        if matches!(self.state, ClientState::Closed(_)) {
            return Vec::new();
        }

        self.state = ClientState::Closed(closed_state);
        std::mem::take(&mut self.requests).into_values().collect()
    }

    /// Remove one request from local tracking, typically because its stream was
    /// dropped before the engine declared completion.
    pub fn remove_request(&mut self, request_id: &str) -> Option<RequestStreamSender> {
        self.requests.remove(request_id)
    }

    #[cfg(test)]
    pub fn contains(&self, request_id: &str) -> bool {
        self.requests.contains_key(request_id)
    }

    /// Return the terminal client state, if request lifecycle processing has
    /// already been shut down.
    pub fn closed_state(&self) -> Option<&ClientClosedState> {
        match &self.state {
            ClientState::Running => None,
            ClientState::Closed(closed_state) => Some(closed_state),
        }
    }

    /// Report whether request registration and dispatch are still allowed.
    pub fn is_running(&self) -> bool {
        matches!(self.state, ClientState::Running)
    }

    /// Reject request-lifecycle operations after the client has already been
    /// closed by shutdown or dispatcher failure.
    fn ensure_running(&self) -> Result<()> {
        if let Some(closed_state) = self.closed_state() {
            return Err(closed_state.error());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::protocol::{EngineCoreOutput, FinishReason};

    use super::{ClientClosedState, RequestRegistry};

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
