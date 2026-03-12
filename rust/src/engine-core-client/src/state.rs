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

#[derive(Debug, Clone, PartialEq, Eq)]
enum ClientState {
    Running,
    Closed(ClientClosedState),
}

#[derive(Debug, Default)]
pub struct RequestRegistry {
    state: ClientState,
    requests: BTreeMap<String, RequestStreamSender>,
}

impl Default for ClientState {
    fn default() -> Self {
        Self::Running
    }
}

impl RequestRegistry {
    pub fn register(&mut self, request_id: String) -> Result<RequestStreamReceiver> {
        self.ensure_running()?;
        if self.requests.contains_key(&request_id) {
            return Err(Error::DuplicateRequestId { request_id });
        }

        let (tx, rx) = mpsc::unbounded_channel();
        self.requests.insert(request_id, tx);
        Ok(rx)
    }

    pub fn rollback(&mut self, request_id: &str) {
        self.requests.remove(request_id);
    }

    pub fn abortable_request_ids(&self, request_ids: &[String]) -> Result<Vec<String>> {
        self.ensure_running()?;
        Ok(request_ids
            .iter()
            .filter(|request_id| self.requests.contains_key(request_id.as_str()))
            .cloned()
            .collect())
    }

    pub fn sender_for_output(&mut self, output: &EngineCoreOutput) -> Option<RequestStreamSender> {
        if output.finished() {
            self.requests.remove(output.request_id.as_str())
        } else {
            self.requests.get(output.request_id.as_str()).cloned()
        }
    }

    pub fn finish_requests<'a>(
        &mut self,
        request_ids: impl IntoIterator<Item = &'a String>,
    ) -> Vec<RequestStreamSender> {
        request_ids
            .into_iter()
            .filter_map(|request_id| self.requests.remove(request_id.as_str()))
            .collect()
    }

    pub fn close(&mut self, closed_state: ClientClosedState) -> Vec<RequestStreamSender> {
        if matches!(self.state, ClientState::Closed(_)) {
            return Vec::new();
        }

        self.state = ClientState::Closed(closed_state);
        std::mem::take(&mut self.requests).into_values().collect()
    }

    pub fn remove_request(&mut self, request_id: &str) -> Option<RequestStreamSender> {
        self.requests.remove(request_id)
    }

    #[cfg(test)]
    pub fn contains(&self, request_id: &str) -> bool {
        self.requests.contains_key(request_id)
    }

    pub fn closed_state(&self) -> Option<&ClientClosedState> {
        match &self.state {
            ClientState::Running => None,
            ClientState::Closed(closed_state) => Some(closed_state),
        }
    }

    pub fn is_running(&self) -> bool {
        matches!(self.state, ClientState::Running)
    }

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
