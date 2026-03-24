use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Context as _;
use vllm_chat::ChatLlm;

/// Shared router state for the minimal single-model OpenAI server.
pub struct AppState {
    /// Public model ID returned by `/v1/models` and validated on chat requests.
    pub model_id: String,
    /// Shared chat facade used by all requests.
    pub chat: ChatLlm,
    /// Number of in-flight inference requests currently owned by this frontend.
    server_load: AtomicU64,
}

impl AppState {
    /// Construct one application state instance.
    pub fn new(model_id: impl Into<String>, chat: ChatLlm) -> Self {
        Self {
            model_id: model_id.into(),
            chat,
            server_load: AtomicU64::new(0),
        }
    }

    /// Return the current in-flight inference request count for the `/load` endpoint.
    pub fn server_load(&self) -> u64 {
        self.server_load.load(Ordering::Relaxed)
    }

    /// Increment the in-flight inference request count, called by the load tracking middleware.
    pub(crate) fn increment_server_load(&self) {
        self.server_load.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement the in-flight inference request count, called by the load tracking middleware.
    pub(crate) fn decrement_server_load(&self) {
        self.server_load.fetch_sub(1, Ordering::Relaxed);
    }

    /// Shutdown the app. Caller should ensure that no outstanding references to the state remain
    /// before calling this method.
    pub async fn shutdown(self: Arc<Self>) -> anyhow::Result<()> {
        let state = Arc::try_unwrap(self)
            .ok()
            .context("openai server state still has outstanding references")?;
        state.chat.shutdown().await?;
        Ok(())
    }
}
