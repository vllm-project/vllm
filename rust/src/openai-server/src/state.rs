use std::sync::Arc;

use anyhow::Context as _;
use vllm_chat::ChatLlm;

/// Shared router state for the minimal single-model OpenAI server.
#[derive(Clone)]
pub struct AppState {
    /// Public model ID returned by `/v1/models` and validated on chat requests.
    pub model_id: String,
    /// Shared text-only chat facade used by all requests.
    pub chat: Arc<ChatLlm>,
}

impl AppState {
    /// Construct one application state instance.
    pub fn new(model_id: impl Into<String>, chat: Arc<ChatLlm>) -> Self {
        Self {
            model_id: model_id.into(),
            chat,
        }
    }

    /// Shutdown the app. Caller should ensure that no outstanding references to the state remain
    /// before calling this method.
    pub async fn shutdown(self: Arc<Self>) -> anyhow::Result<()> {
        let state = Arc::try_unwrap(self)
            .ok()
            .context("openai server state still has outstanding references")?;
        let chat = Arc::try_unwrap(state.chat)
            .ok()
            .context("openai server chat still has outstanding references")?;

        chat.shutdown().await?;
        Ok(())
    }
}
