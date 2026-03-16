use std::sync::Arc;

use vllm_chat::ChatLlm;

#[derive(Clone)]
pub struct AppState {
    pub model_id: String,
    pub chat: Arc<ChatLlm>,
}

impl AppState {
    pub fn new(model_id: impl Into<String>, chat: Arc<ChatLlm>) -> Self {
        Self {
            model_id: model_id.into(),
            chat,
        }
    }
}
