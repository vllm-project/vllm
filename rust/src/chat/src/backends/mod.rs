use vllm_text::DynTextBackend;

use crate::DynChatBackend;
use crate::error::Result;

pub mod hf;

/// Shared backends loaded from a model id.
pub struct LoadedModelBackends {
    pub text_backend: DynTextBackend,
    pub chat_backend: DynChatBackend,
}

/// Load text and chat backends for the given model id.
pub async fn load_model_backends(model_id: &str) -> Result<LoadedModelBackends> {
    // Currently, we only have HuggingFace backends.
    hf::load_model_backends(model_id).await
}
