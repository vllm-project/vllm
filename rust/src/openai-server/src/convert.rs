use std::time::{SystemTime, UNIX_EPOCH};

mod chat_completions;
mod completions;
mod validate;

pub use chat_completions::prepare_chat_request;
pub use completions::{CompletionRequest, prepare_completion_request};

/// Return the current Unix timestamp in seconds for OpenAI response objects.
pub fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}
