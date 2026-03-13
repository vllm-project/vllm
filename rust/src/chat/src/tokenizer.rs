use std::sync::Arc;

use crate::error::Result;

/// Minimal tokenizer interface needed by `vllm-chat`.
pub trait Tokenizer: Send + Sync {
    /// Encode one prompt string into token IDs.
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>>;
    /// Decode one cumulative token sequence into text.
    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String>;
}

/// Shared trait-object form of [`Tokenizer`].
pub type DynTokenizer = Arc<dyn Tokenizer>;
