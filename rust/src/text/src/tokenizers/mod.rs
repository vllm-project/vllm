use std::sync::Arc;

use crate::error::Result;
use crate::incremental::{DecodeStream, IncrementalDecoder};

mod hf;
mod tekken;
mod tiktoken;

pub use hf::HuggingFaceTokenizer;
pub use tekken::TekkenTokenizer;
pub use tiktoken::TiktokenTokenizer;

pub trait Tokenizer: Send + Sync {
    /// Encode one prompt string into token IDs.
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>>;

    /// Decode one token sequence into text.
    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String>;

    /// Convert one token string into a token ID, returning `None` if the token is not in the
    /// tokenizer vocabulary.
    fn token_to_id(&self, token: &str) -> Option<u32>;

    /// Create a stateful incremental decoder primed with the given prompt tokens.
    ///
    /// The prompt tokens provide left context for the first generated token; the decoder does not
    /// re-emit prompt text.
    fn create_decode_stream(
        &self,
        prompt_token_ids: &[u32],
        skip_special_tokens: bool,
        min_bytes_to_buffer: usize,
    ) -> Box<dyn IncrementalDecoder + '_> {
        Box::new(DecodeStream::new(
            self,
            prompt_token_ids,
            skip_special_tokens,
            min_bytes_to_buffer,
        ))
    }
}

pub type DynTokenizer = Arc<dyn Tokenizer>;
