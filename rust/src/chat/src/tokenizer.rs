use std::sync::Arc;

use crate::error::Result;

pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>>;
    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String>;
}

pub type DynTokenizer = Arc<dyn Tokenizer>;
