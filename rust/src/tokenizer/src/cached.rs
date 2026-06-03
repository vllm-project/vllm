use std::path::PathBuf;

use crate::{Result, Tokenizer};
use llm_tokenizer::cache::CacheConfig as LlmCacheConfigInner;
use llm_tokenizer::{
    CachedTokenizer, Decoder, Encoder, TokenizerTrait, create_tokenizer_from_file,
};

/// Configuration for the llm-tokenizer cache layer.
#[derive(Debug, Clone)]
pub struct LlmCacheConfig {
    /// Enable L0 (whole-string exact match) cache.
    pub enable_l0: bool,
    /// Maximum number of entries in L0 cache.
    pub l0_max_entries: usize,
    /// Enable L1 (prefix matching) cache.
    pub enable_l1: bool,
    /// Maximum memory for L1 cache in bytes.
    pub l1_max_memory: usize,
}

impl Default for LlmCacheConfig {
    fn default() -> Self {
        Self {
            enable_l0: true,
            l0_max_entries: 10_000,
            enable_l1: false,
            l1_max_memory: 50 * 1024 * 1024,
        }
    }
}

impl From<LlmCacheConfig> for LlmCacheConfigInner {
    fn from(c: LlmCacheConfig) -> Self {
        LlmCacheConfigInner {
            enable_l0: c.enable_l0,
            l0_max_entries: c.l0_max_entries,
            enable_l1: c.enable_l1,
            l1_max_memory: c.l1_max_memory,
        }
    }
}

pub struct LlmCachedTokenizer {
    /// The llm-tokenizer CachedTokenizer wrapping our adapter.
    cached: CachedTokenizer,
}

impl LlmCachedTokenizer {
    pub fn new(tokenizer_path: &PathBuf, cache_config: LlmCacheConfig) -> Self {
        let tokenizers_file_path = &tokenizer_path.to_string_lossy();
        let base_tokenizer = create_tokenizer_from_file(tokenizers_file_path).unwrap_or_else(|_| {
            panic!(
                "Failed to create base tokenizer for LlmCachedTokenizer (path: {tokenizers_file_path})"
            )
            });
        let llm_config: LlmCacheConfigInner = cache_config.into();
        tracing::info!(
            enable_l0 = llm_config.enable_l0,
            l0_max_entries = llm_config.l0_max_entries,
            enable_l1 = llm_config.enable_l1,
            l1_max_memory = llm_config.l1_max_memory,
            "LlmCachedTokenizer created with llm-tokenizer cache"
        );
        let cached = CachedTokenizer::new(base_tokenizer, llm_config);
        Self { cached }
    }
}

impl Tokenizer for LlmCachedTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self
            .cached
            .encode(text, add_special_tokens)
            .map_err(|e| crate::error::TokenizerError(format!("{e}")))?;
        let token_ids = encoding.token_ids().to_vec();
        Ok(token_ids)
    }

    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.cached
            .decode(token_ids, skip_special_tokens)
            .map_err(|e| crate::error::TokenizerError(format!("{e}")))
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.cached.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.cached.id_to_token(id)
    }

    fn is_special_id(&self, _: u32) -> bool {
        false
    }
}
