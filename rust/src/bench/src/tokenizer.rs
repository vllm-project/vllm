// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashSet;
use std::future::Future;
use std::path::Path;

use thiserror_ext::AsReport as _;
use tokenizers::Tokenizer;

use crate::error::{BenchError, Result};
use crate::tiktoken::TiktokenTokenizer;

/// Abstraction over local HuggingFace tokenizer, tiktoken, or server-side tokenization.
pub enum TokenizerKind {
    Local(Box<Tokenizer>),
    Tiktoken(TiktokenTokenizer),
    Server(ServerTokenizer),
}

/// Server-side tokenizer using vLLM's /tokenize and /detokenize endpoints.
pub struct ServerTokenizer {
    client: reqwest::Client,
    runtime: tokio::runtime::Handle,
    tokenize_url: String,
    detokenize_url: String,
    model: String,
    cached_vocab_size: u32,
}

impl ServerTokenizer {
    /// Create a new server tokenizer and verify connectivity.
    pub async fn new(base_url: &str, model: &str) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| BenchError::Tokenizer(format!("Failed to build HTTP client: {e}")))?;

        let tokenize_url = format!("{base_url}/tokenize");
        let detokenize_url = format!("{base_url}/detokenize");

        let st = Self {
            client,
            runtime: tokio::runtime::Handle::current(),
            tokenize_url,
            detokenize_url,
            model: model.to_string(),
            cached_vocab_size: 0,
        };

        // Probe the endpoint to verify it works and discover vocab size
        let test_tokens = st.encode_async("test").await?;
        let max_id = test_tokens.iter().copied().max().unwrap_or(0);
        let estimated_vocab = (max_id * 2).max(131072);

        Ok(Self {
            cached_vocab_size: estimated_vocab,
            ..st
        })
    }

    fn encode_inner(&self, text: &str) -> Result<Vec<u32>> {
        self.block_on(self.encode_async(text))
    }

    async fn encode_async(&self, text: &str) -> Result<Vec<u32>> {
        let payload = serde_json::json!({
            "model": self.model,
            "prompt": text,
        });

        let resp = self
            .client
            .post(&self.tokenize_url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| BenchError::Tokenizer(format!("Server tokenize failed: {e}")))?;

        if !resp.status().is_success() {
            return Err(BenchError::Tokenizer(format!(
                "Server tokenize returned HTTP {}",
                resp.status()
            )));
        }

        let data: serde_json::Value = resp.json().await.map_err(|e| {
            BenchError::Tokenizer(format!("Failed to parse tokenize response: {e}"))
        })?;

        let tokens = data
            .get("tokens")
            .and_then(|t| t.as_array())
            .ok_or_else(|| BenchError::Tokenizer("Missing 'tokens' in tokenize response".into()))?;

        tokens
            .iter()
            .map(|v| {
                v.as_u64()
                    .map(|id| id as u32)
                    .ok_or_else(|| BenchError::Tokenizer("Invalid token ID in response".into()))
            })
            .collect()
    }

    fn decode_inner(&self, ids: &[u32]) -> Result<String> {
        self.block_on(self.decode_async(ids))
    }

    async fn decode_async(&self, ids: &[u32]) -> Result<String> {
        let payload = serde_json::json!({
            "model": self.model,
            "tokens": ids,
        });

        let resp = self
            .client
            .post(&self.detokenize_url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| BenchError::Tokenizer(format!("Server detokenize failed: {e}")))?;

        if !resp.status().is_success() {
            return Err(BenchError::Tokenizer(format!(
                "Server detokenize returned HTTP {}",
                resp.status()
            )));
        }

        let data: serde_json::Value = resp.json().await.map_err(|e| {
            BenchError::Tokenizer(format!("Failed to parse detokenize response: {e}"))
        })?;

        data.get("prompt")
            .and_then(|p| p.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| BenchError::Tokenizer("Missing 'prompt' in detokenize response".into()))
    }

    fn block_on<T>(&self, future: impl Future<Output = Result<T>>) -> Result<T> {
        if matches!(
            self.runtime.runtime_flavor(),
            tokio::runtime::RuntimeFlavor::CurrentThread
        ) {
            return Err(BenchError::Tokenizer(
                "Server tokenizer fallback requires a multi-thread Tokio runtime".into(),
            ));
        }

        // Sync tokenizer calls can come from a Tokio worker or a Rayon worker.
        // Tokio workers must enter a blocking region before re-entering the runtime;
        // Rayon workers can drive the future directly with the saved runtime handle.
        if tokio::runtime::Handle::try_current().is_ok() {
            tokio::task::block_in_place(|| self.runtime.block_on(future))
        } else {
            self.runtime.block_on(future)
        }
    }
}

// --- TokenizerKind methods ---

impl TokenizerKind {
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        match self {
            TokenizerKind::Local(tok) => tok
                .encode(text, add_special_tokens)
                .map(|enc| enc.get_ids().to_vec())
                .map_err(|e| BenchError::Tokenizer(format!("Encode failed: {e}"))),
            TokenizerKind::Tiktoken(tok) => Ok(tok.encode(text)),
            TokenizerKind::Server(srv) => srv.encode_inner(text),
        }
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        match self {
            TokenizerKind::Local(tok) => tok
                .decode(ids, skip_special_tokens)
                .map_err(|e| BenchError::Tokenizer(format!("Decode failed: {e}"))),
            TokenizerKind::Tiktoken(tok) => tok.decode(ids),
            TokenizerKind::Server(srv) => srv.decode_inner(ids),
        }
    }

    pub fn vocab_size(&self) -> u32 {
        match self {
            TokenizerKind::Local(tok) => tok.get_vocab_size(true) as u32,
            TokenizerKind::Tiktoken(tok) => tok.vocab_size(),
            TokenizerKind::Server(srv) => srv.cached_vocab_size,
        }
    }

    pub fn num_special_tokens_to_add(&self) -> usize {
        match self {
            TokenizerKind::Local(tok) => match tok.encode("", true) {
                Ok(enc) => enc.get_ids().len(),
                Err(_) => 0,
            },
            TokenizerKind::Tiktoken(_) | TokenizerKind::Server(_) => 0,
        }
    }

    pub fn get_allowed_tokens(&self) -> Vec<u32> {
        match self {
            TokenizerKind::Local(tok) => {
                let vs = tok.get_vocab_size(true) as u32;
                let mut special_ids = HashSet::new();
                for (id, token) in tok.get_added_tokens_decoder() {
                    if token.special {
                        special_ids.insert(id);
                    }
                }
                (0..vs).filter(|id| !special_ids.contains(id)).collect()
            }
            TokenizerKind::Tiktoken(tok) => tok.get_allowed_tokens(),
            TokenizerKind::Server(srv) => (0..srv.cached_vocab_size).collect(),
        }
    }
}

/// Load a tokenizer with fallback chain:
/// 0. Built-in tiktoken encoding (o200k_base, cl100k_base, etc.) — no download needed
/// 1. Local tokenizer.json (HuggingFace fast tokenizer)
/// 2. Tiktoken model file (for Kimi, Qwen, etc.)
/// 3. Server-side /tokenize + /detokenize endpoints
///
/// `server_info` is `Some((base_url, model))` to enable server-side fallback.
pub async fn load_tokenizer(
    model_id: &str,
    _trust_remote_code: bool,
    server_info: Option<(&str, &str)>,
) -> Result<TokenizerKind> {
    // 0. Check for built-in tiktoken encoding names (no HF download needed). These are useful for
    //    consistent cross-model token counting (e.g. Artificial Analysis).
    const BUILTIN_TIKTOKEN: &[&str] = &[
        "o200k_base",
        "cl100k_base",
        "p50k_base",
        "p50k_edit",
        "r50k_base",
        "gpt2",
    ];
    if BUILTIN_TIKTOKEN.contains(&model_id) {
        return crate::tiktoken::load_builtin_tiktoken(model_id).map(TokenizerKind::Tiktoken);
    }

    // 1. Try local HuggingFace tokenizer (tokenizer.json)
    match try_load_local(model_id).await {
        Ok(tok) => {
            tracing::info!(
                model = model_id,
                kind = "local",
                vocab_size = tok.get_vocab_size(true),
                "loaded tokenizer"
            );
            Ok(TokenizerKind::Local(Box::new(tok)))
        }
        Err(local_err) => {
            // 2. Try tiktoken format
            tracing::info!(
                model = model_id,
                error = %local_err.as_report(),
                "local tokenizer unavailable; trying tiktoken"
            );
            match crate::tiktoken::try_load_tiktoken(model_id).await {
                Ok(tok) => {
                    tracing::info!(
                        model = model_id,
                        kind = "tiktoken",
                        vocab_size = tok.vocab_size(),
                        "loaded tokenizer"
                    );
                    Ok(TokenizerKind::Tiktoken(tok))
                }
                Err(tiktoken_err) => {
                    // 3. Try server-side fallback
                    if let Some((base_url, model)) = server_info {
                        tracing::info!(
                            model = model_id,
                            error = %tiktoken_err.as_report(),
                            "tiktoken unavailable; trying server-side tokenization"
                        );
                        match ServerTokenizer::new(base_url, model).await {
                            Ok(srv) => {
                                tracing::info!(
                                    model = model_id,
                                    kind = "server",
                                    vocab_size = srv.cached_vocab_size,
                                    "loaded tokenizer"
                                );
                                return Ok(TokenizerKind::Server(srv));
                            }
                            Err(srv_err) => {
                                return Err(BenchError::Tokenizer(format!(
                                    "All tokenizer loading methods failed:\n  \
                                     Local: {local_err}\n  \
                                     Tiktoken: {tiktoken_err}\n  \
                                     Server: {srv_err}\n  \
                                     Try --tokenizer with a model that has tokenizer.json."
                                )));
                            }
                        }
                    }
                    Err(BenchError::Tokenizer(format!(
                        "Failed to load tokenizer:\n  \
                         Local: {local_err}\n  \
                         Tiktoken: {tiktoken_err}\n  \
                         Try --tokenizer or provide --base-url for server fallback."
                    )))
                }
            }
        }
    }
}

/// Try loading tokenizer.json from local path or HuggingFace Hub.
async fn try_load_local(model_id: &str) -> Result<Tokenizer> {
    // 1. Try local directory with tokenizer.json
    let local_path = Path::new(model_id).join("tokenizer.json");
    if local_path.exists() {
        return Tokenizer::from_file(&local_path).map_err(|e| {
            BenchError::Tokenizer(format!(
                "Failed to load tokenizer from {}: {e}",
                local_path.display()
            ))
        });
    }

    // 2. Try direct path to tokenizer.json
    if Path::new(model_id).exists() && model_id.ends_with("tokenizer.json") {
        return Tokenizer::from_file(model_id)
            .map_err(|e| BenchError::Tokenizer(format!("Failed to load tokenizer: {e}")));
    }

    // 3. If model_id is a local directory, don't try HF Hub — let tiktoken fallback handle it
    if Path::new(model_id).is_dir() {
        return Err(BenchError::Tokenizer(format!(
            "No tokenizer.json in local directory '{model_id}'"
        )));
    }

    // 4. Download from HuggingFace Hub (hf-hub handles auth via HF_TOKEN / cached token)
    let repo = crate::hub::HubRepo::model(model_id.to_string()).map_err(BenchError::Tokenizer)?;
    let tokenizer_path = repo
        .get("tokenizer.json")
        .await
        .map_err(|e| BenchError::Tokenizer(format!("No tokenizer.json for '{model_id}': {e}")))?;

    Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| BenchError::Tokenizer(format!("Failed to load downloaded tokenizer: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_server_tokenizer_sync_bridge() {
        let tokenizer = std::sync::Arc::new(ServerTokenizer {
            client: reqwest::Client::new(),
            runtime: tokio::runtime::Handle::current(),
            tokenize_url: String::new(),
            detokenize_url: String::new(),
            model: String::new(),
            cached_vocab_size: 0,
        });

        assert_eq!(tokenizer.block_on(async { Ok(1) }).unwrap(), 1);

        let (tx, rx) = tokio::sync::oneshot::channel();
        rayon::spawn(move || {
            let _ = tx.send(tokenizer.block_on(async { Ok(2) }));
        });
        assert_eq!(rx.await.unwrap().unwrap(), 2);
    }
}
