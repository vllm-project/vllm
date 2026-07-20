// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashSet;
use std::path::Path;

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
    client: reqwest::blocking::Client,
    tokenize_url: String,
    detokenize_url: String,
    model: String,
    cached_vocab_size: u32,
}

impl ServerTokenizer {
    /// Create a new server tokenizer and verify connectivity.
    pub fn new(base_url: &str, model: &str) -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| BenchError::Tokenizer(format!("Failed to build HTTP client: {e}")))?;

        let tokenize_url = format!("{base_url}/tokenize");
        let detokenize_url = format!("{base_url}/detokenize");

        let st = Self {
            client,
            tokenize_url,
            detokenize_url,
            model: model.to_string(),
            cached_vocab_size: 0,
        };

        // Probe the endpoint to verify it works and discover vocab size
        let test_tokens = st.encode_inner("test")?;
        let max_id = test_tokens.iter().copied().max().unwrap_or(0);
        let estimated_vocab = (max_id * 2).max(131072);

        Ok(Self {
            cached_vocab_size: estimated_vocab,
            ..st
        })
    }

    fn encode_inner(&self, text: &str) -> Result<Vec<u32>> {
        let payload = serde_json::json!({
            "model": self.model,
            "prompt": text,
        });

        let resp = self
            .client
            .post(&self.tokenize_url)
            .json(&payload)
            .send()
            .map_err(|e| BenchError::Tokenizer(format!("Server tokenize failed: {e}")))?;

        if !resp.status().is_success() {
            return Err(BenchError::Tokenizer(format!(
                "Server tokenize returned HTTP {}",
                resp.status()
            )));
        }

        let data: serde_json::Value = resp.json().map_err(|e| {
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
        let payload = serde_json::json!({
            "model": self.model,
            "tokens": ids,
        });

        let resp = self
            .client
            .post(&self.detokenize_url)
            .json(&payload)
            .send()
            .map_err(|e| BenchError::Tokenizer(format!("Server detokenize failed: {e}")))?;

        if !resp.status().is_success() {
            return Err(BenchError::Tokenizer(format!(
                "Server detokenize returned HTTP {}",
                resp.status()
            )));
        }

        let data: serde_json::Value = resp.json().map_err(|e| {
            BenchError::Tokenizer(format!("Failed to parse detokenize response: {e}"))
        })?;

        data.get("prompt")
            .and_then(|p| p.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| BenchError::Tokenizer("Missing 'prompt' in detokenize response".into()))
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
pub fn load_tokenizer(
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
    match try_load_local(model_id) {
        Ok(tok) => {
            println!("Tokenizer: Local (vocab_size={})", tok.get_vocab_size(true));
            Ok(TokenizerKind::Local(Box::new(tok)))
        }
        Err(local_err) => {
            // 2. Try tiktoken format
            println!("No tokenizer.json for '{model_id}', trying tiktoken format...");
            match crate::tiktoken::try_load_tiktoken(model_id) {
                Ok(tok) => {
                    println!("Tokenizer: Tiktoken (vocab_size={})", tok.vocab_size());
                    Ok(TokenizerKind::Tiktoken(tok))
                }
                Err(tiktoken_err) => {
                    // 3. Try server-side fallback
                    if let Some((base_url, model)) = server_info {
                        println!(
                            "Tiktoken also not available ({tiktoken_err}), \
                             trying server-side tokenization..."
                        );
                        match ServerTokenizer::new(base_url, model) {
                            Ok(srv) => {
                                println!(
                                    "Tokenizer: Server (vocab_size≈{})",
                                    srv.cached_vocab_size
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
fn try_load_local(model_id: &str) -> Result<Tokenizer> {
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
    let repo = crate::hub::HubRepo::model(model_id.to_string());
    let tokenizer_path = repo
        .get("tokenizer.json")
        .map_err(|e| BenchError::Tokenizer(format!("No tokenizer.json for '{model_id}': {e}")))?;

    Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| BenchError::Tokenizer(format!("Failed to load downloaded tokenizer: {e}")))
}
