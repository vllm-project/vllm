// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;
use std::path::Path;

use rustc_hash::FxHashMap;

use crate::error::{BenchError, Result};

/// Default regex pattern matching cl100k_base (GPT-4, Qwen, etc.)
const DEFAULT_TIKTOKEN_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Number of reserved special token slots (matches Python's
/// TikTokenTokenizer.num_reserved_special_tokens)
const NUM_RESERVED_SPECIAL_TOKENS: u32 = 256;

/// Tiktoken-based tokenizer for models that use tiktoken format (Kimi, Qwen, etc.)
pub struct TiktokenTokenizer {
    bpe: tiktoken_rs::CoreBPE,
    vocab_size: u32,
    #[allow(dead_code)]
    num_base_tokens: u32,
    /// IDs of all special tokens (for filtering from allowed_tokens)
    special_token_ids: Vec<u32>,
    /// Reverse mapping: token_id -> byte sequence, for lossy UTF-8 decoding.
    /// Empty for built-in encodings (use bpe.decode instead).
    decoder: Vec<Vec<u8>>,
    /// True for built-in encodings loaded via tiktoken_rs (o200k_base, cl100k_base, etc.)
    is_builtin: bool,
}

impl TiktokenTokenizer {
    /// Load from a tiktoken .model file.
    ///
    /// `special_tokens_from_config`: tokens from tokenizer_config.json's added_tokens_decoder
    /// `all_special_tokens`: ALL special tokens including the full 256 reserved slots
    pub fn from_file(
        model_path: &Path,
        all_special_tokens: HashMap<String, u32>,
        special_token_ids: Vec<u32>,
        pattern: Option<&str>,
    ) -> Result<Self> {
        let content = std::fs::read_to_string(model_path)
            .map_err(|e| BenchError::Tokenizer(format!("Failed to read tiktoken model: {e}")))?;

        let mut encoder: FxHashMap<Vec<u8>, u32> = FxHashMap::default();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let mut parts = line.split_whitespace();
            let b64 = match parts.next() {
                Some(s) => s,
                None => continue,
            };
            let rank_str = match parts.next() {
                Some(s) => s,
                None => continue,
            };

            use base64::Engine;
            let token_bytes = base64::engine::general_purpose::STANDARD
                .decode(b64)
                .map_err(|e| BenchError::Tokenizer(format!("Invalid base64 in model file: {e}")))?;
            let rank: u32 = rank_str
                .parse()
                .map_err(|e| BenchError::Tokenizer(format!("Invalid rank: {e}")))?;
            encoder.insert(token_bytes, rank);
        }

        if encoder.is_empty() {
            return Err(BenchError::Tokenizer("Empty tiktoken model file".into()));
        }

        let num_base_tokens = encoder.len() as u32;

        // Convert special tokens to FxHashMap for tiktoken-rs
        let special_fx: FxHashMap<String, u32> =
            all_special_tokens.iter().map(|(k, &v)| (k.clone(), v)).collect();

        // vocab_size = base tokens + all reserved special token slots
        let vocab_size = num_base_tokens + NUM_RESERVED_SPECIAL_TOKENS;

        // Build reverse mapping for lossy decode
        let mut decoder = vec![Vec::new(); vocab_size as usize];
        for (bytes, &rank) in &encoder {
            if (rank as usize) < decoder.len() {
                decoder[rank as usize] = bytes.clone();
            }
        }
        for (text, &rank) in &all_special_tokens {
            if (rank as usize) < decoder.len() {
                decoder[rank as usize] = text.as_bytes().to_vec();
            }
        }

        let pat = pattern.unwrap_or(DEFAULT_TIKTOKEN_PATTERN);

        let bpe = tiktoken_rs::CoreBPE::new(encoder, special_fx, pat)
            .map_err(|e| BenchError::Tokenizer(format!("Failed to build tiktoken BPE: {e}")))?;

        Ok(Self {
            bpe,
            vocab_size,
            num_base_tokens,
            special_token_ids,
            decoder,
            is_builtin: false,
        })
    }

    /// Create from a built-in tiktoken encoding (o200k_base, cl100k_base, etc.)
    pub fn from_builtin_bpe(bpe: tiktoken_rs::CoreBPE, vocab_size: u32) -> Self {
        Self {
            bpe,
            vocab_size,
            num_base_tokens: vocab_size,
            special_token_ids: Vec::new(),
            decoder: Vec::new(),
            is_builtin: true,
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        // Use encode_with_special_tokens to match Python's encode(allowed_special="all")
        self.bpe.encode_with_special_tokens(text)
    }

    /// Decode token IDs to text with lossy UTF-8 handling.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        if self.is_builtin {
            // Byte-level decode + lossy UTF-8, matching the file-based path below.
            // (CoreBPE::decode errors on invalid UTF-8, which random token
            // sequences routinely produce with byte-level BPE vocabularies.)
            let bytes: Vec<u8> =
                self.bpe._decode_native_and_split(ids.to_vec()).flatten().collect();
            return Ok(String::from_utf8_lossy(&bytes).into_owned());
        }
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(token_bytes) = self.decoder.get(id as usize) {
                bytes.extend_from_slice(token_bytes);
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    pub fn vocab_size(&self) -> u32 {
        self.vocab_size
    }

    /// Get non-special token IDs whose byte representation is valid UTF-8.
    /// Excludes ALL special tokens (like Python's `set(all_tokens) - set(prohibited_tokens)`).
    pub fn get_allowed_tokens(&self) -> Vec<u32> {
        if self.is_builtin {
            // For built-in encodings, return full token range.
            // Note: when used with random dataset, these IDs are sent to vLLM as-is;
            // ensure the model's tokenizer is compatible (e.g. GPT-4o for o200k_base).
            return (0..self.vocab_size).collect();
        }
        let special_set: std::collections::HashSet<u32> =
            self.special_token_ids.iter().copied().collect();

        self.decoder
            .iter()
            .enumerate()
            .filter(|(id, bytes)| {
                !bytes.is_empty()
                    && std::str::from_utf8(bytes).is_ok()
                    && !special_set.contains(&(*id as u32))
            })
            .map(|(id, _)| id as u32)
            .collect()
    }
}

/// Load a built-in tiktoken encoding by name.
///
/// Supported names: `o200k_base` (GPT-4o), `cl100k_base` (GPT-4/3.5-turbo),
/// `p50k_base`, `r50k_base`, `gpt2`.
///
/// These encodings are bundled with tiktoken-rs — no network download required.
/// Useful for consistent cross-model token counting (e.g. Artificial Analysis methodology).
pub fn load_builtin_tiktoken(encoding: &str) -> Result<TiktokenTokenizer> {
    let (bpe, vocab_size) = match encoding {
        "o200k_base" => (tiktoken_rs::o200k_base(), 200_275u32),
        "cl100k_base" => (tiktoken_rs::cl100k_base(), 100_277u32),
        "p50k_base" => (tiktoken_rs::p50k_base(), 50_281u32),
        "p50k_edit" => (tiktoken_rs::p50k_edit(), 50_281u32),
        "r50k_base" | "gpt2" => (tiktoken_rs::r50k_base(), 50_257u32),
        _ => {
            return Err(BenchError::Tokenizer(format!(
                "Unknown built-in tiktoken encoding: '{encoding}'. \
                 Supported: o200k_base, cl100k_base, p50k_base, r50k_base, gpt2"
            )));
        }
    };
    let bpe = bpe.map_err(|e| BenchError::Tokenizer(format!("Failed to load {encoding}: {e}")))?;
    tracing::info!(
        encoding,
        kind = "built-in-tiktoken",
        vocab_size,
        "loaded tokenizer"
    );
    Ok(TiktokenTokenizer::from_builtin_bpe(bpe, vocab_size))
}

/// Try to load a tiktoken tokenizer from a local directory or HuggingFace model repo.
pub async fn try_load_tiktoken(model_id: &str) -> Result<TiktokenTokenizer> {
    // Phase 1: If model_id is a local directory, look for tiktoken files there
    let local_dir = Path::new(model_id);
    if local_dir.is_dir() {
        return try_load_tiktoken_from_dir(local_dir, model_id);
    }

    // Phase 2: Fall back to HuggingFace Hub download
    try_load_tiktoken_from_hf(model_id).await
}

/// Common tiktoken model filenames to search for.
const TIKTOKEN_MODEL_FILENAMES: &[&str] = &["tiktoken.model", "qwen.tiktoken", "vocab.tiktoken"];

/// Load a tiktoken tokenizer from a local directory.
fn try_load_tiktoken_from_dir(dir: &Path, model_id: &str) -> Result<TiktokenTokenizer> {
    let model_path = TIKTOKEN_MODEL_FILENAMES
        .iter()
        .map(|f| dir.join(f))
        .find(|p| p.exists())
        .ok_or_else(|| {
            BenchError::Tokenizer(format!(
                "No tiktoken model file found in local directory '{model_id}' \
                 (looked for: {})",
                TIKTOKEN_MODEL_FILENAMES.join(", ")
            ))
        })?;

    let num_base_tokens = count_base_tokens(&model_path)?;

    let config_path = dir.join("tokenizer_config.json");
    let config = if config_path.exists() {
        read_tokenizer_config(&config_path)
    } else {
        None
    };

    let pattern = extract_pat_str_from_local_dir(dir);

    build_tiktoken(model_id, &model_path, config, pattern, num_base_tokens)
}

/// Load a tiktoken tokenizer from a HuggingFace model repo.
async fn try_load_tiktoken_from_hf(model_id: &str) -> Result<TiktokenTokenizer> {
    let repo = crate::hub::HubRepo::model(model_id.to_string()).map_err(BenchError::Tokenizer)?;

    let mut model_path = None;
    for filename in TIKTOKEN_MODEL_FILENAMES {
        if let Ok(path) = repo.get(filename).await {
            model_path = Some(path);
            break;
        }
    }
    let model_path = model_path.ok_or_else(|| {
        BenchError::Tokenizer(format!("No tiktoken model file found for '{model_id}'"))
    })?;

    let num_base_tokens = count_base_tokens(&model_path)?;

    let config = match repo.get("tokenizer_config.json").await {
        Ok(config_path) => read_tokenizer_config(&config_path),
        Err(_) => None,
    };

    let pattern = extract_pat_str_from_repo(&repo).await;

    build_tiktoken(model_id, &model_path, config, pattern, num_base_tokens)
}

/// Build a TiktokenTokenizer from discovered model file, config, and pattern.
fn build_tiktoken(
    model_id: &str,
    model_path: &Path,
    config: Option<TokenizerConfig>,
    pattern: Option<String>,
    num_base_tokens: u32,
) -> Result<TiktokenTokenizer> {
    // Build the full 256 reserved special tokens map.
    // Python: {special_tokens_mapping.get(i, f"<|reserved_token_{i}|>"): i
    //          for i in range(num_base_tokens, num_base_tokens + 256)}
    let config_special = config.as_ref().map(|c| &c.added_tokens).cloned().unwrap_or_default();

    // Invert config_special: id -> content
    let id_to_content: HashMap<u32, String> =
        config_special.iter().map(|(content, &id)| (id, content.clone())).collect();

    let mut all_special_tokens: HashMap<String, u32> = HashMap::new();
    let mut special_token_ids: Vec<u32> = Vec::new();

    for i in 0..NUM_RESERVED_SPECIAL_TOKENS {
        let token_id = num_base_tokens + i;
        let content = id_to_content
            .get(&token_id)
            .cloned()
            .unwrap_or_else(|| format!("<|reserved_token_{i}|>"));
        all_special_tokens.insert(content, token_id);
        special_token_ids.push(token_id);
    }

    // Also collect special IDs that Python marks as prohibited
    // (those with "special": true in added_tokens_decoder)
    let prohibited_ids = config.as_ref().map(|c| c.special_ids.clone()).unwrap_or_default();
    for id in &prohibited_ids {
        if !special_token_ids.contains(id) {
            special_token_ids.push(*id);
        }
    }

    tracing::info!(
        model = model_id,
        base_tokens = num_base_tokens,
        special_tokens = all_special_tokens.len(),
        pattern = if pattern.is_some() {
            "custom"
        } else {
            "default"
        },
        "loading tiktoken model"
    );

    TiktokenTokenizer::from_file(
        model_path,
        all_special_tokens,
        special_token_ids,
        pattern.as_deref(),
    )
}

/// Count base tokens in a tiktoken .model file without fully parsing it.
fn count_base_tokens(model_path: &Path) -> Result<u32> {
    let content = std::fs::read_to_string(model_path)
        .map_err(|e| BenchError::Tokenizer(format!("Failed to read tiktoken model: {e}")))?;
    let count = content
        .lines()
        .filter(|l| {
            let l = l.trim();
            !l.is_empty() && l.split_whitespace().count() >= 2
        })
        .count();
    Ok(count as u32)
}

/// Parsed tokenizer config
struct TokenizerConfig {
    /// All tokens from added_tokens_decoder (content -> id)
    added_tokens: HashMap<String, u32>,
    /// IDs of tokens marked "special": true
    special_ids: Vec<u32>,
}

/// Read and parse tokenizer_config.json
fn read_tokenizer_config(config_path: &Path) -> Option<TokenizerConfig> {
    let content = std::fs::read_to_string(config_path).ok()?;
    let config: serde_json::Value = serde_json::from_str(&content).ok()?;

    let mut added_tokens = HashMap::new();
    let mut special_ids = Vec::new();

    if let Some(added) = config.get("added_tokens_decoder").and_then(|v| v.as_object()) {
        for (id_str, token_info) in added {
            if let (Ok(id), Some(content)) = (
                id_str.parse::<u32>(),
                token_info.get("content").and_then(|c| c.as_str()),
            ) {
                added_tokens.insert(content.to_string(), id);

                let is_special =
                    token_info.get("special").and_then(|s| s.as_bool()).unwrap_or(false);
                if is_special {
                    special_ids.push(id);
                }
            }
        }
    }

    Some(TokenizerConfig {
        added_tokens,
        special_ids,
    })
}

/// Try to extract pat_str from Python tokenizer source files in a local directory.
/// Returns None if unavailable or unparsable.
fn extract_pat_str_from_local_dir(dir: &Path) -> Option<String> {
    ["tokenization_kimi.py", "tokenizer.py"]
        .iter()
        .map(|f| dir.join(f))
        .filter(|p| p.exists())
        .find_map(|p| {
            std::fs::read_to_string(p)
                .ok()
                .and_then(|source| extract_pat_str_from_source(&source))
        })
}

/// Try to download the Python tokenizer source file and extract pat_str via regex.
/// Returns None if unavailable or unparsable.
async fn extract_pat_str_from_repo(repo: &crate::hub::HubRepo) -> Option<String> {
    // Try common Python tokenizer filenames
    let py_path = match repo.get("tokenization_kimi.py").await {
        Ok(path) => path,
        Err(_) => repo.get("tokenizer.py").await.ok()?,
    };

    let source = std::fs::read_to_string(&py_path).ok()?;

    // Look for pat_str assignment. Common patterns:
    // pat_str = "..." or pat_str = '...' or pat_str = "|".join([...])
    extract_pat_str_from_source(&source)
}

/// Parse pat_str from Python source code.
fn extract_pat_str_from_source(source: &str) -> Option<String> {
    // Strategy: find `pat_str = "|".join([` and collect the raw string fragments
    // This handles the common Kimi/Qwen pattern of joining a list of regex strings.

    // First try: look for pat_str = "|".join([...]) pattern
    if let Some(join_start) = source.find("pat_str") {
        let after = &source[join_start..];

        // Check for "|".join([ pattern
        if let Some(join_pos) = after.find(".join(") {
            let after_join = &after[join_pos + 6..]; // skip ".join("
            if let Some(bracket_start) = after_join.find('[') {
                let inside = &after_join[bracket_start + 1..];
                // Collect all string literals inside the list
                let mut fragments = Vec::new();
                let mut remaining = inside;

                while let Some(frag) = extract_next_python_string(remaining) {
                    fragments.push(frag.0);
                    remaining = frag.1;
                    // Check if we hit the closing bracket
                    let trimmed = remaining.trim_start();
                    if trimmed.starts_with(']') {
                        break;
                    }
                }

                if !fragments.is_empty() {
                    let pattern = fragments.join("|");
                    tracing::debug!(
                        fragments = fragments.len(),
                        "extracted tiktoken pattern from Python source"
                    );
                    return Some(pattern);
                }
            }
        }

        // Fallback: simple pat_str = r"..." or pat_str = "..."
        if let Some(eq_pos) = after.find('=') {
            let after_eq = after[eq_pos + 1..].trim_start();
            if let Some(frag) = extract_next_python_string(after_eq) {
                return Some(frag.0);
            }
        }
    }

    None
}

/// Extract the next Python string literal (r"...", "...", r'''...''', etc.)
/// Returns (string_content, remaining_text)
fn extract_next_python_string(s: &str) -> Option<(String, &str)> {
    let s = s.trim_start_matches(|c: char| c == ',' || c.is_whitespace());

    // Skip comments
    if s.starts_with('#') {
        let next_line = s.find('\n').map(|i| i + 1).unwrap_or(s.len());
        return extract_next_python_string(&s[next_line..]);
    }

    // Check for r""" (triple-quoted raw string)
    for prefix in &["r\"\"\"", "r'''"] {
        if let Some(inner) = s.strip_prefix(prefix) {
            let delim = &prefix[1..]; // """ or '''
            if let Some(end) = inner.find(delim) {
                let content = &inner[..end];
                let rest = &inner[end + delim.len()..];
                return Some((content.to_string(), rest));
            }
        }
    }

    // Check for r"..." (raw string)
    if let Some(inner) = s.strip_prefix("r\"")
        && let Some(end) = inner.find('"')
    {
        let content = &inner[..end];
        let rest = &inner[end + 1..];
        return Some((content.to_string(), rest));
    }
    if let Some(inner) = s.strip_prefix("r'")
        && let Some(end) = inner.find('\'')
    {
        let content = &inner[..end];
        let rest = &inner[end + 1..];
        return Some((content.to_string(), rest));
    }

    // Check for "..." or '...' (regular string — same as raw for regex patterns)
    if let Some(inner) = s.strip_prefix('"')
        && let Some(end) = find_unescaped(inner, '"')
    {
        let content = &inner[..end];
        let rest = &inner[end + 1..];
        return Some((content.to_string(), rest));
    }
    if let Some(inner) = s.strip_prefix('\'')
        && let Some(end) = find_unescaped(inner, '\'')
    {
        let content = &inner[..end];
        let rest = &inner[end + 1..];
        return Some((content.to_string(), rest));
    }

    // Check for closing bracket — stop
    if s.starts_with(']') {
        return None;
    }

    None
}

/// Find position of `ch` that is not preceded by a backslash.
fn find_unescaped(s: &str, ch: char) -> Option<usize> {
    let mut escaped = false;
    for (i, c) in s.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if c == '\\' {
            escaped = true;
            continue;
        }
        if c == ch {
            return Some(i);
        }
    }
    None
}
