use serde::{Deserialize, Serialize};
use thiserror_ext::AsReport as _;
use tracing::warn;

use crate::Result;

use std::{fs, path::Path};

/// Minimal `tokenizer.json` projection used to patch `added_tokens` while
/// preserving the rest of the tokenizer definition verbatim.
#[derive(Debug, Deserialize, Serialize)]
struct TokenizerJson {
    #[serde(default)]
    added_tokens: Vec<AddedTokenConfig>,
    #[serde(flatten)]
    extra: serde_json::Map<String, serde_json::Value>,
}

/// Minimal `tokenizer_config.json` projection for Hugging Face's
/// `added_tokens_decoder` map. Other config keys are intentionally ignored.
#[derive(Debug, Deserialize)]
struct TokenizerConfigJson {
    #[serde(default)]
    added_tokens_decoder: std::collections::HashMap<String, AddedTokenConfig>,
}

/// Hugging Face added-token payload. `tokenizer.json` stores `id` inside each
/// item, while `tokenizer_config.json` stores it as the map key.
#[derive(Clone, Debug, Deserialize, Serialize)]
struct AddedTokenConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    id: Option<u32>,
    content: String,
    #[serde(default)]
    single_word: bool,
    #[serde(default)]
    lstrip: bool,
    #[serde(default)]
    rstrip: bool,
    #[serde(default)]
    normalized: bool,
    #[serde(default)]
    special: bool,
}

impl AddedTokenConfig {
    /// Return this added-token payload in `tokenizer.json` shape by filling the
    /// numeric token id that came from `added_tokens_decoder`'s string key.
    fn with_id(mut self, id: u32) -> Self {
        self.id = Some(id);
        self
    }
}

/// Read `tokenizer.json`, then merge in extra added tokens from `tokenizer_config.json`.
pub(super) fn load_tokenizer_json_with_extra_tokens(path: &Path) -> Result<serde_json::Value> {
    let tokenizer_json = fs::read_to_string(path)
        .map_err(|error| tokenizer_error!("failed to read {}: {}", path.display(), error))?;
    let mut tokenizer_json: TokenizerJson = serde_json::from_str(&tokenizer_json)
        .map_err(|error| tokenizer_error!("failed to parse {}: {}", path.display(), error))?;

    if let Some(parent) = path.parent() {
        let config_path = parent.join("tokenizer_config.json");
        if config_path.exists() {
            match load_tokenizer_config_json(&config_path) {
                Ok(config_json) => merge_added_tokens_from_config(&mut tokenizer_json, config_json),
                Err(error) => {
                    warn!(
                        path = %config_path.display(),
                        error = %error.as_report(),
                        "failed to load tokenizer_config.json; skipping extra added tokens"
                    );
                }
            }
        }
    }

    serde_json::to_value(tokenizer_json)
        .map_err(|error| tokenizer_error!("failed to serialize tokenizer json: {}", error))
}

/// Read and parse a sibling `tokenizer_config.json`.
fn load_tokenizer_config_json(path: &Path) -> Result<TokenizerConfigJson> {
    let text = fs::read_to_string(path)
        .map_err(|error| tokenizer_error!("failed to read {}: {}", path.display(), error))?;
    serde_json::from_str(&text)
        .map_err(|error| tokenizer_error!("failed to parse {}: {}", path.display(), error))
}

/// Merge added_tokens in `tokenizer.json` and `tokenizer_config.json`.
fn merge_added_tokens_from_config(
    tokenizer_json: &mut TokenizerJson,
    config_json: TokenizerConfigJson,
) {
    use std::collections::HashSet;

    let mut existing_ids: HashSet<u32> =
        tokenizer_json.added_tokens.iter().filter_map(|token| token.id).collect();

    let mut extra_tokens = Vec::with_capacity(config_json.added_tokens_decoder.len());
    for (id_str, token) in config_json.added_tokens_decoder {
        let id = match id_str.parse::<u32>() {
            Ok(id) => id,
            Err(_) => continue,
        };
        extra_tokens.push((id, token));
    }
    extra_tokens.sort_unstable_by_key(|(id, _)| *id);

    for (id, token) in extra_tokens {
        if existing_ids.contains(&id) {
            continue;
        }

        // Convert from decoder format to added_tokens array format by adding the "id" field.
        tokenizer_json.added_tokens.push(token.with_id(id));
        existing_ids.insert(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_added_tokens_from_config_preserves_unmodeled_fields() {
        let mut tokenizer_json: TokenizerJson = serde_json::from_value(serde_json::json!({
            "version": "1.0",
            "added_tokens": [
                {"id": 0, "content": "<unk>", "special": true}
            ],
            "model": {"type": "WordLevel"}
        }))
        .expect("parse tokenizer json");

        let config_json: TokenizerConfigJson = serde_json::from_value(serde_json::json!({
            "chat_template": "{{ messages }}",
            "added_tokens_decoder": {
                "1": {
                    "content": "<|image_pad|>",
                    "special": true,
                    "normalized": false
                }
            }
        }))
        .expect("parse tokenizer config");

        merge_added_tokens_from_config(&mut tokenizer_json, config_json);
        let merged = serde_json::to_value(tokenizer_json).expect("serialize tokenizer json");

        assert_eq!(merged["version"], "1.0");
        assert_eq!(merged["model"]["type"], "WordLevel");
        assert_eq!(merged["added_tokens"][1]["id"], 1);
        assert_eq!(merged["added_tokens"][1]["content"], "<|image_pad|>");
        assert_eq!(merged["added_tokens"][1]["special"], true);
        assert_eq!(merged["added_tokens"][1]["normalized"], false);
    }
}
