use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::Deserialize;
use thiserror_ext::AsReport as _;

use crate::error::{Error, Result};

/// Minimal subset of `tokenizer_config.json` needed by chat/EOS handling.
#[derive(Debug, Default, Deserialize)]
pub(super) struct HfTokenizerConfig {
    #[serde(default)]
    pub eos_token: Option<NamedSpecialToken>,
    /// The `tokenizer_class` field from HuggingFace tokenizer configs. Some tiktoken-based models
    /// (e.g. DeepSeek, Kimi K2) set this to a value containing "Tiktoken" which can be used as a
    /// hint for backend selection.
    #[serde(default)]
    pub tokenizer_class: Option<String>,
}

/// Hugging Face named special tokens may be serialized as a string or an
/// object carrying the token content.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub(super) enum NamedSpecialToken {
    Text(String),
    WithContent { content: String },
}

impl NamedSpecialToken {
    pub(super) fn as_str(&self) -> &str {
        match self {
            Self::Text(value) => value,
            Self::WithContent { content } => content,
        }
    }
}

/// Minimal subset of `config.json` (the model's main HF config).
///
/// This intentionally supports only the two layouts we currently care about in the Rust frontend:
/// - pure text models that keep text metadata at the top level
/// - composite models that expose a single nested `text_config`
///
/// We do not support additional entry points such as `decoder`, `generator`, or `text_encoder`.
#[derive(Debug, Default, Deserialize)]
pub(super) struct ModelConfig {
    #[serde(default)]
    pub max_position_embeddings: Option<u32>,
    #[serde(default)]
    pub num_attention_heads: Option<u32>,
    #[serde(default)]
    pub num_experts: Option<OneOrManyExpertCount>,
    #[serde(default)]
    pub moe_num_experts: Option<OneOrManyExpertCount>,
    #[serde(default)]
    pub n_routed_experts: Option<OneOrManyExpertCount>,
    #[serde(default)]
    pub num_local_experts: Option<OneOrManyExpertCount>,
    #[serde(default)]
    pub block_configs: Vec<BlockConfig>,
    #[serde(default)]
    pub text_config: Option<Box<ModelConfig>>,
}

/// Minimal subset of `generation_config.json`.
#[derive(Debug, Default, Deserialize)]
pub(super) struct GenerationConfig {
    #[serde(default)]
    pub eos_token_id: Option<OneOrManyTokenIds>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub max_new_tokens: Option<u32>,
}

/// HF generation configs allow either one EOS id or a list of EOS ids.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub(super) enum OneOrManyTokenIds {
    One(u32),
    Many(Vec<u32>),
}

impl OneOrManyTokenIds {
    pub(super) fn into_set(self) -> BTreeSet<u32> {
        match self {
            Self::One(id) => BTreeSet::from([id]),
            Self::Many(ids) => ids.into_iter().collect(),
        }
    }
}

/// Hugging Face configs may expose the expert count either as one integer or
/// as a list of repeated integers.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub(super) enum OneOrManyExpertCount {
    One(u32),
    Many(Vec<u32>),
}

impl OneOrManyExpertCount {
    fn first_value(&self) -> u32 {
        match self {
            Self::One(value) => *value,
            // Python currently takes the first value for list[int] expert
            // counts in remote-code configs.
            Self::Many(values) => values.first().copied().unwrap_or(0),
        }
    }
}

/// Heterogeneous block-level MoE metadata used as a fallback when no top-level
/// expert-count field is available.
#[derive(Debug, Default, Deserialize)]
pub(super) struct BlockConfig {
    #[serde(default)]
    pub block_type: String,
    #[serde(default)]
    pub n_routed_experts: u32,
}

impl ModelConfig {
    /// Return the config that the Rust frontend treats as the text/LLM config.
    ///
    /// This is deliberately narrower than Python/transformers: we only support
    /// either the top-level config itself or a single nested `text_config`.
    fn effective_text_config(&self) -> &Self {
        self.text_config.as_deref().unwrap_or(self)
    }

    /// Reject partially nested `text_config` payloads that are unlikely to be
    /// valid LLM configs for our current use.
    ///
    /// This keeps the simplified Rust-side parsing honest: if a model declares
    /// `text_config`, it must at least look like a real text model config.
    fn validate_text_config_selection(&self) -> Result<()> {
        if let Some(text_config) = self.text_config.as_deref()
            && text_config.num_attention_heads.is_none()
        {
            return Err(Error::Tokenizer(
                "the text config extracted from the model config does not have `num_attention_heads`"
                    .to_string(),
            ));
        }

        Ok(())
    }

    /// Match Python's current expert-count priority on the selected text config.
    ///
    /// The only intentional simplification here is how we pick the text config:
    /// Rust only looks at the top level or `text_config`, not the broader
    /// transformers composite-config surface.
    fn num_experts_from_block_configs(&self) -> u32 {
        self.effective_text_config()
            .block_configs
            .iter()
            .filter(|block| block.block_type == "moe")
            .map(|block| block.n_routed_experts)
            .max()
            .unwrap_or(0)
    }

    pub(super) fn num_experts(&self) -> u32 {
        let config = self.effective_text_config();
        let direct = [
            config.num_experts.as_ref(),
            config.moe_num_experts.as_ref(),
            config.n_routed_experts.as_ref(),
            config.num_local_experts.as_ref(),
        ]
        .into_iter()
        .flatten()
        .map(OneOrManyExpertCount::first_value)
        .next()
        .unwrap_or(0);

        if direct > 0 {
            direct
        } else {
            self.num_experts_from_block_configs()
        }
    }

    pub(super) fn is_moe(&self) -> bool {
        self.num_experts() > 0
    }

    pub(super) fn effective_max_position_embeddings(&self) -> Option<u32> {
        self.effective_text_config().max_position_embeddings
    }
}

/// Load the tokenizer-side EOS metadata if a config file is present.
pub(super) fn load_tokenizer_config(path: Option<&Path>) -> Result<HfTokenizerConfig> {
    read_json_file(path)
}

/// Load the generation-side EOS metadata if a config file is present.
pub(super) fn load_generation_config(path: Option<&Path>) -> Result<GenerationConfig> {
    read_json_file(path)
}

/// Load the model-side config (`config.json`) if present.
pub(super) fn load_model_config(path: Option<&Path>) -> Result<ModelConfig> {
    let config: ModelConfig = read_json_file(path)?;
    config.validate_text_config_selection()?;
    Ok(config)
}

fn read_json_file<T>(path: Option<&Path>) -> Result<T>
where
    T: for<'de> Deserialize<'de> + Default,
{
    let Some(path) = path else {
        return Ok(T::default());
    };
    let content = fs::read_to_string(path).map_err(|error| {
        Error::Tokenizer(format!(
            "failed to read {}: {}",
            path.display(),
            error.as_report()
        ))
    })?;
    serde_json::from_str(&content).map_err(|error| {
        Error::Tokenizer(format!(
            "failed to parse {}: {}",
            path.display(),
            error.as_report()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::ModelConfig;

    #[test]
    fn model_config_detects_moe_from_named_expert_fields() {
        let field_names = [
            "num_experts",
            "moe_num_experts",
            "n_routed_experts",
            "num_local_experts",
        ];

        for field_name in field_names {
            let config: ModelConfig =
                serde_json::from_str(&format!(r#"{{"{field_name}": 64}}"#)).unwrap();
            assert_eq!(config.num_experts(), 64, "field_name={field_name}");
            assert!(config.is_moe(), "field_name={field_name}");
        }
    }

    #[test]
    fn model_config_uses_first_value_for_list_expert_counts() {
        let config: ModelConfig = serde_json::from_str(r#"{"num_experts":[16,16]}"#).unwrap();

        assert_eq!(config.num_experts(), 16);
        assert!(config.is_moe());
    }

    #[test]
    fn model_config_falls_back_to_block_configs_maximum() {
        let config: ModelConfig = serde_json::from_str(
            r#"{
                "block_configs": [
                    {"block_type":"attention","n_routed_experts":9},
                    {"block_type":"moe","n_routed_experts":32},
                    {"block_type":"moe","n_routed_experts":64}
                ]
            }"#,
        )
        .unwrap();

        assert_eq!(config.num_experts(), 64);
        assert!(config.is_moe());
    }

    #[test]
    fn model_config_prefers_nested_text_config_like_python_hf_text_config() {
        let config: ModelConfig = serde_json::from_str(
            r#"{
                "num_experts": 64,
                "max_position_embeddings": 8192,
                "text_config": {
                    "num_attention_heads": 32,
                    "num_local_experts": 8,
                    "max_position_embeddings": 4096
                }
            }"#,
        )
        .unwrap();

        assert_eq!(config.num_experts(), 8);
        assert_eq!(config.effective_max_position_embeddings(), Some(4096));
        assert!(config.is_moe());
    }

    #[test]
    fn model_config_defaults_to_non_moe_when_no_expert_metadata_exists() {
        let config: ModelConfig =
            serde_json::from_str(r#"{"max_position_embeddings":4096}"#).unwrap();

        assert_eq!(config.num_experts(), 0);
        assert!(!config.is_moe());
        assert_eq!(config.effective_max_position_embeddings(), Some(4096));
    }

    #[test]
    fn model_config_rejects_nested_text_config_without_attention_heads() {
        let config: ModelConfig =
            serde_json::from_str(r#"{"text_config":{"max_position_embeddings":4096}}"#).unwrap();

        let error = config.validate_text_config_selection().unwrap_err();
        assert!(
            error
                .to_string()
                .contains("does not have `num_attention_heads`"),
        );
    }
}
