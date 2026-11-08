// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

mod config;
mod model_files;
mod overrides;

use std::collections::BTreeSet;
use std::sync::Arc;

use tracing::info;
use vllm_tokenizer::{
    DynTokenizer, HuggingFaceTokenizer, TekkenTokenizer, TiktokenTokenizer, Tokenizer,
    UnavailableTokenizer,
};

use self::config::{GenerationConfig, load_generation_config};
pub use self::config::{
    HfChatTemplate, HfNamedChatTemplate, HfSpecialTokens, HfTokenizerConfig, ModelConfig,
    NamedSpecialToken, load_model_config, load_tokenizer_config,
};
pub use self::model_files::{ModelFileSources, ModelSource, ResolvedModelFiles, TokenizerSource};
pub use self::overrides::HfOverrides;
use crate::backend::{GenerationConfigMode, SamplingHints, TextBackend};
use crate::error::Result;

fn load_tokenizer(tokenizer: &TokenizerSource) -> Result<DynTokenizer> {
    match tokenizer {
        TokenizerSource::HuggingFace(path) => Ok(Arc::new(HuggingFaceTokenizer::new(path)?)),
        TokenizerSource::Tiktoken(path) => Ok(Arc::new(TiktokenTokenizer::new(path)?)),
        TokenizerSource::Tekken(path) => Ok(Arc::new(TekkenTokenizer::new(path)?)),
    }
}

/// [`TextBackend`] implementation built on Hugging Face model files.
pub struct HfTextBackend {
    model_id: String,
    files: ResolvedModelFiles,
    /// [`UnavailableTokenizer`] when the files resolve no tokenizer
    /// (`--skip-tokenizer-init`).
    tokenizer: DynTokenizer,
    /// Primary EOS handled by engine-core's dedicated EOS path.
    primary_eos_token_id: Option<u32>,
    /// Additional EOS ids that should flow through stop-token handling.
    extra_eos_token_ids: BTreeSet<u32>,
    /// Generation-config for sampling defaults that may be inherited when the
    /// user does not explicitly override them.
    generation_config: GenerationConfig,
    generation_config_mode: GenerationConfigMode,
    /// Model vocabulary size from the selected text config.
    model_vocab_size: usize,
    /// Model config (`config.json`).
    model_config: ModelConfig,
}

impl HfTextBackend {
    /// Load the text backend from resolved Hugging Face model files.
    pub fn from_resolved_model_files(
        files: ResolvedModelFiles,
        model_id: String,
        generation_config_mode: GenerationConfigMode,
    ) -> Result<Self> {
        let tokenizer_config = load_tokenizer_config(files.tokenizer_config_path.as_deref())?;
        let model_config = load_model_config(files.config_path.as_deref())?;
        let model_vocab_size = model_config.vocab_size()? as usize;
        let tokenizer = match &files.tokenizer {
            Some(tokenizer) => load_tokenizer(tokenizer)?,
            None => Arc::new(UnavailableTokenizer::new(model_vocab_size)),
        };
        let generation_config = load_generation_config(files.generation_config_path.as_deref())?;
        let (primary_eos_token_id, extra_eos_token_ids) = resolve_eos_token_ids(
            &tokenizer_config,
            &model_config,
            &generation_config,
            tokenizer.as_ref(),
        );

        info!(
            model_id,
            "loaded text backend with Hugging Face model files"
        );

        Ok(Self {
            model_id,
            files,
            tokenizer,
            primary_eos_token_id,
            extra_eos_token_ids,
            generation_config,
            generation_config_mode,
            model_vocab_size,
            model_config,
        })
    }

    /// Expose the resolved model files for use by the chat backend to load the
    /// chat template.
    pub fn resolved_model_files(&self) -> &ResolvedModelFiles {
        &self.files
    }
}

/// Resolve EOS hints from tokenizer, model, and generation configs.
///
/// Resolution rules:
/// 1. Use the tokenizer-side EOS token as the primary EOS when it resolves to a
///    token id.
/// 2. Fall back to the first model-config EOS id when the tokenizer config does
///    not provide a primary EOS.
/// 3. Keep any remaining model/generation EOS ids as extra stop-token ids so
///    they still participate in stopping and min-token handling.
fn resolve_eos_token_ids(
    tokenizer_config: &HfTokenizerConfig,
    model_config: &ModelConfig,
    generation_config: &GenerationConfig,
    tokenizer: &dyn Tokenizer,
) -> (Option<u32>, BTreeSet<u32>) {
    let model_config_eos_token_ids = model_config.eos_token_ids();
    let primary_eos_token_id = tokenizer_config
        .special_tokens
        .eos_token
        .as_ref()
        .and_then(|token| tokenizer.token_to_id(token.as_str()))
        .or_else(|| model_config_eos_token_ids.first().copied());

    let mut extra_eos_token_ids = generation_config
        .eos_token_id
        .clone()
        .map(|value| value.into_set())
        .unwrap_or_default();
    extra_eos_token_ids.extend(model_config_eos_token_ids.iter().copied());
    if let Some(primary_eos_token_id) = primary_eos_token_id {
        extra_eos_token_ids.remove(&primary_eos_token_id);
    }

    (primary_eos_token_id, extra_eos_token_ids)
}

impl TextBackend for HfTextBackend {
    fn tokenizer(&self) -> DynTokenizer {
        self.tokenizer.clone()
    }

    fn has_tokenizer(&self) -> bool {
        self.files.tokenizer.is_some()
    }

    fn is_moe(&self) -> bool {
        self.model_config.is_moe()
    }

    fn model_vocab_size(&self) -> usize {
        self.model_vocab_size
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn sampling_hints(&self) -> Result<SamplingHints> {
        Ok(build_sampling_hints(
            self.generation_config_mode,
            &self.generation_config,
            self.primary_eos_token_id,
            self.extra_eos_token_ids.clone(),
        ))
    }
}

fn build_sampling_hints(
    mode: GenerationConfigMode,
    generation_config: &GenerationConfig,
    primary_eos_token_id: Option<u32>,
    extra_eos_token_ids: BTreeSet<u32>,
) -> SamplingHints {
    let sampling_config = match mode {
        // Auto inherits sampling defaults from the model's generation config.
        GenerationConfigMode::Auto => Some(generation_config),
        // Vllm skips model-provided sampling defaults. `lower_sampling_params`
        // applies vLLM fallback values; separately resolved EOS tokens remain.
        GenerationConfigMode::Vllm => None,
    };

    SamplingHints {
        primary_eos_token_id,
        extra_eos_token_ids,
        default_temperature: sampling_config.and_then(|config| config.temperature),
        default_top_p: sampling_config.and_then(|config| config.top_p),
        default_top_k: sampling_config.and_then(|config| config.top_k),
        default_min_p: sampling_config.and_then(|config| config.min_p),
        default_repetition_penalty: sampling_config.and_then(|config| config.repetition_penalty),
        default_presence_penalty: sampling_config.and_then(|config| config.presence_penalty),
        default_frequency_penalty: sampling_config.and_then(|config| config.frequency_penalty),
        default_max_tokens: sampling_config.and_then(|config| config.max_new_tokens),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use vllm_tokenizer::Tokenizer;

    use super::{
        GenerationConfig, GenerationConfigMode, HfTextBackend, HfTokenizerConfig, ModelConfig,
        ModelFileSources, ModelSource, ResolvedModelFiles, build_sampling_hints,
        resolve_eos_token_ids,
    };
    use crate::backend::TextBackend as _;

    struct FakeTokenizer;

    impl Tokenizer for FakeTokenizer {
        fn encode(
            &self,
            _text: &str,
            _add_special_tokens: bool,
        ) -> vllm_tokenizer::Result<Vec<u32>> {
            Ok(vec![])
        }

        fn encode_ordinary(&self, text: &str) -> vllm_tokenizer::Result<Vec<u32>> {
            self.encode(text, false)
        }

        fn decode(
            &self,
            _token_ids: &[u32],
            _skip_special_tokens: bool,
        ) -> vllm_tokenizer::Result<String> {
            Ok(String::new())
        }

        fn token_to_id(&self, token: &str) -> Option<u32> {
            (token == "</s>").then_some(2)
        }

        fn id_to_token(&self, id: u32) -> Option<String> {
            (id == 2).then(|| "</s>".to_string())
        }
    }

    #[tokio::test]
    async fn skip_tokenizer_init_loads_a_model_without_tokenizer_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"vocab_size": 128, "eos_token_id": [7, 9]}"#,
        )
        .unwrap();
        // Unreadable tokenizer-side metadata must not matter when skipping.
        std::fs::write(dir.path().join("tokenizer_config.json"), "not json").unwrap();
        let sources = |skip_tokenizer_init| ModelFileSources {
            model: ModelSource {
                id: dir.path().to_str().unwrap().to_string(),
                revision: None,
            },
            tokenizer: None,
            hf_config: None,
            skip_tokenizer_init,
        };

        assert!(ResolvedModelFiles::from_sources(&sources(false)).await.is_err());

        let files = ResolvedModelFiles::from_sources(&ModelFileSources {
            // Ignored when skipping, even if it does not exist.
            tokenizer: Some(ModelSource {
                id: dir.path().join("missing").to_str().unwrap().to_string(),
                revision: None,
            }),
            ..sources(true)
        })
        .await
        .unwrap();
        assert!(files.tokenizer.is_none());
        assert!(files.tokenizer_config_path.is_none());
        let backend = HfTextBackend::from_resolved_model_files(
            files,
            "test-model".to_string(),
            GenerationConfigMode::Auto,
        )
        .unwrap();
        assert!(!backend.has_tokenizer());
        assert!(backend.tokenizer().encode("hi", false).is_err());
        assert_eq!(backend.tokenizer_vocab_size(), 128);
        // Without a tokenizer, the primary EOS comes from the model config.
        let hints = backend.sampling_hints().unwrap();
        assert_eq!(hints.primary_eos_token_id, Some(7));
        assert_eq!(hints.extra_eos_token_ids, BTreeSet::from([9]));
    }

    #[test]
    fn eos_resolution_uses_model_config_when_tokenizer_has_no_eos() {
        let tokenizer_config: HfTokenizerConfig = serde_json::from_str("{}").unwrap();
        let model_config: ModelConfig =
            serde_json::from_str(r#"{"eos_token_id":[200006,200010]}"#).unwrap();
        let generation_config: GenerationConfig = serde_json::from_str("{}").unwrap();

        let (primary, extra) = resolve_eos_token_ids(
            &tokenizer_config,
            &model_config,
            &generation_config,
            &FakeTokenizer,
        );

        assert_eq!(primary, Some(200006));
        assert_eq!(extra, BTreeSet::from([200010]));
    }

    #[test]
    fn eos_resolution_keeps_tokenizer_eos_primary() {
        let tokenizer_config: HfTokenizerConfig =
            serde_json::from_str(r#"{"eos_token":"</s>"}"#).unwrap();
        let model_config: ModelConfig =
            serde_json::from_str(r#"{"eos_token_id":[2,200006]}"#).unwrap();
        let generation_config: GenerationConfig =
            serde_json::from_str(r#"{"eos_token_id":[2,200010]}"#).unwrap();

        let (primary, extra) = resolve_eos_token_ids(
            &tokenizer_config,
            &model_config,
            &generation_config,
            &FakeTokenizer,
        );

        assert_eq!(primary, Some(2));
        assert_eq!(extra, BTreeSet::from([200006, 200010]));
    }

    #[test]
    fn vllm_generation_config_keeps_eos_and_uses_neutral_sampling_defaults() {
        let generation_config: GenerationConfig = serde_json::from_str(
            r#"{
                "eos_token_id": [2, 3],
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.1,
                "repetition_penalty": 1.1,
                "presence_penalty": 1.5,
                "frequency_penalty": 0.5,
                "max_new_tokens": 4096
            }"#,
        )
        .unwrap();
        let extra_eos_token_ids = BTreeSet::from([3]);

        let hints = (
            build_sampling_hints(
                GenerationConfigMode::Auto,
                &generation_config,
                Some(2),
                extra_eos_token_ids.clone(),
            ),
            build_sampling_hints(
                GenerationConfigMode::Vllm,
                &generation_config,
                Some(2),
                extra_eos_token_ids,
            ),
        );

        expect_test::expect![[r#"
            (
                SamplingHints {
                    primary_eos_token_id: Some(
                        2,
                    ),
                    extra_eos_token_ids: {
                        3,
                    },
                    default_temperature: Some(
                        0.6,
                    ),
                    default_top_p: Some(
                        0.95,
                    ),
                    default_top_k: Some(
                        20,
                    ),
                    default_min_p: Some(
                        0.1,
                    ),
                    default_repetition_penalty: Some(
                        1.1,
                    ),
                    default_presence_penalty: Some(
                        1.5,
                    ),
                    default_frequency_penalty: Some(
                        0.5,
                    ),
                    default_max_tokens: Some(
                        4096,
                    ),
                },
                SamplingHints {
                    primary_eos_token_id: Some(
                        2,
                    ),
                    extra_eos_token_ids: {
                        3,
                    },
                    default_temperature: None,
                    default_top_p: None,
                    default_top_k: None,
                    default_min_p: None,
                    default_repetition_penalty: None,
                    default_presence_penalty: None,
                    default_frequency_penalty: None,
                    default_max_tokens: None,
                },
            )
        "#]]
        .assert_debug_eq(&hints);
    }
}
