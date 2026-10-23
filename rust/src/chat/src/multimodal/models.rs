// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Multimodal model support that lives in this repository until
//! `llm-multimodal` ships it.
//!
//! The table is consulted before the `llm-multimodal` registry. Once
//! `llm-multimodal` supports a model, delete its entry in the same change that
//! bumps the dependency.

use std::sync::LazyLock;

use llm_multimodal::{ModelMetadata, ModelProcessorSpec, ModelRegistry};

mod muse_spark;
mod step3_vl;

/// Specs for models that `llm-multimodal` does not have yet, one model per
/// submodule.
static SPECS: &[&dyn ModelProcessorSpec] = &[
    &step3_vl::Step3VLVisionSpec, // Step3-VL, Step5-VL
    &muse_spark::MuseSparkSpec,   // MuseSpark (image + video + audio)
];

/// `(model_type, target model_type)` pairs for checkpoints that reuse another
/// family's `llm-multimodal` spec unchanged. The target spec still sees this
/// model's own metadata.
const MODEL_TYPE_ALIASES: &[(&str, &str)] = &[];

/// Resolve the model processor spec for one loaded model: a local spec first,
/// then the `llm-multimodal` registry, through a model type alias if any.
pub(super) fn lookup_spec(metadata: &ModelMetadata<'_>) -> Option<&'static dyn ModelProcessorSpec> {
    static LLM_MULTIMODAL_REGISTRY: LazyLock<ModelRegistry> = LazyLock::new(ModelRegistry::new);

    if let Some(spec) = SPECS.iter().copied().find(|spec| spec.matches(metadata)) {
        return Some(spec);
    }
    let target = metadata.config_model_type().and_then(|model_type| {
        MODEL_TYPE_ALIASES
            .iter()
            .find_map(|&(alias, target)| (alias == model_type).then_some(target))
    });
    let Some(target) = target else {
        return LLM_MULTIMODAL_REGISTRY.lookup(metadata);
    };
    let mut config = metadata.config.clone();
    config["model_type"] = target.into();
    LLM_MULTIMODAL_REGISTRY.lookup(&ModelMetadata {
        config: &config,
        ..*metadata
    })
}

/// Ported from `llm-multimodal`'s `registry::test_helpers`, which is
/// `pub(super)` there and cannot be imported.
#[cfg(test)]
pub(crate) mod test_helpers {
    use std::collections::HashMap;

    use llm_multimodal::registry::Tokenizer;
    use llm_multimodal::{ImageSize, PreprocessedEncoderInputs};

    pub struct TestTokenizer {
        vocab: HashMap<String, u32>,
    }

    impl TestTokenizer {
        pub fn new(pairs: &[(&str, u32)]) -> Self {
            let vocab = pairs.iter().map(|(token, id)| ((*token).to_string(), *id)).collect();
            Self { vocab }
        }
    }

    impl Tokenizer for TestTokenizer {
        fn token_to_id(&self, token: &str) -> Option<u32> {
            self.vocab.get(token).copied()
        }

        fn id_to_token(&self, id: u32) -> Option<String> {
            self.vocab.iter().find(|&(_, &v)| v == id).map(|(k, _)| k.clone())
        }

        fn encode_text(&self, _text: &str) -> Option<Vec<u32>> {
            Some(Vec::new())
        }
    }

    pub fn test_preprocessed_with_tokens(
        item_sizes: &[ImageSize],
        feature_token_counts: &[usize],
    ) -> PreprocessedEncoderInputs {
        let sizes: Vec<(u32, u32)> = item_sizes.iter().map(|s| (s.height, s.width)).collect();
        PreprocessedEncoderInputs {
            encoder_input: ndarray::ArrayD::zeros(vec![1, 3, 336, 336]).into(),
            feature_token_counts: feature_token_counts.to_vec(),
            item_sizes: sizes,
            model_specific: HashMap::new(),
        }
    }
}
