// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Step3-VL vision model contract (also used by Step5-VL in vLLM).
//!
//! Mirrors vLLM's `transformers_utils/processors/step3_vl.py`: each `<im_patch>`
//! placeholder expands to per-patch spans (`<patch_start>` + 81 embed tokens +
//! `<patch_end>`, with `<patch_newline>` after row-final patches) followed by the
//! global-view span (`<im_start>` + 169 embed tokens + `<im_end>`). The per-image
//! patch counts and newline flags come from the preprocessor's `num_patches` and
//! `patch_newline_mask` model-specific outputs.

use std::collections::HashMap;

use serde_json::{Value, json};

use llm_multimodal::registry::RegistryResult;
use llm_multimodal::{
    FieldLayout, Modality, ModelMetadata, ModelProcessorSpec, ModelSpecificValue,
    PreprocessedEncoderInputs, PromptReplacement, TokenId,
};

pub(super) use processor::Step3VLProcessor;

mod processor;

pub(super) struct Step3VLVisionSpec;

impl Step3VLVisionSpec {
    const IMAGE_TOKEN: &'static str = "<im_patch>";
    const IMAGE_START_TOKEN: &'static str = "<im_start>";
    const IMAGE_END_TOKEN: &'static str = "<im_end>";
    const PATCH_START_TOKEN: &'static str = "<patch_start>";
    const PATCH_END_TOKEN: &'static str = "<patch_end>";
    const PATCH_NEWLINE_TOKEN: &'static str = "<patch_newline>";

    // Defaults from Step3VLImageProcessor; the processor reads overrides from
    // PreProcessorConfig extras, and `processor_kwargs` exposes these knobs.
    const NUM_IMAGE_FEATURE_SIZE: usize = 169;
    const NUM_PATCH_FEATURE_SIZE: usize = 81;

    /// Per-image patch counts from the preprocessor output. Missing or
    /// malformed data falls back to zero patches per image (global view only).
    fn extract_num_patches(preprocessed: &PreprocessedEncoderInputs) -> Vec<usize> {
        if let Some(ModelSpecificValue::IntTensor { data, .. }) =
            preprocessed.model_specific.get("num_patches")
        {
            return data.iter().map(|&n| n.max(0) as usize).collect();
        }
        vec![0; preprocessed.feature_token_counts.len()]
    }

    /// Flat per-patch newline flags from the preprocessor output.
    fn extract_newline_mask(preprocessed: &PreprocessedEncoderInputs) -> Vec<u32> {
        if let Some(ModelSpecificValue::UintTensor { data, .. }) =
            preprocessed.model_specific.get("patch_newline_mask")
        {
            return data.clone();
        }
        Vec::new()
    }
}

impl ModelProcessorSpec for Step3VLVisionSpec {
    fn name(&self) -> &'static str {
        "step3_vl"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        if metadata
            .config_model_type()
            .is_some_and(|mt| mt == "step5_vl" || mt == "step3_vl")
        {
            return true;
        }
        let id = metadata.model_id.to_ascii_lowercase();
        id.contains("step5-vl")
            || id.contains("step5_vl")
            || id.contains("step3-vl")
            || id.contains("step3_vl")
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok(Self::IMAGE_TOKEN.to_string())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata.token_id(Self::IMAGE_TOKEN)
    }

    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([(Modality::Image, usize::MAX)]))
    }

    fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
        Ok(json!({
            "image_size": 728,
            "patch_size": 504,
            "num_image_feature_size": Self::NUM_IMAGE_FEATURE_SIZE,
            "num_patch_feature_size": Self::NUM_PATCH_FEATURE_SIZE,
            "enable_patch": true,
        }))
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let image_token_id = metadata.token_id(Self::IMAGE_TOKEN)?;
        let im_start_id = metadata.token_id(Self::IMAGE_START_TOKEN)?;
        let im_end_id = metadata.token_id(Self::IMAGE_END_TOKEN)?;
        let patch_start_id = metadata.token_id(Self::PATCH_START_TOKEN)?;
        let patch_end_id = metadata.token_id(Self::PATCH_END_TOKEN)?;
        let patch_newline_id = metadata.token_id(Self::PATCH_NEWLINE_TOKEN)?;

        let num_patches = Self::extract_num_patches(preprocessed);
        let newline_mask = Self::extract_newline_mask(preprocessed);

        let mut replacements = Vec::with_capacity(num_patches.len());
        let mut offset = 0usize;
        for &n in &num_patches {
            // Mirrors Step3VLProcessor.get_image_repl_feature_ids: patch spans
            // first, then the global-view span.
            let mut tokens = Vec::with_capacity(
                n * (Self::NUM_PATCH_FEATURE_SIZE + 2) + Self::NUM_IMAGE_FEATURE_SIZE + 2,
            );
            for p in 0..n {
                tokens.push(patch_start_id);
                tokens.extend(std::iter::repeat_n(
                    image_token_id,
                    Self::NUM_PATCH_FEATURE_SIZE,
                ));
                tokens.push(patch_end_id);
                if newline_mask.get(offset + p).copied().unwrap_or(0) == 1 {
                    tokens.push(patch_newline_id);
                }
            }
            offset += n;

            tokens.push(im_start_id);
            tokens.extend(std::iter::repeat_n(
                image_token_id,
                Self::NUM_IMAGE_FEATURE_SIZE,
            ));
            tokens.push(im_end_id);

            replacements.push(PromptReplacement::sequence(
                Modality::Image,
                Self::IMAGE_TOKEN,
                tokens,
            ));
        }
        Ok(replacements)
    }

    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        HashMap::from([
            ("pixel_values".to_string(), FieldLayout::Batched),
            (
                "patch_pixel_values".to_string(),
                FieldLayout::flat("num_patches"),
            ),
            (
                "patch_newline_mask".to_string(),
                FieldLayout::flat("num_patches"),
            ),
            ("num_patches".to_string(), FieldLayout::Batched),
        ])
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use llm_multimodal::registry::ModelRegistryError;
    use llm_multimodal::{ImageSize, Tokenizer};

    use super::*;

    struct TestTokenizer {
        vocab: HashMap<String, u32>,
    }

    impl TestTokenizer {
        fn new(pairs: &[(&str, u32)]) -> Self {
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

    fn test_preprocessed_with_tokens(
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

    const IM_PATCH: u32 = 100;
    const IM_START: u32 = 101;
    const IM_END: u32 = 102;
    const PATCH_START: u32 = 103;
    const PATCH_END: u32 = 104;
    const PATCH_NEWLINE: u32 = 105;

    fn step3_tokenizer() -> TestTokenizer {
        TestTokenizer::new(&[
            ("<im_patch>", IM_PATCH),
            ("<im_start>", IM_START),
            ("<im_end>", IM_END),
            ("<patch_start>", PATCH_START),
            ("<patch_end>", PATCH_END),
            ("<patch_newline>", PATCH_NEWLINE),
        ])
    }

    fn metadata<'a>(tokenizer: &'a TestTokenizer, config: &'a Value) -> ModelMetadata<'a> {
        ModelMetadata {
            model_id: "stepfun/Step5-VL",
            tokenizer,
            config,
        }
    }

    fn preprocessed_with_patches(
        num_patches: &[i64],
        newline_mask: &[u32],
    ) -> PreprocessedEncoderInputs {
        let feature_counts: Vec<usize> =
            num_patches.iter().map(|&n| 169 + 81 * n as usize).collect();
        let sizes: Vec<ImageSize> = num_patches.iter().map(|_| ImageSize::new(1024, 768)).collect();
        test_preprocessed_with_tokens(&sizes, &feature_counts)
            .with_extra(
                "num_patches",
                ModelSpecificValue::int_1d(num_patches.to_vec()),
            )
            .with_extra(
                "patch_newline_mask",
                ModelSpecificValue::uint_1d(newline_mask.to_vec()),
            )
    }

    #[test]
    fn matches_step5_vl_and_step3_vl_model_types() {
        let tokenizer = step3_tokenizer();
        for model_type in ["step5_vl", "step3_vl"] {
            let config = json!({"model_type": model_type});
            let metadata = ModelMetadata {
                model_id: "some-custom-model",
                tokenizer: &tokenizer,
                config: &config,
            };
            let spec = super::super::lookup_spec(&metadata).expect("step3_vl spec");
            assert_eq!(spec.name(), "step3_vl");
        }
    }

    #[test]
    fn matches_model_id_case_insensitively() {
        let tokenizer = step3_tokenizer();
        let config = json!({});
        for model_id in ["stepfun/Step5-VL", "stepfun/step3-vl", "STEP5_VL"] {
            let metadata = ModelMetadata {
                model_id,
                tokenizer: &tokenizer,
                config: &config,
            };
            assert!(Step3VLVisionSpec.matches(&metadata), "{model_id}");
        }
    }

    #[test]
    fn does_not_match_unrelated_models() {
        let tokenizer = step3_tokenizer();
        let config = json!({"model_type": "step5"});
        let metadata = ModelMetadata {
            model_id: "stepfun/Step5-Text",
            tokenizer: &tokenizer,
            config: &config,
        };
        assert!(!Step3VLVisionSpec.matches(&metadata));
    }

    #[test]
    fn prompt_replacements_expand_patches_and_global_view() {
        let tokenizer = step3_tokenizer();
        let config = json!({"model_type": "step5_vl"});
        let metadata = metadata(&tokenizer, &config);

        // Image 0: 2 patches, newline after the first; image 1: 1 patch, no newline.
        let preprocessed = preprocessed_with_patches(&[2, 1], &[1, 0, 0]);
        let replacements = Step3VLVisionSpec.prompt_replacements(&metadata, &preprocessed).unwrap();

        assert_eq!(replacements.len(), 2);

        let mut expected0 = vec![PATCH_START as TokenId];
        expected0.extend(std::iter::repeat_n(IM_PATCH as TokenId, 81));
        expected0.push(PATCH_END as TokenId);
        expected0.push(PATCH_NEWLINE as TokenId);
        expected0.push(PATCH_START as TokenId);
        expected0.extend(std::iter::repeat_n(IM_PATCH as TokenId, 81));
        expected0.push(PATCH_END as TokenId);
        expected0.push(IM_START as TokenId);
        expected0.extend(std::iter::repeat_n(IM_PATCH as TokenId, 169));
        expected0.push(IM_END as TokenId);
        assert_eq!(replacements[0].tokens, expected0);
        assert_eq!(replacements[0].modality, Modality::Image);
        assert_eq!(replacements[0].placeholder_token, "<im_patch>");

        let mut expected1 = vec![PATCH_START as TokenId];
        expected1.extend(std::iter::repeat_n(IM_PATCH as TokenId, 81));
        expected1.push(PATCH_END as TokenId);
        expected1.push(IM_START as TokenId);
        expected1.extend(std::iter::repeat_n(IM_PATCH as TokenId, 169));
        expected1.push(IM_END as TokenId);
        assert_eq!(replacements[1].tokens, expected1);
    }

    #[test]
    fn prompt_replacements_handle_zero_patches() {
        let tokenizer = step3_tokenizer();
        let config = json!({"model_type": "step5_vl"});
        let metadata = metadata(&tokenizer, &config);

        let preprocessed = preprocessed_with_patches(&[0], &[]);
        let replacements = Step3VLVisionSpec.prompt_replacements(&metadata, &preprocessed).unwrap();

        assert_eq!(replacements.len(), 1);
        let mut expected = vec![IM_START as TokenId];
        expected.extend(std::iter::repeat_n(IM_PATCH as TokenId, 169));
        expected.push(IM_END as TokenId);
        assert_eq!(replacements[0].tokens, expected);
    }

    #[test]
    fn prompt_replacements_require_special_tokens_in_vocab() {
        let tokenizer = TestTokenizer::new(&[("<im_patch>", IM_PATCH)]);
        let config = json!({"model_type": "step5_vl"});
        let metadata = metadata(&tokenizer, &config);

        let preprocessed = preprocessed_with_patches(&[1], &[0]);
        let result = Step3VLVisionSpec.prompt_replacements(&metadata, &preprocessed);
        assert!(matches!(
            result,
            Err(ModelRegistryError::TokenNotFound { .. })
        ));
    }

    #[test]
    fn field_layouts_mark_patch_fields_flat() {
        let layouts = Step3VLVisionSpec.field_layouts();
        assert_eq!(
            layouts.get("patch_pixel_values"),
            Some(&FieldLayout::flat("num_patches"))
        );
        assert_eq!(
            layouts.get("patch_newline_mask"),
            Some(&FieldLayout::flat("num_patches"))
        );
        assert_eq!(layouts.get("pixel_values"), Some(&FieldLayout::Batched));
        assert_eq!(layouts.get("num_patches"), Some(&FieldLayout::Batched));
    }
}
