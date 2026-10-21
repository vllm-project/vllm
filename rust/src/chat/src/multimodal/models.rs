// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Multimodal model support that lives in this repository until
//! `llm-multimodal` ships it.
//!
//! Both tables are consulted before the `llm-multimodal` registry. Once
//! `llm-multimodal` supports a model, delete its entry in the same change that
//! bumps the dependency.

use std::sync::LazyLock;

use llm_multimodal::{
    ModelMetadata, ModelProcessorSpec, ModelRegistry, VisionPreProcessor, VisionProcessorRegistry,
};

mod step3_vl;

/// Specs for models that `llm-multimodal` does not have yet, one model per
/// submodule.
static SPECS: &[&dyn ModelProcessorSpec] = &[
    &step3_vl::Step3VLVisionSpec, // Step3-VL, Step5-VL
];

/// Vision processors for the local specs above, in the same order: a model
/// its spec serves is preprocessed by the paired processor.
static VISION_PROCESSORS: &[&dyn VisionPreProcessor] = &[&step3_vl::Step3VLProcessor];

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

/// Resolve the vision preprocessor for one loaded model: the processor paired
/// with a matching local spec first, then the `llm-multimodal` registry.
pub(super) fn lookup_vision_processor(
    metadata: &ModelMetadata<'_>,
) -> Option<&'static dyn VisionPreProcessor> {
    static LLM_MULTIMODAL_REGISTRY: LazyLock<VisionProcessorRegistry> =
        LazyLock::new(VisionProcessorRegistry::with_defaults);

    if let Some(index) = SPECS.iter().position(|spec| spec.matches(metadata)) {
        return Some(VISION_PROCESSORS[index]);
    }
    LLM_MULTIMODAL_REGISTRY.find(metadata.model_id, metadata.config_model_type())
}
