// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context as _, Result};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest as _, Sha256};
use vllm_chat::multimodal::MmLimitPerPrompt;
use vllm_engine_core_client::protocol::dtype::ModelDtype;
use vllm_text::backend::hf::ResolvedModelFiles;

use crate::config::Config;
use crate::render::RenderConfig;

const RENDER_FINGERPRINT_VERSION: u8 = 1;

#[derive(Serialize)]
struct RenderFingerprintOptions {
    version: u8,
    vllm_version: &'static str,
    model: String,
    renderer: String,
    tool_call_parser: String,
    reasoning_parser: String,
    tool_strict_level: Option<String>,
    configured_chat_template: Option<String>,
    hf_overrides: BTreeMap<String, Value>,
    default_chat_template_kwargs: BTreeMap<String, Value>,
    reasoning_effort_map: BTreeMap<String, String>,
    trust_request_chat_template: bool,
    exclude_tools_when_tool_choice_none: bool,
    chat_template_content_format: String,
    max_model_len: u64,
    language_model_only: bool,
    model_dtype: Option<String>,
    multimodal_limits: BTreeMap<String, Value>,
}

pub(crate) fn for_render_server(
    config: &RenderConfig,
    model_dtype: Option<ModelDtype>,
    files: &ResolvedModelFiles,
) -> Result<String> {
    fingerprint(
        RenderFingerprintOptions {
            version: RENDER_FINGERPRINT_VERSION,
            vllm_version: vllm_build_info::VERSION,
            model: config.model.clone(),
            renderer: config.renderer.to_string(),
            tool_call_parser: config.tool_call_parser.to_string(),
            reasoning_parser: config.reasoning_parser.to_string(),
            tool_strict_level: config.tool_strict_level.map(|level| level.to_string()),
            configured_chat_template: configured_chat_template(config.chat_template.as_deref())?,
            hf_overrides: canonical_map(&config.hf_overrides.0),
            default_chat_template_kwargs: canonical_map(&config.default_chat_template_kwargs),
            reasoning_effort_map: config
                .reasoning_effort_map
                .iter()
                .map(|(effort, value)| (effort.as_str().to_string(), value.clone()))
                .collect(),
            trust_request_chat_template: config.trust_request_chat_template,
            exclude_tools_when_tool_choice_none: config.exclude_tools_when_tool_choice_none,
            chat_template_content_format: config.chat_template_content_format.to_string(),
            max_model_len: config.max_model_len.map(u64::from).unwrap_or(u64::MAX),
            language_model_only: config.language_model_only,
            model_dtype: model_dtype.map(|dtype| dtype.as_str().to_string()),
            multimodal_limits: multimodal_limits(&config.limit_mm_per_prompt),
        },
        &file_fingerprints(files)?,
    )
}

pub(crate) fn for_server(
    config: &Config,
    max_model_len: u64,
    model_dtype: ModelDtype,
    files: &ResolvedModelFiles,
) -> Result<String> {
    fingerprint(
        RenderFingerprintOptions {
            version: RENDER_FINGERPRINT_VERSION,
            vllm_version: vllm_build_info::VERSION,
            model: config.model.clone(),
            renderer: config.renderer.to_string(),
            tool_call_parser: config.tool_call_parser.to_string(),
            reasoning_parser: config.reasoning_parser.to_string(),
            tool_strict_level: config.tool_strict_level.map(|level| level.to_string()),
            configured_chat_template: configured_chat_template(config.chat_template.as_deref())?,
            hf_overrides: canonical_map(&config.hf_overrides.0),
            default_chat_template_kwargs: config
                .default_chat_template_kwargs
                .as_ref()
                .map_or_else(BTreeMap::new, canonical_map),
            reasoning_effort_map: config
                .reasoning_effort_map
                .iter()
                .map(|(effort, value)| (effort.as_str().to_string(), value.clone()))
                .collect(),
            trust_request_chat_template: config.trust_request_chat_template,
            exclude_tools_when_tool_choice_none: config.exclude_tools_when_tool_choice_none,
            chat_template_content_format: config.chat_template_content_format.to_string(),
            max_model_len,
            language_model_only: config.language_model_only,
            model_dtype: Some(model_dtype.as_str().to_string()),
            multimodal_limits: multimodal_limits(&config.limit_mm_per_prompt),
        },
        &file_fingerprints(files)?,
    )
}

fn fingerprint(
    options: RenderFingerprintOptions,
    file_fingerprints: &BTreeMap<String, Option<String>>,
) -> Result<String> {
    #[derive(Serialize)]
    struct Inputs<'a> {
        options: RenderFingerprintOptions,
        files: &'a BTreeMap<String, Option<String>>,
    }

    let payload = serde_json::to_vec(&Inputs {
        options,
        files: file_fingerprints,
    })
    .context("failed to serialize render fingerprint inputs")?;
    Ok(format!(
        "vllm-render-v{RENDER_FINGERPRINT_VERSION}-{:x}",
        Sha256::digest(payload)
    ))
}

fn file_fingerprints(files: &ResolvedModelFiles) -> Result<BTreeMap<String, Option<String>>> {
    [
        ("chat_template", files.chat_template_path.as_deref()),
        ("config", files.config_path.as_deref()),
        ("generation_config", files.generation_config_path.as_deref()),
        (
            "preprocessor_config",
            files.preprocessor_config_path.as_deref(),
        ),
        ("processor_config", files.processor_config_path.as_deref()),
        (
            "tokenizer",
            files.tokenizer.as_ref().map(|source| source.path()),
        ),
        ("tokenizer_config", files.tokenizer_config_path.as_deref()),
        (
            "video_preprocessor_config",
            files.video_preprocessor_config_path.as_deref(),
        ),
    ]
    .into_iter()
    .map(|(name, path)| Ok((name.to_string(), fingerprint_file(path)?)))
    .collect()
}

fn fingerprint_file(path: Option<&Path>) -> Result<Option<String>> {
    path.map(|path| {
        let bytes = std::fs::read(path)
            .with_context(|| format!("failed to read renderer input {}", path.display()))?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    })
    .transpose()
}

fn configured_chat_template(template: Option<&str>) -> Result<Option<String>> {
    template.map(vllm_chat::resolve_chat_template).transpose().map_err(Into::into)
}

fn multimodal_limits(limits: &MmLimitPerPrompt) -> BTreeMap<String, Value> {
    limits
        .iter()
        .map(|(modality, limit)| {
            (
                modality.as_str().to_string(),
                serde_json::to_value(limit).expect("multimodal limits serialize"),
            )
        })
        .collect()
}

fn canonical_map<'a>(
    values: impl IntoIterator<Item = (&'a String, &'a Value)>,
) -> BTreeMap<String, Value> {
    values
        .into_iter()
        .map(|(key, value)| (key.clone(), canonical_json(value)))
        .collect()
}

fn canonical_json(value: &Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.iter().map(canonical_json).collect()),
        Value::Object(values) => Value::Object(
            values.iter().map(|(key, value)| (key.clone(), canonical_json(value))).collect(),
        ),
        value => value.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options() -> RenderFingerprintOptions {
        RenderFingerprintOptions {
            version: RENDER_FINGERPRINT_VERSION,
            vllm_version: "0.1.0",
            model: "org/model".to_string(),
            renderer: "hf".to_string(),
            tool_call_parser: "auto".to_string(),
            reasoning_parser: "auto".to_string(),
            tool_strict_level: None,
            configured_chat_template: None,
            hf_overrides: BTreeMap::new(),
            default_chat_template_kwargs: BTreeMap::new(),
            reasoning_effort_map: BTreeMap::new(),
            trust_request_chat_template: false,
            exclude_tools_when_tool_choice_none: false,
            chat_template_content_format: "auto".to_string(),
            max_model_len: 4096,
            language_model_only: false,
            model_dtype: Some("bfloat16".to_string()),
            multimodal_limits: BTreeMap::new(),
        }
    }

    #[test]
    fn render_fingerprint_is_stable_for_the_same_inputs() {
        let files = BTreeMap::from([("tokenizer".to_string(), Some("abc".to_string()))]);

        let first = fingerprint(options(), &files).unwrap();
        let second = fingerprint(options(), &files).unwrap();

        assert_eq!(first, second);
        assert!(first.starts_with("vllm-render-v1-"));
    }

    #[test]
    fn render_fingerprint_changes_with_rendering_inputs() {
        let files = BTreeMap::from([("tokenizer".to_string(), Some("abc".to_string()))]);
        let baseline = fingerprint(options(), &files).unwrap();

        let mut changed_options = options();
        changed_options
            .default_chat_template_kwargs
            .insert("enable_thinking".to_string(), serde_json::json!(false));
        assert_ne!(baseline, fingerprint(changed_options, &files).unwrap());

        let mut changed_options = options();
        changed_options
            .hf_overrides
            .insert("rope_scaling".to_string(), serde_json::json!({"factor": 4}));
        assert_ne!(baseline, fingerprint(changed_options, &files).unwrap());

        let changed_files = BTreeMap::from([("tokenizer".to_string(), Some("def".to_string()))]);
        assert_ne!(baseline, fingerprint(options(), &changed_files).unwrap());
    }

    #[test]
    fn file_fingerprint_tracks_content() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("tokenizer.json");
        std::fs::write(&file, "first").unwrap();
        let first = fingerprint_file(Some(&file)).unwrap();

        std::fs::write(&file, "second").unwrap();
        let second = fingerprint_file(Some(&file)).unwrap();

        assert_ne!(first, second);
    }
}
