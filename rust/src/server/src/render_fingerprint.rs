// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use anyhow::{Context as _, Result};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest as _, Sha256};
use tracing::{debug, info};
use vllm_chat::RendererSelection;
use vllm_text::backend::hf::ResolvedModelFiles;

use crate::config::ServerConfig;
use crate::render::RenderConfig;

const RENDER_FINGERPRINT_SCHEMA: u8 = 1;

#[derive(Debug, Serialize)]
struct RenderFingerprintManifest {
    schema: u8,
    files: BTreeMap<String, Option<String>>,
    flags: RenderFingerprintFlags,
}

#[derive(Debug, Serialize)]
struct RenderFingerprintFlags {
    tokenizer_mode: RendererSelection,
    config_format: &'static str,
    trust_remote_code: bool,
    chat_template: Option<String>,
    chat_template_fallback: Option<String>,
    chat_template_content_format: String,
    default_chat_template_kwargs: Option<BTreeMap<String, Value>>,
    reasoning_parser: Option<String>,
    reasoning_effort_map: Option<BTreeMap<String, String>>,
    exclude_tools_when_tool_choice_none: bool,
    hf_overrides: Option<BTreeMap<String, Value>>,
    mm_processor_kwargs: Option<BTreeMap<String, Value>>,
    media_io_kwargs: Option<Value>,
    language_model_only: bool,
    interleave_mm_strings: bool,
    video_pruning_rate: Option<Value>,
    video_pruning_method: Option<String>,
    max_model_len: Option<u64>,
    trust_request_chat_template: bool,
}

pub(crate) async fn for_render_server(
    config: &RenderConfig,
    renderer: Option<RendererSelection>,
    tokenizer_library: Option<&str>,
    multimodal: bool,
    files: &ResolvedModelFiles,
) -> Result<String> {
    fingerprint_with_identity(
        renderer,
        tokenizer_library,
        manifest(
            flags(
                config.renderer,
                config.chat_template.as_deref(),
                optional_canonical_map(&config.hf_overrides.0),
                optional_canonical_map(&config.default_chat_template_kwargs),
                optional_reasoning_effort_map(&config.reasoning_effort_map),
                reasoning_parser(&config.reasoning_parser),
                config.exclude_tools_when_tool_choice_none,
                config.chat_template_content_format.to_string(),
                optional_canonical_map(&config.mm_processor_kwargs),
                config.language_model_only,
                multimodal.then(|| config.max_model_len.map(u64::from)).flatten(),
                config.trust_request_chat_template,
            )?,
            file_fingerprints(files).await?,
        ),
    )
}

pub(crate) async fn for_server(
    config: &ServerConfig,
    renderer: Option<RendererSelection>,
    tokenizer_library: Option<&str>,
    max_model_len: u64,
    multimodal: bool,
    files: &ResolvedModelFiles,
) -> Result<String> {
    let default_chat_template_kwargs =
        config.default_chat_template_kwargs.as_ref().and_then(optional_canonical_map);
    fingerprint_with_identity(
        renderer,
        tokenizer_library,
        manifest(
            flags(
                config.renderer,
                config.chat_template.as_deref(),
                optional_canonical_map(&config.hf_overrides.0),
                default_chat_template_kwargs,
                optional_reasoning_effort_map(&config.reasoning_effort_map),
                reasoning_parser(&config.reasoning_parser),
                config.exclude_tools_when_tool_choice_none,
                config.chat_template_content_format.to_string(),
                optional_canonical_map(&config.mm_processor_kwargs),
                config.language_model_only,
                multimodal.then_some(max_model_len),
                config.trust_request_chat_template,
            )?,
            file_fingerprints(files).await?,
        ),
    )
}

fn flags(
    tokenizer_mode: RendererSelection,
    chat_template: Option<&str>,
    hf_overrides: Option<BTreeMap<String, Value>>,
    default_chat_template_kwargs: Option<BTreeMap<String, Value>>,
    reasoning_effort_map: Option<BTreeMap<String, String>>,
    reasoning_parser: Option<String>,
    exclude_tools_when_tool_choice_none: bool,
    chat_template_content_format: String,
    mm_processor_kwargs: Option<BTreeMap<String, Value>>,
    language_model_only: bool,
    max_model_len: Option<u64>,
    trust_request_chat_template: bool,
) -> Result<RenderFingerprintFlags> {
    Ok(RenderFingerprintFlags {
        tokenizer_mode,
        config_format: "auto",
        trust_remote_code: false,
        chat_template: configured_chat_template(chat_template)?,
        chat_template_fallback: None,
        chat_template_content_format,
        default_chat_template_kwargs,
        reasoning_parser,
        reasoning_effort_map,
        exclude_tools_when_tool_choice_none,
        hf_overrides,
        mm_processor_kwargs,
        media_io_kwargs: None,
        language_model_only,
        interleave_mm_strings: false,
        video_pruning_rate: None,
        video_pruning_method: None,
        max_model_len,
        trust_request_chat_template,
    })
}

fn manifest(
    flags: RenderFingerprintFlags,
    files: BTreeMap<String, Option<String>>,
) -> RenderFingerprintManifest {
    RenderFingerprintManifest {
        schema: RENDER_FINGERPRINT_SCHEMA,
        files,
        flags,
    }
}

fn fingerprint_with_identity(
    renderer: Option<RendererSelection>,
    tokenizer_library: Option<&str>,
    manifest: RenderFingerprintManifest,
) -> Result<String> {
    let payload =
        serde_jcs::to_vec(&manifest).context("failed to serialize render fingerprint manifest")?;
    let renderer = renderer.map(|renderer| renderer.to_string()).unwrap_or_else(|| "none".into());
    let tokenizer_library = tokenizer_library.unwrap_or("none");
    let tag = format!(
        "vllm-rs/{};renderer={renderer};tokenizer={tokenizer_library}",
        vllm_build_info::VERSION,
    );
    let render_fingerprint = format!(
        "rf{RENDER_FINGERPRINT_SCHEMA}:{:x}:{tag}",
        Sha256::digest(payload)
    );
    debug!(?manifest, "computed render fingerprint manifest");
    info!(%render_fingerprint, "computed render fingerprint");
    Ok(render_fingerprint)
}

#[cfg(test)]
fn fingerprint(renderer: RendererSelection, manifest: RenderFingerprintManifest) -> Result<String> {
    fingerprint_with_identity(Some(renderer), Some("fastokens/test"), manifest)
}

async fn file_fingerprints(files: &ResolvedModelFiles) -> Result<BTreeMap<String, Option<String>>> {
    files
        .fingerprint_files()
        .await?
        .into_iter()
        .map(|(name, path)| Ok((name, fingerprint_file(path.as_deref())?)))
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
    template
        .map(|template| {
            let path = Path::new(template);
            if path.exists() {
                std::fs::read_to_string(path)
                    .with_context(|| format!("failed to read chat template {}", path.display()))
            } else {
                vllm_chat::resolve_chat_template(template).map_err(Into::into)
            }
        })
        .transpose()
}

fn optional_canonical_map<'a>(
    values: impl IntoIterator<Item = (&'a String, &'a Value)>,
) -> Option<BTreeMap<String, Value>> {
    let values = values
        .into_iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect::<BTreeMap<_, _>>();
    (!values.is_empty()).then_some(values)
}

fn optional_reasoning_effort_map(
    values: &HashMap<crate::ReasoningEffort, String>,
) -> Option<BTreeMap<String, String>> {
    let values = values
        .into_iter()
        .map(|(effort, value)| (effort.as_str().to_string(), value.clone()))
        .collect::<BTreeMap<_, _>>();
    (!values.is_empty()).then_some(values)
}

fn reasoning_parser(selection: &vllm_chat::ParserSelection) -> Option<String> {
    (!matches!(selection, vllm_chat::ParserSelection::Auto)).then(|| selection.to_string())
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use vllm_chat::{ChatTemplateContentFormatOption, ParserSelection, RendererSelection};
    use vllm_engine_core_client::protocol::structured_outputs::StructuredOutputBackend;

    use super::*;
    use crate::config::HttpListenerMode;
    use crate::fingerprint::FingerprintMode;

    const TEST_TOKENIZER_LIBRARY: &str = "fastokens/test";

    fn render_config(model: &str) -> RenderConfig {
        RenderConfig {
            model: model.to_string(),
            revision: None,
            tokenizer: None,
            hf_config: None,
            skip_tokenizer_init: false,
            hf_token: None,
            hf_overrides: Default::default(),
            served_model_name: Vec::new(),
            skip_model_name_validation: false,
            host: "127.0.0.1".to_string(),
            port: 8000,
            grpc_port: None,
            tool_call_parser: ParserSelection::Auto,
            reasoning_parser: ParserSelection::Auto,
            tool_strict_level: None,
            renderer: RendererSelection::Hf,
            chat_template: None,
            default_chat_template_kwargs: HashMap::new(),
            reasoning_effort_map: HashMap::new(),
            trust_request_chat_template: false,
            exclude_tools_when_tool_choice_none: false,
            chat_template_content_format: ChatTemplateContentFormatOption::Auto,
            max_model_len: Some(4096),
            max_logprobs: None,
            lora_modules: Vec::new(),
            language_model_only: false,
            dtype: None,
            limit_mm_per_prompt: Default::default(),
            mm_processor_kwargs: HashMap::new(),
            tls: None,
        }
    }

    fn server_config(model: &str) -> ServerConfig {
        ServerConfig {
            model: model.to_string(),
            revision: None,
            tokenizer: None,
            hf_config: None,
            skip_tokenizer_init: false,
            hf_token: None,
            hf_overrides: Default::default(),
            generation_config: Default::default(),
            override_generation_config: Default::default(),
            served_model_name: Vec::new(),
            listener_mode: HttpListenerMode::InheritedFd { fd: -1 },
            grpc_listener_mode: None,
            tool_call_parser: ParserSelection::Auto,
            reasoning_parser: ParserSelection::Auto,
            tool_strict_level: None,
            structured_outputs_backend: StructuredOutputBackend::default(),
            renderer: RendererSelection::Hf,
            language_model_only: false,
            chat_template: None,
            decisions_read_strategy: None,
            default_chat_template_kwargs: None,
            reasoning_effort_map: HashMap::new(),
            trust_request_chat_template: false,
            exclude_tools_when_tool_choice_none: false,
            limit_mm_per_prompt: Default::default(),
            mm_processor_kwargs: HashMap::new(),
            chat_template_content_format: ChatTemplateContentFormatOption::Auto,
            max_logprobs: None,
            api_server_options: Default::default(),
            fingerprint_mode: FingerprintMode::default(),
            cors: Default::default(),
            access_log: Default::default(),
            tls: None,
            root_path: None,
            api_keys: Vec::new(),
            keep_alive_on_engine_death: false,
            shutdown_timeout: Duration::ZERO,
            keep_alive_timeout: Duration::ZERO,
        }
    }

    async fn model_files() -> (tempfile::TempDir, ResolvedModelFiles) {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"tokenizer_class":"TokenizersBackend"}"#,
        )
        .unwrap();
        std::fs::write(dir.path().join("config.json"), "{}").unwrap();
        let files = ResolvedModelFiles::new(dir.path().to_str().unwrap(), None).await.unwrap();
        (dir, files)
    }

    fn test_manifest() -> RenderFingerprintManifest {
        manifest(
            RenderFingerprintFlags {
                tokenizer_mode: RendererSelection::Auto,
                config_format: "auto",
                trust_remote_code: false,
                chat_template: None,
                chat_template_fallback: None,
                chat_template_content_format: "auto".to_string(),
                default_chat_template_kwargs: None,
                reasoning_parser: None,
                reasoning_effort_map: None,
                exclude_tools_when_tool_choice_none: false,
                hf_overrides: None,
                mm_processor_kwargs: None,
                media_io_kwargs: None,
                language_model_only: false,
                interleave_mm_strings: false,
                video_pruning_rate: None,
                video_pruning_method: None,
                max_model_len: None,
                trust_request_chat_template: false,
            },
            BTreeMap::from([("tokenizer.json".to_string(), Some("abc".to_string()))]),
        )
    }

    #[test]
    fn render_fingerprint_is_stable_for_the_same_inputs() {
        let first = fingerprint(RendererSelection::Hf, test_manifest()).unwrap();
        let second = fingerprint(RendererSelection::Hf, test_manifest()).unwrap();

        assert_eq!(first, second);
        assert!(first.starts_with("rf1:"));
        assert_eq!(first.split(':').count(), 3);
        assert!(first.contains(":vllm-rs/"));
        assert!(first.contains(";renderer=hf;tokenizer=fastokens/test"));
    }

    #[test]
    fn render_fingerprint_changes_with_rendering_inputs() {
        let baseline = fingerprint(RendererSelection::Hf, test_manifest()).unwrap();

        let mut changed_manifest = test_manifest();
        changed_manifest.flags.default_chat_template_kwargs = Some(BTreeMap::from([(
            "enable_thinking".to_string(),
            serde_json::json!(false),
        )]));
        assert_ne!(
            baseline,
            fingerprint(RendererSelection::Hf, changed_manifest).unwrap()
        );

        let mut changed_manifest = test_manifest();
        changed_manifest.flags.hf_overrides = Some(BTreeMap::from([(
            "rope_scaling".to_string(),
            serde_json::json!({"factor": 4}),
        )]));
        assert_ne!(
            baseline,
            fingerprint(RendererSelection::Hf, changed_manifest).unwrap()
        );

        let mut changed_manifest = test_manifest();
        changed_manifest
            .files
            .insert("tokenizer.json".to_string(), Some("def".to_string()));
        assert_ne!(
            baseline,
            fingerprint(RendererSelection::Hf, changed_manifest).unwrap()
        );
    }

    #[test]
    fn render_fingerprint_canonicalizes_nested_json_keys() {
        let mut first_nested = serde_json::Map::new();
        first_nested.insert("second".to_string(), serde_json::json!(2));
        first_nested.insert("first".to_string(), serde_json::json!(1));
        let mut second_nested = serde_json::Map::new();
        second_nested.insert("first".to_string(), serde_json::json!(1));
        second_nested.insert("second".to_string(), serde_json::json!(2));

        let mut first_manifest = test_manifest();
        first_manifest.flags.hf_overrides = Some(BTreeMap::from([(
            "nested".to_string(),
            serde_json::json!([{"value": first_nested}]),
        )]));
        let mut second_manifest = test_manifest();
        second_manifest.flags.hf_overrides = Some(BTreeMap::from([(
            "nested".to_string(),
            serde_json::json!([{"value": second_nested}]),
        )]));

        assert_eq!(
            fingerprint(RendererSelection::Hf, first_manifest).unwrap(),
            fingerprint(RendererSelection::Hf, second_manifest).unwrap()
        );
    }

    #[tokio::test]
    async fn text_only_render_and_server_fingerprints_match() {
        let (_dir, files) = model_files().await;
        let mut render_config = render_config("org/model");
        render_config.max_model_len = None;
        let mut server_config = server_config("/models/model");
        server_config.tool_call_parser = ParserSelection::Explicit("hermes".to_string());
        server_config.tool_strict_level = Some(vllm_chat::ToolStrictLevel::Parameter);

        assert_eq!(
            for_render_server(
                &render_config,
                Some(RendererSelection::Hf),
                Some(TEST_TOKENIZER_LIBRARY),
                false,
                &files,
            )
            .await
            .unwrap(),
            for_server(
                &server_config,
                Some(RendererSelection::Hf),
                Some(TEST_TOKENIZER_LIBRARY),
                4096,
                false,
                &files,
            )
            .await
            .unwrap()
        );
    }

    #[tokio::test]
    async fn multimodal_render_and_server_fingerprints_require_matching_max_model_len() {
        let (_dir, files) = model_files().await;
        let render_config = render_config("org/model");
        let server_config = server_config("/models/model");
        let render_fingerprint = for_render_server(
            &render_config,
            Some(RendererSelection::Hf),
            Some(TEST_TOKENIZER_LIBRARY),
            true,
            &files,
        )
        .await
        .unwrap();
        let matching_server_fingerprint = for_server(
            &server_config,
            Some(RendererSelection::Hf),
            Some(TEST_TOKENIZER_LIBRARY),
            4096,
            true,
            &files,
        )
        .await
        .unwrap();
        let mismatched_server_fingerprint = for_server(
            &server_config,
            Some(RendererSelection::Hf),
            Some(TEST_TOKENIZER_LIBRARY),
            2048,
            true,
            &files,
        )
        .await
        .unwrap();

        assert_eq!(render_fingerprint, matching_server_fingerprint);
        assert_ne!(render_fingerprint, mismatched_server_fingerprint);
    }

    #[tokio::test]
    async fn render_fingerprint_ignores_non_rendering_options() {
        let (_dir, files) = model_files().await;
        let baseline = for_render_server(
            &render_config("org/model"),
            Some(RendererSelection::Hf),
            Some(TEST_TOKENIZER_LIBRARY),
            false,
            &files,
        )
        .await
        .unwrap();
        let mut changed = render_config("/models/model");
        changed.tool_call_parser = ParserSelection::Explicit("hermes".to_string());
        changed.tool_strict_level = Some(vllm_chat::ToolStrictLevel::Parameter);

        assert_eq!(
            baseline,
            for_render_server(
                &changed,
                Some(RendererSelection::Hf),
                Some(TEST_TOKENIZER_LIBRARY),
                false,
                &files,
            )
            .await
            .unwrap()
        );
    }

    #[tokio::test]
    async fn render_fingerprint_changes_with_resolved_rendering_inputs() {
        let (dir, files) = model_files().await;
        let baseline = for_render_server(
            &render_config("org/model"),
            Some(RendererSelection::Hf),
            Some(TEST_TOKENIZER_LIBRARY),
            false,
            &files,
        )
        .await
        .unwrap();

        let mut changed_reasoning = render_config("org/model");
        changed_reasoning.reasoning_parser = ParserSelection::None;
        assert_ne!(
            baseline,
            for_render_server(
                &changed_reasoning,
                Some(RendererSelection::Hf),
                Some(TEST_TOKENIZER_LIBRARY),
                false,
                &files,
            )
            .await
            .unwrap()
        );

        let mut changed_tokenizer_mode = render_config("org/model");
        changed_tokenizer_mode.renderer = RendererSelection::Auto;
        let changed = for_render_server(
            &changed_tokenizer_mode,
            Some(RendererSelection::Hf),
            Some(TEST_TOKENIZER_LIBRARY),
            false,
            &files,
        )
        .await
        .unwrap();
        assert_ne!(baseline, changed);

        let mut changed_mm_kwargs = render_config("org/model");
        changed_mm_kwargs
            .mm_processor_kwargs
            .insert("image".to_string(), serde_json::json!({"size": 256}));
        assert_ne!(
            baseline,
            for_render_server(
                &changed_mm_kwargs,
                Some(RendererSelection::Hf),
                Some(TEST_TOKENIZER_LIBRARY),
                false,
                &files,
            )
            .await
            .unwrap()
        );

        assert_ne!(
            baseline,
            for_render_server(
                &render_config("org/model"),
                Some(RendererSelection::Harmony),
                Some(TEST_TOKENIZER_LIBRARY),
                false,
                &files,
            )
            .await
            .unwrap()
        );

        std::fs::write(dir.path().join("config.json"), r#"{"changed":true}"#).unwrap();
        assert_ne!(
            baseline,
            for_render_server(
                &render_config("org/model"),
                Some(RendererSelection::Hf),
                Some(TEST_TOKENIZER_LIBRARY),
                false,
                &files,
            )
            .await
            .unwrap()
        );
    }

    #[tokio::test]
    async fn render_fingerprint_preserves_configured_template_file_text() {
        let (_dir, files) = model_files().await;
        let template = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(template.path(), "{{ messages }}\r\n").unwrap();

        let mut from_file = render_config("org/model");
        from_file.chat_template = Some(template.path().to_str().unwrap().to_string());
        let mut inline = render_config("org/model");
        inline.chat_template = Some("{{ messages }}\r\n".to_string());

        assert_eq!(
            for_render_server(
                &from_file,
                Some(RendererSelection::Hf),
                Some(TEST_TOKENIZER_LIBRARY),
                false,
                &files,
            )
            .await
            .unwrap(),
            for_render_server(
                &inline,
                Some(RendererSelection::Hf),
                Some(TEST_TOKENIZER_LIBRARY),
                false,
                &files,
            )
            .await
            .unwrap()
        );
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
