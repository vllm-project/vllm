// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::path::{Path, PathBuf};

use hf_hub::{HFClient, HFError, HFRepository, RepoTypeModel, split_id};

use super::config::{HfTokenizerConfig, load_tokenizer_config};
use crate::error::{Error, Result};

/// The tokenizer source selected for a model.
#[derive(Debug, Clone)]
pub enum TokenizerSource {
    /// Path to `tokenizer.json` in HuggingFace format.
    HuggingFace(PathBuf),
    /// Path to `tiktoken.model` or `*.tiktoken` file for tiktoken-based models.
    Tiktoken(PathBuf),
    /// Path to `tekken.json` when present (Mistral native tokenizer format).
    ///
    /// When set, the Tekken tokenizer should be preferred over the Hugging Face
    /// tokenizer because the HuggingFace `tokenizer.json` for Mistral
    /// models has a known regex bug that produces incorrect token IDs for
    /// some inputs.
    Tekken(PathBuf),
}

impl TokenizerSource {
    pub fn path(&self) -> &Path {
        match self {
            Self::HuggingFace(path) | Self::Tiktoken(path) | Self::Tekken(path) => path,
        }
    }
}

/// A typed Hugging Face model repository with shared client configuration.
struct HubModel<'a> {
    model_id: &'a str,
    repo: HFRepository<RepoTypeModel>,
}

impl<'a> HubModel<'a> {
    fn from_env(model_id: &'a str) -> Result<Self> {
        let client = HFClient::new().map_err(|source| Error::HuggingFaceHubClient {
            source: Box::new(source),
        })?;
        Ok(Self::new(model_id, client))
    }

    fn new(model_id: &'a str, client: HFClient) -> Self {
        let (owner, name) = split_id(model_id);
        Self {
            model_id,
            repo: client.model(owner, name),
        }
    }

    async fn cached_snapshot(&self) -> Result<Option<PathBuf>> {
        let result = self.repo.snapshot_download().local_files_only(true).send().await;
        match result {
            Ok(path) => Ok(Some(path)),
            Err(HFError::LocalEntryNotFound { .. } | HFError::EntryNotFound { .. }) => Ok(None),
            Err(source) => Err(Error::HuggingFaceHubSnapshot {
                operation: "resolve cached",
                model_id: self.model_id.to_owned(),
                source: Box::new(source),
            }),
        }
    }

    async fn download_file(&self, filename: &str, revision: &str) -> Result<PathBuf> {
        self.repo
            .download_file()
            .filename(filename)
            .revision(revision)
            .send()
            .await
            .map_err(|source| Error::HuggingFaceHubFile {
                operation: "download",
                filename: filename.to_owned(),
                model_id: self.model_id.to_owned(),
                source: Box::new(source),
            })
    }
}

/// Concrete tokenizer/config file locations resolved for one HF model id.
#[derive(Debug, Clone)]
pub struct ResolvedModelFiles {
    /// The selected tokenizer source for this model.
    pub tokenizer: TokenizerSource,
    pub tokenizer_config_path: Option<PathBuf>,
    pub generation_config_path: Option<PathBuf>,
    pub preprocessor_config_path: Option<PathBuf>,
    /// Video-specific preprocessor config, when provided by the model repo.
    pub video_preprocessor_config_path: Option<PathBuf>,
    /// Combined processor config, which may embed a `video_processor` section.
    pub processor_config_path: Option<PathBuf>,
    pub chat_template_path: Option<PathBuf>,
    pub config_path: Option<PathBuf>,
}

impl ResolvedModelFiles {
    /// Resolve tokenizer/config files from a local model directory first when
    /// `model_id` points to one, otherwise consult the local HF cache and
    /// finally the Hub.
    pub async fn new(model_id: &str) -> Result<Self> {
        if Path::new(model_id).is_dir() {
            return resolve_model_dir(Path::new(model_id));
        }
        let model = HubModel::from_env(model_id)?;
        if let Some(files) = resolve_cached_model_files(&model).await? {
            return Ok(files);
        }
        resolve_remote_model_files(&model).await
    }
}

fn resolve_model_dir(model_dir: &Path) -> Result<ResolvedModelFiles> {
    try_resolve_model_dir(model_dir)?.ok_or_else(|| {
        Error::Tokenizer(format!(
            "model directory '{}' does not contain a supported tokenizer file \
             (tokenizer.json, tiktoken.model, or *.tiktoken)",
            model_dir.display()
        ))
    })
}

fn try_resolve_model_dir(model_dir: &Path) -> Result<Option<ResolvedModelFiles>> {
    let tokenizer_config_path = local_file_if_exists(model_dir, "tokenizer_config.json");
    let tokenizer_config = load_tokenizer_config(tokenizer_config_path.as_deref())?;
    let Some(tokenizer) = resolve_tokenizer_source_in_dir(model_dir, &tokenizer_config) else {
        return Ok(None);
    };

    Ok(Some(ResolvedModelFiles {
        tokenizer,
        tokenizer_config_path,
        generation_config_path: local_file_if_exists(model_dir, "generation_config.json"),
        preprocessor_config_path: local_file_if_exists(model_dir, "preprocessor_config.json"),
        video_preprocessor_config_path: local_file_if_exists(
            model_dir,
            "video_preprocessor_config.json",
        ),
        processor_config_path: local_file_if_exists(model_dir, "processor_config.json"),
        chat_template_path: discover_chat_template_in_dir(model_dir),
        config_path: local_file_if_exists(model_dir, "config.json"),
    }))
}

async fn resolve_remote_model_files(model: &HubModel<'_>) -> Result<ResolvedModelFiles> {
    let info = model.repo.info().send().await.map_err(|source| Error::HuggingFaceHubModel {
        model_id: model.model_id.to_owned(),
        source: Box::new(source),
    })?;

    let revision = info.sha.as_deref().ok_or_else(|| Error::HuggingFaceHubModelRevision {
        model_id: model.model_id.to_owned(),
    })?;
    let mut filenames = collect_remote_model_files(
        info.siblings.iter().flatten().map(|sibling| sibling.rfilename.as_str()),
    );
    let root_filename = filenames
        .iter()
        .copied()
        .find(|filename| is_tokenizer_artifact(filename))
        .ok_or_else(|| {
            Error::Tokenizer(format!(
                "model '{}' does not expose a supported tokenizer file \
             (tokenizer.json, tiktoken.model, or *.tiktoken) on Hugging Face",
                model.model_id
            ))
        })?;

    let root_path = model.download_file(root_filename, revision).await?;
    let model_dir = model_dir_from_download(&root_path, root_filename)?;
    filenames.remove(root_filename);
    for filename in filenames {
        model.download_file(filename, revision).await?;
    }

    resolve_model_dir(&model_dir)
}

async fn resolve_cached_model_files(model: &HubModel<'_>) -> Result<Option<ResolvedModelFiles>> {
    let Some(model_dir) = model.cached_snapshot().await? else {
        return Ok(None);
    };
    try_resolve_model_dir(&model_dir)
}

const REMOTE_METADATA_FILES: &[&str] = &[
    "tokenizer_config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "processor_config.json",
    "chat_template.json",
    "config.json",
];

fn collect_remote_model_files<'a>(
    siblings: impl IntoIterator<Item = &'a str>,
) -> std::collections::BTreeSet<&'a str> {
    siblings
        .into_iter()
        .filter(|filename| {
            REMOTE_METADATA_FILES.contains(filename)
                || is_tokenizer_artifact(filename)
                || filename.ends_with(".jinja")
        })
        .collect()
}

fn is_tokenizer_artifact(filename: &str) -> bool {
    matches!(
        filename,
        "tekken.json" | "tokenizer.json" | "tiktoken.model"
    ) || filename.ends_with(".tiktoken")
}

/// Choose the tokenizer from a materialized model directory.
///
/// Selection order:
/// 1. `tekken.json` — Mistral native tokenizer (preferred over HF `tokenizer.json` because the HF
///    version has a known regex bug for Mistral models).
/// 2. `tokenizer.json` — classes containing "Tiktoken" in `tokenizer_config.json` redirect to a
///    sibling BPE file; other classes use the Hugging Face tokenizer.
/// 3. `tiktoken.model` or `*.tiktoken` — use tiktoken directly from BPE data.
fn resolve_tokenizer_source_in_dir(
    model_dir: &Path,
    tokenizer_config: &HfTokenizerConfig,
) -> Option<TokenizerSource> {
    if let Some(tekken_path) = local_file_if_exists(model_dir, "tekken.json") {
        return Some(TokenizerSource::Tekken(tekken_path));
    }

    let tokenizer_path = local_file_if_exists(model_dir, "tokenizer.json")
        .or_else(|| local_file_if_exists(model_dir, "tiktoken.model"))
        .or_else(|| discover_tiktoken_in_dir(model_dir))?;

    Some(resolve_tokenizer_source(
        tokenizer_path,
        tokenizer_config.tokenizer_class.as_deref(),
    ))
}

fn resolve_tokenizer_source(
    tokenizer_path: PathBuf,
    tokenizer_class: Option<&str>,
) -> TokenizerSource {
    if is_tiktoken_file(&tokenizer_path) {
        return TokenizerSource::Tiktoken(tokenizer_path);
    }

    if tokenizer_class.is_some_and(|cls| cls.to_ascii_lowercase().contains("tiktoken"))
        && let Some(dir) = tokenizer_path.parent()
        && let Some(tiktoken_path) = discover_tiktoken_in_dir(dir)
    {
        return TokenizerSource::Tiktoken(tiktoken_path);
    }

    TokenizerSource::HuggingFace(tokenizer_path)
}

fn model_dir_from_download(path: &Path, filename: &str) -> Result<PathBuf> {
    path.ancestors()
        .nth(Path::new(filename).components().count())
        .map(Path::to_path_buf)
        .ok_or_else(|| {
            Error::Tokenizer(format!(
                "downloaded model file '{}' does not contain repository path '{filename}'",
                path.display()
            ))
        })
}

fn local_file_if_exists(dir: &Path, filename: &str) -> Option<PathBuf> {
    let path = dir.join(filename);
    path.is_file().then_some(path)
}

/// Discover a tiktoken model file in a local directory.
pub(super) fn discover_tiktoken_in_dir(dir: &std::path::Path) -> Option<PathBuf> {
    let tiktoken_model = dir.join("tiktoken.model");
    if tiktoken_model.exists() {
        return Some(tiktoken_model);
    }
    discover_file_recursively(dir, |path| {
        path.file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.ends_with(".tiktoken"))
    })
}

/// Returns `true` if `path` points to a tiktoken-format file (by name).
pub(super) fn is_tiktoken_file(path: &std::path::Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|name| name == "tiktoken.model" || name.ends_with(".tiktoken"))
}

/// Chat templates are sometimes stored as dedicated .jinja files rather than as
/// a fixed-name config entry, so we scan the model dir.
fn discover_chat_template_in_dir(dir: &std::path::Path) -> Option<PathBuf> {
    let json_template_path = dir.join("chat_template.json");
    if json_template_path.exists() {
        return Some(json_template_path);
    }

    let jinja_path = dir.join("chat_template.jinja");
    if jinja_path.exists() {
        return Some(jinja_path);
    }

    discover_file_recursively(dir, |path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".jinja"))
    })
}

fn discover_file_recursively(dir: &Path, matches: impl Fn(&Path) -> bool) -> Option<PathBuf> {
    let mut pending = vec![dir.to_path_buf()];
    let mut selected: Option<PathBuf> = None;

    while let Some(dir) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if entry.file_type().is_ok_and(|file_type| file_type.is_dir()) {
                pending.push(path);
            } else if matches(&path) && selected.as_ref().is_none_or(|current| path < *current) {
                selected = Some(path);
            }
        }
    }

    selected
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use hf_hub::HFClient;
    use tempfile::tempdir;
    use vllm_tokenizer::{TiktokenTokenizer, Tokenizer};

    use super::{
        HubModel, ResolvedModelFiles, TokenizerSource, collect_remote_model_files,
        model_dir_from_download, resolve_cached_model_files,
    };

    #[tokio::test]
    async fn resolved_model_files_prefers_absolute_local_model_dir() {
        let dir = tempdir().expect("create temp dir");
        fs::write(dir.path().join("tokenizer.json"), "{}").expect("write tokenizer");
        fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"tokenizer_class":"TokenizersBackend"}"#,
        )
        .expect("write tokenizer config");
        fs::write(dir.path().join("config.json"), "{}").expect("write config");

        let files = ResolvedModelFiles::new(dir.path().to_str().expect("utf8 path"))
            .await
            .expect("resolve local model files");

        match files.tokenizer {
            TokenizerSource::HuggingFace(path) => {
                assert_eq!(path, dir.path().join("tokenizer.json"));
            }
            other => panic!("expected HuggingFace tokenizer, got {other:?}"),
        }
        assert_eq!(files.config_path, Some(dir.path().join("config.json")));
        assert_eq!(
            files.tokenizer_config_path,
            Some(dir.path().join("tokenizer_config.json"))
        );
    }

    #[tokio::test]
    async fn resolved_model_files_uses_hf_cache_without_network() {
        let cache = tempdir().expect("create cache dir");
        let repo_dir = cache.path().join("models--test--cached");
        let commit = "0123456789abcdef0123456789abcdef01234567";
        let snapshot_dir = repo_dir.join("snapshots").join(commit);
        fs::create_dir_all(&snapshot_dir).expect("create snapshot dir");
        fs::create_dir_all(repo_dir.join("refs")).expect("create refs dir");
        fs::write(repo_dir.join("refs/main"), commit).expect("write main ref");
        fs::write(
            snapshot_dir.join("tokenizer_config.json"),
            r#"{"tokenizer_class":"TiktokenTokenizer"}"#,
        )
        .expect("write tokenizer config");
        fs::write(snapshot_dir.join("tiktoken.model"), "fixture").expect("write tokenizer");
        fs::write(snapshot_dir.join("config.json"), "{}").expect("write config");
        fs::write(snapshot_dir.join("generation_config.json"), "{}")
            .expect("write generation config");
        fs::write(snapshot_dir.join("chat_template.jinja"), "{{ messages }}")
            .expect("write chat template");

        let client = HFClient::builder()
            .endpoint("http://127.0.0.1:1")
            .cache_dir(cache.path())
            .build()
            .expect("build hf-hub client");
        let model = HubModel::new("test/cached", client);
        let files = resolve_cached_model_files(&model)
            .await
            .expect("resolve cached model files")
            .expect("cached tokenizer is complete");

        match files.tokenizer {
            TokenizerSource::Tiktoken(path) => {
                assert_eq!(path, snapshot_dir.join("tiktoken.model"));
            }
            other => panic!("expected tiktoken tokenizer, got {other:?}"),
        }
        assert_eq!(files.config_path, Some(snapshot_dir.join("config.json")));
        assert_eq!(
            files.generation_config_path,
            Some(snapshot_dir.join("generation_config.json"))
        );
        assert_eq!(
            files.chat_template_path,
            Some(snapshot_dir.join("chat_template.jinja"))
        );
    }

    #[tokio::test]
    async fn cached_snapshot_without_tokenizer_is_a_cache_miss() {
        let cache = tempdir().expect("create cache dir");
        let repo_dir = cache.path().join("models--test--partial");
        let commit = "0123456789abcdef0123456789abcdef01234567";
        let snapshot_dir = repo_dir.join("snapshots").join(commit);
        fs::create_dir_all(&snapshot_dir).expect("create snapshot dir");
        fs::create_dir_all(repo_dir.join("refs")).expect("create refs dir");
        fs::write(repo_dir.join("refs/main"), commit).expect("write main ref");
        fs::write(snapshot_dir.join("config.json"), "{}").expect("write config");

        let client = HFClient::builder()
            .endpoint("http://127.0.0.1:1")
            .cache_dir(cache.path())
            .build()
            .expect("build hf-hub client");
        let model = HubModel::new("test/partial", client);

        assert!(
            resolve_cached_model_files(&model)
                .await
                .expect("inspect cached snapshot")
                .is_none()
        );
    }

    #[test]
    fn downloaded_nested_repo_file_resolves_snapshot_root() {
        let path =
            Path::new("/cache/models--test--nested/snapshots/commit/tokenizers/model.tiktoken");
        assert_eq!(
            model_dir_from_download(path, "tokenizers/model.tiktoken")
                .expect("resolve snapshot root"),
            Path::new("/cache/models--test--nested/snapshots/commit")
        );
    }

    #[test]
    fn directory_resolver_finds_nested_tiktoken_file() {
        let dir = tempdir().expect("create temp dir");
        let tokenizer_dir = dir.path().join("tokenizers");
        fs::create_dir_all(&tokenizer_dir).expect("create tokenizer dir");
        fs::write(tokenizer_dir.join("model.tiktoken"), "fixture").expect("write tokenizer");

        let files = super::resolve_model_dir(dir.path()).expect("resolve model directory");
        match files.tokenizer {
            TokenizerSource::Tiktoken(path) => {
                assert_eq!(path, tokenizer_dir.join("model.tiktoken"));
            }
            other => panic!("expected tiktoken tokenizer, got {other:?}"),
        }
    }

    #[test]
    fn directory_resolver_owns_tokenizer_and_chat_template_priority() {
        let dir = tempdir().expect("create temp dir");
        for filename in [
            "tekken.json",
            "tokenizer.json",
            "tiktoken.model",
            "chat_template.json",
            "chat_template.jinja",
        ] {
            fs::write(dir.path().join(filename), "fixture").expect("write model file");
        }

        let files = super::resolve_model_dir(dir.path()).expect("resolve model directory");
        match files.tokenizer {
            TokenizerSource::Tekken(path) => {
                assert_eq!(path, dir.path().join("tekken.json"));
            }
            other => panic!("expected Tekken tokenizer, got {other:?}"),
        }
        assert_eq!(
            files.chat_template_path,
            Some(dir.path().join("chat_template.json"))
        );
    }

    #[test]
    fn remote_model_file_collection_includes_all_candidates() {
        let siblings = std::collections::BTreeSet::from([
            "README.md",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "tiktoken.model",
            "tekken.json",
            "tokenizers/model.tiktoken",
            "chat_template.json",
            "chat_template.jinja",
            "templates/tool_use.jinja",
            "model.safetensors",
        ]);
        assert_eq!(
            collect_remote_model_files(siblings),
            std::collections::BTreeSet::from([
                "chat_template.jinja",
                "chat_template.json",
                "config.json",
                "tekken.json",
                "templates/tool_use.jinja",
                "tiktoken.model",
                "tokenizer.json",
                "tokenizer_config.json",
                "tokenizers/model.tiktoken",
            ])
        );
    }

    #[tokio::test]
    #[ignore = "too slow for CI and requires network access to Hugging Face"]
    async fn tiktoken_real_kimi_k25_tokenizer_files_load_and_handle_special_tokens() {
        let files = ResolvedModelFiles::new("moonshotai/Kimi-K2.5")
            .await
            .expect("resolve real Kimi K2.5 model files");

        let tokenizer_path = match &files.tokenizer {
            TokenizerSource::Tiktoken(path) => path.clone(),
            other => panic!("expected tiktoken tokenizer source, got {other:?}"),
        };

        for backend in [
            TiktokenTokenizer::new_riptoken(&tokenizer_path).expect("load riptoken backend"),
            TiktokenTokenizer::new_tiktoken_rs(&tokenizer_path).expect("load tiktoken-rs backend"),
        ] {
            let think_id = backend.token_to_id("<think>").expect("resolve <think>");
            let end_think_id = backend.token_to_id("</think>").expect("resolve </think>");
            let tool_section_id = backend
                .token_to_id("<|tool_calls_section_begin|>")
                .expect("resolve tool call section marker");
            let contraction_heavy_text =
                "I'm sure it's fine, but I can't say I'd trust that it's what we'd ship.";
            let contraction_heavy_ids = backend.encode(contraction_heavy_text, false).unwrap();

            assert_eq!(
                (think_id, end_think_id, tool_section_id),
                (163606, 163607, 163595)
            );
            assert_eq!(backend.decode(&[think_id], true).unwrap(), "<think>");
            assert_eq!(backend.decode(&[end_think_id], true).unwrap(), "</think>");
            assert_eq!(
                backend.decode(&[tool_section_id], true).unwrap(),
                "<|tool_calls_section_begin|>"
            );

            // This demonstrates that we're using Kimi's custom BPE pattern.
            // With CL100K this will be 23 tokens instead.
            assert_eq!(
                contraction_heavy_ids,
                vec![
                    17172, 3287, 4643, 8201, 11, 996, 374, 8971, 3637, 20020, 8173, 473, 4643,
                    1573, 56229, 13922, 13,
                ]
            );
            assert_eq!(contraction_heavy_ids.len(), 17);
            assert_eq!(
                backend.decode(&contraction_heavy_ids, false).unwrap(),
                contraction_heavy_text
            );

            // Special-looking text that is not actually registered should fail gracefully.
            assert_eq!(backend.token_to_id("◁think▷"), None);
            assert_eq!(backend.token_to_id("<|definitely_not_registered|>"), None);
        }
    }
}
