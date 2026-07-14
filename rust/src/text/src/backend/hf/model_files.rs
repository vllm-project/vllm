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

    async fn cached_file(&self, filename: &str) -> Result<Option<PathBuf>> {
        let result =
            self.repo.download_file().filename(filename).local_files_only(true).send().await;
        match result {
            Ok(path) => Ok(Some(path)),
            Err(HFError::LocalEntryNotFound { .. } | HFError::EntryNotFound { .. }) => Ok(None),
            Err(source) => Err(Error::HuggingFaceHubFile {
                operation: "resolve cached",
                filename: filename.to_owned(),
                model_id: self.model_id.to_owned(),
                source: Box::new(source),
            }),
        }
    }

    async fn download_file(&self, filename: &str) -> Result<PathBuf> {
        self.repo.download_file().filename(filename).send().await.map_err(|source| {
            Error::HuggingFaceHubFile {
                operation: "download",
                filename: filename.to_owned(),
                model_id: self.model_id.to_owned(),
                source: Box::new(source),
            }
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
            return resolve_local_model_files(Path::new(model_id));
        }
        let model = HubModel::from_env(model_id)?;
        if let Some(files) = resolve_cached_model_files(&model).await? {
            return Ok(files);
        }
        resolve_remote_model_files(&model).await
    }
}

fn resolve_local_model_files(model_dir: &Path) -> Result<ResolvedModelFiles> {
    let tokenizer_config_path = local_file_if_exists(model_dir, "tokenizer_config.json");
    let tokenizer_config = load_tokenizer_config(tokenizer_config_path.as_deref())?;
    let tokenizer = resolve_local_tokenizer_source(model_dir, &tokenizer_config)?;

    Ok(ResolvedModelFiles {
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
    })
}

async fn resolve_remote_model_files(model: &HubModel<'_>) -> Result<ResolvedModelFiles> {
    let info = model.repo.info().send().await.map_err(|source| Error::HuggingFaceHubModel {
        model_id: model.model_id.to_owned(),
        source: Box::new(source),
    })?;

    let siblings = info
        .siblings
        .iter()
        .flatten()
        .map(|sibling| sibling.rfilename.as_str())
        .collect::<std::collections::BTreeSet<_>>();

    let tokenizer_config_path =
        download_if_present(model, &siblings, "tokenizer_config.json").await?;
    let tokenizer_config = load_tokenizer_config(tokenizer_config_path.as_deref())?;

    let tokenizer = resolve_remote_tokenizer_source(
        model,
        &siblings,
        tokenizer_config.tokenizer_class.as_deref(),
    )
    .await?;

    let generation_config_path =
        download_if_present(model, &siblings, "generation_config.json").await?;
    let preprocessor_config_path =
        download_if_present(model, &siblings, "preprocessor_config.json").await?;
    let video_preprocessor_config_path =
        download_if_present(model, &siblings, "video_preprocessor_config.json").await?;
    let processor_config_path =
        download_if_present(model, &siblings, "processor_config.json").await?;
    let chat_template_name = siblings
        .contains("chat_template.json")
        .then_some("chat_template.json")
        .or_else(|| siblings.contains("chat_template.jinja").then_some("chat_template.jinja"))
        .or_else(|| siblings.iter().copied().find(|name| name.ends_with(".jinja")));
    let chat_template_path = match chat_template_name {
        Some(name) => Some(model.download_file(name).await?),
        None => None,
    };
    let config_path = download_if_present(model, &siblings, "config.json").await?;

    Ok(ResolvedModelFiles {
        tokenizer,
        tokenizer_config_path,
        generation_config_path,
        preprocessor_config_path,
        video_preprocessor_config_path,
        processor_config_path,
        chat_template_path,
        config_path,
    })
}

async fn resolve_cached_model_files(model: &HubModel<'_>) -> Result<Option<ResolvedModelFiles>> {
    let tokenizer_config_path = model.cached_file("tokenizer_config.json").await?;
    let tokenizer_config = load_tokenizer_config(tokenizer_config_path.as_deref())?;
    let config_path = model.cached_file("config.json").await?;
    let tokenizer =
        match resolve_cached_tokenizer_source(model, &tokenizer_config, config_path.as_deref())
            .await?
        {
            Some(tokenizer) => tokenizer,
            None => return Ok(None),
        };

    let model_dir = tokenizer.path().parent().ok_or_else(|| {
        Error::Tokenizer("resolved tokenizer file has no parent directory".to_string())
    })?;
    let generation_config_path = model.cached_file("generation_config.json").await?;
    let preprocessor_config_path = model.cached_file("preprocessor_config.json").await?;
    let video_preprocessor_config_path =
        model.cached_file("video_preprocessor_config.json").await?;
    let processor_config_path = model.cached_file("processor_config.json").await?;
    let chat_template_path = discover_chat_template_in_dir(model_dir);

    Ok(Some(ResolvedModelFiles {
        tokenizer,
        tokenizer_config_path,
        generation_config_path,
        preprocessor_config_path,
        video_preprocessor_config_path,
        processor_config_path,
        chat_template_path,
        config_path,
    }))
}

async fn resolve_remote_tokenizer_source(
    model: &HubModel<'_>,
    siblings: &std::collections::BTreeSet<&str>,
    tokenizer_class: Option<&str>,
) -> Result<TokenizerSource> {
    if let Some(tekken_path) = download_if_present(model, siblings, "tekken.json").await? {
        return Ok(TokenizerSource::Tekken(tekken_path));
    }

    let tokenizer_path = if siblings.contains("tokenizer.json") {
        model.download_file("tokenizer.json").await?
    } else if let Some(tiktoken_name) = find_tiktoken_sibling(siblings) {
        model.download_file(tiktoken_name).await?
    } else {
        return Err(Error::Tokenizer(format!(
            "model '{}' does not expose a supported tokenizer file \
             (tokenizer.json, tiktoken.model, or *.tiktoken) on Hugging Face",
            model.model_id
        )));
    };

    Ok(resolve_tokenizer_source(
        tokenizer_path,
        tokenizer_class,
        None,
    ))
}

async fn resolve_cached_tokenizer_source(
    model: &HubModel<'_>,
    tokenizer_config: &HfTokenizerConfig,
    config_path: Option<&Path>,
) -> Result<Option<TokenizerSource>> {
    if let Some(tekken_path) = model.cached_file("tekken.json").await? {
        return Ok(Some(TokenizerSource::Tekken(tekken_path)));
    }

    let tokenizer_path = match model.cached_file("tokenizer.json").await? {
        Some(path) => Some(path),
        None => match model.cached_file("tiktoken.model").await? {
            Some(path) => Some(path),
            None => config_path.and_then(Path::parent).and_then(discover_tiktoken_in_dir),
        },
    };
    let Some(tokenizer_path) = tokenizer_path else {
        return Ok(None);
    };

    Ok(Some(resolve_tokenizer_source(
        tokenizer_path,
        tokenizer_config.tokenizer_class.as_deref(),
        None,
    )))
}

fn resolve_local_tokenizer_source(
    model_dir: &Path,
    tokenizer_config: &HfTokenizerConfig,
) -> Result<TokenizerSource> {
    let tekken_path = local_file_if_exists(model_dir, "tekken.json");
    if let Some(tekken_path) = tekken_path {
        return Ok(TokenizerSource::Tekken(tekken_path));
    }

    let tokenizer_path = local_file_if_exists(model_dir, "tokenizer.json")
        .or_else(|| local_file_if_exists(model_dir, "tiktoken.model"))
        .or_else(|| discover_tiktoken_in_dir(model_dir))
        .ok_or_else(|| {
            Error::Tokenizer(format!(
                "local model directory '{}' does not contain a supported tokenizer file \
                 (tokenizer.json, tiktoken.model, or *.tiktoken)",
                model_dir.display()
            ))
        })?;

    Ok(resolve_tokenizer_source(
        tokenizer_path,
        tokenizer_config.tokenizer_class.as_deref(),
        None,
    ))
}

/// Choose the tokenizer.
///
/// Selection order:
/// 1. `tekken.json` — Mistral native tokenizer (preferred over HF `tokenizer.json` because the HF
///    version has a known regex bug for Mistral models).
/// 2. File extension — `.tiktoken` / `tiktoken.model` files use tiktoken from BPE data.
/// 3. `tokenizer_class` in `tokenizer_config.json` — classes containing "Tiktoken" (case-
///    insensitive) trigger tiktoken loading from a sibling BPE file.
/// 4. Default — `tokenizer.json` in HuggingFace format.
fn resolve_tokenizer_source(
    tokenizer_path: PathBuf,
    tokenizer_class: Option<&str>,
    tekken_path: Option<PathBuf>,
) -> TokenizerSource {
    if let Some(tekken_path) = tekken_path {
        return TokenizerSource::Tekken(tekken_path);
    }

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

/// Download `filename` only if it exists in `siblings`.
async fn download_if_present(
    model: &HubModel<'_>,
    siblings: &std::collections::BTreeSet<&str>,
    filename: &str,
) -> Result<Option<PathBuf>> {
    match siblings.contains(filename) {
        true => model.download_file(filename).await.map(Some),
        false => Ok(None),
    }
}

fn local_file_if_exists(dir: &Path, filename: &str) -> Option<PathBuf> {
    let path = dir.join(filename);
    path.is_file().then_some(path)
}

/// Find a tiktoken file name among repo siblings, preferring `tiktoken.model`.
fn find_tiktoken_sibling<'a>(siblings: &std::collections::BTreeSet<&'a str>) -> Option<&'a str> {
    if siblings.contains("tiktoken.model") {
        return Some("tiktoken.model");
    }
    siblings.iter().copied().find(|name| name.ends_with(".tiktoken"))
}

/// Discover a tiktoken model file in a local directory.
pub(super) fn discover_tiktoken_in_dir(dir: &std::path::Path) -> Option<PathBuf> {
    let tiktoken_model = dir.join("tiktoken.model");
    if tiktoken_model.exists() {
        return Some(tiktoken_model);
    }
    std::fs::read_dir(dir).ok()?.flatten().find_map(|entry| {
        let path = entry.path();
        if path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.ends_with(".tiktoken"))
        {
            Some(path)
        } else {
            None
        }
    })
}

/// Returns `true` if `path` points to a tiktoken-format file (by name).
pub(super) fn is_tiktoken_file(path: &std::path::Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|name| name == "tiktoken.model" || name.ends_with(".tiktoken"))
}

/// Chat templates are sometimes stored as dedicated .jinja files rather than as
/// a fixed-name config entry, so we scan the cached model dir.
fn discover_chat_template_in_dir(dir: &std::path::Path) -> Option<PathBuf> {
    let json_template_path = dir.join("chat_template.json");
    if json_template_path.exists() {
        return Some(json_template_path);
    }

    let jinja_path = dir.join("chat_template.jinja");
    if jinja_path.exists() {
        return Some(jinja_path);
    }

    std::fs::read_dir(dir).ok()?.flatten().map(|entry| entry.path()).find(|path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".jinja"))
    })
}

#[cfg(test)]
mod tests {
    use std::fs;

    use hf_hub::HFClient;
    use tempfile::tempdir;
    use vllm_tokenizer::{TiktokenTokenizer, Tokenizer};

    use super::{HubModel, ResolvedModelFiles, TokenizerSource, resolve_cached_model_files};

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
