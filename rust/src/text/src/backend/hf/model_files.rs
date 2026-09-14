// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::path::{Path, PathBuf};

use hf_hub::api::tokio::{Api, ApiBuilder, ApiRepo};
use hf_hub::{Cache, Repo, RepoType};
use thiserror_ext::AsReport as _;

use super::config::{HfTokenizerConfig, load_tokenizer_config};
use crate::error::{Error, Result};

const HF_TOKEN_ENV: &str = "HF_TOKEN";

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
    pub async fn new(model_id: &str, revision: Option<&str>) -> Result<Self> {
        if Path::new(model_id).is_dir() {
            return resolve_local_model_files(Path::new(model_id));
        }
        let repo = Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            revision.unwrap_or("main").to_string(),
        );
        // TODO: Let hf-hub read HF_HUB_CACHE after upgrading to 1.0.
        let cache = std::env::var_os("HF_HUB_CACHE")
            .map(|path| Cache::new(path.into()))
            .unwrap_or_else(Cache::from_env);
        if let Some(files) = resolve_cached_model_files(&cache, &repo)? {
            return Ok(files);
        }
        if is_hf_hub_offline() {
            return Err(Error::Tokenizer(format!(
                "model '{model_id}' at revision '{}' has no supported tokenizer in cache '{}' \
                 and HF_HUB_OFFLINE is enabled",
                repo.revision(),
                cache.path().display(),
            )));
        }
        resolve_remote_model_files(model_id, repo, &cache).await
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

async fn resolve_remote_model_files(
    model_id: &str,
    model_repo: Repo,
    cache: &Cache,
) -> Result<ResolvedModelFiles> {
    let api = build_api(cache).map_err(|error| Error::Tokenizer(error.to_report_string()))?;
    let repo = api.repo(model_repo);
    let info = repo.info().await.map_err(|error| {
        Error::Tokenizer(format!(
            "failed to fetch model '{model_id}': {}",
            error.as_report()
        ))
    })?;

    let siblings = info
        .siblings
        .iter()
        .map(|sibling| sibling.rfilename.as_str())
        .collect::<std::collections::BTreeSet<_>>();

    let tokenizer_config_path =
        download_if_present(&repo, model_id, &siblings, "tokenizer_config.json").await?;
    let tokenizer_config = load_tokenizer_config(tokenizer_config_path.as_deref())?;

    let tokenizer = resolve_remote_tokenizer_source(
        &repo,
        model_id,
        &siblings,
        tokenizer_config.tokenizer_class.as_deref(),
    )
    .await?;

    let generation_config_path =
        download_if_present(&repo, model_id, &siblings, "generation_config.json").await?;
    let preprocessor_config_path =
        download_if_present(&repo, model_id, &siblings, "preprocessor_config.json").await?;
    let video_preprocessor_config_path =
        download_if_present(&repo, model_id, &siblings, "video_preprocessor_config.json").await?;
    let processor_config_path =
        download_if_present(&repo, model_id, &siblings, "processor_config.json").await?;
    let chat_template_name = siblings
        .contains("chat_template.json")
        .then_some("chat_template.json")
        .or_else(|| siblings.contains("chat_template.jinja").then_some("chat_template.jinja"))
        .or_else(|| siblings.iter().copied().find(|name| name.ends_with(".jinja")));
    let chat_template_path = match chat_template_name {
        Some(name) => Some(download_known_file(&repo, model_id, name).await?),
        None => None,
    };
    let config_path = download_if_present(&repo, model_id, &siblings, "config.json").await?;

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

fn resolve_cached_model_files(cache: &Cache, repo: &Repo) -> Result<Option<ResolvedModelFiles>> {
    let cache_repo = cache.repo(repo.clone());
    // hf-hub 0.5 looks up refs even for commit SHAs; Python caches have no such ref.
    let snapshot_dir = (repo.revision().len() == 40
        && repo.revision().bytes().all(|byte| byte.is_ascii_hexdigit()))
    .then(|| cache.path().join(repo.folder_name()).join("snapshots").join(repo.revision()));
    let get_file = |name: &str| match &snapshot_dir {
        Some(dir) => local_file_if_exists(dir, name),
        None => cache_repo.get(name),
    };

    let tokenizer_config_path = get_file("tokenizer_config.json");
    let tokenizer_config = load_tokenizer_config(tokenizer_config_path.as_deref())?;
    let tokenizer = match resolve_cached_tokenizer_source(&get_file, &tokenizer_config)? {
        Some(tokenizer) => tokenizer,
        None => return Ok(None),
    };

    let model_dir = tokenizer.path().parent().ok_or_else(|| {
        Error::Tokenizer("resolved tokenizer file has no parent directory".to_string())
    })?;
    let generation_config_path = get_file("generation_config.json");
    let preprocessor_config_path = get_file("preprocessor_config.json");
    let video_preprocessor_config_path = get_file("video_preprocessor_config.json");
    let processor_config_path = get_file("processor_config.json");
    let chat_template_path = discover_chat_template_in_dir(model_dir);
    let config_path = get_file("config.json");

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
    repo: &ApiRepo,
    model_id: &str,
    siblings: &std::collections::BTreeSet<&str>,
    tokenizer_class: Option<&str>,
) -> Result<TokenizerSource> {
    if let Some(tekken_path) = download_if_present(repo, model_id, siblings, "tekken.json").await? {
        return Ok(TokenizerSource::Tekken(tekken_path));
    }

    let tokenizer_path = if siblings.contains("tokenizer.json") {
        download_known_file(repo, model_id, "tokenizer.json").await?
    } else if let Some(tiktoken_name) = find_tiktoken_sibling(siblings) {
        download_known_file(repo, model_id, tiktoken_name).await?
    } else {
        return Err(Error::Tokenizer(format!(
            "model '{model_id}' does not expose a supported tokenizer file \
             (tokenizer.json, tiktoken.model, or *.tiktoken) on Hugging Face"
        )));
    };

    Ok(resolve_tokenizer_source(
        tokenizer_path,
        tokenizer_class,
        None,
    ))
}

fn resolve_cached_tokenizer_source(
    get_file: &impl Fn(&str) -> Option<PathBuf>,
    tokenizer_config: &HfTokenizerConfig,
) -> Result<Option<TokenizerSource>> {
    let tekken_path = get_file("tekken.json");

    if let Some(tekken_path) = tekken_path {
        return Ok(Some(TokenizerSource::Tekken(tekken_path)));
    }

    let Some(tokenizer_path) = get_file("tokenizer.json").or_else(|| {
        // tiktoken.model is the most common name, try it first.
        get_file("tiktoken.model").or_else(|| {
            // Scan for any *.tiktoken file in the cache snapshot directory.
            let snapshot_dir = get_file("config.json")?.parent()?.to_path_buf();
            discover_tiktoken_in_dir(&snapshot_dir)
        })
    }) else {
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
    repo: &ApiRepo,
    model_id: &str,
    siblings: &std::collections::BTreeSet<&str>,
    filename: &str,
) -> Result<Option<PathBuf>> {
    match siblings.contains(filename) {
        true => download_known_file(repo, model_id, filename).await.map(Some),
        false => Ok(None),
    }
}

async fn download_known_file(repo: &ApiRepo, model_id: &str, filename: &str) -> Result<PathBuf> {
    repo.get(filename).await.map_err(|error| {
        Error::Tokenizer(format!(
            "failed to download '{filename}' for model '{model_id}': {}",
            error.as_report()
        ))
    })
}

fn is_hf_hub_offline() -> bool {
    std::env::var("HF_HUB_OFFLINE").is_ok_and(|value| {
        matches!(
            value.to_ascii_uppercase().as_str(),
            "1" | "ON" | "YES" | "TRUE"
        )
    })
}

fn build_api(cache: &Cache) -> anyhow::Result<Api> {
    let mut builder =
        ApiBuilder::from_env().with_cache_dir(cache.path().clone()).with_progress(true);
    if let Ok(token) = std::env::var(HF_TOKEN_ENV)
        && !token.is_empty()
    {
        builder = builder.with_token(Some(token));
    }
    Ok(builder.build()?)
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
    use std::path::PathBuf;

    use tempfile::tempdir;
    use vllm_tokenizer::{TiktokenTokenizer, Tokenizer};

    use super::{ResolvedModelFiles, TokenizerSource};

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

        let files = ResolvedModelFiles::new(dir.path().to_str().expect("utf8 path"), None)
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

    #[test]
    fn cached_model_files_respect_branch_tag_and_commit_revisions() {
        let dir = tempdir().unwrap();
        let cache = hf_hub::Cache::new(dir.path().to_path_buf());
        let model = "test/model";
        let main_sha = "a".repeat(40);
        let pinned_sha = "b".repeat(40);
        for (revision, sha) in [("main", &main_sha), ("release/v1", &pinned_sha)] {
            let repo =
                hf_hub::Repo::with_revision(model.into(), hf_hub::RepoType::Model, revision.into());
            cache.repo(repo.clone()).create_ref(sha).unwrap();
            let snapshot = dir.path().join(repo.folder_name()).join("snapshots").join(sha);
            fs::create_dir_all(&snapshot).unwrap();
            fs::write(snapshot.join("tokenizer.json"), "{}").unwrap();
            fs::write(snapshot.join("config.json"), "{}").unwrap();
        }
        for (revision, expected_sha) in [
            ("main", Some(&main_sha)),
            ("release/v1", Some(&pinned_sha)),
            (pinned_sha.as_str(), Some(&pinned_sha)),
            ("missing", None),
        ] {
            let repo =
                hf_hub::Repo::with_revision(model.into(), hf_hub::RepoType::Model, revision.into());
            let files = super::resolve_cached_model_files(&cache, &repo).unwrap();
            match expected_sha {
                Some(sha) => {
                    let snapshot = dir.path().join(repo.folder_name()).join("snapshots").join(sha);
                    let files = files.unwrap();
                    assert_eq!(files.tokenizer.path(), snapshot.join("tokenizer.json"));
                    assert_eq!(files.config_path, Some(snapshot.join("config.json")));
                }
                None => assert!(files.is_none()),
            }
        }
    }

    #[tokio::test]
    async fn offline_resolution_uses_hub_cache_and_reports_cache_misses() {
        const CHILD: &str = "VLLM_TEST_HF_OFFLINE_CHILD";
        if std::env::var_os(CHILD).is_some() {
            let files = ResolvedModelFiles::new("test/model", Some("release")).await.unwrap();
            let cache = PathBuf::from(std::env::var_os("HF_HUB_CACHE").unwrap());
            assert_eq!(
                files.tokenizer.path(),
                cache.join("models--test--model/snapshots/commit/tokenizer.json")
            );
            let error = ResolvedModelFiles::new("test/model", Some("missing")).await.unwrap_err();
            let message = thiserror_ext::AsReport::as_report(&error).to_string();
            assert!(message.contains("HF_HUB_OFFLINE is enabled"), "{message}");
            assert!(message.contains("revision 'missing'"), "{message}");
            return;
        }

        // Isolate environment variables from other tests and Tokio worker threads.
        let dir = tempdir().unwrap();
        let cache = dir.path().join("hub-cache");
        let repo = cache.join("models--test--model");
        fs::create_dir_all(repo.join("refs")).unwrap();
        fs::create_dir_all(repo.join("snapshots/commit")).unwrap();
        fs::write(repo.join("refs/release"), "commit").unwrap();
        fs::write(repo.join("snapshots/commit/tokenizer.json"), "{}").unwrap();
        for value in ["1", "ON", "yes", "True"] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "backend::hf::model_files::tests::offline_resolution_uses_hub_cache_and_reports_cache_misses", "--nocapture"])
                .env(CHILD, "1")
                .env("HF_HOME", dir.path().join("hf-home"))
                .env("HF_HUB_CACHE", &cache)
                .env("HF_HUB_OFFLINE", value)
                .env("HF_ENDPOINT", "http://127.0.0.1:1")
                .output().unwrap();
            assert!(
                output.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }

    #[tokio::test]
    #[ignore = "too slow for CI and requires network access to Hugging Face"]
    async fn tiktoken_real_kimi_k25_tokenizer_files_load_and_handle_special_tokens() {
        let files = ResolvedModelFiles::new("moonshotai/Kimi-K2.5", None)
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
