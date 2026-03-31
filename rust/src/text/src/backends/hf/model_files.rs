use std::path::{Path, PathBuf};

use hf_hub::Cache;
use hf_hub::api::tokio::{Api, ApiBuilder, ApiRepo};
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
    /// When set, the Tekken tokenizer should be preferred over the Hugging Face tokenizer
    /// because the HuggingFace `tokenizer.json` for Mistral models has a known regex bug that
    /// produces incorrect token IDs for some inputs.
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
    pub chat_template_path: Option<PathBuf>,
    pub config_path: Option<PathBuf>,
}

impl ResolvedModelFiles {
    /// Resolve tokenizer/config files from the local HF cache first, then fall back to downloading
    /// the known metadata files from the Hub.
    pub async fn new(model_id: &str) -> Result<Self> {
        if let Some(files) = resolve_cached_model_files(model_id)? {
            return Ok(files);
        }
        resolve_remote_model_files(model_id).await
    }
}

async fn resolve_remote_model_files(model_id: &str) -> Result<ResolvedModelFiles> {
    let api = build_api().map_err(|error| Error::Tokenizer(error.to_report_string()))?;
    let repo = api.model(model_id.to_string());
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
    let chat_template_name = siblings
        .contains("chat_template.json")
        .then_some("chat_template.json")
        .or_else(|| {
            siblings
                .contains("chat_template.jinja")
                .then_some("chat_template.jinja")
        })
        .or_else(|| {
            siblings
                .iter()
                .copied()
                .find(|name| name.ends_with(".jinja"))
        });
    let chat_template_path = match chat_template_name {
        Some(name) => Some(download_known_file(&repo, model_id, name).await?),
        None => None,
    };
    let config_path = download_if_present(&repo, model_id, &siblings, "config.json").await?;

    Ok(ResolvedModelFiles {
        tokenizer,
        tokenizer_config_path,
        generation_config_path,
        chat_template_path,
        config_path,
    })
}

fn resolve_cached_model_files(model_id: &str) -> Result<Option<ResolvedModelFiles>> {
    let cache_repo = Cache::from_env().model(model_id.to_string());

    let tokenizer_config_path = cache_repo.get("tokenizer_config.json");
    let tokenizer_config = load_tokenizer_config(tokenizer_config_path.as_deref())?;
    let tokenizer = match resolve_cached_tokenizer_source(&cache_repo, &tokenizer_config)? {
        Some(tokenizer) => tokenizer,
        None => return Ok(None),
    };

    let model_dir = tokenizer.path().parent().ok_or_else(|| {
        Error::Tokenizer("resolved tokenizer file has no parent directory".to_string())
    })?;
    let generation_config_path = cache_repo.get("generation_config.json");
    let chat_template_path = discover_chat_template_in_dir(model_dir);
    let config_path = cache_repo.get("config.json");

    Ok(Some(ResolvedModelFiles {
        tokenizer,
        tokenizer_config_path,
        generation_config_path,
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
    cache_repo: &hf_hub::CacheRepo,
    tokenizer_config: &HfTokenizerConfig,
) -> Result<Option<TokenizerSource>> {
    let tekken_path = cache_repo.get("tekken.json");

    if let Some(tekken_path) = tekken_path {
        return Ok(Some(TokenizerSource::Tekken(tekken_path)));
    }

    let Some(tokenizer_path) = cache_repo.get("tokenizer.json").or_else(|| {
        // tiktoken.model is the most common name, try it first.
        cache_repo.get("tiktoken.model").or_else(|| {
            // Scan for any *.tiktoken file in the cache snapshot directory.
            let snapshot_dir = cache_repo.get("config.json")?.parent()?.to_path_buf();
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
        true => download_known_file(repo, model_id, filename)
            .await
            .map(Some),
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

fn build_api() -> anyhow::Result<Api> {
    let mut builder = ApiBuilder::from_env().with_progress(true);
    if let Ok(token) = std::env::var(HF_TOKEN_ENV)
        && !token.is_empty()
    {
        builder = builder.with_token(Some(token));
    }
    Ok(builder.build()?)
}

/// Find a tiktoken file name among repo siblings, preferring `tiktoken.model`.
fn find_tiktoken_sibling<'a>(siblings: &std::collections::BTreeSet<&'a str>) -> Option<&'a str> {
    if siblings.contains("tiktoken.model") {
        return Some("tiktoken.model");
    }
    siblings
        .iter()
        .copied()
        .find(|name| name.ends_with(".tiktoken"))
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

/// Chat templates are sometimes stored as dedicated .jinja files rather than as a fixed-name config
/// entry, so we scan the cached model dir.
fn discover_chat_template_in_dir(dir: &std::path::Path) -> Option<PathBuf> {
    let json_template_path = dir.join("chat_template.json");
    if json_template_path.exists() {
        return Some(json_template_path);
    }

    let jinja_path = dir.join("chat_template.jinja");
    if jinja_path.exists() {
        return Some(jinja_path);
    }

    std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with(".jinja"))
        })
}
