use std::path::{Path, PathBuf};

use hf_hub::Cache;
use hf_hub::api::tokio::{Api, ApiBuilder, ApiRepo};
use thiserror_ext::AsReport as _;

use crate::config::{TokenizerConfig, load_tokenizer_config};
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
    /// Select a tokenizer source from a tokenizer file path.
    pub fn from_path(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let file_name = path.file_name().and_then(|name| name.to_str()).ok_or_else(|| {
            Error::new(format!(
                "tokenizer path has no file name: {}",
                path.display()
            ))
        })?;

        match file_name {
            "tekken.json" => Ok(Self::Tekken(path)),
            "tokenizer.json" => Ok(Self::HuggingFace(path)),
            _ if is_tiktoken_file(&path) => Ok(Self::Tiktoken(path)),
            _ => Err(Error::new(format!(
                "unsupported tokenizer file '{}'",
                path.display()
            ))),
        }
    }

    /// Return the local filesystem path for this tokenizer source.
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
    /// Path to `tokenizer_config.json` when present.
    pub tokenizer_config_path: Option<PathBuf>,
    /// Path to `generation_config.json` when present.
    pub generation_config_path: Option<PathBuf>,
    /// Path to `preprocessor_config.json` when present.
    pub preprocessor_config_path: Option<PathBuf>,
    /// Path to a discovered chat template file when present.
    pub chat_template_path: Option<PathBuf>,
    /// Path to `config.json` when present.
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
        if let Some(files) = resolve_cached_model_files(model_id)? {
            return Ok(files);
        }
        resolve_remote_model_files(model_id).await
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
        chat_template_path: discover_chat_template_in_dir(model_dir),
        config_path: local_file_if_exists(model_dir, "config.json"),
    })
}

async fn resolve_remote_model_files(model_id: &str) -> Result<ResolvedModelFiles> {
    let api = build_api().map_err(|error| Error::new(format!("{}", error.as_report())))?;
    let repo = api.model(model_id.to_string());
    let info = repo.info().await.map_err(|error| {
        Error::new(format!(
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

    let model_dir = tokenizer
        .path()
        .parent()
        .ok_or_else(|| Error::new("resolved tokenizer file has no parent directory"))?;
    let generation_config_path = cache_repo.get("generation_config.json");
    let preprocessor_config_path = cache_repo.get("preprocessor_config.json");
    let chat_template_path = discover_chat_template_in_dir(model_dir);
    let config_path = cache_repo.get("config.json");

    Ok(Some(ResolvedModelFiles {
        tokenizer,
        tokenizer_config_path,
        generation_config_path,
        preprocessor_config_path,
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
    let tokenizer_path = if siblings.contains("tekken.json") {
        download_known_file(repo, model_id, "tekken.json").await?
    } else if siblings.contains("tokenizer.json") {
        download_known_file(repo, model_id, "tokenizer.json").await?
    } else if let Some(tiktoken_name) = find_tiktoken_sibling(siblings) {
        download_known_file(repo, model_id, tiktoken_name).await?
    } else {
        return Err(Error::new(format!(
            "model '{model_id}' does not expose a supported tokenizer file \
             (tokenizer.json, tiktoken.model, or *.tiktoken) on Hugging Face"
        )));
    };

    resolve_tokenizer_source(tokenizer_path, tokenizer_class)
}

fn resolve_cached_tokenizer_source(
    cache_repo: &hf_hub::CacheRepo,
    tokenizer_config: &TokenizerConfig,
) -> Result<Option<TokenizerSource>> {
    let Some(tokenizer_path) = cache_repo
        .get("tekken.json")
        .or_else(|| cache_repo.get("tokenizer.json"))
        .or_else(|| {
            // tiktoken.model is the most common name, try it first.
            cache_repo.get("tiktoken.model").or_else(|| {
                // Scan for any *.tiktoken file in the cache snapshot directory.
                let snapshot_dir = cache_repo.get("config.json")?.parent()?.to_path_buf();
                discover_tiktoken_in_dir(&snapshot_dir)
            })
        })
    else {
        return Ok(None);
    };

    Ok(Some(resolve_tokenizer_source(
        tokenizer_path,
        tokenizer_config.tokenizer_class.as_deref(),
    )?))
}

fn resolve_local_tokenizer_source(
    model_dir: &Path,
    tokenizer_config: &TokenizerConfig,
) -> Result<TokenizerSource> {
    let tokenizer_path = local_file_if_exists(model_dir, "tekken.json")
        .or_else(|| local_file_if_exists(model_dir, "tokenizer.json"))
        .or_else(|| local_file_if_exists(model_dir, "tiktoken.model"))
        .or_else(|| discover_tiktoken_in_dir(model_dir))
        .ok_or_else(|| {
            Error::new(format!(
                "local model directory '{}' does not contain a supported tokenizer file \
                 (tokenizer.json, tiktoken.model, or *.tiktoken)",
                model_dir.display()
            ))
        })?;

    resolve_tokenizer_source(tokenizer_path, tokenizer_config.tokenizer_class.as_deref())
}

/// Choose the tokenizer.
///
/// Selection order:
/// 1. File extension — `.tiktoken` / `tiktoken.model` files use tiktoken from BPE data.
/// 2. `tokenizer_class` in `tokenizer_config.json` — classes containing "Tiktoken" (case-
///    insensitive) trigger tiktoken loading from a sibling BPE file.
/// 3. Default — `tokenizer.json` in HuggingFace format.
fn resolve_tokenizer_source(
    tokenizer_path: PathBuf,
    tokenizer_class: Option<&str>,
) -> Result<TokenizerSource> {
    let tokenizer = TokenizerSource::from_path(tokenizer_path)?;

    if let TokenizerSource::HuggingFace(path) = &tokenizer
        && tokenizer_class.is_some_and(|cls| cls.to_ascii_lowercase().contains("tiktoken"))
        && let Some(dir) = path.parent()
        && let Some(tiktoken_path) = discover_tiktoken_in_dir(dir)
    {
        return Ok(TokenizerSource::Tiktoken(tiktoken_path));
    }

    Ok(tokenizer)
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
        Error::new(format!(
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
fn discover_tiktoken_in_dir(dir: &std::path::Path) -> Option<PathBuf> {
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
fn is_tiktoken_file(path: &std::path::Path) -> bool {
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

    use tempfile::tempdir;

    use super::{ResolvedModelFiles, TokenizerSource};

    #[tokio::test]
    async fn resolved_model_files_prefers_absolute_local_model_dir() {
        let dir = tempdir().expect("create temp dir");
        fs::write(dir.path().join("tokenizer.json"), "{}").expect("write tokenizer");
        fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"tokenizer_class":"PreTrainedTokenizerFast"}"#,
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
}
