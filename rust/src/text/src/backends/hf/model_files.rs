use std::path::PathBuf;

use hf_hub::Cache;
use hf_hub::api::tokio::{Api, ApiBuilder, ApiRepo};
use thiserror_ext::AsReport as _;

use crate::error::{Error, Result};

const HF_TOKEN_ENV: &str = "HF_TOKEN";

/// Concrete tokenizer/config file locations resolved for one HF model id.
#[derive(Debug, Clone)]
pub struct ResolvedModelFiles {
    pub tokenizer_path: PathBuf,
    pub tokenizer_config_path: Option<PathBuf>,
    pub generation_config_path: Option<PathBuf>,
    pub chat_template_path: Option<PathBuf>,
    pub config_path: Option<PathBuf>,
}

/// Resolve tokenizer/config files from the local HF cache first, then fall back
/// to downloading the known metadata files from the Hub.
pub(super) async fn resolve_model_files(model_id: &str) -> Result<ResolvedModelFiles> {
    if let Some(files) = resolve_cached_model_files(model_id) {
        return Ok(files);
    }

    resolve_remote_model_files(model_id).await
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

    if !siblings.contains("tokenizer.json") {
        return Err(Error::Tokenizer(format!(
            "model '{model_id}' does not expose tokenizer.json on Hugging Face"
        )));
    }
    let tokenizer_path = download_known_file(&repo, model_id, "tokenizer.json").await?;

    let tokenizer_config_path = if siblings.contains("tokenizer_config.json") {
        Some(download_known_file(&repo, model_id, "tokenizer_config.json").await?)
    } else {
        None
    };
    let generation_config_path = if siblings.contains("generation_config.json") {
        Some(download_known_file(&repo, model_id, "generation_config.json").await?)
    } else {
        None
    };
    let chat_template_name = if siblings.contains("chat_template.json") {
        Some("chat_template.json")
    } else if siblings.contains("chat_template.jinja") {
        Some("chat_template.jinja")
    } else {
        siblings
            .iter()
            .copied()
            .find(|name| name.ends_with(".jinja"))
    };
    let chat_template_path = if let Some(chat_template_name) = chat_template_name {
        Some(download_known_file(&repo, model_id, chat_template_name).await?)
    } else {
        None
    };
    let config_path = if siblings.contains("config.json") {
        Some(download_known_file(&repo, model_id, "config.json").await?)
    } else {
        None
    };

    Ok(ResolvedModelFiles {
        tokenizer_path,
        tokenizer_config_path,
        generation_config_path,
        chat_template_path,
        config_path,
    })
}

fn resolve_cached_model_files(model_id: &str) -> Option<ResolvedModelFiles> {
    let cache_repo = Cache::from_env().model(model_id.to_string());
    let tokenizer_path = cache_repo.get("tokenizer.json")?;
    let model_dir = tokenizer_path.parent()?;
    let tokenizer_config_path = cache_repo.get("tokenizer_config.json");
    let generation_config_path = cache_repo.get("generation_config.json");
    let chat_template_path = discover_chat_template_in_dir(model_dir);
    let config_path = cache_repo.get("config.json");

    Some(ResolvedModelFiles {
        tokenizer_path,
        tokenizer_config_path,
        generation_config_path,
        chat_template_path,
        config_path,
    })
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
