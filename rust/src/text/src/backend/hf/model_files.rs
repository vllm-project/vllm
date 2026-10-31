// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use hf_hub::api::tokio::{Api, ApiBuilder, ApiRepo};
use hf_hub::{Cache, Repo, RepoType};
use serde::Serialize;
use thiserror_ext::AsReport as _;

use super::HfOverrides;
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

/// A local model directory or a Hugging Face repo at a revision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelSource {
    /// Hugging Face repo id or local directory.
    pub id: String,
    /// Revision on the Hugging Face Hub (branch, tag, or commit SHA).
    pub revision: Option<String>,
}

/// Where each group of model files is read from, matching Python's
/// `ModelConfig`: the tokenizer files and the chat template follow
/// `--tokenizer`, `config.json` and `generation_config.json` follow
/// `--hf-config-path`, and the multimodal processor configs follow the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelFileSources {
    pub model: ModelSource,
    /// Source of the tokenizer files and the chat template, or `None` for `model`.
    pub tokenizer: Option<ModelSource>,
    /// Source of `config.json` and `generation_config.json`, or `None` for
    /// `model`. When set, it must provide `config.json`.
    pub hf_config: Option<ModelSource>,
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
    /// Keep a patched config alive across clones and backend ownership.
    config_temp_path: Option<Arc<tempfile::TempPath>>,
}

impl ResolvedModelFiles {
    /// Materialize a merge-patched config owned by these resolved model files.
    pub fn apply_overrides(&mut self, overrides: &HfOverrides) -> Result<()> {
        if overrides.is_empty() {
            return Ok(());
        }
        let path = materialize_config(self.config_path.as_deref(), overrides).map_err(|error| {
            Error::Tokenizer(format!(
                "failed to materialize HF overrides: {}",
                error.as_report()
            ))
        })?;
        self.config_path = Some(path.to_path_buf());
        self.config_temp_path = Some(Arc::new(path));
        Ok(())
    }

    /// Resolve tokenizer/config files from a local model directory first when
    /// `model_id` points to one, otherwise consult the local HF cache and
    /// finally the Hub.
    pub async fn new(model_id: &str, revision: Option<&str>) -> Result<Self> {
        Self::from_sources(&ModelFileSources {
            model: ModelSource {
                id: model_id.to_string(),
                revision: revision.map(str::to_string),
            },
            tokenizer: None,
            hf_config: None,
        })
        .await
    }

    /// Resolve each group of files from its source in `sources`.
    pub async fn from_sources(sources: &ModelFileSources) -> Result<Self> {
        let ModelFileSources {
            model,
            tokenizer,
            hf_config,
        } = sources;
        // A cached model snapshot must contain the files that only the model provides.
        let model_required = match (tokenizer, hf_config) {
            (None, _) => RequiredFile::Tokenizer,
            (Some(_), None) => RequiredFile::Config,
            (Some(_), Some(_)) => RequiredFile::Nothing,
        };
        let model_files = SourceFiles::open(model, model_required).await?;
        let tokenizer_files = match tokenizer {
            Some(source) => Some(SourceFiles::open(source, RequiredFile::Tokenizer).await?),
            None => None,
        };
        let tokenizer_files = tokenizer_files.as_ref().unwrap_or(&model_files);
        let config_files = match hf_config {
            Some(source) => Some(SourceFiles::open(source, RequiredFile::Config).await?),
            None => None,
        };
        let config_files = config_files.as_ref().unwrap_or(&model_files);

        let tokenizer_config_path = tokenizer_files.get("tokenizer_config.json").await?;
        let tokenizer_config = load_tokenizer_config(tokenizer_config_path.as_deref())?;
        let config_path = config_files.get("config.json").await?;
        if config_path.is_none()
            && let Some(source) = hf_config
        {
            return Err(Error::Tokenizer(format!(
                "config source '{}' at revision '{}' does not contain config.json",
                source.id,
                source.revision.as_deref().unwrap_or("main"),
            )));
        }

        Ok(Self {
            tokenizer: tokenizer_files.tokenizer(&tokenizer_config).await?,
            tokenizer_config_path,
            generation_config_path: config_files.get("generation_config.json").await?,
            preprocessor_config_path: model_files.get("preprocessor_config.json").await?,
            video_preprocessor_config_path: model_files
                .get("video_preprocessor_config.json")
                .await?,
            processor_config_path: model_files.get("processor_config.json").await?,
            chat_template_path: tokenizer_files.chat_template().await?,
            config_path,
            config_temp_path: None,
        })
    }
}

fn materialize_config(
    path: Option<&Path>,
    overrides: &HfOverrides,
) -> anyhow::Result<tempfile::TempPath> {
    let mut config: serde_json::Value = match path {
        Some(path) => serde_json::from_slice(&std::fs::read(path)?)?,
        None => serde_json::json!({}),
    };
    anyhow::ensure!(config.is_object(), "HF model config must be a JSON object");
    overrides.apply(&mut config);
    let file = tempfile::Builder::new()
        .prefix("vllm-rs-hf-config-")
        .suffix(".json")
        .tempfile()?;
    serde_json::to_writer(file.as_file(), &config)?;
    Ok(file.into_temp_path())
}

/// Appended to tokenizer-not-found errors for repos that ship only the slow BPE
/// tokenizer files, such as `Qwen/Qwen3-ASR-1.7B`.
const SLOW_BPE_TOKENIZER_HINT: &str = ". Found `vocab.json` and `merges.txt` but no \
     `tokenizer.json`; generate one with transformers \
     (`AutoTokenizer.from_pretrained(<model>).save_pretrained(<dir>)`) and pass `--tokenizer <dir>`";

/// The file a cached snapshot must contain to be used for a source.
#[derive(Debug, Clone, Copy)]
enum RequiredFile {
    Tokenizer,
    Config,
    Nothing,
}

impl std::fmt::Display for RequiredFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Tokenizer => "supported tokenizer",
            Self::Config => "config.json",
            Self::Nothing => "snapshot",
        })
    }
}

/// Model files of one source, read from disk or downloaded from the Hub on demand.
enum SourceFiles {
    /// A local model directory or a cached Hub snapshot.
    Dir(PathBuf),
    /// A Hub repo with its listed files.
    Remote {
        model_id: String,
        repo: ApiRepo,
        siblings: BTreeSet<String>,
    },
}

impl SourceFiles {
    /// Open a local model directory when `source` points to one, otherwise a
    /// cached snapshot that contains `required`, and finally the Hub.
    async fn open(source: &ModelSource, required: RequiredFile) -> Result<Self> {
        let model_id = source.id.as_str();
        if Path::new(model_id).is_dir() {
            return Ok(Self::Dir(PathBuf::from(model_id)));
        }
        let repo = Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            source.revision.as_deref().unwrap_or("main").to_string(),
        );
        // TODO: Let hf-hub read HF_HUB_CACHE after upgrading to 1.0.
        let cache = std::env::var_os("HF_HUB_CACHE")
            .map(|path| Cache::new(path.into()))
            .unwrap_or_else(Cache::from_env);
        if let Some(snapshot) = cached_snapshot_dir(&cache, &repo).map(Self::Dir)
            && snapshot.contains(required).await?
        {
            return Ok(snapshot);
        }
        if is_hf_hub_offline() {
            let hint = match required {
                RequiredFile::Tokenizer => cached_snapshot_dir(&cache, &repo)
                    .map(Self::Dir)
                    .map_or("", |snapshot| snapshot.slow_bpe_tokenizer_hint()),
                RequiredFile::Config | RequiredFile::Nothing => "",
            };
            return Err(Error::Tokenizer(format!(
                "model '{model_id}' at revision '{}' has no {required} in cache '{}' \
                 and HF_HUB_OFFLINE is enabled{hint}",
                repo.revision(),
                cache.path().display(),
            )));
        }

        let api = build_api(&cache).map_err(|error| Error::Tokenizer(error.to_report_string()))?;
        let repo = api.repo(repo);
        let info = repo.info().await.map_err(|error| {
            Error::Tokenizer(format!(
                "failed to fetch model '{model_id}': {}",
                error.as_report()
            ))
        })?;
        let siblings = info.siblings.into_iter().map(|sibling| sibling.rfilename).collect();
        Ok(Self::Remote {
            model_id: model_id.to_string(),
            repo,
            siblings,
        })
    }

    /// Return the path of `filename`, downloading it if the source lists it.
    async fn get(&self, filename: &str) -> Result<Option<PathBuf>> {
        match self {
            Self::Dir(dir) => Ok(local_file_if_exists(dir, filename)),
            Self::Remote {
                model_id,
                repo,
                siblings,
            } => match siblings.contains(filename) {
                true => download_known_file(repo, model_id, filename).await.map(Some),
                false => Ok(None),
            },
        }
    }

    /// Return whether this source provides `required`.
    async fn contains(&self, required: RequiredFile) -> Result<bool> {
        Ok(match required {
            RequiredFile::Tokenizer => self.tokenizer_file().await?.is_some(),
            RequiredFile::Config => self.get("config.json").await?.is_some(),
            RequiredFile::Nothing => true,
        })
    }

    /// Find the tokenizer file: `tekken.json`, `tokenizer.json`, or a tiktoken file.
    async fn tokenizer_file(&self) -> Result<Option<PathBuf>> {
        for filename in ["tekken.json", "tokenizer.json"] {
            if let Some(path) = self.get(filename).await? {
                return Ok(Some(path));
            }
        }
        match self {
            Self::Dir(dir) => Ok(discover_tiktoken_in_dir(dir)),
            Self::Remote {
                model_id,
                repo,
                siblings,
            } => match find_tiktoken_sibling(siblings) {
                Some(filename) => download_known_file(repo, model_id, filename).await.map(Some),
                None => Ok(None),
            },
        }
    }

    /// Return [`SLOW_BPE_TOKENIZER_HINT`] if this source ships `vocab.json` and
    /// `merges.txt`, checked without downloading them; otherwise an empty string.
    fn slow_bpe_tokenizer_hint(&self) -> &'static str {
        let has = |filename: &str| match self {
            Self::Dir(dir) => dir.join(filename).is_file(),
            Self::Remote { siblings, .. } => siblings.contains(filename),
        };
        if has("vocab.json") && has("merges.txt") {
            SLOW_BPE_TOKENIZER_HINT
        } else {
            ""
        }
    }

    /// Select the tokenizer source from the files found by [`Self::tokenizer_file`].
    async fn tokenizer(&self, tokenizer_config: &HfTokenizerConfig) -> Result<TokenizerSource> {
        let Some(tokenizer_path) = self.tokenizer_file().await? else {
            let message = match self {
                Self::Dir(dir) => format!(
                    "local model directory '{}' does not contain a supported tokenizer file \
                     (tokenizer.json, tiktoken.model, or *.tiktoken)",
                    dir.display()
                ),
                Self::Remote { model_id, .. } => format!(
                    "model '{model_id}' does not expose a supported tokenizer file \
                     (tokenizer.json, tiktoken.model, or *.tiktoken) on Hugging Face"
                ),
            };
            return Err(Error::Tokenizer(format!(
                "{message}{}",
                self.slow_bpe_tokenizer_hint()
            )));
        };
        if tokenizer_path.ends_with("tekken.json") {
            return Ok(TokenizerSource::Tekken(tokenizer_path));
        }

        Ok(resolve_tokenizer_source(
            tokenizer_path,
            tokenizer_config.tokenizer_class.as_deref(),
            None,
        ))
    }

    /// Find the dedicated chat template file, if any.
    async fn chat_template(&self) -> Result<Option<PathBuf>> {
        match self {
            Self::Dir(dir) => Ok(discover_chat_template_in_dir(dir)),
            Self::Remote {
                model_id,
                repo,
                siblings,
            } => match remote_chat_template_name(siblings) {
                Some(filename) => download_known_file(repo, model_id, filename).await.map(Some),
                None => Ok(None),
            },
        }
    }
}

/// Locate the cached snapshot directory of `repo` at its revision.
fn cached_snapshot_dir(cache: &Cache, repo: &Repo) -> Option<PathBuf> {
    let cache_repo = cache.repo(repo.clone());
    // hf-hub 0.5 looks up refs even for commit SHAs; Python caches have no such ref.
    let commit = if repo.revision().len() == 40
        && repo.revision().bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        repo.revision().to_string()
    } else {
        let ref_path = cache.path().join(repo.folder_name()).join("refs").join(repo.revision());
        std::fs::read_to_string(ref_path).ok()?
    };
    Some(cache_repo.pointer_path(&commit)).filter(|dir| dir.is_dir())
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
fn find_tiktoken_sibling(siblings: &BTreeSet<String>) -> Option<&str> {
    if siblings.contains("tiktoken.model") {
        return Some("tiktoken.model");
    }
    siblings.iter().map(String::as_str).find(|name| name.ends_with(".tiktoken"))
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

/// Select a root-level template, keeping auxiliary tokenizer templates separate.
fn remote_chat_template_name(siblings: &BTreeSet<String>) -> Option<&str> {
    siblings
        .contains("chat_template.json")
        .then_some("chat_template.json")
        .or_else(|| siblings.contains("chat_template.jinja").then_some("chat_template.jinja"))
        .or_else(|| {
            siblings
                .iter()
                .map(String::as_str)
                .find(|name| !name.contains('/') && name.ends_with(".jinja"))
        })
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

    use super::{ModelFileSources, ModelSource, ResolvedModelFiles, TokenizerSource};

    fn source(id: impl AsRef<std::ffi::OsStr>, revision: Option<&str>) -> ModelSource {
        ModelSource {
            id: id.as_ref().to_str().unwrap().to_string(),
            revision: revision.map(str::to_string),
        }
    }

    #[test]
    fn remote_chat_template_selection_stays_with_main_tokenizer() {
        let mut siblings = std::collections::BTreeSet::from(
            [
                "audio_tokenizer/chat_template.jinja",
                "tokenizer_config.json",
            ]
            .map(String::from),
        );
        assert_eq!(super::remote_chat_template_name(&siblings), None);
        for name in [
            "template.jinja",
            "chat_template.jinja",
            "chat_template.json",
        ] {
            siblings.insert(name.to_string());
            assert_eq!(super::remote_chat_template_name(&siblings), Some(name));
        }
    }

    #[tokio::test]
    async fn overrides_keep_configs_alive_across_clones_and_isolate_instances() {
        let source = tempdir().unwrap();
        let original = r#"{"vocab_size":10}"#;
        fs::write(source.path().join("config.json"), original).unwrap();
        fs::write(source.path().join("tokenizer.json"), "{}").unwrap();
        let mut first =
            ResolvedModelFiles::new(source.path().to_str().unwrap(), None).await.unwrap();
        let patch = serde_json::from_value(serde_json::json!({"vocab_size":20})).unwrap();
        first.apply_overrides(&patch).unwrap();
        let first_path = first.config_path.clone().unwrap();

        let mut second = first.clone();
        second.apply_overrides(&patch).unwrap();
        let second_path = second.config_path.clone().unwrap();
        assert_ne!(first_path, second_path);

        let retained = first.clone();
        drop(first);
        assert!(first_path.exists());
        drop(second);
        assert!(!second_path.exists());
        assert!(first_path.exists());
        drop(retained);
        assert!(!first_path.exists());
        assert_eq!(
            fs::read_to_string(source.path().join("config.json")).unwrap(),
            original
        );
    }

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

    #[tokio::test]
    async fn resolved_model_files_read_each_group_from_its_source() {
        let dirs = ["model", "tokenizer", "config"].map(|_| tempdir().unwrap());
        for dir in &dirs {
            for file in [
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
                "config.json",
                "generation_config.json",
                "preprocessor_config.json",
                "video_preprocessor_config.json",
                "processor_config.json",
            ] {
                fs::write(dir.path().join(file), "{}").unwrap();
            }
        }
        let [model, tokenizer, config] = &dirs;

        let files = ResolvedModelFiles::from_sources(&ModelFileSources {
            model: source(model.path(), None),
            tokenizer: Some(source(tokenizer.path(), None)),
            hf_config: Some(source(config.path(), None)),
        })
        .await
        .unwrap();

        let origin = |path: Option<&std::path::Path>| {
            let dir = path.unwrap().parent().unwrap();
            ["model", "tokenizer", "config"][dirs.iter().position(|d| d.path() == dir).unwrap()]
        };
        let origins = [
            ("tokenizer", Some(files.tokenizer.path())),
            ("tokenizer_config", files.tokenizer_config_path.as_deref()),
            ("chat_template", files.chat_template_path.as_deref()),
            ("config", files.config_path.as_deref()),
            ("generation_config", files.generation_config_path.as_deref()),
            (
                "preprocessor_config",
                files.preprocessor_config_path.as_deref(),
            ),
            (
                "video_preprocessor_config",
                files.video_preprocessor_config_path.as_deref(),
            ),
            ("processor_config", files.processor_config_path.as_deref()),
        ]
        .map(|(name, path)| format!("{name}: {}\n", origin(path)))
        .concat();
        expect_test::expect![[r#"
            tokenizer: tokenizer
            tokenizer_config: tokenizer
            chat_template: tokenizer
            config: config
            generation_config: config
            preprocessor_config: model
            video_preprocessor_config: model
            processor_config: model
        "#]]
        .assert_eq(&origins);
    }

    #[tokio::test]
    async fn slow_bpe_tokenizer_files_get_a_tokenizer_json_hint() {
        let dir = tempdir().unwrap();
        for file in ["vocab.json", "merges.txt", "config.json"] {
            fs::write(dir.path().join(file), "{}").unwrap();
        }

        let error = ResolvedModelFiles::new(dir.path().to_str().unwrap(), None).await.unwrap_err();
        let message = thiserror_ext::AsReport::as_report(&error).to_string();
        assert!(
            message.contains("Found `vocab.json` and `merges.txt`"),
            "{message}"
        );
        assert!(message.contains("--tokenizer <dir>"), "{message}");
    }

    #[tokio::test]
    async fn model_without_tokenizer_resolves_with_tokenizer_and_config_sources() {
        let model = tempdir().unwrap();
        fs::write(model.path().join("model.gguf"), "").unwrap();
        let tokenizer = tempdir().unwrap();
        fs::write(tokenizer.path().join("tokenizer.json"), "{}").unwrap();
        fs::write(tokenizer.path().join("config.json"), "{}").unwrap();

        let error =
            ResolvedModelFiles::new(model.path().to_str().unwrap(), None).await.unwrap_err();
        let message = thiserror_ext::AsReport::as_report(&error).to_string();
        assert!(
            message.contains("does not contain a supported tokenizer file"),
            "{message}"
        );
        assert!(!message.contains("vocab.json"), "{message}");

        let files = ResolvedModelFiles::from_sources(&ModelFileSources {
            model: source(model.path(), None),
            tokenizer: Some(source(tokenizer.path(), None)),
            hf_config: Some(source(tokenizer.path(), None)),
        })
        .await
        .unwrap();
        assert_eq!(
            files.tokenizer.path(),
            tokenizer.path().join("tokenizer.json")
        );
        assert_eq!(
            files.config_path,
            Some(tokenizer.path().join("config.json"))
        );
    }

    #[tokio::test]
    async fn only_explicit_config_source_requires_config_json() {
        let model = tempdir().unwrap();
        fs::write(model.path().join("tokenizer.json"), "{}").unwrap();
        let files = ResolvedModelFiles::new(model.path().to_str().unwrap(), None).await.unwrap();
        assert_eq!(files.config_path, None);

        let error = ResolvedModelFiles::from_sources(&ModelFileSources {
            model: source(model.path(), None),
            tokenizer: None,
            hf_config: Some(source(model.path(), Some("release"))),
        })
        .await
        .unwrap_err();
        let message = thiserror_ext::AsReport::as_report(&error).to_string();
        assert!(
            message.contains("at revision 'release' does not contain config.json"),
            "{message}"
        );
    }

    #[test]
    fn cached_snapshot_dir_respects_branch_tag_and_commit_revisions() {
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
        }
        for (revision, expected_sha) in [
            ("main", Some(&main_sha)),
            ("release/v1", Some(&pinned_sha)),
            (pinned_sha.as_str(), Some(&pinned_sha)),
            ("missing", None),
        ] {
            let repo =
                hf_hub::Repo::with_revision(model.into(), hf_hub::RepoType::Model, revision.into());
            let expected = expected_sha
                .map(|sha| dir.path().join(repo.folder_name()).join("snapshots").join(sha));
            assert_eq!(super::cached_snapshot_dir(&cache, &repo), expected);
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
            assert!(!message.contains("vocab.json"), "{message}");

            let error = ResolvedModelFiles::new("test/slow", Some("release")).await.unwrap_err();
            let message = thiserror_ext::AsReport::as_report(&error).to_string();
            assert!(
                message.contains("Found `vocab.json` and `merges.txt`"),
                "{message}"
            );

            // A weights-only snapshot serves the model once tokenizer and config come from elsewhere.
            let sources = ModelFileSources {
                model: source("test/weights", Some("release")),
                tokenizer: Some(source("test/model", Some("release"))),
                hf_config: Some(source("test/config", Some("release"))),
            };
            let files = ResolvedModelFiles::from_sources(&sources).await.unwrap();
            assert_eq!(
                files.tokenizer.path(),
                cache.join("models--test--model/snapshots/commit/tokenizer.json")
            );
            assert_eq!(
                files.config_path,
                Some(cache.join("models--test--config/snapshots/commit/config.json"))
            );
            let error = ResolvedModelFiles::from_sources(&ModelFileSources {
                hf_config: Some(source("test/model", Some("release"))),
                ..sources
            })
            .await
            .unwrap_err();
            let message = thiserror_ext::AsReport::as_report(&error).to_string();
            assert!(message.contains("has no config.json in cache"), "{message}");
            return;
        }

        // Isolate environment variables from other tests and Tokio worker threads.
        let dir = tempdir().unwrap();
        let cache = dir.path().join("hub-cache");
        for (repo, file) in [
            ("models--test--model", "tokenizer.json"),
            ("models--test--weights", "model.gguf"),
            ("models--test--config", "config.json"),
            ("models--test--slow", "vocab.json"),
            ("models--test--slow", "merges.txt"),
        ] {
            let repo = cache.join(repo);
            fs::create_dir_all(repo.join("refs")).unwrap();
            fs::create_dir_all(repo.join("snapshots/commit")).unwrap();
            fs::write(repo.join("refs/release"), "commit").unwrap();
            fs::write(repo.join("snapshots/commit").join(file), "{}").unwrap();
        }
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
