// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::io;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use indexmap::IndexMap;
use thiserror::Error;
use thiserror_ext::Macro;
use tokio::sync::{Mutex, RwLock};
use vllm_engine_core_client::EngineCoreClient;
use vllm_engine_core_client::protocol::lora::LoraRequest;

use crate::config::LoraModulePath;
use crate::model_names::ServedModelNames;

const RUNTIME_LORA_ALLOWED_PATH_PREFIXES_ENV: &str = "VLLM_RUNTIME_LORA_ALLOWED_PATH_PREFIXES";

#[derive(Debug, Error, Macro)]
pub(crate) enum LoraPathAccessError {
    #[error("No adapter found for {lora_path}")]
    NotFound { lora_path: String },
    #[error("{message}")]
    InvalidPath { message: String },
    #[error("{message}")]
    InvalidConfiguration { message: String },
}

fn runtime_lora_allowed_path_prefixes() -> Option<Vec<PathBuf>> {
    let prefixes = std::env::var_os(RUNTIME_LORA_ALLOWED_PATH_PREFIXES_ENV)?;
    let prefixes: Vec<_> = std::env::split_paths(&prefixes)
        .filter(|path| !path.as_os_str().is_empty())
        .collect();
    (!prefixes.is_empty()).then_some(prefixes)
}

fn looks_like_local_lora_path(lora_path: &str) -> bool {
    let path = Path::new(lora_path);
    path.is_absolute()
        || lora_path.starts_with('~')
        || lora_path.starts_with('.')
        || path.components().any(|component| matches!(component, Component::ParentDir))
}

fn validate_lora_path_access(
    lora_path: &str,
    allowed_prefixes: Option<&[PathBuf]>,
) -> Result<Option<String>, LoraPathAccessError> {
    let path = Path::new(lora_path);
    if !looks_like_local_lora_path(lora_path) && !path.exists() {
        return Ok(None);
    }

    let Some(allowed_prefixes) = allowed_prefixes else {
        return Err(invalid_path!(
            "Local LoRA adapter paths require {RUNTIME_LORA_ALLOWED_PATH_PREFIXES_ENV} to be configured."
        ));
    };

    if !path.is_absolute() {
        return Err(invalid_path!(
            "Local LoRA adapter paths must be absolute and under one of the prefixes configured by {RUNTIME_LORA_ALLOWED_PATH_PREFIXES_ENV}."
        ));
    }

    let canonical_path = path.canonicalize().map_err(|error| match error.kind() {
        io::ErrorKind::NotFound => LoraPathAccessError::NotFound {
            lora_path: lora_path.to_string(),
        },
        _ => invalid_path!("Local LoRA adapter path must exist and be accessible."),
    })?;
    let canonical_prefixes = allowed_prefixes
        .iter()
        .map(|prefix| {
            prefix.canonicalize().map_err(|_| {
                invalid_configuration!(
                    "configured {RUNTIME_LORA_ALLOWED_PATH_PREFIXES_ENV} path prefix must exist and be accessible"
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    if !canonical_prefixes.iter().any(|prefix| canonical_path.starts_with(prefix)) {
        return Err(invalid_path!(
            "Local LoRA adapter path is outside the configured allowed prefixes."
        ));
    }

    Ok(Some(canonical_path.to_string_lossy().into_owned()))
}

/// A LoRA adapter registered under a public model name.
///
/// Holds what identifies the loaded adapter, not the request that loaded it:
/// per-load options such as `load_inplace` stay on that load, so the engine
/// requests built from the registry never carry them.
#[derive(Debug, Clone)]
pub(crate) struct LoadedLora {
    pub lora_name: String,
    pub lora_int_id: u64,
    pub lora_path: String,
    pub base_model_name: Option<String>,
    pub is_3d_lora_weight: bool,
}

impl LoadedLora {
    /// The engine request that selects this adapter.
    fn to_request(&self) -> LoraRequest {
        LoraRequest {
            lora_name: self.lora_name.clone(),
            lora_int_id: self.lora_int_id,
            lora_path: self.lora_path.clone(),
            base_model_name: self.base_model_name.clone(),
            load_inplace: false,
            is_3d_lora_weight: self.is_3d_lora_weight,
        }
    }
}

/// Snapshot of the currently served model names plus the requested LoRA, if
/// the model name resolves to a dynamic adapter.
#[derive(Debug, Clone)]
pub(crate) struct LoraModelResolution {
    pub model_names: ServedModelNames,
    pub lora_request: Option<LoraRequest>,
}

/// Runtime registry for dynamically loaded LoRA adapters.
pub(crate) struct LoraManager {
    /// Dynamically loaded LoRA adapters keyed by public model name, in load order.
    adapters: RwLock<IndexMap<String, LoadedLora>>,
    /// Monotonic adapter id allocator. LoRA ids are one-indexed.
    id_counter: AtomicU64,
    /// Serialize dynamic LoRA registry updates around engine utility calls.
    update_lock: Mutex<()>,
}

#[derive(Debug, Error)]
#[error("engine was not started with LoRA enabled")]
pub(crate) struct LoraDisabledError;

#[derive(Debug, Error, Macro)]
pub(crate) enum LoadLoraError {
    #[error(transparent)]
    Disabled(#[from] LoraDisabledError),
    #[error("{message}")]
    InvalidAdapter { message: String },
    #[error(transparent)]
    PathAccess(#[from] LoraPathAccessError),
    #[error("LoRA adapter `{lora_name}` is already loaded")]
    AlreadyLoaded { lora_name: String },
    #[error("LoRA adapter `{lora_name}` conflicts with a served base model")]
    BaseModelName { lora_name: String },
    #[error("failed to load LoRA adapter `{lora_name}`")]
    Engine {
        lora_name: String,
        #[source]
        source: vllm_engine_core_client::Error,
    },
    #[error("one or more engine ranks rejected LoRA adapter `{lora_name}`")]
    NotLoaded { lora_name: String },
}

#[derive(Debug, Error)]
pub(crate) enum UnloadLoraError {
    #[error(transparent)]
    Disabled(#[from] LoraDisabledError),
    #[error("LoRA adapter `{lora_name}` is not loaded")]
    NotFound { lora_name: String },
    #[error(
        "requested lora_int_id {actual} does not match loaded adapter `{lora_name}` with id {expected}"
    )]
    IntIdMismatch {
        lora_name: String,
        expected: u64,
        actual: u64,
    },
    #[error("failed to unload LoRA adapter `{lora_name}`")]
    Engine {
        lora_name: String,
        #[source]
        source: vllm_engine_core_client::Error,
    },
    #[error("engine rejected removal of LoRA adapter `{lora_name}` with id {lora_int_id}")]
    NotRemoved { lora_name: String, lora_int_id: u64 },
}

impl LoraManager {
    pub fn new() -> Self {
        Self {
            adapters: RwLock::new(IndexMap::new()),
            id_counter: AtomicU64::new(0),
            update_lock: Mutex::new(()),
        }
    }

    /// Snapshot loaded LoRA adapters in load order.
    pub async fn served_loras(&self) -> Vec<LoadedLora> {
        self.adapters.read().await.values().cloned().collect()
    }

    /// Resolve the requested model against one consistent LoRA registry
    /// snapshot.
    pub async fn resolve_model(
        &self,
        base_model_names: ServedModelNames,
        model_name: Option<&str>,
    ) -> LoraModelResolution {
        let adapters = self.adapters.read().await;
        let model_names = base_model_names.with_names(adapters.keys().cloned());
        let lora_request =
            model_name.and_then(|name| adapters.get(name)).map(LoadedLora::to_request);

        LoraModelResolution {
            model_names,
            lora_request,
        }
    }

    /// Load one dynamic LoRA adapter and register it as a public model name.
    pub async fn load_lora(
        &self,
        engine_core_client: &EngineCoreClient,
        base_model_names: &[String],
        mut module: LoraModulePath,
        load_inplace: bool,
    ) -> Result<LoadedLora, LoadLoraError> {
        let allowed_prefixes = runtime_lora_allowed_path_prefixes();
        if let Some(canonical_path) =
            validate_lora_path_access(&module.path, allowed_prefixes.as_deref())?
        {
            module.path = canonical_path;
        }
        self.register(engine_core_client, base_model_names, module, load_inplace).await
    }

    /// Load an operator-configured adapter (`--lora-modules`). Unlike the
    /// runtime endpoint, the path is trusted as given: no allowed-prefix
    /// check applies.
    pub async fn load_static_lora(
        &self,
        engine_core_client: &EngineCoreClient,
        base_model_names: &[String],
        module: &LoraModulePath,
    ) -> Result<LoadedLora, LoadLoraError> {
        self.register(engine_core_client, base_model_names, module.clone(), false).await
    }

    async fn register(
        &self,
        engine_core_client: &EngineCoreClient,
        base_model_names: &[String],
        module: LoraModulePath,
        load_inplace: bool,
    ) -> Result<LoadedLora, LoadLoraError> {
        let LoraModulePath {
            name: lora_name,
            path: lora_path,
            base_model_name,
            is_3d_lora_weight,
        } = module;
        if lora_name.trim().is_empty() {
            bail_invalid_adapter!("lora_name must not be empty");
        }
        if lora_path.trim().is_empty() {
            bail_invalid_adapter!("lora_path must not be empty");
        }
        let _guard = self.update_lock.lock().await;
        if base_model_names.iter().any(|name| name == &lora_name) {
            return Err(LoadLoraError::BaseModelName { lora_name });
        }
        let adapters = self.adapters.read().await;
        let existing_lora_int_id = adapters.get(&lora_name).map(|adapter| adapter.lora_int_id);
        if !load_inplace && existing_lora_int_id.is_some() {
            return Err(LoadLoraError::AlreadyLoaded { lora_name });
        }

        let lora_int_id = existing_lora_int_id
            .unwrap_or_else(|| self.id_counter.fetch_add(1, Ordering::Relaxed) + 1);
        let adapter = LoadedLora {
            lora_name: lora_name.clone(),
            lora_int_id,
            lora_path,
            base_model_name,
            is_3d_lora_weight,
        };
        drop(adapters);

        let lora_request = LoraRequest {
            load_inplace,
            ..adapter.to_request()
        };
        let loaded = engine_core_client.add_lora(&lora_request).await.map_err(|source| {
            LoadLoraError::Engine {
                lora_name: lora_name.clone(),
                source,
            }
        })?;
        if !loaded {
            return Err(LoadLoraError::NotLoaded { lora_name });
        }
        self.adapters.write().await.insert(lora_name, adapter.clone());
        Ok(adapter)
    }

    /// Remove one dynamic LoRA adapter from the engine and public model
    /// registry.
    pub async fn unload_lora(
        &self,
        engine_core_client: &EngineCoreClient,
        lora_name: &str,
        requested_lora_int_id: Option<u64>,
    ) -> Result<LoadedLora, UnloadLoraError> {
        let _guard = self.update_lock.lock().await;
        let adapter = self.adapters.read().await.get(lora_name).cloned().ok_or_else(|| {
            UnloadLoraError::NotFound {
                lora_name: lora_name.to_string(),
            }
        })?;

        if let Some(actual) = requested_lora_int_id
            && actual != adapter.lora_int_id
        {
            return Err(UnloadLoraError::IntIdMismatch {
                lora_name: lora_name.to_string(),
                expected: adapter.lora_int_id,
                actual,
            });
        }

        let removed =
            engine_core_client.remove_lora(adapter.lora_int_id).await.map_err(|source| {
                UnloadLoraError::Engine {
                    lora_name: adapter.lora_name.clone(),
                    source,
                }
            })?;
        if !removed {
            return Err(UnloadLoraError::NotRemoved {
                lora_name: adapter.lora_name,
                lora_int_id: adapter.lora_int_id,
            });
        }

        Ok(self.adapters.write().await.shift_remove(lora_name).unwrap_or(adapter))
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{LoadedLora, LoraManager, LoraPathAccessError, validate_lora_path_access};
    use crate::model_names::ServedModelNames;

    fn temp_lora_dir(test_name: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "vllm-lora-{test_name}-{}-{suffix}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("create temp lora dir");
        path
    }

    #[tokio::test]
    async fn resolve_model_selects_adapter_when_any_model_name_is_accepted() {
        let manager = LoraManager::new();
        manager.adapters.write().await.insert(
            "adapter-a".to_string(),
            LoadedLora {
                lora_name: "adapter-a".to_string(),
                lora_int_id: 1,
                lora_path: "org/adapter-a".to_string(),
                base_model_name: None,
                is_3d_lora_weight: false,
            },
        );
        let base = ServedModelNames::new(vec!["base".to_string()], true);

        let resolution = manager.resolve_model(base.clone(), Some("adapter-a")).await;
        assert_eq!(
            resolution.lora_request.expect("adapter selected").lora_name,
            "adapter-a"
        );

        let resolution = manager.resolve_model(base, Some("unknown")).await;
        assert!(resolution.lora_request.is_none());
        assert!(resolution.model_names.accepts(Some("unknown")));
        assert_eq!(resolution.model_names.primary(), "base");
    }

    #[test]
    fn lora_path_allows_hf_repo_ids_without_prefixes() {
        assert_eq!(
            validate_lora_path_access("org/adapter-a", None).expect("hf repo id should be allowed"),
            None
        );
    }

    #[test]
    fn lora_path_rejects_local_paths_without_prefixes() {
        assert!(validate_lora_path_access("/tmp/adapter-a", None).is_err());
        assert!(validate_lora_path_access("./adapter-a", None).is_err());
        assert!(validate_lora_path_access("~/adapter-a", None).is_err());
        assert!(validate_lora_path_access("subdir/../../../etc/sensitive", None).is_err());
    }

    #[test]
    fn lora_path_rejects_existing_bare_relative_paths_without_prefixes() {
        let root =
            PathBuf::from("target").join(format!("vllm-lora-relative-{}", std::process::id()));
        let adapter = root.join("adapter-a");
        fs::create_dir_all(&adapter).expect("create relative adapter dir");

        assert!(
            validate_lora_path_access(adapter.to_str().expect("utf-8 temp path"), None).is_err()
        );

        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn lora_path_allows_absolute_paths_under_configured_prefixes() {
        let root = temp_lora_dir("allowed-prefix");
        let allowed = root.join("allowed");
        let adapter = allowed.join("adapter-a");
        fs::create_dir_all(&adapter).expect("create adapter dir");

        let prefixes = [allowed];
        let resolved =
            validate_lora_path_access(adapter.to_str().expect("utf-8 temp path"), Some(&prefixes))
                .expect("path under configured prefix should be allowed");
        assert_eq!(
            resolved.as_deref(),
            Some(
                adapter
                    .canonicalize()
                    .expect("canonical adapter")
                    .to_str()
                    .expect("utf-8 temp path")
            )
        );

        fs::remove_dir_all(root).ok();
    }

    /// Python answers a missing adapter with 404 `NotFoundError`, so the
    /// guard reports a missing path apart from an inaccessible one.
    #[test]
    fn lora_path_reports_missing_paths_as_not_found() {
        let root = temp_lora_dir("missing-path");
        let missing = root.join("missing-adapter");

        let prefixes = [root.clone()];
        let error =
            validate_lora_path_access(missing.to_str().expect("utf-8 temp path"), Some(&prefixes))
                .expect_err("missing adapter path should be rejected");
        assert!(
            matches!(&error, LoraPathAccessError::NotFound { lora_path } if *lora_path == missing.to_str().unwrap()),
            "{error:?}"
        );

        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn lora_path_rejects_parent_escape_from_configured_prefixes() {
        let root = temp_lora_dir("parent-escape");
        let allowed = root.join("allowed");
        let private_adapter = root.join("private").join("adapter-a");
        fs::create_dir_all(&allowed).expect("create allowed dir");
        fs::create_dir_all(&private_adapter).expect("create private adapter dir");

        let escaped = allowed.join("../private/adapter-a");
        let prefixes = [allowed];
        assert!(
            validate_lora_path_access(escaped.to_str().expect("utf-8 temp path"), Some(&prefixes))
                .is_err()
        );

        fs::remove_dir_all(root).ok();
    }
}
