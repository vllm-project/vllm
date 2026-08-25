// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::atomic::{AtomicU64, Ordering};

use indexmap::IndexMap;
use thiserror::Error;
use tokio::sync::{Mutex, RwLock};
use vllm_engine_core_client::EngineCoreClient;
use vllm_engine_core_client::protocol::lora::{LoraRequest, LoraRequestError};

/// Snapshot of the currently served model names plus the requested LoRA, if
/// the model name resolves to a dynamic adapter.
#[derive(Debug, Clone)]
pub(crate) struct LoraModelResolution {
    pub model_names: Vec<String>,
    pub lora_request: Option<LoraRequest>,
}

/// Runtime registry for dynamically loaded LoRA adapters.
pub(crate) struct LoraManager {
    /// Dynamically loaded LoRA adapters keyed by public model name, in load order.
    requests: RwLock<IndexMap<String, LoraRequest>>,
    /// Monotonic adapter id allocator. LoRA ids are one-indexed.
    id_counter: AtomicU64,
    /// Serialize dynamic LoRA registry updates around engine utility calls.
    update_lock: Mutex<()>,
}

#[derive(Debug, Error)]
#[error("engine was not started with LoRA enabled")]
pub(crate) struct LoraDisabledError;

#[derive(Debug, Error)]
pub(crate) enum LoadLoraError {
    #[error(transparent)]
    Disabled(#[from] LoraDisabledError),
    #[error(transparent)]
    InvalidRequest(#[from] LoraRequestError),
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
            requests: RwLock::new(IndexMap::new()),
            id_counter: AtomicU64::new(0),
            update_lock: Mutex::new(()),
        }
    }

    /// Snapshot loaded LoRA adapters in load order.
    pub async fn served_lora_requests(&self) -> Vec<LoraRequest> {
        self.requests.read().await.values().cloned().collect()
    }

    /// Resolve the requested model against one consistent LoRA registry
    /// snapshot.
    pub async fn resolve_model(
        &self,
        base_model_names: &[String],
        model_name: Option<&str>,
    ) -> LoraModelResolution {
        let requests = self.requests.read().await;
        let mut model_names = base_model_names.to_vec();
        model_names.extend(requests.keys().cloned());
        let lora_request = model_name.and_then(|name| requests.get(name).cloned());

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
        lora_name: String,
        lora_path: String,
        load_inplace: bool,
        is_3d_lora_weight: bool,
    ) -> Result<LoraRequest, LoadLoraError> {
        let _guard = self.update_lock.lock().await;
        if base_model_names.iter().any(|name| name == &lora_name) {
            return Err(LoadLoraError::BaseModelName { lora_name });
        }
        let requests = self.requests.read().await;
        let existing_lora_int_id = requests.get(&lora_name).map(|request| request.lora_int_id);
        if !load_inplace && existing_lora_int_id.is_some() {
            return Err(LoadLoraError::AlreadyLoaded { lora_name });
        }

        let lora_int_id = existing_lora_int_id
            .unwrap_or_else(|| self.id_counter.fetch_add(1, Ordering::Relaxed) + 1);
        let lora_request = LoraRequest::new(
            lora_name.clone(),
            lora_int_id,
            lora_path,
            load_inplace,
            is_3d_lora_weight,
        )
        .map_err(LoadLoraError::InvalidRequest)?;
        drop(requests);

        let loaded = engine_core_client.add_lora(&lora_request).await.map_err(|source| {
            LoadLoraError::Engine {
                lora_name: lora_name.clone(),
                source,
            }
        })?;
        if !loaded {
            return Err(LoadLoraError::NotLoaded { lora_name });
        }
        self.requests.write().await.insert(lora_name, lora_request.clone());
        Ok(lora_request)
    }

    /// Remove one dynamic LoRA adapter from the engine and public model
    /// registry.
    pub async fn unload_lora(
        &self,
        engine_core_client: &EngineCoreClient,
        lora_name: &str,
        requested_lora_int_id: Option<u64>,
    ) -> Result<LoraRequest, UnloadLoraError> {
        let _guard = self.update_lock.lock().await;
        let lora_request = self.requests.read().await.get(lora_name).cloned().ok_or_else(|| {
            UnloadLoraError::NotFound {
                lora_name: lora_name.to_string(),
            }
        })?;

        if let Some(actual) = requested_lora_int_id
            && actual != lora_request.lora_int_id
        {
            return Err(UnloadLoraError::IntIdMismatch {
                lora_name: lora_name.to_string(),
                expected: lora_request.lora_int_id,
                actual,
            });
        }

        let removed =
            engine_core_client
                .remove_lora(lora_request.lora_int_id)
                .await
                .map_err(|source| UnloadLoraError::Engine {
                    lora_name: lora_request.lora_name.clone(),
                    source,
                })?;
        if !removed {
            return Err(UnloadLoraError::NotRemoved {
                lora_name: lora_request.lora_name,
                lora_int_id: lora_request.lora_int_id,
            });
        }

        Ok(self.requests.write().await.shift_remove(lora_name).unwrap_or(lora_request))
    }
}
