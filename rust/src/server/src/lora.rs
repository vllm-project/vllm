use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::sync::{Mutex, RwLock};
use vllm_engine_core_client::EngineCoreClient;
use vllm_engine_core_client::protocol::lora::LoraRequest;

/// Snapshot of the currently served model names plus the requested LoRA, if
/// the model name resolves to a dynamic adapter.
#[derive(Debug, Clone)]
pub(crate) struct LoraModelResolution {
    pub model_names: Vec<String>,
    pub lora_request: Option<LoraRequest>,
}

/// Runtime registry for dynamically loaded LoRA adapters.
pub(crate) struct LoraManager {
    /// Dynamically loaded LoRA adapters keyed by public model name.
    requests: RwLock<BTreeMap<String, LoraRequest>>,
    /// Monotonic adapter id allocator. LoRA ids are one-indexed.
    id_counter: AtomicU64,
    /// Serialize dynamic LoRA registry updates around engine utility calls.
    update_lock: Mutex<()>,
}

#[derive(Debug)]
pub(crate) enum LoadLoraError {
    AlreadyLoaded { lora_name: String },
    BaseModelName { lora_name: String },
    Engine(vllm_engine_core_client::Error),
    NotLoaded { lora_name: String },
}

#[derive(Debug)]
pub(crate) enum UnloadLoraError {
    NotFound {
        lora_name: String,
    },
    IntIdMismatch {
        lora_name: String,
        expected: u64,
        actual: u64,
    },
    Engine(vllm_engine_core_client::Error),
    NotRemoved {
        lora_name: String,
        lora_int_id: u64,
    },
}

impl LoraManager {
    pub fn new() -> Self {
        Self {
            requests: RwLock::new(BTreeMap::new()),
            id_counter: AtomicU64::new(0),
            update_lock: Mutex::new(()),
        }
    }

    /// Return base served model names plus dynamically loaded LoRA adapter
    /// names.
    pub async fn served_model_names(&self, base_model_names: &[String]) -> Vec<String> {
        let mut names = base_model_names.to_vec();
        names.extend(self.requests.read().await.keys().cloned());
        names
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
        if !load_inplace && self.requests.read().await.contains_key(&lora_name) {
            return Err(LoadLoraError::AlreadyLoaded { lora_name });
        }

        let lora_int_id = self
            .requests
            .read()
            .await
            .get(&lora_name)
            .map(|request| request.lora_int_id)
            .unwrap_or_else(|| self.id_counter.fetch_add(1, Ordering::Relaxed) + 1);
        let lora_request = LoraRequest::new(
            lora_name.clone(),
            lora_int_id,
            lora_path,
            load_inplace,
            is_3d_lora_weight,
        );

        let loaded = engine_core_client
            .add_lora(&lora_request)
            .await
            .map_err(LoadLoraError::Engine)?;
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

        let removed = engine_core_client
            .remove_lora(lora_request.lora_int_id)
            .await
            .map_err(UnloadLoraError::Engine)?;
        if !removed {
            return Err(UnloadLoraError::NotRemoved {
                lora_name: lora_request.lora_name,
                lora_int_id: lora_request.lora_int_id,
            });
        }

        Ok(self.requests.write().await.remove(lora_name).unwrap_or(lora_request))
    }
}
