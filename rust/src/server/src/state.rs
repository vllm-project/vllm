use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::sync::{Mutex, RwLock};
use tokio::time::{Duration, Instant, sleep_until};
use tracing::warn;
use vllm_chat::ChatLlm;
use vllm_engine_core_client::EngineCoreClient;
use vllm_engine_core_client::protocol::lora::LoRARequest;

const SHUTDOWN_REFCOUNT_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Shared router state for the minimal single-model OpenAI server.
pub struct AppState {
    /// All public model IDs served by this frontend. The first entry is the
    /// primary ID used in responses; all entries are valid in requests.
    served_model_names: Vec<String>,
    /// Shared chat facade used by all requests.
    pub chat: ChatLlm,
    /// Whether to log a summary line for each completed request.
    pub enable_log_requests: bool,
    /// Number of in-flight inference requests currently owned by this frontend.
    server_load: AtomicU64,
    /// Dynamically loaded LoRA adapters keyed by public model name.
    lora_requests: RwLock<BTreeMap<String, LoRARequest>>,
    /// Monotonic adapter id allocator. LoRA ids are one-indexed.
    lora_id_counter: AtomicU64,
    /// Serialize dynamic LoRA registry updates around engine utility calls.
    lora_update_lock: Mutex<()>,
}

#[derive(Debug)]
pub(crate) enum LoadLoRAError {
    AlreadyLoaded { lora_name: String },
    BaseModelName { lora_name: String },
    Engine(vllm_engine_core_client::Error),
    NotLoaded { lora_name: String },
}

#[derive(Debug)]
pub(crate) enum UnloadLoRAError {
    NotFound { lora_name: String },
    Engine(vllm_engine_core_client::Error),
    NotRemoved { lora_name: String, lora_int_id: u64 },
}

impl AppState {
    /// Construct one application state instance.
    ///
    /// `served_model_names` must be non-empty; the first entry is the primary
    /// model ID returned in API responses.
    ///
    /// # Panics
    ///
    /// Panics if `served_model_names` is empty.
    pub fn new(served_model_names: Vec<String>, chat: ChatLlm) -> Self {
        assert!(
            !served_model_names.is_empty(),
            "served_model_names must not be empty"
        );
        Self {
            served_model_names,
            chat,
            enable_log_requests: false,
            server_load: AtomicU64::new(0),
            lora_requests: RwLock::new(BTreeMap::new()),
            lora_id_counter: AtomicU64::new(0),
            lora_update_lock: Mutex::new(()),
        }
    }

    /// Enable per-request completion logging.
    pub fn with_log_requests(mut self, enabled: bool) -> Self {
        self.enable_log_requests = enabled;
        self
    }

    /// The primary model name echoed back in API responses (the first served
    /// name).
    pub fn primary_model_name(&self) -> &str {
        self.served_model_names.first().map(String::as_str).unwrap_or_default()
    }

    /// All model names served by this frontend.
    pub fn served_model_names(&self) -> &[String] {
        &self.served_model_names
    }

    /// Return base served model names plus dynamically loaded LoRA adapter
    /// names.
    pub async fn served_model_names_with_loras(&self) -> Vec<String> {
        let mut names = self.served_model_names.clone();
        names.extend(self.lora_requests.read().await.keys().cloned());
        names
    }

    /// Resolve a dynamically loaded LoRA adapter by public model name.
    pub async fn resolve_lora_request(&self, model_name: &str) -> Option<LoRARequest> {
        self.lora_requests.read().await.get(model_name).cloned()
    }

    /// Load one dynamic LoRA adapter and register it as a public model name.
    pub async fn load_lora(
        &self,
        lora_name: String,
        lora_path: String,
        load_inplace: bool,
        is_3d_lora_weight: bool,
    ) -> Result<LoRARequest, LoadLoRAError> {
        let _guard = self.lora_update_lock.lock().await;
        if self.served_model_names.iter().any(|name| name == &lora_name) {
            return Err(LoadLoRAError::BaseModelName { lora_name });
        }
        if !load_inplace && self.lora_requests.read().await.contains_key(&lora_name) {
            return Err(LoadLoRAError::AlreadyLoaded { lora_name });
        }

        let lora_int_id = self
            .lora_requests
            .read()
            .await
            .get(&lora_name)
            .map(|request| request.lora_int_id)
            .unwrap_or_else(|| self.lora_id_counter.fetch_add(1, Ordering::Relaxed) + 1);
        let lora_request = LoRARequest::new(
            lora_name.clone(),
            lora_int_id,
            lora_path,
            load_inplace,
            is_3d_lora_weight,
        );

        let loaded = self
            .engine_core_client()
            .add_lora(&lora_request)
            .await
            .map_err(LoadLoRAError::Engine)?;
        if !loaded {
            return Err(LoadLoRAError::NotLoaded { lora_name });
        }
        self.lora_requests.write().await.insert(lora_name, lora_request.clone());
        Ok(lora_request)
    }

    /// Remove one dynamic LoRA adapter from the engine and public model
    /// registry.
    pub async fn unload_lora(&self, lora_name: &str) -> Result<LoRARequest, UnloadLoRAError> {
        let _guard = self.lora_update_lock.lock().await;
        let lora_request =
            self.lora_requests.read().await.get(lora_name).cloned().ok_or_else(|| {
                UnloadLoRAError::NotFound {
                    lora_name: lora_name.to_string(),
                }
            })?;

        let removed = self
            .engine_core_client()
            .remove_lora(lora_request.lora_int_id)
            .await
            .map_err(UnloadLoRAError::Engine)?;
        if !removed {
            return Err(UnloadLoRAError::NotRemoved {
                lora_name: lora_request.lora_name,
                lora_int_id: lora_request.lora_int_id,
            });
        }

        Ok(self.lora_requests.write().await.remove(lora_name).unwrap_or(lora_request))
    }

    /// Return a reference to the underlying engine core client for utility
    /// calls.
    pub(crate) fn engine_core_client(&self) -> &EngineCoreClient {
        self.chat.engine_core_client()
    }

    /// Return the current in-flight inference request count for the `/load`
    /// endpoint.
    pub fn server_load(&self) -> u64 {
        self.server_load.load(Ordering::Relaxed)
    }

    /// Increment the in-flight inference request count, called by the load
    /// tracking middleware.
    pub(crate) fn increment_server_load(&self) {
        self.server_load.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement the in-flight inference request count, called by the load
    /// tracking middleware.
    pub(crate) fn decrement_server_load(&self) {
        self.server_load.fetch_sub(1, Ordering::Relaxed);
    }

    /// Wait until all request-owned references are dropped, then shut down the
    /// engine client.
    ///
    /// If the deadline elapses while request/connection tasks still hold state
    /// references, skip the clean engine-client shutdown and let process
    /// teardown reclaim the remaining resources.
    pub async fn shutdown(mut self: Arc<Self>, deadline: Instant) -> anyhow::Result<()> {
        loop {
            match Arc::try_unwrap(self) {
                Ok(state) => {
                    state.chat.shutdown().await?;
                    return Ok(());
                }
                Err(state) => self = state,
            }
            let ref_count = Arc::strong_count(&self);

            let now = Instant::now();
            if now >= deadline {
                warn!(
                    ref_count,
                    "shutdown deadline elapsed before app state became idle; skipping engine-client shutdown"
                );
                return Ok(());
            }

            sleep_until(std::cmp::min(
                deadline,
                now + SHUTDOWN_REFCOUNT_POLL_INTERVAL,
            ))
            .await;
        }
    }
}
