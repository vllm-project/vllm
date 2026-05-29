use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::time::{Duration, Instant, sleep_until};
use tracing::warn;
use vllm_chat::ChatLlm;
use vllm_engine_core_client::EngineCoreClient;
use vllm_engine_core_client::protocol::lora::LoraRequest;

use crate::lora::{LoadLoraError, LoraManager, LoraModelResolution, UnloadLoraError};

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
    /// Dynamic LoRA adapter registry.
    lora_manager: LoraManager,
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
            lora_manager: LoraManager::new(),
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
        self.lora_manager.served_model_names(&self.served_model_names).await
    }

    /// Resolve the requested model against one dynamic LoRA registry snapshot.
    pub async fn resolve_model_with_loras(&self, model_name: Option<&str>) -> LoraModelResolution {
        self.lora_manager.resolve_model(&self.served_model_names, model_name).await
    }

    /// Load one dynamic LoRA adapter and register it as a public model name.
    pub async fn load_lora(
        &self,
        lora_name: String,
        lora_path: String,
        load_inplace: bool,
        is_3d_lora_weight: bool,
    ) -> Result<LoraRequest, LoadLoraError> {
        self.lora_manager
            .load_lora(
                self.engine_core_client(),
                &self.served_model_names,
                lora_name,
                lora_path,
                load_inplace,
                is_3d_lora_weight,
            )
            .await
    }

    /// Remove one dynamic LoRA adapter from the engine and public model
    /// registry.
    pub async fn unload_lora(
        &self,
        lora_name: &str,
        lora_int_id: Option<u64>,
    ) -> Result<LoraRequest, UnloadLoraError> {
        self.lora_manager
            .unload_lora(self.engine_core_client(), lora_name, lora_int_id)
            .await
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
