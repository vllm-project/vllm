use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::time::{Duration, Instant, sleep_until};
use tracing::warn;
use vllm_chat::ChatLlm;
use vllm_engine_core_client::EngineCoreClient;
use vllm_engine_core_client::protocol::lora::LoraRequest;

use crate::config::{ApiServerOptions, CorsConfig};
use crate::lora::{LoadLoraError, LoraManager, LoraModelResolution, UnloadLoraError};
use crate::server_info::{ServerInfoConfigFormat, ServerInfoSnapshot};

const SHUTDOWN_REFCOUNT_POLL_INTERVAL: Duration = Duration::from_millis(100);

pub(crate) type ApiKeyHash = [u8; 32];

pub(crate) fn hash_api_key(api_key: &str) -> ApiKeyHash {
    Sha256::digest(api_key.as_bytes()).into()
}

/// Shared router state for the minimal single-model OpenAI server.
pub struct AppState {
    /// All public model IDs served by this frontend. The first entry is the
    /// primary ID used in responses; all entries are valid in requests.
    served_model_names: Vec<String>,
    /// Shared chat facade used by all requests.
    pub chat: ChatLlm,
    /// HTTP/API-server behavior switches.
    pub api_server_options: ApiServerOptions,
    /// CORS settings applied to every HTTP response.
    pub cors: CorsConfig,
    /// Runtime server information returned by `/server_info`, when available.
    server_info: Option<ServerInfoSnapshot>,
    /// SHA-256 hashes of API keys accepted as bearer tokens for guarded routes.
    api_key_hashes: Vec<ApiKeyHash>,
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
            api_server_options: ApiServerOptions::default(),
            cors: CorsConfig::default(),
            server_info: None,
            api_key_hashes: Vec::new(),
            server_load: AtomicU64::new(0),
            lora_manager: LoraManager::new(),
        }
    }

    /// Set HTTP/API-server behavior switches.
    pub fn with_api_server_options(mut self, options: ApiServerOptions) -> Self {
        self.api_server_options = options;
        self
    }

    /// Set the CORS settings applied to every HTTP response.
    pub fn with_cors(mut self, cors: CorsConfig) -> Self {
        self.cors = cors;
        self
    }

    /// Attach the runtime server information snapshot used by `/server_info`.
    pub(crate) fn with_server_info(mut self, server_info: ServerInfoSnapshot) -> Self {
        self.server_info = Some(server_info);
        self
    }

    /// Build a `/server_info` response payload.
    pub(crate) fn server_info_response(
        &self,
        config_format: ServerInfoConfigFormat,
    ) -> Option<Value> {
        self.server_info.as_ref().map(|server_info| server_info.response(config_format))
    }

    /// Configure API keys accepted by guarded HTTP routes.
    pub fn with_api_keys(mut self, api_keys: Vec<String>) -> Self {
        self.api_key_hashes = api_keys
            .into_iter()
            .filter(|key| !key.is_empty())
            .map(|key| hash_api_key(&key))
            .collect();
        self
    }

    pub(crate) fn has_api_keys(&self) -> bool {
        !self.api_key_hashes.is_empty()
    }

    pub(crate) fn api_key_hashes(&self) -> &[ApiKeyHash] {
        &self.api_key_hashes
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
