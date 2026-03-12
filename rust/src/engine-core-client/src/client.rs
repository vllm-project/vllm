use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::protocol::{EngineCoreOutputs, EngineCoreRequest};

/// Configuration for connecting a Rust frontend client to an already running
/// Python `EngineCoreProc`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZmqEngineCoreClientConfig {
    /// Startup handshake address that the Python engine connects to first.
    pub handshake_address: String,
    /// Local host/interface used when allocating the frontend input/output addresses.
    pub local_host: String,
    /// Timeout while waiting for each step of the startup handshake.
    pub ready_timeout: Duration,
    /// Frontend client index stamped onto every request.
    pub client_index: u32,
}

impl ZmqEngineCoreClientConfig {
    pub fn new(handshake_address: impl Into<String>) -> Self {
        Self {
            handshake_address: handshake_address.into(),
            local_host: "127.0.0.1".to_string(),
            ready_timeout: Duration::from_secs(30),
            client_index: 0,
        }
    }
}

/// Decoded engine ready-handshake payload.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReadyMessage {
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub local: Option<bool>,
    #[serde(default)]
    pub headless: Option<bool>,
    #[serde(default)]
    pub num_gpu_blocks: Option<u64>,
    #[serde(default)]
    pub dp_stats_address: Option<String>,
    #[serde(default)]
    pub parallel_config_hash: Option<u64>,
}

/// Minimal async engine-core client surface for the first-stage Rust frontend.
#[async_trait]
pub trait EngineCoreClient {
    /// Add a new request to the engine.
    async fn add_request(&self, req: EngineCoreRequest) -> Result<()>;
    /// Abort currently in-flight requests by request ID.
    async fn abort_requests(&self, ids: &[String]) -> Result<()>;
    /// Wait for the next batch of outputs from the engine.
    async fn next_output(&mut self) -> Result<EngineCoreOutputs>;
    /// Shut down local client tasks and close transport state.
    async fn shutdown(&mut self) -> Result<()>;
}
