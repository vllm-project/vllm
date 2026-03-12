use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::protocol::{EngineCoreOutputs, EngineCoreRequest};

/// Configuration for connecting a Rust frontend client to an already running
/// Python `EngineCoreProc`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZmqEngineCoreClientConfig {
    /// ROUTER bind address used for frontend -> engine requests.
    pub input_address: String,
    /// PULL connect address used for engine -> frontend outputs.
    pub output_address: String,
    /// Expected engine DEALER identity used for ready-handshake validation and routing.
    pub engine_identity: Vec<u8>,
    /// Timeout while waiting for the engine ready message.
    pub ready_timeout: Duration,
    /// Frontend client index stamped onto every request.
    pub client_index: u32,
}

impl ZmqEngineCoreClientConfig {
    pub fn new(
        input_address: impl Into<String>,
        output_address: impl Into<String>,
        engine_identity: Vec<u8>,
    ) -> Self {
        Self {
            input_address: input_address.into(),
            output_address: output_address.into(),
            engine_identity,
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
