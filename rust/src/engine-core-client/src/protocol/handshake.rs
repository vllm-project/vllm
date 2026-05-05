use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::protocol::OpaqueValue;

/// Decoded engine startup-handshake payload sent on the handshake socket.
///
/// Original Python payload construction:
/// <https://github.com/vllm-project/vllm/blob/c8d98f81f6/vllm/v1/engine/core.py#L1000-L1035>
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReadyMessage {
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub local: Option<bool>,
    #[serde(default)]
    pub headless: Option<bool>,
    #[serde(default)]
    pub parallel_config_hash: Option<String>,
}

/// Post-initialization configuration sent from each engine on the input socket
/// registration message, after the handshake completes.
///
/// Contains values that may differ from the original config (e.g.
/// `max_model_len` after KV cache auto-fitting, `num_gpu_blocks` after
/// profiling).
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/c8d98f81f6/vllm/v1/engine/__init__.py#L67-L77>
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineCoreReadyResponse {
    /// Engine-reported maximum model context length (auto-fitted after
    /// KV cache profiling and may differ from the original config value).
    pub max_model_len: u64,
    /// Number of GPU blocks available for KV cache on this engine.
    pub num_gpu_blocks: u64,
    /// DP coordinator stats publish address, if applicable.
    pub dp_stats_address: Option<String>,
}

/// Frontend-owned ZMQ addresses that are sent to the engine during startup
/// handshake initialization.
///
/// Original Python definition (`EngineZmqAddresses`):
/// <https://github.com/vllm-project/vllm/blob/f22d6e026798a74e6542a52ef776c054f2de572a/vllm/v1/engine/utils.py#L53-L67>
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakeAddresses {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub coordinator_input: Option<String>,
    pub coordinator_output: Option<String>,
    pub frontend_stats_publish_address: Option<String>,
}

/// Startup handshake payload sent from the frontend to initialize an engine
/// after receiving `HELLO`.
///
/// Original Python definition (`EngineHandshakeMetadata`):
/// <https://github.com/vllm-project/vllm/blob/f22d6e026798a74e6542a52ef776c054f2de572a/vllm/v1/engine/utils.py#L69-L77>
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakeInitMessage {
    pub addresses: HandshakeAddresses,
    pub parallel_config: BTreeMap<String, OpaqueValue>,
}
