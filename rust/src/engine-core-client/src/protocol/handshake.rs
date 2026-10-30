// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde::{Deserialize, Serialize};

use crate::protocol::dtype::ModelDtype;

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

/// KV-event publisher configuration reported by EngineCore.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvEventsConfig {
    pub enable_kv_cache_events: bool,
    pub publisher: String,
    pub endpoint: String,
    pub replay_endpoint: Option<String>,
    pub buffer_steps: u32,
    pub hwm: u32,
    pub max_queue_size: u32,
    pub topic: String,
}

/// Post-initialization configuration sent from each engine on the input socket
/// registration message, after the handshake completes.
///
/// Contains values that may differ from the original config (e.g.
/// `max_model_len` after KV cache auto-fitting, `num_gpu_blocks` after
/// profiling).
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/c9340e6f35/vllm/v1/engine/__init__.py#L68-L80>
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineCoreReadyResponse {
    /// Engine-reported maximum model context length (auto-fitted after
    /// KV cache profiling and may differ from the original config value).
    pub max_model_len: u64,
    /// Number of GPU blocks available for KV cache on this engine.
    pub num_gpu_blocks: u64,
    /// KV cache block size (tokens per block).
    pub block_size: u64,
    /// DP coordinator stats publish address, if applicable.
    pub dp_stats_address: Option<String>,
    /// Effective model dtype after Python vLLM resolves `--dtype`.
    pub dtype: ModelDtype,
    /// Python vLLM version reported by the engine process.
    pub vllm_version: String,
    /// Per-DP-engine world size (TP * PP * PCP) from the parallel config.
    /// Deployment-wide world size also multiplies by `data_parallel_size`.
    pub world_size: u64,
    /// Data-parallel size from this EngineCore's effective parallel config.
    /// Dense independent-DP ranks are reconfigured to report `1`; the client
    /// transport owns the deployment-wide data-parallel size.
    #[serde(rename = "data_parallel_size")]
    pub effective_data_parallel_size: u64,
    // Required discovery metadata; EngineCore and client versions must match.
    /// Tensor-parallel size of this engine.
    pub tensor_parallel_size: u32,
    /// Pipeline-parallel size of this engine.
    pub pipeline_parallel_size: u32,
    /// Decode-context-parallel size of this engine.
    pub decode_context_parallel_size: u32,
    /// This engine's data-parallel rank.
    pub data_parallel_rank: u32,
    /// Scheduler cap on concurrently running sequences.
    pub max_num_seqs: u64,
    /// Scheduler cap on batched tokens per step.
    pub max_num_batched_tokens: u64,
    /// Unique identifier for this server instance.
    pub instance_id: String,
    /// Whether the engine was started with LoRA support enabled.
    pub supports_lora: bool,
    /// Maximum number of LoRA adapters the engine may keep active.
    pub max_loras: u32,
    /// Total KV cache capacity in tokens, if reported.
    pub kv_cache_size_tokens: Option<u64>,
    /// Maximum achievable request concurrency given the KV cache, if reported.
    pub kv_cache_max_concurrency: Option<f64>,
    /// KV-event publisher configuration, if configured.
    #[serde(default)]
    pub kv_events_config: Option<KvEventsConfig>,
    /// Configured RL weight-transfer backend, if weight transfer is enabled.
    #[serde(default)]
    pub weight_transfer_backend: Option<String>,
    /// Whether the engine was started with sleep mode enabled.
    #[serde(default)]
    pub enable_sleep_mode: bool,
    /// Whether the engine has a speculative draft model that can be updated.
    #[serde(default)]
    pub supports_draft_weight_updates: bool,
    /// Full-attention block size in tokens after initialization, or unavailable.
    #[serde(default)]
    pub effective_attention_block_size: Option<u64>,
    /// JSON-mode dump of this engine's `VllmConfig`, reported only when the
    /// engine runs with `VLLM_SERVER_DEV_MODE` (for `/server_info`).
    #[serde(default)]
    pub vllm_config: Option<serde_json::Value>,
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
    pub parallel_config: HandshakeParallelConfig,
}

/// `ParallelConfig` fields that the engine overwrites from INIT, encoded as a
/// map.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HandshakeParallelConfig {
    /// Sent for coordinated (MoE) DP only; otherwise the map is empty.
    #[serde(flatten)]
    pub data_parallel_rendezvous: Option<DataParallelRendezvous>,
}

/// Where all DP ranks rendezvous to form their cross-rank torch process
/// groups.
///
/// `_coord_store_port` is not sent: no coordination store runs, so engines
/// take the listed ports.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DataParallelRendezvous {
    pub data_parallel_master_ip: String,
    pub data_parallel_master_port: u16,
    /// Free ports that the engine takes one per process group before falling
    /// back to incrementing `data_parallel_master_port`.
    #[serde(rename = "_data_parallel_master_port_list")]
    pub data_parallel_master_port_list: Vec<u16>,
    pub data_parallel_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{decode_value, encode_msgpack};

    #[test]
    fn handshake_parallel_config_encodes_rendezvous_fields_or_empty_map() {
        let encode = |parallel_config: HandshakeParallelConfig| {
            decode_value(&encode_msgpack(&parallel_config).unwrap()).unwrap().to_string()
        };

        let rendezvous = DataParallelRendezvous {
            data_parallel_master_ip: "10.0.0.1".to_string(),
            data_parallel_master_port: 29500,
            data_parallel_master_port_list: vec![29501, 29502, 29503, 29504],
            data_parallel_size: 4,
        };

        expect_test::expect!["{}"].assert_eq(&encode(HandshakeParallelConfig::default()));
        expect_test::expect![[r#"{"data_parallel_master_ip": "10.0.0.1", "data_parallel_master_port": 29500, "_data_parallel_master_port_list": [29501, 29502, 29503, 29504], "data_parallel_size": 4}"#]]
            .assert_eq(&encode(HandshakeParallelConfig {
                data_parallel_rendezvous: Some(rendezvous),
            }));
    }
}
