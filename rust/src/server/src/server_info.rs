// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;

use serde_json::{Value, json};
use vllm_engine_core_client::EngineCoreClient;

use crate::config::Config;

const SENSITIVE_VLLM_ENV_PATTERNS: &[&str] =
    &["KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL", "AUTH"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ServerInfoConfigFormat {
    Text,
    Json,
}

/// Snapshot returned by `/server_info`.
#[derive(Debug, Clone)]
pub(crate) struct ServerInfoSnapshot {
    /// The engine's `VllmConfig` as seen from this frontend, or `None` if the
    /// engine reports none (it was started without `VLLM_SERVER_DEV_MODE`).
    vllm_config: Option<Value>,
    /// The Rust frontend's own `Config`, which has no Python counterpart.
    rust_frontend_config: Value,
    vllm_env: BTreeMap<String, String>,
    system_env: BTreeMap<String, String>,
}

impl ServerInfoSnapshot {
    pub(crate) fn new(config: &Config, client: &EngineCoreClient) -> Self {
        Self {
            vllm_config: engine_vllm_config(client),
            rust_frontend_config: serde_json::to_value(config)
                .expect("server info value must serialize"),
            vllm_env: collect_vllm_env(),
            system_env: collect_system_env(),
        }
    }

    pub(crate) fn response(&self, config_format: ServerInfoConfigFormat) -> Value {
        let render = |config: &Value| match config_format {
            ServerInfoConfigFormat::Text => Value::String(render_config_text(config)),
            ServerInfoConfigFormat::Json => config.clone(),
        };

        json!({
            "vllm_config": self.vllm_config.as_ref().map(render),
            "rust_frontend_config": render(&self.rust_frontend_config),
            "vllm_env": self.vllm_env.clone(),
            "system_env": self.system_env.clone(),
        })
    }
}

/// The `VllmConfig` reported by the first engine, as seen from this frontend.
fn engine_vllm_config(client: &EngineCoreClient) -> Option<Value> {
    let engine_config = client.ready_response().vllm_config.clone()?;
    Some(frontend_vllm_config(engine_config, client))
}

/// Replace the fields of one engine's `VllmConfig` that Python's `/server_info`
/// reports from the API server's view: values it syncs from every engine's
/// ready response, and its own API process fields.
///
/// An engine reports its own values instead: `data_parallel_size` 1 for an
/// independent dense DP rank, its own `num_gpu_blocks`, and the launcher's API
/// process fields. The DP size and rank match the gRPC `GetServerInfo`
/// parallelism view, and this frontend is the only API process.
fn frontend_vllm_config(mut vllm_config: Value, client: &EngineCoreClient) -> Value {
    let overrides = [
        ("/parallel_config/_api_process_count", json!(1)),
        ("/parallel_config/_api_process_rank", json!(0)),
        (
            "/parallel_config/data_parallel_size",
            json!(client.data_parallel_size()),
        ),
        (
            "/parallel_config/data_parallel_rank",
            json!(client.ready_response().data_parallel_rank),
        ),
        (
            "/cache_config/num_gpu_blocks",
            json!(client.total_num_gpu_blocks()),
        ),
        (
            "/cache_config/effective_attention_block_size",
            json!(client.effective_attention_block_size()),
        ),
        ("/model_config/max_model_len", json!(client.max_model_len())),
    ];
    for (pointer, value) in overrides {
        if let Some(field) = vllm_config.pointer_mut(pointer) {
            *field = value;
        }
    }
    vllm_config
}

fn render_config_text(config: &Value) -> String {
    match config {
        Value::Object(fields) => fields
            .iter()
            .map(|(key, value)| format!("{key}={}", render_config_text_value(value)))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => render_config_text_value(config),
    }
}

fn render_config_text_value(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::String(value) => value.clone(),
        _ => value.to_string(),
    }
}

fn collect_vllm_env() -> BTreeMap<String, String> {
    std::env::vars().filter(|(key, _)| is_public_vllm_env_key(key)).collect()
}

fn is_public_vllm_env_key(key: &str) -> bool {
    let key = key.to_ascii_uppercase();
    key.starts_with("VLLM_")
        && !SENSITIVE_VLLM_ENV_PATTERNS.iter().any(|pattern| key.contains(pattern))
}

fn collect_system_env() -> BTreeMap<String, String> {
    BTreeMap::from([
        ("arch".to_string(), std::env::consts::ARCH.to_string()),
        ("family".to_string(), std::env::consts::FAMILY.to_string()),
        ("os".to_string(), std::env::consts::OS.to_string()),
    ])
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::os::fd::IntoRawFd;
    use std::os::unix::net::UnixListener;
    use std::time::Duration;

    use serde_json::{Value, json};
    use vllm_engine_core_client::mock_engine::{
        MockEngineConfig, connect_to_bootstrapped_frontend, default_ready_response,
    };
    use vllm_engine_core_client::test_utils::IpcNamespace;
    use vllm_engine_core_client::{
        EngineCoreClient, EngineCoreClientConfig, EngineId, TransportMode,
    };

    use super::{engine_vllm_config, is_public_vllm_env_key, render_config_text};

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn engine_vllm_config_reports_frontend_view() {
        // Two dense DP ranks (4 and 5) of a DP=8 deployment, each reconfigured
        // to DP=1 on the engine side, behind one API server.
        let engine_config = json!({
            "model_config": {"model": "test-model", "max_model_len": 8192},
            "cache_config": {
                "block_size": 16,
                "num_gpu_blocks": 10,
                "effective_attention_block_size": 64,
            },
            "parallel_config": {
                "tensor_parallel_size": 2,
                "data_parallel_size": 1,
                "data_parallel_rank": 0,
                "_api_process_count": 4,
                "_api_process_rank": -1,
            },
        });
        let ipc = IpcNamespace::new().expect("create ipc namespace");
        let input_address = ipc.input_endpoint();
        let output_address = ipc.output_endpoint();
        let listener_fd = |address: &str| {
            let path = address.strip_prefix("ipc://").expect("IPC test endpoint");
            UnixListener::bind(path).expect("bind inherited test listener").into_raw_fd()
        };
        let client_task = tokio::spawn(EngineCoreClient::connect(EngineCoreClientConfig {
            transport_mode: TransportMode::Bootstrapped {
                input_listener_fd: listener_fd(&input_address),
                output_listener_fd: listener_fd(&output_address),
                engine_start_index: 4,
                engine_count: 2,
                data_parallel_size: 8,
                ready_timeout: Duration::from_secs(2),
            },
            coordinator_mode: None,
            model_name: "test-model".to_string(),
            client_index: 0,
            engine_stats_enabled: true,
        }));
        let mut engine_sockets = Vec::new();
        for (rank, max_model_len, num_gpu_blocks, attention_block_size) in
            [(4_u16, 8192, 10, 64), (5, 4096, 20, 128)]
        {
            let mut ready = default_ready_response();
            ready.data_parallel_rank = rank.into();
            ready.max_model_len = max_model_len;
            ready.num_gpu_blocks = num_gpu_blocks;
            ready.effective_attention_block_size = Some(attention_block_size);
            ready.vllm_config = Some(engine_config.clone());
            engine_sockets.push(
                connect_to_bootstrapped_frontend(
                    &input_address,
                    &output_address,
                    EngineId::from_engine_index(rank),
                    MockEngineConfig {
                        ready_response: ready,
                        ..Default::default()
                    },
                )
                .await
                .expect("connect mock engine"),
            );
        }
        let client = client_task.await.unwrap().expect("connect multi-engine client");

        assert_eq!(
            engine_vllm_config(&client).expect("engine reports config"),
            json!({
                "model_config": {"model": "test-model", "max_model_len": 4096},
                "cache_config": {
                    "block_size": 16,
                    "num_gpu_blocks": 30,
                    "effective_attention_block_size": null,
                },
                "parallel_config": {
                    "tensor_parallel_size": 2,
                    "data_parallel_size": 8,
                    "data_parallel_rank": 4,
                    "_api_process_count": 1,
                    "_api_process_rank": 0,
                },
            })
        );

        drop(engine_sockets);
        client.shutdown().await.unwrap();
    }

    #[test]
    fn render_config_text_formats_config_snapshot() {
        let rendered = render_config_text(&json!({
            "model": "test-model",
            "served_model_name": ["served-model"],
            "chat_template": null,
            "enable_log_requests": true,
        }));
        let lines = rendered.lines().collect::<BTreeSet<_>>();

        assert_eq!(
            lines,
            BTreeSet::from([
                "chat_template=None",
                "enable_log_requests=true",
                "model=test-model",
                "served_model_name=[\"served-model\"]",
            ])
        );
        assert_eq!(
            render_config_text(&Value::String("inline".to_string())),
            "inline"
        );
        assert_eq!(render_config_text(&Value::Null), "None");
    }

    #[test]
    fn server_info_env_filter_excludes_sensitive_vllm_keys() {
        for key in [
            "VLLM_API_KEY",
            "VLLM_AUTH_TOKEN",
            "VLLM_SECRET",
            "VLLM_PASSWORD",
            "VLLM_CREDENTIAL_FILE",
            "vllm_token",
        ] {
            assert!(!is_public_vllm_env_key(key), "{key}");
        }
    }

    #[test]
    fn server_info_env_filter_includes_public_vllm_keys() {
        assert!(is_public_vllm_env_key("VLLM_LOGGING_LEVEL"));
        assert!(is_public_vllm_env_key("VLLM_USE_MODELSCOPE"));
        assert!(!is_public_vllm_env_key("OTHER_ENV"));
    }
}
