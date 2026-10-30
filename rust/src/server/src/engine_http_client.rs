// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::future::BoxFuture;
use reqwest::{Client, RequestBuilder, Response};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use thiserror_ext::AsReport as _;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use url::Url;
use vllm_engine_core_client::protocol::OpaqueValue;
use vllm_engine_core_client::protocol::dtype::ModelDtype;
use vllm_engine_core_client::protocol::handshake::{EngineCoreReadyResponse, KvEventsConfig};
use vllm_engine_core_client::protocol::lora::LoraRequest;
use vllm_engine_core_client::protocol::request::EngineCoreRequest;
use vllm_engine_core_client::protocol::utility::{CheckpointState, PauseMode, SleepLevel};
use vllm_engine_core_client::{EngineClient, EngineOutputStream, Error, Result};

mod stream;
#[cfg(test)]
pub(crate) mod tests;

/// An HTTP generation backend. No engine-core transport is created in this mode.
pub(crate) struct EngineHTTPClient {
    client: Client,
    base: Url,
    model: String,
    ready: EngineCoreReadyResponse,
    max_model_len: u32,
    active: Arc<Mutex<HashMap<String, CancellationToken>>>,
    loras: Mutex<HashMap<u64, String>>,
    health: watch::Sender<bool>,
    health_task: tokio::task::JoinHandle<()>,
}

pub(crate) fn generate_url(base: &str) -> anyhow::Result<Url> {
    let mut url = Url::parse(base)?;
    anyhow::ensure!(
        matches!(url.scheme(), "http" | "https") && url.host_str().is_some(),
        "generation URL must use HTTP or HTTPS"
    );
    anyhow::ensure!(
        url.query().is_none() && url.fragment().is_none(),
        "generation URL must not contain a query or fragment"
    );
    url.set_path(&format!("{}/", url.path().trim_end_matches('/')));
    Ok(url.join("inference/v1/generate")?)
}

impl EngineHTTPClient {
    pub(crate) async fn connect(
        base: &str,
        model: String,
        max_model_len: u32,
    ) -> anyhow::Result<Self> {
        let mut base = generate_url(base)?;
        let path = base.path().strip_suffix("inference/v1/generate").unwrap().to_owned();
        base.set_path(&path);
        let client = Client::builder()
            // Neither retries nor redirects may submit generation twice.
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .connect_timeout(Duration::from_secs(30))
            .build()?;
        let info: Value = decode(
            send(client.get(base.join("server_info")?).query(&[("config_format", "json")])).await?,
        )
        .await?;
        let config = info
            .get("vllm_config")
            .ok_or_else(|| remote_error(502, "server_info omitted vllm_config"))?;
        let version: Value = decode(send(client.get(base.join("version")?)).await?).await?;
        let sources: BTreeMap<u32, KvEventsConfig> =
            decode(send(client.get(base.join("kv_event_sources")?)).await?).await?;
        // /server_info cannot recover individual ready responses when this
        // frontend manages several DP engines, including engines on other nodes.
        anyhow::ensure!(
            field::<u64>(config, "/parallel_config/data_parallel_size_local")? == 1
                && (field::<u64>(config, "/parallel_config/data_parallel_size")? == 1
                    || field::<bool>(config, "/parallel_config/data_parallel_external_lb")?
                    || field::<bool>(config, "/parallel_config/data_parallel_hybrid_lb")?)
                && sources.len() <= 1,
            "HTTP discovery requires one managed DP engine per backend; /server_info does not expose per-engine metadata"
        );
        let rank: u32 = field(config, "/parallel_config/data_parallel_rank")?;
        anyhow::ensure!(
            sources.is_empty() || sources.contains_key(&rank),
            "KV-event source rank does not match the discovered DP engine"
        );
        let dtype: String = field(config, "/model_config/dtype")?;
        let lora = config.get("lora_config").filter(|v| !v.is_null());
        let weight_transfer_backend: Option<String> = config
            .pointer("/weight_transfer_config/backend")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let supports_draft_weight_updates = if weight_transfer_backend.is_some() {
            let result: Value = decode(
                send(
                    client
                        .post(base.join("collective_rpc")?)
                        .json(&json!({"method": "supports_draft_weight_updates"})),
                )
                .await?,
            )
            .await?;
            let results: Vec<Value> = field(&result, "/results")?;
            anyhow::ensure!(
                !results.is_empty(),
                "draft capability RPC returned no worker results"
            );
            results
                .iter()
                .map(|value| match value {
                    Value::Bool(value) => Ok(*value),
                    Value::String(value) if value == "True" => Ok(true),
                    Value::String(value) if value == "False" => Ok(false),
                    _ => Err(remote_error(502, "invalid draft capability result")),
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .all(|value| value)
        } else {
            false
        };
        let ready = EngineCoreReadyResponse {
            max_model_len: field(config, "/model_config/max_model_len")?,
            num_gpu_blocks: field(config, "/cache_config/num_gpu_blocks")?,
            block_size: field(config, "/cache_config/block_size")?,
            dp_stats_address: None,
            dtype: serde_json::from_value(json!(dtype.strip_prefix("torch.").unwrap_or(&dtype)))?,
            vllm_version: field(&version, "/version")?,
            world_size: field(config, "/parallel_config/world_size")?,
            effective_data_parallel_size: field(config, "/parallel_config/data_parallel_size")?,
            tensor_parallel_size: field(config, "/parallel_config/tensor_parallel_size")?,
            pipeline_parallel_size: field(config, "/parallel_config/pipeline_parallel_size")?,
            decode_context_parallel_size: field(
                config,
                "/parallel_config/decode_context_parallel_size",
            )?,
            data_parallel_rank: rank,
            max_num_seqs: field(config, "/scheduler_config/max_num_seqs")?,
            max_num_batched_tokens: field(config, "/scheduler_config/max_num_batched_tokens")?,
            instance_id: field(config, "/instance_id")?,
            supports_lora: lora.is_some(),
            max_loras: lora.map(|value| field(value, "/max_loras")).transpose()?.unwrap_or(0),
            kv_cache_size_tokens: field(config, "/cache_config/kv_cache_size_tokens")?,
            kv_cache_max_concurrency: field(config, "/cache_config/kv_cache_max_concurrency")?,
            kv_events_config: sources.get(&rank).cloned(),
            weight_transfer_backend,
            enable_sleep_mode: field(config, "/model_config/enable_sleep_mode")?,
            supports_draft_weight_updates,
            effective_attention_block_size: field(
                config,
                "/cache_config/effective_attention_block_size",
            )?,
            vllm_config: Some(config.clone()),
        };
        anyhow::ensure!(
            max_model_len > 0 && u64::from(max_model_len) <= ready.max_model_len,
            "--max-model-len exceeds the downstream context limit ({})",
            ready.max_model_len
        );
        send(client.get(base.join("health")?).timeout(Duration::from_secs(5))).await?;
        let (health, _) = watch::channel(true);
        let health_task = tokio::spawn({
            let client = client.clone();
            let url = base.join("health")?;
            let health = health.clone();
            async move {
                let mut interval = tokio::time::interval(Duration::from_secs(5));
                loop {
                    interval.tick().await;
                    let healthy = client
                        .get(url.clone())
                        .timeout(Duration::from_secs(5))
                        .send()
                        .await
                        .is_ok_and(|response| response.status().is_success());
                    health.send_if_modified(|value| {
                        let changed = *value != healthy;
                        *value = healthy;
                        changed
                    });
                }
            }
        });
        Ok(Self {
            client,
            base,
            model,
            ready,
            max_model_len,
            active: Default::default(),
            loras: Default::default(),
            health,
            health_task,
        })
    }

    fn get(&self, path: &str) -> RequestBuilder {
        self.client.get(self.base.join(path).expect("constant endpoint path"))
    }

    fn post(&self, path: &str) -> RequestBuilder {
        self.client.post(self.base.join(path).expect("constant endpoint path"))
    }
}

impl Drop for EngineHTTPClient {
    fn drop(&mut self) {
        self.health_task.abort();
        for token in self.active.lock().expect("active requests mutex").values() {
            token.cancel();
        }
    }
}

fn remote_error(status: u16, message: impl Into<String>) -> Error {
    Error::Http {
        status,
        message: message.into(),
    }
}

fn transport_error(error: reqwest::Error) -> Error {
    remote_error(502, error.to_report_string())
}

async fn send(request: RequestBuilder) -> Result<Response> {
    let response = request.send().await.map_err(transport_error)?;
    if response.status().is_success() {
        Ok(response)
    } else {
        let status = response.status().as_u16();
        Err(remote_error(
            status,
            response.text().await.map_err(transport_error)?,
        ))
    }
}

async fn decode<T: DeserializeOwned>(response: Response) -> Result<T> {
    response.json().await.map_err(transport_error)
}

fn field<T: DeserializeOwned>(value: &Value, pointer: &str) -> Result<T> {
    let value = value
        .pointer(pointer)
        .ok_or_else(|| remote_error(502, format!("HTTP response omitted {pointer}")))?;
    serde_json::from_value(value.clone())
        .map_err(|error| remote_error(502, error.to_report_string()))
}

impl EngineClient for EngineHTTPClient {
    fn call(
        &self,
        request: EngineCoreRequest,
    ) -> BoxFuture<'_, Result<Box<dyn EngineOutputStream>>> {
        Box::pin(stream::submit(self, request))
    }

    fn abort<'a>(&'a self, ids: &'a [String]) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let active = self.active.lock().expect("active requests mutex");
            for id in ids {
                if let Some(cancel) = active.get(id) {
                    cancel.cancel();
                }
            }
            Ok(())
        })
    }

    fn shutdown(self: Box<Self>) -> BoxFuture<'static, Result<()>> {
        Box::pin(async move {
            drop(self);
            Ok(())
        })
    }

    fn model_name(&self) -> &str {
        &self.model
    }
    // A gateway does not expose the selected engine's index in token responses.
    // Account for requests under this HTTP backend, with engine stats disabled.
    fn engine_indices(&self) -> Vec<u32> {
        vec![0]
    }
    fn engine_stats_enabled(&self) -> bool {
        false
    }
    fn max_model_len(&self) -> u32 {
        self.max_model_len
    }
    fn model_dtype(&self) -> ModelDtype {
        self.ready.dtype
    }
    fn is_healthy(&self) -> bool {
        *self.health.borrow()
    }
    fn supports_lora(&self) -> bool {
        self.ready.supports_lora
    }
    fn ready_response(&self) -> &EngineCoreReadyResponse {
        &self.ready
    }
    fn ready_responses(&self) -> Vec<&EngineCoreReadyResponse> {
        vec![&self.ready]
    }
    fn data_parallel_size(&self) -> usize {
        self.ready.effective_data_parallel_size as usize
    }
    fn engine_count(&self) -> usize {
        1
    }
    fn world_size(&self) -> u64 {
        self.ready.world_size
    }
    fn vllm_version(&self) -> &str {
        &self.ready.vllm_version
    }
    fn total_num_gpu_blocks(&self) -> u64 {
        self.ready.num_gpu_blocks
    }
    fn effective_attention_block_size(&self) -> Option<u64> {
        self.ready.effective_attention_block_size
    }
    fn subscribe_health(&self) -> watch::Receiver<bool> {
        self.health.subscribe()
    }

    fn collective_rpc<'a>(
        &'a self,
        method: &'a str,
        timeout: Option<f64>,
        args: Vec<Value>,
        kwargs: BTreeMap<String, Value>,
    ) -> BoxFuture<'a, Result<Vec<OpaqueValue>>> {
        Box::pin(async move {
            let response = send(self.post("collective_rpc").json(
                &json!({"method": method, "timeout": timeout, "args": args, "kwargs": kwargs}),
            ))
            .await?;
            let body = response.bytes().await.map_err(transport_error)?;
            if body.is_empty() {
                return Ok(Vec::new());
            }
            let value: Value = serde_json::from_slice(&body)
                .map_err(|error| remote_error(502, error.to_report_string()))?;
            let values: Vec<Value> = field(&value, "/results")?;
            values
                .into_iter()
                .map(|value| {
                    rmpv::ext::to_value(value)
                        .map_err(|error| remote_error(502, error.to_report_string()))
                })
                .collect()
        })
    }
    fn shutdown_weight_transfer_engine<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            send(self.post("shutdown_weight_transfer_engine")).await?;
            Ok(())
        })
    }
    fn start_weight_update<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            send(self.post("start_weight_update")).await?;
            Ok(())
        })
    }
    fn start_draft_weight_update<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            send(self.post("start_draft_weight_update")).await?;
            Ok(())
        })
    }
    fn finish_weight_update<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            send(self.post("finish_weight_update")).await?;
            Ok(())
        })
    }
    fn reset_mm_cache<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            send(self.post("reset_mm_cache")).await?;
            Ok(())
        })
    }
    fn reset_encoder_cache<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            send(self.post("reset_encoder_cache")).await?;
            Ok(())
        })
    }
    fn release_kv_cache_memory<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            send(self.post("release_kv_cache_memory")).await?;
            Ok(())
        })
    }
    fn init_weight_transfer_engine<'a>(&'a self, init_info: Value) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            send(self.post("init_weight_transfer_engine").json(&json!({"init_info": init_info})))
                .await?;
            Ok(())
        })
    }
    fn update_weights<'a>(&'a self, update_info: Value) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            send(self.post("update_weights").json(&json!({"update_info": update_info}))).await?;
            Ok(())
        })
    }
    fn get_weight_version<'a>(&'a self) -> BoxFuture<'a, Result<String>> {
        Box::pin(async move {
            let value: Value = decode(send(self.get("weight_info")).await?).await?;
            field(&value, "/weight_version")
        })
    }
    fn is_sleeping<'a>(&'a self) -> BoxFuture<'a, Result<bool>> {
        Box::pin(async move {
            let value: Value = decode(send(self.get("is_sleeping")).await?).await?;
            field(&value, "/is_sleeping")
        })
    }
    fn is_scheduler_paused<'a>(&'a self) -> BoxFuture<'a, Result<bool>> {
        Box::pin(async move {
            let value: Value = decode(send(self.get("is_paused")).await?).await?;
            field(&value, "/is_paused")
        })
    }
    fn set_weight_version<'a>(&'a self, weight_version: &'a str) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            send(self.post("update_weight_version").json(&json!({"new_version": weight_version})))
                .await?;
            Ok(())
        })
    }
    fn reset_prefix_cache<'a>(
        &'a self,
        reset_running_requests: bool,
        reset_connector: bool,
    ) -> BoxFuture<'a, Result<bool>> {
        Box::pin(async move {
            let value: Value = decode(
                send(self.post("reset_prefix_cache").query(&[
                    ("reset_running_requests", reset_running_requests),
                    ("reset_external", reset_connector),
                ]))
                .await?,
            )
            .await?;
            field(&value, "/success")
        })
    }
    fn add_lora<'a>(&'a self, request: &'a LoraRequest) -> BoxFuture<'a, Result<bool>> {
        Box::pin(async move {
            send(self.post("v1/load_lora_adapter").json(&json!({"lora_name": request.lora_name, "lora_path": request.lora_path, "load_inplace": request.load_inplace, "is_3d_lora_weight": request.is_3d_lora_weight}))).await?;
            self.loras
                .lock()
                .expect("LoRA mutex")
                .insert(request.lora_int_id, request.lora_name.clone());
            Ok(true)
        })
    }
    fn remove_lora<'a>(&'a self, lora_id: u64) -> BoxFuture<'a, Result<bool>> {
        Box::pin(async move {
            let name = self.loras.lock().expect("LoRA mutex").get(&lora_id).cloned();
            let Some(name) = name else {
                return Ok(false);
            };
            // The remote frontend assigns its own integer IDs. Use the shared name.
            send(self.post("v1/unload_lora_adapter").json(&json!({"lora_name": name}))).await?;
            self.loras.lock().expect("LoRA mutex").remove(&lora_id);
            Ok(true)
        })
    }
    fn sleep<'a>(&'a self, level: SleepLevel, mode: PauseMode) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            send(self.post("sleep").query(&json!({"level": u32::from(level), "mode": mode})))
                .await?;
            Ok(())
        })
    }
    fn wake_up<'a>(&'a self, tags: Option<Vec<String>>) -> BoxFuture<'a, Result<bool>> {
        Box::pin(async move {
            let tags: Vec<_> =
                tags.unwrap_or_default().into_iter().map(|tag| ("tags", tag)).collect();
            let value: Value = decode(send(self.post("wake_up").query(&tags)).await?).await?;
            match field::<String>(&value, "/status")?.as_str() {
                "awake" => Ok(true),
                "sleeping" => Ok(false),
                _ => Err(remote_error(502, "invalid wake_up status")),
            }
        })
    }
    fn pause_generation<'a>(
        &'a self,
        mode: PauseMode,
        clear_cache: bool,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            send(self.post("pause").query(&json!({"mode": mode, "clear_cache": clear_cache})))
                .await?;
            Ok(())
        })
    }
    fn resume_scheduler<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            send(self.post("resume")).await?;
            Ok(())
        })
    }
    fn checkpoint_prepare<'a>(&'a self, mode: PauseMode) -> BoxFuture<'a, Result<Option<String>>> {
        Box::pin(async move {
            match send(self.post("checkpoint/prepare").query(&[("mode", mode)])).await {
                Ok(_) => Ok(None),
                Err(Error::Http {
                    status: 400,
                    message,
                }) => {
                    let value: Value = serde_json::from_str(&message)
                        .map_err(|error| remote_error(502, error.to_report_string()))?;
                    Ok(Some(
                        field::<Option<String>>(&value, "/error")?.unwrap_or(message),
                    ))
                }
                Err(error) => Err(error),
            }
        })
    }
    fn checkpoint_restore<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            send(self.post("checkpoint/restore")).await?;
            Ok(())
        })
    }
    fn checkpoint_state<'a>(&'a self) -> BoxFuture<'a, Result<(CheckpointState, Option<String>)>> {
        Box::pin(async move {
            let response = self.get("checkpoint/status").send().await.map_err(transport_error)?;
            let status = response.status().as_u16();
            let body = response.text().await.map_err(transport_error)?;
            if !matches!(status, 200 | 500 | 503) {
                return Err(remote_error(status, body));
            }
            let value: Value = serde_json::from_str(&body)
                .map_err(|error| remote_error(502, error.to_report_string()))?;
            Ok((
                field(&value, "/state")?,
                value.get("error").and_then(Value::as_str).map(str::to_owned),
            ))
        })
    }
    fn start_profile<'a>(
        &'a self,
        profile_prefix: Option<&'a str>,
        delay_iterations: Option<u64>,
        max_iterations: Option<u64>,
        profiler_kwargs: Option<&'a serde_json::Map<String, Value>>,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let body = json!({"profile_prefix": profile_prefix, "delay_iterations": delay_iterations, "max_iterations": max_iterations, "profiler_kwargs": profiler_kwargs});
            match send(self.post("start_profile").json(&body)).await {
                Err(Error::Http { status: 409, .. }) => Err(Error::ProfileAlreadyActive),
                result => {
                    result?;
                    Ok(())
                }
            }
        })
    }
    fn stop_profile<'a>(&'a self, profile_prefix: Option<&'a str>) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            if profile_prefix.is_some() {
                return Err(remote_error(
                    400,
                    "HTTP stop_profile does not accept a profile prefix",
                ));
            }
            send(self.post("stop_profile")).await?;
            Ok(())
        })
    }
}
