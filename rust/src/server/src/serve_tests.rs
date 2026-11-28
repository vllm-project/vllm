// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! End-to-end tests of a [`Server`] over a caller-built `Llm`
//! ([`LlmSource::Provided`]), serving HTTP on a Unix socket.

use std::collections::HashMap;
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
use tokio::net::UnixStream;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use vllm_chat::{ChatTemplateContentFormatOption, ParserSelection, RendererSelection};
use vllm_engine_core_client::protocol::structured_outputs::StructuredOutputBackend;
use vllm_llm::{BoxGenerateOutputStream, GenerateRequest, Llm, LlmFingerprint, LlmInfo};

use crate::config::{
    AccessLogConfig, ApiServerOptions, CorsConfig, HttpListenerMode, ServerConfig,
};
use crate::{LlmSource, Server};

/// An `Llm` that cannot take requests, and records whether it was shut down.
struct TestLlm {
    info: LlmInfo,
    health: watch::Sender<bool>,
    readiness: watch::Sender<bool>,
    shut_down: Arc<AtomicBool>,
}

impl TestLlm {
    /// The `Llm`, and a flag that turns `true` once it is shut down.
    fn tracked() -> (Box<dyn Llm>, Arc<AtomicBool>) {
        let shut_down = Arc::new(AtomicBool::new(false));
        let llm = Self {
            info: LlmInfo {
                max_model_len: 4096,
                model_dtype: None,
                supports_lora: false,
                allow_missing_mm_embeddings: false,
                vllm_version: "test".to_string(),
                fingerprint: LlmFingerprint {
                    vllm_version: "test".to_string(),
                    config_hash: None,
                    tensor_parallel_size: 1,
                    pipeline_parallel_size: 1,
                    data_parallel_size: 1,
                    enable_expert_parallel: false,
                },
            },
            health: watch::channel(true).0,
            readiness: watch::channel(true).0,
            shut_down: shut_down.clone(),
        };
        (Box::new(llm), shut_down)
    }
}

#[async_trait::async_trait]
impl Llm for TestLlm {
    fn info(&self) -> &LlmInfo {
        &self.info
    }

    fn subscribe_health(&self) -> watch::Receiver<bool> {
        self.health.subscribe()
    }

    fn subscribe_readiness(&self) -> watch::Receiver<bool> {
        self.readiness.subscribe()
    }

    async fn check_health_generate(
        &self,
        _timeout: std::time::Duration,
    ) -> Result<(), vllm_llm::NotReady> {
        Ok(())
    }

    async fn generate_n(
        &self,
        _request: GenerateRequest,
        _n: NonZeroU32,
    ) -> vllm_llm::Result<Vec<BoxGenerateOutputStream>> {
        Err(vllm_llm::Error::Unavailable {
            message: "no engine can take the request".to_string(),
        })
    }

    async fn abort(&self, _external_ids: &[String]) -> vllm_llm::Result<()> {
        Ok(())
    }

    async fn shutdown(self: Box<Self>) -> vllm_llm::Result<()> {
        self.shut_down.store(true, Ordering::SeqCst);
        Ok(())
    }
}

/// A config that serves the model in `model_dir`, which only needs a
/// `config.json` since no tokenizer is loaded, on the Unix socket `socket`,
/// with dev mode on.
fn server_config(model_dir: &Path, socket: &Path) -> ServerConfig {
    ServerConfig {
        model: model_dir.to_str().expect("utf-8 path").to_string(),
        revision: None,
        tokenizer: None,
        hf_config: None,
        skip_tokenizer_init: true,
        hf_token: None,
        hf_overrides: Default::default(),
        generation_config: Default::default(),
        override_generation_config: Default::default(),
        served_model_name: vec!["test-model".to_string()],
        listener_mode: HttpListenerMode::BindUnix {
            path: socket.to_str().expect("utf-8 path").to_string(),
        },
        grpc_listener_mode: None,
        tool_call_parser: ParserSelection::None,
        reasoning_parser: ParserSelection::None,
        tool_strict_level: None,
        structured_outputs_backend: StructuredOutputBackend::default(),
        renderer: RendererSelection::Auto,
        language_model_only: false,
        chat_template: None,
        decisions_read_strategy: None,
        default_chat_template_kwargs: None,
        reasoning_effort_map: HashMap::new(),
        trust_request_chat_template: false,
        exclude_tools_when_tool_choice_none: false,
        limit_mm_per_prompt: HashMap::new(),
        mm_processor_kwargs: HashMap::new(),
        chat_template_content_format: ChatTemplateContentFormatOption::Auto,
        max_logprobs: None,
        api_server_options: ApiServerOptions {
            server_dev_mode: true,
            ..ApiServerOptions::default()
        },
        fingerprint_mode: Default::default(),
        cors: CorsConfig::default(),
        access_log: AccessLogConfig::default(),
        tls: None,
        root_path: None,
        api_keys: Vec::new(),
        shutdown_timeout: Duration::from_secs(5),
        keep_alive_on_engine_death: false,
        keep_alive_timeout: Duration::from_secs(5),
    }
}

/// Send one HTTP/1.1 request over the Unix socket at `socket`, retrying the
/// connection until the server listens, and return the status code.
async fn status_over_unix(socket: &Path, method: &str, uri: &str, body: &str) -> u16 {
    let mut stream = None;
    for _ in 0..500 {
        if let Ok(connected) = UnixStream::connect(socket).await {
            stream = Some(connected);
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let mut stream = stream.expect("server should listen on its socket");
    let request = format!(
        "{method} {uri} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(request.as_bytes()).await.expect("send request");
    let mut response = String::new();
    stream.read_to_string(&mut response).await.expect("read response");
    let status = response.split_whitespace().nth(1).expect("status line");
    status.parse().expect("numeric status")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn provided_llm_is_served_and_shut_down() {
    let model_dir = tempfile::tempdir().expect("create temp dir");
    std::fs::write(
        model_dir.path().join("config.json"),
        r#"{"model_type":"qwen3","vocab_size":128}"#,
    )
    .expect("write config.json");
    let socket = model_dir.path().join("server.sock");
    let config = server_config(model_dir.path(), &socket);
    let (llm, shut_down) = TestLlm::tracked();
    let shutdown = CancellationToken::new();
    let server =
        tokio::spawn(Server::new(config, LlmSource::Provided(llm)).serve(shutdown.clone()));

    assert_eq!(status_over_unix(&socket, "GET", "/health", "").await, 200);
    // Generation goes through the provided `Llm`, whose unavailability maps to
    // 503.
    let completion = r#"{"model":"test-model","prompt":[1,2,3],"max_tokens":1}"#;
    assert_eq!(
        status_over_unix(&socket, "POST", "/v1/completions", completion).await,
        503
    );
    // Routes that drive the engine-core client are not served, even in dev
    // mode.
    assert_eq!(
        status_over_unix(&socket, "POST", "/reset_prefix_cache", "").await,
        404
    );
    assert!(!shut_down.load(Ordering::SeqCst));

    shutdown.cancel();
    server.await.expect("join server").expect("serve");
    assert!(shut_down.load(Ordering::SeqCst));
}
