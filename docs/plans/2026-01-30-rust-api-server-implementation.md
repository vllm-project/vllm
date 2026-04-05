# Rust API Server Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use 10x-engineer:executing-plans to implement this plan task-by-task.

**Goal:** Build a Rust API server that exposes OpenAI-compatible `/v1/chat/completions` and communicates with vLLM's Python gRPC server.

**Architecture:** Axum HTTP server receives OpenAI requests, tokenizes with HuggingFace tokenizers, calls Python gRPC server via tonic, detokenizes responses, streams SSE back to client.

**Tech Stack:** Rust, Axum, Tonic, HuggingFace tokenizers, prost

---

## Task 1: Create Rust Workspace Structure

**Files:**
- Create: `rust/Cargo.toml`
- Create: `rust/vllm-api-server/Cargo.toml`
- Create: `rust/vllm-api-server/src/main.rs`
- Create: `rust/proto` (symlink)

**Step 1: Create workspace directory structure**

```bash
mkdir -p rust/vllm-api-server/src
```

**Step 2: Create workspace Cargo.toml**

Create `rust/Cargo.toml`:

```toml
[workspace]
resolver = "2"
members = ["vllm-api-server"]

[workspace.package]
version = "0.1.0"
edition = "2021"
license = "Apache-2.0"
```

**Step 3: Create vllm-api-server Cargo.toml**

Create `rust/vllm-api-server/Cargo.toml`:

```toml
[package]
name = "vllm-api-server"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
# Web framework
axum = { version = "0.7", features = ["macros"] }
tokio = { version = "1", features = ["full"] }
tower = "0.5"
tower-http = { version = "0.6", features = ["cors", "trace"] }

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# gRPC
tonic = "0.12"
prost = "0.13"

# Tokenizer
tokenizers = "0.21"

# Streaming
tokio-stream = "0.1"
async-stream = "0.3"

# CLI & Utilities
clap = { version = "4", features = ["derive"] }
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
uuid = { version = "1", features = ["v4"] }
thiserror = "2"

[build-dependencies]
tonic-build = "0.12"
```

**Step 4: Create minimal main.rs**

Create `rust/vllm-api-server/src/main.rs`:

```rust
fn main() {
    println!("vllm-api-server placeholder");
}
```

**Step 5: Create proto symlink**

```bash
ln -s ../../vllm/grpc rust/proto
```

**Step 6: Verify workspace compiles**

```bash
cd rust && cargo check
```

Expected: Compiles successfully with no errors.

**Step 7: Commit**

```bash
sl commit -m "feat(rust): initialize Rust workspace for API server

- Add workspace Cargo.toml with vllm-api-server member
- Add dependencies: axum, tonic, tokenizers, etc.
- Symlink proto directory for shared .proto files"
```

---

## Task 2: Set Up Proto Compilation

**Files:**
- Create: `rust/vllm-api-server/build.rs`
- Modify: `rust/vllm-api-server/src/main.rs`

**Step 1: Create build.rs for proto compilation**

Create `rust/vllm-api-server/build.rs`:

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(false)  // We only need client
        .build_client(true)
        .out_dir("src/generated")
        .compile_protos(
            &["../proto/vllm_engine.proto"],
            &["../proto"],
        )?;
    Ok(())
}
```

**Step 2: Create generated directory**

```bash
mkdir -p rust/vllm-api-server/src/generated
```

**Step 3: Create generated module**

Create `rust/vllm-api-server/src/generated/mod.rs`:

```rust
#![allow(clippy::all)]
#![allow(warnings)]

pub mod vllm {
    pub mod grpc {
        pub mod engine {
            tonic::include_proto!("vllm.grpc.engine");
        }
    }
}

pub use vllm::grpc::engine::*;
```

**Step 4: Update main.rs to include generated module**

Replace `rust/vllm-api-server/src/main.rs`:

```rust
mod generated;

use generated::vllm_engine_client::VllmEngineClient;

#[tokio::main]
async fn main() {
    println!("vllm-api-server - proto types available");

    // Verify types compile
    let _: Option<generated::GenerateRequest> = None;
    let _: Option<generated::SamplingParams> = None;
}
```

**Step 5: Build and verify proto compilation**

```bash
cd rust && cargo build
```

Expected: Compiles successfully, generates `src/generated/vllm.grpc.engine.rs`.

**Step 6: Commit**

```bash
sl commit -m "feat(rust): add proto compilation with tonic-build

- Add build.rs to compile vllm_engine.proto
- Generate Rust types for gRPC client
- Verify GenerateRequest and SamplingParams types available"
```

---

## Task 3: Create OpenAI Request/Response Types

**Files:**
- Create: `rust/vllm-api-server/src/openai/mod.rs`
- Create: `rust/vllm-api-server/src/openai/types.rs`

**Step 1: Create openai module directory**

```bash
mkdir -p rust/vllm-api-server/src/openai
```

**Step 2: Create types.rs with OpenAI structs**

Create `rust/vllm-api-server/src/openai/types.rs`:

```rust
use serde::{Deserialize, Serialize};

/// OpenAI Chat Completion Request
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<i64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// OpenAI Chat Completion Response (non-streaming)
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// OpenAI Chat Completion Chunk (streaming)
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatChunkChoice {
    pub index: u32,
    pub delta: ChatDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

impl ChatCompletionRequest {
    pub fn is_streaming(&self) -> bool {
        self.stream.unwrap_or(false)
    }
}
```

**Step 3: Create openai mod.rs**

Create `rust/vllm-api-server/src/openai/mod.rs`:

```rust
mod types;

pub use types::*;
```

**Step 4: Update main.rs to verify types**

Replace `rust/vllm-api-server/src/main.rs`:

```rust
mod generated;
mod openai;

use openai::{ChatCompletionRequest, ChatMessage};

#[tokio::main]
async fn main() {
    // Verify OpenAI types work with serde
    let json = r#"{
        "model": "test",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": true
    }"#;

    let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.model, "test");
    assert!(req.is_streaming());

    println!("OpenAI types working correctly");
}
```

**Step 5: Build and run to verify**

```bash
cd rust && cargo run
```

Expected: Prints "OpenAI types working correctly".

**Step 6: Commit**

```bash
sl commit -m "feat(rust): add OpenAI request/response types

- ChatCompletionRequest with all common fields
- ChatCompletionResponse for non-streaming
- ChatCompletionChunk for SSE streaming
- Serde derive for JSON serialization"
```

---

## Task 4: Create gRPC Client Module

**Files:**
- Create: `rust/vllm-api-server/src/grpc/mod.rs`
- Create: `rust/vllm-api-server/src/grpc/client.rs`

**Step 1: Create grpc module directory**

```bash
mkdir -p rust/vllm-api-server/src/grpc
```

**Step 2: Create client.rs with gRPC client wrapper**

Create `rust/vllm-api-server/src/grpc/client.rs`:

```rust
use tonic::transport::Channel;
use crate::generated::{
    vllm_engine_client::VllmEngineClient,
    GenerateRequest, GenerateResponse, SamplingParams,
    TokenizedInput, HealthCheckRequest,
    generate_request::Input,
};
use crate::openai::ChatCompletionRequest;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GrpcError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(#[from] tonic::transport::Error),
    #[error("RPC failed: {0}")]
    RpcFailed(#[from] tonic::Status),
}

#[derive(Clone)]
pub struct VllmClient {
    client: VllmEngineClient<Channel>,
}

impl VllmClient {
    pub async fn connect(addr: &str) -> Result<Self, GrpcError> {
        let client = VllmEngineClient::connect(format!("http://{}", addr)).await?;
        Ok(Self { client })
    }

    pub async fn health_check(&mut self) -> Result<bool, GrpcError> {
        let response = self.client.health_check(HealthCheckRequest {}).await?;
        Ok(response.into_inner().healthy)
    }

    pub fn build_generate_request(
        request_id: String,
        token_ids: Vec<u32>,
        original_text: String,
        req: &ChatCompletionRequest,
        stream: bool,
    ) -> GenerateRequest {
        GenerateRequest {
            request_id,
            input: Some(Input::Tokenized(TokenizedInput {
                original_text,
                input_ids: token_ids,
            })),
            sampling_params: Some(Self::build_sampling_params(req)),
            stream,
        }
    }

    fn build_sampling_params(req: &ChatCompletionRequest) -> SamplingParams {
        SamplingParams {
            temperature: req.temperature,
            top_p: req.top_p.unwrap_or(1.0),
            max_tokens: req.max_tokens,
            frequency_penalty: req.frequency_penalty.unwrap_or(0.0),
            presence_penalty: req.presence_penalty.unwrap_or(0.0),
            seed: req.seed.map(|s| s as i32),
            stop: req.stop.clone().unwrap_or_default(),
            // Defaults for other fields
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            min_tokens: 0,
            stop_token_ids: vec![],
            skip_special_tokens: true,
            spaces_between_special_tokens: true,
            ignore_eos: false,
            n: 1,
            logprobs: None,
            prompt_logprobs: None,
            include_stop_str_in_output: false,
            logit_bias: std::collections::HashMap::new(),
            truncate_prompt_tokens: None,
            constraint: None,
        }
    }

    pub async fn generate(
        &mut self,
        request: GenerateRequest,
    ) -> Result<tonic::Streaming<GenerateResponse>, GrpcError> {
        let response = self.client.generate(request).await?;
        Ok(response.into_inner())
    }
}
```

**Step 3: Create grpc mod.rs**

Create `rust/vllm-api-server/src/grpc/mod.rs`:

```rust
mod client;

pub use client::{VllmClient, GrpcError};
```

**Step 4: Update main.rs to verify compilation**

Replace `rust/vllm-api-server/src/main.rs`:

```rust
mod generated;
mod openai;
mod grpc;

use grpc::VllmClient;

#[tokio::main]
async fn main() {
    println!("Modules compiled successfully");

    // Note: This would fail without a running server, just verifying types compile
    let _ = VllmClient::connect("localhost:50051");
}
```

**Step 5: Build to verify**

```bash
cd rust && cargo build
```

Expected: Compiles successfully.

**Step 6: Commit**

```bash
sl commit -m "feat(rust): add gRPC client wrapper

- VllmClient wraps tonic VllmEngineClient
- build_generate_request converts OpenAI params to proto
- build_sampling_params maps temperature, top_p, etc.
- health_check for connection verification"
```

---

## Task 5: Create Axum Server Skeleton

**Files:**
- Create: `rust/vllm-api-server/src/server.rs`
- Create: `rust/vllm-api-server/src/routes/mod.rs`
- Create: `rust/vllm-api-server/src/routes/health.rs`

**Step 1: Create routes directory**

```bash
mkdir -p rust/vllm-api-server/src/routes
```

**Step 2: Create health.rs with health endpoint**

Create `rust/vllm-api-server/src/routes/health.rs`:

```rust
use axum::{extract::State, http::StatusCode, Json};
use serde::Serialize;
use std::sync::Arc;
use crate::server::AppState;

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
}

pub async fn health_check(
    State(_state): State<Arc<AppState>>,
) -> (StatusCode, Json<HealthResponse>) {
    (
        StatusCode::OK,
        Json(HealthResponse {
            status: "healthy".to_string(),
        }),
    )
}
```

**Step 3: Create routes mod.rs**

Create `rust/vllm-api-server/src/routes/mod.rs`:

```rust
mod health;

pub use health::health_check;
```

**Step 4: Create server.rs with Axum setup**

Create `rust/vllm-api-server/src/server.rs`:

```rust
use axum::{routing::get, Router};
use std::sync::Arc;
use tokio::sync::Mutex;
use tokenizers::Tokenizer;
use tower_http::trace::TraceLayer;

use crate::grpc::VllmClient;
use crate::routes;

pub struct AppState {
    pub tokenizer: Tokenizer,
    pub grpc_client: Mutex<VllmClient>,
    pub model_name: String,
}

impl AppState {
    pub fn new(tokenizer: Tokenizer, grpc_client: VllmClient, model_name: String) -> Self {
        Self {
            tokenizer,
            grpc_client: Mutex::new(grpc_client),
            model_name,
        }
    }
}

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(routes::health_check))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

pub async fn run_server(addr: &str, state: Arc<AppState>) -> Result<(), std::io::Error> {
    let app = create_router(state);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("Server listening on {}", addr);
    axum::serve(listener, app).await
}
```

**Step 5: Update main.rs with CLI and server startup**

Replace `rust/vllm-api-server/src/main.rs`:

```rust
mod generated;
mod grpc;
mod openai;
mod routes;
mod server;

use clap::Parser;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use grpc::VllmClient;
use server::AppState;

#[derive(Parser, Debug)]
#[command(name = "vllm-api-server")]
#[command(about = "Rust API server for vLLM")]
struct Args {
    /// HTTP port to listen on
    #[arg(long, default_value = "8000")]
    port: u16,

    /// gRPC server address
    #[arg(long, default_value = "localhost:50051")]
    grpc_addr: String,

    /// Model name (for tokenizer and response metadata)
    #[arg(long)]
    model: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = Args::parse();

    tracing::info!("Loading tokenizer for model: {}", args.model);
    let tokenizer = Tokenizer::from_pretrained(&args.model, None)
        .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

    tracing::info!("Connecting to gRPC server at {}", args.grpc_addr);
    let grpc_client = VllmClient::connect(&args.grpc_addr).await?;

    let state = Arc::new(AppState::new(tokenizer, grpc_client, args.model));

    let addr = format!("0.0.0.0:{}", args.port);
    server::run_server(&addr, state).await?;

    Ok(())
}
```

**Step 6: Build to verify**

```bash
cd rust && cargo build
```

Expected: Compiles successfully.

**Step 7: Commit**

```bash
sl commit -m "feat(rust): add Axum server skeleton with CLI

- AppState holds tokenizer, gRPC client, model name
- CLI args: --port, --grpc-addr, --model
- /health endpoint for health checks
- TraceLayer for request logging"
```

---

## Task 6: Implement Non-Streaming Chat Completions

**Files:**
- Create: `rust/vllm-api-server/src/routes/chat.rs`
- Modify: `rust/vllm-api-server/src/routes/mod.rs`
- Modify: `rust/vllm-api-server/src/server.rs`

**Step 1: Create chat.rs with chat completions handler**

Create `rust/vllm-api-server/src/routes/chat.rs`:

```rust
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

use crate::generated::generate_response::Response as GrpcResponse;
use crate::grpc::VllmClient;
use crate::openai::{
    ChatChoice, ChatCompletionRequest, ChatCompletionResponse,
    ChatMessage, Usage,
};
use crate::server::AppState;

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    if request.is_streaming() {
        // TODO: Implement streaming in next task
        return (
            StatusCode::NOT_IMPLEMENTED,
            Json(serde_json::json!({"error": "streaming not yet implemented"})),
        )
            .into_response();
    }

    match handle_non_streaming(state, request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(e) => {
            tracing::error!("Chat completion failed: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response()
        }
    }
}

async fn handle_non_streaming(
    state: Arc<AppState>,
    request: ChatCompletionRequest,
) -> Result<ChatCompletionResponse, Box<dyn std::error::Error + Send + Sync>> {
    let request_id = Uuid::new_v4().to_string();

    // Apply chat template and tokenize
    let prompt = apply_chat_template(&request.messages);
    let encoding = state
        .tokenizer
        .encode(prompt.clone(), false)
        .map_err(|e| format!("Tokenization failed: {}", e))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    let prompt_tokens = token_ids.len() as u32;

    tracing::debug!(
        "Request {}: {} prompt tokens",
        request_id,
        prompt_tokens
    );

    // Build gRPC request
    let grpc_request = VllmClient::build_generate_request(
        request_id.clone(),
        token_ids,
        prompt,
        &request,
        false, // non-streaming
    );

    // Call gRPC server
    let mut client = state.grpc_client.lock().await;
    let mut stream = client.generate(grpc_request).await?;

    // Collect response (for non-streaming, we get chunks then a complete message)
    let mut output_tokens: Vec<u32> = Vec::new();
    let mut finish_reason = String::new();
    let mut completion_tokens = 0u32;

    while let Some(response) = stream.message().await? {
        if let Some(grpc_response) = response.response {
            match grpc_response {
                GrpcResponse::Chunk(chunk) => {
                    output_tokens.extend(chunk.token_ids);
                }
                GrpcResponse::Complete(complete) => {
                    if !complete.output_ids.is_empty() {
                        output_tokens = complete.output_ids;
                    }
                    finish_reason = complete.finish_reason;
                    completion_tokens = complete.completion_tokens;
                    break;
                }
            }
        }
    }

    // Detokenize output
    let output_text = state
        .tokenizer
        .decode(&output_tokens, true)
        .map_err(|e| format!("Detokenization failed: {}", e))?;

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    Ok(ChatCompletionResponse {
        id: format!("chatcmpl-{}", request_id),
        object: "chat.completion".to_string(),
        created,
        model: state.model_name.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: output_text,
            },
            finish_reason: Some(finish_reason),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    })
}

fn apply_chat_template(messages: &[ChatMessage]) -> String {
    // Simple chat template - for production, load from tokenizer config
    let mut prompt = String::new();
    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                prompt.push_str(&format!("<|system|>\n{}\n", msg.content));
            }
            "user" => {
                prompt.push_str(&format!("<|user|>\n{}\n", msg.content));
            }
            "assistant" => {
                prompt.push_str(&format!("<|assistant|>\n{}\n", msg.content));
            }
            _ => {
                prompt.push_str(&format!("<|{}|>\n{}\n", msg.role, msg.content));
            }
        }
    }
    prompt.push_str("<|assistant|>\n");
    prompt
}
```

**Step 2: Update routes mod.rs**

Replace `rust/vllm-api-server/src/routes/mod.rs`:

```rust
mod chat;
mod health;

pub use chat::chat_completions;
pub use health::health_check;
```

**Step 3: Update server.rs to add chat route**

Replace `rust/vllm-api-server/src/server.rs`:

```rust
use axum::{routing::{get, post}, Router};
use std::sync::Arc;
use tokio::sync::Mutex;
use tokenizers::Tokenizer;
use tower_http::trace::TraceLayer;

use crate::grpc::VllmClient;
use crate::routes;

pub struct AppState {
    pub tokenizer: Tokenizer,
    pub grpc_client: Mutex<VllmClient>,
    pub model_name: String,
}

impl AppState {
    pub fn new(tokenizer: Tokenizer, grpc_client: VllmClient, model_name: String) -> Self {
        Self {
            tokenizer,
            grpc_client: Mutex::new(grpc_client),
            model_name,
        }
    }
}

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(routes::health_check))
        .route("/v1/chat/completions", post(routes::chat_completions))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

pub async fn run_server(addr: &str, state: Arc<AppState>) -> Result<(), std::io::Error> {
    let app = create_router(state);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("Server listening on {}", addr);
    axum::serve(listener, app).await
}
```

**Step 4: Build to verify**

```bash
cd rust && cargo build
```

Expected: Compiles successfully.

**Step 5: Commit**

```bash
sl commit -m "feat(rust): implement non-streaming chat completions

- /v1/chat/completions endpoint (POST)
- Tokenize messages, call gRPC, detokenize response
- Apply simple chat template
- Return OpenAI-compatible ChatCompletionResponse"
```

---

## Task 7: Implement Streaming Chat Completions (SSE)

**Files:**
- Create: `rust/vllm-api-server/src/openai/streaming.rs`
- Modify: `rust/vllm-api-server/src/openai/mod.rs`
- Modify: `rust/vllm-api-server/src/routes/chat.rs`

**Step 1: Create streaming.rs with SSE helpers**

Create `rust/vllm-api-server/src/openai/streaming.rs`:

```rust
use crate::openai::{ChatCompletionChunk, ChatChunkChoice, ChatDelta};
use std::time::{SystemTime, UNIX_EPOCH};

pub fn create_chunk(
    id: &str,
    model: &str,
    content: Option<String>,
    role: Option<String>,
    finish_reason: Option<String>,
) -> ChatCompletionChunk {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.to_string(),
        choices: vec![ChatChunkChoice {
            index: 0,
            delta: ChatDelta { role, content },
            finish_reason,
        }],
    }
}

pub fn format_sse_data(chunk: &ChatCompletionChunk) -> String {
    format!("data: {}\n\n", serde_json::to_string(chunk).unwrap())
}

pub fn format_sse_done() -> String {
    "data: [DONE]\n\n".to_string()
}
```

**Step 2: Update openai mod.rs**

Replace `rust/vllm-api-server/src/openai/mod.rs`:

```rust
mod streaming;
mod types;

pub use streaming::{create_chunk, format_sse_data, format_sse_done};
pub use types::*;
```

**Step 3: Update chat.rs with streaming support**

Replace `rust/vllm-api-server/src/routes/chat.rs`:

```rust
use axum::{
    body::Body,
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio_stream::StreamExt;
use uuid::Uuid;

use crate::generated::generate_response::Response as GrpcResponse;
use crate::grpc::VllmClient;
use crate::openai::{
    create_chunk, format_sse_data, format_sse_done,
    ChatChoice, ChatCompletionRequest, ChatCompletionResponse,
    ChatMessage, Usage,
};
use crate::server::AppState;

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    if request.is_streaming() {
        match handle_streaming(state, request).await {
            Ok(response) => response,
            Err(e) => {
                tracing::error!("Streaming chat completion failed: {}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": e.to_string()})),
                )
                    .into_response()
            }
        }
    } else {
        match handle_non_streaming(state, request).await {
            Ok(response) => (StatusCode::OK, Json(response)).into_response(),
            Err(e) => {
                tracing::error!("Chat completion failed: {}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": e.to_string()})),
                )
                    .into_response()
            }
        }
    }
}

async fn handle_streaming(
    state: Arc<AppState>,
    request: ChatCompletionRequest,
) -> Result<Response, Box<dyn std::error::Error + Send + Sync>> {
    let request_id = Uuid::new_v4().to_string();
    let chat_id = format!("chatcmpl-{}", request_id);
    let model_name = state.model_name.clone();

    // Apply chat template and tokenize
    let prompt = apply_chat_template(&request.messages);
    let encoding = state
        .tokenizer
        .encode(prompt.clone(), false)
        .map_err(|e| format!("Tokenization failed: {}", e))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    tracing::debug!(
        "Streaming request {}: {} prompt tokens",
        request_id,
        token_ids.len()
    );

    // Build gRPC request
    let grpc_request = VllmClient::build_generate_request(
        request_id,
        token_ids,
        prompt,
        &request,
        true, // streaming
    );

    // Call gRPC server
    let mut client = state.grpc_client.lock().await;
    let grpc_stream = client.generate(grpc_request).await?;

    // Create SSE stream
    let tokenizer = state.tokenizer.clone();
    let stream = async_stream::stream! {
        let mut grpc_stream = grpc_stream;
        let mut first_chunk = true;

        while let Some(result) = grpc_stream.next().await {
            match result {
                Ok(response) => {
                    if let Some(grpc_response) = response.response {
                        match grpc_response {
                            GrpcResponse::Chunk(chunk) => {
                                if !chunk.token_ids.is_empty() {
                                    let text = tokenizer
                                        .decode(&chunk.token_ids, true)
                                        .unwrap_or_default();

                                    let sse_chunk = if first_chunk {
                                        first_chunk = false;
                                        create_chunk(
                                            &chat_id,
                                            &model_name,
                                            Some(text),
                                            Some("assistant".to_string()),
                                            None,
                                        )
                                    } else {
                                        create_chunk(
                                            &chat_id,
                                            &model_name,
                                            Some(text),
                                            None,
                                            None,
                                        )
                                    };
                                    yield Ok::<_, std::io::Error>(format_sse_data(&sse_chunk));
                                }
                            }
                            GrpcResponse::Complete(complete) => {
                                // Send final chunk with finish_reason
                                let final_chunk = create_chunk(
                                    &chat_id,
                                    &model_name,
                                    None,
                                    None,
                                    Some(complete.finish_reason),
                                );
                                yield Ok(format_sse_data(&final_chunk));
                                yield Ok(format_sse_done());
                                break;
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("gRPC stream error: {}", e);
                    break;
                }
            }
        }
    };

    let body = Body::from_stream(stream);

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .body(body)
        .unwrap())
}

async fn handle_non_streaming(
    state: Arc<AppState>,
    request: ChatCompletionRequest,
) -> Result<ChatCompletionResponse, Box<dyn std::error::Error + Send + Sync>> {
    let request_id = Uuid::new_v4().to_string();

    // Apply chat template and tokenize
    let prompt = apply_chat_template(&request.messages);
    let encoding = state
        .tokenizer
        .encode(prompt.clone(), false)
        .map_err(|e| format!("Tokenization failed: {}", e))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    let prompt_tokens = token_ids.len() as u32;

    tracing::debug!(
        "Request {}: {} prompt tokens",
        request_id,
        prompt_tokens
    );

    // Build gRPC request
    let grpc_request = VllmClient::build_generate_request(
        request_id.clone(),
        token_ids,
        prompt,
        &request,
        false,
    );

    // Call gRPC server
    let mut client = state.grpc_client.lock().await;
    let mut stream = client.generate(grpc_request).await?;

    // Collect response
    let mut output_tokens: Vec<u32> = Vec::new();
    let mut finish_reason = String::new();
    let mut completion_tokens = 0u32;

    while let Some(response) = stream.message().await? {
        if let Some(grpc_response) = response.response {
            match grpc_response {
                GrpcResponse::Chunk(chunk) => {
                    output_tokens.extend(chunk.token_ids);
                }
                GrpcResponse::Complete(complete) => {
                    if !complete.output_ids.is_empty() {
                        output_tokens = complete.output_ids;
                    }
                    finish_reason = complete.finish_reason;
                    completion_tokens = complete.completion_tokens;
                    break;
                }
            }
        }
    }

    // Detokenize output
    let output_text = state
        .tokenizer
        .decode(&output_tokens, true)
        .map_err(|e| format!("Detokenization failed: {}", e))?;

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    Ok(ChatCompletionResponse {
        id: format!("chatcmpl-{}", request_id),
        object: "chat.completion".to_string(),
        created,
        model: state.model_name.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: output_text,
            },
            finish_reason: Some(finish_reason),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    })
}

fn apply_chat_template(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                prompt.push_str(&format!("<|system|>\n{}\n", msg.content));
            }
            "user" => {
                prompt.push_str(&format!("<|user|>\n{}\n", msg.content));
            }
            "assistant" => {
                prompt.push_str(&format!("<|assistant|>\n{}\n", msg.content));
            }
            _ => {
                prompt.push_str(&format!("<|{}|>\n{}\n", msg.role, msg.content));
            }
        }
    }
    prompt.push_str("<|assistant|>\n");
    prompt
}
```

**Step 4: Build to verify**

```bash
cd rust && cargo build
```

Expected: Compiles successfully.

**Step 5: Commit**

```bash
sl commit -m "feat(rust): implement streaming chat completions with SSE

- SSE stream for stream: true requests
- Incremental detokenization of token chunks
- Proper OpenAI chunk format with delta
- data: [DONE] termination"
```

---

## Task 8: Add Chat Template from Tokenizer

**Files:**
- Modify: `rust/vllm-api-server/src/routes/chat.rs`
- Modify: `rust/vllm-api-server/src/server.rs`

**Step 1: Update AppState to include chat template**

Replace `rust/vllm-api-server/src/server.rs`:

```rust
use axum::{routing::{get, post}, Router};
use std::sync::Arc;
use tokio::sync::Mutex;
use tokenizers::Tokenizer;
use tower_http::trace::TraceLayer;

use crate::grpc::VllmClient;
use crate::routes;

pub struct AppState {
    pub tokenizer: Tokenizer,
    pub grpc_client: Mutex<VllmClient>,
    pub model_name: String,
    pub chat_template: Option<String>,
}

impl AppState {
    pub fn new(
        tokenizer: Tokenizer,
        grpc_client: VllmClient,
        model_name: String,
        chat_template: Option<String>,
    ) -> Self {
        Self {
            tokenizer,
            grpc_client: Mutex::new(grpc_client),
            model_name,
            chat_template,
        }
    }
}

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(routes::health_check))
        .route("/v1/chat/completions", post(routes::chat_completions))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

pub async fn run_server(addr: &str, state: Arc<AppState>) -> Result<(), std::io::Error> {
    let app = create_router(state);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("Server listening on {}", addr);
    axum::serve(listener, app).await
}
```

**Step 2: Update main.rs to extract chat template**

Replace `rust/vllm-api-server/src/main.rs`:

```rust
mod generated;
mod grpc;
mod openai;
mod routes;
mod server;

use clap::Parser;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use grpc::VllmClient;
use server::AppState;

#[derive(Parser, Debug)]
#[command(name = "vllm-api-server")]
#[command(about = "Rust API server for vLLM")]
struct Args {
    /// HTTP port to listen on
    #[arg(long, default_value = "8000")]
    port: u16,

    /// gRPC server address
    #[arg(long, default_value = "localhost:50051")]
    grpc_addr: String,

    /// Model name (for tokenizer and response metadata)
    #[arg(long)]
    model: String,

    /// Override chat template (Jinja2 format)
    #[arg(long)]
    chat_template: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = Args::parse();

    tracing::info!("Loading tokenizer for model: {}", args.model);
    let tokenizer = Tokenizer::from_pretrained(&args.model, None)
        .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

    // Try to get chat template from tokenizer config
    let chat_template = args.chat_template.or_else(|| {
        // Note: tokenizers crate doesn't directly expose chat_template
        // For now, we use the fallback template
        tracing::warn!("Using fallback chat template - consider providing --chat-template");
        None
    });

    tracing::info!("Connecting to gRPC server at {}", args.grpc_addr);
    let grpc_client = VllmClient::connect(&args.grpc_addr).await?;

    let state = Arc::new(AppState::new(
        tokenizer,
        grpc_client,
        args.model,
        chat_template,
    ));

    let addr = format!("0.0.0.0:{}", args.port);
    server::run_server(&addr, state).await?;

    Ok(())
}
```

**Step 3: Update chat.rs to use chat template from state**

In `rust/vllm-api-server/src/routes/chat.rs`, update the `apply_chat_template` function and its usages:

Find the `apply_chat_template` function at the bottom and replace it with:

```rust
fn apply_chat_template(messages: &[ChatMessage], template: Option<&str>) -> String {
    // If a custom template is provided, use minijinja to render it
    // For now, use the fallback since minijinja adds complexity
    // TODO: Add minijinja for proper Jinja2 template support

    let mut prompt = String::new();
    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                prompt.push_str(&format!("<|system|>\n{}\n", msg.content));
            }
            "user" => {
                prompt.push_str(&format!("<|user|>\n{}\n", msg.content));
            }
            "assistant" => {
                prompt.push_str(&format!("<|assistant|>\n{}\n", msg.content));
            }
            _ => {
                prompt.push_str(&format!("<|{}|>\n{}\n", msg.role, msg.content));
            }
        }
    }
    prompt.push_str("<|assistant|>\n");
    prompt
}
```

Also update the calls to `apply_chat_template` in both `handle_streaming` and `handle_non_streaming`:

Change:
```rust
let prompt = apply_chat_template(&request.messages);
```

To:
```rust
let prompt = apply_chat_template(&request.messages, state.chat_template.as_deref());
```

**Step 4: Build to verify**

```bash
cd rust && cargo build
```

Expected: Compiles successfully.

**Step 5: Commit**

```bash
sl commit -m "feat(rust): add chat template configuration

- AppState now holds optional chat_template
- CLI flag --chat-template for custom templates
- TODO: Add minijinja for full Jinja2 support"
```

---

## Task 9: Build Release Binary

**Files:**
- None (build only)

**Step 1: Build release binary**

```bash
cd rust && cargo build --release
```

Expected: Creates `rust/target/release/vllm-api-server`.

**Step 2: Check binary size**

```bash
ls -lh rust/target/release/vllm-api-server
```

Expected: Binary around 10-30 MB.

**Step 3: Verify binary runs**

```bash
./rust/target/release/vllm-api-server --help
```

Expected: Shows CLI help with --port, --grpc-addr, --model options.

**Step 4: Commit (no changes, just verification)**

No commit needed - this is a build verification step.

---

## Task 10: Create Benchmark Script

**Files:**
- Create: `scripts/benchmark_rust_vs_python.sh`

**Step 1: Create scripts directory if needed**

```bash
mkdir -p scripts
```

**Step 2: Create benchmark script**

Create `scripts/benchmark_rust_vs_python.sh`:

```bash
#!/bin/bash
set -e

# Configuration
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
GRPC_PORT="${GRPC_PORT:-50051}"
HTTP_PORT="${HTTP_PORT:-8000}"
MAX_TIME="${MAX_TIME:-3}"
MAX_REQUESTS="${MAX_REQUESTS:-500}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-./experiments}"

echo "============================================"
echo "vLLM Rust vs Python API Server Benchmark"
echo "============================================"
echo "Model: $MODEL"
echo "gRPC Port: $GRPC_PORT"
echo "HTTP Port: $HTTP_PORT"
echo ""

# Ensure experiment directory exists
mkdir -p "$EXPERIMENT_DIR"

# Function to cleanup background processes
cleanup() {
    echo "Cleaning up..."
    kill $GRPC_PID 2>/dev/null || true
    kill $API_PID 2>/dev/null || true
}
trap cleanup EXIT

# Start Python gRPC backend
echo "[1/4] Starting Python gRPC server..."
python -m vllm.entrypoints.grpc_server \
    --model "$MODEL" \
    --port "$GRPC_PORT" &
GRPC_PID=$!

echo "Waiting for model to load (this may take a while)..."
sleep 60

# Verify gRPC server is healthy
echo "Checking gRPC server health..."
if ! grpcurl -plaintext "localhost:$GRPC_PORT" vllm.grpc.engine.VllmEngine/HealthCheck; then
    echo "ERROR: gRPC server not responding"
    exit 1
fi
echo "gRPC server is healthy"

# Benchmark 1: Python API Server
echo ""
echo "[2/4] Benchmarking Python API Server..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$HTTP_PORT" &
API_PID=$!
sleep 10

genai-bench benchmark \
    --api-backend vllm \
    --api-base "http://localhost:$HTTP_PORT" \
    --task text-to-text \
    --model-tokenizer "$MODEL" \
    --api-model-name "$MODEL" \
    --traffic-scenario "D(100,100)" \
    --traffic-scenario "D(2000,200)" \
    --num-concurrency 1 \
    --num-concurrency 4 \
    --num-concurrency 16 \
    --num-concurrency 64 \
    --max-time-per-run "$MAX_TIME" \
    --max-requests-per-run "$MAX_REQUESTS" \
    --experiment-base-dir "$EXPERIMENT_DIR" \
    --experiment-folder-name "python-baseline"

kill $API_PID
wait $API_PID 2>/dev/null || true
sleep 2

# Benchmark 2: Rust API Server
echo ""
echo "[3/4] Benchmarking Rust API Server..."
./rust/target/release/vllm-api-server \
    --grpc-addr "localhost:$GRPC_PORT" \
    --port "$HTTP_PORT" \
    --model "$MODEL" &
API_PID=$!
sleep 5

genai-bench benchmark \
    --api-backend vllm \
    --api-base "http://localhost:$HTTP_PORT" \
    --task text-to-text \
    --model-tokenizer "$MODEL" \
    --api-model-name "$MODEL" \
    --traffic-scenario "D(100,100)" \
    --traffic-scenario "D(2000,200)" \
    --num-concurrency 1 \
    --num-concurrency 4 \
    --num-concurrency 16 \
    --num-concurrency 64 \
    --max-time-per-run "$MAX_TIME" \
    --max-requests-per-run "$MAX_REQUESTS" \
    --experiment-base-dir "$EXPERIMENT_DIR" \
    --experiment-folder-name "rust-prototype"

kill $API_PID
wait $API_PID 2>/dev/null || true

# Generate comparison report
echo ""
echo "[4/4] Generating comparison report..."
genai-bench excel \
    --experiment-folder "$EXPERIMENT_DIR" \
    --excel-name rust-vs-python-comparison \
    --metric-percentile p90 p99 mean

echo ""
echo "============================================"
echo "Benchmark complete!"
echo "Results saved to: $EXPERIMENT_DIR"
echo "============================================"
```

**Step 3: Make script executable**

```bash
chmod +x scripts/benchmark_rust_vs_python.sh
```

**Step 4: Commit**

```bash
sl commit -m "feat: add benchmark script for Rust vs Python comparison

- Starts shared gRPC backend
- Runs genai-bench against Python API server
- Runs genai-bench against Rust API server
- Generates Excel comparison report"
```

---

## Task 11: Add Integration Test Script

**Files:**
- Create: `scripts/test_rust_server.sh`

**Step 1: Create test script**

Create `scripts/test_rust_server.sh`:

```bash
#!/bin/bash
set -e

# Quick integration test for Rust API server
# Requires: running gRPC server and Rust API server

HTTP_PORT="${HTTP_PORT:-8000}"
BASE_URL="http://localhost:$HTTP_PORT"

echo "Testing Rust API Server at $BASE_URL"

# Test 1: Health check
echo ""
echo "Test 1: Health check"
curl -s "$BASE_URL/health" | jq .
echo "PASS"

# Test 2: Non-streaming chat completion
echo ""
echo "Test 2: Non-streaming chat completion"
curl -s "$BASE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "test",
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 10,
        "stream": false
    }' | jq .
echo "PASS"

# Test 3: Streaming chat completion
echo ""
echo "Test 3: Streaming chat completion"
curl -s -N "$BASE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "test",
        "messages": [{"role": "user", "content": "Count to 3"}],
        "max_tokens": 20,
        "stream": true
    }' | head -10
echo ""
echo "PASS"

echo ""
echo "All tests passed!"
```

**Step 2: Make script executable**

```bash
chmod +x scripts/test_rust_server.sh
```

**Step 3: Commit**

```bash
sl commit -m "feat: add integration test script for Rust API server

- Tests /health endpoint
- Tests non-streaming /v1/chat/completions
- Tests streaming /v1/chat/completions"
```

---

## Summary

| Task | Description | Key Files |
|------|-------------|-----------|
| 1 | Create workspace structure | `rust/Cargo.toml`, `rust/vllm-api-server/Cargo.toml` |
| 2 | Set up proto compilation | `rust/vllm-api-server/build.rs` |
| 3 | Create OpenAI types | `rust/vllm-api-server/src/openai/types.rs` |
| 4 | Create gRPC client | `rust/vllm-api-server/src/grpc/client.rs` |
| 5 | Create Axum server | `rust/vllm-api-server/src/server.rs`, `src/main.rs` |
| 6 | Non-streaming chat | `rust/vllm-api-server/src/routes/chat.rs` |
| 7 | Streaming chat (SSE) | `rust/vllm-api-server/src/openai/streaming.rs` |
| 8 | Chat template config | Update `server.rs`, `main.rs` |
| 9 | Build release | `cargo build --release` |
| 10 | Benchmark script | `scripts/benchmark_rust_vs_python.sh` |
| 11 | Integration tests | `scripts/test_rust_server.sh` |

**Total estimated tasks:** 11 tasks with ~45 steps
