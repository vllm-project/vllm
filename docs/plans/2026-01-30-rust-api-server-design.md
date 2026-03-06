# Rust API Server for vLLM

**Date:** 2026-01-30
**Author:** Jake Weissman
**Status:** Draft

## Overview

This document describes a prototype Rust-based API server for vLLM that communicates with the Python EngineCore via gRPC. The goal is to benchmark throughput and TTFT improvements compared to the current Python FastAPI server.

## Motivation

The current vLLM API server is implemented in Python using FastAPI/Uvicorn. While functional, Python's performance characteristics in HTTP parsing, JSON serialization, and request handling may limit throughput at high concurrency. A Rust implementation could reduce API layer overhead and improve:

- **Throughput** (requests/second)
- **TTFT P90/P99** (Time to First Token latency)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     External Clients                             │
│              (genai-bench, curl, applications)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP/JSON (OpenAI-compatible)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Rust API Server (NEW)                          │
│                                                                  │
│  1. Parse HTTP/JSON request                                      │
│  2. Apply chat template (messages → prompt string)               │
│  3. Tokenize prompt → token IDs                                  │
│  4. Build SamplingParams                                         │
│  5. Select backend (round-robin for single host)                 │
│  6. Serialize to protobuf (TokenizedInput)                       │
│  7. Send gRPC GenerateRequest                                    │
│  8. Receive token ID stream from engine                          │
│  9. Detokenize token IDs → text                                  │
│  10. Format as OpenAI SSE chunks                                 │
│  11. Stream HTTP response                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ gRPC (token IDs in, token IDs out)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Python gRPC Server (grpc_server.py)                 │
│         Receives TokenizedInput, returns token ID stream         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     vLLM EngineCore                              │
│              (GPU inference, scheduling)                         │
└─────────────────────────────────────────────────────────────────┘
```

The Rust server handles all API-layer work: HTTP parsing, JSON serialization, tokenization, SSE streaming. The Python gRPC server is a thin wrapper around AsyncLLM that accepts token IDs and returns token IDs.

## Project Structure

```
vllm/
├── rust/                           # New Rust workspace
│   ├── Cargo.toml                  # Workspace manifest
│   ├── vllm-api-server/            # Main API server crate
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── main.rs             # Entry point, CLI args
│   │   │   ├── server.rs           # Axum server setup
│   │   │   ├── routes/
│   │   │   │   ├── mod.rs
│   │   │   │   └── chat.rs         # /v1/chat/completions handler
│   │   │   ├── openai/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── types.rs        # OpenAI request/response types
│   │   │   │   └── streaming.rs    # SSE streaming logic
│   │   │   └── grpc/
│   │   │       ├── mod.rs
│   │   │       └── client.rs       # Tonic client wrapper
│   │   └── build.rs                # Proto compilation
│   └── proto/                      # Symlink to vllm/grpc/*.proto
├── grpc/
│   └── vllm_engine.proto           # Existing proto (shared)
└── scripts/
    └── benchmark_rust_vs_python.sh # Benchmark runner script
```

## API Surface

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (streaming & non-streaming) |
| `/health` | GET | Health check |

### Request Format (OpenAI-compatible)

```json
POST /v1/chat/completions
{
  "model": "meta-llama/Llama-3-8B-Instruct",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 100,
  "temperature": 0.7,
  "stream": true
}
```

### Response Format (Streaming)

```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"}}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"!"}}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

## Key Components

### Dependencies

```toml
[dependencies]
# Web framework
axum = { version = "0.7", features = ["macros"] }
tokio = { version = "1", features = ["full"] }
tower = "0.4"
tower-http = { version = "0.5", features = ["cors", "trace"] }

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# gRPC
tonic = "0.12"
prost = "0.13"

# Tokenizer (HuggingFace's Rust implementation)
tokenizers = "0.20"

# Streaming
tokio-stream = "0.1"
async-stream = "0.3"

# CLI & Config
clap = { version = "4", features = ["derive"] }
tracing = "0.1"
tracing-subscriber = "0.3"

[build-dependencies]
tonic-build = "0.12"
```

### Core Types

```rust
// OpenAI request types
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stream: Option<bool>,
    pub stop: Option<Vec<String>>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub seed: Option<i64>,
}

pub struct ChatMessage {
    pub role: String,       // "system", "user", "assistant"
    pub content: String,
}

// Server state
pub struct AppState {
    pub tokenizer: Tokenizer,           // HuggingFace tokenizer
    pub grpc_client: VllmEngineClient,  // Tonic gRPC client
    pub model_name: String,
}
```

### Streaming Handler

```rust
async fn chat_completions(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    if request.stream.unwrap_or(false) {
        stream_response(state, request).await
    } else {
        non_stream_response(state, request).await
    }
}

async fn stream_response(
    state: AppState,
    req: ChatCompletionRequest
) -> Sse<impl Stream<Item = Event>> {
    // 1. Apply chat template & tokenize
    let prompt = apply_chat_template(&req.messages, &state.tokenizer);
    let encoding = state.tokenizer.encode(prompt, false).unwrap();
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    // 2. Build gRPC request
    let grpc_req = GenerateRequest {
        request_id: uuid::Uuid::new_v4().to_string(),
        input: Some(Input::Tokenized(TokenizedInput {
            original_text: prompt,
            input_ids: token_ids,
        })),
        sampling_params: Some(build_sampling_params(&req)),
        stream: true,
    };

    // 3. Call gRPC streaming endpoint
    let mut stream = state.grpc_client
        .generate(grpc_req)
        .await
        .unwrap()
        .into_inner();

    // 4. Transform gRPC stream → SSE stream
    let sse_stream = async_stream::stream! {
        while let Some(chunk) = stream.message().await.unwrap() {
            if let Some(response) = chunk.response {
                match response {
                    Response::Chunk(c) => {
                        let text = state.tokenizer
                            .decode(&c.token_ids, false)
                            .unwrap();
                        let delta = format_chunk(&text);
                        yield Event::default().data(delta);
                    }
                    Response::Complete(c) => {
                        let done = format_complete(&c);
                        yield Event::default().data(done);
                        yield Event::default().data("[DONE]");
                    }
                }
            }
        }
    };

    Sse::new(sse_stream)
}
```

## Benchmarking

### Tool

Use `genai-bench` with the following configuration:

```bash
genai-bench benchmark \
    --api-backend vllm \
    --api-base "http://localhost:8000" \
    --task text-to-text \
    --model-tokenizer "meta-llama/Llama-3-8B-Instruct" \
    --api-model-name "meta-llama/Llama-3-8B-Instruct" \
    --traffic-scenario "D(100,100)" \
    --traffic-scenario "D(2000,200)" \
    --num-concurrency 1 --num-concurrency 2 --num-concurrency 4 \
    --num-concurrency 8 --num-concurrency 16 --num-concurrency 32 \
    --num-concurrency 64 \
    --max-time-per-run 3 \
    --max-requests-per-run 500
```

### Traffic Scenarios

| Scenario | Description |
|----------|-------------|
| `D(100,100)` | Chatbot/dialog - short input, short output |
| `D(2000,200)` | Typical RAG - long context, medium output |

### Metrics

| Metric | Target |
|--------|--------|
| Throughput (req/s) | Higher than Python baseline |
| TTFT P90 | Lower than Python baseline |
| TTFT P99 | Lower than Python baseline |

### Benchmark Script

```bash
#!/bin/bash
# scripts/benchmark_rust_vs_python.sh

MODEL="meta-llama/Llama-3-8B-Instruct"
GRPC_PORT=50051
HTTP_PORT=8000

# Start Python gRPC backend (shared by both servers)
python -m vllm.entrypoints.grpc_server \
    --model $MODEL \
    --port $GRPC_PORT &
GRPC_PID=$!
echo "Waiting for gRPC server to load model..."
sleep 60

# Benchmark 1: Python API Server (baseline)
echo "Starting Python API server benchmark..."
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --port $HTTP_PORT &
PY_PID=$!
sleep 10

genai-bench benchmark \
    --api-backend vllm \
    --api-base "http://localhost:$HTTP_PORT" \
    --task text-to-text \
    --model-tokenizer $MODEL \
    --api-model-name $MODEL \
    --traffic-scenario "D(100,100)" \
    --traffic-scenario "D(2000,200)" \
    --num-concurrency 1 --num-concurrency 4 --num-concurrency 16 --num-concurrency 64 \
    --max-time-per-run 3 \
    --max-requests-per-run 500 \
    --experiment-folder-name "python-baseline"

kill $PY_PID
sleep 2

# Benchmark 2: Rust API Server
echo "Starting Rust API server benchmark..."
./rust/target/release/vllm-api-server \
    --grpc-addr "localhost:$GRPC_PORT" \
    --port $HTTP_PORT \
    --model $MODEL &
RUST_PID=$!
sleep 5

genai-bench benchmark \
    --api-backend vllm \
    --api-base "http://localhost:$HTTP_PORT" \
    --task text-to-text \
    --model-tokenizer $MODEL \
    --api-model-name $MODEL \
    --traffic-scenario "D(100,100)" \
    --traffic-scenario "D(2000,200)" \
    --num-concurrency 1 --num-concurrency 4 --num-concurrency 16 --num-concurrency 64 \
    --max-time-per-run 3 \
    --max-requests-per-run 500 \
    --experiment-folder-name "rust-prototype"

kill $RUST_PID $GRPC_PID

# Generate comparison report
echo "Generating comparison report..."
genai-bench excel \
    --experiment-folder ./experiments \
    --excel-name rust-vs-python-comparison \
    --metric-percentile p90 p99 mean

echo "Benchmark complete. Results in ./experiments/"
```

## Implementation Phases

### Phase 1: Minimal Working Prototype
- Rust server with `/v1/chat/completions` (non-streaming)
- Hardcoded model/tokenizer path
- Single gRPC backend connection
- Basic error handling

### Phase 2: Streaming Support
- Add SSE streaming for `stream: true`
- Incremental detokenization
- Proper OpenAI chunk formatting

### Phase 3: Production Features
- CLI arguments (model, port, grpc-addr)
- Health check endpoint
- Graceful shutdown
- Request logging/tracing

### Phase 4: Benchmarking & Optimization
- Run genai-bench comparisons
- Profile hotspots
- Tune buffer sizes, connection pooling

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Chat template incompatibility | Incorrect prompt formatting | Use HuggingFace tokenizers with chat template support |
| Tokenizer output mismatch | Different token IDs than Python | Validate against Python tokenizer output with test cases |
| gRPC connection failures | Request failures | Add retry logic with exponential backoff |
| Streaming backpressure | Memory growth | Use bounded channels, apply backpressure |

## Success Criteria

The prototype is successful if:

1. **Functional parity**: `/v1/chat/completions` works with streaming and non-streaming modes
2. **Throughput improvement**: Measurable increase in requests/second at high concurrency
3. **Latency improvement**: Lower TTFT P90/P99 compared to Python baseline
4. **Stability**: No crashes or memory leaks during benchmark runs

## Future Work (Out of Scope)

- Full OpenAI API surface (`/v1/completions`, `/v1/embeddings`, `/v1/models`)
- Multi-host load balancing with health checks
- TLS/authentication
- Prometheus metrics export
- Integration with vLLM's data parallel coordinator
