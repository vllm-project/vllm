# vLLM 主干逻辑架构概览

## 整体分层

vLLM 的架构可以分为以下几个层次（从上到下）：

### 1. 入口层 (`vllm/entrypoints/`)

- **离线推理**: `llm.py` — 用户通过 `LLM` 类直接调用，批量生成
- **在线服务**: `api_server.py` — 启动 HTTP 服务器，提供 OpenAI 兼容 API (`openai/`)、gRPC、MCP 等接口

### 2. 引擎层 (`vllm/v1/engine/`)

当前主力是 **v1 引擎**，核心组件：

- **`async_llm.py`** — 异步引擎前端，接收请求、返回流式输出
- **`core.py`** — Engine Core，运行在独立进程中，负责调度 + 模型执行的主循环
- **`core_client.py`** — 前端与 EngineCore 之间的 IPC 通信客户端
- **`input_processor.py`** — 请求预处理（tokenize、多模态输入处理等）
- **`output_processor.py`** — 输出后处理（detokenize、组装结果）
- **`detokenizer.py`** — 增量 detokenize

### 3. 调度层 (`vllm/v1/core/sched/`)

- **`scheduler.py`** — 核心调度器，决定每步哪些请求参与计算（prefill vs decode），管理抢占
- **KV Cache 管理** (`kv_cache_manager.py`, `block_pool.py`) — 分块式 KV cache 分配、prefix caching

### 4. 执行层 (`vllm/v1/worker/` + `vllm/v1/executor/`)

- **Executor** — 管理 worker 进程（支持单 GPU、多 GPU tensor parallel、Ray 分布式）
- **Worker** — 每个 GPU 上的工作进程，持有模型和 KV cache
- **`model_runner.py`** — 构造输入张量、调用模型 forward、采样

### 5. 模型层 (`vllm/models/`)

- 各种模型实现（LLaMA, GPT, Qwen, Mixtral 等）
- 统一的 attention 后端接口 (`vllm/v1/attention/`)

### 6. 关键横切模块

- **`sampling_params.py` / `vllm/v1/sample/`** — 采样逻辑（top-k, top-p, temperature 等）
- **`vllm/distributed/`** — tensor parallel、pipeline parallel 通信
- **`vllm/v1/spec_decode/`** — 投机解码
- **`vllm/multimodal/`** — 多模态输入处理
- **`vllm/lora/`** — LoRA 适配器

---

## 一次请求的生命周期（简化版）

```
用户请求 → API Server → AsyncLLM (前端)
  → InputProcessor (tokenize/multimodal)
  → EngineCore (独立进程)
    → Scheduler (调度，分配 KV block)
    → ModelRunner (构建 batch, GPU forward)
    → Sampler (采样 next token)
  → OutputProcessor (detokenize, 流式返回)
→ 响应给用户
```

---

## 深入方向索引

后续可以继续深入的方向：

- 调度器是怎么决定 prefill/decode 批次的？
- KV Cache 的 PagedAttention 块管理机制？
- EngineCore 的主循环具体做了什么？
- ModelRunner 如何构建输入张量并执行 forward？
- 分布式执行（Tensor Parallel / Pipeline Parallel）如何协调？
- 投机解码（Speculative Decoding）的工作流程？
