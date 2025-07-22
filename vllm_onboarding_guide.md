# vLLM v1 Developer Onboarding Guide

## 1. 🧭 Overview

vLLM v1 is a high-performance large language model serving framework designed for **easy, fast, and cheap LLM serving**. It represents a major architectural upgrade from v0 with significant performance improvements and cleaner code organization.

### Key Features
- **High-throughput serving** with state-of-the-art performance
- **PagedAttention** for efficient memory management of attention key-value pairs
- **Continuous batching** of incoming requests for optimal resource utilization
- **Speculative decoding** and **chunked prefill** for faster inference
- **Multi-modal support** (text, vision, audio) with unified processing
- **Distributed inference** with tensor and pipeline parallelism
- **Prefix caching** for improved efficiency on repeated prompts
- **Multiple hardware support** (NVIDIA GPUs, AMD, Intel, TPU, AWS Neuron)

### Technologies Used
- **Language**: Python 3.8+ with C++/CUDA kernels
- **Framework**: PyTorch with custom CUDA kernels
- **Distributed**: Ray for multi-node coordination, multiprocessing for local parallelism
- **Memory Management**: Custom block-based KV cache with PagedAttention
- **API**: OpenAI-compatible REST API server
- **Build System**: CMake for C++/CUDA components, setuptools for Python

---

## 2. 🧱 High-Level Architecture Diagram

```mermaid
graph TB
    subgraph "Client Layer"
        API[API Server]
        CLI[CLI Interface]
        SDK[Python SDK]
    end

    subgraph "Engine Layer"
        LLMEngine[LLM Engine]
        Processor[Input Processor]
        OutputProcessor[Output Processor]
    end

    subgraph "Core Execution"
        EngineCore[Engine Core]
        Scheduler[Scheduler]
        KVManager[KV Cache Manager]
        StructuredOutput[Structured Output Manager]
    end

    subgraph "Execution Layer"
        Executor[Executor]
        subgraph "Workers"
            GPUWorker[GPU Worker]
            CPUWorker[CPU Worker]
            TPUWorker[TPU Worker]
        end
        ModelRunner[Model Runner]
    end

    subgraph "Storage & Cache"
        KVCache[(KV Cache Blocks)]
        EncoderCache[(Encoder Cache)]
        PrefixCache[(Prefix Cache)]
    end

    subgraph "External Systems"
        HF[Hugging Face Models]
        Distributed[Ray Cluster]
        Monitoring[Prometheus Metrics]
    end

    API --> LLMEngine
    CLI --> LLMEngine
    SDK --> LLMEngine

    LLMEngine --> Processor
    LLMEngine --> OutputProcessor
    LLMEngine --> EngineCore

    EngineCore --> Scheduler
    EngineCore --> KVManager
    EngineCore --> StructuredOutput

    Scheduler --> Executor
    Executor --> GPUWorker
    Executor --> CPUWorker
    Executor --> TPUWorker

    GPUWorker --> ModelRunner
    CPUWorker --> ModelRunner
    TPUWorker --> ModelRunner

    KVManager --> KVCache
    KVManager --> EncoderCache
    KVManager --> PrefixCache

    ModelRunner --> HF
    Executor --> Distributed
    LLMEngine --> Monitoring
```

### Component Explanations

- **LLM Engine**: Main orchestrator that coordinates all components and provides the public API
- **Engine Core**: Core execution engine that manages request lifecycle and coordinates scheduling
- **Scheduler**: Intelligent request scheduler that manages resource allocation and batching decisions
- **KV Cache Manager**: Sophisticated memory manager using PagedAttention for efficient key-value storage
- **Workers**: Hardware-specific execution units that run the actual model inference
- **Executor**: Coordinates distributed execution across multiple workers/nodes

---

## 3. 🔎 Component Breakdown

### Component: LLM Engine (`/data/users/yeq/gitrepos/vllm/vllm/v1/engine/llm_engine.py`)

**Purpose**:
Main entry point and orchestrator for the entire vLLM system. Provides backward compatibility with v0 API while leveraging v1 architecture improvements.

**Key Elements**:
- `LLMEngine.__init__()`: Initializes all core components and establishes communication channels
- `add_request()`: Processes and queues new inference requests with validation
- `step()`: Executes one inference iteration, coordinating scheduling and execution
- `abort_request()`: Handles request cancellation and resource cleanup
- `get_tokenizer_group()`: Provides access to tokenization services

**Depends On**:
- Internal: `EngineCoreClient`, `Processor`, `OutputProcessor`, `Executor`
- External: PyTorch, Hugging Face Transformers, Ray (optional)

---

### Component: Engine Core (`/data/users/yeq/gitrepos/vllm/vllm/v1/engine/core.py`)

**Purpose**:
Core execution engine that manages the request lifecycle, coordinates between scheduler and workers, and handles distributed execution.

**Key Elements**:
- `EngineCore.add_request()`: Validates and queues requests for scheduling
- `EngineCore.get_output()`: Retrieves completed inference results
- `EngineCore.abort_requests()`: Handles request cancellation
- `EngineCoreClient`: Client interface for multiprocess communication

**Depends On**:
- Internal: `Scheduler`, `Executor`, `ModelRunner`
- External: Multiprocessing, asyncio

---

### Component: Scheduler (`/data/users/yeq/gitrepos/vllm/vllm/v1/core/sched/scheduler.py`)

**Purpose**:
Intelligent request scheduler that makes optimal batching decisions, manages resource allocation, and handles advanced features like speculative decoding and prefix caching.

**Key Elements**:
- `Scheduler.schedule()`: Core scheduling algorithm that batches requests optimally
- `_try_schedule_encoder_inputs()`: Handles multi-modal input scheduling
- `update_from_output()`: Processes model outputs and updates request states
- `_make_cached_request_data()`: Optimizes data structures for cached requests

**Depends On**:
- Internal: `KVCacheManager`, `StructuredOutputManager`, `RequestQueue`
- External: None (pure Python logic)

---

### Component: KV Cache Manager (`/data/users/yeq/gitrepos/vllm/vllm/v1/core/kv_cache_manager.py`)

**Purpose**:
Sophisticated memory management system implementing PagedAttention for efficient key-value cache storage and retrieval.

**Key Elements**:
- `KVCacheManager.allocate_slots()`: Allocates memory blocks for new requests
- `get_computed_blocks()`: Retrieves cached computation results
- `free()`: Releases memory blocks when requests complete
- `cache_blocks()`: Implements prefix caching for repeated prompts

**Depends On**:
- Internal: `BlockPool`, `KVCacheUtils`
- External: PyTorch tensors

---

### Component: Workers (`/data/users/yeq/gitrepos/vllm/vllm/v1/worker/`)

**Purpose**:
Hardware-specific execution units that perform the actual model inference on different accelerators.

**Key Elements**:
- `GPUWorker`: NVIDIA GPU-optimized execution with CUDA kernels
- `CPUWorker`: CPU-based inference for cost-effective serving
- `TPUWorker`: Google TPU integration for specialized workloads
- `ModelRunner`: Coordinates model execution and batch processing

**Depends On**:
- Internal: `InputBatch`, `BlockTable`, model loading utilities
- External: PyTorch, hardware-specific libraries (CUDA, TPU)

---

### Component: Executors (`/data/users/yeq/gitrepos/vllm/vllm/v1/executor/`)

**Purpose**:
Coordinates distributed execution across multiple workers and handles different parallelism strategies.

**Key Elements**:
- `MultiprocessExecutor`: Local multi-GPU execution
- `RayDistributedExecutor`: Multi-node distributed execution via Ray
- `AbstractExecutor`: Base interface for all execution strategies

**Depends On**:
- Internal: `Worker` implementations
- External: Ray (for distributed), multiprocessing

---

### Component: Request Processing (`/data/users/yeq/gitrepos/vllm/vllm/v1/request.py`, `/data/users/yeq/gitrepos/vllm/vllm/v1/outputs.py`)

**Purpose**:
Handles request lifecycle management, input validation, and output formatting.

**Key Elements**:
- `Request`: Core request data structure with state management
- `RequestStatus`: Enum tracking request lifecycle states
- `ModelRunnerOutput`: Structured output from model execution
- `SamplerOutput`: Token sampling results with logprobs

**Depends On**:
- Internal: Sampling/pooling parameters, multi-modal inputs
- External: PyTorch tensors

---

## 4. 🔁 Data Flow & Call Flow Examples

### Example Flow: Single Request Processing

**Description**:
A client submits a text generation request that goes through the complete vLLM pipeline from input processing to response generation.

**Sequence Diagram**:

```mermaid
sequenceDiagram
    participant Client
    participant LLMEngine
    participant Processor
    participant EngineCore
    participant Scheduler
    participant KVManager
    participant Executor
    participant Worker
    participant ModelRunner

    Client->>LLMEngine: add_request(prompt, sampling_params)
    LLMEngine->>Processor: process_inputs(prompt, params)
    Processor-->>LLMEngine: EngineCoreRequest
    LLMEngine->>EngineCore: add_request(core_request)
    EngineCore->>Scheduler: add_request(request)

    Note over Scheduler: Request queued in WAITING state

    Client->>LLMEngine: step() - Execute inference
    LLMEngine->>EngineCore: get_output()
    EngineCore->>Scheduler: schedule()

    Scheduler->>KVManager: allocate_slots(request, num_tokens)
    KVManager-->>Scheduler: allocated_blocks
    Scheduler-->>EngineCore: SchedulerOutput

    EngineCore->>Executor: execute_model(scheduler_output)
    Executor->>Worker: execute_model_async(model_input)
    Worker->>ModelRunner: execute_model(model_input)

    Note over ModelRunner: Forward pass through transformer

    ModelRunner-->>Worker: ModelRunnerOutput
    Worker-->>Executor: ModelRunnerOutput
    Executor-->>EngineCore: ModelRunnerOutput

    EngineCore->>Scheduler: update_from_output(output)
    Scheduler-->>EngineCore: EngineCoreOutputs
    EngineCore-->>LLMEngine: EngineCoreOutputs

    LLMEngine->>LLMEngine: output_processor.process_outputs()
    LLMEngine-->>Client: RequestOutput
```

---

### Example Flow: Batched Request Processing

**Description**:
Multiple requests are intelligently batched together for efficient GPU utilization, demonstrating vLLM's continuous batching capabilities.

**Sequence Diagram**:

```mermaid
sequenceDiagram
    participant Client1
    participant Client2
    participant Client3
    participant LLMEngine
    participant Scheduler
    participant KVManager
    participant Worker

    Client1->>LLMEngine: add_request(req1)
    Client2->>LLMEngine: add_request(req2)
    Client3->>LLMEngine: add_request(req3)

    Note over LLMEngine: Multiple requests queued

    LLMEngine->>Scheduler: schedule()

    Note over Scheduler: Batch optimization logic
    Scheduler->>Scheduler: calculate_token_budget()
    Scheduler->>Scheduler: select_requests_for_batch()

    loop For each selected request
        Scheduler->>KVManager: allocate_slots(request)
        KVManager-->>Scheduler: blocks_allocated
    end

    Scheduler-->>LLMEngine: SchedulerOutput(batched_requests)

    LLMEngine->>Worker: execute_model(batch)

    Note over Worker: Single forward pass for all requests

    Worker-->>LLMEngine: ModelRunnerOutput(batch_results)

    LLMEngine->>LLMEngine: split_batch_outputs()
    LLMEngine-->>Client1: RequestOutput(req1_result)
    LLMEngine-->>Client2: RequestOutput(req2_result)
    LLMEngine-->>Client3: RequestOutput(req3_result)
```

---

### Example Flow: Prefix Caching Hit

**Description**:
A request benefits from prefix caching when its prompt shares a common prefix with a previously processed request.

**Sequence Diagram**:

```mermaid
sequenceDiagram
    participant Client
    participant LLMEngine
    participant Scheduler
    participant KVManager
    participant PrefixCache

    Client->>LLMEngine: add_request("Explain quantum physics...")
    LLMEngine->>Scheduler: schedule()

    Scheduler->>KVManager: get_computed_blocks(request)
    KVManager->>PrefixCache: lookup_prefix_hash(prompt_tokens)

    alt Cache Hit
        PrefixCache-->>KVManager: cached_blocks(num_cached_tokens=50)
        KVManager-->>Scheduler: computed_blocks + cache_info

        Note over Scheduler: Skip computation for cached tokens
        Scheduler->>Scheduler: schedule_remaining_tokens(total-cached)

    else Cache Miss
        PrefixCache-->>KVManager: no_cache_found
        KVManager-->>Scheduler: empty_blocks

        Note over Scheduler: Full computation required
        Scheduler->>Scheduler: schedule_all_tokens()
    end

    Scheduler-->>LLMEngine: SchedulerOutput
    Note over LLMEngine: Execution continues with optimized token count
```

---

### Example Flow: Multi-Modal Request Processing

**Description**:
Processing a request that includes both text and image inputs, demonstrating vLLM's multi-modal capabilities.

**Sequence Diagram**:

```mermaid
sequenceDiagram
    participant Client
    participant LLMEngine
    participant Processor
    participant Scheduler
    participant EncoderCache
    participant Worker
    participant VisionEncoder

    Client->>LLMEngine: add_request(text="Describe image", image=img_data)
    LLMEngine->>Processor: process_inputs(multimodal_input)

    Processor->>Processor: tokenize_text()
    Processor->>Processor: process_image_placeholders()
    Processor-->>LLMEngine: Request(mm_inputs, mm_positions)

    LLMEngine->>Scheduler: schedule()
    Scheduler->>Scheduler: _try_schedule_encoder_inputs()

    alt Encoder Input Needed
        Scheduler->>EncoderCache: can_allocate(request, input_id)
        EncoderCache-->>Scheduler: cache_available

        Scheduler->>EncoderCache: allocate(request, input_id)
        Scheduler-->>LLMEngine: SchedulerOutput(encoder_inputs=[0])

        LLMEngine->>Worker: execute_model(scheduler_output)
        Worker->>VisionEncoder: encode_image(image_data)
        VisionEncoder-->>Worker: image_embeddings

        Worker->>Worker: merge_text_image_embeddings()
        Worker-->>LLMEngine: ModelRunnerOutput

    else Encoder Cached
        Scheduler->>EncoderCache: get_cached_embeddings()
        EncoderCache-->>Scheduler: cached_embeddings
        Note over Scheduler: Skip encoder computation
    end

    LLMEngine-->>Client: RequestOutput(generated_text)
```

---

## 5. 🗃️ Data Models (Entities)

### Entity: Request

- **Class**: `Request` in `/data/users/yeq/gitrepos/vllm/vllm/v1/request.py`
- **Fields**:
  - `request_id: str` – unique identifier for the request
  - `prompt_token_ids: list[int]` – tokenized input prompt
  - `sampling_params: SamplingParams` – generation parameters (temperature, top_p, etc.)
  - `pooling_params: PoolingParams` – for embedding/pooling requests
  - `status: RequestStatus` – current lifecycle state (WAITING, RUNNING, FINISHED_*)
  - `num_computed_tokens: int` – number of tokens already processed
  - `max_tokens: int` – maximum tokens to generate
  - `arrival_time: float` – timestamp when request was received
  - `priority: int` – scheduling priority (higher = more important)

- **Relations**:
  - Contains `MultiModalKwargs` for vision/audio inputs
  - References `LoRARequest` for adapter-specific inference
  - Links to `StructuredOutputRequest` for guided generation

- **Notes**:
  - Immutable token lists use `ConstantList` wrapper for safety
  - Supports speculative decoding with `spec_token_ids`
  - Tracks prefix cache hits via `num_cached_tokens`

---

### Entity: RequestStatus

- **Enum**: `RequestStatus` in `/data/users/yeq/gitrepos/vllm/vllm/v1/request.py`
- **Values**:
  - `WAITING` – queued for scheduling
  - `WAITING_FOR_FSM` – waiting for structured output compilation
  - `WAITING_FOR_REMOTE_KVS` – waiting for distributed KV transfer
  - `RUNNING` – actively being processed
  - `PREEMPTED` – temporarily paused for higher priority requests
  - `FINISHED_STOPPED` – completed normally (stop token/string)
  - `FINISHED_LENGTH_CAPPED` – completed due to max length
  - `FINISHED_ABORTED` – cancelled by client
  - `FINISHED_IGNORED` – rejected due to constraints

- **Relations**:
  - Maps to `FinishReason` enum for API compatibility
  - Used by scheduler for state transitions

- **Notes**:
  - States > PREEMPTED are considered finished
  - Supports graceful degradation and error handling

---

### Entity: ModelRunnerOutput

- **Class**: `ModelRunnerOutput` in `/data/users/yeq/gitrepos/vllm/vllm/v1/outputs.py`
- **Fields**:
  - `req_ids: list[str]` – request identifiers in batch order
  - `req_id_to_index: dict[str, int]` – mapping for efficient lookup
  - `sampled_token_ids: list[list[int]]` – generated tokens per request
  - `spec_token_ids: list[list[int]]` – speculative tokens (if enabled)
  - `logprobs: LogprobsLists` – token probabilities for each request
  - `prompt_logprobs_dict: dict[str, LogprobsTensors]` – prompt token probabilities
  - `pooler_output: list[torch.Tensor]` – embeddings for pooling requests

- **Relations**:
  - Consumed by `Scheduler.update_from_output()`
  - Converted to `RequestOutput` by `OutputProcessor`

- **Notes**:
  - Uses lists instead of tensors for efficient serialization
  - Supports variable-length outputs per request in batch

---

### Entity: SchedulerOutput

- **Class**: `SchedulerOutput` in `/data/users/yeq/gitrepos/vllm/vllm/v1/core/sched/output.py`
- **Fields**:
  - `scheduled_new_reqs: list[NewRequestData]` – first-time scheduled requests
  - `scheduled_cached_reqs: CachedRequestData` – continuing requests
  - `num_scheduled_tokens: dict[str, int]` – tokens per request this step
  - `total_num_scheduled_tokens: int` – total batch size
  - `scheduled_encoder_inputs: dict[str, list[int]]` – multi-modal inputs to process
  - `num_common_prefix_blocks: list[int]` – shared prefix optimization data

- **Relations**:
  - Produced by `Scheduler.schedule()`
  - Consumed by `Executor.execute_model()`

- **Notes**:
  - Optimizes memory layout for different request types
  - Includes metadata for advanced features (speculative decoding, prefix caching)

---

### Entity: KVCacheConfig

- **Class**: `KVCacheConfig` in `/data/users/yeq/gitrepos/vllm/vllm/v1/kv_cache_interface.py`
- **Fields**:
  - `block_size: int` – tokens per memory block (typically 16)
  - `num_gpu_blocks: int` – total GPU memory blocks available
  - `num_cpu_blocks: int` – CPU memory blocks for offloading
  - `cache_dtype: torch.dtype` – data type for cache storage
  - `kv_cache_groups: list[KVCacheGroup]` – cache organization

- **Relations**:
  - Used by `KVCacheManager` for memory allocation
  - Configured based on model and hardware constraints

- **Notes**:
  - Block-based design enables efficient memory management
  - Supports heterogeneous memory hierarchies (GPU/CPU)

---

### Entity: SamplingParams

- **Class**: `SamplingParams` in `vllm/sampling_params.py`
- **Fields**:
  - `n: int` – number of output sequences to generate
  - `max_tokens: int` – maximum tokens to generate
  - `temperature: float` – sampling randomness (0.0 = deterministic)
  - `top_p: float` – nucleus sampling threshold
  - `top_k: int` – top-k sampling limit
  - `stop: list[str]` – stop strings to terminate generation
  - `logprobs: int` – number of log probabilities to return

- **Relations**:
  - Embedded in `Request` objects
  - Used by sampling kernels during generation

- **Notes**:
  - Supports advanced sampling strategies (beam search, parallel sampling)
  - Extensible for custom sampling algorithms# vLLM Developer Onboarding Guide
