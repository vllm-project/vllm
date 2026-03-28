Overview:
    description:
        vLLM is a high-throughput, memory-efficient inference and serving engine
        for large language models.  It provides an OpenAI-compatible API server
        with continuous batching, PagedAttention, CUDA graph support, and
        torch.compile integration.

    subsystems:
        engine:
            The core scheduling and request lifecycle management layer.
            Accepts requests, tokenizes, schedules batches, and returns outputs.
        model_executor:
            Loads and runs model forward passes.  Contains model definitions,
            quantization, LoRA, and custom ops (including steering).
        worker:
            Manages GPU resources, the persistent input batch, and per-step
            buffer updates (KV cache, steering tables, LoRA adapters).
        entrypoints:
            HTTP/gRPC server (OpenAI-compatible) and offline LLM API.

    data_flow:
        Request arrives at entrypoints → engine validates and enqueues →
        scheduler picks requests respecting capacity constraints (tokens,
        KV blocks, LoRA slots, steering config slots) → model runner
        prepares inputs and updates GPU buffers → model forward pass →
        sampler selects tokens → engine streams output back to client.

Features Index:
    steering:
        description: >
            Activation steering — inject additive vectors into the residual
            stream during decode to steer model behaviour.  Supports both
            global (server-wide) and per-request steering vectors.
        entry_points:
            - POST /v1/steering/set (global)
            - POST /v1/steering/clear (global)
            - GET /v1/steering (status)
            - SamplingParams.steering_vectors (per-request)
        depends_on: []
        doc: docs/features/steering.md
