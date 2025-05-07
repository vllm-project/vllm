(optimization-and-tuning)=

# Optimization and Tuning

This guide covers optimization strategies and performance tuning for vLLM V1.

## Preemption

Due to the auto-regressive nature of transformer architecture, there are times when KV cache space is insufficient to handle all batched requests.
In such cases, vLLM can preempt requests to free up KV cache space for other requests. Preempted requests are recomputed when sufficient KV cache space becomes
available again. When this occurs, you may see the following warning:

```text
WARNING 05-09 00:49:33 scheduler.py:1057 Sequence group 0 is preempted by PreemptionMode.RECOMPUTE mode because there is not enough KV cache space. This can affect the end-to-end performance. Increase gpu_memory_utilization or tensor_parallel_size to provide more KV cache memory. total_cumulative_preemption_cnt=1
```

While this mechanism ensures system robustness, preemption and recomputation can adversely affect end-to-end latency.
If you frequently encounter preemptions, consider the following actions:

- Increase `gpu_memory_utilization`. vLLM pre-allocates GPU cache using this percentage of memory. By increasing utilization, you can provide more KV cache space.
- Decrease `max_num_seqs` or `max_num_batched_tokens`. This reduces the number of concurrent requests in a batch, thereby requiring less KV cache space.
- Increase `tensor_parallel_size`. This shards model weights across GPUs, allowing each GPU to have more memory available for KV cache. However, increasing this value may cause excessive synchronization overhead.
- Increase `pipeline_parallel_size`. This distributes model layers across GPUs, reducing the memory needed for model weights on each GPU, indirectly leaving more memory available for KV cache. However, increasing this value may cause latency penalties.

You can monitor the number of preemption requests through Prometheus metrics exposed by vLLM. Additionally, you can log the cumulative number of preemption requests by setting `disable_log_stats=False`.

In vLLM V1, the default preemption mode is `RECOMPUTE` rather than `SWAP`, as recomputation has lower overhead in the V1 architecture.

(chunked-prefill)=

## Chunked Prefill

Chunked prefill allows vLLM to process large prefills in smaller chunks and batch them together with decode requests. This feature helps improve both throughput and latency by better balancing compute-bound (prefill) and memory-bound (decode) operations.

In vLLM V1, **chunked prefill is always enabled by default**. This is different from vLLM V0, where it was conditionally enabled based on model characteristics.

With chunked prefill enabled, the scheduling policy prioritizes decode requests. It batches all pending decode requests before scheduling any prefill operations. When there are available tokens in the `max_num_batched_tokens` budget, it schedules pending prefills. If a pending prefill request cannot fit into `max_num_batched_tokens`, it automatically chunks it.

This policy has two benefits:

- It improves ITL and generation decode because decode requests are prioritized.
- It helps achieve better GPU utilization by locating compute-bound (prefill) and memory-bound (decode) requests to the same batch.

### Performance Tuning with Chunked Prefill

You can tune the performance by adjusting `max_num_batched_tokens`:

- Smaller values (e.g., 2048) achieve better inter-token latency (ITL) because there are fewer prefills slowing down decodes.
- Higher values achieve better time to first token (TTFT) as you can process more prefill tokens in a batch.
- For optimal throughput, we recommend setting `max_num_batched_tokens > 8096` especially for smaller models on large GPUs.
- If `max_num_batched_tokens` is the same as `max_model_len`, that's almost the equivalent to the V0 default scheduling policy (except that it still prioritizes decodes).

```python
from vllm import LLM

# Set max_num_batched_tokens to tune performance
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", max_num_batched_tokens=16384)
```

See related papers for more details (<https://arxiv.org/pdf/2401.08671> or <https://arxiv.org/pdf/2308.16369>).

## Parallelism Strategies

vLLM supports multiple parallelism strategies that can be combined to optimize performance across different hardware configurations.

### Tensor Parallelism (TP)

Tensor parallelism shards model parameters across multiple GPUs within each model layer. This is the most common strategy for large model inference within a single node.

**When to use:**

- When the model is too large to fit on a single GPU
- When you need to reduce memory pressure per GPU to allow more KV cache space for higher throughput

```python
from vllm import LLM

# Split model across 4 GPUs
llm = LLM(model="meta-llama/Llama-3.3-70B-Instruct", tensor_parallel_size=4)
```

For models that are too large to fit on a single GPU (like 70B parameter models), tensor parallelism is essential.

### Pipeline Parallelism (PP)

Pipeline parallelism distributes model layers across multiple GPUs. Each GPU processes different parts of the model in sequence.

**When to use:**

- When you've already maxed out efficient tensor parallelism but need to distribute the model further, or across nodes
- For very deep and narrow models where layer distribution is more efficient than tensor sharding

Pipeline parallelism can be combined with tensor parallelism for very large models:

```python
from vllm import LLM

# Combine pipeline and tensor parallelism
llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct,
    tensor_parallel_size=4,
    pipeline_parallel_size=2
)
```

### Expert Parallelism (EP)

Expert parallelism is a specialized form of parallelism for Mixture of Experts (MoE) models, where different expert networks are distributed across GPUs.

**When to use:**

- Specifically for MoE models (like DeepSeekV3, Qwen3MoE, Llama-4)
- When you want to balance the expert computation load across GPUs

Expert parallelism is enabled by setting `enable_expert_parallel=True`, which will use expert parallelism instead of tensor parallelism for MoE layers.
It will use the same degree of parallelism as what you have set for tensor parallelism.

### Data Parallelism (DP)

Data parallelism replicates the entire model across multiple GPU sets and processes different batches of requests in parallel.

**When to use:**

- When you have enough GPUs to replicate the entire model
- When you need to scale throughput rather than model size
- In multi-user environments where isolation between request batches is beneficial

Data parallelism can be combined with the other parallelism strategies and is set by `data_parallel_size=N`.
Note that MoE layers will be sharded according to the product of the tensor parallel size and data parallel size.

## Reducing Memory Usage

If you encounter out-of-memory issues, consider these strategies:

### Context Length and Batch Size

You can reduce memory usage by limiting the context length and batch size:

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    max_model_len=2048,  # Limit context window
    max_num_seqs=4       # Limit batch size
)
```

### Adjust CUDA Graph Compilation

CUDA graph compilation in V1 uses more memory than in V0. You can reduce memory usage by adjusting the compilation level:

```python
from vllm import LLM
from vllm.config import CompilationConfig, CompilationLevel

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    compilation_config=CompilationConfig(
        level=CompilationLevel.PIECEWISE,
        cudagraph_capture_sizes=[1, 2, 4, 8]  # Capture fewer batch sizes
    )
)
```

Or, if you are not concerned about latency or overall performance, disable CUDA graph compilation entirely with `enforce_eager=True`:

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enforce_eager=True  # Disable CUDA graph compilation
)
```

### Multimodal Models

For multi-modal models, you can reduce memory usage by limiting the number of images/videos per request:

```python
from vllm import LLM

# Accept up to 2 images per prompt
llm = LLM(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    limit_mm_per_prompt={"image": 2}
)
```
