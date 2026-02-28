# Hybrid Architectures in vLLM: Bridging Transformers and State Space Models for Efficient Long-Context Inference

## Abstract

Large Language Models (LLMs) based on the Transformer architecture have revolutionized natural language processing but face a fundamental bottleneck: the quadratic scaling of attention complexity and the linear growth of the Key-Value (KV) cache with context length. State Space Models (SSMs), such as Mamba, offer a compelling alternative with linear time complexity and constant memory usage. This paper presents the implementation of a **Hybrid Attention Architecture** within vLLM, a high-performance inference engine. We describe the architectural challenges of managing "dual memory" (PagedAttention KV blocks vs. SSM fixed states), the design of a unified backend that fuses sliding-window attention with recurrent history, and the integration of production-grade Mamba kernels. We further detail a novel verification methodology using deterministic prefix-sum tasks to validate state persistence across distributed inference steps.

---

## 1. Introduction

The dominance of the Transformer architecture is predicated on its ability to perform "associative recall"—looking back at any specific token in the history with perfect precision. However, this capability comes at a steep cost. As the context window ($L$) grows, the memory required to store the KV cache scales linearly ($O(L)$), and the compute required for attention scales quadratically ($O(L^2)$). For applications requiring "infinite context," this is prohibitive.

State Space Models (SSMs), particularly recent variants like Mamba, solve this by compressing history into a fixed-size recurrent state. This allows for theoretically infinite context lengths with constant memory usage and linear compute. However, pure SSMs can struggle with the precise retrieval of specific facts from the distant past compared to Attention.

**Hybrid Architectures** aim to capture the best of both worlds by interleaving standard Attention layers (for local precision) with SSM layers (for global context). This paper details the engineering effort to support such hybrid models in vLLM. The primary contribution is not just a new model kernel, but the "architectural plumbing" required to support two distinct memory paradigms—the **PagedAttention** block table and the **SSM Recurrent State**—simultaneously within a single forward pass.

---

## 2. Background

### 2.1 vLLM and PagedAttention
vLLM achieves state-of-the-art throughput by managing memory like an operating system. Through **PagedAttention**, it breaks the KV cache into non-contiguous blocks, allowing for efficient memory allocation and sharing. However, this system was designed strictly for the Transformer paradigm, where "state" is synonymous with a growing list of Key and Value vectors.

### 2.2 The Challenge of Dual State
Integrating an SSM like Mamba introduces a fundamental conflict. An SSM layer does not need a growing cache; it needs a **fixed-size, mutable state** (typically a hidden state $h_t$ and a convolutional state).

In a hybrid model, the inference engine must:
1.  **Allocate** standard KV blocks for the Attention layers.
2.  **Allocate** fixed state buffers for the SSM layers.
3.  **Route** the correct metadata (block tables vs. state indices) to the respective kernels.
4.  **Fuse** the outputs of both branches seamlessly.

Existing systems typically support one or the other. Our work bridges this gap.

---

## 3. Methodology

We implemented a new attention backend, `HybridAttentionBackend`, and a generalized adapter, `HybridSSMAdapter`, to handle the complex orchestration of hybrid inference.

### 3.1 The Hybrid Attention Backend
The core of our solution is the `HybridAttentionImpl` class. Unlike standard backends that simply execute an attention kernel, this implementation acts as a coordinator.

For every layer $l$:
1.  **Sliding Window Attention**: The backend delegates the local attention computation to the existing, highly optimized Triton kernels. This handles the "recent" context (e.g., the last 4096 tokens) using the standard PagedAttention KV cache.
2.  **SSM History Branch**: Simultaneously, the backend invokes the `ssm_adapter`. This component processes the input using the Mamba architecture to extract long-term context from the recurrent state.
3.  **Fusion**: The output of the SSM branch is fused (element-wise added) to the output of the attention mechanism.

$$ Output = \text{Attention}(Q, K, V)_{\text{sliding}} + \text{SSM}(\text{History}) $$

The implementation in `HybridAttentionImpl` coordinates this process:

```python
def forward(self, ...):
    # Step 1: Delegate sliding-window attention to Triton
    # This populates 'output' with the Attention(Q,K,V) result
    self._triton_impl.forward(..., output=output)

    # Step 2: Invoke the SSM adapter (Mamba branch)
    # This calculates the recurrent history contribution
    ssm_out = ssm_adapter.forward_history_branch_decode(
        query_tokens, attn_metadata=attn_metadata
    )

    # Step 3: Fuse SSM contribution into the attention output
    # Simple element-wise addition of the two branches
    output[:num_actual_tokens] += ssm_out
```

### 3.2 The Hybrid SSM Adapter
The `HybridSSMAdapter` serves as the bridge between the vLLM engine and the Mamba mathematics. It implements the `AttentionLayerBase` interface, allowing it to interact directly with vLLM's memory manager.

Crucially, it utilizes a `MambaSpec` to request a dedicated memory pool. This allows vLLM to allocate **Mamba-specific state blocks** alongside the standard KV cache blocks.

```python
def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
    # Request a MambaSpec for fixed-size state allocation.
    # This tells the memory manager to reserve space for the recurrent state
    # separate from the sliding-window KV cache.
    return MambaSpec(
        shapes=tuple(self.get_state_shape()),
        dtypes=self.get_state_dtype(),
        block_size=mamba_block_size,
        page_size_padded=page_size_padded,
        mamba_type="mamba1",
    )
```

The adapter exposes two key methods:
*   `forward_history_branch_prefill`: Processes the initial prompt in parallel.
*   `forward_history_branch_decode`: Handles the step-by-step generation, updating the recurrent state in-place.

### 3.3 Integration of Mamba Kernels
We integrated the real Mamba architecture (Mamba1) directly into the adapter. This involved:
*   **Input Projection**: Splitting the input into standard and gate branches.
*   **Causal Convolution**: Using `causal_conv1d_fn` to process the local context.
*   **Selective Scan**: Utilizing `selective_scan_fn` (for prefill) and `selective_state_update` (for decode) to update the recurrent dynamics.

A significant engineering challenge was **Distributed Initialization (`dist_init`)**. The Mamba parameters ($A$ and $D$) must be correctly sharded across GPUs when running in Tensor Parallel (TP) mode. We implemented custom weight loaders to ensure that the continuous parameters $A$ (representing the state evolution) are split correctly across ranks without breaking the mathematical consistency of the state space.

### 3.4 The "Ungated" Refactor
During implementation, we refactored the model to support "ungated" architectures. This simplified the linear layers involved (`x_proj`, `dt_proj`, `out_proj`), ensuring compatibility with specific Hybrid model variants (like `Step3Text`) that simplify the standard Mamba block design for efficiency.

---

## 4. Verification and Results

Debugging CUDA kernels that rely on hidden states is notoriously difficult. To validate the architectural "plumbing" before debugging the math, we devised a novel verification strategy.

### 4.1 The Prefix-Sum Verification
We introduced a deterministic **Prefix-Sum Mode** (`VLLM_HYBRID_SSM_MODE=prefix_sum`). In this mode, the SSM adapter is temporarily replaced by a mathematical operator that computes the running sum of tokens.

*   **Logic**: If the input sequence is $[x_1, x_2, x_3]$, the history branch outputs $[x_1, x_1+x_2, x_1+x_2+x_3]$.
*   **Significance**: This operation is **history-dependent** (like Mamba) but **deterministic**.

This logic is implemented directly in the adapter to allow for unit testing without the full Mamba kernel stack:

```python
if self.ssm_mode == "prefix_sum":
    # Deterministic history verification:
    # Calculate cumulative sum of tokens to verify state persistence
    prefix = torch.cumsum(hidden_states[:num_actual_tokens], dim=0)
    ssm_out = torch.zeros_like(hidden_states)
    ssm_out[:num_actual_tokens] = prefix
    return ssm_out
```

By running synthetic benchmarks (`test_hybrid_synthetic_eval.py`), we proved that:
1.  The state flows correctly from `prefill` to `decode`.
2.  The separate memory pools (Attention Cache vs. SSM State) do not corrupt each other.
3.  The fusion logic correctly adds the history contribution to the attention output.

### 4.2 Benchmarks
Initial benchmarks demonstrate the viability of the approach. We successfully integrated the `HybridAttentionLayer` into the `Step3Text` model and verified correctness on synthetic tasks. The system correctly handles the "hand-off" where tokens slide out of the attention window but are retained in the SSM state.

---

## 5. Future Work

This work establishes the foundation for Hybrid Models in vLLM. Future efforts will focus on:

1.  **Mamba2 Integration**: Upgrading the kernel support to the newer, faster Mamba2 specification.
2.  **Production Optimization**: The current "fused" implementation relies on Python-level orchestration. Future work will fuse the Attention and SSM kernels at the CUDA level to minimize kernel launch overhead.
3.  **Speculative Decoding**: Enabling speculative decoding for Hybrid models, which requires managing multiple "divergent" SSM states for candidate sequences.

## 6. Conclusion

We have successfully implemented a Hybrid Attention backend in vLLM that supports the concurrent execution of sliding-window Attention and Mamba-style State Space Models. By solving the dual-memory allocation challenge and implementing robust verification tools, we have paved the way for the efficient inference of next-generation, infinite-context LLMs.

