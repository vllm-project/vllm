# Activation Extraction for SAE Training

This document describes how to extract intermediate activations from vLLM during inference, primarily for Sparse Autoencoder (SAE) training data collection.

[TOC]

## Overview

vLLM's model execution pipeline provides several extension points for extracting intermediate activations. The key locations are:

1. **Auxiliary hidden state extraction** - Built-in mechanism for capturing residual stream activations at specific layers
2. **Layer-level hooks** - Custom extraction points within attention and MLP layers
3. **Forward context** - Runtime access to layer state during forward passes

## Architecture Background

### Activation Flow

In a typical transformer decoder layer (e.g., Llama), activations flow as follows:

```
input_ids → Embedding → [DecoderLayer × N] → Final LayerNorm → LM Head → logits
                              │
                              ├── input_layernorm(hidden_states, residual)
                              ├── self_attn(hidden_states) → attention output
                              ├── post_attention_layernorm(attn_output, residual)
                              └── mlp(hidden_states) → layer output
```

Key tensors at each layer:
- **hidden_states**: Shape `[num_tokens, hidden_size]` - main activation tensor
- **residual**: Accumulated residual connections, same shape as hidden_states

### Key Files

| File | Purpose |
|------|---------|
| `vllm/model_executor/models/llama.py` | Reference model implementation with aux_hidden_state support |
| `vllm/sequence.py` | `IntermediateTensors` data structure |
| `vllm/forward_context.py` | Runtime forward context for layer access |
| `vllm/v1/worker/gpu_model_runner.py` | Model execution and aux_hidden_states processing |

## Built-in Auxiliary Hidden State Extraction

vLLM includes a built-in mechanism for extracting hidden states at specific layers, originally designed for Eagle3 speculative decoding.

### How It Works

The `LlamaModel` (and similar models) maintains an `aux_hidden_state_layers` tuple that specifies which layer indices to capture:

```python
# vllm/model_executor/models/llama.py:392
self.aux_hidden_state_layers = tuple[int, ...]()
```

During forward pass, activations are captured at specified layers:

```python
# vllm/model_executor/models/llama.py:420-428
aux_hidden_states = []
for idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):
    if idx in self.aux_hidden_state_layers:
        aux_hidden_states.append(hidden_states + residual)  # Residual stream
    hidden_states, residual = layer(positions, hidden_states, residual)
```

The model returns a tuple when aux states are collected:

```python
# vllm/model_executor/models/llama.py:437-439
if len(aux_hidden_states) > 0:
    return hidden_states, aux_hidden_states
return hidden_states
```

### Configuration API

Models supporting this feature implement the `SupportsEagle3` protocol:

```python
# Set which layers to extract (0-indexed relative to start_layer)
model.set_aux_hidden_state_layers((2, num_layers // 2, num_layers - 3))

# Get default layers for Eagle3 (model-specific)
layers = model.get_eagle3_aux_hidden_state_layers()
```

### Extending for SAE

To capture more granular activations for SAE training, extend the existing mechanism:

```python
# Example: Capture post-MLP activations in LlamaDecoderLayer.forward()
def forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    capture_mlp_output: bool = False,  # New parameter
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Self Attention
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
    hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

    # Fully Connected
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    mlp_output = self.mlp(hidden_states)

    if capture_mlp_output:
        return mlp_output, residual, hidden_states  # Include pre-MLP for SAE
    return mlp_output, residual
```

## Layer-Level Extraction Points

For fine-grained control, extract activations directly within layer implementations.

### MLP Activations

The `LlamaMLP` class (`llama.py:80-120`) is a key location for SAE:

```python
class LlamaMLP(nn.Module):
    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)  # [tokens, 2*intermediate_size]
        # ↑ Extract here for pre-activation SAE

        x = self.act_fn(gate_up)  # SiLU activation + gating
        # ↑ Extract here for post-activation SAE

        x, _ = self.down_proj(x)
        return x
```

Extraction points for SAE training:
1. **Pre-activation** (`gate_up`): Before SiLU, captures raw projections
2. **Post-activation** (`x` after `act_fn`): After SiLU gating, before down projection
3. **MLP output** (final `x`): Full MLP transformation

### Attention Activations

For attention-based SAE, extract from `LlamaAttention.forward()`:

```python
class LlamaAttention(nn.Module):
    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([...], dim=-1)
        # ↑ Extract Q, K, V projections

        # ... rotary embeddings applied ...

        attn_output = self.attn(q, k, v, ...)
        # ↑ Extract attention output (pre o_proj)

        output, _ = self.o_proj(attn_output)
        return output
```

## Streaming Implementation Strategy

For SAE training data collection, you need to stream activations without blocking inference.

### Option 1: Async Queue with Background Writer

```python
import queue
import threading
from dataclasses import dataclass

@dataclass
class ActivationBatch:
    layer_idx: int
    activation_type: str  # "mlp_pre", "mlp_post", "residual", etc.
    data: torch.Tensor
    metadata: dict  # request_ids, positions, etc.

class ActivationStreamer:
    def __init__(self, output_path: str, max_queue_size: int = 100):
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.output_path = output_path
        self._stop = threading.Event()
        self._writer_thread = threading.Thread(target=self._writer_loop)
        self._writer_thread.start()

    def capture(self, batch: ActivationBatch):
        # Non-blocking put with timeout
        try:
            # Clone and move to CPU to free GPU memory
            cpu_batch = ActivationBatch(
                layer_idx=batch.layer_idx,
                activation_type=batch.activation_type,
                data=batch.data.detach().cpu(),
                metadata=batch.metadata,
            )
            self.queue.put(cpu_batch, timeout=0.1)
        except queue.Full:
            pass  # Drop if queue is full to avoid blocking inference

    def _writer_loop(self):
        while not self._stop.is_set():
            try:
                batch = self.queue.get(timeout=1.0)
                self._write_batch(batch)
            except queue.Empty:
                continue

    def _write_batch(self, batch: ActivationBatch):
        # Implement your storage format (HDF5, safetensors, etc.)
        pass
```

### Option 2: Hook into Model Runner

Modify `vllm/v1/worker/gpu_model_runner.py` to handle extracted activations:

```python
# In GPUModelRunner._execute_model() or similar
output = self.model(...)

if isinstance(output, tuple):
    hidden_states, aux_hidden_states = output
    # Stream aux_hidden_states to your collection pipeline
    if self.activation_streamer is not None:
        for idx, activation in enumerate(aux_hidden_states):
            self.activation_streamer.capture(ActivationBatch(
                layer_idx=self.aux_layer_indices[idx],
                activation_type="residual",
                data=activation,
                metadata={"batch_id": batch_id},
            ))
else:
    hidden_states = output
```

### Option 3: ForwardContext Extension

Use `ForwardContext` to pass extraction configuration:

```python
# In forward_context.py, extend ForwardContext
@dataclass
class ForwardContext:
    # ... existing fields ...
    activation_capture_config: dict | None = None  # New field

# In model forward, check context
from vllm.forward_context import get_forward_context

class LlamaDecoderLayer(nn.Module):
    def forward(self, positions, hidden_states, residual):
        ctx = get_forward_context()

        # ... normal forward ...

        if ctx.activation_capture_config:
            if self.layer_idx in ctx.activation_capture_config.get("layers", []):
                # Capture and store activation
                pass
```

## Integration with Model Runner

The model runner (`vllm/v1/worker/gpu_model_runner.py`) already handles auxiliary hidden states:

```python
# Existing code pattern
if isinstance(model_output, tuple):
    hidden_states, aux_hidden_states = model_output
else:
    hidden_states = model_output
    aux_hidden_states = None
```

To add streaming support:

1. Add configuration for activation capture in `VllmConfig`
2. Initialize streamer in `GPUModelRunner.__init__()`
3. Call streamer after model forward in execution loop
4. Handle cleanup in `GPUModelRunner.shutdown()`

## Memory Considerations

### GPU Memory

- Clone activations before queueing: `activation.detach().clone()`
- Move to CPU immediately: `.cpu()` to free GPU memory
- Consider capturing only every N batches for large-scale collection

### CPU Memory / Disk I/O

- Use memory-mapped files or streaming formats (HDF5, Zarr)
- Compress activations if bandwidth-limited
- Buffer writes to minimize I/O overhead

### Tensor Shapes

Typical activation shapes:
- Residual stream: `[num_tokens, hidden_size]` (e.g., `[2048, 4096]`)
- MLP intermediate: `[num_tokens, intermediate_size]` (e.g., `[2048, 14336]`)
- Per batch memory: ~100-500 MB depending on batch size and layer

## Example: Minimal SAE Data Collection

```python
from vllm import LLM, SamplingParams

# Subclass or patch the model to capture activations
class SAEDataCollector:
    def __init__(self, llm: LLM, output_dir: str, layers: list[int]):
        self.llm = llm
        self.output_dir = output_dir
        self.layers = layers
        self._setup_hooks()

    def _setup_hooks(self):
        # Access internal model and set aux_hidden_state_layers
        model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        model.model.set_aux_hidden_state_layers(tuple(self.layers))

    def collect(self, prompts: list[str], sampling_params: SamplingParams):
        # Run inference - activations captured via aux mechanism
        outputs = self.llm.generate(prompts, sampling_params)
        # Access captured activations through model runner
        return outputs

# Usage
llm = LLM(model="meta-llama/Llama-2-7b-hf")
collector = SAEDataCollector(llm, "./sae_data", layers=[8, 16, 24])
outputs = collector.collect(["Hello world"], SamplingParams(max_tokens=100))
```

## Related Resources

- [Live Activation Steering](STEERING.md) - Modifying activations for steering and intervention
- [Steering API Implementation](API.md) - Exposing steering via the vLLM API
- [Architecture Overview](design/arch_overview.md) - vLLM system architecture
- [HuggingFace Integration](design/huggingface_integration.md) - Model loading and configuration
- Eagle3 implementation in `vllm/v1/worker/gpu/spec_decode/eagle/` - Reference for aux_hidden_states usage
