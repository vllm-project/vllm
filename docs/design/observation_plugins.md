# Observation Plugins

## Introduction

The vLLM Observation Plugin provides a mechanism to safely and efficiently extract internal hidden states (activations) from the model during inference. This capability is primarily designed to facilitate "Zero-Code" safety monitoring, numerical stability checks, and forensic investigations on production workloads without modifying the core vLLM engine source code.

## Overview

Traditionally, extracting intermediate states from a large language model requires invasive, hardcoded modifications to the model's forward pass, which is unmaintainable and frequently breaks highly optimized inference engines like vLLM.

The Observation Plugin solves this by exposing a stable, plugin-based architecture. Users can inject custom Python logic that executes synchronously during the model's forward pass. Crucially, plugins can process these activations and return `ObservationAction` enum values (such as `CONTINUE` or `ABORT`) allowing them to immediately halt a generation request if unsafe or mathematically invalid states are detected.

## How it works

The architecture relies on PyTorch forward hooks (`register_forward_hook`) injected into the model runner dynamically.

When vLLM processes a batch of prompts or generates tokens:

1. The `ObservationHook` machinery registers hooks on specific transformer boundary layers as requested by loaded plugins.
2. During the forward pass, the hook captures the output tensor resulting from that layer.
3. Because vLLM often computes mixed batches (containing both prefill and decode sequences simultaneously), the hook unpacks and untangles the flattened tensor into per-request segments.
4. The hook invokes the `on_step_batch` callback for every registered `ObservationPlugin`, passing the isolated tensors and a `RequestContext` indicating whether the sequence is currently in the prefill or decode phase.
5. The engine aggregates the responses. If any plugin requests an `ABORT` for a specific request ID, vLLM immediately terminates that sequence, drops its KV cache, and flushes it from the scheduler with an aborted finish reason.

## How to use it

To use an observation plugin, you must inherit from the abstract base class `vllm.plugins.observation.ObservationPlugin`.

```python
from typing import Dict, List
import torch
from vllm import LLM
from vllm.plugins.observation import (
    ObservationPlugin, ObservationResult, ObservationAction, RequestContext
)

class NaNDetectorPlugin(ObservationPlugin):
    def get_observation_layers(self) -> List[int]:
        # -1 represents the final layer before the lm_head
        return [-1]

    def on_step_batch(
        self, 
        batch_hidden_states: Dict[int, torch.Tensor], 
        request_contexts: List[RequestContext]
    ) -> List[ObservationResult]:
        results = []
        for ctx in request_contexts:
            # Look at the final layer tensor for this specific request
            start = ctx.batch_offset
            end = start + ctx.num_tokens
            tensor = batch_hidden_states[-1][start:end]
            
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                results.append(ObservationResult(action=ObservationAction.ABORT))
            else:
                results.append(ObservationResult(action=ObservationAction.CONTINUE))
        return results

# Register the plugin during LLM initialization
llm = LLM(
    model="facebook/opt-125m", 
    observation_plugins=[NaNDetectorPlugin()]
)
```

## The Workflow

1. **Instantiation**: The user creates a concrete plugin instance, loading any necessary external state (like projection vectors or classifier weights).
2. **Registration**: The plugin instance is passed to `LLM(observation_plugins=[...])` or loaded dynamically.
3. **Execution**:
   - For every request, `on_request_start` is invoked.
   - For every forward pass step, `on_step_batch` is invoked with layer tensors.
   - If an `ObservationAction.ABORT` is returned, generation halts.
   - Once a request finishes (naturally or aborted), `on_request_complete` is invoked to clean up local plugin state.

## Limitations

- **Decode Phase Overhead**: Inspecting every single token generated during the decode phase requires breaking vLLM's CUDA Graphs (`enforce_eager=True`). This fundamentally degrades overall engine throughput. It is highly recommended to design plugins that *only* observe the prefill phase, which is generally memory bandwidth bound and suffers virtually zero latency degradation from PyTorch hooks.
- **Architecture Support**: Currently, determining the actual `nn.Module` string paths to hook into generic layers (e.g., layer `-1`) is reliant on hardcoded architecture mapping within vLLM. Highly exotic custom model architectures may not resolve layers automatically.
- **MoE Interaction**: Mixture-of-Expert (MoE) architectures are complex. Hooks are placed at the end of the transformer block, capturing the combined, post-routing aggregated state, not the individual expert internal states.

## Debugging Suggestions

1. **Check the Layers**: If your plugin is not receiving data, ensure `get_observation_layers()` returns mathematical integers (e.g. `[0, 15, -1]`). If the model has fewer layers than requested, initialization will fail.
2. **Phase Awareness**: If you are trying to intercept toxic prompts, ensure you are not accidentally writing logic that only runs during decode. The `RequestContext.is_prefill` boolean differentiates these inputs.
3. **Batch Splitting**: Remember that `batch_hidden_states` in `on_step_batch` is a flattened concatenation of all requests currently executing on the GPU. You *must* iterate over `request_contexts` and slice the tensor chunks according to `ctx.batch_offset` and `ctx.num_tokens`.
4. **CUDA Graphs**: If you notice severe performance degradation, verify if your plugin implements `observe_decode = True`. If it does, vLLM will be forced to disable CUDA graphs to allow the PyTorch hooks to fire dynamically.
