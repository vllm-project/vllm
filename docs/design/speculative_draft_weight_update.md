# Runtime Draft Weight Update for Speculative Decoding

## Background

### Target-model weight update today

vLLM exposes three weight update paths for RL training loops:

| Path | Entry points | Transport |
| ---- | ------------ | --------- |
| Checkpoint / layerwise | `start_weight_update` → `update_weights` → `finish_weight_update` | IPC or NCCL |
| Direct reload | `reload_weights` | disk / iterator |
| Kernel-format | `start_weight_update(is_checkpoint_format=False)` → `update_weights` | IPC or NCCL |

All three paths operate exclusively on `self.model_runner.model` (the verifier).
No path touches the speculative draft model.

### Draft model architecture

vLLM's `GPUModelRunner` holds two independent `nn.Module` objects:

```text
GPUModelRunner
  ├── self.model               ← verifier (target model)
  └── self.drafter / self.speculator  ← Proposer, contains draft model
        └── .model             ← draft nn.Module (independent parameters)
```

Eagle, Eagle3, and DFlash all follow the same pattern: an independent draft
`nn.Module` is loaded at startup, with `embed_tokens` and `lm_head` shared
by Python reference to the corresponding verifier parameters.  The draft's own
parameters (FC layers, Transformer layers, normalization) are loaded once and
never updated again.

### What breaks in an RL training loop

When the RL trainer updates verifier weights via the standard paths, three
things happen to the draft model:

1. **Independent draft parameters are never updated.** The weight iterator
   produced by the trainer (Megatron → HF name mapping) only covers verifier
   parameter names. `model.load_weights` in all three paths is called on
   `self.model_runner.model` only; the draft's own parameters are untouched.

2. **Shared references may break.** `finalize_layerwise_reload` can rebuild
   parameter objects rather than updating them in-place.  When it does, the
   Python references that connected `draft.embed_tokens` / `draft.lm_head` to
   the verifier are silently severed, and the draft begins reading stale
   embeddings.

3. **Draft parameters are lost after `sleep(level=2)`.** Level-2 sleep calls
   `allocator.sleep(offload_tags=tuple())`, which discards every untagged cumem
   allocation.  Draft parameters live in the untagged pool.  `wake_up` restores
   only the verifier's named buffers, leaving the draft module with invalid GPU
   memory.

The same gap exists in SGLang: the NCCL distributed path never updates draft
weights (sglang#27718, fix under review in sglang#28257).  SGLang's IPC
(colocate) path works only because `EAGLEWorker` overrides
`update_weights_from_tensor` to dispatch to both verifier and draft workers.
vLLM has no equivalent override on any path.

## What this PR adds

Five additions that together close the gap:

1. `Worker.get_draft_model()` — stable accessor for the draft `nn.Module`
2. `Worker.update_speculative_model_weights(weights)` — stable draft weight loader
3. Draft parameter preservation across `sleep(level=2)` / `wake_up`
4. `EAGLEConfig._normalize_dflash_layer_ids()` — remove a DFlash config footgun
5. Spec decode acceptance stats always at INFO level
