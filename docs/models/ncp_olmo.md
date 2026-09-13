# NCP-ArchPreview

NCP-ArchPreview is a decoder-only language model with token-level encoder and
decoder towers around a chunk-level high-level model (HLM).

## Checkpoint contract

The implementation accepts pure Hugging Face checkpoints with:

- `architectures: ["NCPOlmo3ForCausalLM"]`
- `model_type: "ncp_olmo3"`
- `weight_key_format: "huggingface_state_dict"`
- `conceptlm_chunk_merge_method: "meanpooling"`
- `conceptlm_layer_norm_option: "normed_add"`
- `conceptlm_hlm_attention_mode: "backbone_window"`
- `conceptlm_v22_vq_merge_mode: "raw_logits"`
- `conceptlm_v21_dd_self_dd_mode: "dd"`
- `qk_layernorm: true` with full-hidden Q/K normalization
- `position_embedding_type: "rope"` and `rotary_interleaved: false`
- `conceptlm_shift_feature: true` and `conceptlm_hlm_ffn_hidden_size: null`
- split `q_proj`, `k_proj`, `v_proj` tensors
- split `gate_proj` and `up_proj` tensors

Native Megatron/DCP state-dict keys are intentionally not accepted. Convert a
training checkpoint into this standalone HF schema before loading it with vLLM.

All non-DFlash checkpoints in the public
[NCP-ArchPreview collection](https://huggingface.co/collections/ArchSpace-Collection/ncp-archpreview)
are supported, including Stage 1 (and its intermediate checkpoints) and Stage 2
v1/v2/v3. DFlash checkpoints require the separate DFlash integration.
The model-registry initialization test uses the Stage 2 v1 configuration with
a contract-preserving layer-count override and vLLM's dummy loader; a separate
dummy-weight checkpoint is not required.

The loader maps those tensors into vLLM fused QKV and SwiGLU parameters. The
weight-loading test checks every tensor in the NCP-ArchPreview graph shared by
the published Stage 1 and Stage 2 checkpoints.

## Attention backends

The token encoder and decoder instantiate vLLM `Attention` layers. Their KV
cache therefore uses the regular vLLM block table and PagedAttention lifecycle.
The concrete CUDA implementation is not fixed in the model. vLLM selects the
compatible backend for the device and can use FlashAttention decode kernels
when available.

FlashInfer sampling is independent of the attention backend. Enable its
top-k/top-p kernels with `VLLM_USE_FLASHINFER_SAMPLER=1` or use the offline
example's `--flashinfer-sampler` option. The FlashInfer package and AOT kernel
cache must match the vLLM CUDA and PyTorch build.

NCP-ArchPreview interleaves full and sliding-window attention. On Hopper (SM90), vLLM
currently rejects FlashInfer attention for sliding-window groups because that
combination is not reliable. A supported mixed configuration keeps
sliding-window groups on FlashAttention 3 and assigns only full-attention
groups to FlashInfer:

```python
attention_config = {
    "backend": "FLASH_ATTN",
    "flash_attn_version": 3,
    "use_trtllm_attention": False,
    "backend_per_kind": {
        "full_attention": "FLASHINFER",
        "sliding_window": "FLASH_ATTN",
    },
}
```

Disabling TRTLLM attention in this configuration selects native FlashInfer and
avoids a remote cubin lookup in air-gapped deployments. Do not force every
attention group to FlashInfer on SM90 until the upstream sliding-window issue
is resolved.

The HLM currently retains request-owned dense state. Its add, remove, reorder,
and replay lifecycle is connected to the V2 runner through the model-state
interface, while its tensors remain outside the vLLM block manager. Prefix
caching is rejected because a cached token prefix does not yet carry the
corresponding HLM state snapshot. Enabling it requires a cache payload for the
pending encoder fragment, per-layer HLM K/V and raw states, and predicted
concepts at the same token boundary, together with copy-on-write ownership when
requests share that prefix. Token-tower KV blocks alone are insufficient.
Speculative decoding, tensor parallelism, and pipeline parallelism remain
disabled in this first implementation. Quantized checkpoints are also rejected
until their fused weight mappings and numerical parity are validated.

## Offline inference

```bash
python examples/offline_inference/ncp_olmo.py /path/to/pure-hf-model \
  --prompt "The capital of France is" \
  --max-tokens 32
```

To use FlashInfer sampling while keeping vLLM's default attention selection:

```bash
python examples/offline_inference/ncp_olmo.py /path/to/pure-hf-model \
  --flashinfer-sampler
```

To test the mixed FlashAttention/FlashInfer attention configuration:

```bash
python examples/offline_inference/ncp_olmo.py /path/to/pure-hf-model \
  --attention-mode mixed-flashinfer \
  --flashinfer-sampler
```

The example uses the default V2 model runner. Request-local HLM state is
initialized, reordered, and released by the V2 model-state lifecycle. The first
version requires `enforce_eager=True` and prefix caching to remain disabled.
CUDA graph capture will be enabled only after request-scoped HLM state has an
explicit graph-safe contract.

## Validation before upstream review

The registry test uses the public Stage 2 v1 checkpoint with small dummy-weight
overrides. Before opening a pull request, run:

- model registry import and dummy-weight initialization
- full pure-HF checkpoint loading
- Hugging Face versus vLLM logprob and greedy-output parity
- mixed-length continuous batching with more queued requests than
  `max_num_seqs`, in-flight slot refill/reordering, and request cleanup
- chunked-prefill and preemption/recompute tests
- GPU confirmation of the selected PagedAttention/FlashAttention backend
- FlashInfer-sampler parity and mixed-backend parity on a sliding-window model
