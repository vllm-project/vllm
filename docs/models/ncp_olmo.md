# NCP-OLMo

NCP-OLMo is a decoder-only language model with token-level encoder and decoder
towers around a chunk-level high-level model (HLM).

## Checkpoint contract

The first implementation accepts pure Hugging Face checkpoints with:

- `architectures: ["NCPOlmo3ForCausalLM"]`
- `model_type: "ncp_olmo3"`
- `weight_key_format: "huggingface_state_dict"`
- split `q_proj`, `k_proj`, `v_proj` tensors
- split `gate_proj` and `up_proj` tensors

Native Megatron/DCP state-dict keys and the historical
`ConceptLMV22VQForCausalLM` export identity are intentionally not accepted.
Convert a training checkpoint into this standalone HF schema before loading it
with vLLM.

Public reference checkpoints are available for
[Stage 1](https://huggingface.co/ArchSpace-Collection/NCP_ArchPreview_dolma3_8.9B_Stage1)
and
[Stage 2 v1](https://huggingface.co/ArchSpace-Collection/NCP_ArchPreview_dolma3_8.9B_Stage2_v1).
The model-registry initialization test uses the Stage 2 v1 configuration with
a contract-preserving layer-count override and vLLM's dummy loader; a separate
dummy-weight checkpoint is not required.

The loader maps those tensors into vLLM fused QKV and SwiGLU parameters. The
weight-loading test checks every tensor in the NCP-OLMo Stage3 graph.

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

NCP-OLMo interleaves full and sliding-window attention. On Hopper (SM90), vLLM
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

The architecture and checkpoint contract are based on the Concept OLMo
reference implementation in `Liu-yuliang/concept_olmo`. The pull request must
identify any code adapted from that repository and preserve the applicable
source notices.
