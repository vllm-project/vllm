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
The loader maps those tensors into vLLM fused QKV and SwiGLU parameters. The
token towers use vLLM paged attention; the HLM retains request-owned dense state.

## Attention backends

The token encoder and decoder instantiate vLLM `Attention` layers. Their KV
cache therefore uses the regular vLLM block table and PagedAttention lifecycle.
The concrete CUDA implementation is not fixed in the model. vLLM selects the
compatible backend for the device and can use FlashAttention decode kernels
when available.

FlashInfer sampling is independent of the attention backend. Enable its
top-k/top-p kernels with `VLLM_USE_FLASHINFER_SAMPLER=1`. The FlashInfer package
and kernel cache must match the vLLM CUDA and PyTorch build. Greedy requests
bypass the fused sampler; top-k 1 sampling can select any exactly tied maximum
and need not match greedy's tie-break.

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
avoids a TRTLLM cubin lookup in air-gapped deployments. Native FlashInfer still
requires its matching kernels or a working JIT toolchain. This mixed mode
requires `VLLM_BATCH_INVARIANT=0`: native FlashInfer attention does not implement
batch invariance. Do not force every
attention group to FlashInfer on SM90 until the upstream sliding-window issue
is resolved.

## Runtime requirements

The model requires the V2 model runner, `enforce_eager=True`,
`enable_prefix_caching=False`, and TP=PP=1. CUDA graph capture, speculative
decoding, and quantized checkpoints are not supported.

The V2 model-state interface manages HLM request creation, reordering, replay,
and cleanup. HLM tensors remain outside the paged KV block manager, so reusing
token-tower KV blocks alone cannot restore a cached prefix's HLM state.

## Offline inference

Use a public model ID or a local pure-HF checkpoint directory:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="ArchSpace-Collection/NCP_ArchPreview_dolma3_8.9B_Stage2_v1",
    trust_remote_code=True,
    dtype="bfloat16",
    enforce_eager=True,
    enable_prefix_caching=False,
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
)
outputs = llm.generate(
    ["The capital of France is"],
    SamplingParams(temperature=0.0, max_tokens=32),
)
print(outputs[0].outputs[0].text)
```
