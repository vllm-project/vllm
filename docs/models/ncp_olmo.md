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
avoids a TRTLLM cubin lookup in air-gapped deployments. Native FlashInfer still
requires its matching kernels or a working JIT toolchain. This mixed mode
requires `VLLM_BATCH_INVARIANT=0`: native FlashInfer attention does not implement
batch invariance. Do not force every
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

Pass a public model ID or a local checkpoint directory to the example:

```bash
.venv/bin/python examples/offline_inference/ncp_olmo.py \
  ArchSpace-Collection/NCP_ArchPreview_dolma3_8.9B_Stage2_v1 \
  --prompt "The capital of France is" \
  --max-tokens 32
```

To use FlashInfer sampling while keeping vLLM's default attention selection:

```bash
.venv/bin/python examples/offline_inference/ncp_olmo.py /path/to/pure-hf-model \
  --flashinfer-sampler
```

This option uses top-k 1 with temperature 1 and no per-request seed. FlashInfer
may select any exactly tied maximum, so its output can differ from greedy's
fixed tie-break. Ordinary greedy requests bypass FlashInfer's fused sampler even when
the environment flag is enabled. Without the option, the example uses greedy
generation; `--seed` continues to configure the engine in both modes.

To test the mixed FlashAttention/FlashInfer attention configuration:

```bash
.venv/bin/python examples/offline_inference/ncp_olmo.py /path/to/pure-hf-model \
  --attention-mode mixed-flashinfer \
  --flashinfer-sampler
```

The example uses the default V2 model runner. Request-local HLM state is
initialized, reordered, and released by the V2 model-state lifecycle. The first
version requires `enforce_eager=True` and prefix caching to remain disabled.
CUDA graph capture will be enabled only after request-scoped HLM state has an
explicit graph-safe contract.

## Validation before upstream review

Two NCP-OLMo correctness CI steps run the real-layer unit tests, registry
initialization and generation suite on one H200 GPU each, using the public
Stage 1 and Stage 2 v1 models. `NCP_OLMO_TEST_MODEL` can also select a compatible
local checkpoint. The vLLM environment retains the upstream CUDA test
dependencies. Only the NCP HF reference runs in a separate, HF-only environment
with Transformers 5.16.1 and huggingface-hub 1.33.0: the public checkpoint's
cached-generation implementation requires the earlier attention-mask API.
Neither the public checkpoint nor the global vLLM dependency pins are patched.

Run the same entry point as CI from a CUDA test environment with `uv` available:

```bash
NCP_OLMO_TEST_MODEL=ArchSpace-Collection/NCP_ArchPreview_dolma3_8.9B_Stage1 \
NCP_OLMO_TEST_REVISION=d642693c0efaca7329e544076397105238bda3bf \
NCP_TEST_PYTHON="$PWD/.venv/bin/python" \
  bash .buildkite/scripts/run-ncp-olmo.sh
```

For the Stage 2 v1 CI case, use
`NCP_OLMO_TEST_MODEL=ArchSpace-Collection/NCP_ArchPreview_dolma3_8.9B_Stage2_v1`
and `NCP_OLMO_TEST_REVISION=715aead3f9fd5af3787205b0329c2cc19b92fd5e`.
The revision applies to Hub IDs; local directories are used as supplied.

The script installs the hash-locked `requirements/test/ncp_hf.txt` into
`.ncp-hf` (override with `NCP_OLMO_HF_ENV_DIR`). Regenerate that lock using
`.venv/bin/python tools/pre_commit/compile_ncp_hf.py`; shared dependencies are constrained
by `requirements/test/cuda.txt`, excluding only Transformers and Hub. A
preflight checks matching Torch, torchvision, NumPy, tokenizers and safetensors
versions and rejects a reference environment containing vLLM. `--setup-only`
performs this setup without starting model validation. To invoke individual
tests afterwards, export `NCP_OLMO_HF_PYTHON="$PWD/.ncp-hf/bin/python"`.

The HF parity test generates a fresh reference in an isolated subprocess using
the real public model, tokenizer and processor: BF16, eager attention, cached
greedy generation, 16 new tokens and top-5 scores. Request hashes, a fresh
nonce, Git commit, test/helper source hashes, checkpoint code/config hashes,
runtime versions and input IDs bind the reference to the current run. Greedy
tokens must match exactly, and the original shared logprob checker is also
applied separately at every offset. Historical golden outputs are not used.
The other comparisons generate fresh vLLM baselines: sequential execution for
batching/preemption and FlashAttention 3 for sampler/mixed-backend parity.

MRV2 and `VLLM_GPU_SYNC_CHECK=error` are required throughout. The main parity,
batching, preemption and FlashInfer-sampler tests use
`VLLM_BATCH_INVARIANT=1`. Only the mixed-attention candidate disables batch
invariance, retaining the same exact-token and per-offset top-5 assertions.
Only the sampler comparison uses temperature 1 and top-k 1 without a request
seed to exercise MRV2's real FlashInfer sampling path. A fresh FlashAttention 3
greedy engine scores every exact token prefix of the FI trajectory. Unique
maxima must match exactly; different tokens are accepted only if both are
exactly tied maxima in both score maps, without an epsilon. The original
top-5 checker still runs at every offset. This tests conditional choices, not
whole-sequence equality after a tie. All other comparisons retain their exact
token assertions. Sampler candidate/reference JSON retains the prefixes,
tokens, raw scores and source/model hashes; a separate parity marker records
successful checks. Backend inspection and GPU profiler traces must confirm FlashInfer
kernels actually ran; selecting an environment flag alone is insufficient.

The CI entry point rejects skipped tests, journals failures immediately, bounds
each complete test (including cleanup) to 15 minutes, and bounds the whole
invocation to 60 minutes with a 30-second termination grace. JUnit, environment
versions, reference/parity JSON and GPU traces are retained under
`tests/artifacts/ncp-olmo/` (override with `NCP_OLMO_CI_RESULT_DIR`).
In Buildkite, the step uploads these artifacts on both success and failure.

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
