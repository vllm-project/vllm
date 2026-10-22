# EAGLE-4

EAGLE-4 runs through the external `eagle-harness` plugin. Select
`method="eagle_harness"`; installing the plugin alone does not enable speculation.
The plugin supplies the trained draft model, efficient candidate head, tree
construction, cost policy, attention kernels and acceptance logic. vLLM supplies
scheduling, model execution, sampling and cache allocation through generic hooks.

Both chains and trees are available. Chains use vLLM's causal attention and
rejection sampler. Trees additionally use the plugin's `CUSTOM` attention backend
and linear-attention verification extension. No EAGLE-4 core code is vendored in
vLLM. The plugin and model weights retain their own licenses.

## Installation

Build and install **this vLLM checkout** using its normal CUDA installation
instructions. In the same environment, install the plugin without replacing the
engine with the EAGLE project's pinned fork:

```bash
uv pip install --no-deps eagle-harness==0.1.1
eagle-harness-check
```

For development against a local EAGLE-4 checkout, use
`uv pip install --no-deps -e /path/to/EAGLE-4` instead. The package has numpy,
packaging and torch dependencies, already supplied by the vLLM environment.
Kernel compilation requires a compatible CUDA toolkit and C++ compiler. Do not
install the `[vllm]` extra as a substitute for installing this checkout.

The external package checks the host hooks before registering. If the method is
unknown, check the output of `eagle-harness-check` and the `VLLM_PLUGINS` allowlist.
If an allowlist is set, include `eagle_harness`. A source build's version can be
outside the external package's tested version range even when the hooks match.

## Chain example

This uses the EAGLE-4 candidate head without tree attention:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve Qwen/Qwen3.8-27B \
  --max-model-len 32768 --max-num-seqs 8 --no-enable-prefix-caching \
  --speculative-config '{"method":"eagle_harness","model":"EAGLE-Corp/Qwen3.8-27B-EAGLE4","method_config":{"mode":"static","node_top_k":1,"tree_depth":5,"max_verify_width":6}}'
```

`max_verify_width` includes the root; vLLM's `num_speculative_tokens` excludes it.
Let the plugin derive `num_speculative_tokens` from this configuration.

## Tree example

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve Qwen/Qwen3.8-27B \
  --tensor-parallel-size 4 --max-model-len 32768 --max-num-seqs 8 \
  --no-enable-prefix-caching --attention-backend CUSTOM \
  --linear-backend flashinfer_cutedsl \
  --speculative-config '{"method":"eagle_harness","model":"EAGLE-Corp/Qwen3.8-27B-EAGLE4","method_config":{"mode":"static","node_top_k":4,"tree_depth":14,"max_verify_width":16}}'
```

The plugin sizes draft scratch KV separately from the verification budget. For
example, a top-k of 4 and depth of 14 needs more draft slots than a verification
width of 16. Tree positions, sampled-token rows and accepted cache states are
reordered by the plugin's verification hooks.

Static mode works without cost tables. Without tables the early-stop rule only
avoids steps that cannot change the selected verification tree; it does not price
the benefit of additional work. To enable cost-based stopping, measure costs for
the served model, draft, GPU, TP, dtype and engine, then set `method_config.data_path`:

```bash
eagle-harness calibrate --engine vllm --model Qwen/Qwen3.8-27B \
  --draft EAGLE-Corp/Qwen3.8-27B-EAGLE4 --tp 4 \
  --engine-args '{"linear_backend":"flashinfer_cutedsl"}' \
  --output calibration-vllm-tp4
```

Calibration is not an end-to-end serving benchmark. Do not reuse B300 TP4 cost
tables or narrowing thresholds on another setup without measuring it. Native
target-only draft suspension is not implemented by this integration; the plugin
rejects automatic policies that require it. A zero-draft round can still incur
draft-extension work.

## Supported scope and validation

This integration targets single-node NVIDIA CUDA execution with matched target
and draft TP sizes. Other parallel execution modes (PP, DP, context parallelism
and dual batch overlap) require the registered method to opt in explicitly.
EAGLE-4 0.1.1 does not declare those capabilities. Use the model pair above as the
initial qualification target. Loading another architecture is not evidence of
correctness or acceleration.

Batch-sharded sampling is also rejected unless the method opts in: gathering
sampled tokens alone does not synchronize the verifier's accepted-path state
across TP ranks.

Tree verification requires an unquantized KV cache. The plugin handles GDN and
Mamba2 speculative rows through an extension; ReplaySSM cannot supply the per-row
state it needs and is rejected. Unsupported recurrent layers must not fall back
silently to chain verification. Chains keep the built-in verification path.

Tree batches with grammar-constrained requests only accept the root token for
every request in the batch. Prefix-dependent logits processors use the plugin's
chain fallback. Watermarking has no registered-method support. Consult the
[external request restrictions](https://github.com/eagle-hq/EAGLE-4/blob/main/docs/known-limits.md)
before choosing sampling parameters or enabling additional request features.

The host integration has not been GPU-qualified on the development macOS host.
The external project's published speedups and numerical checks cover its pinned
engine fork, not this checkout. Its current support matrix also records an
unresolved high-concurrency vLLM divergence. Do not infer production precision or
performance from successful registration.

Run the real-weight tests in a CUDA environment with the plugin installed:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 .venv/bin/python -m pytest -v \
  tests/v1/spec_decode/test_method_config.py \
  tests/v1/spec_decode/test_method_registry.py \
  tests/v1/e2e/spec_decode/eagle4/
```

The GPU CI step installs the pinned plugin and runs chain/tree, eager/graph,
greedy/seeded sampling, output logprob, request-slot reuse and multi-token
acceptance checks. These tests use the existing target logprob comparison rules;
they are not a full sampled-distribution or model-quality evaluation.

For performance qualification, compare the same model, precision, requests and
server limits without speculation, with the best existing speculative method,
and with EAGLE-4. Warm each server, repeat independent starts and sweep concurrency
1, 8 and 64. For example, against each separately started server:

```bash
vllm bench serve --backend vllm --model Qwen/Qwen3.8-27B \
  --dataset-name random --random-input-len 512 --random-output-len 256 \
  --num-prompts 128 --max-concurrency 8 --ignore-eos \
  --save-result --result-dir results/eagle4
```

Record TPOT, TTFT, tail latency, throughput, accepted tokens per round and peak
memory. Check quality on the served workload, including stopping, structured
outputs, cache hits and preemption. A larger acceptance length alone does not
establish a speedup. No measured performance claim is made for this integration.

## Host extension API

General plugins register a `SpeculativeMethodSpec` with
`vllm.v1.spec_decode.registry.register_speculative_method`. Built-in names cannot
be overridden. Registered names use Model Runner V2, and explicit
`VLLM_USE_V2_MODEL_RUNNER=0` is rejected. `method_config` is passed to the plugin
and contributes to the compilation cache key. The optional `configure` callback
runs after draft configuration loading and before final budget validation.

`BaseSpeculator.prepare_verify_inputs` stages non-chain positions and metadata
after slot mappings exist. `make_rejection_sampler` can supply a verifier with
the standard sampler output contract. `post_accept` commits accepted KV/state
before request state advances and drafting begins. Built-in speculators retain
no-op/default behavior. `uses_varlen_verify_graphs` additionally requires the
target's backends to support the captured query-length range.

`SpeculativeMethodSpec.linear_attn_verify_extension` names a factory returning
`LinearAttnVerifyExtension` or `None`. GDN speculative rows and Mamba2 decode rows
are passed as typed views through `GDNSpecRows` and `Mamba2SpecRows`. Extensions
write outputs and states in the existing layout and commit the accepted path
through `post_accept`. A GDN-only extension rejects Mamba2 rather than running
chain kernels on a tree.

The generic host hooks were adapted from the Apache-2.0 files in the
[official EAGLE vLLM fork](https://github.com/eagle-hq/vllm/tree/eagle/v0.1.0),
including registry, verification, linear-state and shared-expert stream changes.
The EAGLE-4 algorithm and kernels remain external.
