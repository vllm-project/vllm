# SiFive P550 Component Bring-up Plan

This plan records the extra work needed when the P550 board cannot use the
standard vLLM CPU dependency path.

## Current State

- Board: Ubuntu 24.04.3 LTS, `riscv64`, 4 CPUs, 9.6 GiB RAM.
- Source: vLLM repository checkout on branch `zp550`.
- CPU ISA: `rv64imafdch_zicntr_zicsr_zifencei_zihpm_zba_zbb_sscofpmf`.
- RVV: no `zvl128b` or `zvl256b`; use `VLLM_RVV_VLEN=0`.
- System dependencies installed: `python3-pip`, `python3-venv`, `ninja-build`, `python3-dev`, `libnuma-dev`, `libtcmalloc-minimal4t64`.
- Official PyPI and PyTorch CPU index do not provide `torch==2.11.0` for `riscv64`.
- A third-party `torch 2.4.1` wheel is available from `https://ext.kmtea.eu/simple`.
- `torch 2.4.1` does not ship `torch/headeronly/util/Exception.h`; the P550 path
  needs a guarded fallback for `STD_TORCH_CHECK` while building scalar CPU code.
- `torch 2.4.1` provides `torch._library.infer_schema.infer_schema` but does not
  export it as `torch.library.infer_schema`; vLLM imports need a guarded fallback
  before `import vllm` can pass.
- The older internal `infer_schema` also needs Python annotation normalization for
  PEP 585 and PEP 604 forms such as `list[int]` and `torch.Tensor | None`.
  It also needs safe resolution of string annotations produced by
  `from __future__ import annotations`.
  Some import-time custom-op registrations also use string default values, which
  the older infer-schema helper cannot encode; strip only unsupported defaults
  while inferring schemas.
- `torch 2.4.1` does not provide `torch._inductor.custom_graph_pass` or
  `torch.distributed._symmetric_memory`; CPU eager smoke should treat these as
  optional compile/distributed accelerators and guard imports.
- `openai-harmony` cannot currently build with the board's Rust 1.75 toolchain;
  offline LLM smoke should not import MCP/Harmony tool-server types unless those
  optional features are requested.
- `torch 2.4.1` on riscv64 does not provide `torch.cpu._is_amx_tile_supported`;
  P550 should shim this to `False` because AMX is x86-only.
- `torch 2.4.1` does not provide newer FP4 dtypes such as
  `torch.float4_e2m1fn_x2`; dtype-keyed metadata should only include dtypes that
  exist in the active Torch build.
- `torch 2.4.1` provides `torch.Tag` but lacks `torch.Tag.flexible_layout`; map
  it to an existing registration tag for the CPU eager smoke path.
- `torch 2.4.1` does not provide `torch.fx._graph_pickler`; eager CPU smoke can
  use a pickle-based import fallback because AOT compile artifact caching is not
  exercised.
- `torch 2.4.1` has an older `torch._dynamo.utils.dynamo_timed` signature; vLLM
  compilation imports need a wrapper for newer label/context-manager usages.
- `torch 2.4.1` does not provide `torch.accelerator`; CPU worker startup needs a
  no-op accelerator shim.
- `xgrammar` has no usable riscv64 wheel in the tested environment; structured
  output imports should not require it unless that backend is selected.
- The P550 board may time out when downloading from Hugging Face; smoke testing
  needs a local tiny random model generator so offline inference does not depend
  on external model downloads.
- Ubuntu's system `pyzmq 24.0.1` has an older `Socket.shadow(int)` API, while
  current vLLM passes a `zmq.Socket` object; install `pyzmq>=26` inside the P550
  venv before running the smoke test.
- CPU attention dispatch does not support very small Llama head dimensions such
  as `head_dim=8`; the local tiny smoke model should stay tiny by using one
  layer and one attention head with `head_dim=64`.

## Decision Gates

1. Prefer an existing binary PyTorch wheel over building PyTorch from source.
   - Accept `torch 2.4.1` only for an exploratory minimum smoke test.
   - Treat any vLLM build/import/runtime failure caused by missing newer PyTorch APIs as a blocker, not as a vLLM regression.
2. If Python dependencies fail only because a wheel download times out, retry with a longer timeout or download the wheel separately.
3. If dependencies fail because no `riscv64` wheel or source build is practical, document the package and decide whether to:
   - remove the feature from the P550 minimum path,
   - vendor a small compatibility shim,
   - or build the missing package from source.
4. Build PyTorch from source only if the `torch 2.4.1` path cannot build/import vLLM and the missing APIs cannot be patched safely.

## Work Plan

1. Install non-PyTorch requirements with the existing `torch 2.4.1` preserved.
   - Generate temporary requirements files that remove only `torch==...` lines.
   - Keep `VLLM_TARGET_DEVICE=cpu` and `VLLM_RVV_VLEN=0`.
   - Record logs as `p550_pip_*.log`.
2. Try a no-dependency editable vLLM build:
   - `VLLM_TARGET_DEVICE=cpu VLLM_RVV_VLEN=0 python -m pip install -e . --no-build-isolation --no-deps`
   - Record logs as `p550_vllm_build.log`.
3. If the build fails on PyTorch API/version checks:
   - identify the exact API or version gate,
   - add the smallest P550-only compatibility patch,
   - keep the patch guarded by architecture or environment so normal platforms remain unchanged.
   - for missing Torch header-only check helpers, prefer `__has_include` guarded
     C++ fallbacks over changing the public Python or model API.
   - for missing Python helpers that exist under older internal Torch modules,
     add import fallbacks only at vLLM's existing helper import sites.
   - centralize schema inference fallbacks in `vllm.utils.torch_schema` so
     custom op registration behavior stays consistent.
   - for optional compile or distributed accelerator modules, add no-op import
     guards that keep eager single-node CPU execution importable.
   - for optional OpenAI serving or MCP typing-only dependencies, move imports
     behind `TYPE_CHECKING` instead of installing Rust-backed packages.
   - for missing CPU feature probes, add conservative Torch shims only when the
     attribute is absent.
   - for structured-output backends, keep imports lazy so plain offline
     generation does not require unavailable optional backends.
4. If import succeeds, run the minimum smoke path:
   - `python -c "import vllm; print(vllm.__version__)"`
   - `bash tools/p550_smoke_test.sh`
5. If the smoke test fails due to model/runtime dependencies:
   - switch to the smallest supported local/Hugging Face model,
   - disable optional features that are not required for offline generation,
   - document any skipped feature in the P550 installation page.
   - if Hugging Face is unreachable, generate a local tiny Llama checkpoint with
     `tools/p550_make_tiny_llama.py` and point `VLLM_P550_MODEL` to it.
   - keep the generated model at a CPU-attention-supported head dimension
     (`hidden_size=64`, `num_attention_heads=1`) instead of shrinking
     `head_dim` below the dispatch table.

## Acceptance Criteria

- The local vLLM checkout is on `zp550` and includes this plan.
- `tools/p550_probe.sh` reports `VLLM_RVV_VLEN=0`.
- vLLM builds or the first non-buildable component is documented with logs.
- If vLLM builds, `tools/p550_smoke_test.sh` generates text from a tiny model.

## PyTorch Source Build Fallback

Only use this fallback after the binary `torch 2.4.1` path is exhausted. A
native PyTorch build on this board is expected to be slow and memory constrained.
Use a separate directory outside the vLLM checkout, add swap before
building if needed, and preserve all build logs. Do not mix a source-built
PyTorch tree into the vLLM checkout.
