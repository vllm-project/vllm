# Genesis — `vllm/_genesis/` package (engineering README)

> **This is the developer / contributor README for the `vllm/_genesis/` Python
> package.** It documents package layout, design principles, programmatic
> usage, testing, and troubleshooting at the *code* level.
>
> **Looking for the operator-facing docs?** → [`../../README.md`](../../README.md)
> (live PROD benchmarks, install one-liner, quick start, all 10 chart embeds,
> reference configs).
> **Looking for command reference?** → [`../../docs/COMMANDS.md`](../../docs/COMMANDS.md).
> **Looking for the per-release notes?** → [`../../CHANGELOG.md`](../../CHANGELOG.md)
> (public-facing) and [`CHANGELOG.md`](CHANGELOG.md) (engineering log,
> per-commit detail).

**Modular drop-in architecture for Genesis vLLM patches — v7.0 → v7.72 (current).**

Replaces the legacy monolithic `patch_genesis_unified.py` (v5.14.1) with a clean
package structure that:

- Works on NVIDIA CUDA / AMD ROCm / Intel XPU / CPU with graceful skip
  (philosophy: **МЫ ЧИНИМ, НЕ ЛОМАЕМ** — we fix, we don't break)
- Follows TDD discipline (tests first, implementation second) — **1858 tests across the package, 73 skipped, 0 failures** (full local sweep, 2026-05-05)
- Is upstream-ready — kernels can be submitted as vLLM PRs directly
- Self-documents via `genesis doctor` and the curated model registry

## Where to find the things this README used to contain

| Topic | Now lives at |
|:---|:---|
| Live PROD benchmarks (TPS / VRAM / latency / 10 charts) | [`../../README.md`](../../README.md) §"Live PROD benchmarks" |
| `vllm` pin requirement | [`../../README.md`](../../README.md) §"Minimum vLLM pin" |
| Operator install + quick start | [`../../README.md`](../../README.md) §"Installation" + §"Quick start" |
| Command reference (every CLI command) | [`../../docs/COMMANDS.md`](../../docs/COMMANDS.md) |
| Per-release notes | [`../../CHANGELOG.md`](../../CHANGELOG.md) |
| Engineering log (per-commit, per-A/B) | [`CHANGELOG.md`](CHANGELOG.md) |

The rest of *this* file is engineering content that has no equivalent in the
operator docs.

## Package layout

```text
vllm/_genesis/
├── __init__.py              Public API entry
├── dispatcher.py            PATCH_REGISTRY (123 entries) + A3/D2 validator
├── guards.py                Canonical vendor/chip/model/dep detection
├── prealloc.py              GenesisPreallocBuffer framework
│
├── compat/                  Unified compat / UX / diagnostic layer (v7.63.x → present)
│   ├── cli.py               `genesis <subcommand>` entry point (13 commands)
│   ├── doctor.py            `genesis doctor` — full diagnostic
│   ├── init_wizard.py       `genesis init` — interactive setup
│   ├── version_check.py     vllm/torch/cuda/triton/driver range matching
│   ├── predicates.py        AND/OR/NOT applies_to evaluator
│   ├── lifecycle.py         patch lifecycle state machine
│   ├── preflight_checks.py  Boot-time quant-arg validator (PN60)
│   ├── env_flag_guard.py    Typo-detection for GENESIS_* env vars
│   ├── schema_validator.py  PATCH_REGISTRY entry shape check
│   ├── gpu_profile.py       16-GPU spec database + per-patch predicate engine
│   ├── model_detect.py      Hybrid + GDN detection
│   ├── config_detect.py     Quant scheme detection
│   ├── models/
│   │   ├── registry.py      SUPPORTED_MODELS dict (5 entries)
│   │   ├── pull.py          HF download + verify + launch script gen
│   │   └── list_cli.py      `genesis list-models`
│   └── fingerprints/        Reference benchmark JSONs for ablation
│       └── rtx_a5000_x2_qwen3_6_27b_int4_v794.json
│
├── kernels/                 Genesis-original Triton kernels
│   ├── ffn_intermediate_cache.py   PN12 — Cliff 1 fix
│   ├── p67_multi_query_kernel.py   P67 — TQ K+1 verify
│   ├── pn26_sparse_v.py            PN26b — sparse-V SM86 kernel
│   ├── streaming_gdn_driver.py     PN59 — Cliff 2b breakthrough
│   ├── gdn_scratch_pool.py         GdnScratchPool (future infra; PN59 ships windowing)
│   ├── pn40_dflash_omnibus.py      PN40 — DFlash multi-sub-kernel
│   ├── marlin_tuning.py            P17/P18/PN64 per-arch Marlin tuning
│   ├── block_verify_sampler.py     P71 — Sun 2024 ICLR (A4-hardened)
│   ├── router_softmax.py           P31
│   ├── dequant_buffer.py           P22/P26
│   ├── gdn_dual_stream.py          P7
│   ├── gdn_core_attn_manager.py    P28
│   ├── fla_kkt_buffer.py           P39a
│   └── ...
│
├── wiring/                  Text-patch wiring (~80 modules in 11 dirs)
│   ├── text_patch.py        TextPatcher framework + result_to_wiring_status helper
│   │                        + MultiFilePatchTransaction (true rollback as of G-POST-08)
│   ├── rebind.py            runtime class-method rebind helpers
│   ├── spec_decode/         P56-P79c, P82-83, P86, P94, PN8-9, PN22, PN40
│   ├── structured_output/   P59, P61/61b, P62, P64, P68/69, PN56, PN58, PN66
│   ├── perf_hotfix/         P98-101, PN51, PN52, PN55, PN57, PN67
│   ├── compile_safety/      P72, P74, P78
│   ├── kv_cache/            P84-85
│   ├── kernels/             P81, P87, P91, PN14
│   ├── hybrid/              P95, P103, PN11-13, PN50, PN59
│   ├── memory/              PN8 inputs_embeds skip, PN61 vl-loader, PN62 vit-skip
│   ├── middleware/          PN16 lazy_reasoner, PN65 access_log
│   ├── loader/              PN61
│   ├── streaming/           chunked-prefill helpers
│   └── legacy/              P1-P55 pre-PATCH_REGISTRY series
│
├── middleware/              Request-level pre-engine logic
│   ├── lazy_reasoner.py            PN16 — hybrid policy (variants 1+3+5)
│   ├── long_ctx_tool_adherence.py  P68/P69
│   └── response_cache_middleware.py
│
├── patches/                 Orchestration + upstream tracking
│   ├── apply_all.py         Boot-time orchestrator
│   └── upstream_compat.py   PR marker registry (auto-retire on merge)
│
├── utils/
│   └── gdn_composability.py composes_with / conflicts_with matrix evaluator
│
├── configs/
│   └── moe_tuning/          Pre-tuned MoE Triton configs (community-contributed,
│                            see configs/moe_tuning/README.md for honest A5000 caveat)
│
└── tests/                   pytest TDD suite — 1858 pass, 73 skipped, 0 failures
    ├── conftest.py          AST-scans test files for `import torch`,
    │                        auto-applies `requires_torch` skip on CPU-only env
    ├── compat/              CLI / doctor / lifecycle / models / plugins / telemetry
    ├── integration/         Cross-module tests (composability, full-stack)
    ├── oracle/              Reference-output regression tests
    └── test_*.py            Per-patch TDD files (1 per major patch)
```

## Design principles

### 1. МЫ ЧИНИМ, НЕ ЛОМАЕМ (We fix, we don't break)

Every patch uses a 5-layer defensive guard: file exists → idempotency marker
→ upstream merge check → vendor/chip compat → model/backend arch.
If any layer fails, patch returns `("skipped", reason)`, never raises.

### 2. Three-source truth tracking

Verify each patch against three vLLM sources simultaneously:

- Release tag (e.g. `v0.20.0`)
- `main` branch HEAD
- `nightly` docker image

Patch ready for deploy only when all three are green. See
`compat/version_check.py` for the matrix evaluator.

### 3. TDD discipline

For each new kernel module:

```
1. Write test → run → see RED (ImportError or assertion failure)
2. Implement minimal code → run → see GREEN
3. Refactor keeping GREEN
```

Genesis enforces this via `test_apply_all_dispatcher_sync.py` (every
`apply_patch_*` function must have a corresponding `PATCH_REGISTRY` entry).

### 4. Canonical vendor detection

No duplication of detection logic across patches. All helpers live in
`guards.py`:

- `is_nvidia_cuda()` — strict NVIDIA (not ROCm trap)
- `is_sm_at_least(major, minor)` — compute capability gate
- `is_rocm_cdna3()` — MI300X/MI325X detection
- `is_model_arch(cfg, "Qwen3")` — architecture match
- `has_turboquant_support(cache_dtype)` — backend gate

### 5. Status-helper centralization (B2 / 2026-05-05 audit)

Every `wiring/*/patch_*.py` module returns `(status, reason)` from `apply()`.
The helper `result_to_wiring_status()` in `wiring/text_patch.py:251`
centralizes the mapping `(TextPatchResult, TextPatchFailure) → ("applied" |
"skipped" | "failed", reason)` so SKIPPED never gets masked as APPLIED.

## Programmatic usage

### As container entrypoint

```yaml
# docker-compose.yml
services:
  vllm-server:
    image: vllm/vllm-openai:nightly
    volumes:
      - ./vllm/_genesis:/usr/local/lib/python3.12/dist-packages/vllm/_genesis:ro
    entrypoint: [
      "/bin/bash", "-c",
      "python3 -m vllm._genesis.patches.apply_all && exec vllm serve \"$@\"",
      "--"
    ]
```

For the 6 reference launch scripts that ship in the repo see [`../../scripts/`](../../scripts/) and the [Reference configs](../../README.md#-reference-configs) table in the root README.

### Standalone for testing

```bash
# Apply all patches (dry inspection — does NOT actually patch the install
# unless run inside the vllm install). Useful for diagnostic output.
python3 -m vllm._genesis.patches.apply_all

# Expected output: structured boot summary with system info + per-category
# tables of APPLIED / SKIPPED / FAILED. See genesis doctor for the same
# information without booting vLLM.
```

### From Python code

```python
from vllm._genesis.guards import is_nvidia_cuda, is_ampere_consumer
from vllm._genesis.kernels.router_softmax import router_softmax
from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB

# Platform-aware code
if is_nvidia_cuda() and is_ampere_consumer():
    print("Running on A5000-class GPU")

# Drop-in replacement
weights = router_softmax(gating_output)  # instead of torch.softmax(...)

# Safe pre-allocation
buf = GPB.get_or_create(
    namespace="my_kernel_scratch",
    shape=(4, 128),
    dtype=torch.bfloat16,
    device="cuda",
)
slice_view = GPB.slice_to(buf, 2)  # view, pointer-stable
```

## Testing

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests (1858 expected pass + 73 skipped on CPU-only env)
cd /path/to/genesis-vllm-patches
python3 -m pytest vllm/_genesis/tests/ --no-header -q

# With coverage
python3 -m pytest vllm/_genesis/tests/ --cov=vllm._genesis --cov-report=term-missing

# Only CPU tests (skip GPU-required) — useful on macOS dev box
python3 -m pytest vllm/_genesis/tests/ -m 'not cuda_required'

# One specific patch's TDD
python3 -m pytest vllm/_genesis/tests/test_pn59_streaming_gdn.py -v
python3 -m pytest vllm/_genesis/tests/test_pn65_access_log.py -v
python3 -m pytest vllm/_genesis/tests/test_multi_file_transaction.py -v
```

> **Note about pytest namespace package import:** Genesis uses a PEP 420
> namespace package layout under `vllm/_genesis/`. Running `pytest` directly
> from the repo root with no `pythonpath` flag works on Linux but fails on
> macOS with `ModuleNotFoundError`. Always use
> `python3 -m pytest vllm/_genesis/tests/` (with the `-m`) to pick up the
> repo root as the search path. Documented in `tests/conftest.py`.

## Troubleshooting

### "Patches all skipped — anchor not found"

Genesis text-patches edit specific upstream files at known anchors. If the
upstream pin drifted, anchors won't match and the patcher will silent-skip.
Symptom: structured boot summary shows many `SKIPPED (anchor not found)`.

**Fix:** roll vLLM back to the pinned commit (see
[`../../README.md`](../../README.md) §"Minimum vLLM pin"), or open an issue
with your `vllm --version` output.

### "Boot log says X errors in Genesis registry"

The boot validator runs on every `apply_all.run()` and surfaces shape or
dependency issues in `PATCH_REGISTRY`. Errors look like:

```text
[ERROR:genesis.apply_all] [Genesis registry] PXX: <message>
```

Common causes: a contributor added an entry with malformed `env_flag`, a
typo in a known field name, a `requires_patches` referencing a non-existent
ID, or a deprecated patch missing `superseded_by`. Run `genesis
validate-schema` to see the same issues outside the boot flow.

### "Plugin not loading"

If a third-party Genesis plugin isn't being discovered:

1. Confirm `GENESIS_ALLOW_PLUGINS=1` is set (Genesis loads zero foreign
   code by default — opt-in is required).
2. Confirm the plugin is `pip install`-ed (`pip show <plugin-name>`).
3. Run `genesis plugins list` — discovered plugins will be listed even if
   their env_flag is unset.
4. See [`../../docs/PLUGINS.md`](../../docs/PLUGINS.md) for the full plugin
   guide and reference example at
   [`../../tools/examples/genesis-plugin-hello-world/`](../../tools/examples/genesis-plugin-hello-world/).

### "How do I tell which patches are active?"

```bash
genesis doctor --patches              # full apply matrix without booting vllm
genesis explain P67                   # one patch in detail
genesis lifecycle-audit               # patches near retirement / broken anchors
```

The structured boot summary (printed by `apply_all` at vllm boot) also
gives a per-patch one-line entry with applied / skipped (with reason) /
failed annotations.

## Migration status

As of v7.65 (2026-05-02) the historical pre-dispatcher patches (P1-P46)
have been promoted to first-class `PATCH_REGISTRY` entries with
`lifecycle: legacy` (minimal metadata by design — they predate the
registry). All `apply_patch_*` functions now have a corresponding registry
entry — pinned by `tests/test_apply_all_dispatcher_sync.py`.

The skeleton kernel modules listed in earlier README revisions (P22 / P7 /
P17 / P1 dequant_buffer etc.) all shipped under their `wiring/` text-patches;
the kernel-side rewrites are tracked in [`CHANGELOG.md`](CHANGELOG.md).

## Upstream attribution

Genesis stands on prior work; full credit list at [`../../docs/CREDITS.md`](../../docs/CREDITS.md).
Highlights:

- **vLLM core team** ([@WoosukKwon](https://github.com/WoosukKwon),
  [@zhuohan123](https://github.com/zhuohan123),
  [@robertgshaw2-redhat](https://github.com/robertgshaw2-redhat),
  [@bnellnm](https://github.com/bnellnm),
  [@simon-mo](https://github.com/simon-mo)) — for the engine + responsive
  community
- **DeepSeek-V3 team** — fp32 router upcast pattern, basis for Patch 31
- **[@JartX](https://github.com/JartX)** — TurboQuant author (vllm#39931
  hybrid TQ supersedes Genesis P5+P9 on merge)
- **[@noonghunna](https://github.com/noonghunna)** — Cliff 2 reproducer
  suite + cross-rig culture
- **[@Quentin-M](https://github.com/Quentin-M)** — P64 sub-patch E +
  rapid bug-class triage
- **[@apnar](https://github.com/apnar)** — first-ever real RTX 5090
  sm_120 datapoints
- **[@tfriedel](https://github.com/tfriedel)** — cross-engine framing
  (vLLM ⇄ llama.cpp) that keeps Genesis honest about scope
- **[@allanchan339](https://github.com/allanchan339)** — bundled
  `qwen3.6-enhanced.jinja` chat template
- **[@thc1006](https://github.com/thc1006)** — spec-decode acceptance
  benchmarking
- **[@MidasMining](https://github.com/MidasMining)** — TurboQuant cross-rig
  confirms (H20, R6000 Blackwell)
- **[@webcodes-cz](https://github.com/webcodes-cz)** — OpenAI tool-call
  validator
- **[@jhsmith409](https://github.com/jhsmith409)** — `llama-cpp-turboquant`
  cross-engine port

Full PR attribution in each kernel module docstring + [`CHANGELOG.md`](CHANGELOG.md).

## Author

**Sandermage (Sander) — Barzov Aleksandr**
Ukraine, Odessa
GitHub: [@Sandermage](https://github.com/Sandermage)
Project: [genesis-vllm-patches](https://github.com/Sandermage/genesis-vllm-patches)

---

*Part of Genesis vLLM Master Plan v7.0 (2026-04-24), iterated through v7.72 (2026-05-05).*
