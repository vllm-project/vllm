# P550 Latest Main Dependency Plan

<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright contributors to the vLLM project
SPDX-FileCopyrightText: Copyright (c) 2026 zyz
-->

This note records the extra dependency work required when validating the P550
enablement on top of the latest upstream vLLM `main` branch.

## Context

The original P550 bring-up used a local vLLM source tree that did not require
the latest dependency set from upstream `main`. After rebasing the P550
enablement onto the current upstream `main`, editable installation attempted to
install `blake3`.

On the P550 RISC-V Python environment, `pip` resolved `blake3` from source. The
source build requires `maturin`. The board already has the Rust toolchain
available through system packages, so the least invasive path is to install the
missing Python build helper and then build/install `blake3` inside the existing
P550 virtual environment.

## Plan

1. Keep model weights, virtual environments, board addresses, and credentials
   outside the repository.
2. Install or build only the missing Python dependencies in the P550 virtual
   environment:
   - `maturin`
   - `blake3==1.0.6`
3. Re-run editable installation from the latest-main validation checkout:

   ```bash
   VLLM_TARGET_DEVICE=cpu python -m pip install -e . --no-build-isolation --no-deps
   ```

   `blake3==1.0.6` is used because newer `blake3` source releases require a
   Cargo lockfile format newer than the Rust 1.75 toolchain available on the
   tested P550 environment.

4. Verify that `import vllm` imports from the latest-main validation checkout.
5. Use a RISC-V-only FLA stub during CPU extension builds. The latest upstream
   `csrc/cpu/sgl-kernels/fla.cpp` requires newer PyTorch CPU BLAS internals
   than the validated P550 `torch==2.4.1` environment provides. Qwen2.5 text
   generation does not use these GDN/FLA ops, so the stub keeps symbols
   registered while failing explicitly if those unsupported ops are called.
6. Run Qwen2.5 0.5B validation through the P550 minimal vLLM service.
7. Accept the branch only after 20 short chat sessions return expected answers.

## Acceptance Criteria

- `python -c "import blake3, vllm"` succeeds in the P550 virtual environment.
- `vllm.__file__` points to the latest-main P550 validation checkout.
- The Qwen2.5 0.5B minimal service starts with the CPU backend.
- The active Qwen2.5 path uses the CPU attention native op and does not call
  the RISC-V FLA stubs.
- 20 chat requests complete successfully with expected answers.

## Verified Result

The latest-main validation was performed against upstream `main` commit
`f0204358d9b811bde5320c037236e97b8fb6199d`. The P550 validation checkout built
successfully after installing `maturin` and building `blake3==1.0.6`.

Import validation:

```text
blake3 1.0.6
vllm file /home/ubuntu/vllm_p550dev_validation/vllm/__init__.py
vllm version 0.1.dev17215+gd7c6d51f4.d20260603
vllm _C /home/ubuntu/vllm_p550dev_validation/vllm/_C.abi3.so
```

The Qwen2.5 0.5B service became healthy with:

```text
{"status": "ok", "model": "qwen2.5-0.5b-instruct"}
```

Twenty short chat sessions were executed through `/v1/chat/completions` with
`temperature=0.0` and `max_tokens=24`; all passed:

```text
01. 1+2 -> 3
02. 2*3 -> 6
03. 10-4 -> 6
04. 7+5 -> 12
05. 9-3 -> 6
06. 5+5 -> 10
07. 3+4 -> 7
08. 8-2 -> 6
09. days in a week -> 7
10. square sides -> 4
11. capital of France -> Paris
12. clear sky color -> Blue
13. water wet -> Yes.
14. opposite of hot -> Cold
15. month after January -> February
16. first English alphabet letter -> A
17. humans live on planet -> Earth
18. healthy grass color -> Green
19. larger number, 9 or 4 -> 9
20. animal says woof -> Dog
```

Observed latency remained about 19 to 21 seconds per short request on the P550
scalar RISC-V CPU path.
