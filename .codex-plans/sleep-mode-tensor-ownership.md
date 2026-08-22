# Sleep Mode Tensor Ownership and Recovery

Status: APPROVED

## Goal

Prevent persistent non-weight and non-KV tensors from inheriting destructive
sleep semantics merely because they are created while the `weights` or
`kv_cache` memory pool is active. Preserve actual KV-cache reclamation and
level-2 weight reload behavior.

## Milestones

1. Narrow the KV-cache pool in MRV1 and MRV2 to the backing allocation call;
   build and bind persistent metadata outside the pool.
2. Convert confirmed model and kernel constants to non-persistent buffers on
   their owning modules, and add explicit reset/reload handling for stateful
   Humming and Lamport workspaces.
3. Add deterministic remap-poison tests: allocator-level reproducers that
   demonstrate ordinary tensors inherit tag semantics, plus production
   regressions proving vLLM no longer places persistent metadata in a
   discardable allocation.
4. Run focused unit/GPU tests and compare the final diff against this plan.

## Invariants

- Only discardable KV/Mamba cache backing storage receives `kv_cache`.
- Level-1 host-backed weights recover their values; level-2 parameters remain
  caller-reloaded; registered buffers recover before inference resumes.
- CUDA-graph-visible storage is restored or reset in place.
- Synchronization state is reset to its protocol-defined initial state, not
  assumed to be valid because newly mapped pages often appear zeroed.
- The existing untracked RFC document is not included in code commits.

## Validation

- Fresh-process allocator reproducer with poisoned remaps and stable-pointer
  assertions for ordinary `torch.Tensor`, weight, and fake KV allocations.
- MRV1/MRV2 allocation-boundary tests and targeted attention metadata tests.
- Level-2 buffer/reload tests plus Humming and Lamport reset tests.
- Existing sleep-mode, model-runner, MoE, and platform-gated backend suites.

## Decisions

- Keep metadata resident instead of adding a third allocator tag.
- Use test-only poisoning; do not add production debug configuration.
- Deliver two local signed-off commits: KV boundary/tests, then weight
  recovery/tests. Do not push or create a PR.

## Progress

- [x] Narrow MRV1/MRV2 KV allocation scopes.
- [x] Add KV-tag, level-2 weights-tag, and CUDA-graph poison reproducers.
- [x] Convert confirmed model and MoE constants to sleep-managed buffers.
- [x] Add reload rebinding plus Humming and Lamport reset hooks.
- [x] Add focused allocation-boundary and ownership unit tests.
- [ ] Run GPU poison and platform-specific backend tests on supported hardware.

## Discoveries

- Layerwise reload restores the original non-persistent buffer objects after
  kernel processing. Non-module owners therefore need a post-reload rebinding
  hook in addition to buffer registration, otherwise helper objects retain a
  newly-created tensor that the layer no longer owns.
- The local Windows environment cannot collect the normal vLLM pytest suite
  because its test import path requires Linux-only `uvloop`. Direct CPU tests,
  bytecode compilation, Ruff, and diff checks are used locally; CUDA/ROCm/XPU
  cases remain hardware validation items.
