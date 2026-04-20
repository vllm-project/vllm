# Per-version CRIU + cuda-checkpoint as forkserver replacement

## Goal

Replace Python's `multiprocessing.forkserver` (currently used by exp 22)
with a CRIU + cuda-checkpoint based snapshot-and-restore mechanism that:

1. **Captures everything we're willing to pay for once per (vllm-version,
   python-version, torch-version, cuda-driver, gpu-arch) tuple.** This
   includes the Python import state, all pure-Python module init, and
   the per-process CUDA context.

2. **Does NOT capture model weights, model config, user argv, open
   sockets, or any state that differs between serve invocations.** The
   snapshot is truly per-version, not per-(version × model).

3. **Runs with zero idle cost**, unlike a long-running zygote daemon:
   a snapshot is a directory on disk; restore runs on demand.

4. **Integrates with `vllm serve` transparently**: if a matching
   snapshot exists, restore from it; otherwise fall back to the existing
   forkserver path we already ship.

The savings target: ~15-20 s cold-start reduction on any model size by
amortizing Python imports (10-12 s) + CUDA context init (3-5 s) across
invocations. Composes with the three already-kept wins (parent weight
prefetch, bg torch preload, async forkserver).

## What the snapshot contains vs excludes

### Captured (paid once, reused forever across serve invocations)

| state | cost saved per restore | notes |
|---|---|---|
| Python bytecode for ~500 imported modules | 3-5 s | vllm + torch + transformers + fastapi + stdlib |
| `vllm.env_override` monkey-patches on `torch._inductor.*` | negligible | but correctness-critical |
| Torch op registry, dtype tables, TensorBase metaclasses | 1-2 s | torch internal state |
| `transformers.models.*` class registry | 0.5-1 s | |
| pydantic model schemas pre-compiled | 0.5 s | vllm config classes |
| fastapi + starlette + uvicorn route decorators pre-evaluated | 0.5 s | |
| **CUDA driver context for the process's GPU** | 3-5 s | `cuda-checkpoint --toggle` required |
| NCCL pynccl .so loaded (but **not** initialized) | 0.5-1 s | see note below |

### Explicitly **not** captured

| state | reason |
|---|---|
| Model weights | per-model, 10-100s of GB; would break "one snapshot" invariant |
| Engine config / vllm_config | depends on user argv |
| torch.compile cache artifacts | per-(model, shape, compile_config); lives in `~/.cache/vllm/torch_compile_cache/` already |
| CUDA graph captures | per-(model, compilation_config); per-process by nature |
| KV cache allocation | per-(model, gpu-mem-util) |
| Open sockets/pipes | per-invocation |
| FastAPI app instance | depends on args.disable_fastapi_docs etc. |
| Routes that depend on supported_tasks | depends on model |

### The NCCL handling note

NCCL communicator creation is tied to rank + world_size + master-addr,
all of which are per-invocation. The snapshot captures the *loaded*
pynccl .so and any hardware probe results, but the communicator is
`init_process_group`'d fresh after restore. This saves the ~0.5-1 s of
library loading but leaves the ~2 s of NCCL init per-invocation.

## Invocation flow

### Snapshot creation (once per version triple)

```
┌─────────────────────────────────────────────────────────────┐
│ vllm snapshot create                                         │
│                                                              │
│ 1. Derive snapshot key:                                      │
│    key = hash(vllm_version, py_version, torch_version,       │
│                cuda_driver_version, nvidia_device_arch)      │
│    snap_dir = $VLLM_SNAPSHOT_ROOT/<key>/                     │
│                                                              │
│ 2. Abort if snap_dir/MANIFEST exists (already built).        │
│                                                              │
│ 3. Start snapshot helper:                                    │
│    python -u -m vllm.snapshot.helper                        │
│      → all heavy imports                                     │
│      → monkey-patches applied                                │
│      → torch.cuda.init()  [explicit, opt-in via env var]    │
│      → torch.cuda.synchronize()                             │
│      → writes $snap_dir/.ready                              │
│      → pauses forever in signal.pause()                      │
│                                                              │
│ 4. When .ready appears:                                      │
│    cuda-checkpoint --toggle --pid <helper_pid>               │
│      # GPU memory copied to host pinned memory               │
│                                                              │
│    criu dump -t <helper_pid> \                              │
│      -D $snap_dir/imgs/ \                                   │
│      --external "file[/dev/nvidia*]:ignore" \               │
│      --tcp-established --ext-unix-sk --file-locks \         │
│      --action-script $snap_dir/pre-restore.sh               │
│                                                              │
│    kill $helper_pid                                         │
│                                                              │
│ 5. Write manifest:                                           │
│    $snap_dir/MANIFEST = {                                    │
│      vllm: "0.5.0.post2.dev14263+g219bb5b8c",              │
│      python: "3.12.13",                                      │
│      torch: "2.8.0",                                         │
│      cuda_runtime: "12.8",                                   │
│      cuda_driver: "590.48.01",                               │
│      gpu_arch: "sm_90a",                                     │
│      created_at: "2026-04-20T...",                          │
│      bytes_on_disk: 3_200_000_000                           │
│    }                                                         │
└─────────────────────────────────────────────────────────────┘
```

Total time: ~30-60 s one-time cost. Snapshot file ~2-4 GB.

### Restore on `vllm serve <args>`

```
┌─────────────────────────────────────────────────────────────┐
│ vllm serve ...                                               │
│                                                              │
│ 1. Compute current key from live env.                        │
│                                                              │
│ 2. If $VLLM_SNAPSHOT_ROOT/<key>/MANIFEST exists:             │
│    a. Write argv payload:                                    │
│         /tmp/vllm-resume-<uuid>.json = {                    │
│           argv, env, cwd,                                    │
│           stdin_fd, stdout_fd, stderr_fd  (passed via...)   │
│         }                                                    │
│                                                              │
│    b. Spawn restore:                                         │
│         criu restore -D $snap_dir/imgs/ --shell-job &       │
│         # criu's --action-script hook triggers post-resume   │
│                                                              │
│    c. Wait on restored-process PID:                          │
│         waitpid() → exit code                                │
│         exit with child's return code                        │
│                                                              │
│ 3. Else:                                                     │
│    # Fallback — original code path unchanged                 │
│    cli.main.main()                                           │
└─────────────────────────────────────────────────────────────┘
```

Restore time: ~200-500 ms for CRIU + ~500 ms for cuda-checkpoint resume ≈
**~1 s total**.

### Post-restore execution in the restored process

The restored process was paused in `signal.pause()` during snapshot.
When `criu restore` wakes it, a pre-installed SIGUSR2 handler fires:

```python
# Installed BEFORE pause() so it's part of the snapshot
def _resume_handler(signum, frame):
    payload_path = os.environ["VLLM_RESUME_PAYLOAD"]
    with open(payload_path) as f:
        payload = json.load(f)

    # Replace argv / env / cwd
    sys.argv = payload["argv"]
    os.environ.clear()
    os.environ.update(payload["env"])
    os.chdir(payload["cwd"])

    # Dup2 stdin/out/err from FDs passed via payload's socketpair
    for tgt_fd, src_fd in ((0, payload["stdin_fd"]), ...):
        os.dup2(src_fd, tgt_fd)
        os.close(src_fd)

    # Now execute the normal CLI main from the warmed state
    from vllm.entrypoints.cli.main import main
    main()
    os._exit(0)

signal.signal(signal.SIGUSR2, _resume_handler)
signal.pause()  # <-- snapshot taken here
```

## CRIU + cuda-checkpoint command details

### Dependencies

- `criu` ≥ 3.17 (`apt install criu` or build from source)
- NVIDIA driver ≥ 535 (for `cuda-checkpoint`)
- `cuda-checkpoint` binary from https://github.com/NVIDIA/cuda-checkpoint
- Kernel support for `/proc/<pid>/mem` access (standard)
- One-time `setcap cap_sys_admin,cap_sys_ptrace,cap_net_admin+ep /usr/bin/criu`
  OR run via a setuid wrapper OR run from a user namespace

### Dump sequence (exact order matters)

```bash
# 1. Quiesce all CUDA work and migrate device memory to host
cuda-checkpoint --toggle --pid $HELPER_PID
# at this point, GPU state has been copied to CPU mapped memory,
# and no further CUDA work can happen in the process until toggled back

# 2. CRIU dump while GPU is "parked" on CPU
criu dump \
  --tree $HELPER_PID \
  --images-dir $SNAP_DIR/imgs/ \
  --log-file $SNAP_DIR/dump.log \
  --leave-stopped \
  --external "file[/dev/nvidiactl]:ignore" \
  --external "file[/dev/nvidia-uvm]:ignore" \
  --external "file[/dev/nvidia0]:ignore" \
  --ext-unix-sk \
  --file-locks \
  --tcp-established

# 3. Helper can be killed now; snapshot is complete
kill -KILL $HELPER_PID
```

### Restore sequence

```bash
# 1. CRIU restore recreates the process from the snapshot. At this
#    point the process is suspended; cuda-checkpoint state is still
#    on the CPU side.
criu restore \
  --images-dir $SNAP_DIR/imgs/ \
  --log-file $RESTORE_LOG \
  --shell-job \
  --ext-unix-sk \
  --external "file[/dev/nvidiactl]:map:/dev/nvidiactl" \
  ... (device remap args) \
  &
RESTORE_PID=$!

# 2. Re-activate CUDA state back onto the GPU
cuda-checkpoint --toggle --pid $RESTORE_PID

# 3. Signal the restored process to resume
kill -SIGUSR2 $RESTORE_PID

# 4. The restored process runs main() and eventually exits
wait $RESTORE_PID
```

## Integration points in the vllm codebase

### 1. New module: `vllm/snapshot/`

```
vllm/snapshot/
  __init__.py         (public re-exports)
  helper.py           (the process that gets snapshotted)
  cli.py              (vllm snapshot create|restore|list|drop)
  keying.py           (compute version-triple hash)
  criu_wrapper.py     (subprocess invocations for criu + cuda-checkpoint)
  resume_protocol.py  (argv/env/fd handover)
```

### 2. CLI entry hook: `vllm/entrypoints/cli/main.py`

Currently the `serve` subcommand falls into the `async forkserver
prewarm` logic (exp 22). Add a pre-dispatch hook:

```python
# In main.py's main()
if len(sys.argv) > 1 and sys.argv[1] == "serve":
    try:
        from vllm.snapshot.cli import try_restore_and_dispatch
        if try_restore_and_dispatch():  # returns False if no snapshot
            return  # restored; restored process handled the work
    except Exception as e:
        logger.debug("Snapshot restore failed, falling back: %r", e)
    # Existing forkserver prewarm thread + normal main flow
```

If a snapshot exists AND restore succeeds, we `execvp criu_restore_wrapper`
and never return to this process's main(). If restore fails or
snapshot is absent, fall through to the existing path (exp 22).

### 3. New subcommand: `vllm snapshot`

`vllm snapshot create` — builds a new snapshot for the current version
triple. Prints paths and size.

`vllm snapshot list` — lists all snapshots, their MANIFEST keys, sizes,
creation times.

`vllm snapshot drop <key|--all>` — removes snapshot(s).

`vllm snapshot verify <key>` — does a dry-run restore and immediately
quits, to validate the snapshot is usable on this host.

### 4. Install hook (opt-in)

In `pyproject.toml` or a post-install script, offer:

```bash
vllm snapshot create --if-missing
```

User can run this manually after install. Not automatic because CRIU
requires one-time root setup (setcap).

## Error-handling and failure modes

| failure | detection | recovery |
|---|---|---|
| CRIU not installed | `shutil.which("criu") is None` | silent fallback to normal flow |
| cuda-checkpoint not installed | same | fallback to CRIU-only snapshot (no CUDA state), saves ~10s instead of ~20s |
| Driver version mismatch | MANIFEST check | log warning, rebuild snapshot or fallback |
| GPU arch mismatch | MANIFEST check | fallback; rebuild if user is moving cache between hosts |
| Restore fails mid-way | `criu restore` nonzero exit | kill zombie pid, log, fallback |
| Snapshot dir missing or corrupt | stat | fallback |
| User runs `vllm serve` before snapshot created | `MANIFEST` check | transparent fallback (no error) |

## Per-invocation work that still runs after restore

For full honesty about what the snapshot does NOT cover:

1. **argv parsing** — user's flags must go through argparse.
2. **`engine_args.create_engine_config()`** — HF hub probes for config,
   ModelConfig resolution.
3. **EngineCore subprocess spawn** — child process fork + CUDA init
   (for the child, not the parent! Still need forkserver for the child).
4. **Model weight load** — from disk (or HF cache) per-model.
5. **torch.compile for the specific model shapes** — cache-hit on warm
   via `~/.cache/vllm/torch_compile_cache/`.
6. **CUDA graph capture** — always per-process.
7. **Profile + KV cache allocation** — per-model.
8. **init_app_state** — FastAPI lifespan, serving-class init.
9. **First inference** — triton autotune on kernels if not in cache.

So the snapshot shaves the ~15-20 s of Python import + parent-CUDA-init.
The remaining ~60-90 s of engine work is unchanged.

## Composability with existing kept wins

The three existing keeps (exp 18, 19, 22) continue to apply to the
**restored process**, not the snapshot itself:

- **Parent weight prefetch (exp 18)**: bg thread at `build_async_engine_client_from_engine_args`.
  Runs AFTER restore. Unaffected.
- **BG torch preload (exp 19)**: runs at CLI entry. With a CRIU snapshot,
  torch is ALREADY loaded in the restored process, so this preload is a
  no-op on the restore path. Still needed on the fallback path.
- **Async forkserver (exp 22)**: for the ENGINE subprocess, not the
  APIServer itself. Still needed for the EngineCore fork, runs
  concurrently with engine_args parsing post-restore.

Effectively, the snapshot replaces exp 19 + the APIServer side of the
preload. exp 18 and exp 22 continue to save on the child side.

## Roadmap

### Phase 1 — prototype scaffolding (session-scale, ~1 day)

- Write `vllm/snapshot/helper.py` — the process that imports everything
  and waits.
- Write `vllm/snapshot/cli.py` with `create`, `list`, `drop` subcommands.
  In "dry-run" mode (CRIU not installed), it logs what would happen.
- Write `vllm/entrypoints/cli/main.py` pre-dispatch hook that calls into
  `try_restore_and_dispatch()`.
- Add a `--dry-run-snapshot` env flag that logs all the actions but
  doesn't invoke criu.

### Phase 2 — real CRIU integration (~3-5 days)

- Implement `criu dump` + `criu restore` subprocess calls.
- Handle stdio FD passing via socketpair + SCM_RIGHTS.
- Signal forwarding (SIGINT from client → restored process).
- Argv handover via file in /tmp + `VLLM_RESUME_PAYLOAD` env var.
- Test on a box with CRIU installed and setcap'd.

### Phase 3 — cuda-checkpoint integration (~2-3 days)

- Add `cuda-checkpoint --toggle` bracket around `criu dump` / `criu restore`.
- Handle GPU device permission propagation.
- Validate that restored process can call `torch.cuda.device_count()` etc.

### Phase 4 — polish, install-hook, docs (~2 days)

- Post-install script that offers `vllm snapshot create --if-missing`.
- systemd user-service option (for always-available snapshots on boot).
- Documentation for ops teams.

Total: ~1.5-2 weeks engineering for a shippable implementation.

## Open questions

1. **Multi-GPU**: does `cuda-checkpoint` handle multi-GPU processes?
   Per docs, single-process multi-GPU is supported in driver 535+.
   Needs verification on an 8×H100 box.

2. **Container deployment**: Docker + CRIU requires `--cap-add`
   or privileged. Kubernetes needs custom runtime class. Likely
   won't work out of the box in managed k8s.

3. **Snapshot size**: 2-4 GB baseline, higher if user sets
   `TORCH_ENABLE_ONEDNN_FUSION=1` or similar that pre-compiles more.
   Should MANIFEST track these env vars in the key too.

4. **Restore on a different host**: possible if host has identical
   (driver, CUDA runtime, GPU arch). Useful for cluster deployments.

5. **Concurrent restores**: two `vllm serve` invocations simultaneously.
   Each `criu restore` is independent; should work but untested.

## Non-goals (explicitly)

- Model-weight snapshotting — orthogonal, handled by separate "per-model"
  overlay snapshots if user opts in.
- Torch compile cache snapshotting — already handled via vllm's built-in
  torch_compile_cache directory.
- Cross-machine snapshot distribution — out of scope for v1.
- Supporting rootless CRIU on arbitrary kernels — v1 requires the
  setcap setup step.
