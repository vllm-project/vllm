# Unified Communication Abstraction Layer (`unified_comm`)

A pluggable, device-agnostic communication layer for vllm-hust.

It sits between vLLM's `GroupCoordinator` and the underlying backends
(NCCL / HCCL / future MetaX CCL) and is **strictly opt-in**: when the
environment variable `UNIFIED_COMM_ENABLED` is unset or `0`, this code
is **never imported** at runtime and the default vLLM code path is
unaffected.

The design has been validated end-to-end on Ascend 910B2 with TP=2
inference (Qwen2.5-7B-Instruct, random-online, 200 prompts, 0 failures).

---

## Architecture

```text
                +-----------------------------------+
                |  GroupCoordinator (vllm/upstream) |
                |  _all_reduce_out_place(...)       |
                |  _all_gather_out_place(...)       |
                |  _reduce_scatter_out_place(...)   |
                |  broadcast(...)                   |
                +-----------------+-----------------+
                                  | hook: _get_unified_adapter()
                                  v
                       +----------+----------+         (lazy build,
                       |  UnifiedCommAdapter |          one per group)
                       +----------+----------+
                                  |
   Layer 4 ------------ +---------+---------+ ----- selects
   Strategy             |  CommStrategy     |        algorithm + link
                        +---------+---------+
                                  |
   Layer 3 ------------ +---------+---------+ ----- KV / EC / Weight
   TransferPlane        |  TransferPlane    |        transfers reuse
                        +---------+---------+        the same group
                                  |
   Layer 2 ------------ +---------+---------+ ----- AllReduce / AllGather
   CollectiveOps        |  CollectiveGroup  |        / Broadcast / RS
                        +---------+---------+
                                  |
   Layer 1 ------------ +---------+---------+ ----- NCCL / HCCL / ...
   CommBackend          |  CommBackend      |        (registry pattern)
                        +-------------------+
```

| Layer | Responsibility | Key types |
| --- | --- | --- |
| 1 - `CommBackend` | hide library-level differences | `CommBackend`, `NCCLBackend`, `HCCLBackend`, `CommBackendRegistry` |
| 2 - `CollectiveOps` | device-agnostic collectives, lifecycle | `CollectiveGroup`, `CollectiveOps`, `ReduceOp` |
| 3 - `TransferPlane` | unified transfers (KV / EC / weight) | `TransferPlane`, `TransferPlaneRegistry`, `TransferType` |
| 4 - `CommStrategy` | choose algorithm by topology / size | `CommStrategy`, `DefaultStrategy`, `ConfigDrivenStrategy` |
| Bridge | adapt the above into vLLM | `UnifiedCommAdapter`, `is_unified_comm_enabled` |

---

## How it integrates into vLLM

The only change in `vllm/distributed/parallel_state.py` is a tiny hook
inside `GroupCoordinator`:

```python
class GroupCoordinator:
    # class-level defaults survive subclasses that override __init__()
    # without calling super().__init__()
    _unified_adapter: Any | None = None
    _unified_adapter_built: bool = False

    def _get_unified_adapter(self):
        # built lazily on first access; idempotent on failure.
        ...

    def _all_reduce_out_place(self, x):
        adapter = self._get_unified_adapter()
        if adapter is not None:
            out = adapter.all_reduce(x)
            if out is not None:
                return out
        # default code path below is untouched
        return self.device_communicator.all_reduce(x)
```

Every collective routes through the same pattern, so a single boolean
flag enables/disables the whole layer.

The hook is intentionally **lazy** rather than constructor-based so
that subclasses such as `vllm-ascend-hust`'s `GroupCoordinatorPatch`

- which overrides `__init__` without calling `super().__init__()` -
still pick up the adapter and the safe fallback defaults.

---

## How to enable

Single environment variable:

```bash
export UNIFIED_COMM_ENABLED=1
```

Optional knobs:

| Variable | Default | Effect |
| --- | --- | --- |
| `UNIFIED_COMM_ENABLED` | `0` | master switch; everything else has no effect when this is off |
| `UNIFIED_COMM_USE_DIRECT_HCCL` | `0` | (HCCL only) bypass `torch.distributed` and call `libhccl.so` directly via `pyhccl_wrapper` |
| `UNIFIED_COMM_USE_DIRECT_NCCL` | `0` | (NCCL only) bypass `torch.distributed` and call `libnccl.so` directly |

When enabled successfully, the worker logs three confirmation lines
during `GroupCoordinator` construction::

```text
[hccl_backend.py] HCCL comm group initialized: rank=0, world_size=2, mode=torch.distributed
[adapter.py]      [UnifiedCommAdapter] Created for ranks=[0,1], device=npu:0, backend=hccl
[parallel_state.py] [unified_comm] adapter attached to group 'tp:0' (rank=0/2, device=npu:0)
```

If any step fails the adapter returns `None` and a single warning is
emitted; the default code path resumes silently.

---

## End-to-end validation

| metric | value |
| --- | --- |
| HW | Ascend 910B2 x 2 |
| Model | Qwen2.5-7B-Instruct, FP16, TP=2 |
| Workload | random-online, input=1024, output=256, num_prompts=200, rate=1.0 |
| Output throughput | **233.0 tok/s** |
| Mean / P99 TTFT | 214.0 / 291.8 ms |
| Mean / P99 TPOT | 77.0 / 79.7 ms |
| Failures | 0 / 200 |

Smoke benchmarks of the adapter itself (compared against `dist.all_reduce`
on the same group) show `<10 us` overhead on small messages and
parity at >=1 MB messages. After the small-tensor fast-path the
`unified_comm` adapter recovers ~80% of the early-version
small-message gap; the Mode B (direct HCCL C API) variant additionally
gives **+1.4% throughput / -1.6% TPOT** on the end-to-end path under
graph mode.

---

## Tests

```bash
# 2-card smoke test (skips automatically when fewer than 2 accelerators)
pytest tests/distributed/test_unified_comm.py -v
```

The tests cover:

- feature-flag plumbing (default off, multiple truthy/falsy spellings)
- adapter construction returning `None` when disabled
- numerical parity for `all_reduce`, `all_gather`, `broadcast` against
  values produced by the underlying `torch.distributed` PG.

---

## Extending

### Adding a new device backend

1. Subclass `CommBackend` from
   `vllm/distributed/unified_comm/backend.py`.
2. Register it in `CommBackendRegistry` by calling
   `register_backend("my_device", MyBackend)` from your plugin's
   initialization code (or by adding it to
   `unified_comm/backends/__init__.py` and the dispatch table in
   `initialize.py`).
3. The bridging code in `parallel_state.py` is device-agnostic; it
   normalizes `self.device` to `torch.device` before calling
   `try_create`.

### Adding a new transfer plane

1. Subclass `TransferPlane` from
   `vllm/distributed/unified_comm/transfer_plane.py`.
2. Register an instance with
   `TransferPlaneRegistry().register(TransferType.MY_TYPE, plane)`.
3. The plane will be wired up to the `CollectiveGroup` automatically
   the first time a `UnifiedCommAdapter` is created.

---

## Caveats / follow-ups

- A subset of inline comments and method-level docstrings inside
  `collective.py`, `strategy.py`, `transfer_plane.py`, `backend.py`,
  `initialize.py`, and the two backend files are still mixed Chinese.
  These will be translated in a follow-up PR.
- The NCCL backend has been kept in tree to keep the layer
  symmetric and to satisfy the registry, but it has not been exercised
  on a GPU machine yet; it is gated behind the same opt-in flag.
- The adapter currently delegates only the four primitives that
  `GroupCoordinator._*_out_place` and `broadcast` use. `send` /
  `recv` / `all_to_all` are exposed at the adapter layer for future
  use but are not wired into `parallel_state.py` yet.
