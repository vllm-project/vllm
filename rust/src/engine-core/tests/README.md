# Python reference traces

The engine-core replay tests compare the Rust port against traces recorded
from the Python implementation. Each suite has a generator next to its replay
test and a small checked-in set, so `cargo test -p vllm-engine-core` stays
meaningful offline. CI records many more traces with
`.buildkite/scripts/run-rust-engine-core-traces.sh` and replays them.

| Suite | Generator | Replay test | Checked-in traces |
| --- | --- | --- | --- |
| `kv_cache` | `src/kv_cache/tests/gen_traces.py` | `kv_cache::tests::differential` | `src/kv_cache/tests/traces` |
| `sched` | `src/sched/scheduler/tests/gen_traces.py` | `sched::scheduler::tests::differential` | `src/sched/scheduler/tests/traces` |
| `nixl` | `src/sched/scheduler/tests/gen_nixl_traces.py` | `sched::scheduler::tests::nixl_differential` | `src/sched/scheduler/tests/nixl_traces` |
| `mooncake` | `src/sched/scheduler/tests/gen_mooncake_traces.py` | `sched::scheduler::tests::mooncake_store` | `src/sched/scheduler/tests/mooncake_traces` |

The generators need vLLM installed (CPU is enough) and run from the
repository root. They share one command line:

- no arguments: regenerate the checked-in traces;
- `--seeds N` or `--seeds LO-HI`: record every kind for each seed, at the
  default number of steps, instead of the checked-in set;
- `--out DIR`: write there instead of the checked-in directory;
- `--steps N`: override the number of steps per trace;
- `--kinds a,b` (`kv_cache`: positional config names): only these kinds.

The output depends only on the arguments: each generator re-executes itself
with `PYTHONHASHSEED=0`, and every random choice is seeded. The DeepSeek-V4.1
layouts need a pickled KV cache config in `RUST_CORE_DIFF_DSV41_KVCFG` and are
skipped without it.

To replay freshly recorded traces, point `ENGINE_CORE_TRACE_DIR` at a
directory with one subdirectory per suite:

```bash
T=rust/src/engine-core/src
.venv/bin/python $T/sched/scheduler/tests/gen_traces.py --seeds 0-50 --out /tmp/traces/sched
ENGINE_CORE_TRACE_DIR=/tmp/traces cargo test --manifest-path rust/Cargo.toml \
    -p vllm-engine-core sched::scheduler::tests::differential
```

Rerun the generators without arguments, and commit the result, when a change
to the Python implementation changes the checked-in traces.

The JSON fixtures in `tests/fixtures` (request queues, the encoder cache
manager, and CPython `set` iteration order) come from
`tests/fixtures/gen_sched_fixtures.py`.
