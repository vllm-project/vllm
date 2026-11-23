# Community sprint workspace

This tree holds dependency-free design notes, analytic models, tests and micro-benchmarks
contributed by the community working group. Nothing here is imported by the serving path:
the goal is to make tuning decisions reproducible without a GPU in the loop.

## Layout

| path | contents |
| --- | --- |
| `docs/` | One design note per topic: mechanism, knobs, failure modes, instrumentation. |
| `src/vllm_community/` | Analytic models, standard-library only. |
| `tests/` | pytest suites that validate the models. |
| `benchmarks/` | Micro-benchmarks that measure the models themselves, so numbers compare across CI machines. |

## Conventions

- Every model exposes `Budget`, `DEFAULTS`, `simulate`, `recommend`, `check_invariants`.
- Every note states the metric it expects you to watch before and after a change.
- No file in this tree may import `torch` or `vllm`; the tree must stay importable in a bare interpreter.

## Running

```bash
python -m pytest community/tests -q
python community/benchmarks/bench_<topic>.py --requests 256
```
