# Experimental Adaptive Prefill Validation

## Fixed-overhead follow-up on November 19, 2026

The new real-scheduler regression uses the same synthetic timing in sync and
async modes: `14.76 + 0.53 * decode_requests + 0.07 * weighted_prefill_tokens`
milliseconds. Requests and schedulers use their real constructors, timing enters
through `ModelRunnerOutput`, and async completion is delayed by one scheduled
step. No controller state or clock is patched.

Before the controller change, the four cases produced **2 failed, 2 passed**:
the async three-long/late-short case never admitted the short prompt, and the
async four-long case retained four unfinished requests after the bounded run.
The corresponding sync cases passed. After the change, **all four passed**.

```text
Before:
FAILED ...test_adaptive_prefill_progress_with_fixed_step_overhead[3-True-True]
  assert (None is not None)
FAILED ...test_adaptive_prefill_progress_with_fixed_step_overhead[4-False-True]
  assert not 4
2 failed, 2 passed, 300 deselected in 4.00s

After:
4 passed, 300 deselected in 4.54s
```

The before run used PR head `4053e4e4c2f8` plus the new test; the after run adds
the controller fix without changing its completion/admission assertions.

```bash
VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 \
  uv run --no-project .venv/bin/python -m pytest tests/v1/core/test_scheduler.py \
  -k fixed_step_overhead -q --disable-warnings --tb=short
```

Controller/protocol validation now contains 21 passing unittest cases, including
fixed-versus-marginal decode cost, isolated timing spikes, zero fixed overhead,
prefill-only calibration, and bounded async reservations.

The complete CPU sync/async scheduler suites also pass: **323 passed in 161.84s**.
These include the four regressions above, which must not be counted twice.

```bash
VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 \
  uv run --no-project .venv/bin/python -m pytest \
  tests/v1/core/test_scheduler.py tests/v1/core/test_async_scheduler.py \
  -q --disable-warnings --tb=short
uv run --no-project .venv/bin/python -m unittest discover \
  -s tests/v1/core -p 'test_adaptive_prefill*.py' -v
```

```text
323 passed in 161.84s (0:02:41)
Ran 21 tests in 1.734s
OK
```

The model-backed `test_scheduler_e2e.py` now selects baseline/adaptive cases and
mixed-length completion. All six cases collect successfully, but they have
**not been executed locally**. Buildkite's V1
Core CPU/GPU selections cover the changed test paths; upstream CI still requires
the project's normal authorization. Real GPU execution, model-quality checks,
decode p99, TTFT, throughput and the original workload reproduction remain
**pending**. The contributor's A10G stall reports are not independently validated
by these CPU runs.

### Updated declared simulation, not a GPU measurement

The unchanged synthetic formula and arrival trace from the historical section
below produce these follow-up mixed-load results:

| Synthetic metric | Fixed-budget baseline | Adaptive follow-up |
| --- | ---: | ---: |
| Short decode interval p99, ms | 52.040 | 22.452 |
| Short TTFT p95, ms | 90.484 | 36.854 |
| Long TTFT maximum, ms | 685.528 | 853.131 |
| Output tokens/s | 1767.433 | 1730.480 |
| All bounded-workload prefills completed | yes | yes |

Short-only controls remain identical. Long-only output throughput is 40.822
versus 39.908 synthetic tokens/s. Mixed throughput decreases about 2.1%, and
maximum long TTFT increases about 24.5%. This experiment does not exercise the
real model, CUDA timing, or delayed feedback; the scheduler regression above
separately exercises delayed synthetic feedback. Neither is a GPU speedup claim.

## Historical validation on November 16, 2026

Current upstream base: vLLM `71c3541492cf`. The initial simulation used
`919d45f`; it was repeated after rebasing without changing the results below.
This is an experimental patch, not an accepted upstream fix for #62574.

## Hardware and blocked work

The Windows host reports AMD Radeon integrated graphics. WSL PyTorch
`2.13.0+cu130` reports `torch.cuda.is_available() == False` and zero CUDA devices.
No live vLLM GPU server or GB10/DeepSeek reproduction was run. Real GPU decode
p99, TTFT and throughput are **not available**.

The CPU scheduler test environment now runs the real scheduler and KV managers
with synthetic worker outputs. The complete `test_scheduler.py` and
`test_async_scheduler.py` run passed: **319 passed** in 228.06 seconds, including
the four new enabled/disabled and sync/async controller cases. Two existing
structured-output tests bypass scheduler construction; their fixtures now
explicitly initialize the optional controller to its disabled default.

The extended run of `test_async_scheduler.py` and
`test_priority_scheduler_random.py` also passed: **31 passed** in 803.91 seconds.
This adds 12 randomized priority/preemption cases; its 19 async cases overlap the
319-case run and must not be counted twice. The stress run fetched the missing
Qwen2.5-VL model configuration online; no model weights or GPU execution were
needed. The opt-in policy remains deliberately unsupported for priority mode;
these stress cases check default-policy compatibility, not adaptive priority
scheduling.

Environment: Python 3.12.16, PyTorch 2.15.0+cpu, Triton CPU 3.8.0,
Transformers 5.23.0 and pytest 9.2.0. PyTorch differs from the checkout's 2.14.0
pin, so this is local CPU integration evidence, not a matching upstream CI image.
Source CPU package metadata was generated with `VLLM_TARGET_DEVICE=cpu` and
`python setup.py egg_info`; this does not build model-execution kernels. An
optional editable install was stopped after it started compiling the unrelated
Rust frontend; no successful complete wheel build is claimed.

Reproduction after installing CPU test dependencies and caching model configs:

```bash
VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 python -m pytest \
  tests/v1/core/test_scheduler.py tests/v1/core/test_async_scheduler.py -q
```

`test_scheduler_e2e.py` requires model execution and is not part of this CPU-only
run. No GPU result may be inferred from any of the tests below.

## Executed checks

- 16 standard-library unittest cases pass: controller admission/credits,
  context cost, pending reservations, rollback refunds, invalid feedback,
  bounded state, aged minimum progress, fair shares, and local HTTP protocol
  handling (including truncated and multi-token streams).
- Modified Python files pass Ruff lint/format checks; `git diff --check` passes.
- The streaming fixture is only a protocol test, not inference evidence.

## Synthetic policy experiment only

`PYTHONPATH=. python benchmarks/adaptive_prefill/simulate.py` compares both
policies under the explicitly declared artificial device-cost formula:

`0.5 + 0.18 * decode_requests + 0.006 * context_weighted_prefill_tokens` ms.

Mixed scenario: four 16,384-token prompts and 32 short prompts; prompt chunks
capped at 1,024 tokens, aggregate token budget 8,192, identical arrival trace.
Percentiles use nearest rank.

| Synthetic metric | Fixed-budget baseline | Adaptive |
| --- | ---: | ---: |
| Short decode gap p99, ms | 52.040 | 25.045 |
| Short TTFT p95, ms | 90.484 | 35.985 |
| Long TTFT maximum, ms | 685.528 | 851.958 |
| Output tokens/s | 1767.433 | 1732.728 |
| All long prefills completed | yes | yes |

Short-only controls are identical. Long-only output throughput decreases from
40.822 to 39.997 synthetic tokens/s. Mixed-load throughput decreases about 2.0%;
long-request TTFT increases about 24.3%. These tradeoffs must remain visible.

The experiment supports testing the control logic, not a claim about vLLM GPU
performance, real p99 protection or starvation freedom under arbitrary overload.
The original issue's speculative hybrid-model configuration is intentionally
unsupported by this first implementation.

## Next required gate

Install this commit in a CUDA-capable vLLM environment. Run the added real
streaming benchmark against baseline and opt-in servers on identical hardware,
model, token budgets and arrival traces; repeat and sweep targets, then execute
the original reported configuration only after adding its missing support.
