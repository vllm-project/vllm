# tests

Read this before adding or changing tests. A test is worth merging only if it
fails when the behavior it guards breaks and passes every time otherwise.

## Bug fixes: show the test catches the bug

When the bug is reproducible, run the new test on the base commit (without the
fix) and confirm it fails for the reported reason: read the assertion or
traceback, not just the exit code. Then confirm it passes with the fix. If the
fix touches `csrc/`, rebuild before each run. Put both commands and the relevant
output in the PR's Validation section; "tests pass" alone is not evidence.

If no automated test can reproduce the bug (hardware, checkpoint size, timing),
say so and give the manual reproducer with before/after output.

## Pick the level

- Use the cheapest level that actually reproduces the problem.
- A fix to user-visible behavior needs at least one test through a real entry
  point: `LLM` via the `vllm_runner` fixture, or the OpenAI server via
  `RemoteOpenAIServer` (`tests/utils.py`). Testing an extracted helper does not
  show that a scheduler, model runner, or sampler fix works for users.
- Kernel tests compare against a reference implementation with explicit
  tolerances; custom ops also go through `opcheck` (`tests/kernels/utils.py`).
- Mock only boundary dependencies (external services, absent hardware), never
  the code under test. A test that only asserts `assert_called*` or that no
  exception was raised is not accepted.
- Keep end-to-end tests cheap: the smallest model already used in that area,
  short prompts, small `max_tokens` and `max_model_len`, and `enforce_eager=True`
  unless the behavior involves CUDA graphs or `torch.compile`.
- If you cannot name a code change that would turn the test red, it guards
  nothing; drop it.

## Determinism

- Use greedy sampling or a fixed `SamplingParams(seed=...)`; seed random inputs
  with `set_random_seed`. Pin settings that change numerics (dtype, attention
  backend, chunked prefill, prefix caching) instead of relying on defaults.
- Compare against a reference with explicit tolerances: HF via `hf_runner` and
  `check_logprobs_close` (`tests/models/utils.py`), or
  `torch.testing.assert_close` against a PyTorch reference. vLLM is not batch
  invariant by default: batch size, chunked prefill, prefix-cache hits, backend,
  and TP size change kernel shapes and can flip greedy tokens. Assert exact
  tokens or bitwise equality across such changes only with
  `VLLM_BATCH_INVARIANT=1` (see `docs/features/batch_invariance.md`).
- Do not synchronize with fixed `sleep`s; poll an explicit condition (health
  endpoint, future, event) with a timeout. Reproduce concurrency bugs by forcing
  the ordering (events, barriers), not by hoping for it. Let
  `RemoteOpenAIServer` pick its port. Serve URL inputs from the
  `local_asset_server` fixture rather than the internet.
- Do not hide flakiness with new `@pytest.mark.flaky(reruns=...)` markers, retry
  loops, or wider tolerances; find the cause. For tests involving timing, async
  code, or multiple processes, run them repeatedly before submitting
  (`.buildkite/scripts/rerun-test.sh`, see `docs/contributing/ci/failures.md`)
  and report the number of passing runs in the PR.

## Isolation

- Create engines in context managers (`with vllm_runner(...) as llm:`,
  `with RemoteOpenAIServer(...) as server:`) so GPU memory and child processes
  are released before the next test.
- Use `@create_new_process_for_each_test()` (`tests/utils.py`) when a test
  initializes distributed state or other process-global state.
- Pass server environment variables through `RemoteOpenAIServer(...,
  env_dict=...)` (works in module-scoped fixtures, unlike `monkeypatch`). In
  process, prefer `monkeypatch.setenv` or `patch.dict(os.environ, ...)` so
  changes are undone; plain assignment is fine only in a child process's entry
  point.
- Declare hardware needs with `@multi_gpu_test(num_gpus=...)` and
  `@large_gpu_test(min_gb=...)`.

## Assertions

- Assert the expected result: tokens or text, field values, status code and
  error message, not `len(out) > 0` or `out is not None`.
- Prove the new path ran rather than a fallback: the selected backend, a log
  line (`caplog_vllm`), or a nonzero spec-decode acceptance rate.
- Include inputs and both values in failure messages, or use a helper that
  prints a diff.

## Do not weaken existing tests

- Do not delete, loosen, or skip an existing test to make a PR pass. If the
  expected behavior changes on purpose, explain why in the PR.
- Use `skip` only for unsupported environments. Mark a known bug with `xfail`
  linking the issue, so the test reports when it starts passing.

## Placement and CI

- Put tests in the `tests/` subdirectory that mirrors the code under test and
  extend an existing file when one fits.
- CI runs only what a step in `.buildkite/test_areas/*.yaml` selects. Check
  that a step's `commands` run your test and its `source_file_dependencies`
  cover the changed source paths; update the step if not. A test no step
  selects never runs, which looks the same as passing. If CI hardware cannot
  run the test, say so in the PR and show a local run.
