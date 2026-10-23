# tests

- **Test real objects on the lightest real path.** Construct vLLM objects with
  their real constructors; do not stand them in with `SimpleNamespace` or
  `MagicMock`, set private attributes, or patch methods to reach a state. Mock
  only external services. Pick the cheapest setup that exercises the change: a
  layer or kernel test does not need `vllm_runner`, and single-rank parallel
  state is enough when parallelism is irrelevant. A fix to user-visible behavior
  needs at least one regression test through `LLM` (`vllm_runner`) or
  `RemoteOpenAIServer`, with real weights when loading is involved. A test that
  only asserts `assert_called*` or "no exception" guards nothing.
- **Show the failure before the fix.** When the bug is reproducible, run the new
  test on the base commit (rebuild if the fix touches `csrc/`) and confirm it
  fails for the reported reason, then passes with the fix. Paste both commands
  and the relevant output in the PR; "fails before" without output is not
  evidence.
- **Never buy a green CI.** Do not add `flaky`/reruns, retry loops, `skip`,
  `xfail`, lowered thresholds, wider `atol`/`rtol`, or deletions to make a test
  pass. Find out whether the failure is caused by your change or also happens
  on main (`.buildkite/scripts/rerun-test.sh`,
  `docs/contributing/ci/failures.md`); the only exception is
  `xfail(strict=True)` linking an open issue for a bug on main.
- **Deterministic and exact by construction.** Use greedy or seeded sampling.
  Assert equality where the result is exact (a change that must not alter
  output is compared against the code it replaces, not another reference), and
  keep tolerances as tight as the math allows. vLLM is not batch invariant by default: batch size, chunked
  prefill, prefix caching, backend, and TP size can flip greedy tokens, so
  compare against a reference with `check_logprobs_close` /
  `check_embeddings_close`, or require `VLLM_BATCH_INVARIANT=1` for exact
  equality. Never synchronize with a fixed `sleep`; poll an explicit condition
  with a timeout, and force race orderings with events.
- **Make sure CI runs it.** A test no step in `.buildkite/test_areas/*.yaml`
  selects never runs, which looks like passing. Check that the step's
  `commands` and pytest markers (`core_model`, `slow_test`, `distributed`)
  select the test and its `source_file_dependencies` cover the changed files;
  update the step if not.
