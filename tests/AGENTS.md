# tests

- **User-visible fixes need a real-path test.** Use the cheapest level that
  reproduces the problem, but a fix to user-visible behavior needs at least one
  regression test through `LLM` (`vllm_runner`) or the OpenAI server
  (`RemoteOpenAIServer`), not only a unit test on hand-built inputs. Mock only
  external dependencies; a test that only asserts `assert_called*` or "no
  exception" guards nothing.
- **Show the failure before the fix.** When the bug is reproducible, run the new
  test on the base commit (rebuild if the fix touches `csrc/`) and confirm it
  fails for the reported reason, then passes with the fix. Paste both commands
  and the relevant output in the PR; "fails before" without output is not
  evidence.
- **Never buy a green CI.** Do not add `@pytest.mark.flaky(reruns=...)`, retry
  loops, wider tolerances, lowered thresholds, `skip`, or deletions to make a
  test pass. Find out whether the failure is caused by your change or also
  happens on main (`.buildkite/scripts/rerun-test.sh`,
  `docs/contributing/ci/failures.md`). Mark a known bug with `xfail` linking the
  issue, not `skip`.
- **Deterministic by construction.** Use greedy or seeded sampling. vLLM is not
  batch invariant by default: batch size, chunked prefill, prefix caching,
  backend, and TP size can flip greedy tokens, so compare against a reference
  with tolerances (`check_logprobs_close`, `check_embeddings_close`) or require
  `VLLM_BATCH_INVARIANT=1` for exact equality. Never synchronize with a fixed
  `sleep`; poll an explicit condition with a timeout, and force race orderings
  with events.
- **Make sure CI runs it.** A test no step in `.buildkite/test_areas/*.yaml`
  selects never runs, which looks like passing. Check that the step's
  `commands` and pytest markers (`core_model`, `slow_test`, `distributed`)
  select the test and its `source_file_dependencies` cover the changed files;
  update the step if not.
