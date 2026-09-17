# Uno (MRV2) — PR notes

Working notes for the Uno Model Runner V2 pull request. This file records the
2026-09-17 cleanup pass on top of the merge of upstream `91a4c40b45`
(merge commit `ba14f1e40a`).

## Cleanup 2026-09-17

The review asked to remove the custom timing logic and to shrink the changes to
core components that were not directly related to Uno. This pass does both and
leaves Uno's serving behaviour unchanged.

### Removed

- The per-step timing tracer `vllm/v1/worker/gpu/uno_step_timing.py`, its
  `VLLM_UNO_STEP_TIMING_DEBUG` switch, and the `debug_uno_step_id` /
  `debug_schedule_wall_ms` scheduler-output fields it read.
- The launch-key receipt recorder `vllm/v1/worker/gpu/launch_key_debug.py`,
  its `VLLM_UNO_LAUNCH_KEY_DEBUG` path in `uno_prepare.py`, and the
  `launch_key_phase` / `record_topk_topp_launches` / `serving_launches`
  wrappers in the worker.
- The post-warmup JIT self-check `run_uno_served_jit_self_check` in
  `warmup.py`, together with `capture_topk_topp_launches`,
  `capture_sampler_branches`, `UnoJitSelfCheck` and
  `uno_self_check_token_count`.
- The `capture_compilations` hooks that had been added to
  `vllm/utils/jit_monitor.py`, and the one added test case in
  `tests/jit_monitor/test_hooks.py`.
- A cosmetic `launch_grid` local in `vllm/v1/worker/gpu/sample/gumbel.py`.
- The scheduler trace helpers `_will_finish_after_next_sample`, the
  `uno_tail_debug` row builder, and the `UNO_STEP_TIMING_*` log sites, plus the
  matching test cases.

Taken together the removals above delete 1,841 lines across code and tests
(`git diff --shortstat` of the cleanup commits).

### Contained

The Uno length-tail policy remains, because it changes scheduling behaviour and
not just observation. It now lives in one small object,
`vllm/v1/core/sched/uno_tail.py` (`UnoTailPolicy.apply` / `in_tail` /
`forget`), and the scheduler keeps only the construction and calls at the
existing sites.

Why the policy stays: Uno's draft writes one seed row and `K-1` noisy rows
beyond the target's query rows, so every drafted step reserves `K` extra KV
slots. On the step where a request can reach `max_tokens` (or the model context
limit) the draft cannot be used at all, because a single target sample ends the
request. Without the policy the scheduler pads that terminal step to `K+1`
tokens, allocates KV for rows it will never consume, and under KV pressure can
preempt other requests for nothing. The policy chooses `K=0` before KV admission
on a possibly-terminal step and remembers that choice across preemption (but
not across a new streaming-input turn or the end of the request). The tail
behaviour is covered by `tests/v1/core/test_scheduler.py`,
`tests/v1/spec_decode/test_uno_tail_worker.py` and
`tests/v1/spec_decode/test_uno_preemption.py`.

The result of the pass is the core touch below (this notes file excluded from
the stat so the block is the literal output of its command); everything else is
Uno-specific code, configuration validation and tests.

```text
$ git diff --stat upstream/main HEAD -- . ':(exclude)docs/PR-DESCRIPTION-MRV2.md'
 .buildkite/test_areas/spec_decode.yaml            |   13 +
 docs/features/speculative_decoding/README.md      |    3 +-
 docs/features/speculative_decoding/uno.md         |   52 +
 tests/v1/core/test_prefix_caching.py              |   59 +
 tests/v1/core/test_scheduler.py                   |  556 ++++++++
 tests/v1/core/utils.py                            |   22 +-
 tests/v1/e2e/spec_decode/test_uno.py              | 1662 ++++++++++++++++++++++
 tests/v1/e2e/spec_decode/uno_kv_budget.py         |  767 ++++++++++
 tests/v1/engine/test_preprocess_error_handling.py |   25 +
 tests/v1/spec_decode/test_uno_config.py           |  390 +++++
 tests/v1/spec_decode/test_uno_lora_mrv2.py        |  614 ++++++++
 tests/v1/spec_decode/test_uno_mrv2.py             | 3123 +++++++++++++++++++++++++++++++++++++++++
 tests/v1/spec_decode/test_uno_preemption.py       |  616 ++++++++
 tests/v1/spec_decode/test_uno_prepare_mrv2.py     |  462 ++++++
 tests/v1/spec_decode/test_uno_tail_worker.py      |  207 +++
 tests/v1/worker/test_gpu_model_runner_v2_eplb.py  |    2 +-
 vllm/config/speculative.py                        |   97 +-
 vllm/config/vllm.py                               |   79 +-
 vllm/v1/core/sched/async_scheduler.py             |    6 +-
 vllm/v1/core/sched/output.py                      |   11 +-
 vllm/v1/core/sched/scheduler.py                   |   46 +-
 vllm/v1/core/sched/uno_tail.py                    |   70 +
 vllm/v1/engine/input_processor.py                 |    5 +
 vllm/v1/spec_decode/uno_noise.py                  |   52 +
 vllm/v1/worker/gpu/model_runner.py                |  378 ++++-
 vllm/v1/worker/gpu/spec_decode/__init__.py        |    6 +-
 vllm/v1/worker/gpu/spec_decode/uno.py             |  972 +++++++++++++
 vllm/v1/worker/gpu/spec_decode/uno_lora.py        |  514 +++++++
 vllm/v1/worker/gpu/spec_decode/uno_prepare.py     |  532 +++++++
 vllm/v1/worker/gpu/warmup.py                      |  135 +-
 30 files changed, 11417 insertions(+), 59 deletions(-)
```

The non-Uno core touch is the configuration validation, the runner's
install/propose/publish path and one contained scheduler policy.

### Plugin hook

Once the speculator plugin mechanism is in place, the Uno modules
(`uno.py`, `uno_prepare.py`, `uno_lora.py`, `uno_noise.py` and the runner's
install/propose/publish path) can move behind it. The one thing the hook needs
for the scheduler change to disappear as well is a per-request "next draft
width" override that the plugin can set to zero on a possibly-terminal step.
