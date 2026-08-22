---
name: ci-fails-buildkite
description: Fetch and diagnose vLLM Buildkite CI failure logs, including non-default pipelines such as intel-ci. Use when investigating failing CI jobs on a PR or build, when the user pastes a buildkite.com URL, or asks to fetch/diagnose CI logs.
---

# Diagnosing vLLM Buildkite CI Failures

Buildkite logs are public; no login needed.

`.buildkite/scripts/ci-fetch-log.sh` saves each log as `ci-<build>-<job-name>.log` for the default `ci` pipeline, or `ci-<pipeline>-<build>-<job-name>.log` for non-default pipelines, stripped of timestamps and ANSI codes. Existing files are kept; set `CI_FETCH_LOG_FORCE=1` to refetch.

## Fetching logs

```bash
# All failed jobs in a PR's latest ci build (current branch's PR if omitted):
.buildkite/scripts/ci-fetch-log.sh --pr <PR>

# All failed jobs in a PR's latest non-default pipeline build:
.buildkite/scripts/ci-fetch-log.sh --pipeline intel-ci --pr <PR>

# One job by build number and job UUID in a non-default pipeline:
.buildkite/scripts/ci-fetch-log.sh --pipeline intel-ci <N> <job_uuid>

# All failed jobs in a build (--soft also includes soft-failed jobs;
# --all fetches every finished job):
.buildkite/scripts/ci-fetch-log.sh "https://buildkite.com/vllm/ci/builds/<N>"

# Pipeline is inferred from Buildkite URLs, including intel-ci:
.buildkite/scripts/ci-fetch-log.sh "https://buildkite.com/vllm/intel-ci/builds/<N>"

# One job — `gh pr checks` URLs (#<job_uuid>) and web UI URLs (?sid=) both
# work; pass "-" as a second argument to stream to stdout:
.buildkite/scripts/ci-fetch-log.sh "https://buildkite.com/vllm/ci/builds/<N>#<job_uuid>"
.buildkite/scripts/ci-fetch-log.sh "https://buildkite.com/vllm/intel-ci/builds/<N>#<job_uuid>"
```

To clean an already-downloaded log with `.buildkite/scripts/ci-clean-log.sh`:

```bash
./ci-clean-log.sh ci.log
```

## Reference

See [docs/contributing/ci/failures.md](../../../docs/contributing/ci/failures.md) for the full guide: filing CI failure issues, investigating/bisecting, reproducing flaky tests, and daily triage.
