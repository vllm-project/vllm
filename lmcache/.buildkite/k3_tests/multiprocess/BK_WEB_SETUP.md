# Buildkite Web UI Setup: Multiprocess Tests

**Steps editor**: paste contents of `buildkite-pipeline.yml` (fill in `HF_TOKEN`).

**GitHub trigger settings**:
- Filter: `build.pull_request.labels includes "mp" || build.pull_request.labels includes "full" || build.branch == 'dev'`
- Rebuild on PR label change: Yes
- Skip queued / cancel running branch builds: Yes

Heavy test (2 GPUs, Docker-in-Docker, ~45 min) — run on `"mp"`/`"full"` label or dev push, not every PR.

> Builds whose only changes are docs/`*.md`/`LICENSE`/`.github/**` auto-pass
> via the [path filter](../README.md#path-based-skip-auto-pass-on-docs-only-changes).
> Changes under `.buildkite/` always run. Add `force-ci` label to the PR to
> bypass.
