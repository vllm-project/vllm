# Buildkite Web UI Setup: SGLang + LMCache MP

**Steps editor**: paste contents of `buildkite-pipeline.yml` (fill in `HF_TOKEN`).

**GitHub trigger settings**:
- Filter (the SGLang MP adapter's real dependency surface spans most of `lmcache/`):
  - `lmcache/**`
  - `.buildkite/k3_tests/sglang/**`
- Skip queued / cancel running branch builds: Yes

Two GPU jobs (correctness + performance), ~5 min each — lightweight enough for a required PR status check on changes that touch the MP integration.

> Builds whose only changes are docs/`*.md`/`LICENSE`/`.github/**` auto-pass
> via the [path filter](../README.md#path-based-skip-auto-pass-on-docs-only-changes).
> Changes under `.buildkite/` always run. Add the `force-ci` label to a PR to
> bypass.
