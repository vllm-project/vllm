# Buildkite Web UI Setup: XPU Smoke Test

**Steps editor**: paste the contents of `buildkite-pipeline.yml`.

**GitHub trigger settings**:
- Filter: `build.pull_request.labels includes "xpu" || build.pull_request.labels includes "full" || build.branch == 'dev'`
- Rebuild on PR label change: Yes
- Skip queued / cancel running branch builds: Yes

This pipeline now has a single step: it runs the XPU smoke test directly in a
prebuilt public vLLM image and installs LMCache from source inside the job pod.

### Trigger strategy

The XPU pipeline is intentionally lightweight, so it is label/branch gated:

| Condition | Result |
|-----------|--------|
| PR label includes `full` | upload the XPU pipeline |
| branch is `dev` | upload the XPU pipeline |
| any docs/asset-only change | path filter skips upload |
| any change under `.buildkite/` | path filter forces upload |

The path filter treats the following as trivial for the k3 test harness:

- `*.md`, `LICENSE*`, `NOTICE*`
- `.gitignore`, `.gitattributes`, `.editorconfig`, `.mailmap`, `CODEOWNERS`
- anything under `docs/`, `asset/`, or `.github/`

If you need the XPU pipeline to run for a docs/asset-only PR, add the
`force-ci` label.


## Required host setup

Before creating the pipeline, prepare the machine that will run the `intel-xpu` queue:

1. Run [setup-cluster.sh](../../k3_harness/setup-cluster.sh) to install K3s, the GPU Operator, and the shared host volumes.
2. Run [install-agent-stack.sh](../../k3_harness/install-agent-stack.sh) with a Buildkite agent token and a GitHub token.


## Buildkite UI snippet

If you want to create the pipeline manually, paste this into the Steps editor:

```yaml
agents:
  queue: "intel-xpu"

steps:
  - label: ":pipeline: Upload pipeline"
    command: bash .buildkite/k3_tests/common_scripts/upload-pipeline.sh .buildkite/k3_tests/xpu/pipeline.yml
```

## What this pipeline does

- Runs the XPU smoke test on the `intel-xpu` queue
- Uses the prebuilt public vLLM image
- Installs LMCache from source via `setup-lmcache-only-env.sh`
- Verifies `torch.xpu.is_available()` inside the job pod

## TODO

- Enable vLLM/LMCache nightly build to catch up latest code changes

- Refine the XPU path filter if additional XPU-only subtrees need to be excluded

- Refactor unit tests targeting multiple devices.

- Enable more `xpu` tests within current CI architecture.
