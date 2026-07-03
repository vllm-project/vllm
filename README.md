<!-- markdownlint-disable MD013 MD033 MD041 -->

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM logo" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width="360">
  </picture>
</p>

<h3 align="center">
HUST-maintained vLLM fork for upstream-compatible Ascend/NPU serving research
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Upstream Docs</b></a> |
<a href="https://github.com/vllm-project/vllm"><b>Upstream vLLM</b></a> |
<a href="https://github.com/vLLM-HUST/vllm-ascend-hust"><b>HUST Ascend Plugin</b></a> |
<a href="https://github.com/vLLM-HUST/vllm-hust-dev-hub"><b>Dev Hub</b></a> |
</p>

# vLLM-HUST

`vLLM-HUST` is the HUST-maintained fork of
[`vllm-project/vllm`](https://github.com/vllm-project/vllm). It keeps the
upstream vLLM serving surface while carrying the local patches needed for
Ascend/NPU research, managed service experiments, and HUST deployment workflows.

This repository is the core runtime fork. The paired Ascend plugin is
[`vllm-ascend-hust`](https://github.com/vLLM-HUST/vllm-ascend-hust); the two
repositories should be installed, tested, synced, and released together.

## What Comes From Upstream vLLM

This fork remains a vLLM distribution. It inherits the upstream project goal:
fast, flexible, OpenAI-compatible LLM inference and serving.

Upstream vLLM provides:

- high-throughput serving with continuous batching
- efficient KV-cache management with PagedAttention
- OpenAI-compatible online serving APIs
- offline generation, scoring, pooling, and embedding-style workflows
- tensor, pipeline, data, and expert parallelism support
- speculative decoding, chunked prefill, prefix caching, and structured output
- broad Hugging Face model support
- quantization and optimized execution paths maintained by the upstream project

For general vLLM usage, model support, and command-line documentation, start
with the official docs: <https://docs.vllm.ai/>.

## What HUST Adds

HUST-specific changes are intentionally scoped around keeping upstream vLLM
usable on the local Ascend/NPU stack:

- Ascend/NPU compatibility fixes in core runtime paths.
- Knorm environment hooks and manager registration points.
- Plugin pooling and offline encode support in `LLM`.
- Routing headers for managed OpenAI-compatible services.
- Shutdown and health-path fixes for service supervisors.
- Compatibility with the paired `vllm-ascend-hust` plugin.
- Upstream merge/version metadata used to keep fork drift visible.

The goal is not to fork vLLM into a separate framework. The desired state is:
upstream behavior by default, HUST deltas only where they are needed and tested.

## Repository Map

| Path | Purpose |
| --- | --- |
| `vllm/` | Core vLLM runtime with HUST patches. |
| `vllm/knorm/` | Optional Knorm integration points. |
| `vllm/platforms/` | Platform discovery and custom device support hooks. |
| `vllm/entrypoints/` | CLI, offline, and OpenAI-compatible entrypoints. |
| `upstream_version.json` | Current upstream anchor and HUST release base. |
| `AGENTS.md` | Required workflow rules for AI-assisted changes. |

## Paired HUST Repositories

| Repository | Role |
| --- | --- |
| [`vllm-hust`](https://github.com/vLLM-HUST/vllm-hust) | Core vLLM fork. |
| [`vllm-ascend-hust`](https://github.com/vLLM-HUST/vllm-ascend-hust) | Ascend/NPU plugin fork paired with this runtime. |
| [`vllm-hust-dev-hub`](https://github.com/vLLM-HUST/vllm-hust-dev-hub) | Multi-repo workspace, managed service scripts, and NPU smoke-test entrypoint. |
| [`vllm-hust-benchmark`](https://github.com/vLLM-HUST/vllm-hust-benchmark) | Benchmark orchestration and result export. |
| [`vllm-hust-perf-analyzer`](https://github.com/vLLM-HUST/vllm-hust-perf-analyzer) | Offline profiler timeline analysis. |

## Versioning

This fork uses an upstream-anchored version:

```text
<upstream release>.post1.dev<HUST-only commit count>+g<short sha>
```

`upstream_version.json` records the anchor:

- `upstream_commit`: exact upstream `main` commit included in the fork graph.
- `upstream_version`: upstream-compatible version string, including rc suffix
  when upstream is on an rc.
- `release_version`: the same version line without the rc suffix.

After a completed upstream sync, the fork should be zero commits behind
upstream:

```bash
git fetch upstream main
git rev-list --left-right --count origin/main...upstream/main
# <HUST-only commits>  0
```

The left side is HUST-only delta; the right side should be `0`.

## Install For Development

Use `uv` and the project virtual environment. Do not install with system
`python3` or bare `pip`.

```bash
cd /path/to/vllm-hust
uv venv --python 3.12
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

Then install the paired Ascend plugin:

```bash
cd /path/to/vllm-ascend-hust
COMPILE_CUSTOM_KERNELS=0 uv pip install -e . --no-deps
```

On HUST local hosts, prefer the plugin install helper:

```bash
cd /path/to/vllm-ascend-hust
bash scripts/install_local_ascend_plugin.sh /path/to/vllm-ascend-hust
```

## Run On Ascend/NPU

For managed HUST services, use the dev-hub scripts:

```bash
cd /path/to/vllm-hust-dev-hub
./manage.sh status
./manage.sh restart
./manage.sh health --json
```

For direct local experiments:

```bash
VLLM_PLUGINS=ascend \
VLLM_TARGET_DEVICE=npu \
vllm serve /path/to/model \
  --host 0.0.0.0 \
  --port 8000
```

On shared machines, use only the allocated device. The current HUST smoke-test
workflow is constrained to NPU 1 unless the operator explicitly assigns another
device.

## Validation Checklist

README-only changes:

```bash
git diff --check -- README.md
```

Python/runtime changes:

```bash
.venv/bin/python -m py_compile path/to/file.py
pre-commit run --files path/to/file.py
```

Upstream merges:

```bash
git diff --name-only --diff-filter=U
git rev-list --left-right --count origin/main...upstream/main
```

NPU runtime changes should also be tested through `manage.sh` from
`vllm-hust-dev-hub`, using NPU 1 unless another device is assigned.

## Upstream Sync Workflow

1. Fetch upstream:

   ```bash
   git fetch upstream main
   ```

2. Merge upstream into a staging branch:

   ```bash
   git checkout -B sync/upstream-main-YYYYMMDD origin/main
   git merge --no-ff upstream/main
   ```

3. Resolve conflicts by preserving small HUST deltas and taking upstream
   behavior wherever possible.
4. Update `upstream_version.json` and verify the version rule in `AGENTS.md`.
5. Run syntax checks, targeted tests, and managed NPU smoke tests.
6. Merge the staging PR to `main`, then confirm the fork is zero commits behind
   upstream.

Avoid routine cherry-pick/backport stacks when a real upstream merge can keep
the fork graph honest.

## Documentation And Community

- Upstream vLLM docs: <https://docs.vllm.ai/>
- Upstream vLLM repository: <https://github.com/vllm-project/vllm>
- Upstream paper: <https://arxiv.org/abs/2309.06180>
- vLLM user forum: <https://discuss.vllm.ai/>
- HUST organization: <https://github.com/vLLM-HUST>
- HUST agent workflow: [`AGENTS.md`](AGENTS.md)
- Security policy inherited from upstream: [`SECURITY.md`](SECURITY.md)

## License

This repository follows the upstream vLLM license. See [`LICENSE`](LICENSE) and
upstream notices for details.
