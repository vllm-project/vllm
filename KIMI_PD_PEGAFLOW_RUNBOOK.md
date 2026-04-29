# Kimi PD PegaFlow Runbook

This runbook is the persistent handoff for this worktree. Read it first when
opening a new session in:

```text
/Users/ppio-dn-289/Documents/llmd-vllm-workspace/vllm-kimi-pd-pegaflow
```

It records where code should be pushed, how to sync code to the test server,
how to run unit and end-to-end tests, and how to build the patched
`vllm-with-pegaflow` image.

## Current Source State

### vLLM Worktree

```text
Local path:
/Users/ppio-dn-289/Documents/llmd-vllm-workspace/vllm-kimi-pd-pegaflow

Branch:
kimi-pd-pegaflow-v0.20

Base:
vLLM v0.20.0

Latest known HEAD:
a1cb3d2ce fix(spec_decode): allow reasoning logits processor
```

Remotes:

```text
origin  https://github.com/wz1qqx/vllm.git
novita  https://github.com/novitalabs/vllm-int.git
upstream https://github.com/vllm-project/vllm.git
```

Push local commits to both active remotes:

```bash
git push origin kimi-pd-pegaflow-v0.20
git push novita kimi-pd-pegaflow-v0.20
```

The branch should usually be aligned like this:

```text
HEAD == origin/kimi-pd-pegaflow-v0.20 == novita/kimi-pd-pegaflow-v0.20
```

Check:

```bash
git status --short --branch
git log --oneline -8 --decorate
git ls-remote origin refs/heads/kimi-pd-pegaflow-v0.20
git ls-remote novita refs/heads/kimi-pd-pegaflow-v0.20
```

### PegaFlow Worktree

The image build also overlays PegaFlow patches from a separate worktree:

```text
Local path:
/Users/ppio-dn-289/Documents/dynamo-vllm-workspace/pegaflow-kimi-pd-pegaflow

Branch:
kimi-pd-pegaflow

Base:
338c426 / tag v0.0.20

Latest known HEAD:
ba60953 fix(connector): import NIXL connector from vLLM 0.20 package
```

Remotes:

```text
origin   https://github.com/wz1qqx/pegaflow.git
upstream https://github.com/novitalabs/pegaflow.git
```

Push PegaFlow commits to both active remotes:

```bash
cd /Users/ppio-dn-289/Documents/dynamo-vllm-workspace/pegaflow-kimi-pd-pegaflow
git push origin kimi-pd-pegaflow
git push upstream kimi-pd-pegaflow
```

## Important Commits

Recent vLLM commit stack:

```text
a1cb3d2ce fix(spec_decode): allow reasoning logits processor
3acae96fd docs: add Kimi PD PegaFlow runbook
ccd9b95e8 fix(engine): keep idle polling on unfinished requests
a1c8dc5cc fix(metrics): compute TTFT prompt length from prefill stats
65c7265b4 fix(nixl): align PP stage descriptors by region label
0e469d0b0 fix(nixl): wire PP metadata and handshake targets for kv transfer
a93d2e44f fix(tests): use create_spec_decode_metadata in logit bias spec decode tests
cf03bdd77 test: update test_nixl_connector_pp to use v0.20.0 TransferTopology API
fa737bf43 feat(spec_decode): add LogitBiasLogitsProcessor.apply_with_spec_decode()
71be30fa7 fix(tests): adapt kimi test fixtures for vllm v0.20.0 API changes
```

Recent PegaFlow commit stack:

```text
ba60953 fix(connector): import NIXL connector from vLLM 0.20 package
0ff6175 fix(connector): add pp_size to namespace and fix unregister guard for PP
1074b90 style(connector): fix ruff lint errors in pd_connector
82d68de feat(connector): add PegaPdConnector for Dynamo PD disaggregated serving
338c426 feat(connector): switch most connector logs to debug and bump 0.20.0 (#221)
```

## Development Rules for This Worktree

Follow the repository `AGENTS.md`.

Key local rules:

- Do not use system `python3` or bare `pip` for vLLM tests.
- Use the server venv Python for server-side validation:

```text
/ppio1/venvs/vllm-kimi-pd-v0.20/bin/python
```

- Keep unrelated local changes out of commits unless explicitly asked.
- For this branch, push committed code to both:

```text
origin/kimi-pd-pegaflow-v0.20
novita/kimi-pd-pegaflow-v0.20
```

## Server Test Environment

SSH target:

```text
root@10.121.196.3
```

Server source directory:

```text
/ppio1/vllm-kimi-src
```

Server venv:

```text
/ppio1/venvs/vllm-kimi-pd-v0.20
```

Python:

```text
/ppio1/venvs/vllm-kimi-pd-v0.20/bin/python
```

vLLM CLI:

```text
/ppio1/venvs/vllm-kimi-pd-v0.20/bin/vllm
```

NIXL is installed in this venv. Check with:

```bash
ssh root@10.121.196.3 \
  'source /ppio1/venvs/vllm-kimi-pd-v0.20/bin/activate && python -c "import nixl; print(nixl.__file__)"'
```

## Syncing Local vLLM Code to the Server

The server source directory is not assumed to be a git checkout. Use `rsync`.

From the local vLLM worktree:

```bash
cd /Users/ppio-dn-289/Documents/llmd-vllm-workspace/vllm-kimi-pd-pegaflow
```

Sync the current patched production and test files:

```bash
rsync -av --relative \
  vllm/distributed/kv_transfer/kv_connector/utils.py \
  vllm/distributed/kv_transfer/kv_connector/v1/nixl/metadata.py \
  vllm/distributed/kv_transfer/kv_connector/v1/nixl/scheduler.py \
  vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py \
  vllm/v1/worker/gpu_worker.py \
  vllm/v1/metrics/stats.py \
  vllm/v1/core/sched/scheduler.py \
  vllm/v1/engine/coordinator.py \
  vllm/v1/engine/core.py \
  vllm/v1/worker/ubatching.py \
  tests/v1/kv_connector/unit/test_nixl_connector.py \
  tests/v1/kv_connector/unit/test_nixl_connector_pp.py \
  tests/tool_parsers/test_kimi_k2_tool_parser.py \
  tests/entrypoints/openai/test_kimi_params_validation.py \
  tests/v1/sample/test_rejection_sampler.py \
  root@10.121.196.3:/ppio1/vllm-kimi-src/
```

For production service runs, syncing to `/ppio1/vllm-kimi-src` is not enough.
Patch the venv site-packages too:

```bash
ssh root@10.121.196.3 'set -e
VENV=/ppio1/venvs/vllm-kimi-pd-v0.20/lib/python3.12/site-packages
SRC=/ppio1/vllm-kimi-src

cp $SRC/vllm/distributed/kv_transfer/kv_connector/utils.py \
  $VENV/vllm/distributed/kv_transfer/kv_connector/utils.py
cp $SRC/vllm/distributed/kv_transfer/kv_connector/v1/nixl/metadata.py \
  $VENV/vllm/distributed/kv_transfer/kv_connector/v1/nixl/metadata.py
cp $SRC/vllm/distributed/kv_transfer/kv_connector/v1/nixl/scheduler.py \
  $VENV/vllm/distributed/kv_transfer/kv_connector/v1/nixl/scheduler.py
cp $SRC/vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py \
  $VENV/vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py
cp $SRC/vllm/v1/worker/gpu_worker.py \
  $VENV/vllm/v1/worker/gpu_worker.py
cp $SRC/vllm/v1/metrics/stats.py \
  $VENV/vllm/v1/metrics/stats.py
cp $SRC/vllm/v1/core/sched/scheduler.py \
  $VENV/vllm/v1/core/sched/scheduler.py
cp $SRC/vllm/v1/engine/coordinator.py \
  $VENV/vllm/v1/engine/coordinator.py
cp $SRC/vllm/v1/engine/core.py \
  $VENV/vllm/v1/engine/core.py
cp $SRC/vllm/v1/worker/ubatching.py \
  $VENV/vllm/v1/worker/ubatching.py
'
```

Verify the server venv is loading the patched NIXL files:

```bash
ssh root@10.121.196.3 'cd /ppio1/vllm-kimi-src && /ppio1/venvs/vllm-kimi-pd-v0.20/bin/python - << "PY"
import inspect
import vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker as w
import vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata as m
print("worker", inspect.getfile(w))
print("metadata_version", m.NIXL_CONNECTOR_VERSION)
print("has_local_region_mapper", hasattr(w.NixlConnectorWorker, "_get_local_region_ids_for_remote_stage"))
PY'
```

Expected:

```text
metadata_version 3
has_local_region_mapper True
```

## Unit Tests on the Server

Run tests from the server source directory:

```bash
cd /ppio1/vllm-kimi-src
```

Kimi tool parser and Kimi params:

```bash
PYTHONPATH=. /ppio1/venvs/vllm-kimi-pd-v0.20/bin/python -m pytest \
  tests/tool_parsers/test_kimi_k2_tool_parser.py \
  tests/entrypoints/openai/test_kimi_params_validation.py -v
```

Known result:

```text
50 passed
22 passed
```

Spec decode + logit bias:

```bash
PYTHONPATH=. /ppio1/venvs/vllm-kimi-pd-v0.20/bin/python -m pytest \
  tests/v1/sample/test_rejection_sampler.py -v
```

Known result:

```text
68 passed
```

NIXL PP tests:

```bash
PYTHONPATH=. /ppio1/venvs/vllm-kimi-pd-v0.20/bin/python -m pytest \
  tests/v1/kv_connector/unit/test_nixl_connector_pp.py -v
```

Known result after descriptor mapping fix:

```text
60 passed
```

## End-to-End PD NIXL Tests

Main test runner:

```text
/ppio1/run_pd_nixl_case.sh
```

Usage:

```bash
/ppio1/run_pd_nixl_case.sh \
  <label> \
  <prefill_gpus> <prefill_tp> <prefill_pp> \
  <decode_gpus> <decode_tp> <decode_pp> \
  <base_port> <max_model_len>
```

Logs:

```text
/ppio1/pd-case-logs/<label>-prefill.log
/ppio1/pd-case-logs/<label>-decode.log
/ppio1/pd-case-logs/<label>-proxy.log
/ppio1/pd-case-logs/<label>-result.json
/ppio1/pd-case-logs/<label>-summary.txt
```

Baseline:

```bash
ssh root@10.121.196.3 \
  '/ppio1/run_pd_nixl_case.sh baseline 0,1,2,3 4 1 4,5,6,7 4 1 25000 128'
```

Known result:

```text
status=success
http_code=200
text='verybeautifulcity.Itisveryclean'
```

PP2:

```bash
ssh root@10.121.196.3 \
  '/ppio1/run_pd_nixl_case.sh pp2fix 0,1,2,3 2 2 4,5,6,7 4 1 26000 128'
```

Known result:

```text
status=success
http_code=200
text='verygoodplaceforpeoplewhowant'
```

PP4:

```bash
ssh root@10.121.196.3 \
  '/ppio1/run_pd_nixl_case.sh pp4fix 0,1,2,3 1 4 4,5,6,7 4 1 27000 128'
```

Known result:

```text
status=success
http_code=200
text='verygoodplaceforpeoplewhowant'
```

Check for NIXL transfer failures:

```bash
ssh root@10.121.196.3 \
  'grep -nE "transfer_setup_failed|NIXL_ERR_INVALID_PARAM|NIXL transfer failure" /ppio1/pd-case-logs/baseline-*.log /ppio1/pd-case-logs/pp2fix-*.log /ppio1/pd-case-logs/pp4fix-*.log || true'
```

Expected:

```text
no matches
```

### Interpreting Output Differences

The PP2/PP4 output differs from baseline, but logprobs showed the first
divergence is a near-tie (`beautiful` vs `good`) under `temperature=0`, not
an obvious KV corruption.

Conclusion from the last validation:

```text
NIXL data-plane setup is fixed.
PP2 and PP4 return HTTP 200.
No NIXL_ERR_INVALID_PARAM is observed.
The remaining text difference is consistent with topology/numeric near-tie
behavior across different prefill TP/PP layouts.
```

## Image Build

Target image:

```text
image.paigpu.com/library/vllm-with-pegaflow:kimi-llmd-test
```

Current versioned tag:

```text
image.paigpu.com/library/vllm-with-pegaflow:kimi-llmd-test-v1
```

Next image version:

```text
v2
```

Final known digest:

```text
image.paigpu.com/library/vllm-with-pegaflow@sha256:de0f2541cda799cde8717e7f94abb71f4edf147bd8bab851fcc56094dc51c19c
```

Build-server local image id:

```text
sha256:91711c883d9a9a6c10dc02f3874e7ae19e2381db6e51f9d442def3d4d33ec6ee
```

Build script:

```text
/Users/ppio-dn-289/Documents/dynamo-vllm-workspace/build/build-vllm-pegaflow.sh
```

Dockerfile template:

```text
/Users/ppio-dn-289/Documents/dynamo-vllm-workspace/build/dockerfiles/Dockerfile.vllm-pegaflow-patch
```

Build config:

```text
/Users/ppio-dn-289/Documents/dynamo-vllm-workspace/build/config.yaml
```

Remote build context:

```text
build-server:/ppio1/gailun/dynamo-images-build/vllm-pegaflow/
```

Base image:

```text
vllm/vllm-openai:v0.20.0-cu130-ubuntu2404
```

PegaFlow wheel:

```text
pegaflow-llm-cu13==0.21.0
```

DeepEP:

```text
Do not pass --deepep-whl for the kimi-llmd-test image unless explicitly needed.
The validated image keeps the base image deep_ep version:
1.2.1+73b6ea4
```

### Image Versioning Policy

Every new image build must have a monotonically increasing versioned tag:

```text
image.paigpu.com/library/vllm-with-pegaflow:kimi-llmd-test-v<N>
```

Rules:

- Do not reuse old version tags.
- Keep `kimi-llmd-test` as the moving "latest test" tag.
- Build and push the versioned tag first.
- After verifying the versioned image, move `kimi-llmd-test` to the same image
  only if this version should become the current test image.
- Update this runbook after every build. Record:
  - versioned tag,
  - moving tag if updated,
  - registry digest,
  - build-server image id,
  - vLLM commit,
  - PegaFlow commit,
  - patch file counts,
  - build command,
  - verification command and result,
  - short change summary.

Future build command pattern:

```bash
cd /Users/ppio-dn-289/Documents/dynamo-vllm-workspace

VERSION=v2

./build/build-vllm-pegaflow.sh \
  --image image.paigpu.com/library/vllm-with-pegaflow:kimi-llmd-test-${VERSION} \
  --vllm-wt /Users/ppio-dn-289/Documents/llmd-vllm-workspace/vllm-kimi-pd-pegaflow \
  --push
```

If the version should also become the moving test tag:

```bash
ssh build-server 'set -e
VERSION=v2
docker tag \
  image.paigpu.com/library/vllm-with-pegaflow:kimi-llmd-test-${VERSION} \
  image.paigpu.com/library/vllm-with-pegaflow:kimi-llmd-test
docker push image.paigpu.com/library/vllm-with-pegaflow:kimi-llmd-test
'
```

### Image Build History

#### v1 - 2026-04-29

Tags:

```text
image.paigpu.com/library/vllm-with-pegaflow:kimi-llmd-test-v1
image.paigpu.com/library/vllm-with-pegaflow:kimi-llmd-test
```

Digest:

```text
image.paigpu.com/library/vllm-with-pegaflow@sha256:de0f2541cda799cde8717e7f94abb71f4edf147bd8bab851fcc56094dc51c19c
```

Build-server image id:

```text
sha256:91711c883d9a9a6c10dc02f3874e7ae19e2381db6e51f9d442def3d4d33ec6ee
```

Source commits:

```text
vLLM:    a1cb3d2ce fix(spec_decode): allow reasoning logits processor
PegaFlow: ba60953 fix(connector): import NIXL connector from vLLM 0.20 package
```

Patch counts:

```text
vllm -> 15 files
pegaflow -> 5 files
DeepEP whl: not provided, base image keeps deep_ep 1.2.1+73b6ea4
```

Build commands:

```bash
cd /Users/ppio-dn-289/Documents/dynamo-vllm-workspace

./build/build-vllm-pegaflow.sh \
  --image image.paigpu.com/library/vllm-with-pegaflow:kimi-llmd-test \
  --vllm-wt /Users/ppio-dn-289/Documents/llmd-vllm-workspace/vllm-kimi-pd-pegaflow \
  --push

ssh build-server 'set -e
docker tag \
  image.paigpu.com/library/vllm-with-pegaflow:kimi-llmd-test \
  image.paigpu.com/library/vllm-with-pegaflow:kimi-llmd-test-v1
docker push image.paigpu.com/library/vllm-with-pegaflow:kimi-llmd-test-v1
'
```

Change summary:

```text
Includes all prior Kimi params/tool-parser, logit_bias/spec-decode, NIXL PP,
metrics, and PegaFlow PD connector patches. Adds the spec-decode startup fix
for explicit ReasoningLogitsProcessor and applies its spec-decode hook in the
rejection sampler.
```

Verification:

```text
Server unit test:
PYTHONPATH=. /ppio1/venvs/vllm-kimi-pd-v0.20/bin/python -m pytest \
  tests/v1/sample/test_rejection_sampler.py -v
Result: 71 passed, 16 warnings

Image source check:
spec_decode_processors ['MinTokensLogitsProcessor', 'LogitBiasLogitsProcessor', 'ReasoningLogitsProcessor']
reasoning_has_spec_hook True
rejection_uses_hook True
deep_ep_version 1.2.1+73b6ea4
```

### Patch Files Included in the Image

vLLM patch list:

```bash
cd /Users/ppio-dn-289/Documents/llmd-vllm-workspace/vllm-kimi-pd-pegaflow
git diff v0.20.0..HEAD --name-only --diff-filter=AM -- 'vllm/'
```

Known vLLM files:

```text
vllm/distributed/kv_transfer/kv_connector/utils.py
vllm/distributed/kv_transfer/kv_connector/v1/nixl/metadata.py
vllm/distributed/kv_transfer/kv_connector/v1/nixl/scheduler.py
vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py
vllm/entrypoints/openai/chat_completion/protocol.py
vllm/entrypoints/openai/chat_completion/serving.py
vllm/sampling_params.py
vllm/tool_parsers/abstract_tool_parser.py
vllm/tool_parsers/kimi_k2_tool_parser.py
vllm/v1/metrics/loggers.py
vllm/v1/metrics/stats.py
vllm/v1/sample/logits_processor/__init__.py
vllm/v1/sample/logits_processor/builtin.py
vllm/v1/sample/rejection_sampler.py
vllm/v1/worker/gpu_worker.py
```

PegaFlow patch list:

```bash
cd /Users/ppio-dn-289/Documents/dynamo-vllm-workspace/pegaflow-kimi-pd-pegaflow
git diff 338c426..HEAD --name-only --diff-filter=AM -- 'python/pegaflow/'
```

Known PegaFlow files:

```text
python/pegaflow/connector/__init__.py
python/pegaflow/connector/common.py
python/pegaflow/connector/pd_connector.py
python/pegaflow/connector/worker.py
python/pegaflow/vllm_plugin.py
```

### Dry Run

Always dry-run first:

```bash
cd /Users/ppio-dn-289/Documents/dynamo-vllm-workspace

VERSION=v2

./build/build-vllm-pegaflow.sh \
  --image image.paigpu.com/library/vllm-with-pegaflow:kimi-llmd-test-${VERSION} \
  --vllm-wt /Users/ppio-dn-289/Documents/llmd-vllm-workspace/vllm-kimi-pd-pegaflow \
  --dry-run
```

Expected summary:

```text
vllm -> 15 files
pegaflow -> 5 files
DeepEP whl: (not provided, skip reinstall)
```

### Build and Push

Important: pass `--vllm-wt`. The build script default vLLM worktree is not this
worktree.

```bash
cd /Users/ppio-dn-289/Documents/dynamo-vllm-workspace

VERSION=v2

./build/build-vllm-pegaflow.sh \
  --image image.paigpu.com/library/vllm-with-pegaflow:kimi-llmd-test-${VERSION} \
  --vllm-wt /Users/ppio-dn-289/Documents/llmd-vllm-workspace/vllm-kimi-pd-pegaflow \
  --push
```

The script will:

1. collect patch files into a temporary context,
2. generate `Dockerfile`,
3. rsync the context to `build-server:/ppio1/gailun/dynamo-images-build/vllm-pegaflow/`,
4. run Docker BuildKit on build-server,
5. push the image to `image.paigpu.com/library`.

## Image Verification

Run:

```bash
VERSION=v2

ssh build-server 'docker run --rm -i --entrypoint python3 \
  image.paigpu.com/library/vllm-with-pegaflow:kimi-llmd-test-'"${VERSION}"' - << "PY"
import importlib.metadata
import inspect
import vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata as metadata
import vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker as worker
import vllm.v1.sample.logits_processor as logits_processor
import vllm.v1.sample.logits_processor.builtin as builtin
import vllm.v1.sample.rejection_sampler as rejection_sampler
from pegaflow.connector.pd_connector import PegaPdConnector

print("vllm_version", importlib.metadata.version("vllm"))
print("nixl_connector_version", metadata.NIXL_CONNECTOR_VERSION)
print("worker_file", inspect.getfile(worker))
print("has_region_mapper", hasattr(worker.NixlConnectorWorker, "_get_local_region_ids_for_remote_stage"))
print("pegaflow_pd_connector", PegaPdConnector.__name__)
print("spec_decode_processors", [c.__name__ for c in logits_processor.SPEC_DECODE_LOGITS_PROCESSORS])
print("reasoning_has_spec_hook", hasattr(builtin.ReasoningLogitsProcessor, "apply_with_spec_decode"))
print("rejection_uses_hook", "apply_with_spec_decode" in inspect.getsource(rejection_sampler.RejectionSampler.apply_logits_processors))
try:
    print("deep_ep_version", importlib.metadata.version("deep_ep"))
except importlib.metadata.PackageNotFoundError:
    print("deep_ep_version", "not installed")
PY'
```

Expected:

```text
vllm_version 0.20.0
nixl_connector_version 3
worker_file /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py
has_region_mapper True
pegaflow_pd_connector PegaPdConnector
spec_decode_processors ['MinTokensLogitsProcessor', 'LogitBiasLogitsProcessor', 'ReasoningLogitsProcessor']
reasoning_has_spec_hook True
rejection_uses_hook True
deep_ep_version 1.2.1+73b6ea4
```

## Common Failure Modes

### `PegaPdConnector` Import Fails

Symptom:

```text
ModuleNotFoundError: No module named 'vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector'
```

Cause:

The PegaFlow branch is missing:

```text
ba60953 fix(connector): import NIXL connector from vLLM 0.20 package
```

Fix:

```bash
cd /Users/ppio-dn-289/Documents/dynamo-vllm-workspace/pegaflow-kimi-pd-pegaflow
git fetch upstream
git checkout kimi-pd-pegaflow
git pull upstream kimi-pd-pegaflow
```

### `make_prepped_xfer` Fails with `NIXL_ERR_INVALID_PARAM`

Likely cause:

The vLLM branch is missing the region label descriptor mapping fix:

```text
65c7265b4 fix(nixl): align PP stage descriptors by region label
```

Verify server venv:

```bash
ssh root@10.121.196.3 'cd /ppio1/vllm-kimi-src && /ppio1/venvs/vllm-kimi-pd-v0.20/bin/python - << "PY"
import vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata as m
import vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker as w
print(m.NIXL_CONNECTOR_VERSION)
print(hasattr(w.NixlConnectorWorker, "_get_local_region_ids_for_remote_stage"))
PY'
```

Expected:

```text
3
True
```

### End-to-End Output Differs from Baseline

`temperature=0` is set in `/ppio1/run_pd_nixl_case.sh`.

The observed PP2/PP4 output difference was investigated with `logprobs=5`.
The first divergence was a near-tie in greedy decoding, not an obvious transfer
failure.

Treat link success criteria as:

```text
status=success
http_code=200
no NIXL_ERR_INVALID_PARAM
no transfer_setup_failed
```

Use logprobs if correctness needs deeper numerical comparison.

## Feishu Handoff Docs

Detailed Feishu docs created during this work:

```text
vllm-with-pegaflow:kimi-llmd-test 镜像构建交接文档
https://www.feishu.cn/docx/Ok4tdvTdEolO2jxc27QcP6q8nxe

b300-kimi-prod-v1 ~ v4 镜像构建历史与交接文档
https://www.feishu.cn/docx/Qo8ndYoaUoRkO2xMMUrcv8DUnAf
```

## New Session Checklist

When starting a new session:

1. Read this file.
2. Check vLLM status:

   ```bash
   git status --short --branch
   git log --oneline -5 --decorate
   ```

3. Check that PegaFlow is current:

   ```bash
   git -C /Users/ppio-dn-289/Documents/dynamo-vllm-workspace/pegaflow-kimi-pd-pegaflow \
     log --oneline -5 --decorate
   ```

4. After code changes, commit and push vLLM to:

   ```bash
   git push origin kimi-pd-pegaflow-v0.20
   git push novita kimi-pd-pegaflow-v0.20
   ```

5. Sync code to `root@10.121.196.3`, patch venv, run unit tests.
6. Run E2E cases with `/ppio1/run_pd_nixl_case.sh`.
7. Dry-run the image build.
8. Build and push the image.
9. Verify the image import checks.
